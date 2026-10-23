import argparse
import logging
import os
from collections import defaultdict

import pandas as pd
import torch
from pydantic import ValidationError
import wandb
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from hipredict.diffusion.config import DataConfig, DDPMConfig, FMConfig, TrainingConfig
from hipredict.diffusion.dataset import DiffusionDataset, get_loader
from hipredict.diffusion.dataset.utils import split_by_date
from hipredict.diffusion.models.utils import DDPMScheduler, load_edge_model
from hipredict.diffusion.utils import get_model, resume_checkpoint, save_checkpoint, set_seed

from .steps import train_step

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", type=str, default="HiRISE")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        default="UnidirectionalDDPM",
        choices=[
            "UnidirectionalDDPM",
            "BidirectionalDDPM",
            "UnidirectionalFM",
            "BidirectionalFM",
        ],
    )
    parser.add_argument(
        "--edge_type",
        type=str,
        default="sobel",
        choices=["sobel", "dexined", "soft_canny", "none"],
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Record CSV to train on (default: files/data_record_alls.csv)",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=None)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Train DataLoader worker processes (overrides config if set).",
    )
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=None,
        help="Validation DataLoader worker processes (overrides config if set).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--direction",
        type=str,
        default="ir2red",
        choices=["ir2red", "red2ir", "bidirectional"],
    )
    parser.add_argument(
        "--FM_mode", type=str, default="Bridge", choices=["Bridge", "Condition"]
    )

    # outputs
    parser.add_argument("--save_dir", type=str, default="outputs/models/")
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest.pt in save_dir if it exists.",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        default=None,
        help="Disable the EMA of model weights during training (on by default).",
    )
    parser.add_argument(
        "--lambda_grad",
        type=float,
        default=None,
        help="Weight for signed Sobel loss on the FM velocity field.",
    )

    args = parser.parse_args()

    # Bidirectional architectures need a direction label for every sample.
    if args.model_type.startswith("Bidirectional") and args.direction != "bidirectional":
        parser.error(f"{args.model_type} must be trained with --direction bidirectional")
    if args.model_type.startswith("Unidirectional") and args.direction == "bidirectional":
        parser.error(f"{args.model_type} cannot be trained with --direction bidirectional")
    if args.direction == "bidirectional" and args.batch_size is not None and args.batch_size < 2:
        parser.error("Bidirectional training needs --batch_size of at least 2")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        # Model configuration
        cfg_model = None
        if "DDPM" in args.model_type:
            cfg_model = DDPMConfig()
        elif "FM" in args.model_type:
            cfg_model = FMConfig()
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        if args.direction:
            cfg_model.direction = args.direction
        if args.edge_type:
            cfg_model.edge_mode = args.edge_type
        if args.model_type:
            cfg_model.type = args.model_type

        # Training configuration
        cfg_train = TrainingConfig()
        for name in ("batch_size", "lr", "total_steps", "log_every", "val_every"):
            if getattr(args, name) is not None:
                setattr(cfg_train, name, getattr(args, name))
        if args.num_workers is not None:
            cfg_train.num_workers = args.num_workers
        if args.val_num_workers is not None:
            cfg_train.val_num_workers = args.val_num_workers
        if args.save_dir:
            cfg_train.save_dir = args.save_dir
        if args.save_every is not None:
            cfg_train.save_every = args.save_every
        if args.resume:
            cfg_train.resume = args.resume
        if args.seed is not None:
            cfg_train.seed = args.seed
        if args.no_ema:
            cfg_train.use_ema = False
        if args.lambda_grad is not None:
            cfg_train.lambda_grad = args.lambda_grad
        if args.FM_mode and "FM" in args.model_type:
            cfg_model.Mode = args.FM_mode
        if args.edge_type == "none":
            cfg_model.in_channels = 2
        if cfg_model.direction == "bidirectional" and cfg_train.batch_size < 2:
            parser.error("Bidirectional training needs a batch size of at least 2")

        # Dataset configuration
        cfg_dataset = DataConfig()
        if args.csv_path:
            cfg_dataset.csv_path = args.csv_path
        # The model records the preprocessing it is trained with.
        cfg_model.normalization = cfg_dataset.normalization
    except ValidationError as error:
        parser.error(str(error))

    set_seed(cfg_train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting training with the following configuration:")
    logger.info("Using device: %s", device)
    logger.info("Model configuration: %s", cfg_model.model_dump())
    logger.info("Training configuration: %s", cfg_train.model_dump())
    logger.info("Dataset configuration: %s", cfg_dataset.model_dump())

    use_wandb = not args.no_wandb
    if use_wandb:
        # Persist a stable wandb run id in save_dir so resumed jobs append to
        # the same run instead of creating a new one each submission.
        os.makedirs(cfg_train.save_dir, exist_ok=True)
        wandb_id_path = os.path.join(cfg_train.save_dir, "wandb_id.txt")
        if cfg_train.resume and os.path.exists(wandb_id_path):
            with open(wandb_id_path) as f:
                wandb_id = f.read().strip()
        else:
            wandb_id = wandb.util.generate_id()
            with open(wandb_id_path, "w") as f:
                f.write(wandb_id)

        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            id=wandb_id,
            resume="allow",
            config={
                **cfg_model.model_dump(),
                **cfg_train.model_dump(),
                **cfg_dataset.model_dump(),
            },
        )

    df = pd.read_csv(cfg_dataset.csv_path)
    train_sets, test_sets, val_sets, train_val_date, val_test_date = split_by_date(
        df, 0.7, 0.2, 0.1
    )
    train_dataset = DiffusionDataset(cfg_dataset, allowed_sets=train_sets)
    val_dataset = DiffusionDataset(cfg_dataset, allowed_sets=val_sets)
    train_loader = get_loader(
        train_dataset,
        batch_size=cfg_train.batch_size,
        num_workers=cfg_train.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_loader = get_loader(
        val_dataset,
        batch_size=cfg_train.batch_size,
        num_workers=cfg_train.val_num_workers,
        shuffle=False,
        # Full batches keep both directions populated in bidirectional training.
        drop_last=True,
    )

    # Model setup
    model = get_model(cfg_model, device)
    scheduler = None
    if "DDPM" in args.model_type:
        scheduler = DDPMScheduler(
            num_timesteps=cfg_model.num_timesteps,
            beta_start=cfg_model.beta_start,
            beta_end=cfg_model.beta_end,
        ).to(device)

    edge_model = load_edge_model(cfg_model.edge_mode, device)

    # Optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_train.lr)
    schedule_lr = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg_train.lr_decay_every, gamma=cfg_train.lr_decay
    )

    # Resume from checkpoint if requested
    start_step = 0
    ema_state = None
    if cfg_train.resume:
        ckpt_path = os.path.join(cfg_train.save_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            last_step, cfg_model_resume, cfg_train_resume, ema_state = (
                resume_checkpoint(
                    ckpt_path, model, optimizer, schedule_lr, cfg_model, cfg_train
                )
            )
            start_step = last_step + 1
            logger.info("Resumed from %s at step %d", ckpt_path, last_step)
        else:
            logger.warning(
                "resume=True but no checkpoint found at %s; starting from scratch",
                ckpt_path,
            )

    # Use EMA for smoother training
    ema = None
    if cfg_train.use_ema:
        ema = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(cfg_train.ema_decay),
            use_buffers=True,
        )
        if ema_state is not None:
            # ema_state holds the unwrapped model weights
            ema.module.load_state_dict(ema_state)
            # inform AveragedModel that the EMA is already initialized
            ema.n_averaged.fill_(1)
            logger.info("Restored EMA weights from checkpoint")

    # Training loop
    step = start_step
    train_iter = iter(train_loader)
    while step <= cfg_train.total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad()
        model.train()
        loss, loss_ir2red, loss_red2ir, aux = train_step(
            batch,
            model,
            args.model_type,
            edge_model,
            scheduler,
            device,
            cfg_model,
            cfg_train,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        schedule_lr.step()

        # Update EMA weights
        if ema is not None:
            ema.update_parameters(model)

        # Logging
        if step % cfg_train.log_every == 0:
            log = f"Step {step}: Loss = {loss.item():.4f}, Grad Norm = {grad_norm:.4f}"
            if loss_ir2red is not None:
                log += f", Loss ir2red = {loss_ir2red.item():.4f}"
            if loss_red2ir is not None:
                log += f", Loss red2ir = {loss_red2ir.item():.4f}"
            if aux is not None:
                log += "".join(f", {k} = {v.item():.4f}" for k, v in aux.items())

            logger.info(log)

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )
                if loss_ir2red is not None:
                    wandb.log({"train/loss_ir2red": loss_ir2red.item()}, step=step)
                if loss_red2ir is not None:
                    wandb.log({"train/loss_red2ir": loss_red2ir.item()}, step=step)
                if aux is not None:
                    wandb.log(
                        {f"train/loss_{k}": v.item() for k, v in aux.items()}, step=step
                    )

        # Validation
        if step % cfg_train.val_every == 0:
            val_losses = []
            val_losses_ir2red = []
            val_losses_red2ir = []
            val_losses_aux = defaultdict(list)
            with torch.no_grad():
                val_model = ema.module if ema is not None else model
                val_model.eval()
                for val_batch in val_loader:
                    val_loss, val_loss_ir2red, val_loss_red2ir, val_aux = train_step(
                        val_batch,
                        val_model,
                        args.model_type,
                        edge_model,
                        scheduler,
                        device,
                        cfg_model,
                        cfg_train,
                    )
                    val_losses.append(val_loss.item())
                    if val_loss_red2ir is not None:
                        val_losses_red2ir.append(val_loss_red2ir.item())
                    if val_loss_ir2red is not None:
                        val_losses_ir2red.append(val_loss_ir2red.item())
                    if val_aux is not None:
                        for k, v in val_aux.items():
                            val_losses_aux[k].append(v.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            logger.info(f"Validation at step {step}: Avg Loss = {avg_val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": avg_val_loss}, step=step)
                if val_losses_ir2red:
                    wandb.log(
                        {
                            "val/loss_ir2red": sum(val_losses_ir2red)
                            / len(val_losses_ir2red)
                        },
                        step=step,
                    )
                if val_losses_red2ir:
                    wandb.log(
                        {
                            "val/loss_red2ir": sum(val_losses_red2ir)
                            / len(val_losses_red2ir)
                        },
                        step=step,
                    )
                for k, vals in val_losses_aux.items():
                    wandb.log({f"val/loss_{k}": sum(vals) / len(vals)}, step=step)

        # Checkpoint
        if step > 0 and step % cfg_train.save_every == 0:
            save_checkpoint(
                os.path.join(cfg_train.save_dir, f"model_step_{step}.pt"),
                step,
                model,
                optimizer,
                schedule_lr,
                cfg_model,
                cfg_train,
                ema,
            )
            save_checkpoint(
                os.path.join(cfg_train.save_dir, "latest.pt"),
                step,
                model,
                optimizer,
                schedule_lr,
                cfg_model,
                cfg_train,
                ema,
            )
            logger.info("Saved checkpoint at step %d", step)

        step += 1

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
