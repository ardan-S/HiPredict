import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from pydantic import ValidationError

from hipredict.diffusion.config import DataConfig, InferenceConfig, ModelConfig
from hipredict.diffusion.dataset import DiffusionDataset, get_loader
from hipredict.diffusion.dataset.utils import batch_set_ids, split_by_date
from hipredict.diffusion.inference.inference import inference_batch
from hipredict.diffusion.models.brightness import load_brightness_ctx
from hipredict.diffusion.models.utils import DDPMScheduler, load_edge_model
from hipredict.diffusion.utils import load_checkpoint, set_seed

from .metrics import compute_metrics
from .reporting import (
    DrawEnsemble,
    plot_nmae_histogram,
    plot_pixel_histogram,
    summarise,
    write_outputs,
)

logger = logging.getLogger(__name__)


def eval_batch(
    batch: dict,
    model: torch.nn.Module,
    model_type: str,
    edge_model: torch.nn.Module,
    scheduler: DDPMScheduler,
    device: torch.device,
    cfg: ModelConfig,
    cfg_inference: InferenceConfig,
    return_denorm_images: bool = False,
    dc_correct: str = "none",
    cond_fit: dict = None,
    brightness_ctx: dict = None,
    return_per_image_metrics: bool = False,
):
    """
    Reconstruct the target with inference_batch and score it against the
    ground truth with HiPredict's metrics.

    Args:
        batch: A dictionary containing the input data.
        model: The diffusion model to evaluate.
        model_type: Type of the diffusion model (e.g. "UnidirectionalDDPM").
        edge_model: Edge model (DexiNed) or None for sobel.
        scheduler: The noise scheduler (only used by DDPM).
        device: The device to run on (CPU or GPU).
        cfg: Model configuration (uses cfg.direction, cfg.edge_mode).
        cfg_inference: Inference configuration.
        return_denorm_images: If True, also return the predicted and target images (denormalized) for further analysis.
        dc_correct: Method for post-hoc brightness (DC) correction
        cond_fit: Dictionary containing the linear fit parameters for the conditional mean correction (used if dc_correct is "cond").
        brightness_ctx: Dictionary containing the brightness head model and related information (used if dc_correct is "head").
        return_per_image_metrics: If True, also return the per-image denormalized metrics.
    Returns:
        out: A tuple containing:
            metrics: Dict of HiPredict metrics (nmae, bias, pwt, ssim) in physical units.
            pred_denorm (optional): The denormalized predicted image (if return_denorm_images).
            target_denorm (optional): The denormalized target image (if return_denorm_images).
            per_image_metrics (optional): Dict of [B] denormalized metrics (if
                return_per_image_metrics), appended last so the images keep a fixed
                position.
    """
    target_key = "raw_red4" if cfg_inference.direction == "ir2red" else "raw_ir10"
    if target_key not in batch:
        raise ValueError(f"Evaluation requires the unclipped physical target: {target_key}")
    outputs = inference_batch(
        batch,
        model,
        model_type,
        edge_model,
        scheduler,
        device,
        cfg,
        cfg_inference,
        dc_correct=dc_correct,
        cond_fit=cond_fit,
        brightness_ctx=brightness_ctx,
    )
    pred_denorm = outputs["pred_denorm"]
    target_denorm = outputs["target_denorm"]

    # Metrics are computed in physical units, as in HiPredict.
    denorm = compute_metrics(
        pred_denorm, target_denorm, per_image=return_per_image_metrics
    )
    if return_per_image_metrics:
        denorm, per_image_metrics = denorm
    metrics = {f"denorm/{k}": v for k, v in denorm.items()}

    out = [metrics]
    if return_denorm_images:
        out.append(pred_denorm)
        out.append(target_denorm)
    if return_per_image_metrics:
        out.append(per_image_metrics)

    return tuple(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader worker processes (overrides config if set).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--direction",
        type=str,
        default="ir2red",
        choices=["ir2red", "red2ir", "bidirectional"],
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Cap number of test batches (full reverse sampling is expensive); "
        "None = whole test set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval/",
        help="Directory to save the eval metrics summary",
    )
    parser.add_argument(
        "--cond_fit",
        type=str,
        default="outputs/dataset/cond_mean_fit.pt",
        help="Conditional-mean fit written by dataset.EDA (used by --dc_correct cond)",
    )
    parser.add_argument(
        "--FM_ODE_steps",
        type=int,
        default=10,
        help="Number of ODE steps for FM evaluation",
    )
    parser.add_argument(
        "--FM_solver",
        type=str,
        default=None,
        choices=["euler", "rk4"],
        help="ODE solver for FM evaluation (default: euler).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train", "val", "test"],
        default="val",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-batch metrics during evaluation",
    )
    parser.add_argument(
        "--plot_pixel_histogram",
        action="store_true",
        help="A denormalised predicted/true pixel histogram",
    )
    parser.add_argument(
        "--plot_nmae_histogram",
        action="store_true",
        help="A per-image NMAE histogram, one figure per direction",
    )
    parser.add_argument(
        "--histogram_full_range",
        action="store_true",
        help="Use the full min/max range for the histogram x-axis instead of the 0.5-99.5 percentile range.",
    )
    parser.add_argument(
        "--dc_correct",
        type=str,
        default="boundary",
        choices=["none", "oracle", "cond", "head", "boundary"],
        help="Post-hoc brightness correction (oracle uses the true target mean; "
        "reference upper bound only)",
    )
    parser.add_argument(
        "--n_draws",
        type=int,
        default=1,
        help="Sample the model this many times per batch and report the spread "
        "under <output_dir>/uncertainty. 1 keeps the plain eval.",
    )
    parser.add_argument(
        "--head_ckpt",
        type=str,
        default="outputs/ckpt/brightness/model.pt",
    )
    parser.add_argument(
        "--no_dc_anchor",
        action="store_true",
        help="Apply the dc_correct mean only to the final prediction, not after "
        "every FM ODE step",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Record CSV to evaluate (default: files/data_record_alls.csv)",
    )

    args = parser.parse_args()

    if args.n_draws < 1:
        parser.error("--n_draws must be at least 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        # Eval configuration (reuse InferenceConfig)
        cfg_eval = InferenceConfig()
        if args.batch_size:
            cfg_eval.batch_size = args.batch_size
        if args.num_workers is not None:
            cfg_eval.num_workers = args.num_workers
        if args.ckpt_path:
            cfg_eval.ckpt_path = args.ckpt_path
        if args.output_dir:
            cfg_eval.output_dir = args.output_dir
        if args.direction:
            cfg_eval.direction = args.direction
        cfg_eval.dc_anchor = args.dc_correct != "none" and not args.no_dc_anchor
        if args.FM_ODE_steps is not None:
            cfg_eval.FM_ODE_steps = args.FM_ODE_steps
        if args.FM_solver is not None:
            cfg_eval.FM_solver = args.FM_solver

        # Dataset configuration
        cfg_dataset = DataConfig()
        if args.csv_path:
            cfg_dataset.csv_path = args.csv_path
    except ValidationError as error:
        parser.error(str(error))

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and its configurations
    model, cfg_model, _, _ = load_checkpoint(cfg_eval.ckpt_path, device)
    model_type = cfg_model.type
    cfg_dataset.normalization = cfg_model.normalization

    if "DDPM" in model_type:
        cfg_eval.FM_ODE_steps = None  # not used for DDPM

    # A Bridge-mode FM starts its ODE from the source image instead of from noise, so its sampler is deterministic and every draw gives one answer.
    if (
        args.n_draws > 1
        and "FM" in model_type
        and getattr(cfg_model, "Mode", "Condition") == "Bridge"
    ):
        raise ValueError(
            f"{cfg_eval.ckpt_path} is a Bridge-mode FM checkpoint: its sampler is "
            "deterministic, so a draw ensemble would report zero spread."
        )

    # check model compatibility with the requested eval direction
    if "Unidirectional" in model_type and cfg_eval.direction == "bidirectional":
        raise AttributeError(
            "Unidirectional models cannot be evaluated in bidirectional mode."
        )
    if cfg_model.direction not in ("bidirectional", cfg_eval.direction):
        raise ValueError(
            f"Checkpoint direction ({cfg_model.direction}) does not support "
            f"eval direction ({cfg_eval.direction})"
        )

    logger.info(f"Evaluating model from checkpoint: {cfg_eval.ckpt_path}")
    logger.info(f"Model configuration: {cfg_model.model_dump()}")
    logger.info(f"Eval configuration: {cfg_eval.model_dump()}")
    logger.info(f"Using device: {device}")

    scheduler = None
    if "DDPM" in model_type:
        scheduler = DDPMScheduler(
            num_timesteps=cfg_model.num_timesteps,
            beta_start=cfg_model.beta_start,
            beta_end=cfg_model.beta_end,
        ).to(device)

    edge_model = load_edge_model(cfg_model.edge_mode, device)

    # cond_mean fit (a, b) per direction for --dc_correct cond.
    cond_fit = cond_fit_ir2red = cond_fit_red2ir = None
    if args.dc_correct == "cond":
        cond_all = torch.load(args.cond_fit, map_location=device, weights_only=False)

        def _get_cond_fit(d):
            return {
                "a": torch.tensor(float(d["a"]), device=device),
                "b": torch.tensor(float(d["b"]), device=device),
            }

        if cfg_eval.direction == "bidirectional":
            cond_fit_ir2red = _get_cond_fit(cond_all["ir2red"])
            cond_fit_red2ir = _get_cond_fit(cond_all["red2ir"])
        else:
            cond_fit = _get_cond_fit(cond_all[cfg_eval.direction])

    # Brightness head for --dc_correct head (shared across directions).
    brightness_ctx = None
    required_metacols = None
    if args.dc_correct == "head":
        brightness_ctx = load_brightness_ctx(args.head_ckpt, device)
        required_metacols = brightness_ctx["meta_cols"]
        logger.info("Loaded brightness head from %s", args.head_ckpt)

    # Load the target dataset
    df = pd.read_csv(cfg_dataset.csv_path)
    train_sets, test_sets, val_sets, _, _ = split_by_date(df, 0.7, 0.2, 0.1)
    if args.dataset == "train":
        test_sets = train_sets
    elif args.dataset == "val":
        test_sets = val_sets
    elif args.dataset == "test":
        test_sets = test_sets
    test_dataset = DiffusionDataset(
        cfg_dataset,
        allowed_sets=test_sets,
        required_metacols=required_metacols,
        boundary_sets="RED3",
    )
    test_loader = get_loader(
        test_dataset,
        batch_size=cfg_eval.batch_size,
        num_workers=cfg_eval.num_workers,
        shuffle=False,
    )

    n_batches = (
        len(test_loader)
        if args.max_batches is None
        else min(args.max_batches, len(test_loader))
    )
    logger.info(
        f"Evaluating on {args.dataset}.\nIn total {len(test_loader)} samples over {n_batches} batches"
    )

    # One pass over the split, sampling n_draws times per batch.
    ensemble = DrawEnsemble() if args.n_draws > 1 else None
    collect_pixels = args.plot_pixel_histogram
    collect_nmae = args.plot_nmae_histogram
    if (collect_pixels or collect_nmae) and args.n_draws > 1:
        logger.info("Histogram values are collected from the first draw only")

    # One running total per draw; draw d over the whole split is one replicate.
    sums = [defaultdict(float) for _ in range(args.n_draws)]
    total_count = 0
    pix = defaultdict(list)  # label -> 1-D denormalised pixel tensors
    nmae = defaultdict(list)  # label -> per-image NMAE
    # The ensemble scores samples individually, so it needs the images too.
    return_denorm_images = collect_pixels or ensemble is not None

    for i, batch in enumerate(test_loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        B = batch["ir10"].size(0)
        # Only the ensemble keys results by sample, so a plain eval skips this.
        set_ids = (
            batch_set_ids(batch, B, offset=total_count)
            if ensemble is not None
            else None
        )

        if cfg_eval.direction == "bidirectional":
            passes = [
                (
                    "ir2red",
                    cfg_eval.model_copy(update={"direction": "ir2red"}),
                    cond_fit_ir2red,
                ),
                (
                    "red2ir",
                    cfg_eval.model_copy(update={"direction": "red2ir"}),
                    cond_fit_red2ir,
                ),
            ]
        else:
            passes = [(cfg_eval.direction, cfg_eval, cond_fit)]

        batch_metrics = {}
        for direction, cfg_dir, cond_dir in passes:
            # Unidirectional runs keep the unprefixed keys they always had.
            prefix = f"{direction}/" if cfg_eval.direction == "bidirectional" else ""
            preds = []
            for d in range(args.n_draws):
                out = eval_batch(
                    batch,
                    model,
                    model_type,
                    edge_model,
                    scheduler,
                    device,
                    cfg_model,
                    cfg_dir,
                    return_denorm_images=return_denorm_images,
                    dc_correct=args.dc_correct,
                    cond_fit=cond_dir,
                    brightness_ctx=brightness_ctx,
                    return_per_image_metrics=collect_nmae,
                )
                # eval_batch returns metrics, then the images, then the per-image metrics, each only when its flag is set.
                metrics = out[0]
                pred_denorm = target_denorm = per_image_metrics = None
                if return_denorm_images:
                    pred_denorm, target_denorm = out[1], out[2]
                if collect_nmae:
                    per_image_metrics = out[-1]

                for k, v in metrics.items():
                    sums[d][f"{prefix}{k}"] += v * B
                preds.append(pred_denorm)

                # Record all pixels from the produced image using the first draw to plot the histogram
                if d == 0:
                    batch_metrics[direction] = metrics
                    if collect_pixels:
                        if cfg_eval.direction == "bidirectional":
                            pix[f"{direction} pred"].append(
                                pred_denorm.detach().flatten().cpu()
                            )
                            pix[f"{direction} true"].append(
                                target_denorm.detach().flatten().cpu()
                            )
                        else:
                            pix["prediction"].append(
                                pred_denorm.detach().flatten().cpu()
                            )
                            pix["ground truth"].append(
                                target_denorm.detach().flatten().cpu()
                            )
                    if collect_nmae:
                        nmae[direction].append(per_image_metrics["nmae"])

            if ensemble is not None:
                ensemble.add_batch(direction, set_ids, preds, target_denorm)

        total_count += B
        if args.verbose:
            logger.info(
                "Batch %d/%d:\n%s",
                i + 1,
                n_batches,
                {
                    dir_key: {k: round(v, 6) for k, v in m.items()}
                    for dir_key, m in batch_metrics.items()
                },
            )

    if total_count == 0:
        raise RuntimeError(f"No samples were evaluated on the {args.dataset} split")
    avgs = [{key: total[key] / total_count for key in total} for total in sums]

    # eval_metrics.json reports draw 0
    avg = avgs[0]
    if ensemble is not None:
        for d, draw_avg in enumerate(avgs):
            ensemble.end_draw(draw_avg)

    logger.info("Evaluation complete over %d samples:", total_count)
    for k in sorted(avg):
        logger.info("  %-12s = %.6f", k, avg[k])

    # Save a metrics summary
    os.makedirs(cfg_eval.output_dir, exist_ok=True)
    summary = {
        "metrics": avg,
        "num_samples": total_count,
        "direction": cfg_eval.direction,
        "model_type": model_type,
        "ODE_steps": cfg_eval.FM_ODE_steps,
        "FM_solver": cfg_eval.FM_solver,
        "edge_mode": cfg_model.edge_mode,
        "fm_mode": getattr(cfg_model, "Mode", "Condition"),
        "ckpt_path": cfg_eval.ckpt_path,
        "head_ckpt": args.head_ckpt if args.dc_correct == "head" else None,
        "dataset": args.dataset,
        "seed": args.seed,
        "dc_correct": args.dc_correct,
        "dc_anchor": cfg_eval.dc_anchor,
    }

    metrics_path = os.path.join(cfg_eval.output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # histogram plot
    if args.plot_pixel_histogram:
        named = {k: torch.cat(v).numpy() for k, v in pix.items() if v}
        if cfg_eval.direction == "bidirectional":
            ir2red_named = {k: v for k, v in named.items() if k.startswith("ir2red")}
            plot_pixel_histogram(
                ir2red_named,
                os.path.join(cfg_eval.output_dir, "ir2red_histogram.png"),
                title=f"{args.dataset} — denormalised pixels (ir2red)",
                percentile_plot=not args.histogram_full_range,
            )
            red2ir_named = {k: v for k, v in named.items() if k.startswith("red2ir")}
            plot_pixel_histogram(
                red2ir_named,
                os.path.join(cfg_eval.output_dir, "red2ir_histogram.png"),
                title=f"{args.dataset} — denormalised pixels (red2ir)",
                percentile_plot=not args.histogram_full_range,
            )
        else:
            plot_pixel_histogram(
                named,
                os.path.join(cfg_eval.output_dir, "histogram.png"),
                title=f"{args.dataset} — denormalised pixels ({cfg_eval.direction})",
                percentile_plot=not args.histogram_full_range,
            )

    # NMAE histogram plot
    if args.plot_nmae_histogram:
        for direction, values in nmae.items():
            prefix = f"{direction}_" if cfg_eval.direction == "bidirectional" else ""
            plot_nmae_histogram(
                {direction: np.concatenate(values)},
                os.path.join(cfg_eval.output_dir, f"{prefix}nmae_histogram.png"),
                title=f"{args.dataset} — per-image NMAE ({direction})",
                percentile_plot=not args.histogram_full_range,
            )

    # Draw-ensemble statistics land in their own subdirectory, so the outputs
    # above are exactly what a single-draw run produces.
    if ensemble is not None:
        results = ensemble.results()
        write_outputs(
            os.path.join(cfg_eval.output_dir, "uncertainty"),
            summarise(results),
            results,
            {
                "ckpt_path": cfg_eval.ckpt_path,
                "head_ckpt": args.head_ckpt if args.dc_correct == "head" else None,
                "model_type": model_type,
                "dataset": args.dataset,
                "direction": cfg_eval.direction,
                "n_draws": args.n_draws,
                "base_seed": args.seed,
                "batch_size": cfg_eval.batch_size,
                "max_batches": args.max_batches,
                "FM_ODE_steps": cfg_eval.FM_ODE_steps,
                "FM_solver": cfg_eval.FM_solver,
                "dc_correct": args.dc_correct,
                "dc_anchor": cfg_eval.dc_anchor,
            },
        )


if __name__ == "__main__":
    main()
