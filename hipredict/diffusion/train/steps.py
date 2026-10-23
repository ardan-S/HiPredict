import torch
import torch.nn.functional as F

from hipredict.diffusion.config import ModelConfig, TrainingConfig
from hipredict.diffusion.models.utils import (
    DDPMScheduler,
    compute_edge,
    fm_interpolate,
    fm_velocity_target,
    sobel_gradients,
)


def train_step(
    batch: dict,
    model: torch.nn.Module,
    model_type: str,
    edge_model: torch.nn.Module,
    scheduler: DDPMScheduler,
    device: torch.device,
    cfg: ModelConfig,
    cfg_train: TrainingConfig = None,
):
    """
    Perform a single training step.

    Resolves the source/target images, then calls the DDPM or flow-matching step.

    Args:
        batch: A dictionary containing the input data.
        model: The diffusion model being trained.
        scheduler: The noise scheduler (only used by DDPM).
        device: The device to train on (CPU or GPU).
        cfg: Configuration parameters for the model and training.
        model_type: Type of the diffusion model, e.g. "BidirectionalFM".
    Returns:
        (loss, loss_ir2red, loss_red2ir, aux): the training loss, the per-direction
        losses of a bidirectional batch (else None), and the logged loss terms.
    """
    ir = batch["ir10"].to(device)
    red = batch["red4"].to(device)
    direction = None
    if cfg.direction == "red2ir":
        x_source, x_target = red, ir
    elif cfg.direction == "ir2red":
        x_source, x_target = ir, red
    elif cfg.direction == "bidirectional":
        B = ir.size(0)
        if B < 2:
            raise ValueError("Bidirectional training needs a batch size of at least 2")
        half = B // 2
        # half of the batch is ir2red, half is red2ir
        direction = torch.cat(
            [
                torch.zeros(half, dtype=torch.long, device=device),
                torch.ones(B - half, dtype=torch.long, device=device),
            ]
        )
        m = (direction == 0)[:, None, None, None]
        x_source = torch.where(m, ir, red)
        x_target = torch.where(m, red, ir)
    else:
        raise ValueError(f"Unknown direction: {cfg.direction}")

    edge = compute_edge(x_source, cfg.edge_mode, edge_model)

    if "DDPM" in model_type:
        return _train_ddpm_step(
            model, scheduler, x_source, x_target, edge, direction, device
        )
    elif "FM" in model_type:
        mode = cfg.Mode
        return _train_fm_step(
            model, x_source, x_target, edge, mode, direction, device, cfg_train
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _train_ddpm_step(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    x_source: torch.Tensor,
    x_target: torch.Tensor,
    edge: torch.Tensor,
    direction: torch.Tensor,
    device: torch.device,
):
    """
    Perform a single DDPM training step (noise prediction).
    """
    B = x_target.size(0)

    t = torch.randint(0, scheduler.num_timesteps, (B,), device=device).long()
    noise = torch.randn_like(x_target)
    x_t = scheduler.add_noise(x_target, t, noise)

    noise_pred = model(x_t, x_source, edge, direction, t)

    loss = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=[1, 2, 3])

    if direction is not None:
        loss_ir2red = loss[direction == 0].mean()
        loss_red2ir = loss[direction == 1].mean()
        loss = 0.5 * loss_ir2red + 0.5 * loss_red2ir
    else:
        loss_ir2red = None
        loss_red2ir = None
        loss = loss.mean()

    return loss, loss_ir2red, loss_red2ir, None


def _train_fm_step(
    model: torch.nn.Module,
    x_source: torch.Tensor,
    x_target: torch.Tensor,
    edge: torch.Tensor,
    mode: str,
    direction: torch.Tensor,
    device: torch.device,
    cfg_train: TrainingConfig = None,
):
    """
    Perform a single flow-matching training step.
    """

    B = x_target.size(0)
    if mode == "Bridge":
        # Bridge mode samples t from a fixed four-point grid.
        grid = torch.tensor([0.0, 0.25, 0.50, 0.75], device=device)
        idx = torch.randint(0, len(grid), (B,), device=device)
        t = grid[idx]
    elif mode == "Condition":
        # Train on the full range of timesteps for conditional generation.
        t = torch.rand(B, device=device)

    # Model prediction
    if mode == "Condition":
        # Path from noise (t=0) to target (t=1): x_t = (1-t)*noise + t*x_target
        noise = torch.randn_like(x_target)
        x_t = fm_interpolate(x_target, noise, t)
        v_target = fm_velocity_target(x_target, noise)
    elif mode == "Bridge":
        # Path from source (t=0) to target (t=1): x_t = (1-t)*x_source + t*x_target.
        x_t = fm_interpolate(x_target, x_source, t)
        v_target = fm_velocity_target(x_target, x_source)
    else:
        raise ValueError(f"Unknown FM mode: {mode}. Must be 'Condition' or 'Bridge'.")

    v_pred = model(x_t, x_source, edge, direction, t)

    loss_velocity = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=[1, 2, 3])

    # Endpoint MAE is logged as a diagnostic only.
    x1_pred = x_t + (1 - t[:, None, None, None]) * v_pred
    loss_mae = F.l1_loss(x1_pred, x_target)

    if direction is not None:
        loss_ir2red = loss_velocity[direction == 0].mean()
        loss_red2ir = loss_velocity[direction == 1].mean()
        loss_velocity_balanced = 0.5 * loss_ir2red + 0.5 * loss_red2ir
    else:
        loss_ir2red = None
        loss_red2ir = None
        loss_velocity_balanced = loss_velocity.mean()

    loss = loss_velocity_balanced
    aux = {"velocity": loss_velocity_balanced, "mae": loss_mae}

    # Signed Sobel loss on the velocity field.
    if cfg_train.lambda_grad > 0:
        gx_p, gy_p = sobel_gradients(v_pred, padding_mode="reflect")
        gx_t, gy_t = sobel_gradients(v_target, padding_mode="reflect")
        loss_grad_v = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
        loss = loss + cfg_train.lambda_grad * loss_grad_v
        aux["grad_v"] = loss_grad_v

    return loss, loss_ir2red, loss_red2ir, aux
