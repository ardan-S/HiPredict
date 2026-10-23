import logging
import os
import pickle
import random

import numpy as np
import torch

from hipredict.diffusion.config import DDPMConfig, FMConfig, ModelConfig, TrainingConfig
from hipredict.diffusion.models import (
    BidirectionalDDPM,
    BidirectionalFM,
    UnidirectionalDDPM,
    UnidirectionalFM,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "hipredict.diffusion"
CHECKPOINT_VERSION = 1


def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproduction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


MODELS = {
    "UnidirectionalDDPM": UnidirectionalDDPM,
    "BidirectionalDDPM": BidirectionalDDPM,
    "UnidirectionalFM": UnidirectionalFM,
    "BidirectionalFM": BidirectionalFM,
}

# Training settings that may change when a run is resumed; all others must match.
RESUMABLE_TRAINING_FIELDS = {
    "num_workers",
    "val_num_workers",
    "save_dir",
    "save_every",
    "log_every",
    "val_every",
    "total_steps",
    "resume",
}


def get_model(cfg: ModelConfig, device: str = "cuda") -> torch.nn.Module:
    """
    Build the network described by a model configuration.
    """
    model_class = getattr(cfg, "type", None)
    if model_class not in MODELS:
        raise ValueError(f"Unknown model class: {model_class}")
    model = MODELS[model_class](
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_channels=cfg.base_channels,
        num_res_blocks=cfg.num_res_blocks,
        dropout=cfg.dropout,
    )
    return model.to(device)


def model_config_from_dict(values: dict) -> ModelConfig:
    """Rebuild a DDPM or FM model configuration from its stored values."""
    cls = DDPMConfig if "DDPM" in str(values.get("type")) else FMConfig
    return cls(**values)


def checkpoint_dict(
    step: int,
    model_state: dict,
    cfg_model: ModelConfig,
    cfg_train: TrainingConfig = None,
    ema_state: dict = None,
    optimizer_state: dict = None,
    scheduler_state: dict = None,
) -> dict:
    """Assemble a checkpoint in the versioned format read by load_checkpoint."""
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "step": int(step),
        "model_config": cfg_model.model_dump(),
        "train_config": cfg_train.model_dump() if cfg_train is not None else None,
        "model": model_state,
        "ema_model": ema_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
    }


def read_checkpoint(path: str) -> dict:
    """Load a checkpoint file and check its format; no pickled classes are allowed."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(
            f"{path} is not a supported {CHECKPOINT_FORMAT} checkpoint"
        ) from error
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} has checkpoint version {checkpoint.get('version')}; "
            f"this code reads version {CHECKPOINT_VERSION}"
        )
    return checkpoint


def save_checkpoint(
    path: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg_model: ModelConfig,
    cfg_train: TrainingConfig,
    ema: torch.nn.Module = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        checkpoint_dict(
            step,
            model.state_dict(),
            cfg_model,
            cfg_train,
            ema_state=ema.module.state_dict() if ema is not None else None,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
        ),
        path,
    )


def resume_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg_model: ModelConfig,
    cfg_train: TrainingConfig,
) -> tuple:
    """
    Restore a training run, requiring the same model and training settings.

    Only the fields in RESUMABLE_TRAINING_FIELDS may differ from the checkpoint.
    """
    checkpoint = read_checkpoint(path)
    if checkpoint["optimizer"] is None or checkpoint["train_config"] is None:
        raise ValueError(f"{path} holds weights only and cannot resume training")
    cfg_model_ckpt = model_config_from_dict(checkpoint["model_config"])
    cfg_train_ckpt = TrainingConfig(**checkpoint["train_config"])

    current_model = cfg_model.model_dump()
    mismatches = [
        f"model.{k}: checkpoint={v!r}, current={current_model.get(k)!r}"
        for k, v in cfg_model_ckpt.model_dump().items()
        if current_model.get(k) != v
    ]
    current_train = cfg_train.model_dump()
    mismatches += [
        f"train.{k}: checkpoint={v!r}, current={current_train[k]!r}"
        for k, v in cfg_train_ckpt.model_dump().items()
        if k not in RESUMABLE_TRAINING_FIELDS and current_train[k] != v
    ]
    if mismatches:
        raise ValueError(
            f"Cannot resume from {path}; these settings differ from the checkpoint:\n  "
            + "\n  ".join(mismatches)
        )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return (
        checkpoint["step"],
        cfg_model_ckpt,
        cfg_train_ckpt,
        checkpoint["ema_model"],
    )


def load_checkpoint(path: str, device: torch.device = "cuda"):
    """
    Rebuild a model from the config stored in its checkpoint and load the weights.

    Prefers the EMA weights ("ema_model") when present, falling back to the raw
    model weights for checkpoints trained without EMA. Returns
    ``(model, cfg_model, cfg_train, step)``; ``cfg_train`` is None when the
    checkpoint does not record it.
    """
    checkpoint = read_checkpoint(path)
    cfg_model = model_config_from_dict(checkpoint["model_config"])
    cfg_train = checkpoint["train_config"]
    if cfg_train is not None:
        cfg_train = TrainingConfig(**cfg_train)
    model = get_model(cfg_model, device)
    state = checkpoint["ema_model"]
    if state is not None:
        logger.info("Loading EMA weights from checkpoint")
    else:
        logger.info("No EMA weights in checkpoint; loading raw model weights")
        state = checkpoint["model"]
    model.load_state_dict(state)
    return model, cfg_model, cfg_train, checkpoint["step"]
