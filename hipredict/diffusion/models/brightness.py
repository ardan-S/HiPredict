import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from hipredict.diffusion.config import DataConfig
from hipredict.diffusion.dataset import (
    META_COLS,
    STAT_NAMES,
    DiffusionDataset,
    _image_stats,
    get_loader,
    split_by_date,
)

logger = logging.getLogger(__name__)

DIRECTIONS = ["ir2red", "red2ir"]
CYCLIC_METADATA = {"Solar_longitude": 360.0, "Local_time": 24.0}


class BrightnessModel(nn.Module):
    """Predict a correction to the RED3 boundary-based target mean."""

    def __init__(self, n_feat: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def _metadata_array(
    batch_meta: dict, direction: str, meta_cols: list[str]
) -> np.ndarray:
    """Return source- then target-side metadata with cyclic sin/cos encoding."""
    src = "ir" if direction == "ir2red" else "red"
    target = "red" if direction == "ir2red" else "ir"
    features = []
    for side in (src, target):
        for col in meta_cols:
            values = (
                batch_meta[f"{side}_{col}"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .reshape(-1)
            )
            if col in CYCLIC_METADATA:
                angle = 2.0 * np.pi * values / CYCLIC_METADATA[col]
                features.append(np.stack([np.sin(angle), np.cos(angle)], axis=1))
            else:
                features.append(values[:, None])
    return np.concatenate(features, axis=1)


def _raw_features_from_stats(
    source_stats: np.ndarray,
    batch_meta: dict,
    boundary_offset: np.ndarray,
    boundary_stats: np.ndarray,
    direction: str,
    meta_cols: list[str],
) -> np.ndarray:
    """Build unstandardized head features for one translation direction."""
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown brightness direction: {direction}")
    parts = [
        source_stats.astype(np.float64),
        np.asarray(boundary_offset, dtype=np.float64).reshape(-1, 1),
        np.asarray(boundary_stats, dtype=np.float64).reshape(-1, 15),
    ]
    parts.append(_metadata_array(batch_meta, direction, meta_cols))
    return np.concatenate(parts, axis=1)


def fit_feature_stats(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit train-only feature normalization while safely handling NaNs."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(raw, axis=0)
        std = np.nanstd(raw, axis=0)
    invalid_mean = ~np.isfinite(mean)
    if invalid_mean.any():
        logger.warning(
            "%d brightness features are entirely non-finite; using zero fill",
            int(invalid_mean.sum()),
        )
        mean[invalid_mean] = 0.0
    std[(~np.isfinite(std)) | (std < 1e-8)] = 1.0
    return mean, std


def make_model_features(
    raw: np.ndarray,
    direction: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
) -> np.ndarray:
    """Standardize continuous features and append direction one-hot."""
    z = (raw - feat_mean) / feat_std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    onehot = np.eye(len(DIRECTIONS), dtype=np.float64)[direction]
    return np.concatenate([z, onehot], axis=1).astype(np.float32)


def extract_brightness_features(
    dataset: DiffusionDataset,
    meta_cols: list[str],
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    """Read a DiffusionDataset once and cache both brightness directions.

    Each loader batch contains:
        ir10, red4: normalized images with shape (B, 1, H, W)
        stats: normalization parameters, collated per sample
        metadata: source dictionaries named ir_<column> and red_<column>, shape (B,)
        boundary_offset: RED-IR median offset, shape (B,)
        boundary_stats: RED3 IR, RED and RED-IR stats, shape (B, 15)
        has_boundary: availability flag, shape (B,)
    """
    loader = get_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    collected = {key: [] for key in ("raw", "residual", "base", "target", "direction")}
    for batch in loader:
        if "boundary_offset" not in batch:
            raise KeyError("DiffusionDataset must be created with boundary_sets='RED3'")
        if "has_boundary" in batch and not bool(
            torch.as_tensor(batch["has_boundary"]).all()
        ):
            raise RuntimeError(
                "A RED3 boundary is missing; brightness training cannot continue"
            )

        ir_stats = _image_stats(batch["ir10"]).numpy().astype(np.float64)
        red_stats = _image_stats(batch["red4"]).numpy().astype(np.float64)
        offset = batch["boundary_offset"].numpy().astype(np.float64).reshape(-1)
        boundary_stats = batch["boundary_stats"].numpy().astype(np.float64)
        batch_size_actual = len(offset)

        for direction, source_stats, target_stats in (
            ("ir2red", ir_stats, red_stats),
            ("red2ir", red_stats, ir_stats),
        ):
            base = source_stats[:, 0] + (offset if direction == "ir2red" else -offset)
            target = target_stats[:, 0]
            raw = _raw_features_from_stats(
                source_stats,
                batch["metadata"],
                offset,
                boundary_stats,
                direction,
                meta_cols,
            )
            collected["raw"].append(raw)
            collected["residual"].append(target - base)
            collected["base"].append(base)
            collected["target"].append(target)
            direction_idx = DIRECTIONS.index(direction)
            collected["direction"].append(
                np.full(batch_size_actual, direction_idx, dtype=np.int64)
            )

    return {key: np.concatenate(parts) for key, parts in collected.items()}


def load_brightness_ctx(ckpt_path: str, device: torch.device) -> dict:
    """Load the brightness model and its feature-normalization metadata."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BrightnessModel(ckpt["n_feat"], hidden=ckpt.get("hidden", 64)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return {
        "head": model,
        "meta_cols": list(ckpt["meta_cols"]),
        "feat_mean": np.asarray(ckpt["feat_mean"], dtype=np.float64),
        "feat_std": np.asarray(ckpt["feat_std"], dtype=np.float64),
        "directions": list(ckpt["directions"]),
        "device": device,
    }


def boundary_mu_pred(
    x_source: torch.Tensor,
    batch: dict,
    direction: str,
) -> torch.Tensor:
    """Predict target mean by transferring the neighbour IR-RED offset.
    Allows to be used alone without head residual prediction."""
    dims = [1, 2, 3]
    device = x_source.device
    offset = batch["boundary_offset"].to(device).float().view(-1, 1, 1, 1)
    has_boundary = batch.get("has_boundary")
    if has_boundary is not None and not bool(has_boundary.bool().all()):
        raise RuntimeError("RED3 boundary is required for brightness correction")

    mu_source = x_source.mean(dim=dims, keepdim=True)
    sign = 1.0 if direction == "ir2red" else -1.0
    return mu_source + sign * offset


def head_mu_pred(
    x_source: torch.Tensor,
    batch: dict,
    direction: str,
    brightness_ctx: dict,
) -> torch.Tensor:
    """Predict target mean as neighbour offset baseline plus a learned residual."""
    if "boundary_offset" not in batch:
        raise KeyError("Brightness head requires batch['boundary_offset'] from RED3")
    batch_meta = batch["metadata"]
    stats = _image_stats(x_source).detach().cpu().numpy().astype(np.float64)
    offset = batch["boundary_offset"].detach().cpu().numpy().astype(np.float64)
    boundary_stats = batch["boundary_stats"].detach().cpu().numpy().astype(np.float64)
    raw = _raw_features_from_stats(
        stats,
        batch_meta,
        offset,
        boundary_stats,
        direction,
        brightness_ctx["meta_cols"],
    )

    direction_idx = np.full(len(raw), brightness_ctx["directions"].index(direction))
    features = make_model_features(
        raw,
        direction_idx,
        brightness_ctx["feat_mean"],
        brightness_ctx["feat_std"],
    )

    device = brightness_ctx["device"]
    with torch.no_grad():
        prediction = brightness_ctx["head"](
            torch.as_tensor(features, dtype=torch.float32, device=device)
        ).view(-1, 1, 1, 1)

    return boundary_mu_pred(x_source, batch, direction) + prediction


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    """MAE / MSE / R2 of a target-mean prediction against the truth."""
    err = pred - y
    ss_res = float((err**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "mse": float((err**2).mean()),
        "r2": 1.0 - ss_res / max(ss_tot, 1e-12),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a RED3 boundary-residual brightness model."
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/ckpt/brightness",
    )
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--feature_batch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_values = {}
    if args.csv_path:
        config_values["csv_path"] = args.csv_path
    if args.data_path:
        config_values["data_path"] = args.data_path
    cfg = DataConfig(**config_values)

    table = pd.read_csv(cfg.csv_path)
    train_sets, _, val_sets, _, _ = split_by_date(table, 0.7, 0.2, 0.1)
    meta_cols = [col for col in META_COLS if col != "Orbit_number"]

    logger.info(
        "Loading %d train and %d validation sets", len(train_sets), len(val_sets)
    )
    train_dataset = DiffusionDataset(
        cfg,
        allowed_sets=train_sets,
        required_metacols=meta_cols,
        boundary_sets="RED3",
    )
    val_dataset = DiffusionDataset(
        cfg,
        allowed_sets=val_sets,
        required_metacols=meta_cols,
        boundary_sets="RED3",
    )
    train_data = extract_brightness_features(
        train_dataset, meta_cols, args.feature_batch, args.num_workers
    )
    val_data = extract_brightness_features(
        val_dataset, meta_cols, args.feature_batch, args.num_workers
    )

    feat_mean, feat_std = fit_feature_stats(train_data["raw"])
    x_train = torch.from_numpy(
        make_model_features(
            train_data["raw"], train_data["direction"], feat_mean, feat_std
        )
    )
    y_train = torch.from_numpy(train_data["residual"].astype(np.float32))
    x_val = torch.from_numpy(
        make_model_features(val_data["raw"], val_data["direction"], feat_mean, feat_std)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    model = BrightnessModel(x_train.shape[1], hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss()

    n = len(y_train)
    # Treat the zero-residual boundary estimator as the first checkpoint
    model.eval()
    with torch.no_grad():
        initial_residual = model(x_val).cpu().numpy()
    best_mae = float(np.abs(initial_residual - val_data["residual"]).mean())
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    logger.info("boundary baseline: val target-mean MAE = %.6f", best_mae)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val).cpu().numpy()
        val_mae = float(np.abs(val_pred - val_data["residual"]).mean())
        if epoch % 10 == 0 or val_mae < best_mae:
            logger.info("epoch %d: val target-mean MAE = %.6f", epoch, val_mae)
        if val_mae < best_mae - 1e-7:
            best_mae = val_mae
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    # Final model evaluation on the validation set, reporting per-direction metrics
    model.eval()
    with torch.no_grad():
        residual_pred = model(x_val).cpu().numpy()
    pred_target = val_data["base"] + residual_pred
    report = {"val": {}}
    for dir_idx, dir_name in enumerate(DIRECTIONS):
        rows = val_data["direction"] == dir_idx
        model_metrics = metrics(pred_target[rows], val_data["target"][rows])
        base_metrics = metrics(val_data["base"][rows], val_data["target"][rows])
        report["val"][dir_name] = {"model": model_metrics, "boundary": base_metrics}
        logger.info(
            "[%s] model    MAE=%.6f MSE=%.6f R2=%.6f",
            dir_name,
            model_metrics["mae"],
            model_metrics["mse"],
            model_metrics["r2"],
        )
        logger.info(
            "[%s] boundary MAE=%.6f MSE=%.6f R2=%.6f",
            dir_name,
            base_metrics["mae"],
            base_metrics["mse"],
            base_metrics["r2"],
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_feat": x_train.shape[1],
            "hidden": args.hidden,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "stat_names": STAT_NAMES,
            "meta_cols": meta_cols,
            "directions": DIRECTIONS,
        },
        out_dir / "model.pt",
    )
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved model.pt and report.json to %s", out_dir)


if __name__ == "__main__":
    main()
