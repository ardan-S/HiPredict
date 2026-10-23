from pathlib import Path

import pandas as pd
import torch

from hipredict.diffusion.config import IR10_GLOBAL_CENTER, IR10_GLOBAL_SCALE

TILE_ROOT = "npy_files_all"


def resolve_record_path(csv_value, csv_path, data_path, marker=TILE_ROOT):
    """Resolve a record 'Path' entry to the tile on disk.
    """
    path = Path(csv_value)
    resolved = path if path.is_absolute() else Path(csv_path).resolve().parent / path
    parts = path.parts
    if marker in parts and not resolved.exists():
        return Path(data_path).joinpath(*parts[parts.index(marker) + 1 :])
    return resolved


def batch_set_ids(batch: dict, batch_size: int, offset: int) -> list:
    """Return a list of sample identifiers for this batch."""
    ids = batch.get("set_id")
    if ids is None:
        return [f"idx{offset + j}" for j in range(batch_size)]
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return [str(i) for i in ids]


def _image_stats(x: torch.Tensor) -> torch.Tensor:
    """Per-image [mean, std, q25, median, q75] over (B, C, H, W), in STAT_NAMES
    order. Used by the brightness head."""
    x = x.flatten(1).float()
    mean = x.mean(dim=1)
    std = x.std(dim=1)
    quants = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device), dim=1)
    return torch.stack([mean, std, quants[0], quants[1], quants[2]], dim=1)


def split_by_date(data_record, train_size=0.7, test_size=0.2, val_size=0.1):
    """
    Split the data into training and testing sets based on the date.

    Args:
        data_record (pd.DataFrame): The dataframe containing the data records, must have 'Set
            and 'Date' columns.
        train_size (float): Proportion of the data to be used for training.
        test_size (float): Proportion of the data to be used for testing.
        val_size (float): Proportion of the data to be used for validation.
    Returns:
        train_sets (list): List of Set ids assigned to the training set.
        test_sets (list): List of Set ids assigned to the testing set.
        val_sets (list): List of Set ids assigned to the validation set.
        train_val_date (pd.Timestamp): The date that separates the training and validation sets.
        val_test_date (pd.Timestamp): The date that separates the validation and testing sets.
    """
    data_record["Date"] = pd.to_datetime(data_record["Date"])
    unique_sets = data_record.groupby("Set")["Date"].first().sort_values()
    total = len(unique_sets)

    train_end = int(total * train_size)
    remaining = total - train_end
    val_count = int(remaining * (val_size / (test_size + val_size)))

    train_sets = unique_sets.iloc[:train_end].index.tolist()
    val_sets = unique_sets.iloc[train_end : train_end + val_count].index.tolist()
    test_sets = unique_sets.iloc[train_end + val_count :].index.tolist()

    train_val_date = unique_sets.iloc[train_end]
    val_test_date = unique_sets.iloc[train_end + val_count]

    return train_sets, test_sets, val_sets, train_val_date, val_test_date


def global_normalize(
    ir10, red4, center=IR10_GLOBAL_CENTER, scale=IR10_GLOBAL_SCALE, clamp=5.0
):
    """Global, shared, invertible normalisation.

    Args:
        ir10, red4: tensors, shape (1, H, W).
        center, scale: global constants; default to the training IR10 median
            and std so normalised IR10 has mean ~ 0 and std ~ 1.
        clamp: symmetric clip in normalised units to tame rare outliers; pass
            None for an exactly invertible (lossless) transform.
    Returns:
        ir_norm, red_norm, norm_stats ({center, scale}) for denormalisation.
    """
    center = torch.as_tensor(float(center))
    scale = torch.as_tensor(float(scale))

    ir_norm = (ir10 - center) / scale
    red_norm = (red4 - center) / scale
    if clamp is not None:
        ir_norm = ir_norm.clamp(-clamp, clamp)
        red_norm = red_norm.clamp(-clamp, clamp)

    norm_stats = {"center": center, "scale": scale}
    return ir_norm, red_norm, norm_stats


def global_denormalize(x, norm_stats):
    """Inverse of global_normalize

    Args:
        x: normalised tensor, shape (1, H, W) or batched (B, 1, H, W).
        norm_stats: dict {center, scale}
    Returns:
        x_denorm with the same shape as x.
    """
    center = torch.as_tensor(norm_stats["center"])
    scale = torch.as_tensor(norm_stats["scale"])

    # Broadcast scalar (per-sample) or (B,) stats over the trailing image dims.
    while center.dim() < x.dim():
        center, scale = center[..., None], scale[..., None]

    return x * scale + center

