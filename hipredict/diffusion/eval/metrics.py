import numpy as np
import torch

# The metrics HiPredict reports, computed with its own per-image definitions.
METRIC_KEYS = ["nmae", "bias", "pwt", "ssim"]


def compute_metrics(
    pred: torch.Tensor, target: torch.Tensor, per_image: bool = False
) -> dict:
    """
    Reconstruction metrics between predicted and target images.

    Uses ``compute_per_img_metrics`` and ``summarize_metric`` from
    ``hipredict.post.run_evals`` (requires the ``postprocess`` extra), so the
    values are directly comparable with HiPredict's evaluation.

    Args:
        pred: predicted images, shape (B, 1, H, W)
        target: target images, shape (B, 1, H, W)
        per_image: also return the per-image values of every metric

    Returns:
        The batch means, or (means, per-image values) when per_image.
    """
    from hipredict.post.run_evals import compute_per_img_metrics, summarize_metric

    per_image_metrics = compute_per_img_metrics(target.float(), pred.float())
    metrics = {k: summarize_metric(per_image_metrics[k]) for k in METRIC_KEYS}
    if not per_image:
        return metrics
    return metrics, {k: np.asarray(per_image_metrics[k]) for k in METRIC_KEYS}
