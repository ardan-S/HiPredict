import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import ValidationError
import torch.nn.functional as F

from hipredict.diffusion.config import DataConfig, InferenceConfig, ModelConfig
from hipredict.diffusion.dataset import (
    DiffusionDataset,
    get_loader,
    global_denormalize,
    global_normalize,
)
from hipredict.diffusion.dataset.dataset import red3_boundary_stats
from hipredict.diffusion.dataset.utils import split_by_date
from hipredict.diffusion.models.brightness import boundary_mu_pred, head_mu_pred, load_brightness_ctx
from hipredict.diffusion.models.utils import (
    DDPMScheduler,
    compute_edge,
    load_edge_model,
)
from hipredict.diffusion.utils import load_checkpoint, set_seed

from .samplers import _inference_ddpm, _inference_fm
from .tiling import TILE_OVERLAP, TILE_SIZE, context_rows, predict_tiled

logger = logging.getLogger(__name__)

SOURCE_CCD = {"ir2red": "IR10", "red2ir": "RED4"}
TARGET_CCD = {"ir2red": "RED4", "red2ir": "IR10"}

# The U-Nets downsample five times, so inputs must be a multiple of 32.
MODEL_STRIDE = 32


def inference_batch(
    batch: dict,
    model: torch.nn.Module,
    model_type: str,
    edge_model: torch.nn.Module,
    scheduler: DDPMScheduler,
    device: torch.device,
    cfg: ModelConfig,
    cfg_inference: InferenceConfig,
    dc_correct: str = "none",
    cond_fit: dict = None,
    brightness_ctx: dict = None,
):
    """
    Perform a single inference step for the given batch.

    Args:
        batch: A dictionary containing the input data.
        model: The diffusion model.
        model_type: Type of the diffusion model, e.g. "BidirectionalFM".
        scheduler: The noise scheduler (only used by DDPM).
        device: The device to run inference on (CPU or GPU).
        cfg: Configuration parameters for the model.
        cfg_inference: Configuration parameters for inference.
        dc_correct: Method for post-hoc brightness (DC) correction
        cond_fit: Dictionary containing the linear fit parameters for the conditional mean correction (used if dc_correct is "cond").
        brightness_ctx: Dictionary containing the RED3 residual brightness head and feature statistics.
    Returns:
        dict with pred_norm, target_norm, pred_denorm, and target_denorm tensors.
    """
    ir = batch["ir10"].to(device)
    red = batch["red4"].to(device)
    direction = cfg_inference.direction
    if direction == "red2ir":
        x_source, x_target = red, ir
    elif direction == "ir2red":
        x_source, x_target = ir, red
    else:
        raise ValueError(
            f"Unknown direction: {direction}. Must be 'ir2red' or 'red2ir'."
        )

    edge = compute_edge(x_source, cfg.edge_mode, edge_model)
    # Per-image target-mean estimate, used for the post-hoc DC shift and
    # (if cfg_inference.dc_anchor) for per-step anchoring inside the FM ODE.
    # oracle: the true target mean (upper bound) for reference only
    # cond: a*mean(source)+b from the train cond_mean fit.
    # head: the learned brightness head's prediction.
    dims = [1, 2, 3]
    mu_pred = None
    if dc_correct != "none":
        if dc_correct == "oracle":
            mu_pred = x_target.mean(dim=dims, keepdim=True)
        elif dc_correct == "cond":
            if cond_fit is None:
                raise ValueError("dc_correct='cond' requires cond_fit")
            mu_src = x_source.mean(dim=dims, keepdim=True)
            mu_pred = cond_fit["a"] * mu_src + cond_fit["b"]
        elif dc_correct == "head":
            if brightness_ctx is None:
                raise ValueError("dc_correct='head' requires brightness_ctx")
            mu_pred = head_mu_pred(
                x_source, batch, cfg_inference.direction, brightness_ctx
            )
        elif dc_correct == "boundary":
            mu_pred = boundary_mu_pred(x_source, batch, cfg_inference.direction)
        else:
            raise ValueError(f"Unknown dc_correct: {dc_correct}")

    # Perform inference based on the model type
    if "DDPM" in model_type:
        pred_norm = _inference_ddpm(model, scheduler, x_source, edge, device, cfg_inference)
    elif "FM" in model_type:
        mode = cfg.Mode
        mu_anchor = mu_pred if cfg_inference.dc_anchor else None
        pred_norm = _inference_fm(
            model, x_source, edge, mode, device, cfg_inference, mu_anchor
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if mu_pred is not None:
        pred_norm = pred_norm - pred_norm.mean(dim=dims, keepdim=True) + mu_pred

    stats = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch["stats"].items()
    }
    pred_denorm = global_denormalize(pred_norm, stats)
    raw_target = batch.get(f"raw_{TARGET_CCD[direction].lower()}")
    target_denorm = (
        raw_target.to(device)
        if raw_target is not None
        else global_denormalize(x_target, stats)
    )

    return {
        "pred_norm": pred_norm,
        "target_norm": x_target,
        "pred_denorm": pred_denorm,
        "target_denorm": target_denorm,
    }


def prediction_name(input_path: Path, direction: str) -> str:
    """Name a prediction after its band: X_IR10.npy -> X_RED4.npy for ir2red."""
    stem = input_path.stem
    source, target = SOURCE_CCD[direction], TARGET_CCD[direction]
    if stem.endswith(f"_{source}"):
        stem = stem[: -len(source)] + target
    else:
        stem = f"{stem}_{target}"
    return f"{stem}.npy"


def load_single_image(input_path, red3_path=None):
    """
    Load a source image, padded to a multiple of MODEL_STRIDE, and its RED3 context.

    Returns the padded image as a (1, 1, H, W) tensor, the (rows, columns)
    padding to crop from the prediction, and the context (or None).
    """
    image = np.load(input_path).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"{input_path}: expected a 2-D image, got shape {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"{input_path}: the image contains non-finite values")

    height, width = image.shape
    pad = ((-height) % MODEL_STRIDE, (-width) % MODEL_STRIDE)
    source = torch.from_numpy(image)[None, None]
    if any(pad):
        source = F.pad(source, (0, pad[1], 0, pad[0]), mode="reflect")

    red3 = None
    if red3_path is not None:
        red3 = torch.from_numpy(np.load(red3_path).astype(np.float32))
        if red3.ndim != 3 or red3.shape[-1] < 3:
            raise ValueError(
                f"{red3_path}: expected RED3 context of shape (H, W, 4), got {tuple(red3.shape)}"
            )
    return source, pad, red3


def run_single_image_inference(
    args,
    cfg_inference: InferenceConfig,
    cfg_model: ModelConfig,
    model: torch.nn.Module,
    model_type: str,
    edge_model: torch.nn.Module,
    scheduler: DDPMScheduler,
    device: torch.device,
) -> None:
    """Predict the target band for each --input_path and save it under output_dir.

    Images larger than --tile_size are predicted as overlapping tiles; each
    tile's brightness comes from the RED3 context rows around it.
    """
    output_dir = Path(cfg_inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    red3_paths = args.red3_path or [None] * len(args.input_path)
    normalization = cfg_model.normalization

    for input_path, red3_path in zip(args.input_path, red3_paths):
        input_path = Path(input_path)
        source, (pad_h, pad_w), red3 = load_single_image(input_path, red3_path)
        source_norm, _, stats = global_normalize(source, source, **normalization.model_dump())
        height, width = source.shape[-2:]
        image_height = height - pad_h

        def predict(windows):
            tiles = torch.cat([source_norm[..., y:y + h, x:x + w] for y, x, h, w in windows])
            batch = {
                "ir10": tiles if cfg_inference.direction == "ir2red" else torch.zeros_like(tiles),
                "red4": tiles if cfg_inference.direction == "red2ir" else torch.zeros_like(tiles),
                "stats": {k: v.expand(len(windows)) for k, v in stats.items()},
            }
            if red3 is not None:
                boundary_stats = torch.stack([
                    red3_boundary_stats(
                        red3[context_rows(red3.shape[0], image_height, y, h)], normalization
                    )
                    for y, _, h, _ in windows
                ])
                batch["boundary_offset"] = boundary_stats[:, 13]
                batch["boundary_stats"] = boundary_stats
                batch["has_boundary"] = torch.ones(len(windows))
            outputs = inference_batch(
                batch,
                model,
                model_type,
                edge_model,
                scheduler,
                device,
                cfg_model,
                cfg_inference,
                dc_correct=args.dc_correct,
            )
            return outputs["pred_denorm"][:, 0]

        prediction = predict_tiled(
            predict,
            height,
            width,
            tile=args.tile_size,
            overlap=args.tile_overlap,
            batch_size=cfg_inference.batch_size,
        ).numpy()
        prediction = prediction[: height - pad_h, : width - pad_w]

        output_path = output_dir / prediction_name(input_path, cfg_inference.direction)
        np.save(output_path, prediction.astype(np.float32))
        logger.info("Saved %s", output_path)


def _relative_tile_path(csv_value: str) -> Path:
    """Return the tile path relative to its tile root.
    """
    path = Path(csv_value)
    parts = path.parts
    if "npy_files_all" in parts:
        return Path(*parts[parts.index("npy_files_all") + 1 :])
    if not path.is_absolute() and len(parts) > 1:
        return Path(*parts[1:])
    raise ValueError(
        f"Path is neither under 'npy_files_all' nor relative to the CSV: {csv_value}"
    )


def run_dataset_inference(
    args,
    cfg_inference: InferenceConfig,
    cfg_model: ModelConfig,
    model: torch.nn.Module,
    model_type: str,
    edge_model: torch.nn.Module,
    scheduler: DDPMScheduler,
    device: torch.device,
) -> None:
    """
    Reconstruct the canonical test split and save each denormalized prediction
    under output_dir at the path of the target tile it replaces, so the output
    tree mirrors the input record's layout.
    """
    brightness_ctx = None
    required_metacols = None
    if args.dc_correct == "head":
        brightness_ctx = load_brightness_ctx(args.head_ckpt, device)
        required_metacols = brightness_ctx["meta_cols"]

    dataframe = pd.read_csv(args.test_csv)
    _, test_sets, _, _, _ = split_by_date(dataframe, 0.7, 0.2, 0.1)
    cfg_dataset = DataConfig(
        csv_path=args.test_csv, normalization=cfg_model.normalization
    )
    if args.data_root:
        cfg_dataset.data_path = args.data_root
    dataset = DiffusionDataset(
        cfg_dataset,
        allowed_sets=test_sets,
        required_metacols=required_metacols,
        boundary_sets="RED3" if args.dc_correct in ("head", "boundary") else None,
    )
    loader = get_loader(
        dataset,
        batch_size=cfg_inference.batch_size,
        num_workers=cfg_inference.num_workers,
        shuffle=False,
    )

    target_ccd = TARGET_CCD[cfg_inference.direction]
    output_root = Path(cfg_inference.output_dir)
    processed = 0
    for batch in loader:
        outputs = inference_batch(
            batch,
            model,
            model_type,
            edge_model,
            scheduler,
            device,
            cfg_model,
            cfg_inference,
            dc_correct=args.dc_correct,
            brightness_ctx=brightness_ctx,
        )
        preds = outputs["pred_denorm"].detach().cpu().numpy().astype(np.float32)
        for offset in range(preds.shape[0]):
            set_id = dataset.sets[processed + offset]
            csv_value = dataset.set_records[set_id][target_ccd]["Path"]
            tile_path = output_root / _relative_tile_path(csv_value)
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(tile_path, preds[offset, 0])
        processed += preds.shape[0]
        logger.info("Reconstructed %d/%d test tiles", processed, len(dataset))

    logger.info("Saved %d reconstructions under %s", processed, output_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--direction",
        type=str,
        default="ir2red",
        choices=["ir2red", "red2ir"],
    )

    # inputs — either source images or the canonical test split of a record CSV
    parser.add_argument(
        "--input_path",
        type=str,
        nargs="+",
        default=None,
        help="Source .npy image(s): IR10 for ir2red, RED4 for red2ir",
    )
    parser.add_argument(
        "--red3_path",
        type=str,
        nargs="+",
        default=None,
        help="RED3 context .npy for each --input_path (needed for --dc_correct boundary)",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=TILE_SIZE,
        help="Images larger than this are predicted as overlapping tiles (multiple of 32)",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=TILE_OVERLAP,
        help="Overlap between neighbouring tiles, in pixels",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="HiRISE record CSV; reconstructs the canonical test split and "
        "mirrors the record's tile layout under --output_dir",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root that 'npy_files_all' record paths are re-rooted under",
    )
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--dc_correct",
        type=str,
        default="boundary",
        choices=["none", "head", "boundary", "oracle"],
        help="Post-hoc brightness correction (oracle uses the true target mean; "
        "reference upper bound only)",
    )
    parser.add_argument(
        "--no_dc_anchor",
        action="store_true",
        help="Apply the dc_correct mean only to the final prediction, not after "
        "every FM ODE step",
    )
    parser.add_argument(
        "--head_ckpt",
        type=str,
        default="outputs/ckpt/brightness/model.pt",
        help="Brightness-head checkpoint (used when --dc_correct is head)",
    )

    # outputs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference/",
        help="Directory to save inference results",
    )

    parser.add_argument(
        "--FM_ODE_steps",
        type=int,
        default=10,
        help="Number of ODE steps for FM inference",
    )
    parser.add_argument(
        "--FM_solver",
        type=str,
        default=None,
        choices=["euler", "rk4"],
        help="ODE solver for FM inference (default: euler).",
    )

    args = parser.parse_args()

    if bool(args.input_path) == bool(args.test_csv):
        parser.error("Provide exactly one of --input_path or --test_csv.")
    if args.input_path:
        if args.dc_correct not in ("none", "boundary"):
            parser.error("Single-image inference supports --dc_correct none or boundary")
        if args.dc_correct == "boundary" and not args.red3_path:
            parser.error(
                "--dc_correct boundary (the default) needs a --red3_path for each "
                "input; pass --dc_correct none to skip brightness correction"
            )
        if args.red3_path and len(args.red3_path) != len(args.input_path):
            parser.error("Give one --red3_path per --input_path")
        if args.tile_size <= 0 or args.tile_size % MODEL_STRIDE:
            parser.error(f"--tile_size must be a positive multiple of {MODEL_STRIDE}")
        if not 0 <= args.tile_overlap < args.tile_size:
            parser.error("--tile_overlap must be at least 0 and smaller than --tile_size")
    elif args.red3_path:
        parser.error("--red3_path is only used with --input_path")

    set_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        # Inference configuration
        cfg_inference = InferenceConfig()
        if args.batch_size:
            cfg_inference.batch_size = args.batch_size
        if args.ckpt_path:
            cfg_inference.ckpt_path = args.ckpt_path
        if args.output_dir:
            cfg_inference.output_dir = args.output_dir
        if args.direction:
            cfg_inference.direction = args.direction
        if args.FM_ODE_steps is not None:
            cfg_inference.FM_ODE_steps = args.FM_ODE_steps
        if args.FM_solver:
            cfg_inference.FM_solver = args.FM_solver
        if args.num_workers is not None:
            cfg_inference.num_workers = args.num_workers
        cfg_inference.dc_anchor = args.dc_correct != "none" and not args.no_dc_anchor
    except ValidationError as error:
        parser.error(str(error))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and its configurations
    model, cfg_model, _, _ = load_checkpoint(cfg_inference.ckpt_path, device)
    model_type = cfg_model.type

    if "DDPM" in model_type:
        cfg_inference.FM_ODE_steps = None  # not used for DDPM

    if cfg_model.direction not in ("bidirectional", cfg_inference.direction):
        raise ValueError(
            f"Checkpoint direction ({cfg_model.direction}) does not support "
            f"inference direction ({cfg_inference.direction})"
        )

    logger.info(f"Loading model from checkpoint: {cfg_inference.ckpt_path}")
    logger.info(f"Model configuration: {cfg_model.model_dump()}")
    logger.info(f"Inference configuration: {cfg_inference.model_dump()}")
    logger.info(f"Using device: {device}")

    scheduler = None
    if "DDPM" in model_type:
        scheduler = DDPMScheduler(
            num_timesteps=cfg_model.num_timesteps,
            beta_start=cfg_model.beta_start,
            beta_end=cfg_model.beta_end,
        ).to(device)

    edge_model = load_edge_model(cfg_model.edge_mode, device)

    runner = run_dataset_inference if args.test_csv else run_single_image_inference
    runner(
        args,
        cfg_inference,
        cfg_model,
        model,
        model_type,
        edge_model,
        scheduler,
        device,
    )


if __name__ == "__main__":
    main()
