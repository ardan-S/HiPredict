"""Run diffusion flow-matching checkpoints through the HiPredict inference API.

The adapter gives an FM model HiPredict's ``model(x_band, x_neigh, x_meta)``
interface, so ``HiPredict.load`` and ``HiPredict.predict`` work unchanged on
patches and full strips; inputs larger than the training patches are tiled. The HiPredict API predicts RED4, so only ``ir2red`` is
supported here.
"""

from typing import Union

import torch
from torch import nn

from hipredict.data.sample_build import Normalizer
from hipredict.diffusion.config import InferenceConfig, NormalizationConfig
from hipredict.diffusion.inference.samplers import _inference_fm
from hipredict.diffusion.inference.tiling import (
    TILE_OVERLAP,
    TILE_SIZE,
    context_rows,
    predict_tiled,
)
from hipredict.diffusion.models.utils import compute_edge, load_edge_model
from hipredict.diffusion.utils import load_checkpoint


class DiffusionGlobalNormalizer(Normalizer):
    """The fixed affine a diffusion checkpoint was trained with."""

    def __init__(self, normalization: NormalizationConfig = None):
        self.normalization = normalization or NormalizationConfig()

    def stats(self, bands, neighbours=None):
        return self.normalization.center, self.normalization.scale


class HiPredictFMAdapter(nn.Module):
    """Expose an FM checkpoint as ``model(x_band, x_neigh, x_meta)``.

    Images larger than ``tile_size`` are predicted as overlapping tiles (see
    ``inference.tiling``). Each tile's RED4 mean is set from the RED3 boundary
    offset, after every ODE step when ``dc_anchor`` is enabled and once on the
    final prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg_model,
        *,
        ode_steps: int = 10,
        solver: str = "euler",
        dc_anchor: bool = True,
        edge_model: nn.Module = None,
        tile_size: int = TILE_SIZE,
        tile_overlap: int = TILE_OVERLAP,
        tile_batch_size: int = 16,
    ):
        super().__init__()
        self.model = model
        self.mode = cfg_model.Mode
        self.edge_mode = cfg_model.edge_mode
        self.edge_model = edge_model
        clamp = cfg_model.normalization.clamp
        self.clamp = float(clamp) if clamp is not None else float("inf")
        self.dc_anchor = bool(dc_anchor)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_batch_size = tile_batch_size
        self.inference_config = InferenceConfig(
            direction="ir2red",
            FM_ODE_steps=ode_steps,
            FM_solver=solver,
            dc_anchor=self.dc_anchor,
        )

    def forward(self, x_band, x_neigh, x_meta=None):
        x_source = x_band[:, 0:1].clamp(-self.clamp, self.clamp)
        # Both patch and strip layouts place RED3 IR and RED at indices 4 and 5.
        boundary = x_neigh[:, 4:6].clamp(-self.clamp, self.clamp)
        height, width = x_source.shape[-2:]

        predictions = []
        for source, context in zip(x_source, boundary):

            def predict(windows, source=source, context=context):
                tiles = torch.stack([source[:, y:y + h, x:x + w] for y, x, h, w in windows])
                contexts = [
                    context[:, context_rows(context.shape[1], height, y, h)]
                    for y, _, h, _ in windows
                ]
                return self._predict_tiles(tiles, contexts)[:, 0]

            predictions.append(
                predict_tiled(
                    predict,
                    height,
                    width,
                    tile=self.tile_size,
                    overlap=self.tile_overlap,
                    batch_size=self.tile_batch_size,
                )
            )
        return torch.stack(predictions)[:, None].to(x_band.device)

    def _predict_tiles(self, x_source, contexts):
        """Predict (N, 1, h, w) source tiles, each with its [IR, RED] RED3 context."""
        boundary_offset = torch.stack(
            [torch.nanmedian(context[1] - context[0]) for context in contexts]
        ).view(-1, 1, 1, 1)
        if not bool(torch.isfinite(boundary_offset).all()):
            raise ValueError(
                "RED3 boundary channels produced a non-finite median offset"
            )

        mu_pred = x_source.mean(dim=(1, 2, 3), keepdim=True) + boundary_offset
        edge = compute_edge(x_source, self.edge_mode, self.edge_model)
        mu_anchor = mu_pred if self.dc_anchor else None

        prediction = _inference_fm(
            self.model,
            x_source,
            edge,
            self.mode,
            x_source.device,
            self.inference_config,
            mu_anchor=mu_anchor,
        )

        return prediction - prediction.mean(dim=(1, 2, 3), keepdim=True) + mu_pred


def hipredict_adapter_config():
    """Return the minimal HiPredict config needed by the FM adapter."""
    return {
        "data": {
            "meta_cols": [],
            "meta_ccd": "RED3",
            "normalisation_mode": None,
        },
        # BidirectionalFM has five downsampling stages, so its stride is 32.
        "architecture": {"args": {"depth": 5}},
    }


def build_hipredict_predictor(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
    *,
    ode_steps: int = 10,
    solver: str = "euler",
    tile_size: int = TILE_SIZE,
    tile_overlap: int = TILE_OVERLAP,
    verbose: bool = True,
):
    """Load an FM checkpoint as a HiPredict predictor for RED4.

    Inputs larger than ``tile_size`` (patches or full strips) are predicted as
    tiles overlapping by ``tile_overlap`` pixels.

    Returns ``(predictor, cfg_model, checkpoint_step)``. Use the predictor like
    one built from ``get_model``: ``predictor.predict(predictor.load(...))``.
    """
    from hipredict.inference.hipredict import HiPredict

    model, cfg_model, _, checkpoint_step = load_checkpoint(checkpoint_path, device)
    if cfg_model.direction not in ("bidirectional", "ir2red"):
        raise ValueError(
            f"Checkpoint direction {cfg_model.direction!r} cannot predict RED4"
        )
    if "DDPM" in cfg_model.type:
        raise ValueError("The HiPredict adapter supports flow-matching checkpoints only")
    adapter = HiPredictFMAdapter(
        model,
        cfg_model,
        ode_steps=ode_steps,
        solver=solver,
        dc_anchor=True,
        edge_model=load_edge_model(cfg_model.edge_mode, device),
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    predictor = HiPredict(
        adapter,
        hipredict_adapter_config(),
        device=device,
        normalizer=DiffusionGlobalNormalizer(cfg_model.normalization),
        verbose=verbose,
    )
    return predictor, cfg_model, checkpoint_step
