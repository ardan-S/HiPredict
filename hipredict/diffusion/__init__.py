"""Conditional diffusion (DDPM) and flow-matching (FM) models for RED4 <-> IR10.

Install with the ``diffusion`` extra. See docs/diffusion.md.
"""

from hipredict.diffusion.adapter import HiPredictFMAdapter, build_hipredict_predictor
from hipredict.diffusion.utils import get_model, load_checkpoint

__all__ = [
    "build_hipredict_predictor",
    "HiPredictFMAdapter",
    "get_model",
    "load_checkpoint",
]
