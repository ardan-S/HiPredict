"""Overlapping-tile prediction for images larger than the training patches.

The FM networks attend over every position of their input, so memory grows with
the square of the image area, and they were trained on 256 x 256 patches. Larger
images are therefore predicted as overlapping 256 x 256 tiles whose overlaps are
blended with linear ramps.
"""

import torch

# Training patch size, and the RED3 context rows each training patch was paired with.
TILE_SIZE = 256
TILE_OVERLAP = 64
BOUNDARY_CONTEXT_ROWS = 1024


def tile_starts(length: int, tile: int, overlap: int) -> list:
    """Start offsets of tiles of size ``tile`` covering ``length`` with at least ``overlap``."""
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _ramp(length: int, overlap: int) -> torch.Tensor:
    weight = torch.ones(length)
    ramp_length = min(overlap, length // 2)
    if ramp_length > 0:
        ramp = torch.arange(1, ramp_length + 1, dtype=torch.float32) / (ramp_length + 1)
        weight[:ramp_length] = ramp
        weight[-ramp_length:] = ramp.flip(0)
    return weight


def context_rows(context_height: int, image_height: int, y0: int, height: int) -> slice:
    """Rows of a RED3 context to use for the tile starting at row ``y0``.

    A context with the image's height is row-aligned (full strips): the tile
    uses the BOUNDARY_CONTEXT_ROWS rows centred on it, as in training. Any other
    context (a patch record's fixed context array) is used whole.
    """
    if context_height != image_height or context_height <= BOUNDARY_CONTEXT_ROWS:
        return slice(None)
    start = int(y0 + height / 2 - BOUNDARY_CONTEXT_ROWS / 2)
    start = min(max(start, 0), context_height - BOUNDARY_CONTEXT_ROWS)
    return slice(start, start + BOUNDARY_CONTEXT_ROWS)


def predict_tiled(
    predict,
    height: int,
    width: int,
    tile: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
    batch_size: int = 16,
) -> torch.Tensor:
    """Predict a (height, width) image tile by tile and blend the overlaps.

    ``predict(windows)`` receives a list of ``(y0, x0, h, w)`` windows of equal
    size and returns their predictions as a tensor of shape (N, h, w). Height
    and width must be multiples of the model stride; tiles are clipped to the
    image when it is smaller than ``tile``.
    """
    if not 0 <= overlap < tile:
        raise ValueError(f"tile overlap must be in [0, {tile}), got {overlap}")
    tile_h, tile_w = min(tile, height), min(tile, width)
    windows = [
        (y0, x0, tile_h, tile_w)
        for y0 in tile_starts(height, tile_h, overlap)
        for x0 in tile_starts(width, tile_w, overlap)
    ]
    weight = _ramp(tile_h, overlap)[:, None] * _ramp(tile_w, overlap)[None, :]

    total = torch.zeros(height, width)
    norm = torch.zeros(height, width)
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        predictions = predict(chunk).detach().float().cpu()
        for (y0, x0, h, w), prediction in zip(chunk, predictions):
            total[y0 : y0 + h, x0 : x0 + w] += weight * prediction
            norm[y0 : y0 + h, x0 : x0 + w] += weight
    return total / norm
