import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hipredict.data.sample_build import ConstantNormalizer
from hipredict.diffusion.config import NormalizationConfig
from hipredict.predict.dataset import BaseHiRISEDataset

from .utils import global_normalize, resolve_record_path

# Observation metadata columns available to the brightness head.
META_COLS = [
    "TDI",
    "IEA_PWS_temp",
    "Incidence_angle",
    "Orbit_number",
    "Local_time",
    "Phase_angle",
    "Solar_longitude",
    "Solar_distance",
    "FPA_negY_temp",
]
STAT_NAMES = ["mean", "std", "q25", "median", "q75"]

logger = logging.getLogger(__name__)


def red3_boundary_stats(
    red3: torch.Tensor, normalization: NormalizationConfig = None
) -> torch.Tensor:
    """
    Statistics of the RED3 context that overlaps RED4, in normalised units.

    ``red3`` holds the four context channels [BG, IR, RED, RED4] in its last
    dimension. Returns [mean, std, q25, median, q75] of IR, RED and RED - IR
    (15 values); index 13 is the median RED - IR offset.
    """
    red = red3[..., 2:3]
    ir = red3[..., 1:2]
    normalization = normalization or NormalizationConfig()
    red_norm, ir_norm, _ = global_normalize(red, ir, **normalization.model_dump())
    stats = []
    for values in (ir_norm, red_norm, red_norm - ir_norm):
        values = values.numpy().reshape(-1)
        stats.extend(
            [
                np.nanmean(values),
                np.nanstd(values),
                *np.nanquantile(values, [0.25, 0.5, 0.75]),
            ]
        )
    stats = torch.tensor(stats, dtype=torch.float32)
    if not bool(torch.isfinite(stats).all()):
        raise ValueError("RED3 statistics contain non-finite values")
    return stats


class DiffusionDataset(BaseHiRISEDataset):
    """Paired RED4 and IR10 patches, with the RED3 boundary when it is requested.

    Every set must hold one row per required CCD (RED4 and IR10, plus RED3 when
    ``boundary_sets`` is given). Sets missing a required CCD are excluded and
    counted in the log; duplicate CCD rows are an error.
    """

    def __init__(
        self, config, allowed_sets=None, required_metacols=None, boundary_sets=None
    ):
        self.config = config
        self.allowed_sets = set(allowed_sets) if allowed_sets is not None else None
        self.required_metacols = required_metacols
        self.boundary_sets = list(boundary_sets) if boundary_sets is not None else None
        self.required_ccds = ("RED4", "IR10") + (("RED3",) if self.boundary_sets else ())
        self.records, self.set_records = self.load_data(self.allowed_sets)
        self.sets = list(self.set_records.keys())
        normalization = config.normalization
        super().__init__(
            self.records,
            data_root=Path(config.csv_path).resolve().parent,
            norm_mode=None,
            normalizer=ConstantNormalizer(normalization.center, normalization.scale),
        )

    def load_data(self, allowed_sets):
        csv_path = self.config.csv_path
        df = pd.read_csv(csv_path)

        if allowed_sets is not None:
            df = df[df["Set"].isin(allowed_sets)]

        records = df[df["CCD"].isin(self.required_ccds)].reset_index(drop=True).copy()
        duplicates = records[records.duplicated(["Set", "CCD"], keep=False)]
        if not duplicates.empty:
            sets = sorted(duplicates["Set"].unique().tolist())
            raise ValueError(
                f"{csv_path} has duplicate CCD rows in {len(sets)} sets, e.g. {sets[:5]}"
            )
        records["resolved_path"] = records["Path"].apply(self._resolve_data_path)

        set_records = {}
        for set_id, group in records.groupby("Set", sort=False):
            ccd_records = {row["CCD"]: row for _, row in group.iterrows()}
            if all(ccd in ccd_records for ccd in self.required_ccds):
                set_records[set_id] = ccd_records

        requested = df["Set"].nunique()
        excluded = requested - len(set_records)
        if excluded:
            logger.warning(
                "Excluded %d of %d sets missing one of %s",
                excluded,
                requested,
                ", ".join(self.required_ccds),
            )
        logger.info("Loaded %d sets from %s", len(set_records), csv_path)
        return records, set_records

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        set_id = self.sets[idx]
        ccd_records = self.set_records[set_id]

        images = self.load_ccd_arrays(set_id, {
            ccd: Path(ccd_records[ccd]["resolved_path"]).resolve()
            for ccd in ("IR10", "RED4")
        })
        red4 = torch.from_numpy(images["RED4"].astype(np.float32)).unsqueeze(0)
        ir10 = torch.from_numpy(images["IR10"].astype(np.float32)).unsqueeze(0)

        normalization = self.config.normalization
        center, scale = self.normalizer.stats()
        ir_norm, red_norm, norm_stats = global_normalize(
            ir10, red4, center=center, scale=scale, clamp=normalization.clamp
        )

        # set_id lets consumers key results by observation rather than by position
        out = dict(
            red4=red_norm, ir10=ir_norm, stats=norm_stats, set_id=set_id,
            raw_red4=red4, raw_ir10=ir10,
        )

        # Per-CCD metadata from the CSV rows backing this set (NaN preserved).
        if self.required_metacols:
            out["metadata"] = {
                f"{prefix}_{col}": float(ccd_records[ccd][col])
                for prefix, ccd in (("ir", "IR10"), ("red", "RED4"))
                for col in self.required_metacols
            }

        if self.boundary_sets:
            # RED3 context, shape (H, 32, 4): [BG, IR, RED, RED4]
            try:
                boundary = self.load_ccd_arrays(set_id, {
                    "RED3": Path(ccd_records["RED3"]["resolved_path"]).resolve()
                })
                red3 = torch.from_numpy(boundary["RED3"].astype(np.float32))
                boundary_stats = red3_boundary_stats(red3, normalization)
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Cannot read the RED3 boundary of set {set_id}") from error
            out["boundary_offset"] = boundary_stats[13]  # RED3 median(RED-IR)
            out["boundary_stats"] = boundary_stats
            out["has_boundary"] = 1.0

        return out

    def _resolve_data_path(self, csv_value):
        return resolve_record_path(
            csv_value, self.config.csv_path, self.config.data_path
        )


def get_loader(
    dataset,
    batch_size,
    shuffle=True,
    pin_memory=True,
    prefetch_factor=2,
    num_workers=4,
    collate_fn=None,
    worker_init_fn=None,
    persistent_workers=True,
    drop_last=False,
):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    # prefetch_factor / persistent_workers are only valid with worker processes.
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **kwargs)
