import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

from hipredict.data.sample_build import (
    SceneRobust,
    assemble_patch_inputs_np,
    assemble_patch_neigh_from_stacked,
)
from hipredict.data.normalization import DEFAULT_NORMALIZATION_MODE, normalization_kwargs_from_config

OBS_CENTER_COLUMN = "Obs_center"
OBS_SCALE_COLUMN = "Obs_scale"
PATCH_X_COLUMN = "Patch_x"


def _infer_observation_width_from_binning(df):
    if "Binning" not in df.columns:
        return 1024.0

    binning = pd.to_numeric(df["Binning"], errors="coerce").dropna()
    if binning.empty:
        return 1024.0
    if (binning == 1).any():
        return 2048.0
    if (binning == 2).all():
        return 1024.0

    raise ValueError(f"Unexpected binning values for set: {sorted(binning.unique().tolist())}")


def _load_observation_norm_from_record(df, dtype):
    if OBS_CENTER_COLUMN not in df.columns or OBS_SCALE_COLUMN not in df.columns:
        return None

    center_vals = pd.to_numeric(df[OBS_CENTER_COLUMN], errors="coerce").dropna().unique()
    scale_vals = pd.to_numeric(df[OBS_SCALE_COLUMN], errors="coerce").dropna().unique()
    if len(center_vals) != 1 or len(scale_vals) != 1:
        return None

    a_val = float(center_vals[0])
    s_val = float(scale_vals[0])
    if not np.isfinite(a_val) or not np.isfinite(s_val) or s_val <= 0:
        return None
    return torch.tensor(a_val, dtype=dtype), torch.tensor(s_val, dtype=dtype)


class BaseHiRISEDataset(Dataset):
    """Shared scaffolding for HiRISE map-style datasets over patch sets.

    Holds the pieces that training and inference genuinely share -- the data
    record, set indexing, root resolution, and the (injected) normalization
    strategy -- while leaving the concrete per-sample loading/assembly to
    subclasses, because those differ meaningfully (torch vs numpy, normalize-now
    vs deferred, data_root-relative vs record-absolute paths).

    The ``normalizer`` is the pluggable hook: pass a custom
    :class:`hipredict.data.sample_build.Normalizer` to override the built-in.
    For ``scene_robust`` it defaults to ``SceneRobust``; for ``global_robust`` and
    the identity/None case it stays ``None`` and the subclass keeps its existing
    inline scalar handling (preserving prior behaviour and error timing).
    """

    band_ccds = ['IR10', 'BG12']
    neighbour_ccds = ['RED5', 'RED3']

    def __init__(self, data_record, data_root=None, norm_mode=DEFAULT_NORMALIZATION_MODE,
                 global_center=None, global_scale=None, normalizer=None, meta_cols=None):
        if norm_mode == "global_robust" and (global_center is None or global_scale is None):
            defaults = normalization_kwargs_from_config({}, norm_mode=norm_mode)
            global_center = defaults["global_center"] if global_center is None else global_center
            global_scale = defaults["global_scale"] if global_scale is None else global_scale
        self.data_record = data_record
        self.data_root = data_root if data_root is not None else data_record.attrs.get("record_dir")
        self.meta_cols = meta_cols
        self.norm_mode = norm_mode
        self.global_center = global_center
        self.global_scale = global_scale
        self.unique_sets = sorted(self.data_record['Set'].unique())

        if normalizer is not None:
            self.normalizer = normalizer
        elif norm_mode == "scene_robust":
            self.normalizer = SceneRobust(include_red_means=False)
        else:
            self.normalizer = None

    def __len__(self):
        return self.data_record['Set'].nunique()

    def _root(self):
        return self.data_root if self.data_root is not None else os.getcwd()

    def load_ccd_arrays(self, set_id, paths):
        """Read raw CCD arrays relative to the dataset's record directory."""
        arrays = {}
        for ccd, value in paths.items():
            path = Path(self._root()) / Path(value)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing patch file for set {set_id}, CCD {ccd}: {path}"
                )
            arrays[ccd] = np.load(path)
        return arrays


class HiRISEDataset(BaseHiRISEDataset):
    """
    A custom Dataset for loading images.
    A dir contains several sets of images.
    Eg:
    - npy_files/                                    # Root directory
        - ESP_026721_1635/                          # Observation directory
            - ESP_026721_1635_0_RED5.npy            # Image array - shape either (32, 1024) or (256, 256)
            - ESP_026721_1635_0_RED4.npy
            - ESP_026721_1635_0_RED3.npy
            - ESP_026721_1635_0_IR10.npy
            - ESP_026721_1635_0_BG12.npy

            - ESP_026721_1635_1_RED5.npy
            - ESP_026721_1635_1_RED4.npy
            - ESP_026721_1635_1_RED3.npy
            - ESP_026721_1635_1_IR10.npy
            - ESP_026721_1635_1_BG12.npy

            - etc

        - ESP_011290_1800/
            - etc

    Args:
        data_record (pd.DataFrame): Data record containing the paths to the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, data_record, transform=None, meta_cols=None, scale_pix=False, norm_mode=DEFAULT_NORMALIZATION_MODE, global_center=None, global_scale=None, pix_min=None, pix_max=None, meta_scaler=None, data_root=None, normalizer=None):
        super().__init__(
            data_record, data_root=data_root, norm_mode=norm_mode,
            global_center=global_center, global_scale=global_scale,
            normalizer=normalizer, meta_cols=meta_cols,
        )
        self.transform = transform              # Image transformations
        self.scale_pix = scale_pix              # Whether to scale BG&IR to RED mean

        if self.meta_cols is not None:
            missing = [c for c in self.meta_cols if c not in self.data_record.columns]
            assert not missing, f"Meta columns {missing} not found in data record"
            if meta_scaler is None:
                self.meta_scaler = StandardScaler().fit(self.data_record[self.meta_cols])
            else:
                self.meta_scaler = meta_scaler

        if pix_min is None or pix_max is None:
            pix_max = self.data_record['Pix_max'].max()
            pix_min = self.data_record['Pix_min'].min()
        self.pix_max = pix_max
        self.pix_min = pix_min
        self.denom = (pix_max - pix_min) + 1e-8

        assert pix_max <= 1 and pix_max >= 0, f"Pixel maximum value should be in the range [0, 1] but found {pix_max}"
        assert pix_min < pix_max and pix_min >= -1, (
            f"Pixel minimum should be less than pixel max and greater than -1 but found {pix_min} (max: {pix_max})")
        assert norm_mode in ['scene_robust', 'global_robust', None], f"Invalid norm_mode given: {norm_mode}"

        # Resolve the root directory for image paths
        root = self.data_root if self.data_root is not None else os.getcwd()

        rand_line = self.data_record.sample(1, random_state=42).iloc[0]
        try:
            img = np.load(os.path.join(root, rand_line['Path']))
            assert img.dtype == np.float32, f"Image should be of type float32, not {img.dtype}"
        except FileNotFoundError:
            pass

        # Confirm every path in dr exists
        invalid_obs = []
        for idx, row in self.data_record.iterrows():
            path = os.path.join(root, row['Path'])
            if not os.path.isfile(path):
                observation = row['Observation']
                invalid_obs.append(observation)

        if invalid_obs:
            original_count = self.data_record['Observation'].nunique()
            # Remove whole obs from dr
            invalid_obs = set(invalid_obs)
            self.data_record = self.data_record[~self.data_record['Observation'].isin(invalid_obs)]
            raise ValueError(f"WARNING: {len(invalid_obs)} observations out of {original_count} were removed from the dataset due to missing files\n"
                  f"Example of invalid path: {path}")

    def __len__(self):
        return self.data_record['Set'].nunique()

    def __getitem__(self, idx):
        set_idx = self.unique_sets[idx]
        df = self.data_record[self.data_record['Set'] == set_idx].set_index('CCD')

        # ------ LOAD IMAGES ------
        try:
            images = self.load_ccd_arrays(set_idx, {
                ccd: df.at[ccd, 'Path']
                for ccd in [*self.band_ccds, *self.neighbour_ccds, 'RED4']
            })
            band_imgs = [images[ccd] for ccd in self.band_ccds]
            neighbour_imgs = [images[ccd] for ccd in self.neighbour_ccds]
            label_img = images['RED4']
        except FileNotFoundError as e:
            raise FileNotFoundError(f"(Inside __getitem__) File not found for set {set_idx}: {e}")

        assert band_imgs[0].dtype == np.float32, f"Band images should be of type float32, not {band_imgs[0].dtype}"
        assert neighbour_imgs[0].dtype == np.float32, f"Neighbour images should be of type float32, not {neighbour_imgs[0].dtype}"
        assert label_img.dtype == np.float32, f"Label image should be of type float32, not {label_img.dtype}"

        # ------ LOAD META VALUES ------
        if self.meta_cols is not None and len(self.meta_cols) > 0:
            # !!!! CAUTION - Binning and TDI vary by CCD !!!
            raw_meta = pd.DataFrame([df.loc['RED3', self.meta_cols]], columns=self.meta_cols)    # (1, n_meta)
            meta_vals = self.meta_scaler.transform(raw_meta).squeeze(0)
            meta_tensor = torch.tensor(meta_vals, dtype=torch.float32)                # (n_meta,)
        else:
            meta_tensor = torch.empty(0, dtype=torch.float32)

        # ------ BUILD SAMPLE  ------
        obs_id = df.at['IR10', 'Observation']
        set_name = df.at['IR10', 'Set']
        date = df.at['IR10', 'Date']
        obs_width = _infer_observation_width_from_binning(df)

        patch_x = 0.0
        if PATCH_X_COLUMN in df.columns:
            patch_x_values = pd.to_numeric(df[PATCH_X_COLUMN], errors="coerce").dropna().unique()
            if len(patch_x_values) > 0:
                patch_x = float(patch_x_values[0])
        patch_x_tensor = torch.tensor(patch_x, dtype=torch.float32)
        obs_width_tensor = torch.tensor(obs_width, dtype=torch.float32)

        bands = torch.from_numpy(np.stack(band_imgs)).float()               # bands: (2, Hb, Wb)
        neighbours = torch.from_numpy(np.stack(neighbour_imgs)).float()     # neighbours: (2, Hn, Wn, 4)
        label = torch.from_numpy(label_img).float().unsqueeze(0)            # label: (1, Hb, Wb)

        # choose a shared affine (a,s)
        if self.norm_mode == "scene_robust":
            obs_norm = _load_observation_norm_from_record(df, bands.dtype)
            if obs_norm is not None:
                a, s = obs_norm
            else:
                a, s = self.normalizer.stats(bands, neighbours)
        elif self.norm_mode == "global_robust":
            assert self.global_center is not None and self.global_scale is not None, "global stats required"
            a = torch.tensor(float(self.global_center), dtype=bands.dtype)
            s = torch.tensor(float(self.global_scale), dtype=bands.dtype)
        else:
            # norm_mode not recognised, defaulting to a=0.0, s=1.0
            a = torch.tensor(0.0, dtype=bands.dtype)
            s = torch.tensor(1.0, dtype=bands.dtype)

        # apply the SAME affine to inputs and label (train-time)
        bands = (bands - a) / s
        neighbours = (neighbours - a) / s
        label = (label - a) / s

        x_band = bands                # (2, Hb, Wb)
        x_neigh = assemble_patch_neigh_from_stacked(neighbours)    # (6, Hn, Wn)

        if meta_tensor.numel() > 0:
            x_meta = meta_tensor.to(dtype=bands.dtype)  # (n_meta,)
        else:
            x_meta = torch.empty(0, dtype=bands.dtype)

        y = label

        # Capture stats for denormalization
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=bands.dtype)
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=bands.dtype)
        stats = torch.stack([a, s])

        chan_spec_band = ['IR10', 'BG12']
        chan_spec_neigh = [
            'R5_BG',
            'R5_IR',
            'R5_RE',
            'R3_BG',
            'R3_IR',
            'R3_RE',
        ]

        if self.meta_cols is not None and len(self.meta_cols) > 0:
            meta_spec = [f'META:{c}' for c in self.meta_cols]
        else:
            meta_spec = []

        sample = dict(
            x_band=x_band,                  # (2, Hb, Wb)
            x_neigh=x_neigh,                # (6, Hn, Wn)
            x_meta=x_meta,                  # (n_meta,)
            y=y,                            # (1, Hb, Wb)
            stats=stats,
            chan_spec_band=chan_spec_band,
            chan_spec_neigh=chan_spec_neigh,
            meta_spec=meta_spec,
            obs_id=obs_id,
            set_name=set_name,
            date=date,
            patch_x=patch_x_tensor,
            obs_width=obs_width_tensor,
        )

        assert x_band.shape[0] == len(chan_spec_band), (f"bands channels mismatch: {x_band.shape[0]=} vs {len(chan_spec_band)=}")
        assert x_neigh.shape[0] == len(chan_spec_neigh), (f"neigh channels mismatch: {x_neigh.shape[0]=} vs {len(chan_spec_neigh)=}")
        assert x_meta.numel() == len(meta_spec) or x_meta.numel() == 0, (f"meta length mismatch: {x_meta.numel()=} vs {len(meta_spec)=}")

        if self.transform:
            sample = self.transform(sample)
        return sample


class FilteredHiRISEDataset(HiRISEDataset):
    def __init__(self, data_record, sweep, transform=None, allowed_sets=None, meta_cols=None, norm_mode=DEFAULT_NORMALIZATION_MODE, global_center=None, global_scale=None, pix_min=None, pix_max=None, meta_scaler=None, data_root=None, normalizer=None):
        # Filter data record to only include allowed sets
        if allowed_sets is not None:
            set_orig = data_record['Set'].nunique()
            data_record = data_record[data_record['Set'].isin(allowed_sets)]
            if data_record.empty:
                raise ValueError("No data found for the specified allowed sets. None of the requested sets are present.")
            if not sweep:
                print("Filtered dataset:")
                print(f"\tObservations: {data_record['Observation'].nunique():,}")
                print(f"\tSets: {data_record['Set'].nunique():,}")
                print(f"\tImages: {len(data_record):,}")
                print(f"\tProportion of sets: {data_record['Set'].nunique() / set_orig:.2%}")
                print()

        # Call the parent constructor with updated data record
        super().__init__(data_record, transform, meta_cols, pix_min=pix_min, pix_max=pix_max, meta_scaler=meta_scaler, norm_mode=norm_mode, global_center=global_center, global_scale=global_scale, data_root=data_root, normalizer=normalizer)


class PatchInferenceDataset(BaseHiRISEDataset):
    """Map-style dataset yielding one raw inference patch sample per set.

    Each item matches exactly one element previously produced by
    ``HiPredict._load_patch_samples``: raw (unnormalized) assembled arrays plus the
    record-provided ``obs_center``/``obs_scale``. Normalization stays deferred to
    ``predict()`` (Option A: assembly here, affine applied at the edge).
    """

    required_ccds = ["BG12", "IR10", "RED3", "RED4", "RED5"]

    def __init__(self, data_record, obs_ids=None, sets=None, data_root=None,
                 norm_mode=DEFAULT_NORMALIZATION_MODE, global_center=None, global_scale=None, normalizer=None):
        df = data_record
        if obs_ids is not None:
            df = df[df["Observation"].isin(obs_ids)]
        if sets is not None:
            df = df[df["Set"].isin(list(sets))]
        if df.empty:
            raise ValueError("No rows remain after applying data_record filters.")
        super().__init__(
            df, data_root=data_root, norm_mode=norm_mode,
            global_center=global_center, global_scale=global_scale,
            normalizer=normalizer, meta_cols=None,
        )

    def __getitem__(self, i):
        set_id = self.unique_sets[i]
        set_df = self.data_record[self.data_record["Set"] == set_id].copy()

        if set_df["CCD"].duplicated().any():
            dupes = sorted(set_df.loc[set_df["CCD"].duplicated(), "CCD"].tolist())
            raise ValueError(f"Set {set_id} has duplicate CCD rows: {dupes}")

        set_df = set_df.set_index("CCD", drop=False)
        missing_ccds = [ccd for ccd in self.required_ccds if ccd not in set_df.index]
        if missing_ccds:
            raise KeyError(f"Set {set_id} is missing required CCD rows: {missing_ccds}")

        images = self.load_ccd_arrays(set_id, {
            ccd: set_df.at[ccd, "Path"] for ccd in self.required_ccds
        })
        bg, ir, r3, r4, r5 = (images[ccd] for ccd in self.required_ccds)

        if bg.ndim != 2 or ir.ndim != 2 or r4.ndim != 2:
            raise ValueError(
                f"Set {set_id} expected 2D BG12/IR10/RED4 patches, got "
                f"{bg.shape=}, {ir.shape=}, {r4.shape=}"
            )
        if r3.ndim != 3 or r3.shape[-1] < 3:
            raise ValueError(f"Set {set_id} expected RED3 context patch with shape (H, W, >=3), got {r3.shape}")
        if r5.ndim != 3 or r5.shape[-1] < 3:
            raise ValueError(f"Set {set_id} expected RED5 context patch with shape (H, W, >=3), got {r5.shape}")

        x_band, x_neigh = assemble_patch_inputs_np(bg, ir, r3, r5)

        obs_id = str(set_df.at["RED4", "Observation"])
        patch_idx = int(set_df.at["RED4", "Patch"]) if "Patch" in set_df.columns else None
        obs_center = None
        obs_scale = None
        if {"Obs_center", "Obs_scale"}.issubset(set_df.columns):
            center_val = set_df.at["RED4", "Obs_center"]
            scale_val = set_df.at["RED4", "Obs_scale"]
            if pd.notna(center_val) and pd.notna(scale_val) and float(scale_val) > 0:
                obs_center = float(center_val)
                obs_scale = float(scale_val)

        return {
            "x_band": x_band,
            "x_neigh": x_neigh,
            "obs_id": obs_id,
            "set_name": set_id,
            "patch_idx": patch_idx,
            "obs_center": obs_center,
            "obs_scale": obs_scale,
            "y": r4,
        }


def collate_fn(batch):
    x_band = torch.stack([b['x_band'] for b in batch], dim=0)    # (B, 2, Hb, Wb)
    x_neigh = torch.stack([b['x_neigh'] for b in batch], dim=0)  # (B, 6, Hn, Wn)
    x_meta = torch.stack([b['x_meta'] for b in batch], dim=0)      # (B, n_meta) or (B, 0)
    y = torch.stack([b['y'] for b in batch], dim=0)    # (B, 1, H, W)
    stats = torch.stack([b['stats'] for b in batch], dim=0)
    cspecs_band = [b['chan_spec_band'] for b in batch]
    cspecs_neigh = [b['chan_spec_neigh'] for b in batch]
    patch_x = torch.stack([b['patch_x'] for b in batch], dim=0)
    obs_width = torch.stack([b['obs_width'] for b in batch], dim=0)
    cspecs_meta = [b['meta_spec'] for b in batch]
    return dict(
        x_band=x_band,
        x_neigh=x_neigh,
        x_meta=x_meta,
        y=y,
        stats=stats,
        chan_spec_band=cspecs_band[0],
        chan_spec_neigh=cspecs_neigh[0],
        patch_x=patch_x,
        obs_width=obs_width,
        meta_spec=cspecs_meta[0],
        obs_id=[b['obs_id'] for b in batch],
        set_name=[b['set_name'] for b in batch],
        date=[b['date'] for b in batch],
    )


def identity_collate(batch):
    """Return the batch as a plain list of samples, unchanged.

    Used for inference so ``predict()`` receives the same list-of-dicts it has
    always consumed, rather than tensor-stacked batches from the default collate.
    """
    return list(batch)


def get_loader(dataset, batch_size, shuffle=True, pin_memory=True, prefetch_factor=2,
               num_workers=4, collate_fn=None, worker_init_fn=None, persistent_workers=True, seed=42):
    torch.manual_seed(seed)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
    )
    # prefetch_factor / persistent_workers are only valid with worker processes;
    # torch raises if they are passed when num_workers == 0 (the inference default).
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **kwargs)
