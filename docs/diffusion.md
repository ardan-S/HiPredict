# Diffusion and flow-matching models

`hipredict.diffusion` adds conditional diffusion (DDPM) and flow-matching (FM) models that reconstruct HiRISE **RED4 ↔ IR10**. FM models can also be run through the HiPredict inference API. The module is self-contained: nothing else in HiPredict imports it.

Pretrained FM weights are available on Hugging Face; see the [checkpoint links and download example](../hipredict/diffusion/README.md). To train a model, see [Train](#train).

## Install

```bash
uv sync --extra diffusion
```

Evaluation uses HiPredict's metric code, which also needs the `postprocess` extra (`uv sync --extra diffusion --extra postprocess`). The `all` extra includes both. Training and sampling need a CUDA GPU in practice.

The tests for this module run with the rest of the suite: `uv run --extra all pytest`.

## Data

The models train on the [HiPredict dataset](workflows.md#the-hipredict-dataset). By default they read `files/data_record_alls.csv` (or `$HIPREDICT_FILES_DIR/data_record_alls.csv`). Record paths relative to the CSV are resolved against its folder, as in HiPredict, and absolute paths are used as they are. A record path under `npy_files_all/` that does not exist is looked up under `DataConfig.data_path` instead (default `files/npy_files_all`), so records can follow a moved tile tree.

Each set needs one IR10 and one RED4 row, plus RED3 when a boundary-based brightness correction is used. Sets missing a required CCD are excluded and counted in the log; duplicate CCD rows are an error.

The diffusion dataset extends HiPredict's `BaseHiRISEDataset` and shares its raw CCD reader and `ConstantNormalizer` strategy. Splits use the same chronological 70% / 10% / 20% split as HiPredict, so test sets match. Model inputs and training targets are normalised as `(x - center) / scale` and clipped to `±clamp`, with fixed IR10 statistics by default (`NormalizationConfig` in `hipredict/diffusion/config.py`), not HiPredict's `global_stats.json`. The values are stored in every checkpoint, and evaluation, inference and the HiPredict API use the stored values. Evaluation scores predictions against the original, unclipped RED4 or IR10 arrays; it never reconstructs ground truth from a clipped training target.

## Entry points

Run from the directory where outputs should go. Relative defaults such as `outputs/...` resolve against the working directory.

| Command | Purpose |
| --- | --- |
| `python -m hipredict.diffusion.train.train` | Train a DDPM or FM model |
| `python -m hipredict.diffusion.eval.eval` | Score a checkpoint on a split |
| `python -m hipredict.diffusion.inference.inference` | Predict single images, or a whole test split |
| `python -m hipredict.diffusion.dataset.EDA` | Dataset statistics and the conditional-mean fit |
| `python -m hipredict.diffusion.models.brightness` | Train the RED3 boundary brightness head |

## Train

```bash
python -m hipredict.diffusion.train.train \
    --run_name my_run \
    --model_type BidirectionalFM \
    --direction bidirectional \
    --edge_type sobel \
    --FM_mode Condition \
    --lambda_grad 0.1 \
    --save_dir outputs/train/bidirectional/my_run/ \
    --save_every 20000 \
    --no_wandb
```

- `--model_type`: `UnidirectionalDDPM`, `BidirectionalDDPM`, `UnidirectionalFM`, `BidirectionalFM`.
- `--direction`: bidirectional models train with `bidirectional`; unidirectional models with `ir2red` or `red2ir`. Other combinations are rejected. Bidirectional training splits each batch between the two directions, so it needs a batch size of at least 2.
- `--edge_type`: structural conditioning, `sobel` (default), `dexined`, `soft_canny` or `none`. `dexined` downloads pretrained weights through kornia on first use.
- `--FM_mode`: `Condition` (ODE from noise) or `Bridge` (ODE from the source image), FM models only.
- `--lambda_grad`: weight of the signed Sobel loss on the FM velocity field (default `0.0`).
- `--no_ema` disables the weight EMA.
- `--csv_path`, `--batch_size`, `--lr`, `--total_steps`, `--log_every`, `--val_every` override the defaults in `TrainingConfig` (`hipredict/diffusion/config.py`), which also holds the learning-rate decay and EMA settings. Invalid values are rejected.
- `--resume` continues from `latest.pt` in `--save_dir`. The model settings and the training settings must match the checkpoint; only worker counts, `--save_dir`, the save, log and validation intervals and `total_steps` may change. A mismatch is an error that lists the differing settings.

## Brightness head and conditional-mean fit

The source image constrains the target's structure but not its overall brightness, so evaluation and inference set the mean of each prediction (`--dc_correct`). The default, `boundary`, takes it from the RED3 context, which observes both RED and IR where it overlaps RED4. `head` adds a small MLP that predicts a correction from the boundary, the source statistics and observation metadata:

```bash
python -m hipredict.diffusion.models.brightness --out_dir outputs/ckpt/brightness
```

It writes `model.pt`, used by `--dc_correct head` (read from `outputs/ckpt/brightness/model.pt` unless `--head_ckpt` is given).

`python -m hipredict.diffusion.dataset.EDA --output_dir outputs/dataset` writes per-split statistics and distributions, and `cond_mean_fit.pt`, the linear fit used by `--dc_correct cond` (pass it with `--cond_fit`). Histograms and summary statistics use a seeded sample of `--pixels_per_image` pixels per image (default 4,096; `0` keeps every pixel, which needs memory for the whole split); the per-image means behind the fit use every pixel.

## Eval

```bash
python -m hipredict.diffusion.eval.eval \
    --ckpt_path outputs/train/bidirectional/my_run/latest.pt \
    --direction bidirectional \
    --dataset test \
    --batch_size 16 \
    --FM_ODE_steps 10 \
    --FM_solver euler \
    --output_dir outputs/eval/my_run/
```

Reports NMAE, bias, PWT and SSIM in physical units, computed with HiPredict's own per-image definitions (`hipredict.post.run_evals`), so the values are directly comparable with HiPredict's evaluation. Every sample of the split is scored, and results are written to `<output_dir>/eval_metrics.json`.

- `--dataset`: `train`, `val` or `test` (default `val`). `--max_batches` caps the number of batches, which DDPM sampling usually needs.
- `--csv_path` evaluates another record (default `files/data_record_alls.csv`).
- `--dc_correct`: `boundary` (default), `head`, `cond`, `none` or `oracle`. `oracle` uses the true target mean and is a reference upper bound only.
- The corrected mean is also applied to the FM clean estimate after every ODE step; `--no_dc_anchor` applies it to the final prediction only.
- `--n_draws N` samples N times per batch and writes the spread, error decomposition and CRPS under `<output_dir>/uncertainty`.
- `--plot_pixel_histogram` and `--plot_nmae_histogram` save the pixel distribution and the per-image NMAE distribution.

## Inference

**Single images.** Pass one or more source images: IR10 for `ir2red`, RED4 for `red2ir`. Images of any size are padded to a multiple of 32 and cropped back. Images larger than the 256 × 256 training patches are predicted as overlapping tiles (`--tile_size`, `--tile_overlap`, default 256 and 64) blended with linear ramps; when the RED3 context has the image's height, each tile takes its brightness from the 1,024 context rows around it, as in training:

```bash
python -m hipredict.diffusion.inference.inference \
    --ckpt_path outputs/train/bidirectional/my_run/latest.pt \
    --input_path ESP_034899_2230_0_IR10.npy \
    --red3_path ESP_034899_2230_0_RED3.npy \
    --direction ir2red \
    --output_dir outputs/inference/
```

Each prediction is named after the band it predicts (`ESP_034899_2230_0_IR10.npy` → `ESP_034899_2230_0_RED4.npy`). `--dc_correct` is `boundary` (default) or `none`; `boundary` takes the brightness from the RED3 context array (shape `(H, 32, 4)`) given with `--red3_path`, one per input.

**Test split.** Reconstruct the test split of a record and mirror its tile layout under `--output_dir`:

```bash
python -m hipredict.diffusion.inference.inference \
    --ckpt_path outputs/train/bidirectional/my_run/latest.pt \
    --test_csv files/data_record_alls.csv \
    --direction ir2red \
    --output_dir outputs/reconstructions/ir2red/my_run/
```

Each prediction is written at the path of the target tile it replaces: RED4 tiles for `ir2red`, IR10 tiles for `red2ir`. `--dc_correct` takes `boundary` (default), `head`, `none` or `oracle`. `--data_root` is only needed for moved `npy_files_all` records.

## HiPredict API

`build_hipredict_predictor` wraps an FM checkpoint so it is used like the model returned by `get_model`, on patches or full strips. The HiPredict API predicts RED4, so the checkpoint must be `ir2red` or `bidirectional`:

```python
from hipredict.diffusion import build_hipredict_predictor

predictor, cfg_model, step = build_hipredict_predictor(
    "outputs/train/bidirectional/my_run/latest.pt", device="cuda"
)
samples = predictor.load("files/data_record_evals.csv", mode="patches")
outputs = predictor.predict(samples)
```

Inputs are normalised with the checkpoint's stored statistics rather than `global_stats.json`, and the RED4 brightness is set from the RED3 boundary with anchoring, the same default as the command-line tools. `ode_steps` and `solver` correspond to `--FM_ODE_steps` and `--FM_solver`. All edge modes are supported.

The FM networks attend over every position of their input, so memory grows with the square of the image area. Inputs larger than `tile_size` (default 256), such as full strips, are therefore predicted as overlapping tiles in the same way as single images; `tile_overlap` sets the overlap. Each tile is sampled and brightness-anchored independently, so faint seams are possible where the brightness changes across a tile boundary.

## Checkpoints

Training writes versioned checkpoints holding the model and training settings (including the normalisation), the weights, the EMA weights and the optimiser state. They are plain tensors and dictionaries and load with `torch.load(..., weights_only=True)`; no pickled classes are involved. `load_checkpoint` uses the EMA weights when present.
