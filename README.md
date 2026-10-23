# HiPredict

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23414377.svg)](https://doi.org/10.5281/zenodo.23414377)
[![Release](https://img.shields.io/github/v/release/ardan-S/HiPredict)](https://github.com/ardan-S/HiPredict/releases)
[![License: MIT](https://img.shields.io/github/license/ardan-S/HiPredict)](LICENSE)

**Paper:** [ArXiv](https://doi.org/10.48550/arXiv.2610.23335) </br>
**Model:** [ardan-S/HiPredict](https://huggingface.co/ardan-S/HiPredict) on Hugging Face ([doi:10.57967/hf/11018](https://doi.org/10.57967/hf/11018))</br>
**Dataset:** [ardan-S/HiPredict](https://huggingface.co/datasets/ardan-S/HiPredict) on Hugging Face ([doi:10.57967/hf/11017](https://doi.org/10.57967/hf/11017))</br>
**Code:** archived on Zenodo ([doi:10.5281/zenodo.23414377](https://doi.org/10.5281/zenodo.23414377))</br>

HiPredict reconstructs the RED4 channel of the HiRISE camera on the Mars Reconnaissance Orbiter. The RED4 detector has partially failed. HiPredict predicts RED4 from the instrument's other channels: the IR10 and BG12 colour bands, which image the same ground as RED4, and the neighbouring RED3 and RED5 detectors.

This repository contains the full workflow: downloading HiRISE observations, radiometric preprocessing, inference on patches or full strips, evaluation, and training. Each step can be used on its own. It also provides **SYN4**, a simple baseline that scales IR10 to the brightness of RED5 and needs no trained weights.

## Results

### Example reconstructions

Each example shows a false-colour composite made with the true RED4, the same composite made with HiPredict's RED4, and the absolute difference between the true and predicted RED4.

<table>
  <tr>
    <th width="33%">True RED4</th>
    <th width="33%">HiPredict RED4</th>
    <th width="33%">Absolute error</th>
  </tr>
  <tr>
    <td colspan="3" align="center"><b>ESP_012940_1655</b> (−14.0974°, 318.282°E), 512 × 512 pixels</td>
  </tr>
  <tr>
    <td width="33%"><img src="docs/figures/example_1_true.png" width="100%" alt="ESP_012940_1655: false-colour composite with the true RED4"></td>
    <td width="33%"><img src="docs/figures/example_1_hipredict.png" width="100%" alt="ESP_012940_1655: false-colour composite with HiPredict's RED4"></td>
    <td width="33%"><img src="docs/figures/example_1_error.png" width="100%" alt="ESP_012940_1655: absolute RED4 error map"></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><b>ESP_055232_1595</b> (−20.1668°, 226.817°E), 256 × 256 pixels</td>
  </tr>
  <tr>
    <td width="33%"><img src="docs/figures/example_2_true.png" width="100%" alt="ESP_055232_1595: false-colour composite with the true RED4"></td>
    <td width="33%"><img src="docs/figures/example_2_hipredict.png" width="100%" alt="ESP_055232_1595: false-colour composite with HiPredict's RED4"></td>
    <td width="33%"><img src="docs/figures/example_2_error.png" width="100%" alt="ESP_055232_1595: absolute RED4 error map"></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><b>ESP_056705_1440</b> (−35.4419°, 342.328°E), 256 × 256 pixels</td>
  </tr>
  <tr>
    <td width="33%"><img src="docs/figures/example_3_true.png" width="100%" alt="ESP_056705_1440: false-colour composite with the true RED4"></td>
    <td width="33%"><img src="docs/figures/example_3_hipredict.png" width="100%" alt="ESP_056705_1440: false-colour composite with HiPredict's RED4"></td>
    <td width="33%"><img src="docs/figures/example_3_error.png" width="100%" alt="ESP_056705_1440: absolute RED4 error map"></td>
  </tr>
</table>

The composites show IR10, RED4 and BG12 as red, green and blue, each contrast-stretched independently. In the error maps, brighter means a larger error. In the second and third examples the error is mostly salt-and-pepper noise with little spatial structure. This is likely due to binning artefacts and genuine instrument noise in the true RED4, which the model is not expected to reproduce. Image data: NASA/JPL/University of Arizona.

### Metrics

Performance on unseen data, compared with the SYN4 baseline (↓ lower is better, ↑ higher is better):

| Model | NMAE mean (%) ↓ | NMAE median (%) ↓ | SSIM mean ↑ | SSIM median ↑ | PWT 1% mean (%) ↑ | PWT 1% median (%) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SYN4 | 3.118 | 1.545 | 0.663 | 0.722 | 11.069 | 9.148 |
| **HiPredict** | **1.110** | **0.943** | **0.725** | **0.788** | **20.844** | **18.801** |

- **NMAE**: normalised mean absolute error, the mean absolute error as a percentage of the image's mean brightness.
- **SSIM**: structural similarity index.
- **PWT 1%**: percentage of pixels whose absolute error is within 1% of the image's dynamic range.

HiPredict's mean NMAE is about a third of SYN4's, and about twice as many of its pixels fall within the 1% tolerance.

### Where HiPredict helps most

![HiPredict error per image plotted against SYN4 error on log axes. The median HiPredict error stays between about 0.7% and 3% while SYN4 error rises to around 50%; the median line crosses the parity line at roughly 1.5% SYN4 error.](docs/figures/hipredict_vs_syn4_error.png)

HiPredict's error per image against SYN4's, grouped by SYN4 error. The line is the median and the band the 25th–75th percentile. Below the dashed line HiPredict is more accurate. On the simplest images, where SYN4's error is below about 1.5%, SYN4 is as good or slightly better. As images get harder, SYN4's error grows far faster than HiPredict's.

## How it works

The model is a U-Net. For each RED4 location it receives the registered IR10 and BG12 bands plus 32-pixel-wide context strips from the adjacent edges of RED3 and RED5. It is also conditioned on nine observation metadata values (TDI, incidence and phase angles, local time, solar longitude and distance, orbit number, and two instrument temperatures). The U-Net reconstructs spatial detail, and a brightness head sets the local level from the neighbouring RED context. Inputs are normalised with fixed dataset-wide statistics, and predictions are returned in the same calibrated units as the processed cubes.

## Installation

HiPredict is developed and tested on Linux. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/ardan-S/HiPredict.git
cd HiPredict
uv sync
```

This installs everything needed to download observations, load the model and run inference on CPU or GPU. Optional extras:

| Extra | Adds |
| --- | --- |
| `train` | Training and evaluation dependencies |
| `postprocess` | Evaluation plots and maps |
| `cube` | GDAL bindings for ISIS cube export |
| `dev` | The test suite |
| `notebooks` | IPython and Jupyter |
| `diffusion` | Conditional diffusion and flow-matching models |
| `all` | Everything except `cube` |

For example, `uv sync --extra train --extra postprocess`. The GDAL Python bindings in `cube` must match a system GDAL installation.

Raw radiometric preprocessing also needs a separate HiPROC + ISIS environment; see [Preprocessing](docs/preprocessing.md). It is not needed to run inference on observations that are already processed.

## Model weights

The trained weights are hosted on Hugging Face at [`ardan-S/HiPredict`](https://huggingface.co/ardan-S/HiPredict). They are downloaded and cached on first use, pinned to the release this version of the code was validated against:

```python
from hipredict import get_model, HiPredict

model, config = get_model("hipredict", device="cpu")
predictor = HiPredict(model, config, device="cpu")
```

`get_model` also accepts `hf_repo_id` and `revision` to load another release, `local_files_only=True` for offline use, and `model_save_dir` for a local copy. See [Model weights](docs/model.md).

## Quick start

Download, preprocess and predict one observation:

```bash
uv run -m hipredict.download --obs ESP_028693_1535 --workers 2
uv run -m hipredict.process --obs ESP_028693_1535 --workers 2
uv run -m hipredict.inference.predict --obs ESP_028693_1535 --model both
```

- **Download** fetches the ten raw `.IMG` files (two per detector) for BG12, IR10, RED3, RED4 and RED5. The first run also downloads and caches the HiRISE PDS EDR index, which takes a few minutes.
- **Process** needs the HiPROC + ISIS backend. It writes five `*_processed.cub` files to `files/isis_imgs/observations/<observation>/`.
- **Predict** writes float32 NumPy arrays and TIFFs to `files/predictions/`, named `<observation>_HiPredict` and `<observation>_SYN4`. Values keep the calibrated units of the input cubes. Pixels outside the valid crop are NaN, and the TIFFs are neither stretched nor georeferenced. Add `--format npy tif cub` for an ISIS cube (needs the `cube` extra and the backend).

Every command accepts `--obs-list file.txt` (one observation ID per line) and `--img-dir`. Prediction also accepts `--output-dir`, `--model hipredict|SYN4|both` and `--device`, and can run all three stages with `--download --preprocess`. SYN4 alone (`--model SYN4`) needs no model download.

## Python API

```python
from hipredict import Download, PrepObservations, get_model, HiPredict

observations = ["ESP_028693_1535"]
image_root = "files/isis_imgs/observations"

Download(observations, image_root, workers=2)
PrepObservations(obs_list=observations, img_base_dir=image_root, workers=2)

model, config = get_model("hipredict")
predictor = HiPredict(model, config)
samples = predictor.load(image_root, mode="strips", obs_ids=observations)
hipredict_output = predictor.predict(samples, window_size=8192, pad_x=256, pad_y=64)
syn4_output = predictor.predict(samples, syn4=True)
```

`load(..., mode="strips")` expects all five processed cubes. RED4 is used only as the reference for evaluation. To predict from arrays you have already loaded, pass samples of the form `{"bg": bg, "ir": ir, "r3": r3, "r5": r5, "obs_id": observation}`. Observations with unsupported binning or TDI settings are skipped during preprocessing.

## Training and evaluation

The dataset used to train and evaluate HiPredict, including its patch records, is published on Hugging Face at [`ardan-S/HiPredict`](https://huggingface.co/datasets/ardan-S/HiPredict). You can also generate patches from your own observations with `--mode patches`. See [Workflows](docs/workflows.md) for patch generation, training, evaluation, runtime paths and storage requirements.

## Documentation

- [Preprocessing](docs/preprocessing.md): the HiPROC + ISIS backend and the processing pipeline.
- [Model weights](docs/model.md): how weights are located, verified and used offline.
- [Workflows](docs/workflows.md): patches, training, evaluation, exports and storage.
- [Diffusion and flow matching](docs/diffusion.md): installation, training, evaluation and the HiPredict adapter.
- [`files/`](files/README.md): the observation lists and statistics shipped with the code.

## Data and software

HiRISE data are from the NASA Planetary Data System, produced by the HiRISE team at the University of Arizona. Preprocessing uses [USGS ISIS](https://isis.astrogeology.usgs.gov/) and [HiPROC](https://github.com/rbeyer/hiproc).

## License

HiPredict is released under the [MIT License](LICENSE).
