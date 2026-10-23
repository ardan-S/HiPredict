from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hipredict.diffusion.config import DataConfig
from hipredict.diffusion.dataset import DiffusionDataset, global_normalize, split_by_date

CCDS = ("RED4", "IR10")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/dataset")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Record CSV (default: files/data_record_alls.csv)",
    )
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument(
        "--pixels_per_image",
        type=int,
        default=4096,
        help="Pixels sampled per image for histograms and summaries; 0 keeps all",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def compute_histogram(pixels, bins=256, hist_min=None, hist_max=None):
    """
    Probability histogram + bin edges for a 1-D pixel array.
    """
    if hist_min is None or hist_max is None:
        lo, hi = float(pixels.min()), float(pixels.max())
        margin = (hi - lo) * 0.1
        hist_min = float(np.floor((lo - margin)))
        hist_max = float(np.ceil((hi + margin)))

    counts, edges = np.histogram(pixels, bins=bins, range=(hist_min, hist_max))
    counts = counts.astype(np.float64)
    counts = counts / (counts.sum() + 1e-12)
    return counts.astype(np.float32), edges.astype(np.float32), hist_min, hist_max


def collect_split(config, set_ids, pixels_per_image, rng):
    """One pass over a split's images.

    Returns sampled raw and normalized pixels per CCD (the same pixel positions
    in both domains) and every image's mean in the normalized domain.
    """
    dataset = DiffusionDataset(config, allowed_sets=set_ids)
    raw = {ccd: [] for ccd in CCDS}
    norm = {ccd: [] for ccd in CCDS}
    means = {ccd: [] for ccd in CCDS}
    for set_id in dataset.sets:
        records = dataset.set_records[set_id]
        images = {
            ccd: torch.from_numpy(np.load(records[ccd]["resolved_path"]).astype(np.float32))
            for ccd in CCDS
        }
        ir_norm, red_norm, _ = global_normalize(
            images["IR10"], images["RED4"], **config.normalization.model_dump()
        )
        normalized = {"IR10": ir_norm, "RED4": red_norm}
        for ccd in CCDS:
            flat = images[ccd].reshape(-1).numpy()
            flat_norm = normalized[ccd].reshape(-1).numpy()
            if 0 < pixels_per_image < flat.size:
                index = rng.choice(flat.size, size=pixels_per_image, replace=False)
                flat, flat_norm = flat[index], flat_norm[index]
            raw[ccd].append(flat)
            norm[ccd].append(flat_norm)
            means[ccd].append(float(normalized[ccd].mean()))
    return (
        {ccd: np.concatenate(v) for ccd, v in raw.items()},
        {ccd: np.concatenate(v) for ccd, v in norm.items()},
        {ccd: np.asarray(v, dtype=np.float64) for ccd, v in means.items()},
    )


def plot_distribution(pixels, split_name, out_dir, kind):
    """Save a boxplot + histogram for one split. kind is 'raw' or 'norm'."""
    suffix = "" if kind == "raw" else "_norm"
    tag = " (normalized)" if kind == "norm" else ""

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([pixels[c] for c in CCDS], showfliers=False)
    ax.set_xticks([1, 2], CCDS)
    ax.set_title(f"{split_name}{tag} pixel value distribution")
    ax.set_ylabel("Pixel value")
    fig.tight_layout()
    fig.savefig(out_dir / f"{split_name}_boxplot{suffix}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for ccd in CCDS:
        ax.hist(pixels[ccd], bins=100, density=True, histtype="step", label=ccd)
    ax.set_title(f"{split_name}{tag} pixel histogram")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{split_name}_histogram{suffix}.png", dpi=150)
    plt.close(fig)


def summarize(pixels, split_name, kind):
    rows = []
    for ccd in CCDS:
        p = pixels[ccd]
        rows.append(
            {
                "split": split_name,
                "ccd": ccd,
                "kind": kind,
                "pixels": p.size,
                "mean": p.mean(),
                "std": p.std(),
                "min": p.min(),
                "median": np.median(p),
                "max": p.max(),
            }
        )
    return rows


def _fit_pair(x, y):
    """OLS + robust (Huber) linear fit of y / x. The robust slope/intercept go
    into a/b (used by guidance); OLS is kept for reference."""
    from sklearn.linear_model import HuberRegressor

    ols_a, ols_b = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    huber = HuberRegressor(max_iter=1000).fit(x.reshape(-1, 1), y)
    return {
        "a": float(huber.coef_[0]),
        "b": float(huber.intercept_),
        "ols_a": float(ols_a),
        "ols_b": float(ols_b),
        "r": r,
    }


def fit_conditional_mean(train_means, out_dir, val_means=None):
    """Fit the per-image target mean as a linear function of the source mean
    on the train split (OLS + robust Huber).
    """
    tr_ir, tr_red = train_means["IR10"], train_means["RED4"]
    fit_r = _fit_pair(tr_ir, tr_red)  # ir2red
    fit_i = _fit_pair(tr_red, tr_ir)  # red2ir

    # Val best-fit slope for comparison only.
    val_ols = {}
    if val_means is not None:
        v_ir, v_red = val_means["IR10"], val_means["RED4"]
        val_ols["ir2red"] = tuple(float(v) for v in np.polyfit(v_ir, v_red, 1))
        val_ols["red2ir"] = tuple(float(v) for v in np.polyfit(v_red, v_ir, 1))

    panels = [
        (tr_ir, tr_red, fit_r, "mean(IR10)", "mean(RED4)", "ir2red"),
        (tr_red, tr_ir, fit_i, "mean(RED4)", "mean(IR10)", "red2ir"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (x, y, f, xl, yl, name) in zip(axes, panels):
        ax.scatter(x, y, s=6, alpha=0.3)
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(
            xs,
            f["ols_a"] * xs + f["ols_b"],
            "g--",
            lw=1.5,
            label=f"train OLS: {f['ols_a']:.3f}x+{f['ols_b']:.3f}",
        )
        ax.plot(
            xs,
            f["a"] * xs + f["b"],
            "r-",
            lw=2,
            label=f"train Huber: {f['a']:.3f}x+{f['b']:.3f}",
        )
        if name in val_ols:
            va, vb = val_ols[name]
            ax.plot(
                xs, va * xs + vb, "b:", lw=1.5, label=f"val OLS: {va:.3f}x+{vb:.3f}"
            )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{name}  (r={f['r']:.3f})")
        ax.legend(fontsize=8)
    fig.suptitle("Per-image mean: train OLS vs Huber vs val OLS (normalized)")
    fig.tight_layout()
    fig.savefig(out_dir / "cond_mean_scatter.png", dpi=150)
    plt.close(fig)

    coeffs = {"ir2red": fit_r, "red2ir": fit_i, "n_images": int(len(tr_ir))}
    torch.save(coeffs, out_dir / "cond_mean_fit.pt")

    return coeffs


def main():
    params = parse_args()
    out_dir = Path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = DataConfig()
    if params.csv_path:
        config.csv_path = params.csv_path
    df = pd.read_csv(config.csv_path)
    train_sets, test_sets, val_sets, *_ = split_by_date(df, 0.7, 0.2, 0.1)
    splits = {"train": train_sets, "val": val_sets, "test": test_sets}
    rng = np.random.default_rng(params.seed)

    summary = []
    hist_store = {}  # flat arrays for the analysis .npz
    split_means = {}
    for split_name, set_ids in splits.items():
        print(f"Processing {split_name}: {len(set_ids)} sets")
        raw, norm, split_means[split_name] = collect_split(
            config, set_ids, params.pixels_per_image, rng
        )

        # "raw" is the physical domain; "norm" is the model's normalized domain.
        for kind, pixels in (("raw", raw), ("norm", norm)):
            plot_distribution(pixels, split_name, out_dir, kind=kind)
            summary += summarize(pixels, split_name, kind)

        # Histograms for both domains; "denorm" is the raw physical domain.
        for domain, pixels in (("denorm", raw), ("norm", norm)):
            for ccd in CCDS:
                counts, edges, _, _ = compute_histogram(
                    pixels[ccd], bins=params.bins
                )
                hist_store[f"{split_name}_{domain}_{ccd}_counts"] = counts
                hist_store[f"{split_name}_{domain}_{ccd}_edges"] = edges

    pd.DataFrame(summary).to_csv(out_dir / "eda_summary.csv", index=False)

    # All-splits, both-domains histograms for analysis.
    np.savez(out_dir / "eda_histograms.npz", **hist_store)

    # Conditional per-image mean fit (train); val means for slope comparison.
    fit_conditional_mean(split_means["train"], out_dir, val_means=split_means["val"])

    print(f"Saved plots, summary, histograms and the conditional-mean fit to {out_dir}")


if __name__ == "__main__":
    main()
