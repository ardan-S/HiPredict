import json
import logging
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


logger = logging.getLogger(__name__)


class DrawEnsemble:
    """Collects per-draw and per-sample statistics from a draw ensemble.
    Every draw of an image arrives together, so a sample is finished in a single visit.
    """

    def __init__(self):
        """Three granularities, each answering a different question:
            draws_results: [N] one dict per draw, appended by end_draw(), ordinary eval metrics results.
            per_sample: one entry per image, keyed "<direction>/<set_id>"
                    mse_of_draw_mean     mean_pix (p_mean-y)^2
                    mean_pixel_variance  mean_pix Var_s(p)
                    mean_sigma           mean_pix sqrt(Var_s(p))
                    mae_draw_mean        mean_pix |p_mean-y|
                    crps                 mean_pix CRPS(draws, y)
            decomposition_sums: the same five scalars under the same names summed over every image
        """
        self.draws_results = []
        self.per_sample: dict[str, dict[str, list]] = {}
        self.decomposition_sums: dict[str, float] = dict.fromkeys(
            (
                "mse_of_draw_mean",
                "mean_pixel_variance",
                "mae_draw_mean",
                "mean_sigma",
                "crps",
            ),
            0.0,
        )

    def add_batch(self, direction, set_ids, preds_denorm, target_denorm) -> None:
        """Record every draw's result on one same batch.

        Args:
            direction: "ir2red" or "red2ir"
            set_ids:[B] list of sample identifiers
            preds_denorm: [N, B, 1, H, W] list of N tensors, each [B, 1, H, W], one per draw
            target_denorm:[B, 1, H, W] the ground truth for this batch
        """
        stack = torch.stack(preds_denorm).double()
        target = target_denorm.double()

        for j, set_id in enumerate(set_ids):
            key = f"{direction}/{set_id}"
            # Pixel errors of each draw, for the error/variance decomposition.
            errors = stack[:, j, 0] - target[j, 0]  # [N, H, W]
            self.per_sample[key] = {
                "draw_mae": errors.abs().mean(dim=(1, 2)).tolist(),
                "draw_mse": errors.pow(2).mean(dim=(1, 2)).tolist(),
            }

            # E_s[(p-y)^2] = (p_mean-y)^2 + Var_s(p)
            draws = stack[:, j, 0]  # [N, H, W] Same image predicted from N draws
            y = target[j, 0] # [H, W] Ground truth for this image
            draw_mean = draws.mean(0) # [H, W] Mean prediction over N draws
            variance = draws.var(0, unbiased=False) # [H, W] Variance of predictions over N draws

            mse_of_draw_mean = ((draw_mean - y) ** 2).mean().item() # Mean squared error of the mean prediction
            mae_draw_mean = (draw_mean - y).abs().mean().item() # Mean absolute error of the mean prediction
            mean_pixel_variance = variance.mean().item() # Mean of the pixel-wise variance
            mean_sigma = variance.sqrt().mean().item()# Mean of the pixel-wise standard deviation (sigma)
            # CRPS
            n = draws.shape[0]
            ordered, _ = draws.sort(0)
            i = torch.arange(n, device=draws.device, dtype=draws.dtype).view(-1, 1, 1)
            spread = (2 * i - n + 1).mul(ordered).sum(0) / (n * (n - 1))
            crps = ((draws - y).abs().mean(0) - spread).mean().item()
            self.per_sample[key].update(
                {
                    "mse_of_draw_mean": mse_of_draw_mean,
                    "mean_pixel_variance": mean_pixel_variance,
                    "mean_sigma": mean_sigma,
                    "mae_draw_mean": mae_draw_mean,
                    "crps": crps,
                }
            )

            self.decomposition_sums["mse_of_draw_mean"] += mse_of_draw_mean
            self.decomposition_sums["mean_pixel_variance"] += mean_pixel_variance
            self.decomposition_sums["mae_draw_mean"] += mae_draw_mean
            self.decomposition_sums["mean_sigma"] += mean_sigma
            self.decomposition_sums["crps"] += crps

    def end_draw(self, split_metrics: dict) -> None:
        self.draws_results.append(dict(split_metrics))

    def results(self) -> dict:
        return {
            "draws_results": self.draws_results,
            "per_sample": self.per_sample,
            "decomposition_sums": self.decomposition_sums,
            "n_samples": len(self.per_sample),
        }


def _plot_step_histogram(
    named_arrays,
    out_path,
    xlabel,
    bins,
    title=None,
    add_inset=True,
    inset_pct=(0.5, 99.5),
    percentile_plot=True,
):
    """
    Overlay one step histogram per named array on a shared x-axis.
    """
    arrs = {
        k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in named_arrays.items()
    }
    arrs = {k: v[np.isfinite(v)] for k, v in arrs.items() if v.size}

    all_vals = np.concatenate(list(arrs.values()))

    if percentile_plot:
        lo, hi = np.percentile(all_vals, inset_pct)
        rng = (float(lo), float(hi))
    else:
        lo, hi = np.min(all_vals), np.max(all_vals)
        rng = (float(lo), float(hi))

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, arr in arrs.items():
        ax.hist(
            arr,
            bins=bins,
            range=rng,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=f"{label}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if title:
        ax.set_title(title)

    ax.legend(loc="upper left", fontsize=10)

    # Inset: when plotting the full min/max range, zoom into the normal bulk.
    if add_inset and not percentile_plot:
        zoom_lo, zoom_hi = np.percentile(all_vals, inset_pct)
        axins = inset_axes(ax, width="45%", height="45%", loc="upper right")

        for label, arr in arrs.items():
            axins.hist(
                arr,
                bins=bins,
                range=(float(zoom_lo), float(zoom_hi)),
                density=True,
                histtype="step",
                linewidth=1.2,
            )

        axins.set_title(f"Zoom {inset_pct[0]}–{inset_pct[1]}%", fontsize=9)
        axins.tick_params(axis="both", labelsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    logger.info("Saved histogram to %s", out_path)


def plot_pixel_histogram(
    named_pixels,
    out_path,
    bins=300,
    title=None,
    add_inset=True,
    inset_pct=(0.5, 99.5),
    percentile_plot=True,
):
    """Plot distribution of denormalized pixel values, one curve per named array."""
    _plot_step_histogram(
        named_pixels,
        out_path,
        "Calibrated value",
        bins,
        title=title,
        add_inset=add_inset,
        inset_pct=inset_pct,
        percentile_plot=percentile_plot,
    )


def plot_nmae_histogram(
    named_nmae,
    out_path,
    bins=80,
    title=None,
    inset_pct=(0.5, 99.5),
    percentile_plot=True,
):
    """Plot per-image NMAE.

    Each array holds one NMAE per image of the split, so the spread shows how
    consistent a method is rather than how it scores on average.
    """
    _plot_step_histogram(
        named_nmae,
        out_path,
        "Per-image NMAE (%)",
        bins,
        title=title,
        add_inset=False,
        inset_pct=inset_pct,
        percentile_plot=percentile_plot,
    )


def summarise(results: dict) -> dict:
    """Summarise the results of a draw ensemble."""
    draws_results = results["draws_results"]
    n_draws = len(draws_results)
    summary = {"n_draws": n_draws, "n_samples": results["n_samples"]}

    # Split metrics across draws; each draw is one whole eval run.
    aggregate = {}
    for key in sorted(draws_results[0]):
        values = np.array([r[key] for r in draws_results], dtype=np.float64)
        sd = float(values.std(ddof=1)) if n_draws > 1 else 0.0
        mean = float(values.mean())
        aggregate[key] = {
            "mean": mean,
            "sd": sd,
            "cv_pct": float(100.0 * sd / abs(mean)) if mean else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            # Standard error of the mean: how well N draws pin the mean down.
            "sem": float(sd / np.sqrt(n_draws)) if n_draws > 1 else 0.0,
            "running_mean": [float(values[: k + 1].mean()) for k in range(n_draws)],
            "running_sd": [
                float(values[: k + 1].std(ddof=1)) if k > 0 else 0.0
                for k in range(n_draws)
            ],
        }
    summary["aggregate"] = aggregate

    # Spread across draws for each sample: which observations are least stable.
    per_sample = {}
    for key, metrics in results["per_sample"].items():
        maes = np.array(metrics["draw_mae"], dtype=np.float64)
        sd = float(maes.std(ddof=1)) if len(maes) > 1 else 0.0
        mse_total = float(np.mean(metrics["draw_mse"]))
        mean_pixel_variance = metrics["mean_pixel_variance"]
        per_sample[key] = {
            "mae_mean": float(maes.mean()),
            "mae_sd": sd,
            "mae_cv_pct": float(100.0 * sd / maes.mean()) if maes.mean() else 0.0,
            "mean_sigma": metrics["mean_sigma"],
            "mse_total": mse_total,
            "mse_of_draw_mean": metrics["mse_of_draw_mean"],
            "mean_pixel_variance": mean_pixel_variance,
            "variance_fraction": float(mean_pixel_variance / mse_total) if mse_total else 0.0,
            "mae_draw_mean": metrics["mae_draw_mean"],
            "crps": metrics["crps"],
        }
    summary["per_sample"] = per_sample

    # Error decomposition over the whole split: how much of the reported error
    # is sampler noise that averaging draws would remove.
    n = max(results["n_samples"], 1)
    sums = results["decomposition_sums"]
    mse_of_mean = sums["mse_of_draw_mean"] / n
    mean_var = sums["mean_pixel_variance"] / n
    # Independent measurement of the same quantity, straight from the per-draw
    # metrics -- if the accumulators are right these agree.
    measured = float(
        np.mean([np.mean(m["draw_mse"]) for m in results["per_sample"].values()])
    )
    identity = mse_of_mean + mean_var
    mae_single = float(
        np.mean([np.mean(m["draw_mae"]) for m in results["per_sample"].values()])
    )
    summary["decomposition"] = {
        "mse_total_measured": measured,
        "mse_of_draw_mean": mse_of_mean,
        "mean_pixel_variance": mean_var,
        "identity_check": identity,
        "identity_rel_gap": float(abs(identity - measured) / measured)
        if measured
        else 0.0,
        "variance_fraction": float(mean_var / measured) if measured else 0.0,
        "mae_single_draw": mae_single,
        "mae_draw_mean": sums["mae_draw_mean"] / n,
        "mean_sigma": sums["mean_sigma"] / n,
        # CRPS of a one-member ensemble is MAE, so a single draw is the
        # deterministic baseline the skill score is measured against.
        "crps": sums["crps"] / n,
        "crps_skill_score": float(1.0 - (sums["crps"] / n) / mae_single)
        if mae_single
        else 0.0,
    }

    return summary


def write_outputs(output_dir: Path, summary: dict, results: dict, config: dict) -> None:
    """Write summary.json and a readable summary.txt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, **summary, "per_draw": results["draws_results"]}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    n_draws = summary["n_draws"]
    lines = [
        f"Draw-ensemble eval: {n_draws} draws x {summary['n_samples']} sample-directions",
        f"checkpoint: {config.get('ckpt_path')}",
        f"split={config.get('dataset')} direction={config.get('direction')} "
        f"batch_size={config.get('batch_size')} dc_correct={config.get('dc_correct')} "
        f"dc_anchor={config.get('dc_anchor')}",
        "",
        "Metric spread across whole-split runs",
        "| metric | mean | sd | cv% | sem | min | max |",
    ]
    # Bidirectional runs prefix every key, so select on the suffix.
    for key in sorted(summary["aggregate"]):
        stat = summary["aggregate"][key]
        lines.append(
            f"| {key} | {stat['mean']:.6g} | {stat['sd']:.3g} | {stat['cv_pct']:.2f} "
            f"| {stat['sem']:.3g} | {stat['min']:.6g} | {stat['max']:.6g} |"
        )

    # Ranked by absolute sigma, i.e. how far a rerun moves the picture.
    fragile = sorted(summary["per_sample"].items(), key=lambda kv: -kv[1]["mean_sigma"])
    lines += [
        "",
        "Samples the model is least sure about, by mean sigma",
        "| sample | mean sigma | mae | mse total | mse of draw mean "
        "| mean pixel variance | var frac |",
    ]
    lines += [
        f"| {key} | {s['mean_sigma']:.3g} | {s['mae_mean']:.6g} "
        f"| {s['mse_total']:.4g} | {s['mse_of_draw_mean']:.4g} "
        f"| {s['mean_pixel_variance']:.4g} | {s['variance_fraction']:.3f} |"
        for key, s in fragile[:10]
    ]

    d = summary["decomposition"]
    lines += [
        "",
        f"Error decomposition over all {summary['n_samples']} sample-directions",
        "  E_s[(p-y)^2] = (p_mean-y)^2 + Var_s(p)",
        f"| mse total measured   | {d['mse_total_measured']:.6g} |",
        f"| mse of draw mean     | {d['mse_of_draw_mean']:.6g} |",
        f"| mean pixel variance  | {d['mean_pixel_variance']:.6g} |",
        f"| variance fraction    | {d['variance_fraction']:.4f} |",
        f"| mean sigma           | {d['mean_sigma']:.6g} |",
        f"| mae single draw      | {d['mae_single_draw']:.6g} |",
        f"| mae draw mean        | {d['mae_draw_mean']:.6g} |",
        f"| crps                 | {d['crps']:.6g} |",
        f"| crps skill score     | {d['crps_skill_score']:.4f} |",
        f"| identity rel. gap    | {d['identity_rel_gap']:.2e} |",
    ]

    for line in lines:
        logger.info(line)
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", output_dir / "summary.json")
