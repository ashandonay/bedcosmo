"""Compare joint and ILR KDE output priors in physical galaxy coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from validate_joint_empirical_mixture import fit_ilr_draws


def contour_thresholds(histogram, probabilities=(0.68, 0.95)):
    """Density thresholds enclosing requested masses on an equal-area grid."""
    flat = np.sort(np.asarray(histogram, dtype=float).ravel())[::-1]
    if np.any(flat < 0) or flat.sum() <= 0:
        raise ValueError("Contour histogram needs nonnegative, nonzero mass")
    cumulative = np.cumsum(flat) / flat.sum()
    return np.unique(
        [flat[min(np.searchsorted(cumulative, p), len(flat) - 1)] for p in probabilities]
    )


def plot_priors(comparison_dir, output_dir):
    comparison_dir, output_dir = Path(comparison_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(comparison_dir / "samples.npz") as saved:
        populations = {key: saved[key] for key in ["heldout", "joint", "ilr"]}
        names = saved["feature_names"].tolist()
    with np.load(comparison_dir / "joint_fit.npz") as saved:
        training = saved["training_rows"]
        joint_bandwidth = float(saved["bandwidth"])
    comparison_metadata = json.loads((comparison_dir / "comparison.json").read_text())
    seed = comparison_metadata["split_seed"]
    populations["ilr_narrow"] = fit_ilr_draws(
        training, names, 0.05, len(populations["joint"]), seed
    )
    labels = {
        "heldout": "Held-out DESI fits",
        "joint": f"New: joint mixture, h={joint_bandwidth:g}",
        "ilr": "Old: ILR KDE, h=0.3",
        "ilr_narrow": "Old: ILR KDE, h=0.05",
    }
    palette = {"heldout": "#333333", "joint": "#168556", "ilr": "#246bb2", "ilr_narrow": "#a35db4"}
    styles = {"heldout": ":", "joint": "-", "ilr": "-", "ilr_narrow": "--"}
    k = len(names) - 2
    pretty = {
        **{f"a{i+1}": f"Template {i+1} fraction" for i in range(k)},
        "log_c_scale": "Log flux scale",
        "z": "Redshift z",
    }
    handles = [
        Line2D([], [], color=palette[key], ls=styles[key], lw=2, label=labels[key])
        for key in populations
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), layout="constrained")
    for i, ax in enumerate(axes.flat):
        if i >= len(names):
            ax.set_visible(False)
            continue
        bins = (
            np.linspace(0, 1, 41)
            if i < k
            else np.linspace(
                min(data[:, i].min() for data in populations.values()),
                max(data[:, i].max() for data in populations.values()),
                41,
            )
        )
        for key, rows in populations.items():
            counts, _ = np.histogram(rows[:, i], bins)
            ax.stairs(counts / len(rows) * 100, bins, color=palette[key], ls=styles[key], lw=1.7)
        ax.set(xlabel=pretty[names[i]], ylabel="Population per bin (%)")
        ax.grid(alpha=0.18)
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=10)
    fig.suptitle(
        "Output prior in physical coordinates\nSame training galaxies and template bank; KDE stage, before any prior-flow approximation",
        fontsize=15,
    )
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"physical_marginals.{ext}", dpi=170)
    plt.close(fig)

    selected = ["a3", "a6", "log_c_scale", "z"]
    indices = [names.index(name) for name in selected]
    # Identical axis ranges include every plotted sample; contours are display
    # summaries, with 0.7-bin smoothing applied equally to every population.
    ranges = {
        i: (
            min(rows[:, i].min() for rows in populations.values()),
            max(rows[:, i].max() for rows in populations.values()),
        )
        for i in indices
    }
    fig, axes = plt.subplots(4, 4, figsize=(11, 10), layout="constrained")
    for row in range(4):
        for col in range(4):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            ix, iy = indices[col], indices[row]
            xedges = np.linspace(*ranges[ix], 51)
            if row == col:
                for key, data in populations.items():
                    counts, _ = np.histogram(data[:, ix], xedges)
                    ax.stairs(
                        counts / len(data) * 100, xedges, color=palette[key], ls=styles[key], lw=1.5
                    )
                ax.set_ylabel("% / bin")
            else:
                yedges = np.linspace(*ranges[iy], 51)
                for key, data in populations.items():
                    hist, _, _ = np.histogram2d(data[:, ix], data[:, iy], bins=[xedges, yedges])
                    smooth = gaussian_filter(hist, 0.7)
                    thresholds = contour_thresholds(smooth)
                    ax.contour(
                        (xedges[1:] + xedges[:-1]) / 2,
                        (yedges[1:] + yedges[:-1]) / 2,
                        smooth.T,
                        levels=thresholds,
                        colors=[palette[key]],
                        linestyles=styles[key],
                        linewidths=1.3,
                    )
                ax.set_ylim(ranges[iy])
                if col == 0:
                    ax.set_ylabel(pretty[selected[row]])
            ax.set_xlim(ranges[ix])
            if row == 3:
                ax.set_xlabel(pretty[selected[col]])
            elif row != col:
                ax.tick_params(labelbottom=False)
            ax.grid(alpha=0.12)
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle(
        "Joint structure of the output priors\nContours enclose approximately 68% and 95% of each sample population\nDisplay smoothing only: 0.7 histogram bin; exact-zero masses shown separately",
        fontsize=12,
    )
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"physical_corner.{ext}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    positions = np.arange(k)
    for j, (key, data) in enumerate(populations.items()):
        axes[0].bar(
            positions + (j - 1.5) * 0.2,
            (data[:, :k] == 0).mean(axis=0) * 100,
            width=0.2,
            color=palette[key],
            alpha=0.85,
            label=labels[key],
        )
        counts = (data[:, :k] > 0).sum(axis=1)
        histogram = np.bincount(counts, minlength=k + 1)
        axes[1].plot(
            np.arange(k + 1),
            histogram / len(data) * 100,
            color=palette[key],
            ls=styles[key],
            marker="o",
            label=labels[key],
        )
    axes[0].set(
        xticks=positions,
        xticklabels=names[:k],
        ylabel="Population with exact zero (%)",
        title="Probability that a template is absent",
    )
    axes[1].set(
        xlabel="Number of nonzero template fractions",
        ylabel="Population (%)",
        xticks=np.arange(1, k + 1),
        title="Number of active templates per draw",
        xlim=(0.8, k + 0.2),
    )
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle(
        "The structural difference: discrete zeros versus continuous near-zero weights\nILR represents every component with a positive weight, including very small weights",
        fontsize=13,
    )
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"zero_structure.{ext}", dpi=170)
    plt.close(fig)
    np.savez_compressed(
        output_dir / "physical_prior_samples.npz", **populations, feature_names=names
    )
    (output_dir / "plot_metadata.json").write_text(
        json.dumps(
            {
                "source_comparison": str(comparison_dir),
                "joint_bandwidth": joint_bandwidth,
                "ilr_bandwidths": [0.3, 0.05],
                "n_training": len(training),
                "n_per_population": {key: len(rows) for key, rows in populations.items()},
                "corner_coordinates": selected,
                "notes": [
                    "All KDEs use the same saved training population; held-out fits are a reference, not a prior model.",
                    "Joint samples preserve exact zeros. ILR samples are decoded to physical fractions for comparison.",
                    "Bandwidth units differ across model families; no prior-flow approximation is shown.",
                    "Marginal histograms include exact-zero observations in the first bin; separate zero plot distinguishes atoms from small positive values.",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    plot_priors(**vars(parser.parse_args()))
