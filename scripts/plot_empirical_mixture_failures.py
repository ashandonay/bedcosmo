"""Plot dependence failures from saved offline mixture comparison samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def binned_means(x, y, edges):
    """Return means and counts, retaining empty bins as NaN."""
    counts, _ = np.histogram(x, edges)
    totals, _ = np.histogram(x, edges, weights=y)
    means = np.divide(totals, counts, out=np.full(len(counts), np.nan), where=counts > 0)
    return means, counts


def plot_failures(samples_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(samples_path) as saved:
        names = saved["feature_names"].tolist()
        populations = [saved[key] for key in ["training", "mixture", "ilr_kde"]]
    labels = ["DESI fitted population", "Independent P0 + KDE", "Existing ILR KDE"]
    colors = ["#202020", "#d66a12", "#246bb2"]
    # Fixed pairs illustrate coefficient dependence separately from the deliberate
    # independence of coefficients and redshift in this prototype.
    pairs = [("a4", "a6"), ("z", "a6"), ("z", "a3")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), layout="constrained")
    for row, (xname, yname) in enumerate(pairs):
        ix, iy = names.index(xname), names.index(yname)
        xlim = (0, 1) if xname != "z" else (0, max(p[:, ix].max() for p in populations))
        edges_x = np.linspace(*xlim, 46)
        edges_y = np.linspace(0, 1, 46)
        for col, data in enumerate(populations):
            ax = axes[row, col]
            image = ax.hist2d(
                data[:, ix],
                data[:, iy],
                bins=[edges_x, edges_y],
                weights=np.full(len(data), 1 / len(data)),
                norm=LogNorm(vmin=1e-4, vmax=0.1),
                cmap="magma_r",
            )[3]
            means, counts = binned_means(data[:, ix], data[:, iy], edges_x)
            means[counts < 30] = np.nan
            ax.plot(
                (edges_x[:-1] + edges_x[1:]) / 2,
                means,
                color="#00b3a4",
                lw=2,
                label="Mean in bin (≥30 objects)",
            )
            r = np.corrcoef(data[:, ix], data[:, iy])[0, 1]
            ax.text(
                0.95,
                0.94,
                f"Pearson r = {r:+.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
            )
            ax.set(
                xlabel="Redshift z" if xname == "z" else f"Template fraction {xname}",
                ylabel=f"Template fraction {yname}",
                xlim=xlim,
                ylim=(0, 1),
            )
            if row == 0:
                ax.set_title(f"{labels[col]}\nn = {len(data):,}")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.colorbar(image, ax=axes, label="Fraction of population per bin (log scale)", shrink=0.7)
    fig.suptitle(
        "The independent mixture loses coefficient relationships\nSame axes and density scale across populations; exact zeros included",
        fontsize=15,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"dependence_failures.{suffix}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    k = len(names) - 2
    positions = np.arange(k)
    for i, (data, label, color) in enumerate(zip(populations, labels, colors)):
        axes[0].bar(
            positions + (i - 1) * 0.25,
            (data[:, :k] == 0).mean(axis=0) * 100,
            width=0.25,
            label=label,
            color=color,
        )
    axes[0].set(
        xticks=positions,
        xticklabels=names[:k],
        ylabel="Exactly zero (%)",
        title="Absence frequencies agree",
    )
    axes[0].legend(fontsize=8)
    for ax, name in zip(axes[1:], ["a1", "a7"]):
        idx = names.index(name)
        for data, label, color in zip(populations, labels, colors):
            positive = np.sort(data[data[:, idx] > 0, idx])
            ax.step(
                positive,
                np.arange(1, len(positive) + 1) / len(positive),
                where="post",
                label=label,
                color=color,
            )
        ax.set(
            xlabel=f"Template fraction {name}",
            ylabel="Cumulative fraction among positive values",
            xlim=(0, 1),
            ylim=(0, 1),
            title=f"{name}: positive magnitudes shift",
        )
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Matching zeros does not preserve the rest of the distribution\nILR samples have no exact zeros; their positive CDF includes smoothed near-zero weights",
        fontsize=13,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"zeros_and_positive_values.{suffix}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    plot_failures(**vars(parser.parse_args()))
