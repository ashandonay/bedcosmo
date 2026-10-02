"""DESI wavelength coverage and fitted-template activation diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from speclite.filters import load_filters

from .paths import get_desi_training_data_dir, get_prior_build_dir
from .templates import load_two_column_template, read_template_param


def contributor_density(valid, redshift, edges):
    """Count valid spectra in disjoint redshift bins, including the last edge."""
    bins = np.searchsorted(edges, redshift, side="right") - 1
    bins[redshift == edges[-1]] = len(edges) - 2
    if np.any((bins < 0) | (bins >= len(edges) - 1)):
        raise ValueError("Redshift bins must cover the entire population")
    counts = np.zeros((len(edges) - 1, valid.shape[1]), dtype=np.int64)
    np.add.at(counts, bins, valid)
    return counts


def plot_coverage(training_matrix, prior_dir=None, redshift_bins=70):
    """Show DESI contributors, optionally overlaying a saved build's support."""
    with np.load(training_matrix) as data:
        wave = data["wave_rest_aa"]
        z = data["redshift"]
        valid = np.isfinite(data["relative_ivar"]) & (data["relative_ivar"] > 0)
    if prior_dir is not None:
        meta = json.loads((prior_dir / "build_provenance.json").read_text())
        with np.load(prior_dir / "desi_basis.npz") as data:
            learned = data["wave_rest_aa"]
            saved = data["wavelength_contributors"]
        args = meta["arguments"]
        train = np.random.default_rng(args["split_seed"]).permutation(len(z))[
            : int(args["train_fraction"] * len(z))
        ]
        counts = valid[train].sum(axis=0)
        if not np.array_equal(counts, saved):
            raise ValueError("Training matrix/split does not match saved basis contributors")
    edges = np.linspace(z.min(), z.max(), redshift_bins + 1)
    heat = contributor_density(valid, z, edges)
    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 0.035],
        height_ratios=[1.25, 1],
        hspace=0.16,
        wspace=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    lower = fig.add_subplot(gs[1, 0], sharex=ax)
    cax = fig.add_subplot(gs[0, 1])
    wave_edges = np.r_[
        wave[0] - (wave[1] - wave[0]) / 2,
        (wave[:-1] + wave[1:]) / 2,
        wave[-1] + (wave[-1] - wave[-2]) / 2,
    ]
    mesh = ax.pcolormesh(
        wave_edges,
        edges,
        np.ma.masked_equal(heat, 0),
        cmap="Greys",
        norm=LogNorm(vmin=1, vmax=max(2, heat.max())),
        rasterized=True,
    )
    zz = np.linspace(z.min(), z.max(), 500)
    ax.plot(3600 / (1 + zz), zz, color="tab:blue", lw=2, label="DESI blue edge: 3600 Å observed")
    ax.plot(9824 / (1 + zz), zz, color="tab:red", lw=2, label="DESI red edge: 9824 Å observed")
    if prior_dir is not None:
        lo, hi = meta["selection"]["prior_z_min"], meta["selection"]["prior_z_max"]
        ax.axhline(lo, color=".25", ls=":", label=f"Later prior redshift cut: {lo:g}–{hi:g}")
        ax.axhline(hi, color=".25", ls=":")
    ax.legend(loc="upper right", fontsize=9)
    ax.set(ylim=(z.min(), z.max()), ylabel="DESI redshift z")
    ax.set_title(
        f"DESI contributor density in {np.median(np.diff(wave)):g} Å × "
        f"Δz={edges[1]-edges[0]:.3f} bins\n"
        f"Full population: {len(z):,} spectra; blank bins have no data"
    )
    ax.tick_params(axis="x", labelbottom=False)
    fig.colorbar(mesh, cax=cax).set_label("Spectra per redshift–wavelength bin (log scale)")
    lower.plot(wave, valid.sum(axis=0), color=".6", label=f"Full population (N={len(z):,})")
    if prior_dir is not None:
        lower.plot(
            wave,
            counts,
            color="tab:blue",
            label=f"Training spectra used to select wavelength range (N={len(train):,})",
        )
        threshold = meta["factorization"]["required_wavelength_contributors"]
        lower.axhline(
            threshold, color="tab:red", ls=":", label=f"Minimum {threshold} training contributors"
        )
        lower.axvspan(
            learned.min(),
            learned.max(),
            color="tab:blue",
            alpha=0.08,
            label=f"Retained templates: {learned.min():,.0f}–{learned.max():,.0f} Å",
        )
        for panel in (ax, lower):
            for cut in (learned.min(), learned.max()):
                panel.axvline(cut, color=".3", ls="--", lw=1.2)
    lower.set(
        xlim=(wave.min(), wave.max()),
        ylim=(1, len(z) * 1.3),
        yscale="log",
        xlabel="Rest wavelength [Å]",
        ylabel="Number of spectra with valid data",
    )
    lower.set_title(
        "Contributors summed across redshift"
        + (
            "\nDashed vertical lines mark the retained template endpoints"
            if prior_dir is not None
            else ""
        )
    )
    lower.grid(alpha=0.18)
    lower.legend(loc="lower left", fontsize=9)
    fig.suptitle(
        "DESI observations" + (" and template wavelength support" if prior_dir is not None else ""),
        y=0.975,
    )
    fig.subplots_adjust(left=0.10, right=0.89, top=0.89, bottom=0.075)
    return fig


def plot_template_redshifts(prior_dir, activation_threshold=0.1, redshift_bins=33):
    """Show templates and redshifts of quality-passing fits with a_k >= cutoff."""
    if not 0 < activation_threshold <= 1:
        raise ValueError("Activation threshold must be in (0, 1]")
    frame = pd.read_csv(prior_dir / "desi_eazy_empirical_weights.csv")
    frame = frame.loc[frame["quality_pass"] == True]  # noqa: E712
    if frame.empty:
        raise ValueError("No quality-passing fitted spectra")
    template_paths = read_template_param(prior_dir / "templates" / f"{prior_dir.name}.param")
    filters = load_filters("lsst2023-*")
    observed_limits = (
        min(f.wavelength.min() for f in filters),
        max(f.wavelength.max() for f in filters),
    )
    k = len(template_paths)
    colors = plt.get_cmap("tab10" if k <= 10 else "tab20")
    fig = plt.figure(figsize=(3 * k, 10))
    gs = fig.add_gridspec(3, k, height_ratios=[1, 2, 0.95], hspace=0.38, wspace=0.3)
    edges = np.linspace(frame.z.min(), frame.z.max(), redshift_bins + 1)
    hist_axes = []
    for i, filename in enumerate(template_paths):
        color = colors(i % colors.N)
        wave, shape = load_two_column_template(prior_dir / "templates" / filename)
        if not np.isfinite(shape).all() or shape.mean() <= 0:
            raise ValueError(f"Invalid template shape: {filename}")
        shape = shape / shape.mean()
        selected = frame.loc[frame[f"a{i+1}"] >= activation_threshold, "z"]
        if selected.empty:
            raise ValueError(f"No spectra pass activation threshold for B{i+1}")
        rest = fig.add_subplot(gs[0, i])
        rest.plot(wave, shape, color=color)
        rest.set(
            xlim=(wave.min(), wave.max()),
            title=f"B{i+1}: full rest-frame shape",
            xlabel="Rest wavelength [Å]",
        )
        pair = gs[1, i].subgridspec(2, 1, hspace=0)
        upper = fig.add_subplot(pair[0])
        lower = fig.add_subplot(pair[1], sharex=upper, sharey=upper)
        for ax, z in ((upper, selected.min()), (lower, selected.max())):
            ax.plot(wave * (1 + z), shape, color=color)
            ax.text(0.04, 0.88, f"z = {z:.3f}", transform=ax.transAxes)
            ax.set_xlim(*observed_limits)
            ax.grid(alpha=0.15)
        upper.tick_params(axis="x", labelbottom=False)
        upper.set_title("Observed shape\nLSST wavelength range")
        lower.set_xlabel("Observed wavelength [Å]")
        hist = fig.add_subplot(gs[2, i])
        hist.hist(
            frame.z, bins=edges, density=True, histtype="step", color=".6", label="All passed fits"
        )
        hist.hist(
            selected,
            bins=edges,
            density=True,
            color=color,
            alpha=0.45,
            label=f"a{i+1} ≥ {activation_threshold:g}",
        )
        hist.axvline(selected.median(), color=color, ls="--", label="Selected median")
        hist.set(xlabel="Redshift z", title=f"N = {len(selected):,}")
        hist_axes.append(hist)
        if i == 0:
            rest.set_ylabel("Template / full-grid mean")
            upper.set_ylabel("Template / full-grid mean")
            hist.set_ylabel("Probability density")
            hist.legend(fontsize=8)
    ymax = max(ax.get_ylim()[1] for ax in hist_axes)
    for ax in hist_axes:
        ax.set_ylim(0, ymax)
    fig.suptitle(f"{prior_dir.name.upper()}: fitted-template activation and redshift", y=0.99)
    fig.subplots_adjust(top=0.93, bottom=0.07)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("coverage", "template-redshifts"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--prior-dir",
            type=Path,
            default=None if name == "coverage" else get_prior_build_dir("empirical_prior/desi8"),
            help=(
                "Optional saved DESI build overlay"
                if name == "coverage"
                else "Prior build directory"
            ),
        )
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--redshift-bins", type=int, default=70 if name == "coverage" else 33)
        if name == "coverage":
            sub.add_argument(
                "--training-matrix",
                type=Path,
                default=get_desi_training_data_dir() / "desi_rest_frame_training_matrix.npz",
            )
        else:
            sub.add_argument("--activation-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.redshift_bins < 1:
        parser.error("--redshift-bins must be positive")
    if args.command == "coverage":
        fig = plot_coverage(args.training_matrix, args.prior_dir, args.redshift_bins)
    else:
        fig = plot_template_redshifts(args.prior_dir, args.activation_threshold, args.redshift_bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
