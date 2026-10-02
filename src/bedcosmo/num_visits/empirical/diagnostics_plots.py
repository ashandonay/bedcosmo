"""DESI wavelength coverage and fitted-template activation diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from speclite.filters import load_filters
from matplotlib.colors import LogNorm

from .desi.support import largest_contiguous_region, lsst_support_limits
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
        threshold = meta["factorization"]["required_wavelength_contributors"]
        lo, hi = meta["selection"]["prior_z_min"], meta["selection"]["prior_z_max"]
    else:
        train = np.random.default_rng(42).permutation(len(z))[: int(0.70 * len(z))]
        counts = valid[train].sum(axis=0)
        threshold = 100
        support = largest_contiguous_region(counts >= threshold)
        if not np.any(support):
            raise ValueError("No wavelength bins have at least 100 training contributors")
        learned = wave[support]
    lsst_blue, lsst_red, supported_lo, supported_hi = lsst_support_limits(
        learned.min(), learned.max()
    )
    if prior_dir is None:
        lo, hi = supported_lo, supported_hi
        if lo >= hi:
            raise ValueError("Retained support cannot cover all LSST filters at any redshift")
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
    ax.plot(
        lsst_blue / (1 + zz),
        zz,
        color="tab:blue",
        ls="--",
        lw=1.5,
        label=f"LSST blue edge: {lsst_blue:,.0f} Å observed",
    )
    ax.plot(
        lsst_red / (1 + zz),
        zz,
        color="tab:red",
        ls="--",
        lw=1.5,
        label=f"LSST red edge: {lsst_red:,.0f} Å observed",
    )
    # Intersections delimit complete filter coverage, not the catalog redshift cut.
    if supported_lo <= supported_hi:
        ax.scatter(
            [learned.max(), learned.min()],
            [supported_lo, supported_hi],
            color=".4",
            edgecolors="white",
            s=65,
            zorder=5,
            label=f"Full LSST support: z={supported_lo:.3f}–{supported_hi:.3f}",
        )
    ax.axhspan(lo, hi, color=".5", alpha=0.08)
    ax.axhline(lo, color=".4", ls=":", lw=1.2)
    ax.axhline(hi, color=".4", ls=":", lw=1.2)
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
    lower.plot(
        wave,
        counts,
        color="tab:blue",
        label=f"Training spectra used to select wavelength range (N={len(train):,})",
    )
    lower.axhline(
        threshold, color="tab:red", ls=":", label=f"Minimum {threshold} training contributors"
    )
    lower.axvspan(
        learned.min(),
        learned.max(),
        color="tab:blue",
        alpha=0.08,
        label=f"Retained support: {learned.min():,.0f}–{learned.max():,.0f} Å",
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
        "Contributors summed across redshift\n"
        "Dashed vertical lines mark the retained wavelength endpoints"
    )
    lower.grid(alpha=0.18)
    lower.legend(loc="lower left", fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.89, top=0.89, bottom=0.075)
    return fig


def weighted_redshift_summary(redshift, shares):
    """Weighted empirical-CDF percentiles (5/50/95)."""
    redshift = np.asarray(redshift, dtype=float)
    shares = np.asarray(shares, dtype=float)
    if (redshift.shape != shares.shape or redshift.ndim != 1
            or not np.isfinite(redshift).all() or not np.isfinite(shares).all()
            or np.any(shares < 0) or shares.sum() <= 0):
        raise ValueError("Redshifts and coefficient shares must be finite, with positive total weight")
    order = np.argsort(redshift, kind="stable")
    cumulative = np.cumsum(shares[order]) / shares.sum()
    indices = np.searchsorted(cumulative, [.05, .5, .95], side="left")
    return redshift[order][indices]


def plot_template_redshifts(prior_dir, redshift_bins=24, log_flux=False,
                            flux_max=None, template_param=None):
    """Show templates and coefficient-weighted redshifts of quality-passing fits."""
    if flux_max is not None and (not np.isfinite(flux_max) or flux_max <= 0):
        raise ValueError("Flux maximum must be finite and positive")
    frame = pd.read_csv(prior_dir / "desi_eazy_empirical_weights.csv")
    frame = frame.loc[frame["quality_pass"] == True]  # noqa: E712
    if frame.empty:
        raise ValueError("No quality-passing fitted spectra")
    template_param = template_param or f"{prior_dir.name}.param"
    template_paths = read_template_param(prior_dir / "templates" / template_param)
    filters = load_filters("lsst2023-*")
    wavelength_limits = (min(f.wavelength.min() for f in filters),
                         max(f.wavelength.max() for f in filters))
    k = len(template_paths)
    bank = [load_two_column_template(prior_dir / "templates" / path) for path in template_paths]
    _, _, z_min, z_max = lsst_support_limits(
        max(wave.min() for wave, _ in bank), min(wave.max() for wave, _ in bank)
    )
    if z_min >= z_max:
        raise ValueError("Template support cannot cover all LSST filters at any redshift")
    if frame.z.min() < z_min or frame.z.max() > z_max:
        raise ValueError("Quality-passing redshifts exceed the derived template support limits")
    fig = plt.figure(figsize=(3 * k, 9))
    gs = fig.add_gridspec(3, k, height_ratios=[1, 1.2, 0.95], hspace=0.38, wspace=0.3)
    edges = np.linspace(z_min, z_max, redshift_bins + 1)
    hist_axes = []
    wavelength_axes = []
    for i, filename in enumerate(template_paths):
        wave, shape = bank[i]
        if not np.isfinite(shape).all() or shape.mean() <= 0:
            raise ValueError(f"Invalid template shape: {filename}")
        shape = shape / shape.mean()
        if log_flux:
            shape = np.ma.masked_less_equal(shape, 0)
        shares = frame[f"a{i+1}"].to_numpy(float)
        z05, z50, z95 = weighted_redshift_summary(frame.z, shares)
        rest = fig.add_subplot(gs[0, i])
        rest.plot(wave, shape, color="black")
        if log_flux:
            rest.set_yscale("log")
        rest.set(
            title=f"Template B{i+1}",
            xlim=(wave.min(), wave.max()),
            xlabel="Rest wavelength [Å]",
        )
        observed = fig.add_subplot(gs[1, i], sharex=wavelength_axes[0] if wavelength_axes else None)
        wavelength_axes.append(observed)
        for z, color, label in (
            (z05, "tab:blue", "5%"),
            (z50, ".4", "median"),
            (z95, "tab:red", "95%"),
        ):
            observed.plot(wave * (1 + z), shape, color=color,
                          alpha=.7 if label != "median" else 1.,
                          label=f"z {label} = {z:.3f}")
        observed.set_xlabel("Observed wavelength [Å]")
        if log_flux:
            observed.set_yscale("log")
        if flux_max is not None:
            for ax in (rest, observed):
                if log_flux:
                    ax.set_ylim(top=flux_max)
                else:
                    ax.set_ylim(0, flux_max)
        observed.grid(alpha=.15)
        observed.legend(fontsize=8, loc="upper right")
        hist = fig.add_subplot(gs[2, i])
        hist.hist(
            frame.z, bins=edges, density=True, histtype="stepfilled",
            facecolor="white", edgecolor="black", linewidth=1.3, label="All passed fits"
        )
        hist.hist(frame.z, bins=edges, weights=shares, density=True,
                  histtype="stepfilled", color=".4", alpha=.45,
                  label="Coefficient-weighted (all fits)")
        hist.axvline(z50, color=".4", alpha=.45, ls="--", label="Weighted median")
        hist.set(xlabel="Redshift z", xlim=(z_min, z_max))
        hist_axes.append(hist)
        if i == 0:
            rest.set_ylabel("Template / full-grid mean")
            observed.set_ylabel("Template / full-grid mean")
            hist.set_ylabel("Probability density")
            hist.legend(fontsize=8)
    wavelength_axes[0].set_xlim(*wavelength_limits)
    ymax = max(ax.get_ylim()[1] for ax in hist_axes)
    for ax in hist_axes:
        ax.set_ylim(0, ymax)
    fig.suptitle(
        f"{prior_dir.name.upper()}: coefficient-weighted template redshifts "
        f"({len(frame):,} fitted spectra)", y=0.99
    )
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
        sub.add_argument("--redshift-bins", type=int, default=70 if name == "coverage" else 24)
        if name == "coverage":
            sub.add_argument(
                "--training-matrix",
                type=Path,
                default=get_desi_training_data_dir() / "desi_rest_frame_training_matrix.npz",
            )
        else:
            sub.add_argument("--log-flux", action="store_true",
                             help="Logarithmic template flux axes; mask nonpositive values")
            sub.add_argument("--flux-max", type=float, default=None,
                             help="Common upper y-axis limit for template panels")
    args = parser.parse_args(argv)
    if args.redshift_bins < 1:
        parser.error("--redshift-bins must be positive")
    if args.command == "coverage":
        fig = plot_coverage(args.training_matrix, args.prior_dir, args.redshift_bins)
    else:
        fig = plot_template_redshifts(args.prior_dir, args.redshift_bins,
                                      log_flux=args.log_flux, flux_max=args.flux_max)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
