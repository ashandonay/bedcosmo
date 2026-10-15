"""DESI wavelength coverage and fitted-template activation diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from speclite.filters import load_filters

from .desi.support import lsst_support_limits, select_wavelength_support
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


def plot_coverage(training_matrix, prior_dir=None, redshift_bins=70, population_matrix=None):
    """Show measured contributors and distinguish learned from exported support."""
    with np.load(training_matrix) as data:
        wave = data["wave_rest_aa"]
        z = data["redshift"]
        targetids = data["targetid"]
        weights = data["relative_ivar"]
        valid = np.isfinite(weights) & (weights > 0)
        if "coverage_kind" not in data.files or data["coverage_kind"].item() != "measured":
            raise ValueError("Coverage diagnostic requires a measured-only training matrix")
    if prior_dir is not None:
        meta = json.loads((prior_dir / "build_provenance.json").read_text())
        with np.load(prior_dir / "desi_basis.npz") as data:
            learned = data["wave_rest_aa"]
            saved = data["wavelength_contributors"]
            exported = data["export_wave_rest_aa"]
        args = meta["arguments"]
        split_seed, train_fraction = args["split_seed"], args["train_fraction"]
        threshold = meta["factorization"]["required_wavelength_contributors"]
        lo, hi = meta["selection"]["prior_z_min"], meta["selection"]["prior_z_max"]
    else:
        split_seed, train_fraction, threshold = 42, 0.70, 100
        lo, hi = z.min(), z.max()
        exported = None
    train = np.random.default_rng(split_seed).permutation(len(z))[: int(train_fraction * len(z))]
    counts = valid[train].sum(axis=0)
    if prior_dir is None:
        support, _, _ = select_wavelength_support(
            weights[train], [1], support_rank=1, minimum_contributors=threshold
        )
        if not np.any(support):
            raise ValueError("No wavelength interval meets the measured-contributor threshold")
        learned = wave[support]
    elif not np.array_equal(counts, saved):
        raise ValueError("Training matrix/split does not match saved basis contributors")
    if population_matrix is None:
        population_wave, population_z, population_valid = wave, z, valid
    else:
        with np.load(population_matrix) as data:
            if "coverage_kind" not in data.files or data["coverage_kind"].item() != "measured":
                raise ValueError("Heatmap population matrix must also contain measured-only data")
            population_wave, population_z = data["wave_rest_aa"], data["redshift"]
            population_valid = data["relative_ivar"] > 0
            if not np.all(np.isin(targetids, data["targetid"])):
                raise ValueError("Heatmap population must contain every training target")
    edges = np.linspace(population_z.min(), population_z.max(), redshift_bins + 1)
    heat = contributor_density(population_valid, population_z, edges)
    wave_edges = np.r_[
        population_wave[0] - np.diff(population_wave)[0] / 2,
        (population_wave[:-1] + population_wave[1:]) / 2,
        population_wave[-1] + np.diff(population_wave)[-1] / 2,
    ]
    fig, (ax, bottom) = plt.subplots(2, 1, figsize=(13, 10), layout="constrained")
    mesh = ax.pcolormesh(
        wave_edges,
        edges,
        np.ma.masked_equal(heat, 0),
        cmap="Greys",
        norm=LogNorm(vmin=1, vmax=max(2, heat.max())),
        rasterized=True,
    )
    fig.colorbar(mesh, ax=ax, pad=0.015, label="Measured spectra per redshift–wavelength bin")
    zz = np.linspace(population_z.min(), population_z.max(), 500)
    blue, red, _, _ = lsst_support_limits(learned.min(), learned.max())
    ax.plot(blue / (1 + zz), zz, color="tab:blue", ls="--", label="LSST blue demand")
    ax.plot(red / (1 + zz), zz, color="tab:red", ls="--", label="LSST red demand")
    for cut in (lo, hi):
        ax.axhline(cut, color="#7b3294", ls="-.", lw=1.5)
    ax.plot([], [], color="#7b3294", ls="-.", label=f"Prior range z={lo:g}–{hi:g}")
    ax.set(
        xlabel="Rest wavelength [Å]",
        ylabel="Galaxy redshift z",
        title=f"Measured DESI coverage: {len(population_z):,} galaxies; {len(z):,} selected for training",
    )
    bottom.plot(wave, valid.sum(axis=0), color=".6", label=f"All selected galaxies (N={len(z):,})")
    bottom.plot(wave, counts, color="black", label=f"Training split (N={len(train):,})")
    bottom.axhline(
        threshold, color="tab:red", ls=":", label=f"Minimum {threshold} measured contributors"
    )
    bottom.axvspan(
        learned.min(),
        learned.max(),
        color="tab:blue",
        alpha=0.08,
        label=f"Learned support: {learned.min():,.0f}–{learned.max():,.0f} Å",
    )
    for panel in (ax, bottom):
        for cut in (learned.min(), learned.max()):
            panel.axvline(cut, color="tab:blue", ls=":", lw=1.5)
        if exported is not None:
            for cut in (exported.min(), exported.max()):
                panel.axvline(cut, color="#00856a", ls="--", lw=1.5)
            panel.plot(
                [],
                [],
                color="#00856a",
                ls="--",
                label=f"Exported template limits: {exported.min():,.0f}–{exported.max():,.0f} Å",
            )
    ax.legend(loc="upper right", fontsize=9)
    bottom.set(
        yscale="log",
        ylim=(1, len(z) * 1.3),
        xlabel="Rest wavelength [Å]",
        ylabel="Measured contributors",
        title="Contributors summed across redshift; no synthetic training pixels",
    )
    bottom.legend(fontsize=9)
    bottom.grid(alpha=0.18)
    return fig


def plot_template_extrapolation(prior_dir):
    """Plot exported tails and conditional template errors; skip unextended banks."""
    prior_dir = Path(prior_dir)
    with np.load(prior_dir / "desi_basis.npz") as saved:
        wave = saved["wave_rest_aa"]
        basis = saved["basis"]
        exported_wave = saved["export_wave_rest_aa"]
        exported_basis = saved["export_basis"]
        blue = exported_wave < wave[0]
        red = exported_wave > wave[-1]
        if not np.any(blue | red):
            return None
        ivar = saved["estimated_template_ivar"]
    if basis.shape != ivar.shape or exported_basis.shape != (len(basis), len(exported_wave)):
        raise ValueError("Learned/exported template and estimated IVAR dimensions do not match")
    if not np.isfinite(ivar).all() or np.any(ivar < 0):
        raise ValueError("Estimated template IVAR must be finite and nonnegative")
    columns = np.searchsorted(exported_wave, wave)
    if np.any(columns >= len(exported_wave)) or not np.array_equal(exported_wave[columns], wave):
        raise ValueError("Exported grid does not contain the learned template grid")
    if not np.array_equal(exported_basis[:, columns], basis):
        raise ValueError("Exported templates changed learned wavelength values")
    if not np.isfinite(exported_basis).all() or np.any(exported_basis.mean(axis=1) <= 0):
        raise ValueError("Exported templates must be finite with positive mean flux")
    metadata = json.loads((prior_dir / "build_provenance.json").read_text())
    method = metadata["template"]["edge_extrapolation"]
    extrapolation_label = {
        "constant": "Constant extrapolation",
        "powerlaw": "Power-law extrapolation",
    }[method]
    selection = metadata["selection"]
    learned = (exported_wave >= wave[0]) & (exported_wave <= wave[-1])
    edges = []
    if np.any(blue):
        edges.append(
            ("Blue", blue, wave[0], wave[0] - exported_wave[0], exported_wave <= wave[0] + 180)
        )
    if np.any(red):
        edges.append(
            ("Red", red, wave[-1], exported_wave[-1] - wave[-1], exported_wave >= wave[-1] - 180)
        )
    ncols = min(4, len(basis))
    nrows = int(np.ceil(len(basis) / ncols))
    fig = plt.figure(figsize=(4.5 * ncols, 6 * nrows + 1))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.09)
    outer = fig.add_gridspec(nrows, ncols, hspace=0.30, wspace=0.30)
    for row, component in enumerate(exported_basis):
        grid = outer[row // ncols, row % ncols].subgridspec(
            2, len(edges), height_ratios=[2, 1], hspace=0.50, wspace=0.50
        )
        full = fig.add_subplot(grid[0, :])
        panels = [(full, np.ones(len(exported_wave), dtype=bool))]
        for edge_index, (name, _, _, distance, window) in enumerate(edges):
            zoom = fig.add_subplot(grid[1, edge_index])
            zoom.set_title(f"{name} edge: +{distance:g} Å", fontsize=10)
            zoom.set_xlabel("Rest wavelength [Å]", fontsize=9)
            zoom.locator_params(axis="x", nbins=3)
            panels.append((zoom, window))
        scale = component.mean()
        flux = component / scale
        sigma = np.full(len(exported_wave), np.nan)
        sigma[columns] = (
            np.divide(1.0, np.sqrt(ivar[row]), out=np.full(len(wave), np.nan), where=ivar[row] > 0)
            / scale
        )
        for panel, window in panels:
            selected = learned & window
            panel.fill_between(
                exported_wave[selected],
                (flux - sigma)[selected],
                (flux + sigma)[selected],
                color="#2767c5",
                alpha=0.20,
                linewidth=0,
                zorder=1,
            )
            panel.plot(exported_wave[selected], flux[selected], color="#2767c5", lw=1.2, zorder=3)
            for _, tail, boundary, _, _ in edges:
                selected_tail = tail & window
                if np.any(selected_tail):
                    # Include the adjacent learned endpoint to show the saved bank's
                    # interpolation without creating a one-bin gap between colors.
                    connected = selected_tail | ((exported_wave == boundary) & window)
                    panel.plot(
                        exported_wave[connected], flux[connected], color="#e68117", lw=2.2, zorder=4
                    )
                    panel.axvspan(
                        min(exported_wave[selected_tail].min(), boundary),
                        max(exported_wave[selected_tail].max(), boundary),
                        color="#e68117",
                        alpha=0.1,
                    )
                    panel.axvline(boundary, color=".45", ls=":", lw=0.9)
            limits = exported_wave[window]
            panel.set_xlim(limits[0], limits[-1])
            panel.grid(alpha=0.16)
            panel.tick_params(labelsize=9)
        full.set_title(f"B{row + 1}")
        full.locator_params(axis="x", nbins=4)
        if row % ncols == 0:
            full.set_ylabel("Mean-normalized template flux")
    fig.suptitle(
        f"Rank-{len(basis)} templates: prior z={selection['prior_z_min']:g}–{selection['prior_z_max']:g}\n"
        f"{extrapolation_label} to {exported_wave[0]:,.0f}–{exported_wave[-1]:,.0f} Å\n"
        "blue shading = conditional ±1σ from estimated template IVAR",
        fontsize=15,
    )
    fig.legend(
        handles=[
            Line2D([], [], color="#2767c5", lw=2, label="Learned template"),
            Line2D([], [], color="#e68117", lw=2, label=extrapolation_label),
            Line2D([], [], color=".45", ls=":", label="Learned-support boundary"),
            Patch(facecolor="#2767c5", alpha=0.20, label="±1σ = 1 / √(estimated template IVAR)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
    )
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
                            flux_max=None):
    """Show templates and coefficient-weighted redshifts of quality-passing fits."""
    if flux_max is not None and (not np.isfinite(flux_max) or flux_max <= 0):
        raise ValueError("Flux maximum must be finite and positive")
    frame = pd.read_csv(prior_dir / "desi_eazy_empirical_weights.csv")
    frame = frame.loc[frame["quality_pass"] == True]  # noqa: E712
    if frame.empty:
        raise ValueError("No quality-passing fitted spectra")
    metadata = json.loads((prior_dir / "build_provenance.json").read_text())
    template_param = metadata["template"]["template_param"]
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
        if not np.isfinite(shares).all() or np.any(shares < 0):
            raise ValueError("Coefficient shares must be finite and nonnegative")
        active = shares.sum() > 0
        redshift_curves = []
        if active:
            z05, z50, z95 = weighted_redshift_summary(frame.z, shares)
            redshift_curves = [(z05, "tab:blue", "5%"), (z50, ".4", "median"),
                              (z95, "tab:red", "95%")]
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
        for z, color, label in redshift_curves:
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
        if active:
            observed.legend(fontsize=8, loc="upper right")
        else:
            observed.text(.5, .5, "No contribution", transform=observed.transAxes,
                          ha="center", va="center")
        hist = fig.add_subplot(gs[2, i])
        hist.hist(
            frame.z, bins=edges, density=True, histtype="stepfilled",
            facecolor="white", edgecolor="black", linewidth=1.3, label="All passed fits"
        )
        if active:
            hist.hist(frame.z, bins=edges, weights=shares, density=True,
                      histtype="stepfilled", color=".4", alpha=.45,
                      label="Coefficient-weighted (all fits)")
            hist.axvline(z50, color=".4", alpha=.45, ls="--", label="Weighted median")
        else:
            hist.text(.5, .9, "No contribution", transform=hist.transAxes,
                      ha="center", va="top")
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
                "--population-matrix", type=Path, default=None,
                help="Full DESI population matrix for the top heatmap; training counts use --training-matrix",
            )
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
        fig = plot_coverage(args.training_matrix, args.prior_dir, args.redshift_bins, args.population_matrix)
    else:
        fig = plot_template_redshifts(args.prior_dir, args.redshift_bins,
                                      log_flux=args.log_flux, flux_max=args.flux_max)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
