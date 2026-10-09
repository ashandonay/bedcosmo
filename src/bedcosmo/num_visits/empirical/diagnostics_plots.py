"""DESI wavelength coverage and fitted-template activation diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from speclite.filters import load_filters

from .desi.support import lsst_required_mask, lsst_support_limits, select_wavelength_support
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
    """Show DESI contributors, optionally overlaying a saved build's support."""
    with np.load(training_matrix) as data:
        wave = data["wave_rest_aa"]
        z = data["redshift"]
        targetids = data["targetid"]
        weights = data["relative_ivar"]
        valid = np.isfinite(weights) & (weights > 0)
        requested_bounds = data["prior_redshift_bounds"] if "prior_redshift_bounds" in data.files else None
    if prior_dir is not None:
        meta = json.loads((prior_dir / "build_provenance.json").read_text())
        with np.load(prior_dir / "desi_basis.npz") as data:
            learned = data["wave_rest_aa"]
            saved = data["wavelength_contributors"]
        args = meta["arguments"]
        split_seed, train_fraction = args["split_seed"], args["train_fraction"]
        train = np.random.default_rng(args["split_seed"]).permutation(len(z))[
            : int(args["train_fraction"] * len(z))
        ]
        counts = valid[train].sum(axis=0)
        if not np.array_equal(counts, saved):
            raise ValueError("Training matrix/split does not match saved basis contributors")
        threshold = meta["factorization"]["required_wavelength_contributors"]
        lo, hi = meta["selection"]["prior_z_min"], meta["selection"]["prior_z_max"]
    else:
        if requested_bounds is None:
            split_seed, train_fraction, minimum = 42, 0.70, 100
        else:
            plan = json.loads(
                (training_matrix.parent / "desi_training_matrix_provenance.json").read_text()
            )
            split_seed = plan["split_seed"]
            train_fraction = plan["train_fraction"]
            minimum = plan["minimum_wavelength_contributors"]
        train = np.random.default_rng(split_seed).permutation(len(z))[
            : int(train_fraction * len(z))
        ]
        demand = None if requested_bounds is None else lsst_required_mask(wave, requested_bounds)
        support, counts, threshold = select_wavelength_support(
            valid[train], [1], support_rank=1, minimum_contributors=minimum, required_mask=demand
        )
        if not np.any(support):
            raise ValueError("No wavelength bins have at least 100 training contributors")
        learned = wave[support]
    lsst_blue, lsst_red, supported_lo, supported_hi = lsst_support_limits(
        learned.min(), learned.max()
    )
    if prior_dir is None:
        lo, hi = (supported_lo, supported_hi) if requested_bounds is None else requested_bounds
        if lo >= hi:
            raise ValueError("Retained support cannot cover all LSST filters at any redshift")
    if population_matrix is None:
        population_wave, population_z, population_valid = wave, z, valid
    else:
        with np.load(population_matrix) as data:
            population_wave = data["wave_rest_aa"]
            population_z = data["redshift"]
            population_targetids = data["targetid"]
            population_weights = data["relative_ivar"]
            population_valid = np.isfinite(population_weights) & (population_weights > 0)
        if not np.all(np.isin(targetids, population_targetids)):
            raise ValueError("Heatmap population must contain every selected training target")
    edges = np.linspace(population_z.min(), population_z.max(), redshift_bins + 1)
    heat = contributor_density(population_valid, population_z, edges)
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1, 0.035],
        height_ratios=[1.5, .85, 1],
        hspace=0.37,
        wspace=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    bottom = fig.add_subplot(gs[2, 0], sharex=ax)
    buffer_grid = gs[1, 0].subgridspec(1, 2, wspace=0.20)
    low_panel = fig.add_subplot(buffer_grid[0, 0])
    high_panel = fig.add_subplot(buffer_grid[0, 1], sharey=low_panel)
    cax = fig.add_subplot(gs[0, 1])
    wave_edges = np.r_[
        population_wave[0] - (population_wave[1] - population_wave[0]) / 2,
        (population_wave[:-1] + population_wave[1:]) / 2,
        population_wave[-1] + (population_wave[-1] - population_wave[-2]) / 2,
    ]
    mesh = ax.pcolormesh(
        wave_edges,
        edges,
        np.ma.masked_equal(heat, 0),
        cmap="Greys",
        norm=LogNorm(vmin=1, vmax=max(2, heat.max())),
        rasterized=True,
    )
    zz = np.linspace(population_z.min(), population_z.max(), 500)
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
    prior_color, buffer_color = "#7b3294", "#00856a"
    ax.axhspan(lo, hi, color=prior_color, alpha=0.045)
    for index, cut in enumerate((lo, hi)):
        ax.axhline(
            cut, color=prior_color, ls="-.", lw=1.8,
            label=f"Requested prior: z={lo:.3f}–{hi:.3f}" if index == 0 else None,
        )
    buffer_count = np.count_nonzero((z < lo) | (z > hi))
    if buffer_count:
        for index, cut in enumerate((z.min(), z.max())):
            ax.axhline(
                cut, color=buffer_color, ls=":", lw=2,
                label=f"Training buffer limits: z={z.min():.3f}–{z.max():.3f}"
                if index == 0 else None,
            )
        for start, stop, n_buffer in (
            (z.min(), lo, np.count_nonzero(z < lo)),
            (hi, z.max(), np.count_nonzero(z > hi)),
        ):
            if start >= stop:
                continue
            ax.axhspan(start, stop, color=buffer_color, alpha=0.10)
            ax.text(
                .51, stop + .035 if stop - start < .025 else (start + stop) / 2,
                f"Buffer: Δz={stop-start:.3f}; {n_buffer:,} galaxies",
                transform=ax.get_yaxis_transform(), fontsize=9, va="center",
                color=buffer_color, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2},
            )
    ax.legend(loc="center right", fontsize=8, framealpha=.95)
    margin = .025 * (population_z.max() - population_z.min())
    ax.set(ylim=(max(0, population_z.min() - margin), population_z.max() + margin), ylabel="DESI redshift z")
    ax.set_title(
        f"DESI contributor density in {np.median(np.diff(wave)):g} Å × "
        f"Δz={edges[1]-edges[0]:.3f} bins\n"
        f"Full DESI galaxy sample: {len(population_z):,}; selected for training: {len(z):,} ({buffer_count:,} buffer)"
    )
    ax.set(xlim=(population_wave.min(), population_wave.max()), xlabel="Rest wavelength [Å]")
    fig.colorbar(mesh, cax=cax).set_label("Spectra per redshift–wavelength bin (log scale)")
    for cut in (learned.min(), learned.max()):
        ax.axvline(cut, color=".3", ls="--", lw=1.2)

    inside_rows = np.flatnonzero((z >= lo) & (z <= hi))
    baseline_train = inside_rows[np.random.default_rng(split_seed).permutation(len(inside_rows))[
        : int(train_fraction * len(inside_rows))
    ]]
    baseline_counts = valid[baseline_train].sum(axis=0)
    bottom.plot(
        wave, valid.sum(axis=0), color=".6",
        label=f"Selected population, all splits (N={len(z):,})",
    )
    bottom.plot(
        wave, counts, color=buffer_color, ls="-", lw=1.8,
        label=f"Extended z={z.min():.4f}–{z.max():.4f}: training N={len(train):,}",
    )
    bottom.plot(
        wave, baseline_counts, color=prior_color, ls="-.", lw=1.4,
        label=f"Requested z={lo:g}–{hi:g}: training N={len(baseline_train):,}",
    )
    bottom.axhline(
        threshold, color="tab:red", ls=":",
        label=f"Minimum {threshold} training contributors",
    )
    bottom.axvspan(
        learned.min(), learned.max(), color="tab:blue", alpha=.08,
        label=f"Retained support: {learned.min():,.0f}–{learned.max():,.0f} Å",
    )
    for cut in (learned.min(), learned.max()):
        bottom.axvline(cut, color=".3", ls="--", lw=1.2)
    bottom.set(
        yscale="log", ylim=(1, len(z) * 1.3), xlabel="Rest wavelength [Å]",
        ylabel="Contributors (measured + extrapolated)",
        title="Contributors summed across redshift",
    )
    bottom.legend(loc="upper center", fontsize=8)
    bottom.grid(alpha=.18)

    required_columns = np.flatnonzero(lsst_required_mask(wave, np.array([lo, hi])))
    training_z = z[train]
    peak_count = threshold
    for panel, prior_cut, buffer_cut, column, red_edge in (
        (low_panel, lo, z.min(), required_columns[-1], True),
        (high_panel, hi, z.max(), required_columns[0], False),
    ):
        # Hold membership of the final training split fixed while revealing its buffer.
        candidates = np.sort(np.unique(np.r_[prior_cut, buffer_cut, training_z[
            (training_z >= min(prior_cut, buffer_cut)) &
            (training_z <= max(prior_cut, buffer_cut))
        ]]))
        contributors = valid[train, column]
        totals = np.array([
            np.count_nonzero(contributors & (training_z >= bound if red_edge else training_z <= bound))
            for bound in candidates
        ])
        color = "tab:red" if red_edge else "tab:blue"
        panel.step(candidates, totals, where="pre" if red_edge else "post", color=color, lw=2)
        final_count = int(np.count_nonzero(contributors))
        baseline_count = int(np.count_nonzero(
            contributors & (training_z >= prior_cut if red_edge else training_z <= prior_cut)
        ))
        panel.axhline(threshold, color=".35", ls="--", lw=1.2)
        panel.text(.03, .91, f"Required: {threshold}", transform=panel.transAxes, fontsize=10)
        panel.axvline(prior_cut, color=prior_color, ls="-.", lw=1.5)
        panel.axvline(buffer_cut, color=buffer_color, ls=":", lw=1.8)
        panel.axvspan(prior_cut, buffer_cut, color=buffer_color, alpha=.08)
        panel.scatter([prior_cut], [baseline_count], color=prior_color, s=40, zorder=5)
        panel.annotate(
            f"Chosen z={buffer_cut:.4f}\n{final_count} contributors",
            xy=(buffer_cut, final_count), xytext=(-10, 20), textcoords="offset points",
            ha="right", fontsize=10, color=buffer_color,
        )
        panel.annotate(
            f"Start z={prior_cut:g}\n{baseline_count} contributors",
            xy=(prior_cut, baseline_count), xytext=(10, 12), textcoords="offset points",
            ha="left", fontsize=10, color=prior_color,
        )
        span = max(abs(buffer_cut - prior_cut), .001)
        if red_edge:
            panel.set_xlim(prior_cut + .06 * span, buffer_cut - .06 * span)
        else:
            panel.set_xlim(prior_cut - .06 * span, buffer_cut + .06 * span)
        panel.set(
            xlabel="Training lower bound z (decreases →)" if red_edge else "Training upper bound z (increases →)",
            title=("Low-z buffer supplies the red edge" if red_edge else "High-z buffer supplies the blue edge")
            + f"\nRequired rest wavelength: {wave[column]:,.0f} Å",
        )
        panel.ticklabel_format(axis="x", style="plain", useOffset=False)
        panel.grid(alpha=.15)
        peak_count = max(peak_count, final_count)
    low_panel.set(ylabel="Cumulative training contributors", ylim=(0, peak_count * 1.45))
    high_panel.tick_params(labelleft=False)
    fig.text(
        .495, .30,
        "Final training split held fixed; curves reveal contributions as each buffer is included.\n"
        "The build checks ≥100 contributors at every required wavelength, not only these endpoints.",
        ha="center", fontsize=10, color=".3",
    )
    fig.subplots_adjust(left=0.10, right=0.89, top=0.92, bottom=0.06)
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
