"""DESI template edge fits, LSST extension, and paired magnitude diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .support import lsst_support_limits


def extrapolate_edge(
    wave, flux, ivar, edge_wave, target_wave, *, side, method="constant", window_aa=500.0
):
    """Continue one edge using retained samples and their measurement precision.

    Both methods fix the amplitude to the IVAR-weighted mean of the nearest
    100 rest-frame Å. Power-law slopes use window_aa. No uncertainty is
    assigned to the predicted tail.
    """
    wave, flux, ivar = (np.asarray(a, dtype=float) for a in (wave, flux, ivar))
    target_wave = np.asarray(target_wave, dtype=float)
    if wave.ndim != 1 or flux.shape != wave.shape or ivar.shape != wave.shape:
        raise ValueError("Wavelength, flux and IVAR must be matching 1D arrays")
    if (
        not np.isfinite(wave).all()
        or np.any(wave <= 0)
        or np.any(np.diff(wave) <= 0)
        or not np.isfinite(flux).all()
        or not np.isfinite(ivar).all()
        or np.any(ivar < 0)
        or not np.isfinite(edge_wave)
        or edge_wave <= 0
        or target_wave.ndim != 1
        or not np.isfinite(target_wave).all()
        or np.any(target_wave <= 0)
    ):
        raise ValueError("Invalid edge-fit wavelengths, flux or IVAR")
    if side not in {"blue", "red"}:
        raise ValueError("side must be 'blue' or 'red'")
    if method not in {"constant", "powerlaw"}:
        raise ValueError("Unknown edge extrapolation method")
    if not np.isfinite(window_aa) or window_aa <= 0:
        raise ValueError("window_aa must be finite and positive")
    distance = wave - edge_wave if side == "blue" else edge_wave - wave
    outside = target_wave <= edge_wave if side == "blue" else target_wave >= edge_wave
    if not np.all(outside):
        raise ValueError("Target wavelengths must lie outside the retained edge")
    available = (distance >= 0) & (ivar > 0)
    fit = available & (distance <= window_aa)
    local = available & (distance <= 100)
    if not np.any(local):
        raise ValueError("Local anchor window has no positive template IVAR")
    amplitude = float(np.average(flux[local], weights=ivar[local]))
    if method == "constant":
        return np.full(target_wave.shape, amplitude)
    return _extrapolate_powerlaw(
        wave[fit],
        flux[fit],
        ivar[fit],
        edge_wave,
        target_wave,
        anchor_amplitude=amplitude,
    )


def _extrapolate_powerlaw(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    edge_wave: float,
    target_wave: np.ndarray,
    *,
    anchor_amplitude: float,
) -> np.ndarray:
    """Fit only alpha in A (lambda / lambda_edge)**alpha with fixed local A."""
    if len(wave) < 3:
        raise ValueError("Power-law extrapolation requires at least three measured edge bins")
    log_ratio = np.log(wave / edge_wave)
    root_weight = np.sqrt(ivar)
    if not np.isfinite(anchor_amplitude) or anchor_amplitude < 0:
        raise ValueError("Power-law anchor amplitude must be finite and nonnegative")
    if anchor_amplitude == 0:
        return np.zeros_like(target_wave, dtype=float)
    log_amplitude = np.log(anchor_amplitude)

    def residual(parameters):
        prediction = np.exp(log_amplitude + parameters[0] * log_ratio)
        return (prediction - flux) * root_weight

    result = least_squares(residual, [0.0], loss="soft_l1", max_nfev=2000)
    if not result.success:
        raise ValueError(f"Power-law continuum fit failed: {result.message}")
    prediction = np.exp(log_amplitude + result.x[0] * np.log(target_wave / edge_wave))
    # A steep declining tail can underflow to zero at long extension distances.
    # Zero is a valid nonnegative template flux; overflowing fits still fail.
    if not np.all(np.isfinite(prediction)) or np.any(prediction < 0):
        raise ValueError("Power-law extrapolation produced nonfinite or negative flux")
    return prediction


def extend_basis_for_lsst(
    wave, basis, template_ivar, prior_bounds, *, method="constant", continuum_window_aa=500.0
):
    """Extend finished components, preserving every learned wavelength value."""
    template_ivar = np.asarray(template_ivar, dtype=float)
    if (
        template_ivar.shape != basis.shape
        or not np.isfinite(template_ivar).all()
        or np.any(template_ivar < 0)
    ):
        raise ValueError(
            "Estimated template IVAR must match the basis and be finite and nonnegative"
        )
    lower, upper = np.asarray(prior_bounds, dtype=float)
    if not np.isfinite([lower, upper]).all() or not 0 <= lower < upper:
        raise ValueError("Prior redshift bounds must be finite, nonnegative and ordered")
    blue, red, _, _ = lsst_support_limits(wave[0], wave[-1])
    step = float(np.median(np.diff(wave)))
    if not np.allclose(np.diff(wave), step, rtol=0, atol=1e-7):
        raise ValueError("Template extension requires a uniformly spaced learned grid")
    first = min(0, int(np.floor((blue / (1 + upper) - wave[0]) / step)))
    last = max(len(wave) - 1, int(np.ceil((red / (1 + lower) - wave[0]) / step)))
    extended_wave = wave[0] + step * np.arange(first, last + 1)
    learned_columns = slice(-first, -first + len(wave))
    extended_basis = np.zeros((len(basis), len(extended_wave)), dtype=float)
    for row, component in enumerate(basis):
        extended_basis[row, learned_columns] = component
        for side, edge in (("blue", wave[0]), ("red", wave[-1])):
            tail = extended_wave < edge if side == "blue" else extended_wave > edge
            if np.any(tail):
                extended_basis[row, tail] = extrapolate_edge(
                    wave,
                    component,
                    template_ivar[row],
                    edge,
                    extended_wave[tail],
                    side=side,
                    method=method,
                    window_aa=continuum_window_aa,
                )
    if not np.isfinite(extended_basis).all() or np.any(extended_basis < 0):
        raise ValueError("Exported template values must be finite and nonnegative")
    return extended_wave, extended_basis


def plot_template_extrapolation(prior_dir):
    """Plot exported tails and conditional template errors; skip unextended banks."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

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
        # Use the learned flux and uncertainty to scale the overview independently of tails.
        learned_limits = np.r_[flux[learned], (flux - sigma)[learned], (flux + sigma)[learned]]
        learned_limits = learned_limits[np.isfinite(learned_limits)]
        padding = 0.08 * (learned_limits.max() - learned_limits.min())
        full.set_ylim(min(0, learned_limits.min() - padding), learned_limits.max() + padding)
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


def compare_template_extensions(prior_dir, output_dir, *, powerlaw_windows=(500.0, 2000.0)):
    """Refit actual LSST tails from a completed learned bank; preserve source files."""
    import matplotlib.pyplot as plt

    prior_dir, output_dir = Path(prior_dir), Path(output_dir)
    if prior_dir.resolve() == output_dir.resolve():
        raise ValueError("Use a separate extension-comparison output directory")
    windows = tuple(float(w) for w in powerlaw_windows)
    if (
        not windows
        or len(set(windows)) != len(windows)
        or not np.isfinite(windows).all()
        or min(windows) <= 0
    ):
        raise ValueError("Power-law windows must be unique, finite and positive")
    with np.load(prior_dir / "desi_basis.npz") as saved:
        wave, basis, ivar = saved["wave_rest_aa"], saved["basis"], saved["estimated_template_ivar"]
    provenance = json.loads((prior_dir / "build_provenance.json").read_text())
    bounds = [provenance["selection"]["prior_z_min"], provenance["selection"]["prior_z_max"]]
    methods = {"constant": 100.0, **{f"powerlaw_{w:g}_anchored": w for w in windows}}
    banks = {}
    for name, width in methods.items():
        export_wave, banks[name] = extend_basis_for_lsst(
            wave,
            basis,
            ivar,
            bounds,
            method="constant" if name == "constant" else "powerlaw",
            continuum_window_aa=width,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "template_extensions.npz",
        wave=wave,
        basis=basis,
        estimated_template_ivar=ivar,
        export_wave=export_wave,
        **{f"actual_{name}": bank for name, bank in banks.items()},
    )
    metadata = {
        "prior_dir": str(prior_dir.resolve()),
        "basis_sha256": hashlib.sha256((prior_dir / "desi_basis.npz").read_bytes()).hexdigest(),
        "prior_bounds": bounds,
        "fit_windows_rest_aa": methods,
        "amplitude_window_rest_aa": 100.0,
        "learned_bounds_rest_aa": [float(wave[0]), float(wave[-1])],
        "export_bounds_rest_aa": [float(export_wave[0]), float(export_wave[-1])],
        "description": "Actual full-prior LSST extension; no artificial holdout, no NMF or coefficient-prior refit; learned values preserved exactly.",
    }
    (output_dir / "template_extensions.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig = _plot_template_comparison(
        export_wave,
        banks,
        wave,
        basis,
        ivar,
        wave[0],
        wave[-1],
        f"DESI templates: actual LSST extension for z={bounds[0]:.4f}–{bounds[1]:.4f}",
    )
    fig.savefig(output_dir / "templates_actual.png", dpi=170)
    fig.savefig(output_dir / "templates_actual.pdf")
    plt.close(fig)
    return export_wave, banks


def _comparison_labels(methods):
    palette = ["#269763", "#df761b", "#8c4ab0", "#c54d52", "#8a6846"]
    colors, labels = {"constant": "#2868bd"}, {"constant": "Constant (100 Å)"}
    for index, method in enumerate(m for m in methods if m != "constant"):
        labels[method] = f"PL {method.split('_')[1]} Å"
        colors[method] = {
            "500": "#269763",
            "1000": "#df761b",
            "2000": "#8c4ab0",
        }.get(method.split("_")[1], palette[index % len(palette)])
    return colors, labels


def _plot_template_comparison(wave, banks, reference_wave, reference, ivar, first, last, title):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = list(banks)
    colors, labels = _comparison_labels(methods)
    ncols = min(4, len(reference))
    nrows = int(np.ceil(len(reference) / ncols))
    fig = plt.figure(figsize=(5 * ncols, 6 * nrows))
    grid = fig.add_gridspec(
        nrows, ncols, left=0.055, right=0.99, bottom=0.11, top=0.87, wspace=0.25, hspace=0.38
    )

    norm = reference.mean(axis=1)
    sigma = np.zeros_like(reference)
    np.divide(1, np.sqrt(ivar), out=sigma, where=ivar > 0)
    for k in range(len(reference)):
        sub = grid[k // ncols, k % ncols].subgridspec(
            2, 2, height_ratios=[1.55, 1], wspace=0.3, hspace=0.48
        )
        full = fig.add_subplot(sub[0, :])
        blue = fig.add_subplot(sub[1, 0])
        red = fig.add_subplot(sub[1, 1])
        for ax in [full, blue, red]:
            ax.plot(
                reference_wave,
                reference[k] / norm[k],
                color="#39434d",
                lw=0.85,
                zorder=2,
            )
            valid = (reference_wave >= first) & (reference_wave <= last) & (ivar[k] > 0)
            if ax is full:
                ax.fill_between(
                    reference_wave,
                    np.where(valid, (reference[k] - sigma[k]) / norm[k], np.nan),
                    np.where(valid, (reference[k] + sigma[k]) / norm[k], np.nan),
                    color="#88a9ce",
                    alpha=0.22,
                    zorder=1,
                )
            else:
                ax.errorbar(
                    reference_wave[valid],
                    reference[k, valid] / norm[k],
                    yerr=sigma[k, valid] / norm[k],
                    fmt="none",
                    ecolor="#88a9ce",
                    alpha=0.7,
                    elinewidth=0.8,
                    capsize=1.5,
                    zorder=1,
                )
            for method in methods:
                color = colors[method]
                values = banks[method][k] / norm[k]
                bmask = wave <= first
                rmask = wave >= last
                ax.plot(wave[bmask], values[bmask], color=color, lw=1.8, zorder=4)
                ax.plot(wave[rmask], values[rmask], color=color, lw=1.8, zorder=4)
            ax.axvline(first, color=".5", ls=":", lw=0.9)
            ax.axvline(last, color=".5", ls=":", lw=0.9)
            ax.grid(alpha=0.14)
            ax.tick_params(labelsize=9)
        # Keep runaway tails from setting the overview scale; edge zooms show them fully.
        learned_low = (reference[k] - sigma[k]) / norm[k]
        learned_high = (reference[k] + sigma[k]) / norm[k]
        padding = 0.08 * (learned_high.max() - learned_low.min())
        full.set(
            xlim=(wave[0], wave[-1]),
            title=f"B{k+1}",
            ylim=(min(0, learned_low.min() - padding), learned_high.max() + padding),
        )
        blue.set(xlim=(wave[0], first + 200), title="Blue edge", xlabel="Rest wavelength [Å]")
        red.set(xlim=(last - 200, wave[-1]), title="Red edge", xlabel="Rest wavelength [Å]")
        # Keep runaway extrapolation from hiding the measured edge continuum.
        for ax in [blue, red]:
            lo, hi = ax.get_xlim()
            wr = (reference_wave >= lo) & (reference_wave <= hi)
            we = (wave >= lo) & (wave <= hi)
            vals = [
                reference[k, wr] / norm[k],
                (reference[k, wr] - sigma[k, wr]) / norm[k],
                (reference[k, wr] + sigma[k, wr]) / norm[k],
            ] + [b[k, we] / norm[k] for b in banks.values()]
            minimum = min(float(v.min()) for v in vals)
            maximum = max(float(v.max()) for v in vals)
            cap = 10 * max(float(np.max(np.abs(np.concatenate(vals[:3])))), 1e-8)
            if maximum > cap:
                maximum = cap
                clipped = [labels[m] for m, b in banks.items() if np.any(b[k, we] / norm[k] > cap)]
                ax.text(
                    0.03,
                    0.97,
                    "Above range: " + ", ".join(clipped),
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7,
                )
            pad = 0.08 * max(maximum - minimum, maximum * 0.05, 1e-8)
            ax.set_ylim(min(0, minimum - pad), maximum + pad)
        if k % ncols == 0:
            full.set_ylabel("Flux / mean learned flux")
            blue.set_ylabel("Flux / mean learned flux")
    handles = (
        [Line2D([], [], color="#39434d", label="Learned template")]
        + [Line2D([], [], color=colors[m], lw=2, label=labels[m]) for m in methods]
        + [Line2D([], [], color="#88a9ce", label="±1σ from estimated template IVAR")]
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.015), fontsize=10
    )
    fig.suptitle(
        title + "\nSame learned basis; uncertainty bars = ±1σ from estimated template IVAR",
        fontsize=17,
        y=0.97,
    )
    return fig


def compare_filter_response_holdouts(
    prior_dir,
    output_dir,
    *,
    response_fractions=(0.1, 0.25, 0.5),
    powerlaw_windows=(500.0, 2000.0),
    response_tail_fraction=0.001,
    n_draws=60000,
    max_samples=1000,
    seed=20261015,
):
    """Compare paired template mixtures at redshift-specific response cutoffs.

    Cutoffs integrate lambda * R(lambda), the photon-counting response for flat
    f_lambda, without using reference spectral values. Their wavelengths are
    continuous rather than snapped to the template grid. Only the relevant edge
    is hidden; every method receives exactly the same available template bins.
    """
    import pandas as pd
    from scipy.integrate import cumulative_trapezoid
    from speclite.filters import FilterResponse, load_filter

    from ..fit_sed_prior_kde import load_sed_prior_kde, sample_sed_prior
    from ..simplex import split_feature_matrix

    levels = tuple(float(f) for f in response_fractions)
    windows = tuple(float(w) for w in powerlaw_windows)
    if (
        not levels
        or not np.isfinite(levels).all()
        or len(set(levels)) != len(levels)
        or min(levels) <= 0
        or max(levels) >= 1
    ):
        raise ValueError("Hidden response fractions must be unique and between zero and one")
    if (
        not windows
        or not np.isfinite(windows).all()
        or len(set(windows)) != len(windows)
        or min(windows) <= 0
        or n_draws < 1
        or max_samples < 1
    ):
        raise ValueError(
            "Fit windows must be unique positive widths; sample counts must be positive"
        )
    if not np.isfinite(response_tail_fraction) or not 0 <= response_tail_fraction < 0.1:
        raise ValueError("Discarded response fraction must be finite and in [0, 0.1)")
    prior_dir, output_dir = Path(prior_dir), Path(output_dir)
    if prior_dir.resolve() == output_dir.resolve():
        raise ValueError("Use a separate response-comparison output directory")
    with np.load(prior_dir / "desi_basis.npz") as saved:
        wave, basis, ivar = saved["wave_rest_aa"], saved["basis"], saved["estimated_template_ivar"]
    if (
        basis.ndim != 2
        or basis.shape != ivar.shape
        or basis.shape[1] != len(wave)
        or not np.isfinite(basis).all()
        or np.any(basis < 0)
        or not np.isfinite(ivar).all()
        or np.any(ivar < 0)
    ):
        raise ValueError("Invalid learned templates or estimated template IVAR")
    filters, grids, cutoffs, response_support = {}, {}, {}, {}
    for band in ("u", "y"):
        original = load_filter("lsst2023-" + band)
        full_obs = np.unique(
            np.r_[
                original.wavelength, np.arange(original.wavelength[0], original.wavelength[-1], 2.0)
            ]
        )
        full_integral = cumulative_trapezoid(full_obs * original(full_obs), full_obs, initial=0)
        full_cdf = full_integral / full_integral[-1]
        lower = float(np.interp(response_tail_fraction / 2, full_cdf, full_obs))
        upper = float(np.interp(1 - response_tail_fraction / 2, full_cdf, full_obs))
        # A narrow endpoint taper permits speclite's zero-ended filter convention
        # without adding unavailable spectral wavelengths to the reference.
        obs = np.unique(
            np.r_[
                lower,
                lower + 1e-5,
                full_obs[(full_obs > lower) & (full_obs < upper)],
                upper - 1e-5,
                upper,
            ]
        )
        response = original(obs)
        response[[0, -1]] = 0
        filt = FilterResponse(obs, response, meta={"group_name": "edgeeval", "band_name": band})
        integral = cumulative_trapezoid(obs * response, obs, initial=0)
        retained = float(integral[-1] / full_integral[-1])
        if retained < 1 - response_tail_fraction - 1e-5:
            raise ValueError("Cropped response loses more than the declared tolerance")
        filters[band], grids[band] = filt, obs
        response_support[band] = {
            "observed_bounds_aa": [lower, upper],
            "retained_response_fraction": retained,
            "coverage_z_min": upper / wave[-1] - 1,
            "coverage_z_max": lower / wave[0] - 1,
        }
        cdf = integral / integral[-1]
        cutoffs[band] = {
            level: float(np.interp(level if band == "u" else 1 - level, cdf, obs))
            for level in levels
        }
    kde = load_sed_prior_kde(prior_dir / "sed_prior_kde_native.joblib")
    features = sample_sed_prior(kde, n_draws, seed=seed)
    fractions, log_scale, redshift = split_feature_matrix(
        features, len(basis), parameterization=kde["parameterization"]
    )
    chosen_by_band, coverage = {}, {}
    rng = np.random.default_rng(seed)
    for band, filt in filters.items():
        covered = (filt.wavelength[0] / (1 + redshift) >= wave[0]) & (
            filt.wavelength[-1] / (1 + redshift) <= wave[-1]
        )
        eligible = covered.copy()
        for cutoff in cutoffs[band].values():
            edge = cutoff / (1 + redshift)
            eligible &= (
                (edge + max(windows) <= wave[-1])
                if band == "u"
                else (edge - max(windows) >= wave[0])
            )
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            raise ValueError(f"No eligible prior draws for {band} after response/fit coverage cuts")
        chosen_by_band[band] = rng.choice(
            candidates, min(max_samples, len(candidates)), replace=False
        )
        coverage[band] = {
            **response_support[band],
            "n_candidate_draws": len(redshift),
            "n_response_covered": int(covered.sum()),
            "response_covered_percent": float(100 * covered.mean()),
            "n_fit_eligible": len(candidates),
            "fit_eligible_percent": float(100 * eligible.mean()),
            "n_selected": len(chosen_by_band[band]),
        }
    coefficients = np.exp(log_scale)[:, None] * fractions
    methods = {"constant": 100.0, **{f"powerlaw_{w:g}_anchored": w for w in windows}}
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for band, filt in filters.items():
        chosen = chosen_by_band[band]
        for level in levels:
            cutoff_obs = cutoffs[band][level]
            # Resolve a possible amplitude jump at the cutoff without blending it
            # across a broad numerical integration interval.
            obs = np.unique(np.r_[grids[band], cutoff_obs - 1e-4, cutoff_obs, cutoff_obs + 1e-4])
            response = obs * filt(obs)
            wing = obs <= cutoff_obs if band == "u" else obs >= cutoff_obs
            realized_response = np.trapz(response[wing], obs[wing]) / np.trapz(response, obs)
            for position, idx in enumerate(chosen):
                z, coeff = redshift[idx], coefficients[idx]
                query, edge = obs / (1 + z), cutoff_obs / (1 + z)
                reference_components = np.stack([np.interp(query, wave, b) for b in basis])
                reference_flux = coeff @ reference_components * 1e-17 / (1 + z)
                reference_maggies = float(filt.get_ab_maggies(reference_flux, obs))
                if not np.isfinite(reference_maggies) or reference_maggies <= 0:
                    raise ValueError("Reference band flux is not finite and positive")
                reference_mag = -2.5 * np.log10(reference_maggies)
                flux_fraction = np.trapz((reference_flux * response)[wing], obs[wing]) / np.trapz(
                    reference_flux * response, obs
                )
                for method, width in methods.items():
                    reconstructed = reference_components.copy()
                    status, failure = "ok", ""
                    for k in range(len(basis)):
                        try:
                            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                                reconstructed[k, wing] = extrapolate_edge(
                                    wave,
                                    basis[k],
                                    ivar[k],
                                    edge,
                                    query[wing],
                                    side="blue" if band == "u" else "red",
                                    method=("constant" if method == "constant" else "powerlaw"),
                                    window_aa=width,
                                )
                        except ValueError as exc:
                            message = str(exc)
                            if not (
                                message.startswith("Power-law continuum fit failed:")
                                or message.startswith("Power-law extrapolation produced")
                                or message
                                in {
                                    "Residuals are not finite in the initial point.",
                                    "array must not contain infs or NaNs",
                                }
                            ):
                                raise
                            status, failure = "invalid_fit", f"B{k+1}: {message}"
                            break
                    reconstructed_mag = np.nan
                    if status == "ok":
                        predicted_flux = coeff @ reconstructed * 1e-17 / (1 + z)
                        maggies = float(filt.get_ab_maggies(predicted_flux, obs))
                        if np.isfinite(maggies) and maggies > 0:
                            reconstructed_mag = -2.5 * np.log10(maggies)
                        else:
                            status, failure = (
                                "invalid_photometry",
                                "Nonfinite or nonpositive reconstructed band flux",
                            )
                    records.append(
                        {
                            "draw_index": int(idx),
                            "band": band,
                            "target_response_fraction": level,
                            "realized_response_fraction": realized_response,
                            "reference_flux_fraction_hidden": flux_fraction,
                            "cutoff_observed_aa": cutoff_obs,
                            "cutoff_rest_aa": edge,
                            "z": z,
                            "method": method,
                            "reference_mag": reference_mag,
                            "reconstructed_mag": reconstructed_mag,
                            "delta_mag": reconstructed_mag - reference_mag,
                            "status": status,
                            "failure": failure,
                        }
                    )
                if (position + 1) % 100 == 0:
                    print(
                        f"Response holdout {band} {level:.0%}: {position+1}/{len(chosen)} paired draws",
                        flush=True,
                    )
            # Keep completed panels on disk during longer comparisons.
            pd.DataFrame(records).to_csv(output_dir / "response_errors.csv", index=False)
    errors = pd.DataFrame(records)
    summaries = []
    for (band, level, method), group in errors.groupby(
        ["band", "target_response_fraction", "method"]
    ):
        valid = group.status == "ok"
        delta = group.loc[valid, "delta_mag"].to_numpy()
        summaries.append(
            {
                "band": band,
                "target_response_fraction": level,
                "method": method,
                "n": len(group),
                "n_failed": int((~valid).sum()),
                "median_delta_mag": float(np.median(delta)) if len(delta) else None,
                "median_abs_delta_mag": float(np.median(abs(delta))) if len(delta) else None,
                "p95_abs_delta_mag": float(np.quantile(abs(delta), 0.95)) if len(delta) else None,
                "fraction_failed_or_abs_above_0.1": float(
                    ((~valid) | (group.delta_mag.abs() > 0.1)).mean()
                ),
                "realized_response_fraction": float(group.realized_response_fraction.iloc[0]),
            }
        )
    pd.DataFrame(summaries).to_csv(output_dir / "response_summary.csv", index=False)
    np.savez_compressed(
        output_dir / "response_samples.npz",
        candidate_redshifts=redshift,
        **{f"{band}_feature_rows": features[ids] for band, ids in chosen_by_band.items()},
        **{f"{band}_draw_indices": ids for band, ids in chosen_by_band.items()},
        **{f"{band}_coefficient_fractions": fractions[ids] for band, ids in chosen_by_band.items()},
        **{f"{band}_log_c_scale": log_scale[ids] for band, ids in chosen_by_band.items()},
        **{f"{band}_redshifts": redshift[ids] for band, ids in chosen_by_band.items()},
    )
    z_edges = np.arange(
        np.floor(redshift.min() / 0.2) * 0.2, np.ceil(redshift.max() / 0.2) * 0.2 + 0.1, 0.2
    )
    binned = []
    for (band, level, method), group in errors.groupby(
        ["band", "target_response_fraction", "method"]
    ):
        assignments = np.searchsorted(z_edges, group.z.to_numpy(), side="right") - 1
        for index in range(len(z_edges) - 1):
            selected = group.loc[assignments == index]
            if selected.empty:
                continue
            valid = selected.status == "ok"
            delta = selected.loc[valid, "delta_mag"].abs()
            binned.append(
                {
                    "band": band,
                    "target_response_fraction": level,
                    "method": method,
                    "z_min": z_edges[index],
                    "z_max": z_edges[index + 1],
                    "n": len(selected),
                    "n_failed": int((~valid).sum()),
                    "median_abs_delta_mag": float(delta.median()) if len(delta) else None,
                    "p95_abs_delta_mag": float(delta.quantile(0.95)) if len(delta) else None,
                    "fraction_failed_or_abs_above_0.1": float(
                        ((~valid) | (selected.delta_mag.abs() > 0.1)).mean()
                    ),
                }
            )
    pd.DataFrame(binned).to_csv(output_dir / "response_redshift_summary.csv", index=False)
    protocol = {
        "prior_dir": str(prior_dir.resolve()),
        "basis_sha256": hashlib.sha256((prior_dir / "desi_basis.npz").read_bytes()).hexdigest(),
        "kde_sha256": hashlib.sha256(
            (prior_dir / "sed_prior_kde_native.joblib").read_bytes()
        ).hexdigest(),
        "seed": seed,
        "candidate_draws": n_draws,
        "coverage": coverage,
        "learned_bounds_rest_aa": [float(wave[0]), float(wave[-1])],
        "discarded_response_fraction_target": response_tail_fraction,
        "n_paired_draws_per_panel": {band: len(ids) for band, ids in chosen_by_band.items()},
        "response_fractions": levels,
        "fit_windows_rest_aa": methods,
        "cutoffs_observed_aa": cutoffs,
        "response_definition": "Integral lambda * R(lambda) d(lambda), normalized over the declared cropped band response; independent of reference spectral flux",
        "cutoff": "Continuous observed response quantile divided by (1+sampled redshift); no template-grid snapping",
        "selection": "Separate eligible cohort per band; identical joint draws within each band across methods and fractions; full cropped reference and maximum inward-fit coverage required",
        "evaluation": "Hide only the relevant blue/red wing, fit each learned component using retained values/IVAR, anchor at nearest 100 rest-AA weighted mean, mix identical coefficients; integrate the same cropped-band AB response for reference and reconstruction",
        "failures": "Numerically invalid fits/photometry remain as failed rows; summaries count failures separately and include them in the outlier rate",
        "limitations": [
            "Reference is a learned-template mixture, not independent observed truth.",
            "Matching response fraction does not match hidden galaxy-flux fraction.",
            "Equal response does not imply equal extrapolation distance or continuum complexity.",
            "Each band-specific eligible cohort is a subset of the full prior; aggregate u/y errors use different galaxy populations.",
            "Discarded response fraction is not a bound on omitted spectral-flux fraction or full-band magnitude error.",
            "Template uncertainties are conditional on fixed fitted coefficients; draws share one learned bank.",
        ],
        "summary": summaries,
    }
    # JSON object keys must be strings; preserve numerical fractions in each record.
    protocol["cutoffs_observed_aa"] = {
        b: {str(f): c for f, c in cuts.items()} for b, cuts in cutoffs.items()
    }
    (output_dir / "response_comparison.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return errors


def plot_filter_response_holdouts(output_dir):
    """Show saved holdout boundaries on observed-frame LSST photon responses."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from speclite.filters import load_filter

    output_dir = Path(output_dir)
    meta = json.loads((output_dir / "response_comparison.json").read_text())
    levels = meta["response_fractions"]
    fig, axes = plt.subplots(2, len(levels), figsize=(5 * len(levels), 7), squeeze=False)
    for row, band in enumerate(("u", "y")):
        filt = load_filter("lsst2023-" + band)
        lower, upper = meta["coverage"][band]["observed_bounds_aa"]
        wave = np.linspace(lower, upper, 12000)
        response = wave * filt(wave)
        response /= response.max()
        main = wave[response > 0.01]
        padding = 0.08 * (main[-1] - main[0])
        for col, level in enumerate(levels):
            ax = axes[row, col]
            cut = meta["cutoffs_observed_aa"][band][str(level)]
            grid = np.unique(np.r_[wave, cut])
            curve = grid * filt(grid) / np.max(wave * filt(wave))
            hidden = grid <= cut if band == "u" else grid >= cut
            ax.fill_between(grid, 0, curve, color=".8", alpha=0.5)
            ax.fill_between(grid, 0, curve, where=hidden, color="#df761b", alpha=0.65)
            ax.plot(grid, curve, color=".25", lw=1.5)
            ax.axvline(cut, color="#b35d16", ls="--", lw=1.3)
            ax.text(
                cut,
                1.03,
                f"{cut:.0f} Å",
                ha="center",
                va="bottom",
                fontsize=10,
            )
            ax.set(
                title=f"{band}: {level:.0%} held out",
                xlabel="Observed wavelength [Å]",
                ylabel="Normalized photon response λR(λ)",
                xlim=(main[0] - padding, main[-1] + padding),
                ylim=(0, 1.17),
            )
            ax.grid(alpha=0.15)
    fig.suptitle("LSST filter regions used for extrapolation tests", fontsize=16)
    fig.legend(
        handles=[
            Patch(color="#df761b", alpha=0.65, label="Hidden / extrapolated"),
            Patch(color=".8", alpha=0.5, label="Retained"),
        ],
        loc="lower center",
        ncol=2,
    )
    fig.text(
        0.5,
        0.07,
        "Main-band zooms; fractions include weak tails within the retained 99.9% response.\n"
        "Same observed boundaries at every redshift; rest wavelength = observed wavelength / (1 + z).",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(output_dir / "response_holdout_regions.png", dpi=170)
    fig.savefig(output_dir / "response_holdout_regions.pdf")
    plt.close(fig)


def plot_filter_response_comparison(output_dir):
    """Show error distributions and dependence on hidden flux for each holdout."""
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "response_comparison.json").read_text())
    errors = pd.read_csv(output_dir / "response_errors.csv")
    methods = list(metadata["fit_windows_rest_aa"])
    colors, labels = _comparison_labels(methods)
    for level in metadata["response_fractions"]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        for row, band in enumerate(("u", "y")):
            panel = errors[
                (errors.band == band) & np.isclose(errors.target_response_fraction, level)
            ]
            valid = panel[panel.status == "ok"]
            if valid.empty:
                raise ValueError(f"No finite comparison values for {band}, response={level}")
            distribution, hidden_flux = axes[row]
            # Symmetric-log bin spacing resolves the central error distribution
            # while retaining every finite catastrophic tail.
            transformed = np.arcsinh(valid.delta_mag.to_numpy() / 0.1)
            bins = 0.1 * np.sinh(
                np.linspace(transformed.min() - 0.02, transformed.max() + 0.02, 75)
            )
            distribution.set_xscale("symlog", linthresh=0.1)
            distribution.set_yscale("log")
            distribution.axvline(0, color=".5", ls="--", lw=1)
            distribution.set(
                xlabel="Δmag = reconstructed − reference",
                ylabel="Number of prior draws",
                title=band,
            )
            hidden_flux.set_yscale("symlog", linthresh=0.1)
            hidden_flux.axhline(0, color=".5", ls="--", lw=1)
            hidden_flux.set(
                xlabel=f"Reference {band}-band flux from hidden region [%]",
                ylabel="Δmag = reconstructed − reference",
            )
            counts = []
            for method in methods:
                group = panel[panel.method == method]
                finite = group[group.status == "ok"]
                distribution.hist(
                    finite.delta_mag,
                    bins=bins,
                    color=colors[method],
                    histtype="step",
                    lw=1.7,
                    label=labels[method],
                )
                hidden_flux.scatter(
                    100 * finite.reference_flux_fraction_hidden,
                    finite.delta_mag,
                    color=colors[method],
                    s=10,
                    alpha=0.35,
                    label=labels[method],
                )
                failed = int((group.status != "ok").sum())
                if failed:
                    counts.append(f"{labels[method]}: {failed} failed")
            if counts:
                distribution.text(
                    0.02,
                    0.98,
                    "\n".join(counts),
                    transform=distribution.transAxes,
                    va="top",
                    fontsize=8,
                )
            hidden_flux.legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=0.15)
        sample_counts = metadata["n_paired_draws_per_panel"]
        sample_label = (
            f"n={sample_counts['u']} per band"
            if sample_counts["u"] == sample_counts["y"]
            else f"u: n={sample_counts['u']}; y: n={sample_counts['y']}"
        )
        fig.suptitle(
            f"u/y extrapolation comparison: {level:.0%} hidden filter response; {sample_label}\n"
            "Cutoff follows sampled redshift; all finite error tails shown",
            fontsize=15,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        stem = f"response_comparison_{100*level:g}pct"
        fig.savefig(output_dir / f"{stem}.png", dpi=170)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)


def plot_filter_response_redshift_summary(output_dir):
    """Expose band-specific population selection and error rates versus redshift."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir)
    meta = json.loads((output_dir / "response_comparison.json").read_text())
    with np.load(output_dir / "response_samples.npz") as saved:
        population = saved["candidate_redshifts"]
    summary = pd.read_csv(output_dir / "response_redshift_summary.csv")
    methods = list(meta["fit_windows_rest_aa"])
    colors, labels = _comparison_labels(methods)
    edges = np.arange(0, np.ceil(population.max() / 0.1) * 0.1 + 0.05, 0.1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, band in zip(axes, ("u", "y"), strict=True):
        coverage = meta["coverage"][band]
        eligible = (population >= coverage["coverage_z_min"]) & (
            population <= coverage["coverage_z_max"]
        )
        for cut in meta["cutoffs_observed_aa"][band].values():
            edge = cut / (1 + population)
            width = max(meta["fit_windows_rest_aa"].values())
            eligible &= (
                (edge + width <= meta["learned_bounds_rest_aa"][1])
                if band == "u"
                else (edge - width >= meta["learned_bounds_rest_aa"][0])
            )
        ax.hist(
            population,
            bins=edges,
            weights=np.full(len(population), 100 / len(population)),
            color=".75",
            alpha=0.45,
            label="All candidate prior draws",
        )
        ax.hist(
            population[eligible],
            bins=edges,
            weights=np.full(int(eligible.sum()), 100 / len(population)),
            color="#2868bd",
            histtype="step",
            lw=2,
            label="Eligible for this band",
        )
        for boundary in (coverage["coverage_z_min"], coverage["coverage_z_max"]):
            if population.min() < boundary < population.max():
                ax.axvline(boundary, color=".4", ls=":", lw=1)
        ax.set(
            ylabel="Fraction of all prior draws [%]",
            title=f"{band}: {coverage['fit_eligible_percent']:.1f}% eligible; {100*coverage['retained_response_fraction']:.3f}% filter response retained",
        )
        ax.grid(alpha=0.15)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Redshift")
    fig.tight_layout()
    fig.savefig(output_dir / "response_coverage.png", dpi=170)
    fig.savefig(output_dir / "response_coverage.pdf")
    plt.close(fig)
    levels = meta["response_fractions"]
    fig, axes = plt.subplots(
        2, len(levels), figsize=(6 * len(levels), 9), squeeze=False, sharex=True, sharey=True
    )
    for row, band in enumerate(("u", "y")):
        for col, level in enumerate(levels):
            ax = axes[row, col]
            panel = summary[
                (summary.band == band) & np.isclose(summary.target_response_fraction, level)
            ]
            for method in methods:
                group = panel[panel.method == method].sort_values("z_min")
                ax.plot(
                    (group.z_min + group.z_max) / 2,
                    100 * group["fraction_failed_or_abs_above_0.1"],
                    color=colors[method],
                    lw=1.4,
                    marker="o",
                    ms=3,
                )
            counts = panel[panel.method == "constant"]
            for _, cell in counts.iterrows():
                ax.text(
                    (cell.z_min + cell.z_max) / 2,
                    0.99,
                    str(int(cell.n)),
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                )
            ax.set(
                xlabel="Redshift",
                ylabel="Error > 0.1 mag or failed fit [%]",
                ylim=(0, 105),
                title=f"{band}: {level:.0%} hidden response",
                xlim=(0, edges[-1]),
            )
            ax.grid(alpha=0.15)
    fig.suptitle(
        "Extrapolation errors by redshift\nCounts along panel tops give paired draws in each 0.2-wide redshift bin",
        fontsize=16,
    )
    fig.legend(
        handles=[Line2D([], [], color=colors[m], lw=2, label=labels[m]) for m in methods],
        loc="lower center",
        ncol=len(methods),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.90))
    fig.savefig(output_dir / "response_redshift_summary.png", dpi=170)
    fig.savefig(output_dir / "response_redshift_summary.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extend = commands.add_parser(
        "extend", help="Regenerate actual constant and power-law template tails"
    )
    extend.add_argument("--prior-dir", type=Path, required=True)
    extend.add_argument("--output-dir", type=Path, required=True)
    extend.add_argument("--powerlaw-windows", type=float, nargs="+", default=[500, 2000])
    response = commands.add_parser(
        "response-compare", help="Evaluate matched hidden u/y filter response"
    )
    response.add_argument("--prior-dir", type=Path, required=True)
    response.add_argument("--output-dir", type=Path, required=True)
    response.add_argument("--response-fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    response.add_argument("--powerlaw-windows", type=float, nargs="+", default=[500, 2000])
    response.add_argument(
        "--response-tail-fraction",
        type=float,
        default=0.001,
        help="Total photon response discarded from both tails for band coverage",
    )
    response.add_argument("--n-draws", type=int, default=60000)
    response.add_argument("--max-samples", type=int, default=1000)
    response.add_argument("--seed", type=int, default=20261015)
    redraw = commands.add_parser(
        "response-plot", help="Redraw saved response comparisons without fitting"
    )
    redraw.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "extend":
        compare_template_extensions(
            args.prior_dir, args.output_dir, powerlaw_windows=args.powerlaw_windows
        )
        return
    if args.command == "response-compare":
        compare_filter_response_holdouts(
            args.prior_dir,
            args.output_dir,
            response_fractions=args.response_fractions,
            response_tail_fraction=args.response_tail_fraction,
            powerlaw_windows=args.powerlaw_windows,
            n_draws=args.n_draws,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    plot_filter_response_holdouts(args.output_dir)
    plot_filter_response_comparison(args.output_dir)
    plot_filter_response_redshift_summary(args.output_dir)


if __name__ == "__main__":
    main()
