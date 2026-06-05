#!/usr/bin/env python3
"""
Fit EAZY rest-frame template mixtures to DESI HEALPix coadd spectra.

This estimates empirical template-mixture coefficients a_k for galaxies:

    f_DESI(lambda_obs) ~= sum_k c_k T_k(lambda_obs / (1 + z)) / (1 + z)

with c_k >= 0, then stores normalized mixture weights

    a_k = c_k / sum_j c_j   (NNLS; WLS uses sum_j |c_j| in the denominator)

Fits store raw c_k, normalized a_k, scale s = sum_k |c_k| (L1), and log s = log(s).
Optional --coeff-norm max uses a_k = c_k / max_j |c_j| instead of L1 (rare).
Prior coordinates: (a_1, ..., a_K, log s, z) with c_k = exp(log s) * a_k.

Rows with poor fit quality (default: chi2/dof > 1.2) are written to dropped_fits.csv
and excluded from prior-quality triangles. See README.md.

Example:

  python fit_eazy_weights_to_desi.py \\
    --healpix 23040 \\
    --build-name empirical_prior_full \\
    --plot-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from tqdm import tqdm

from .desi_data import ensure_desi_healpix, get_local_desi_paths
from .paths import (
    DEFAULT_EMPIRICAL_PRIOR_DIR,
    add_desi_dir_argument,
    get_healpix_fit_dir,
    resolve_desi_dir,
)
from .templates import (
    DEFAULT_PARAM_12D,
    DEFAULT_TEMPLATES_DIR,
    load_eazy_templates,
)

DEFAULT_PARAM_6D = "templates/eazy_v1.0.spectra.param"
EAZY_TEMPLATES_DIR = DEFAULT_TEMPLATES_DIR


def read_redrock(redrock_path: Path):
    """Read REDSHIFTS table."""
    return fits.getdata(redrock_path, "REDSHIFTS")


def read_fibermap(coadd_path: Path):
    """Read FIBERMAP table from coadd file."""
    return fits.getdata(coadd_path, "FIBERMAP")


def read_desi_arms(coadd_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """
    Read B/R/Z wavelength, flux, ivar, mask arrays.

    Flux/ivar/mask shapes are usually (n_target, n_wave).
    Wavelength shape is (n_wave,).
    """
    arms: dict[str, dict[str, np.ndarray]] = {}

    with fits.open(coadd_path, memmap=True) as hdul:
        for arm in ["B", "R", "Z"]:
            required = [
                f"{arm}_WAVELENGTH",
                f"{arm}_FLUX",
                f"{arm}_IVAR",
                f"{arm}_MASK",
            ]
            missing = [name for name in required if name not in hdul]
            if missing:
                raise KeyError(f"Missing HDUs in {coadd_path}: {missing}")

            arms[arm] = {
                "wave": np.asarray(hdul[f"{arm}_WAVELENGTH"].data, dtype=float),
                "flux": np.asarray(hdul[f"{arm}_FLUX"].data, dtype=float),
                "ivar": np.asarray(hdul[f"{arm}_IVAR"].data, dtype=float),
                "mask": np.asarray(hdul[f"{arm}_MASK"].data),
            }

    return arms


def build_template_matrix_on_observed_grid(
    wave_obs: np.ndarray,
    z: float,
    template_waves: list[np.ndarray],
    template_fluxes: list[np.ndarray],
) -> np.ndarray:
    """
    Redshift rest-frame templates onto observed wavelength grid.

    T_obs(lambda_obs) = T_rest(lambda_obs / (1+z)) / (1+z)

    The 1/(1+z) factor is an f_lambda shape factor. It is degenerate with
    the fitted amplitude for a single object, but included for consistency.
    """
    wave_obs = np.asarray(wave_obs, dtype=float)
    wave_rest = wave_obs / (1.0 + float(z))

    cols = []

    for tw, tf in zip(template_waves, template_fluxes):
        tw = np.asarray(tw, dtype=float)
        tf = np.asarray(tf, dtype=float)

        if tw.ndim != 1 or tf.ndim != 1:
            raise ValueError("Template wavelength and flux arrays must be 1D")
        if len(tw) != len(tf):
            raise ValueError("Template wavelength and flux arrays must have same length")

        col = np.interp(wave_rest, tw, tf, left=0.0, right=0.0)
        col = col / (1.0 + float(z))
        cols.append(col)

    return np.stack(cols, axis=1)


def concatenate_target_spectrum(
    arms: dict[str, dict[str, np.ndarray]],
    target_row_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate B/R/Z wavelength, flux, ivar, mask arrays for one target."""
    waves = []
    fluxes = []
    ivars = []
    masks = []

    for arm in ["B", "R", "Z"]:
        waves.append(arms[arm]["wave"])
        fluxes.append(arms[arm]["flux"][target_row_index])
        ivars.append(arms[arm]["ivar"][target_row_index])
        masks.append(arms[arm]["mask"][target_row_index])

    wave = np.concatenate(waves).astype(float)
    flux = np.concatenate(fluxes).astype(float)
    ivar = np.concatenate(ivars).astype(float)
    mask = np.concatenate(masks)

    order = np.argsort(wave)
    return wave[order], flux[order], ivar[order], mask[order]


def _fit_failure(
    n_templates: int,
    reason: str,
    n_good: int = 0,
) -> dict[str, object]:
    return {
        "success": False,
        "reason": reason,
        "coeffs": np.full(n_templates, np.nan),
        "a": np.full(n_templates, np.nan),
        "c_scale": np.nan,
        "log_c_scale": np.nan,
        "amplitude": np.nan,
        "chi2": np.nan,
        "dof": 0,
        "chi2_dof": np.nan,
        "n_good": n_good,
    }


def normalize_coeffs_for_prior(
    coeffs: np.ndarray,
    method: str = "l1",
) -> tuple[np.ndarray, float, float]:
    """
    Per-galaxy normalization for prior coordinates (a_k, log s).

    l1 (default): a_k = c_k / sum|c|, s = sum|c|.
        Each |a_k| <= 1; sum_k |a_k| = 1. Avoids forcing one coeff to +/-1.
    max: a_k = c_k / max|c|, s = max|c|. At least one |a_k| = 1 exactly.

    Reconstruct with c_k = exp(log s) * a_k.
    """
    method = method.lower()
    c = np.asarray(coeffs, dtype=float)

    if method == "l1":
        scale = float(np.sum(np.abs(c)))
    elif method == "max":
        scale = float(np.max(np.abs(c)))
    else:
        raise ValueError(f"coeff_norm must be 'l1' or 'max', got {method!r}")

    if not np.isfinite(scale) or scale <= 0:
        return np.full_like(c, np.nan), np.nan, np.nan
    return c / scale, scale, float(np.log(scale))


def fit_one_spectrum(
    wave_obs: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    z: float,
    template_waves: list[np.ndarray],
    template_fluxes: list[np.ndarray],
    wave_obs_min: float | None = None,
    wave_obs_max: float | None = None,
    min_good_pixels: int = 50,
    fit_method: str = "nnls",
    coeff_norm: str = "l1",
) -> dict[str, object]:
    """
    Fit template coefficients c_k to one DESI spectrum.

    fit_method:
      nnls -- nonnegative weighted least squares (c_k >= 0)
      wls  -- unconstrained weighted least squares (signed c_k)

    Returns a dict containing:
      success, coeffs c, normalized a, c_scale, log_c_scale, amplitude,
      chi2, dof, chi2_dof, n_good
    """
    fit_method = fit_method.lower()
    if fit_method not in {"nnls", "wls"}:
        raise ValueError(f"fit_method must be 'nnls' or 'wls', got {fit_method!r}")

    M = build_template_matrix_on_observed_grid(
        wave_obs=wave_obs,
        z=z,
        template_waves=template_waves,
        template_fluxes=template_fluxes,
    )

    flux = np.asarray(flux, dtype=float)
    ivar = np.asarray(ivar, dtype=float)

    good = (
        np.isfinite(wave_obs)
        & np.isfinite(flux)
        & np.isfinite(ivar)
        & (ivar > 0)
        & (mask == 0)
        & np.all(np.isfinite(M), axis=1)
        & np.any(M > 0, axis=1)
    )

    if wave_obs_min is not None:
        good &= wave_obs >= wave_obs_min
    if wave_obs_max is not None:
        good &= wave_obs <= wave_obs_max

    n_good = int(good.sum())
    n_templates = M.shape[1]

    if n_good < max(min_good_pixels, n_templates + 1):
        return _fit_failure(n_templates, "too_few_good_pixels", n_good=n_good)

    w = np.sqrt(ivar[good])
    X = M[good, :] * w[:, None]
    y = flux[good] * w

    if fit_method == "nnls":
        try:
            coeffs, _ = nnls(X, y)
        except Exception as exc:
            return _fit_failure(n_templates, f"nnls_failed:{exc}", n_good=n_good)
    else:
        try:
            coeffs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception as exc:
            return _fit_failure(n_templates, f"wls_failed:{exc}", n_good=n_good)

        if rank < n_templates:
            return _fit_failure(n_templates, "wls_rank_deficient", n_good=n_good)

        if not np.all(np.isfinite(coeffs)):
            return _fit_failure(n_templates, "wls_nonfinite_coeffs", n_good=n_good)

    a, c_scale, log_c_scale = normalize_coeffs_for_prior(coeffs, method=coeff_norm)
    if fit_method == "nnls":
        amplitude = float(np.sum(coeffs))
        success = np.isfinite(amplitude) and amplitude > 0
        reason = "ok" if success else "nonpositive_amplitude"
    else:
        amplitude = float(np.sum(np.abs(coeffs)))
        success = np.isfinite(amplitude) and amplitude > 0
        reason = "ok" if success else "zero_l1_norm"
    if not success:
        a = np.full(n_templates, np.nan)
        c_scale = np.nan
        log_c_scale = np.nan

    resid = y - X @ coeffs
    chi2 = float(np.sum(resid**2))
    dof = max(n_good - n_templates, 1)
    chi2_dof = chi2 / dof

    return {
        "success": success,
        "reason": reason,
        "coeffs": coeffs,
        "a": a,
        "amplitude": amplitude,
        "c_scale": c_scale,
        "log_c_scale": log_c_scale,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2_dof,
        "n_good": n_good,
    }


def fit_one_spectrum_nnls(
    wave_obs: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    z: float,
    template_waves: list[np.ndarray],
    template_fluxes: list[np.ndarray],
    wave_obs_min: float | None = None,
    wave_obs_max: float | None = None,
    min_good_pixels: int = 50,
) -> dict[str, object]:
    """Backward-compatible alias for fit_one_spectrum(..., fit_method='nnls')."""
    return fit_one_spectrum(
        wave_obs=wave_obs,
        flux=flux,
        ivar=ivar,
        mask=mask,
        z=z,
        template_waves=template_waves,
        template_fluxes=template_fluxes,
        wave_obs_min=wave_obs_min,
        wave_obs_max=wave_obs_max,
        min_good_pixels=min_good_pixels,
        fit_method="nnls",
    )


def template_mixture_flux(
    wave_obs: np.ndarray,
    z: float,
    coeffs: np.ndarray,
    template_waves: list[np.ndarray],
    template_fluxes: list[np.ndarray],
) -> np.ndarray:
    """Observed-frame template mixture flux: M(z) @ c."""
    M = build_template_matrix_on_observed_grid(
        wave_obs=wave_obs,
        z=z,
        template_waves=template_waves,
        template_fluxes=template_fluxes,
    )
    return M @ np.asarray(coeffs, dtype=float)


def _iter_good_segments(
    wave: np.ndarray,
    good: np.ndarray,
    max_gap_aa: float = 80.0,
) -> list[np.ndarray]:
    """Split a sorted wavelength grid into contiguous unmasked segments."""
    idx = np.flatnonzero(good)
    if idx.size == 0:
        return []

    segments: list[list[int]] = [[int(idx[0])]]
    for j in range(1, idx.size):
        if wave[idx[j]] - wave[idx[j - 1]] > max_gap_aa:
            segments.append([int(idx[j])])
        else:
            segments[-1].append(int(idx[j]))

    return [np.asarray(seg, dtype=int) for seg in segments]


def _gaussian_smooth_segments(
    wave: np.ndarray,
    y: np.ndarray,
    good: np.ndarray,
    sigma_aa: float,
    max_gap_aa: float = 80.0,
) -> np.ndarray:
    """Gaussian smooth within each spectrograph arm segment (display only)."""
    out = np.asarray(y, dtype=float).copy()
    if sigma_aa <= 0:
        return out

    for seg in _iter_good_segments(wave, good, max_gap_aa=max_gap_aa):
        if seg.size < 3:
            continue
        dlam = float(np.median(np.diff(wave[seg])))
        if not np.isfinite(dlam) or dlam <= 0:
            continue
        sigma_pix = max(sigma_aa / dlam, 0.8)
        out[seg] = gaussian_filter1d(y[seg].astype(float), sigma=sigma_pix, mode="nearest")

    return out


def _divide_by_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    good: np.ndarray,
    cont_sigma_aa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide flux by a heavily smoothed continuum estimate (display only).

    Returns normalized flux and the continuum array.
    """
    cont = _gaussian_smooth_segments(wave, flux, good, sigma_aa=cont_sigma_aa)
    normed = np.full_like(flux, np.nan, dtype=float)
    safe = good & np.isfinite(cont) & (cont > 0)
    normed[safe] = flux[safe] / cont[safe]
    return normed, cont


def _robust_ylim(
    *arrays: np.ndarray,
    good: np.ndarray,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    pad: float = 0.08,
) -> tuple[float, float]:
    vals = np.concatenate([a[good] for a in arrays if np.any(good)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def _plot_coefficient_bars(
    ax: plt.Axes,
    coeffs: np.ndarray,
    z: float,
    *,
    fit_method: str,
) -> None:
    """Bar chart of fitted template amplitudes c_k."""
    n_templates = coeffs.size
    x = np.arange(1, n_templates + 1)
    colors = np.where(
        coeffs > 0,
        "tab:blue",
        np.where(coeffs < 0, "tab:red", "0.75"),
    )
    ax.bar(x, coeffs, color=colors, alpha=0.85, edgecolor="0.2", linewidth=0.4)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_ylabel(r"$c_k$")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in x], rotation=45, ha="right", fontsize=6)
    ax.set_title(rf"$\sum_k c_k = {np.sum(coeffs):.4g}$", fontsize=8, pad=4)
    if fit_method == "nnls":
        ax.set_ylim(bottom=0.0)


DEFAULT_MAX_CHI2_DOF = 1.2
DEFAULT_Z_MIN = 0.01


def apply_quality_cuts(
    df: pd.DataFrame,
    *,
    max_chi2_dof: float | None = DEFAULT_MAX_CHI2_DOF,
) -> pd.DataFrame:
    """Mark rows that pass fit-quality cuts for the empirical prior (column quality_pass)."""
    df = df.copy()
    qp = np.asarray(df["success"], dtype=bool)
    if max_chi2_dof is not None:
        chi2 = np.asarray(df["chi2_dof"], dtype=float)
        qp = qp & np.isfinite(chi2) & (chi2 <= max_chi2_dof)
    df["quality_pass"] = qp
    return df


def prior_quality_mask(df: pd.DataFrame) -> np.ndarray:
    if "quality_pass" in df.columns:
        return np.asarray(df["quality_pass"], dtype=bool)
    return np.asarray(df["success"], dtype=bool)


def parse_targetid_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def n_template_coeff_columns(df: pd.DataFrame) -> int:
    return len([c for c in df.columns if len(c) > 1 and c[0] == "c" and c[1:].isdigit()])


def prior_a_column_names(df: pd.DataFrame) -> list[str]:
    """CSV columns a1..aK for normalized coefficients."""
    n = n_template_coeff_columns(df)
    a_cols = [f"a{k + 1}" for k in range(n)]
    missing = [c for c in a_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing normalized coefficient columns: {missing}")
    return a_cols


def prior_arrays_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a_k, raw c_k, and log s from a weights CSV."""
    n = n_template_coeff_columns(df)
    a_cols = prior_a_column_names(df)
    c_cols = [f"c{k + 1}" for k in range(n)]
    missing_c = [c for c in c_cols if c not in df.columns]
    if missing_c:
        raise KeyError(f"Missing raw coefficient columns: {missing_c}")
    if "log_c_scale" not in df.columns:
        raise KeyError("Missing log_c_scale column")
    return (
        df[a_cols].to_numpy(dtype=float),
        df[c_cols].to_numpy(dtype=float),
        df["log_c_scale"].to_numpy(dtype=float),
    )


def outlier_scores(
    coeffs: np.ndarray,
    *,
    metric: str,
    chi2_dof: np.ndarray | None = None,
) -> np.ndarray:
    if metric == "l1":
        return np.sum(np.abs(coeffs), axis=1)
    if metric == "chi2_dof":
        if chi2_dof is None:
            raise ValueError("chi2_dof required for metric='chi2_dof'")
        return np.asarray(chi2_dof, dtype=float)
    return np.max(np.abs(coeffs), axis=1)


def filter_df_to_loaded_healpix(df: pd.DataFrame, healpix: int) -> pd.DataFrame:
    """When weights table spans multiple HEALPix, keep rows for the loaded coadd only."""
    if "healpix" not in df.columns:
        return df
    hp = int(healpix)
    sub = df[df["healpix"].astype(int) == hp]
    if len(sub) < len(df):
        print(
            f"Spectrum plots: using {len(sub)} rows on HEALPIX {hp} "
            f"({len(df) - len(sub)} rows from other patches skipped)."
        )
    return sub


def select_rows_for_spectrum_plots(
    df: pd.DataFrame,
    *,
    targetids: list[int],
    n_outliers: int,
    outlier_metric: str,
) -> pd.DataFrame:
    """Pick rows by explicit TARGETID and/or largest outlier score (successful fits)."""
    good_df = df[df["success"].astype(bool)].copy()
    if good_df.empty:
        return good_df

    n_templates = n_template_coeff_columns(good_df)
    c_cols = [f"c{k + 1}" for k in range(n_templates)]
    order: list[int] = []

    if targetids:
        for tid in targetids:
            match = good_df[good_df["targetid"].astype(np.int64) == tid]
            if match.empty:
                print(f"Warning: TARGETID {tid} not found among successful fits.")
            else:
                order.append(int(match.index[0]))

    if n_outliers > 0:
        coeffs = good_df[c_cols].to_numpy(dtype=float)
        scores = outlier_scores(
            coeffs,
            metric=outlier_metric,
            chi2_dof=good_df["chi2_dof"].to_numpy(dtype=float),
        )
        top_idx = np.argsort(scores)[-n_outliers:][::-1]
        order.extend(int(good_df.index[i]) for i in top_idx)

    if not order:
        return good_df.iloc[0:0]

    seen: set[int] = set()
    unique_order = []
    for idx in order:
        if idx not in seen:
            seen.add(idx)
            unique_order.append(idx)
    return good_df.loc[unique_order]


def write_outlier_summary(
    outdir: Path,
    df: pd.DataFrame,
    *,
    outlier_metric: str,
    filename: str = "outlier_fits_summary.csv",
) -> Path | None:
    good_df = df[df["success"].astype(bool)].copy()
    if good_df.empty:
        return None

    n_templates = n_template_coeff_columns(good_df)
    c_cols = [f"c{k + 1}" for k in range(n_templates)]
    coeffs = good_df[c_cols].to_numpy(dtype=float)
    scores = outlier_scores(
        coeffs,
        metric=outlier_metric,
        chi2_dof=good_df["chi2_dof"].to_numpy(dtype=float),
    )

    summary = good_df[
        [
            "targetid",
            "z",
            "zwarn",
            "spectype",
            "chi2_dof",
            "c_scale",
            "log_c_scale",
            "amplitude",
            "n_good",
        ]
    ].copy()
    summary["max_abs_c"] = np.max(np.abs(coeffs), axis=1)
    summary["sum_abs_c"] = np.sum(np.abs(coeffs), axis=1)
    summary["outlier_score"] = scores
    summary = summary.sort_values("outlier_score", ascending=False)
    out_path = outdir / filename
    summary.to_csv(out_path, index=False)
    return out_path


def build_spectrum_examples(
    pick: pd.DataFrame,
    *,
    targetid_to_index: dict[int, int],
    arms: object,
    template_waves: list[np.ndarray],
    template_fluxes: list[np.ndarray],
    n_templates: int,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for _, row in pick.iterrows():
        targetid = int(row["targetid"])
        coadd_idx = targetid_to_index[targetid]
        wave, flux, ivar, mask = concatenate_target_spectrum(arms, coadd_idx)
        coeffs = np.array([row[f"c{k + 1}"] for k in range(n_templates)], dtype=float)
        examples.append(
            {
                "wave": wave,
                "flux": flux,
                "ivar": ivar,
                "mask": mask,
                "z": float(row["z"]),
                "targetid": targetid,
                "coeffs": coeffs,
                "chi2_dof": float(row["chi2_dof"]),
                "template_waves": template_waves,
                "template_fluxes": template_fluxes,
            }
        )
    return examples


def save_example_spectrum_plots(
    outdir: Path,
    examples: list[dict[str, object]],
    *,
    basename: str = "spectrum_fit_examples",
    smooth_sigma_aa: float = 8.0,
    cont_sigma_aa: float = 350.0,
    normalize_continuum: bool = True,
    fit_method: str = "nnls",
) -> list[Path]:
    """
    Plot DESI coadd flux vs. template fit with fitted c_k and z for each target.

    Each example dict must contain:
      wave, flux, ivar, mask, z, targetid, coeffs, chi2_dof

    Writes:
      {basename}.png       -- shape view (optional continuum normalization)
      {basename}_flux.png  -- absolute coadd flux vs M(z) @ c
    """
    if not examples:
        return []

    outdir.mkdir(parents=True, exist_ok=True)
    n = len(examples)
    outputs: list[Path] = []

    for normalize, filename, suptitle_extra in (
        (
            normalize_continuum,
            f"{basename}.png",
            "continuum-normalized spectra (display only)",
        ),
        (
            False,
            f"{basename}_flux.png",
            "absolute coadd flux (same units as fit)",
        ),
    ):
        row_h = 3.0 if n <= 4 else 3.35
        fig = plt.figure(figsize=(13.5, row_h * n))
        hspace = 0.52 if n <= 3 else (0.62 if n <= 5 else 0.72)
        gs = fig.add_gridspec(
            nrows=n,
            ncols=2,
            width_ratios=[2.6, 1.05],
            wspace=0.35,
            hspace=hspace,
        )

        for i, ex in enumerate(examples):
            ax_spec = fig.add_subplot(gs[i, 0])
            ax_coeff = fig.add_subplot(gs[i, 1])

            wave = np.asarray(ex["wave"], dtype=float)
            flux = np.asarray(ex["flux"], dtype=float)
            ivar = np.asarray(ex["ivar"], dtype=float)
            mask = np.asarray(ex["mask"])
            z = float(ex["z"])
            targetid = int(ex["targetid"])
            coeffs = np.asarray(ex["coeffs"], dtype=float)
            chi2_dof = float(ex["chi2_dof"])

            model = template_mixture_flux(
                wave_obs=wave,
                z=z,
                coeffs=coeffs,
                template_waves=ex["template_waves"],
                template_fluxes=ex["template_fluxes"],
            )

            good = (
                np.isfinite(wave)
                & np.isfinite(flux)
                & np.isfinite(model)
                & np.isfinite(ivar)
                & (ivar > 0)
                & (mask == 0)
            )

            if normalize:
                flux_plot, cont = _divide_by_continuum(
                    wave, flux, good, cont_sigma_aa=cont_sigma_aa
                )
                model_plot = np.full_like(model, np.nan, dtype=float)
                safe = good & np.isfinite(cont) & (cont > 0) & np.isfinite(model)
                model_plot[safe] = model[safe] / cont[safe]
                ylabel = r"$f_\lambda / f_{\mathrm{cont}}$"
            else:
                flux_plot = flux.astype(float)
                model_plot = model.astype(float)
                ylabel = r"$f_\lambda$ (coadd)"

            if smooth_sigma_aa > 0:
                flux_plot = _gaussian_smooth_segments(
                    wave, flux_plot, good, sigma_aa=smooth_sigma_aa
                )
                model_plot = _gaussian_smooth_segments(
                    wave, model_plot, good, sigma_aa=smooth_sigma_aa
                )

            plot_good = good & np.isfinite(flux_plot) & np.isfinite(model_plot)
            if np.any(plot_good):
                ax_spec.plot(
                    wave[plot_good],
                    flux_plot[plot_good],
                    color="0.15",
                    lw=0.9,
                    label="DESI coadd",
                )
                ax_spec.plot(
                    wave[plot_good],
                    model_plot[plot_good],
                    color="C3",
                    lw=1.1,
                    label=r"$M(z)\,c$",
                )
                ymin, ymax = _robust_ylim(flux_plot, model_plot, good=plot_good)
                ax_spec.set_ylim(ymin, ymax)

            ax_spec.set_ylabel(ylabel)
            if i == n - 1:
                ax_spec.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
            else:
                ax_spec.tick_params(labelbottom=False)
            ax_spec.legend(loc="upper right", fontsize=7)
            ax_spec.set_title(
                f"TARGETID {targetid}   "
                rf"$z = {z:.4f}$   "
                rf"$\chi^2/\mathrm{{dof}} = {chi2_dof:.2f}$",
                fontsize=9,
            )

            _plot_coefficient_bars(ax_coeff, coeffs, z, fit_method=fit_method)
            if i < n - 1:
                ax_coeff.tick_params(labelbottom=False)
            else:
                ax_coeff.set_xlabel("Template index", fontsize=8)

        smooth_note = (
            f", smoothed ~{smooth_sigma_aa:.0f} A"
            if smooth_sigma_aa > 0
            else ""
        )
        top_margin = max(0.90, 0.97 - 0.012 * n)
        fig.tight_layout(rect=[0, 0, 1, top_margin], h_pad=0.6, w_pad=0.4)
        fig.suptitle(
            f"{suptitle_extra}{smooth_note}   fit={fit_method}",
            fontsize=10,
            y=top_margin + 0.018,
        )
        out_path = outdir / filename
        fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        outputs.append(out_path)

    return outputs


def _param_labels(n_templates: int, symbol: str) -> list[str]:
    return [rf"${symbol}_{{{k + 1}}}$" for k in range(n_templates)]


def save_raw_coeff_triangle(
    outdir: Path,
    coeffs: np.ndarray,
    z: np.ndarray,
    *,
    filename: str = "coeffs_raw_triangle.png",
    panel_size: float = 1.5,
    fit_method: str = "wls",
) -> None:
    """Corner plot of raw c_1..c_K and z (no normalization)."""
    labels = _param_labels(coeffs.shape[1], "c") + [r"$z$"]
    joint = np.column_stack([coeffs, np.asarray(z, dtype=float)])
    save_triangle_plot(
        outdir,
        joint,
        labels,
        filename=filename,
        title=rf"Raw fitted $c_k$ and $z$ ({fit_method})",
        panel_size=panel_size,
        diagonal_hist=True,
    )


def build_prior_parameter_samples(
    a: np.ndarray,
    log_c_scale: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Prior sampling coordinates: all a_k, log s, and z (no PCA)."""
    a = np.asarray(a, dtype=float)
    log_s = np.asarray(log_c_scale, dtype=float)
    z = np.asarray(z, dtype=float)
    joint = np.column_stack([a, log_s, z])
    labels = [rf"$a_{{{k + 1}}}$" for k in range(a.shape[1])] + [
        r"$\log s$",
        r"$z$",
    ]
    return joint, labels


def save_triangle_plot(
    outdir: Path,
    joint: np.ndarray,
    param_labels: list[str],
    *,
    filename: str,
    title: str,
    panel_size: float = 2.2,
    diagonal_hist: bool = True,
) -> None:
    """Lower-triangle corner plot for a joint sample matrix."""
    n_params = joint.shape[1]
    if n_params < 2 or joint.shape[0] < 3:
        return

    fig, axes = plt.subplots(
        n_params,
        n_params,
        figsize=(panel_size * n_params, panel_size * n_params),
        squeeze=False,
    )

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                if diagonal_hist:
                    vals = joint[:, i]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        ax.text(
                            0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
                        )
                    else:
                        bins = np.linspace(vals.min(), vals.max(), 31)
                        if bins[0] == bins[-1]:
                            bins = np.linspace(bins[0] - 0.5, bins[-1] + 0.5, 31)
                        ax.hist(vals, bins=bins, color="0.35", alpha=0.85, histtype="stepfilled")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        param_labels[i],
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=8,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                xj = joint[:, j]
                yi = joint[:, i]
                ok = np.isfinite(xj) & np.isfinite(yi)
                ax.scatter(
                    xj[ok],
                    yi[ok],
                    s=10,
                    alpha=0.65,
                    color="0.25",
                    edgecolors="none",
                )

            ax.tick_params(axis="both", labelsize=7)
            if i == n_params - 1:
                ax.set_xlabel(param_labels[j], fontsize=8)
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(param_labels[i], fontsize=8)
            else:
                ax.tick_params(labelleft=False)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if not ax.get_visible():
                continue
            if i != n_params - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_chi2_dof_histogram(
    outdir: Path,
    df: pd.DataFrame,
    *,
    max_chi2_dof: float | None = DEFAULT_MAX_CHI2_DOF,
    filename: str = "chi2_dof_histogram.png",
) -> None:
    """Histogram of chi2/dof for successful fits, with optional quality-cutoff line."""
    success = df["success"].astype(bool)
    if not success.any():
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    chi2 = df.loc[success, "chi2_dof"].to_numpy(dtype=float)
    chi2 = chi2[np.isfinite(chi2)]
    if chi2.size == 0:
        plt.close(fig)
        return

    bin_edges = np.histogram_bin_edges(chi2, bins=40)

    if "quality_pass" in df.columns and max_chi2_dof is not None:
        qp = df.loc[success, "quality_pass"].astype(bool).to_numpy()
        chi2_pass = chi2[qp]
        chi2_drop = chi2[~qp]
        # Shared bin edges so dropped vs passed bars align on the same x grid.
        if chi2_pass.size:
            ax.hist(
                chi2_pass,
                bins=bin_edges,
                alpha=0.85,
                color="0.35",
                histtype="stepfilled",
                label="prior-quality",
            )
        if chi2_drop.size:
            ax.hist(
                chi2_drop,
                bins=bin_edges,
                alpha=0.55,
                color="C3",
                histtype="stepfilled",
                label="dropped",
            )
        ax.axvline(
            max_chi2_dof,
            color="C3",
            ls="--",
            lw=1.8,
            label=rf"cutoff = {max_chi2_dof:g}",
        )
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.hist(chi2, bins=bin_edges, alpha=0.85, color="0.35", histtype="stepfilled")

    ax.set_xlabel(r"$\chi^2 / \mathrm{dof}$")
    ax.set_ylabel("count")
    ax.set_title("Template-mixture fit quality")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / filename, dpi=180)
    plt.close(fig)


def save_diagnostic_plots(
    outdir: Path,
    df: pd.DataFrame,
    a: np.ndarray,
    coeffs: np.ndarray,
    log_c_scale: np.ndarray,
    template_names: list[str],
    *,
    triangle_panel_size: float = 2.2,
    triangle_plots: bool = True,
    raw_coeff_triangle: bool = True,
    fit_method: str = "wls",
    coeff_norm: str = "l1",
    max_chi2_dof: float | None = DEFAULT_MAX_CHI2_DOF,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        print("No fits attempted; skipping diagnostic plots.")
        return

    good = prior_quality_mask(df)
    cg = coeffs[good]
    a_prior = a[good]
    log_s = log_c_scale[good]
    z_good = np.asarray(df.loc[good, "z"], dtype=float)

    if cg.size == 0:
        print("No prior-quality fits; skipping diagnostic plots.")
        return

    n_templates = len(template_names)

    if triangle_plots:
        joint, labels = build_prior_parameter_samples(a_prior, log_s, z_good)
        n_params = joint.shape[1]
        panel_eff = triangle_panel_size * min(6, n_templates) / max(n_params, 1)
        save_triangle_plot(
            outdir,
            joint,
            labels,
            filename="prior_params_triangle.png",
            title=(
                rf"Prior coords ({fit_method}, {coeff_norm} norm): "
                rf"$a_k$, $\log s$, $z$; $c_k = e^{{\log s}}\, a_k$"
            ),
            panel_size=max(panel_eff, 1.4),
            diagonal_hist=True,
        )

    if raw_coeff_triangle:
        panel_raw = 1.4 * min(6, n_templates) / max(n_templates + 1, 1)
        save_raw_coeff_triangle(
            outdir,
            cg,
            z_good,
            panel_size=max(panel_raw, 1.2),
            fit_method=fit_method,
        )

    save_chi2_dof_histogram(outdir, df, max_chi2_dof=max_chi2_dof)


def main() -> None:
    parser = argparse.ArgumentParser()

    add_desi_dir_argument(parser)
    parser.add_argument("--specprod", default="iron")
    parser.add_argument("--survey", default="main")
    parser.add_argument("--program", default="dark")
    parser.add_argument("--healpix", type=int, default=23040)

    parser.add_argument(
        "--build-name",
        default=DEFAULT_EMPIRICAL_PRIOR_DIR,
        help="Prior build name; fit output goes under num_visits/<build-name>/healpix/hp<HEALPIX>/.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Fit output directory (default: .../num_visits/<build-name>/healpix/hp<HEALPIX>).",
    )
    parser.add_argument("--param", default=DEFAULT_PARAM_12D)
    parser.add_argument("--overwrite-templates", action="store_true")

    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Max spectra to fit after selection (random subsample). Default: fit all passing candidates.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=DEFAULT_Z_MIN,
        help=(
            "Minimum redshift for candidate selection. "
            f"Default {DEFAULT_Z_MIN:g} drops near-zero redshifts."
        ),
    )
    parser.add_argument(
        "--no-z-min",
        dest="z_min",
        action="store_const",
        const=None,
        help="Disable the default redshift floor.",
    )
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--target-spectype", default="GALAXY")

    parser.add_argument("--require-zwarn-zero", action="store_true", default=True)
    parser.add_argument(
        "--no-require-zwarn-zero",
        dest="require_zwarn_zero",
        action="store_false",
    )

    parser.add_argument("--norm-min", type=float, default=4000.0)
    parser.add_argument("--norm-max", type=float, default=8000.0)
    parser.add_argument("--wave-obs-min", type=float, default=None)
    parser.add_argument("--wave-obs-max", type=float, default=None)
    parser.add_argument("--min-good-pixels", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--plot-n-examples",
        type=int,
        default=6,
        help="Number of successful fits to plot (random sample). 0 disables.",
    )
    parser.add_argument(
        "--plot-targetids",
        default=None,
        help="Comma-separated TARGETIDs to plot (spectrum + c_k bars).",
    )
    parser.add_argument(
        "--plot-top-outliers",
        type=int,
        default=0,
        help="Plot this many successful fits with largest outlier score.",
    )
    parser.add_argument(
        "--outlier-metric",
        choices=("max_abs", "l1", "chi2_dof"),
        default="max_abs",
        help="Score for --plot-top-outliers and outlier_fits_summary.csv ranking.",
    )
    parser.add_argument(
        "--max-chi2-dof",
        type=float,
        default=DEFAULT_MAX_CHI2_DOF,
        help="Drop successful fits with chi2/dof above this (prior outputs only).",
    )
    parser.add_argument(
        "--no-quality-cuts",
        action="store_true",
        help="Keep all successful fits in prior CSV/NPZ and diagnostic plots.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip fitting; load weights CSV from --outdir and only make plots.",
    )
    parser.add_argument(
        "--weights-csv",
        default=None,
        help="Weights table for --plot-only (default: <outdir>/desi_eazy_empirical_weights.csv).",
    )
    parser.add_argument(
        "--plot-smooth-angstrom",
        type=float,
        default=8.0,
        help="Gaussian smoothing scale for example spectrum plots (display only).",
    )
    parser.add_argument(
        "--plot-cont-angstrom",
        type=float,
        default=350.0,
        help="Continuum scale for example-plot normalization (display only).",
    )
    parser.add_argument(
        "--no-plot-normalize",
        dest="plot_normalize",
        action="store_false",
        help="Do not continuum-normalize example spectrum plots.",
    )
    parser.set_defaults(plot_normalize=True)

    parser.add_argument(
        "--triangle-panel-size",
        type=float,
        default=2.2,
        help="Inches per panel in prior-parameter triangle plot.",
    )
    parser.add_argument(
        "--no-triangle-plots",
        action="store_true",
        help="Skip prior-parameter triangle (a_k, log s, z).",
    )
    parser.add_argument(
        "--no-raw-coeff-triangle",
        action="store_true",
        help="Skip raw c_k vs z corner plot (coeffs_raw_triangle.png).",
    )
    parser.add_argument(
        "--fit-method",
        choices=("nnls", "wls"),
        default="wls",
        help=(
            "wls: unconstrained WLS (default). "
            "nnls: nonnegative coefficients."
        ),
    )
    parser.add_argument(
        "--coeff-norm",
        choices=("l1", "max"),
        default="l1",
        help=(
            "l1: a_k=c_k/sum|c| (default; softer than max, no forced +/-1). "
            "max: a_k=c_k/max|c| (at least one |a_k|=1 per galaxy)."
        ),
    )

    parser.add_argument(
        "--auto-download-desi",
        action="store_true",
        default=True,
        help="Download missing DESI coadd/redrock FITS before fitting (default: on).",
    )
    parser.add_argument(
        "--no-auto-download-desi",
        dest="auto_download_desi",
        action="store_false",
        help="Do not download DESI data; require local coadd/redrock files.",
    )

    args = parser.parse_args()

    desi_dir = resolve_desi_dir(args.desi_dir)
    outdir = (
        Path(args.outdir)
        if args.outdir
        else get_healpix_fit_dir(args.healpix, build_name=args.build_name)
    )
    outdir.mkdir(parents=True, exist_ok=True)

    if args.auto_download_desi:
        coadd_path, redrock_path = ensure_desi_healpix(
            args.healpix,
            desi_dir=desi_dir,
            specprod=args.specprod,
            survey=args.survey,
            program=args.program,
        )
    else:
        coadd_path, redrock_path = get_local_desi_paths(
            desi_dir=desi_dir,
            specprod=args.specprod,
            survey=args.survey,
            program=args.program,
            healpix=args.healpix,
        )
        if not coadd_path.exists():
            raise FileNotFoundError(
                f"Missing coadd file: {coadd_path}. Re-run with --auto-download-desi or "
                "download data via desi_get_dr_subset.py."
            )
        if not redrock_path.exists():
            raise FileNotFoundError(
                f"Missing redrock file: {redrock_path}. Re-run with --auto-download-desi or "
                "download data via desi_get_dr_subset.py."
            )

    print(f"Using coadd:   {coadd_path}")
    print(f"Using redrock: {redrock_path}")
    print(f"Fit method:    {args.fit_method}")
    print(f"Coeff norm:    {args.coeff_norm}")

    print(f"\nLoading EAZY templates from {EAZY_TEMPLATES_DIR}...")
    template_waves, template_fluxes, template_files = load_eazy_templates(
        param=args.param,
        overwrite=args.overwrite_templates,
        norm_min=args.norm_min,
        norm_max=args.norm_max,
    )

    n_templates = len(template_files)
    print(f"Loaded {n_templates} templates.")

    print("\nReading DESI redrock and coadd files...")
    rr = read_redrock(redrock_path)
    fibermap = read_fibermap(coadd_path)
    arms = read_desi_arms(coadd_path)

    fiber_targetids = np.asarray(fibermap["TARGETID"])
    targetid_to_index = {int(tid): i for i, tid in enumerate(fiber_targetids)}

    select = np.ones(len(rr), dtype=bool)

    if args.target_spectype:
        spectype = np.char.strip(np.asarray(rr["SPECTYPE"]).astype(str))
        select &= spectype == args.target_spectype

    if args.require_zwarn_zero:
        select &= np.asarray(rr["ZWARN"]) == 0

    if args.z_min is not None:
        select &= np.asarray(rr["Z"]) >= args.z_min

    if args.z_max is not None:
        select &= np.asarray(rr["Z"]) <= args.z_max

    candidate_rows = np.where(select)[0]

    candidate_rows = [
        i for i in candidate_rows
        if int(rr["TARGETID"][i]) in targetid_to_index
    ]

    rng = np.random.default_rng(args.seed)

    n_candidates = len(candidate_rows)
    if args.n_max is not None and n_candidates > args.n_max:
        print(
            f"Subsampled {n_candidates} passing candidates to {args.n_max} "
            f"(--n-max; use --n-max {n_candidates} or larger to fit all)"
        )
        candidate_rows = list(rng.choice(candidate_rows, size=args.n_max, replace=False))

    plot_targetids = parse_targetid_list(args.plot_targetids)
    want_outlier_plots = args.plot_top_outliers > 0 or bool(plot_targetids)
    want_random_examples = args.plot_n_examples > 0

    if args.plot_only:
        weights_csv = (
            Path(args.weights_csv)
            if args.weights_csv
            else outdir / "desi_eazy_empirical_weights.csv"
        )
        if not weights_csv.exists():
            raise FileNotFoundError(f"--plot-only requires weights CSV: {weights_csv}")
        df = pd.read_csv(weights_csv)
        print(f"\n--plot-only: loaded {len(df)} rows from {weights_csv}")
        max_chi2 = None if args.no_quality_cuts else args.max_chi2_dof
        df = apply_quality_cuts(df, max_chi2_dof=max_chi2)
        a_arr, coeffs_arr, log_c_scale_arr = prior_arrays_from_df(df)
        csv_path = weights_csv
        save_diagnostic_plots(
            outdir,
            df,
            a_arr,
            coeffs_arr,
            log_c_scale_arr,
            template_files,
            triangle_panel_size=args.triangle_panel_size,
            triangle_plots=not args.no_triangle_plots,
            raw_coeff_triangle=not args.no_raw_coeff_triangle,
            fit_method=args.fit_method,
            coeff_norm=args.coeff_norm,
            max_chi2_dof=None if args.no_quality_cuts else args.max_chi2_dof,
        )
    else:
        print(f"Candidate spectra to fit: {len(candidate_rows)}")

        rows = []
        all_a = []
        all_coeffs = []
        all_log_c_scale = []

        for rr_idx in tqdm(candidate_rows, desc="Fitting DESI spectra"):
            targetid = int(rr["TARGETID"][rr_idx])
            z = float(rr["Z"][rr_idx])
            coadd_idx = targetid_to_index[targetid]

            wave, flux, ivar, mask = concatenate_target_spectrum(arms, coadd_idx)

            fit = fit_one_spectrum(
                wave_obs=wave,
                flux=flux,
                ivar=ivar,
                mask=mask,
                z=z,
                template_waves=template_waves,
                template_fluxes=template_fluxes,
                wave_obs_min=args.wave_obs_min,
                wave_obs_max=args.wave_obs_max,
                min_good_pixels=args.min_good_pixels,
                fit_method=args.fit_method,
                coeff_norm=args.coeff_norm,
            )

            a = np.asarray(fit["a"], dtype=float)
            coeffs = np.asarray(fit["coeffs"], dtype=float)

            amp = fit["amplitude"]
            c_scale = fit["c_scale"]
            log_s = fit["log_c_scale"]
            chi2 = fit["chi2"]
            chi2_dof = fit["chi2_dof"]

            row = {
                "targetid": targetid,
                "z": z,
                "zwarn": int(rr["ZWARN"][rr_idx]),
                "spectype": str(rr["SPECTYPE"][rr_idx]).strip(),
                "success": bool(fit["success"]),
                "reason": str(fit["reason"]),
                "amplitude": float(amp) if np.isfinite(amp) else np.nan,
                "c_scale": float(c_scale) if np.isfinite(c_scale) else np.nan,
                "log_c_scale": float(log_s) if np.isfinite(log_s) else np.nan,
                "chi2": float(chi2) if np.isfinite(chi2) else np.nan,
                "dof": int(fit["dof"]),
                "chi2_dof": float(chi2_dof) if np.isfinite(chi2_dof) else np.nan,
                "n_good": int(fit["n_good"]),
            }

            for k in range(n_templates):
                row[f"a{k+1}"] = a[k]
                row[f"c{k+1}"] = coeffs[k]

            rows.append(row)
            all_a.append(a)
            all_coeffs.append(coeffs)
            all_log_c_scale.append(
                float(log_s) if np.isfinite(log_s) else np.nan
            )

        df = pd.DataFrame(rows)

        a_arr = (
            np.vstack(all_a)
            if all_a
            else np.empty((0, n_templates), dtype=float)
        )

        coeffs_arr = (
            np.vstack(all_coeffs)
            if all_coeffs
            else np.empty((0, n_templates), dtype=float)
        )

        log_c_scale_arr = (
            np.asarray(all_log_c_scale, dtype=float)
            if all_log_c_scale
            else np.empty(0, dtype=float)
        )

        max_chi2 = None if args.no_quality_cuts else args.max_chi2_dof
        df = apply_quality_cuts(df, max_chi2_dof=max_chi2)
        prior_mask = prior_quality_mask(df)

        dropped = df[df["success"].astype(bool) & ~df["quality_pass"].astype(bool)]
        dropped_path = outdir / "dropped_fits.csv"
        if len(dropped):
            dropped.to_csv(dropped_path, index=False)
            print(
                f"\nDropped {len(dropped)} successful fits failing quality cuts "
                f"(chi2/dof > {max_chi2}): {dropped_path}"
            )
            for _, row in dropped.iterrows():
                print(
                    f"  TARGETID {int(row['targetid'])}  z={row['z']:.4f}  "
                    f"chi2/dof={row['chi2_dof']:.3f}"
                )
        elif not args.no_quality_cuts:
            print(f"\nAll successful fits pass quality cuts (chi2/dof <= {max_chi2}).")

        csv_path = outdir / "desi_eazy_empirical_weights.csv"
        df.to_csv(csv_path, index=False)

        save_diagnostic_plots(
            outdir,
            df,
            a_arr,
            coeffs_arr,
            log_c_scale_arr,
            template_files,
            triangle_panel_size=args.triangle_panel_size,
            triangle_plots=not args.no_triangle_plots,
            raw_coeff_triangle=not args.no_raw_coeff_triangle,
            fit_method=args.fit_method,
            coeff_norm=args.coeff_norm,
            max_chi2_dof=None if args.no_quality_cuts else args.max_chi2_dof,
        )

    summary_path = write_outlier_summary(
        outdir,
        df,
        outlier_metric=args.outlier_metric,
    )
    if summary_path is not None:
        print(f"\nOutlier ranking: {summary_path}")

    plot_kwargs = dict(
        smooth_sigma_aa=args.plot_smooth_angstrom,
        cont_sigma_aa=args.plot_cont_angstrom,
        normalize_continuum=args.plot_normalize,
        fit_method=args.fit_method,
    )

    plot_df = filter_df_to_loaded_healpix(df, args.healpix)

    if want_outlier_plots and len(plot_df) > 0:
        pick = select_rows_for_spectrum_plots(
            plot_df,
            targetids=plot_targetids,
            n_outliers=args.plot_top_outliers,
            outlier_metric=args.outlier_metric,
        )
        if len(pick) > 0:
            print("\nPlotting outlier / targeted spectra:")
            for _, row in pick.iterrows():
                print(
                    f"  TARGETID {int(row['targetid'])}  z={row['z']:.4f}  "
                    f"chi2/dof={row['chi2_dof']:.3f}  c_scale={row['c_scale']:.4g}"
                )
            examples = build_spectrum_examples(
                pick,
                targetid_to_index=targetid_to_index,
                arms=arms,
                template_waves=template_waves,
                template_fluxes=template_fluxes,
                n_templates=n_templates,
            )
            save_example_spectrum_plots(
                outdir,
                examples,
                basename="spectrum_outlier_examples",
                **plot_kwargs,
            )

    if want_random_examples and len(plot_df) > 0:
        good_df = (
            plot_df[plot_df["quality_pass"]]
            if "quality_pass" in plot_df.columns
            else plot_df[plot_df["success"]]
        )
        if len(good_df) > 0:
            n_plot = min(args.plot_n_examples, len(good_df))
            pick = good_df.sample(n=n_plot, random_state=args.seed)
            examples = build_spectrum_examples(
                pick,
                targetid_to_index=targetid_to_index,
                arms=arms,
                template_waves=template_waves,
                template_fluxes=template_fluxes,
                n_templates=n_templates,
            )
            save_example_spectrum_plots(
                outdir,
                examples,
                basename="spectrum_fit_examples",
                **plot_kwargs,
            )

    n_success = int(df["success"].sum()) if len(df) else 0
    n_prior = int(df["quality_pass"].sum()) if len(df) and "quality_pass" in df.columns else n_success

    print("\nDone.")
    print(f"Attempted fits: {len(df)}")
    print(f"Successful fits: {n_success}")
    if "quality_pass" in df.columns and not args.no_quality_cuts:
        print(
            f"Prior-quality fits (chi2/dof <= {args.max_chi2_dof}): {n_prior} "
            f"({n_success - n_prior} dropped)"
        )
    elif "quality_pass" in df.columns:
        print(f"Prior-quality fits (no chi2 cut): {n_prior}")
    print("Outputs:")
    print(f"  {csv_path}")

    for plot_name in (
        "spectrum_fit_examples.png",
        "spectrum_fit_examples_flux.png",
        "spectrum_outlier_examples.png",
        "spectrum_outlier_examples_flux.png",
        "outlier_fits_summary.csv",
        "dropped_fits.csv",
        "coeffs_raw_triangle.png",
        "prior_params_triangle.png",
        "chi2_dof_histogram.png",
    ):
        plot_path = outdir / plot_name
        if plot_path.exists():
            print(f"  {plot_path}")


if __name__ == "__main__":
    main()