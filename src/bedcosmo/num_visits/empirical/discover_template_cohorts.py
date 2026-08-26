"""Discover DESI cohorts represented by a fixed number of EAZY templates.

For every size-N subset of the configured template bank, this module refits all
quality-passing DESI spectra with nonnegative coefficients. It reports both
overlapping coverage (every spectrum passing the quality cuts for a subset)
and a disjoint best-subset assignment suitable for cohort-specific priors.

The expensive pixel-to-template products are cached as per-spectrum weighted
least-squares sufficient statistics, so subsequent searches over N are fast.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from astropy.io import fits
from speclite import filters as speclite_filters

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .desi_data import get_local_desi_paths
from .fit_eazy_weights_to_desi import build_template_matrix_on_observed_grid
from .paths import (
    DEFAULT_EMPIRICAL_PRIOR_DIR,
    DEFAULT_PROGRAM,
    DEFAULT_SPECPROD,
    DEFAULT_SURVEY,
    get_desi_data_dir,
    get_prior_build_dir,
    get_template_dir,
)
from .templates import (
    DEFAULT_TEMPLATE_NORM_MAX_AA,
    DEFAULT_TEMPLATE_NORM_MIN_AA,
    DEFAULT_TEMPLATE_PARAM_12D,
    load_eazy_template_bank,
    load_eazy_templates,
)

INK = "#25272B"
MUTED = "#6B7280"
GRID = "#D9DDE3"
COVERAGE_COLOR = "#9DB7DF"
ASSIGNED_COLOR = "#3366CC"


def _coefficient_columns(table: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        (name for name in table if name.startswith(prefix) and name[len(prefix) :].isdigit()),
        key=lambda name: int(name[len(prefix) :]),
    )


def load_quality_fit_table(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "targetid",
        "healpix",
        "z",
        "success",
        "quality_pass",
        "chi2",
        "dof",
        "chi2_dof",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    keep = table["success"].astype(bool) & table["quality_pass"].astype(bool)
    table = table.loc[keep].copy().reset_index(drop=True)
    if table.empty:
        raise ValueError(f"No quality-passing rows in {path}")
    return table


def _read_target_spectrum(
    hdul: fits.HDUList,
    arm_waves: dict[str, np.ndarray],
    row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wave = np.concatenate([arm_waves[arm] for arm in "BRZ"])
    flux = np.concatenate([np.asarray(hdul[f"{arm}_FLUX"].data[row], dtype=float) for arm in "BRZ"])
    ivar = np.concatenate([np.asarray(hdul[f"{arm}_IVAR"].data[row], dtype=float) for arm in "BRZ"])
    mask = np.concatenate([np.asarray(hdul[f"{arm}_MASK"].data[row]) for arm in "BRZ"])
    order = np.argsort(wave)
    return wave[order], flux[order], ivar[order], mask[order]


def build_sufficient_statistics(
    table: pd.DataFrame,
    *,
    desi_dir: Path,
    template_waves: list[np.ndarray],
    template_fluxes: list[np.ndarray],
    specprod: str = DEFAULT_SPECPROD,
    survey: str = DEFAULT_SURVEY,
    program: str = DEFAULT_PROGRAM,
    wave_obs_min: float | None = None,
    wave_obs_max: float | None = None,
    min_good_pixels: int = 200,
) -> dict[str, np.ndarray]:
    """Build G=X'X, b=X'y, and y2=y'y for each fitted DESI spectrum."""
    n_rows = len(table)
    n_templates = len(template_waves)
    gram = np.full((n_rows, n_templates, n_templates), np.nan, dtype=np.float64)
    cross = np.full((n_rows, n_templates), np.nan, dtype=np.float64)
    data_norm = np.full(n_rows, np.nan, dtype=np.float64)
    n_good = np.zeros(n_rows, dtype=np.int32)
    available = np.zeros(n_rows, dtype=bool)

    for healpix, patch in table.groupby("healpix", sort=True):
        coadd_path, _ = get_local_desi_paths(desi_dir, specprod, survey, program, int(healpix))
        if not coadd_path.is_file():
            raise FileNotFoundError(coadd_path)
        with fits.open(coadd_path, memmap=True) as hdul:
            targetids = np.asarray(hdul["FIBERMAP"].data["TARGETID"], dtype=np.int64)
            target_to_row = {int(targetid): i for i, targetid in enumerate(targetids)}
            arm_waves = {
                arm: np.asarray(hdul[f"{arm}_WAVELENGTH"].data, dtype=float) for arm in "BRZ"
            }
            for table_index, fit_row in patch.iterrows():
                targetid = int(fit_row["targetid"])
                coadd_row = target_to_row.get(targetid)
                if coadd_row is None:
                    continue
                wave, flux, ivar, mask = _read_target_spectrum(hdul, arm_waves, coadd_row)
                matrix = build_template_matrix_on_observed_grid(
                    wave,
                    float(fit_row["z"]),
                    template_waves,
                    template_fluxes,
                )
                good = (
                    np.isfinite(wave)
                    & np.isfinite(flux)
                    & np.isfinite(ivar)
                    & (ivar > 0)
                    & (mask == 0)
                    & np.all(np.isfinite(matrix), axis=1)
                    & np.any(matrix > 0, axis=1)
                )
                if wave_obs_min is not None:
                    good &= wave >= wave_obs_min
                if wave_obs_max is not None:
                    good &= wave <= wave_obs_max
                count = int(good.sum())
                if count < max(int(min_good_pixels), n_templates + 1):
                    continue
                weights = np.sqrt(ivar[good])
                design = matrix[good] * weights[:, None]
                response = flux[good] * weights
                gram[table_index] = design.T @ design
                cross[table_index] = design.T @ response
                data_norm[table_index] = response @ response
                n_good[table_index] = count
                available[table_index] = True
        print(f"Cached sufficient statistics for HEALPix {int(healpix)} ({len(patch):,} rows)")

    if not np.all(available):
        print(f"Warning: {int((~available).sum())} rows could not be rebuilt from coadds")
    return {
        "gram": gram,
        "cross": cross,
        "data_norm": data_norm,
        "n_good": n_good,
        "available": available,
    }


def save_statistics_cache(
    path: str | Path,
    table: pd.DataFrame,
    statistics: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        targetid=table["targetid"].to_numpy(np.int64),
        healpix=table["healpix"].to_numpy(np.int64),
        redshift=table["z"].to_numpy(float),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **statistics,
    )
    return path


def load_statistics_cache(
    path: str | Path,
    table: pd.DataFrame,
    expected_metadata: dict[str, Any],
) -> dict[str, np.ndarray] | None:
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata"].item()))
        targetids = cached["targetid"]
        if metadata != expected_metadata or not np.array_equal(
            targetids, table["targetid"].to_numpy(np.int64)
        ):
            print(f"Ignoring incompatible sufficient-statistics cache: {path}")
            return None
        return {
            key: np.asarray(cached[key])
            for key in ("gram", "cross", "data_norm", "n_good", "available")
        }


def solve_subset_nnls(
    gram: np.ndarray,
    cross: np.ndarray,
    data_norm: np.ndarray,
    subset: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a batch of small NNLS problems by enumerating active faces.

    This is exact for the convex quadratic objective and avoids returning to
    the several-thousand-pixel design matrix for each candidate subset.
    """
    subset_array = np.asarray(subset, dtype=int)
    n_rows = len(gram)
    n_subset = len(subset)
    candidate_gram = gram[:, subset_array][:, :, subset_array]
    candidate_cross = cross[:, subset_array]
    best_chi2 = np.asarray(data_norm, dtype=float).copy()
    best_coefficients = np.zeros((n_rows, n_subset), dtype=float)

    for active_size in range(1, n_subset + 1):
        for active in itertools.combinations(range(n_subset), active_size):
            active_array = np.asarray(active, dtype=int)
            active_gram = candidate_gram[:, active_array][:, :, active_array]
            active_cross = candidate_cross[:, active_array]
            diagonal_scale = np.maximum(
                np.max(np.abs(np.diagonal(active_gram, axis1=1, axis2=2)), axis=1),
                1.0,
            )
            regularized = active_gram.copy()
            diagonal = np.arange(active_size)
            regularized[:, diagonal, diagonal] += diagonal_scale[:, None] * 1e-12
            try:
                coefficients = np.linalg.solve(regularized, active_cross[..., None])[..., 0]
            except np.linalg.LinAlgError:
                coefficients = np.einsum(
                    "nij,nj->ni", np.linalg.pinv(regularized, rcond=1e-10), active_cross
                )
            feasible = np.all(coefficients >= -1e-10, axis=1)
            coefficients = np.maximum(coefficients, 0.0)
            chi2 = (
                data_norm
                - 2.0 * np.einsum("ni,ni->n", coefficients, active_cross)
                + np.einsum("ni,nij,nj->n", coefficients, active_gram, coefficients)
            )
            improve = feasible & np.isfinite(chi2) & (chi2 < best_chi2)
            if not np.any(improve):
                continue
            best_chi2[improve] = np.maximum(chi2[improve], 0.0)
            best_coefficients[improve] = 0.0
            for local_column, candidate_column in enumerate(active):
                best_coefficients[improve, candidate_column] = coefficients[improve, local_column]
    return best_coefficients, best_chi2


def full_fit_chi2_dof_from_statistics(
    table: pd.DataFrame,
    statistics: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate the stored full-fit coefficients against the cached objective."""
    coefficients = table[_coefficient_columns(table, "c")].to_numpy(float)
    chi2 = (
        statistics["data_norm"]
        - 2.0 * np.einsum("ni,ni->n", coefficients, statistics["cross"])
        + np.einsum("ni,nij,nj->n", coefficients, statistics["gram"], coefficients)
    )
    dof = np.maximum(statistics["n_good"] - coefficients.shape[1], 1)
    return chi2 / dof


def lsst_template_responses(
    wave_rest: np.ndarray,
    templates: np.ndarray,
    redshift: np.ndarray,
    *,
    chunk_size: int = 500,
) -> np.ndarray:
    """Photon-weighted LSST ugrizy responses for each galaxy and template."""
    filter_data = []
    for band in "ugrizy":
        loaded = speclite_filters.load_filter("lsst2023-" + band)
        wave = np.asarray(loaded.wavelength, dtype=float)[::5]
        filter_data.append((wave, np.asarray(loaded(wave), dtype=float)))
    n_wave = max(len(wave) for wave, _ in filter_data)
    wave_obs = np.linspace(
        min(wave.min() for wave, _ in filter_data),
        max(wave.max() for wave, _ in filter_data),
        n_wave,
    )
    transmission = np.stack(
        [np.interp(wave_obs, wave, response, left=0.0, right=0.0) for wave, response in filter_data]
    )
    photon_kernel = transmission * wave_obs[None, :]
    responses = np.empty((len(redshift), len(templates), 6), dtype=float)
    for start in range(0, len(redshift), chunk_size):
        stop = min(start + chunk_size, len(redshift))
        z = redshift[start:stop]
        rest_wave = wave_obs[None, :] / (1.0 + z[:, None])
        observed = np.stack(
            [
                np.interp(rest_wave.ravel(), wave_rest, flux, left=0.0, right=0.0).reshape(
                    len(z), n_wave
                )
                / (1.0 + z[:, None])
                for flux in templates
            ],
            axis=1,
        )
        responses[start:stop] = np.trapz(
            observed[:, :, None, :] * photon_kernel[None, None, :, :],
            wave_obs,
            axis=-1,
        )
    return responses


def centered_magnitudes(flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.all(np.isfinite(flux) & (flux > 0), axis=1)
    magnitudes = np.full_like(flux, np.nan)
    magnitudes[valid] = -2.5 * np.log10(flux[valid])
    magnitudes[valid] -= magnitudes[valid].mean(axis=1, keepdims=True)
    return magnitudes, valid


def subset_color_rms(
    coefficients: np.ndarray,
    subset: tuple[int, ...],
    responses: np.ndarray,
    reference_colors: np.ndarray,
    reference_valid: np.ndarray,
) -> np.ndarray:
    candidate_flux = np.einsum("ni,nib->nb", coefficients, responses[:, subset, :])
    colors, valid = centered_magnitudes(candidate_flux)
    valid &= reference_valid
    rms = np.full(len(coefficients), np.inf, dtype=float)
    rms[valid] = np.sqrt(np.mean((colors[valid] - reference_colors[valid]) ** 2, axis=1))
    return rms


def _subset_label(subset: tuple[int, ...]) -> str:
    return "+".join(f"T{index + 1}" for index in subset)


def discover_cohorts(
    table: pd.DataFrame,
    statistics: dict[str, np.ndarray],
    responses: np.ndarray,
    *,
    n_templates_per_subset: int,
    max_chi2_dof: float,
    max_delta_chi2_dof: float,
    max_color_rms: float,
    min_component_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    c_columns = _coefficient_columns(table, "c")
    n_templates = statistics["cross"].shape[1]
    if len(c_columns) != n_templates:
        raise ValueError(f"Expected {n_templates} raw coefficient columns, found {c_columns}")
    full_coefficients = table[c_columns].to_numpy(float)
    full_flux = np.einsum("nk,nkb->nb", full_coefficients, responses)
    reference_colors, reference_valid = centered_magnitudes(full_flux)
    subsets = list(itertools.combinations(range(n_templates), n_templates_per_subset))
    n_rows = len(table)
    n_subsets = len(subsets)
    spectral_quality = np.full((n_rows, n_subsets), np.inf, dtype=np.float32)
    reduced_chi2_quality = np.full((n_rows, n_subsets), np.inf, dtype=np.float32)
    color_quality = np.full((n_rows, n_subsets), np.inf, dtype=np.float32)
    pass_matrix = np.zeros((n_rows, n_subsets), dtype=bool)
    best_score = np.full(n_rows, np.inf, dtype=float)
    best_subset = np.full(n_rows, -1, dtype=np.int32)
    best_coefficients = np.full((n_rows, n_templates_per_subset), np.nan, dtype=float)
    best_chi2_dof = np.full(n_rows, np.nan, dtype=float)
    best_delta = np.full(n_rows, np.nan, dtype=float)
    best_color = np.full(n_rows, np.nan, dtype=float)
    available = statistics["available"].astype(bool)
    full_chi2 = table["chi2"].to_numpy(float)
    full_dof = np.maximum(table["dof"].to_numpy(float), 1.0)
    summary_rows = []
    membership_frames: list[pd.DataFrame] = []

    for subset_index, subset in enumerate(subsets):
        coefficients, chi2 = solve_subset_nnls(
            statistics["gram"],
            statistics["cross"],
            statistics["data_norm"],
            subset,
        )
        dof = np.maximum(statistics["n_good"] - n_templates_per_subset, 1)
        chi2_dof = chi2 / dof
        # Extra chi-square per original-fit degree of freedom. Unlike the
        # difference of two reduced chi2 values, this cannot become negative
        # merely because the smaller model has a larger nominal dof.
        delta = np.maximum((chi2 - full_chi2) / full_dof, 0.0)
        color_rms = subset_color_rms(
            coefficients,
            subset,
            responses,
            reference_colors,
            reference_valid,
        )
        coefficient_sum = coefficients.sum(axis=1)
        normalized_coefficients = np.divide(
            coefficients,
            coefficient_sum[:, None],
            out=np.zeros_like(coefficients),
            where=coefficient_sum[:, None] > 0,
        )
        all_components_active = np.all(normalized_coefficients >= min_component_weight, axis=1)
        passing = (
            available
            & all_components_active
            & np.isfinite(chi2_dof)
            & (chi2_dof <= max_chi2_dof)
            & (delta <= max_delta_chi2_dof)
            & (color_rms <= max_color_rms)
        )
        spectral_quality[:, subset_index] = delta.astype(np.float32)
        reduced_chi2_quality[:, subset_index] = chi2_dof.astype(np.float32)
        color_quality[:, subset_index] = color_rms.astype(np.float32)
        pass_matrix[:, subset_index] = passing
        passing_rows = np.flatnonzero(passing)
        if len(passing_rows):
            membership = table.loc[
                passing_rows, ["targetid", "healpix", "z", "chi2", "dof", "chi2_dof"]
            ].copy()
            membership["subset_index"] = subset_index
            membership["templates"] = _subset_label(subset)
            membership["reduced_chi2_dof"] = chi2_dof[passing_rows]
            membership["delta_chi2_dof"] = delta[passing_rows]
            membership["lsst_color_rms"] = color_rms[passing_rows]
            reduced_scale = coefficient_sum[passing_rows]
            membership["reduced_log_c_scale"] = np.log(reduced_scale)
            for position, template_index in enumerate(subset):
                membership[f"template_{position + 1}"] = template_index + 1
                membership[f"c_{position + 1}"] = coefficients[passing_rows, position]
                membership[f"a_{position + 1}"] = (
                    coefficients[passing_rows, position] / reduced_scale
                )
            membership_frames.append(membership)
        score = delta / max(max_delta_chi2_dof, 1e-12) + color_rms / max(max_color_rms, 1e-12)
        improve = passing & (score < best_score)
        best_score[improve] = score[improve]
        best_subset[improve] = subset_index
        best_coefficients[improve] = coefficients[improve]
        best_chi2_dof[improve] = chi2_dof[improve]
        best_delta[improve] = delta[improve]
        best_color[improve] = color_rms[improve]
        covered = passing
        summary_rows.append(
            {
                "subset_index": subset_index,
                "templates": _subset_label(subset),
                "template_indices": ",".join(str(index + 1) for index in subset),
                "coverage_count": int(covered.sum()),
                "coverage_fraction": float(covered.mean()),
                "coverage_median_delta_chi2_dof": (
                    float(np.median(delta[covered])) if np.any(covered) else np.nan
                ),
                "coverage_p95_delta_chi2_dof": (
                    float(np.percentile(delta[covered], 95)) if np.any(covered) else np.nan
                ),
                "coverage_median_reduced_chi2_dof": (
                    float(np.median(chi2_dof[covered])) if np.any(covered) else np.nan
                ),
                "coverage_p95_reduced_chi2_dof": (
                    float(np.percentile(chi2_dof[covered], 95)) if np.any(covered) else np.nan
                ),
                "coverage_p95_color_rms": (
                    float(np.percentile(color_rms[covered], 95)) if np.any(covered) else np.nan
                ),
            }
        )
        if (subset_index + 1) % 25 == 0 or subset_index + 1 == n_subsets:
            print(f"Evaluated {subset_index + 1}/{n_subsets} subsets")

    assigned = best_subset >= 0
    assigned_counts = np.bincount(best_subset[assigned], minlength=n_subsets)
    membership_multiplicity = pass_matrix.sum(axis=1)
    exclusive_counts = (pass_matrix & (membership_multiplicity[:, None] == 1)).sum(axis=0)
    summary = pd.DataFrame(summary_rows)
    summary["exclusive_count"] = exclusive_counts
    summary["exclusive_fraction"] = exclusive_counts / n_rows
    summary["assigned_count"] = assigned_counts
    summary["assigned_fraction"] = assigned_counts / n_rows
    for subset_index in range(n_subsets):
        rows = best_subset == subset_index
        summary.loc[subset_index, "assigned_median_delta_chi2_dof"] = (
            np.median(best_delta[rows]) if np.any(rows) else np.nan
        )
        summary.loc[subset_index, "assigned_p95_color_rms"] = (
            np.percentile(best_color[rows], 95) if np.any(rows) else np.nan
        )
        summary.loc[subset_index, "assigned_median_z"] = (
            np.median(table.loc[rows, "z"]) if np.any(rows) else np.nan
        )
    summary = summary.sort_values(
        ["coverage_count", "assigned_count"], ascending=False
    ).reset_index(drop=True)

    assignments = table.copy()
    assignments["reduced_subset_index"] = best_subset
    assignments["reduced_templates"] = [
        _subset_label(subsets[index]) if index >= 0 else "" for index in best_subset
    ]
    assignments["reduced_chi2_dof"] = best_chi2_dof
    assignments["delta_chi2_dof"] = best_delta
    assignments["lsst_color_rms"] = best_color
    assignments["reduced_quality_pass"] = assigned
    for position in range(n_templates_per_subset):
        template_number = np.full(n_rows, -1, dtype=int)
        valid_rows = np.flatnonzero(assigned)
        template_number[valid_rows] = np.asarray(
            [subsets[best_subset[row]][position] + 1 for row in valid_rows]
        )
        assignments[f"reduced_template_{position + 1}"] = template_number
        assignments[f"reduced_c_{position + 1}"] = best_coefficients[:, position]
    coefficient_sum = np.nansum(best_coefficients, axis=1)
    positive_scale = assigned & (coefficient_sum > 0)
    reduced_log_scale = np.full(n_rows, np.nan, dtype=float)
    reduced_log_scale[positive_scale] = np.log(coefficient_sum[positive_scale])
    assignments["reduced_log_c_scale"] = reduced_log_scale
    for position in range(n_templates_per_subset):
        normalized = np.full(n_rows, np.nan, dtype=float)
        normalized[positive_scale] = (
            best_coefficients[positive_scale, position] / coefficient_sum[positive_scale]
        )
        assignments[f"reduced_a_{position + 1}"] = normalized

    matrices = {
        "subsets": np.asarray(subsets, dtype=np.int16) + 1,
        "delta_chi2_dof": spectral_quality,
        "reduced_chi2_dof": reduced_chi2_quality,
        "lsst_color_rms": color_quality,
        "passes": pass_matrix,
    }
    memberships = (
        pd.concat(membership_frames, ignore_index=True) if membership_frames else pd.DataFrame()
    )
    return summary, assignments, memberships, matrices


def plot_cohort_summary(
    summary: pd.DataFrame,
    assignments: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    *,
    output: str | Path,
    min_cohort_size: int,
    max_chi2_dof: float,
    max_delta_chi2_dof: float,
    max_sets: int = 20,
) -> None:
    shown = summary[summary["coverage_count"] >= min_cohort_size].head(max_sets)
    if shown.empty:
        shown = summary.head(min(max_sets, len(summary)))
    shown = shown.iloc[::-1]
    y = np.arange(len(shown))
    fig, (ax_count, ax_delta, ax_color, ax_redshift) = plt.subplots(
        1,
        4,
        figsize=(18, max(6.5, 0.38 * len(shown) + 2.0)),
        constrained_layout=True,
        sharey=True,
        gridspec_kw={"width_ratios": [1.4, 0.75, 0.75, 0.9]},
    )
    ax_count.barh(
        y,
        shown["coverage_count"],
        color=COVERAGE_COLOR,
        edgecolor=ASSIGNED_COLOR,
        linewidth=0.8,
    )
    ax_count.barh(y, shown["exclusive_count"], color=ASSIGNED_COLOR)
    ax_count.set_yticks(y, shown["templates"])
    ax_count.set_xlabel("Independently represented DESI spectra")
    ax_count.set_title("Coverage: light = total, dark = unique", loc="left")

    subset_indices = shown["subset_index"].to_numpy(int)
    pass_matrix = matrices["passes"]
    n_input = pass_matrix.shape[0]
    n_passing = int(np.any(pass_matrix, axis=1).sum())
    n_templates_per_subset = matrices["subsets"].shape[1]
    n_full_templates = int(np.max(matrices["subsets"]))
    passing_fraction = 100.0 * n_passing / n_input

    def draw_distributions(ax: plt.Axes, values: np.ndarray) -> None:
        for position, subset_index in zip(y, subset_indices):
            data = np.asarray(values[pass_matrix[:, subset_index], subset_index], dtype=float)
            data = data[np.isfinite(data)]
            if len(data) == 0:
                continue
            if len(data) > 1 and np.ptp(data) > 1e-12:
                violin = ax.violinplot(
                    [data],
                    positions=[position],
                    vert=False,
                    widths=0.72,
                    showextrema=False,
                )
                body = violin["bodies"][0]
                body.set_facecolor(COVERAGE_COLOR)
                body.set_edgecolor(ASSIGNED_COLOR)
                body.set_alpha(0.55)
            q05, median, q95 = np.percentile(data, [5, 50, 95])
            ax.hlines(position, q05, q95, color=INK, lw=1.1)
            ax.scatter(median, position, color=ASSIGNED_COLOR, edgecolor=INK, s=27, zorder=3)

    draw_distributions(ax_delta, matrices["delta_chi2_dof"])
    ax_delta.axvline(max_delta_chi2_dof, color=MUTED, ls=":", lw=1.2)
    ax_delta.set_xlabel(r"Extra $\chi^2$ / original dof vs full-template fit")
    ax_delta.set_title("DESI fit degradation", loc="left")
    ax_delta.tick_params(axis="y", labelleft=False)

    draw_distributions(ax_color, matrices["reduced_chi2_dof"])
    ax_color.axvline(max_chi2_dof, color=MUTED, ls=":", lw=1.2)
    ax_color.set_xlabel(r"Reduced-fit $\chi^2/\mathrm{dof}$")
    ax_color.set_title("Fit to observed DESI spectrum", loc="left")
    ax_color.tick_params(axis="y", labelleft=False)

    assigned = assignments.loc[assignments["reduced_quality_pass"].astype(bool)]
    for position, subset_index in zip(y, subset_indices):
        data = assigned.loc[
            assigned["reduced_subset_index"].astype(int) == subset_index, "z"
        ].to_numpy(float)
        data = data[np.isfinite(data)]
        if len(data) == 0:
            continue
        if len(data) > 1 and np.ptp(data) > 1e-12:
            violin = ax_redshift.violinplot(
                [data],
                positions=[position],
                vert=False,
                widths=0.72,
                showextrema=False,
            )
            body = violin["bodies"][0]
            body.set_facecolor(COVERAGE_COLOR)
            body.set_edgecolor(ASSIGNED_COLOR)
            body.set_alpha(0.55)
        q05, median, q95 = np.percentile(data, [5, 50, 95])
        ax_redshift.hlines(position, q05, q95, color=INK, lw=1.1)
        ax_redshift.scatter(median, position, color=ASSIGNED_COLOR, edgecolor=INK, s=27, zorder=3)
    ax_redshift.set_xlabel("Assigned spectra redshift $z$")
    ax_redshift.set_title("Assigned spectra redshift", loc="left")
    ax_redshift.tick_params(axis="y", labelleft=False)

    for ax in (ax_count, ax_delta, ax_color, ax_redshift):
        ax.grid(True, color=GRID, alpha=0.65, linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"Full {n_full_templates}-template EAZY basis → N={n_templates_per_subset}: "
        f"{n_passing:,} spectra pass ({passing_fraction:.1f}% of {n_input:,})",
        fontsize=15,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-templates", type=int, required=True)
    parser.add_argument("--build-name", default=DEFAULT_EMPIRICAL_PRIOR_DIR)
    parser.add_argument("--weights-csv", type=Path, default=None)
    parser.add_argument("--desi-dir", type=Path, default=None)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--template-param", default=DEFAULT_TEMPLATE_PARAM_12D)
    parser.add_argument("--norm-min", type=float, default=DEFAULT_TEMPLATE_NORM_MIN_AA)
    parser.add_argument("--norm-max", type=float, default=DEFAULT_TEMPLATE_NORM_MAX_AA)
    parser.add_argument("--specprod", default=DEFAULT_SPECPROD)
    parser.add_argument("--survey", default=DEFAULT_SURVEY)
    parser.add_argument("--program", default=DEFAULT_PROGRAM)
    parser.add_argument("--wave-obs-min", type=float, default=None)
    parser.add_argument("--wave-obs-max", type=float, default=None)
    parser.add_argument("--min-good-pixels", type=int, default=200)
    parser.add_argument("--max-chi2-dof", type=float, default=1.2)
    parser.add_argument("--max-delta-chi2-dof", type=float, default=0.05)
    parser.add_argument("--max-color-rms", type=float, default=0.02)
    parser.add_argument(
        "--min-component-weight",
        type=float,
        default=0.01,
        help=(
            "Minimum normalized coefficient for every member of the subset. "
            "This prevents an exact N-template cohort from being padded by unused templates."
        ),
    )
    parser.add_argument("--min-cohort-size", type=int, default=100)
    parser.add_argument("--max-plot-sets", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--statistics-cache", type=Path, default=None)
    parser.add_argument("--force-statistics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prior_dir = get_prior_build_dir(args.build_name)
    weights_csv = args.weights_csv or prior_dir / "desi_eazy_empirical_weights.csv"
    desi_dir = args.desi_dir or get_desi_data_dir()
    template_dir = args.template_dir or get_template_dir()
    output_dir = args.output_dir or prior_dir / "reduced_template_cohorts" / f"n{args.n_templates}"
    cache_path = args.statistics_cache or output_dir.parent / "sufficient_statistics.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = load_quality_fit_table(weights_csv)
    a_columns = _coefficient_columns(table, "a")
    if not 1 <= args.n_templates <= len(a_columns):
        raise ValueError(f"--n-templates must be between 1 and {len(a_columns)}")
    if args.norm_min >= args.norm_max:
        raise ValueError("--norm-min must be below --norm-max")
    for label, value in (
        ("--max-chi2-dof", args.max_chi2_dof),
        ("--max-delta-chi2-dof", args.max_delta_chi2_dof),
        ("--max-color-rms", args.max_color_rms),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    if not 0.0 <= args.min_component_weight < 1.0 / args.n_templates:
        raise ValueError(
            "--min-component-weight must be nonnegative and below 1/N so all components "
            "can satisfy it"
        )

    template_waves, template_fluxes, template_paths = load_eazy_templates(
        args.template_param,
        template_dir=template_dir,
        norm_min=args.norm_min,
        norm_max=args.norm_max,
    )
    if len(template_waves) != len(a_columns):
        raise ValueError(
            f"Template bank has {len(template_waves)} templates but fit table has {len(a_columns)}"
        )
    cache_metadata = {
        "template_param": args.template_param,
        "template_paths": template_paths,
        "norm_min": float(args.norm_min),
        "norm_max": float(args.norm_max),
        "desi_dir": str(Path(desi_dir).expanduser().resolve()),
        "specprod": args.specprod,
        "survey": args.survey,
        "program": args.program,
        "wave_obs_min": args.wave_obs_min,
        "wave_obs_max": args.wave_obs_max,
        "min_good_pixels": int(args.min_good_pixels),
    }
    statistics = (
        None if args.force_statistics else load_statistics_cache(cache_path, table, cache_metadata)
    )
    if statistics is None:
        statistics = build_sufficient_statistics(
            table,
            desi_dir=Path(desi_dir),
            template_waves=template_waves,
            template_fluxes=template_fluxes,
            specprod=args.specprod,
            survey=args.survey,
            program=args.program,
            wave_obs_min=args.wave_obs_min,
            wave_obs_max=args.wave_obs_max,
            min_good_pixels=args.min_good_pixels,
        )
        save_statistics_cache(cache_path, table, statistics, cache_metadata)
        print(f"Wrote sufficient-statistics cache: {cache_path}")
    else:
        print(f"Loaded sufficient-statistics cache: {cache_path}")

    rebuilt_full_chi2_dof = full_fit_chi2_dof_from_statistics(table, statistics)
    available = statistics["available"].astype(bool)
    discrepancy = np.abs(
        rebuilt_full_chi2_dof[available] - table.loc[available, "chi2_dof"].to_numpy(float)
    )
    print(
        "Full-fit cache consistency: "
        f"median |delta chi2/dof|={np.median(discrepancy):.3g}, "
        f"p99={np.percentile(discrepancy, 99):.3g}"
    )
    if np.percentile(discrepancy, 99) > 1e-5:
        raise RuntimeError(
            "Cached sufficient statistics do not reproduce the original full-template fits; "
            "check template normalization and fit wavelength settings."
        )

    wave_rest, template_stack, _ = load_eazy_template_bank(
        args.template_param,
        template_dir=template_dir,
        norm_min=args.norm_min,
        norm_max=args.norm_max,
    )
    print("Computing LSST ugrizy responses...")
    responses = lsst_template_responses(wave_rest, template_stack, table["z"].to_numpy(float))
    summary, assignments, memberships, matrices = discover_cohorts(
        table,
        statistics,
        responses,
        n_templates_per_subset=args.n_templates,
        max_chi2_dof=args.max_chi2_dof,
        max_delta_chi2_dof=args.max_delta_chi2_dof,
        max_color_rms=args.max_color_rms,
        min_component_weight=args.min_component_weight,
    )
    summary_path = output_dir / "subset_summary.csv"
    assignment_path = output_dir / "spectrum_assignments.csv"
    membership_path = output_dir / "subset_memberships.csv"
    matrix_path = output_dir / "subset_quality_matrices.npz"
    plot_path = output_dir / "subset_discovery.png"
    metadata_path = output_dir / "discovery_parameters.json"
    summary.to_csv(summary_path, index=False)
    assignments.to_csv(assignment_path, index=False)
    memberships.to_csv(membership_path, index=False)
    np.savez_compressed(
        matrix_path,
        targetid=table["targetid"].to_numpy(np.int64),
        **matrices,
    )
    metadata = {
        **cache_metadata,
        "n_templates": int(args.n_templates),
        "max_chi2_dof": float(args.max_chi2_dof),
        "max_delta_chi2_dof": float(args.max_delta_chi2_dof),
        "max_color_rms": float(args.max_color_rms),
        "min_component_weight": float(args.min_component_weight),
        "min_cohort_size": int(args.min_cohort_size),
        "n_input_spectra": int(len(table)),
        "n_assigned_spectra": int(assignments["reduced_quality_pass"].sum()),
        "n_independent_memberships": int(len(memberships)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    plot_cohort_summary(
        summary,
        assignments,
        matrices,
        output=plot_path,
        min_cohort_size=args.min_cohort_size,
        max_chi2_dof=args.max_chi2_dof,
        max_delta_chi2_dof=args.max_delta_chi2_dof,
        max_sets=args.max_plot_sets,
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {assignment_path}")
    print(f"Wrote {membership_path}")
    print(f"Wrote {matrix_path}")
    print(f"Wrote {plot_path}")
    print(
        summary.loc[
            summary["coverage_count"] >= args.min_cohort_size,
            [
                "templates",
                "coverage_count",
                "exclusive_count",
                "coverage_p95_delta_chi2_dof",
                "coverage_p95_reduced_chi2_dof",
            ],
        ]
        .head(args.max_plot_sets)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
