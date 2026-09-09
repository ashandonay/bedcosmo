#!/usr/bin/env python
"""Discover DESI spectral families and decode each into a sparse EAZY basis.

The full-fit EAZY weights are compositional, so this analysis first maps them
to the 11-dimensional ILR shape space. PCA finds the populated directions in
that space and HDBSCAN identifies dense families without prescribing their
number. Existing fixed-N cohort quality matrices are then used only as a
decoder: for every family, choose the smallest original-template subset that
passes the DESI-fit and LSST-color cuts for a requested fraction of members.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

try:
    from sklearn.cluster import HDBSCAN
except ImportError as error:  # pragma: no cover - depends on the local analysis environment
    raise ImportError(
        "Spectral-family discovery requires scikit-learn >= 1.3 for HDBSCAN"
    ) from error

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..paths import (  # noqa: E402
    DEFAULT_EMPIRICAL_PRIOR_DIR,
    get_prior_build_dir,
    get_template_dir,
)
from ..simplex import (  # noqa: E402
    DEFAULT_CLR_EPS,
    ilr_basis,
    weights_to_ilr,
)
from ..templates import (  # noqa: E402
    DEFAULT_TEMPLATE_NORM_MAX_AA,
    DEFAULT_TEMPLATE_NORM_MIN_AA,
    DEFAULT_TEMPLATE_PARAM_12D,
    load_eazy_templates,
)

INK = "#25272B"
MUTED = "#6B7280"
GRID = "#D9DDE3"
TEMPLATE_COLORS = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#003F5C",
    "#E6AB02",
)

LINE_DEFINITIONS = {
    "ew_oii_3727": ((3716.0, 3739.0), (3655.0, 3705.0), (3755.0, 3805.0), 3727.0),
    "ew_hbeta": ((4851.0, 4872.0), (4800.0, 4835.0), (4885.0, 4920.0), 4861.0),
    "ew_oiii_5007": ((4997.0, 5018.0), (4950.0, 4985.0), (5025.0, 5060.0), 5007.0),
    "ew_halpha": ((6553.0, 6574.0), (6485.0, 6525.0), (6605.0, 6645.0), 6563.0),
}


def _numbered_columns(table: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        (
            name
            for name in table.columns
            if name.startswith(prefix) and name[len(prefix) :].isdigit()
        ),
        key=lambda name: int(name[len(prefix) :]),
    )


def load_fit_population(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    table = pd.read_csv(path)
    required = {"targetid", "healpix", "z", "success", "quality_pass"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    table = table.loc[
        table["success"].astype(bool) & table["quality_pass"].astype(bool)
    ].copy()
    table = table.reset_index(drop=True)
    columns = _numbered_columns(table, "a")
    if len(columns) < 2:
        raise ValueError(f"Expected normalized a1...aK columns in {path}")
    weights = table[columns].to_numpy(float)
    finite = np.all(np.isfinite(weights), axis=1) & (weights.sum(axis=1) > 0)
    if not np.all(finite):
        table = table.loc[finite].reset_index(drop=True)
        weights = weights[finite]
    weights /= weights.sum(axis=1, keepdims=True)
    return table, weights


def fit_population_embedding(
    weights: np.ndarray,
    *,
    variance_threshold: float,
    clr_eps: float,
    min_cluster_size: int,
    min_samples: int,
    pca_scaling: str = "standardized",
) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray]:
    ilr = weights_to_ilr(weights, eps=clr_eps)
    pca = PCA().fit(ilr)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_retained = min(
        len(cumulative), int(np.searchsorted(cumulative, variance_threshold) + 1)
    )
    raw_scores = pca.transform(ilr)[:, :n_retained]
    if pca_scaling == "standardized":
        scores = raw_scores / np.sqrt(pca.explained_variance_[:n_retained])[None, :]
    elif pca_scaling == "raw":
        scores = raw_scores
    else:
        raise ValueError(f"Unknown PCA scaling {pca_scaling!r}")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        store_centers="centroid",
    )
    raw_labels = clusterer.fit_predict(scores)

    # Stable, visually useful family labels: order dense groups by PC1, then PC2.
    clusters = [int(value) for value in np.unique(raw_labels) if value >= 0]
    clusters.sort(
        key=lambda value: tuple(
            np.mean(scores[raw_labels == value, : min(2, scores.shape[1])], axis=0)
        )
    )
    relabel = {old: new + 1 for new, old in enumerate(clusters)}
    labels = np.asarray([relabel.get(int(value), 0) for value in raw_labels], dtype=int)
    probabilities = np.asarray(clusterer.probabilities_, dtype=float)
    return pca, scores, labels, probabilities


def cluster_sensitivity(
    scores: np.ndarray,
    reference_labels: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
) -> pd.DataFrame:
    """Measure whether nearby density settings preserve the clustered core."""
    rows: list[dict[str, float | int]] = []
    for factor in (2.0 / 3.0, 5.0 / 6.0, 1.0, 7.0 / 6.0, 4.0 / 3.0):
        candidate_size = max(2, int(round(min_cluster_size * factor)))
        candidate_samples = max(1, int(round(min_samples * factor)))
        labels = HDBSCAN(
            min_cluster_size=candidate_size,
            min_samples=candidate_samples,
            cluster_selection_method="eom",
        ).fit_predict(scores)
        shared = (reference_labels > 0) & (labels >= 0)
        ari = (
            adjusted_rand_score(reference_labels[shared], labels[shared])
            if np.any(shared)
            else np.nan
        )
        rows.append(
            {
                "min_cluster_size": candidate_size,
                "min_samples": candidate_samples,
                "n_families": len(set(labels).difference({-1})),
                "clustered_fraction": float(np.mean(labels >= 0)),
                "shared_with_reference_fraction": float(np.mean(shared)),
                "adjusted_rand_on_shared_core": float(ari),
            }
        )
    return pd.DataFrame(rows)


def load_subset_searches(
    cohort_root: Path,
    targetids: np.ndarray,
    *,
    max_chi2_dof: float,
    max_delta_chi2_dof: float,
    max_color_rms: float,
) -> dict[int, dict[str, np.ndarray]]:
    searches: dict[int, dict[str, np.ndarray]] = {}
    for directory in sorted(cohort_root.glob("n[0-9]*")):
        try:
            n_templates = int(directory.name[1:])
        except ValueError:
            continue
        path = directory / "subset_quality_matrices.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as saved:
            cached_targetids = np.asarray(saved["targetid"], dtype=np.int64)
            if not np.array_equal(cached_targetids, targetids):
                raise ValueError(f"TARGETID order in {path} does not match the fit table")
            delta = np.asarray(saved["delta_chi2_dof"], dtype=float)
            chi2_dof = np.asarray(saved["reduced_chi2_dof"], dtype=float)
            color_rms = np.asarray(saved["lsst_color_rms"], dtype=float)
            passes = (
                np.isfinite(delta)
                & np.isfinite(chi2_dof)
                & np.isfinite(color_rms)
                & (delta <= max_delta_chi2_dof)
                & (chi2_dof <= max_chi2_dof)
                & (color_rms <= max_color_rms)
            )
            searches[n_templates] = {
                "subsets": np.asarray(saved["subsets"], dtype=int),
                "delta": delta,
                "chi2_dof": chi2_dof,
                "color_rms": color_rms,
                "passes": passes,
            }
    if not searches:
        raise FileNotFoundError(f"No n*/subset_quality_matrices.npz files below {cohort_root}")
    return searches


def subset_label(subset: np.ndarray) -> str:
    return "+".join(f"T{int(value)}" for value in subset)


def decode_family_bases(
    labels: np.ndarray,
    searches: dict[int, dict[str, np.ndarray]],
    *,
    required_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for family in sorted(value for value in np.unique(labels) if value > 0):
        members = labels == family
        family_candidates: list[dict[str, object]] = []
        best_by_n: list[dict[str, object]] = []
        for n_templates, search in sorted(searches.items()):
            coverage = search["passes"][members].mean(axis=0)
            for subset_index, subset in enumerate(search["subsets"]):
                passing = members & search["passes"][:, subset_index]
                row = {
                    "family": f"F{family:02d}",
                    "n_templates": n_templates,
                    "templates": subset_label(subset),
                    "coverage_fraction": float(coverage[subset_index]),
                    "passing_count": int(passing.sum()),
                    "median_delta_chi2_dof": (
                        float(np.median(search["delta"][passing, subset_index]))
                        if np.any(passing)
                        else np.nan
                    ),
                    "p95_delta_chi2_dof": (
                        float(np.percentile(search["delta"][passing, subset_index], 95))
                        if np.any(passing)
                        else np.nan
                    ),
                    "p95_reduced_chi2_dof": (
                        float(np.percentile(search["chi2_dof"][passing, subset_index], 95))
                        if np.any(passing)
                        else np.nan
                    ),
                    "p95_lsst_color_rms": (
                        float(np.percentile(search["color_rms"][passing, subset_index], 95))
                        if np.any(passing)
                        else np.nan
                    ),
                }
                family_candidates.append(row)
            same_n = family_candidates[-len(search["subsets"]) :]
            best_by_n.append(
                sorted(
                    same_n,
                    key=lambda row: (
                        -float(row["coverage_fraction"]),
                        float(row["p95_delta_chi2_dof"]),
                        float(row["p95_lsst_color_rms"]),
                    ),
                )[0]
            )
        candidates.extend(family_candidates)
        selected = next(
            (
                row
                for row in best_by_n
                if float(row["coverage_fraction"]) >= required_coverage
            ),
            None,
        )
        best_tested = max(best_by_n, key=lambda row: float(row["coverage_fraction"]))
        summaries.append(
            {
                "family": f"F{family:02d}",
                "member_count": int(members.sum()),
                "population_fraction": float(members.mean()),
                "selected_n": int(selected["n_templates"]) if selected else np.nan,
                "selected_templates": str(selected["templates"]) if selected else "",
                "selected_coverage_fraction": (
                    float(selected["coverage_fraction"]) if selected else np.nan
                ),
                "meets_required_coverage": selected is not None,
                "best_tested_n": int(best_tested["n_templates"]),
                "best_tested_templates": str(best_tested["templates"]),
                "best_tested_coverage_fraction": float(best_tested["coverage_fraction"]),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(candidates)


def _window_mean(
    wave: np.ndarray,
    templates: np.ndarray,
    bounds: tuple[float, float],
    *,
    fnu: bool = False,
) -> np.ndarray:
    select = (wave >= bounds[0]) & (wave <= bounds[1])
    values = templates[:, select]
    if fnu:
        values = values * wave[select][None, :] ** 2
    return np.trapz(values, wave[select], axis=1) / (bounds[1] - bounds[0])


def _line_ew(
    wave: np.ndarray,
    templates: np.ndarray,
    weights: np.ndarray,
    definition: tuple[tuple[float, float], tuple[float, float], tuple[float, float], float],
) -> np.ndarray:
    line, blue, red, center = definition
    blue_flux = weights @ _window_mean(wave, templates, blue)
    red_flux = weights @ _window_mean(wave, templates, red)
    blue_center = 0.5 * sum(blue)
    red_center = 0.5 * sum(red)
    continuum_center = blue_flux + (red_flux - blue_flux) * (
        (center - blue_center) / (red_center - blue_center)
    )
    select = (wave >= line[0]) & (wave <= line[1])
    line_wave = wave[select]
    line_flux = weights @ templates[:, select]
    continuum = blue_flux[:, None] + (red_flux - blue_flux)[:, None] * (
        (line_wave[None, :] - blue_center) / (red_center - blue_center)
    )
    excess = np.trapz(line_flux - continuum, line_wave, axis=1)
    return np.divide(
        excess,
        continuum_center,
        out=np.full_like(excess, np.nan),
        where=continuum_center > 0,
    )


def compute_spectral_features(
    wave: np.ndarray, templates: np.ndarray, weights: np.ndarray
) -> pd.DataFrame:
    blue = weights @ _window_mean(wave, templates, (3850.0, 3950.0), fnu=True)
    red = weights @ _window_mean(wave, templates, (4000.0, 4100.0), fnu=True)
    uv = weights @ _window_mean(wave, templates, (2000.0, 3000.0), fnu=True)
    optical = weights @ _window_mean(wave, templates, (4500.0, 5500.0), fnu=True)
    nir = weights @ _window_mean(wave, templates, (8000.0, 10000.0), fnu=True)
    values: dict[str, np.ndarray] = {
        "dn4000": np.divide(red, blue, out=np.full_like(red, np.nan), where=blue > 0),
        "uv_to_optical_fnu": np.divide(
            uv, optical, out=np.full_like(uv, np.nan), where=optical > 0
        ),
        "nir_to_optical_fnu": np.divide(
            nir, optical, out=np.full_like(nir, np.nan), where=optical > 0
        ),
    }
    for name, definition in LINE_DEFINITIONS.items():
        values[name] = _line_ew(wave, templates, weights, definition)
    return pd.DataFrame(values)


def summarize_family_properties(
    summary: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    table: pd.DataFrame,
    weights: np.ndarray,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary.copy().set_index("family", drop=False)
    weight_rows: list[dict[str, object]] = []
    for family_number in sorted(value for value in np.unique(labels) if value > 0):
        family = f"F{family_number:02d}"
        members = labels == family_number
        summary.loc[family, "median_membership_probability"] = float(
            np.median(probabilities[members])
        )
        summary.loc[family, "median_z"] = float(np.median(table.loc[members, "z"]))
        summary.loc[family, "z_p10"] = float(np.percentile(table.loc[members, "z"], 10))
        summary.loc[family, "z_p90"] = float(np.percentile(table.loc[members, "z"], 90))
        means = weights[members].mean(axis=0)
        order = np.argsort(means)[::-1]
        summary.loc[family, "top_full_fit_templates"] = "+".join(
            f"T{index + 1}" for index in order[:4]
        )
        for index, mean in enumerate(means):
            values = weights[members, index]
            weight_rows.append(
                {
                    "family": family,
                    "template": f"T{index + 1}",
                    "mean_weight": float(mean),
                    "median_weight": float(np.median(values)),
                    "p10_weight": float(np.percentile(values, 10)),
                    "p90_weight": float(np.percentile(values, 90)),
                    "dominant_fraction": float(
                        np.mean(np.argmax(weights[members], axis=1) == index)
                    ),
                }
            )
        for feature in features.columns:
            values = features.loc[members, feature].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values):
                summary.loc[family, f"median_{feature}"] = float(np.median(values))
                summary.loc[family, f"p10_{feature}"] = float(np.percentile(values, 10))
                summary.loc[family, f"p90_{feature}"] = float(np.percentile(values, 90))
    return summary.reset_index(drop=True), pd.DataFrame(weight_rows)


def make_overview_figure(
    pca: PCA,
    scores: np.ndarray,
    labels: np.ndarray,
    family_summary: pd.DataFrame,
    family_weights: pd.DataFrame,
    *,
    variance_threshold: float,
    pca_scaling: str,
    output: Path,
) -> None:
    families = family_summary["family"].tolist()
    family_numbers = np.asarray([int(value[1:]) for value in families])
    family_colors = plt.get_cmap("tab20")(
        np.linspace(0.0, 0.95, max(len(families), 2))
    )
    color_by_family = {
        number: family_colors[index] for index, number in enumerate(family_numbers)
    }
    fig = plt.figure(figsize=(16, max(10.5, 0.42 * len(families) + 6.3)))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, max(1.0, len(families) / 9.0)])
    ax_components = fig.add_subplot(grid[0, 0])
    ax_scores = fig.add_subplot(grid[0, 1])
    ax_weights = fig.add_subplot(grid[1, 0])
    ax_basis = fig.add_subplot(grid[1, 1])

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    retained = min(
        len(cumulative), int(np.searchsorted(cumulative, variance_threshold) + 1)
    )
    # A unit step in the standardized score for PC j corresponds to
    # sqrt(lambda_j) times its unit loading. Mapping the ILR direction back to
    # CLR makes the signed effect readable as a template log-ratio contrast.
    clr_loadings = pca.components_[:retained] @ ilr_basis(len(TEMPLATE_COLORS)).T
    one_sigma_contrasts = clr_loadings * np.sqrt(
        pca.explained_variance_[:retained, None]
    )
    contrast_limit = float(np.max(np.abs(one_sigma_contrasts)))
    component_image = ax_components.imshow(
        one_sigma_contrasts,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-contrast_limit,
        vmax=contrast_limit,
    )
    ax_components.set_xticks(
        np.arange(len(TEMPLATE_COLORS)),
        [f"T{index + 1}" for index in range(len(TEMPLATE_COLORS))],
    )
    ax_components.set_yticks(
        np.arange(retained),
        [
            f"PC{index + 1} ({100 * pca.explained_variance_ratio_[index]:.1f}%)"
            for index in range(retained)
        ],
    )
    colorbar = fig.colorbar(component_image, ax=ax_components, fraction=0.046, pad=0.03)
    colorbar.set_label("1-SD template log-contrast", fontsize=9)
    ax_components.set_title(
        f"PC contents: {retained} components retain {100 * cumulative[retained - 1]:.1f}%",
        loc="left",
    )

    rng = np.random.default_rng(42)
    shown = rng.choice(len(scores), min(6000, len(scores)), replace=False)
    noise = shown[labels[shown] == 0]
    ax_scores.scatter(
        scores[noise, 0],
        scores[noise, 1],
        s=5,
        color="#C9CDD3",
        alpha=0.28,
        linewidths=0,
        label="unclustered",
    )
    for number in family_numbers:
        rows = shown[labels[shown] == number]
        ax_scores.scatter(
            scores[rows, 0],
            scores[rows, 1],
            s=7,
            color=color_by_family[number],
            alpha=0.55,
            linewidths=0,
            label=f"F{number:02d}",
        )
    score_suffix = "Standardized Score" if pca_scaling == "standardized" else "Raw Score"
    ax_scores.set(xlabel=f"PC1 {score_suffix}", ylabel=f"PC2 {score_suffix}")
    ax_scores.set_title(
        f"Density-discovered spectral families using {scores.shape[1]} PCs",
        loc="left",
    )
    ax_scores.legend(ncol=3, fontsize=8, frameon=False, markerscale=2.0)

    y = np.arange(len(families))
    left = np.zeros(len(families))
    for template_number in range(1, 13):
        means = []
        for family in families:
            row = family_weights.loc[
                (family_weights["family"] == family)
                & (family_weights["template"] == f"T{template_number}"),
                "mean_weight",
            ]
            means.append(float(row.iloc[0]))
        means_array = np.asarray(means)
        ax_weights.barh(
            y,
            means_array,
            left=left,
            color=TEMPLATE_COLORS[template_number - 1],
            edgecolor="white",
            linewidth=0.35,
            label=f"T{template_number}",
        )
        left += means_array
    family_labels = [
        f"{row.family}  (n={int(row.member_count):,})"
        for row in family_summary.itertuples(index=False)
    ]
    ax_weights.set_yticks(y, family_labels)
    ax_weights.invert_yaxis()
    ax_weights.set_xlim(0, 1)
    ax_weights.set_xlabel("Mean full-fit template weight")
    ax_weights.set_title("How each family occupies the 12D basis", loc="left")
    handles, legend_labels = ax_weights.get_legend_handles_labels()

    coverage = family_summary["selected_coverage_fraction"].fillna(
        family_summary["best_tested_coverage_fraction"]
    )
    passed_counts = np.rint(coverage * family_summary["member_count"]).astype(int)
    total_counts = family_summary["member_count"].astype(int)
    failed_counts = total_counts - passed_counts
    ax_basis.barh(
        y,
        passed_counts,
        color="#3366CC",
        edgecolor="#3366CC",
        linewidth=0.8,
        label="passed",
    )
    ax_basis.barh(
        y,
        failed_counts,
        left=passed_counts,
        color="#C8D6EC",
        edgecolor="#3366CC",
        linewidth=0.8,
        label="did not pass",
    )
    subset_labels = []
    text_offset = max(float(total_counts.max()) * 0.012, 5.0)
    for position, row in enumerate(family_summary.itertuples(index=False)):
        subset_labels.append(
            row.selected_templates if row.meets_required_coverage else row.best_tested_templates
        )
        ax_basis.text(
            float(total_counts.iloc[position]) + text_offset,
            position,
            f"{passed_counts.iloc[position]:,} / {total_counts.iloc[position]:,}",
            va="center",
            fontsize=8.5,
        )
    ax_basis.set_yticks(y, subset_labels)
    ax_basis.invert_yaxis()
    ax_basis.set_xlim(0, float(total_counts.max()) * 1.18)
    ax_basis.set_xlabel("DESI spectra")
    ax_basis.set_title("Subset fits: dark=passed; full bar=family size", loc="left")

    for ax in (ax_scores, ax_weights, ax_basis):
        ax.grid(True, color=GRID, alpha=0.65, linewidth=0.7)
        ax.set_axisbelow(True)
    clustered = np.mean(labels > 0)
    fig.suptitle(
        f"EAZY12 DESI family discovery: {len(families)} families contain "
        f"{clustered:.1%} of spectra",
        fontsize=16,
    )
    ax_weights.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=6,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_spectra_figure(
    wave: np.ndarray,
    templates: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    family_summary: pd.DataFrame,
    *,
    output: Path,
) -> None:
    families = family_summary["family"].tolist()
    ncols = 3
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15.5, 3.15 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    shown_wave = (wave >= 3500.0) & (wave <= 7000.0)
    rng = np.random.default_rng(42)
    for ax, row in zip(axes.flat, family_summary.itertuples(index=False)):
        number = int(row.family[1:])
        family_weights = weights[labels == number]
        if len(family_weights) > 1200:
            family_weights = family_weights[
                rng.choice(len(family_weights), 1200, replace=False)
            ]
        spectra = family_weights @ templates[:, shown_wave]
        low, median, high = np.percentile(spectra, [10, 50, 90], axis=0)
        color = plt.get_cmap("tab20")((number - 1) % 20)
        ax.fill_between(wave[shown_wave], low, high, color=color, alpha=0.23)
        ax.plot(wave[shown_wave], median, color=color, lw=1.5)
        basis = (
            row.selected_templates
            if row.meets_required_coverage
            else f"no ≤{int(row.best_tested_n)}-template basis"
        )
        ax.set_title(
            f"{row.family}: {basis}; n={int(row.member_count):,}\n"
            f"Dn4000={row.median_dn4000:.2f}, UV/opt={row.median_uv_to_optical_fnu:.2f}",
            fontsize=10,
        )
        ax.grid(True, color=GRID, alpha=0.55, linewidth=0.6)
        ax.set_yscale("log")
    for ax in axes.flat[len(families) :]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Rest wavelength [$\AA$]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Reconstructed $f_\lambda$")
    fig.suptitle("Median full-fit family spectra (shading: 10th–90th percentile)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_pca_loading_figure(
    pca: PCA,
    n_retained: int,
    n_templates: int,
    *,
    output: Path,
) -> None:
    """Show each retained PC as a template log-ratio contrast."""
    loadings = pca.components_[:n_retained] @ ilr_basis(n_templates).T
    limit = float(np.max(np.abs(loadings)))
    fig, ax = plt.subplots(
        figsize=(12.5, max(5.5, 0.58 * n_retained + 1.8)),
        constrained_layout=True,
    )
    image = ax.imshow(loadings, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(n_templates), [f"T{index + 1}" for index in range(n_templates)])
    ax.set_yticks(
        np.arange(n_retained),
        [
            f"PC{index + 1}  ({100 * pca.explained_variance_ratio_[index]:.1f}%)"
            for index in range(n_retained)
        ],
    )
    for row in range(n_retained):
        for column in range(n_templates):
            color = "white" if abs(loadings[row, column]) > 0.55 * limit else INK
            ax.text(
                column,
                row,
                f"{loadings[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    fig.colorbar(image, ax=ax, label="Template log-contrast loading")
    ax.set_title(
        "Retained ILR principal components decoded into the original EAZY templates",
        loc="left",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_feature_figure(
    family_summary: pd.DataFrame,
    *,
    output: Path,
) -> None:
    features = [
        ("median_dn4000", r"$D_n(4000)$"),
        ("median_uv_to_optical_fnu", r"UV / optical $f_\nu$"),
        ("median_nir_to_optical_fnu", r"NIR / optical $f_\nu$"),
        ("median_ew_oii_3727", r"[O II] EW"),
        ("median_ew_hbeta", r"H$\beta$ EW"),
        ("median_ew_oiii_5007", r"[O III] EW"),
        ("median_ew_halpha", r"H$\alpha$ EW"),
    ]
    matrix = family_summary[[name for name, _ in features]].to_numpy(float)
    center = np.nanmedian(matrix, axis=0)
    scale = np.nanpercentile(matrix, 75, axis=0) - np.nanpercentile(matrix, 25, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    standardized = np.clip((matrix - center) / scale, -2.5, 2.5)
    fig, ax = plt.subplots(
        figsize=(11.5, max(6.5, 0.48 * len(family_summary) + 2.0)),
        constrained_layout=True,
    )
    image = ax.imshow(standardized, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(features)), [label for _, label in features], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(family_summary)), family_summary["family"])
    for row in range(len(family_summary)):
        for column in range(len(features)):
            value = matrix[row, column]
            text = f"{value:.2f}" if np.isfinite(value) else "—"
            color = "white" if abs(standardized[row, column]) > 1.25 else INK
            ax.text(column, row, text, ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, label="Difference from population median [interquartile ranges]")
    ax.set_title("Physical diagnostics of reconstructed family spectra", loc="left")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--build-name", default=DEFAULT_EMPIRICAL_PRIOR_DIR)
    parser.add_argument("--weights-csv", type=Path, default=None)
    parser.add_argument("--cohort-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--template-param", default=DEFAULT_TEMPLATE_PARAM_12D)
    parser.add_argument("--norm-min", type=float, default=DEFAULT_TEMPLATE_NORM_MIN_AA)
    parser.add_argument("--norm-max", type=float, default=DEFAULT_TEMPLATE_NORM_MAX_AA)
    parser.add_argument("--variance-threshold", type=float, default=0.90)
    parser.add_argument(
        "--pca-scaling",
        choices=("standardized", "raw"),
        default="standardized",
        help="Scale supplied to HDBSCAN; standardized is PCA whitening.",
    )
    parser.add_argument("--clr-eps", type=float, default=DEFAULT_CLR_EPS)
    parser.add_argument("--min-cluster-size", type=int, default=300)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--required-family-coverage", type=float, default=0.80)
    parser.add_argument("--max-chi2-dof", type=float, default=1.2)
    parser.add_argument("--max-delta-chi2-dof", type=float, default=0.05)
    parser.add_argument("--max-color-rms", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prior_dir = get_prior_build_dir(args.build_name)
    weights_csv = args.weights_csv or prior_dir / "desi_eazy_empirical_weights.csv"
    cohort_root = args.cohort_root or prior_dir / "reduced_template_cohorts"
    output_dir = args.output_dir or cohort_root / "spectral_families"
    template_dir = args.template_dir or get_template_dir()
    if not 0 < args.variance_threshold <= 1:
        raise ValueError("--variance-threshold must be in (0, 1]")
    if not 0 < args.required_family_coverage <= 1:
        raise ValueError("--required-family-coverage must be in (0, 1]")

    table, weights = load_fit_population(Path(weights_csv))
    pca, scores, labels, probabilities = fit_population_embedding(
        weights,
        variance_threshold=args.variance_threshold,
        clr_eps=args.clr_eps,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        pca_scaling=args.pca_scaling,
    )
    searches = load_subset_searches(
        Path(cohort_root),
        table["targetid"].to_numpy(np.int64),
        max_chi2_dof=args.max_chi2_dof,
        max_delta_chi2_dof=args.max_delta_chi2_dof,
        max_color_rms=args.max_color_rms,
    )
    family_summary, candidates = decode_family_bases(
        labels,
        searches,
        required_coverage=args.required_family_coverage,
    )
    sensitivity = cluster_sensitivity(
        scores,
        labels,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    template_waves, template_fluxes, template_paths = load_eazy_templates(
        args.template_param,
        template_dir=template_dir,
        norm_min=args.norm_min,
        norm_max=args.norm_max,
    )
    wave = np.arange(1200.0, 11000.1, 2.0)
    templates = np.stack(
        [np.interp(wave, tw, tf, left=0.0, right=0.0) for tw, tf in zip(template_waves, template_fluxes)]
    )
    if templates.shape[0] != weights.shape[1]:
        raise ValueError(
            f"Loaded {templates.shape[0]} templates for {weights.shape[1]} fitted weights"
        )
    features = compute_spectral_features(wave, templates, weights)
    family_summary, family_weights = summarize_family_properties(
        family_summary,
        labels,
        probabilities,
        table,
        weights,
        features,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = table[["targetid", "healpix", "z"]].copy()
    assignments["family"] = [f"F{value:02d}" if value > 0 else "" for value in labels]
    assignments["family_number"] = labels
    assignments["membership_probability"] = probabilities
    for index in range(scores.shape[1]):
        assignments[f"pc{index + 1}"] = scores[:, index]
    assignments = pd.concat([assignments, features], axis=1)
    assignments.to_csv(output_dir / "spectrum_family_assignments.csv", index=False)
    family_summary.to_csv(output_dir / "family_summary.csv", index=False)
    family_weights.to_csv(output_dir / "family_template_weights.csv", index=False)
    candidates.to_csv(output_dir / "family_basis_candidates.csv", index=False)
    sensitivity.to_csv(output_dir / "cluster_sensitivity.csv", index=False)

    clr_loadings = pca.components_ @ ilr_basis(weights.shape[1]).T
    loading_table = pd.DataFrame(
        clr_loadings,
        columns=[f"T{index + 1}_log_contrast" for index in range(weights.shape[1])],
    )
    loading_table.insert(0, "explained_variance_ratio", pca.explained_variance_ratio_)
    loading_table.insert(0, "component", np.arange(1, len(loading_table) + 1))
    loading_table.to_csv(output_dir / "pca_template_log_contrasts.csv", index=False)

    parameters = vars(args).copy()
    parameters.update(
        {
            "weights_csv": str(Path(weights_csv).expanduser().resolve()),
            "cohort_root": str(Path(cohort_root).expanduser().resolve()),
            "template_dir": str(Path(template_dir).expanduser().resolve()),
            "template_paths": template_paths,
            "n_input_spectra": len(table),
            "n_retained_pcs": scores.shape[1],
            "n_families": int(labels.max()),
            "clustered_fraction": float(np.mean(labels > 0)),
            "searched_subset_sizes": sorted(searches),
        }
    )
    for key, value in list(parameters.items()):
        if isinstance(value, Path):
            parameters[key] = str(value)
    (output_dir / "family_discovery_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n"
    )

    make_overview_figure(
        pca,
        scores,
        labels,
        family_summary,
        family_weights,
        variance_threshold=args.variance_threshold,
        pca_scaling=args.pca_scaling,
        output=output_dir / "family_discovery_overview.png",
    )
    make_spectra_figure(
        wave,
        templates,
        weights,
        labels,
        family_summary,
        output=output_dir / "family_reconstructed_spectra.png",
    )
    make_pca_loading_figure(
        pca,
        scores.shape[1],
        weights.shape[1],
        output=output_dir / "pca_template_log_contrasts.png",
    )
    make_feature_figure(
        family_summary,
        output=output_dir / "family_spectral_features.png",
    )

    print(
        f"Retained {scores.shape[1]} PCs; found {labels.max()} families containing "
        f"{np.mean(labels > 0):.1%} of {len(labels):,} spectra"
    )
    shown = family_summary[
        [
            "family",
            "member_count",
            "selected_n",
            "selected_templates",
            "selected_coverage_fraction",
            "best_tested_templates",
            "best_tested_coverage_fraction",
            "median_dn4000",
            "median_uv_to_optical_fnu",
        ]
    ]
    print(shown.to_string(index=False))
    print(f"Wrote family analysis to {output_dir}")


if __name__ == "__main__":
    main()
