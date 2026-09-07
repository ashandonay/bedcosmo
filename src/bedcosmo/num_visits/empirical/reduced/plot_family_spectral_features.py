#!/usr/bin/env python
"""Compare spectral shapes and feature power for reduced EAZY family bases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .discover_template_cohorts import (  # noqa: E402
    solve_subset_nnls,
)
from ..templates import load_eazy_templates  # noqa: E402

GRID = "#D9DDE3"
INK = "#25272B"
COLORS = (
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
    "#A51C61",
)
DEFAULT_GROUPS = (
    "F12:T1+T7",
    "F13:T1+T7+T8",
    "F11:T1+T7+T9",
    "F07:T7+T10",
    "F06:T1+T8",
)
EMISSION_WINDOWS = {
    "oii": ((3717.0, 3737.0), (3655.0, 3705.0), (3755.0, 3805.0)),
    "hbeta": ((4851.0, 4871.0), (4800.0, 4835.0), (4885.0, 4920.0)),
    "oiii": ((4997.0, 5017.0), (4950.0, 4985.0), (5025.0, 5060.0)),
    "halpha": ((6553.0, 6573.0), (6485.0, 6525.0), (6605.0, 6645.0)),
}

# Lick/IDS passbands from Worthey et al. (1994) and Worthey & Ottaviani (1997).
# Ca H+K is a deliberately broad custom pseudo-equivalent-width diagnostic.
ABSORPTION_WINDOWS = {
    "Ca H+K": ((3925.0, 3980.0), (3890.0, 3910.0), (4000.0, 4020.0)),
    "G4300": (
        (4281.375, 4316.375),
        (4266.375, 4282.625),
        (4318.875, 4335.125),
    ),
    r"Mg $b$": (
        (5160.125, 5192.625),
        (5142.625, 5161.375),
        (5191.375, 5206.375),
    ),
    "Na D": (
        (5876.875, 5909.375),
        (5860.625, 5875.625),
        (5922.125, 5948.125),
    ),
}

FE5270_WINDOWS = (
    (5245.650, 5285.650),
    (5233.150, 5248.150),
    (5285.650, 5318.150),
)
FE5335_WINDOWS = (
    (5312.125, 5352.125),
    (5304.625, 5315.875),
    (5353.375, 5363.375),
)
EMISSION_LABELS = {
    "oii": "[O II]",
    "hbeta": r"H$\beta$",
    "oiii": "[O III]",
    "halpha": r"H$\alpha$",
}


def _format_angstrom(value: float) -> str:
    return str(int(round(value)))


def ew_label(name: str, *features: tuple[float, float]) -> str:
    """Panel title with the feature integration window(s) in Angstroms."""
    ranges = ", ".join(
        f"{_format_angstrom(lo)}–{_format_angstrom(hi)}" for lo, hi in features
    )
    return rf"{name} EW $[{ranges}\,\AA]$"


@dataclass(frozen=True)
class FamilyBasis:
    family: str
    templates: str
    template_indices: tuple[int, ...]


def parse_family_basis(value: str) -> FamilyBasis:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected FAMILY:T1+T2, e.g. F12:T1+T7")
    family, template_label = value.split(":", 1)
    try:
        indices = tuple(int(token.removeprefix("T")) - 1 for token in template_label.split("+"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid template label {template_label!r}") from error
    if not family or not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError(f"Invalid family basis {value!r}")
    normalized = "+".join(f"T{index + 1}" for index in indices)
    return FamilyBasis(family=family, templates=normalized, template_indices=indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--family-dir", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--statistics-cache", type=Path, default=None)
    parser.add_argument(
        "--family-basis",
        type=parse_family_basis,
        action="append",
        default=None,
        help="Family and reduced basis; repeat for multiple groups.",
    )
    parser.add_argument(
        "--all-families",
        action="store_true",
        help="Use every displayed subset in family_summary.csv.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-chi2-dof", type=float, default=1.2)
    parser.add_argument("--max-delta-chi2-dof", type=float, default=0.05)
    parser.add_argument("--max-color-rms", type=float, default=0.02)
    return parser.parse_args()


def load_reduced_weights(
    group: FamilyBasis,
    assignments: pd.DataFrame,
    cohort_root: Path,
    statistics: dict[str, np.ndarray],
    *,
    max_chi2_dof: float,
    max_delta_chi2_dof: float,
    max_color_rms: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_templates = len(group.template_indices)
    quality_path = cohort_root / f"n{n_templates}" / "subset_quality_matrices.npz"
    with np.load(quality_path, allow_pickle=False) as quality:
        targetids = np.asarray(quality["targetid"], dtype=np.int64)
        expected_targetids = assignments["targetid"].to_numpy(np.int64)
        if not np.array_equal(targetids, expected_targetids):
            raise ValueError(f"TARGETID order in {quality_path} does not match assignments")
        wanted = np.asarray(group.template_indices, dtype=int) + 1
        matching = np.flatnonzero(np.all(quality["subsets"] == wanted[None, :], axis=1))
        if len(matching) != 1:
            raise ValueError(f"Could not uniquely locate {group.templates} in {quality_path}")
        subset_index = int(matching[0])
        passes = (
            np.isfinite(quality["delta_chi2_dof"][:, subset_index])
            & (quality["delta_chi2_dof"][:, subset_index] <= max_delta_chi2_dof)
            & (quality["reduced_chi2_dof"][:, subset_index] <= max_chi2_dof)
            & (quality["lsst_color_rms"][:, subset_index] <= max_color_rms)
        )
    selected = (assignments["family"].to_numpy() == group.family) & passes
    if not np.any(selected):
        raise ValueError(f"No passing {group.family} members found for {group.templates}")
    coefficients, _ = solve_subset_nnls(
        statistics["gram"],
        statistics["cross"],
        statistics["data_norm"],
        group.template_indices,
    )
    weights = coefficients[selected]
    weights /= weights.sum(axis=1, keepdims=True)
    return targetids[selected], weights


def window_mean(
    spectra: np.ndarray,
    wave: np.ndarray,
    bounds: tuple[float, float],
    *,
    fnu: bool = False,
) -> np.ndarray:
    select = (wave >= bounds[0]) & (wave <= bounds[1])
    values = spectra[:, select]
    if fnu:
        values = values * wave[select][None, :] ** 2
    return np.trapz(values, wave[select], axis=1) / (bounds[1] - bounds[0])


def interpolated_continuum(
    spectra: np.ndarray,
    wave: np.ndarray,
    feature_wave: np.ndarray,
    blue: tuple[float, float],
    red: tuple[float, float],
) -> np.ndarray:
    blue_flux = window_mean(spectra, wave, blue)
    red_flux = window_mean(spectra, wave, red)
    blue_center = 0.5 * sum(blue)
    red_center = 0.5 * sum(red)
    return blue_flux[:, None] + (red_flux - blue_flux)[:, None] * (
        (feature_wave[None, :] - blue_center) / (red_center - blue_center)
    )


def pseudo_equivalent_width(
    spectra: np.ndarray,
    wave: np.ndarray,
    feature: tuple[float, float],
    blue: tuple[float, float],
    red: tuple[float, float],
) -> np.ndarray:
    """Integrate EW with one convention: absorption positive, emission negative."""
    select = (wave >= feature[0]) & (wave <= feature[1])
    feature_wave = wave[select]
    continuum = interpolated_continuum(spectra, wave, feature_wave, blue, red)
    ratio = np.divide(
        spectra[:, select],
        continuum,
        out=np.full_like(spectra[:, select], np.nan),
        where=continuum > 0,
    )
    return np.trapz(1.0 - ratio, feature_wave, axis=1)


def diagnostics(spectra: np.ndarray, wave: np.ndarray) -> dict[str, np.ndarray]:
    blue = window_mean(spectra, wave, (3850.0, 3950.0), fnu=True)
    red = window_mean(spectra, wave, (4000.0, 4100.0), fnu=True)
    uv = window_mean(spectra, wave, (2000.0, 3000.0))
    optical_uv = window_mean(spectra, wave, (4500.0, 5500.0))
    nir = window_mean(spectra, wave, (8000.0, 10000.0))
    optical_nir = window_mean(spectra, wave, (4000.0, 8000.0))
    result: dict[str, np.ndarray] = {
        r"$D_n(4000)$": np.divide(
            red, blue, out=np.full_like(red, np.nan), where=blue > 0
        ),
        "UV / optical": np.divide(
            uv,
            optical_uv,
            out=np.full_like(uv, np.nan),
            where=optical_uv > 0,
        ),
        "NIR / optical": np.divide(
            nir,
            optical_nir,
            out=np.full_like(nir, np.nan),
            where=optical_nir > 0,
        ),
    }
    for name, (feature, blue_window, red_window) in ABSORPTION_WINDOWS.items():
        result[ew_label(name, feature)] = pseudo_equivalent_width(
            spectra, wave, feature, blue_window, red_window
        )
    fe5270 = pseudo_equivalent_width(spectra, wave, *FE5270_WINDOWS)
    fe5335 = pseudo_equivalent_width(spectra, wave, *FE5335_WINDOWS)
    result[ew_label(r"$\langle$Fe$\rangle$", FE5270_WINDOWS[0], FE5335_WINDOWS[0])] = (
        0.5 * (fe5270 + fe5335)
    )

    for key, name in EMISSION_LABELS.items():
        feature, blue_window, red_window = EMISSION_WINDOWS[key]
        result[ew_label(name, feature)] = pseudo_equivalent_width(
            spectra, wave, feature, blue_window, red_window
        )
    return result


def summarize_diagnostics(
    group: FamilyBasis,
    targetids: np.ndarray,
    values: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for diagnostic, data in values.items():
        finite = np.asarray(data)[np.isfinite(data)]
        if len(finite):
            p10, median, p90 = np.percentile(finite, [10, 50, 90])
        else:
            p10 = median = p90 = np.nan
        rows.append(
            {
                "family": group.family,
                "templates": group.templates,
                "passing_spectra": len(targetids),
                "diagnostic": diagnostic.replace("$", ""),
                "valid_spectra": len(finite),
                "p10": float(p10),
                "median": float(median),
                "p90": float(p90),
            }
        )
    return rows


def make_figure(
    groups: list[FamilyBasis],
    wave: np.ndarray,
    spectra_by_group: dict[str, np.ndarray],
    diagnostics_by_group: dict[str, dict[str, np.ndarray]],
    counts: dict[str, int],
    output: Path,
) -> None:
    diagnostic_labels = list(next(iter(diagnostics_by_group.values())).keys())
    diagnostic_columns = 2
    diagnostic_rows = int(np.ceil(len(diagnostic_labels) / diagnostic_columns))
    fig = plt.figure(figsize=(22, 13), constrained_layout=True)
    grid = fig.add_gridspec(
        diagnostic_rows, 3, width_ratios=[2.5, 1.0, 1.0], hspace=0.10
    )
    ax_spectra = fig.add_subplot(grid[:, 0])
    feature_axes = []
    for index in range(len(diagnostic_labels)):
        column = index // diagnostic_rows + 1
        row = index % diagnostic_rows
        feature_axes.append(fig.add_subplot(grid[row, column]))
    display = (wave >= 3400.0) & (wave <= 7000.0)

    for index, group in enumerate(groups):
        key = f"{group.family}:{group.templates}"
        color = COLORS[index % len(COLORS)]
        spectra = spectra_by_group[key]
        low, median, high = np.percentile(spectra[:, display], [10, 50, 90], axis=0)
        label = f"{group.family} {group.templates} (n={counts[key]:,})"
        ax_spectra.fill_between(wave[display], low, high, color=color, alpha=0.12)
        ax_spectra.plot(wave[display], median, color=color, lw=1.7, label=label)

        offset = (index - (len(groups) - 1) / 2) * 0.12
        for ax_feature, diagnostic in zip(feature_axes, diagnostic_labels):
            values = diagnostics_by_group[key][diagnostic]
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            q10, med, q90 = np.percentile(values, [10, 50, 90])
            ax_feature.hlines(offset, q10, q90, color=color, lw=1.4)
            ax_feature.scatter(
                med,
                offset,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                s=46,
                zorder=3,
            )

    for feature, _, _ in EMISSION_WINDOWS.values():
        center = 0.5 * sum(feature)
        if 3400 <= center <= 7000:
            ax_spectra.axvline(center, color="#7A7F87", ls=":", lw=0.8)
    ax_spectra.set_yscale("log")
    ax_spectra.set_xlim(3400, 7000)
    ax_spectra.set_xlabel(r"Rest wavelength $\lambda$ [$\AA$]")
    ax_spectra.set_ylabel(r"Pipeline-normalized $f_\lambda$ [$\AA^{-1}$]")
    ax_spectra.set_title("Reduced-fit spectral shapes", loc="left")
    ax_spectra.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
    )

    ratio_diagnostics = {
        r"$D_n(4000)$",
        "UV / optical",
        "NIR / optical",
    }
    for ax_feature, diagnostic in zip(feature_axes, diagnostic_labels):
        reference = 1.0 if diagnostic in ratio_diagnostics else 0.0
        ax_feature.set_ylim(-0.9, 0.9)
        ax_feature.set_yticks([])
        ax_feature.set_title(diagnostic, loc="left", fontsize=10, pad=2)
        ax_feature.margins(x=0.06)
        natural_xlim = ax_feature.get_xlim()
        if natural_xlim[0] <= reference <= natural_xlim[1]:
            ax_feature.axvline(reference, color="#7A7F87", ls="--", lw=1.0)
            ax_feature.set_xlim(natural_xlim)
    feature_axes[0].set_title(
        "Continuum and feature power\n" + diagnostic_labels[0], loc="left"
    )
    if len(diagnostic_labels) < diagnostic_rows * diagnostic_columns:
        key_ax = fig.add_subplot(grid[-1, -1])
        key_ax.axis("off")
        key_ax.text(
            0.5,
            0.5,
            r"Independent x scales; EW: $+$ absorption, $-$ emission",
            ha="center",
            va="center",
            transform=key_ax.transAxes,
        )

    for ax in (ax_spectra, *feature_axes):
        ax.grid(True, color=GRID, alpha=0.65, linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Spectral features encoded by reduced EAZY12 family bases", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.all_families and args.family_basis:
        raise ValueError("Use either --all-families or --family-basis, not both")
    if args.all_families:
        family_summary = pd.read_csv(args.family_dir / "family_summary.csv")
        groups = []
        for row in family_summary.itertuples(index=False):
            templates = (
                row.selected_templates
                if bool(row.meets_required_coverage)
                else row.best_tested_templates
            )
            groups.append(parse_family_basis(f"{row.family}:{templates}"))
    else:
        groups = args.family_basis or [parse_family_basis(value) for value in DEFAULT_GROUPS]
    assignments = pd.read_csv(args.family_dir / "spectrum_family_assignments.csv")
    statistics_cache = args.statistics_cache or args.cohort_root / "sufficient_statistics.npz"
    with np.load(statistics_cache, allow_pickle=False) as saved:
        cache_targetids = np.asarray(saved["targetid"], dtype=np.int64)
        if not np.array_equal(cache_targetids, assignments["targetid"].to_numpy(np.int64)):
            raise ValueError(f"TARGETID order in {statistics_cache} does not match assignments")
        statistics = {
            name: np.asarray(saved[name]) for name in ("gram", "cross", "data_norm")
        }
    template_waves, template_fluxes, _ = load_eazy_templates(
        template_dir=args.template_dir
    )
    wave = np.arange(1200.0, 11000.1, 2.0)
    template_bank = np.stack(
        [
            np.interp(wave, template_wave, template_flux, left=0.0, right=0.0)
            for template_wave, template_flux in zip(template_waves, template_fluxes)
        ]
    )

    spectra_by_group: dict[str, np.ndarray] = {}
    diagnostics_by_group: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, int] = {}
    summary_rows: list[dict[str, object]] = []
    for group in groups:
        targetids, weights = load_reduced_weights(
            group,
            assignments,
            args.cohort_root,
            statistics,
            max_chi2_dof=args.max_chi2_dof,
            max_delta_chi2_dof=args.max_delta_chi2_dof,
            max_color_rms=args.max_color_rms,
        )
        spectra = weights @ template_bank[np.asarray(group.template_indices)]
        values = diagnostics(spectra, wave)
        key = f"{group.family}:{group.templates}"
        spectra_by_group[key] = spectra
        diagnostics_by_group[key] = values
        counts[key] = len(targetids)
        summary_rows.extend(summarize_diagnostics(group, targetids, values))
        print(f"{group.family} {group.templates}: {len(targetids):,} passing spectra")

    make_figure(
        groups,
        wave,
        spectra_by_group,
        diagnostics_by_group,
        counts,
        args.output,
    )
    output_csv = args.output_csv or args.output.with_suffix(".csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_csv, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
