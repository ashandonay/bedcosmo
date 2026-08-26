#!/usr/bin/env python3
"""Connect dominant EAZY-template cohorts to physical DESI spectral traits.

The 12-template fit is used only to define empirical cohorts. For spectra led
by T1 or T7, this script returns to the DESI coadds and measures rest-frame
Dn4000 and simple local-continuum emission equivalent widths. It also plots
median normalized rest spectra and the conditional mean EAZY coefficients.

The output CSV retains TARGETID and HEALPix, allowing every plotted object to
be traced back to its DESI spectrum.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _paths import plot_path  # noqa: E402

from bedcosmo.num_visits.empirical.desi_data import get_local_desi_paths  # noqa: E402
from bedcosmo.num_visits.empirical.paths import (  # noqa: E402
    DEFAULT_PROGRAM,
    DEFAULT_SPECPROD,
    DEFAULT_SURVEY,
    get_desi_data_dir,
)

INK = "#25272B"
MUTED = "#6B7280"
GRID = "#D9DDE3"
COHORTS = {"T1-led": (0, "#1f77b4"), "T7-led": (6, "#8c564b")}
REST_GRID = np.arange(3500.0, 7000.1, 2.0)


@dataclass(frozen=True)
class LineDefinition:
    center: float
    line: tuple[float, float]
    blue: tuple[float, float]
    red: tuple[float, float]


LINES = {
    "ew_oii_3727": LineDefinition(3727.0, (3716.0, 3739.0), (3655.0, 3705.0), (3755.0, 3805.0)),
    "ew_hbeta": LineDefinition(4861.0, (4851.0, 4872.0), (4800.0, 4835.0), (4885.0, 4920.0)),
    "ew_oiii_5007": LineDefinition(5007.0, (4997.0, 5018.0), (4950.0, 4985.0), (5025.0, 5060.0)),
    "ew_halpha": LineDefinition(6563.0, (6553.0, 6574.0), (6485.0, 6525.0), (6605.0, 6645.0)),
}


def _window_values(
    wave_rest: np.ndarray,
    flux: np.ndarray,
    good: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    select = good & (wave_rest >= bounds[0]) & (wave_rest <= bounds[1])
    return wave_rest[select], flux[select]


def measure_dn4000(wave_rest: np.ndarray, flux: np.ndarray, good: np.ndarray) -> float:
    """Balogh narrow Dn4000 using median f_nu in its two continuum windows."""
    blue_wave, blue_flux = _window_values(wave_rest, flux, good, (3850.0, 3950.0))
    red_wave, red_flux = _window_values(wave_rest, flux, good, (4000.0, 4100.0))
    if min(len(blue_flux), len(red_flux)) < 8:
        return np.nan
    blue = np.median(blue_flux * blue_wave**2)
    red = np.median(red_flux * red_wave**2)
    if not np.isfinite(blue) or not np.isfinite(red) or blue <= 0:
        return np.nan
    return float(red / blue)


def measure_emission_ew(
    wave_rest: np.ndarray,
    flux: np.ndarray,
    good: np.ndarray,
    definition: LineDefinition,
) -> float:
    """Rest-frame emission EW from a linear local continuum; emission is positive."""
    blue_wave, blue_flux = _window_values(wave_rest, flux, good, definition.blue)
    red_wave, red_flux = _window_values(wave_rest, flux, good, definition.red)
    line_wave, line_flux = _window_values(wave_rest, flux, good, definition.line)
    if min(len(blue_flux), len(red_flux), len(line_flux)) < 4:
        return np.nan
    blue = float(np.median(blue_flux))
    red = float(np.median(red_flux))
    blue_center = float(np.median(blue_wave))
    red_center = float(np.median(red_wave))
    continuum = blue + (red - blue) * (line_wave - blue_center) / (red_center - blue_center)
    continuum_at_line = blue + (red - blue) * (definition.center - blue_center) / (
        red_center - blue_center
    )
    if not np.isfinite(continuum_at_line) or continuum_at_line <= 0:
        return np.nan
    return float(np.trapz(line_flux - continuum, line_wave) / continuum_at_line)


def normalized_rest_spectrum(
    wave_rest: np.ndarray,
    flux: np.ndarray,
    good: np.ndarray,
) -> np.ndarray | None:
    _, norm_flux = _window_values(wave_rest, flux, good, (4050.0, 4250.0))
    if len(norm_flux) < 12:
        return None
    norm = float(np.median(norm_flux))
    if not np.isfinite(norm) or norm <= 0:
        return None
    wave_good = wave_rest[good]
    flux_good = flux[good] / norm
    order = np.argsort(wave_good)
    wave_good = wave_good[order]
    flux_good = flux_good[order]
    unique = np.concatenate(([True], np.diff(wave_good) > 0))
    wave_good = wave_good[unique]
    flux_good = flux_good[unique]
    out = np.full_like(REST_GRID, np.nan)
    covered = (REST_GRID >= wave_good[0]) & (REST_GRID <= wave_good[-1])
    out[covered] = np.interp(REST_GRID[covered], wave_good, flux_good)
    return out


def select_cohorts(weights_csv: Path, min_anchor_weight: float) -> pd.DataFrame:
    table = pd.read_csv(weights_csv)
    coeff_columns = sorted(
        (column for column in table if column[0] == "a" and column[1:].isdigit()),
        key=lambda name: int(name[1:]),
    )
    if len(coeff_columns) != 12:
        raise ValueError(f"Expected 12 normalized coefficients, found {coeff_columns}")
    keep = table["success"].astype(bool) & table["quality_pass"].astype(bool)
    table = table.loc[keep].copy()
    weights = table[coeff_columns].to_numpy(float)
    dominant = np.argmax(weights, axis=1)
    second = np.partition(weights, -2, axis=1)[:, -2]
    table["dominant_template"] = dominant + 1
    table["dominance_margin"] = weights.max(axis=1) - second
    labels = np.full(len(table), "", dtype=object)
    for label, (anchor, _) in COHORTS.items():
        selected = (dominant == anchor) & (weights[:, anchor] >= min_anchor_weight)
        labels[selected] = label
    table["cohort"] = labels
    return table.loc[table["cohort"] != ""].copy()


def measure_cohort_spectra(
    selected: pd.DataFrame,
    *,
    desi_dir: Path,
    specprod: str,
    survey: str,
    program: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    records: list[dict] = []
    stacks: dict[str, list[np.ndarray]] = {label: [] for label in COHORTS}
    for healpix, patch in selected.groupby("healpix", sort=True):
        coadd_path, _ = get_local_desi_paths(desi_dir, specprod, survey, program, int(healpix))
        if not coadd_path.is_file():
            raise FileNotFoundError(coadd_path)
        with fits.open(coadd_path, memmap=True) as hdul:
            targetids = np.asarray(hdul["FIBERMAP"].data["TARGETID"], dtype=np.int64)
            target_to_row = {int(targetid): i for i, targetid in enumerate(targetids)}
            arm_waves = {
                arm: np.asarray(hdul[f"{arm}_WAVELENGTH"].data, dtype=float) for arm in "BRZ"
            }
            for _, row in patch.iterrows():
                targetid = int(row["targetid"])
                if targetid not in target_to_row:
                    continue
                index = target_to_row[targetid]
                wave = np.concatenate([arm_waves[arm] for arm in "BRZ"])
                flux = np.concatenate(
                    [np.asarray(hdul[f"{arm}_FLUX"].data[index], dtype=float) for arm in "BRZ"]
                )
                ivar = np.concatenate(
                    [np.asarray(hdul[f"{arm}_IVAR"].data[index], dtype=float) for arm in "BRZ"]
                )
                mask = np.concatenate(
                    [np.asarray(hdul[f"{arm}_MASK"].data[index]) for arm in "BRZ"]
                )
                wave_rest = wave / (1.0 + float(row["z"]))
                good = (
                    np.isfinite(wave_rest)
                    & np.isfinite(flux)
                    & np.isfinite(ivar)
                    & (ivar > 0)
                    & (mask == 0)
                )
                record = row.to_dict()
                record["dn4000"] = measure_dn4000(wave_rest, flux, good)
                for name, definition in LINES.items():
                    record[name] = measure_emission_ew(wave_rest, flux, good, definition)
                records.append(record)
                normalized = normalized_rest_spectrum(wave_rest, flux, good)
                if normalized is not None:
                    stacks[str(row["cohort"])].append(normalized)
        print(f"Measured HEALPix {int(healpix)}: {len(patch):,} selected spectra")
    stack_arrays = {
        label: np.asarray(values, dtype=float) for label, values in stacks.items() if values
    }
    return pd.DataFrame.from_records(records), stack_arrays


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)


def _cohort_violin(ax: plt.Axes, table: pd.DataFrame, column: str, ylabel: str) -> None:
    values = []
    colors = []
    labels = []
    pooled = []
    for label, (_, color) in COHORTS.items():
        data = table.loc[table["cohort"] == label, column].to_numpy(float)
        data = data[np.isfinite(data)]
        values.append(data)
        pooled.append(data)
        colors.append(color)
        labels.append(f"{label}\n$n={len(data):,}$")
    violin = ax.violinplot(values, showmedians=True, showextrema=False, widths=0.75)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.38)
    violin["cmedians"].set_color(INK)
    violin["cmedians"].set_linewidth(1.4)
    finite = np.concatenate([item for item in pooled if len(item)])
    low, high = np.percentile(finite, [1, 99])
    pad = max((high - low) * 0.08, 0.02)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xticks([1, 2], labels)
    ax.set_ylabel(ylabel)
    _style_axis(ax)


def make_figure(
    table: pd.DataFrame,
    stacks: dict[str, np.ndarray],
    *,
    output: Path,
    min_anchor_weight: float,
) -> None:
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.35, 1.0, 1.0])
    ax_stack = fig.add_subplot(grid[0, :])
    ax_dn = fig.add_subplot(grid[1, 0])
    ax_ha = fig.add_subplot(grid[1, 1])
    ax_oiii = fig.add_subplot(grid[1, 2])
    ax_oii = fig.add_subplot(grid[2, 0])
    ax_z = fig.add_subplot(grid[2, 1])
    ax_coeff = fig.add_subplot(grid[2, 2])

    for label, (_, color) in COHORTS.items():
        spectra = stacks[label]
        median = np.nanmedian(spectra, axis=0)
        valid = np.isfinite(median) & (median > 0)
        smoothed = gaussian_filter1d(median[valid], sigma=2.0)
        ax_stack.plot(
            REST_GRID[valid],
            median[valid],
            color=color,
            lw=0.65,
            alpha=0.22,
        )
        ax_stack.plot(
            REST_GRID[valid],
            smoothed,
            color=color,
            lw=2.0,
            label=f"{label} ($n={len(spectra):,}$)",
        )
    for wavelength, label in (
        (3727, "[O II]"),
        (3934, "Ca K"),
        (4000, "4000 Å"),
        (4861, r"H$\beta$"),
        (5007, "[O III]"),
        (6563, r"H$\alpha$"),
    ):
        ax_stack.axvline(wavelength, color=MUTED, lw=0.7, alpha=0.5)
        ax_stack.text(
            wavelength + 10,
            0.98,
            label,
            transform=ax_stack.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="left",
            color=MUTED,
            fontsize=8,
        )
    ax_stack.set_xlim(3500, 7000)
    median_values = np.concatenate([np.nanmedian(values, axis=0) for values in stacks.values()])
    median_values = median_values[np.isfinite(median_values) & (median_values > 0)]
    ax_stack.set_yscale("log")
    ax_stack.set_ylim(np.percentile(median_values, 1) * 0.8, median_values.max() * 1.25)
    ax_stack.set_xlabel(r"Rest wavelength $\lambda$ [$\AA$]")
    ax_stack.set_ylabel(r"Median DESI $f_\lambda$ / median$(4050$–$4250\,\AA)$")
    ax_stack.set_title(
        f"Median DESI spectra (thick: 4 Å smoothing; faint: unsmoothed); log flux axis "
        rf"($a_{{anchor}}\geq {min_anchor_weight:g}$)",
        loc="left",
    )
    ax_stack.legend(frameon=False, loc="upper right")
    _style_axis(ax_stack)

    _cohort_violin(ax_dn, table, "dn4000", r"$D_n(4000)$")
    ax_dn.set_title("4000 Å break (older/quiescent → larger)", loc="left")
    _cohort_violin(ax_ha, table, "ew_halpha", r"H$\alpha$ emission EW [$\AA$]")
    ax_ha.set_title("Recent star formation / ionized gas", loc="left")
    _cohort_violin(ax_oiii, table, "ew_oiii_5007", r"[O III] 5007 EW [$\AA$]")
    ax_oiii.set_title("High-excitation ionized gas", loc="left")
    _cohort_violin(ax_oii, table, "ew_oii_3727", r"[O II] 3727 EW [$\AA$]")
    ax_oii.set_title("Star formation / ionized gas", loc="left")

    bins = np.linspace(table["z"].min(), table["z"].max(), 35)
    for label, (_, color) in COHORTS.items():
        values = table.loc[table["cohort"] == label, "z"].to_numpy(float)
        ax_z.hist(values, bins=bins, density=True, histtype="step", lw=2, color=color, label=label)
    ax_z.set_xlabel("DESI fitted redshift $z$")
    ax_z.set_ylabel("Density")
    ax_z.set_title("Redshift selection", loc="left")
    ax_z.legend(frameon=False)
    _style_axis(ax_z)

    coeff_columns = [f"a{i}" for i in range(1, 13)]
    x = np.arange(1, 13)
    width = 0.38
    for offset, (label, (_, color)) in zip((-width / 2, width / 2), COHORTS.items()):
        mean = table.loc[table["cohort"] == label, coeff_columns].mean().to_numpy(float)
        ax_coeff.bar(x + offset, mean, width=width, color=color, alpha=0.72, label=label)
    ax_coeff.set_xticks(x, [f"T{i}" for i in x], rotation=45)
    ax_coeff.set_xlabel("EAZY template")
    ax_coeff.set_ylabel("Conditional mean fitted weight")
    ax_coeff.set_title("Companion templates for reduced priors", loc="left")
    ax_coeff.legend(frameon=False)
    _style_axis(ax_coeff)

    fig.suptitle("Physical spectral traits of dominant EAZY-template DESI cohorts", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-csv",
        type=Path,
        default=None,
        help="EAZY12 combined weights table (default: eazy12 scratch build)",
    )
    parser.add_argument("--desi-dir", type=Path, default=None)
    parser.add_argument("--specprod", default=DEFAULT_SPECPROD)
    parser.add_argument("--survey", default=DEFAULT_SURVEY)
    parser.add_argument("--program", default=DEFAULT_PROGRAM)
    parser.add_argument(
        "--min-anchor-weight",
        type=float,
        default=0.5,
        help="Require the dominant T1 or T7 coefficient to exceed this value",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scratch = Path(os.environ.get("SCRATCH", Path.home() / "scratch"))
    weights_csv = args.weights_csv or (
        scratch / "bedcosmo/num_visits/empirical_prior/eazy12/desi_eazy_empirical_weights.csv"
    )
    desi_dir = args.desi_dir or get_desi_data_dir()
    output = args.output or plot_path("eazy12_dominant_cohort_spectral_traits.png")
    output_csv = args.output_csv or output.with_suffix(".csv")
    if not 0.0 <= args.min_anchor_weight <= 1.0:
        raise ValueError("--min-anchor-weight must be between 0 and 1")

    selected = select_cohorts(weights_csv, args.min_anchor_weight)
    print(selected["cohort"].value_counts().sort_index().to_string())
    measured, stacks = measure_cohort_spectra(
        selected,
        desi_dir=desi_dir,
        specprod=args.specprod,
        survey=args.survey,
        program=args.program,
    )
    measured.to_csv(output_csv, index=False)
    make_figure(
        measured,
        stacks,
        output=output,
        min_anchor_weight=args.min_anchor_weight,
    )
    print(f"Wrote {output}")
    print(f"Wrote {output_csv}")
    for label in COHORTS:
        cohort = measured[measured["cohort"] == label]
        summary = {
            name: float(np.nanmedian(cohort[name]))
            for name in ("dn4000", "ew_halpha", "ew_oiii_5007", "ew_oii_3727", "z")
        }
        print(label, {key: round(value, 3) for key, value in summary.items()})


if __name__ == "__main__":
    main()
