"""Plot DESI fit examples for one or more reduced EAZY template subsets.

Each gallery compares the observed DESI coadd with the original full-template
fit and the reduced-subset refit.  The reduced model is also decomposed into
its individual template contributions.  Examples are chosen to include a
representative cohort member and one member rich in each selected template.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from astropy.io import fits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .desi_data import get_local_desi_paths
from .discover_template_cohorts import (
    _coefficient_columns,
    _read_target_spectrum,
    load_quality_fit_table,
)
from .fit_eazy_weights_to_desi import (
    _divide_by_continuum,
    _gaussian_smooth_segments,
    build_template_matrix_on_observed_grid,
)
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
    load_eazy_templates,
)

INK = "#25272B"
OBSERVED = "#7A7F87"
FULL_FIT = "#D1495B"
REDUCED_FIT = "#2563B8"
GRID = "#D9DDE3"
COMPONENT_COLORS = ("#2A9D8F", "#E9C46A", "#F4A261", "#8E6BBE", "#4C78A8", "#E45756")


def parse_template_label(label: str) -> tuple[int, ...]:
    pieces = label.upper().replace(",", "+").split("+")
    try:
        values = tuple(int(piece.strip().removeprefix("T")) for piece in pieces if piece.strip())
    except ValueError as error:
        raise ValueError(f"Invalid template subset: {label}") from error
    if not values or len(values) != len(set(values)) or any(value < 1 for value in values):
        raise ValueError(f"Invalid template subset: {label}")
    return tuple(sorted(values))


def template_label(subset: tuple[int, ...]) -> str:
    return "+".join(f"T{value}" for value in subset)


def select_examples(members: pd.DataFrame, subset: tuple[int, ...]) -> pd.DataFrame:
    """Choose one representative and one component-rich member per template."""
    a_columns = [f"a_{position + 1}" for position in range(len(subset))]
    missing = set(a_columns).difference(members.columns)
    if missing:
        raise ValueError(f"Membership table lacks columns: {sorted(missing)}")
    weights = members[a_columns].to_numpy(float)
    features = np.column_stack(
        [
            weights,
            members["z"].to_numpy(float),
            members["delta_chi2_dof"].to_numpy(float),
        ]
    )
    center = np.nanmedian(features, axis=0)
    scale = np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    distance = np.nansum(((features - center) / scale) ** 2, axis=1)
    chosen: list[int] = [int(np.nanargmin(distance))]
    roles = ["representative"]
    for position, template in enumerate(subset):
        for index in np.argsort(weights[:, position])[::-1]:
            if int(index) not in chosen:
                chosen.append(int(index))
                roles.append(f"T{template}-rich")
                break
    selected = members.iloc[chosen].copy().reset_index(drop=True)
    selected.insert(0, "example_role", roles)
    return selected


def load_spectra(
    selected: pd.DataFrame,
    *,
    desi_dir: Path,
    specprod: str,
    survey: str,
    program: str,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    spectra = {}
    for healpix, patch in selected.groupby("healpix", sort=True):
        coadd_path, _ = get_local_desi_paths(desi_dir, specprod, survey, program, int(healpix))
        if not coadd_path.is_file():
            raise FileNotFoundError(coadd_path)
        with fits.open(coadd_path, memmap=True) as hdul:
            targetids = np.asarray(hdul["FIBERMAP"].data["TARGETID"], dtype=np.int64)
            target_to_row = {int(targetid): row for row, targetid in enumerate(targetids)}
            arm_waves = {
                arm: np.asarray(hdul[f"{arm}_WAVELENGTH"].data, dtype=float) for arm in "BRZ"
            }
            for row in patch.itertuples(index=False):
                targetid = int(row.targetid)
                if targetid not in target_to_row:
                    raise KeyError(f"TARGETID {targetid} is missing from {coadd_path}")
                spectra[targetid] = _read_target_spectrum(hdul, arm_waves, target_to_row[targetid])
    return spectra


def _robust_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    low, high = np.percentile(values, [1.0, 99.5])
    span = max(high - low, 1e-8)
    return low - 0.08 * span, high + 0.12 * span


def divide_by_display_continuum(
    values: np.ndarray, continuum: np.ndarray, safe: np.ndarray
) -> np.ndarray:
    """Divide only valid display pixels, leaving all other positions NaN."""
    divided = np.full_like(values, np.nan, dtype=float)
    np.divide(values, continuum, out=divided, where=safe)
    return divided


def valid_display_continuum(
    continuum: np.ndarray,
    good: np.ndarray,
    *,
    min_fraction: float,
    continuum_ivar: np.ndarray | None = None,
    min_snr: float = 0.0,
) -> np.ndarray:
    """Select continuum values that are reliably positive for display division."""
    positive = good & np.isfinite(continuum) & (continuum > 0)
    if not np.any(positive):
        return positive
    # Anchor the cutoff to the well-measured continuum rather than the median:
    # spectra with a long near-zero blue tail can otherwise make the reference
    # itself nearly zero and permit unstable display division.
    reference = float(np.nanpercentile(continuum[positive], 75.0))
    if not np.isfinite(reference) or reference <= 0:
        return np.zeros_like(good, dtype=bool)
    valid = positive & (continuum >= min_fraction * reference)
    if continuum_ivar is not None and min_snr > 0:
        continuum_snr = continuum * np.sqrt(np.clip(continuum_ivar, 0.0, None))
        valid &= np.isfinite(continuum_snr) & (continuum_snr >= min_snr)
    return valid


def continuum_sigma_in_observed_frame(sigma_aa: float, frame: str, redshift: float) -> float:
    """Convert a continuum scale to the coordinates of the DESI coadd."""
    if frame == "observed":
        return sigma_aa
    if frame == "rest":
        return sigma_aa * (1.0 + redshift)
    raise ValueError(f"Unknown continuum frame: {frame}")


def ivar_bin_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    good: np.ndarray,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance-bin a spectrum for display without changing the fit."""
    wave_good = wave[good]
    flux_good = flux[good]
    ivar_good = ivar[good]
    start = np.floor(wave_good.min() / bin_width) * bin_width
    indices = np.floor((wave_good - start) / bin_width).astype(int)
    n_bins = int(indices.max()) + 1
    weight_sum = np.bincount(indices, weights=ivar_good, minlength=n_bins)
    weighted_wave = np.bincount(indices, weights=ivar_good * wave_good, minlength=n_bins)
    weighted_flux = np.bincount(indices, weights=ivar_good * flux_good, minlength=n_bins)
    keep = weight_sum > 0
    return (
        weighted_wave[keep] / weight_sum[keep],
        weighted_flux[keep] / weight_sum[keep],
        np.sqrt(1.0 / weight_sum[keep]),
    )


def make_gallery(
    selected: pd.DataFrame,
    full_table: pd.DataFrame,
    spectra: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    subset: tuple[int, ...],
    template_waves: list[np.ndarray],
    template_fluxes: list[np.ndarray],
    output: Path,
    y_scale: str = "linear",
    display_bin_aa: float = 25.0,
    continuum_normalize: bool = True,
    continuum_sigma_aa: float = 250.0,
    continuum_frame: str = "rest",
    continuum_min_fraction: float = 0.05,
    continuum_min_snr: float = 3.0,
    display_smooth_aa: float = 8.0,
    wavelength_frame: str = "observed",
    shared_x: bool = False,
    redshift_window: tuple[float, float] | None = None,
) -> None:
    n_examples = len(selected)
    n_full_templates = len(template_waves)
    full_by_targetid = full_table.set_index("targetid", drop=False)
    full_c_columns = _coefficient_columns(full_table, "c")
    shared_wave_min = shared_wave_max = np.nan
    if shared_x:
        covered_waves = []
        for member in selected.itertuples(index=False):
            wave, flux, ivar, mask = spectra[int(member.targetid)]
            good = (
                np.isfinite(wave) & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (mask == 0)
            )
            display_wave = (
                wave if wavelength_frame == "observed" else wave / (1.0 + float(member.z))
            )
            covered_waves.append(display_wave[good])
        shared_wave_min = min(np.min(wave) for wave in covered_waves)
        shared_wave_max = max(np.max(wave) for wave in covered_waves)
    fig, axes = plt.subplots(
        n_examples,
        2,
        figsize=(15, max(7.0, 3.15 * n_examples)),
        gridspec_kw={"width_ratios": [4.5, 1.0]},
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.055,
        top=0.895,
        hspace=0.42,
        wspace=0.08,
    )
    component_handles = []
    main_handles = []
    for example_index, member in selected.iterrows():
        ax, ax_weights = axes[example_index]
        targetid = int(member["targetid"])
        full_row = full_by_targetid.loc[targetid]
        wave, flux, ivar, mask = spectra[targetid]
        redshift = float(member["z"])
        matrix = build_template_matrix_on_observed_grid(
            wave, redshift, template_waves, template_fluxes
        )
        full_coefficients = full_row[full_c_columns].to_numpy(float)
        reduced_coefficients = np.asarray(
            [member[f"c_{position + 1}"] for position in range(len(subset))], dtype=float
        )
        subset_indices = np.asarray(subset, dtype=int) - 1
        components = matrix[:, subset_indices] * reduced_coefficients[None, :]
        full_model = matrix @ full_coefficients
        reduced_model = components.sum(axis=1)
        good = (
            np.isfinite(wave)
            & np.isfinite(flux)
            & np.isfinite(ivar)
            & (ivar > 0)
            & (mask == 0)
            & np.isfinite(full_model)
            & np.isfinite(reduced_model)
        )
        display_wave = wave if wavelength_frame == "observed" else wave / (1.0 + redshift)
        full_plot = np.where(good, full_model, np.nan)
        reduced_plot = np.where(good, reduced_model, np.nan)
        component_plots = [np.where(good, component, np.nan) for component in components.T]
        if continuum_normalize:
            continuum_sigma_observed = continuum_sigma_in_observed_frame(
                continuum_sigma_aa, continuum_frame, redshift
            )
            observed_plot, continuum, continuum_ivar = _divide_by_continuum(
                wave,
                flux,
                good,
                cont_sigma_aa=continuum_sigma_observed,
                ivar=ivar,
            )
            safe = valid_display_continuum(
                continuum,
                good,
                min_fraction=continuum_min_fraction,
                continuum_ivar=continuum_ivar,
                min_snr=continuum_min_snr,
            )
            observed_plot = np.where(safe, observed_plot, np.nan)
            full_plot = divide_by_display_continuum(full_model, continuum, safe)
            reduced_plot = divide_by_display_continuum(reduced_model, continuum, safe)
            component_plots = [
                divide_by_display_continuum(component, continuum, safe)
                for component in components.T
            ]
            if display_smooth_aa > 0:
                observed_plot = _gaussian_smooth_segments(
                    wave, observed_plot, safe, sigma_aa=display_smooth_aa
                )
                full_plot = _gaussian_smooth_segments(
                    wave, full_plot, safe, sigma_aa=display_smooth_aa
                )
                reduced_plot = _gaussian_smooth_segments(
                    wave, reduced_plot, safe, sigma_aa=display_smooth_aa
                )
                component_plots = [
                    _gaussian_smooth_segments(wave, component, safe, sigma_aa=display_smooth_aa)
                    for component in component_plots
                ]
            observed_for_limits = observed_plot
            (observed_handle,) = ax.plot(
                display_wave,
                observed_plot,
                color=INK,
                lw=0.8,
                alpha=0.8,
                label=rf"DESI coadd ({display_smooth_aa:g} $\AA$ display smoothing)",
            )
        elif display_bin_aa > 0:
            observed_wave, observed_flux, observed_error = ivar_bin_spectrum(
                display_wave, flux, ivar, good, display_bin_aa
            )
            observed_handle = ax.errorbar(
                observed_wave,
                observed_flux,
                yerr=observed_error,
                color=OBSERVED,
                fmt=".",
                ms=2.2,
                elinewidth=0.45,
                capsize=0,
                alpha=0.55,
                label=f"DESI ({display_bin_aa:g} Å bins; no smoothing)",
            )
            observed_for_limits = np.concatenate(
                [observed_flux - observed_error, observed_flux + observed_error]
            )
        else:
            observed_for_limits = np.where(good, flux, np.nan)
            observed_handle = ax.scatter(
                display_wave[good],
                flux[good],
                color=OBSERVED,
                s=1.0,
                alpha=0.2,
                linewidths=0,
                rasterized=True,
                label="DESI pixels",
            )
        (full_handle,) = ax.plot(
            display_wave, full_plot, color=FULL_FIT, lw=1.0, ls="--", label="full fit"
        )
        (reduced_handle,) = ax.plot(
            display_wave, reduced_plot, color=REDUCED_FIT, lw=1.25, label="reduced fit"
        )
        if example_index == 0:
            main_handles = [observed_handle, full_handle, reduced_handle]
        for position, template in enumerate(subset):
            contribution = component_plots[position]
            (handle,) = ax.plot(
                display_wave,
                contribution,
                color=COMPONENT_COLORS[position],
                lw=0.9,
                alpha=0.9,
                label=f"T{template} contribution",
            )
            if example_index == 0:
                component_handles.append(handle)
        ax.set_ylim(*_robust_limits(observed_for_limits, full_plot, reduced_plot))
        if y_scale == "symlog":
            model_amplitude = np.nanpercentile(np.abs(reduced_plot[good]), 95)
            linthresh = max(model_amplitude * 0.01, np.finfo(float).tiny)
            ax.set_yscale("symlog", linthresh=linthresh, linscale=0.7)
            ymin, ymax = ax.get_ylim()
            largest_power = int(np.ceil(np.log10(max(abs(ymin), abs(ymax)))))
            powers = 10.0 ** np.arange(largest_power - 6, largest_power + 1)
            positive_ticks = powers[(powers >= 10.0 * linthresh) & (powers <= ymax)][-3:]
            negative_ticks = -powers[(powers >= 10.0 * linthresh) & (powers <= -ymin)][-3:]
            ax.set_yticks([*negative_ticks[::-1], 0.0, *positive_ticks])
        if shared_x:
            ax.set_xlim(shared_wave_min, shared_wave_max)
        else:
            ax.set_xlim(np.nanmin(display_wave[good]), np.nanmax(display_wave[good]))
        ax.grid(True, color=GRID, alpha=0.65, linewidth=0.7)
        ax.set_axisbelow(True)
        if continuum_normalize:
            ax.set_ylabel(r"$f_\lambda / f_{\mathrm{cont}}$")
        else:
            ax.set_ylabel(rf"DESI $f_\lambda$ ({y_scale})")
        if example_index == n_examples - 1:
            frame_label = "Observed" if wavelength_frame == "observed" else "Rest"
            ax.set_xlabel(rf"{frame_label} wavelength [$\mathrm{{\AA}}$]")
        role = str(member["example_role"])
        ax.set_title(
            f"{role}: TARGETID {targetid}, z={redshift:.3f}  |  "
            rf"full $\chi^2$/dof={float(full_row['chi2_dof']):.3f}, "
            rf"reduced={float(member['reduced_chi2_dof']):.3f}",
            loc="left",
            fontsize=9.5,
        )

        normalized = np.asarray(
            [member[f"a_{position + 1}"] for position in range(len(subset))], dtype=float
        )
        positions = np.arange(len(subset))
        ax_weights.barh(
            positions,
            normalized,
            color=COMPONENT_COLORS[: len(subset)],
            edgecolor=INK,
            linewidth=0.5,
        )
        ax_weights.set_yticks(positions, [f"T{template}" for template in subset])
        ax_weights.invert_yaxis()
        ax_weights.set_xlim(0, 1)
        ax_weights.set_xlabel("Reduced weight")
        for position, value in enumerate(normalized):
            ax_weights.text(
                min(value + 0.025, 0.96),
                position,
                f"{value:.0%}",
                va="center",
                ha="left" if value < 0.9 else "right",
                fontsize=8,
                color=INK,
            )
        ax_weights.grid(True, axis="x", color=GRID, alpha=0.65, linewidth=0.7)
        ax_weights.set_axisbelow(True)

    fig.legend(
        [*main_handles, *component_handles],
        [handle.get_label() for handle in [*main_handles, *component_handles]],
        loc="upper center",
        ncol=min(3 + len(subset), 7),
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
    )
    title = f"Full {n_full_templates}-template EAZY fits vs reduced {template_label(subset)}"
    if redshift_window is not None:
        title += rf"  |  {redshift_window[0]:.2f} $\leq z \leq$ {redshift_window[1]:.2f}"
    if continuum_normalize:
        title += (
            rf"  |  continuum-normalized display "
            rf"($\sigma={continuum_sigma_aa:g}$ $\mathrm{{\AA}}$ {continuum_frame})"
        )
    fig.suptitle(title, fontsize=15, y=0.985)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument(
        "--templates",
        action="append",
        default=[],
        help="Subset such as T1+T7+T8; repeat to plot multiple subsets.",
    )
    parser.add_argument("--top-sets", type=int, default=0)
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
    parser.add_argument("--y-scale", choices=("linear", "symlog"), default="linear")
    parser.add_argument(
        "--display-bin-aa",
        type=float,
        default=25.0,
        help="Inverse-variance bin width for the unnormalized DESI trace; 0 shows pixels.",
    )
    parser.add_argument(
        "--no-continuum-normalize",
        action="store_true",
        help="Show flux directly instead of dividing all traces by the DESI-derived continuum.",
    )
    parser.add_argument(
        "--continuum-sigma-aa",
        type=float,
        default=250.0,
        help="Gaussian sigma used for display-only continuum estimation.",
    )
    parser.add_argument(
        "--continuum-frame",
        choices=("rest", "observed"),
        default="rest",
        help="Frame in which --continuum-sigma-aa is defined.",
    )
    parser.add_argument(
        "--display-smooth-aa",
        type=float,
        default=8.0,
        help="Observed-frame Gaussian sigma applied after continuum normalization for display.",
    )
    parser.add_argument(
        "--continuum-min-fraction",
        type=float,
        default=0.05,
        help="Mask continuum below this fraction of its within-spectrum 75th percentile.",
    )
    parser.add_argument(
        "--continuum-min-snr",
        type=float,
        default=3.0,
        help="Mask regions where the smoothed continuum estimate is below this S/N.",
    )
    parser.add_argument(
        "--wavelength-frame",
        choices=("observed", "rest"),
        default="observed",
        help="Wavelength coordinates used in the gallery.",
    )
    parser.add_argument(
        "--shared-x",
        action="store_true",
        help="Use the union rest-wavelength range for every row instead of local coverage.",
    )
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument(
        "--distinct-across-subsets",
        action="store_true",
        help="Do not reuse a TARGETID when plotting several subsets.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.display_bin_aa < 0:
        raise ValueError("--display-bin-aa must be nonnegative")
    if args.continuum_sigma_aa <= 0:
        raise ValueError("--continuum-sigma-aa must be positive")
    if args.display_smooth_aa < 0:
        raise ValueError("--display-smooth-aa must be nonnegative")
    if not 0 <= args.continuum_min_fraction < 1:
        raise ValueError("--continuum-min-fraction must be in [0, 1)")
    if args.continuum_min_snr < 0:
        raise ValueError("--continuum-min-snr must be nonnegative")
    if (args.z_min is None) != (args.z_max is None):
        raise ValueError("Provide both --z-min and --z-max")
    if args.z_min is not None and args.z_min > args.z_max:
        raise ValueError("--z-min must not exceed --z-max")
    prior_dir = get_prior_build_dir(args.build_name)
    weights_csv = args.weights_csv or prior_dir / "desi_eazy_empirical_weights.csv"
    memberships = pd.read_csv(args.cohort_dir / "subset_memberships.csv")
    labels = list(args.templates)
    if args.top_sets:
        summary = pd.read_csv(args.cohort_dir / "subset_summary.csv")
        labels.extend(summary.head(args.top_sets)["templates"].astype(str).tolist())
    if not labels:
        raise ValueError("Provide --templates and/or a positive --top-sets")
    subsets = list(dict.fromkeys(parse_template_label(label) for label in labels))
    full_table = load_quality_fit_table(weights_csv)
    template_waves, template_fluxes, _ = load_eazy_templates(
        args.template_param,
        template_dir=args.template_dir or get_template_dir(),
        norm_min=args.norm_min,
        norm_max=args.norm_max,
    )
    output_dir = args.output_dir or args.cohort_dir / "fit_examples"
    used_targetids: set[int] = set()
    redshift_window = (args.z_min, args.z_max) if args.z_min is not None else None
    for subset in subsets:
        label = template_label(subset)
        members = memberships.loc[memberships["templates"] == label].copy()
        if redshift_window is not None:
            members = members.loc[members["z"].between(*redshift_window)].copy()
        if args.distinct_across_subsets:
            members = members.loc[~members["targetid"].isin(used_targetids)].copy()
        if members.empty:
            raise ValueError(f"No eligible passing members found for {label}")
        selected = select_examples(members, subset)
        if args.distinct_across_subsets:
            used_targetids.update(selected["targetid"].astype(int))
        spectra = load_spectra(
            selected,
            desi_dir=args.desi_dir or get_desi_data_dir(),
            specprod=args.specprod,
            survey=args.survey,
            program=args.program,
        )
        stem = label.lower().replace("+", "-")
        output = output_dir / f"{stem}_fit_examples.png"
        make_gallery(
            selected,
            full_table,
            spectra,
            subset=subset,
            template_waves=template_waves,
            template_fluxes=template_fluxes,
            output=output,
            y_scale=args.y_scale,
            display_bin_aa=args.display_bin_aa,
            continuum_normalize=not args.no_continuum_normalize,
            continuum_sigma_aa=args.continuum_sigma_aa,
            continuum_frame=args.continuum_frame,
            continuum_min_fraction=args.continuum_min_fraction,
            continuum_min_snr=args.continuum_min_snr,
            display_smooth_aa=args.display_smooth_aa,
            wavelength_frame=args.wavelength_frame,
            shared_x=args.shared_x,
            redshift_window=redshift_window,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        selected.to_csv(output_dir / f"{stem}_selected_examples.csv", index=False)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
