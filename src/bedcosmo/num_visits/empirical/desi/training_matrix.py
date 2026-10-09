"""Construct the masked rest-frame training matrix from DESI coadds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

from ..desi_data import get_local_desi_paths
from ..paths import DEFAULT_PROGRAM, DEFAULT_SPECPROD, DEFAULT_SURVEY


def discover_desi_manifest(
    healpix: list[int] | tuple[int, ...],
    *,
    desi_dir: Path,
    target_spectype: str = "GALAXY",
    z_min: float | None = 0.01,
    z_max: float | None = None,
    allow_nonzero_zwarn: bool = False,
    zwarn_forbid_mask: int | None = None,
    specprod: str = DEFAULT_SPECPROD,
    survey: str = DEFAULT_SURVEY,
    program: str = DEFAULT_PROGRAM,
) -> pd.DataFrame:
    """Select basis-training targets directly from DESI Redrock and FIBERMAP."""
    frames: list[pd.DataFrame] = []
    for hp in healpix:
        coadd_path, redrock_path = get_local_desi_paths(
            desi_dir, specprod, survey, program, int(hp)
        )
        if not coadd_path.is_file():
            raise FileNotFoundError(coadd_path)
        if not redrock_path.is_file():
            raise FileNotFoundError(redrock_path)
        redrock = fits.getdata(redrock_path, "REDSHIFTS")
        fibermap = fits.getdata(coadd_path, "FIBERMAP")
        names = set(redrock.dtype.names or ())
        required = {"TARGETID", "Z", "ZWARN", "SPECTYPE"}
        missing = required.difference(names)
        if missing:
            raise ValueError(f"Redrock table {redrock_path} lacks {sorted(missing)}")

        targetid = np.asarray(redrock["TARGETID"], dtype=np.int64)
        redshift = np.asarray(redrock["Z"], dtype=float)
        zwarn = np.asarray(redrock["ZWARN"], dtype=np.int64)
        spectype = np.char.strip(np.asarray(redrock["SPECTYPE"]).astype(str))
        fiber_targetid = np.asarray(fibermap["TARGETID"], dtype=np.int64)
        select = np.isfinite(redshift) & np.isin(targetid, fiber_targetid)
        if target_spectype:
            select &= spectype == target_spectype
        if zwarn_forbid_mask is not None:
            select &= (zwarn & int(zwarn_forbid_mask)) == 0
        elif not allow_nonzero_zwarn:
            select &= zwarn == 0
        if z_min is not None:
            select &= redshift >= float(z_min)
        if z_max is not None:
            select &= redshift <= float(z_max)

        patch = pd.DataFrame(
            {
                "targetid": targetid[select],
                "healpix": np.full(np.count_nonzero(select), int(hp), dtype=np.int64),
                "z": redshift[select],
            }
        ).drop_duplicates("targetid")
        frames.append(patch)
        print(
            f"Selected direct DESI galaxies for HEALPix {int(hp)}: "
            f"{len(patch):,}/{len(redrock):,} Redrock rows"
        )
    if not frames:
        return pd.DataFrame(columns=["targetid", "healpix", "z"])
    manifest = pd.concat(frames, ignore_index=True)
    return manifest.drop_duplicates("targetid").reset_index(drop=True)


def load_desi_manifest(path: Path | str) -> pd.DataFrame:
    """Load the DESI object identity/redshift columns used for basis training."""
    table = pd.read_csv(path)
    required = {"targetid", "healpix", "z"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if {"success", "quality_pass"}.issubset(table.columns):
        table = table.loc[table["success"].astype(bool) & table["quality_pass"].astype(bool)]
    table = table.loc[np.isfinite(table["z"].to_numpy(float)) & (table["z"].to_numpy(float) >= 0)]
    return table[["targetid", "healpix", "z"]].drop_duplicates("targetid").reset_index(drop=True)


def _robust_flux_scale(rest_wave: np.ndarray, flux: np.ndarray, ivar: np.ndarray) -> float:
    preferred = (
        (rest_wave >= 3600.0)
        & (rest_wave <= 7000.0)
        & np.isfinite(flux)
        & np.isfinite(ivar)
        & (ivar > 0)
    )
    values = flux[preferred]
    if len(values) < 50:
        values = flux[np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)]
    if not len(values):
        return np.nan
    scale = float(np.median(values))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.percentile(np.abs(values), 60))
    return scale if np.isfinite(scale) and scale > 0 else np.nan


def bin_rest_frame_spectrum(
    wave_obs: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    redshift: float,
    rest_wave: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Inverse-variance bin one masked DESI spectrum onto a rest-frame grid."""
    wave_obs = np.asarray(wave_obs, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    mask = np.asarray(mask)
    rest = wave_obs / (1.0 + float(redshift))
    good = (
        np.isfinite(rest)
        & np.isfinite(flux)
        & np.isfinite(ivar)
        & (ivar > 0)
        & (mask == 0)
        & (rest >= rest_wave[0])
        & (rest <= rest_wave[-1])
    )
    scale = _robust_flux_scale(rest[good], flux[good], ivar[good])
    values = np.zeros(len(rest_wave), dtype=np.float32)
    weights = np.zeros(len(rest_wave), dtype=np.float32)
    if not np.isfinite(scale):
        return values, weights, np.nan

    step = float(np.median(np.diff(rest_wave)))
    index = np.rint((rest[good] - rest_wave[0]) / step).astype(int)
    valid = (index >= 0) & (index < len(rest_wave))
    index = index[valid]
    scaled_flux = flux[good][valid] / scale
    scaled_ivar = ivar[good][valid] * scale**2
    weighted_sum = np.bincount(index, weights=scaled_flux * scaled_ivar, minlength=len(rest_wave))
    weight_sum = np.bincount(index, weights=scaled_ivar, minlength=len(rest_wave))
    observed = weight_sum > 0
    values[observed] = (weighted_sum[observed] / weight_sum[observed]).astype(np.float32)

    # This is inverse variance in the normalized-flux units. Preserve its
    # absolute scale so high-S/N spectra determine the learned component shapes
    # more strongly than noise-dominated spectra. Clip only extreme pixels
    # within an object (typically residual sky-line weights).
    if np.any(observed):
        cap = float(np.percentile(weight_sum[observed], 99))
        weights[observed] = np.minimum(weight_sum[observed], cap).astype(np.float32)
    return values, weights, scale


def extrapolate_spectrum_edges(
    rest_wave: np.ndarray,
    flux: np.ndarray,
    weights: np.ndarray,
    *,
    window_aa: float = 100.0,
    continuum_window_aa: float = 500.0,
    median_width: int = 5,
    method: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Extend spectrum edges with a constant level or robust broadband continuum."""
    wave = np.asarray(rest_wave, dtype=float)
    values = np.asarray(flux, dtype=float).copy()
    ivar = np.asarray(weights, dtype=float).copy()
    if wave.ndim != 1 or values.shape != wave.shape or ivar.shape != wave.shape:
        raise ValueError("Wavelength, flux, and weight arrays must be matching 1D arrays")
    if not np.all(np.isfinite(wave)) or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
        raise ValueError("Wavelengths must be finite, positive, and strictly increasing")
    if not np.isfinite(window_aa) or window_aa <= 0:
        raise ValueError("window_aa must be finite and positive")
    if not np.isfinite(continuum_window_aa) or continuum_window_aa <= 0:
        raise ValueError("continuum_window_aa must be finite and positive")
    if median_width <= 0 or median_width % 2 == 0:
        raise ValueError("median_width must be a positive odd integer")
    if method not in {"constant", "linear", "powerlaw"}:
        raise ValueError("method must be 'constant', 'linear', or 'powerlaw'")
    observed = np.flatnonzero((ivar > 0) & np.isfinite(ivar) & np.isfinite(values))
    if len(observed) < 2:
        return values, ivar

    blue = observed[0]
    red = observed[-1]
    available_width = len(observed) if len(observed) % 2 else len(observed) - 1
    filter_width = min(median_width, available_width)
    smoothed = median_filter(values[observed], size=filter_width, mode="reflect")
    edge_window = window_aa if method == "constant" else continuum_window_aa
    edge_windows = (
        wave[observed] <= wave[blue] + edge_window,
        wave[observed] >= wave[red] - edge_window,
    )
    for indices, target in zip(edge_windows, (blue, red), strict=True):
        tail = slice(None, blue) if target == blue else slice(red + 1, None)
        if not len(wave[tail]):
            continue
        edge_wave = wave[observed[indices]]
        edge_flux = smoothed[indices]
        value = float(np.mean(edge_flux))
        weight = float(np.median(ivar[observed[indices]])) * 0.1
        if method == "powerlaw":
            values[tail] = _extrapolate_powerlaw(
                edge_wave,
                values[observed[indices]],
                ivar[observed[indices]],
                wave[target],
                wave[tail],
            )
        elif method == "constant" or len(edge_wave) < 2:
            values[tail] = value
        else:
            values[tail] = _fit_edge_continuum(
                edge_wave,
                edge_flux,
                ivar[observed[indices]],
                wave[target],
                direction=1 if target == blue else -1,
                fit_window_aa=continuum_window_aa,
            )(wave[tail])
        ivar[tail] = weight
    return values, ivar


def _extrapolate_powerlaw(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    edge_wave: float,
    target_wave: np.ndarray,
) -> np.ndarray:
    """Fit A (lambda / lambda_edge)**alpha in flux space, retaining signed data."""
    if len(wave) < 3:
        raise ValueError("Power-law extrapolation requires at least three measured edge bins")
    log_ratio = np.log(wave / edge_wave)
    root_weight = np.sqrt(ivar)
    initial_amplitude = max(float(np.median(flux)), float(np.median(1 / root_weight)))

    def residual(parameters):
        prediction = np.exp(parameters[0] + parameters[1] * log_ratio)
        return (prediction - flux) * root_weight

    result = least_squares(
        residual,
        [np.log(initial_amplitude), 0.0],
        loss="soft_l1",
        max_nfev=2000,
    )
    if not result.success:
        raise ValueError(f"Power-law continuum fit failed: {result.message}")
    prediction = np.exp(result.x[0] + result.x[1] * np.log(target_wave / edge_wave))
    if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0):
        raise ValueError("Power-law extrapolation produced nonfinite or nonpositive flux")
    return prediction


def _fit_edge_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    weights: np.ndarray,
    edge_wave: float,
    *,
    direction: int,
    fit_window_aa: float = 500.0,
    bin_width_aa: float = 50.0,
):
    """Fit a clipped weighted line to median-flux bins near one spectrum edge."""
    distance = direction * (wave - edge_wave)
    inside = (distance >= 0) & (distance <= fit_window_aa)
    distance, flux, weights = distance[inside], flux[inside], weights[inside]
    bin_index = np.floor(distance / bin_width_aa).astype(int)
    centers, levels, bin_weights = [], [], []
    for index in np.unique(bin_index):
        select = bin_index == index
        centers.append(float(np.median(distance[select])))
        levels.append(float(np.median(flux[select])))
        bin_weights.append(float(np.median(weights[select])))
    centers = np.asarray(centers)
    levels = np.asarray(levels)
    bin_weights = np.asarray(bin_weights)
    if len(centers) < 3:
        slope, intercept = np.polyfit(distance, flux, 1)
        return lambda target: slope * (direction * (target - edge_wave)) + intercept

    design = np.column_stack((np.ones(len(centers)), centers / fit_window_aa))
    relative_weights = bin_weights / np.median(bin_weights)
    keep = np.ones(len(centers), dtype=bool)
    for _ in range(5):
        root_weight = np.sqrt(relative_weights[keep])
        coefficients = np.linalg.lstsq(
            design[keep] * root_weight[:, None], levels[keep] * root_weight, rcond=None
        )[0]
        standardized_residual = (levels - design @ coefficients) * np.sqrt(relative_weights)
        residual_center = float(np.median(standardized_residual[keep]))
        residual_scale = 1.4826 * float(
            np.median(np.abs(standardized_residual[keep] - residual_center))
        )
        if residual_scale == 0:
            break
        updated = np.abs(standardized_residual - residual_center) <= 3.5 * residual_scale
        if np.count_nonzero(updated) < 3 or np.array_equal(updated, keep):
            break
        keep = updated

    def continuum(target: np.ndarray) -> np.ndarray:
        target_distance = direction * (np.asarray(target) - edge_wave)
        target_design = np.column_stack(
            (np.ones(len(target_distance)), target_distance / fit_window_aa)
        )
        return target_design @ coefficients

    return continuum


def derive_rest_frame_grid(
    manifest: pd.DataFrame,
    *,
    desi_dir: Path,
    wave_step: float,
    wave_min: float | None = None,
    wave_max: float | None = None,
) -> np.ndarray:
    """Round selected coadds' valid rest-frame pixel endpoints outward."""
    if not np.isfinite(wave_step) or wave_step <= 0:
        raise ValueError("wave_step must be finite and positive")
    lower, upper = np.inf, -np.inf
    for healpix, patch in manifest.groupby("healpix", sort=True):
        coadd_path, _ = get_local_desi_paths(
            desi_dir, DEFAULT_SPECPROD, DEFAULT_SURVEY, DEFAULT_PROGRAM, int(healpix)
        )
        with fits.open(coadd_path, memmap=True) as hdul:
            row_by_target = {
                int(target): row for row, target in enumerate(hdul["FIBERMAP"].data["TARGETID"])
            }
            arm_wave = {arm: np.asarray(hdul[f"{arm}_WAVELENGTH"].data, float) for arm in "BRZ"}
            for item in patch.itertuples():
                row = row_by_target[int(item.targetid)]
                for arm in "BRZ":
                    wave = arm_wave[arm]
                    flux = hdul[f"{arm}_FLUX"].data[row]
                    ivar = hdul[f"{arm}_IVAR"].data[row]
                    mask = hdul[f"{arm}_MASK"].data[row]
                    valid = (
                        np.isfinite(wave)
                        & np.isfinite(flux)
                        & np.isfinite(ivar)
                        & (ivar > 0)
                        & (mask == 0)
                    )
                    if np.any(valid):
                        rest = wave[valid] / (1 + item.z)
                        lower = min(lower, float(rest.min()))
                        upper = max(upper, float(rest.max()))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("No valid DESI pixels available to derive wavelength grid")
    lower = np.floor(lower / wave_step) * wave_step if wave_min is None else wave_min
    upper = np.ceil(upper / wave_step) * wave_step if wave_max is None else wave_max
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= lower:
        raise ValueError("Candidate wavelength limits must be finite, positive and ordered")
    return np.arange(lower, upper + 0.5 * wave_step, wave_step)


def build_rest_frame_matrix(
    manifest: pd.DataFrame,
    *,
    desi_dir: Path,
    rest_wave: np.ndarray,
    specprod: str = DEFAULT_SPECPROD,
    survey: str = DEFAULT_SURVEY,
    program: str = DEFAULT_PROGRAM,
    min_good_pixels: int = 100,
    edge_extrapolation: str = "constant",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Read DESI coadds and return flux, relative inverse variance, and scales."""
    flux_matrix = np.zeros((len(manifest), len(rest_wave)), dtype=np.float32)
    weight_matrix = np.zeros_like(flux_matrix)
    observed_pixel_count = np.zeros(len(manifest), dtype=int)
    scales = np.full(len(manifest), np.nan, dtype=float)
    found = np.zeros(len(manifest), dtype=bool)

    for healpix, patch in manifest.groupby("healpix", sort=True):
        coadd_path, _ = get_local_desi_paths(desi_dir, specprod, survey, program, int(healpix))
        if not coadd_path.is_file():
            raise FileNotFoundError(coadd_path)
        with fits.open(coadd_path, memmap=True) as hdul:
            targetids = np.asarray(hdul["FIBERMAP"].data["TARGETID"], dtype=np.int64)
            row_by_target = {int(value): index for index, value in enumerate(targetids)}
            arm_wave = {
                arm: np.asarray(hdul[f"{arm}_WAVELENGTH"].data, dtype=float) for arm in "BRZ"
            }
            # itertuples preserves the int64 TARGETID. DataFrame.iterrows would
            # coerce this mixed numeric row to float and corrupt 18-digit IDs.
            for item in patch.itertuples():
                matrix_row = int(item.Index)
                coadd_row = row_by_target.get(int(item.targetid))
                if coadd_row is None:
                    continue
                wave = np.concatenate([arm_wave[arm] for arm in "BRZ"])
                flux = np.concatenate(
                    [np.asarray(hdul[f"{arm}_FLUX"].data[coadd_row], dtype=float) for arm in "BRZ"]
                )
                ivar = np.concatenate(
                    [np.asarray(hdul[f"{arm}_IVAR"].data[coadd_row], dtype=float) for arm in "BRZ"]
                )
                mask = np.concatenate(
                    [np.asarray(hdul[f"{arm}_MASK"].data[coadd_row]) for arm in "BRZ"]
                )
                values, weights, scale = bin_rest_frame_spectrum(
                    wave, flux, ivar, mask, float(item.z), rest_wave
                )
                observed_pixel_count[matrix_row] = np.count_nonzero(weights > 0)
                values, weights = extrapolate_spectrum_edges(
                    rest_wave, values, weights, method=edge_extrapolation
                )
                flux_matrix[matrix_row] = values
                weight_matrix[matrix_row] = weights
                scales[matrix_row] = scale
                found[matrix_row] = True
        accepted = np.isfinite(scales[patch.index]) & (
            observed_pixel_count[patch.index] >= int(min_good_pixels)
        )
        print(
            f"Loaded direct DESI spectra for HEALPix {int(healpix)}: "
            f"{int(np.sum(accepted)):,}/{len(patch):,} usable"
        )

    keep = np.isfinite(scales) & (observed_pixel_count >= int(min_good_pixels))
    if np.any(~found):
        print(f"Warning: {int(np.sum(~found)):,} manifest targets were absent from coadds")
    return (
        manifest.loc[keep].reset_index(drop=True),
        flux_matrix[keep],
        weight_matrix[keep],
        scales[keep],
    )
