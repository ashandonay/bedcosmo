"""Construct a masked rest-frame training matrix from DESI coadds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from ..desi_data import get_local_desi_paths
from ..paths import DEFAULT_PROGRAM, DEFAULT_SPECPROD, DEFAULT_SURVEY


def load_desi_manifest(path: Path | str) -> pd.DataFrame:
    """Load the DESI object identity/redshift columns used for basis training."""
    table = pd.read_csv(path)
    required = {"targetid", "healpix", "z"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if {"success", "quality_pass"}.issubset(table.columns):
        table = table.loc[
            table["success"].astype(bool) & table["quality_pass"].astype(bool)
        ]
    table = table.loc[
        np.isfinite(table["z"].to_numpy(float)) & (table["z"].to_numpy(float) >= 0)
    ]
    return table[["targetid", "healpix", "z"]].drop_duplicates("targetid").reset_index(
        drop=True
    )


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
    weighted_sum = np.bincount(
        index, weights=scaled_flux * scaled_ivar, minlength=len(rest_wave)
    )
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


def build_rest_frame_matrix(
    manifest: pd.DataFrame,
    *,
    desi_dir: Path,
    rest_wave: np.ndarray,
    specprod: str = DEFAULT_SPECPROD,
    survey: str = DEFAULT_SURVEY,
    program: str = DEFAULT_PROGRAM,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Read DESI coadds and return flux, relative inverse variance, and scales."""
    flux_matrix = np.zeros((len(manifest), len(rest_wave)), dtype=np.float32)
    weight_matrix = np.zeros_like(flux_matrix)
    scales = np.full(len(manifest), np.nan, dtype=float)
    found = np.zeros(len(manifest), dtype=bool)

    for healpix, patch in manifest.groupby("healpix", sort=True):
        coadd_path, _ = get_local_desi_paths(
            desi_dir, specprod, survey, program, int(healpix)
        )
        if not coadd_path.is_file():
            raise FileNotFoundError(coadd_path)
        with fits.open(coadd_path, memmap=True) as hdul:
            targetids = np.asarray(hdul["FIBERMAP"].data["TARGETID"], dtype=np.int64)
            row_by_target = {int(value): index for index, value in enumerate(targetids)}
            arm_wave = {
                arm: np.asarray(hdul[f"{arm}_WAVELENGTH"].data, dtype=float)
                for arm in "BRZ"
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
                flux_matrix[matrix_row] = values
                weight_matrix[matrix_row] = weights
                scales[matrix_row] = scale
                found[matrix_row] = True
        accepted = np.isfinite(scales[patch.index]) & (
            np.sum(weight_matrix[patch.index] > 0, axis=1) >= 100
        )
        print(
            f"Loaded direct DESI spectra for HEALPix {int(healpix)}: "
            f"{int(np.sum(accepted)):,}/{len(patch):,} usable"
        )

    keep = np.isfinite(scales) & (np.sum(weight_matrix > 0, axis=1) >= 100)
    if np.any(~found):
        print(f"Warning: {int(np.sum(~found)):,} manifest targets were absent from coadds")
    return (
        manifest.loc[keep].reset_index(drop=True),
        flux_matrix[keep],
        weight_matrix[keep],
        scales[keep],
    )
