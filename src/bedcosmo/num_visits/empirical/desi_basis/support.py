"""Wavelength-support diagnostics for a direct-DESI spectral basis."""

from __future__ import annotations

import numpy as np
from speclite import filters as speclite_filters


def largest_contiguous_region(mask: np.ndarray) -> np.ndarray:
    """Keep the longest contiguous True region of a one-dimensional mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("wavelength-support mask must be one-dimensional")
    selected = np.zeros_like(mask)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return selected
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, breaks)
    longest = max(runs, key=len)
    selected[longest] = True
    return selected


def select_wavelength_support(
    weights: np.ndarray,
    ranks: list[int] | tuple[int, ...] | np.ndarray,
    *,
    observations_per_component: int = 10,
    minimum_contributors: int | None = None,
    support_rank: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select a common, contiguous wavelength interval with enough constraints.

    Each wavelength column contains ``max(ranks)`` unknown basis values.  The
    default is sized for a rank-10 basis and requires ten observed spectra per
    unknown at every retained wavelength.  This is independent of the total
    catalog size, unlike a global population-fraction cutoff, and keeps the
    support fixed while comparing or incrementally adding ranks up to ten.
    """
    weights = np.asarray(weights)
    if weights.ndim != 2:
        raise ValueError("weights must have shape (n_spectra, n_wavelengths)")
    ranks = np.asarray(ranks, dtype=int)
    if ranks.size == 0 or np.any(ranks < 1):
        raise ValueError("at least one positive basis rank is required")
    if observations_per_component < 1:
        raise ValueError("observations_per_component must be positive")
    if support_rank < 1:
        raise ValueError("support_rank must be positive")
    effective_rank = max(int(np.max(ranks)), int(support_rank))
    required = (
        int(minimum_contributors)
        if minimum_contributors is not None
        else int(observations_per_component * effective_rank)
    )
    if required < effective_rank:
        raise ValueError("minimum contributors cannot be smaller than the largest rank")
    contributors = np.sum(np.isfinite(weights) & (weights > 0), axis=0)
    selected = largest_contiguous_region(contributors >= required)
    return selected, contributors, required


def lsst_demand_weighted_coverage(
    wave_rest_aa: np.ndarray,
    redshift: np.ndarray,
    weights: np.ndarray,
    *,
    bands: str = "ugrizy",
    chunk_size: int = 512,
) -> np.ndarray:
    """Fraction of LSST demand at each rest wavelength observed by DESI.

    LSST transmission is evaluated at ``wave_rest * (1 + z)`` separately for
    every spectrum.  The numerator includes only spectra with valid DESI
    pixels; the denominator includes all spectra for which an LSST band needs
    that rest wavelength.  Each band's response is normalized by its peak so
    no single band dominates solely because of throughput normalization.
    """
    wave_rest_aa = np.asarray(wave_rest_aa, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    weights = np.asarray(weights)
    if weights.shape != (len(redshift), len(wave_rest_aa)):
        raise ValueError("weights shape must match redshift and wavelength arrays")
    loaded = [speclite_filters.load_filter(f"lsst2023-{band}") for band in bands]
    observed = np.isfinite(weights) & (weights > 0)
    numerator = np.zeros(len(wave_rest_aa), dtype=float)
    denominator = np.zeros(len(wave_rest_aa), dtype=float)
    for start in range(0, len(redshift), chunk_size):
        stop = min(start + chunk_size, len(redshift))
        wave_observed = wave_rest_aa[None, :] * (1.0 + redshift[start:stop, None])
        demand = np.zeros_like(wave_observed)
        for loaded_filter in loaded:
            response = np.asarray(loaded_filter(wave_observed), dtype=float)
            demand += response / float(np.max(loaded_filter.response))
        denominator += np.sum(demand, axis=0)
        numerator += np.sum(demand * observed[start:stop], axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
