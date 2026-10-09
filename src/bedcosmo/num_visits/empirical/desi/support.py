"""Wavelength-support diagnostics for a direct DESI spectral basis."""

from __future__ import annotations

import numpy as np
from speclite import filters as speclite_filters


def lsst_support_limits(rest_min, rest_max):
    """Full tabulated ugrizy coverage requires both LSST edges inside support."""
    if not 0 < rest_min < rest_max or not np.isfinite([rest_min, rest_max]).all():
        raise ValueError("Rest wavelength endpoints must be finite, positive and ordered")
    filters = speclite_filters.load_filters("lsst2023-*")
    blue = min(f.wavelength.min() for f in filters)
    red = max(f.wavelength.max() for f in filters)
    return blue, red, max(0.0, red / rest_max - 1.0), blue / rest_min - 1.0


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


def lsst_required_mask(wave: np.ndarray, redshift: np.ndarray) -> np.ndarray:
    """Common LSST demand with grid centers bracketing its exact endpoints."""
    blue, red, _, _ = lsst_support_limits(wave[0], wave[-1])
    lower = blue / (1 + np.max(redshift))
    upper = red / (1 + np.min(redshift))
    if lower < wave[0] or upper > wave[-1]:
        raise ValueError("Training matrix does not span LSST for every selected redshift")
    first = max(0, np.searchsorted(wave, lower, side="right") - 1)
    last = min(len(wave) - 1, np.searchsorted(wave, upper, side="left"))
    selected = np.zeros(len(wave), dtype=bool)
    selected[first : last + 1] = True
    return selected


def select_training_redshift_buffer(
    wave, weights, redshift, prior_bounds, *, train_fraction, split_seed, minimum_contributors
):
    """Grow low/high buffers independently to cover deficient training wavelengths."""
    lower, upper = prior_bounds
    required = lsst_required_mask(wave, np.asarray(prior_bounds))
    inside = (redshift >= lower) & (redshift <= upper)
    if not np.any(inside):
        raise ValueError("No usable galaxies lie in the requested prior redshift interval")
    low_rows = np.flatnonzero(redshift < lower)
    high_rows = np.flatnonzero(redshift > upper)
    sides = [
        low_rows[np.argsort(-redshift[low_rows], kind="stable")],
        high_rows[np.argsort(redshift[high_rows], kind="stable")],
    ]
    added = [0, 0]
    valid = np.isfinite(weights[:, required]) & (weights[:, required] > 0)

    def coverage():
        selected = inside.copy()
        for rows, count in zip(sides, added):
            selected[rows[:count]] = True
        rows = np.flatnonzero(selected)
        train = np.random.default_rng(split_seed).permutation(len(rows))[
            : int(train_fraction * len(rows))
        ]
        counts = valid[rows[train]].sum(axis=0)
        return selected, counts < minimum_contributors

    selected, missing = coverage()
    while np.any(missing):
        available = [i for i in range(2) if added[i] < len(sides[i])]
        if not available:
            break
        # Grow only the side whose next nearby galaxies best cover deficient columns.
        scores = [valid[sides[i][added[i] : added[i] + 25]][:, missing].sum() for i in available]
        if not any(scores):
            scores = [valid[sides[i][added[i] :]][:, missing].sum() for i in available]
        side = available[int(np.argmax(scores))]
        previous = added[side]
        added[side] = min(previous + 25, len(sides[side]))
        selected, missing = coverage()
        if not np.any(missing):
            # Refine the last batch against the exact saved-row split.
            for count in range(previous + 1, added[side] + 1):
                added[side] = count
                selected, missing = coverage()
                if not np.any(missing):
                    return selected
    if not np.any(missing):
        return selected
    raise ValueError(
        f"Available galaxies cannot supply {minimum_contributors} training contributors "
        f"throughout LSST coverage for prior z={lower:g}–{upper:g}"
    )


def plan_edge_extensions(wave, weights, prior_bounds, train, minimum_contributors):
    """Extend nearest exterior tails until the fixed training split covers LSST.

    Rank all galaxies by extra rest-wavelength distance, using row order to
    break ties. Held-out rows follow the same geometric ordering; their fluxes
    are never used. Interior masked gaps remain missing.
    """
    valid = np.isfinite(weights) & (weights > 0)
    if np.any(~valid.any(axis=1)):
        raise ValueError("Cannot extend a spectrum without a supported endpoint")
    first = valid.argmax(axis=1)
    last = len(wave) - 1 - valid[:, ::-1].argmax(axis=1)
    original = np.column_stack((wave[first], wave[last]))
    bounds = original.copy()
    training = np.zeros(len(weights), dtype=bool)
    training[train] = True
    if len(train) < minimum_contributors:
        raise ValueError("Too few training galaxies for the requested contributor threshold")
    required = np.flatnonzero(lsst_required_mask(wave, np.asarray(prior_bounds)))
    counts = valid[train].sum(axis=0)
    # Work inward from both required endpoints; updates include the entire tail.
    for column in list(required) + list(required[::-1]):
        if counts[column] >= minimum_contributors:
            continue
        candidates = np.flatnonzero((first > column) | (last < column))
        distance = np.maximum(first[candidates] - column, column - last[candidates])
        for row in candidates[np.argsort(distance, kind="stable")]:
            if first[row] > column:
                added = slice(column, first[row])
                first[row] = column
            else:
                added = slice(last[row] + 1, column + 1)
                last[row] = column
            if training[row]:
                counts[added] += 1
            if counts[column] >= minimum_contributors:
                break
        if counts[column] < minimum_contributors:
            raise ValueError(
                f"Cannot support wavelength {wave[column]:g} without filling interior gaps"
            )
    bounds[:, 0], bounds[:, 1] = wave[first], wave[last]
    return bounds, original


def select_wavelength_support(
    weights: np.ndarray,
    ranks: list[int] | tuple[int, ...] | np.ndarray,
    *,
    observations_per_component: int = 10,
    minimum_contributors: int | None = None,
    support_rank: int = 10,
    required_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select a sufficiently populated interval containing all required LSST columns.

    Each wavelength column contains ``max(ranks)`` unknown basis values.  The
    default is sized for a rank-10 basis and requires ten observed spectra per
    unknown at every retained wavelength.  This is independent of the total
    catalog size, unlike a global population-fraction cutoff, and keeps the
    support fixed while comparing or incrementally adding ranks up to ten.
    Required LSST columns must meet the same threshold; insufficient coverage
    raises an error rather than silently narrowing the selected redshift range.
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
    if required_mask is not None:
        if len(weights) < required:
            raise ValueError(f"Need at least {required} training spectra to retain LSST support")
        required_mask = np.asarray(required_mask, dtype=bool)
        if required_mask.shape != contributors.shape:
            raise ValueError("Required wavelength mask must match the matrix columns")
        insufficient = required_mask & (contributors < required)
        if np.any(insufficient):
            raise ValueError(
                f"{np.count_nonzero(insufficient)} LSST-required wavelength bins have fewer "
                f"than {required} training contributors (minimum "
                f"{contributors[required_mask].min()}); use a larger training sample "
                "or revise the selected redshift range"
            )
        selected |= required_mask
        if np.any(selected):
            first, last = np.flatnonzero(selected)[[0, -1]]
            selected[first : last + 1] = True
        if np.any(selected & (contributors < required)):
            raise ValueError("Required LSST support is separated by underpopulated columns")
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
