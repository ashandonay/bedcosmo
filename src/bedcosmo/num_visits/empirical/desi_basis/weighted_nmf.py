"""Alternating nonnegative least squares with missing, weighted observations."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from sklearn.decomposition import NMF


def infer_coefficients(
    flux: np.ndarray, weights: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    """Infer nonnegative amplitudes for fixed component spectra."""
    coefficients = np.zeros((len(flux), len(basis)), dtype=float)
    for row in range(len(flux)):
        observed = weights[row] > 0
        if np.count_nonzero(observed) < len(basis):
            continue
        sqrt_weight = np.sqrt(weights[row, observed])
        design = basis[:, observed].T * sqrt_weight[:, None]
        response = flux[row, observed] * sqrt_weight
        coefficients[row], _ = nnls(design, response)
    return coefficients


def _initialize_basis(
    flux: np.ndarray, weights: np.ndarray, rank: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    observed = weights > 0
    column_weight = weights.sum(axis=0)
    column_mean = np.divide(
        (weights * flux).sum(axis=0),
        column_weight,
        out=np.zeros(flux.shape[1], dtype=float),
        where=column_weight > 0,
    )
    filled = np.where(observed, flux, column_mean[None, :])
    filled = np.clip(filled, 0.0, None)
    model = NMF(
        n_components=rank,
        init="nndsvda",
        solver="cd",
        max_iter=400,
        random_state=seed,
    )
    coefficients = model.fit_transform(filled)
    return coefficients, model.components_


def fit_weighted_nmf(
    flux: np.ndarray,
    weights: np.ndarray,
    rank: int,
    *,
    iterations: int = 8,
    smooth_sigma_pixels: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a nonnegative basis while ignoring missing and masked pixels."""
    flux = np.asarray(flux, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if flux.shape != weights.shape:
        raise ValueError("flux and weights must have the same shape")
    if rank < 1 or rank > min(flux.shape):
        raise ValueError("rank is incompatible with the input matrix")

    coefficients, basis = _initialize_basis(flux, weights, rank, seed)
    losses: list[float] = []
    for _ in range(iterations):
        coefficients = infer_coefficients(flux, weights, basis)
        for column in range(flux.shape[1]):
            observed = weights[:, column] > 0
            if np.count_nonzero(observed) < rank:
                basis[:, column] = 0.0
                continue
            sqrt_weight = np.sqrt(weights[observed, column])
            design = coefficients[observed] * sqrt_weight[:, None]
            response = flux[observed, column] * sqrt_weight
            basis[:, column], _ = nnls(design, response)
        if smooth_sigma_pixels > 0:
            basis = gaussian_filter1d(
                basis, smooth_sigma_pixels, axis=1, mode="nearest"
            )
        norms = np.trapz(basis, axis=1)
        valid = norms > 0
        basis[valid] /= norms[valid, None]
        coefficients[:, valid] *= norms[valid]
        residual = flux - coefficients @ basis
        losses.append(float(np.sum(weights * residual**2) / np.sum(weights)))
    return coefficients, basis, np.asarray(losses)


def weighted_reconstruction_error(
    flux: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Per-object weighted RMS in the per-spectrum normalized flux units."""
    squared = weights * (flux - coefficients @ basis) ** 2
    return np.sqrt(
        np.divide(
            squared.sum(axis=1),
            weights.sum(axis=1),
            out=np.full(len(flux), np.nan),
            where=weights.sum(axis=1) > 0,
        )
    )
