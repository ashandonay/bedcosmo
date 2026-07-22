"""2D toy helpers mirroring the production KDE prior entropy path."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
    fit_sed_prior_kde,
    pack_kde_artifact,
)
from bedcosmo.num_visits.empirical.sed_prior import score_kde_artifact

LOG_2PI = math.log(2.0 * math.pi)


def bits(x_nats: float) -> float:
    """Convert differential entropy from nats to bits."""
    return float(x_nats / math.log(2.0))


def mvn_analytic_entropy(cov: np.ndarray, *, nats: bool = True) -> float:
    """Closed-form differential entropy of a multivariate Gaussian."""
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("cov must be positive definite")
    h_nats = 0.5 * (d * (1.0 + LOG_2PI) + logdet)
    return float(h_nats if nats else bits(h_nats))


def standard_normal_log_prob(z: np.ndarray) -> np.ndarray:
    """log N(0, I)(z) for rows of shape (n, d)."""
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    d = arr.shape[1]
    return -0.5 * (np.sum(arr * arr, axis=1) + d * LOG_2PI)


def fit_prod_kde(
    x_train: np.ndarray,
    *,
    bandwidth: float | str = 0.3,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Fit a production-style KDE artifact on ``x_train``."""
    x_train = np.asarray(x_train, dtype=np.float64)
    if x_train.ndim != 2 or x_train.shape[1] != 2:
        raise ValueError("fit_prod_kde expects shape (n, 2)")
    names = feature_names or ["x0", "x1"]
    kde, scaler = fit_sed_prior_kde(x_train, bandwidth=bandwidth)
    # Generic 2D KDE used only for scoring; the simplex parameterization is a
    # formality (features are never decoded to weights). n_templates is passed
    # in metadata so pack_kde_artifact does not derive it from feature_names.
    return pack_kde_artifact(
        kde,
        scaler,
        names,
        x_train,
        metadata={"n_train": int(x_train.shape[0]), "toy": True, "n_templates": len(names)},
        parameterization="ilr",
    )


def kde_plugin_entropy(artifact: dict[str, Any], x_eval: np.ndarray) -> float:
    """Plug-in entropy H ≈ -E[log p_KDE(x)] in nats (prod convention)."""
    x_eval = np.asarray(x_eval, dtype=np.float64)
    log_p = score_kde_artifact(artifact, x_eval)
    return float(-np.mean(log_p))


def cov_entropy_from_base(
    z: np.ndarray,
    log_p_z: np.ndarray,
    log_abs_det_j: np.ndarray,
) -> tuple[float, float, float]:
    """Monte Carlo entropy via change-of-variables from a known base density.

    Returns:
        h_cov: -mean(log p_z - log|det J|)
        h_identity: H(z) + mean(log|det J|) [should match h_cov]
        h_base: -mean(log p_z)
    """
    log_p_y = np.asarray(log_p_z, dtype=np.float64) - np.asarray(log_abs_det_j, dtype=np.float64)
    h_cov = float(-np.mean(log_p_y))
    h_base = float(-np.mean(log_p_z))
    h_identity = h_base + float(np.mean(log_abs_det_j))
    return h_cov, h_identity, h_base


def naive_standard_normal_entropy(y: np.ndarray) -> float:
    """H ≈ -E[log N(0,I)(y)] — the transform_input=True shortcut."""
    return float(-np.mean(standard_normal_log_prob(y)))


def sample_mvn(
    n: int,
    cov: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``n`` rows from N(0, cov)."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(2), cov, size=n)


def sample_standard_normal(n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n, 2))


def linear_transform(z: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return z @ np.asarray(matrix, dtype=np.float64).T


def linear_log_abs_det(matrix: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(np.asarray(matrix, dtype=np.float64))
    if sign <= 0:
        raise ValueError("linear map must be invertible")
    return float(logdet)


def transform_exp_z2(z: np.ndarray) -> np.ndarray:
    """y1 = z1, y2 = exp(z2)."""
    z = np.asarray(z, dtype=np.float64)
    return np.column_stack([z[:, 0], np.exp(z[:, 1])])


def log_abs_det_exp_z2(z: np.ndarray) -> np.ndarray:
    return np.asarray(z[:, 1], dtype=np.float64)


def transform_quadratic_z1(z: np.ndarray) -> np.ndarray:
    """y1 = z1, y2 = z1^2 + z2 (det = 1, non-Gaussian conditional structure)."""
    z = np.asarray(z, dtype=np.float64)
    return np.column_stack([z[:, 0], z[:, 0] ** 2 + z[:, 1]])


def log_abs_det_quadratic_z1(z: np.ndarray) -> np.ndarray:
    return np.zeros(z.shape[0], dtype=np.float64)


def apply_transform(
    z: np.ndarray,
    forward: Callable[[np.ndarray], np.ndarray],
    log_abs_det_fn: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    y = forward(z)
    return y, log_abs_det_fn(z)
