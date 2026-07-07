"""
Simplex (template mixture) parameterizations for normalized weights a_k.

Two log-ratio coordinate systems on the K-template simplex:

* CLR (centered log-ratio): clr_i = log a_i - mean_j log a_j. K coordinates that
  sum to zero *exactly* (rank K-1). Used as the internal intermediate and as a
  readable legacy stored format.
* ILR (isometric log-ratio): CLR rotated onto an orthonormal basis of its own
  sum-zero hyperplane -> K-1 unconstrained, full-rank coordinates. The map has
  |det| = 1, so differential entropy is well-posed and basis-independent. This is
  the stored prior format.

    ilr = clr @ V,   clr = ilr @ V.T,   a = softmax(clr),   V = ilr_basis(K).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

PARAMETERIZATION_CLR = "clr"
PARAMETERIZATION_ILR = "ilr"
DEFAULT_CLR_EPS = 1e-5


@lru_cache(maxsize=None)
def _ilr_basis_cached(n_templates: int) -> tuple:
    if n_templates < 2:
        raise ValueError("n_templates must be >= 2 for ILR parameterization")
    m = np.eye(n_templates) - np.ones((n_templates, n_templates)) / n_templates
    # m projects onto the sum-zero hyperplane: eigenvalue 0 for the all-ones
    # direction (first, ascending), 1 for the remaining K-1 orthonormal axes.
    _w, vecs = np.linalg.eigh(m)
    V = vecs[:, 1:]  # (K, K-1)
    return tuple(map(tuple, V))


def ilr_basis(n_templates: int) -> np.ndarray:
    """Fixed orthonormal (K, K-1) basis of the CLR sum-zero hyperplane."""
    return np.asarray(_ilr_basis_cached(int(n_templates)), dtype=float)


def prior_clr_feature_names(n_templates: int) -> list[str]:
    """Feature names for CLR rows: f1..fK, log_c_scale, z."""
    if n_templates < 2:
        raise ValueError("n_templates must be >= 2 for CLR simplex parameterization")
    return [f"f{k + 1}" for k in range(n_templates)] + ["log_c_scale", "z"]


def prior_ilr_feature_names(n_templates: int) -> list[str]:
    """Feature names for ILR rows: f1..f_{K-1}, log_c_scale, z."""
    if n_templates < 2:
        raise ValueError("n_templates must be >= 2 for ILR simplex parameterization")
    return [f"f{k + 1}" for k in range(n_templates - 1)] + ["log_c_scale", "z"]


def prior_feature_names(n_templates: int, *, parameterization: str = PARAMETERIZATION_ILR) -> list[str]:
    if parameterization == PARAMETERIZATION_ILR:
        return prior_ilr_feature_names(n_templates)
    if parameterization == PARAMETERIZATION_CLR:
        return prior_clr_feature_names(n_templates)
    raise ValueError(f"Unknown parameterization {parameterization!r}")


def weights_to_clr(w: np.ndarray, eps: float = DEFAULT_CLR_EPS) -> np.ndarray:
    """Map simplex weights (..., K) to centered log-ratios clr_i = log a_i - mean_j log a_j."""
    w = np.asarray(w, dtype=float)
    w = np.clip(w, eps, None)
    s = w.sum(axis=-1, keepdims=True)
    w = w / np.where(s > 0, s, 1.0)
    logw = np.log(w)
    return logw - logw.mean(axis=-1, keepdims=True)


def clr_to_weights(clr: np.ndarray) -> np.ndarray:
    """Map centered log-ratio coordinates (..., K) back to simplex weights."""
    clr = np.asarray(clr, dtype=float)
    x = clr - clr.max(axis=-1, keepdims=True)
    expx = np.exp(x)
    return expx / expx.sum(axis=-1, keepdims=True)


def weights_to_ilr(w: np.ndarray, eps: float = DEFAULT_CLR_EPS) -> np.ndarray:
    """Map simplex weights (..., K) to ILR coordinates (..., K-1)."""
    clr = weights_to_clr(w, eps=eps)
    return clr @ ilr_basis(clr.shape[-1])


def ilr_to_weights(ilr: np.ndarray) -> np.ndarray:
    """Map ILR coordinates (..., K-1) back to simplex weights (..., K)."""
    ilr = np.asarray(ilr, dtype=float)
    V = ilr_basis(ilr.shape[-1] + 1)
    return clr_to_weights(ilr @ V.T)


def ilr_to_weights_torch(ilr: torch.Tensor) -> torch.Tensor:
    """Torch counterpart of ilr_to_weights for the runtime sampling path."""
    V = torch.as_tensor(ilr_basis(ilr.shape[-1] + 1), device=ilr.device, dtype=ilr.dtype)
    clr = ilr @ V.T
    clr = clr - clr.amax(dim=-1, keepdim=True)
    exp = torch.exp(clr)
    return exp / exp.sum(dim=-1, keepdim=True)


def feature_matrix_from_weights(
    a: np.ndarray,
    log_s: np.ndarray,
    z: np.ndarray,
    *,
    parameterization: str = PARAMETERIZATION_ILR,
) -> np.ndarray:
    """Stack (a, log_s, z) columns into KDE feature rows."""
    a = np.asarray(a, dtype=float)
    log_s = np.asarray(log_s, dtype=float).reshape(-1, 1)
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    if parameterization == PARAMETERIZATION_ILR:
        return np.hstack([weights_to_ilr(a), log_s, z])
    if parameterization == PARAMETERIZATION_CLR:
        return np.hstack([weights_to_clr(a), log_s, z])
    raise ValueError(f"Unknown parameterization {parameterization!r}")


def split_feature_matrix(
    x: np.ndarray,
    n_templates: int,
    *,
    parameterization: str = PARAMETERIZATION_ILR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a, log_s, z) from a feature matrix row block."""
    x = np.asarray(x, dtype=float)
    if parameterization == PARAMETERIZATION_ILR:
        ilr = x[:, : n_templates - 1]
        a = ilr_to_weights(ilr)
        log_s = x[:, n_templates - 1]
        z = x[:, n_templates]
        return a, log_s, z
    if parameterization == PARAMETERIZATION_CLR:
        clr = x[:, :n_templates]
        a = clr_to_weights(clr)
        log_s = x[:, n_templates]
        z = x[:, n_templates + 1]
        return a, log_s, z
    raise ValueError(f"Unknown parameterization {parameterization!r}")
