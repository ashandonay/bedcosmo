"""
Simplex (template mixture) parameterization for normalized weights a_k.

Uses K-1 log-ratios relative to template K (softmax with zero reference):
    eta_k = log(a_k / a_K),  k = 1..K-1
    a = softmax([eta_1, ..., eta_{K-1}, 0])

This enforces a_k >= 0 and sum_k a_k = 1 by construction.
"""

from __future__ import annotations

import numpy as np
import torch

PARAMETERIZATION_WEIGHTS = "weights"
PARAMETERIZATION_LOGITS = "logits"


def prior_logit_feature_names(n_templates: int) -> list[str]:
    """Feature names for KDE / pool rows: f1..f_{K-1}, log_c_scale, z."""
    if n_templates < 2:
        raise ValueError("n_templates must be >= 2 for logit simplex parameterization")
    return [f"f{k + 1}" for k in range(n_templates - 1)] + ["log_c_scale", "z"]


def prior_weights_feature_names(n_templates: int) -> list[str]:
    """Legacy feature names: a1..aK, log_c_scale, z."""
    return [f"a{k + 1}" for k in range(n_templates)] + ["log_c_scale", "z"]


def prior_feature_names(n_templates: int, *, parameterization: str = PARAMETERIZATION_LOGITS) -> list[str]:
    if parameterization == PARAMETERIZATION_LOGITS:
        return prior_logit_feature_names(n_templates)
    if parameterization == PARAMETERIZATION_WEIGHTS:
        return prior_weights_feature_names(n_templates)
    raise ValueError(f"Unknown parameterization {parameterization!r}")


def weights_to_logits(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Map simplex weights (..., K) to log-ratios (..., K-1) vs template K.
    """
    w = np.asarray(w, dtype=float)
    w = np.clip(w, eps, None)
    s = w.sum(axis=-1, keepdims=True)
    w = w / np.where(s > 0, s, 1.0)
    ref = w[..., -1:]
    return np.log(w[..., :-1] / ref)


def logits_to_weights(eta: np.ndarray) -> np.ndarray:
    """
    Map log-ratios (..., K-1) to simplex weights (..., K) via stable softmax.
    """
    eta = np.asarray(eta, dtype=float)
    z = np.concatenate([eta, np.zeros((*eta.shape[:-1], 1), dtype=eta.dtype)], axis=-1)
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def weights_to_logits_torch(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    w = torch.clamp(w, min=eps)
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=eps)
    ref = w[..., -1:].clamp(min=eps)
    return torch.log(w[..., :-1] / ref)


def logits_to_weights_torch(eta: torch.Tensor) -> torch.Tensor:
    z = torch.cat([eta, torch.zeros(*eta.shape[:-1], 1, device=eta.device, dtype=eta.dtype)], dim=-1)
    z = z - z.amax(dim=-1, keepdim=True)
    exp = torch.exp(z)
    return exp / exp.sum(dim=-1, keepdim=True)


def feature_matrix_from_weights(
    a: np.ndarray,
    log_s: np.ndarray,
    z: np.ndarray,
    *,
    parameterization: str = PARAMETERIZATION_LOGITS,
) -> np.ndarray:
    """Stack (a, log_s, z) columns into KDE feature rows."""
    a = np.asarray(a, dtype=float)
    log_s = np.asarray(log_s, dtype=float).reshape(-1, 1)
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    if parameterization == PARAMETERIZATION_LOGITS:
        eta = weights_to_logits(a)
        return np.hstack([eta, log_s, z])
    return np.hstack([a, log_s, z])


def split_feature_matrix(
    x: np.ndarray,
    n_templates: int,
    *,
    parameterization: str = PARAMETERIZATION_LOGITS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a, log_s, z) from a feature matrix row block."""
    x = np.asarray(x, dtype=float)
    if parameterization == PARAMETERIZATION_LOGITS:
        eta = x[:, : n_templates - 1]
        a = logits_to_weights(eta)
        log_s = x[:, n_templates - 1]
        z = x[:, n_templates]
        return a, log_s, z
    a = x[:, :n_templates]
    log_s = x[:, n_templates]
    z = x[:, n_templates + 1]
    return a, log_s, z
