"""Tests for simplex template-weight parameterizations (logits and CLR)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bedcosmo.num_visits.empirical.simplex import (
    PARAMETERIZATION_CLR,
    PARAMETERIZATION_LOGITS,
    clr_to_weights,
    logits_to_weights,
    logits_to_weights_torch,
    prior_clr_feature_names,
    prior_logit_feature_names,
    split_feature_matrix,
    weights_to_clr,
    weights_to_logits,
    weights_to_logits_torch,
)

N_TEMPLATES = 12


def test_prior_logit_feature_names():
    assert prior_logit_feature_names(N_TEMPLATES) == [f"f{k}" for k in range(1, N_TEMPLATES)] + [
        "log_c_scale",
        "z",
    ]


def test_prior_clr_feature_names():
    assert prior_clr_feature_names(N_TEMPLATES) == [
        f"f{k}" for k in range(1, N_TEMPLATES + 1)
    ] + ["log_c_scale", "z"]


def test_weights_logits_roundtrip_dense():
    w = np.full((1, N_TEMPLATES), 1.0 / N_TEMPLATES)
    eta = weights_to_logits(w)
    w2 = logits_to_weights(eta)
    assert np.allclose(w, w2, atol=1e-10)
    assert np.allclose(w2.sum(axis=1), 1.0)


def test_weights_logits_sparse_smoothed():
    """Exact zeros are floored before log-ratios (matches KDE training)."""
    w = np.array([[0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    w2 = logits_to_weights(weights_to_logits(w))
    assert np.all(w2 > 0)
    assert np.allclose(w2.sum(axis=1), 1.0)


def test_weights_clr_roundtrip_dense():
    w = np.array([[0.1, 0.05, 0.05, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025]])
    w = w / w.sum(axis=1, keepdims=True)
    clr = weights_to_clr(w)
    assert np.allclose(clr.sum(axis=1), 0.0, atol=1e-12)
    w2 = clr_to_weights(clr)
    assert np.allclose(w, w2, atol=1e-10)
    assert np.allclose(w2.sum(axis=1), 1.0)


def test_weights_clr_sparse_smoothed():
    w = np.array([[0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    w2 = clr_to_weights(weights_to_clr(w))
    assert np.all(w2 > 0)
    assert np.allclose(w2.sum(axis=1), 1.0)
    assert w2[0, 3] > w2[0, 0]


def test_uniform_at_zero_logits():
    eta = np.zeros((5, N_TEMPLATES - 1))
    w = logits_to_weights(eta)
    assert np.allclose(w, 1.0 / N_TEMPLATES, atol=1e-8)


def test_uniform_at_zero_clr():
    clr = np.zeros((5, N_TEMPLATES))
    w = clr_to_weights(clr)
    assert np.allclose(w, 1.0 / N_TEMPLATES, atol=1e-8)


def test_split_feature_matrix_logits():
    eta = np.zeros((2, N_TEMPLATES - 1))
    log_s = np.array([7.0, 7.5])
    z = np.array([0.5, 1.0])
    x = np.column_stack([eta, log_s, z])
    a, ls, zz = split_feature_matrix(x, N_TEMPLATES, parameterization=PARAMETERIZATION_LOGITS)
    assert a.shape == (2, N_TEMPLATES)
    assert np.allclose(a.sum(axis=1), 1.0)


def test_split_feature_matrix_clr():
    clr = np.zeros((2, N_TEMPLATES))
    log_s = np.array([7.0, 7.5])
    z = np.array([0.5, 1.0])
    x = np.column_stack([clr, log_s, z])
    a, ls, zz = split_feature_matrix(x, N_TEMPLATES, parameterization=PARAMETERIZATION_CLR)
    assert a.shape == (2, N_TEMPLATES)
    assert np.allclose(a.sum(axis=1), 1.0)


def test_torch_matches_numpy():
    w = torch.tensor([[0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7]])
    eta = weights_to_logits_torch(w)
    w_t = logits_to_weights_torch(eta)
    assert torch.allclose(w, w_t, atol=1e-10)
