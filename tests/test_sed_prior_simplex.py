"""Tests for simplex (logit) template-weight parameterization."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bedcosmo.num_visits.sed_prior.simplex import (
    logits_to_weights,
    logits_to_weights_torch,
    prior_logit_feature_names,
    split_feature_matrix,
    weights_to_logits,
    weights_to_logits_torch,
)


def test_prior_logit_feature_names():
    assert prior_logit_feature_names(12) == [f"f{k}" for k in range(1, 12)] + [
        "log_c_scale",
        "z",
    ]


def test_weights_logits_roundtrip():
    w = np.array([[0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    eta = weights_to_logits(w)
    w2 = logits_to_weights(eta)
    assert np.allclose(w, w2, atol=1e-10)
    assert np.allclose(w2.sum(axis=1), 1.0)


def test_uniform_at_zero_logits():
    eta = np.zeros((5, 11))
    w = logits_to_weights(eta)
    assert np.allclose(w, 1.0 / 12.0, atol=1e-8)


def test_split_feature_matrix_logits():
    eta = np.zeros((2, 11))
    log_s = np.array([7.0, 7.5])
    z = np.array([0.5, 1.0])
    x = np.column_stack([eta, log_s, z])
    a, ls, zz = split_feature_matrix(x, 12, parameterization="logits")
    assert a.shape == (2, 12)
    assert np.allclose(a.sum(axis=1), 1.0)


def test_torch_matches_numpy():
    w = torch.tensor([[0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7]])
    eta = weights_to_logits_torch(w)
    w_t = logits_to_weights_torch(eta)
    assert torch.allclose(w, w_t, atol=1e-10)
