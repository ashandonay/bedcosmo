"""Tests for simplex template-weight parameterizations (ILR and legacy CLR)."""

from __future__ import annotations

import numpy as np
import torch

from bedcosmo.num_visits.empirical.simplex import (
    PARAMETERIZATION_CLR,
    PARAMETERIZATION_ILR,
    clr_to_weights,
    ilr_basis,
    ilr_to_weights,
    ilr_to_weights_torch,
    prior_clr_feature_names,
    prior_ilr_feature_names,
    split_feature_matrix,
    weights_to_clr,
    weights_to_ilr,
)

N_TEMPLATES = 12


def test_prior_ilr_feature_names():
    assert prior_ilr_feature_names(N_TEMPLATES) == [f"f{k}" for k in range(1, N_TEMPLATES)] + [
        "log_c_scale",
        "z",
    ]


def test_prior_clr_feature_names():
    assert prior_clr_feature_names(N_TEMPLATES) == [
        f"f{k}" for k in range(1, N_TEMPLATES + 1)
    ] + ["log_c_scale", "z"]


def test_ilr_basis_orthonormal_and_sum_zero():
    V = ilr_basis(N_TEMPLATES)
    assert V.shape == (N_TEMPLATES, N_TEMPLATES - 1)
    assert np.allclose(V.T @ V, np.eye(N_TEMPLATES - 1))  # orthonormal
    assert np.allclose(V.sum(axis=0), 0.0, atol=1e-12)  # spans the sum-zero hyperplane


def test_weights_ilr_roundtrip_dense():
    w = np.array([[0.1, 0.05, 0.05, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025]])
    w = w / w.sum(axis=1, keepdims=True)
    ilr = weights_to_ilr(w)
    assert ilr.shape == (1, N_TEMPLATES - 1)
    w2 = ilr_to_weights(ilr)
    assert np.allclose(w, w2, atol=1e-10)
    assert np.allclose(w2.sum(axis=1), 1.0)


def test_ilr_matches_clr_projection():
    """ilr = clr @ V exactly (the definition)."""
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(N_TEMPLATES), size=50)
    V = ilr_basis(N_TEMPLATES)
    assert np.allclose(weights_to_ilr(w), weights_to_clr(w) @ V)


def test_weights_ilr_sparse_smoothed():
    w = np.array([[0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    w2 = ilr_to_weights(weights_to_ilr(w))
    assert np.all(w2 > 0)
    assert np.allclose(w2.sum(axis=1), 1.0)
    assert w2[0, 3] > w2[0, 0]


def test_weights_clr_roundtrip_dense():
    w = np.array([[0.1, 0.05, 0.05, 0.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025]])
    w = w / w.sum(axis=1, keepdims=True)
    clr = weights_to_clr(w)
    assert np.allclose(clr.sum(axis=1), 0.0, atol=1e-12)
    w2 = clr_to_weights(clr)
    assert np.allclose(w, w2, atol=1e-10)
    assert np.allclose(w2.sum(axis=1), 1.0)


def test_uniform_at_zero_ilr():
    ilr = np.zeros((5, N_TEMPLATES - 1))
    w = ilr_to_weights(ilr)
    assert np.allclose(w, 1.0 / N_TEMPLATES, atol=1e-8)


def test_uniform_at_zero_clr():
    clr = np.zeros((5, N_TEMPLATES))
    w = clr_to_weights(clr)
    assert np.allclose(w, 1.0 / N_TEMPLATES, atol=1e-8)


def test_split_feature_matrix_ilr():
    ilr = np.zeros((2, N_TEMPLATES - 1))
    log_s = np.array([7.0, 7.5])
    z = np.array([0.5, 1.0])
    x = np.column_stack([ilr, log_s, z])
    a, ls, zz = split_feature_matrix(x, N_TEMPLATES, parameterization=PARAMETERIZATION_ILR)
    assert a.shape == (2, N_TEMPLATES)
    assert np.allclose(a.sum(axis=1), 1.0)
    assert np.allclose(ls, log_s)
    assert np.allclose(zz, z)


def test_split_feature_matrix_clr():
    clr = np.zeros((2, N_TEMPLATES))
    log_s = np.array([7.0, 7.5])
    z = np.array([0.5, 1.0])
    x = np.column_stack([clr, log_s, z])
    a, ls, zz = split_feature_matrix(x, N_TEMPLATES, parameterization=PARAMETERIZATION_CLR)
    assert a.shape == (2, N_TEMPLATES)
    assert np.allclose(a.sum(axis=1), 1.0)


def test_ilr_torch_matches_numpy():
    rng = np.random.default_rng(1)
    ilr = rng.normal(size=(8, N_TEMPLATES - 1))
    w_np = ilr_to_weights(ilr)
    w_t = ilr_to_weights_torch(torch.as_tensor(ilr)).numpy()
    assert np.allclose(w_np, w_t, atol=1e-12)
