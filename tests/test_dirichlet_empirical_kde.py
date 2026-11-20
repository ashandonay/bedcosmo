"""Verify mask, amplitude, redshift, donor means, and singleton behavior."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_dirichlet_empirical_kde import dirichlet_draws, fit_logratio_kernels, logratio_draws
from prototype_joint_empirical_mixture import JointPatternMixture


def test_draws_respect_simplex_and_preserve_donor_mean():
    rows = np.array([[0.1, 0.9, 0, 2, 0.2], [0.1, 0.9, 0, 3, 0.3], [0, 0.4, 0.6, 5, 0.7]])
    model = JointPatternMixture(rows, bandwidth=0.1)
    gaussian = model.sample(20000, seed=10)
    donors = np.random.default_rng(10).integers(len(rows), size=len(gaussian))
    samples, floored = dirichlet_draws(model, donors, gaussian, 30, np.random.default_rng(33))
    np.testing.assert_array_equal(samples[:, :-2] > 0, gaussian[:, :-2] > 0)
    np.testing.assert_array_equal(samples[:, -2:], gaussian[:, -2:])
    np.testing.assert_allclose(samples[:, :-2].sum(axis=1), 1)
    np.testing.assert_array_equal(samples[donors == 2], gaussian[donors == 2])
    assert abs(samples[donors != 2, 0].mean() - 0.1) < 0.003
    assert samples[donors != 2, 0].std() > 0.04
    assert floored == 0


def test_tiny_positive_fraction_keeps_its_mask():
    rows = np.array([[1e-15, 1, 2, 0.2], [1e-15, 1, 3, 0.3]])
    model = JointPatternMixture(rows, bandwidth=0.1)
    gaussian = model.sample(10, seed=8)
    donors = np.random.default_rng(8).integers(len(rows), size=10)
    draws, floored = dirichlet_draws(model, donors, gaussian, 30, np.random.default_rng(33))
    assert np.all(draws[:, :-2] > 0)
    np.testing.assert_allclose(draws[:, :-2].sum(axis=1), 1)
    assert floored == 10


def test_logratio_kernel_preserves_support_and_uses_fitted_full_covariance():
    rng = np.random.default_rng(22)
    coords = rng.multivariate_normal([0, 0], [[0.6, 0.3], [0.3, 0.5]], size=500)
    from scipy.linalg import helmert

    basis = helmert(3, full=False)
    a = np.exp(coords @ basis)
    a /= a.sum(axis=1, keepdims=True)
    rows = np.column_stack([a, np.zeros(500), np.ones(500), np.full(500, 0.4)])
    rows = np.vstack([rows, [0, 0, 0, 1, 2, 0.5]])
    model = JointPatternMixture(rows, bandwidth=0.1)
    kernels = fit_logratio_kernels(model)
    gaussian = model.sample(20000, seed=17)
    donors = np.random.default_rng(17).integers(len(rows), size=len(gaussian))
    output, floored = logratio_draws(model, kernels, donors, gaussian, 0.3, rng)
    smooth = donors < 500
    np.testing.assert_array_equal(output[:, :-2] > 0, gaussian[:, :-2] > 0)
    np.testing.assert_array_equal(output[:, -2:], gaussian[:, -2:])
    np.testing.assert_array_equal(output[~smooth], gaussian[~smooth])
    np.testing.assert_allclose(output[:, :-2].sum(axis=1), 1)
    residual = (np.log(output[smooth, :3]) - np.log(rows[donors[smooth], :3])) @ basis.T
    fitted_root = kernels[model.group[0]][1]
    assert fitted_root is not None
    expected = 0.3**2 * fitted_root @ fitted_root.T
    assert expected[0, 1] > 0.01
    np.testing.assert_allclose(np.cov(residual, rowvar=False), expected, atol=0.002)
    unsmoothed, _ = logratio_draws(model, kernels, donors, gaussian, 0, rng)
    np.testing.assert_array_equal(unsmoothed[smooth, :-2], rows[donors[smooth], :-2])
    assert floored == 0
