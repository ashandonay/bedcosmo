"""Checks for joint diagnostics and the induced face-density calculation."""

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_empirical_mixture_density import (
    joint_log_density,
    reflected_log_pdf,
    simplex_kernel_log_density,
)
from prototype_joint_empirical_mixture import JointPatternMixture
from validate_joint_empirical_mixture import joint_color_metrics


def test_joint_metric_detects_dependence_loss_with_unchanged_marginals():
    rng = np.random.default_rng(2)
    latent = rng.normal(size=800)
    reference = np.column_stack([latent + rng.normal(0, 0.05, len(latent)) for _ in range(5)])
    shuffled = np.column_stack([rng.permutation(column) for column in reference.T])
    same = joint_color_metrics(reference, reference, reference, max_rows=800)
    different = joint_color_metrics(reference, shuffled, reference, max_rows=800)
    assert abs(same["rbf_mmd2_biased"]) < 1e-14
    assert different["rbf_mmd2_biased"] > 0.01
    assert different["max_marginal_color_ks"] == 0
    assert different["sliced_wasserstein"] > same["sliced_wasserstein"] + 0.05


def test_joint_metric_is_reproducible_and_does_not_hide_constant_coordinates():
    rng = np.random.default_rng(7)
    training = rng.normal(size=(100, 5))
    reference = rng.normal(size=(100, 5))
    samples = rng.normal(size=(100, 5))
    assert joint_color_metrics(reference, samples, training) == joint_color_metrics(
        reference, samples, training
    )
    with pytest.raises(ValueError, match="nonzero"):
        joint_color_metrics(reference, samples, np.ones((100, 5)))


@pytest.mark.parametrize("sigma", [0.01, 0.2, 2.0])
def test_reflected_density_normalizes_even_with_multiple_reflections(sigma):
    nodes, weights = leggauss(512)
    x = (nodes + 1) / 2
    for center in [0, 0.3, 1]:
        value = np.sum(np.exp(reflected_log_pdf(x, center, sigma)) * weights / 2)
        assert value == pytest.approx(1, abs=2e-10)


def test_closed_simplex_density_integrates_to_one_and_matches_sampler():
    centers = np.array([[0.2, 0.8], [0.7, 0.3]])
    sigma = np.array([0.12, 0.12])
    nodes, weights = leggauss(200)
    a = (nodes + 1) / 2
    logpdf = simplex_kernel_log_density(np.column_stack([a, 1 - a]), centers, sigma, order=200)
    np.testing.assert_allclose(np.sum(np.exp(logpdf) * weights[:, None] / 2, axis=0), 1, atol=1e-8)
    # Independently generate reflected pre-closure kernels and compare a simple
    # event probability against the induced density integral.
    rng = np.random.default_rng(9)
    donors = centers[rng.integers(2, size=100000)]
    preclosure = donors + rng.normal(size=donors.shape) * sigma
    reflected = 1 - np.abs(np.remainder(preclosure, 2) - 1)
    samples = reflected[:, 0] / reflected.sum(axis=1)
    event_x = (nodes + 1) * 0.3
    event_logpdf = simplex_kernel_log_density(
        np.column_stack([event_x, 1 - event_x]), centers, sigma, order=200
    )
    probability = np.sum(np.exp(event_logpdf).mean(axis=1) * weights * 0.3)
    assert abs((samples < 0.6).mean() - probability) < 0.006


def test_pattern_density_includes_discrete_mass_and_rejects_atoms():
    rows = np.array([[1, 0, 2, 0.1], [1, 0, 4, 0.8], [0, 1, 3, 0.4]])
    model = JointPatternMixture(rows)
    query = np.array([1, 0, 3, 0.4])
    expected = reflected_log_pdf(3, rows[:2, -2], model.widths[model.group[0], -2], 2, 4)
    expected += reflected_log_pdf(0.4, rows[:2, -1], model.widths[model.group[0], -1], 0.1, 0.8)
    assert np.exp(joint_log_density(model, query)[0]) == pytest.approx(np.exp(expected).sum() / 3)
    assert np.isneginf(joint_log_density(model, [0.5, 0.5, 3, 0.4])[0])
    with pytest.raises(ValueError, match="singular"):
        joint_log_density(model, [0, 1, 3, 0.4])
