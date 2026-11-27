"""Checks for the offline sparse coefficient experiment."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "mixture_prototype",
    Path(__file__).resolve().parents[1] / "scripts/prototype_empirical_mixture.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
CoefficientMixture = _MODULE.CoefficientMixture


def test_conditional_zero_masses_and_simplex():
    weights = np.array([[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0], [0.2, 0.8, 0]])
    model = CoefficientMixture(weights)
    draws = model.sample(50000, seed=3)
    expected = (model.p0 - np.prod(model.p0)) / (1 - np.prod(model.p0))
    np.testing.assert_allclose((draws == 0).mean(axis=0), expected, atol=0.008)
    np.testing.assert_allclose(draws.sum(axis=1), 1)
    assert np.all((draws >= 0) & (draws <= 1))
    np.testing.assert_array_equal(draws, model.sample(50000, seed=3))


def test_reflected_kde_retains_positive_variation_without_boundary_atoms():
    model = CoefficientMixture(np.tile([0.001, 0.999], (10, 1)), bandwidth=0.2)
    draws = model.sample(10000)
    assert np.all((draws > 0) & (draws < 1))
    assert np.std(draws[:, 0]) > 0.05


def test_threshold_and_singleton_support():
    model = CoefficientMixture([[0.001, 0.999]], zero_threshold=0.01)
    np.testing.assert_array_equal(model.sample(30), np.tile([0, 1], (30, 1)))
    with pytest.raises(ValueError, match="every component"):
        CoefficientMixture([[0.5, 0.5]], zero_threshold=0.6)


@pytest.mark.parametrize("weights", [[[0, 0]], [[-1, 2]], [[np.nan, 1]], [[0.2, 0.2]], []])
def test_invalid_training_rows(weights):
    with pytest.raises(ValueError, match="simplex"):
        CoefficientMixture(weights)


def test_diagnostics_detect_lost_dependence():
    rng = np.random.default_rng(1)
    a = rng.uniform(size=1000)
    reference = np.column_stack([a, 1 - a, a * 2, a * 3])
    independent = reference.copy()
    independent[:, 2:] = independent[rng.permutation(len(reference)), 2:]
    assert _MODULE.diagnostics(reference, reference)["correlation_frobenius"] == 0
    assert _MODULE.diagnostics(reference, independent)["correlation_frobenius"] > 1
