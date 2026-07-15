"""Validate the reliable H(z) estimators at the real problem's dimensionality.

The transformed prior is 14-dimensional (``input_transform_type="joint"``), where
the KDE plug-in is unreliable (see ``test_kde_entropy_toy_2d``). These tests check
the replacement estimators on a known correlated, non-Gaussian distribution with a
tight analytic-MC reference entropy:

- :func:`cumulant_negentropy_entropy` improves on the Gaussian-only baseline
  (captures low-order non-Gaussianity) and is exact on a pure Gaussian.
- :func:`flow_plugin_entropy` recovers the entropy closely and beats the cumulant
  estimator, since it captures higher-order structure the cumulants miss.

The flow tests are marked ``slow`` (CPU NSF training). Run the fast subset with
``pytest -m "not slow"``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.helpers.entropy_estimators import (
    cumulant_negentropy_entropy,
    flow_plugin_entropy,
    gaussian_entropy,
    make_correlated_nongaussian,
)

BITS = 1.0 / math.log(2.0)


def _bits(nats: float) -> float:
    return nats * BITS


def test_ground_truth_is_self_consistent():
    """A pure Gaussian (strength=0) has entropy equal to its Gaussian baseline."""
    y, h_true = make_correlated_nongaussian(8, 40_000, seed=1, strength=0.0)
    h_gauss = gaussian_entropy(np.cov(y, rowvar=False))
    # No non-Gaussianity: analytic-MC reference and Gaussian entropy must agree.
    assert abs(_bits(h_gauss - h_true)) < 0.1


def test_cumulant_exact_on_gaussian():
    """Negentropy ~ 0 for Gaussian data, so the estimate matches the truth."""
    y, h_true = make_correlated_nongaussian(8, 40_000, seed=2, strength=0.0)
    h_cum, diag = cumulant_negentropy_entropy(y, return_diagnostics=True)
    # Sample-noise negentropy only; well under the non-Gaussian signal scale.
    assert _bits(diag["negentropy"]) < 0.2
    assert abs(_bits(h_cum - h_true)) < 0.2


def test_cumulant_beats_gaussian_baseline_14d():
    """At D=14 the cumulant correction reduces the Gaussian-only overestimate."""
    y, h_true = make_correlated_nongaussian(14, 20_000, seed=0, strength=0.6)
    h_gauss = gaussian_entropy(np.cov(y, rowvar=False))
    h_cum = cumulant_negentropy_entropy(y)
    err_gauss = _bits(h_gauss - h_true)
    err_cum = _bits(h_cum - h_true)
    # Both overestimate (they under-account for non-Gaussianity), but the
    # cumulant correction moves strictly toward the truth.
    assert err_gauss > 0.5
    assert 0.0 < err_cum < err_gauss


@pytest.mark.slow
def test_flow_recovers_entropy_14d():
    """NF plug-in is accurate at D=14 and beats the cumulant cross-check."""
    y, h_true = make_correlated_nongaussian(14, 20_000, seed=0, strength=0.6)
    h_cum = cumulant_negentropy_entropy(y)
    h_flow = flow_plugin_entropy(
        y, epochs=80, hidden_features=(64, 64), batch_size=2048, seed=0
    )
    err_flow = abs(_bits(h_flow - h_true))
    err_cum = abs(_bits(h_cum - h_true))
    # Flow lands within ~0.5 bits and is closer to the truth than the cumulant.
    assert err_flow < 0.5
    assert err_flow < err_cum
