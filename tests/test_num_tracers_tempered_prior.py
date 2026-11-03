"""The tempered DESI DR1 BAO posterior target and its ensemble sampler."""

import math
import os

import numpy as np
import pytest
import torch

from bedcosmo.num_tracers.prior import (
    TemperedDR1Posterior,
    integrated_autocorr_time,
    sample_ensemble,
)

_DATA = os.path.join(os.environ.get("HOME", ""), "data", "desi", "bao_dr1")


def test_ensemble_recovers_a_correlated_gaussian():
    cov = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=torch.float64)
    precision = torch.linalg.inv(cov)
    mean = torch.tensor([3.0, -1.0], dtype=torch.float64)

    def log_prob(x):
        r = x - mean
        return -0.5 * torch.einsum("ni,ij,nj->n", r, precision, r)

    generator = torch.Generator().manual_seed(0)
    x0 = mean + 0.1 * torch.randn(512, 2, generator=generator, dtype=torch.float64)
    chain, acceptance = sample_ensemble(log_prob, x0, 600, generator)
    x = chain[300:].reshape(-1, 2)
    assert 0.3 < acceptance < 0.9
    torch.testing.assert_close(x.mean(0), mean, atol=0.05, rtol=0)
    torch.testing.assert_close(torch.cov(x.T), cov, atol=0.05, rtol=0)


def test_autocorr_time_of_ar1():
    # x_t = phi x_{t-1} + noise has tau = (1 + phi) / (1 - phi) = 9 for phi = 0.8.
    rng = np.random.default_rng(0)
    phi, n = 0.8, 20_000
    x = np.zeros((n, 32, 1))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(size=(32, 1))
    assert integrated_autocorr_time(torch.from_numpy(x))[0] == pytest.approx(9.0, rel=0.1)


@pytest.mark.skipif(not os.path.isdir(_DATA), reason="desi bao_dr1 data not available")
def test_tempering_divides_the_log_likelihood():
    cold = TemperedDR1Posterior("base_w_wa", 1)
    hot = TemperedDR1Posterior("base_w_wa", 16)
    theta = torch.tensor(
        [[0.30, -0.8, -0.5, 100.0], [0.35, -0.5, -1.5, 98.0], [0.30, 0.5, 0.5, 100.0]],
        dtype=torch.float64,
    )
    lp_cold, lp_hot = cold.log_prob(theta), hot.log_prob(theta)
    torch.testing.assert_close(lp_hot[:2], lp_cold[:2] / 16)
    assert lp_cold[2] == lp_hot[2] == -math.inf  # w0 + wa = 1 breaks high_z_matter_dom


@pytest.mark.skipif(not os.path.isdir(_DATA), reason="desi bao_dr1 data not available")
def test_dr1_best_fit_chi2():
    # DESI 2024 VI's LCDM best fit to DR1 BAO alone fits its 12 points well.
    target = TemperedDR1Posterior("base", 1)
    chi2 = -2 * target.log_likelihood(torch.tensor([[0.295, 101.8]], dtype=torch.float64))
    assert chi2.item() < 15
