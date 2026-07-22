"""Tests for marginal-EIG prior sampling and sample-count helpers."""

import numpy as np
import pytest
import torch

from bedcosmo.evaluate import Evaluator
from bedcosmo.num_visits.empirical.sed_prior import (
    EmpiricalPriorPool,
    sample_prior_batch,
    sample_prior_pool_unique,
)


def _dummy_pool(n_pool: int = 100) -> EmpiricalPriorPool:
    pool = torch.arange(n_pool * 3, dtype=torch.float64).reshape(n_pool, 3)
    return EmpiricalPriorPool(
        pool=pool,
        feature_names=["f1", "log_c_scale", "z"],
        n_templates=1,
        bounds_min=pool.min(dim=0).values,
        bounds_max=pool.max(dim=0).values,
    )


def test_sample_prior_pool_unique_no_replacement():
    pool = _dummy_pool(50)
    rows = sample_prior_pool_unique(pool, 50, generator=torch.Generator().manual_seed(0))
    assert rows.shape == (50, 3)
    assert len(torch.unique(rows, dim=0)) == 50


def test_sample_prior_batch_allows_duplicates():
    pool = _dummy_pool(10)
    rows = sample_prior_batch(pool, 100, generator=torch.Generator().manual_seed(0))
    assert len(torch.unique(rows, dim=0)) < 100


def test_marginal_prior_sample_count_matches_posterior_mc():
    assert Evaluator._marginal_prior_sample_count(200, 8) == 4096
    assert Evaluator._marginal_prior_sample_count(2000, 64) == 128000


def test_pooled_posterior_entropy_underestimates_conditional_average():
    """Pooling K×M posterior rows biases EIG low vs per-y k-NN averaging."""
    from bedcosmo.entropy import knn_entropy

    rng = np.random.default_rng(0)
    K, M = 80, 12
    # Two outer y values: tight clusters far apart → low per-y H, high mixture H.
    cols = np.zeros((K, M, 2))
    cols[:, : M // 2, 0] = rng.normal(-5.0, 0.15, size=(K, M // 2))
    cols[:, : M // 2, 1] = rng.normal(0.0, 0.15, size=(K, M // 2))
    cols[:, M // 2 :, 0] = rng.normal(5.0, 0.15, size=(K, M - M // 2))
    cols[:, M // 2 :, 1] = rng.normal(0.0, 0.15, size=(K, M - M // 2))

    per_y = float(np.mean([knn_entropy(cols[:, m, :], k=3) for m in range(M)]))
    pooled = knn_entropy(cols.reshape(-1, 2), k=3)
    assert pooled > per_y + 0.5


def test_knn_entropy_survives_clamped_duplicates():
    """Rows stacked on one value (a bound clamp) must not collapse the estimate.

    Regression: with ``k + 1`` identical rows the k-th neighbor distance is
    exactly 0, and the old ``max(eps, tiny)`` guard turned that into
    ``log(5e-324) ~ -744 nats`` per row -- 4 ties out of 200 took H from +2.6 to
    -18 bits, which silently drove the marginal-EIG design ranking.
    """
    from scipy.stats import gamma

    from bedcosmo.entropy import knn_entropy

    clean = gamma.rvs(3.0, size=200, random_state=np.random.default_rng(0))
    baseline = knn_entropy(clean[:, None], k=3)

    for n_clamped in (4, 10, 50):
        x = clean.copy()
        idx = np.random.default_rng(1).choice(x.size, n_clamped, replace=False)
        x[idx] = np.quantile(clean, 0.05)  # exact clamp onto a single bound value
        with pytest.warns(RuntimeWarning, match="duplicate"):
            h = knn_entropy(x[:, None], k=3)
        assert abs(h - baseline) < 0.2, f"{n_clamped} ties moved H by {h - baseline:.2f} bits"


def test_knn_entropy_does_not_mutate_caller_array():
    """The degenerate-column jitter must not write through to the caller."""
    from bedcosmo.entropy import knn_entropy

    x = np.zeros((50, 1))
    before = x.copy()
    knn_entropy(x, k=3)
    assert np.array_equal(x, before)


def test_estimators_agree_on_a_known_gaussian():
    """k-NN / KDE / Gaussian plug-ins must agree with the analytic 1-D entropy."""
    from bedcosmo.entropy import gaussian_entropy, kde_entropy, knn_entropy

    sigma = 0.7
    x = np.random.default_rng(0).normal(0.0, sigma, size=4000)[:, None]
    truth = 0.5 * np.log2(2 * np.pi * np.e * sigma**2)

    assert abs(gaussian_entropy(x) - truth) < 0.05  # exact family -> tightest
    assert abs(kde_entropy(x) - truth) < 0.10
    assert abs(knn_entropy(x, k=3) - truth) < 0.10


def test_kde_entropy_subsamples_eval_points():
    """max_eval_points bounds the O(n_eval * n_fit) logpdf without moving the answer."""
    from bedcosmo.entropy import kde_entropy

    x = np.random.default_rng(1).normal(size=3000)[:, None]
    assert abs(kde_entropy(x) - kde_entropy(x, max_eval_points=500)) < 0.1


def test_nats_bits_round_trip():
    from bedcosmo.entropy import bits_to_nats, nats_to_bits

    assert np.allclose(nats_to_bits(np.log(2.0)), 1.0)
    assert np.allclose(bits_to_nats(1.0), np.log(2.0))
    assert np.allclose(bits_to_nats(nats_to_bits([1.0, 2.0])), [1.0, 2.0])


def test_plugin_entropy_from_log_probs_joint_and_factorized():
    """Joint and factorized priors must give the same H when the parts agree."""
    from bedcosmo.entropy import plugin_entropy_from_log_probs

    a = torch.full((10, 3), -1.5, dtype=torch.float64)
    b = torch.full((10, 3), -0.5, dtype=torch.float64)

    joint = plugin_entropy_from_log_probs({"joint": a + b}, ["x", "y"])
    factored = plugin_entropy_from_log_probs({"x": a, "y": b}, ["x", "y"])
    assert torch.allclose(joint, factored)
    assert torch.allclose(joint, torch.full((3,), 2.0, dtype=torch.float64))


def test_physical_samples_valid_mask_flags_out_of_support():
    """The mask counterpart to the clamp: out-of-bounds/non-finite -> False."""
    from bedcosmo.base import BaseExperiment
    from bedcosmo.custom_dist import EmpiricalPrior

    class _Stub:  # BaseExperiment is abstract; the method only needs these attrs.
        cosmo_params = ["z", "other"]
        prior = {"z": EmpiricalPrior(0.01, 2.0)}

    exp = _Stub()

    samples = torch.tensor(
        [
            [0.5, 1.0],  # in support
            [0.005, 1.0],  # below low
            [3.0, 1.0],  # above high
            [float("nan"), 1.0],  # non-finite
        ],
        dtype=torch.float64,
    )
    mask = BaseExperiment._physical_samples_valid_mask(exp, samples)
    assert mask[:, 0].tolist() == [True, False, False, False]
    # A param without an EmpiricalPrior is only checked for finiteness.
    assert mask[:, 1].all()
