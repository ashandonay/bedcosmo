"""Tests for marginal-EIG prior sampling and sample-count helpers."""

import numpy as np
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
