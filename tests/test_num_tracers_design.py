"""num_tracers design.py: pool subsampling keeps every pinned design."""
import numpy as np
import pytest

from bedcosmo.num_tracers import design

PINNED = [[0.05, 0.30, 0.40, 0.25], [0.04, 0.26, 0.40, 0.30], [0.03, 0.27, 0.35, 0.35]]


def _pool(n_target):
    return design.budget_pool(sum_lower=1.0, sum_upper=1.0, include=PINNED, n_target=n_target)


def test_n_target_equal_to_the_pinned_count_keeps_only_the_pinned_designs():
    pool = _pool(len(PINNED))
    pinned = np.asarray(PINNED)
    np.testing.assert_allclose(pool, pinned[np.lexsort(pinned.T[::-1])])


def test_n_target_above_the_pinned_count_adds_grid_designs():
    pool = _pool(len(PINNED) + 5)
    assert pool.shape[0] == len(PINNED) + 5
    assert design.is_pinned(pool, np.asarray(PINNED)).sum() == len(PINNED)


def test_n_target_below_the_pinned_count_raises():
    with pytest.raises(ValueError, match="below the 3 pinned designs"):
        _pool(len(PINNED) - 1)
