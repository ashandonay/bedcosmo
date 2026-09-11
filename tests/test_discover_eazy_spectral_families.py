"""Tests for decoding clustered spectral families into EAZY subsets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bedcosmo.num_visits.empirical.reduced.discover_eazy_spectral_families import (
    decode_family_bases,
    select_balanced_family_bases,
)
from bedcosmo.num_visits.empirical.reduced.plot_family_subset_tradeoffs import (
    select_family_candidates,
    select_family_template_pool,
)


def test_decode_family_bases_reports_completeness_and_purity():
    labels = np.array([1, 1, 1, 2, 2, 0])
    passes = np.array(
        [
            [True, True],
            [True, True],
            [False, True],
            [True, False],
            [False, True],
            [True, False],
        ]
    )
    zeros = np.zeros_like(passes, dtype=float)
    searches = {
        2: {
            "subsets": np.array([[1, 2], [1, 3]]),
            "passes": passes,
            "delta": zeros,
            "chi2_dof": zeros,
            "color_rms": zeros,
        }
    }

    summary, candidates = decode_family_bases(
        labels,
        searches,
        required_coverage=0.8,
    )

    f01_t1_t2 = candidates.loc[
        (candidates["family"] == "F01") & (candidates["templates"] == "T1+T2")
    ].iloc[0]
    assert f01_t1_t2["passing_count"] == 2
    assert f01_t1_t2["family_member_count"] == 3
    assert f01_t1_t2["subset_passing_count"] == 4
    assert f01_t1_t2["coverage_fraction"] == 2 / 3
    assert f01_t1_t2["purity_fraction"] == 1 / 2

    f01 = summary.loc[summary["family"] == "F01"].iloc[0]
    assert f01["selected_templates"] == "T1+T3"
    assert f01["selected_passing_count"] == 3
    assert f01["selected_subset_passing_count"] == 4
    assert f01["selected_coverage_fraction"] == 1.0
    assert f01["selected_purity_fraction"] == 3 / 4


def test_select_family_candidates_filters_and_orders_subsets():
    candidates = pd.DataFrame(
        {
            "family": ["F01", "F01", "F01", "F02"],
            "n_templates": [2, 3, 2, 2],
            "templates": ["T1+T2", "T1+T2+T3", "T1+T3", "T1+T2"],
            "coverage_fraction": [0.8, 0.95, 0.9, 1.0],
            "purity_fraction": [0.7, 0.9, 0.6, 1.0],
        }
    )
    selected = select_family_candidates(
        candidates,
        "f01",
        min_completeness=0.85,
        min_purity=0.5,
        max_templates=2,
    )
    assert selected["templates"].tolist() == ["T1+T3"]


def test_family_supported_candidates_use_cumulative_mean_weight_pool():
    family_weights = pd.DataFrame(
        {
            "family": ["F01", "F01", "F01", "F02"],
            "template": ["T1", "T2", "T3", "T1"],
            "mean_weight": [0.6, 0.3, 0.1, 1.0],
        }
    )
    pool, retained = select_family_template_pool(
        family_weights,
        "f01",
        required_weight=0.85,
    )
    assert pool == ["T1", "T2"]
    assert retained == pytest.approx(0.9)

    candidates = pd.DataFrame(
        {
            "family": ["F01", "F01", "F01"],
            "n_templates": [1, 2, 2],
            "templates": ["T1", "T1+T2", "T1+T3"],
            "coverage_fraction": [0.5, 0.8, 0.9],
            "purity_fraction": [0.5, 0.6, 0.7],
        }
    )
    selected = select_family_candidates(
        candidates,
        "F01",
        supported_templates=set(pool),
    )
    assert selected["templates"].tolist() == ["T1", "T1+T2"]


def test_balanced_family_basis_uses_harmonic_mean():
    candidates = pd.DataFrame(
        {
            "family": ["F01", "F01"],
            "n_templates": [1, 2],
            "templates": ["T1", "T1+T2"],
            "coverage_fraction": [0.9, 0.7],
            "purity_fraction": [0.2, 0.6],
            "passing_count": [90, 70],
            "subset_passing_count": [450, 117],
        }
    )
    family_weights = pd.DataFrame(
        {
            "family": ["F01", "F01"],
            "template": ["T1", "T2"],
            "mean_weight": [0.7, 0.3],
        }
    )
    result = select_balanced_family_bases(
        candidates,
        family_weights,
        family_weight_coverage=0.99,
    ).iloc[0]
    assert result["balanced_templates"] == "T1+T2"
    assert result["balanced_f1"] == pytest.approx(2 * 0.7 * 0.6 / (0.7 + 0.6))
