"""Tests for downstream family reduced-basis utility metrics."""

import pandas as pd

from bedcosmo.num_visits.empirical.reduced.compare_family_reduced_basis_utility import (
    mark_pareto_front,
    summarize_selected_bases,
)


def test_summarize_selected_bases_uses_end_to_end_population_denominators():
    selected = pd.DataFrame(
        {
            "family": ["F01", "F02"],
            "balanced_passing_count": [60, 20],
            "balanced_subset_passing_count": [75, 100],
            "balanced_f1": [0.8, 0.4],
            "balanced_n": [2, 4],
        }
    )
    members = pd.Series([100, 50], index=["F01", "F02"])
    result = summarize_selected_bases(selected, members, total_population=300)
    assert result["assigned_fraction"] == 0.5
    assert result["recovered_population_fraction"] == 80 / 300
    assert result["micro_completeness"] == 80 / 150
    assert result["micro_purity"] == 80 / 175
    assert result["member_weighted_family_f1"] == (100 * 0.8 + 50 * 0.4) / 150
    assert result["median_selected_n"] == 3


def test_mark_pareto_front_preserves_quality_tradeoffs():
    candidates = pd.DataFrame(
        {
            "completeness": [0.9, 0.8, 0.7, 0.9],
            "purity": [0.5, 0.7, 0.4, 0.5],
            "passing_count": [100, 80, 200, 120],
            "n_templates": [2, 2, 2, 1],
        }
    )
    marked = mark_pareto_front(candidates)
    assert marked["pareto_completeness_purity"].tolist() == [True, True, False, True]
    assert marked["pareto_quality_size_dimension"].tolist() == [False, True, True, True]
