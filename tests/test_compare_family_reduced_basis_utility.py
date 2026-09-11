"""Tests for downstream family reduced-basis utility metrics."""

import pandas as pd

from bedcosmo.num_visits.empirical.reduced.compare_family_reduced_basis_utility import (
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
