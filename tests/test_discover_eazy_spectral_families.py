"""Tests for decoding clustered spectral families into EAZY subsets."""

from __future__ import annotations

import numpy as np

from bedcosmo.num_visits.empirical.reduced.discover_eazy_spectral_families import (
    decode_family_bases,
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
    assert f01_t1_t2["subset_passing_count"] == 4
    assert f01_t1_t2["coverage_fraction"] == 2 / 3
    assert f01_t1_t2["purity_fraction"] == 1 / 2

    f01 = summary.loc[summary["family"] == "F01"].iloc[0]
    assert f01["selected_templates"] == "T1+T3"
    assert f01["selected_passing_count"] == 3
    assert f01["selected_subset_passing_count"] == 4
    assert f01["selected_coverage_fraction"] == 1.0
    assert f01["selected_purity_fraction"] == 3 / 4
