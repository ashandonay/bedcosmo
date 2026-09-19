"""Tests for bootstrap stability of PCA/HDBSCAN spectral families."""

import numpy as np

from bedcosmo.num_visits.empirical.reduced.bootstrap_hdbscan_family_stability import (
    ClusterConfig,
    pairwise_cluster_scores,
    parse_config,
    run_bootstrap_stability,
)


def test_pairwise_scores_distinguish_splits_and_merges():
    reference = np.array([0, 0, 0, 1, 1, 1])
    split = np.array([0, 0, 1, 2, 2, 2])
    merged = np.zeros(6, dtype=int)
    split_scores = pairwise_cluster_scores(reference, split)
    merge_scores = pairwise_cluster_scores(reference, merged)
    assert split_scores["pair_precision"] == 1.0
    assert split_scores["pair_recall"] < 1.0
    assert merge_scores["pair_recall"] == 1.0
    assert merge_scores["pair_precision"] < 1.0


def test_bootstrap_stability_runs_repeated_pca_and_clustering():
    rng = np.random.default_rng(3)
    ilr = np.concatenate(
        [
            rng.normal((-4, 0, 0), 0.15, size=(50, 3)),
            rng.normal((4, 0, 0), 0.15, size=(50, 3)),
            rng.normal((0, 4, 0), 0.15, size=(50, 3)),
        ]
    )
    config = ClusterConfig(2, 10, 5, "eom")
    results, family_results, references = run_bootstrap_stability(
        ilr,
        [config],
        repetitions=3,
        sample_fraction=0.8,
        seed=9,
    )
    assert len(results) == 3
    assert len(references) == 1
    assert references.iloc[0]["n_families"] == 3
    assert results["pair_f1"].min() > 0.95
    assert set(family_results["reference_family"]) == {0, 1, 2}


def test_parse_config():
    assert parse_config("8:300:30:leaf") == ClusterConfig(8, 300, 30, "leaf")
