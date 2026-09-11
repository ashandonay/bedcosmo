"""Tests for the HDBSCAN family hyperparameter scan."""

import numpy as np

from bedcosmo.num_visits.empirical.reduced.scan_hdbscan_family_hyperparameters import (
    run_scan,
)


def test_run_scan_varies_primary_hyperparameters_independently():
    rng = np.random.default_rng(7)
    scores = np.concatenate(
        [
            rng.normal((-3, 0, 0), 0.2, size=(40, 3)),
            rng.normal((3, 0, 0), 0.2, size=(40, 3)),
            rng.normal((0, 3, 0), 0.2, size=(40, 3)),
        ]
    )
    results = run_scan(
        {2: scores[:, :2], 3: scores},
        np.array([0.6, 0.3, 0.1]),
        min_cluster_sizes=[10, 20],
        min_samples_values=[5, 10],
        selection_methods=["eom", "leaf"],
        baseline=(2, 10, 5, "eom"),
    )
    assert len(results) == 16
    assert set(results["n_components"]) == {2, 3}
    assert set(results["min_cluster_size"]) == {10, 20}
    assert set(results["min_samples"]) == {5, 10}
    assert set(results["selection_method"]) == {"eom", "leaf"}
    baseline = results.loc[
        (results["n_components"] == 2)
        & (results["min_cluster_size"] == 10)
        & (results["min_samples"] == 5)
        & (results["selection_method"] == "eom")
    ].iloc[0]
    assert baseline["ari_including_unclustered"] == 1.0
