"""Tests for empirical SED KDE sampling."""

from __future__ import annotations

import numpy as np

from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
    fit_sed_prior_kde,
    sample_sed_prior,
)


def test_training_bounds_use_rejection_without_endpoint_atoms():
    training = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    kde, scaler = fit_sed_prior_kde(training, bandwidth=2.0)
    artifact = {
        "kde": kde,
        "scaler": scaler,
        "feature_bounds_min": training.min(axis=0),
        "feature_bounds_max": training.max(axis=0),
        "n_templates": 2,
        "parameterization": "ilr",
        "support_mode": "smooth",
    }

    unrestricted = sample_sed_prior(
        artifact,
        500,
        seed=12,
        renormalize_a=False,
        restrict_to_training_bounds=False,
    )
    bounded = sample_sed_prior(
        artifact,
        500,
        seed=12,
        renormalize_a=False,
    )

    assert np.any((unrestricted < 0.0) | (unrestricted > 1.0))
    assert np.all((bounded >= 0.0) & (bounded <= 1.0))
    assert not np.any((bounded == 0.0) | (bounded == 1.0))
    assert np.array_equal(
        bounded,
        sample_sed_prior(
            artifact,
            500,
            seed=12,
            renormalize_a=False,
        ),
    )
