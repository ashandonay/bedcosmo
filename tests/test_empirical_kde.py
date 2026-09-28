"""Tests for empirical SED KDE sampling."""

from __future__ import annotations

import joblib
import numpy as np

from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
    fit_empirical_gaussianizer,
    fit_sed_prior_kde,
    get_empirical_gaussianizer,
    pack_kde_artifact,
    refit_gaussianizer,
    sample_sed_prior,
    save_sed_prior_kde,
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


def _cdf_rows(gaussianizer, name):
    """Rows behind a CDF table: its steps are multiples of 1 / n_rows."""
    du = np.diff(gaussianizer.cdfs[name]["cdf_values"].double().numpy())[1:-1]  # ends are eps-clamped
    return round(1 / du[du > 0].min())


def test_refit_gaussianizer_uses_every_draw_and_keeps_the_kde(tmp_path):
    rng = np.random.default_rng(0)
    names = ["f1", "log_c_scale", "z"]
    training = np.column_stack(
        [rng.normal(0, 1, 400), rng.normal(5, 0.2, 400), rng.uniform(0.1, 2.0, 400)]
    )
    kde, scaler = fit_sed_prior_kde(training, bandwidth=0.3)
    metadata = {
        "n_templates": 2,
        "gaussianizer_fit_source": "kde",
        "gaussianizer_fit_seed": 3,
        "gaussianizer_fit_samples": 1000,
        "gaussianizer_fit_n_rows": 1000,
        "gaussianizer_shrinkage": 1e-3,
        "gaussianizer_eps": 1e-3,
        "kde_parameters": {"seed": 7, "gaussianizer_fit_samples": 1000},
    }
    old = fit_empirical_gaussianizer(training, names)
    path = tmp_path / "sed_prior_kde_native.joblib"
    save_sed_prior_kde(
        path,
        pack_kde_artifact(kde, scaler, names, training, metadata=metadata, gaussianizer=old),
    )

    refit_gaussianizer(path, n_samples=60_000)

    artifact = joblib.load(path)
    new = get_empirical_gaussianizer(artifact)
    # All 60k draws reach the CDF tables (the old 50k cap would leave 50k).
    assert all(_cdf_rows(new, n) == 60_000 for n in names)
    assert new.joint_state is not None
    assert artifact["metadata"]["gaussianizer_fit_n_rows"] == 60_000
    assert artifact["metadata"]["kde_parameters"]["gaussianizer_fit_samples"] == 60_000
    assert np.array_equal(artifact["training_x"], training)
    # Same KDE and seed: the prior's draws are unchanged.
    assert np.array_equal(
        sample_sed_prior(artifact, 200, seed=1), sample_sed_prior(
            pack_kde_artifact(kde, scaler, names, training, metadata=metadata), 200, seed=1
        )
    )
