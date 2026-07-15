"""2D KDE entropy toy validation (mirrors production prior scoring)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers.kde_entropy_toy import (
    apply_transform,
    bits,
    cov_entropy_from_base,
    fit_prod_kde,
    kde_plugin_entropy,
    linear_log_abs_det,
    linear_transform,
    log_abs_det_exp_z2,
    log_abs_det_quadratic_z1,
    mvn_analytic_entropy,
    naive_standard_normal_entropy,
    sample_mvn,
    sample_standard_normal,
    standard_normal_log_prob,
    transform_exp_z2,
    transform_quadratic_z1,
)

N_TRAIN = 10_000
N_EVAL = 5_000
PROD_BANDWIDTH = 0.3

# Correlated 2D MVN: sigma1=1, sigma2=0.5, rho=0.5
MVN_COV = np.array([[1.0, 0.25], [0.25, 0.25]], dtype=np.float64)

LINEAR_A = np.array([[1.0, 0.5], [0.0, 1.5]], dtype=np.float64)


@pytest.mark.parametrize("seed", [0, 1])
def test_mvn_kde_matches_analytic_entropy(seed: int) -> None:
    x_train = sample_mvn(N_TRAIN, MVN_COV, seed=seed)
    x_eval = sample_mvn(N_EVAL, MVN_COV, seed=seed + 10_000)
    artifact = fit_prod_kde(x_train, bandwidth=PROD_BANDWIDTH)

    h_analytic = mvn_analytic_entropy(MVN_COV)
    h_kde = kde_plugin_entropy(artifact, x_eval)

    assert abs(bits(h_kde) - bits(h_analytic)) < 1.2


@pytest.mark.parametrize("seed", [0, 1])
def test_mvn_kde_training_bias(seed: int) -> None:
    x_train = sample_mvn(N_TRAIN, MVN_COV, seed=seed)
    x_eval = sample_mvn(N_EVAL, MVN_COV, seed=seed + 10_000)
    artifact = fit_prod_kde(x_train, bandwidth=PROD_BANDWIDTH)

    from bedcosmo.num_visits.empirical.sed_prior import score_kde_artifact

    log_p_train = score_kde_artifact(artifact, x_train)
    log_p_eval = score_kde_artifact(artifact, x_eval)
    # In-sample rows tend to score higher log p (lower H) than fresh draws.
    assert float(np.mean(log_p_train)) >= float(np.mean(log_p_eval)) - 0.05


@pytest.mark.parametrize("seed", [0, 1])
def test_linear_transform_jacobian_matches_kde(seed: int) -> None:
    z_train = sample_standard_normal(N_TRAIN, seed=seed)
    z_eval = sample_standard_normal(N_EVAL, seed=seed + 10_000)
    y_train = linear_transform(z_train, LINEAR_A)
    y_eval = linear_transform(z_eval, LINEAR_A)

    log_det = linear_log_abs_det(LINEAR_A)
    log_p_z = standard_normal_log_prob(z_eval)
    log_det_rows = np.full(z_eval.shape[0], log_det)
    h_cov, h_identity, _ = cov_entropy_from_base(z_eval, log_p_z, log_det_rows)
    assert abs(h_cov - h_identity) < 1e-10

    artifact = fit_prod_kde(y_train, bandwidth=PROD_BANDWIDTH)
    h_kde = kde_plugin_entropy(artifact, y_eval)

    assert abs(bits(h_kde) - bits(h_cov)) < 0.75


@pytest.mark.parametrize("seed", [0, 1])
def test_nonlinear_kde_vs_cov_entropy(seed: int) -> None:
    z_train = sample_standard_normal(N_TRAIN, seed=seed)
    z_eval = sample_standard_normal(N_EVAL, seed=seed + 10_000)
    y_train, log_det_train = apply_transform(
        z_train, transform_quadratic_z1, log_abs_det_quadratic_z1
    )
    y_eval, log_det_eval = apply_transform(
        z_eval, transform_quadratic_z1, log_abs_det_quadratic_z1
    )
    del log_det_train

    log_p_z = standard_normal_log_prob(z_eval)
    h_cov, _, _ = cov_entropy_from_base(z_eval, log_p_z, log_det_eval)

    artifact = fit_prod_kde(y_train, bandwidth=PROD_BANDWIDTH)
    h_kde = kde_plugin_entropy(artifact, y_eval)
    h_naive = naive_standard_normal_entropy(y_eval)

    assert abs(bits(h_kde) - bits(h_cov)) < 0.75
    assert abs(bits(h_naive) - bits(h_cov)) > 0.5


@pytest.mark.parametrize("seed", [0])
def test_exp_transform_cov_identity(seed: int) -> None:
    """y2 = exp(z2): H(y) equals H(z) but naive MVN on y fails."""
    z_eval = sample_standard_normal(N_EVAL, seed=seed)
    y_eval, log_det = apply_transform(z_eval, transform_exp_z2, log_abs_det_exp_z2)
    log_p_z = standard_normal_log_prob(z_eval)

    h_z = float(-np.mean(log_p_z))
    h_cov, h_identity, _ = cov_entropy_from_base(z_eval, log_p_z, log_det)
    assert abs(h_cov - h_identity) < 1e-10
    assert abs(h_cov - h_z) < 0.05

    h_naive = naive_standard_normal_entropy(y_eval)
    assert abs(bits(h_naive) - bits(h_cov)) > 0.3
