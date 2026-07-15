"""Tests for joint whitening on :class:`bedcosmo.transform.Bijector`."""

import numpy as np
import torch
from pyro import distributions as dist

from bedcosmo.transform import Bijector


class _MockExperiment:
    """Minimal experiment stub for Bijector CDF construction."""

    def __init__(self, prior, device="cpu"):
        self.prior = prior
        self.device = device
        self.cosmo_params = list(prior.keys())

    def sample_parameters(self, sample_shape, prior=None, use_prior_flow=True):
        p = prior if prior is not None else self.prior
        return {k: p[k].sample(sample_shape) for k in p}


def _build_bijector_on_gaussian_marginals(n_samples=50_000):
    prior = {
        "a": dist.Normal(0.0, 1.0),
        "b": dist.Normal(2.0, 0.5),
    }
    exp = _MockExperiment(prior)
    bj = Bijector(exp, cdf_bins=500, cdf_samples=n_samples, param_keys=["a", "b"])
    return bj, exp


def _correlated_physical(n, rho_target=0.7, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.array([0.0, 2.0])
    cov = np.array([[1.0, rho_target * 0.5], [rho_target * 0.5, 0.25]])
    return rng.multivariate_normal(mean, cov, size=n)


def test_joint_round_trip():
    bj, _ = _build_bijector_on_gaussian_marginals()
    x = _correlated_physical(5000, rho_target=0.65)
    bj.fit_joint_gaussianizer(
        x, param_names=["a", "b"], param_indices=[0, 1], shrinkage=1e-3
    )
    xt = torch.as_tensor(x[:256], dtype=torch.float64)
    z = bj.prior_to_gaussian_joint(xt)
    xr = bj.gaussian_to_prior_joint(z)
    err = (xr - xt).abs().max().item()
    assert err < 0.35, f"round-trip max error {err}"


def test_joint_log_det_finite():
    bj, _ = _build_bijector_on_gaussian_marginals()
    x = _correlated_physical(1000)
    bj.fit_joint_gaussianizer(
        x, param_names=["a", "b"], param_indices=[0, 1]
    )
    xt = torch.as_tensor(x, dtype=torch.float64)
    log_det = bj.joint_log_abs_det_jacobian(xt)
    assert torch.isfinite(log_det).all()
    assert log_det.shape == (xt.shape[0],)


def test_bijector_state_round_trip():
    bj, _ = _build_bijector_on_gaussian_marginals()
    x = _correlated_physical(3000)
    bj.fit_joint_gaussianizer(
        x, param_names=["a", "b"], param_indices=[0, 1], shrinkage=1e-3
    )
    state = bj.get_state()
    assert "joint_state" in state
    assert state["joint_state"] is not None
    assert "cdfs" in state

    bj2 = Bijector(
        _MockExperiment({"a": dist.Normal(0, 1), "b": dist.Normal(2, 0.5)}),
        skip_sampling=True,
    )
    bj2.set_state(state)
    assert bj2.uses_joint_gaussianizer()

    xt = torch.as_tensor(x[:32], dtype=torch.float64)
    z1 = bj.prior_to_gaussian_joint(xt)
    z2 = bj2.prior_to_gaussian_joint(xt)
    assert torch.allclose(z1, z2, atol=1e-5)


def test_marginal_only_bijector_state():
    bj, exp = _build_bijector_on_gaussian_marginals(n_samples=5000)
    flat = bj.get_state()
    bj2 = Bijector(exp, skip_sampling=True)
    bj2.set_state(flat)
    assert not bj2.uses_joint_gaussianizer()
    assert set(bj2.cdfs.keys()) == {"a", "b"}


def test_fit_from_matrix_marginal_vs_joint():
    x = _correlated_physical(2000)
    names = ["a", "b"]
    bj_marg = Bijector.fit_from_matrix(
        x, names, input_transform_type="marginal", cdf_bins=200
    )
    bj_joint = Bijector.fit_from_matrix(
        x, names, input_transform_type="joint", cdf_bins=200
    )
    assert not bj_marg.uses_joint_gaussianizer()
    assert bj_joint.uses_joint_gaussianizer()
    xt = torch.as_tensor(x[:64], dtype=torch.float64)
    y_m = bj_marg.matrix_to_gaussian(xt, apply_joint=False)
    y_j = bj_joint.matrix_to_gaussian(xt, apply_joint=True)
    assert y_m.shape == y_j.shape == xt.shape
