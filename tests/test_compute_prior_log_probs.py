"""Tests for ``LikelihoodDataset._compute_prior_log_probs``.

The function is the place where the prior log-probabilities entering the EIG
estimator are computed. There are two branches:

1. **No prior_flow.** Returns ``trace.compute_log_prob()`` per parameter and,
   when ``experiment.transform_input=True``, applies a change-of-variables
   correction ``- log|dT_cur/dx|`` so the prior log-prob lives in the same
   space as the posterior (the flow's natural output space).
2. **With prior_flow.** Evaluates ``prior_flow.log_prob`` in the prior_flow's
   native space (using ``prior_flow_bijector`` if it was trained with
   ``transform_input=True``), then symmetric change-of-variables: ``+ log|dT_pf/dx|``
   to land in theta-space, then ``- log|dT_cur/dx|`` if the current flow uses
   ``transform_input=True``.

These tests exercise both branches and the four ``(cur_T, pf_T)`` combinations
without spinning up a real flow -- a tiny stub stands in for ``prior_flow``.
"""

from types import SimpleNamespace

import math
import numpy as np
import pyro
import pytest
import torch
from pyro import distributions as dist

from bedcosmo.base import BaseExperiment
from bedcosmo.pyro_oed_src import LikelihoodDataset
from bedcosmo.transform import Bijector

# Smaller-than-default bijector resolution so each fixture builds in <1s.
_CDF_SAMPLES = 30_000
_CDF_BINS = 1_000

# Tolerances. The empirical-CDF bijector is approximate, so equalities only
# hold to ~percent level after averaging over a meaningful sample.
_TOL_TIGHT = 5e-3   # exact algebraic identities (e.g. cancellation)
_TOL_LOOSE = 5e-2   # statistical agreement of two estimators


# ============================================================================
# Minimal experiment stub; modeled on tests/test_transform_input.py::_Stub.
# ============================================================================

class _Stub(BaseExperiment):
    def __init__(self, prior, transform_input=False, prior_flow=None,
                 prior_flow_metadata=None):
        self.name = "stub_prior_logprob"
        self.device = "cpu"
        self.prior = prior
        self.cosmo_params = list(prior.keys())
        self.latex_labels = list(prior.keys())
        self.observation_labels = ["y"]
        self.transform_input = transform_input
        self.global_rank = 0
        self.profile = False
        self.prior_flow = prior_flow
        if prior_flow_metadata is not None:
            self.prior_flow_metadata = prior_flow_metadata

    def init_designs(self, **kwargs):
        self.designs = torch.zeros(1, len(self.cosmo_params))

    def init_prior(self, parameters, **kwargs):
        return self.prior, {}, self.latex_labels, None, None

    def pyro_model(self, design):
        # Used to draw samples + observations; we register all priors and a
        # dummy observation so trace.compute_log_prob() works.
        params = self.sample_parameters(())
        # Concatenate to a single feature vector.
        theta_vec = torch.cat([params[k] for k in self.cosmo_params], dim=-1)
        # Trivial observation model -- y deterministic given theta.
        pyro.sample("y", dist.Normal(theta_vec.sum(-1), torch.tensor(1.0)))
        return theta_vec

    def sample_parameters(self, sample_shape, prior=None, use_prior_flow=True, **kwargs):
        if prior is None:
            prior = self.prior
        parameters = {}
        for k, v in prior.items():
            parameters[k] = pyro.sample(k, v).unsqueeze(-1)
        return parameters


def _make_stub(prior, **kwargs):
    stub = _Stub(prior, **kwargs)
    stub.init_designs()
    if stub.transform_input:
        stub.param_bijector = Bijector(
            stub, cdf_bins=_CDF_BINS, cdf_samples=_CDF_SAMPLES, use_prior_flow=False,
        )
    return stub


def _make_prior_flow_bijector(stub):
    """Build a separate bijector with the same prior as ``stub``."""
    return Bijector(
        stub, cdf_bins=_CDF_BINS, cdf_samples=_CDF_SAMPLES, use_prior_flow=False,
    )


def _draw_samples(stub, n=2000, seed=0):
    """Draw n iid prior samples in the (n, D) shape that
    ``_compute_prior_log_probs`` expects."""
    torch.manual_seed(seed)
    cols = []
    for name in stub.cosmo_params:
        cols.append(stub.prior[name].sample((n,)).reshape(-1, 1))
    return torch.cat(cols, dim=-1)


def _build_dataset(stub):
    """Return a LikelihoodDataset wired to a stub experiment.

    We pass evaluation=False so __init__ doesn't need real designs; we only
    invoke _compute_prior_log_probs directly, which doesn't depend on dataset
    fields besides particle_batch_size and device.
    """
    ds = LikelihoodDataset.__new__(LikelihoodDataset)
    ds.experiment = stub
    ds.device = "cpu"
    ds.evaluation = True
    ds.profile = False
    ds.global_rank = 0
    ds.particle_batch_size = 1024
    return ds


def _trace_for(stub, samples):
    """Build a pyro trace whose param sites carry exactly the supplied samples,
    so ``trace.compute_log_prob()`` returns the analytical prior log_prob at
    those samples (no random sampling).
    """
    # Use poutine.condition to pin each param site to the given column.
    cond_data = {
        name: samples[..., i] for i, name in enumerate(stub.cosmo_params)
    }
    cond_data["y"] = torch.zeros(samples.shape[:-1])  # value irrelevant here
    trace = pyro.poutine.trace(
        pyro.poutine.condition(stub.pyro_model, data=cond_data)
    ).get_trace(stub.designs)
    return trace


# ============================================================================
# Branch 1: No prior_flow.
# ============================================================================

class TestNoPriorFlow:
    """``_compute_prior_log_probs`` without a prior_flow."""

    def test_no_transform_returns_analytical_log_prob(self):
        """When transform_input=False, output equals dist.log_prob per param."""
        prior = {
            "z": dist.Gamma(torch.tensor(3.0), torch.tensor(10.0 / 3.0)),
        }
        stub = _make_stub(prior, transform_input=False)
        ds = _build_dataset(stub)

        samples = _draw_samples(stub, n=1024)
        trace = _trace_for(stub, samples)
        out = ds._compute_prior_log_probs(samples, trace)

        expected = prior["z"].log_prob(samples[..., 0])
        assert torch.allclose(out["z"], expected, atol=1e-6)

    def test_transform_change_of_variables_identity(self):
        """log p_T(T(x)) = log p_x(x) - log|dT/dx| should hold pointwise.

        This is a stronger check than the entropy test: we evaluate the bare
        algebraic identity at each sample.
        """
        prior = {
            "z": dist.Gamma(torch.tensor(3.0), torch.tensor(10.0 / 3.0)),
        }
        stub = _make_stub(prior, transform_input=True)
        ds = _build_dataset(stub)

        samples = _draw_samples(stub, n=512, seed=2)
        trace = _trace_for(stub, samples)
        out = ds._compute_prior_log_probs(samples, trace)

        log_px = prior["z"].log_prob(samples[..., 0])
        log_det = stub.param_bijector.log_abs_det_jacobian(
            samples[..., 0:1], "z",
        ).reshape(log_px.shape)
        expected = log_px - log_det
        assert torch.allclose(out["z"], expected, atol=1e-6)


# ============================================================================
# Branch 2: With prior_flow.
# ============================================================================

class _StandardNormalPriorFlow(torch.nn.Module):
    """Stub prior_flow: returns a standard-normal distribution regardless of
    context. Used to exercise the prior_flow code path with a known density.
    """

    def forward(self, context):
        # `prior_dist.log_prob(input)` then evaluates Sum_i log N(input_i; 0, 1).
        loc = torch.zeros(context.shape[0], 1)
        scale = torch.ones(context.shape[0], 1)
        return dist.Normal(loc, scale).to_event(1)


class _GammaPriorFlow(torch.nn.Module):
    """Stub prior_flow whose log_prob equals the analytical Gamma density."""

    def __init__(self, gamma_dist):
        super().__init__()
        self.gamma_dist = gamma_dist

    def forward(self, context):
        # Expand to match context batch shape.
        loc = self.gamma_dist.concentration.expand(context.shape[0])
        rate = self.gamma_dist.rate.expand(context.shape[0])
        return dist.Gamma(loc, rate).to_event(0)


class _DummyForLogProb:
    """Wrap a Gamma so .log_prob accepts an (..., 1) tensor and returns (...,)."""

    def __init__(self, gamma_dist, batch):
        self.gamma_dist = gamma_dist
        self.batch = batch

    def log_prob(self, x):
        return self.gamma_dist.log_prob(x[..., 0])


class _GammaPriorFlowFlat(torch.nn.Module):
    """Same as _GammaPriorFlow but unwraps the trailing param dim itself, so
    the returned distribution's ``log_prob`` accepts an ``(N, 1)`` tensor and
    produces an ``(N,)`` output -- matching how the prior_flow branch in
    ``_compute_prior_log_probs`` calls it.
    """

    def __init__(self, gamma_dist):
        super().__init__()
        self.gamma_dist = gamma_dist

    def forward(self, context):
        return _DummyForLogProb(self.gamma_dist, context.shape[0])


def _make_prior_flow_metadata(transform_input, bijector_state=None):
    return {
        "transform_input": transform_input,
        # 1D nominal_context -- _compute_prior_log_probs unsqueezes it to add
        # a leading batch dim and then expands to the chunk size.
        "nominal_context": torch.zeros(2),
        "bijector_state": bijector_state,
    }


class TestWithPriorFlow:
    """``_compute_prior_log_probs`` with a stub prior_flow."""

    def test_pf_false_cur_false_passthrough(self):
        """pf_T=False, cur_T=False: output == prior_flow.log_prob(theta)."""
        prior = {
            "z": dist.Gamma(torch.tensor(3.0), torch.tensor(10.0 / 3.0)),
        }
        gamma = prior["z"]
        flow = _GammaPriorFlowFlat(gamma)
        meta = _make_prior_flow_metadata(transform_input=False)
        stub = _make_stub(prior, transform_input=False,
                          prior_flow=flow, prior_flow_metadata=meta)
        ds = _build_dataset(stub)

        samples = _draw_samples(stub, n=1024, seed=3)
        trace = _trace_for(stub, samples)
        out = ds._compute_prior_log_probs(samples, trace)

        expected = gamma.log_prob(samples[..., 0])
        assert torch.allclose(out["joint"], expected, atol=1e-6)

    def test_pf_true_cur_false_recovers_theta_density(self):
        """pf_T=True, cur_T=False: output ~ log p_theta(theta).

        With a standard-normal prior_flow (so prior_flow.log_prob(z_pf)
        equals log N(0,1)(z_pf) by construction) and the change-of-variables
        + log|dT_pf/dtheta|, the result reconstructs log p_theta(theta) up
        to bijector discretization. We compare to the analytical Gamma
        log_prob in the bulk.
        """
        prior = {
            "z": dist.Gamma(torch.tensor(3.0), torch.tensor(10.0 / 3.0)),
        }
        flow = _StandardNormalPriorFlow()

        # Build an independent bijector that we'll register as the prior_flow's.
        helper = _make_stub(prior, transform_input=True)  # builds param_bijector
        pf_bijector = helper.param_bijector
        meta = _make_prior_flow_metadata(
            transform_input=True, bijector_state=pf_bijector.get_state(),
        )

        stub = _make_stub(prior, transform_input=False,
                          prior_flow=flow, prior_flow_metadata=meta)
        # Wire the prior_flow_bijector explicitly (mirrors what
        # _init_param_bijector does at experiment init time).
        stub.prior_flow_bijector = pf_bijector

        ds = _build_dataset(stub)
        samples = _draw_samples(stub, n=4096, seed=4)
        # Restrict to the bulk to avoid CDF-tail noise dominating the test.
        bulk = (samples[..., 0] > 0.05) & (samples[..., 0] < 2.5)
        samples = samples[bulk].reshape(-1, 1)

        trace = _trace_for(stub, samples)
        out = ds._compute_prior_log_probs(samples, trace)

        expected = prior["z"].log_prob(samples[..., 0])
        diff = (out["joint"] - expected)
        # Pointwise tolerance is loose because the empirical CDF is piecewise;
        # the *mean* difference however is the EIG-relevant quantity and is
        # tight.
        assert diff.mean().abs().item() < _TOL_LOOSE

    def test_pf_true_cur_true_jacobians_cancel(self):
        """pf_T=True, cur_T=True with T_pf == T_cur: output == prior_flow.log_prob(T(theta)).

        The two change-of-variables terms cancel exactly when the prior_flow's
        bijector is the same object as the experiment's, so the result reduces
        to the bare prior_flow.log_prob in T-space (the legacy behavior).
        """
        prior = {
            "z": dist.Gamma(torch.tensor(3.0), torch.tensor(10.0 / 3.0)),
        }
        flow = _StandardNormalPriorFlow()

        helper = _make_stub(prior, transform_input=True)
        shared_state = helper.param_bijector.get_state()
        meta = _make_prior_flow_metadata(
            transform_input=True, bijector_state=shared_state,
        )
        stub = _make_stub(prior, transform_input=True,
                          prior_flow=flow, prior_flow_metadata=meta)
        # Wire prior_flow_bijector to the same state -- guaranteeing T_pf == T_cur.
        stub.prior_flow_bijector = Bijector(
            stub, cdf_bins=_CDF_BINS, cdf_samples=_CDF_SAMPLES, skip_sampling=True,
        )
        stub.prior_flow_bijector.set_state(shared_state, device="cpu")
        # Replace param_bijector to share the exact tensors (not a freshly
        # sampled CDF) -- otherwise tiny CDF-noise differences leak in.
        stub.param_bijector = Bijector(
            stub, cdf_bins=_CDF_BINS, cdf_samples=_CDF_SAMPLES, skip_sampling=True,
        )
        stub.param_bijector.set_state(shared_state, device="cpu")

        ds = _build_dataset(stub)
        samples = _draw_samples(stub, n=1024, seed=5)
        trace = _trace_for(stub, samples)
        out = ds._compute_prior_log_probs(samples, trace)

        # When T_pf == T_cur, the two log|dT/dx| corrections cancel exactly.
        z_pf = stub.prior_flow_bijector.prior_to_gaussian(samples, "z").reshape(-1)
        expected = -0.5 * z_pf.pow(2) - 0.5 * math.log(2 * math.pi)
        assert torch.allclose(out["joint"], expected, atol=_TOL_TIGHT)

    def test_pf_false_cur_true_subtracts_jacobian(self):
        """pf_T=False, cur_T=True: output == log_prior_flow(theta) - log|dT_cur/dtheta|."""
        prior = {
            "z": dist.Gamma(torch.tensor(3.0), torch.tensor(10.0 / 3.0)),
        }
        gamma = prior["z"]
        flow = _GammaPriorFlowFlat(gamma)
        meta = _make_prior_flow_metadata(transform_input=False)

        stub = _make_stub(prior, transform_input=True,
                          prior_flow=flow, prior_flow_metadata=meta)
        ds = _build_dataset(stub)
        samples = _draw_samples(stub, n=1024, seed=6)
        trace = _trace_for(stub, samples)
        out = ds._compute_prior_log_probs(samples, trace)

        log_pf = gamma.log_prob(samples[..., 0])
        log_det_cur = stub.param_bijector.log_abs_det_jacobian(samples, "z").reshape(-1)
        expected = log_pf - log_det_cur
        assert torch.allclose(out["joint"], expected, atol=1e-6)

    def test_pf_true_no_bijector_state_raises(self):
        """Loading a prior_flow that declares transform_input=True without a
        bijector_state must fail loudly at experiment init.
        """
        prior = {
            "z": dist.Gamma(torch.tensor(3.0), torch.tensor(10.0 / 3.0)),
        }
        flow = _StandardNormalPriorFlow()
        meta = _make_prior_flow_metadata(transform_input=True, bijector_state=None)
        stub = _Stub(prior, transform_input=False,
                     prior_flow=flow, prior_flow_metadata=meta)
        with pytest.raises(RuntimeError, match="bijector_state"):
            stub._init_param_bijector()
