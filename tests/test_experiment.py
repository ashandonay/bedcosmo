"""Unit tests for experiment base class methods: get_prior_samples, apply_multipliers, prior flow sampling."""

import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

import pyro
from pyro import distributions as dist

from bedcosmo.base import BaseExperiment


# ============================================================================
# Concrete stub for testing BaseExperiment (which is abstract)
# ============================================================================

class StubExperiment(BaseExperiment):
    """Minimal concrete subclass of BaseExperiment for testing."""

    def __init__(self, cosmo_params, latex_labels, device="cpu", prior=None, **kwargs):
        # bypass abstract __init__
        self.name = "stub"
        self.device = device
        self.cosmo_params = list(cosmo_params)
        self.latex_labels = list(latex_labels)
        self.transform_input = False
        self.global_rank = 0
        self.profile = False

        if prior is None:
            # default: uniform priors for each param
            self.prior = {
                p: dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))
                for p in self.cosmo_params
            }
        else:
            self.prior = prior

        # Apply any extra attributes (e.g. multipliers, prior_flow)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def init_designs(self, **kwargs):
        self.designs = torch.zeros(1, len(self.cosmo_params))

    def init_prior(self, parameters, **kwargs):
        return self.prior, {}, self.latex_labels, None, None

    def pyro_model(self, design):
        raise NotImplementedError

    def sample_valid_parameters(self, sample_shape, prior=None, use_prior_flow=True, **kwargs):
        if prior is None:
            prior = self.prior
        parameters = {}
        for k, v in prior.items():
            if isinstance(v, dist.Distribution):
                parameters[k] = pyro.sample(k, v).unsqueeze(-1)
            else:
                parameters[k] = v
        return parameters


# ============================================================================
# apply_multipliers tests
# ============================================================================

class TestApplyMultipliers:
    """Tests for BaseExperiment.apply_multipliers."""

    def test_no_multipliers(self):
        """apply_multipliers is a no-op when no multipliers are set."""
        exp = StubExperiment(["Om", "hrdrag"], ["\\Omega_m", "H_0r_d"])
        samples = torch.ones(10, 2)
        result = exp.apply_multipliers(samples)
        assert torch.allclose(result, torch.ones(10, 2))

    def test_single_multiplier(self):
        """apply_multipliers scales the correct column."""
        exp = StubExperiment(
            ["Om", "hrdrag"], ["\\Omega_m", "H_0r_d"],
            hrdrag_multiplier=100.0,
        )
        samples = torch.ones(5, 2)
        exp.apply_multipliers(samples)
        assert torch.allclose(samples[:, 0], torch.ones(5))
        assert torch.allclose(samples[:, 1], torch.full((5,), 100.0))

    def test_multiple_multipliers(self):
        """apply_multipliers handles multiple multipliers correctly."""
        exp = StubExperiment(
            ["a", "b", "c"], ["A", "B", "C"],
            a_multiplier=2.0,
            c_multiplier=0.5,
        )
        samples = torch.tensor([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]])
        exp.apply_multipliers(samples)
        expected = torch.tensor([[2.0, 1.0, 0.5], [8.0, 4.0, 2.0]])
        assert torch.allclose(samples, expected)

    def test_multiplier_correct_index(self):
        """Multiplier is applied by param name, not hardcoded index."""
        exp = StubExperiment(
            ["Om", "w0", "hrdrag"], ["\\Omega_m", "w_0", "H_0r_d"],
            hrdrag_multiplier=100.0,
        )
        samples = torch.ones(3, 3)
        exp.apply_multipliers(samples)
        # hrdrag is at index 2
        assert samples[0, 0].item() == 1.0
        assert samples[0, 1].item() == 1.0
        assert samples[0, 2].item() == 100.0

    def test_batched_samples(self):
        """apply_multipliers works on higher-dimensional tensors."""
        exp = StubExperiment(
            ["x", "y"], ["X", "Y"],
            y_multiplier=10.0,
        )
        samples = torch.ones(4, 3, 2)
        exp.apply_multipliers(samples)
        assert torch.allclose(samples[..., 0], torch.ones(4, 3))
        assert torch.allclose(samples[..., 1], torch.full((4, 3), 10.0))


# ============================================================================
# get_prior_samples tests (without prior_flow)
# ============================================================================

class TestGetPriorSamplesNoPriorFlow:
    """Tests for BaseExperiment.get_prior_samples when no prior_flow is set."""

    def test_returns_mcsamples(self):
        """get_prior_samples returns a getdist.MCSamples object."""
        exp = StubExperiment(["Om", "hrdrag"], ["\\Omega_m", "H_0r_d"])
        result = exp.get_prior_samples(num_samples=100)
        import getdist
        assert isinstance(result, getdist.MCSamples)

    def test_correct_shape(self):
        """Returned samples have the expected shape."""
        exp = StubExperiment(["Om", "hrdrag"], ["\\Omega_m", "H_0r_d"])
        result = exp.get_prior_samples(num_samples=200)
        assert result.samples.shape == (200, 2)

    def test_param_names(self):
        """Returned samples have correct parameter names."""
        params = ["Om", "w0", "hrdrag"]
        exp = StubExperiment(params, ["\\Omega_m", "w_0", "H_0r_d"])
        result = exp.get_prior_samples(num_samples=50)
        names = [p.name for p in result.getParamNames().names]
        assert names == params

    def test_multiplier_applied(self):
        """Multipliers are applied to the returned samples."""
        exp = StubExperiment(
            ["Om", "hrdrag"], ["\\Omega_m", "H_0r_d"],
            prior={
                "Om": dist.Uniform(torch.tensor(0.2), torch.tensor(0.4)),
                "hrdrag": dist.Uniform(torch.tensor(0.9), torch.tensor(1.1)),
            },
            hrdrag_multiplier=100.0,
        )
        result = exp.get_prior_samples(num_samples=500)
        # hrdrag samples should be in [90, 110] after multiplier
        hrdrag_col = result.samples[:, 1]
        assert hrdrag_col.min() >= 89.0
        assert hrdrag_col.max() <= 111.0

    def test_no_multiplier_no_scaling(self):
        """Without multipliers, samples match prior range."""
        exp = StubExperiment(
            ["Om"], ["\\Omega_m"],
            prior={"Om": dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))},
        )
        result = exp.get_prior_samples(num_samples=1000)
        assert result.samples[:, 0].min() >= 0.0
        assert result.samples[:, 0].max() <= 1.0

    def test_samples_are_float64(self):
        """Returned samples should be float64."""
        exp = StubExperiment(["Om"], ["\\Omega_m"])
        result = exp.get_prior_samples(num_samples=50)
        assert result.samples.dtype == np.float64


# ============================================================================
# get_prior_samples tests (with prior_flow)
# ============================================================================

class TestGetPriorSamplesWithPriorFlow:
    """Tests for BaseExperiment.get_prior_samples when prior_flow is set."""

    def _make_exp_with_flow(self, n_params=2, cache_size=500):
        """Helper to create experiment with a mock prior flow."""
        params = [f"p{i}" for i in range(n_params)]
        labels = [f"P_{i}" for i in range(n_params)]

        # Create a mock flow that returns uniform samples
        mock_flow = Mock()
        mock_dist = Mock()
        mock_dist.sample.return_value = torch.rand(10000, n_params)
        mock_flow.return_value = mock_dist

        exp = StubExperiment(
            params, labels,
            prior_flow=mock_flow,
            prior_flow_nominal_context=torch.zeros(4),
            prior_flow_transform_input=False,
            prior_flow_batch_size=10000,
            prior_flow_cache_size=cache_size,
            prior_flow_use_cache=True,
            nominal_context=torch.zeros(4),
        )
        return exp

    def test_uses_flow_when_available(self):
        """get_prior_samples uses prior_flow path when prior_flow is set."""
        exp = self._make_exp_with_flow()
        result = exp.get_prior_samples(num_samples=100)
        import getdist
        assert isinstance(result, getdist.MCSamples)
        assert result.samples.shape == (100, 2)

    def test_multiplier_applied_with_flow(self):
        """Multipliers are applied even when using prior_flow."""
        exp = self._make_exp_with_flow(n_params=2)
        exp.p1_multiplier = 10.0
        result = exp.get_prior_samples(num_samples=100)
        # p1 is at index 1, all original samples are in [0, 1], so after * 10 -> [0, 10]
        assert result.samples[:, 1].max() <= 10.1


# ============================================================================
# _sample_prior_flow (direct, non-cached) tests
# ============================================================================

class TestSamplePriorFlowDirect:
    """Tests for BaseExperiment._sample_prior_flow (direct sampling)."""

    def _make_exp(self, n_params=2):
        params = [f"p{i}" for i in range(n_params)]
        labels = [f"P_{i}" for i in range(n_params)]

        mock_flow = Mock()
        mock_dist = Mock()
        mock_dist.sample.return_value = torch.rand(100, n_params)
        mock_flow.return_value = mock_dist

        exp = StubExperiment(
            params, labels,
            prior_flow=mock_flow,
            prior_flow_nominal_context=torch.zeros(4),
            prior_flow_transform_input=False,
            prior_flow_batch_size=100,
            nominal_context=torch.zeros(4),
        )
        return exp

    def test_returns_correct_shape(self):
        """_sample_prior_flow returns (total_samples, n_params) tensor."""
        exp = self._make_exp(n_params=3)
        exp.prior_flow.return_value.sample.return_value = torch.rand(50, 3)
        result = exp._sample_prior_flow(50)
        assert result.shape == (50, 3)

    def test_raises_without_flow(self):
        """_sample_prior_flow raises RuntimeError if prior_flow is not set."""
        exp = StubExperiment(["Om"], ["\\Omega_m"])
        with pytest.raises(RuntimeError, match="prior_flow not set"):
            exp._sample_prior_flow(10)

    def test_raises_without_context(self):
        """_sample_prior_flow raises RuntimeError if no nominal context."""
        exp = StubExperiment(["Om"], ["\\Omega_m"], prior_flow=Mock())
        with pytest.raises(RuntimeError, match="No nominal context"):
            exp._sample_prior_flow(10)

    def test_batched_sampling(self):
        """_sample_prior_flow batches correctly when total > batch_size."""
        exp = self._make_exp(n_params=2)
        exp.prior_flow_batch_size = 30
        # Each call to flow returns 30 samples (or fewer for last batch)
        def make_samples(ctx):
            n = ctx.shape[0]
            mock_d = Mock()
            mock_d.sample.return_value = torch.rand(n, 2)
            return mock_d
        exp.prior_flow.side_effect = make_samples

        result = exp._sample_prior_flow(100)
        assert result.shape == (100, 2)


# ============================================================================
# _sample_prior_flow_cache tests
# ============================================================================

class TestSamplePriorFlowCache:
    """Tests for BaseExperiment._sample_prior_flow_cache (cached sampling)."""

    def _make_exp(self, n_params=2, cache_size=200):
        params = [f"p{i}" for i in range(n_params)]
        labels = [f"P_{i}" for i in range(n_params)]

        mock_flow = Mock()
        mock_dist = Mock()
        mock_dist.sample.return_value = torch.rand(cache_size, n_params)
        mock_flow.return_value = mock_dist

        exp = StubExperiment(
            params, labels,
            prior_flow=mock_flow,
            prior_flow_nominal_context=torch.zeros(4),
            prior_flow_transform_input=False,
            prior_flow_batch_size=cache_size,
            prior_flow_cache_size=cache_size,
            prior_flow_use_cache=True,
            nominal_context=torch.zeros(4),
        )
        return exp

    def test_returns_correct_shape_int(self):
        """_sample_prior_flow_cache with int returns (n, n_params)."""
        exp = self._make_exp()
        result = exp._sample_prior_flow_cache(50)
        assert result.shape == (50, 2)

    def test_returns_correct_shape_tuple(self):
        """_sample_prior_flow_cache with tuple returns (*shape, n_params)."""
        exp = self._make_exp()
        result = exp._sample_prior_flow_cache((10, 5))
        assert result.shape == (10, 5, 2)

    def test_cache_initialized_on_first_call(self):
        """Cache is created on first call."""
        exp = self._make_exp(cache_size=100)
        assert not hasattr(exp, "_prior_flow_cache") or exp._prior_flow_cache is None
        exp._sample_prior_flow_cache(10)
        assert exp._prior_flow_cache is not None
        assert exp._prior_flow_cache.shape[0] == 100

    def test_cache_index_advances(self):
        """Cache index advances after each draw."""
        exp = self._make_exp(cache_size=100)
        exp._sample_prior_flow_cache(10)
        assert exp._prior_flow_cache_idx == 10
        exp._sample_prior_flow_cache(20)
        assert exp._prior_flow_cache_idx == 30

    def test_cache_wraps_around(self):
        """Cache shuffles and resets index when exhausted."""
        exp = self._make_exp(cache_size=50)
        # Draw enough to exceed cache
        exp._sample_prior_flow_cache(30)
        assert exp._prior_flow_cache_idx == 30
        # This exceeds remaining 20, so cache should wrap
        exp._sample_prior_flow_cache(30)
        assert exp._prior_flow_cache_idx == 30  # reset to 0 then advanced 30

    def test_no_cache_mode(self):
        """When prior_flow_use_cache=False, samples directly."""
        exp = self._make_exp(cache_size=100)
        exp.prior_flow_use_cache = False
        # Need to make flow return correct size for direct sampling
        def make_samples(ctx):
            n = ctx.shape[0]
            mock_d = Mock()
            mock_d.sample.return_value = torch.rand(n, 2)
            return mock_d
        exp.prior_flow.side_effect = make_samples
        result = exp._sample_prior_flow_cache(25)
        assert result.shape == (25, 2)
        # Cache should not be initialized
        assert not hasattr(exp, "_prior_flow_cache") or exp._prior_flow_cache is None

    def test_clear_cache(self):
        """clear_prior_flow_cache resets state."""
        exp = self._make_exp(cache_size=100)
        exp._sample_prior_flow_cache(10)
        assert exp._prior_flow_cache is not None
        exp.clear_prior_flow_cache()
        assert exp._prior_flow_cache is None
        assert exp._prior_flow_cache_idx == 0
