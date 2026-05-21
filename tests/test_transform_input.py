"""Tests for the ``transform_input`` parameter-space transformation.

Covers two layers:

1. ``Bijector._bin_bracket`` (util.py): the rule for picking CDF-bin support
   per prior distribution type.
2. ``BaseExperiment.params_to_unconstrained`` / ``params_from_unconstrained``
   (base.py): the generic per-parameter transform built on top of the bijector.

Round-trip tests build a Bijector once per class (CDF construction is the
expensive step at ~1s per 30k samples), then share it across methods.
"""

import numpy as np
import pytest
import pyro
import torch
from pyro import distributions as dist

from bedcosmo.base import BaseExperiment
from bedcosmo.util import Bijector


# Bijector resolution settings used across round-trip tests. 30k samples + 1k
# bins is enough to keep round-trip error under ~2% in the bulk while keeping
# class setup under ~1s.
_CDF_SAMPLES = 30_000
_CDF_BINS = 1_000


# ============================================================================
# Test stub: minimal concrete subclass of BaseExperiment.
# ============================================================================

class _Stub(BaseExperiment):
    def __init__(self, prior, transform_input=False):
        self.name = "stub_transform"
        self.device = "cpu"
        self.prior = prior
        self.cosmo_params = list(prior.keys())
        self.latex_labels = list(prior.keys())
        self.transform_input = transform_input
        self.global_rank = 0
        self.profile = False

    # Required abstract methods (unused in these tests).
    def init_designs(self, **kwargs):
        self.designs = torch.zeros(1, len(self.cosmo_params))

    def init_prior(self, parameters, **kwargs):
        return self.prior, {}, self.latex_labels, None, None

    def pyro_model(self, design):
        raise NotImplementedError

    def sample_parameters(self, sample_shape, prior=None, use_prior_flow=True, **kwargs):
        if prior is None:
            prior = self.prior
        parameters = {}
        for k, v in prior.items():
            parameters[k] = pyro.sample(k, v).unsqueeze(-1)
        return parameters


def _make_stub_with_bijector(prior):
    stub = _Stub(prior, transform_input=True)
    stub.param_bijector = Bijector(
        stub, cdf_bins=_CDF_BINS, cdf_samples=_CDF_SAMPLES, use_prior_flow=False,
    )
    return stub


# ============================================================================
# Bijector._bin_bracket -- pure unit tests, no CDF construction.
# ============================================================================

class TestBinBracket:
    """Static-method bracket selection for each supported prior type."""

    def test_uniform_returns_exact_support(self):
        d = dist.Uniform(torch.tensor(-2.5), torch.tensor(7.0))
        low, high = Bijector._bin_bracket(d, "x")
        assert torch.isclose(low, torch.tensor(-2.5))
        assert torch.isclose(high, torch.tensor(7.0))

    def test_gamma_lower_is_zero(self):
        d = dist.Gamma(torch.tensor(2.0), torch.tensor(2.0))
        low, _ = Bijector._bin_bracket(d, "z")
        assert torch.isclose(low, torch.tensor(0.0))

    def test_gamma_upper_is_mean_plus_n_std(self):
        shape = torch.tensor(3.0)
        rate = torch.tensor(2.0)  # z_0 = 0.5
        d = dist.Gamma(shape, rate)
        n_std = 12.0
        _, high = Bijector._bin_bracket(d, "z", n_std=n_std)
        expected_mean = shape / rate
        expected_std = torch.sqrt(shape) / rate
        assert torch.isclose(high, expected_mean + n_std * expected_std)

    def test_normal_is_symmetric_around_mean(self):
        loc = torch.tensor(1.5)
        scale = torch.tensor(0.4)
        d = dist.Normal(loc, scale)
        n_std = 8.0
        low, high = Bijector._bin_bracket(d, "p", n_std=n_std)
        assert torch.isclose(low, loc - n_std * scale)
        assert torch.isclose(high, loc + n_std * scale)

    def test_n_std_changes_width(self):
        d = dist.Gamma(torch.tensor(2.0), torch.tensor(1.0))
        _, h_small = Bijector._bin_bracket(d, "z", n_std=3.0)
        _, h_large = Bijector._bin_bracket(d, "z", n_std=20.0)
        assert h_large > h_small

    def test_unsupported_raises(self):
        d = dist.Beta(torch.tensor(2.0), torch.tensor(5.0))
        with pytest.raises(NotImplementedError):
            Bijector._bin_bracket(d, "x")


# ============================================================================
# Identity short-circuit: transform_input=False -> methods are no-ops.
# ============================================================================

class TestTransformDisabled:
    """When transform_input is False, both methods return the input unchanged."""

    def test_to_unconstrained_is_identity(self):
        stub = _Stub({"a": dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))})
        x = torch.rand(7, 1)
        assert torch.equal(stub.params_to_unconstrained(x), x)

    def test_from_unconstrained_is_identity(self):
        stub = _Stub({"a": dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))})
        y = torch.randn(7, 1)
        assert torch.equal(stub.params_from_unconstrained(y), y)

    def test_no_bijector_required_when_disabled(self):
        stub = _Stub({"a": dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))})
        assert not hasattr(stub, "param_bijector")
        # Must not raise (would AttributeError if the short-circuit broke).
        stub.params_to_unconstrained(torch.rand(3, 1))
        stub.params_from_unconstrained(torch.randn(3, 1))


# ============================================================================
# Round-trip + shape behavior with transform_input=True.
#
# Bijector is built once per class via setup_class to keep total runtime low.
# ============================================================================

class TestTransformUniform:
    """Single uniform parameter (D=1) -- regression coverage for shape bug."""

    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)
        cls.stub = _make_stub_with_bijector(
            {"z": dist.Uniform(torch.tensor(0.0), torch.tensor(5.0))}
        )

    def test_round_trip_d1(self):
        """Regression: D=1 used to fail because scalar indexing dropped the trailing dim."""
        x = torch.linspace(0.1, 4.9, 100, dtype=torch.float64).unsqueeze(-1)
        y = self.stub.params_to_unconstrained(x)
        x_back = self.stub.params_from_unconstrained(y)
        assert x.shape == y.shape == x_back.shape == (100, 1)
        assert torch.allclose(x, x_back, atol=2e-2)

    def test_forward_is_approximately_standard_normal(self):
        """Uniform samples through the transform should be ~ N(0, 1)."""
        with pyro.plate("p", 20_000):
            samples = pyro.sample(
                "z", dist.Uniform(torch.tensor(0.0), torch.tensor(5.0))
            )
        x = samples.unsqueeze(-1).to(torch.float64)
        y = self.stub.params_to_unconstrained(x)
        # Drop the extreme tails for stability since the empirical CDF is finite-resolution.
        y_inner = y[(y > -3) & (y < 3)]
        assert abs(y_inner.mean().item()) < 0.1
        assert abs(y_inner.std().item() - 1.0) < 0.15

    def test_batched_shape(self):
        """Round-trip works for >2D inputs with arbitrary leading batch dims."""
        x = torch.rand(4, 16, 1, dtype=torch.float64) * 5.0
        y = self.stub.params_to_unconstrained(x)
        x_back = self.stub.params_from_unconstrained(y)
        assert y.shape == x.shape == x_back.shape == (4, 16, 1)
        assert torch.allclose(x, x_back, atol=2e-2)


class TestTransformGamma:
    """Gamma prior -- exercises the (0, mean + n_std*std) bracket path."""

    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)
        # shape=2, rate=2 -> mean=1, std=sqrt(0.5) ~ 0.707; bracket high ~ 9.49.
        cls.stub = _make_stub_with_bijector(
            {"z": dist.Gamma(torch.tensor(2.0), torch.tensor(2.0))}
        )

    def test_round_trip_in_bulk(self):
        """Round-trip is faithful for samples in the bulk of the gamma support.

        Restrict to z in [0.05, 2.5] where the empirical CDF is strictly
        increasing -- in the upper tail (z > ~4) the empirical CDF saturates
        at 1.0 and the inverse interpolation can divide by zero on flat bins.
        """
        with pyro.plate("p", 5_000):
            samples = pyro.sample(
                "z", dist.Gamma(torch.tensor(2.0), torch.tensor(2.0))
            )
        bulk_mask = (samples > 0.05) & (samples < 2.5)
        x = samples[bulk_mask].unsqueeze(-1).to(torch.float64)
        y = self.stub.params_to_unconstrained(x)
        x_back = self.stub.params_from_unconstrained(y)
        # Coarse tolerance to absorb finite-bin discretization (1k bins over ~9.5).
        assert torch.allclose(x, x_back, atol=5e-2)

    def test_bracket_covers_typical_samples(self):
        """Bracket should be wide enough that essentially no samples are clamped."""
        bracket_high = self.stub.param_bijector.cdfs["z"]["bins"][-1].item()
        with pyro.plate("p", 10_000):
            samples = pyro.sample(
                "z", dist.Gamma(torch.tensor(2.0), torch.tensor(2.0))
            )
        frac_inside = (samples < bracket_high).float().mean().item()
        assert frac_inside > 0.999


class TestTransformMultiParam:
    """Multiple cosmo params with mixed prior types are transformed independently."""

    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)
        cls.stub = _make_stub_with_bijector(
            {
                "Om": dist.Uniform(torch.tensor(0.1), torch.tensor(0.9)),
                "z":  dist.Gamma(torch.tensor(2.0), torch.tensor(2.0)),
            }
        )
        cls.stub_swapped = _make_stub_with_bijector(
            {
                "z":  dist.Gamma(torch.tensor(2.0), torch.tensor(2.0)),
                "Om": dist.Uniform(torch.tensor(0.1), torch.tensor(0.9)),
            }
        )

    def test_each_column_uses_its_own_bijector_key(self):
        """Permuting cosmo_params order permutes output columns correspondingly."""
        x = torch.tensor([[0.5, 1.0], [0.3, 0.5]], dtype=torch.float64)
        y = self.stub.params_to_unconstrained(x)
        x_swapped = x[:, [1, 0]]
        y_swapped = self.stub_swapped.params_to_unconstrained(x_swapped)
        # Column 0 of y is Om; column 1 of y_swapped is also Om.
        assert torch.allclose(y[:, 0], y_swapped[:, 1], atol=5e-2)
        assert torch.allclose(y[:, 1], y_swapped[:, 0], atol=5e-2)

    def test_round_trip(self):
        x = torch.tensor(
            [[0.30, 0.5], [0.50, 1.2], [0.75, 2.0]], dtype=torch.float64
        )
        y = self.stub.params_to_unconstrained(x)
        x_back = self.stub.params_from_unconstrained(y)
        assert x.shape == y.shape == x_back.shape == (3, 2)
        assert torch.allclose(x, x_back, atol=5e-2)


# ============================================================================
# Default-bijector resolution: methods fall back to self.param_bijector.
# ============================================================================

class TestDefaultBijectorResolution:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)
        cls.stub = _make_stub_with_bijector(
            {"a": dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))}
        )

    def test_uses_param_bijector_attribute(self):
        x = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
        y_default = self.stub.params_to_unconstrained(x)
        y_explicit = self.stub.params_to_unconstrained(x, bijector_class=self.stub.param_bijector)
        assert torch.allclose(y_default, y_explicit)

    def test_missing_bijector_raises(self):
        """transform_input=True without a param_bijector attribute is a programmer error."""
        stub = _Stub(
            {"a": dist.Uniform(torch.tensor(0.0), torch.tensor(1.0))},
            transform_input=True,
        )
        x = torch.tensor([[0.5]], dtype=torch.float64)
        with pytest.raises(AttributeError):
            stub.params_to_unconstrained(x)
