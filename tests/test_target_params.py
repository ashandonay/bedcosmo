"""Tests for BaseExperiment._init_target_params (focused-target guide config).

The method is shared by all experiments (num_tracers, variable_redshift,
num_visits). These tests exercise its branching in isolation via a minimal
BaseExperiment subclass, plus verify the three experiments inherit it and
accept the ``target_params`` kwarg.
"""

from __future__ import annotations

import inspect

import pytest

from bedcosmo.base import BaseExperiment


class _StubExperiment(BaseExperiment):
    """Minimal concrete BaseExperiment for pure-logic testing of target params."""

    def __init__(self, cosmo_params, transform_input=False):
        self.cosmo_params = list(cosmo_params)
        self.transform_input = transform_input

    # Abstract methods are stubbed out below via __abstractmethods__.
    def init_designs(self, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def init_prior(self, parameters, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def pyro_model(self, design):  # pragma: no cover
        raise NotImplementedError

    def sample_parameters(self, sample_shape, prior=None, **kwargs):  # pragma: no cover
        raise NotImplementedError


CP = ["f1", "f2", "log_c_scale", "z"]


@pytest.mark.parametrize("target", [None, [], list(CP), ["z", "f1", "f2", "log_c_scale"]])
def test_none_empty_or_full_resolves_to_all_params(target):
    exp = _StubExperiment(CP)
    exp._init_target_params(target)
    assert exp.target_params == CP
    assert exp.target_indices == [0, 1, 2, 3]
    assert exp.n_targets == 4


def test_single_target_subset():
    exp = _StubExperiment(CP)
    exp._init_target_params(["z"])
    assert exp.target_params == ["z"]
    assert exp.target_indices == [3]
    assert exp.n_targets == 1


def test_subset_preserves_requested_order():
    exp = _StubExperiment(CP)
    exp._init_target_params(["z", "f2"])
    assert exp.target_params == ["z", "f2"]
    assert exp.target_indices == [3, 1]
    assert exp.n_targets == 2


def test_unknown_param_raises():
    exp = _StubExperiment(CP)
    with pytest.raises(ValueError, match="not in cosmo_params"):
        exp._init_target_params(["q0"])


def test_subset_with_transform_input_raises():
    exp = _StubExperiment(CP, transform_input=True)
    with pytest.raises(ValueError, match="transform_input=False"):
        exp._init_target_params(["z"])


def test_full_set_with_transform_input_ok():
    # transform_input is only incompatible with a strict subset, not the full set.
    exp = _StubExperiment(CP, transform_input=True)
    exp._init_target_params(None)
    assert exp.n_targets == 4


def test_experiments_inherit_and_accept_kwarg():
    from bedcosmo.num_tracers import NumTracers
    from bedcosmo.num_visits import NumVisits
    from bedcosmo.variable_redshift import VariableRedshift

    for cls in (NumTracers, VariableRedshift, NumVisits):
        assert hasattr(cls, "_init_target_params")
        # inherited from BaseExperiment, not redefined on the subclass
        assert "_init_target_params" not in vars(cls)
        assert "target_params" in inspect.signature(cls.__init__).parameters


# Ensure the ABC abstractmethods don't block instantiation of the stub.
_StubExperiment.__abstractmethods__ = frozenset()
