"""Tests for default-eval central-context posterior npz persistence."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from bedcosmo.artifacts import load_posterior_samples, load_posterior_samples_file
from bedcosmo.evaluate import Evaluator


def _make_mcsamples(theta: np.ndarray, names: list[str]):
    """Minimal stand-in for getdist.MCSamples used by the save helper."""
    return SimpleNamespace(
        samples=theta,
        paramNames=SimpleNamespace(list=lambda: list(names)),
    )


def _bare_evaluator(tmp_path, *, n_designs=3, n_obs=4, n_params=2, guide_samples=50):
    """Build an Evaluator-like object without running ``__init__``."""
    ev = Evaluator.__new__(Evaluator)
    ev.save_path = str(tmp_path / "artifacts")
    ev.run_id = "run123"
    ev.run_obj = MagicMock()
    ev.run_args = {"total_steps": 100}
    ev.seed = 7
    ev.guide_samples = guide_samples
    ev.nf_transform_output = True
    ev.param_space = "physical"
    ev.cosmo_exp = "num_visits"
    ev.device = "cpu"
    ev.global_rank = 0
    ev.eig_file_path = None
    ev.output_path = None
    ev.profile = False
    ev.eig_data = {
        "step_100": {
            "variable": {
                "eigs_avg": [0.1, 0.9, 0.2],
            }
        }
    }
    ev.input_designs = torch.randn(n_designs, 3, dtype=torch.float64)
    central = torch.randn(n_obs, dtype=torch.float64)
    nominal_design = torch.randn(3, dtype=torch.float64)
    ev.experiment = SimpleNamespace(
        central_val=central,
        nominal_design=nominal_design,
        nominal_context=torch.cat([nominal_design, central]),
        cosmo_params=[f"p{i}" for i in range(n_params)],
        get_guide_samples=MagicMock(),
    )
    return ev


def test_save_run_posterior_samples_writes_central_bundle(tmp_path):
    n_params = 2
    n_guide = 40
    n_obs = 4
    param_names = ["p0", "p1"]
    ev = _bare_evaluator(tmp_path, n_obs=n_obs, n_params=n_params, guide_samples=n_guide)

    rng = np.random.default_rng(0)
    theta_opt = rng.normal(size=(n_guide, n_params))
    theta_nom = rng.normal(size=(n_guide, n_params))
    ev.experiment.get_guide_samples.side_effect = [
        _make_mcsamples(theta_opt, param_names),
        _make_mcsamples(theta_nom, param_names),
    ]

    fake_flow = MagicMock()
    with patch("bedcosmo.evaluate.load_model", return_value=(fake_flow, 100)):
        out_path = ev._save_run_posterior_samples(100)

    assert out_path is not None
    bundle = load_posterior_samples(out_path)
    assert bundle["meta"]["generated_by"] == "Evaluator.run"
    assert bundle["meta"]["num_data_samples"] == 1
    assert bundle["meta"]["conditioning"] == "central_val"
    assert bundle["meta"]["central"] is True
    assert bundle["meta"]["step"] == 100
    assert bundle["meta"]["optimal_design_index"] == 1  # argmax of [0.1, 0.9, 0.2]
    assert list(bundle["series_names"]) == ["optimal", "nominal"]
    assert list(bundle["param_names"]) == param_names
    assert bundle["theta"].shape == (2, 1, n_guide, n_params)
    assert bundle["y"].shape == (2, 1, n_obs)
    assert bundle["design"].shape == (2, 3)
    np.testing.assert_allclose(bundle["theta"][0, 0], theta_opt)
    np.testing.assert_allclose(bundle["theta"][1, 0], theta_nom)
    np.testing.assert_allclose(
        bundle["y"][0, 0], ev.experiment.central_val.numpy().reshape(-1)
    )
    np.testing.assert_allclose(bundle["y"][0], bundle["y"][1])

    # Discoverable via the same load helper used by --reuse-posterior-samples
    discovered = load_posterior_samples_file(ev.save_path, step=100)
    assert discovered["path"] == out_path
    assert discovered["meta"]["generated_by"] == "Evaluator.run"

    assert ev.experiment.get_guide_samples.call_count == 2


def test_resolve_optimal_design_single_design(tmp_path):
    ev = _bare_evaluator(tmp_path, n_designs=1)
    ev.eig_data = {"step_5": {"variable": {"eigs_avg": [1.23]}}}
    design, idx = ev._resolve_optimal_design(5)
    assert idx == 0
    torch.testing.assert_close(design, ev.input_designs[0])


def test_run_calls_save_before_generate_posterior(tmp_path):
    """``run`` should attempt to persist plot samples even if plotting fails later."""
    ev = _bare_evaluator(tmp_path)
    ev.total_steps = 100
    ev.levels = [0.68]
    ev.plot_prior = False
    ev.sort = True
    ev.include_nominal = False
    ev.step_diagnostics = False
    ev.marginal_eig_subsets = []
    ev.other_eig_data = None
    ev.cosmo_exp = "num_visits"
    ev.timestamp = None
    ev.plotter = MagicMock()
    ev.plotter.generate_posterior.side_effect = RuntimeError("plot boom")

    ev.get_eig = MagicMock(side_effect=[(0.5, 0.01), (np.array([0.1, 0.9, 0.2]), np.zeros(3))])
    ev._update_runtime = MagicMock()
    ev._eig_data_save_path = MagicMock(return_value=str(tmp_path / "eig.json"))
    ev._target_prior_entropy = MagicMock(return_value=None)
    ev._save_run_posterior_samples = MagicMock(return_value="/fake.npz")

    with patch("bedcosmo.evaluate.render_overlay"):
        ev.run(eval_step=100)

    ev._save_run_posterior_samples.assert_called_once_with(100)
    ev.plotter.generate_posterior.assert_called()
