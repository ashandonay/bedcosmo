"""Tests for default-eval: sample → save_posterior_samples → plot_posterior_display."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from bedcosmo.artifacts import load_posterior_samples_file, save_posterior_samples
from bedcosmo.evaluate import Evaluator
from bedcosmo.plotting import BasePlotter, RunPlotter


def _make_mcsamples(theta: np.ndarray, names: list[str]):
    return SimpleNamespace(
        samples=theta,
        paramNames=SimpleNamespace(
            list=lambda: list(names),
            names=[SimpleNamespace(name=n) for n in names],
        ),
    )


def test_plot_posterior_display_does_not_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    theta = np.random.default_rng(0).normal(size=(10, 2))
    entries = [
        {
            "name": "nominal",
            "samples": _make_mcsamples(theta, ["p0", "p1"]),
            "label": "Nominal",
            "color": "tab:blue",
            "line_style": "-",
            "alpha": 1.0,
        }
    ]
    experiment = SimpleNamespace(
        cosmo_params=["p0", "p1"],
        latex_labels=["p0", "p1"],
        device="cpu",
        central_params=None,
        get_guide_samples=MagicMock(side_effect=AssertionError("should not sample")),
    )
    fake_g = MagicMock()
    fake_g.fig.legends = []
    fake_g.subplots = [[MagicMock()]]
    fake_g.param_names_for_root.return_value = SimpleNamespace(
        names=[SimpleNamespace(name="p0"), SimpleNamespace(name="p1")]
    )
    plotter = BasePlotter(cosmo_exp="num_visits")
    with patch.object(plotter, "plot_posterior", return_value=fake_g), \
         patch.object(plotter, "save_figure"), \
         patch("bedcosmo.plotting.Line2D"), \
         patch.object(plotter, "_nf_display_samples") as mock_nf:
        plotter.plot_posterior_display(experiment, entries, guide_samples=10)
    mock_nf.assert_not_called()
    experiment.get_guide_samples.assert_not_called()


def test_sample_nf_posterior_lives_on_evaluator_not_plotter():
    assert hasattr(Evaluator, "sample_nf_posterior")
    assert not hasattr(RunPlotter, "sample_nf_posterior")


def test_run_sample_save_then_plot(tmp_path):
    """Evaluator.run samples on Evaluator, saves, then plots via plotter."""
    n_guide, n_params, n_obs = 20, 2, 3
    rng = np.random.default_rng(1)
    theta_nom = rng.normal(size=(n_guide, n_params))
    theta_opt = rng.normal(size=(n_guide, n_params))
    central = torch.zeros(n_obs, dtype=torch.float64)
    design_nom = np.ones(2)
    design_opt = np.array([1.0, 1.0])

    nf_entries = [
        {
            "name": "nominal",
            "color": "tab:blue",
            "design": design_nom,
            "samples": _make_mcsamples(theta_nom, ["p0", "p1"]),
            "label": "n",
            "line_style": "-",
            "alpha": 1.0,
        },
        {
            "name": "optimal",
            "color": "tab:orange",
            "design": design_opt,
            "samples": _make_mcsamples(theta_opt, ["p0", "p1"]),
            "label": "o",
            "line_style": "-",
            "alpha": 1.0,
        },
    ]
    experiment = SimpleNamespace(
        central_val=central,
        nominal_design=torch.zeros(3, dtype=torch.float64),
    )
    data = {
        "experiment": experiment,
        "eig_values": np.array([0.1, 0.9, 0.2]),
        "title": "Posterior Evaluation",
        "nominal_grid_eig": None,
        "nominal_prior_entropy": None,
    }

    ev = Evaluator.__new__(Evaluator)
    ev.save_path = str(tmp_path / "artifacts")
    ev.run_id = "run123"
    ev.seed = 7
    ev.guide_samples = n_guide
    ev.nf_transform_output = True
    ev.param_space = "physical"
    ev.cosmo_exp = "num_visits"
    ev.device = "cpu"
    ev.global_rank = 0
    ev.eig_file_path = None
    ev.output_path = None
    ev.profile = False
    ev.total_steps = 100
    ev.levels = [0.68]
    ev.plot_prior = False
    ev.sort = True
    ev.include_nominal = False
    ev.step_diagnostics = False
    ev.marginal_eig_subsets = []
    ev.other_eig_data = None
    ev.timestamp = None
    ev.eig_data = {}
    ev.input_designs = torch.randn(2, 3, dtype=torch.float64)
    ev.experiment = experiment
    ev.plotter = MagicMock()
    ev.sample_nf_posterior = MagicMock(return_value=(nf_entries, data))
    ev.get_eig = MagicMock(side_effect=[(0.5, 0.01), (np.array([0.2, 0.8]), np.zeros(2))])
    ev._update_runtime = MagicMock()
    ev._eig_data_save_path = MagicMock(return_value=str(tmp_path / "eig.json"))
    ev._target_prior_entropy = MagicMock(return_value=None)

    with patch("bedcosmo.evaluate.render_overlay"), \
         patch("bedcosmo.evaluate.save_posterior_samples", wraps=save_posterior_samples) as save_spy:
        ev.run(eval_step=100)

    ev.sample_nf_posterior.assert_called_once()
    assert not hasattr(ev.plotter, "sample_nf_posterior") or not ev.plotter.sample_nf_posterior.called
    assert save_spy.called
    ev.plotter.plot_posterior_display.assert_called_once()
    plotted_entries = ev.plotter.plot_posterior_display.call_args.args[0]
    assert plotted_entries is nf_entries

    bundle = load_posterior_samples_file(ev.save_path, step=100)
    assert list(bundle["series_names"]) == ["optimal", "nominal"]
    assert bundle["meta"]["generated_by"] == "Evaluator.run"
    assert bundle["meta"]["num_data_samples"] == 1
    assert bundle["theta"].shape == (2, 1, n_guide, n_params)
    np.testing.assert_allclose(bundle["theta"][0, 0], theta_opt)
    np.testing.assert_allclose(bundle["theta"][1, 0], theta_nom)
