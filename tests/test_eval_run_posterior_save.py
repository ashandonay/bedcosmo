"""Tests for default-eval: sample_nf → save_posterior_samples → plot_posterior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from bedcosmo.artifacts import load_posterior_samples_file, save_posterior_samples
from bedcosmo.evaluate import Evaluator
from bedcosmo.plotting import BasePlotter, RunPlotter
from bedcosmo.util import sample_nf


def _make_mcsamples(theta: np.ndarray, names: list[str]):
    return SimpleNamespace(
        samples=theta,
        paramNames=SimpleNamespace(
            list=lambda: list(names),
            names=[SimpleNamespace(name=n) for n in names],
        ),
    )


def test_sample_nf_conditions_on_design_and_y():
    """sample_nf takes design + y only; returns MCSamples (no display labels)."""
    n_guide, n_params = 15, 2
    rng = np.random.default_rng(2)
    theta = rng.normal(size=(n_guide, n_params))
    design = torch.tensor([1.0, 1.0], dtype=torch.float64)
    y = torch.zeros(3, dtype=torch.float64)
    experiment = SimpleNamespace(
        device="cpu",
        get_guide_samples=MagicMock(return_value=_make_mcsamples(theta, ["p0", "p1"])),
    )
    samples = sample_nf(
        experiment,
        MagicMock(name="flow"),
        design,
        y,
        num_samples=n_guide,
        device="cpu",
    )
    np.testing.assert_allclose(samples.samples, theta)
    experiment.get_guide_samples.assert_called_once()
    args, kwargs = experiment.get_guide_samples.call_args
    context = args[1]
    np.testing.assert_allclose(
        context.detach().cpu().numpy(),
        np.concatenate([design.numpy(), y.numpy()]),
    )
    assert kwargs["num_samples"] == n_guide


def test_plot_posterior_does_not_sample(tmp_path, monkeypatch):
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
    with patch.object(plotter, "plot_triangle", return_value=fake_g), \
         patch.object(plotter, "save_figure"), \
         patch("bedcosmo.plotting.Line2D"), \
         patch.object(plotter, "_sample_nf_entries") as mock_nf:
        plotter.plot_posterior(experiment, entries, guide_samples=10)
    mock_nf.assert_not_called()
    experiment.get_guide_samples.assert_not_called()


def test_sample_nf_not_on_evaluator_or_plotter():
    assert not hasattr(Evaluator, "sample_nf")
    assert not hasattr(RunPlotter, "sample_nf")


def test_sample_nf_entries_selects_then_calls_sample_nf():
    """_sample_nf_entries owns nominal/optimal selection; sample_nf only samples."""
    n_guide, n_params = 12, 2
    rng = np.random.default_rng(3)
    theta_nom = rng.normal(size=(n_guide, n_params))
    theta_opt = rng.normal(size=(n_guide, n_params))
    central = torch.zeros(3, dtype=torch.float64)
    nominal_design = torch.ones(2, dtype=torch.float64)
    experiment = SimpleNamespace(
        central_val=central,
        nominal_design=nominal_design,
        device="cpu",
    )
    plotter = BasePlotter(cosmo_exp="num_visits")
    input_designs = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    with patch(
        "bedcosmo.plotting.sample_nf",
        side_effect=[
            _make_mcsamples(theta_nom, ["p0", "p1"]),
            _make_mcsamples(theta_opt, ["p0", "p1"]),
        ],
    ) as mock_sample:
        entries, _ = plotter._sample_nf_entries(
            ("nominal", "optimal"),
            n_guide,
            experiment=experiment,
            posterior_flow=MagicMock(),
            input_designs=input_designs,
            eig_values=np.array([0.1, 0.9, 0.2]),
            device="cpu",
        )
    assert mock_sample.call_count == 2
    nom_call, opt_call = mock_sample.call_args_list
    np.testing.assert_allclose(
        nom_call.args[2].detach().cpu().numpy(), nominal_design.numpy()
    )
    np.testing.assert_allclose(nom_call.args[3].detach().cpu().numpy(), central.numpy())
    np.testing.assert_allclose(opt_call.args[2], input_designs[1])
    np.testing.assert_allclose(opt_call.args[3].detach().cpu().numpy(), central.numpy())
    assert [e["name"] for e in entries] == ["nominal", "optimal"]
    assert entries[0]["color"] == "tab:blue"
    assert entries[1]["color"] == "tab:orange"


def test_run_sample_save_then_plot(tmp_path):
    """Evaluator.run samples via _sample_nf_entries, saves, then plots."""
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
    extract_data = {
        "experiment": experiment,
        "posterior_flow": MagicMock(),
        "input_designs": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "eig_values": np.array([0.1, 0.9]),
        "nominal_eig": 0.5,
        "nominal_prior_entropy": None,
        "nominal_posterior_entropy": None,
        "prior_entropy_by_design": None,
        "posterior_entropy_by_design": None,
        "title": "Posterior Evaluation",
        "nominal_grid_eig": None,
        "marginal_eig": False,
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
    ev.plotter._extract_run_posterior_data.return_value = dict(extract_data)
    ev.plotter._sample_nf_entries.return_value = (nf_entries, None)
    ev.get_eig = MagicMock(side_effect=[(0.5, 0.01), (np.array([0.2, 0.8]), np.zeros(2))])
    ev._update_runtime = MagicMock()
    ev._eig_data_save_path = MagicMock(return_value=str(tmp_path / "eig.json"))
    ev._target_prior_entropy = MagicMock(return_value=None)

    with patch("bedcosmo.evaluate.render_overlay"), \
         patch("bedcosmo.evaluate.save_posterior_samples", wraps=save_posterior_samples) as save_spy:
        ev.run(eval_step=100)

    ev.plotter._sample_nf_entries.assert_called_once()
    assert save_spy.called
    ev.plotter.plot_posterior.assert_called_once()
    plotted_entries = ev.plotter.plot_posterior.call_args.args[0]
    assert plotted_entries is nf_entries

    bundle = load_posterior_samples_file(ev.save_path, step=100)
    assert list(bundle["series_names"]) == ["optimal", "nominal"]
    assert bundle["meta"]["generated_by"] == "Evaluator.run"
    assert bundle["meta"]["num_data_samples"] == 1
    assert bundle["theta"].shape == (2, 1, n_guide, n_params)
    np.testing.assert_allclose(bundle["theta"][0, 0], theta_opt)
    np.testing.assert_allclose(bundle["theta"][1, 0], theta_nom)
