"""Tests for default-eval: sample_nf → save_posterior_samples → plot_posterior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from bedcosmo.artifacts import (
    load_posterior_samples_file,
    make_posterior_samples_path,
    save_posterior_samples,
)
from bedcosmo.evaluate import Evaluator
from bedcosmo.plotting import BasePlotter, nf_entries_from_posterior_bundle
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
         patch("bedcosmo.plotting.sample_nf") as mock_sample:
        plotter.plot_posterior(experiment=experiment, nf_entries=entries, guide_samples=10)
    mock_sample.assert_not_called()
    experiment.get_guide_samples.assert_not_called()


def test_plot_posterior_loads_npz_when_entries_omitted(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    n_guide, n_params = 12, 2
    rng = np.random.default_rng(4)
    theta = np.stack(
        [
            rng.normal(size=(1, n_guide, n_params)),
            rng.normal(size=(1, n_guide, n_params)),
        ],
        axis=0,
    )
    y = np.zeros((2, 1, 3))
    design = np.array([[1.0, 0.0], [0.0, 1.0]])
    artifacts = tmp_path / "artifacts"
    out = make_posterior_samples_path(str(artifacts), step=50)
    save_posterior_samples(
        out,
        theta=theta,
        y=y,
        design=design,
        series_names=["optimal", "nominal"],
        param_names=["p0", "p1"],
        meta={
            "status": "complete",
            "step": 50,
            "series": [
                {"name": "optimal", "color": "tab:orange"},
                {"name": "nominal", "color": "tab:blue"},
            ],
        },
    )
    experiment = SimpleNamespace(
        cosmo_params=["p0", "p1"],
        latex_labels=["p0", "p1"],
        device="cpu",
        central_params=None,
    )
    fake_g = MagicMock()
    fake_g.fig.legends = []
    fake_g.subplots = [[MagicMock()]]
    fake_g.param_names_for_root.return_value = SimpleNamespace(
        names=[SimpleNamespace(name="p0"), SimpleNamespace(name="p1")]
    )
    plotter = BasePlotter(cosmo_exp="num_visits")
    with patch.object(plotter, "plot_triangle", return_value=fake_g) as mock_tri, \
         patch.object(plotter, "save_figure"), \
         patch("bedcosmo.plotting.Line2D"):
        plotter.plot_posterior(
            experiment=experiment,
            artifacts_dir=str(artifacts),
            eval_step=50,
        )
    mock_tri.assert_called_once()
    plotted = mock_tri.call_args.args[0]
    assert len(plotted) == 2
    np.testing.assert_allclose(plotted[0].samples, theta[0, 0])


def test_nf_entries_from_posterior_bundle_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    theta = rng.normal(size=(2, 1, 8, 2))
    bundle = {
        "theta": theta,
        "design": np.array([[1.0, 0.0], [0.5, 0.5]]),
        "series_names": ["optimal", "nominal"],
        "param_names": ["a", "b"],
        "meta": {"series": [{"name": "optimal", "color": "tab:orange"}]},
    }
    experiment = SimpleNamespace(cosmo_params=["a", "b"], latex_labels=["a", "b"])
    entries = nf_entries_from_posterior_bundle(bundle, experiment)
    assert [e["name"] for e in entries] == ["optimal", "nominal"]
    assert entries[0]["color"] == "tab:orange"
    assert entries[1]["color"] == "tab:blue"
    np.testing.assert_allclose(entries[0]["samples"].samples, theta[0, 0])


def test_plot_posterior_requires_artifact_when_no_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    plotter = BasePlotter(cosmo_exp="num_visits")
    experiment = SimpleNamespace(cosmo_params=["p0"], latex_labels=["p0"], device="cpu")
    try:
        plotter.plot_posterior(experiment=experiment)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "artifacts_dir" in str(e) or "posterior_samples_path" in str(e)


def test_parse_eig_for_posterior_joint():
    """parse_eig_for_posterior returns designs, EIGs, and entropy from eig_data."""
    from bedcosmo.util import parse_eig_for_posterior

    eig_data = {
        "input_designs": [[0.0, 0.0], [1.0, 1.0]],
        "step_10": {
            "variable": {
                "eigs_avg": [0.2, 0.8],
                "prior_entropy_avg": [1.0, 1.1],
                "posterior_entropy_avg": [0.5, 0.4],
            },
            "nominal": {
                "eigs_avg": 0.3,
                "prior_entropy_avg": 1.0,
                "posterior_entropy_avg": 0.6,
            },
        },
    }
    input_designs, eig_values, nominal_eig, entropy = parse_eig_for_posterior(
        eig_data, eval_step=10
    )
    np.testing.assert_allclose(input_designs, [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(eig_values, [0.2, 0.8])
    assert nominal_eig == 0.3
    assert entropy["nominal_prior_entropy"] == 1.0
    assert entropy["nominal_posterior_entropy"] == 0.6


def test_run_sample_save_then_plot(tmp_path):
    """Evaluator.run samples, saves NPZ, then plot_posterior loads that NPZ."""
    n_guide, n_params, n_obs = 20, 2, 3
    rng = np.random.default_rng(1)
    theta_nom = rng.normal(size=(n_guide, n_params))
    theta_opt = rng.normal(size=(n_guide, n_params))
    central = torch.zeros(n_obs, dtype=torch.float64)
    design_nom = torch.ones(2, dtype=torch.float64)

    experiment = SimpleNamespace(
        central_val=central,
        nominal_design=design_nom,
        device="cpu",
    )

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
    ev.eig_data = {
        "input_designs": [[0.0, 0.0], [1.0, 1.0]],
        "step_100": {
            "variable": {"eigs_avg": [0.1, 0.9]},
            "nominal": {"eigs_avg": 0.5},
        },
    }
    ev.input_designs = torch.randn(2, 3, dtype=torch.float64)
    ev.experiment = experiment
    ev.run_obj = MagicMock()
    ev.run_obj.info.run_id = "run123"
    ev.run_args = {"total_steps": 100}
    ev.plotter = MagicMock()
    ev.plotter._entropy_legend_suffix = MagicMock(return_value="")
    ev.get_eig = MagicMock(side_effect=[(0.5, 0.01), (np.array([0.2, 0.8]), np.zeros(2))])
    ev._update_runtime = MagicMock()
    ev._eig_data_save_path = MagicMock(return_value=str(tmp_path / "eig.json"))
    ev._target_prior_entropy = MagicMock(return_value=None)

    mcsamples = [
        _make_mcsamples(theta_nom, ["p0", "p1"]),
        _make_mcsamples(theta_opt, ["p0", "p1"]),
    ]
    with patch("bedcosmo.evaluate.render_overlay"), \
         patch("bedcosmo.evaluate.load_model", return_value=(MagicMock(name="flow"), 100)) as mock_load, \
         patch("bedcosmo.evaluate.sample_nf", side_effect=mcsamples) as mock_sample, \
         patch("bedcosmo.evaluate.save_posterior_samples", wraps=save_posterior_samples) as save_spy:
        ev.run(eval_step=100)

    mock_load.assert_called_once()
    assert mock_sample.call_count == 2
    assert save_spy.called
    ev.plotter.plot_posterior.assert_called_once()
    plot_kwargs = ev.plotter.plot_posterior.call_args.kwargs
    assert plot_kwargs.get("artifacts_dir") == ev.save_path
    assert plot_kwargs.get("eval_step") == 100
    # No in-memory nf_entries — plot loads from the just-saved NPZ.
    assert plot_kwargs.get("nf_entries") is None
    assert len(ev.plotter.plot_posterior.call_args.args) == 0

    bundle = load_posterior_samples_file(ev.save_path, step=100)
    assert list(bundle["series_names"]) == ["optimal", "nominal"]
    assert bundle["meta"]["generated_by"] == "Evaluator.run"
    assert bundle["meta"]["num_data_samples"] == 1
    assert bundle["meta"]["optimal_design_index"] == 1
    assert bundle["theta"].shape == (2, 1, n_guide, n_params)
    np.testing.assert_allclose(bundle["theta"][0, 0], theta_opt)
    np.testing.assert_allclose(bundle["theta"][1, 0], theta_nom)
