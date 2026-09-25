"""Tests for default-eval: sample_nf → save_posterior_samples → plot_posterior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from bedcosmo.artifacts import (
    load_posterior_samples_file,
    make_posterior_samples_path,
    save_posterior_samples,
)
from bedcosmo.evaluate import Evaluator
from bedcosmo.plotting import BasePlotter, nf_entries_from_posterior_bundle
from bedcosmo.util import nf_posterior_entries, sample_nf


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
         patch("bedcosmo.util.sample_nf") as mock_sample:
        plotter.plot_posterior(
            experiment=experiment,
            nf_entries=entries,
            display="nominal",
            guide_samples=10,
        )
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
            "generated_by": "Evaluator.run",
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
    # Default display=('nominal','optimal') reorders relative to NPZ series order.
    assert len(plotted) == 2
    np.testing.assert_allclose(plotted[0].samples, theta[1, 0])  # nominal
    np.testing.assert_allclose(plotted[1].samples, theta[0, 0])  # optimal


def test_plot_posterior_display_filters_loaded_and_provided(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    n_guide, n_params = 10, 2
    rng = np.random.default_rng(6)
    theta = np.stack(
        [
            rng.normal(size=(1, n_guide, n_params)),
            rng.normal(size=(1, n_guide, n_params)),
        ],
        axis=0,
    )
    artifacts = tmp_path / "artifacts"
    out = make_posterior_samples_path(str(artifacts), step=7)
    save_posterior_samples(
        out,
        theta=theta,
        y=np.zeros((2, 1, 2)),
        design=np.array([[1.0], [2.0]]),
        series_names=["optimal", "nominal"],
        param_names=["p0", "p1"],
        meta={"status": "complete", "step": 7, "generated_by": "Evaluator.run"},
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
    plotter = BasePlotter(cosmo_exp="num_visits")

    with patch.object(plotter, "plot_triangle", return_value=fake_g) as mock_tri, \
         patch.object(plotter, "save_figure"), \
         patch("bedcosmo.plotting.Line2D"):
        plotter.plot_posterior(
            experiment=experiment,
            artifacts_dir=str(artifacts),
            eval_step=7,
            display="nominal",
        )
    plotted = mock_tri.call_args.args[0]
    assert len(plotted) == 1
    np.testing.assert_allclose(plotted[0].samples, theta[1, 0])

    entries = nf_entries_from_posterior_bundle(
        {
            "theta": theta,
            "design": np.array([[1.0], [2.0]]),
            "series_names": ["optimal", "nominal"],
            "param_names": ["p0", "p1"],
            "meta": {"series": []},
        },
        experiment,
    )
    with patch.object(plotter, "plot_triangle", return_value=fake_g) as mock_tri, \
         patch.object(plotter, "save_figure"), \
         patch("bedcosmo.plotting.Line2D"):
        plotter.plot_posterior(
            experiment=experiment,
            nf_entries=entries,
            display=("optimal",),
        )
    plotted = mock_tri.call_args.args[0]
    assert len(plotted) == 1
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
    from bedcosmo.artifacts import parse_eig_for_posterior

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


def _eig_data_with_marginal():
    return {
        "input_designs": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        "step_100": {
            "variable": {
                "eigs_avg": [0.1, 0.9, 0.5],
                "prior_entropy_avg": [3.0, 3.0, 3.0],
                "posterior_entropy_avg": [2.9, 2.1, 2.5],
            },
            "nominal": {
                "eigs_avg": 0.5,
                "prior_entropy_avg": 3.0,
                "posterior_entropy_avg": 2.5,
            },
            "marginal": {
                "p0": {"eigs_avg": [0.7, 0.2, 0.3], "nominal": {"eigs_avg": 0.4}},
            },
        },
    }


def _entries_experiment():
    return SimpleNamespace(
        central_val=torch.zeros(3, dtype=torch.float64),
        nominal_design=torch.tensor([9.0, 9.0], dtype=torch.float64),
        device="cpu",
    )


def test_nf_posterior_entries_joint_labels_and_optimal():
    """Optimal = joint EIG argmax; legends carry EIG and entropy."""
    eig_data = _eig_data_with_marginal()
    with patch("bedcosmo.util.sample_nf", return_value="samples") as mock_sample:
        entries = nf_posterior_entries(_entries_experiment(), "flow", eig_data, 100)
    assert [e["name"] for e in entries] == ["nominal", "optimal"]
    nominal, optimal = entries
    np.testing.assert_allclose(optimal["design"], [1.0, 1.0])
    np.testing.assert_allclose(nominal["design"], [9.0, 9.0])
    assert nominal["label"] == (
        "Nominal Design (NF), EIG: 0.500 bits, H_prior: 3.00 bits, H_post: 2.50 bits"
    )
    assert optimal["label"] == (
        "Optimal Design (NF), EIG: 0.900 bits, H_prior: 3.00 bits, H_post: 2.10 bits"
    )
    np.testing.assert_allclose(mock_sample.call_args_list[1].args[2], [1.0, 1.0])


def test_nf_posterior_entries_marginal_uses_marginal_optimal():
    """With params, optimal comes from the marginal EIG, not the joint EIG."""
    eig_data = _eig_data_with_marginal()
    with patch("bedcosmo.util.sample_nf", return_value="samples") as mock_sample:
        entries = nf_posterior_entries(
            _entries_experiment(), "flow", eig_data, 100, params=["p0"]
        )
    nominal, optimal = entries
    np.testing.assert_allclose(optimal["design"], [0.0, 0.0])
    assert optimal["label"].startswith("Optimal Design (NF), Marginal EIG: 0.700 bits")
    assert nominal["label"].startswith("Nominal Design (NF), Marginal EIG: 0.400 bits")
    assert all(c.kwargs["params"] == ["p0"] for c in mock_sample.call_args_list)


def test_nf_posterior_entries_nominal_only_without_eig_data():
    with patch("bedcosmo.util.sample_nf", return_value="samples"):
        entries = nf_posterior_entries(
            _entries_experiment(), "flow", None, display=("nominal",)
        )
    assert [e["label"] for e in entries] == ["Nominal Design (NF)"]
    with pytest.raises(ValueError, match="eig_data"):
        nf_posterior_entries(_entries_experiment(), "flow", None)


def _make_evaluator(tmp_path, experiment, eig_data):
    ev = Evaluator.__new__(Evaluator)
    ev.save_path = str(tmp_path / "artifacts")
    ev.run_id = "run123"
    ev.seed = 7
    ev.guide_samples = 20
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
    ev.eig_data = eig_data
    ev.input_designs = torch.randn(3, 2, dtype=torch.float64)
    ev.experiment = experiment
    ev.run_obj = MagicMock()
    ev.run_obj.info.run_id = "run123"
    ev.run_args = {"total_steps": 100}
    ev.plotter = MagicMock()
    ev._update_runtime = MagicMock()
    ev._eig_data_save_path = MagicMock(return_value=str(tmp_path / "eig.json"))
    ev._target_prior_entropy = MagicMock(return_value=None)
    return ev


def test_run_plots_sampled_entries_and_saves_npz(tmp_path):
    """Evaluator.run plots the entries it sampled (EIG legends) and saves them as NPZ."""
    n_guide, n_params = 20, 2
    rng = np.random.default_rng(1)
    theta_nom = rng.normal(size=(n_guide, n_params))
    theta_opt = rng.normal(size=(n_guide, n_params))
    ev = _make_evaluator(tmp_path, _entries_experiment(), _eig_data_with_marginal())
    ev.get_eig = MagicMock(
        side_effect=[(0.5, 0.01), (np.array([0.1, 0.9, 0.5]), np.zeros(3))]
    )
    ev.plotter.plot_raw_posterior.side_effect = lambda **kw: plt.figure()

    mcsamples = [
        _make_mcsamples(theta_nom, ["p0", "p1"]),
        _make_mcsamples(theta_opt, ["p0", "p1"]),
    ]
    with patch("bedcosmo.evaluate.render_overlay"), \
         patch("bedcosmo.evaluate.load_model", return_value=(MagicMock(name="flow"), 100)), \
         patch("bedcosmo.util.sample_nf", side_effect=mcsamples):
        ev.run(eval_step=100)

    ev.plotter.plot_posterior.assert_called_once()
    args = ev.plotter.plot_posterior.call_args.args
    assert args[0] is ev.experiment
    plotted = args[1]
    assert [e["name"] for e in plotted] == ["nominal", "optimal"]
    assert "EIG: 0.900 bits" in plotted[1]["label"]

    bundle = load_posterior_samples_file(ev.save_path, step=100, generated_by="Evaluator.run")
    assert list(bundle["series_names"]) == ["nominal", "optimal"]
    assert bundle["meta"]["num_data_samples"] == 1
    assert [s["label"] for s in bundle["meta"]["series"]] == [e["label"] for e in plotted]
    assert bundle["theta"].shape == (2, 1, n_guide, n_params)
    np.testing.assert_allclose(bundle["theta"][0, 0], theta_nom)
    np.testing.assert_allclose(bundle["theta"][1, 0], theta_opt)
    np.testing.assert_allclose(bundle["design"][1], [1.0, 1.0])

    # Raw-sample triangles come from the saved NPZ: full range, then plot ranges.
    raw_calls = ev.plotter.plot_raw_posterior.call_args_list
    assert [c.kwargs for c in raw_calls] == [
        {"posterior_samples_path": bundle["path"], "plot_ranges": False},
        {"posterior_samples_path": bundle["path"], "plot_ranges": True},
    ]

    # Replot from the NPZ keeps the saved legend labels.
    entries = nf_entries_from_posterior_bundle(
        bundle, SimpleNamespace(cosmo_params=["p0", "p1"], latex_labels=["a", "b"])
    )
    assert [e["label"] for e in entries] == [e["label"] for e in plotted]


def test_run_saves_npz_before_plotting(tmp_path):
    """A failing posterior plot must not lose the sampled NPZ."""
    ev = _make_evaluator(tmp_path, _entries_experiment(), _eig_data_with_marginal())
    ev.get_eig = MagicMock(
        side_effect=[(0.5, 0.01), (np.array([0.1, 0.9, 0.5]), np.zeros(3))]
    )
    ev.plotter.plot_posterior.side_effect = RuntimeError("plot failed")
    samples = _make_mcsamples(np.zeros((20, 2)), ["p0", "p1"])
    with patch("bedcosmo.evaluate.render_overlay"), \
         patch("bedcosmo.evaluate.load_model", return_value=(MagicMock(name="flow"), 100)), \
         patch("bedcosmo.util.sample_nf", return_value=samples):
        ev.run(eval_step=100)

    ev.plotter.plot_posterior.assert_called_once()
    bundle = load_posterior_samples_file(ev.save_path, step=100, generated_by="Evaluator.run")
    assert list(bundle["series_names"]) == ["nominal", "optimal"]


def test_run_marginal_samples_without_saved_npz(tmp_path):
    """--marginal plots sample fresh at the marginal-optimal design (no NPZ needed)."""
    ev = _make_evaluator(tmp_path, _entries_experiment(), _eig_data_with_marginal())
    ev.marginal_eig_subsets = [["p0"]]
    ev._subset_id = lambda subset: "+".join(subset)
    ev.get_marginal_eig = MagicMock()
    with patch("bedcosmo.evaluate.load_model", return_value=(MagicMock(name="flow"), 100)), \
         patch("bedcosmo.util.sample_nf", return_value="samples"):
        ev.run_marginal(eval_step=100)

    ev.plotter.plot_posterior.assert_called_once()
    call = ev.plotter.plot_posterior.call_args
    assert call.kwargs["params"] == ["p0"]
    assert call.kwargs["filename"] == "posterior_marginal_p0"
    optimal = call.args[1][1]
    np.testing.assert_allclose(optimal["design"], [0.0, 0.0])


def test_render_overlay_checkpoint_branch_plots_nf_and_grid(tmp_path):
    """The explicit-checkpoint overlay samples NF entries and reaches plot_posterior."""
    import json

    from bedcosmo.util import render_overlay

    own = _eig_data_with_marginal()
    sibling = tmp_path / "grid_eig.json"
    sibling.write_text(json.dumps({
        "status": "complete",
        "step_100": {"nominal": {"grid": {"eigs_avg": 0.45}}},
    }))
    grid_experiment = _entries_experiment()
    grid_experiment.name = "num_visits"
    with patch("bedcosmo.util.load_posterior_flow_from_checkpoint_file", return_value="flow"), \
         patch("bedcosmo.util.sample_nf", return_value="samples"), \
         patch.object(BasePlotter, "plot_posterior", autospec=True) as mock_plot, \
         patch.object(BasePlotter, "eig_designs", autospec=True, create=True):
        render_overlay(
            own,
            "nf",
            str(sibling),
            BasePlotter(cosmo_exp="num_visits"),
            100,
            device="cpu",
            nf_checkpoint_path="ckpt.pt",
            grid_experiment=grid_experiment,
            overlay_save_dir=str(tmp_path),
        )
    mock_plot.assert_called_once()
    kwargs = mock_plot.call_args.kwargs
    assert [e["name"] for e in kwargs["nf_entries"]] == ["nominal", "optimal"]
    assert kwargs["nominal_grid_eig"] == 0.45
    assert kwargs["save_dir"] == str(tmp_path)


def test_run_skips_raw_posterior_outside_physical_space(tmp_path):
    """Prior bounds/plot windows are physical-space only, so no raw plots otherwise."""
    ev = _make_evaluator(tmp_path, _entries_experiment(), _eig_data_with_marginal())
    ev.param_space = "unconstrained"
    ev.get_eig = MagicMock(
        side_effect=[(0.5, 0.01), (np.array([0.1, 0.9, 0.5]), np.zeros(3))]
    )
    samples = _make_mcsamples(np.zeros((20, 2)), ["p0", "p1"])
    with patch("bedcosmo.evaluate.render_overlay"), \
         patch("bedcosmo.evaluate.load_model", return_value=(MagicMock(name="flow"), 100)), \
         patch("bedcosmo.util.sample_nf", return_value=samples):
        ev.run(eval_step=100)

    ev.plotter.plot_posterior.assert_called_once()
    ev.plotter.plot_raw_posterior.assert_not_called()
