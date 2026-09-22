"""Tests for default-eval persistence of _nf_display_samples via generate_posterior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from bedcosmo.artifacts import (
    load_posterior_samples,
    load_posterior_samples_file,
    save_central_posterior_from_nf_entries,
)
from bedcosmo.evaluate import Evaluator
from bedcosmo.plotting import BasePlotter


def _make_mcsamples(theta: np.ndarray, names: list[str]):
    return SimpleNamespace(
        samples=theta,
        paramNames=SimpleNamespace(
            list=lambda: list(names),
            names=[SimpleNamespace(name=n) for n in names],
        ),
    )


def test_save_central_posterior_from_nf_entries_packs_schema(tmp_path):
    n_guide, n_params, n_obs = 40, 2, 4
    rng = np.random.default_rng(0)
    theta_opt = rng.normal(size=(n_guide, n_params))
    theta_nom = rng.normal(size=(n_guide, n_params))
    design_opt = rng.normal(size=3)
    design_nom = rng.normal(size=3)
    central = torch.randn(n_obs, dtype=torch.float64)
    experiment = SimpleNamespace(central_val=central)

    # Plotter order is nominal then optimal; saver reorders to optimal, nominal.
    entries = [
        {
            "name": "nominal",
            "color": "tab:blue",
            "design": design_nom,
            "samples": _make_mcsamples(theta_nom, ["p0", "p1"]),
        },
        {
            "name": "optimal",
            "color": "tab:orange",
            "design": design_opt,
            "samples": _make_mcsamples(theta_opt, ["p0", "p1"]),
        },
    ]
    out = save_central_posterior_from_nf_entries(
        str(tmp_path / "artifacts"),
        entries,
        experiment=experiment,
        step=100,
        meta={"run_id": "run123", "generated_by": "Evaluator.run", "seed": 7},
        eig_values=[0.1, 0.9, 0.2],
    )
    assert out is not None
    bundle = load_posterior_samples(out)
    assert list(bundle["series_names"]) == ["optimal", "nominal"]
    assert bundle["meta"]["num_data_samples"] == 1
    assert bundle["meta"]["conditioning"] == "central_val"
    assert bundle["meta"]["generated_by"] == "Evaluator.run"
    assert bundle["meta"]["optimal_design_index"] == 1
    assert bundle["theta"].shape == (2, 1, n_guide, n_params)
    np.testing.assert_allclose(bundle["theta"][0, 0], theta_opt)
    np.testing.assert_allclose(bundle["theta"][1, 0], theta_nom)
    np.testing.assert_allclose(bundle["y"][0, 0], central.numpy().reshape(-1))
    discovered = load_posterior_samples_file(str(tmp_path / "artifacts"), step=100)
    assert discovered["path"] == out


def test_generate_posterior_persists_nf_display_samples(tmp_path, monkeypatch):
    """generate_posterior saves the same entries produced by _nf_display_samples."""
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    n_guide, n_params, n_obs = 20, 2, 3
    rng = np.random.default_rng(1)
    theta_nom = rng.normal(size=(n_guide, n_params))
    theta_opt = rng.normal(size=(n_guide, n_params))
    central = torch.zeros(n_obs, dtype=torch.float64)
    nominal_design = torch.ones(2, dtype=torch.float64)
    experiment = SimpleNamespace(
        central_val=central,
        nominal_design=nominal_design,
        nominal_context=torch.cat([nominal_design, central]),
        cosmo_params=["p0", "p1"],
        latex_labels=["p0", "p1"],
        device="cpu",
        central_params=None,
        get_guide_samples=MagicMock(
            side_effect=[
                _make_mcsamples(theta_nom, ["p0", "p1"]),
                _make_mcsamples(theta_opt, ["p0", "p1"]),
            ]
        ),
    )
    input_designs = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    eig_values = np.array([0.1, 0.9, 0.2])
    artifacts = tmp_path / "artifacts"

    fake_g = MagicMock()
    fake_g.fig.legends = []
    fake_g.subplots = [[MagicMock()]]
    fake_g.param_names_for_root.return_value = SimpleNamespace(
        names=[SimpleNamespace(name="p0"), SimpleNamespace(name="p1")]
    )

    plotter = BasePlotter(cosmo_exp="num_visits")
    with patch.object(plotter, "plot_posterior", return_value=fake_g), \
         patch.object(plotter, "save_figure"), \
         patch("bedcosmo.plotting.Line2D"):
        plotter.generate_posterior(
            experiment=experiment,
            posterior_flow=MagicMock(),
            input_designs=input_designs,
            eig_values=eig_values,
            display=("nominal", "optimal"),
            guide_samples=n_guide,
            device="cpu",
            seed=1,
            persist_posterior_samples=True,
            posterior_samples_dir=str(artifacts),
            posterior_samples_step=50,
            posterior_samples_meta={"generated_by": "Evaluator.run", "run_id": "abc"},
        )

    assert experiment.get_guide_samples.call_count == 2  # sampled once, not twice
    bundle = load_posterior_samples_file(str(artifacts), step=50)
    assert bundle["meta"]["generated_by"] == "Evaluator.run"
    assert bundle["meta"]["num_data_samples"] == 1
    assert bundle["theta"].shape == (2, 1, n_guide, n_params)
    np.testing.assert_allclose(bundle["theta"][0, 0], theta_opt)
    np.testing.assert_allclose(bundle["theta"][1, 0], theta_nom)


def test_run_passes_persist_flags_to_generate_posterior(tmp_path):
    ev = Evaluator.__new__(Evaluator)
    ev.save_path = str(tmp_path / "artifacts")
    ev.run_id = "run123"
    ev.seed = 7
    ev.guide_samples = 100
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
    ev.experiment = SimpleNamespace(
        nominal_design=torch.zeros(3, dtype=torch.float64),
    )
    ev.plotter = MagicMock()
    ev.get_eig = MagicMock(side_effect=[(0.5, 0.01), (np.array([0.2, 0.8]), np.zeros(2))])
    ev._update_runtime = MagicMock()
    ev._eig_data_save_path = MagicMock(return_value=str(tmp_path / "eig.json"))
    ev._target_prior_entropy = MagicMock(return_value=None)

    with patch("bedcosmo.evaluate.render_overlay"):
        ev.run(eval_step=100)

    kwargs = ev.plotter.generate_posterior.call_args.kwargs
    assert kwargs["persist_posterior_samples"] is True
    assert kwargs["posterior_samples_dir"] == ev.save_path
    assert kwargs["posterior_samples_step"] == 100
    assert kwargs["posterior_samples_meta"]["generated_by"] == "Evaluator.run"
    assert kwargs["posterior_samples_meta"]["run_id"] == "run123"
    # No parallel Evaluator sampling wrapper
    assert not hasattr(ev, "_save_run_posterior_samples") or not callable(
        getattr(Evaluator, "_save_run_posterior_samples", None)
    )
