"""NumTracers ``lya_accounting="qso_coupled"``: Lya QSO follows the design's QSO class.

Tracer and Lya QSOs share one target selection, so Lya's count scales with the QSO class
fraction and its nominal DESI covariance block is rescaled by ``lya_sigma_scaling``. These
tests pin the nominal identity (nominal design -> nominal Lya count and block), the
scaling of a halved QSO class under each law, and the budget (Lya's forest fibers only).
"""
import os

import numpy as np
import pytest
import torch

from bedcosmo.num_tracers.lya import (
    LYA_FOREST_FRACTION, LYA_NOISE_POWER_X, LYA_SIGMA_SCALINGS, resolve_lya_accounting)
from bedcosmo.util import init_experiment

DATASETS = ["dr1", "dr2"]
# sigma ratio of a halved Lya count under the default noise-power law.
_NP_HALVED = (1 + 2 / LYA_NOISE_POWER_X) / (1 + 1 / LYA_NOISE_POWER_X)
DESIGN_ARGS = {"dr1": "design_args_dr1.yaml", "dr2": "design_args_dr2.yaml"}
_DATA_ROOT = os.path.join(os.environ.get("HOME", ""), "data", "desi")
_CKPT_DIR = os.path.join(
    os.environ.get("SCRATCH", ""), "bedcosmo", "num_tracers", "emulator", "bao",
    "models", "dr1", "base", "fourier")


def _skip_without(ds):
    if not os.path.isdir(os.path.join(_DATA_ROOT, f"bao_{ds}")):
        pytest.skip(f"desi dataset bao_{ds} not available")


def _make_exp(ds, likelihood_mode="scaling", **overrides):
    kwargs = dict(lya_accounting="qso_coupled")
    kwargs.update(overrides)
    if likelihood_mode == "emulator":
        kwargs.setdefault("emulator_space", "fourier")
    return init_experiment(
        cosmo_exp="num_tracers", prior_args_path="prior_args_hrdrag.yaml",
        design_args_path=DESIGN_ARGS[ds], dataset=ds, analysis="bao", cosmo_model="base",
        likelihood_mode=likelihood_mode, device="cpu", mode="eval", **kwargs)


def _halve_qso(exp):
    """The nominal design with half the QSO class moved to ELG (budget unchanged)."""
    nominal = exp.nominal_design.to(torch.float64)
    halved = nominal.clone()
    i_qso, i_elg = exp.design_labels.index("QSO"), exp.design_labels.index("ELG")
    halved[i_elg] += halved[i_qso] / 2
    halved[i_qso] /= 2
    return nominal.view(1, -1), halved.view(1, -1)


def _lya_n(exp, design):
    return float(exp._passed_ratio_to_n_tracers(exp.calc_passed(design))["Lya_QSO"].squeeze())


@pytest.mark.parametrize("ds", DATASETS)
def test_budget_holds_lya_forest_fibers_only(ds):
    _skip_without(ds)
    exp = _make_exp(ds)
    bins = exp.desi_data.drop_duplicates("tracer").set_index("tracer")["observed"]
    expected = bins.drop("Lya QSO").sum() + LYA_FOREST_FRACTION[ds] * bins["Lya QSO"]
    assert exp.nominal_total_obs == int(expected)
    assert float(exp.nominal_design.sum()) == pytest.approx(1.0)
    dt = exp.desi_tracers.set_index("tracer")["observed"]
    qso_fibers = dt["QSO"] + LYA_FOREST_FRACTION[ds] * dt["Lya QSO"]
    share = qso_fibers / (dt.sum() - (1 - LYA_FOREST_FRACTION[ds]) * dt["Lya QSO"])
    assert float(exp.nominal_design[exp.design_labels.index("QSO")]) == pytest.approx(share)


@pytest.mark.parametrize("ds", DATASETS)
def test_nominal_design_gives_nominal_counts_and_halving_qso_halves_them(ds):
    _skip_without(ds)
    exp = _make_exp(ds)
    nominal, halved = _halve_qso(exp)
    n_nominal = exp._passed_ratio_to_n_tracers(exp.calc_passed(nominal))
    for tracer_bin, desi_name in exp._EMULATOR_TRACER_TO_DESI.items():
        passed = exp.desi_data.loc[exp.desi_data["tracer"] == desi_name, "passed"]
        assert float(n_nominal[tracer_bin].squeeze()) == pytest.approx(float(passed.iloc[0]), rel=1e-6)
    n_halved = exp._passed_ratio_to_n_tracers(exp.calc_passed(halved))
    for tracer_bin in ("QSO", "Lya_QSO"):
        assert float(n_halved[tracer_bin].squeeze()) == pytest.approx(
            float(n_nominal[tracer_bin].squeeze()) / 2, rel=1e-6)


@pytest.mark.parametrize("ds", DATASETS)
@pytest.mark.parametrize("law, halved_factor", [
    ("noise_power", _NP_HALVED), ("inverse_sqrt_n", np.sqrt(2.0)), ("inverse_n", 2.0)])
def test_scaling_mode_lya_sigma_follows_the_law(ds, law, halved_factor):
    _skip_without(ds)
    exp = _make_exp(ds, lya_sigma_scaling=law)
    nominal, halved = _halve_qso(exp)
    rows = exp._lya_qso_rows
    at_nominal = exp.sigma_scaling_factor(exp.calc_passed(nominal), rows)
    assert torch.allclose(at_nominal, torch.ones_like(at_nominal))
    at_halved = exp.sigma_scaling_factor(exp.calc_passed(halved), rows)
    assert torch.allclose(at_halved, torch.full_like(at_halved, halved_factor))
    # The tracer QSO bin keeps scaling mode's 1/sqrt(N) whatever the Lya law.
    qso_row = np.where(exp.desi_data["tracer"] == "QSO")[0]
    qso = exp.sigma_scaling_factor(exp.calc_passed(halved), qso_row)
    assert torch.allclose(qso, torch.full_like(qso, np.sqrt(2.0)))


@pytest.mark.parametrize("accounting", ["outside", "in_budget_fixed"])
def test_other_accountings_hold_the_lya_error_nominal(accounting):
    _skip_without("dr1")
    exp = _make_exp("dr1", lya_accounting=accounting)
    _, halved = _halve_qso(exp)
    factor = exp.sigma_scaling_factor(exp.calc_passed(halved), exp._lya_qso_rows)
    assert torch.allclose(factor, torch.ones_like(factor))


def test_noise_power_law_limits():
    """1 at nominal; local slope -1/(1+x); 1/N when sparse; flat when signal dominated."""
    law = LYA_SIGMA_SCALINGS["noise_power"]
    assert law(1.0) == pytest.approx(1.0)
    eps = 1e-6
    slope = (np.log(law(1 + eps)) - np.log(law(1 - eps))) / (2 * eps)
    assert slope == pytest.approx(-1 / (1 + LYA_NOISE_POWER_X), rel=1e-6)
    assert law(1e-4, x=1e-3) / law(2e-4, x=1e-3) == pytest.approx(2.0, rel=1e-3)
    assert law(1e4, x=1.0) / law(2e4, x=1.0) == pytest.approx(1.0, abs=1e-3)
    got = law(torch.tensor([0.5, 2.0], dtype=torch.float64))
    assert torch.allclose(got, torch.tensor([law(0.5), law(2.0)], dtype=torch.float64))


def test_default_law_is_noise_power():
    _skip_without("dr1")
    assert _make_exp("dr1").lya_sigma_scaling == "noise_power"


def test_legacy_flag_maps_onto_the_accountings():
    assert resolve_lya_accounting(None, None) == "outside"
    assert resolve_lya_accounting(None, False) == "outside"
    assert resolve_lya_accounting(None, True) == "in_budget_fixed"
    assert resolve_lya_accounting("qso_coupled", None) == "qso_coupled"
    with pytest.raises(ValueError, match="contradicts"):
        resolve_lya_accounting("qso_coupled", False)
    with pytest.raises(ValueError, match="must be one of"):
        resolve_lya_accounting("coupled", None)


def test_unknown_scaling_law_is_rejected():
    _skip_without("dr1")
    with pytest.raises(ValueError, match="lya_sigma_scaling"):
        _make_exp("dr1", lya_sigma_scaling="inverse_cube")


def test_runs_logged_with_lya_accounting_keep_it(tmp_path):
    """init_experiment's old-run default (lya_in_budget=True) must not override a logged accounting."""
    _skip_without("dr1")
    from types import SimpleNamespace

    import yaml

    from bedcosmo.util import get_experiment_config_path

    def load(name):
        with open(get_experiment_config_path("num_tracers", name)) as f:
            return yaml.safe_load(f)

    run_obj = SimpleNamespace(info=SimpleNamespace(artifact_uri=f"file://{tmp_path}"))
    run_args = dict(cosmo_exp="num_tracers", dataset="dr1", analysis="bao", cosmo_model="base",
                    likelihood_mode="scaling", device="cpu", mode="eval",
                    lya_accounting="qso_coupled", lya_sigma_scaling="inverse_n")
    exp = init_experiment(run_obj=run_obj, run_args=run_args, design_args=load(DESIGN_ARGS["dr1"]),
                          prior_args=load("prior_args_hrdrag.yaml"))
    assert exp.lya_accounting == "qso_coupled"
    assert exp.lya_sigma_scaling == "inverse_n"


@pytest.mark.skipif(not os.path.isdir(_CKPT_DIR),
                    reason="bao dr1/base fourier emulator checkpoints not available")
@pytest.mark.parametrize("law, halved_variance", [
    ("noise_power", _NP_HALVED ** 2), ("inverse_sqrt_n", 2.0), ("inverse_n", 4.0)])
@pytest.mark.parametrize("sqrtn_ref", [None, "fiducial"])
def test_emulator_fallback_lya_block_follows_the_law(law, halved_variance, sqrtn_ref):
    exp = _make_exp("dr1", likelihood_mode="emulator", lya_sigma_scaling=law,
                    emulator_covariance="fiducial", emulator_sqrtn_ref=sqrtn_ref)
    assert "Lya_QSO" in exp._emulator_fallback_bins
    rows = exp._lya_qso_rows
    block = np.ix_(rows, rows)
    nominal, halved = _halve_qso(exp)
    cov_nominal = exp._build_emulator_covariance(exp.calc_passed(nominal), {})[0].numpy()
    np.testing.assert_allclose(cov_nominal[block], exp.ref_cov[block], rtol=1e-6)
    cov_halved = exp._build_emulator_covariance(exp.calc_passed(halved), {})[0].numpy()
    np.testing.assert_allclose(cov_halved[block], halved_variance * exp.ref_cov[block], rtol=1e-6)


@pytest.mark.parametrize("accounting", ["outside", "in_budget_fixed", "qso_coupled"])
def test_design_tool_nominal_matches_the_experiment(accounting):
    """design.py's nominal vector (pinned pool rows) is the experiment's under each accounting."""
    _skip_without("dr1")
    from bedcosmo.num_tracers.design import nominal_vector

    exp = _make_exp("dr1", lya_accounting=accounting)
    np.testing.assert_allclose(nominal_vector("dr1", accounting), exp.nominal_design.numpy(), rtol=1e-12)
