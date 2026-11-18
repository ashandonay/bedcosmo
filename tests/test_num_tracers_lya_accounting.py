"""NumTracers ``lya_accounting="qso_coupled"``: Lya QSO follows the design's QSO class.

Tracer and Lya QSOs share one target selection, so Lya's count scales with the QSO class
fraction. Without a Lya table its nominal DESI covariance block scales as 1/sqrt(N), like
every other row. These tests pin the nominal identity (nominal design -> nominal Lya count
and block), the scaling of a halved QSO class, and the budget (Lya's forest fibers only).
"""
import os
import warnings

import numpy as np
import pytest
import torch

from bedcosmo.num_tracers.lya import LYA_FOREST_FRACTION, resolve_lya_accounting
from bedcosmo.util import init_experiment

DATASETS = ["dr1", "dr2"]
DESIGN_ARGS = {"dr1": "design_args_dr1.yaml", "dr2": "design_args_dr2.yaml"}
_DATA_ROOT = os.path.join(os.environ.get("HOME", ""), "data", "desi")
_CKPT_DIR = os.path.join(
    os.environ.get("SCRATCH", ""), "bedcosmo", "num_tracers", "emulator", "bao",
    "models", "dr1", "base", "fourier")


def _skip_without(ds):
    if not os.path.isdir(os.path.join(_DATA_ROOT, f"bao_{ds}")):
        pytest.skip(f"desi dataset bao_{ds} not available")


def _design_args(ds, accounting="qso_coupled"):
    """The dataset's design_args. Its pool is built for qso_coupled; another accounting
    gets the grid instead, since the pool guard refuses a pool built for another one."""
    import yaml

    from bedcosmo.util import get_experiment_config_path

    with open(get_experiment_config_path("num_tracers", DESIGN_ARGS[ds])) as f:
        args = yaml.safe_load(f)
    return args if accounting == "qso_coupled" else dict(args, input_path=None)


def _make_exp(ds, likelihood_mode="scaling", **overrides):
    kwargs = dict(lya_accounting="qso_coupled")
    kwargs.update(overrides)
    if likelihood_mode == "emulator":
        kwargs.setdefault("emulator_space", "fourier")
    return init_experiment(
        cosmo_exp="num_tracers", prior_args_path="prior_args_hrdrag.yaml",
        design_args=_design_args(ds, kwargs["lya_accounting"]), dataset=ds, analysis="bao",
        cosmo_model="base", likelihood_mode=likelihood_mode, device="cpu", mode="eval", **kwargs)


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
def test_scaling_mode_lya_sigma_scales_as_inverse_sqrt_n(ds):
    _skip_without(ds)
    exp = _make_exp(ds)
    nominal, halved = _halve_qso(exp)
    rows = exp._lya_qso_rows
    at_nominal = exp.sigma_scaling_factor(exp.calc_passed(nominal), rows)
    assert torch.allclose(at_nominal, torch.ones_like(at_nominal))
    at_halved = exp.sigma_scaling_factor(exp.calc_passed(halved), rows)
    assert torch.allclose(at_halved, torch.full_like(at_halved, np.sqrt(2.0)))
    # The tracer QSO bin scales the same way.
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


def test_legacy_flag_maps_onto_the_accountings():
    assert resolve_lya_accounting(None, None) == "qso_coupled"
    assert resolve_lya_accounting(None, False) == "outside"
    assert resolve_lya_accounting(None, True) == "in_budget_fixed"
    assert resolve_lya_accounting("qso_coupled", None) == "qso_coupled"
    with pytest.raises(ValueError, match="contradicts"):
        resolve_lya_accounting("qso_coupled", False)
    with pytest.raises(ValueError, match="must be one of"):
        resolve_lya_accounting("coupled", None)


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
                    lya_accounting="qso_coupled")
    exp = init_experiment(run_obj=run_obj, run_args=run_args, design_args=load(DESIGN_ARGS["dr1"]),
                          prior_args=load("prior_args_hrdrag.yaml"))
    assert exp.lya_accounting == "qso_coupled"


def test_runs_logged_with_a_lya_law_still_load(tmp_path):
    """A run logged with the removed lya_sigma_scaling reloads (key dropped), warning when its
    law was not 1/sqrt(N)."""
    _skip_without("dr1")
    from types import SimpleNamespace

    import yaml

    from bedcosmo.util import get_experiment_config_path

    def load(name):
        with open(get_experiment_config_path("num_tracers", name)) as f:
            return yaml.safe_load(f)

    run_obj = SimpleNamespace(info=SimpleNamespace(artifact_uri=f"file://{tmp_path}"))
    for law, warns in (("noise_power", True), ("inverse_sqrt_n", False)):
        run_args = dict(cosmo_exp="num_tracers", dataset="dr1", analysis="bao",
                        cosmo_model="base", likelihood_mode="scaling", device="cpu", mode="eval",
                        lya_accounting="qso_coupled", lya_sigma_scaling=law)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exp = init_experiment(run_obj=run_obj, run_args=run_args,
                                  design_args=load(DESIGN_ARGS["dr1"]),
                                  prior_args=load("prior_args_hrdrag.yaml"))
        assert exp.lya_accounting == "qso_coupled"
        assert not hasattr(exp, "lya_sigma_scaling")
        assert any("lya_sigma_scaling" in str(w.message) for w in caught) == warns


@pytest.mark.skipif(not os.path.isdir(_CKPT_DIR),
                    reason="bao dr1/base fourier emulator checkpoints not available")
@pytest.mark.parametrize("sqrtn_ref", [None, "fiducial"])
def test_emulator_fallback_lya_block_scales_as_inverse_sqrt_n(sqrtn_ref):
    halved_variance = 2.0
    exp = _make_exp("dr1", likelihood_mode="emulator",
                    emulator_covariance="fiducial", emulator_sqrtn_ref=sqrtn_ref)
    _as_fallback(exp)
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


# ---------------------------------------------------------------- Lya_QSO tables
_TABLE_PATH = os.path.join(
    os.environ.get("SCRATCH", ""), "bedcosmo", "num_tracers", "emulator", "bao", "models",
    "dr1", "base", "lya", "v1_ratio_to_nominal", "Lya_QSO.npz")


def _as_fallback(exp):
    """Make Lya_QSO a null-checkpoint bin, whatever emulators.yaml gives it."""
    exp._emulator_tables.pop("Lya_QSO", None)
    if "Lya_QSO" not in exp._emulator_fallback_bins:
        exp._emulator_fallback_bins.append("Lya_QSO")


def _write_table(path, **over):
    fields = dict(kind="table", output="ratio_to_nominal", tracer_bin="Lya_QSO",
                  param_names=np.array(["N_tracers"]),
                  target_names=np.array(["ratio_sigma_DH_over_rd", "ratio_sigma_DM_over_rd", "delta_rho"]),
                  n_nominal=709_565.0, log_n_ratio=np.array([-1.0, 0.0, 1.0]),
                  values=np.array([[2.0, 3.0, 0.1], [1.0, 1.0, 0.0], [0.5, 0.4, -0.05]]),
                  provenance="test")
    fields.update(over)
    np.savez(path, **fields)
    return path


def _with_table(exp, path):
    exp._emulator_fallback_bins = [b for b in exp._emulator_fallback_bins if b != "Lya_QSO"]
    exp._emulator_tables["Lya_QSO"] = exp._load_emulator_table("Lya_QSO", str(path))
    return exp


def _lya_block(exp, n_lya):
    rows = exp._lya_qso_rows
    q = exp.desi_data.loc[rows, "quantity"].tolist()
    dh, dm = rows[q.index("DH_over_rs")], rows[q.index("DM_over_rs")]
    cov = torch.zeros(len(exp.desi_data), len(exp.desi_data), dtype=torch.float64)
    ref = torch.as_tensor(exp.ref_cov, dtype=torch.float64)
    exp._fill_table_block(cov, "Lya_QSO", {"Lya_QSO": torch.tensor(float(n_lya))}, ref)
    s_dh, s_dm = cov[dh, dh].sqrt(), cov[dm, dm].sqrt()
    r_dh, r_dm = ref[dh, dh].sqrt(), ref[dm, dm].sqrt()
    return (float(s_dh / r_dh), float(s_dm / r_dm),
            float(cov[dh, dm] / (s_dh * s_dm) - ref[dh, dm] / (r_dh * r_dm)))


@pytest.mark.skipif(not os.path.isdir(_CKPT_DIR),
                    reason="bao dr1/base fourier emulator checkpoints not available")
def test_table_rescales_the_reference_block_per_component(tmp_path):
    exp = _with_table(_make_exp("dr1", likelihood_mode="emulator", emulator_covariance="fiducial"),
                      _write_table(tmp_path / "t.npz"))
    assert _lya_block(exp, 709_565) == pytest.approx((1.0, 1.0, 0.0), abs=1e-12)
    # halfway between the first two rows in log(n_ratio)
    got = _lya_block(exp, 709_565 * np.exp(-0.5))
    assert got == pytest.approx((1.5, 2.0, 0.05), rel=1e-9)
    # outside the grid: clamped to the edge
    assert _lya_block(exp, 709_565 * np.exp(-3)) == pytest.approx((2.0, 3.0, 0.1), rel=1e-9)


@pytest.mark.skipif(not os.path.isdir(_CKPT_DIR),
                    reason="bao dr1/base fourier emulator checkpoints not available")
@pytest.mark.parametrize("accounting", ["outside", "in_budget_fixed"])
def test_table_is_held_nominal_outside_qso_coupled(accounting, tmp_path):
    exp = _with_table(_make_exp("dr1", likelihood_mode="emulator", emulator_covariance="fiducial",
                                lya_accounting=accounting), _write_table(tmp_path / "t.npz"))
    assert _lya_block(exp, 709_565 * np.exp(-0.5)) == (1.0, 1.0, 0.0)


@pytest.mark.skipif(not os.path.isdir(_CKPT_DIR),
                    reason="bao dr1/base fourier emulator checkpoints not available")
@pytest.mark.parametrize("bad, match", [
    (dict(kind="network"), "unknown emulator file kind"),
    (dict(output="absolute"), "not supported"),
    (dict(n_nominal=1_289_874.0), "n_nominal"),
    (dict(tracer_bin="QSO"), "slot is"),
])
def test_table_metadata_is_enforced(bad, match, tmp_path):
    exp = _make_exp("dr1", likelihood_mode="emulator", emulator_covariance="fiducial")
    with pytest.raises(ValueError, match=match):
        exp._load_emulator_table("Lya_QSO", str(_write_table(tmp_path / "t.npz", **bad)))


@pytest.mark.skipif(not os.path.isdir(_CKPT_DIR),
                    reason="bao dr1/base fourier emulator checkpoints not available")
def test_checkpoint_declaring_a_non_absolute_output_is_refused(tmp_path):
    exp = _make_exp("dr1", likelihood_mode="emulator", emulator_covariance="fiducial")
    path = tmp_path / "x.pt"
    torch.save({"output": "ratio_to_nominal"}, path)
    with pytest.raises(ValueError, match="output='absolute'"):
        exp._load_emulator(str(path))


@pytest.mark.skipif(not (os.path.isdir(_CKPT_DIR) and os.path.exists(_TABLE_PATH)),
                    reason="bao dr1/base emulators or the deployed Lya_QSO table not available")
def test_deployed_table_is_loaded_and_halving_qso_follows_it():
    exp = _make_exp("dr1", likelihood_mode="emulator", emulator_covariance="fiducial")
    assert "Lya_QSO" in exp._emulator_tables and "Lya_QSO" not in exp._emulator_fallback_bins
    nominal, halved = _halve_qso(exp)
    rows = exp._lya_qso_rows
    block = np.ix_(rows, rows)
    cov_nom = exp._build_emulator_covariance(exp.calc_passed(nominal), {})[0].numpy()
    # float32 design arithmetic puts nominal N at 709,565 (1 +- 1e-7), as for the law test
    np.testing.assert_allclose(cov_nom[block], exp.ref_cov[block], rtol=1e-6)
    t = np.load(_TABLE_PATH)
    i = int(np.argmin(np.abs(t["log_n_ratio"] - np.log(0.5))))
    r_dh, r_dm, _ = _lya_block(exp, _lya_n(exp, halved))
    # the grid is 0.05 dex, so n_ratio = 0.5 sits between points: tolerance of one step
    assert r_dh == pytest.approx(t["values"][i, 0], rel=0.03)
    assert r_dm == pytest.approx(t["values"][i, 1], rel=0.03)
    assert r_dm > r_dh > 1.0


@pytest.mark.skipif(not (os.path.isdir(_CKPT_DIR) and os.path.exists(_TABLE_PATH)),
                    reason="bao dr1/base emulators or the deployed Lya_QSO table not available")
def test_run_snapshot_keeps_the_table_and_a_reload_finds_it(tmp_path):
    """create_run snapshots Lya_QSO as a .npz (not renamed .pt) and NumTracers reloads it."""
    import importlib.util
    from types import SimpleNamespace

    import yaml

    from bedcosmo.util import get_experiment_config_path

    spec = importlib.util.spec_from_file_location(
        "create_run", os.path.join(os.path.dirname(__file__), "..", "scripts", "create_run.py"))
    create_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(create_run)
    args = SimpleNamespace(cosmo_exp="num_tracers", likelihood_mode="emulator", analysis="bao",
                           cosmo_model="base", dataset="dr1", emulator_space="fourier")
    create_run._snapshot_emulators(args, str(tmp_path))
    assert (tmp_path / "emulators" / "Lya_QSO.npz").exists()
    assert not (tmp_path / "emulators" / "Lya_QSO.pt").exists()

    def load(name):
        with open(get_experiment_config_path("num_tracers", name)) as f:
            return yaml.safe_load(f)

    run_obj = SimpleNamespace(info=SimpleNamespace(artifact_uri=f"file://{tmp_path}"))
    run_args = dict(cosmo_exp="num_tracers", dataset="dr1", analysis="bao", cosmo_model="base",
                    likelihood_mode="emulator", emulator_space="fourier",
                    emulator_covariance="fiducial", device="cpu", mode="eval",
                    lya_accounting="qso_coupled")
    exp = init_experiment(run_obj=run_obj, run_args=run_args, design_args=load(DESIGN_ARGS["dr1"]),
                          prior_args=load("prior_args_hrdrag.yaml"))
    assert "Lya_QSO" in exp._emulator_tables
    assert exp._emulator_tables["Lya_QSO"]["path"].startswith(str(tmp_path))


# ---------------------------------------------------------------- design-tool bounds
def test_design_default_bounds_outside_are_unchanged():
    from bedcosmo.num_tracers import design

    lower, upper = design.default_bounds("dr1", "outside")
    assert lower == design.DEFAULT_LOWER and upper == design.DEFAULT_UPPER


@pytest.mark.parametrize("accounting", ["qso_coupled", "in_budget_fixed"])
def test_design_default_bounds_keep_each_class_relative_to_nominal(accounting):
    """Converted bounds sit at the same multiple of nominal; the caps stay availability."""
    _skip_without("dr1")
    from bedcosmo.num_tracers import design

    lower, upper = map(np.asarray, design.default_bounds("dr1", accounting))
    nom_out = design.nominal_vector("dr1", "outside")
    nom = design.nominal_vector("dr1", accounting)
    np.testing.assert_allclose(upper / nom, np.asarray(design.DEFAULT_UPPER) / nom_out, rtol=1e-12)
    keep = [0, 1, 3]   # ELG's lower is snapped onto the sum lattice
    np.testing.assert_allclose((lower / nom)[keep], (np.asarray(design.DEFAULT_LOWER) / nom_out)[keep],
                               rtol=1e-12)
    elg = design.LABELS.index("ELG")
    assert 0 <= design.DEFAULT_LOWER[elg] * nom[elg] / nom_out[elg] - lower[elg] < 0.01
    # sum 1 on the lattice: (1 - sum(lower)) is a whole number of 0.01 steps
    gap = (1 - lower.sum()) / 0.01
    assert gap == pytest.approx(round(gap), abs=1e-9)
    # the availability caps are NumTracers' own (num_targets over the budget)
    exp = _make_exp("dr1", lya_accounting=accounting)
    avail = (exp.num_targets / exp.nominal_total_obs).values
    i = [design.LABELS.index(c) for c in ("BGS", "LRG", "QSO")]
    np.testing.assert_allclose(upper[i], avail[i], rtol=2e-3)   # DEFAULT_UPPER is rounded to 4 d.p.


def test_design_coupled_pool_reaches_past_nominal_qso():
    _skip_without("dr1")
    from bedcosmo.num_tracers import design

    pool = design.budget_pool(sum_lower=1.0, sum_upper=1.0, dataset="dr1",
                              lya_accounting="qso_coupled", include_nominal_scales=[1.0])
    nom = design.nominal_vector("dr1", "qso_coupled")
    qso = design.LABELS.index("QSO")
    assert pool.shape[0] > 50
    assert pool[:, qso].max() > nom[qso]
    np.testing.assert_allclose(pool.sum(1), 1.0, atol=1e-9)
