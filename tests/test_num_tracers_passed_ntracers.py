"""Regression tests for NumTracers passed N_tracers (calc_passed + _passed_ratio_to_n_tracers).

The BAO emulator's ``N_tracers`` feature is the redshift-confirmed *passed* count
(tracer_forecast_emulator ``_get_ntracers`` reads the ``passed`` column). bedcosmo must feed
the emulator passed counts, not observed. The passed *ratio* comes from the single shared
``calc_passed`` computation (``passed = observed * efficiency``); ``_passed_ratio_to_n_tracers``
converts that ratio to an absolute per-bin count with ``* nominal_total_obs``.

These tests lock in two properties:
  1. Nominal identity: at the nominal design, per-bin passed N == ``desi_data["passed"]``
     (i.e. the emulator's own ``_get_ntracers`` values).
  2. Off-nominal merged bin: the LRG3+ELG1 blend is design-dependent
     (``LRG3*eff_LRG3 + ELG1*eff_ELG1``), which a fixed per-bin efficiency would get wrong.

Instantiated in scaling mode: the passed-tracer computation is mode-independent and this
avoids requiring emulator checkpoints.
"""
import os

import pytest
import torch

from bedcosmo.util import init_experiment

DATASETS = ["dr1", "dr2"]
DESIGN_ARGS = {"dr1": "design_args_dr1.yaml", "dr2": "design_args_dr2.yaml"}

# desi_data lives under $HOME/data/desi/bao_<dataset>; skip cleanly if absent.
_DATA_ROOT = os.path.join(os.environ.get("HOME", ""), "data", "desi")


def _has_dataset(ds):
    return os.path.isdir(os.path.join(_DATA_ROOT, f"bao_{ds}"))


def _passed_n_tracers(exp, class_ratio):
    """passed N_tracers per emulator bin: calc_passed (ratio) -> _passed_ratio_to_n_tracers."""
    return exp._passed_ratio_to_n_tracers(exp.calc_passed(class_ratio))


def _make_exp(ds):
    return init_experiment(
        cosmo_exp="num_tracers",
        prior_args_path="prior_args_hrdrag.yaml",
        design_args_path=DESIGN_ARGS[ds],
        dataset=ds,
        analysis="bao",
        cosmo_model="base",
        likelihood_mode="scaling",
        include_D_M=True,
        include_D_V=True,
        device="cpu",
        mode="eval",
    )


@pytest.mark.parametrize("ds", DATASETS)
def test_nominal_passed_matches_desi_passed(ds):
    """At the nominal design, passed N per bin == desi_data['passed'] (emulator's _get_ntracers)."""
    if not _has_dataset(ds):
        pytest.skip(f"desi dataset bao_{ds} not available")
    exp = _make_exp(ds)
    nd = exp.nominal_design.to(torch.float64).view(1, 1, len(exp.design_labels))
    passed = _passed_n_tracers(exp, nd)

    for tracer_bin, desi_name in exp._EMULATOR_TRACER_TO_DESI.items():
        rows = exp.desi_data.index[exp.desi_data["tracer"] == desi_name].tolist()
        if not rows:
            # Bin absent from this dataset's data vector: must not appear in the output.
            assert tracer_bin not in passed
            continue
        expected = float(exp.desi_data.loc[rows[0], "passed"])
        got = float(passed[tracer_bin].squeeze())
        assert got == pytest.approx(expected, rel=1e-4), (
            f"{ds} {tracer_bin}: passed N {got} != desi_data passed {expected}"
        )


@pytest.mark.parametrize("ds", DATASETS)
def test_offnominal_merged_bin_uses_design_dependent_blend(ds):
    """LRG3+ELG1 passed count = LRG3*eff_LRG3 + ELG1*eff_ELG1 for an off-nominal LRG/ELG mix."""
    if not _has_dataset(ds):
        pytest.skip(f"desi dataset bao_{ds} not available")
    exp = _make_exp(ds)
    dt = exp.desi_tracers

    def observed_frac(cls):
        s = dt.loc[dt["class"] == cls]["observed"]
        return (s / s.sum()).values

    lrg_f = observed_frac("LRG")
    elg_f = observed_frac("ELG")
    eff_lrg3 = float(dt.loc[dt["tracer"] == "LRG3", "efficiency"].values[0])
    eff_elg1 = float(dt.loc[dt["tracer"] == "ELG1", "efficiency"].values[0])
    total = exp.nominal_total_obs

    labels = list(exp.design_labels)
    i_lrg, i_elg = labels.index("LRG"), labels.index("ELG")

    # A design whose LRG/ELG ratio differs from nominal (so a fixed 0.828 blend would be wrong).
    design = torch.tensor([[[0.05, 0.30, 0.45, 0.20]]], dtype=torch.float64)
    got = float(_passed_n_tracers(exp, design)["LRG3_ELG1"].squeeze())
    expected = (
        design[0, 0, i_lrg].item() * lrg_f[2] * eff_lrg3
        + design[0, 0, i_elg].item() * elg_f[0] * eff_elg1
    ) * total
    assert got == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("shape", [(1, 4), (5, 4), (3, 2, 4)])
def test_arbitrary_batch_shapes(shape):
    """passed N_tracers (via calc_passed) accepts arbitrary leading batch dims."""
    if not _has_dataset("dr1"):
        pytest.skip("desi dataset bao_dr1 not available")
    exp = _make_exp("dr1")
    x = torch.rand(*shape, dtype=torch.float64)
    x = x / x.sum(-1, keepdim=True)
    out = _passed_n_tracers(exp, x)
    assert out["BGS"].shape == shape[:-1]


def _lya_passed(exp, class_ratio):
    return float(_passed_n_tracers(exp, class_ratio)["Lya_QSO"].squeeze())


@pytest.mark.parametrize("ds", DATASETS)
def test_lya_counts_are_fixed_outside_the_budget(ds):
    """Lya QSO keeps the dataset's passed count whatever the design, low QSO and budget > 1 included."""
    if not _has_dataset(ds):
        pytest.skip(f"desi dataset bao_{ds} not available")
    exp = _make_exp(ds)
    rows = exp.desi_data.index[exp.desi_data["tracer"] == "Lya QSO"].tolist()
    if not rows:
        pytest.skip("no Lya QSO rows")
    expected = float(exp.desi_data.loc[rows[0], "passed"])
    nominal = exp.nominal_design.to(torch.float64)
    low_qso = nominal.clone()
    i_qso, i_bgs = exp.design_labels.index("QSO"), exp.design_labels.index("BGS")
    low_qso[i_bgs] += low_qso[i_qso] - 0.02
    low_qso[i_qso] = 0.02
    for design in (nominal, low_qso, 1.2 * nominal):
        assert _lya_passed(exp, design.view(1, 1, -1)) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("ds", DATASETS)
def test_qso_class_is_the_qso_bin(ds):
    """The design's whole QSO fraction goes to the QSO bin; the budget excludes Lya fibers."""
    if not _has_dataset(ds):
        pytest.skip(f"desi dataset bao_{ds} not available")
    exp = _make_exp(ds)
    bins = exp.desi_data.drop_duplicates(subset=["tracer"])
    assert exp.nominal_total_obs == int(bins.loc[bins["tracer"] != "Lya QSO", "observed"].sum())
    assert float(exp.nominal_design.sum()) == pytest.approx(1.0)
    eff_qso = float(exp.desi_tracers.loc[exp.desi_tracers["tracer"] == "QSO", "efficiency"].values[0])
    design = torch.tensor([[[0.05, 0.30, 0.45, 0.20]]], dtype=torch.float64)
    got = float(_passed_n_tracers(exp, design)["QSO"].squeeze())
    assert got == pytest.approx(0.20 * eff_qso * exp.nominal_total_obs, rel=1e-6)


def test_lya_in_budget_restores_the_earlier_accounting():
    """lya_in_budget=True: Lya fibers sit in the QSO class and move with it."""
    if not _has_dataset("dr1"):
        pytest.skip("desi dataset bao_dr1 not available")
    exp = init_experiment(
        cosmo_exp="num_tracers", prior_args_path="prior_args_hrdrag.yaml",
        design_args_path=DESIGN_ARGS["dr1"], dataset="dr1", analysis="bao", cosmo_model="base",
        likelihood_mode="scaling", device="cpu", mode="eval", lya_in_budget=True,
    )
    dt = exp.desi_tracers
    qso_share = float(dt.loc[dt["class"] == "QSO", "observed"].sum())  # QSO bin + Lya
    assert float(exp.nominal_design[exp.design_labels.index("QSO")]) == pytest.approx(qso_share)
    bins = exp.desi_data.drop_duplicates(subset=["tracer"])
    assert exp.nominal_total_obs == int(bins["observed"].sum())
    nominal = exp.nominal_design.to(torch.float64)
    halved = nominal.clone()
    i_qso, i_elg = exp.design_labels.index("QSO"), exp.design_labels.index("ELG")
    halved[i_elg] += halved[i_qso] / 2
    halved[i_qso] /= 2
    assert _lya_passed(exp, halved.view(1, 1, -1)) == pytest.approx(
        _lya_passed(exp, nominal.view(1, 1, -1)) / 2, rel=1e-6)


def test_runs_logged_without_the_option_load_with_lya_in_budget(tmp_path):
    """init_experiment reads a run logged before lya_in_budget existed with the old accounting."""
    if not _has_dataset("dr1"):
        pytest.skip("desi dataset bao_dr1 not available")
    from types import SimpleNamespace

    import yaml

    from bedcosmo.util import get_experiment_config_path

    def load(name):
        with open(get_experiment_config_path("num_tracers", name)) as f:
            return yaml.safe_load(f)

    run_obj = SimpleNamespace(info=SimpleNamespace(artifact_uri=f"file://{tmp_path}"))
    run_args = dict(cosmo_exp="num_tracers", dataset="dr1", analysis="bao", cosmo_model="base",
                    likelihood_mode="scaling", device="cpu", mode="eval")
    exp = init_experiment(run_obj=run_obj, run_args=dict(run_args),
                          design_args=load(DESIGN_ARGS["dr1"]), prior_args=load("prior_args_hrdrag.yaml"))
    assert exp.lya_in_budget is True
    exp = init_experiment(run_obj=run_obj, run_args=dict(run_args, lya_in_budget=False),
                          design_args=load(DESIGN_ARGS["dr1"]), prior_args=load("prior_args_hrdrag.yaml"))
    assert exp.lya_in_budget is False


@pytest.mark.parametrize("ds", DATASETS)
def test_passed_map_is_additive_over_sub_tracers(ds):
    """A combined bin is the sum of its sub-tracers' maps, so BAO's LRG3+ELG1 row and
    ShapeFit's LRG3-only bin come from one split."""
    if not _has_dataset(ds):
        pytest.skip(f"desi data for {ds} not available")
    exp = _make_exp(ds)
    W, b = exp._passed_map([["LRG3", "ELG1"], ["LRG3"], ["ELG1"]])
    torch.testing.assert_close(W[0], W[1] + W[2])
    torch.testing.assert_close(b[0], b[1] + b[2])
    row = list(exp.desi_data["tracer"]).index("LRG3+ELG1")
    torch.testing.assert_close(exp._passed_weights[row], W[0].to(exp._passed_weights))


def test_calc_passed_rejects_non_tensor():
    if not _has_dataset("dr1"):
        pytest.skip("desi data for dr1 not available")
    with pytest.raises(TypeError):
        _make_exp("dr1").calc_passed([0.1, 0.3, 0.4, 0.2])
