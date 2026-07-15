"""Regression tests for NumTracers passed N_tracers (calc_passed + _passed_ratio_to_n_tracers).

The BAO emulator's ``N_tracers`` feature is the redshift-confirmed *passed* count
(desilike_emulator ``_get_ntracers`` reads the ``passed`` column). bedcosmo must feed
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
