"""NumTracers ``vary_z_eff``: BAO means at the design's effective redshift.

With ``vary_z_eff`` the mean of each tracer bin is evaluated at
``desilike_emulator.util.effective_redshift(bin, N)`` for the design's passed
``N_tracers``, rather than at desi_data.csv's fixed DR1 ``z``. The emulator's sigma
labels are defined at that redshift, so this keeps mean and covariance consistent.

The helper is exercised on a scaling-mode instance (no emulator checkpoints needed)
with the flag switched on after construction; the constructor's own validation, which
restricts the flag to emulator mode on DR1, is tested separately.
"""
import os

import pytest
import torch

from bedcosmo.util import init_experiment

_DATA_ROOT = os.path.join(os.environ.get("HOME", ""), "data", "desi")
QUANTITIES = ("DH_over_rs", "DM_over_rs", "DV_over_rs")


def _has_dataset(ds):
    return os.path.isdir(os.path.join(_DATA_ROOT, f"bao_{ds}"))


def _make_exp(**overrides):
    kwargs = dict(
        cosmo_exp="num_tracers",
        prior_args_path="prior_args_hrdrag.yaml",
        design_args_path="design_args_dr1.yaml",
        dataset="dr1",
        analysis="bao",
        cosmo_model="base",
        likelihood_mode="scaling",
        include_D_M=True,
        include_D_V=True,
        device="cpu",
        mode="eval",
    )
    kwargs.update(overrides)
    return init_experiment(**kwargs)


@pytest.fixture(scope="module")
def exp():
    if not _has_dataset("dr1"):
        pytest.skip("desi dataset bao_dr1 not available")
    pytest.importorskip("desilike_emulator.util")
    e = _make_exp()
    e.vary_z_eff = True
    return e


def _passed(exp, scale=1.0, batch=(1,)):
    design = exp.nominal_design.to(torch.float64) * scale
    return exp.calc_passed(design.expand(*batch, design.shape[-1]).clone())


def _rows(exp, quantity):
    return exp.desi_data[exp.desi_data["quantity"] == quantity]


def test_default_is_off():
    if not _has_dataset("dr1"):
        pytest.skip("desi dataset bao_dr1 not available")
    e = _make_exp()
    assert e.vary_z_eff is False and e._effective_redshift is None
    z = e._mean_z_eff(_passed(e))["DH_over_rs"]
    assert torch.equal(z, torch.tensor(_rows(e, "DH_over_rs")["z"].to_list()))


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_nominal_design_reproduces_desi_z(exp, quantity):
    """At the nominal design N is DR1's, where z_eff matches DESI's published z."""
    z = exp._mean_z_eff(_passed(exp))[quantity]
    fixed = torch.tensor(_rows(exp, quantity)["z"].to_list(), dtype=z.dtype)
    assert z.shape == (1, len(fixed))
    torch.testing.assert_close(z[0], fixed, rtol=1.5e-3, atol=0)


@pytest.mark.parametrize("quantity", QUANTITIES)
def test_matches_effective_redshift_off_nominal(exp, quantity):
    from desilike_emulator.util import effective_redshift

    passed = _passed(exp, scale=0.6)
    z = exp._mean_z_eff(passed)[quantity]
    n = exp._passed_ratio_to_n_tracers(passed)
    desi_to_bin = {v: k for k, v in exp._EMULATOR_TRACER_TO_DESI.items()}
    for i, tracer in enumerate(_rows(exp, quantity)["tracer"].to_list()):
        if tracer in exp._FIXED_Z_EFF_TRACERS:
            continue
        b = desi_to_bin[tracer]
        assert float(z[0, i]) == pytest.approx(float(effective_redshift(b, n[b])), rel=1e-12)


def test_lya_rows_stay_fixed(exp):
    rows = _rows(exp, "DH_over_rs")
    lya = [i for i, t in enumerate(rows["tracer"].to_list()) if t in exp._FIXED_Z_EFF_TRACERS]
    if not lya:
        pytest.skip("no Lya QSO rows in this data vector")
    for scale in (0.6, 1.0, 1.3):
        z = exp._mean_z_eff(_passed(exp, scale))["DH_over_rs"]
        for i in lya:
            assert float(z[0, i]) == pytest.approx(float(rows["z"].iloc[i]), abs=1e-12)


@pytest.mark.parametrize("batch", [(1,), (5,), (3, 2)])
def test_batch_shapes_feed_distance_functions(exp, batch):
    passed = _passed(exp, batch=batch)
    z = exp._mean_z_eff(passed)["DM_over_rs"]
    assert z.shape == batch + (len(_rows(exp, "DM_over_rs")),)
    params = {"Om": torch.full(batch + (1,), 0.3152, dtype=torch.float64),
              "hrdrag": torch.full(batch + (1,), 99.08, dtype=torch.float64)}
    d = exp.D_M_func(z, **params)
    assert d.shape == z.shape and torch.all(torch.isfinite(d))


def test_gradient_flows_through_design(exp):
    design = exp.nominal_design.to(torch.float64).unsqueeze(0).clone().requires_grad_(True)
    z = exp._mean_z_eff(exp.calc_passed(design))["DH_over_rs"]
    z.sum().backward()
    assert torch.all(torch.isfinite(design.grad)) and torch.any(design.grad != 0)


def test_requires_emulator_mode():
    if not _has_dataset("dr1"):
        pytest.skip("desi dataset bao_dr1 not available")
    with pytest.raises(ValueError, match="likelihood_mode='emulator'"):
        _make_exp(vary_z_eff=True)


def test_rejects_non_dr1():
    if not _has_dataset("dr2"):
        pytest.skip("desi dataset bao_dr2 not available")
    with pytest.raises(ValueError, match="DR1-only"):
        _make_exp(vary_z_eff=True, likelihood_mode="emulator", dataset="dr2",
                  design_args_path="design_args_dr2.yaml")
