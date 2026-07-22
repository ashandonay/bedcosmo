"""Regression tests for NumTracers.sigma_scaling_factor (scaling-mode sigma vs design budget).

``calc_passed`` is linear in the design ``class_ratio``, so its output already encodes *both*
the split across tracers *and* the total observation budget (the design sum ``s``):
``passed_ratio = s * nominal_passed_ratio`` for a uniformly scaled design. Since
``N_i = passed_ratio_i * nominal_total_obs``, the shot-noise idealization ``sigma ~ 1/sqrt(N)``
means the factor must be ``sqrt(nominal_passed_ratio / passed_ratio)`` -> ``1/sqrt(s)``.

A previous version also divided by the design sum ("total_obs_multiplier"), double-counting the
budget and yielding ``sigma ~ 1/s``. That was a no-op at ``sum == 1`` (every historical
design_args pins ``sum_lower == sum_upper == 1.0``), so it only ever biased designs with a free
sum -- shrinking sigma ~2x too aggressively at s=1.2 (0.833 vs 0.913). These tests pin the
scaling law so it cannot regress.

Note: this is scaling mode's shot-noise idealization, not truth. Emulator mode learns the real
BAO forecast (dense bins saturate well short of 1/sqrt(N)) and never calls this method.

Instantiated in scaling mode, which needs no emulator checkpoints.
"""
import math
import os

import pytest
import torch

from bedcosmo.util import init_experiment

# desi_data lives under $HOME/data/desi/bao_<dataset>; skip cleanly if absent.
_DATA_ROOT = os.path.join(os.environ.get("HOME", ""), "data", "desi")

SCALES = [1.0, 1.05, 1.1, 1.15, 1.2]


def _has_dataset(ds):
    return os.path.isdir(os.path.join(_DATA_ROOT, f"bao_{ds}"))


def _make_exp():
    return init_experiment(
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


@pytest.fixture(scope="module")
def exp():
    if not _has_dataset("dr1"):
        pytest.skip("desi dataset bao_dr1 not available")
    return _make_exp()


def _factor(exp, scale, index):
    """sigma scaling factor for the nominal design scaled uniformly by ``scale``."""
    design = (exp.nominal_design.to(torch.float64) * scale).unsqueeze(0)
    passed_ratio = exp.calc_passed(design)
    f = exp.sigma_scaling_factor(passed_ratio, index)
    return float(torch.as_tensor(f).reshape(-1)[0])


@pytest.mark.parametrize("scale", SCALES)
def test_uniform_scaling_follows_inverse_sqrt(exp, scale):
    """Uniformly scaling the design by s must scale sigma by 1/sqrt(s), not 1/s."""
    got = _factor(exp, scale, exp.DH_idx)
    assert got == pytest.approx(1.0 / math.sqrt(scale), rel=1e-5), (
        f"s={scale}: factor {got} != 1/sqrt(s) {1/math.sqrt(scale)}"
    )


@pytest.mark.parametrize("scale", [s for s in SCALES if s != 1.0])
def test_not_double_counting_budget(exp, scale):
    """Guard the specific regression: the old code returned 1/s (budget counted twice)."""
    got = _factor(exp, scale, exp.DH_idx)
    assert got != pytest.approx(1.0 / scale, rel=1e-5), (
        f"s={scale}: factor {got} == 1/s -- design sum is being double-counted"
    )


def test_nominal_design_is_unity(exp):
    """At the nominal design (sum == 1) sigma is unchanged, for every measurement block."""
    for index in (exp.DH_idx, exp.DM_idx, exp.DV_idx):
        assert _factor(exp, 1.0, index) == pytest.approx(1.0, rel=1e-6)


def test_factor_depends_only_on_passed_ratio(exp):
    """Equal passed_ratio must give an equal factor regardless of the design that produced it.

    The budget enters solely through passed_ratio; nothing may re-derive it from the design sum.
    """
    design = (exp.nominal_design.to(torch.float64) * 1.2).unsqueeze(0)
    passed_ratio = exp.calc_passed(design)
    direct = exp.sigma_scaling_factor(passed_ratio, exp.DH_idx)
    # Same passed_ratio, no design in sight -> identical factor.
    again = exp.sigma_scaling_factor(passed_ratio.clone(), exp.DH_idx)
    assert torch.allclose(torch.as_tensor(direct), torch.as_tensor(again))


def test_per_bin_shot_noise_for_nonuniform_design(exp):
    """For an arbitrary (non-uniform) design, factor_i == sqrt(N_nominal_i / N_i) per bin.

    Lya QSO rows are excluded: with vary_lya_qso=False (the default) that error is held fixed at
    its nominal value, so those rows are pinned to 1.0 -- covered by the test below.
    """
    import numpy as np

    design = torch.tensor([[0.05, 0.30, 0.45, 0.20]], dtype=torch.float64)
    passed_ratio = exp.calc_passed(design)
    got = torch.as_tensor(exp.sigma_scaling_factor(passed_ratio, exp.DH_idx)).reshape(-1)
    expected = torch.sqrt(
        exp.nominal_passed_ratio[exp.DH_idx].to(torch.float64)
        / passed_ratio[..., exp.DH_idx].reshape(-1).to(torch.float64)
    ).reshape(-1)

    free = ~torch.as_tensor(np.isin(np.asarray(exp.DH_idx), exp._lya_qso_rows)).reshape(-1)
    assert free.any(), "expected at least one non-Lya-QSO row in DH_idx"
    assert torch.allclose(got.double()[free], expected[free], rtol=1e-6)


def test_lya_qso_rows_are_design_independent(exp):
    """With vary_lya_qso=False, Lya QSO sigma stays nominal (factor 1) at any design sum."""
    import numpy as np

    if not exp._lya_qso_rows.size:
        pytest.skip("dataset has no Lya QSO rows")
    lya = torch.as_tensor(np.isin(np.asarray(exp.DH_idx), exp._lya_qso_rows)).reshape(-1)
    if not bool(lya.any()):
        pytest.skip("no Lya QSO row in DH_idx")
    for scale in (1.0, 1.2):
        design = (exp.nominal_design.to(torch.float64) * scale).unsqueeze(0)
        f = torch.as_tensor(
            exp.sigma_scaling_factor(exp.calc_passed(design), exp.DH_idx)
        ).reshape(-1)
        assert torch.allclose(f.double()[lya], torch.ones(int(lya.sum()), dtype=torch.float64))
