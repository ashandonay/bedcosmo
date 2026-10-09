"""NumTracers BAO ``emulator_covariance`` as a cosmology dict: a fixed covariance off DESI's fiducial.

The fixed covariance at a moved fiducial must equal the theta-dependent covariance at
that cosmology (hrdrag given in reported km/s units), and differ from DESI's fiducial.

Needs the BAO fourier emulator checkpoints under $SCRATCH; skipped when they are absent.
"""
import os

import numpy as np
import pytest
import torch

from bedcosmo.util import init_experiment

_CKPT_DIR = os.path.join(
    os.environ.get("SCRATCH", ""), "bedcosmo", "num_tracers", "emulator", "bao",
    "models", "dr1", "base", "fourier")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_CKPT_DIR), reason="bao dr1/base fourier emulator checkpoints not available")


def _make_exp(**overrides):
    return init_experiment(cosmo_exp="num_tracers", prior_args_path="prior_args_hrdrag.yaml",
                           design_args_path="design_args_dr1.yaml", dataset="dr1", analysis="bao",
                           cosmo_model="base", likelihood_mode="emulator", emulator_space="fourier",
                           device="cpu", mode="eval", **overrides)


def test_cosmology_dict_moves_the_fixed_fiducial():
    fixed = _make_exp(emulator_covariance="fiducial")
    moved = _make_exp(emulator_covariance={"Om": 0.36, "hrdrag": 10300.0})
    theta = _make_exp(emulator_covariance="cosmology")
    designs = fixed.designs[:2].double().unsqueeze(0)                  # (1, design, 4)
    passed_ratio = fixed.calc_passed(designs)
    # Raw hrdrag 1.03 x multiplier 1e4 = 10300 km/s, the reported value the dict takes.
    params = {"Om": torch.full((1, 2, 1), 0.36, dtype=torch.float64),
              "hrdrag": torch.full((1, 2, 1), 1.03, dtype=torch.float64)}
    cov_moved = moved._build_emulator_covariance(passed_ratio, params)
    assert torch.allclose(cov_moved, theta._build_emulator_covariance(passed_ratio, params))
    assert not torch.allclose(cov_moved, fixed._build_emulator_covariance(passed_ratio, params))
    assert not torch.allclose(cov_moved[0, 0], cov_moved[0, 1])        # still follows the design


@pytest.mark.parametrize("override, match", [
    ({"emulator_covariance": {"omega_cdm": 0.12}}, "unknown parameters"),
    ({"emulator_covariance": "fixed"}, "must be 'cosmology', 'fiducial'"),
    ({"emulator_covariance": {"Om": 0.28}, "emulator_sqrtn_ref": "sampled"}, "contradicts"),
])
def test_emulator_covariance_validation(override, match):
    with pytest.raises(ValueError, match=match):
        _make_exp(**override)


def test_cli_parses_a_cosmology_dict_and_keeps_the_strings():
    from bedcosmo.util import finalize_train_run_args

    yaml = {"emulator_covariance": "fiducial"}
    run_args = finalize_train_run_args({"emulator_covariance": '{"Om": 0.28, "hrdrag": 10180}'}, yaml)
    assert run_args["emulator_covariance"] == {"Om": 0.28, "hrdrag": 10180}
    assert finalize_train_run_args({"emulator_covariance": "cosmology"}, yaml)["emulator_covariance"] == "cosmology"


def test_covariance_fiducials_figure(tmp_path):
    from bedcosmo.num_tracers.covariance_fiducials import main, nominal_covariances
    from bedcosmo.num_tracers.experiment import NumTracers

    fid = NumTracers._BAO_FIDUCIAL["Om"]
    _, covs = nominal_covariances("bao", "Om", [0.25, fid])
    fixed = _make_exp(emulator_covariance="fiducial")
    passed = fixed.calc_passed(fixed.nominal_design.double().view(1, -1))
    np.testing.assert_allclose(covs[fid], fixed._build_emulator_covariance(passed, {})[0].numpy())
    # Lower Om gives tighter emulated errors; Lya (no emulator) is unchanged.
    assert np.all(np.diag(covs[0.25]) <= np.diag(covs[fid]))

    out = tmp_path / "cov.png"
    main(["--values", "0.25", str(fid), "--out", str(out)])
    assert out.stat().st_size > 0
    for argv in (["--values", "0.25", "0.4"],                      # fiducial missing
                 ["--param", "omega_cdm"],                          # not a BAO parameter
                 ["--param", "hrdrag"]):                            # non-default param needs --values
        with pytest.raises(SystemExit):
            main(argv + ["--out", str(out)])
