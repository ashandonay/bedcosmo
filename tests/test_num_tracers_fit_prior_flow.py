"""fit_prior_flow fits a PriorFlow in the target prior_args' units and records its validation."""

import json
import os

import numpy as np
import pytest

from bedcosmo.num_tracers.fit_prior_flow import _hrdrag_scale, main
from bedcosmo.prior_flow import PriorFlow

_DATA = os.path.join(os.environ.get("HOME", ""), "data", "desi", "bao_dr1")


def test_hrdrag_scale_between_prior_args_units():
    # prior_args.yaml keeps h*r_d in Mpc (multiplier 100), prior_args_hrdrag.yaml in 100 Mpc (10000).
    assert _hrdrag_scale("prior_args.yaml", "prior_args_hrdrag.yaml") == pytest.approx(0.01)
    assert _hrdrag_scale("prior_args.yaml", "prior_args.yaml") == 1.0


@pytest.mark.skipif(not os.path.isdir(_DATA), reason="desi bao_dr1 data not available")
def test_fit_writes_flow_in_target_units_with_validation(tmp_path):
    # Stand-in samples near DR1's base posterior, in the sampler's prior_args.yaml units.
    rng = np.random.default_rng(0)
    samples = rng.multivariate_normal(
        [0.295, 101.8], [[0.015**2, -0.0177], [-0.0177, 1.3**2]], 4000
    )
    meta = {
        "cosmo_model": "base",
        "temperature": 1.0,
        "param_names": ["Om", "hrdrag"],
        "prior_args_path": "prior_args.yaml",
    }
    path = tmp_path / "base_T1.npz"
    np.savez(path, samples=samples, metadata=json.dumps(meta))

    main(
        [
            str(path),
            "--prior-args",
            "prior_args_hrdrag.yaml",
            "--epochs",
            "3",
            "--transforms",
            "2",
            "--bins",
            "4",
            "--hidden",
            "16",
            "16",
            "--threads",
            "2",
        ]
    )

    pf = PriorFlow.load(tmp_path / "base_T1_flow.pt")
    assert pf.feature_names == ["Om", "hrdrag"]
    assert pf.sample(2000, seed=0)[:, 1].mean() == pytest.approx(1.018, abs=0.02)  # 101.8 Mpc / 100
    assert pf.meta["prior_args"] == "prior_args_hrdrag.yaml"
    assert set(pf.meta["validation"]) >= {"ks_max", "importance_ess_fraction", "leak_fraction"}
    assert (tmp_path / "base_T1_flow.png").is_file()
