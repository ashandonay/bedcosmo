"""NumTracers (bao) DESI reference posterior: loaded with its dataset-specific legend label."""
import os

import pytest

from bedcosmo.util import init_experiment

DESIGN_ARGS = {"dr1": "design_args_dr1.yaml", "dr2": "design_args_dr2.yaml"}


@pytest.mark.parametrize("ds", ["dr1", "dr2"])
def test_bao_reference_is_labelled_by_dataset(ds):
    chain = os.path.join(os.environ.get("HOME", ""), "data", "desi", f"bao_{ds}", "mcmc_samples", "base.npy")
    if not os.path.exists(chain):
        pytest.skip(f"DESI bao_{ds} reference chain not available")
    exp = init_experiment(
        cosmo_exp="num_tracers",
        prior_args_path="prior_args_hrdrag.yaml",
        design_args_path=DESIGN_ARGS[ds],
        dataset=ds,
        analysis="bao",
        cosmo_model="base",
        likelihood_mode="scaling",
        device="cpu",
        mode="eval",
    )
    gd = exp.get_nominal_samples(num_samples=1000)
    assert gd.label == f"DESI {ds.upper()} BAO"
    assert gd.getParamNames().list() == ["Om", "hrdrag"]
