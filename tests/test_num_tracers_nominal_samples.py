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
    from bedcosmo.util import ReferenceChain
    assert isinstance(gd, ReferenceChain) and gd.smooth_scale_2D == -1
    assert gd.getParamNames().list() == ["Om", "hrdrag"]


def test_reference_chain_keeps_automatic_smoothing_through_restrict_and_plot():
    import contextlib
    import io

    import matplotlib.pyplot as plt
    import numpy as np
    from getdist import MCSamples

    from bedcosmo.plotting import BasePlotter
    from bedcosmo.util import (
        GETDIST_CHAIN_SETTINGS,
        GETDIST_SETTINGS,
        ReferenceChain,
        restrict_mcsamples,
    )

    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 3))
    with contextlib.redirect_stdout(io.StringIO()):
        chain = ReferenceChain(samples=x, names=["a", "b", "c"], label="DESI DR1 BAO",
                               settings=GETDIST_CHAIN_SETTINGS)
        flow = MCSamples(samples=x, names=["a", "b", "c"], label="NF")

    sub = restrict_mcsamples(chain, ["a", "b"])
    assert isinstance(sub, ReferenceChain)
    assert sub.label == "DESI DR1 BAO"
    assert sub.smooth_scale_2D == GETDIST_CHAIN_SETTINGS["smooth_scale_2D"] == -1

    g = BasePlotter(cosmo_exp="test_exp").plot_triangle([flow, chain], ["tab:blue", "black"])
    assert chain.smooth_scale_2D == -1
    assert flow.smooth_scale_2D == GETDIST_SETTINGS["smooth_scale_2D"]
    plt.close(g.fig)
