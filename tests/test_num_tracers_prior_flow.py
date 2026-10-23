"""An unconditional PriorFlow as the num_tracers prior, through ``prior_flow_path``.

The flow is truncated to the prior's box and constraints: samples come from rejection,
and the density is renormalized by the flow's mass inside the support, so it integrates
to 1 there. These checks hold for any flow, so a tiny, barely trained one is enough.
"""

import os

import numpy as np
import pyro
import pytest
import torch
import yaml

from bedcosmo.prior_flow import SPACE_NATIVE, train_prior_flow
from bedcosmo.pyro_oed_src import LikelihoodDataset
from bedcosmo.util import get_experiment_config_path, init_experiment

_DATA = os.path.join(os.environ.get("HOME", ""), "data", "desi", "bao_dr1")

# slow: each experiment draws the bijector's 1e7 CDF samples fresh from the flow.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.isdir(_DATA), reason="desi bao_dr1 data not available"),
]


def _flow_file(tmp_path, feature_names, loc, scale, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc, scale, size=(2000, len(loc)))
    pf = train_prior_flow(
        x,
        space=SPACE_NATIVE,
        feature_names=feature_names,
        epochs=5,
        hidden_features=(16, 16),
        transforms=2,
        bins=6,
        seed=seed,
        verbose=False,
    )
    return str(pf.save(tmp_path / "prior_flow.pt"))


def _experiment(cosmo_model, flow_path):
    with open(get_experiment_config_path("num_tracers", "prior_args.yaml")) as f:
        prior_args = yaml.safe_load(f)
    prior_args["prior_flow_path"] = flow_path
    return init_experiment(
        cosmo_exp="num_tracers",
        prior_args=prior_args,
        cosmo_model=cosmo_model,
        design_args_path="design_args_dr1.yaml",
        dataset="dr1",
        analysis="bao",
        likelihood_mode="scaling",
        device="cpu",
        mode="eval",
    )


@pytest.fixture(scope="module")
def base_experiment(tmp_path_factory):
    # Om centred near its lower bound 0.01, so the truncation removes real mass.
    path = _flow_file(tmp_path_factory.mktemp("flow"), ["Om", "hrdrag"], [0.05, 100.0], [0.05, 5.0])
    return _experiment("base", path)


def test_truncated_density_integrates_to_one(base_experiment):
    exp = base_experiment
    assert exp.prior_flow_metadata["log_acceptance"] < np.log(0.95)
    om = torch.linspace(0.01, 0.99, 801, dtype=torch.float64)
    hr = torch.linspace(60.0, 140.0, 401, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(om, hr, indexing="ij"), dim=-1)
    density = exp.prior_flow_log_prob(grid).exp()
    mass = torch.trapezoid(torch.trapezoid(density, hr, dim=1), om).item()
    assert mass == pytest.approx(1.0, abs=0.01)


def test_samples_stay_inside_support(base_experiment):
    exp = base_experiment
    with pyro.plate("p", 5000):
        params = exp.sample_parameters((5000,))
    om = params["Om"].squeeze(-1)
    assert (om > 0.01).all() and (om < 0.99).all()
    assert om.mean() > 0.05  # rejection lifted the mass cut below Om = 0.01


def test_eig_prior_log_prob_scores_the_truncated_flow(base_experiment):
    exp = base_experiment
    samples = exp._sample_prior_flow(3000).reshape(1000, 3, 2)
    ds = LikelihoodDataset.__new__(LikelihoodDataset)
    ds.experiment, ds.device, ds.particle_batch_size = exp, "cpu", 512
    out = ds._compute_prior_log_probs(samples, trace=None)
    assert out["joint"].shape == (1000, 3)
    torch.testing.assert_close(out["joint"], exp.prior_flow_log_prob(samples))


def test_constraints_truncate_the_flow(tmp_path):
    # w0 + wa < 0 cuts this flow's mass roughly in half.
    path = _flow_file(
        tmp_path, ["Om", "w0", "wa", "hrdrag"], [0.3, -0.5, 0.5, 100.0], [0.02, 0.3, 0.3, 1.0]
    )
    exp = _experiment("base_w_wa", path)
    assert np.exp(exp.prior_flow_metadata["log_acceptance"]) == pytest.approx(0.5, abs=0.1)
    x = exp._sample_prior_flow(5000)
    assert (x[:, 1] + x[:, 2] < 0).all()
    outside = torch.tensor([[0.3, 0.2, 0.2, 100.0]], dtype=torch.float64)
    assert exp.prior_flow_log_prob(outside).item() == -np.inf


def test_flow_over_other_parameters_raises(tmp_path):
    path = _flow_file(tmp_path, ["Om", "w0", "hrdrag"], [0.3, -1.0, 100.0], [0.02, 0.1, 1.0])
    with pytest.raises(ValueError, match="samples"):
        _experiment("base", path)
