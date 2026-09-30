"""Constrained priors record their density in the Pyro trace.

In num_tracers (bao) and variable_redshift, the constrained pairs (Om, Ok) and (w0, wa)
are drawn by ConstrainedUniform2D and registered as PresampledPrior sites. The trace's prior log-prob, which the EIG's H_prior is
computed from, must be the pair's joint density -log(area of the allowed region), not 0.
"""
import math
import os

import pyro
import pytest
import torch
from pyro import poutine

from bedcosmo.util import init_experiment

_DATA = os.path.join(os.environ.get("HOME", ""), "data", "desi", "bao_dr1")

pytestmark = pytest.mark.skipif(not os.path.isdir(_DATA), reason="desi bao_dr1 data not available")

N = 20_000


def _trace(exp):
    def model():
        with pyro.plate("p", N):
            return exp.sample_parameters((N,))

    tr = poutine.trace(model).get_trace()
    tr.compute_log_prob()
    return tr


def _mc_area(exp, a, b, bounds, n=4_000_000):
    """Area of the allowed region, by rejection against the independent uniform box."""
    x = exp.prior[a].sample((n,)).double()
    y = exp.prior[b].sample((n,)).double()
    s = x + y
    ok = torch.ones(n, dtype=torch.bool)
    if "lower" in bounds:
        ok &= s > bounds["lower"]
    if "upper" in bounds:
        ok &= s < bounds["upper"]
    box = float(exp.prior[a].high - exp.prior[a].low) * float(exp.prior[b].high - exp.prior[b].low)
    return box * ok.double().mean().item()


_EXPERIMENT_ARGS = {
    "num_tracers": {
        "design_args_path": "design_args_dr1.yaml",
        "dataset": "dr1",
        "analysis": "bao",
        "likelihood_mode": "scaling",
    },
    "variable_redshift": {"design_args_path": "design_args.yaml"},
}


@pytest.mark.parametrize("cosmo_exp", ["num_tracers", "variable_redshift"])
@pytest.mark.parametrize("cosmo_model, constraint, pair", [
    ("base_omegak", "valid_densities", ("Om", "Ok")),
    ("base_w_wa", "high_z_matter_dom", ("w0", "wa")),
])
def test_constrained_pair_records_uniform_region_density(cosmo_exp, cosmo_model, constraint, pair):
    exp = init_experiment(
        cosmo_exp=cosmo_exp,
        prior_args_path="prior_args_hrdrag.yaml",
        cosmo_model=cosmo_model,
        device="cpu",
        mode="eval",
        **_EXPERIMENT_ARGS[cosmo_exp],
    )
    tr = _trace(exp)
    recorded = tr.nodes[pair[0]]["log_prob"] + tr.nodes[pair[1]]["log_prob"]
    area = _mc_area(exp, *pair, exp.param_constraints[constraint]["bounds"])
    assert torch.allclose(recorded, torch.full_like(recorded, -math.log(area)), atol=2e-3)
