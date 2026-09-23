"""Tests for VariableRedshift models that don't sample every distance parameter."""

import torch
import yaml

from bedcosmo.util import get_experiment_config_path
from bedcosmo.variable_redshift import VariableRedshift
from bedcosmo.variable_redshift.experiment import PLANCK18_FIDUCIAL


def _load(name):
    with open(get_experiment_config_path("variable_redshift", name)) as f:
        return yaml.safe_load(f)


def _experiment(cosmo_model):
    # prior_args.yaml uses the x100 hrdrag multiplier that the fiducial assumes.
    return VariableRedshift(
        prior_args=_load("prior_args.yaml"),
        design_args=_load("design_args.yaml"),
        cosmo_model=cosmo_model,
        device="cpu",
    )


def test_base_om_holds_hrdrag_at_fiducial():
    exp = _experiment("base_om")
    assert exp.cosmo_params == ["Om"]
    assert exp.fixed_params == {"hrdrag": PLANCK18_FIDUCIAL["hrdrag"]}

    designs = exp.designs[:3].unsqueeze(0).expand(4, -1, -1)
    y = exp.pyro_model(designs)
    assert y.shape == (4, 3, exp.observation_dim)
    assert torch.isfinite(y).all()


def test_base_om_distances_match_base_at_fiducial_hrdrag():
    base_om = _experiment("base_om")
    base = _experiment("base")
    z = torch.tensor([0.5, 1.5], dtype=torch.float64)
    Om = torch.tensor([[0.3]], dtype=torch.float64)
    hrdrag = torch.tensor([[PLANCK18_FIDUCIAL["hrdrag"]]], dtype=torch.float64)

    torch.testing.assert_close(
        base_om.D_H_func(z, Om=Om, **base_om.fixed_params),
        base.D_H_func(z, Om=Om, hrdrag=hrdrag),
    )
    torch.testing.assert_close(base_om.central_val, base.central_val)
