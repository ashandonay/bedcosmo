"""NumTracers with analysis='shapefit': per-tracer mean + covar emulators.

Locks in:
  1. N_tracers per bin follows desilike-emulator's shapefit bins: at the nominal design
     it equals ``util.ntracers(bin)``, and LRG3 is LRG-only (no ELG1 contribution).
  2. The likelihood is a 24-dim Gaussian (6 bins x [qiso, qap, f_sigmar, m]) with a
     block-diagonal, positive-definite covariance.
  3. The 4x4 correlation projection repairs a non-PD block and leaves a PD one alone.
  4. Prior draws stay inside the emulators' Omega_m training domain.

Needs the v4 shapefit checkpoints under $SCRATCH; skipped when they are absent.
"""
import os

import pytest
import torch

from bedcosmo.util import init_experiment

_CKPT_DIR = os.path.join(
    os.environ.get("SCRATCH", ""), "bedcosmo", "num_tracers", "emulator", "shapefit",
    "models", "dr1", "base")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_CKPT_DIR), reason="shapefit dr1/base emulator checkpoints not available")


def _make_exp(**overrides):
    kwargs = {
        "cosmo_exp": "num_tracers",
        "prior_args_path": "prior_args_shapefit.yaml",
        "design_args_path": "design_args_dr1.yaml",
        "dataset": "dr1",
        "analysis": "shapefit",
        "cosmo_model": "base",
        "likelihood_mode": "emulator",
        "device": "cpu",
        "mode": "eval",
    }
    kwargs.update(overrides)
    return init_experiment(**kwargs)


@pytest.fixture(scope="module")
def exp():
    return _make_exp()


def test_bins_and_context(exp):
    assert exp.shapefit_bins == ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]
    assert exp.shapefit_quantities == ["qiso", "qap", "f_sigmar", "m"]
    assert exp.central_val.shape == (24,)
    assert exp.context_dim == len(exp.design_labels) + 24


def test_nominal_n_tracers_match_desilike(exp):
    from desilike_emulator.util import ntracers

    n = exp._shapefit_n_tracers(exp.nominal_design.view(1, -1))[0]
    for i, tracer_bin in enumerate(exp.shapefit_bins):
        assert float(n[i]) == pytest.approx(ntracers(tracer_bin, "dr1"), rel=1e-6), tracer_bin


def test_lrg3_excludes_elg1(exp):
    design = exp.nominal_design.to(torch.float64).clone()
    more_elg = design.clone()
    more_elg[exp.design_labels.index("ELG")] *= 1.5
    n, n_elg = exp._shapefit_n_tracers(torch.stack([design, more_elg]))
    i_lrg3 = exp.shapefit_bins.index("LRG3")
    i_elg2 = exp.shapefit_bins.index("ELG2")
    assert float(n_elg[i_lrg3]) == pytest.approx(float(n[i_lrg3]), rel=1e-12)
    assert float(n_elg[i_elg2]) == pytest.approx(1.5 * float(n[i_elg2]), rel=1e-12)


def test_pyro_model_shape_and_block_diagonal_pd_covariance(exp):
    designs = exp.designs[:4].unsqueeze(0).expand(8, -1, -1)
    y = exp.pyro_model(designs)
    assert y.shape == (8, 4, 24)
    assert torch.isfinite(y).all()

    n_tracers = exp._shapefit_n_tracers(designs)
    parameters = exp.sample_parameters(n_tracers.shape[:-1])
    mean, cov = exp._shapefit_likelihood(n_tracers, parameters)
    assert mean.shape == (8, 4, 24)
    assert cov.shape == (8, 4, 24, 24)
    assert (torch.linalg.cholesky_ex(cov).info == 0).all()
    block = torch.block_diag(*[torch.ones(4, 4)] * 6).bool()
    assert (cov[..., ~block] == 0).all()


def test_cov_block_projects_non_pd_correlation(exp):
    sigma = torch.tensor([[0.01, 0.02, 0.03, 0.04]], dtype=torch.float64)
    # Pairwise-valid rho whose 4x4 is not PD: 1-2 and 1-3 strongly correlated, 2-3 anti.
    bad = torch.tensor([[0.9, 0.9, 0.0, -0.9, 0.0, 0.0]], dtype=torch.float64)
    cov = exp._shapefit_cov_block(sigma, bad)
    assert torch.linalg.eigvalsh(cov).min() > 0
    assert torch.allclose(torch.diagonal(cov, dim1=-2, dim2=-1), sigma ** 2)

    good = torch.tensor([[0.3, -0.2, 0.1, 0.4, -0.1, 0.2]], dtype=torch.float64)
    cov = exp._shapefit_cov_block(sigma, good)
    i, j = torch.triu_indices(4, 4, offset=1)
    rho = cov[0, i, j] / (sigma[0, i] * sigma[0, j])
    assert torch.allclose(rho, good[0])


def test_prior_draws_respect_omega_m_domain(exp):
    params = exp.sample_parameters((20000,))
    omega_m = (params["omega_cdm"] + params["omega_b"] + exp._OMEGA_NU_FID) / params["h"] ** 2
    assert omega_m.min() >= 0.01 and omega_m.max() <= 0.99


def test_central_sample_data_shape(exp):
    y = exp.sample_data(exp.nominal_design.view(1, -1), num_samples=5, central=True)
    assert y.shape == (5, 1, 24)


def test_get_nominal_samples_not_available(exp):
    with pytest.raises(NotImplementedError):
        exp.get_nominal_samples()


@pytest.mark.parametrize("override", [
    {"likelihood_mode": "scaling"},
    {"vary_z_eff": True},
    {"emulator_sqrtn_ref": "sampled"},
])
def test_bao_only_options_rejected(override):
    with pytest.raises(ValueError):
        _make_exp(**override)
