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

import numpy as np
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


def test_desi_conversion_null_case():
    from desilike_emulator.shapefit import desi_reference

    from bedcosmo.num_tracers.experiment import NumTracers

    fid = desi_reference.published_fiducial("LRG2")
    at_fid = [fid["DV_over_rd"], fid["DH_over_DM"], fid["f_sigma_s8"], 0.0]
    out = NumTracers.desi_shapefit_to_targets(at_fid, fid, 0.4607)
    np.testing.assert_allclose(out, [1.0, 1.0, 0.4607, 0.0], rtol=1e-12)
    # f_sigmar carries DESI's m-dependence: only the fiducial ratio is applied.
    off = NumTracers.desi_shapefit_to_targets([*at_fid[:2], 1.05 * fid["f_sigma_s8"], 0.06], fid, 0.4607)
    assert off[2] == pytest.approx(0.4607 * 1.05) and off[3] == 0.06


def test_central_val_is_desi_measurement(exp):
    from desilike_emulator.shapefit import desi_reference

    _, measured, _ = desi_reference.datavector("BGS")
    fid = desi_reference.published_fiducial("BGS")
    i = exp.shapefit_bins.index("BGS")
    qiso, qap, _, m = exp.central_val[4 * i:4 * i + 4].tolist()
    assert qiso == pytest.approx(measured[0] / fid["DV_over_rd"])
    assert qap == pytest.approx(measured[1] / fid["DH_over_DM"])
    assert m == pytest.approx(measured[3])


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuSOLVER batch limit is GPU-only")
def test_cov_block_large_batch_on_gpu(exp):
    # Eval's particle batches (50000) exceed cuSOLVER's batched-eigh limit (32768).
    n = 40000
    sigma = torch.full((n, 4), 0.01, dtype=torch.float64, device="cuda")
    rho = torch.zeros(n, 6, dtype=torch.float64, device="cuda")
    rho[0] = torch.tensor([0.9, 0.9, 0.0, -0.9, 0.0, 0.0], dtype=torch.float64)
    cov = exp._shapefit_cov_block(sigma, rho)
    assert (torch.linalg.cholesky_ex(cov).info == 0).all()
    assert torch.equal(cov[1:], torch.diag_embed(sigma[1:] ** 2))


def test_prior_draws_respect_omega_m_domain(exp):
    params = exp.sample_parameters((20000,))
    omega_m = (params["omega_cdm"] + params["omega_b"] + exp._OMEGA_NU_FID) / params["h"] ** 2
    assert omega_m.min() >= 0.01 and omega_m.max() <= 0.99


def test_omega_m_acceptance_matches_quadrature(exp):
    # P(accept) = E_{h, omega_b}[ length of allowed omega_cdm interval ] / omega_cdm range:
    # the cut is linear in omega_cdm at fixed (omega_b, h).
    import numpy as np

    b = exp.param_constraints["omega_m_domain"]["bounds"]
    wc, wb, hp = exp.prior["omega_cdm"], exp.prior["omega_b"], exp.prior["h"]
    wc_lo, wc_hi = float(wc.low), float(wc.high)
    h = np.linspace(float(hp.low), float(hp.high), 200_001)
    x, w = np.polynomial.hermite_e.hermegauss(60)
    omega_b = float(wb.loc) + float(wb.scale) * x
    lo = np.maximum(wc_lo, b["lower"] * h[:, None] ** 2 - omega_b - exp._OMEGA_NU_FID)
    hi = np.minimum(wc_hi, b["upper"] * h[:, None] ** 2 - omega_b - exp._OMEGA_NU_FID)
    frac = np.clip(hi - lo, 0, None) / (wc_hi - wc_lo)
    p_accept = np.trapz(frac @ (w / w.sum()), h) / (float(hp.high) - float(hp.low))
    assert exp._omega_m_log_acceptance == pytest.approx(np.log(p_accept), abs=2e-3)


def test_trace_records_truncated_prior_density(exp):
    import pyro
    from pyro import poutine

    n = 50_000

    def model():
        with pyro.plate("p", n):
            return exp.sample_parameters((n,))

    tr = poutine.trace(model).get_trace()
    tr.compute_log_prob()
    recorded = sum(tr.nodes[k]["log_prob"] for k in exp.cosmo_params)
    expected = sum(exp.prior[k].log_prob(tr.nodes[k]["value"]) for k in exp.cosmo_params)
    expected = expected - exp._omega_m_log_acceptance
    assert torch.allclose(recorded, expected, atol=1e-4)


def test_central_sample_data_shape(exp):
    y = exp.sample_data(exp.nominal_design.view(1, -1), num_samples=5, central=True)
    assert y.shape == (5, 1, 24)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.environ.get("HOME", ""), "data", "desi", "shapefit_dr1",
                                    "mcmc_samples", "base.npy")),
    reason="DESI ShapeFit reference chain not converted")
def test_nominal_samples_are_desi_shapefit_chain(exp):
    gd = exp.get_nominal_samples(num_samples=5000)
    assert gd.getParamNames().list() == exp.cosmo_params
    assert gd.label == "DESI DR1 ShapeFit"
    means = gd.getMeans()
    # DESI DR1 ShapeFit-alone (all-nolya, BBN + ns10) published means, chain.margestats.
    assert means[0] == pytest.approx(0.1233, abs=0.002)    # omega_cdm
    assert means[2] == pytest.approx(0.700, abs=0.005)     # h, not H0
    assert means[4] == pytest.approx(0.969, abs=0.01)      # n_s, not divided by 100


def test_split_param_pair_handles_underscored_names():
    from bedcosmo.plotting import split_param_pair

    names = ["omega_cdm", "omega_b", "h", "ln10A_s", "n_s"]
    assert split_param_pair("omega_cdm_omega_b", names) == ("omega_cdm", "omega_b")
    assert split_param_pair("ln10A_s_n_s", names) == ("ln10A_s", "n_s")
    with pytest.raises(ValueError):
        split_param_pair("omega_cdm_w0", names)


@pytest.mark.parametrize("override", [
    {"likelihood_mode": "scaling"},
    {"vary_z_eff": True},
    {"emulator_sqrtn_ref": "sampled"},
])
def test_bao_only_options_rejected(override):
    with pytest.raises(ValueError):
        _make_exp(**override)


def test_fiducial_covariance_ignores_cosmology_but_keeps_design():
    fixed = _make_exp(emulator_covariance="fiducial")
    designs = fixed.designs[:2].unsqueeze(0).expand(2, -1, -1)       # (cosmology, design, 4)
    n_tracers = fixed._shapefit_n_tracers(designs)
    params = {n: torch.tensor([[fixed._SHAPEFIT_FIDUCIAL[n]], [fixed._SHAPEFIT_FIDUCIAL[n]]],
                              dtype=torch.float64).expand(2, 2).unsqueeze(-1).clone() for n in fixed.cosmo_params}
    params["omega_cdm"][1] *= 1.15
    params["h"][1] *= 1.05
    mean, cov = fixed._shapefit_likelihood(n_tracers, params)
    assert not torch.allclose(mean[0], mean[1])                        # mean follows cosmology
    assert torch.equal(cov[0], cov[1])                                 # covariance does not
    assert not torch.allclose(cov[0, 0], cov[0, 1])                    # but follows the design

    _, cov_theta = _make_exp()._shapefit_likelihood(n_tracers, params)
    assert not torch.allclose(cov_theta[0], cov_theta[1])


def _make_bao_emulator_exp(**overrides):
    return init_experiment(cosmo_exp="num_tracers", prior_args_path="prior_args_hrdrag.yaml",
                           design_args_path="design_args_dr1.yaml", dataset="dr1", analysis="bao",
                           cosmo_model="base", likelihood_mode="emulator", emulator_space="fourier",
                           device="cpu", mode="eval", **overrides)


def test_bao_fiducial_covariance_ignores_cosmology_but_keeps_design():
    fixed = _make_bao_emulator_exp(emulator_covariance="fiducial")
    designs = fixed.designs[:2].double().unsqueeze(0).expand(2, -1, -1)   # (cosmology, design, 4)
    passed_ratio = fixed.calc_passed(designs)
    params = {"Om": torch.tensor([[0.3152], [0.36]], dtype=torch.float64).expand(2, 2).unsqueeze(-1),
              "hrdrag": torch.tensor([[0.9908], [1.03]], dtype=torch.float64).expand(2, 2).unsqueeze(-1)}
    cov = fixed._build_emulator_covariance(passed_ratio, params)
    assert torch.equal(cov[0], cov[1])                                 # covariance ignores cosmology
    assert not torch.allclose(cov[0, 0], cov[0, 1])                    # but follows the design

    cov_theta = _make_bao_emulator_exp(emulator_covariance="cosmology")._build_emulator_covariance(
        passed_ratio, params)
    assert not torch.allclose(cov_theta[0], cov_theta[1])
    assert torch.allclose(cov_theta[0], cov[0])                        # first row sits at the fiducial


def test_sampled_sqrtn_reference_rejects_fiducial_covariance():
    with pytest.raises(ValueError, match="contradicts emulator_covariance='fiducial'"):
        _make_bao_emulator_exp(emulator_covariance="fiducial", emulator_sqrtn_ref="sampled")
