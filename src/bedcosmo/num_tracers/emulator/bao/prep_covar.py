import os

# Set display/backend env vars before any library imports can trigger X11 or GPU probes.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import traceback
import warnings
from typing import Dict, List, Tuple
from scipy.linalg import cho_factor, cho_solve

# In spawned worker processes, additionally suppress all C-level stderr.
if os.environ.get("_PREP_COVAR_WORKER") == "1":
    os.environ.pop("DISPLAY", None)
    warnings.filterwarnings("ignore")
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

# velocileptors uses np.trapezoid (numpy >= 2.0); shim for numpy 1.x.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# Suppress desilike import-time warnings (e.g. missing interpax/jax) before importing
warnings.filterwarnings("ignore")

from desilike import Fisher
from desilike.likelihoods.galaxy_clustering import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import (
    CutskyFootprint,
    ObservablesCovarianceMatrix,
    TracerPowerSpectrumMultipolesObservable,
)
from desilike.theories.galaxy_clustering import (
    BAOPowerSpectrumTemplate,
    DampedBAOWigglesTracerPowerSpectrumMultipoles,
)
from desilike.theories.primordial_cosmology import get_cosmo

from util import (
    TRACER_CONFIGS,
    TRACER_TYPE_CHOICES,
    get_default_save_path,
    get_tracer_config,
    latin_hypercube_samples,
    parse_priors,
    save_dataset,
)

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", message=".*EisensteinHu.*")

# Priors matching experiments/num_tracers/prior_args.yaml, plus N_tracers.
DEFAULT_PRIORS = {
    "N_tracers": {"dist": "uniform", "low": 1e5, "high": 1e7},
    "Om": {"dist": "uniform", "low": 0.01, "high": 0.99},
    "Ok": {"dist": "uniform", "low": -0.3, "high": 0.3},
    "w0": {"dist": "uniform", "low": -3.0, "high": 1.0},
    "wa": {"dist": "uniform", "low": -3.0, "high": 2.0},
    "hrdrag": {"dist": "uniform", "low": 10.0, "high": 1000.0},
}

# Constraints from prior_args.yaml.
CONSTRAINTS = {
    "valid_densities": {"params": ["Om", "Ok"], "lower": 0.0, "upper": 1.0},
    "high_z_matter_dom": {"params": ["w0", "wa"], "upper": 0.0},
}

COSMO_MODELS = {
    "base":              ["Om", "hrdrag"],
    "base_w":            ["Om", "w0", "hrdrag"],
    "base_w_wa":         ["Om", "w0", "wa", "hrdrag"],
    "base_omegak":       ["Om", "Ok", "hrdrag"],
    "base_omegak_w_wa":  ["Om", "Ok", "w0", "wa", "hrdrag"],
}

# Fiducial values for fixed parameters
PARAM_DEFAULTS = {"Ok": 0.0, "w0": -1.0, "wa": 0.0}

# DESI fiducial values for parameters that set the power spectrum shape
# but are not varied in the BAO prior.
_OMEGA_B_FID = 0.02237
_MNU_FID = 0.06  # sum of neutrino masses in eV (1 massive neutrino)
_OMEGA_NU_FID = _MNU_FID / 93.14  # neutrino density parameter
_N_S_FID = 0.9649
_LN10A_S_FID = 3.044
_H_FID = 0.6736

_PHYS_NAMES = ["DH_over_rd", "DM_over_rd"]
_TRIU_I, _TRIU_J = np.triu_indices(2)
TARGET_NAMES = [f"cov_{_PHYS_NAMES[i]}_{_PHYS_NAMES[j]}" for i, j in zip(_TRIU_I, _TRIU_J)]

# Integration settings for displacement / reconstruction model
_K_DISP_MIN = 1.0e-4
_K_DISP_MAX = 10.0
_NK_DISP = 512
_NMU_DISP = 64


def _sample_constrained_pair(
    low1: float, high1: float,
    low2: float, high2: float,
    n: int,
    rng: np.random.Generator,
    sum_lower: float | None = None,
    sum_upper: float | None = None,
) -> np.ndarray:
    """Sample n points uniformly from a 2D box with a linear sum constraint."""
    out = np.empty((n, 2), dtype=np.float64)
    filled = 0
    while filled < n:
        batch = rng.uniform(
            [low1, low2], [high1, high2], size=(max(n - filled, 256) * 3, 2)
        )
        valid = np.ones(len(batch), dtype=bool)
        if sum_lower is not None:
            valid &= (batch[:, 0] + batch[:, 1]) > sum_lower
        if sum_upper is not None:
            valid &= (batch[:, 0] + batch[:, 1]) < sum_upper
        good = batch[valid]
        take = min(len(good), n - filled)
        out[filled : filled + take] = good[:take]
        filled += take
    return out


def _constrained_samples(
    priors: Dict[str, Dict[str, float]],
    constraints: Dict[str, Dict],
    n_samples: int,
    seed: int,
    sigma_clip: float = 4.0,
) -> List[Dict[str, float]]:
    """Draw samples with LHS for unconstrained params and rejection for constrained pairs."""
    rng = np.random.default_rng(seed)

    constrained_keys: set = set()
    for spec in constraints.values():
        constrained_keys.update(spec["params"])

    unconstrained_priors = {k: v for k, v in priors.items() if k not in constrained_keys}
    if unconstrained_priors:
        lhs_draws = latin_hypercube_samples(
            unconstrained_priors, n_samples=n_samples, seed=seed, sigma_clip=sigma_clip,
        )
    else:
        lhs_draws = [{}] * n_samples

    pair_samples: Dict[str, np.ndarray] = {}
    for _, cspec in constraints.items():
        p1, p2 = cspec["params"]
        pair = _sample_constrained_pair(
            low1=priors[p1]["low"], high1=priors[p1]["high"],
            low2=priors[p2]["low"], high2=priors[p2]["high"],
            n=n_samples,
            rng=rng,
            sum_lower=cspec.get("lower"),
            sum_upper=cspec.get("upper"),
        )
        pair_samples[p1] = pair[:, 0]
        pair_samples[p2] = pair[:, 1]

    rows: List[Dict[str, float]] = []
    for i in range(n_samples):
        row = dict(lhs_draws[i])
        for k, arr in pair_samples.items():
            row[k] = float(arr[i])
        rows.append(row)
    return rows


def _to_bao_cosmo_params(sample: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Convert BAO prior parameters to desilike cosmology parameters.

    Fixes h to the DESI fiducial value, matching the DESI BAO parameterization
    where h is fixed and hrdrag is a free parameter entering through rd = hrdrag / h.
    """
    Om = float(sample["Om"])
    omega_cdm = Om * _H_FID**2 - _OMEGA_NU_FID - _OMEGA_B_FID
    if omega_cdm <= 0:
        raise ValueError(
            f"Om={Om:.4f} too small: omega_cdm={omega_cdm:.4f} < 0 "
            f"(need Om > {(_OMEGA_B_FID + _OMEGA_NU_FID) / _H_FID**2:.4f})"
        )

    theta_cosmo = {
        "Omega_m": Om,
        "Omega_k": float(sample["Ok"]),
        "w0_fld": float(sample["w0"]),
        "wa_fld": float(sample["wa"]),
        "h": _H_FID,
        "omega_b": _OMEGA_B_FID,
        "n_s": _N_S_FID,
        "logA": _LN10A_S_FID,
    }
    return theta_cosmo, float(sample["hrdrag"])


def _linear_pk_1d(fo, z: float):
    """
    Return a callable P_lin(k, z) in (Mpc/h)^3 for delta_cb.
    """
    pk = fo.pk_interpolator(of="delta_cb").to_1d(z=z)
    return lambda k: np.asarray(pk(k), dtype=np.float64)


def _sigma_v_sq_1loop_rsd(pk_lin_1d) -> Tuple[float, float, float]:
    """
    Return the three 1-loop Lagrangian displacement-variance pieces needed to
    build the full 1-loop BAO damping in redshift space.

    Returns (sv2, sv2_dot, sv2_ddot) where
        sv2       = 0.5 * sigmaloop        (density-density, coeff of 1 in Sigma_par^2)
        sv2_dot   = 0.5 * sigmaloopdot     (density-velocity cross, coeff of 2f)
        sv2_ddot  = 0.5 * sigmaloopddot    (velocity-velocity, coeff of f^2)

    With these, the 1-loop contributions are
        Sigma_perp^2 += sv2
        Sigma_par^2  += sv2 + 2f * sv2_dot + f^2 * sv2_ddot.

    The naive (1+f)^2 boost on the density piece assumes sv2_dot = sv2_ddot = sv2,
    which is only exact at linear order. At 1-loop the velocity divergence P_theta-theta
    picks up stronger mode-coupling corrections than P_delta-delta, so sv2_dot and
    sv2_ddot differ from sv2 (typically sv2_dot ~ 2*sv2, sv2_ddot ~ 3*sv2).
    """
    from velocileptors.LPT.velocity_moments_fftw import VelocityMoments

    k = np.geomspace(1.0e-4, 20.0, 2000)
    pk = np.clip(pk_lin_1d(k), 0.0, None)
    vm = VelocityMoments(k, pk, one_loop=True, shear=False, third_order=False)
    return (
        0.5 * float(vm.sigmaloop),
        0.5 * float(vm.sigmaloopdot),
        0.5 * float(vm.sigmaloopddot),
    )


def _sigma_nl_pre_from_pk(
    pk_lin_1d,
    f: float,
    sv2_1loop: float = 0.0,
    sv2_1loop_dot: float = 0.0,
    sv2_1loop_ddot: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute pre-reconstruction BAO damping scales from the linear + 1-loop displacement
    variance, using the full 1-loop redshift-space decomposition.

    Linear:
        sv2_lin = (1/6 pi^2) int dk P_lin(k)
        Sigma_perp^2_lin = sv2_lin
        Sigma_par^2_lin  = (1+f)^2 sv2_lin     (exact, since sv2_dot = sv2_ddot = sv2 at lin)

    1-loop:
        Sigma_perp^2 += sv2_1loop
        Sigma_par^2  += sv2_1loop + 2f sv2_1loop_dot + f^2 sv2_1loop_ddot
    """
    k = np.geomspace(_K_DISP_MIN, _K_DISP_MAX, _NK_DISP)
    pk = np.clip(pk_lin_1d(k), 0.0, None)
    sv2_lin = float(np.trapz(pk, k) / (6.0 * np.pi**2))
    f = float(f)

    sigma_perp_sq = sv2_lin + float(sv2_1loop)
    sigma_par_sq = (
        (1.0 + f) ** 2 * sv2_lin
        + float(sv2_1loop)
        + 2.0 * f * float(sv2_1loop_dot)
        + f**2 * float(sv2_1loop_ddot)
    )
    sigma_perp_pre = np.sqrt(max(sigma_perp_sq, 0.0))
    sigma_par_pre = np.sqrt(max(sigma_par_sq, 0.0))
    return sigma_perp_pre, sigma_par_pre


def _lya_flux_bias(
    z: float,
    bF0: float = -0.117,
    z_ref: float = 2.334,
    gamma: float = 2.9,
) -> float:
    """Compute Ly-alpha flux bias b_F(z)."""
    return float(bF0) * ((1.0 + float(z)) / (1.0 + float(z_ref))) ** float(gamma)


def _sigma_nl_post_from_recon(
    pk_lin_1d,
    b1: float,
    f: float,
    nbar_comoving: float,
    smoothing_scale: float,
    sv2_1loop: float = 0.0,
    sv2_1loop_dot: float = 0.0,
    sv2_1loop_ddot: float = 0.0,
    n_iter: int = 1,
    sigma_fog: float = 0.0,
) -> Tuple[float, float]:
    """
    Anisotropic post-reconstruction residual BAO damping via 2D (k, mu) integration,
    plus the 1-loop redshift-space displacement variance that survives reconstruction.

    The Wiener filter uses the redshift-space galaxy power spectrum
    P_g(k, mu) = (b1 + f*mu^2)^2 P_lin(k), so reconstruction efficiency
    varies with mu (line-of-sight vs transverse).

    W(k, mu) = P_g(k, mu) / [P_g(k, mu) + 1/nbar]
    T_res(k, mu) = [1 - S(k) W(k, mu)]^n_iter

    The exponent n_iter models *iterative* reconstruction (White 2015,
    Schmittfull+2017). Each iteration of IFT recon re-filters the residual
    field, so after N iterations the residual transfer is approximately
    (1 - SW)^N assuming the Wiener filter is held fixed across iterations.
    DESI BAO uses N=3, which is the default here.

    Linear-theory residual:
        sigma_perp^2 = 1/(2 pi^2) int dk P(k) int_0^1 dmu (1-mu^2)/2 T_res^2
        sigma_par^2  = (1+f)^2 * 1/(2 pi^2) int dk P(k) int_0^1 dmu mu^2 T_res^2

    1-loop (beyond-Zel'dovich) contributions survive recon (recon is a linear-theory
    operation) and add the full redshift-space velocity variance:
        sigma_perp^2 += sv2_1loop
        sigma_par^2  += sv2_1loop + 2f sv2_1loop_dot + f^2 sv2_1loop_ddot

    The parallel 1-loop boost differs from the naive (1+f)^2 on sv2_1loop because
    P_theta-theta at 1-loop differs from P_delta-delta at 1-loop.
    """
    if nbar_comoving <= 0:
        raise ValueError(f"nbar_comoving must be positive, got {nbar_comoving}")

    b1 = float(b1)
    f = float(f)
    nbar_inv = 1.0 / float(nbar_comoving)
    R = float(smoothing_scale)

    k = np.geomspace(_K_DISP_MIN, _K_DISP_MAX, _NK_DISP)
    mu = np.linspace(0.0, 1.0, _NMU_DISP)
    kk, mm = np.meshgrid(k, mu, indexing="ij")  # (NK, NMU)

    pk = np.clip(pk_lin_1d(kk), 0.0, None)
    S = np.exp(-0.5 * (kk * R) ** 2)
    Pg = (b1 + f * mm**2) ** 2 * pk
    W = Pg / (Pg + nbar_inv)
    T_res = (1.0 - S * W) ** int(n_iter)

    integrand_perp = pk * T_res**2 * (1.0 - mm**2) / 2.0
    sigma_perp_sq = (
        np.trapz(np.trapz(integrand_perp, mu, axis=1), k) / (2.0 * np.pi**2)
    )

    integrand_par = pk * T_res**2 * mm**2
    sigma_par_sq = (
        (1.0 + f) ** 2
        * np.trapz(np.trapz(integrand_par, mu, axis=1), k)
        / (2.0 * np.pi**2)
    )

    sv2_1loop = float(sv2_1loop)
    sv2_1loop_dot = float(sv2_1loop_dot)
    sv2_1loop_ddot = float(sv2_1loop_ddot)
    sigma_perp_sq += sv2_1loop
    sigma_par_sq += sv2_1loop + 2.0 * f * sv2_1loop_dot + f**2 * sv2_1loop_ddot

    # Finger-of-God: virial redshift scatter adds in quadrature to Sigma_par only.
    sigma_par_sq += float(sigma_fog) ** 2

    sigma_perp_post = np.sqrt(max(float(sigma_perp_sq), 0.0))
    sigma_par_post = np.sqrt(max(float(sigma_par_sq), 0.0))
    return sigma_perp_post, sigma_par_post


def _tinker10_b1(nu: np.ndarray, delta_vir: float = 200.0) -> np.ndarray:
    """Peak-background-split bias from Tinker et al. 2010 (arXiv:1001.3162), Eq. (6)."""
    delta_c = 1.686
    y = np.log10(delta_vir)
    A = 1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4))
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4))
    c = 2.4
    return 1.0 - A * nu**a / (nu**a + delta_c**a) + B * nu**b + C * nu**c


def _sigma_fog_from_bias(
    b1_target: float,
    z: float,
    cosmo,
    fo,
    delta_vir: float = 200.0,
    log_M_min: float = 10.5,
    log_M_max: float = 15.5,
    n_grid: int = 256,
) -> float:
    """
    Tier-1 FoG line-of-sight velocity-dispersion scale from linear bias (no HOD).

    Inverts Tinker+2010 b1(M, z) to find the effective host halo mass M_eff for a
    tracer with linear bias `b1_target`, then uses the virial theorem
    (sigma_v^2 = G M / (2 R_vir)) to convert to a line-of-sight damping scale in Mpc/h.

    This adds in quadrature to Sigma_par (not Sigma_perp) since virial motions
    scatter redshifts but not transverse positions.
    """
    delta_c = 1.686
    log_M = np.linspace(log_M_min, log_M_max, n_grid)
    M = 10.0 ** log_M  # M_sun/h
    rho_m_com = float(cosmo.rho_m(z)) * 1.0e10  # M_sun/h per (Mpc/h)^3, comoving
    R_top_hat = (3.0 * M / (4.0 * np.pi * rho_m_com)) ** (1.0 / 3.0)  # Mpc/h, comoving
    sigma_M = np.asarray(fo.sigma_rz(R_top_hat, z, of="delta_cb"))

    nu = delta_c / sigma_M
    b1_of_M = _tinker10_b1(nu, delta_vir=delta_vir)

    if b1_target <= b1_of_M[0]:
        M_eff = M[0]
    elif b1_target >= b1_of_M[-1]:
        M_eff = M[-1]
    else:
        log_M_eff = np.interp(b1_target, b1_of_M, log_M)
        M_eff = 10.0 ** log_M_eff

    R_vir_com = (3.0 * M_eff / (4.0 * np.pi * delta_vir * rho_m_com)) ** (1.0 / 3.0)
    R_vir_phys = R_vir_com / (1.0 + z)  # physical Mpc/h

    # G in Mpc * (km/s)^2 / M_sun. The h-factors cancel: M[M_sun/h]/R[Mpc/h] = M/R in physical.
    G_Mpc_Msun = 4.30091e-9
    sigma_v_sq_km_s = G_Mpc_Msun * M_eff / (2.0 * R_vir_phys)  # (km/s)^2, 1D dispersion

    E_z = float(cosmo.efunc(z))
    # sigma_FoG [Mpc/h] = sigma_v[km/s] * h / (a H [km/s/Mpc]) = sigma_v * (1+z) / (100 E(z))
    return float(np.sqrt(sigma_v_sq_km_s) * (1.0 + z) / (100.0 * E_z))


# =====================================================================
# Non-Gaussian covariance additions (Tier-1: bias-only, no HOD)
# =====================================================================

def _sigma_b_sq(
    cosmo,
    fo,
    z: float,
    V_survey: float,
    ) -> float:
    """Variance of the super-survey background mode, top-hat sphere approximation.

    sigma_b^2 ≈ sigma^2(R_eff, z) where R_eff = (3 V_survey / 4pi)^(1/3).
    """
    if V_survey <= 0.0:
        return 0.0
    R_eff = (3.0 * V_survey / (4.0 * np.pi)) ** (1.0 / 3.0)
    sigma_R = float(fo.sigma_rz(np.array([R_eff]), z, of="delta_cb")[0])
    return sigma_R ** 2


def _ssc_cov(
    k_centers: np.ndarray,
    ells: Tuple[int, ...],
    pk_multipoles: np.ndarray,
    sigma_b_sq: float,
    ) -> np.ndarray:
    """Super-sample covariance contribution, rank-1 outer product of response.

    C^SSC(k_i, k_j) = sigma_b^2 * R_i * R_j,
    where R_ell(k) = (68/21 - (1/3) d ln P_ell / d ln k) * P_ell(k) is the
    tree-level response (beat-coupling + dilation).

    pk_multipoles : array of shape (n_ells, n_k) with the fiducial galaxy
        multipole power spectrum on k_centers.
    """
    pk = np.asarray(pk_multipoles, dtype=np.float64)
    n_ells, n_k = pk.shape
    assert n_ells == len(ells) and n_k == len(k_centers)

    log_k = np.log(k_centers)
    R_ell = np.empty_like(pk)
    for i in range(n_ells):
        P_i = pk[i]
        # guard against zeros / negative
        safe = np.where(P_i > 0, P_i, np.nan)
        dlnP_dlnk = np.gradient(np.log(safe), log_k)
        dlnP_dlnk = np.nan_to_num(dlnP_dlnk, nan=0.0)
        R_ell[i] = (68.0 / 21.0 - dlnP_dlnk / 3.0) * P_i

    R_flat = R_ell.reshape(-1)  # shape (n_ells * n_k,)
    return sigma_b_sq * np.outer(R_flat, R_flat)


def _get_lya_qso_bao_params(
    cfg: Dict[str, float],
    z_eff: float,
    sigma8_delta: float,
    f: float,
    nbar_comoving: float,
    base_volume: float,
    area: float,
    ) -> Dict[str, float]:
    """Return Ly-alpha QSO BAO nuisance parameters for Fisher forecasts."""
    b1 = _lya_flux_bias(
        z=z_eff,
        bF0=float(cfg.get("bF0", -0.117)),
        gamma=float(cfg.get("gamma_bF", 2.9)),
    )

    Sigma_perp_fid = float(cfg.get("Sigma_perp_fid", 5.0))
    Sigma_par_fid  = float(cfg.get("Sigma_par_fid", 10.0))

    sigma8_fid = float(cfg.get("sigma8_fid", 0.83))
    f_fid      = float(cfg.get("f_fid", 0.97))

    growth_ratio = sigma8_delta / sigma8_fid
    f_ratio = f / f_fid

    sigma_perp_lya = Sigma_perp_fid * growth_ratio
    sigma_par_lya = Sigma_par_fid * growth_ratio * f_ratio**0.5

    # Effective white-noise power:
    # P_N_eff = sigma2_pix * delta_r_pix / n_sightline_3D, then nbar_eff = 1 / P_N_eff.
    P_N_eff = cfg.get("P_N_eff", None)
    if P_N_eff is None:
        sigma2_pix = float(cfg.get("sigma2_pix", 1.0))
        delta_r_pix = float(cfg.get("delta_r_pix", 1.0))
        P_N_eff = sigma2_pix * delta_r_pix / max(nbar_comoving, 1.0e-30)
    P_N_eff = float(P_N_eff)
    if P_N_eff <= 0:
        raise ValueError(f"P_N_eff must be > 0 for LyA tracer, got {P_N_eff}")
    nbar_eff = 1.0 / P_N_eff

    # Convert effective 3D nbar back to an angular density for CutskyFootprint input.
    N_eff = nbar_eff * float(base_volume)
    nbar_for_footprint = N_eff / area

    return {
        "is_lya": True,
        "b1": b1,
        "sigmaper_post": sigma_perp_lya,
        "sigmapar_post": sigma_par_lya,
        "nbar_eff": nbar_eff,
        "P_N_eff": P_N_eff,
        "nbar_for_footprint": nbar_for_footprint,
    }


def _get_standard_tracer_bao_params(
    cfg: Dict[str, float],
    fo,
    sigma8_delta: float,
    f: float,
    pk_lin_1d,
    nbar_comoving: float,
    nbar_ang: float,
    sv2_1loop: float = 0.0,
    sv2_1loop_dot: float = 0.0,
    sv2_1loop_ddot: float = 0.0,
    n_iter: int = 1,
    cosmo=None,
    z: float | None = None,
    include_fog: bool = True,
    ) -> Dict[str, float]:
    """Return standard galaxy-tracer BAO nuisance parameters for Fisher forecasts."""
    b1 = float(cfg["bias_recon"]) * float(fo.sigma8_cb) / sigma8_delta

    if include_fog and (cosmo is not None) and (z is not None):
        sigma_fog = _sigma_fog_from_bias(b1, z=float(z), cosmo=cosmo, fo=fo)
    else:
        sigma_fog = 0.0

    sigma_perp_post, sigma_par_post = _sigma_nl_post_from_recon(
        pk_lin_1d=pk_lin_1d,
        b1=b1,
        f=f,
        nbar_comoving=nbar_comoving,
        smoothing_scale=float(cfg["smoothing_scale"]),
        sv2_1loop=sv2_1loop,
        sv2_1loop_dot=sv2_1loop_dot,
        sv2_1loop_ddot=sv2_1loop_ddot,
        n_iter=n_iter,
        sigma_fog=sigma_fog,
    )
    nbar_eff = nbar_comoving
    P_N_eff = 1.0 / max(nbar_eff, 1.0e-30)
    nbar_for_footprint = nbar_ang

    return {
        "is_lya": False,
        "b1": b1,
        "sigmaper_post": sigma_perp_post,
        "sigmapar_post": sigma_par_post,
        "sigma_fog": sigma_fog,
        "nbar_eff": nbar_eff,
        "P_N_eff": P_N_eff,
        "nbar_for_footprint": nbar_for_footprint,
    }


def get_bao_fisher_covariance(
    N_tracers: float,
    theta_cosmo: Dict[str, float],
    hrdrag: float,
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    area: float = 14000.0,
    resolution: int = 3,
    tracer_config: Dict[str, float] | None = None,
    override_sigmas: Tuple[float, float] | None = None,
    n_iter: int = 1,
    include_fog: bool = True,
    float_sigma_bao: bool = True
) -> Dict[str, float]:
    """
    Compute the 2x2 BAO covariance matrix from a Fisher forecast.

    Realism upgrades:
    - f(z) computed from sampled cosmology
    - pre-reconstruction damping from linear P(k)
    - post-reconstruction residual damping from a semi-analytic reconstruction model
      using tracer bias, smoothing scale, and comoving number density
    """

    # Load tracer-specific configuration from tracers.yaml
    # (bias, smoothing scale, z-bin, fiducial number-density range, etc.)
    cfg = get_tracer_config(tracer_bin)

    # Allow caller-side overrides of tracer config entries.
    if tracer_config is not None:
        cfg.update(tracer_config)

    # Default to tracer-bin redshift configuration if not explicitly supplied.
    if zrange is None:
        zrange = tuple(cfg["zrange"])

    if z_eff is None:
        z_eff = float(cfg["z_eff"])

    z = z_eff

    # Construct DESI fiducial cosmology object and Fourier-space helper.
    cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
    fo = cosmo.get_fourier()

    # Compute linear growth quantities.
    #
    # sigma8_delta  -> RMS density fluctuation amplitude
    # sigma8_theta  -> RMS velocity-divergence amplitude
    #
    # Their ratio gives the linear growth rate:
    #
    #   f(z) = dlnD / dln a
    #
    sigma8_delta = float(fo.sigma8_z(z, of="delta_cb"))
    sigma8_theta = float(fo.sigma8_z(z, of="theta_cb"))

    if sigma8_delta <= 0:
        raise ValueError(f"sigma8_delta <= 0 at z={z:.3f}: {sigma8_delta}")

    f = sigma8_theta / sigma8_delta

    # ------------------------------------------------------------------
    # SURVEY GEOMETRY
    # ------------------------------------------------------------------
    #
    # `area` is assumed to be the EFFECTIVE post-mask survey area (i.e.
    # imaging vetos, bright-star masks, extinction cuts, bad-imaging
    # regions already removed). No further f_mask rescaling is applied.
    #
    # N_tracers is the EFFECTIVE OBSERVED tracer count:
    #
    #   N_observed = f_fiber * f_zsuccess * N_target
    #
    # i.e. fiber-assignment and redshift-success efficiencies are already
    # folded in externally. The pipeline treats N_tracers as the final
    # usable catalog size.
    nbar_ang = N_tracers / area

    # ------------------------------------------------------------------
    # SURVEY FOOTPRINT / EFFECTIVE VOLUME
    # ------------------------------------------------------------------

    # Initial footprint used ONLY to determine the physical survey volume.
    #
    # This is the geometry object:
    #   - survey area
    #   - redshift range
    #   - fiducial cosmology
    #
    # It defines the actual observable Fourier volume.
    base_footprint = CutskyFootprint(
        area=area,
        zrange=zrange,
        nbar=nbar_ang,
        cosmo=cosmo,
    )

    # Convert angular tracer density into comoving number density:
    #
    #   nbar = N / V
    #
    # Units:
    #   (h/Mpc)^3
    #
    nbar_comoving = float(N_tracers) / float(base_footprint.volume)

    # ------------------------------------------------------------------
    # BAO DAMPING MODEL
    # ------------------------------------------------------------------

    # Linear matter power spectrum interpolator.
    pk_lin_1d = _linear_pk_1d(fo, z=z)

    # One-loop displacement variance terms entering nonlinear BAO damping.
    #
    # These encode:
    #   - density-density displacements
    #   - density-velocity coupling
    #   - velocity-velocity coupling
    #
    # and determine anisotropic nonlinear smearing.
    sv2_1loop, sv2_1loop_dot, sv2_1loop_ddot = _sigma_v_sq_1loop_rsd(pk_lin_1d)

    # Pre-reconstruction BAO damping scales:
    #
    #   Sigma_perp
    #   Sigma_parallel
    #
    # including linear + 1-loop corrections.
    sigma_perp_pre, sigma_par_pre = _sigma_nl_pre_from_pk(
        pk_lin_1d,
        f=f,
        sv2_1loop=sv2_1loop,
        sv2_1loop_dot=sv2_1loop_dot,
        sv2_1loop_ddot=sv2_1loop_ddot,
    )

    # ------------------------------------------------------------------
    # TRACER-SPECIFIC EFFECTIVE NOISE / RECONSTRUCTION MODEL
    # ------------------------------------------------------------------

    # Lyα forest tracers use an effective white-noise model rather than
    # standard galaxy shot noise.
    if tracer_bin == "Lya_QSO":

        tracer_settings = _get_lya_qso_bao_params(
            cfg=cfg,
            z_eff=z,
            sigma8_delta=sigma8_delta,
            f=f,
            nbar_comoving=nbar_comoving,
            base_volume=float(base_footprint.volume),
            area=area,
        )

    # Standard galaxy tracers: LRG, ELG, BGS, QSO
    else:
        tracer_settings = _get_standard_tracer_bao_params(
            cfg=cfg,
            fo=fo,
            sigma8_delta=sigma8_delta,
            f=f,
            pk_lin_1d=pk_lin_1d,
            nbar_comoving=nbar_comoving,
            nbar_ang=nbar_ang,
            sv2_1loop=sv2_1loop,
            sv2_1loop_dot=sv2_1loop_dot,
            sv2_1loop_ddot=sv2_1loop_ddot,
            n_iter=n_iter,
            cosmo=cosmo,
            z=z,
            include_fog=include_fog,
        )

    # Extract nuisance / reconstruction parameters.
    is_lya = bool(tracer_settings["is_lya"])

    b1 = float(tracer_settings["b1"])

    sigma_perp_post = float(tracer_settings["sigmaper_post"])
    sigma_par_post = float(tracer_settings["sigmapar_post"])

    # Optional manual override for debugging / validation studies.
    if override_sigmas is not None:
        sigma_perp_post = float(override_sigmas[0])
        sigma_par_post = float(override_sigmas[1])

    nbar_eff = float(tracer_settings["nbar_eff"])
    P_N_eff = float(tracer_settings["P_N_eff"])
    nbar_for_footprint = float(tracer_settings["nbar_for_footprint"])

    # Parameters supplied to the BAO observable model.
    params = {
        "b1": b1,
        "sigmaper": sigma_perp_post,
        "sigmapar": sigma_par_post,
    }

    # ------------------------------------------------------------------
    # FINAL FISHER FOOTPRINT OBJECT
    # ------------------------------------------------------------------

    # This footprint object is used by desilike to build:
    #   - covariance normalization
    #   - effective mode counts
    #   - shot noise
    #
    # attrs are metadata only (not used directly by desilike).
    footprint = CutskyFootprint(
        area=area,
        zrange=zrange,
        nbar=nbar_for_footprint,
        cosmo=cosmo,
        attrs={
            "tracer_bin": tracer_bin,
            "is_lya": is_lya,
            "bias_recon": cfg.get("bias_recon"),
            "smoothing_scale": cfg.get("smoothing_scale"),
            "nbar_comoving": nbar_comoving,
            "nbar_eff": nbar_eff,
            "P_N_eff": P_N_eff,
            "f_cosmo": f,
            "sigmaper_pre": sigma_perp_pre,
            "sigmapar_pre": sigma_par_pre,
            "sigmaper_post": sigma_perp_post,
            "sigmapar_post": sigma_par_post
        },
    )

    # ------------------------------------------------------------------
    # APPROXIMATE SURVEY WINDOW FUNCTION EFFECT
    # ------------------------------------------------------------------

    # Finite survey geometry suppresses access to modes larger than the
    # survey coherence scale.
    #
    # Approximate the survey side length as:
    #
    #   L ~ V^(1/3)
    #
    # which gives a characteristic fundamental Fourier mode:
    #
    #   k_fund ~ 2pi / L
    #
    # Modes below this scale are strongly mixed/suppressed by the survey
    # window function and are not independently measurable.
    #
    # This is a simplified approximation to the full DESI window-function
    # convolution treatment.  [oai_citation:0‡OUP Academic](https://academic.oup.com/mnras/article-pdf/479/4/5168/25209108/sty1814.pdf?utm_source=chatgpt.com)
    L_survey = base_footprint.volume ** (1 / 3)

    # Fundamental observable Fourier mode.
    kmin_window = 2.0 * np.pi / L_survey

    # Prevent unrealistically small k_min values for huge survey volumes.
    #
    # DESI BAO analyses typically do not fit below k ~ 0.02 h/Mpc anyway.
    kmin_eff = max(0.02, kmin_window)

    # ------------------------------------------------------------------
    # BAO MODEL
    # ------------------------------------------------------------------

    template = BAOPowerSpectrumTemplate(
        z=z,
        fiducial=("DESI", dict(theta_cosmo)),
        apmode="qparqper",
    )

    # BAO template matches DESI Y1/Y3 convention:
    #   - mode='recsym' : reconstructed with randoms also shifted (DESI standard).
    #     Lyα QSO is pre-recon so mode='' (no reconstruction template).
    #   - smoothing_radius per tracer from tracers.yaml (15 h^-1 Mpc for
    #     galaxy/ELG, 30 h^-1 Mpc for QSO).
    #   - model='fog-damping' : Beutler+17 model with FoG damping applied to
    #     the wiggle part, used in DESI Y1 BAO (Moon+23, Ross+24).
    if is_lya:
        theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(
            template=template,
            model="fog-damping",
        )
    else:
        theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(
            template=template,
            mode="recsym",
            smoothing_radius=float(cfg["smoothing_scale"]),
            model="fog-damping",
        )

    # Observable power-spectrum multipoles used in the Fisher forecast.
    observable = TracerPowerSpectrumMultipolesObservable(
        data=params,
        klim={
            0: [kmin_eff, 0.30, 0.005],
            2: [kmin_eff, 0.30, 0.005],
        },
        theory=theory,
    )

    # Gaussian covariance matrix model.
    covariance = ObservablesCovarianceMatrix(
        observable,
        footprints=footprint,
        resolution=resolution,
    )
    gauss_cov_obj = covariance(**params)
    C_full = np.array(np.asarray(gauss_cov_obj), dtype=np.float64, copy=True)

    # ------------------------------------------------------------------
    # NON-GAUSSIAN COVARIANCE ADDITIONS (Tier-1: bias-only, no HOD)
    #
    #   C_total = C_gauss + C_{1-halo trispectrum} + C_{SSC}
    #
    # Both additions are rank-1 outer products in (k, k') at Tier-1.
    # For Lyα we skip them (tracer model is not halo-based).
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    # NON-GAUSSIAN COVARIANCE ADDITIONS

    # ------------------------------------------------------------------

    if not is_lya:

        ell_tuple = tuple(int(ell) for ell in observable.ells)
        k_centers = np.asarray(observable.k[0], dtype=np.float64)
        V_survey = float(footprint.volume)

        try:

            # ----------------------------------------------------------

            # Build NG covariance pieces

            # ----------------------------------------------------------

            theory(**params)

            pk_multipoles = np.asarray(theory.power, dtype=np.float64)

            sigma_b_sq = _sigma_b_sq(
                cosmo,
                fo,
                z,
                V_survey,
            )

            C_ssc = _ssc_cov(
                k_centers=k_centers,
                ells=ell_tuple,
                pk_multipoles=pk_multipoles,
                sigma_b_sq=sigma_b_sq,
            )

            if C_ssc.shape == C_full.shape:
                C_full += C_ssc

            # ----------------------------------------------------------

            # Stability check

            # ----------------------------------------------------------

            eigvals = np.linalg.eigvalsh(C_full)
            min_eig = np.min(eigvals)

            if (
                not np.isfinite(min_eig)
                or min_eig <= 1e-12
            ):

                raise ValueError(
                    f"Non-positive-definite covariance "
                    f"(min eigenvalue = {min_eig:.3e})"
                )

        except Exception as exc:

            warnings.warn(
                f"Skipping unstable NG covariance sample: {exc}"
            )

            raise
    augmented_cov = gauss_cov_obj.clone(value=C_full)

    # Gaussian likelihood used for Fisher differentiation.
    likelihood = ObservablesGaussianLikelihood(
        observables=observable,
        covariance=augmented_cov,
    )

    # Give sigmas (streaming) a Gaussian prior matching the DESI Y1 BAO fits
    # instead of fixing it, so the Fisher error on qpar/qper accounts for its
    # marginalization rather than being artificially tight.
    likelihood.all_params["sigmas"].update(
        prior={"dist": "norm", "loc": 2.0, "scale": 2.0}
    )

    # Float the BAO damping scales with Gaussian priors centered on the
    # analytically-computed values (width matches the DESI Y1/Y3 BAO fits).
    # The analytic Σ_perp/Σ_par are only approximations — DESI marginalizes
    # over them at the fit stage, which opens the Σ–B degeneracy direction
    # and inflates σ(qpar, qper).
    # Lyα QSO is pre-recon so we don't float the damping scales.
    if float_sigma_bao and not is_lya:
        _sigma_prior_width = 2.0
        likelihood.all_params["sigmaper"].update(
            fixed=False,
            prior={
                "dist": "norm",
                "loc": float(sigma_perp_post),
                "scale": _sigma_prior_width,
            },
        )
        likelihood.all_params["sigmapar"].update(
            fixed=False,
            prior={
                "dist": "norm",
                "loc": float(sigma_par_post),
                "scale": _sigma_prior_width,
            },
        )

    if is_lya:
        # For Lyα: b1 is negative (~-0.12) which falls outside the yaml prior
        # [0.2, 4.0]. Fixing it avoids prior-bounds issues in the Fisher
        # differentiation and removes the b1-dbeta degeneracy.
        # dbeta is also degenerate with b1 in a BAO-only amplitude rescaling
        # and has no physical meaning without reconstruction.
        likelihood.all_params["b1"].update(fixed=True)
        likelihood.all_params["dbeta"].update(fixed=True)

    fisher = Fisher(likelihood)
    fisher_result = fisher(**params)

    F_matrix = -np.array(fisher_result._hessian)

    # Enforce symmetry
    F_matrix = 0.5 * (F_matrix + F_matrix.T)

    # Stabilize tiny negative modes from numerical noise
    eigvals = np.linalg.eigvalsh(F_matrix)
    min_eig = eigvals.min()

    if min_eig <= 0:
        diag_scale = np.mean(np.diag(F_matrix))
        jitter = max(abs(min_eig), 1e-12 * diag_scale)

        F_matrix += np.eye(F_matrix.shape[0]) * jitter

    all_names = [str(p) for p in fisher_result.names()]
    bao_internal = ["qpar", "qper"]
    bao_idx = [all_names.index(p) for p in bao_internal]

    # Selection matrix
    E = np.zeros((F_matrix.shape[0], len(bao_idx)))

    for j, idx in enumerate(bao_idx):
        E[idx, j] = 1.0

    try:
        c, low = cho_factor(F_matrix, check_finite=False)

        cov_subset = cho_solve(
            (c, low),
            E,
            check_finite=False,
        )

        cov_q = cov_subset[np.ix_(bao_idx, range(len(bao_idx)))]

    except np.linalg.LinAlgError:

        cov_full = np.linalg.pinv(F_matrix, rcond=1e-12)

        cov_q = cov_full[np.ix_(bao_idx, bao_idx)]

    rd = hrdrag / _H_FID
    DH_over_rd_fid = float(template.DH_fid) / rd
    DM_over_rd_fid = float(template.DM_fid) / rd
    J = np.diag([DH_over_rd_fid, DM_over_rd_fid])
    cov_phys = J @ cov_q @ J.T

    upper_tri_vals = cov_phys[_TRIU_I, _TRIU_J]
    return dict(zip(TARGET_NAMES, upper_tri_vals))


def run_fisher(
    sample: Dict[str, float],
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    param_defaults: Dict[str, float] | None = None,
    area: float = 14000.0,
    override_sigmas: Tuple[float, float] | None = None,
    n_iter: int = 1,
    include_fog: bool = True,
    float_sigma_bao: bool = True
) -> Dict[str, float]:
    """Convert a sample dict (with N_tracers + cosmo params) to Fisher covariance elements."""
    if param_defaults:
        sample = {**param_defaults, **sample}

    N_tracers = sample["N_tracers"]
    theta_cosmo, hrdrag = _to_bao_cosmo_params(sample)
    return get_bao_fisher_covariance(
        N_tracers=N_tracers,
        theta_cosmo=theta_cosmo,
        hrdrag=hrdrag,
        tracer_bin=tracer_bin,
        zrange=zrange,
        z_eff=z_eff,
        area=area,
        override_sigmas=override_sigmas,
        n_iter=n_iter,
        include_fog=include_fog,
        float_sigma_bao=float_sigma_bao
    )


def compute_pipeline_sigmas(
    sample: Dict[str, float],
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    param_defaults: Dict[str, float] | None = None,
    area: float = 14000.0,
    n_iter: int = 1,
    include_fog: bool = True,
) -> Tuple[float, float]:
    """Return (sigma_perp_post, sigma_par_post) computed by the pipeline for this sample."""
    if param_defaults:
        sample = {**param_defaults, **sample}

    N_tracers = float(sample["N_tracers"])
    theta_cosmo, _hrdrag = _to_bao_cosmo_params(sample)

    cfg = get_tracer_config(tracer_bin)
    if zrange is None:
        zrange = tuple(cfg["zrange"])
    if z_eff is None:
        z_eff = float(cfg["z_eff"])
    z = z_eff

    cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
    fo = cosmo.get_fourier()
    sigma8_delta = float(fo.sigma8_z(z, of="delta_cb"))
    sigma8_theta = float(fo.sigma8_z(z, of="theta_cb"))
    f = sigma8_theta / sigma8_delta

    nbar_ang = N_tracers / area
    base_footprint = CutskyFootprint(area=area, zrange=zrange, nbar=nbar_ang, cosmo=cosmo)
    nbar_comoving = N_tracers / float(base_footprint.volume)

    pk_lin_1d = _linear_pk_1d(fo, z=z)
    sv2_1loop, sv2_1loop_dot, sv2_1loop_ddot = _sigma_v_sq_1loop_rsd(pk_lin_1d)
    sigma_perp_pre, sigma_par_pre = _sigma_nl_pre_from_pk(
        pk_lin_1d, f=f,
        sv2_1loop=sv2_1loop,
        sv2_1loop_dot=sv2_1loop_dot,
        sv2_1loop_ddot=sv2_1loop_ddot,
    )

    if tracer_bin == "Lya_QSO":
        settings = _get_lya_qso_bao_params(
            cfg=cfg,
            z_eff=z,
            sigma8_delta=sigma8_delta,
            f=f,
            nbar_comoving=nbar_comoving,
            base_volume=float(base_footprint.volume),
            area=area,
        )
    else:
        settings = _get_standard_tracer_bao_params(
            cfg=cfg, fo=fo, sigma8_delta=sigma8_delta, f=f,
            pk_lin_1d=pk_lin_1d, nbar_comoving=nbar_comoving, nbar_ang=nbar_ang,
            sv2_1loop=sv2_1loop,
            sv2_1loop_dot=sv2_1loop_dot,
            sv2_1loop_ddot=sv2_1loop_ddot,
            n_iter=n_iter,
            cosmo=cosmo,
            z=z,
            include_fog=include_fog,
        )
    return float(settings["sigmaper_post"]), float(settings["sigmapar_post"])


def _worker_init():
    """Silence noisy warnings/logging in spawned worker processes."""
    import warnings
    import logging

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.pop("DISPLAY", None)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)


def _worker_run_fisher(args_tuple):
    """Top-level function for multiprocessing (must be picklable)."""
    sample, tracer_bin, zrange, z_eff, param_defaults = args_tuple
    try:
        targets = run_fisher(
            sample,
            tracer_bin=tracer_bin,
            zrange=zrange,
            z_eff=z_eff,
            param_defaults=param_defaults,
        )
        target_vals = [targets[t] for t in TARGET_NAMES]
        if not all(np.isfinite(v) for v in target_vals):
            return None, None
        return sample, target_vals
    except Exception:
        return None, None


def generate_dataset(
    priors: Dict[str, Dict[str, float]],
    n_samples: int,
    tracer_bin: str = "LRG2",
    zrange: Tuple[float, float] | None = None,
    z_eff: float | None = None,
    batch_size: int = 64,
    seed: int = 0,
    sigma_clip: float = 4.0,
    workers: int = 1,
    param_defaults: Dict[str, float] | None = None,
    constraints: Dict[str, Dict] | None = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if constraints is None:
        constraints = CONSTRAINTS

    param_names = list(priors.keys())
    param_rows: List[List[float]] = []
    target_rows: List[List[float]] = []

    total_attempts = 0
    failed = 0
    lhs_seed = seed
    printed_exception = False

    if workers > 1:
        import multiprocessing as mp
        import time as _time

        ctx = mp.get_context("spawn")
        os.environ["_PREP_COVAR_WORKER"] = "1"
        print(f"Using {workers} worker processes (spawn)")
        pool = ctx.Pool(workers, initializer=_worker_init)
        t_start = _time.perf_counter()

        while len(param_rows) < n_samples:
            remaining = n_samples - len(param_rows)
            draw_count = min(max(remaining * 2, batch_size), remaining * 3)
            draws = _constrained_samples(
                priors,
                constraints=constraints,
                n_samples=draw_count,
                seed=lhs_seed,
                sigma_clip=sigma_clip,
            )
            lhs_seed += 1

            tasks = [(s, tracer_bin, zrange, z_eff, param_defaults) for s in draws]
            for sample, target_vals in pool.imap_unordered(_worker_run_fisher, tasks):
                total_attempts += 1
                if sample is None:
                    failed += 1
                else:
                    param_rows.append([sample[p] for p in param_names])
                    target_rows.append(target_vals)

                accepted = len(param_rows)
                if total_attempts % 10 == 0 or accepted >= n_samples:
                    elapsed = _time.perf_counter() - t_start
                    sps = total_attempts / elapsed if elapsed > 0 else 0
                    eta = (n_samples - accepted) / (accepted / elapsed) if accepted > 0 else 0
                    print(
                        f"\r  {accepted:>6}/{n_samples} "
                        f"({100.0 * accepted / n_samples:.1f}%) "
                        f"| {failed} failed | {sps:.1f} samples/s "
                        f"| ETA {eta / 60:.1f}min",
                        end="", flush=True,
                    )

                if accepted >= n_samples:
                    break

        pool.terminate()
        pool.join()
        elapsed = _time.perf_counter() - t_start
        print(
            f"\nDone: {len(param_rows)} accepted, {failed} failed, "
            f"{total_attempts} total in {elapsed:.1f}s"
        )

    else:
        while len(param_rows) < n_samples:
            draws = _constrained_samples(
                priors,
                constraints=constraints,
                n_samples=batch_size,
                seed=lhs_seed,
                sigma_clip=sigma_clip,
            )
            lhs_seed += 1

            for sample in draws:
                total_attempts += 1
                try:
                    targets = run_fisher(
                        sample,
                        tracer_bin=tracer_bin,
                        zrange=zrange,
                        z_eff=z_eff,
                        param_defaults=param_defaults,
                    )
                    target_vals = [targets[t] for t in TARGET_NAMES]
                    if not all(np.isfinite(v) for v in target_vals):
                        failed += 1
                        continue
                    param_rows.append([sample[p] for p in param_names])
                    target_rows.append(target_vals)

                    accepted = len(param_rows)
                    if accepted % 10 == 0 or accepted >= n_samples:
                        rate = accepted / max(total_attempts, 1)
                        print(
                            f"Accepted {accepted:>6}/{n_samples}, "
                            f"failed {failed}, attempts {total_attempts}, "
                            f"acceptance={100.0 * rate:.2f}%"
                        )
                except Exception:
                    failed += 1
                    if not printed_exception:
                        printed_exception = True
                        print("First Fisher failure (showing traceback once):")
                        traceback.print_exc()
                        print(f"Failing sample: {sample}")
                    continue

                if len(param_rows) >= n_samples:
                    break

    X = np.asarray(param_rows, dtype=np.float64)
    y = np.asarray(target_rows, dtype=np.float64)
    return param_names, X, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data for a BAO Fisher error emulator."
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--sigma-clip", type=float, default=4.0)

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1 = serial).",
    )
    parser.add_argument(
        "--tracer-bin",
        dest="tracer_bin",
        type=str,
        default="LRG2",
        choices=TRACER_TYPE_CHOICES,
        help="DESI tracer bin key (must match tracers.yaml).",
    )
    parser.add_argument(
        "--z-eff",
        type=float,
        default=None,
        help="Effective redshift. If omitted, defaults to tracer-bin config.",
    )
    parser.add_argument(
        "--zrange",
        type=float,
        nargs=2,
        default=None,
        metavar=("Z_MIN", "Z_MAX"),
        help="Redshift bin edges for the footprint volume. If omitted, defaults to tracer-bin config.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Tracer name prefix for saved files (e.g. 'LRG1' -> LRG1_train.npz). Defaults to --tracer-bin.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Explicit version number for the training_data/v{N} directory. "
             "If omitted, auto-increments to the next available version.",
    )
    parser.add_argument(
        "--ntracers-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("NTRACERS_LOW", "NTRACERS_HIGH"),
        help="Override the N_tracers prior range. Defaults to tracer-bin low/high from tracers.yaml.",
    )
    parser.add_argument(
        "--priors-json",
        type=str,
        default="",
        help=(
            "JSON dictionary of priors, e.g. "
            '\'{"N_tracers":{"dist":"uniform","low":1e5,"high":1e7}}\''
        ),
    )
    parser.add_argument(
        "--cosmo-model",
        type=str,
        default="base",
        choices=list(COSMO_MODELS.keys()),
        help="Cosmology model determining which params are varied (default: base).",
    )

    sys.argv = [a for a in sys.argv if a.strip()]
    args = parser.parse_args()

    tracer_bin_cfg = get_tracer_config(args.tracer_bin)
    zrange = tuple(args.zrange) if args.zrange is not None else tuple(tracer_bin_cfg["zrange"])
    z_eff = args.z_eff if args.z_eff is not None else float(tracer_bin_cfg["z_eff"])

    cosmo_model = args.cosmo_model
    model_params = COSMO_MODELS[cosmo_model]

    if args.priors_json:
        priors = parse_priors(args.priors_json)
    else:
        varied_keys = ["N_tracers"] + model_params
        priors = {k: dict(DEFAULT_PRIORS[k]) for k in varied_keys}

    if args.ntracers_range is not None:
        priors["N_tracers"] = {
            "dist": "uniform",
            "low": args.ntracers_range[0],
            "high": args.ntracers_range[1],
        }
    else:
        priors["N_tracers"] = {
            "dist": "uniform",
            "low": float(tracer_bin_cfg["low"]),
            "high": float(tracer_bin_cfg["high"]),
        }

    all_cosmo_keys = {"Om", "Ok", "w0", "wa", "hrdrag"}
    fixed_keys = all_cosmo_keys - set(model_params)
    param_defaults = {k: PARAM_DEFAULTS[k] for k in fixed_keys if k in PARAM_DEFAULTS}

    constraints = {
        name: spec for name, spec in CONSTRAINTS.items()
        if all(p in model_params for p in spec["params"])
    }

    save_path = os.path.abspath(
        args.save_path if args.save_path else
        get_default_save_path(analysis="bao", quantity="covar", cosmo_model=cosmo_model)
    )

    print(f"Tracer bin: {args.tracer_bin}")
    print(f"Tracer bin config: {tracer_bin_cfg}")
    print(f"Cosmo model: {cosmo_model} (varied: {model_params})")
    if param_defaults:
        print(f"Fixed params: {param_defaults}")
    print(
        f"N_tracers prior: [{priors['N_tracers']['low']:.3g}, "
        f"{priors['N_tracers']['high']:.3g}]"
    )
    print("Using priors:", priors)
    print(f"Active constraints: {list(constraints.keys())}")
    print(f"Redshift range: {zrange}, z_eff = {z_eff:.3f}")
    print("Writing dataset to:", save_path)

    try:
        param_names, X, y = generate_dataset(
            priors=priors,
            n_samples=args.n_samples,
            tracer_bin=args.tracer_bin,
            zrange=zrange,
            z_eff=z_eff,
            batch_size=args.batch_size,
            seed=args.seed,
            sigma_clip=args.sigma_clip,
            workers=args.workers,
            param_defaults=param_defaults,
            constraints=constraints,
        )
        print(f"Generated dataset with shape X={X.shape}, y={y.shape}")
        save_dataset(
            save_path=save_path,
            param_names=param_names,
            X=X,
            y=y,
            test_size=args.test_size,
            target_names=TARGET_NAMES,
            name=args.name if args.name is not None else args.tracer_bin,
            version=args.version,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
