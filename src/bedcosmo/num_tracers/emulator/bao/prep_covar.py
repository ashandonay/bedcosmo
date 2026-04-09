import os

# Set display/backend env vars before any library imports can trigger X11 or GPU probes.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import traceback
import warnings
from typing import Dict, List, Tuple

# In spawned worker processes, additionally suppress all C-level stderr.
if os.environ.get("_PREP_COVAR_WORKER") == "1":
    os.environ.pop("DISPLAY", None)
    warnings.filterwarnings("ignore")
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

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
    SimpleBAOWigglesTracerPowerSpectrumMultipoles,
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


def _sigma_nl_pre_from_pk(pk_lin_1d, f: float) -> Tuple[float, float]:
    """
    Compute pre-reconstruction BAO damping scales from the linear displacement variance.

    Sigma_nl^2 = (1 / 6 pi^2) * integral dk P_lin(k)
    Sigma_perp_pre = Sigma_nl
    Sigma_par_pre  = (1 + f) * Sigma_nl
    """
    k = np.geomspace(_K_DISP_MIN, _K_DISP_MAX, _NK_DISP)
    pk = np.clip(pk_lin_1d(k), 0.0, None)

    sigma_nl_sq = np.trapz(pk, k) / (6.0 * np.pi**2)
    sigma_nl = np.sqrt(max(float(sigma_nl_sq), 0.0))

    sigma_perp_pre = sigma_nl
    sigma_par_pre = (1.0 + f) * sigma_nl
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
) -> Tuple[float, float]:
    """
    Anisotropic post-reconstruction residual BAO damping via 2D (k, mu) integration.

    The Wiener filter uses the redshift-space galaxy power spectrum
    P_g(k, mu) = (b1 + f*mu^2)^2 P_lin(k), so reconstruction efficiency
    varies with mu (line-of-sight vs transverse).

    W(k, mu) = P_g(k, mu) / [P_g(k, mu) + 1/nbar]
    T_res(k, mu) = 1 - S(k) W(k, mu)

    sigma_perp^2 = 1/(2 pi^2) int dk P(k) int_0^1 dmu (1-mu^2)/2 T_res^2
    sigma_par^2  = (1+f)^2 / (2 pi^2) int dk P(k) int_0^1 dmu mu^2 T_res^2
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
    T_res = 1.0 - S * W

    # Perpendicular: one transverse component, weight (1-mu^2)/2
    integrand_perp = pk * T_res**2 * (1.0 - mm**2) / 2.0
    sigma_perp_sq = (
        np.trapz(np.trapz(integrand_perp, mu, axis=1), k) / (2.0 * np.pi**2)
    )

    # Parallel: LOS component, weight mu^2, times (1+f)^2 from RSD displacement
    integrand_par = pk * T_res**2 * mm**2
    sigma_par_sq = (
        (1.0 + f) ** 2
        * np.trapz(np.trapz(integrand_par, mu, axis=1), k)
        / (2.0 * np.pi**2)
    )

    sigma_perp_post = np.sqrt(max(float(sigma_perp_sq), 0.0))
    sigma_par_post = np.sqrt(max(float(sigma_par_sq), 0.0))
    return sigma_perp_post, sigma_par_post


def _get_lya_qso_bao_params(
    cfg: Dict[str, float],
    z: float,
    sigma_perp_pre: float,
    sigma_par_pre: float,
    nbar_comoving: float,
    base_volume: float,
    area: float,
) -> Dict[str, float]:
    """Return Ly-alpha QSO BAO nuisance parameters for Fisher forecasts."""
    b1 = _lya_flux_bias(
        z=z,
        bF0=float(cfg.get("bF0", -0.117)),
        z_ref=float(cfg.get("z_ref", 2.334)),
        gamma=float(cfg.get("gamma_bF", 2.9)),
    )
    sigma_perp_post, sigma_par_post = sigma_perp_pre, sigma_par_pre

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
        "sigmaper_post": sigma_perp_post,
        "sigmapar_post": sigma_par_post,
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
) -> Dict[str, float]:
    """Return standard galaxy-tracer BAO nuisance parameters for Fisher forecasts."""
    b1 = float(cfg["bias_recon"]) * float(fo.sigma8_cb) / sigma8_delta
    sigma_perp_post, sigma_par_post = _sigma_nl_post_from_recon(
        pk_lin_1d=pk_lin_1d,
        b1=b1,
        f=f,
        nbar_comoving=nbar_comoving,
        smoothing_scale=float(cfg["smoothing_scale"]),
    )
    nbar_eff = nbar_comoving
    P_N_eff = 1.0 / max(nbar_eff, 1.0e-30)
    nbar_for_footprint = nbar_ang

    return {
        "is_lya": False,
        "b1": b1,
        "sigmaper_post": sigma_perp_post,
        "sigmapar_post": sigma_par_post,
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
) -> Dict[str, float]:
    """
    Compute the 2x2 BAO covariance matrix from a Fisher forecast.

    Realism upgrades:
    - f(z) computed from sampled cosmology
    - pre-reconstruction damping from linear P(k)
    - post-reconstruction residual damping from a semi-analytic reconstruction model
      using tracer bias, smoothing scale, and comoving number density
    """
    cfg = get_tracer_config(tracer_bin)
    if tracer_config is not None:
        cfg.update(tracer_config)

    if zrange is None:
        zrange = tuple(cfg["zrange"])
    if z_eff is None:
        z_eff = float(cfg["z_eff"])
    z = z_eff

    cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
    fo = cosmo.get_fourier()

    sigma8_delta = float(fo.sigma8_z(z, of="delta_cb"))
    sigma8_theta = float(fo.sigma8_z(z, of="theta_cb"))
    if sigma8_delta <= 0:
        raise ValueError(f"sigma8_delta <= 0 at z={z:.3f}: {sigma8_delta}")

    f = sigma8_theta / sigma8_delta

    nbar_ang = N_tracers / area  # deg^-2 for scalar CutskyFootprint input
    # First pass footprint gives survey volume for converting surface counts to 3D nbar.
    base_footprint = CutskyFootprint(
        area=area,
        zrange=zrange,
        nbar=nbar_ang,
        cosmo=cosmo,
    )
    nbar_comoving = float(N_tracers) / float(base_footprint.volume)  # (h/Mpc)^3

    pk_lin_1d = _linear_pk_1d(fo, z=z)
    sigma_perp_pre, sigma_par_pre = _sigma_nl_pre_from_pk(pk_lin_1d, f=f)

    if tracer_bin == "Lya_QSO":
        tracer_settings = _get_lya_qso_bao_params(
            cfg=cfg,
            z=z,
            sigma_perp_pre=sigma_perp_pre,
            sigma_par_pre=sigma_par_pre,
            nbar_comoving=nbar_comoving,
            base_volume=float(base_footprint.volume),
            area=area,
        )
    else:
        tracer_settings = _get_standard_tracer_bao_params(
            cfg=cfg,
            fo=fo,
            sigma8_delta=sigma8_delta,
            f=f,
            pk_lin_1d=pk_lin_1d,
            nbar_comoving=nbar_comoving,
            nbar_ang=nbar_ang,
        )
    is_lya = bool(tracer_settings["is_lya"])
    b1 = float(tracer_settings["b1"])
    sigma_perp_post = float(tracer_settings["sigmaper_post"])
    sigma_par_post = float(tracer_settings["sigmapar_post"])
    nbar_eff = float(tracer_settings["nbar_eff"])
    P_N_eff = float(tracer_settings["P_N_eff"])
    nbar_for_footprint = float(tracer_settings["nbar_for_footprint"])

    params = {
        "b1": b1,
        "sigmaper": sigma_perp_post,
        "sigmapar": sigma_par_post,
    }

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
            "sigmapar_post": sigma_par_post,
        },
    )

    template = BAOPowerSpectrumTemplate(
        z=z,
        fiducial=("DESI", dict(theta_cosmo)),
        apmode="qparqper",
    )
    theory = SimpleBAOWigglesTracerPowerSpectrumMultipoles(template=template)

    observable = TracerPowerSpectrumMultipolesObservable(
        data=params,
        klim={0: [0.01, 0.5, 0.01], 2: [0.01, 0.5, 0.01]},
        theory=theory,
    )

    covariance = ObservablesCovarianceMatrix(
        observable, footprints=footprint, resolution=resolution
    )

    likelihood = ObservablesGaussianLikelihood(
        observables=observable,
        covariance=covariance(**params),
    )

    # In this BAO-only setup these can create a singular Fisher direction.
    likelihood.all_params["sigmas"].update(fixed=True)

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
    cov_full = np.linalg.inv(F_matrix)

    all_names = [str(p) for p in fisher_result.names()]
    bao_internal = ["qpar", "qper"]
    bao_idx = [all_names.index(p) for p in bao_internal]
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
    )


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

        pool.close()
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
        default="",
        help="Tracer name prefix for saved files (e.g. 'LRG1' -> LRG1_train.npz, LRG1_test.npz).",
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
            name=args.name,
            version=args.version,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
