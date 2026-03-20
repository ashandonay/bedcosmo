import argparse
import os
import sys
import traceback
import warnings
from typing import Dict, List, Tuple

# In spawned worker processes, suppress all C-level stderr (X11, JAX, etc.)
# before any library imports can trigger it.
if os.environ.get("_PREP_COVAR_WORKER") == "1":
    os.environ.pop("DISPLAY", None)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["JAX_PLATFORMS"] = "cpu"
    warnings.filterwarnings("ignore")
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)

os.environ.setdefault("MPLBACKEND", "Agg")

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
    latin_hypercube_samples,
    parse_priors,
    save_dataset,
    get_default_save_path,
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
_N_S_FID = 0.9649
_LN10A_S_FID = 3.044
_PHYS_NAMES = ["DH_over_rd", "DM_over_rd"]
_TRIU_I, _TRIU_J = np.triu_indices(2)
TARGET_NAMES = [f"cov_{_PHYS_NAMES[i]}_{_PHYS_NAMES[j]}" for i, j in zip(_TRIU_I, _TRIU_J)]


def _sample_constrained_pair(
    low1: float, high1: float,
    low2: float, high2: float,
    n: int,
    rng: np.random.Generator,
    sum_lower: float | None = None,
    sum_upper: float | None = None,
) -> np.ndarray:
    """Sample n points uniformly from a 2D box with a linear sum constraint.

    Mirrors the ConstrainedUniform2D logic from bedcosmo.custom_dist but
    uses pure numpy rejection sampling (fast for 2D).
    """
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
    """Draw samples with LHS for unconstrained params and rejection for constrained pairs.

    Constrained pairs (Om,Ok) and (w0,wa) are sampled uniformly from their
    valid region (matching ConstrainedUniform2D).  Remaining params use LHS.
    """
    rng = np.random.default_rng(seed)

    # Identify which params are in constrained pairs
    constrained_keys: set = set()
    for spec in constraints.values():
        constrained_keys.update(spec["params"])

    # LHS for unconstrained params
    unconstrained_priors = {k: v for k, v in priors.items() if k not in constrained_keys}
    if unconstrained_priors:
        lhs_draws = latin_hypercube_samples(
            unconstrained_priors, n_samples=n_samples, seed=seed, sigma_clip=sigma_clip,
        )
    else:
        lhs_draws = [{}] * n_samples

    # Sample each constrained pair
    pair_samples: Dict[str, np.ndarray] = {}
    for cname, cspec in constraints.items():
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

    # Merge into list of dicts
    rows: List[Dict[str, float]] = []
    for i in range(n_samples):
        row = dict(lhs_draws[i])
        for k, arr in pair_samples.items():
            row[k] = float(arr[i])
        rows.append(row)
    return rows


def _to_bao_cosmo_params(sample: Dict[str, float]) -> Dict[str, float]:
    """Convert BAO prior parameters to desilike cosmology parameters.

    Uses cosmoprimo's ``Cosmology.solve()`` to find h such that
    ``h * r_drag == hrdrag``, following the same pattern desilike uses
    for ``theta_MC_100`` (see ``primordial_cosmology._clone``).
    Early-universe parameters (omega_b, n_s, ln10A_s) are fixed to DESI
    fiducials.

    Raises ValueError if the solver cannot converge (e.g. because
    the (Om, hrdrag) combination is unphysical).
    """
    Om = float(sample["Om"])
    Ok = float(sample["Ok"])
    w0 = float(sample["w0"])
    wa = float(sample["wa"])
    hrdrag = float(sample["hrdrag"])

    cosmo = get_cosmo(("DESI", {
        "Omega_m": Om, "Omega_k": Ok,
        "w0_fld": w0, "wa_fld": wa,
        "omega_b": _OMEGA_B_FID,
        "n_s": _N_S_FID, "ln10A_s": _LN10A_S_FID,
    }))
    cosmo = cosmo.solve(
        param="h",
        func=lambda c: c.h * c.rs_drag,
        target=hrdrag,
    )
    omega_cdm = Om * cosmo.h**2 - _OMEGA_B_FID
    return {
        "h": float(cosmo.h), "omega_cdm": float(omega_cdm),
        "omega_b": _OMEGA_B_FID,
        "n_s": _N_S_FID, "ln10A_s": _LN10A_S_FID,
        "Omega_k": Ok, "w0_fld": w0, "wa_fld": wa,
    }


def get_bao_fisher_covariance(
    N_tracers: float,
    theta_cosmo: Dict[str, float],
    zrange: Tuple[float, float] = (1.2, 1.4),
    z_eff: float | None = None,
    area: float = 14000.0,
    b0: float = 0.84,
    resolution: int = 3,
) -> Dict[str, float]:
    """Compute the 2x2 BAO covariance matrix from a Fisher forecast.

    Returns the upper-triangular elements (3 values) of the physical-basis
    covariance matrix for ``(DH/rd, DM/rd)``.

    desilike's Fisher internally uses ``qpar`` and ``qper``, which relate to
    the physical distances as::

        DH/rd = qpar * (DH/rd)_fid
        DM/rd = qper * (DM/rd)_fid

    A Jacobian transform ``J = diag((DH/rd)_fid, (DM/rd)_fid)`` is applied
    to convert to the physical basis.
    """
    z = z_eff if z_eff is not None else np.mean(zrange)

    cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
    fo = cosmo.get_fourier()

    r = 0.5
    sigmaper = r * 12.4 * 0.758 * fo.sigma8_z(z, of="delta_cb") / 0.9
    f = fo.sigma8_z(z, of="theta_cb") / fo.sigma8_z(z, of="delta_cb")
    b1 = b0 * fo.sigma8_cb / fo.sigma8_z(z, of="delta_cb")
    params = {
        "b1": b1,
        "sigmapar": (1.0 + f) * sigmaper,
        "sigmaper": sigmaper,
    }

    # Convert N_tracers to angular density for the footprint
    nbar = N_tracers / area
    footprint = CutskyFootprint(
        area=area,
        zrange=zrange,
        nbar=nbar,
        cosmo=cosmo,
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
    # Fix sigmas (FoG damping): it does not affect BAO wiggles and would
    # create a flat direction in the Fisher matrix, making it singular.
    likelihood.all_params['sigmas'].update(fixed=True)

    fisher = Fisher(likelihood)
    fisher_result = fisher(**params)

    # Full parameter covariance = inverse of Fisher matrix (= -Hessian)
    F_matrix = -np.array(fisher_result._hessian)
    cov_full = np.linalg.inv(F_matrix)

    # Extract the 2x2 BAO sub-block: (qpar, qper)
    all_names = [str(p) for p in fisher_result.names()]
    bao_internal = ["qpar", "qper"]
    bao_idx = [all_names.index(p) for p in bao_internal]
    cov_q = cov_full[np.ix_(bao_idx, bao_idx)]

    # Jacobian to physical basis (DH/rd, DM/rd):
    # DH/rd = qpar * (DH/rd)_fid,  DM/rd = qper * (DM/rd)_fid
    DH_over_rd_fid = float(template.DH_over_rd_fid)
    DM_over_rd_fid = float(template.DM_over_rd_fid)
    J = np.diag([DH_over_rd_fid, DM_over_rd_fid])
    cov_phys = J @ cov_q @ J.T

    # Extract upper-triangular elements (3 values)
    upper_tri_vals = cov_phys[_TRIU_I, _TRIU_J]
    return dict(zip(TARGET_NAMES, upper_tri_vals))


def run_fisher(
    sample: Dict[str, float],
    zrange: Tuple[float, float] = (1.2, 1.4),
    z_eff: float | None = None,
    param_defaults: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Convert a sample dict (with N_tracers + cosmo params) to Fisher covariance elements."""
    if param_defaults:
        sample = {**param_defaults, **sample}
    N_tracers = sample["N_tracers"]
    theta_cosmo = _to_bao_cosmo_params(sample)
    return get_bao_fisher_covariance(N_tracers, theta_cosmo, zrange=zrange, z_eff=z_eff)


def _worker_init():
    """Silence noisy warnings/logging in spawned worker processes."""
    import warnings
    import logging
    import os
    import sys
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.pop("DISPLAY", None)
    # Redirect the real OS file descriptor 2 to suppress C-level stderr (X11, JAX, etc.)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)


def _worker_run_fisher(args_tuple):
    """Top-level function for multiprocessing (must be picklable)."""
    sample, zrange, z_eff, param_defaults = args_tuple
    try:
        targets = run_fisher(sample, zrange=zrange, z_eff=z_eff, param_defaults=param_defaults)
        target_vals = [targets[t] for t in TARGET_NAMES]
        if not all(np.isfinite(v) for v in target_vals):
            return None, None
        return sample, target_vals
    except Exception:
        return None, None


def generate_dataset(
    priors: Dict[str, Dict[str, float]],
    n_samples: int,
    zrange: Tuple[float, float] = (1.2, 1.4),
    z_eff: float | None = None,
    batch_size: int = 64,
    seed: int = 0,
    verbose_every: int = 200,
    sigma_clip: float = 4.0,
    workers: int = 1,
    checkpoint_fn=None,
    checkpoint_every: int = 1000,
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
    last_checkpoint = 0

    if workers > 1:
        import multiprocessing as mp
        import time as _time

        ctx = mp.get_context("spawn")
        # Flag so workers suppress stderr at module level (before imports)
        os.environ["_PREP_COVAR_WORKER"] = "1"
        print(f"Using {workers} worker processes (spawn)")
        pool = ctx.Pool(workers, initializer=_worker_init)
        t_start = _time.perf_counter()

        while len(param_rows) < n_samples:
            remaining = n_samples - len(param_rows)
            draw_count = min(max(remaining * 2, batch_size), remaining * 3)
            draws = _constrained_samples(
                priors, constraints=constraints,
                n_samples=draw_count, seed=lhs_seed, sigma_clip=sigma_clip,
            )
            lhs_seed += 1

            tasks = [(s, zrange, z_eff, param_defaults) for s in draws]
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
                    rate = accepted / max(total_attempts, 1)
                    sps = total_attempts / elapsed if elapsed > 0 else 0
                    eta = (n_samples - accepted) / (accepted / elapsed) if accepted > 0 else 0
                    print(
                        f"\r  {accepted:>6}/{n_samples} "
                        f"({100.0 * accepted / n_samples:.1f}%) "
                        f"| {failed} failed | {sps:.1f} samples/s "
                        f"| ETA {eta / 60:.1f}min",
                        end="", flush=True,
                    )

                # Periodic checkpoint
                if checkpoint_fn and accepted >= last_checkpoint + checkpoint_every:
                    last_checkpoint = (accepted // checkpoint_every) * checkpoint_every
                    checkpoint_fn(param_names, param_rows, target_rows)

                if accepted >= n_samples:
                    break

        pool.terminate()
        pool.join()
        elapsed = _time.perf_counter() - t_start
        print(f"\nDone: {len(param_rows)} accepted, {failed} failed, "
              f"{total_attempts} total in {elapsed:.1f}s")

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
                    targets = run_fisher(sample, zrange=zrange, z_eff=z_eff, param_defaults=param_defaults)
                    target_vals = [targets[t] for t in TARGET_NAMES]
                    if not all(np.isfinite(v) for v in target_vals):
                        failed += 1
                        continue
                    param_rows.append([sample[p] for p in param_names])
                    target_rows.append(target_vals)
                except Exception:
                    failed += 1
                    if not printed_exception:
                        printed_exception = True
                        print("First Fisher failure (showing traceback once):")
                        traceback.print_exc()
                        print(f"Failing sample: {sample}")
                    continue

                # Periodic checkpoint
                accepted = len(param_rows)
                if checkpoint_fn and accepted >= last_checkpoint + checkpoint_every:
                    last_checkpoint = (accepted // checkpoint_every) * checkpoint_every
                    checkpoint_fn(param_names, param_rows, target_rows)

                if accepted >= n_samples:
                    break
            if total_attempts % verbose_every == 0 or len(param_rows) >= n_samples:
                accepted = len(param_rows)
                rate = accepted / max(total_attempts, 1)
                print(
                    f"Accepted {accepted:>6}/{n_samples}, "
                    f"failed {failed}, attempts {total_attempts}, "
                    f"acceptance={100.0 * rate:.2f}%"
                )

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
    parser.add_argument("--verbose-every", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1 = serial).")
    parser.add_argument(
        "--z-eff", type=float, default=None,
        help="Effective redshift (can differ from midpoint of --zrange).",
    )
    parser.add_argument(
        "--zrange", type=float, nargs=2, default=[1.2, 1.4],
        metavar=("Z_MIN", "Z_MAX"),
        help="Redshift bin edges for the footprint volume (default: 1.2 1.4).",
    )
    parser.add_argument(
        "--name", type=str, default="",
        help="Tracer name prefix for saved files (e.g. 'LRG1' -> LRG1_train.npz, LRG1_test.npz).",
    )
    parser.add_argument(
        "--version", type=int, default=None,
        help="Explicit version number for the training_data/v{N} directory. "
             "If omitted, auto-increments to the next available version.",
    )
    parser.add_argument(
        "--ntracers-range", type=float, nargs=2, default=None,
        metavar=("NTRACERS_LOW", "NTRACERS_HIGH"),
        help="Override the N_tracers prior range (default: from DEFAULT_PRIORS).",
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
        default="base_omegak_w_wa",
        choices=list(COSMO_MODELS.keys()),
        help="Cosmology model defining which parameters to vary (default: base_omegak_w_wa).",
    )
    # Strip empty/whitespace args that can appear from shell line continuation
    sys.argv = [a for a in sys.argv if a.strip()]
    args = parser.parse_args()

    zrange = tuple(args.zrange)
    z_eff = args.z_eff  # None means use midpoint of zrange

    # Build priors: only include params for the selected cosmo model
    cosmo_model = args.cosmo_model
    model_params = COSMO_MODELS[cosmo_model]
    if args.priors_json:
        priors = parse_priors(args.priors_json)
    else:
        varied_keys = ["N_tracers"] + model_params
        priors = {k: dict(DEFAULT_PRIORS[k]) for k in varied_keys}
    if args.ntracers_range is not None:
        priors["N_tracers"] = {"dist": "uniform", "low": args.ntracers_range[0], "high": args.ntracers_range[1]}

    # Fixed defaults for non-varied cosmo params
    all_cosmo_keys = {"Om", "Ok", "w0", "wa", "hrdrag"}
    fixed_keys = all_cosmo_keys - set(model_params)
    param_defaults = {k: PARAM_DEFAULTS[k] for k in fixed_keys if k in PARAM_DEFAULTS}

    # Filter constraints: only keep those whose params are both varied
    constraints = {
        name: spec for name, spec in CONSTRAINTS.items()
        if all(p in model_params for p in spec["params"])
    }

    z_eff_actual = z_eff if z_eff is not None else np.mean(zrange)
    save_path = os.path.abspath(args.save_path if args.save_path else get_default_save_path(analysis="bao", quantity="covar", cosmo_model=cosmo_model))
    print(f"Cosmo model: {cosmo_model} (varied: {model_params})")
    if param_defaults:
        print(f"Fixed params: {param_defaults}")
    print("Using priors:", priors)
    print(f"Active constraints: {list(constraints.keys())}")
    print(f"Redshift range: {zrange}, z_eff = {z_eff_actual:.2f}")
    print("Writing dataset to:", save_path)

    def checkpoint_fn(param_names, param_rows, target_rows):
        X_ckpt = np.asarray(param_rows, dtype=np.float64)
        y_ckpt = np.asarray(target_rows, dtype=np.float64)
        print(f"\n  Checkpoint: saving {len(param_rows)} samples...")
        save_dataset(
            save_path=save_path,
            param_names=param_names,
            X=X_ckpt,
            y=y_ckpt,
            test_size=args.test_size,
            target_names=TARGET_NAMES,
            name=args.name,
            version=args.version,
        )

    try:
        param_names, X, y = generate_dataset(
            priors=priors,
            n_samples=args.n_samples,
            zrange=zrange,
            z_eff=z_eff,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose_every=args.verbose_every,
            sigma_clip=args.sigma_clip,
            workers=args.workers,
            checkpoint_fn=checkpoint_fn,
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
