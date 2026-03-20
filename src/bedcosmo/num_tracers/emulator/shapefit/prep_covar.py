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
    KaiserTracerPowerSpectrumMultipoles,
    ShapeFitPowerSpectrumTemplate,
)
from desilike.theories.primordial_cosmology import get_cosmo

from util import (
    latin_hypercube_samples,
    parse_priors,
    save_dataset,
    to_extractor_params,
    get_default_save_path,
)

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", message=".*EisensteinHu.*")

# Same 5 cosmo priors as prep_shapefit_data, plus N_tracers.
# w0/wa included for base_w_wa model.
DEFAULT_PRIORS = {
    "N_tracers": {"dist": "uniform", "low": 1e5, "high": 1e7},
    "omega_cdm": {"dist": "uniform", "low": 0.01, "high": 0.99},
    "omega_b": {"dist": "normal", "mu": 0.02218, "sigma": 0.00055},
    "h": {"dist": "uniform", "low": 0.2, "high": 1.0},
    "ln10A_s": {"dist": "uniform", "low": 1.61, "high": 3.91},
    "n_s": {"dist": "normal", "mu": 0.9649, "sigma": 0.042},
    "w0": {"dist": "uniform", "low": -3.0, "high": 1.0},
    "wa": {"dist": "uniform", "low": -3.0, "high": 2.0},
}

COSMO_MODELS = {
    "base":     ["omega_cdm", "omega_b", "h", "ln10A_s", "n_s"],
    "base_w_wa": ["omega_cdm", "omega_b", "h", "ln10A_s", "n_s", "w0", "wa"],
}

# Fiducial values for fixed parameters
PARAM_DEFAULTS = {"w0": -1.0, "wa": 0.0}

_PHYS_NAMES = ["qiso", "qap", "f_sigmar", "m"]
_TRIU_I, _TRIU_J = np.triu_indices(4)
TARGET_NAMES = [f"cov_{_PHYS_NAMES[i]}_{_PHYS_NAMES[j]}" for i, j in zip(_TRIU_I, _TRIU_J)]


def get_shapefit_fisher_covariance(
    N_tracers: float,
    theta_cosmo: Dict[str, float],
    zrange: Tuple[float, float] = (1.2, 1.4),
    z_eff: float | None = None,
    area: float = 14000.0,
    b0: float = 0.84,
    resolution: int = 3,
) -> Dict[str, float]:
    """Compute the 4x4 ShapeFit covariance matrix from a Fisher forecast.

    Returns the upper-triangular elements (10 values) of the physical-basis
    covariance matrix for ``(qiso, qap, f_sigmar, m)``.

    desilike's Fisher internally uses ``df`` and ``dm``, which relate to
    the physical quantities as::

        f_sigmar = df * f_sigmar_fid
        m        = m_fid + dm

    A Jacobian transform ``J = diag(1, 1, f_sigmar_fid, 1)`` is applied
    to convert to the physical basis.
    """
    z = z_eff if z_eff is not None else np.mean(zrange)
    dz = np.diff(zrange)[0]

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

    template = ShapeFitPowerSpectrumTemplate(
        z=z,
        fiducial=("DESI", dict(theta_cosmo)),
        apmode='qisoqap'
    )
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)

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

    fisher = Fisher(likelihood)
    fisher_result = fisher(**params)

    # Full parameter covariance = inverse of Fisher matrix (= -Hessian)
    F_matrix = -np.array(fisher_result._hessian)
    cov_full = np.linalg.inv(F_matrix)

    # Extract the 4x4 ShapeFit sub-block: (qiso, qap, df, dm)
    all_names = [str(p) for p in fisher_result.names()]
    sf_internal = ["qiso", "qap", "df", "dm"]
    sf_idx = [all_names.index(p) for p in sf_internal]
    cov_sf = cov_full[np.ix_(sf_idx, sf_idx)]

    # Jacobian to physical basis (qiso, qap, f_sigmar, m):
    # f_sigmar = df * f_sigmar_fid  (index 2), m = dm (index 3)
    f_sigmar_fid = float(template.f_sigmar_fid)
    J = np.diag([1.0, 1.0, f_sigmar_fid, 1.0])
    cov_phys = J @ cov_sf @ J.T

    # Extract upper-triangular elements (10 values)
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
    theta_cosmo = to_extractor_params(sample)
    return get_shapefit_fisher_covariance(N_tracers, theta_cosmo, zrange=zrange, z_eff=z_eff)


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
) -> Tuple[List[str], np.ndarray, np.ndarray]:
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
            draws = latin_hypercube_samples(
                priors, n_samples=draw_count, seed=lhs_seed, sigma_clip=sigma_clip,
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
            draws = latin_hypercube_samples(
                priors,
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
        description="Generate training data for a ShapeFit Fisher error emulator."
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
        default="base",
        choices=list(COSMO_MODELS.keys()),
        help="Cosmology model defining which parameters to vary (default: base).",
    )
    # Strip empty/whitespace args that can appear from shell line continuation
    sys.argv = [a for a in sys.argv if a.strip()]
    args = parser.parse_args()

    zrange = tuple(args.zrange)
    z_eff = args.z_eff

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
    fixed_keys = set(PARAM_DEFAULTS.keys()) - set(model_params)
    param_defaults = {k: PARAM_DEFAULTS[k] for k in fixed_keys}

    z_eff_actual = z_eff if z_eff is not None else np.mean(zrange)
    save_path = os.path.abspath(args.save_path if args.save_path else get_default_save_path(analysis="shapefit", quantity="covar", cosmo_model=cosmo_model))
    print(f"Cosmo model: {cosmo_model} (varied: {model_params})")
    if param_defaults:
        print(f"Fixed params: {param_defaults}")
    print("Using priors:", priors)
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
