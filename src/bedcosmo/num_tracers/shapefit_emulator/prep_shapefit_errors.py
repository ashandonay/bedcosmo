import argparse
import os
import sys
import traceback
import warnings
from typing import Dict, List, Tuple

import numpy as np
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

warnings.filterwarnings("ignore", message=".*EisensteinHu.*")

# Same 5 cosmo priors as prep_shapefit_data, plus nbar.
DEFAULT_PRIORS = {
    "nbar": {"dist": "uniform", "low": 100.0, "high": 5000.0},
    "omega_cdm": {"dist": "uniform", "low": 0.01, "high": 0.99},
    "omega_b": {"dist": "normal", "mu": 0.02218, "sigma": 0.00055},
    "h": {"dist": "uniform", "low": 0.2, "high": 1.0},
    "ln10A_s": {"dist": "uniform", "low": 1.61, "high": 3.91},
    "n_s": {"dist": "normal", "mu": 0.9649, "sigma": 0.042},
}

_PHYS_NAMES = ["qiso", "qap", "f_sigmar", "m"]
_TRIU_I, _TRIU_J = np.triu_indices(4)
TARGET_NAMES = [f"cov_{_PHYS_NAMES[i]}_{_PHYS_NAMES[j]}" for i, j in zip(_TRIU_I, _TRIU_J)]


def get_shapefit_fisher_covariance(
    nbar: float,
    theta_cosmo: Dict[str, float],
    zrange: Tuple[float, float] = (1.2, 1.4),
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
    z = np.mean(zrange)
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

    footprint = CutskyFootprint(
        area=area,
        zrange=zrange,
        nbar=nbar * dz,
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


def run_fisher(sample: Dict[str, float]) -> Dict[str, float]:
    """Convert a sample dict (with nbar + cosmo params) to Fisher covariance elements."""
    nbar = sample["nbar"]
    theta_cosmo = to_extractor_params(sample)
    return get_shapefit_fisher_covariance(nbar, theta_cosmo)


def generate_dataset(
    priors: Dict[str, Dict[str, float]],
    n_samples: int,
    batch_size: int = 64,
    seed: int = 0,
    verbose_every: int = 200,
    sigma_clip: float = 4.0,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    param_names = list(priors.keys())
    param_rows: List[List[float]] = []
    target_rows: List[List[float]] = []

    total_attempts = 0
    failed = 0
    lhs_seed = seed
    printed_exception = False

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
                targets = run_fisher(sample)
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

            if len(param_rows) >= n_samples:
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
    parser.add_argument(
        "--priors-json",
        type=str,
        default="",
        help=(
            "JSON dictionary of priors, e.g. "
            '\'{"nbar":{"dist":"uniform","low":100,"high":5000}}\''
        ),
    )
    # Strip empty/whitespace args that can appear from shell line continuation
    sys.argv = [a for a in sys.argv if a.strip()]
    args = parser.parse_args()

    priors = DEFAULT_PRIORS if not args.priors_json else parse_priors(args.priors_json)
    save_path = os.path.abspath(args.save_path if args.save_path else get_default_save_path(type="shapefit_errors"))
    print("Using priors:", priors)
    print("Writing dataset to:", save_path)

    try:
        param_names, X, y = generate_dataset(
            priors=priors,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose_every=args.verbose_every,
            sigma_clip=args.sigma_clip,
        )
        print(f"Generated dataset with shape X={X.shape}, y={y.shape}")
        save_dataset(
            save_path=save_path,
            param_names=param_names,
            X=X,
            y=y,
            test_size=args.test_size,
            target_names=TARGET_NAMES,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
