import argparse
import json
import os
import traceback
from typing import Dict, List, Tuple

import numpy as np
from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumExtractor
from scipy.stats import qmc
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split


# Fisher-matrix style priors (normals are truncated at +/-4 sigma by default).
DEFAULT_PRIORS = {
    "omega_cdm": {"dist": "uniform", "low": 0.01, "high": 0.99},
    "omega_b": {"dist": "normal", "mu": 0.02218, "sigma": 0.00055},
    "h": {"dist": "uniform", "low": 0.2, "high": 1.0},
    "ln10A_s": {"dist": "uniform", "low": 1.61, "high": 3.91},
    "n_s": {"dist": "normal", "mu": 0.9649, "sigma": 0.042},
}

TARGET_NAMES = ["qiso", "qap", "f_sigmar", "m"]


def get_default_save_path() -> str:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise EnvironmentError("SCRATCH is not set; please pass --save-path explicitly.")
    return os.path.join(scratch, "bedcosmo", "num_tracers", "shapefit")


def latin_hypercube_samples(
    priors: Dict[str, Dict[str, float]],
    n_samples: int,
    seed: int,
    sigma_clip: float = 4.0,
) -> List[Dict[str, float]]:
    keys = list(priors.keys())
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    unit_samples = sampler.random(n=n_samples)

    rows: List[Dict[str, float]] = []
    for urow in unit_samples:
        out: Dict[str, float] = {}
        for key, u in zip(keys, urow):
            spec = priors[key]
            dist = spec["dist"]
            if dist == "uniform":
                low = float(spec["low"])
                high = float(spec["high"])
                out[key] = low + (high - low) * float(u)
            elif dist == "normal":
                mu = float(spec["mu"])
                sigma = float(spec["sigma"])
                a = -sigma_clip
                b = sigma_clip
                out[key] = float(truncnorm.ppf(u, a, b, loc=mu, scale=sigma))
            else:
                raise ValueError(f"Unsupported dist '{dist}' for '{key}'")
        rows.append(out)
    return rows


def to_extractor_params(sample: Dict[str, float]) -> Dict[str, float]:
    # omega_* are physical densities: omega_x = Omega_x * h^2
    # so Omega_m = (omega_cdm + omega_b) / h^2.
    omega_cdm = sample["omega_cdm"]
    omega_b = sample["omega_b"]
    h = sample["h"]
    if h <= 0.0:
        raise ValueError("h must be > 0 to compute Omega_m")
    omega_m = (omega_cdm + omega_b) / (h * h)
    return {
        "Omega_m": float(omega_m),
        "n_s": float(sample["n_s"]),
    }


def run_extractor(
    extractor: ShapeFitPowerSpectrumExtractor, sample: Dict[str, float]
) -> Dict[str, float]:
    extractor_params = to_extractor_params(sample)
    extractor(**extractor_params)
    extractor.get()
    return {
        "qiso": float(extractor.qiso),
        "qap": float(extractor.qap),
        "f_sigmar": float(extractor.f_sigmar),
        "m": float(extractor.m),
    }


def generate_dataset(
    priors: Dict[str, Dict[str, float]],
    n_samples: int,
    batch_size: int = 256,
    seed: int = 0,
    verbose_every: int = 200,
    sigma_clip: float = 4.0,
    omega_m_bounds: Tuple[float, float] = (0.05, 1.0),
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    extractor = ShapeFitPowerSpectrumExtractor()
    param_names = list(priors.keys())
    param_rows: List[List[float]] = []
    target_rows: List[List[float]] = []

    total_attempts = 0
    failed = 0
    lhs_seed = seed

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
                omega_m = (sample["omega_cdm"] + sample["omega_b"]) / (sample["h"] ** 2)
                if not (omega_m_bounds[0] <= omega_m <= omega_m_bounds[1]):
                    failed += 1
                    continue
                targets = run_extractor(extractor, sample)
                param_rows.append([sample[p] for p in param_names])
                target_rows.append([targets[t] for t in TARGET_NAMES])
            except Exception:
                failed += 1
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


def save_dataset(
    save_path: str,
    param_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 1,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    np.savez(
        f"{save_path}/shapefit_train.npz",
        x=X_train,
        y=y_train,
        param_names=np.array(param_names),
        target_names=np.array(TARGET_NAMES),
    )
    np.savez(
        f"{save_path}/shapefit_test.npz",
        x=X_test,
        y=y_test,
        param_names=np.array(param_names),
        target_names=np.array(TARGET_NAMES),
    )
    print(f"Saved train/test files to: {save_path}")


def parse_priors(priors_json: str) -> Dict[str, Dict[str, float]]:
    raw = json.loads(priors_json)
    if not isinstance(raw, dict):
        raise ValueError("Priors JSON must be a dictionary")
    for name, spec in raw.items():
        if not isinstance(spec, dict) or "dist" not in spec:
            raise ValueError(f"Prior '{name}' must be a dictionary with a 'dist' key")
        if spec["dist"] == "uniform":
            if "low" not in spec or "high" not in spec:
                raise ValueError(f"Uniform prior '{name}' needs 'low' and 'high'")
            if float(spec["low"]) >= float(spec["high"]):
                raise ValueError(f"Uniform prior '{name}' must satisfy low < high")
        elif spec["dist"] == "normal":
            if "mu" not in spec or "sigma" not in spec:
                raise ValueError(f"Normal prior '{name}' needs 'mu' and 'sigma'")
            if float(spec["sigma"]) <= 0.0:
                raise ValueError(f"Normal prior '{name}' must have sigma > 0")
        else:
            raise ValueError(f"Unsupported dist '{spec['dist']}' for '{name}'")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data for a ShapeFit parameter emulator."
    )
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--save-path", type=str, default=get_default_save_path())
    parser.add_argument("--sigma-clip", type=float, default=4.0)
    parser.add_argument("--verbose-every", type=int, default=200)
    parser.add_argument("--omega-m-min", type=float, default=0.05)
    parser.add_argument("--omega-m-max", type=float, default=1.0)
    parser.add_argument(
        "--priors-json",
        type=str,
        default="",
        help=(
            "JSON dictionary of priors, e.g. "
            '\'{"omega_b":{"dist":"normal","mu":0.02218,"sigma":0.00055}}\''
        ),
    )
    args = parser.parse_args()

    priors = DEFAULT_PRIORS if not args.priors_json else parse_priors(args.priors_json)
    save_path = os.path.abspath(args.save_path)
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
            omega_m_bounds=(args.omega_m_min, args.omega_m_max),
        )
        print(f"Generated dataset with shape X={X.shape}, y={y.shape}")
        save_dataset(
            save_path=save_path,
            param_names=param_names,
            X=X,
            y=y,
            test_size=args.test_size,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
