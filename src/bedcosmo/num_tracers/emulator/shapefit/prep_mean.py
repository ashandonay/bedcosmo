import argparse
import json
import sys
import os
import traceback
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumExtractor

from util import (
    latin_hypercube_samples,
    parse_priors,
    save_dataset,
    to_extractor_params,
    get_default_save_path,
)

# Fisher-matrix style priors (normals are truncated at +/-4 sigma by default).
# w0/wa included for base_w_wa model.
DEFAULT_PRIORS = {
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

TARGET_NAMES = ["qiso", "qap", "f_sigmar", "m"]

def run_extractor(
    extractor: ShapeFitPowerSpectrumExtractor,
    sample: Dict[str, float],
    param_defaults: Dict[str, float] | None = None,
) -> Dict[str, float]:
    if param_defaults:
        sample = {**param_defaults, **sample}
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
    param_defaults: Dict[str, float] | None = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    extractor = ShapeFitPowerSpectrumExtractor()
    # Preflight sanity check so we fail fast if extractor setup is broken.
    extractor()
    extractor.get()
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
                targets = run_extractor(extractor, sample, param_defaults=param_defaults)
                target_vals = [targets[t] for t in TARGET_NAMES]
                if not all(np.isfinite(target_vals)):
                    failed += 1
                    continue
                param_rows.append([sample[p] for p in param_names])
                target_rows.append(target_vals)
            except Exception:
                failed += 1
                if not printed_exception:
                    printed_exception = True
                    print("First extractor failure (showing traceback once):")
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
        description="Generate training data for a ShapeFit parameter emulator."
    )
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
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
            '\'{"omega_b":{"dist":"normal","mu":0.02218,"sigma":0.00055}}\''
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

    # Build priors: only include params for the selected cosmo model
    cosmo_model = args.cosmo_model
    model_params = COSMO_MODELS[cosmo_model]
    if args.priors_json:
        priors = parse_priors(args.priors_json)
    else:
        priors = {k: dict(DEFAULT_PRIORS[k]) for k in model_params}

    # Fixed defaults for non-varied cosmo params
    fixed_keys = set(PARAM_DEFAULTS.keys()) - set(model_params)
    param_defaults = {k: PARAM_DEFAULTS[k] for k in fixed_keys}

    save_path = os.path.abspath(args.save_path if args.save_path else get_default_save_path(analysis="shapefit", quantity="mean", cosmo_model=cosmo_model))
    print(f"Cosmo model: {cosmo_model} (varied: {model_params})")
    if param_defaults:
        print(f"Fixed params: {param_defaults}")
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
            param_defaults=param_defaults,
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
