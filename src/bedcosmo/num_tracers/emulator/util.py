import os
from typing import Dict, List

import numpy as np
import json
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split

from scipy.stats import qmc
from scipy.stats import truncnorm
import mlflow
import matplotlib.pyplot as plt

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
    # assumes fiducial values for omega_cdm, omega_b, h, ln10A_s, n_s
    omega_cdm = sample.get("omega_cdm", 0.12069)
    omega_b = sample.get("omega_b", 0.02218)
    h = sample.get("h", 0.6736)
    if h <= 0.0:
        raise ValueError("h must be > 0 to compute Omega_m")
    omega_m = (omega_cdm + omega_b) / (h * h)
    result = {
        "h": float(h),
        "Omega_m": float(omega_m),
        "omega_b": float(omega_b),
        "logA": float(sample.get("ln10A_s", 3.036394)),
        "n_s": float(sample.get("n_s", 0.9649)),
    }
    if "w0" in sample:
        result["w0_fde"] = float(sample["w0"])
    if "wa" in sample:
        result["wa_fde"] = float(sample["wa"])
    return result


def save_dataset(
    save_path: str,
    param_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 1,
    target_names: List[str] | None = None,
    name: str = "",
    version: int | None = None,
) -> str:
    if target_names is None:
        target_names = TARGET_NAMES
    if version is None:
        version = _next_version(save_path)
    versioned_path = os.path.join(save_path, f"v{version}")
    os.makedirs(versioned_path, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    prefix = f"{name}_" if name else ""
    np.savez(
        f"{versioned_path}/{prefix}train.npz",
        x=X_train,
        y=y_train,
        param_names=np.array(param_names),
        target_names=np.array(target_names),
    )
    np.savez(
        f"{versioned_path}/{prefix}test.npz",
        x=X_test,
        y=y_test,
        param_names=np.array(param_names),
        target_names=np.array(target_names),
    )
    print(f"Saved {prefix}train/test files to: {versioned_path} (version {version})")
    return versioned_path

def save_scaled_copy(
    analysis: str,
    quantity: str,
    data_version: str,
    scale_by: str,
    suffix: str | None = None,
) -> str:
    """Save a copy of existing training data with targets scaled by an input parameter.

    Args:
        analysis: Analysis name (e.g. 'shapefit').
        quantity: Quantity name (e.g. 'mean', 'errors').
        data_version: Version string (e.g. '5'). Will be prefixed with 'v'.
        scale_by: Name of the input parameter to multiply targets by (must be in param_names).
        suffix: Suffix for the new version directory. Defaults to '{scale_by}_scaled'.

    Returns:
        Path to the new versioned directory.
    """
    root = get_default_save_path(analysis=analysis, quantity=quantity)
    src = os.path.join(root, f"v{data_version}")
    if suffix is None:
        suffix = f"{scale_by}_scaled"
    dst = os.path.join(root, f"v{data_version}_{suffix}")
    os.makedirs(dst, exist_ok=True)

    for fname in os.listdir(src):
        if not fname.endswith(".npz"):
            continue
        d = np.load(os.path.join(src, fname), allow_pickle=True)
        param_names = d["param_names"].tolist()
        if scale_by not in param_names:
            raise ValueError(f"'{scale_by}' not in param_names: {param_names}")
        idx = param_names.index(scale_by)
        y_scaled = d["y"] * d["x"][:, idx : idx + 1]
        np.savez(
            os.path.join(dst, fname),
            x=d["x"],
            y=y_scaled,
            param_names=d["param_names"],
            target_names=d["target_names"],
        )
        print(f"Saved {fname} to {dst}")

    return dst


def _next_version(save_path: str) -> int:
    """Find the next available version number in save_path/v{N} directories."""
    if not os.path.isdir(save_path):
        return 1
    existing = [
        int(d[1:])
        for d in os.listdir(save_path)
        if d.startswith("v") and d[1:].isdigit()
    ]
    return max(existing, default=0) + 1

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

# DESI DR2 redshift bins: name -> (z_min, z_max, z_eff, ntracers_low, ntracers_high)
TRACER_BINS = {
    "BGS":       (0.1, 0.4, 0.295, 6e5, 1.8e6),
    "LRG1":      (0.4, 0.6, 0.510, 5e5, 1.6e6),
    "LRG2":      (0.6, 0.8, 0.706, 8e5, 2.4e6),
    "LRG3_ELG1": (0.8, 1.1, 0.934, 2.3e6, 6.8e6),
    "ELG2":      (1.1, 1.6, 1.321, 1.9e6, 5.7e6),
    "QSO":       (0.8, 2.1, 1.484, 7e5, 2.2e6),
    "Lya_QSO":   (1.8, 4.2, 2.330, 6.5e5, 1.9e6),
}

TRACER_TYPE_CHOICES = list(TRACER_BINS.keys())


def get_tracer_config(tracer: str) -> Dict:
    """Return tracer bin config as a dict with keys: zrange, z_eff, ntracers_low, ntracers_high."""
    if tracer not in TRACER_BINS:
        raise ValueError(f"Unknown tracer type '{tracer}'. Choose from: {TRACER_TYPE_CHOICES}")
    z_min, z_max, z_eff, nt_low, nt_high = TRACER_BINS[tracer]
    return {
        "zrange": (z_min, z_max),
        "z_eff": z_eff,
        "ntracers_low": nt_low,
        "ntracers_high": nt_high,
    }


def _load_module(name: str, path: str):
    """Load a Python module from an explicit file path (avoids name collisions)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_pipeline(analysis: str, quantity: str, tracer: str | None = None):
    """Return (default_priors, target_names, ground_truth_fn, setup) for the given analysis/quantity.

    If tracer is given, overrides N_tracers prior bounds and passes the
    corresponding zrange/z_eff to the ground truth function.
    """
    _here = os.path.dirname(os.path.abspath(__file__))

    tracer_cfg = get_tracer_config(tracer) if tracer else None

    if analysis == "shapefit":
        if quantity == "covar":
            mod = _load_module("shapefit_prep_covar", os.path.join(_here, "shapefit", "prep_covar.py"))
            kw = {}
            if tracer_cfg:
                kw = {"zrange": tracer_cfg["zrange"], "z_eff": tracer_cfg["z_eff"]}
            def ground_truth_fn(_setup, sample, _kw=kw):
                return mod.run_fisher(sample, **_kw)
            priors = dict(mod.DEFAULT_PRIORS)
            if tracer_cfg:
                priors["N_tracers"] = {"dist": "uniform", "low": tracer_cfg["ntracers_low"], "high": tracer_cfg["ntracers_high"]}
            return priors, mod.TARGET_NAMES, ground_truth_fn, None
        elif quantity == "mean":
            mod = _load_module("shapefit_prep_mean", os.path.join(_here, "shapefit", "prep_mean.py"))
            from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumExtractor
            extractor = ShapeFitPowerSpectrumExtractor()
            extractor()
            extractor.get()
            def ground_truth_fn(setup, sample):
                return mod.run_extractor(setup, sample)
            priors = dict(mod.DEFAULT_PRIORS)
            return priors, mod.TARGET_NAMES, ground_truth_fn, extractor
        else:
            raise ValueError(f"Unknown quantity for shapefit: {quantity}")

    elif analysis == "bao":
        if quantity == "covar":
            mod = _load_module("bao_prep_covar", os.path.join(_here, "bao", "prep_covar.py"))
            kw = {}
            if tracer_cfg:
                kw = {"zrange": tracer_cfg["zrange"], "z_eff": tracer_cfg["z_eff"]}
            def ground_truth_fn(_setup, sample, _kw=kw):
                return mod.run_fisher(sample, **_kw)
            priors = dict(mod.DEFAULT_PRIORS)
            if tracer_cfg:
                priors["N_tracers"] = {"dist": "uniform", "low": tracer_cfg["ntracers_low"], "high": tracer_cfg["ntracers_high"]}
            return priors, mod.TARGET_NAMES, ground_truth_fn, None
        else:
            raise ValueError(f"Unknown quantity for bao: {quantity}")

    else:
        raise ValueError(f"Unknown analysis: {analysis}")


def get_default_save_path(analysis: str = "shapefit", quantity: str = "mean", cosmo_model: str | None = None) -> str:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise EnvironmentError("SCRATCH is not set; please pass --save-path explicitly.")
    parts = [scratch, "bedcosmo", "num_tracers", "emulator", "training_data", analysis]
    if cosmo_model is not None:
        parts.append(cosmo_model)
    parts.append(quantity)
    return os.path.join(*parts)

def compare_losses(
        run_ids: list,
        labels: list | None = None,
        log_scale: bool = True,
        y_lim: tuple | None = None,
        per_step: bool = False,
        ) -> None:
    """Compare train/test loss curves across multiple MLflow runs.

    Args:
        run_ids: List of MLflow run IDs to compare.
        labels: Optional display labels for each run. Defaults to run IDs.
        log_scale: Use log scale for y-axis.
        y_lim: Tuple of (min, max) y-axis limits.
        per_step: If True, plot per-batch losses instead of epoch-averaged.
    """
    from mlflow.tracking import MlflowClient

    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    mlflow.set_tracking_uri(f"file:{scratch}/bedcosmo/shapefit_emulator/mlruns")
    client = MlflowClient()

    if labels is None:
        labels = run_ids

    if per_step:
        train_metric, test_metric = "batch_train_loss", "batch_test_loss"
        x_label = "Step"
    else:
        train_metric, test_metric = "epoch_train_loss", "epoch_test_loss"
        x_label = "Epoch"

    fig, ax = plt.subplots(figsize=(7, 4))

    for run_id, label in zip(run_ids, labels):
        train_hist = client.get_metric_history(run_id, train_metric)
        test_hist = client.get_metric_history(run_id, test_metric)

        # Use same color for train/test of a given run, with different styles
        color = None

        if train_hist:
            steps, vals = zip(
                *[(m.step, m.value) for m in train_hist if np.isfinite(m.value)]
            )
            line = ax.plot(
                steps,
                vals,
                label=f"{label} train",
                alpha=0.8,
            )[0]
            color = line.get_color()

        if test_hist:
            steps, vals = zip(
                *[(m.step, m.value) for m in test_hist if np.isfinite(m.value)]
            )
            ax.plot(
                steps,
                vals,
                label=f"{label} test",
                linestyle="--",
                alpha=0.8,
                color=color,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel("MSE Loss")
    ax.set_title("Train/Test Loss")
    if log_scale:
        ax.set_yscale("log")
    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend()

    fig.tight_layout()
    plt.show()