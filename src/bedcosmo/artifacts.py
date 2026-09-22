"""MLflow run artifact helpers: designs, eig_data JSON, posterior sample npz."""

from __future__ import annotations

import glob
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Design args snapshot
# ---------------------------------------------------------------------------


def resolve_design_args_input_path(
    design_args: dict | None,
    config_path: str | Path | None = None,
) -> dict | None:
    """Resolve ``input_designs_path`` from a design-arguments document.

    Environment variables and ``~`` are expanded. Relative paths are anchored
    to the directory containing the YAML file, or to the current directory when
    the arguments were supplied directly as a dictionary.
    """
    if design_args is None:
        return None
    resolved = dict(design_args)
    raw = resolved.get("input_designs_path")
    if raw in (None, ""):
        return resolved
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(raw)))
    if "$" in expanded:
        raise ValueError(
            f"input_designs_path contains an undefined environment variable: {raw}"
        )
    path = Path(expanded)
    if not path.is_absolute():
        base = Path(config_path).expanduser().resolve().parent if config_path else Path.cwd()
        path = base / path
    resolved["input_designs_path"] = str(path.resolve())
    return resolved


def snapshot_design_args_config(
    source_path: str | Path,
    destination_path: str | Path,
) -> dict:
    """Freeze a design YAML and its referenced array into an artifact directory."""
    source_path = Path(source_path).expanduser().resolve()
    destination_path = Path(destination_path).expanduser().resolve()
    with source_path.open() as stream:
        design_args = yaml.safe_load(stream) or {}
    design_args = resolve_design_args_input_path(design_args, source_path)

    input_path = design_args.get("input_designs_path")
    if input_path is not None:
        input_path = Path(input_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"input_designs_path not found: {input_path}")
        frozen_path = destination_path.parent / "designs.npy"
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if input_path.resolve() != frozen_path.resolve():
            shutil.copy2(input_path, frozen_path)
        design_args["input_designs_path"] = str(frozen_path.resolve())

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w") as stream:
        yaml.safe_dump(design_args, stream, default_flow_style=False, sort_keys=False)
    return design_args


# ---------------------------------------------------------------------------
# eig_data JSON discovery
# ---------------------------------------------------------------------------


def _step_keys_for_eval(data, eval_step=None):
    """Return step_* keys to inspect, optionally restricted to eval_step."""
    if eval_step is not None:
        step_str = f"step_{eval_step}" if not str(eval_step).startswith("step_") else str(eval_step)
        return [step_str] if step_str in data else []
    return [k for k in data.keys() if k.startswith("step_")]


def _eig_data_has_variable_eigs(data, eval_step=None):
    """True when eig_data contains joint (variable) EIG averages."""
    for step_key in _step_keys_for_eval(data, eval_step):
        variable = data.get(step_key, {}).get("variable", {})
        if variable.get("eigs_avg") is not None:
            return True
    return False


def _eig_data_has_marginal_eigs(data, eval_step=None):
    """True when eig_data contains marginal EIG blocks."""
    for step_key in _step_keys_for_eval(data, eval_step):
        marginal = data.get(step_key, {}).get("marginal", {})
        if marginal:
            return True
    return False


def load_eig_data_file(artifacts_dir, eval_step=None, eig_kind="any"):
    """
    Load the most recent completed eig_data JSON file from the artifacts directory.

    Args:
        artifacts_dir (str): Path to the artifacts directory containing eig_data files
        eval_step (str or int, optional): If provided, verify that the loaded file contains this step
        eig_kind (str): Which EIG content the file must contain: ``'any'`` (default),
            ``'variable'`` (joint EIG under ``step_*/variable``), or ``'marginal'``
            (marginal EIG under ``step_*/marginal``). Use ``'variable'`` when comparing
            joint EIGs so a newer marginal-only eval file is skipped.

    Returns:
        tuple: (json_path, data) where json_path is the path to the file and data is the loaded JSON.

    Raises:
        ValueError: If no completed eig_data files are found, if file cannot be loaded,
            or if eval_step is not found in the data.
    """
    if eig_kind not in ("any", "variable", "marginal"):
        raise ValueError(f"eig_kind must be 'any', 'variable', or 'marginal', got {eig_kind!r}")

    if not os.path.exists(artifacts_dir):
        raise ValueError(f"Artifacts directory not found: {artifacts_dir}")

    eig_files = glob.glob(f"{artifacts_dir}/eig_data_*.json")

    if len(eig_files) == 0:
        raise ValueError(f"No eig_data JSON files found in {artifacts_dir}")

    # Sort by filename (most recent first)
    eig_files.sort(key=lambda x: os.path.basename(x), reverse=True)

    for json_path in eig_files:
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            status = data.get("status")
            if status != "complete":
                continue

            if eval_step is not None:
                step_str = (
                    f"step_{eval_step}"
                    if not str(eval_step).startswith("step_")
                    else str(eval_step)
                )
                if step_str not in data:
                    continue

            if eig_kind == "variable" and not _eig_data_has_variable_eigs(data, eval_step):
                continue
            if eig_kind == "marginal" and not _eig_data_has_marginal_eigs(data, eval_step):
                continue

            return json_path, data

        except Exception as e:
            print(f"Warning: Error loading {json_path}: {e}, skipping...")
            continue

    kind_suffix = f" with {eig_kind} EIG data" if eig_kind != "any" else ""
    if eval_step is not None:
        raise ValueError(
            f"No completed eig_data files with step {eval_step}{kind_suffix} found in {artifacts_dir}"
        )
    raise ValueError(f"No completed eig_data files{kind_suffix} found in {artifacts_dir}")


# ---------------------------------------------------------------------------
# Posterior sample npz bundles
# ---------------------------------------------------------------------------

POSTERIOR_SAMPLES_SCHEMA_VERSION = 1
POSTERIOR_SAMPLES_SUBDIR = "posterior_samples"
# Backward-compatible aliases
SCHEMA_VERSION = POSTERIOR_SAMPLES_SCHEMA_VERSION
DEFAULT_SUBDIR = POSTERIOR_SAMPLES_SUBDIR


def _meta_to_array(meta: dict) -> np.ndarray:
    """Encode meta dict as a 0-d unicode array (npz-safe, no pickle)."""
    return np.asarray(json.dumps(meta, sort_keys=True), dtype=np.str_)


def _meta_from_array(raw) -> dict:
    if isinstance(raw, np.ndarray):
        raw = raw.item() if raw.ndim == 0 else raw.tolist()
    return json.loads(str(raw))


def observations_to_numpy(data_samples, num_data_samples: int) -> np.ndarray:
    """Flatten observation tensor/array to shape ``(n_data, n_obs)``."""
    if hasattr(data_samples, "detach"):
        arr = data_samples.detach().cpu().numpy()
    else:
        arr = np.asarray(data_samples)
    return np.reshape(arr, (int(num_data_samples), -1))


def posterior_samples_dir(artifacts_dir: str, subdir: str = DEFAULT_SUBDIR) -> str:
    return os.path.join(artifacts_dir, subdir)


def make_posterior_samples_path(
    artifacts_dir: str,
    step: Optional[int] = None,
    timestamp: Optional[str] = None,
    subdir: str = DEFAULT_SUBDIR,
) -> str:
    """Build a timestamped npz path under ``artifacts/posterior_samples/``."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = posterior_samples_dir(artifacts_dir, subdir=subdir)
    os.makedirs(out_dir, exist_ok=True)
    if step is not None:
        return os.path.join(out_dir, f"posterior_step{int(step)}_{timestamp}.npz")
    return os.path.join(out_dir, f"posterior_{timestamp}.npz")


def save_posterior_samples(
    path: str,
    *,
    theta: np.ndarray,
    y: np.ndarray,
    design: np.ndarray,
    series_names: list[str] | np.ndarray,
    param_names: list[str] | np.ndarray,
    meta: dict[str, Any],
    compress: bool = True,
) -> str:
    """
    Write one posterior-sample bundle (arrays + JSON meta) to ``path``.

    Array contract:
      theta:  (n_series, n_data, n_guide, n_params)
      y:      (n_series, n_data, n_obs)
      design: (n_series, n_design_dims)
    """
    theta = np.asarray(theta)
    y = np.asarray(y)
    design = np.asarray(design)
    series_names = np.asarray(list(series_names), dtype=np.str_)
    param_names = np.asarray(list(param_names), dtype=np.str_)

    if theta.ndim != 4:
        raise ValueError(f"theta must be 4-D (n_series, n_data, n_guide, n_params), got {theta.shape}")
    if y.ndim != 3:
        raise ValueError(f"y must be 3-D (n_series, n_data, n_obs), got {y.shape}")
    if design.ndim != 2:
        raise ValueError(f"design must be 2-D (n_series, n_design_dims), got {design.shape}")
    if theta.shape[0] != y.shape[0] or theta.shape[0] != design.shape[0]:
        raise ValueError(
            f"n_series mismatch: theta={theta.shape[0]}, y={y.shape[0]}, design={design.shape[0]}"
        )
    if theta.shape[1] != y.shape[1]:
        raise ValueError(f"n_data mismatch: theta={theta.shape[1]}, y={y.shape[1]}")
    if theta.shape[0] != series_names.shape[0]:
        raise ValueError("series_names length must match n_series")
    if theta.shape[-1] != param_names.shape[0]:
        raise ValueError("param_names length must match n_params")

    meta = dict(meta)
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta.setdefault("status", "complete")
    meta["param_names"] = param_names.tolist()
    meta["series_names"] = series_names.tolist()
    meta["shapes"] = {
        "theta": list(theta.shape),
        "y": list(y.shape),
        "design": list(design.shape),
    }

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if not path.endswith(".npz"):
        path = path + ".npz"

    saver = np.savez_compressed if compress else np.savez
    tmp_path = path + ".writing.npz"
    saver(
        tmp_path,
        theta=theta,
        y=y,
        design=design,
        series_names=series_names,
        param_names=param_names,
        meta=_meta_to_array(meta),
    )
    os.replace(tmp_path, path)
    return path


def read_posterior_samples_meta(path: str) -> dict:
    """Load only the meta JSON from an npz (lazy; does not load arrays)."""
    with np.load(path, allow_pickle=False) as z:
        if "meta" not in z.files:
            raise ValueError(f"No 'meta' key in {path}")
        return _meta_from_array(z["meta"])


def load_posterior_samples(path: str) -> dict[str, Any]:
    """
    Load a posterior-sample bundle.

    Returns dict with keys: path, meta, theta, y, design, series_names, param_names.
    """
    with np.load(path, allow_pickle=False) as z:
        required = ("theta", "y", "design", "series_names", "param_names", "meta")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f"Missing keys {missing} in {path}")
        return {
            "path": path,
            "meta": _meta_from_array(z["meta"]),
            "theta": np.asarray(z["theta"]),
            "y": np.asarray(z["y"]),
            "design": np.asarray(z["design"]),
            "series_names": np.asarray(z["series_names"]).astype(str),
            "param_names": np.asarray(z["param_names"]).astype(str),
        }


def _to_1d_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def save_central_posterior_from_nf_entries(
    artifacts_dir: str,
    nf_entries: list[dict[str, Any]],
    *,
    experiment: Any,
    step,
    meta: Optional[dict[str, Any]] = None,
    eig_values=None,
) -> Optional[str]:
    """
    Persist already-generated central-context NF display entries as an NPZ bundle.

    Does **not** sample. Expects each entry from ``BasePlotter._nf_display_samples``
    to include ``name``, ``color``, ``design``, and ``samples`` (GetDist MCSamples).
    Observation ``y`` is ``experiment.central_val`` with ``n_data=1``.

    Series are written in ``optimal``, ``nominal`` order when both exist (matches
    ``Evaluator.sample_posterior``). Returns the written path, or None if nothing
    to save.
    """
    by_name = {
        entry["name"]: entry
        for entry in nf_entries
        if isinstance(entry, dict) and entry.get("name") in ("optimal", "nominal")
    }
    series_order = [n for n in ("optimal", "nominal") if n in by_name]
    if not series_order:
        return None

    central = _to_1d_numpy(experiment.central_val)
    thetas = []
    ys = []
    designs = []
    series_meta = []
    param_names = None

    for name in series_order:
        entry = by_name[name]
        samples_gd = entry["samples"]
        theta = np.asarray(samples_gd.samples, dtype=np.float64)
        if theta.ndim != 2:
            raise ValueError(f"Expected (n_guide, n_params) samples, got {theta.shape}")
        if param_names is None:
            param_names = list(samples_gd.paramNames.list())
        thetas.append(theta[np.newaxis, ...])  # (1, n_guide, n_params)
        ys.append(central.reshape(1, -1))
        designs.append(_to_1d_numpy(entry["design"]))
        series_meta.append({"name": name, "color": entry.get("color")})

    theta_all = np.stack(thetas, axis=0)
    y_all = np.stack(ys, axis=0)
    design_all = np.stack(designs, axis=0)

    optimal_idx = None
    if eig_values is not None:
        eigs_1d = np.atleast_1d(np.asarray(eig_values))
        if eigs_1d.size > 0:
            optimal_idx = int(np.argmax(eigs_1d))

    out_meta: dict[str, Any] = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "step": int(step) if str(step).isdigit() else step,
        "central": True,
        "conditioning": "central_val",
        "num_data_samples": 1,
        "series": series_meta,
        "generated_by": "generate_posterior",
    }
    if optimal_idx is not None:
        out_meta["optimal_design_index"] = optimal_idx
    if meta:
        out_meta.update(meta)
        # Keep schema fields authoritative for this central-context path.
        out_meta["central"] = True
        out_meta["conditioning"] = "central_val"
        out_meta["num_data_samples"] = 1
        out_meta.setdefault("series", series_meta)

    out_path = make_posterior_samples_path(artifacts_dir, step=step)
    save_posterior_samples(
        out_path,
        theta=theta_all,
        y=y_all,
        design=design_all,
        series_names=series_order,
        param_names=param_names,
        meta=out_meta,
    )
    return out_path


def _step_matches(meta: dict, step) -> bool:
    if step is None:
        return True
    meta_step = meta.get("step")
    if meta_step is None:
        return False
    try:
        return int(meta_step) == int(step)
    except (TypeError, ValueError):
        return str(meta_step) == str(step)


def load_posterior_samples_file(
    artifacts_dir: str,
    step=None,
    *,
    status: str = "complete",
    subdir: str = DEFAULT_SUBDIR,
    path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Load the newest matching posterior-sample bundle under an artifacts dir.

    If ``path`` is given, load that file directly (still validates ``status`` when set).
    Otherwise scan ``artifacts_dir/subdir/posterior*.npz``, prefer files whose meta
    matches ``step`` and ``status``, newest by mtime.
    """
    if path is not None:
        bundle = load_posterior_samples(path)
        if status is not None and bundle["meta"].get("status") != status:
            raise ValueError(
                f"Posterior samples at {path} have status={bundle['meta'].get('status')!r}, "
                f"expected {status!r}"
            )
        if not _step_matches(bundle["meta"], step):
            raise ValueError(
                f"Posterior samples at {path} have step={bundle['meta'].get('step')!r}, "
                f"expected {step!r}"
            )
        return bundle

    search_dir = posterior_samples_dir(artifacts_dir, subdir=subdir)
    patterns = [
        os.path.join(search_dir, "posterior_step*_*.npz"),
        os.path.join(search_dir, "posterior_*.npz"),
        os.path.join(artifacts_dir, "posterior_*.npz"),
    ]
    candidates: list[str] = []
    seen = set()
    for pattern in patterns:
        for p in glob.glob(pattern):
            if p not in seen:
                seen.add(p)
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No posterior_*.npz files found under {search_dir} (or {artifacts_dir})"
        )

    matches: list[tuple[float, str, dict]] = []
    for p in candidates:
        try:
            meta = read_posterior_samples_meta(p)
        except Exception:
            continue
        if status is not None and meta.get("status") != status:
            continue
        if not _step_matches(meta, step):
            continue
        matches.append((os.path.getmtime(p), p, meta))

    if not matches:
        step_msg = f" for step={step}" if step is not None else ""
        raise FileNotFoundError(
            f"No {status!r} posterior sample files{step_msg} under {search_dir}"
        )

    matches.sort(key=lambda t: t[0], reverse=True)
    return load_posterior_samples(matches[0][1])
