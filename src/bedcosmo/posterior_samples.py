"""Save / load posterior sample bundles (theta + design/y provenance) as one npz."""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from typing import Any, Optional

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_SUBDIR = "posterior_samples"


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
    # Write via temp then rename so readers never see a partial file.
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
