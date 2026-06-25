"""Empirical SED KDE prior: MLflow artifacts, GPU pool, log-density, and sampling."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .fit_sed_prior_kde import load_sed_prior_kde, sample_sed_prior
from .paths import get_prior_kde_path

EMPIRICAL_ARTIFACT_DIR = "empirical"
SED_PRIOR_KDE_FILENAME = "sed_prior_kde.joblib"
SED_PRIOR_Y_KDE_FILENAME = "sed_prior_y_kde.joblib"


def _is_null_path(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
        return True
    return False


def sed_prior_kde_artifact_path(artifacts_dir: str | Path) -> Path:
    """Frozen physical KDE path inside a run's MLflow artifacts."""
    return Path(artifacts_dir) / EMPIRICAL_ARTIFACT_DIR / SED_PRIOR_KDE_FILENAME


def sed_prior_y_kde_artifact_path(artifacts_dir: str | Path) -> Path:
    """Frozen NF-coordinate KDE path inside a run's MLflow artifacts."""
    return Path(artifacts_dir) / EMPIRICAL_ARTIFACT_DIR / SED_PRIOR_Y_KDE_FILENAME


def _prior_kde_source_from_args(prior_args: dict[str, Any] | None) -> Any:
    if not prior_args:
        return None
    if "prior_kde_source" in prior_args:
        return prior_args.get("prior_kde_source")
    return prior_args.get("prior_kde_path")


def resolve_prior_kde_source(
    prior_args: dict[str, Any] | None,
    *,
    cosmo_exp: str = "num_visits",
) -> Path | None:
    """Resolve the scratch/source KDE joblib to copy when snapshotting a run."""
    if cosmo_exp != "num_visits" or not prior_args:
        return None
    raw = _prior_kde_source_from_args(prior_args)
    if _is_null_path(raw):
        return get_prior_kde_path()
    return Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve()


def resolve_runtime_prior_kde_path(
    *,
    empirical_artifacts_dir: str | Path | None = None,
    prior_kde_source: str | Path | None = None,
) -> Path:
    """Resolve the physical KDE joblib for training/eval at runtime.

    Prefers ``artifacts/empirical/sed_prior_kde.joblib`` when present, then an
    explicit source override, then the default scratch build.
    """
    if empirical_artifacts_dir is not None:
        frozen = sed_prior_kde_artifact_path(empirical_artifacts_dir)
        if frozen.is_file():
            return frozen.resolve()
    if not _is_null_path(prior_kde_source):
        path = Path(
            os.path.expandvars(os.path.expanduser(str(prior_kde_source)))
        ).resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"prior KDE not found: {path}")
    return get_prior_kde_path()


def resolve_runtime_y_prior_kde_path(
    *,
    empirical_artifacts_dir: str | Path | None = None,
) -> Path | None:
    """Resolve the frozen y-KDE joblib from run artifacts, if present."""
    if empirical_artifacts_dir is None:
        return None
    path = sed_prior_y_kde_artifact_path(empirical_artifacts_dir)
    if path.is_file():
        return path.resolve()
    return None


def snapshot_sed_prior_kde(
    prior_args: dict[str, Any],
    artifacts_dir: str | Path,
    *,
    cosmo_exp: str = "num_visits",
) -> dict[str, Any]:
    """Copy the physical KDE joblib into ``artifacts/empirical/``."""
    src = resolve_prior_kde_source(prior_args, cosmo_exp=cosmo_exp)
    if src is None:
        return prior_args
    if not src.is_file():
        raise FileNotFoundError(f"prior KDE not found: {src}")

    dest = sed_prior_kde_artifact_path(artifacts_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

    out = dict(prior_args)
    for key in ("prior_kde_path", "prior_y_kde_path"):
        out.pop(key, None)
    return out


def copy_sed_prior_artifacts(
    src_artifacts_dir: str | Path,
    dest_artifacts_dir: str | Path,
) -> bool:
    """Copy ``artifacts/empirical/`` from one MLflow run to another."""
    src = Path(src_artifacts_dir) / EMPIRICAL_ARTIFACT_DIR
    if not src.is_dir():
        return False
    shutil.copytree(
        src,
        Path(dest_artifacts_dir) / EMPIRICAL_ARTIFACT_DIR,
        dirs_exist_ok=True,
    )
    return True


@dataclass
class EmpiricalPriorPool:
    """GPU tensor pool of prior draws plus feature metadata."""

    pool: torch.Tensor
    feature_names: list[str]
    n_templates: int
    bounds_min: torch.Tensor
    bounds_max: torch.Tensor


def load_empirical_prior(path: Path | str) -> dict[str, Any]:
    return load_sed_prior_kde(Path(path).expanduser())


def build_gpu_prior_pool(
    artifact: dict[str, Any],
    pool_size: int,
    *,
    seed: int = 7,
    device: str | torch.device = "cpu",
    chunk_size: int = 10000,
) -> EmpiricalPriorPool:
    """
    Fill a pool by CPU KDE sampling, then upload to ``device`` for O(1) indexed draws.
    """
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    device = torch.device(device)
    feature_names = list(artifact["feature_names"])
    n_templates = int(artifact["n_templates"])
    chunks = []
    remaining = pool_size
    chunk_seed = seed
    while remaining > 0:
        n = min(chunk_size, remaining)
        x = sample_sed_prior(artifact, n, seed=chunk_seed)
        chunks.append(x)
        remaining -= n
        chunk_seed = chunk_seed + 1
    x_all = np.vstack(chunks).astype(np.float64)
    pool = torch.tensor(x_all, device=device, dtype=torch.float64)
    bmin = artifact.get("feature_bounds_min")
    bmax = artifact.get("feature_bounds_max")
    if bmin is None or bmax is None:
        bmin = x_all.min(axis=0)
        bmax = x_all.max(axis=0)
    return EmpiricalPriorPool(
        pool=pool,
        feature_names=feature_names,
        n_templates=n_templates,
        bounds_min=torch.tensor(bmin, device=device, dtype=torch.float64),
        bounds_max=torch.tensor(bmax, device=device, dtype=torch.float64),
    )


def sample_prior_batch(
    prior_pool: EmpiricalPriorPool,
    n: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``n`` rows from the GPU pool; shape ``(n, n_features)``."""
    if n <= 0:
        raise ValueError("n must be positive")
    n_pool = prior_pool.pool.shape[0]
    idx = torch.randint(
        0,
        n_pool,
        (n,),
        device=prior_pool.pool.device,
        generator=generator,
    )
    return prior_pool.pool[idx]


def sample_prior_pool_unique(
    prior_pool: EmpiricalPriorPool,
    n: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw up to ``n`` unique pool rows without replacement."""
    if n <= 0:
        raise ValueError("n must be positive")
    n_pool = int(prior_pool.pool.shape[0])
    if n > n_pool:
        raise ValueError(f"Cannot draw {n} unique rows from pool of size {n_pool}")
    perm = torch.randperm(n_pool, device=prior_pool.pool.device, generator=generator)
    return prior_pool.pool[perm[:n]]


def unpack_prior_rows(
    x: torch.Tensor,
    feature_names: list[str],
    n_templates: int,
    *,
    parameterization: str | None = None,
) -> dict[str, torch.Tensor]:
    """Split pool rows into simplex weights a, log_s, and z."""
    from .simplex import (
        PARAMETERIZATION_CLR,
        PARAMETERIZATION_LOGITS,
        PARAMETERIZATION_WEIGHTS,
        split_feature_matrix,
    )

    if parameterization is None:
        n_f = sum(
            1
            for name in feature_names
            if name.startswith("f") and name[1:].isdigit()
        )
        if n_f == n_templates:
            parameterization = PARAMETERIZATION_CLR
        elif n_f == n_templates - 1:
            parameterization = PARAMETERIZATION_LOGITS
        else:
            parameterization = PARAMETERIZATION_WEIGHTS
    arr = x.detach().cpu().numpy()
    a, log_s, z = split_feature_matrix(arr, n_templates, parameterization=parameterization)
    dev, dtype = x.device, x.dtype
    return {
        "a": torch.as_tensor(a, device=dev, dtype=dtype),
        "log_c_scale": torch.as_tensor(log_s, device=dev, dtype=dtype),
        "z": torch.as_tensor(z, device=dev, dtype=dtype),
    }


def score_kde_artifact(artifact: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Log p(x) under the sklearn KDE in artifact feature space (natural log).

    ``x`` must have shape ``(n, D)`` with columns ordered like
    ``artifact['feature_names']``.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    feature_names = list(artifact["feature_names"])
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} KDE features, got {arr.shape[1]}"
        )
    kde = artifact["kde"]
    scaler = artifact["scaler"]
    x_scaled = scaler.transform(arr)
    return kde.score_samples(x_scaled).astype(np.float64, copy=False)


class EmpiricalSedPrior:
    """Masked KDE empirical SED prior (artifact + GPU sample pool)."""

    def __init__(
        self,
        artifact: dict[str, Any],
        pool: EmpiricalPriorPool,
        *,
        path: str | Path | None = None,
    ) -> None:
        self.artifact = artifact
        self.pool = pool
        self.path = str(path) if path is not None else None
        self.y_artifact: dict[str, Any] | None = None
        self.feature_names = list(artifact["feature_names"])
        self.n_templates = int(artifact["n_templates"])
        self.parameterization = artifact.get("parameterization", "weights")

    @classmethod
    def from_kde_path(
        cls,
        kde_path: str | Path,
        *,
        pool_size: int,
        pool_seed: int = 7,
        device: str | torch.device = "cpu",
    ) -> EmpiricalSedPrior:
        """Load a KDE joblib and build the GPU prior pool."""
        path = Path(os.path.expandvars(os.path.expanduser(str(kde_path)))).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"prior_kde_path not found: {path}")
        artifact = load_empirical_prior(path)
        pool = build_gpu_prior_pool(
            artifact,
            int(pool_size),
            seed=int(pool_seed),
            device=device,
        )
        return cls(artifact, pool, path=path)

    def attach_y_artifact(self, y_artifact: dict[str, Any]) -> None:
        """Attach a KDE fit in NF coordinates (see ``fit_y_prior_kde``)."""
        self.y_artifact = y_artifact

    def has_y_prior_kde(self) -> bool:
        return self.y_artifact is not None

    def log_prob_y(
        self,
        y_samples: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Log prior under the y-space KDE at NF coordinates ``y``."""
        if self.y_artifact is None:
            raise RuntimeError("EmpiricalSedPrior has no y_artifact attached")
        batch_shape = y_samples.shape[:-1]
        flat = y_samples.reshape(-1, y_samples.shape[-1])
        total = flat.shape[0]
        if chunk_size is None:
            chunk_size = total
        chunk_size = max(1, min(int(chunk_size), total))

        log_prob_all = torch.empty(total, device=flat.device, dtype=torch.float64)
        with torch.no_grad():
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                chunk_np = flat[start:end].detach().cpu().numpy()
                lp = score_kde_artifact(self.y_artifact, chunk_np)
                log_prob_all[start:end] = torch.as_tensor(
                    lp, device=flat.device, dtype=torch.float64
                )
        return log_prob_all.reshape(batch_shape)

    def log_prob_nf_from_physical(
        self,
        physical_samples: torch.Tensor,
        transform_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        param_names: Sequence[str] | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Score the y-KDE at ``transform_fn(physical_samples)``."""
        flat = self._align_samples(physical_samples, param_names)
        with torch.no_grad():
            y = transform_fn(flat)
        return self.log_prob_y(
            y.reshape(*physical_samples.shape[:-1], y.shape[-1]),
            chunk_size=chunk_size,
        )

    def _align_samples(
        self,
        samples: torch.Tensor,
        param_names: Sequence[str] | None,
    ) -> torch.Tensor:
        """Reorder ``samples`` columns to ``feature_names`` when needed."""
        flat = samples.reshape(-1, samples.shape[-1])
        if param_names is None:
            if flat.shape[-1] != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} parameters, "
                    f"got {flat.shape[-1]}"
                )
            return flat
        if list(param_names) == self.feature_names:
            return flat
        idx = [param_names.index(n) for n in self.feature_names]
        return flat[:, idx]

    def log_prob(
        self,
        samples: torch.Tensor,
        *,
        param_names: Sequence[str] | None = None,
        chunk_size: int | None = None,
        nf_space: bool = False,
        log_abs_det_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Joint log prior under the empirical KDE (natural log).

        Args:
            samples: Physical parameter rows, shape ``(..., n_params)``.
            param_names: Names for each column of ``samples`` (defaults to
                ``feature_names`` order).
            chunk_size: Chunk size for sklearn ``score_samples`` calls.
            nf_space: If True, return ``log p_KDE(theta) - log|det dT/dtheta|`` using
                ``log_abs_det_fn`` (same convention as the experiment bijector).
            log_abs_det_fn: Callable on flattened physical samples returning
                per-row ``log|det dT/dtheta|``; required when ``nf_space=True``.
        """
        batch_shape = samples.shape[:-1]
        flat = self._align_samples(samples, param_names)

        total = flat.shape[0]
        if chunk_size is None:
            chunk_size = total
        chunk_size = max(1, min(int(chunk_size), total))

        log_prob_all = torch.empty(total, device=flat.device, dtype=torch.float64)
        with torch.no_grad():
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                chunk_np = flat[start:end].detach().cpu().numpy()
                lp = score_kde_artifact(self.artifact, chunk_np)
                log_prob_all[start:end] = torch.as_tensor(
                    lp, device=flat.device, dtype=torch.float64
                )

        if nf_space:
            if log_abs_det_fn is None:
                raise RuntimeError(
                    "EmpiricalSedPrior.log_prob(nf_space=True) requires log_abs_det_fn."
                )
            log_det = log_abs_det_fn(samples.reshape(-1, samples.shape[-1]))
            log_prob_all = log_prob_all - log_det.reshape(log_prob_all.shape)

        return log_prob_all.reshape(batch_shape)

    def sample_batch(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw ``n`` rows from the GPU pool; shape ``(n, n_features)``."""
        return sample_prior_batch(self.pool, n, generator=generator)

    def sample_unique(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw up to ``n`` unique pool rows without replacement."""
        return sample_prior_pool_unique(self.pool, n, generator=generator)

    def rows_to_param_dict(
        self,
        rows: torch.Tensor,
        param_names: Sequence[str],
        sample_shape: tuple[int, ...],
    ) -> dict[str, torch.Tensor]:
        """Map feature rows to named parameter tensors with trailing dim 1."""
        col = {n: i for i, n in enumerate(self.feature_names)}
        out: dict[str, torch.Tensor] = {}
        flat = sample_shape if sample_shape else (rows.shape[0],)
        for name in param_names:
            if name not in col:
                raise KeyError(
                    f"Unknown empirical prior parameter '{name}'. "
                    f"Available: {self.feature_names}"
                )
            out[name] = rows[:, col[name]].reshape(*flat, 1)
        return out
