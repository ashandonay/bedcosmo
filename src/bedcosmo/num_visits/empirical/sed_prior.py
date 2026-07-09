"""Empirical SED prior: MLflow artifacts, GPU pool, log-density, and sampling."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .fit_sed_prior_kde import load_sed_prior_kde, sample_sed_prior
from .paths import (
    SED_PRIOR_KDE_GAUSSIANIZED_FILENAME,
    SED_PRIOR_KDE_NATIVE_FILENAME,
    get_prior_build_dir,
)

EMPIRICAL_ARTIFACT_DIR = "empirical"
SED_PRIOR_KDE_FILENAMES = {
    "native": SED_PRIOR_KDE_NATIVE_FILENAME,
    "gaussianized": SED_PRIOR_KDE_GAUSSIANIZED_FILENAME,
}

PRIOR_SOURCE_KDE = "kde"
PRIOR_SOURCE_FLOW = "flow"
VALID_PRIOR_SOURCES = (PRIOR_SOURCE_KDE, PRIOR_SOURCE_FLOW)


def normalize_prior_source(value: Any) -> str:
    """Validate ``prior_source`` to exactly {'kde','flow'}; raise on anything else.

    An absent / null / empty value defaults to ``'kde'``; any other unrecognized
    value (e.g. a typo) is a loud ``ValueError`` rather than a silent KDE fallback.
    """
    if value is None:
        return PRIOR_SOURCE_KDE
    s = str(value).strip().lower()
    if s in ("", "null", "none"):
        return PRIOR_SOURCE_KDE
    if s not in VALID_PRIOR_SOURCES:
        raise ValueError(
            f"prior_source must be one of {VALID_PRIOR_SOURCES} (or null for the "
            f"default 'kde'); got {value!r}."
        )
    return s


def _is_null_path(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
        return True
    return False


def sed_prior_kde_artifact_path(
    artifacts_dir: str | Path,
    space: str = "native",
) -> Path:
    """Frozen KDE path inside a run's MLflow artifacts (``native`` or ``gaussianized``)."""
    try:
        filename = SED_PRIOR_KDE_FILENAMES[space]
    except KeyError as exc:
        raise ValueError(
            f"space must be one of {tuple(SED_PRIOR_KDE_FILENAMES)}; got {space!r}"
        ) from exc
    return Path(artifacts_dir) / EMPIRICAL_ARTIFACT_DIR / filename


def sed_prior_flow_artifact_path(artifacts_dir: str | Path, space: str) -> Path:
    """Frozen prior-flow ``.pt`` path (``space`` in native/gaussianized) in run artifacts."""
    from .prior_flow import SED_PRIOR_FLOW_FILENAMES

    return Path(artifacts_dir) / EMPIRICAL_ARTIFACT_DIR / SED_PRIOR_FLOW_FILENAMES[space]


def resolve_prior_dir(
    prior_args: dict[str, Any] | None = None,
    *,
    prior_dir: str | Path | None = None,
) -> Path:
    """Resolve the empirical prior build directory.

    Contains ``sed_prior_kde_native.joblib`` and, when ``prior_source=flow``, the
    ``sed_prior_flow_*.pt`` files. ``null`` / omitted → default scratch build
    (``$SCRATCH/bedcosmo/num_visits/empirical_prior``).
    """
    raw = prior_dir
    if _is_null_path(raw) and prior_args:
        raw = prior_args.get("prior_dir")
    if _is_null_path(raw):
        return get_prior_build_dir()
    return Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve()


def resolve_runtime_prior_root(
    *,
    artifacts_dir: str | Path | None = None,
    prior_dir: str | Path | None = None,
) -> Path:
    """Directory that holds the empirical prior files for this run.

    Prefers the frozen ``artifacts/empirical/`` tree when present (contains the
    native KDE and, for ``prior_source=flow``, the flow ``.pt`` files). Otherwise
    uses ``prior_dir`` or the default scratch build.
    """
    if artifacts_dir is not None:
        frozen_root = Path(artifacts_dir) / EMPIRICAL_ARTIFACT_DIR
        if (frozen_root / SED_PRIOR_KDE_NATIVE_FILENAME).is_file():
            return frozen_root.resolve()
    return resolve_prior_dir(prior_dir=prior_dir)


def snapshot_sed_prior(
    prior_args: dict[str, Any],
    artifacts_dir: str | Path,
    *,
    cosmo_exp: str = "num_visits",
) -> dict[str, Any]:
    """Freeze the empirical prior into ``artifacts/empirical/``.

    Always copies ``sed_prior_kde_native.joblib`` from ``prior_dir``. When
    ``prior_source == 'flow'`` also copies ``sed_prior_flow_*.pt`` from the same
    directory.
    """
    if cosmo_exp != "num_visits" or not prior_args:
        return prior_args
    src = resolve_prior_dir(prior_args) / SED_PRIOR_KDE_NATIVE_FILENAME
    if not src.is_file():
        raise FileNotFoundError(f"prior KDE not found: {src}")

    dest = sed_prior_kde_artifact_path(artifacts_dir, space="native")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

    # NOTE: sed_prior_kde_gaussianized.joblib is intentionally NOT snapshotted.
    # Runtime empirical entropy uses the trained native/gaussianized PriorFlows
    # (or the N(0,I) shortcut when prior_source=kde); the offline gaussianized
    # KDE is diagnostic only.

    out = dict(prior_args)

    # prior_source=flow: freeze the trained flow(s) from the same prior_dir.
    if normalize_prior_source(out.get("prior_source")) == PRIOR_SOURCE_FLOW:
        _snapshot_sed_prior_flows(artifacts_dir, src_dir=src.parent)

    return out


def _snapshot_sed_prior_flows(
    artifacts_dir: str | Path,
    *,
    src_dir: Path,
) -> None:
    """Copy the native (+ gaussianized) prior-flow ``.pt`` into ``artifacts/empirical/``."""
    from .prior_flow import SED_PRIOR_FLOW_FILENAMES, SPACE_NATIVE

    native_src = src_dir / SED_PRIOR_FLOW_FILENAMES[SPACE_NATIVE]
    if not native_src.is_file():
        raise FileNotFoundError(
            f"prior_source='flow' but native flow not found: {native_src}. Train it with "
            "python -m bedcosmo.num_visits.empirical.prior_flow --space both."
        )
    dest_dir = Path(artifacts_dir) / EMPIRICAL_ARTIFACT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in SED_PRIOR_FLOW_FILENAMES.values():
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dest_dir / name)


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
        raise ValueError(f"Expected {len(feature_names)} KDE features, got {arr.shape[1]}")
    kde = artifact["kde"]
    scaler = artifact["scaler"]
    x_scaled = scaler.transform(arr)
    return kde.score_samples(x_scaled).astype(np.float64, copy=False)


class EmpiricalSedPrior:
    """Empirical SED prior: KDE artifact + GPU sample pool, optional PriorFlows."""

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
        # "kde" draws the pool from the frozen KDE; "flow" rebuilds it from an
        # attached native PriorFlow (see build_pool_from_flow) and scores entropy
        # with the same flow (native) / gaussianized_flow (transform_input).
        self.prior_source = "kde"
        # Trained PriorFlows (see prior_flow.PriorFlow): native_flow models the
        # native/ILR density; gaussianized_flow models the NF-coordinate density.
        self.native_flow: Any | None = None
        self.gaussianized_flow: Any | None = None
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
            raise FileNotFoundError(f"KDE artifact not found: {path}")
        artifact = load_sed_prior_kde(path)
        pool = build_gpu_prior_pool(
            artifact,
            int(pool_size),
            seed=int(pool_seed),
            device=device,
        )
        return cls(artifact, pool, path=path)

    def attach_flow(self, flow: Any) -> None:
        """Attach a trained ``PriorFlow``; dispatched on its ``space`` attribute.

        A ``native`` flow becomes the alternative estimator for :meth:`log_prob`
        (``use_flow=True``); a ``gaussianized`` flow for
        :meth:`log_prob_gaussianized_from_native`.
        """
        from .prior_flow import SPACE_GAUSSIANIZED, SPACE_NATIVE

        space = getattr(flow, "space", None)
        if space == SPACE_NATIVE:
            self.native_flow = flow
        elif space == SPACE_GAUSSIANIZED:
            self.gaussianized_flow = flow
        else:
            raise ValueError(f"PriorFlow has unknown space {space!r}")

    def attach_flow_from_path(self, path: str | Path) -> None:
        """Load a ``PriorFlow`` from ``path`` and attach it (see :meth:`attach_flow`)."""
        from .prior_flow import PriorFlow

        self.attach_flow(PriorFlow.load(path))

    def has_native_flow(self) -> bool:
        return self.native_flow is not None

    def has_gaussianized_flow(self) -> bool:
        return self.gaussianized_flow is not None

    def uses_flow_prior(self) -> bool:
        """True when the native flow (not the KDE) is the active prior."""
        return self.prior_source == "flow" and self.native_flow is not None

    def enable_flow_prior(
        self,
        pool_size: int,
        *,
        seed: int = 7,
        device: str | torch.device | None = None,
        flow_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """Load PriorFlows from ``flow_dir`` (default: beside the KDE) and rebuild the pool.

        Requires ``sed_prior_flow_native.pt``. Attaches ``sed_prior_flow_gaussianized.pt``
        when present (needed for ``transform_input=True`` entropy). Returns the paths
        that were loaded.
        """
        from .prior_flow import SED_PRIOR_FLOW_FILENAMES, SPACE_GAUSSIANIZED, SPACE_NATIVE

        if self.path is None and flow_dir is None:
            raise RuntimeError("enable_flow_prior needs flow_dir or a KDE path on the prior")
        root = Path(flow_dir).expanduser() if flow_dir is not None else Path(self.path).parent
        native_path = root / SED_PRIOR_FLOW_FILENAMES[SPACE_NATIVE]
        if not native_path.is_file():
            raise FileNotFoundError(
                f"prior_source='flow' requires a native flow at {native_path}. Train it with "
                "python -m bedcosmo.num_visits.empirical.prior_flow --space both."
            )
        self.attach_flow_from_path(native_path)
        if self.native_flow.dim != len(self.feature_names):
            raise ValueError(
                f"native flow dim {self.native_flow.dim} != "
                f"{len(self.feature_names)} KDE features; retrain the flow on this artifact."
            )
        loaded = {SPACE_NATIVE: native_path}
        gauss_path = root / SED_PRIOR_FLOW_FILENAMES[SPACE_GAUSSIANIZED]
        if gauss_path.is_file():
            self.attach_flow_from_path(gauss_path)
            loaded[SPACE_GAUSSIANIZED] = gauss_path
        self.build_pool_from_flow(int(pool_size), seed=int(seed), device=device)
        return loaded

    def build_pool_from_flow(
        self,
        pool_size: int,
        *,
        seed: int = 7,
        device: str | torch.device | None = None,
    ) -> None:
        """Replace the GPU pool with draws from the attached native flow.

        Mirrors :func:`build_gpu_prior_pool` but samples the native ``PriorFlow``
        instead of the KDE, and flips ``prior_source`` to ``"flow"`` so entropy is
        scored against the same flow that generated the pool.
        """
        if self.native_flow is None:
            raise RuntimeError(
                "build_pool_from_flow requires a native flow; call attach_flow() first."
            )
        if int(pool_size) <= 0:
            raise ValueError("pool_size must be positive")
        dev = torch.device(device) if device is not None else self.pool.pool.device
        x = self.native_flow.sample(int(pool_size), seed=int(seed)).astype(np.float64)
        pool = torch.tensor(x, device=dev, dtype=torch.float64)
        bmin = self.artifact.get("feature_bounds_min")
        bmax = self.artifact.get("feature_bounds_max")
        if bmin is None or bmax is None:
            bmin = x.min(axis=0)
            bmax = x.max(axis=0)
        self.pool = EmpiricalPriorPool(
            pool=pool,
            feature_names=list(self.pool.feature_names),
            n_templates=int(self.pool.n_templates),
            bounds_min=torch.tensor(bmin, device=dev, dtype=torch.float64),
            bounds_max=torch.tensor(bmax, device=dev, dtype=torch.float64),
        )
        self.prior_source = "flow"

    def log_prob_gaussianized_from_native(
        self,
        native_samples: torch.Tensor,
        transform_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        param_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Score the gaussianized prior at ``y = transform_fn(native_samples)``.

        ``transform_fn`` is typically ``params_to_unconstrained`` (the same
        gaussianizer used by the NF guide). ``log p(y)`` comes from
        ``gaussianized_flow`` directly (no Jacobian).
        """
        if self.gaussianized_flow is None:
            raise RuntimeError(
                "log_prob_gaussianized_from_native requires a gaussianized flow; "
                "call attach_flow() with space='gaussianized'."
            )
        flat = self._align_samples(native_samples, param_names)
        with torch.no_grad():
            y = transform_fn(flat)
        batch_shape = native_samples.shape[:-1]
        y_flat = y.reshape(-1, y.shape[-1])
        lp = self.gaussianized_flow.log_prob(y_flat.detach().cpu().numpy())
        return torch.as_tensor(lp, device=flat.device, dtype=torch.float64).reshape(batch_shape)

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
                    f"Expected {len(self.feature_names)} parameters, " f"got {flat.shape[-1]}"
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
        use_flow: bool = False,
    ) -> torch.Tensor:
        """Joint log prior in physical space under the empirical KDE or flow (natural log).

        Args:
            samples: Physical parameter rows, shape ``(..., n_params)``.
            param_names: Names for each column of ``samples`` (defaults to
                ``feature_names`` order).
            chunk_size: Chunk size for sklearn ``score_samples`` calls.
            use_flow: If True, score with the attached ``native_flow``
                (:meth:`attach_flow`) instead of the KDE plug-in.
        """
        batch_shape = samples.shape[:-1]
        flat = self._align_samples(samples, param_names)

        total = flat.shape[0]
        if chunk_size is None:
            chunk_size = total
        chunk_size = max(1, min(int(chunk_size), total))

        log_prob_all = torch.empty(total, device=flat.device, dtype=torch.float64)
        if use_flow:
            if self.native_flow is None:
                raise RuntimeError(
                    "log_prob(use_flow=True) requires a native flow; call attach_flow()."
                )
            lp = self.native_flow.log_prob(flat.detach().cpu().numpy())
            log_prob_all = torch.as_tensor(lp, device=flat.device, dtype=torch.float64)
        else:
            with torch.no_grad():
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    chunk_np = flat[start:end].detach().cpu().numpy()
                    lp = score_kde_artifact(self.artifact, chunk_np)
                    log_prob_all[start:end] = torch.as_tensor(
                        lp, device=flat.device, dtype=torch.float64
                    )

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
