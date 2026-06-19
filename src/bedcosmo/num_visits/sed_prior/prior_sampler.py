"""GPU-friendly sampling from a masked KDE empirical SED prior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .build_empirical_sed_prior_kde import load_sed_prior_kde, sample_sed_prior


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
