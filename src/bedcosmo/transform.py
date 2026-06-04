"""
Torch prior bijector for bedcosmo experiments and empirical priors.

``Bijector`` maps physical parameters to approximately Gaussian NF coordinates:

1. Per-dimension empirical CDF + normal scores (``input_transform_type="marginal"``).
2. Optional joint Cholesky whitening on a parameter block (``input_transform_type="joint"``).

Fit paths:

- **Experiment:** sample from ``experiment.sample_parameters`` (Pyro prior / prior flow).
- **Matrix:** ``Bijector.fit_from_matrix`` on a fixed ``(N, D)`` reference batch (SED KDE build).

State is a plain dict (``get_state`` / ``set_state``) for checkpoints and joblib artifacts.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pyro
import torch
from pyro import distributions as dist

try:
    from bedcosmo.custom_dist import EmpiricalPrior
except Exception:  # pragma: no cover - keeps standalone notebook imports usable
    EmpiricalPrior = ()  # type: ignore[assignment]


def _normalize_input_transform_type(value: str) -> str:
    if value not in ("marginal", "joint"):
        raise ValueError(
            f"input_transform_type must be 'marginal' or 'joint', got {value!r}"
        )
    return value


def _whitening_to_apply_joint(whitening: str) -> bool:
    if whitening == "none":
        return False
    if whitening == "cholesky":
        return True
    raise ValueError(f"Unknown whitening mode {whitening!r}; expected 'none' or 'cholesky'")


class Bijector:
    """
    Empirical-CDF Gaussianizer with optional joint whitening.

    Used by ``BaseExperiment`` (per-parameter keys) and by the SED KDE pipeline
    (fixed feature matrix columns via ``matrix_columns``).
    """

    def __init__(
        self,
        experiment=None,
        prior=None,
        cdf_bins: int = 1000,
        cdf_samples: int = 500000,
        skip_sampling: bool = False,
        use_prior_flow: bool = True,
        param_keys: Sequence[str] | None = None,
        cdf_eps: float = 1e-10,
        matrix_columns: Sequence[str] | None = None,
    ):
        if not skip_sampling and experiment is None:
            raise ValueError(
                "experiment is required when building CDFs from sampling; "
                "use skip_sampling=True or Bijector.fit_from_matrix()."
            )
        self.experiment = experiment
        self._param_keys = None if param_keys is None else list(param_keys)
        self.use_prior_flow = bool(use_prior_flow)
        if prior is not None:
            self.prior = prior
        elif experiment is not None:
            self.prior = experiment.prior
        else:
            self.prior = None
        self.cdf_eps = float(cdf_eps)
        self.matrix_columns: list[str] | None = (
            None if matrix_columns is None else list(matrix_columns)
        )
        self.joint_state: dict[str, Any] | None = None

        if skip_sampling:
            self.cdfs: dict[str, dict[str, torch.Tensor]] = {}
        else:
            self.cdfs = self.create_cdfs(
                num_bins=int(cdf_bins), num_samples=int(cdf_samples)
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def fit_from_matrix(
        cls,
        x_ref,
        feature_names: Sequence[str],
        *,
        input_transform_type: str = "joint",
        device: str | torch.device = "cpu",
        cdf_bins: int = 5000,
        cdf_eps: float = 1e-6,
        shrinkage: float = 1e-3,
        max_rows: int = 50_000,
        seed: int = 0,
    ) -> "Bijector":
        """Fit CDFs (and optional joint whitening) from a ``(N, D)`` reference matrix."""
        device = torch.device(device)
        x = torch.as_tensor(x_ref, dtype=torch.float64, device=device)
        if x.ndim != 2:
            raise ValueError(f"x_ref must be 2D, got shape {tuple(x.shape)}")
        names = list(feature_names)
        if x.shape[1] != len(names):
            raise ValueError(
                f"feature_names length {len(names)} != x.shape[1] {x.shape[1]}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("x_ref contains non-finite values")

        if x.shape[0] > max_rows:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            idx = torch.randperm(x.shape[0], generator=g, device=device)[:max_rows]
            x = x[idx]

        bj = cls(
            experiment=None,
            skip_sampling=True,
            cdf_eps=cdf_eps,
            matrix_columns=names,
        )
        bj.cdfs = bj._cdfs_from_matrix_columns(x, num_bins=int(cdf_bins))

        if _normalize_input_transform_type(input_transform_type) == "joint":
            bj.fit_joint_gaussianizer(
                x,
                param_names=names,
                param_indices=list(range(len(names))),
                shrinkage=shrinkage,
                max_rows=max_rows,
                device=device,
            )
        return bj

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        experiment=None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Bijector":
        """Restore a bijector without re-sampling."""
        bj = cls(experiment=experiment, skip_sampling=True)
        bj.set_state(state, device=device, dtype=dtype)
        return bj

    def _cdfs_from_matrix_columns(
        self, x: torch.Tensor, *, num_bins: int
    ) -> dict[str, dict[str, torch.Tensor]]:
        names = self._matrix_column_names()
        cdfs: dict[str, dict[str, torch.Tensor]] = {}
        for j, name in enumerate(names):
            col = x[:, j]
            sorted_samples, _ = torch.sort(col)
            n_samples = int(sorted_samples.numel())
            if n_samples < 2:
                raise ValueError(f"Need at least two samples to build CDF for {name!r}")
            low = sorted_samples[0]
            high = sorted_samples[-1]
            bins = torch.linspace(low, high, int(num_bins), device=x.device, dtype=x.dtype)
            counts = torch.searchsorted(sorted_samples, bins, right=True)
            cdf_values = counts.to(x.dtype) / float(n_samples)
            cdf_values = torch.clamp(cdf_values, self.cdf_eps, 1.0 - self.cdf_eps)
            cdfs[name] = {"bins": bins, "cdf_values": cdf_values}
        return cdfs

    def _matrix_column_names(self) -> list[str]:
        if self.matrix_columns is not None:
            return list(self.matrix_columns)
        return list(self.cdfs.keys())

    def _default_device(self) -> torch.device:
        sample = next(iter(self.cdfs.values()), None)
        if sample is not None:
            return sample["bins"].device
        if self.experiment is not None:
            return torch.device(getattr(self.experiment, "device", "cpu"))
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # CDF construction from experiment samples
    # ------------------------------------------------------------------

    def create_cdfs(self, num_bins: int, num_samples: int) -> dict[str, dict[str, torch.Tensor]]:
        """Create marginal empirical-CDF tables from ``experiment.sample_parameters``."""
        with pyro.plate_stack("plate", (num_samples,)):
            empirical_prior = self.experiment.sample_parameters(
                (num_samples,), prior=self.prior, use_prior_flow=self.use_prior_flow
            )

        cdfs: dict[str, dict[str, torch.Tensor]] = {}
        keys = self._param_keys
        for key, samples in empirical_prior.items():
            if keys is not None and key not in keys:
                continue
            flat = samples.detach().flatten()
            sorted_samples, _ = torch.sort(flat)
            n_samples = int(sorted_samples.numel())
            if n_samples < 2:
                raise ValueError(f"Need at least two samples to build CDF for {key!r}")

            low, high = self._bin_bracket(self.prior[key], key)
            low = torch.as_tensor(low, device=samples.device, dtype=samples.dtype).reshape(())
            high = torch.as_tensor(high, device=samples.device, dtype=samples.dtype).reshape(())
            if not torch.isfinite(low) or not torch.isfinite(high) or not (high > low):
                low = sorted_samples[0]
                high = sorted_samples[-1]

            bins = torch.linspace(low, high, int(num_bins), device=samples.device, dtype=samples.dtype)
            counts = torch.searchsorted(sorted_samples, bins, right=True)
            cdf_values = counts.to(samples.dtype) / float(n_samples)
            cdf_values = torch.clamp(cdf_values, self.cdf_eps, 1.0 - self.cdf_eps)
            cdfs[key] = {"bins": bins, "cdf_values": cdf_values}
        return cdfs

    @staticmethod
    def _bin_bracket(prior_dist, key: str, n_std: float = 12.0):
        if isinstance(prior_dist, dist.Uniform):
            return prior_dist.low, prior_dist.high
        if isinstance(prior_dist, dist.Gamma):
            mean = prior_dist.mean
            std = torch.sqrt(prior_dist.variance)
            return torch.zeros_like(mean), mean + n_std * std
        if isinstance(prior_dist, dist.Normal):
            mean = prior_dist.mean
            std = torch.sqrt(prior_dist.variance)
            return mean - n_std * std, mean + n_std * std
        if EmpiricalPrior and isinstance(prior_dist, EmpiricalPrior):
            return prior_dist.low, prior_dist.high
        raise NotImplementedError(
            f"Bijector bin bracket not defined for prior type {type(prior_dist).__name__} "
            f"on parameter {key!r}."
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        cdfs = {
            key: {
                "bins": value["bins"].detach().cpu(),
                "cdf_values": value["cdf_values"].detach().cpu(),
            }
            for key, value in self.cdfs.items()
        }
        state: dict[str, Any] = {
            "class": "Bijector",
            "version": 3,
            "cdfs": cdfs,
            "joint_state": self._joint_state_cpu() if self.joint_state is not None else None,
            "cdf_eps": float(self.cdf_eps),
        }
        if self.matrix_columns is not None:
            state["matrix_columns"] = list(self.matrix_columns)
        return state

    def _joint_state_cpu(self) -> dict[str, Any]:
        assert self.joint_state is not None
        out = {}
        for k, v in self.joint_state.items():
            out[k] = v.detach().cpu() if torch.is_tensor(v) else v
        return out

    def set_state(self, state: dict[str, Any], device=None, dtype=None):
        """Restore CDF and joint-whitening state from :meth:`get_state`."""
        if "cdfs" not in state:
            raise ValueError("Bijector state must contain a 'cdfs' entry")

        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)

        cdf_part = state["cdfs"]
        joint_part = state.get("joint_state")
        if "cdf_eps" in state:
            self.cdf_eps = float(state["cdf_eps"])
        self.matrix_columns = state.get("matrix_columns")
        if self.matrix_columns is not None:
            self.matrix_columns = list(self.matrix_columns)

        sample = next(iter(cdf_part.values()), None)
        default_device = sample["bins"].device if sample else torch.device("cpu")
        default_dtype = sample["bins"].dtype if sample else torch.float64
        target_device = device if device is not None else default_device
        target_dtype = dtype if dtype is not None else default_dtype

        self.cdfs = {}
        for key, tensors in cdf_part.items():
            self.cdfs[key] = {
                "bins": tensors["bins"].to(device=target_device, dtype=target_dtype),
                "cdf_values": tensors["cdf_values"].to(device=target_device, dtype=target_dtype),
            }

        if joint_part is None:
            self.joint_state = None
        else:
            self.joint_state = {}
            for k, v in joint_part.items():
                if torch.is_tensor(v):
                    self.joint_state[k] = v.to(device=target_device, dtype=target_dtype)
                else:
                    self.joint_state[k] = v

    # ------------------------------------------------------------------
    # Matrix-block transforms (SED features, etc.)
    # ------------------------------------------------------------------

    def matrix_to_gaussian(
        self,
        x,
        *,
        apply_joint: bool | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Map ``(..., D)`` physical features to Gaussianized coords (column order = ``matrix_columns``)."""
        if apply_joint is None:
            apply_joint = self.uses_joint_gaussianizer()
        dev = self._default_device() if device is None else torch.device(device)
        x_t = torch.as_tensor(x, dtype=torch.float64, device=dev)
        names = self._matrix_column_names()
        d = len(names)
        if x_t.shape[-1] != d:
            raise ValueError(f"Expected last dim {d}, got {x_t.shape[-1]}")
        if apply_joint and self.joint_state is not None:
            return self.prior_to_gaussian_joint(x_t)
        y = x_t.clone()
        for j, name in enumerate(names):
            y[..., j : j + 1] = self.prior_to_gaussian(x_t[..., j : j + 1], name)
        return y

    def matrix_from_gaussian(
        self,
        y,
        *,
        apply_joint: bool | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Inverse of :meth:`matrix_to_gaussian`."""
        if apply_joint is None:
            apply_joint = self.uses_joint_gaussianizer()
        dev = self._default_device() if device is None else torch.device(device)
        y_t = torch.as_tensor(y, dtype=torch.float64, device=dev)
        names = self._matrix_column_names()
        d = len(names)
        if y_t.shape[-1] != d:
            raise ValueError(f"Expected last dim {d}, got {y_t.shape[-1]}")
        if apply_joint and self.joint_state is not None:
            return self.gaussian_to_prior_joint(y_t)
        x = y_t.clone()
        for j, name in enumerate(names):
            x[..., j : j + 1] = self.gaussian_to_prior(y_t[..., j : j + 1], name)
        return x

    # ------------------------------------------------------------------
    # Per-parameter marginal transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _interp(x_query: torch.Tensor, x0, x1, y0, y1) -> torch.Tensor:
        eps = torch.as_tensor(1e-30, device=x_query.device, dtype=x_query.dtype)
        dx = torch.clamp(x1 - x0, min=eps)
        t = (x_query - x0) / dx
        return y0 + t * (y1 - y0)

    def _cdf_lookup(self, samples: torch.Tensor, param_key: str) -> torch.Tensor:
        bins = self.cdfs[param_key]["bins"].to(device=samples.device, dtype=samples.dtype)
        cdf_values = self.cdfs[param_key]["cdf_values"].to(device=samples.device, dtype=samples.dtype)
        flat = torch.clamp(samples.flatten(), bins[0], bins[-1])
        hi = torch.searchsorted(bins, flat, right=False)
        hi = torch.clamp(hi, 1, len(bins) - 1)
        lo = hi - 1
        u = self._interp(flat, bins[lo], bins[hi], cdf_values[lo], cdf_values[hi])
        return torch.clamp(u, self.cdf_eps, 1.0 - self.cdf_eps).reshape(samples.shape)

    def _icdf_lookup(self, u: torch.Tensor, param_key: str) -> torch.Tensor:
        bins = self.cdfs[param_key]["bins"].to(device=u.device, dtype=u.dtype)
        cdf_values = self.cdfs[param_key]["cdf_values"].to(device=u.device, dtype=u.dtype)
        flat = torch.clamp(u.flatten(), self.cdf_eps, 1.0 - self.cdf_eps)
        hi = torch.searchsorted(cdf_values, flat, right=False)
        hi = torch.clamp(hi, 1, len(cdf_values) - 1)
        lo = hi - 1
        x = self._interp(flat, cdf_values[lo], cdf_values[hi], bins[lo], bins[hi])
        return x.reshape(u.shape)

    def prior_to_gaussian(self, samples: torch.Tensor, param_key: str, target_mean=0.0, target_std=1.0):
        u = self._cdf_lookup(samples, param_key)
        z = np.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        return (target_mean + target_std * z).to(samples.dtype).reshape(samples.shape)

    def gaussian_to_prior(self, gaussian_samples: torch.Tensor, param_key: str, source_mean=0.0, source_std=1.0):
        z = (gaussian_samples - source_mean) / source_std
        u = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
        return self._icdf_lookup(u, param_key).to(gaussian_samples.dtype).reshape(gaussian_samples.shape)

    def log_abs_det_jacobian(self, samples: torch.Tensor, param_key: str):
        bins = self.cdfs[param_key]["bins"].to(device=samples.device, dtype=samples.dtype)
        cdf_values = self.cdfs[param_key]["cdf_values"].to(device=samples.device, dtype=samples.dtype)

        flat = torch.clamp(samples.flatten(), bins[0], bins[-1])
        hi = torch.searchsorted(bins, flat, right=False)
        hi = torch.clamp(hi, 1, len(bins) - 1)
        lo = hi - 1

        db = torch.clamp(bins[hi] - bins[lo], min=1e-30)
        dc = torch.clamp(cdf_values[hi] - cdf_values[lo], min=1e-30)
        log_f_emp = torch.log(dc) - torch.log(db)
        z = self.prior_to_gaussian(samples, param_key).flatten()
        log_phi = -0.5 * z.pow(2) - 0.5 * np.log(2.0 * np.pi)
        return (log_f_emp - log_phi).reshape(samples.shape).to(samples.dtype)

    # ------------------------------------------------------------------
    # Joint whitening
    # ------------------------------------------------------------------

    def fit_joint_gaussianizer(
        self,
        x_phys,
        *,
        param_names: Sequence[str],
        param_indices: Sequence[int],
        shrinkage: float = 1e-3,
        max_rows: int = 50_000,
        device=None,
    ):
        """Fit Cholesky whitening in marginal normal-score space for a parameter block."""
        if device is None:
            if self.experiment is not None:
                device = getattr(self.experiment, "device", "cpu")
            else:
                device = self._default_device()
        device = torch.device(device)
        x = torch.as_tensor(x_phys, dtype=torch.float64, device=device)
        if x.ndim != 2 or x.shape[1] != len(param_names):
            raise ValueError(
                f"x_phys must have shape (N, {len(param_names)}), got {tuple(x.shape)}"
            )
        if x.shape[0] > max_rows:
            idx = torch.randperm(x.shape[0], device=device)[:max_rows]
            x = x[idx]

        cols = []
        for j, name in enumerate(param_names):
            cols.append(self.prior_to_gaussian(x[:, j : j + 1], name).reshape(-1))
        q = torch.stack(cols, dim=1)
        mean = q.mean(dim=0)
        qc = q - mean
        cov = qc.T @ qc / max(q.shape[0] - 1, 1)
        cov = cov + float(shrinkage) * torch.eye(cov.shape[0], dtype=q.dtype, device=q.device)
        chol = torch.linalg.cholesky(cov)
        inv_chol = torch.linalg.inv(chol)
        self.joint_state = {
            "param_names": list(param_names),
            "param_indices": list(param_indices),
            "mean": mean,
            "chol": chol,
            "inv_chol": inv_chol,
            "shrinkage": float(shrinkage),
        }

    def uses_joint_gaussianizer(self) -> bool:
        return self.joint_state is not None

    def prior_to_gaussian_joint(self, samples: torch.Tensor) -> torch.Tensor:
        if self.joint_state is None:
            raise RuntimeError("Joint gaussianizer is not fitted.")
        y = samples.clone()
        names = self.joint_state["param_names"]
        idxs = self.joint_state["param_indices"]
        cols = []
        for name, idx in zip(names, idxs):
            cols.append(self.prior_to_gaussian(samples[..., idx : idx + 1], name))
        q = torch.cat(cols, dim=-1)
        mean = self.joint_state["mean"].to(device=samples.device, dtype=samples.dtype)
        inv_chol = self.joint_state["inv_chol"].to(device=samples.device, dtype=samples.dtype)
        z = (q - mean) @ inv_chol.T
        for j, idx in enumerate(idxs):
            y[..., idx : idx + 1] = z[..., j : j + 1]
        return y

    def gaussian_to_prior_joint(self, samples: torch.Tensor) -> torch.Tensor:
        if self.joint_state is None:
            raise RuntimeError("Joint gaussianizer is not fitted.")
        x = samples.clone()
        names = self.joint_state["param_names"]
        idxs = self.joint_state["param_indices"]
        z = torch.cat([samples[..., idx : idx + 1] for idx in idxs], dim=-1)
        mean = self.joint_state["mean"].to(device=samples.device, dtype=samples.dtype)
        chol = self.joint_state["chol"].to(device=samples.device, dtype=samples.dtype)
        q = z @ chol.T + mean
        for j, (name, idx) in enumerate(zip(names, idxs)):
            x[..., idx : idx + 1] = self.gaussian_to_prior(q[..., j : j + 1], name)
        return x

    def joint_log_abs_det_jacobian(self, samples: torch.Tensor) -> torch.Tensor:
        """log|det dT/dx| for the joint physical -> Gaussianized map."""
        if self.joint_state is None:
            raise RuntimeError("Joint gaussianizer is not fitted.")
        flat = samples.reshape(-1, samples.shape[-1])
        log_det = flat.new_zeros(flat.shape[0])
        for name, idx in zip(self.joint_state["param_names"], self.joint_state["param_indices"]):
            log_det = log_det + self.log_abs_det_jacobian(flat[:, idx : idx + 1], name).reshape(-1)
        inv_chol = self.joint_state["inv_chol"].to(device=samples.device, dtype=samples.dtype)
        _sign, logabs = torch.linalg.slogdet(inv_chol)
        log_det = log_det + logabs
        return log_det.reshape(samples.shape[:-1])

    # ------------------------------------------------------------------
    # Interval / logit helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _logit(x, eps: float = 1e-6):
        x = torch.clamp(x, eps, 1.0 - eps)
        return torch.log(x) - torch.log1p(-x)

    @staticmethod
    def _sigmoid(y: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y)

    @staticmethod
    def _to_unit_interval(x: torch.Tensor, a: float, b: float, eps: float = 1e-6) -> torch.Tensor:
        u = (x - a) / (b - a)
        return torch.clamp(u, eps, 1.0 - eps)

    @staticmethod
    def _from_unit_interval(u: torch.Tensor, a: float, b: float) -> torch.Tensor:
        return a + u * (b - a)

    @staticmethod
    def _interval_to_R(x: torch.Tensor, a: float, b: float, eps: float = 1e-6) -> torch.Tensor:
        u01 = Bijector._to_unit_interval(x, a, b, eps)
        u = torch.clamp(2.0 * u01 - 1.0, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(u) - torch.log1p(-u))

    @staticmethod
    def _R_to_interval(y: torch.Tensor, a: float, b: float) -> torch.Tensor:
        u = torch.tanh(y)
        return Bijector._from_unit_interval(0.5 * (u + 1.0), a, b)

    @staticmethod
    def _log_interval_to_R(x: torch.Tensor, a: float, b: float, H: float = 5.0) -> torch.Tensor:
        a_t = torch.tensor(a, device=x.device, dtype=x.dtype)
        b_t = torch.tensor(b, device=x.device, dtype=x.dtype)
        H_t = torch.tensor(H, device=x.device, dtype=x.dtype)
        loga = torch.log(a_t)
        logb = torch.log(b_t)
        c = 0.5 * (loga + logb)
        s = (2.0 * H_t) / (logb - loga)
        return s * (torch.log(x) - c)

    @staticmethod
    def _R_to_log_interval(y: torch.Tensor, a: float, b: float, H: float = 5.0) -> torch.Tensor:
        a_t = torch.tensor(a, device=y.device, dtype=y.dtype)
        b_t = torch.tensor(b, device=y.device, dtype=y.dtype)
        H_t = torch.tensor(H, device=y.device, dtype=y.dtype)
        loga = torch.log(a_t)
        logb = torch.log(b_t)
        c = 0.5 * (loga + logb)
        s = (2.0 * H_t) / (logb - loga)
        return torch.exp(y / s + c)
