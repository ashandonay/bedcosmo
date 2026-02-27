"""
Generic grid-based EIG helpers using bayesdesign/bed ExperimentDesigner.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from bed.design import ExperimentDesigner
from bed.grid import Grid
from pyro import distributions as dist


def _dist_bounds(distribution, q_lo=0.01, q_hi=0.99) -> Tuple[float, float]:
    # Uniform-like
    if hasattr(distribution, "low") and hasattr(distribution, "high"):
        return float(distribution.low.detach().cpu()), float(distribution.high.detach().cpu())
    # Try analytic quantiles for other distributions.
    q = torch.tensor([q_lo, q_hi], dtype=torch.float64)
    try:
        x = distribution.icdf(q)
        return float(x[0].detach().cpu()), float(x[1].detach().cpu())
    except (NotImplementedError, RuntimeError, AttributeError):
        pass

    # Fallback: Monte Carlo quantiles for distributions lacking icdf.
    with torch.no_grad():
        samples = distribution.sample((20000,)).to(torch.float64)
    lo = float(torch.quantile(samples, q_lo).detach().cpu())
    hi = float(torch.quantile(samples, q_hi).detach().cpu())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"Could not infer valid bounds for distribution {distribution}.")
    return lo, hi


def _create_distribution_from_config(param_config: Dict[str, Any]) -> torch.distributions.Distribution:
    """
    Create a PyTorch distribution from a parameter configuration dict.

    Args:
        param_config: Dictionary with 'distribution' key containing type and bounds.

    Returns:
        PyTorch distribution object.
    """
    dist_config = param_config.get("distribution", {})
    dist_type = dist_config.get("type", "uniform").lower()

    if dist_type == "uniform":
        lower = float(dist_config.get("lower", 0.0))
        upper = float(dist_config.get("upper", 1.0))
        return dist.Uniform(
            torch.tensor(lower, dtype=torch.float64),
            torch.tensor(upper, dtype=torch.float64)
        )
    elif dist_type == "normal" or dist_type == "gaussian":
        mean = float(dist_config.get("mean", 0.0))
        std = float(dist_config.get("std", 1.0))
        return dist.Normal(
            torch.tensor(mean, dtype=torch.float64),
            torch.tensor(std, dtype=torch.float64)
        )
    elif dist_type == "truncatednormal":
        mean = float(dist_config.get("mean", 0.0))
        std = float(dist_config.get("std", 1.0))
        lower = float(dist_config.get("lower", mean - 4 * std))
        upper = float(dist_config.get("upper", mean + 4 * std))
        # Use TruncatedNormal from pyro
        base = dist.Normal(
            torch.tensor(mean, dtype=torch.float64),
            torch.tensor(std, dtype=torch.float64)
        )
        return dist.TruncatedDistribution(
            base,
            low=torch.tensor(lower, dtype=torch.float64),
            high=torch.tensor(upper, dtype=torch.float64)
        )
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


class GridCalculation:
    """
    Class-based interface for grid-based EIG computation.

    Wraps the bed.design.ExperimentDesigner and provides methods for:
    - Building parameter, feature, and design grids
    - Computing prior PDFs from experiment priors or YAML configuration
    - Running grid-based EIG calculation
    - Extracting posteriors and samples
    """

    def __init__(
        self,
        experiment,
        param_pts: int = 231,
        feature_pts: int = 101,
        device: str = "cpu",
        adaptive_features: bool = False,
        adaptive_floor: float = 0.05,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        feature_dense_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        param_dense_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        feature_dense_fraction: float = 0.6,
        param_dense_fraction: float = 0.6,
    ):
        """
        Initialize the GridCalculation.

        Args:
            experiment: Experiment object with prior, cosmo_params, etc.
            param_pts: Number of points per parameter axis.
            feature_pts: Number of points per feature axis.
            device: Torch device (grid-based calculation runs on CPU for stability).
            adaptive_features: If True, use a patched-grid strategy that places
                a dense grid over the peak region (detectable features) and a
                sparse grid spanning the full range, then merges them.
            adaptive_floor: Unused (kept for API compatibility).
            feature_ranges: Per-feature (lo, hi) bounds for the feature grid.
                Keys are feature names (e.g. "y_u", "y_g").
                Features not listed fall back to automatic bounds.
                If None, all features use auto bounds.
            feature_dense_ranges: Per-feature (lo, hi) bounds for the dense sub-region.
                Keys are feature names (e.g. "y_u", "u", "g").
                For each feature with a dense region, the grid uses a
                patched-grid strategy placing ``feature_dense_fraction`` of
                points in the dense region and the rest spanning the full range.
                If None, no dense regions are used (unless adaptive_features
                infers them automatically).
            param_dense_ranges: Per-parameter (lo, hi) bounds for dense
                sub-regions in the parameter grid.  Keys are parameter names
                (e.g. "z", "w0").  Same patched-grid strategy as feature_dense_ranges.
            feature_dense_fraction: Fraction of feature grid points allocated
                to each dense region (default 0.6).
            param_dense_fraction: Fraction of parameter grid points allocated
                to each dense region (default 0.6).
        """
        self.experiment = experiment
        self.param_pts = param_pts
        self.feature_pts = feature_pts
        self.device = device
        self.adaptive_features = adaptive_features
        self.adaptive_floor = adaptive_floor
        self.feature_ranges = feature_ranges or {}
        self.feature_dense_ranges = feature_dense_ranges or {}
        self.param_dense_ranges = param_dense_ranges or {}
        self.feature_dense_fraction = feature_dense_fraction
        self.param_dense_fraction = param_dense_fraction

        # Create grids (design_grid before feature_grid since feature bounds depend on designs)
        self.parameter_grid = self._create_parameter_grid()
        self.design_grid = self._create_design_grid()
        self.feature_grid = self._create_feature_grid()

        # Prior PDF (computed lazily or explicitly)
        self._prior_pdf: Optional[np.ndarray] = None
        self._prior_distributions: Optional[Dict[str, torch.distributions.Distribution]] = None

        # Designer and results (populated after run)
        self.designer: Optional[ExperimentDesigner] = None
        self.result: Optional[Dict[str, Any]] = None
        self.eig: Optional[np.ndarray] = None
        self.best_design: Optional[Any] = None

    def _create_parameter_grid(self) -> Grid:
        """Create parameter grid from experiment prior."""
        axes: Dict[str, np.ndarray] = {}
        names = list(getattr(self.experiment, "cosmo_params", []))
        if not names:
            raise ValueError("Experiment must expose cosmo_params.")

        prior = getattr(self.experiment, "prior", {})
        for name in names:
            if name not in prior:
                raise ValueError(f"Prior missing parameter '{name}'.")
            lo, hi = _dist_bounds(prior[name])
            if name in self.param_dense_ranges:
                d_lo, d_hi = self.param_dense_ranges[name]
                axes[name] = self._dense_axis(lo, hi, d_lo, d_hi, int(self.param_pts), self.param_dense_fraction)
                print(f"  Param grid {name}: [{lo:.4f}, {hi:.4f}] (dense [{d_lo:.4f}, {d_hi:.4f}])")
            else:
                axes[name] = np.linspace(lo, hi, int(self.param_pts), dtype=np.float64)
        return Grid(**axes)

    @staticmethod
    def _adaptive_axis(
        feature_values: np.ndarray,
        errors: np.ndarray,
        lo: float,
        hi: float,
        n_pts: int,
        dense_fraction: float = 0.6,
    ) -> np.ndarray:
        """Build a non-uniform 1D axis using a patched-grid strategy.

        Allocates ``dense_fraction`` of the point budget to a dense grid
        covering the peak region (where detectable features live) and the
        remaining points to a sparse grid spanning the full range.  The two
        grids are merged and deduplicated to produce the final axis.

        The dense region is determined automatically from the features
        with small errors.

        Args:
            feature_values: 1D array of predicted feature values.
            errors: 1D array of corresponding feature errors.
            lo, hi: Full axis bounds.
            n_pts: Total number of grid points.
            dense_fraction: Fraction of points allocated to the dense region.

        Returns:
            Sorted 1D array of length ``n_pts`` spanning [lo, hi].
        """
        detectable_err_thresh = 3.0
        detectable = errors < detectable_err_thresh

        if np.any(detectable):
            det_features = feature_values[detectable]
            det_errs = errors[detectable]
            # Dense region: covers detectable features with generous padding.
            # Use per-object (mag ± n_sigma * err) with a floor so bright objects
            # with tiny errors still get a wide dense region.
            n_sigma_dense = 8.0
            err_floor = 1.0  # minimum 1 feature of sigma
            padded_errs = np.maximum(det_errs, err_floor)
            dense_lo = float(np.min(det_features - n_sigma_dense * padded_errs))
            dense_hi = float(np.max(det_features + n_sigma_dense * padded_errs))
            # Clamp to full axis bounds
            dense_lo = max(dense_lo, lo)
            dense_hi = min(dense_hi, hi)
        else:
            # No detectable features — fall back to uniform
            return np.linspace(lo, hi, n_pts, dtype=np.float64)

        n_dense = max(3, int(round(n_pts * dense_fraction)))
        n_sparse = max(3, n_pts - n_dense)

        dense_grid = np.linspace(dense_lo, dense_hi, n_dense)
        sparse_grid = np.linspace(lo, hi, n_sparse)

        # Merge and deduplicate (keep unique sorted values)
        merged = np.unique(np.concatenate([dense_grid, sparse_grid]))

        # If merge produced more points than budget, thin to n_pts
        # by picking equally-spaced indices (preserving endpoints)
        if len(merged) > n_pts:
            idx = np.round(np.linspace(0, len(merged) - 1, n_pts)).astype(int)
            merged = merged[idx]

        # If merge produced fewer (unlikely with unique), pad with linspace
        if len(merged) < n_pts:
            extra = np.linspace(lo, hi, n_pts - len(merged) + 2)[1:-1]
            merged = np.unique(np.concatenate([merged, extra]))[:n_pts]

        return merged.astype(np.float64)

    @staticmethod
    def _dense_axis(
        lo: float,
        hi: float,
        dense_lo: float,
        dense_hi: float,
        n_pts: int,
        dense_fraction: float = 0.6,
    ) -> np.ndarray:
        """Build a non-uniform 1D axis with a dense sub-region.

        Like ``_adaptive_axis`` but takes explicit feature-dense-range bounds
        instead of inferring them from feature values and errors.

        Args:
            lo, hi: Full axis bounds.
            dense_lo, dense_hi: Bounds of the dense sub-region (clamped to
                [lo, hi]).
            n_pts: Total number of grid points.
            dense_fraction: Fraction of points allocated to the dense region.

        Returns:
            Sorted 1D array of length ``n_pts`` spanning [lo, hi].
        """
        dense_lo = max(dense_lo, lo)
        dense_hi = min(dense_hi, hi)
        if dense_lo >= dense_hi:
            return np.linspace(lo, hi, n_pts, dtype=np.float64)

        n_dense = max(3, int(round(n_pts * dense_fraction)))
        n_sparse = max(3, n_pts - n_dense)

        dense_grid = np.linspace(dense_lo, dense_hi, n_dense)
        sparse_grid = np.linspace(lo, hi, n_sparse)

        merged = np.unique(np.concatenate([dense_grid, sparse_grid]))

        if len(merged) > n_pts:
            idx = np.round(np.linspace(0, len(merged) - 1, n_pts)).astype(int)
            merged = merged[idx]

        if len(merged) < n_pts:
            extra = np.linspace(lo, hi, n_pts - len(merged) + 2)[1:-1]
            merged = np.unique(np.concatenate([merged, extra]))[:n_pts]

        return merged.astype(np.float64)

    def _create_feature_grid(self) -> Grid:
        """Create feature grid from explicit ranges or infer from experiment."""

        # If all feature ranges are explicitly provided, use them directly.
        # This works for any experiment regardless of type.
        if self.feature_ranges and hasattr(self.experiment, "design_labels"):
            feature_names = [f"y_{d}" for d in self.experiment.design_labels]
            all_covered = all(
                d in self.feature_ranges or f"y_{d}" in self.feature_ranges
                for d in self.experiment.design_labels
            )
            if all_covered:
                axes = {}
                for d in self.experiment.design_labels:
                    key = d if d in self.feature_ranges else f"y_{d}"
                    lo, hi = self.feature_ranges[key]
                    name = f"y_{d}"
                    dense_key = d if d in self.feature_dense_ranges else f"y_{d}" if f"y_{d}" in self.feature_dense_ranges else None
                    if dense_key is not None:
                        d_lo, d_hi = self.feature_dense_ranges[dense_key]
                        axes[name] = self._dense_axis(lo, hi, d_lo, d_hi, int(self.feature_pts), self.feature_dense_fraction)
                        print(f"  Feature grid {d}: [{lo:.1f}, {hi:.1f}] (explicit, dense [{d_lo:.1f}, {d_hi:.1f}])")
                    else:
                        axes[name] = np.linspace(lo, hi, int(self.feature_pts), dtype=np.float64)
                        print(f"  Feature grid {d}: [{lo:.1f}, {hi:.1f}] (explicit)")
                return Grid(**axes)

        # Infer feature grid from experiment.
        if hasattr(self.experiment, "design_labels"):
            if self.experiment.name == "num_visits":
                if not hasattr(self.parameter_grid, "z"):
                    raise ValueError("Auto feature grid requires parameter grid to include 'z'.")
                z = np.asarray(getattr(self.parameter_grid, "z"), dtype=np.float64)
                z_tensor = torch.as_tensor(z, device="cpu", dtype=torch.float64)
                with torch.no_grad():
                    features = self.experiment._calculate_magnitudes(z_tensor).detach().cpu().numpy()
            else:
                raise NotImplementedError(f"Feature grid inference not implemented for {self.experiment.name}.")

            features_tensor = torch.as_tensor(features, device="cpu", dtype=torch.float64)
            # Compute feature errors using the worst-case (minimum) visits
            # across all designs so the feature grid covers the full range.
            # Start from nominal design, then override with design grid mins.
            nominal = self.experiment.nominal_design.detach().cpu().to(torch.float64).numpy()
            min_visits = nominal.copy()
            design_grid = self.design_grid
            for j, d in enumerate(self.experiment.design_labels):
                if d in design_grid.names:
                    d_vals = np.asarray(design_grid.axes_in[d], dtype=np.float64)
                    min_visits[j] = d_vals.min()
            min_visits_tensor = torch.as_tensor(
                min_visits, device="cpu", dtype=torch.float64
            ).expand(features_tensor.shape)
            if self.experiment.name == "num_visits":
                feature_errors = self.experiment._magnitude_errors(
                    features_tensor, min_visits_tensor
                ).detach().cpu().numpy()
            else:
                raise NotImplementedError(f"Feature grid inference not implemented for {self.experiment.name}.")

            axes = {}
            for i, d in enumerate(self.experiment.design_labels):
                err_i = feature_errors[..., i]
                feature_i = features[..., i]

                range_key = d if d in self.feature_ranges else f"y_{d}" if f"y_{d}" in self.feature_ranges else None
                if range_key is not None:
                    lo, hi = self.feature_ranges[range_key]
                else:
                    # Auto bounds: use per-object Gaussian envelopes
                    # feature ± n_sigma * err, capped to prevent blow-up.
                    n_sigma = 4.0
                    feature_err_cap = 5.0
                    err_capped = np.minimum(err_i, feature_err_cap)
                    lo = float(np.min(feature_i - n_sigma * err_capped))
                    hi = float(np.max(feature_i + n_sigma * err_capped))
                    padding = 0.10 * (hi - lo)
                    lo -= padding
                    hi += padding

                dense_key = d if d in self.feature_dense_ranges else f"y_{d}" if f"y_{d}" in self.feature_dense_ranges else None
                if dense_key is not None:
                    d_lo, d_hi = self.feature_dense_ranges[dense_key]
                    print(f"  Feature grid {d}: [{lo:.1f}, {hi:.1f}] (dense [{d_lo:.1f}, {d_hi:.1f}])")
                    axes[f"y_{d}"] = self._dense_axis(lo, hi, d_lo, d_hi, int(self.feature_pts), self.feature_dense_fraction)
                elif self.adaptive_features:
                    print(f"  Feature grid {d}: [{lo:.1f}, {hi:.1f}] (adaptive)")
                    axes[f"y_{d}"] = self._adaptive_axis(
                        feature_values=feature_i.ravel(),
                        feature_errors=err_i.ravel(),
                        lo=lo,
                        hi=hi,
                        n_pts=int(self.feature_pts),
                    )
                else:
                    print(f"  Feature grid {d}: [{lo:.1f}, {hi:.1f}]")
                    axes[f"y_{d}"] = np.linspace(lo, hi, int(self.feature_pts), dtype=np.float64)

            return Grid(**axes)

        raise ValueError(
            "Cannot auto-build feature grid for this experiment; provide feature_axes explicitly."
        )

    def _create_design_grid(self) -> Grid:
        """Get design grid from experiment."""
        if not hasattr(self.experiment, "designs_grid") or self.experiment.designs_grid is None:
            raise ValueError(
                "Experiment does not expose designs_grid. "
                "Define it in experiment.init_designs (e.g., via BaseExperiment._build_design_grid)."
            )
        return self.experiment.designs_grid

    @property
    def prior_pdf(self) -> Optional[np.ndarray]:
        """Get the prior PDF array."""
        return self._prior_pdf

    @property
    def prior_distributions(self) -> Optional[Dict[str, torch.distributions.Distribution]]:
        """Get the prior distributions dictionary."""
        return self._prior_distributions

    def compute_prior_pdf(
        self,
        prior_args: Optional[Dict[str, Any]] = None,
        prior_args_path: Optional[str] = None,
        use_experiment_prior: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute the prior PDF enumerated over the parameter grid.

        Computes the joint prior probability at each grid point by evaluating
        the log probability under each marginal prior distribution and summing
        (assuming parameter independence).

        Args:
            prior_args: Dictionary of prior arguments (from YAML structure).
                       If provided, distributions are built from this config.
            prior_args_path: Path to prior_args YAML file.
                            Takes precedence over prior_args if both provided.
            use_experiment_prior: If True and no prior_args/path provided,
                                 uses self.experiment.prior distributions.
            normalize: If True, normalize the PDF to sum to 1.

        Returns:
            np.ndarray: Prior PDF values at each grid point, shape matches parameter_grid.shape.
        """
        # Determine source of prior distributions
        if prior_args_path is not None:
            # Load from YAML file
            resolved_path = Path(prior_args_path)
            if not resolved_path.is_absolute():
                # Try to resolve relative to experiments directory
                from bedcosmo.util import get_experiment_config_path
                cosmo_exp = getattr(self.experiment, "_name", None) or self.experiment.__class__.__name__.lower()
                try:
                    resolved_path = get_experiment_config_path(cosmo_exp, prior_args_path)
                except FileNotFoundError:
                    resolved_path = Path(prior_args_path)

            if not resolved_path.exists():
                raise FileNotFoundError(f"Prior args file not found: {resolved_path}")

            with open(resolved_path, "r") as f:
                prior_args = yaml.safe_load(f)

        if prior_args is not None:
            # Build distributions from prior_args config
            self._prior_distributions = self._build_distributions_from_args(prior_args)
        elif use_experiment_prior:
            # Use experiment's existing prior distributions
            prior = getattr(self.experiment, "prior", None)
            if prior is None:
                raise ValueError("Experiment does not have a prior attribute.")
            self._prior_distributions = prior
        else:
            raise ValueError(
                "Must provide prior_args, prior_args_path, or set use_experiment_prior=True."
            )

        # Compute the PDF over the parameter grid
        self._prior_pdf = self._evaluate_prior_on_grid(normalize=normalize)
        return self._prior_pdf

    def _build_distributions_from_args(
        self, prior_args: Dict[str, Any]
    ) -> Dict[str, torch.distributions.Distribution]:
        """
        Build a dictionary of distributions from prior_args configuration.

        Args:
            prior_args: YAML-loaded prior configuration with 'parameters' key.

        Returns:
            Dictionary mapping parameter names to distributions.
        """
        parameters = prior_args.get("parameters", {})
        param_names = list(self.parameter_grid.names)

        distributions = {}
        for name in param_names:
            if name not in parameters:
                raise ValueError(
                    f"Parameter '{name}' not found in prior_args. "
                    f"Available: {list(parameters.keys())}"
                )
            param_config = parameters[name]
            distributions[name] = _create_distribution_from_config(param_config)

        return distributions

    def _evaluate_prior_on_grid(self, normalize: bool = True) -> np.ndarray:
        """
        Evaluate the joint prior PDF at each point in the parameter grid.

        Assumes parameter independence (joint = product of marginals).

        Returns:
            np.ndarray with shape matching parameter_grid.shape.
        """
        if self._prior_distributions is None:
            raise ValueError("Prior distributions not set. Call compute_prior_pdf first.")

        param_names = list(self.parameter_grid.names)
        grid_shape = tuple(self.parameter_grid.shape)

        # Get axis values for each parameter
        axes = [np.asarray(getattr(self.parameter_grid, name), dtype=np.float64) for name in param_names]

        # Create meshgrid of all parameter values
        # For bed.grid.Grid, the axes are already broadcast-compatible
        # We need to evaluate the joint log prob at each grid point

        # Initialize log prob array
        log_prob = np.zeros(grid_shape, dtype=np.float64)

        # For each parameter, evaluate its marginal log prob and add to joint
        for i, name in enumerate(param_names):
            distribution = self._prior_distributions[name]
            axis_values = axes[i]

            # Create tensor from axis values
            values_tensor = torch.as_tensor(axis_values, dtype=torch.float64)

            # Evaluate log probability
            with torch.no_grad():
                marginal_log_prob = distribution.log_prob(values_tensor).detach().cpu().numpy()

            # Handle non-finite values (outside support)
            marginal_log_prob = np.where(
                np.isfinite(marginal_log_prob), marginal_log_prob, -np.inf
            )

            # Broadcast to grid shape and add to joint log prob
            # The Grid stores axes such that axis[i] broadcasts along dimension i
            # We need to reshape marginal_log_prob to broadcast correctly
            shape_for_broadcast = [1] * len(grid_shape)
            shape_for_broadcast[i] = len(axis_values)
            marginal_log_prob = marginal_log_prob.reshape(shape_for_broadcast)

            log_prob = log_prob + marginal_log_prob

        # Convert to probability
        # Subtract max for numerical stability before exp
        log_prob_max = np.max(log_prob[np.isfinite(log_prob)]) if np.any(np.isfinite(log_prob)) else 0.0
        prob = np.exp(log_prob - log_prob_max)
        prob = np.where(np.isfinite(prob), prob, 0.0)

        if normalize:
            total = np.sum(prob)
            if total > 0:
                prob = prob / total

        return prob.astype(np.float64)

    def run(
        self,
        prior: Optional[np.ndarray] = None,
        lfunc_name: str = "unnorm_lfunc",
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the grid-based EIG calculation.

        Args:
            prior: Prior PDF array. If None, uses self._prior_pdf if computed,
                  otherwise ExperimentDesigner uses uniform prior.
            lfunc_name: Name of likelihood function attribute on experiment.
            debug: Enable debug output from ExperimentDesigner.

        Returns:
            Dictionary with results including 'best_design', 'eig', 'designer'.
        """
        # Use computed prior if available and no override provided
        if prior is None:
            prior = self._prior_pdf

        if not hasattr(self.experiment, lfunc_name):
            raise ValueError(f"Experiment missing likelihood function '{lfunc_name}'.")
        lfunc = getattr(self.experiment, lfunc_name)

        self.designer = ExperimentDesigner(
            self.parameter_grid,
            self.feature_grid,
            self.design_grid,
            lfunc
        )

        self.best_design = self.designer.calculateEIG(prior, debug=debug)
        self.eig = np.asarray(self.designer.EIG, dtype=np.float64)

        self.result = {
            "best_design": self.best_design,
            "eig": self.eig,
            "prior": prior,
            "designer": self.designer,
            "parameter_grid": self.parameter_grid,
            "feature_grid": self.feature_grid,
            "design_grid": self.design_grid,
            "parameter_grid_shape": tuple(self.parameter_grid.shape),
            "feature_grid_shape": tuple(self.feature_grid.shape),
            "design_grid_shape": tuple(self.design_grid.shape),
        }

        return self.result

    def get_posterior(self, nominal: bool = True) -> np.ndarray:
        """
        Get the posterior PDF over the parameter grid.

        Args:
            nominal: If True, condition on nominal design and central features.

        Returns:
            np.ndarray: Posterior PDF values.
        """
        if self.designer is None:
            raise RuntimeError("Must call run() before getting posterior.")

        if not nominal:
            raise NotImplementedError(
                "Non-nominal posterior conditioning is not implemented yet."
            )

        if not hasattr(self.experiment, "nominal_design"):
            raise ValueError("Experiment must expose nominal_design.")
        if not hasattr(self.experiment, "central_val"):
            raise ValueError("Experiment must expose central_val.")

        nominal_design = torch.as_tensor(
            self.experiment.nominal_design, dtype=torch.float64
        ).detach().cpu().numpy().reshape(-1)
        central_features = torch.as_tensor(
            self.experiment.central_val, dtype=torch.float64
        ).detach().cpu().numpy().reshape(-1)

        design_names = list(self.design_grid.names)
        feature_names = list(self.feature_grid.names)

        if nominal_design.size != len(design_names):
            raise ValueError(
                f"Nominal design size mismatch: {nominal_design.size} vs {len(design_names)}."
            )
        if central_features.size != len(feature_names):
            raise ValueError(
                f"Central feature size mismatch: {central_features.size} vs {len(feature_names)}."
            )

        conditioning = {
            name: float(nominal_design[i]) for i, name in enumerate(design_names)
        }
        conditioning.update(
            {name: float(central_features[i]) for i, name in enumerate(feature_names)}
        )

        posterior = self.designer.get_posterior(**conditioning)
        if posterior is None:
            raise RuntimeError(
                "ExperimentDesigner.get_posterior returned None. "
                "Ensure calculateEIG was called."
            )
        return np.asarray(posterior, dtype=np.float64)

    def draw_samples(
        self,
        pdf: Optional[np.ndarray] = None,
        num_samples: int = 50000,
        seed: int = 1,
    ) -> np.ndarray:
        """
        Draw weighted parameter samples from a PDF over the parameter grid.

        Args:
            pdf: PDF array (e.g., posterior). If None, uses posterior from get_posterior().
            num_samples: Number of samples to draw.
            seed: Random seed.

        Returns:
            np.ndarray: Samples with shape (num_samples, n_params).
        """
        if pdf is None:
            pdf = self.get_posterior(nominal=True)

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        posterior = np.asarray(pdf, dtype=np.float64)
        if posterior.size != int(np.prod(self.parameter_grid.shape)):
            raise ValueError(
                f"pdf size mismatch: {posterior.size} vs {int(np.prod(self.parameter_grid.shape))}."
            )

        weights = posterior.reshape(-1).copy()
        weights[~np.isfinite(weights)] = 0.0
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("PDF has no finite positive mass for sampling.")
        weights /= total

        param_axes = [
            np.asarray(getattr(self.parameter_grid, name), dtype=np.float64)
            for name in self.parameter_grid.names
        ]
        broadcast_axes = np.broadcast_arrays(*param_axes)

        rng = np.random.default_rng(seed)
        draw_idx = rng.choice(weights.size, size=int(num_samples), replace=True, p=weights)
        samples = np.column_stack(
            [axis.reshape(-1)[draw_idx] for axis in broadcast_axes]
        ).astype(np.float64, copy=False)
        return samples

    def plot_feature_grid(
        self,
        redshift_range: Optional[Tuple[float, float]] = None,
        n_redshift_pts: int = 200,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ):
        """Plot the feature grid points in feature space.

        For 2D feature grids, plots the grid as a scatter of all (x, y)
        combinations and optionally overlays the redshift track.

        Returns:
            matplotlib Figure and Axes.
        """
        feature_names = list(self.feature_grid.names)
        n_features = len(feature_names)

        axes_vals = [
            np.asarray(self.feature_grid.axes_in[name], dtype=np.float64)
            for name in feature_names
        ]

        if figsize is None:
            figsize = (7, 6)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if n_features == 2:
            xx, yy = np.meshgrid(axes_vals[0], axes_vals[1], indexing="ij")
            ax.scatter(xx.ravel(), yy.ravel(), s=4, alpha=0.6, color="C0", label="grid points")
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
        elif n_features == 1:
            ax.scatter(axes_vals[0], np.zeros_like(axes_vals[0]), s=10, color="C0")
            ax.set_xlabel(feature_names[0])
            ax.set_yticks([])
        else:
            # Higher-D: plot first two dimensions
            xx, yy = np.meshgrid(axes_vals[0], axes_vals[1], indexing="ij")
            ax.scatter(xx.ravel(), yy.ravel(), s=4, alpha=0.6, color="C0")
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])

        # Overlay parameter track
        if self.experiment is not None and n_features >= 2:
            if hasattr(self.experiment, "design_labels"):
                if self.experiment.name == "num_visits":
                    if hasattr(self.experiment, "prior") and "z" in self.experiment.prior:
                        z_prior = self.experiment.prior["z"]
                        if redshift_range is not None:
                            z_min, z_max = redshift_range
                        else:
                            z_min = float(z_prior.icdf(torch.tensor(1e-6, dtype=torch.float64)))
                            z_max = float(z_prior.icdf(torch.tensor(1 - 1e-6, dtype=torch.float64)))
                        p_low = float(z_prior.cdf(torch.tensor(z_min, dtype=torch.float64)))
                        p_high = float(z_prior.cdf(torch.tensor(z_max, dtype=torch.float64)))
                        quantiles = torch.linspace(p_low, p_high, n_redshift_pts, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
                        z_arr = z_prior.icdf(quantiles).clamp(z_min, z_max)
                    else:
                        z_min, z_max = redshift_range or (0.1, 3.0)
                        z_arr = torch.linspace(z_min, z_max, n_redshift_pts, dtype=torch.float64)
                        p_low, p_high = 0.0, 1.0
                    with torch.no_grad():
                        track_features = self.experiment._calculate_magnitudes(z_arr).detach().cpu().numpy()
                else:
                    raise NotImplementedError(f"Feature grid inference not implemented for {self.experiment.name}.")
                filter_to_idx = {f"y_{d}": i for i, d in enumerate(self.experiment.design_labels)}
                ix = filter_to_idx.get(feature_names[0])
                iy = filter_to_idx.get(feature_names[1])
                if ix is not None and iy is not None:
                    ax.plot(track_features[:, ix], track_features[:, iy], color="red",
                            linewidth=1.5, zorder=5, label="z track")
                    z_np = z_arr.numpy()
                    if hasattr(self.experiment, "prior") and "z" in self.experiment.prior:
                        mark_qs = torch.linspace(p_low, p_high, 6, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
                        z_marks = z_prior.icdf(mark_qs).clamp(z_min, z_max).numpy()
                    else:
                        z_marks = np.linspace(z_min, z_max, 6)
                    for z_mark in z_marks:
                        closest = int(np.argmin(np.abs(z_np - z_mark)))
                        ax.plot(track_features[closest, ix], track_features[closest, iy], "o",
                                color="red", markersize=4, zorder=6)
                        ax.annotate(f"z={z_mark:.2f}",
                                    (track_features[closest, ix], track_features[closest, iy]),
                                    textcoords="offset points", xytext=(5, 5),
                                    fontsize=7, color="red", fontweight="bold", zorder=6)
                    ax.legend(fontsize=8)

        ax.set_title("Feature grid points")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig, ax

    def plot_marginal(
        self,
        design_type: str = "nominal",
        design_index: Optional[int] = None,
        label: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        redshift_range: Optional[Tuple[float, float]] = None,
        n_redshift_pts: int = 200,
        param_overlay: bool = False,
        log_scale: bool = True,
    ):
        """
        Plot 2D marginal P(y|xi) for a single design.

        Args:
            design_index: Flat design index into designer.EIG / marginal.
                         If None, chosen by ``which``.
            label: Subplot title. If None, uses ``design_type``.
            figsize: Figure size.
            title: Overall suptitle.
            experiment: Experiment object with design_labels. If provided, overlays a parameter track.
                       design_labels. If provided, overlays a redshift track.
            redshift_range: (z_min, z_max) for the redshift track.
                           Defaults to (0.1, 3.0).
            n_redshift_pts: Number of points along the redshift track.
            design_type: "nominal" (default) or "best". Used when design_index
                  is None to pick the design automatically.
            param_overlay: If True, overlay the parameter track when the marginal is 2D.
            log_scale: If True, plot log10 of the marginal values.

        Returns:
            matplotlib Figure and Axes.
        """
        if self.designer is None:
            raise RuntimeError("Must call run() before plot_marginal.")

        marginal = self.designer.marginal
        feature_names = list(self.feature_grid.names)
        n_features = len(feature_names)

        if design_index is None:
            if design_type == "best":
                design_index = int(np.argmax(self.designer.EIG))
            else:
                design_index = self._nominal_design_index()
        if label is None:
            label = design_type.capitalize()

        # Flatten design dims: (features..., *design_shape) -> (features..., n_designs)
        feature_shape = marginal.shape[:n_features]
        marginal_flat = marginal.reshape(*feature_shape, -1)

        if figsize is None:
            figsize = (7, 6)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        x_vals = np.asarray(self.feature_grid.axes_in[feature_names[0]]).ravel()
        y_vals = np.asarray(self.feature_grid.axes_in[feature_names[1]]).ravel() if n_features > 1 else None

        marg = marginal_flat[..., design_index]

        if log_scale:
            with np.errstate(divide="ignore"):
                marg = np.log(np.maximum(marg, 0.0))
            marg[~np.isfinite(marg)] = np.nan
            colorbar_label = r"$\log{P(y \mid \xi)}$"
            ylabel_1d = r"$\log{P(y \mid \xi)}$"
        else:
            colorbar_label = r"$P(y \mid \xi)$"
            ylabel_1d = r"$P(y \mid \xi)$"

        if n_features == 2 and y_vals is not None:
            im = ax.pcolormesh(x_vals, y_vals, marg.T, shading="gouraud", cmap="viridis")
            fig.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
        else:
            marg_1d = marg.ravel()
            ax.plot(x_vals, marg_1d, linewidth=1.5)
            ax.fill_between(x_vals, marg_1d, alpha=0.3)
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(ylabel_1d)
            if not log_scale:
                ax.set_ylim(0, None)

        # Overlay redshift track if experiment is provided and requested
        if self.experiment.name == "num_visits":
            if (
                param_overlay
                and n_features == 2
                and y_vals is not None
            ):
                if hasattr(self.experiment, "design_labels"):
                    if hasattr(self.experiment, "prior") and "z" in self.experiment.prior:
                        z_prior = self.experiment.prior["z"]
                        if redshift_range is not None:
                            z_min, z_max = redshift_range
                        else:
                            z_min = float(z_prior.icdf(torch.tensor(1e-6, dtype=torch.float64)))
                            z_max = float(z_prior.icdf(torch.tensor(1 - 1e-6, dtype=torch.float64)))
                        p_low = float(z_prior.cdf(torch.tensor(z_min, dtype=torch.float64)))
                        p_high = float(z_prior.cdf(torch.tensor(z_max, dtype=torch.float64)))
                        quantiles = torch.linspace(p_low, p_high, n_redshift_pts, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
                        z_arr = z_prior.icdf(quantiles).clamp(z_min, z_max)
                    else:
                        z_min, z_max = redshift_range or (0.1, 3.0)
                        z_arr = torch.linspace(z_min, z_max, n_redshift_pts, dtype=torch.float64)
                        p_low, p_high = 0.0, 1.0
                    with torch.no_grad():
                        track_features = self.experiment._calculate_magnitudes(z_arr).detach().cpu().numpy()
                    filter_to_idx = {f"y_{d}": i for i, d in enumerate(self.experiment.design_labels)}
                    ix = filter_to_idx.get(feature_names[0])
                    iy = filter_to_idx.get(feature_names[1])
                    if ix is not None and iy is not None:
                        track_x = track_features[:, ix]
                        track_y = track_features[:, iy]
                        ax.plot(track_x, track_y, color="white", linewidth=2, zorder=5)
                        z_np = z_arr.numpy()
                        if hasattr(self.experiment, "prior") and "z" in self.experiment.prior:
                            mark_qs = torch.linspace(p_low, p_high, 6, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
                            z_marks = z_prior.icdf(mark_qs).clamp(z_min, z_max).numpy()
                        else:
                            z_marks = np.linspace(z_min, z_max, 6)
                        for z_mark in z_marks:
                            closest = int(np.argmin(np.abs(z_np - z_mark)))
                            ax.plot(track_x[closest], track_y[closest], "o",
                                    color="black", markersize=4, zorder=6)
                            ax.annotate(f"z={z_mark:.2f}",
                                        (track_x[closest], track_y[closest]),
                                        textcoords="offset points", xytext=(5, 5),
                                        fontsize=7, color="black", fontweight="bold", zorder=6)

        if title is None:
            title = "Marginal P(y|ξ), "
            if self.experiment is not None and hasattr(self.experiment, "temperature"):
                T = self.experiment.temperature
                if hasattr(T, "value"):
                    T = T.value
                title += f"T = {T:.0f} K, "
        fig.suptitle(title + label + " design")
        fig.tight_layout()
        return fig, ax

    def _nominal_design_index(self) -> int:
        """Return the flat design index closest to the experiment's nominal design."""
        nominal_np = self.experiment.nominal_design.detach().cpu().numpy().reshape(1, -1)
        designs_np = self.experiment.designs.detach().cpu().numpy().astype(np.float64)
        return int(np.argmin(np.linalg.norm(
            designs_np - nominal_np[:, :designs_np.shape[1]], axis=1
        )))



def main():
    from bedcosmo.util import init_experiment, get_experiment_config_path, parse_extra_args

    parser = argparse.ArgumentParser(description="Standalone grid-based EIG runner")
    parser.add_argument("cosmo_exp", type=str, help="Experiment type (e.g. num_visits, num_tracers)")
    parser.add_argument(
        "--design-args-path",
        type=str,
        default=None,
        help="Design args config name under experiments/<cosmo_exp>/ (e.g. design_args_2d.yaml)",
    )
    parser.add_argument(
        "--prior-args-path",
        type=str,
        default=None,
        help="Prior args config name under experiments/<cosmo_exp>/ (e.g. prior_args_uniform.yaml)",
    )
    parser.add_argument("--use-experiment-prior", action="store_true", help="Use experiment's prior distributions for PDF")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for experiment calculations")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--param-pts", type=int, default=75, help="Points per parameter axis")
    parser.add_argument("--feature-pts", type=int, default=35, help="Points per feature axis")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--adaptive-features", action="store_true", help="Concentrate feature grid points in high-density regions")
    parser.add_argument("--adaptive-floor", type=float, default=0.05, help="Uniform floor fraction for adaptive feature grid (0=fully adaptive, 1=nearly uniform)")
    parser.add_argument("--feature-range", type=str, action="append", default=[], metavar="NAME:LO,HI",
                        help="Per-feature axis range, e.g. --feature-range u:-10,60 --feature-range g:15,55")
    parser.add_argument("--feature-dense-range", type=str, action="append", default=[], metavar="NAME:LO,HI",
                        help="Per-feature dense sub-region, e.g. --feature-dense-range u:20,30 --feature-dense-range g:22,32")
    parser.add_argument("--param-dense-range", type=str, action="append", default=[], metavar="NAME:LO,HI",
                        help="Per-parameter dense sub-region, e.g. --param-dense-range z:0.5,1.5")
    parser.add_argument("--feature-dense-fraction", type=float, default=0.6,
                        help="Fraction of feature grid points allocated to each dense region (default 0.6)")
    parser.add_argument("--param-dense-fraction", type=float, default=0.6,
                        help="Fraction of parameter grid points allocated to each dense region (default 0.6)")
    args, extra_args = parser.parse_known_args()

    exp_kwargs = parse_extra_args(extra_args)
    if exp_kwargs:
        print(f"Experiment kwargs: {exp_kwargs}")

    if "SCRATCH" not in os.environ:
        raise EnvironmentError("SCRATCH environment variable is required.")

    # If no prior_args_path given, try to find a default from train_args.yaml
    prior_args_path = args.prior_args_path
    if prior_args_path is None:
        try:
            train_args_file = get_experiment_config_path(args.cosmo_exp, "train_args.yaml")
            with open(train_args_file, "r") as f:
                train_args = yaml.safe_load(f)
            # Use the first model config's prior_args_path as default
            first_model = next(iter(train_args.values()))
            if isinstance(first_model, dict):
                prior_args_path = first_model.get("prior_args_path")
                if prior_args_path is not None:
                    print(f"Using default prior_args_path from train_args.yaml: {prior_args_path}")
        except (FileNotFoundError, StopIteration, KeyError):
            pass

    # Initialize experiment via init_experiment
    experiment = init_experiment(
        cosmo_exp=args.cosmo_exp,
        design_args_path=args.design_args_path,
        prior_args_path=prior_args_path,
        device=args.device,
        verbose=args.verbose,
        **exp_kwargs,
    )

    # Build output directory: $SCRATCH/bedcosmo/{exp_name}/grid_calc/{date}
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(os.environ["SCRATCH"]) / "bedcosmo" / args.cosmo_exp / "grid_calc" / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse --feature-range args into dict: {"u": (-10, 60), "g": (15, 55)}
    feature_ranges = {}
    for fr in args.feature_range:
        band, bounds = fr.split(":")
        lo, hi = bounds.split(",")
        feature_ranges[band.strip()] = (float(lo), float(hi))

    # Parse --feature-dense-range args into dict (same syntax as --feature-range)
    feature_dense_ranges = {}
    for dr in args.feature_dense_range:
        band, bounds = dr.split(":")
        lo, hi = bounds.split(",")
        feature_dense_ranges[band.strip()] = (float(lo), float(hi))

    # Parse --param-dense-range args into dict
    param_dense_ranges = {}
    for pr in args.param_dense_range:
        name, bounds = pr.split(":")
        lo, hi = bounds.split(",")
        param_dense_ranges[name.strip()] = (float(lo), float(hi))

    # Save all args (grid_calc + experiment) to args.yaml
    all_args = {**vars(args), "feature_ranges": feature_ranges, "feature_dense_ranges": feature_dense_ranges,
                "param_dense_ranges": param_dense_ranges, **exp_kwargs}
    # Remove non-serializable or redundant fields
    all_args.pop("feature_range", None)
    all_args.pop("feature_dense_range", None)
    all_args.pop("param_dense_range", None)
    with open(out_dir / "args.yaml", "w") as f:
        yaml.dump(all_args, f, default_flow_style=False, sort_keys=False)
    print(f"Saved args to: {out_dir / 'args.yaml'}")

    # Use the class-based interface
    gc = GridCalculation(
        experiment=experiment,
        param_pts=args.param_pts,
        feature_pts=args.feature_pts,
        adaptive_features=args.adaptive_features,
        adaptive_floor=args.adaptive_floor,
        feature_ranges=feature_ranges,
        feature_dense_ranges=feature_dense_ranges,
        param_dense_ranges=param_dense_ranges,
        feature_dense_fraction=args.feature_dense_fraction,
        param_dense_fraction=args.param_dense_fraction,
    )

    # Always save feature grid diagnostic (before run, so it's available on failure)
    fig_fg, _ = gc.plot_feature_grid(
        save_path=str(out_dir / "feature_grid.png"),
    )
    plt.close(fig_fg)
    print(f"  Feature grid plot saved to: {out_dir / 'feature_grid.png'}")

    # Compute prior if requested
    if prior_args_path or args.use_experiment_prior:
        gc.compute_prior_pdf(
            prior_args_path=prior_args_path if not args.use_experiment_prior else None,
            use_experiment_prior=args.use_experiment_prior,
        )
        print(f"Computed prior PDF with shape {gc.prior_pdf.shape}")

    result = gc.run()

    eig = np.asarray(result["eig"], dtype=float)
    eig_flat = eig.reshape(-1)
    best_idx = int(np.argmax(eig_flat))
    payload = {
        "cosmo_exp": args.cosmo_exp,
        "timestamp": datetime.now().isoformat(),
        "best_design": result["best_design"],
        "optimal_eig": float(eig_flat[best_idx]),
        "eigs_avg": eig_flat.tolist(),
        "eigs_std": np.zeros_like(eig_flat).tolist(),
        "design_grid_shape": list(result["design_grid_shape"]),
        "feature_grid_shape": list(result["feature_grid_shape"]),
        "parameter_grid_shape": list(result["parameter_grid_shape"]),
        "param_pts": int(args.param_pts),
        "feature_pts": int(args.feature_pts),
        "used_prior": prior_args_path is not None or args.use_experiment_prior,
    }

    print("Grid-based run complete")
    print(f"  best_design: {payload['best_design']}")
    print(f"  optimal_eig: {payload['optimal_eig']:.6f} bits")
    print(
        f"  grid_shapes: params={payload['parameter_grid_shape']}, "
        f"features={payload['feature_grid_shape']}, designs={payload['design_grid_shape']}"
    )

    # Save eig_data_grid.json
    json_path = out_dir / "eig_data_grid.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {json_path}")

    # Generate plots using refactored BasePlotter
    if not args.no_plots:
        from bedcosmo.plotting import BasePlotter

        print("Generating plots...")
        plotter = BasePlotter(cosmo_exp=args.cosmo_exp)

        # Get the actual constrained designs used in the EIG calculation
        input_designs = experiment.designs.detach().cpu().numpy()

        nominal_design = np.asarray(
            experiment.nominal_design.detach().cpu().numpy(), dtype=np.float64
        ).reshape(-1)

        design_labels = getattr(experiment, 'design_labels', list(gc.design_grid.names))
        plotter.eig_designs(
            eig_values=eig_flat,
            input_designs=input_designs,
            design_labels=design_labels,
            nominal_design=nominal_design,
            title="Grid EIG vs Design",
            save_path=str(out_dir / "eig_designs.png"),
        )

        # Posterior plot
        posterior = gc.get_posterior(nominal=True)
        gc_samples = gc.draw_samples(pdf=posterior, num_samples=50000)

        if experiment.name == "num_visits":
            posterior_title = f"Grid Posterior (nominal), T = {experiment.temperature:.0f} K"

        plotter.generate_posterior(
            experiment=experiment,
            grid_samples=gc_samples,
            title=posterior_title,
            save_path=str(out_dir / "posterior.png"),
            plot_size_ratio=0.8,
            guide_samples=50000,
            plot_prior=True,
        )

        # Marginal P(y|xi) plot for nominal design
        fig_marg, _ = gc.plot_marginal(design_type="nominal")
        fig_marg.savefig(out_dir / "marginal.png", dpi=200, bbox_inches="tight")
        plt.close(fig_marg)

    print(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
