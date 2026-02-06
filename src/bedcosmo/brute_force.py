"""
Generic brute-force EIG helpers using bayesdesign/bed ExperimentDesigner.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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


class BruteForceDesigner:
    """
    Class-based interface for brute-force EIG computation.

    Wraps the bed.design.ExperimentDesigner and provides methods for:
    - Building parameter, feature, and design grids
    - Computing prior PDFs from experiment priors or YAML configuration
    - Running brute-force EIG calculation
    - Extracting posteriors and samples
    """

    def __init__(
        self,
        experiment,
        param_pts: int = 75,
        feature_pts: int = 35,
        device: str = "cpu",
    ):
        """
        Initialize the BruteForceDesigner.

        Args:
            experiment: Experiment object with prior, cosmo_params, etc.
            param_pts: Number of points per parameter axis.
            feature_pts: Number of points per feature axis.
            device: Torch device (brute-force runs on CPU for stability).
        """
        self.experiment = experiment
        self.param_pts = param_pts
        self.feature_pts = feature_pts
        self.device = device

        # Create grids
        self.parameter_grid = self._create_parameter_grid()
        self.feature_grid = self._create_feature_grid()
        self.design_grid = self._create_design_grid()

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
            axes[name] = np.linspace(lo, hi, int(self.param_pts), dtype=np.float64)
        return Grid(**axes)

    def _create_feature_grid(self) -> Grid:
        """Create feature grid (auto for magnitude-based experiments)."""
        experiment = self.experiment
        # Auto mode for magnitude-based experiments (e.g. num_visits).
        if hasattr(experiment, "filters_list") and hasattr(experiment, "_calculate_magnitudes"):
            if not hasattr(self.parameter_grid, "z"):
                raise ValueError("Auto feature grid requires parameter grid to include 'z'.")

            z = np.asarray(getattr(self.parameter_grid, "z"), dtype=np.float64)
            z_tensor = torch.as_tensor(z, device="cpu", dtype=torch.float64)
            with torch.no_grad():
                mags = experiment._calculate_magnitudes(z_tensor).detach().cpu().numpy()
            axes = {}
            for i, band in enumerate(experiment.filters_list):
                lo = float(np.min(mags[..., i]))
                hi = float(np.max(mags[..., i]))
                axes[f"mag_{band}"] = np.linspace(lo, hi, int(self.feature_pts), dtype=np.float64)
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
        Run the brute-force EIG calculation.

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

def main():
    parser = argparse.ArgumentParser(description="Standalone brute-force EIG runner")
    parser.add_argument("cosmo_exp", type=str, help="Experiment type (e.g. num_visits, num_tracers)")
    parser.add_argument("--run_id", type=str, default=None, help="MLflow run ID to load experiment from")
    parser.add_argument(
        "--design_args_path",
        type=str,
        default=None,
        help="Design args config name under experiments/<cosmo_exp>/ (e.g. design_args_2d.yaml)",
    )
    parser.add_argument(
        "--prior_args_path",
        type=str,
        default=None,
        help="Prior args config name under experiments/<cosmo_exp>/ (config mode)",
    )
    parser.add_argument("--use_experiment_prior", action="store_true", help="Use experiment's prior distributions for PDF")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for experiment calculations")
    parser.add_argument("--param_pts", type=int, default=75, help="Points per parameter axis")
    parser.add_argument("--feature_pts", type=int, default=35, help="Points per feature axis")
    parser.add_argument("--save_json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    if "SCRATCH" not in os.environ:
        raise EnvironmentError("SCRATCH environment variable is required.")

    if args.run_id:
        experiment, run_obj = _init_from_run_id(
            run_id=args.run_id,
            cosmo_exp=args.cosmo_exp,
            device=args.device,
            design_args_path=args.design_args_path,
        )
        artifacts_path = Path(run_obj.info.artifact_uri.replace("file://", ""))
    else:
        experiment = _init_from_configs(
            cosmo_exp=args.cosmo_exp,
            device=args.device,
            design_args_path=args.design_args_path,
            prior_args_path=args.prior_args_path,
        )
        artifacts_path = None

    # Use the class-based interface
    bf = BruteForceDesigner(
        experiment=experiment,
        param_pts=args.param_pts,
        feature_pts=args.feature_pts,
    )

    # Compute prior if requested
    if args.prior_args_path or args.use_experiment_prior:
        bf.compute_prior_pdf(
            prior_args_path=args.prior_args_path,
            use_experiment_prior=args.use_experiment_prior,
        )
        print(f"Computed prior PDF with shape {bf.prior_pdf.shape}")

    result = bf.run()

    eig = np.asarray(result["eig"], dtype=float)
    eig_flat = eig.reshape(-1)
    best_idx = int(np.argmax(eig_flat))
    payload = {
        "cosmo_exp": args.cosmo_exp,
        "run_id": args.run_id,
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
        "used_prior": args.prior_args_path is not None or args.use_experiment_prior,
    }

    print("Brute-force run complete")
    print(f"  best_design: {payload['best_design']}")
    print(f"  optimal_eig: {payload['optimal_eig']:.6f} bits")
    print(
        f"  grid_shapes: params={payload['parameter_grid_shape']}, "
        f"features={payload['feature_grid_shape']}, designs={payload['design_grid_shape']}"
    )

    if args.save_json is not None:
        out_path = Path(args.save_json)
    elif artifacts_path is not None:
        out_path = artifacts_path / f"brute_force_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        out_path = Path.cwd() / f"brute_force_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
