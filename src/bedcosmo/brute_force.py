"""
Generic brute-force EIG helpers using bayesdesign/bed ExperimentDesigner.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from bed.design import ExperimentDesigner
from bed.grid import Grid

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


def create_parameter_grid(
    experiment,
    param_pts: int = 75,
) -> Any:
    """
    Create a parameter Grid.

    Derives bounds from experiment.prior.
    """
    axes: Dict[str, np.ndarray] = {}
    names = list(getattr(experiment, "cosmo_params", []))
    if not names:
        raise ValueError("Experiment must expose cosmo_params.")

    prior = getattr(experiment, "prior", {})
    for name in names:
        if name not in prior:
            raise ValueError(f"Prior missing parameter '{name}'.")
        lo, hi = _dist_bounds(prior[name])
        axes[name] = np.linspace(lo, hi, int(param_pts), dtype=np.float64)
    return Grid(**axes)


def create_feature_grid(
    experiment,
    feature_pts: int = 35,
    parameter_grid: Any = None,
) -> Any:
    """
    Create a feature Grid.

    Auto-builds for experiments exposing filters_list + _calculate_magnitudes.
    """
    # Auto mode for magnitude-based experiments (e.g. num_visits).
    if hasattr(experiment, "filters_list") and hasattr(experiment, "_calculate_magnitudes"):
        if parameter_grid is None:
            raise ValueError("parameter_grid is required for auto feature grid construction.")
        if not hasattr(parameter_grid, "z"):
            raise ValueError("Auto feature grid requires parameter grid to include 'z'.")

        z = np.asarray(getattr(parameter_grid, "z"), dtype=np.float64)
        z_tensor = torch.as_tensor(z, device="cpu", dtype=torch.float64)
        with torch.no_grad():
            mags = experiment._calculate_magnitudes(z_tensor).detach().cpu().numpy()
        axes = {}
        for i, band in enumerate(experiment.filters_list):
            lo = float(np.min(mags[..., i]))
            hi = float(np.max(mags[..., i]))
            axes[f"mag_{band}"] = np.linspace(lo, hi, int(feature_pts), dtype=np.float64)
        return Grid(**axes)

    raise ValueError(
        "Cannot auto-build feature grid for this experiment; provide feature_axes explicitly."
    )


def create_design_grid(experiment) -> Any:
    """
    Get design Grid from experiment.
    """
    if not hasattr(experiment, "designs_grid") or experiment.designs_grid is None:
        raise ValueError(
            "Experiment does not expose designs_grid. "
            "Define it in experiment.init_designs (e.g., via BaseExperiment._build_design_grid)."
        )
    return experiment.designs_grid


def create_uniform_prior(parameter_grid) -> np.ndarray:
    prior = np.ones(parameter_grid.shape, dtype=np.float64)
    norm = parameter_grid.sum(prior)
    if norm <= 0:
        raise ValueError("Prior normalization failed: non-positive norm.")
    prior /= norm
    return prior


def run_experiment_designer(
    experiment,
    parameter_grid,
    feature_grid,
    design_grid,
    prior: Optional[np.ndarray] = None,
    lfunc_name: str = "unnorm_lfunc",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run brute-force EIG with ExperimentDesigner.
    """
    if not hasattr(experiment, lfunc_name):
        raise ValueError(f"Experiment missing likelihood function '{lfunc_name}'.")
    lfunc = getattr(experiment, lfunc_name)

    if prior is None:
        prior = create_uniform_prior(parameter_grid)

    designer = ExperimentDesigner(parameter_grid, feature_grid, design_grid, lfunc)
    best_design = designer.calculateEIG(prior, debug=debug)
    eig = np.asarray(designer.EIG, dtype=np.float64)
    return {
        "best_design": best_design,
        "eig": eig,
        "prior": prior,
    }


def brute_force_from_experiment(
    experiment,
    param_pts: int = 75,
    feature_pts: int = 35,
    prior: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper to build grids and run brute-force EIG.
    """
    params_grid = create_parameter_grid(
        experiment=experiment,
        param_pts=param_pts,
    )
    features_grid = create_feature_grid(
        experiment=experiment,
        feature_pts=feature_pts,
        parameter_grid=params_grid,
    )
    designs_grid = create_design_grid(experiment)
    result = run_experiment_designer(
        experiment=experiment,
        parameter_grid=params_grid,
        feature_grid=features_grid,
        design_grid=designs_grid,
        prior=prior,
    )
    result["parameter_grid_shape"] = tuple(params_grid.shape)
    result["feature_grid_shape"] = tuple(features_grid.shape)
    result["design_grid_shape"] = tuple(designs_grid.shape)
    return result


def _init_from_run_id(run_id: str, cosmo_exp: str, device: str, design_args_path: str | None):
    import yaml

    from bedcosmo.util import get_experiment_config_path, get_runs_data, init_experiment

    run_data_list, _, _ = get_runs_data(run_ids=run_id, cosmo_exp=cosmo_exp)
    if not run_data_list:
        raise ValueError(f"Run {run_id} not found for cosmo_exp={cosmo_exp}.")
    run_data = run_data_list[0]
    run_obj = run_data["run_obj"]
    run_args = run_data["params"]

    design_args = None
    if design_args_path is not None:
        resolved_path = get_experiment_config_path(cosmo_exp, design_args_path)
        with open(resolved_path, "r") as f:
            design_args = yaml.safe_load(f)

    experiment = init_experiment(
        run_obj=run_obj,
        run_args=run_args,
        device=device,
        design_args=design_args,
        global_rank=0,
    )
    return experiment, run_obj


def _init_from_configs(
    cosmo_exp: str,
    device: str,
    design_args_path: str | None,
    prior_args_path: str | None,
):
    from bedcosmo.util import init_experiment

    return init_experiment(
        cosmo_exp=cosmo_exp,
        device=device,
        design_args_path=design_args_path,
        prior_args_path=prior_args_path,
        global_rank=0,
    )


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

    result = brute_force_from_experiment(
        experiment=experiment,
        param_pts=args.param_pts,
        feature_pts=args.feature_pts,
    )

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
