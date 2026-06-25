import sys
import os
import torch
import pyro
import mlflow
import zuko
import random
import numpy as np
from mlflow.tracking import MlflowClient
from pyro import distributions as dist
import json
import contextlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pyro.contrib.util import lexpand
import subprocess
from datetime import datetime, timedelta
import traceback
import functools
import time
import inspect
import concurrent.futures
import warnings
from itertools import combinations

# GetDist KDE settings used for all MCSamples constructions. 2D smoothing
# reduces contour pixel noise while preserving broad structure.
GETDIST_SETTINGS = {
    "smooth_scale_2D": 0.2,
    "fine_bins_2D": 256,
    "mult_bias_correction_order": 1,
    "boundary_correction_order": 1,
}

import getdist
import io


def restrict_mcsamples(gd, params):
    """Return a new MCSamples restricted to ``params`` (a list of names).

    Used to build marginal triangle plots so every plotted distribution
    contains only the requested subset of parameters.
    """
    if gd is None or params is None:
        return gd
    names = [p.name for p in gd.paramNames.names]
    idx = [names.index(p) for p in params if p in names]
    if not idx:
        return gd
    sub_labels = [gd.paramNames.names[i].label for i in idx]
    with contextlib.redirect_stdout(io.StringIO()):
        restricted = getdist.MCSamples(
            samples=gd.samples[:, idx],
            names=[names[i] for i in idx],
            labels=sub_labels,
            settings=GETDIST_SETTINGS,
        )
    return restricted


from PIL import Image, ImageDraw, ImageFont
import glob
import argparse
import pandas as pd
import yaml

torch.set_default_dtype(torch.float64)

from pathlib import Path

from bedcosmo.transform import Bijector  # public re-export (implementation in transform.py)


def get_experiments_dir() -> Path:
    """Get path to experiments directory."""
    if "BED_COSMO_EXPERIMENTS" in os.environ:
        return Path(os.environ["BED_COSMO_EXPERIMENTS"])

    # Development mode: relative to package
    pkg_dir = Path(__file__).parent
    dev_experiments = pkg_dir.parent.parent / "experiments"
    if dev_experiments.exists():
        return dev_experiments

    raise FileNotFoundError("Could not find experiments directory")


def get_experiment_config_path(cosmo_exp: str, config_name: str) -> Path:
    """Get full path to experiment config file."""
    return get_experiments_dir() / cosmo_exp / config_name


def extract_run_info_from_checkpoint_path(checkpoint_path: str) -> tuple[str, str, str]:
    """
    Extract run_id, experiment_id, and cosmo_exp from a checkpoint path.

    Expected path: .../bedcosmo/{cosmo_exp}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/...

    Returns: (run_id, experiment_id, cosmo_exp)
    """
    parts = checkpoint_path.split("/")
    try:
        mlruns_idx = parts.index("mlruns")
    except ValueError:
        raise ValueError(
            f"Could not find 'mlruns' in checkpoint path: {checkpoint_path}\n"
            f"Expected: $SCRATCH/bedcosmo/{{cosmo_exp}}/mlruns/{{exp_id}}/{{run_id}}/artifacts/checkpoints/..."
        )

    cosmo_exp = parts[mlruns_idx - 1]
    exp_id = parts[mlruns_idx + 1]
    run_id = parts[mlruns_idx + 2]

    return run_id, exp_id, cosmo_exp


def auto_seed(base_seed=0, rank=0):
    if base_seed < 0:
        base_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    global_seed = base_seed + rank
    # Set all relevant seeds
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)
        torch.cuda.manual_seed_all(global_seed)
        # For completely deterministic results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    pyro.set_rng_seed(global_seed)
    return global_seed

@contextlib.contextmanager
def temporary_seed(seed=None):
    """
    Context manager to temporarily set random seeds and restore previous state.
    
    Args:
        seed (int): The seed to use temporarily
    """
    # Save current RNG states
    old_pyro_state = pyro.util.get_rng_state() # includes torch, random, and numpy states
    old_cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    
    # Set the seed temporarily
    if seed is not None:
        auto_seed(seed)
    try:
        yield
    finally:  # Restore previous RNG states even if try block fails
        pyro.util.set_rng_state(old_pyro_state)
        if old_cuda_states is not None:
            torch.cuda.set_rng_state_all(old_cuda_states)

def print_memory_usage(process, step):
    mem_info = process.memory_info()
    print(f"Step {step}: Memory Usage: {mem_info.rss / 1024**2:.2f} MB")

def save_checkpoint(model, optimizer, filepath, step=None, artifact_path=None, scheduler=None, additional_state=None, global_rank=None):
    """
    Saves the training checkpoint with comprehensive state information.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        filepath (str): Path to save the checkpoint.
        step (int, optional): Current training step.
        artifact_path (str, optional): Path to log the artifact to in MLflow.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        additional_state (dict, optional): Additional state to save (e.g., training metrics, configuration).
        global_rank (int, optional): Global rank in distributed training.
    """
    # Get the state dict from the model, handling DDP wrapper if present
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_type': optimizer.__class__.__name__
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if step is not None:
        checkpoint['step'] = step

    # Save RNG states correctly
    checkpoint['rng_state'] = get_rng_state()
    
    # Also save Pyro's param store state (in case there are any parameters)
    checkpoint['rng_state']['pyro_param_state'] = pyro.get_param_store().get_state()
    
    # Always include global rank information
    if global_rank is not None:
        if additional_state is None:
            additional_state = {}
        additional_state['global_rank'] = global_rank
    
    # Include additional state if provided
    if additional_state is not None:
        checkpoint['additional_state'] = additional_state
    
    # Log the checkpoint to mlflow
    torch.save(checkpoint, filepath)
    if artifact_path:
        mlflow.log_artifact(filepath, artifact_path)

def get_rng_state():
    return {
        'pyro': pyro.util.get_rng_state(),  # This includes torch, random, and numpy states
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

def restore_rng_state(checkpoint, global_rank):
# Restore RNG state at the beginning of each step if resuming from checkpoint

# Handle Pyro's RNG state restoration (includes torch, random, and numpy)
    try:
        rng_state = checkpoint['rng_state']
        pyro_state = rng_state['pyro']
        if pyro_state is None:
            if global_rank == 0:
                print("Warning: Pyro RNG state is None, skipping restoration")
            return
        
        # Ensure the torch state in Pyro's state has the correct dtype and is on CPU
        if isinstance(pyro_state, dict) and 'torch' in pyro_state:
            torch_state = pyro_state['torch']
            if isinstance(torch_state, torch.Tensor):
                # Move to CPU first, then ensure correct dtype
                torch_state = torch_state.cpu()
                if torch_state.dtype != torch.uint8:
                    # Convert to uint8 if it's not already
                    torch_state = torch_state.to(torch.uint8)
            elif isinstance(torch_state, (list, np.ndarray)):
                # Convert from list/array to uint8 tensor on CPU
                torch_state = torch.tensor(torch_state, dtype=torch.uint8, device='cpu')
            else:
                # If we can't convert, skip Pyro RNG restoration
                raise ValueError(f"Cannot convert torch state of type {type(torch_state)} to uint8 tensor")
            pyro_state['torch'] = torch_state
        
        pyro.util.set_rng_state(pyro_state)  # Pyro's RNG state for deterministic sampling
        
        # Restore CUDA RNG states separately (not handled by Pyro)
        if torch.cuda.is_available() and 'cuda' in rng_state and rng_state['cuda'] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng_state['cuda']])
        
        # Also restore Pyro's param store state if it exists in the checkpoint
        if 'pyro_param_state' in rng_state:
            pyro.get_param_store().set_state(rng_state['pyro_param_state'])
        
    except Exception as e:
        if global_rank == 0:
            print(f"Warning: Could not restore RNG state: {e}. Continuing with current state.")
            traceback.print_exc()

def init_nf(
        run_args, 
        input_dim, 
        context_dim,
        device="cuda:0", 
        seed=None, 
        **kwargs
        ):
    """
    Initialize the flow model.

    Returns:
        posterior_flow: zuko.flows.Flow
    """
    with temporary_seed(seed):
        flow_type = run_args.get("flow_type")
        n_transforms = int(run_args.get("n_transforms", 2))
        cond_n_layers = int(run_args.get("cond_n_layers", 2))
        cond_hidden_size = int(run_args.get("cond_hidden_size", 64))
        activation_type = run_args.get("activation", "ReLU").lower()
        if activation_type == "relu":
            activation = torch.nn.ReLU # Rectified Linear Unit
        elif activation_type == "elu":
            activation = torch.nn.ELU # Exponential Linear Unit
        elif activation_type == "sigmoid":
            activation = torch.nn.Sigmoid # Sigmoid
        elif activation_type == "tanh":
            activation = torch.nn.Tanh # Hyperbolic Tangent
        elif activation_type == "softplus":
            activation = torch.nn.Softplus # Softplus
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")

        # Initialize the flow model
        if flow_type == "NSF":
            posterior_flow = zuko.flows.NSF(
                features=input_dim, 
                context=context_dim, 
                transforms=n_transforms,
                bins=int(run_args.get("spline_bins")),
                randperm=True,
                hidden_features=((cond_hidden_size,) * cond_n_layers),
                activation=activation,
                **kwargs
            )
        elif flow_type == "NAF":
            posterior_flow = zuko.flows.NAF(
                features=input_dim, 
                context=context_dim, 
                transforms=n_transforms,
                signal=int(run_args.get("mnn_signal")),
                randperm=True,
                network={
                    "hidden_features": ((int(run_args.get("mnn_hidden_size", 64)),) * int(run_args.get("mnn_n_layers", 2))),
                    "activation": activation
                    },
                hidden_features=((cond_hidden_size,) * cond_n_layers), # for the conditional network
                activation=activation,
                )
        elif flow_type == "MAF":
            transform = run_args.get("nf_transform", None)
            if transform == "affine":
                univariate = zuko.transforms.MonotonicAffineTransform
            elif transform == "rqs":
                univariate = zuko.transforms.MonotonicRQSTransform
            else:
                raise ValueError(f"Unknown transform: {transform}")
            posterior_flow = zuko.flows.MAF(
                features=input_dim, 
                context=context_dim, 
                transforms=n_transforms,
                randperm=True,
                univariate=univariate,
                hidden_features=((cond_hidden_size,) * cond_n_layers),
                activation=activation,
                **kwargs
                )
        elif flow_type == "NICE":
            posterior_flow = zuko.flows.NICE(
                features=input_dim, 
                context=context_dim, 
                transforms=n_transforms,
                activation=activation,
                **kwargs
                )
        elif flow_type == "GF":
            posterior_flow = zuko.flows.GF(
                features=input_dim, 
                context=context_dim, 
                transforms=n_transforms,
                hidden_features=((cond_hidden_size,) * cond_n_layers),
                activation=activation,
                )
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

        # Move to the correct device
        posterior_flow.to(device)

        with torch.no_grad():
            # 1) Generic linear layers: Use He initialization for ReLU networks
            for module in posterior_flow.modules():
                cls = module.__class__.__name__
                if isinstance(module, torch.nn.Linear) or cls.endswith("Linear"):
                    if activation_type == "relu":
                        torch.nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
                    else:
                        torch.nn.init.xavier_uniform_(module.weight, gain=0.8)

                    if getattr(module, "bias", None) is not None:
                        torch.nn.init.zeros_(module.bias)

            # 2) Initialize hypernetworks to produce small but non-zero transform parameters
            for name, sub in posterior_flow.named_modules():
                lname = name.lower()
                if any(k in lname for k in ("condition", "hyper", "context")):
                    last_linear = None
                    for child in sub.modules():
                        if isinstance(child, torch.nn.Linear):
                            last_linear = child
                    if last_linear is not None:
                        try:
                            torch.nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
                            if getattr(last_linear, "bias", None) is not None:
                                torch.nn.init.zeros_(last_linear.bias)
                        except Exception:
                            pass

            # 3) NAF-specific: keep monotonic layers small but not vanishing.
            if flow_type == "NAF":
                for module in posterior_flow.modules():
                    cls = module.__class__.__name__
                    if "Monotonic" in cls and hasattr(module, "weight"):
                        # Use larger gain to break identity behavior and produce more diverse samples
                        torch.nn.init.xavier_uniform_(module.weight, gain=1.0)  # Increased from 0.5 to 1.0
                        if getattr(module, "bias", None) is not None:
                            torch.nn.init.normal_(module.bias, mean=0.0, std=0.1)  # Add small random bias

                # If the base distribution exposes a scale parameter, set it appropriately
                # for unconstrained parameter space.
                try:
                    if hasattr(posterior_flow, "base") and hasattr(posterior_flow.base, "scale"):
                        # Use larger scale to ensure samples cover the full parameter space
                        # when transformed back to physical space
                        posterior_flow.base.scale.data.fill_(2.0)  # Increased from 1.0 to 2.0
                except Exception:
                    pass
                
                # 4) Additional NAF-specific improvements to prevent plateaus
                # Initialize the final layers of univariate networks to be close to identity
                # but with small random perturbations to break symmetry
                for name, module in posterior_flow.named_modules():
                    if "univariate" in name.lower() and hasattr(module, "network"):
                        # Find the last linear layer in the univariate network
                        last_layer = None
                        for child in module.network:
                            if isinstance(child, torch.nn.Linear) and child.out_features == 1:
                                last_layer = child
                                break
                        
                        if last_layer is not None:
                            # Initialize to larger values to break identity behavior
                            # This helps the NAF produce more diverse samples initially
                            torch.nn.init.normal_(last_layer.weight, mean=0.0, std=0.2)  # Increased from 0.05 to 0.2
                            if getattr(last_layer, "bias", None) is not None:
                                torch.nn.init.normal_(last_layer.bias, mean=0.0, std=0.1)  # Add small bias instead of zero
    return posterior_flow


# Keys stored in checkpoints so :func:`init_nf` can rebuild the architecture without MLflow.
NF_INIT_CONFIG_KEY = "nf_init_config"


def build_nf_init_config_from_run_args(run_args: dict) -> dict:
    """Snapshot of parameters read by :func:`init_nf` for self-contained checkpoints."""
    if not run_args:
        raise ValueError("run_args is required to build nf_init_config")
    cfg = {
        "flow_type": run_args.get("flow_type"),
        "n_transforms": int(run_args.get("n_transforms", 2)),
        "cond_n_layers": int(run_args.get("cond_n_layers", 2)),
        "cond_hidden_size": int(run_args.get("cond_hidden_size", 64)),
        "activation": str(run_args.get("activation", "ReLU")),
        "spline_bins": run_args.get("spline_bins"),
        "mnn_signal": run_args.get("mnn_signal"),
        "mnn_hidden_size": run_args.get("mnn_hidden_size"),
        "mnn_n_layers": run_args.get("mnn_n_layers"),
        "nf_transform": run_args.get("nf_transform"),
        "transform_input": run_args.get("transform_input", False),
        "input_transform_type": run_args.get("input_transform_type", "marginal"),
    }
    if cfg["spline_bins"] is not None:
        cfg["spline_bins"] = int(cfg["spline_bins"])
    for k in ("mnn_signal", "mnn_hidden_size", "mnn_n_layers"):
        if cfg[k] is not None:
            cfg[k] = int(cfg[k])
    return cfg


def create_scheduler(optimizer, run_args):
    # Setup
    steps_per_cycle = run_args["total_steps"] // run_args["n_cycles"]
    initial_lr = run_args["initial_lr"]
    final_lr = run_args["final_lr"]
    
    # Get warmup fraction, defaulting to 0.0 (no warmup)
    warmup_fraction = run_args.get("warmup_fraction", 0.0)
    warmup_steps = int(warmup_fraction * run_args["total_steps"])

    if run_args["scheduler_type"] == "constant":
        if warmup_steps > 0:
            # Create a warmup + constant schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to initial_lr
                    return step / warmup_steps
                else:
                    # Constant at initial_lr
                    return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, 
                factor=1.0
                )
    elif run_args["scheduler_type"] == "cosine":
        if warmup_steps > 0:
            # Create a warmup + cosine schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to initial_lr
                    return step / warmup_steps
                else:
                    # Cosine decay from initial_lr to final_lr
                    adjusted_step = step - warmup_steps
                    adjusted_total_steps = run_args["total_steps"] - warmup_steps
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * adjusted_step / adjusted_total_steps))
                    return (final_lr + (initial_lr - final_lr) * cosine_factor) / initial_lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=run_args["total_steps"],
                eta_min=final_lr
                )
    elif run_args["scheduler_type"] == "linear":
        if warmup_steps > 0:
            # Create a warmup + linear schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to initial_lr
                    return step / warmup_steps
                else:
                    # Linear decay from initial_lr to final_lr
                    adjusted_step = step - warmup_steps
                    adjusted_total_steps = run_args["total_steps"] - warmup_steps
                    if adjusted_total_steps <= 1:
                        return final_lr / initial_lr
                    
                    progress = adjusted_step / (adjusted_total_steps - 1)
                    end_factor = final_lr / initial_lr
                    return 1.0 + (end_factor - 1.0) * progress
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # This provides a linear ramp from initial_lr to final_lr.
            # It can handle both increasing and decreasing LR by using LambdaLR.
            def lr_lambda(step):
                total_steps = run_args["total_steps"]
                if initial_lr == 0:
                    if final_lr != 0:
                        raise ValueError("Cannot use 'linear' scheduler for warmup from initial_lr=0, as the optimizer's base LR is 0.")
                    return 1.0  # LR is 0 and stays 0.
                
                if total_steps <= 1:
                    return final_lr / initial_lr
                
                progress = step / (total_steps - 1)
                end_factor = final_lr / initial_lr
                
                # Linear interpolation of the multiplicative factor
                return 1.0 + (end_factor - 1.0) * progress
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif run_args["scheduler_type"] == "exponential":
        if warmup_steps > 0:
            # Create a warmup + exponential schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to initial_lr
                    return step / warmup_steps
                else:
                    # Exponential decay from initial_lr to final_lr
                    adjusted_step = step - warmup_steps
                    adjusted_total_steps = run_args["total_steps"] - warmup_steps
                    gamma = (final_lr / initial_lr) ** (1 / adjusted_total_steps)
                    return (initial_lr * (gamma ** adjusted_step)) / initial_lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # calculate gamma from initial and final lr
            gamma = (final_lr / initial_lr) ** (1 / run_args["total_steps"])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=gamma
                )
    elif run_args["scheduler_type"] == "lambda":
        # Get gamma from run_args for lambda scheduler
        gamma = run_args.get("gamma", 0.1)
        if warmup_steps > 0:
            # Create a warmup + lambda schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to initial_lr
                    return step / warmup_steps
                else:
                    # Original lambda schedule logic
                    adjusted_step = step - warmup_steps
                    adjusted_total_steps = run_args["total_steps"] - warmup_steps
                    steps_per_cycle = adjusted_total_steps // run_args["n_cycles"]
                    cycle = adjusted_step // steps_per_cycle
                    cycle_progress = (adjusted_step % steps_per_cycle) / steps_per_cycle
                    # Decaying peak
                    peak = initial_lr * (gamma ** cycle)
                    # Cosine decay within cycle
                    cosine = 0.5 * (1 + np.cos(np.pi * cycle_progress))
                    lr = final_lr + (peak - final_lr) * cosine
                    return lr / initial_lr  # LambdaLR expects a multiplier of the initial LR
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            def lr_lambda(step):
                cycle = step // steps_per_cycle
                cycle_progress = (step % steps_per_cycle) / steps_per_cycle
                # Decaying peak
                peak = initial_lr * (gamma ** cycle)
                # Cosine decay within cycle
                cosine = 0.5 * (1 + np.cos(np.pi * cycle_progress))
                lr = final_lr + (peak - final_lr) * cosine
                return lr / initial_lr  # LambdaLR expects a multiplier of the initial LR
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, 
                lr_lambda
                )
    else:
        raise ValueError(f"Unknown scheduler_type: {run_args['scheduler_type']}")
    
    return scheduler

def get_runs_data(mlflow_exp=None, run_ids=None, excluded_runs=[], filter_string=None, parse_params=True, cosmo_exp='num_tracers'):
    """
    Fetches run data from MLflow based on experiment name or run IDs.

    Args:
        mlflow_exp (str, optional): Name of the MLflow experiment.
        run_ids (str or list, optional): A single run ID or a list of run IDs.
        excluded_runs (list, optional): A list of run IDs to exclude from the result.
        filter_string (str, optional): An MLflow filter string to apply when searching for runs.
        parse_params (bool): Whether to parse MLflow parameters using `parse_mlflow_params`.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries, where each dictionary represents a run and contains
                    'run_id', 'params', 'run_obj', 'name', and 'exp_id'. Returns empty list on failure.
            - str or None: The experiment ID for the fetched runs.
            - str or None: The experiment name for the fetched runs.
    """
    mlflow.set_tracking_uri(os.environ["SCRATCH"] + f"/bedcosmo/{cosmo_exp}" + "/mlruns")
    client = MlflowClient()
    run_data_list = []
    experiment_id = None
    actual_mlflow_exp = mlflow_exp

    if mlflow_exp is not None:
        try:
            exp = client.get_experiment_by_name(mlflow_exp)
            if exp is None:
                print(f"Error: Experiment '{mlflow_exp}' not found.")
                return [], None, None
            experiment_id = exp.experiment_id
            actual_mlflow_exp = exp.name
            
            run_infos = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string if filter_string else "",
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
            )
            
            for run_obj in run_infos:
                if run_obj.info.run_id not in excluded_runs:
                    params = parse_mlflow_params(run_obj.data.params) if parse_params else run_obj.data.params
                    run_data_list.append({
                        'run_id': run_obj.info.run_id,
                        'params': params,
                        'run_obj': run_obj,
                        'name': run_obj.info.run_name,
                        'exp_id': run_obj.info.experiment_id
                    })
        except Exception as e:
            print(f"Error fetching runs for experiment '{mlflow_exp}': {e}")
            return [], None, None
            
    elif run_ids is not None:
        run_ids_list = [run_ids] if isinstance(run_ids, str) else run_ids
        processed_run_ids = [rid for rid in run_ids_list if rid not in excluded_runs]
        
        if not processed_run_ids:
            print("No valid runs to process after exclusion.")
            return [], None, None

        for rid in processed_run_ids:
            try:
                run_obj = client.get_run(rid)
                if experiment_id is None:
                    experiment_id = run_obj.info.experiment_id
                    if actual_mlflow_exp is None:
                        try:
                            exp = client.get_experiment(experiment_id)
                            actual_mlflow_exp = exp.name if exp else "Selected Runs"
                        except Exception:
                            actual_mlflow_exp = "Selected Runs (from run_id)"

                params = parse_mlflow_params(run_obj.data.params) if parse_params else run_obj.data.params
                run_data_list.append({
                    'run_id': rid,
                    'params': params,
                    'run_obj': run_obj,
                    'name': run_obj.info.run_name,
                    'exp_id': run_obj.info.experiment_id
                })
            except Exception as e:
                print(f"Warning: Could not fetch data for run {rid}: {e}. Skipping this run.")
    else:
        print("Either mlflow_exp or run_ids must be provided.")
        return [], None, None

    if not run_data_list:
        print("No runs found to process.")
        return [], None, None

    return run_data_list, experiment_id, actual_mlflow_exp


def init_experiment(
        run_obj=None,
        run_args=None,
        cosmo_exp=None,
        checkpoint=None,
        global_rank=0,
        design_args=None,
        design_args_path=None,
        prior_args=None,
        prior_args_path=None,
        **kwargs
        ):
    """
    Initializes the experiment class with the run arguments.
    Args:
        run_obj: The MLflow Run object.
        run_args (dict): The run arguments.
        cosmo_exp (str): The cosmology experiment name.
        checkpoint (dict): Checkpoint dictionary.
        global_rank (int): Global rank for distributed training.
        design_args (dict): The design arguments. If None, will try to load from:
            1. Artifacts (design_args.yaml) from the run_obj
            2. File if design_args_path is specified in run_args
        prior_args (dict): The prior arguments. If None, will try to load from:
            Artifacts (prior_args.yaml) from the run_obj
        **kwargs: Additional arguments (e.g., device, profile) that will be added to run_args.
    """
    emulator_artifacts_dir = None
    empirical_artifacts_dir = None
    if run_obj is not None and run_args is not None:

        cosmo_exp = run_args.get("cosmo_exp")

        artifact_uri = run_obj.info.artifact_uri
        if artifact_uri.startswith("file://"):
            artifact_path = artifact_uri[7:]  # Remove "file://" prefix
        else:
            artifact_path = artifact_uri
        # Emulator checkpoints snapshotted into the run's artifacts at submission time
        emu_dir = os.path.join(artifact_path, "emulators")
        if os.path.isdir(emu_dir):
            emulator_artifacts_dir = emu_dir
        empirical_dir = os.path.join(artifact_path, "empirical")
        if os.path.isdir(empirical_dir):
            empirical_artifacts_dir = empirical_dir
        # If design_args is not provided (None), try to load from artifacts first
        if design_args is None:
            design_args_artifact_path = artifact_path + "/design_args.yaml"
            if os.path.exists(design_args_artifact_path):
                with open(design_args_artifact_path, 'r') as f:
                    design_args = yaml.safe_load(f)
            else:
                # Fall back to loading from file if design_args_path is specified
                design_args_path = run_args.get("design_args_path", None)
                if design_args_path is not None:
                    if not os.path.exists(design_args_path):
                        raise FileNotFoundError(f"Design args file not found: {design_args_path}")
                    with open(design_args_path, 'r') as f:
                        design_args = yaml.safe_load(f)
        
        # If prior_args is not provided (None), try to load from artifacts
        if prior_args is None:
            prior_artifact_path = artifact_path + "/prior_args.yaml"
            if os.path.exists(prior_artifact_path):
                with open(prior_artifact_path, 'r') as f:
                    prior_args = yaml.safe_load(f)
            else:
                if global_rank == 0:
                    print(f"Warning: prior_args.yaml not found in artifacts at {prior_artifact_path}")

    else:
        if cosmo_exp is None:
            raise ValueError("cosmo_exp must be specified")
        run_args = {}
        if design_args is None:
            if design_args_path is not None:
                resolved_path = get_experiment_config_path(cosmo_exp, design_args_path)
                if not resolved_path.exists():
                    raise FileNotFoundError(f"Design args file not found: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    design_args = yaml.safe_load(f)

        if prior_args is None:
            if prior_args_path is not None:
                resolved_path = get_experiment_config_path(cosmo_exp, prior_args_path)
                if not resolved_path.exists():
                    raise FileNotFoundError(f"Prior args file not found: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    prior_args = yaml.safe_load(f)
    
    # Initialize run_args if None
    if run_args is None:
        run_args = {}
    
    # Add design_args and prior_args
    run_args['design_args'] = design_args
    run_args['prior_args'] = prior_args
    
    # Merge kwargs into run_args
    run_args.update(kwargs)
    run_args = apply_central_param_cli_flags(run_args)

    if cosmo_exp == 'num_tracers':
        from bedcosmo.num_tracers import NumTracers
        valid_params = inspect.signature(NumTracers.__init__).parameters.keys()
        valid_params = [k for k in valid_params if k != 'self']
        
        # Filter run_args to only include valid NumTracers parameters
        exp_args = {k: v for k, v in run_args.items() if k in valid_params}
        exp_args['global_rank'] = global_rank
        if checkpoint is not None and 'bijector_state' in checkpoint.keys():
            exp_args['bijector_state'] = checkpoint['bijector_state']
        if emulator_artifacts_dir is not None:
            exp_args['emulator_artifacts_dir'] = emulator_artifacts_dir
        experiment = NumTracers(**exp_args)
    
    elif cosmo_exp == 'variable_redshift':
        from bedcosmo.variable_redshift import VariableRedshift
        valid_params = inspect.signature(VariableRedshift.__init__).parameters.keys()
        valid_params = [k for k in valid_params if k != 'self']
        
        # Filter run_args to only include valid VariableRedshift parameters
        exp_args = {k: v for k, v in run_args.items() if k in valid_params}
        exp_args['global_rank'] = global_rank
        if checkpoint is not None and 'bijector_state' in checkpoint.keys():
            exp_args['bijector_state'] = checkpoint['bijector_state']
        experiment = VariableRedshift(**exp_args)
    
    elif cosmo_exp == 'num_visits':
        from bedcosmo.num_visits import NumVisits
        valid_params = inspect.signature(NumVisits.__init__).parameters.keys()
        valid_params = [k for k in valid_params if k != 'self']

        # Filter run_args to only include valid NumVisits parameters
        exp_args = {k: v for k, v in run_args.items() if k in valid_params}
        exp_args['global_rank'] = global_rank
        if checkpoint is not None and 'bijector_state' in checkpoint.keys():
            exp_args['bijector_state'] = checkpoint['bijector_state']
        if empirical_artifacts_dir is not None:
            exp_args['empirical_artifacts_dir'] = empirical_artifacts_dir
        experiment = NumVisits(**exp_args)
    
    else:
        raise ValueError(
            f"Experiment '{cosmo_exp}' not supported. "
            "Supported experiments: 'num_tracers', 'variable_redshift', 'num_visits'"
        )
    
    return experiment

def load_prior_flow_from_file(prior_flow_path, device, global_rank=0):
    """
    Load a prior flow model from a .pt checkpoint file.

    If the checkpoint embeds :data:`NF_INIT_CONFIG_KEY`, no MLflow run is contacted.
    Otherwise the run id is inferred from the checkpoint path to fetch ``run_args``.

    Expects input_dim and context_dim in the checkpoint (saved at train time, or added via
    scripts/add_checkpoint_dims.py). The full experiment is not initialized.

    Args:
        prior_flow_path (str): Path to the .pt checkpoint file
        device (str): Device to load model on
        global_rank (int): Global rank for distributed training

    Returns:
        tuple: (posterior_flow, prior_flow_metadata) where metadata is a dict with:
            - transform_input: bool indicating if the prior flow was trained with transform_input
            - nominal_context: torch.Tensor or None
    """
    if not os.path.exists(prior_flow_path):
        raise FileNotFoundError(f"Prior flow file not found: {prior_flow_path}")

    checkpoint = torch.load(prior_flow_path, map_location=device, weights_only=False)

    if 'input_dim' not in checkpoint or 'context_dim' not in checkpoint:
        raise ValueError(
            f"Checkpoint {prior_flow_path} is missing 'input_dim' and/or 'context_dim'. "
            "Re-save from training, or run: python scripts/add_checkpoint_dims.py --checkpoint <path.pt> --prior_run_id <run_id>"
        )
    input_dim = int(checkpoint['input_dim'])
    context_dim = int(checkpoint['context_dim'])

    nf_cfg = checkpoint.get(NF_INIT_CONFIG_KEY)
    if nf_cfg is not None:
        if global_rank == 0:
            print(f"Using embedded {NF_INIT_CONFIG_KEY} from prior checkpoint (no MLflow run fetch).")
        prior_run_args = nf_cfg
    else:
        prior_run_id, prior_exp_id, prior_cosmo_exp = extract_run_info_from_checkpoint_path(prior_flow_path)

        if global_rank == 0:
            print(f"Extracted run_id {prior_run_id} from checkpoint path")

        storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{prior_cosmo_exp}"
        mlflow.set_tracking_uri(storage_path + "/mlruns")
        client = MlflowClient()
        prior_run = client.get_run(prior_run_id)
        prior_run_args = parse_mlflow_params(prior_run.data.params)

        if global_rank == 0:
            print(f"Found run in {prior_cosmo_exp} experiment")

    if global_rank == 0:
        print(f"Using input_dim={input_dim}, context_dim={context_dim} from checkpoint")
    
    # Initialize the flow model
    posterior_flow = init_nf(
        prior_run_args,
        input_dim,
        context_dim,
        device,
        seed=None
    )
    
    # Load the state dict
    if 'model_state_dict' in checkpoint:
        posterior_flow.load_state_dict(checkpoint['model_state_dict'], strict=True)
    elif 'model' in checkpoint:
        posterior_flow.load_state_dict(checkpoint['model'], strict=True)
    else:
        raise ValueError("Checkpoint does not contain model state dict")
    
    posterior_flow.to(device)
    posterior_flow.eval()
    
    # Store metadata in a dict to return alongside the flow
    transform_input = False
    if isinstance(prior_run_args, dict):
        transform_input = prior_run_args.get("transform_input", False)

    prior_flow_metadata = {
        'transform_input': transform_input,
        'nominal_context': None,
        'bijector_state': checkpoint.get('bijector_state', None)
    }
    
    if global_rank == 0:
        print(f"Successfully loaded prior flow from {prior_flow_path}")
        print(f"  Input dim: {input_dim}, Context dim: {context_dim}")
        if prior_flow_metadata['bijector_state'] is not None:
            print(f"  Bijector state loaded with params: {list(prior_flow_metadata['bijector_state'].keys())}")
    
    # Return both the flow and its metadata
    return posterior_flow, prior_flow_metadata


def load_model(experiment, step, run_obj, run_args, device, global_rank=0):
    """
    Loads the model from the checkpoint of an already existing exp.

    Args:
        experiment (Experiment): The experiment object.
        step (int): The step to load the model from.
        run_obj (Run): The MLflow Run object.
        run_args (dict): The run arguments.
    """
    # Assumes run_obj is the MLflow Run object and parsed_run_params is the output of parse_mlflow_params(run_obj.data.params)
    storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{experiment.name}"

    current_run_id = run_obj.info.run_id
    exp_id = run_obj.info.experiment_id
    input_dim = len(experiment.cosmo_params)

    effective_step = step
    if step == 'last':
        effective_step = run_args.get("total_steps")

    checkpoint_dir = f'{storage_path}/mlruns/{exp_id}/{current_run_id}/artifacts/checkpoints'
    if not os.path.isdir(checkpoint_dir):
         print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
         raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}. Ensure artifacts are downloaded or path is correct.")

    checkpoint, selected_step = get_checkpoint(effective_step, checkpoint_dir, device, global_rank, run_args["total_steps"])
    if selected_step != effective_step:
        print(f"Warning: Step {effective_step} not found in checkpoints. Loading checkpoint for step {selected_step} instead.")
        effective_step = selected_step

    nf_cfg = checkpoint.get(NF_INIT_CONFIG_KEY)
    init_args = nf_cfg if nf_cfg is not None else run_args
    posterior_flow = init_nf(
        init_args,
        input_dim,
        experiment.context_dim,
        device,
        seed=None
    )

    posterior_flow.load_state_dict(checkpoint['model_state_dict'], strict=True)
    posterior_flow.to(device)
    posterior_flow.eval()
    
    return posterior_flow, selected_step


def load_posterior_flow_from_checkpoint_file(
    experiment,
    checkpoint_path: str,
    run_args=None,
    device: str = "cpu",
    global_rank: int = 0,
):
    """
    Load a trained NF posterior flow from a single ``.pt`` checkpoint file.

    New checkpoints embed :data:`NF_INIT_CONFIG_KEY` so :func:`init_nf` can rebuild
    the architecture without MLflow ``run_args``. Older checkpoints require ``run_args``.
    """
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    nf_cfg = checkpoint.get(NF_INIT_CONFIG_KEY)
    if nf_cfg is not None:
        init_args = nf_cfg
        if "input_dim" not in checkpoint or "context_dim" not in checkpoint:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has {NF_INIT_CONFIG_KEY} but is missing "
                "input_dim and/or context_dim."
            )
        input_dim = int(checkpoint["input_dim"])
        context_dim = int(checkpoint["context_dim"])
    else:
        if run_args is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no {NF_INIT_CONFIG_KEY}; pass run_args from "
                "the training run, or re-save checkpoints with a current trainer."
            )
        init_args = run_args
        input_dim = len(experiment.cosmo_params)
        context_dim = experiment.context_dim
    posterior_flow = init_nf(
        init_args,
        input_dim,
        context_dim,
        device,
        seed=None,
    )
    posterior_flow.load_state_dict(checkpoint["model_state_dict"], strict=True)
    posterior_flow.to(device)
    posterior_flow.eval()
    _ = global_rank
    return posterior_flow


def _posterior_kwds_from_merged_eig(combined: dict, step_key: str) -> dict:
    """Build ``generate_posterior`` kwargs from merged NF+grid eig_data (no MLflow read)."""
    step_data = combined[step_key]
    variable_data = step_data.get("variable", {})
    nominal_data = step_data.get("nominal", {})
    input_designs = np.asarray(combined.get("input_designs", []))
    if input_designs.size == 0:
        raise ValueError("No input designs in merged eig_data")
    eig_values = np.asarray(variable_data.get("eigs_avg", []))
    if eig_values.size == 0:
        raise ValueError("No NF EIG values (variable/eigs_avg) in merged eig_data")
    nominal_eig = nominal_data.get("eigs_avg")
    if isinstance(nominal_eig, list):
        nominal_eig = nominal_eig[0] if len(nominal_eig) > 0 else None
    nominal_eig = float(nominal_eig) if nominal_eig is not None else None
    nominal_grid_eig = None
    nominal_grid_data = nominal_data.get("grid", {})
    if isinstance(nominal_grid_data, dict) and "eigs_avg" in nominal_grid_data:
        nominal_grid_eig = nominal_grid_data.get("eigs_avg")
        if isinstance(nominal_grid_eig, list):
            nominal_grid_eig = nominal_grid_eig[0] if len(nominal_grid_eig) > 0 else None
        nominal_grid_eig = float(nominal_grid_eig) if nominal_grid_eig is not None else None
    return dict(
        input_designs=input_designs,
        eig_values=eig_values,
        nominal_eig=nominal_eig,
        nominal_grid_eig=nominal_grid_eig,
        title="Posterior (NF + grid)",
    )


def _eig_design_kwds_from_merged_eig(
    combined: dict,
    step_key: str,
    experiment,
    include_nominal: bool,
):
    """Build ``eig_designs`` kwargs from merged eig_data using a local experiment instance."""
    step_data = combined[step_key]
    variable_data = step_data.get("variable", {})
    nominal_data = step_data.get("nominal", {})
    input_designs = np.asarray(combined.get("input_designs", []))
    if input_designs.size == 0:
        raise ValueError("No input designs in merged eig_data")
    eig_values = np.asarray(variable_data.get("eigs_avg", []))
    eig_std_values = (
        np.asarray(variable_data.get("eigs_std", np.zeros_like(eig_values)))
        if "eigs_std" in variable_data
        else np.zeros_like(eig_values)
    )
    grid_data = variable_data.get("grid", {}) or {}
    grid_eig_values = (
        np.asarray(grid_data["eigs_avg"]) if grid_data and "eigs_avg" in grid_data else None
    )
    grid_eig_std_values = (
        np.asarray(grid_data["eigs_std"]) if grid_data and "eigs_std" in grid_data else None
    )
    nominal_eig = None
    nominal_grid_eig = None
    if include_nominal:
        nominal_eig = nominal_data.get("eigs_avg")
        if isinstance(nominal_eig, list):
            nominal_eig = nominal_eig[0] if len(nominal_eig) > 0 else None
        nominal_eig = float(nominal_eig) if nominal_eig is not None else None
        nominal_grid_data = nominal_data.get("grid", {})
        if isinstance(nominal_grid_data, dict) and "eigs_avg" in nominal_grid_data:
            nominal_grid_eig = nominal_grid_data.get("eigs_avg")
            if isinstance(nominal_grid_eig, list):
                nominal_grid_eig = nominal_grid_eig[0] if len(nominal_grid_eig) > 0 else None
            nominal_grid_eig = float(nominal_grid_eig) if nominal_grid_eig is not None else None
    return dict(
        eig_values=eig_values,
        eig_std_values=eig_std_values,
        grid_eig_values=grid_eig_values,
        grid_eig_std_values=grid_eig_std_values,
        input_designs=input_designs,
        design_labels=experiment.design_labels,
        nominal_design=experiment.nominal_design.detach().cpu().numpy(),
        nominal_eig=nominal_eig,
        nominal_grid_eig=nominal_grid_eig,
        include_nominal=include_nominal,
        title="EIG per design (NF vs grid)",
    )


def load_nominal_samples(cosmo_exp, cosmo_model, dataset='dr2'):
    home_dir = os.environ["HOME"]
    if cosmo_exp == 'num_tracers':
        nominal_samples = np.load(f"{home_dir}/data/desi/bao_{dataset}/mcmc_samples/{cosmo_model}.npy")
        if cosmo_model == 'base':
            target_labels = ['Om', 'hrdrag']
            latex_labels = ['\Omega_m', 'H_0r_d']
        elif cosmo_model == 'base_omegak':
            target_labels = ['Om', 'Ok', 'hrdrag']
            latex_labels = ['\Omega_m', '\Omega_k', 'H_0r_d']
        elif cosmo_model == 'base_w':
            target_labels = ['Om', 'w0', 'hrdrag']
            latex_labels = ['\Omega_m', 'w_0', 'H_0r_d']
        elif cosmo_model == 'base_w_wa':
            target_labels = ['Om', 'w0', 'wa', 'hrdrag']
            latex_labels = ['\Omega_m', 'w_0', 'w_a', 'H_0r_d']
        elif cosmo_model == 'base_omegak_w_wa':
            target_labels = ['Om', 'Ok', 'w0', 'wa', 'hrdrag']
            latex_labels = ['\Omega_m', '\Omega_k', 'w_0', 'w_a', 'H_0r_d']
    else:
        raise NotImplementedError(f"Nominal samples do not exist for {cosmo_exp}")

    return nominal_samples, target_labels, latex_labels

def parse_mlflow_params(params_dict):
    """
    Safely parses a dictionary of MLflow parameters (string values)
    into their likely Python types (int, float, bool, str).
    """
    parsed_params = {}
    for key, value_str in params_dict.items():
        # 1. Try Boolean and None
        if value_str.lower() == 'true':
            parsed_params[key] = True
            continue
        if value_str.lower() == 'false':
            parsed_params[key] = False
            continue
        if value_str.lower() == 'null':
            parsed_params[key] = None
            continue
        if value_str.lower() == 'none':
            parsed_params[key] = None
            continue

        # 2. Try Integer
        try:
            parsed_params[key] = int(value_str)
            continue
        except ValueError:
            pass # Not an integer

        # 3. Try Float
        try:
            parsed_params[key] = float(value_str)
            continue
        except ValueError:
            pass # Not a float

        # 4. Try JSON object/list (optional, if you store complex params)
        # This handles cases where a param might be a stringified list/dict
        try:
            # Be cautious if params could contain very complex/nested JSON
            parsed_value = json.loads(value_str)
            # Only accept if it results in a list or dict (or other simple JSON types)
            if isinstance(parsed_value, (dict, list)):
                 parsed_params[key] = parsed_value
                 continue
            # else: fall through to treat as plain string if it's just a JSON string like "\"hello\""
        except json.JSONDecodeError:
            pass # Not valid JSON

        # 4b. Try Python literal_eval for Python-style literals (e.g., "['u', 'g', 'r']")
        try:
            import ast
            parsed_value = ast.literal_eval(value_str)
            # Only accept if it results in a list, dict, tuple, or other simple types
            if isinstance(parsed_value, (dict, list, tuple)):
                parsed_params[key] = parsed_value
                continue
        except (ValueError, SyntaxError):
            pass # Not a valid Python literal

        # 5. Keep as String (Fallback)
        parsed_params[key] = value_str # Keep original string if no conversion worked

    return parsed_params

def parse_float_or_list(value):
    """
    Parse a string that can be either a single float or a JSON list of floats.
    
    Args:
        value (str): String representation of either a float or a JSON list
        
    Returns:
        float or list: Parsed value as either a single float or list of floats
    """
    try:
        # First try to parse as JSON (for lists)
        parsed = json.loads(value)
        if isinstance(parsed, list):
            # Ensure all elements are numbers
            return [float(x) for x in parsed]
        else:
            # Single value from JSON
            return float(parsed)
    except (json.JSONDecodeError, ValueError):
        try:
            # If JSON parsing fails, try parsing as a single float
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid value '{}'. Must be a float or JSON list of floats.".format(value))


def parse_param_subsets(value):
    """Parse marginal-EIG subsets from a CLI/YAML string.

    Accepts a JSON list-of-lists (e.g. '[["log_c_scale", "z"]]') or a plain
    string of semicolon-separated groups of comma-separated names
    (e.g. 'log_c_scale,z; f1,f2'). Returns a list of lists of names, or None.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    value = value.strip()
    if not value:
        return None
    if value.startswith("["):
        parsed = json.loads(value)
        if parsed and all(isinstance(p, str) for p in parsed):
            return [parsed]
        return parsed
    return [
        [name.strip() for name in group.split(",") if name.strip()]
        for group in value.split(";")
        if group.strip()
    ]


def parse_json_object(value):
    """Parse a YAML/config value into a dict (for ``central_params``)."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected dict or JSON object string, got {type(value)}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        import ast

        parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def _central_params_as_dict(value):
    """Coerce ``central_params`` from YAML, MLflow, or CLI into a dict."""
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return parse_json_object(value)


def apply_central_param_cli_flags(kwargs):
    """
    Fold ``--central-param-<name> <value>`` flags into ``central_params``.

    CLI args become keys like ``central_param_z`` after dash normalization; the
    parameter name is the suffix after ``central_param_``.
    """
    kwargs = dict(kwargs)
    central_params = _central_params_as_dict(kwargs.get("central_params"))

    for key in list(kwargs):
        if not key.startswith("central_param_") or key == "central_params":
            continue
        param_name = key[len("central_param_") :]
        if not param_name:
            raise ValueError(f"Invalid central parameter flag '{key}'; expected central_param_<name>.")
        central_params[param_name] = kwargs.pop(key)

    if central_params:
        kwargs["central_params"] = central_params
    return kwargs


def _coerce_train_arg_override(key, value, yaml_default, project_root):
    """Coerce a single train CLI override according to the YAML default type."""
    if key == "input_designs":
        if isinstance(value, str):
            input_design_str = value.strip()
            if input_design_str.lower() == "nominal":
                return "nominal"
            if input_design_str.endswith(".json") or input_design_str.endswith(".JSON"):
                file_path = input_design_str
                if not os.path.isfile(file_path) and not os.path.isabs(file_path):
                    file_path = os.path.join(project_root, input_design_str)
                if os.path.isfile(file_path):
                    with open(file_path, "r") as f:
                        return json.load(f)
                try:
                    return json.loads(input_design_str)
                except json.JSONDecodeError:
                    return value
            try:
                return json.loads(input_design_str)
            except json.JSONDecodeError:
                return value
        return value

    if isinstance(yaml_default, bool) and isinstance(value, bool):
        return value
    if isinstance(yaml_default, list):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            if os.environ.get("RANK", "0") == "0":
                print(f"Warning: Could not parse '{key}' as JSON: {e}. Keeping default value.")
            return yaml_default
    if isinstance(yaml_default, dict) and key != "central_params":
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            if os.environ.get("RANK", "0") == "0":
                print(f"Warning: Could not parse '{key}' as JSON object: {e}. Keeping default value.")
            return yaml_default
    if not isinstance(yaml_default, float):
        return value
    return value


def finalize_train_run_args(parsed_args, yaml_config, unknown_argv=None, project_root="."):
    """
    Merge argparse output, train_args.yaml defaults, and extension CLI flags.

    Flags registered on the train parser override YAML when set. Extension flags
    (not registered), including ``--central-param-<name>``, are taken from
    ``unknown_argv`` — the return value of ``parse_known_args()`` — and parsed
    via :func:`parse_extra_args`.
    """
    run_args = dict(parsed_args)

    for key, default in yaml_config.items():
        cli_value = run_args.get(key)
        if cli_value is not None:
            run_args[key] = _coerce_train_arg_override(key, cli_value, default, project_root)
        else:
            run_args[key] = default

    if unknown_argv:
        extra = parse_extra_args(unknown_argv)
        if "central_params" in extra:
            base = _central_params_as_dict(run_args.get("central_params"))
            base.update(_central_params_as_dict(extra.pop("central_params")))
            run_args["central_params"] = base
        for key, value in extra.items():
            if value is not None:
                run_args[key] = value

    return apply_central_param_cli_flags(run_args)


def parse_extra_args(extra_args: list) -> dict:
    """
    Parse a list of unknown CLI args into a kwargs dict, auto-converting value types.

    Handles ``--some-flag`` (boolean True) and ``--key value`` pairs.
    Dashes in key names are converted to underscores.
    Values are coerced in order: int → float → bool/None → str.

    Args:
        extra_args: List of unparsed argument strings from ``parse_known_args()``.

    Returns:
        Dictionary of keyword arguments suitable for passing to ``init_experiment``.
    """
    kwargs = {}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-").replace("-", "_")
            if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                val = extra_args[i + 1]
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        if val.lower() == "true":
                            val = True
                        elif val.lower() == "false":
                            val = False
                        elif val.lower() == "none":
                            val = None
                kwargs[key] = val
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            i += 1
    return apply_central_param_cli_flags(kwargs)


def sort_key_for_group_tuple(group_tuple_key):
    key_as_num_or_str = []
    for val_in_tuple in group_tuple_key:
        try:
            # Attempt to convert to float for numeric sorting, otherwise use string
            key_as_num_or_str.append(float(val_in_tuple))
        except (ValueError, TypeError): # TypeError if val_in_tuple is already a non-string number like int
            if isinstance(val_in_tuple, (int, float)):
                key_as_num_or_str.append(val_in_tuple)
            else:
                key_as_num_or_str.append(str(val_in_tuple)) 
    return tuple(key_as_num_or_str)
    
def get_checkpoints(run_id, steps, checkpoint_files, type='all', cosmo_exp='num_tracers', verbose=False):
    storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{cosmo_exp}"
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    client = MlflowClient()
    run = client.get_run(run_id)
    run_args = parse_mlflow_params(run.data.params)
    steps = [step if step != run_args["total_steps"] else 'last' for step in steps]
    if type == 'all':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.endswith('.pt') and not f.endswith('last.pt') and not f.endswith('best.pt')
        ])
    elif type == 'area':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.startswith('checkpoint_nominal_area_')
        ])
    elif type == 'loss':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.startswith('checkpoint_loss_')
        ])
    # print stpes not found in checkpoints
    steps_not_found = [step for step in steps if step not in checkpoints]
    if verbose:
        print(f"Steps not found in checkpoints: {steps_not_found}")
    # get the checkpoints closest to the steps
    plot_checkpoints = [
        checkpoints[np.argmin(np.abs(np.array(checkpoints) - step))]
        for step in steps if step not in ['loss_best', 'nominal_area_best', 'last']
    ]

    # check for duplicates in plot_checkpoints and replace them with the next checkpoint index 
    for i, step in enumerate(plot_checkpoints):
        if step in plot_checkpoints[:i]:
            if np.argmin(np.abs(np.array(checkpoints) - steps[i])) + 1 < len(checkpoints):
                plot_checkpoints[i] = checkpoints[np.argmin(np.abs(np.array(checkpoints) - steps[i])) + 1]
            else:
                print(f"Warning: No next checkpoint found for step {steps[i]}")
                plot_checkpoints.pop(i)

    # add loss_best and nominal_area_best to the plot_checkpoints if requested
    if 'loss_best' in steps:
        plot_checkpoints.append('loss_best')
    if 'nominal_area_best' in steps:
        plot_checkpoints.append('nominal_area_best')
    if 'last' in steps:
        plot_checkpoints.append('last')

    return plot_checkpoints

def get_gpu_utilization(gpu_index=0):
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        lines = output.strip().splitlines()

        if gpu_index < len(lines):
            util_str, mem_str = lines[gpu_index].strip().split(", ")
            return int(util_str), int(mem_str)
        else:
            raise IndexError(f"GPU index {gpu_index} out of range (only {len(lines)} GPUs found).")
    except Exception as e:
        print(f"GPU stat collection failed: {e}")
        return None, None
    
def log_usage_metrics(device_str, process, step, global_rank=0):
    cpu_memory = process.memory_info().rss / 1024**2  # MB
    mlflow.log_metric("cpu_memory_usage_" + str(global_rank), cpu_memory, step=step)

    # CPU utilization (percent)
    cpu_percent = process.cpu_percent(interval=0)  # short interval for up-to-date value
    mlflow.log_metric("cpu_percent_" + str(global_rank), cpu_percent, step=step)

    # Disk I/O (per process)
    try:
        io_counters = process.io_counters()
        # Convert to MB and update metric name
        io_read_mb = io_counters.read_bytes / (1024**2)
        mlflow.log_metric("io_read_" + str(global_rank), io_read_mb, step=step)
        
        # Convert to MB and update metric name
        io_write_mb = io_counters.write_bytes / (1024**2)
        mlflow.log_metric("io_write_" + str(global_rank), io_write_mb, step=step)
    except (NotImplementedError, AttributeError) as e: # Handle systems where io_counters might not be available
        print(f"Rank {global_rank}: Could not log process I/O: {e}")

    # GPU metrics from nvidia-smi
    if torch.cuda.is_available():
        physical_gpu_id_for_nvidia_smi = None
        if isinstance(device_str, str) and device_str.startswith("cuda:"):
            try:
                logical_idx = int(device_str.split(':')[1])

                cuda_visible_devices_env = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cuda_visible_devices_env:
                    # Filter out non-integer values and handle empty strings if any
                    visible_physical_ids_str = [s.strip() for s in cuda_visible_devices_env.split(',') if s.strip()]
                    visible_physical_ids = []
                    for s_id in visible_physical_ids_str:
                        try:
                            visible_physical_ids.append(int(s_id))
                        except ValueError:
                            print(f"Rank {global_rank}: Warning: Non-integer value '{s_id}' in CUDA_VISIBLE_DEVICES='{cuda_visible_devices_env}'. Skipping.")
                    
                    if not visible_physical_ids and cuda_visible_devices_env: # If parsing resulted in empty list but env var was set
                         print(f"Rank {global_rank}: Warning: CUDA_VISIBLE_DEVICES='{cuda_visible_devices_env}' parsed to no valid IDs. Assuming logical index {logical_idx} as physical.")
                         physical_gpu_id_for_nvidia_smi = logical_idx
                    elif 0 <= logical_idx < len(visible_physical_ids):
                        physical_gpu_id_for_nvidia_smi = visible_physical_ids[logical_idx]
                    else:
                        print(f"Rank {global_rank}: Logical index {logical_idx} is out of bounds for parsed CUDA_VISIBLE_DEVICES physical IDs {visible_physical_ids}. Defaulting to logical index as physical.")
                        physical_gpu_id_for_nvidia_smi = logical_idx # Fallback
                else:
                    # CUDA_VISIBLE_DEVICES not set, logical index is the physical index
                    physical_gpu_id_for_nvidia_smi = logical_idx
            except Exception as e:
                print(f"Rank {global_rank}: Error determining physical GPU ID for nvidia-smi from device string '{device_str}': {e}. Attempting with 0.")
                physical_gpu_id_for_nvidia_smi = 0 # Fallback to 0
        
        elif device_str == "cpu":
            pass # No GPU to query with nvidia-smi
        else:
            # This case implies device_str is not "cuda:X" or "cpu"
            print(f"Rank {global_rank}: Unexpected device string '{device_str}' for nvidia-smi. If CUDA is available, attempting with GPU 0.")
            physical_gpu_id_for_nvidia_smi = 0 # Default to 0 if CUDA is available but device_str is unusual

        if physical_gpu_id_for_nvidia_smi is not None:
            gpu_util, gpu_mem_mb = get_gpu_utilization(gpu_index=physical_gpu_id_for_nvidia_smi)
            if gpu_util is not None:
                mlflow.log_metric("gpu_util_nvidia_smi_percent_" + str(global_rank), gpu_util, step=step)
            if gpu_mem_mb is not None: # gpu_mem_mb is in MiB
                mlflow.log_metric("gpu_memory_nvidia_smi_used_" + str(global_rank), gpu_mem_mb, step=step)

    # GPU metrics from PyTorch (uses the device_str like "cuda:0" directly)
    if torch.cuda.is_available() and isinstance(device_str, str) and device_str.startswith("cuda:"):
        try:
            # Log current allocated memory
            gpu_memory_allocated_mb = torch.cuda.memory_allocated(device_str) / 1024**2
            mlflow.log_metric("gpu_memory_torch_allocated_" + str(global_rank), gpu_memory_allocated_mb, step=step)
            
            # Log current reserved memory
            gpu_memory_reserved_mb = torch.cuda.memory_reserved(device_str) / 1024**2
            mlflow.log_metric("gpu_memory_torch_reserved_" + str(global_rank), gpu_memory_reserved_mb, step=step)
            
            # Log max memory allocated since last reset
            gpu_max_memory_allocated_mb = torch.cuda.max_memory_allocated(device_str) / 1024**2
            mlflow.log_metric("gpu_memory_torch_max_allocated_" + str(global_rank), gpu_max_memory_allocated_mb, step=step)

            # Calculate and log GPU memory capacity utilization (memory occupancy)
            total_gpu_memory_bytes = torch.cuda.get_device_properties(device_str).total_memory
            if total_gpu_memory_bytes > 0:
                max_allocated_bytes = torch.cuda.max_memory_allocated(device_str) 
                gpu_memory_capacity_utilization = (max_allocated_bytes / total_gpu_memory_bytes) * 100
                mlflow.log_metric("gpu_memory_capacity_utilization_" + str(global_rank), gpu_memory_capacity_utilization, step=step)
            
            torch.cuda.reset_peak_memory_stats(device_str) 

        except Exception as e:
            print(f"Rank {global_rank}: Error logging PyTorch CUDA memory metrics for device '{device_str}': {e}")
    elif device_str == "cpu" and global_rank == 0: # Log only once for CPU info or if relevant
        print(f"Rank {global_rank}: Process is on CPU. No PyTorch GPU memory metrics to log.")
        pass

def get_checkpoint(target_step, checkpoint_dir, current_pytorch_device, global_rank, total_steps):

    if target_step == 'loss_best' or target_step == 'nominal_area_best':
        checkpoint = torch.load(f"{checkpoint_dir}/checkpoint_{target_step}.pt", map_location=current_pytorch_device, weights_only=False)
        return checkpoint, target_step
    elif target_step == 'last':
        target_step = total_steps
        
    # Load rank-specific checkpoints (default behavior)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    rank_checkpoint_pattern = f"checkpoint_rank_{global_rank}_"
    rank_checkpoint_files = [f for f in checkpoint_files if f.startswith(rank_checkpoint_pattern)]

    if rank_checkpoint_files:
        # Find the checkpoint closest to target_step
        rank_checkpoint_steps = []
        for f in rank_checkpoint_files:
            try:
                if 'last' in f:
                    step = total_steps
                else:
                    step = int(f.split('_')[-1].split('.')[0])
                rank_checkpoint_steps.append((step, f))
            except ValueError:
                continue
        
        rank_checkpoint_steps.sort()
        
        # Find checkpoint closest to target_step
        closest_idx = 0
        if len(rank_checkpoint_steps) > 1:
            if target_step == float('inf'):
                # For 'latest', take the highest step
                closest_idx = len(rank_checkpoint_steps) - 1
            else:
                closest_idx = np.argmin(np.abs(np.array([s[0] for s in rank_checkpoint_steps]) - target_step))
        
        selected_step, selected_file = rank_checkpoint_steps[closest_idx]
        checkpoint_path = f"{checkpoint_dir}/{selected_file}"
    else:
        # Fallback to shared checkpoint if no rank-specific ones found
        # Find the checkpoint closest to target_step
        shared_checkpoint_steps = []
        for f in checkpoint_files:
            if f.startswith('checkpoint_') and f.endswith('.pt') and not f.startswith('checkpoint_rank_'):
                try:
                    step = int(f.split('_')[-1].split('.')[0])
                    shared_checkpoint_steps.append((step, f))
                except ValueError:
                    continue
        
        shared_checkpoint_steps.sort()
        
        # Find checkpoint closest to target_step
        closest_idx = 0
        if len(shared_checkpoint_steps) > 1:
            if target_step == float('inf'):
                # For 'latest', take the highest step
                closest_idx = len(shared_checkpoint_steps) - 1
            else:
                closest_idx = np.argmin(np.abs(np.array([s[0] for s in shared_checkpoint_steps]) - target_step))
        
        selected_step, selected_file = shared_checkpoint_steps[closest_idx]
        checkpoint_path = f"{checkpoint_dir}/{selected_file}"
        
        print(f"No rank-specific checkpoints found for rank {global_rank}, using shared checkpoint")
        print(f"  Requested step: {target_step}")
        print(f"  Available steps: {[s[0] for s in shared_checkpoint_steps]}")
        print(f"  Selected step: {selected_step}")
        print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=current_pytorch_device, weights_only=False)
    return checkpoint, selected_step

def get_runtime(run_id):
    """
    Get the cumulative active training runtime, excluding downtime between sessions.
    Falls back to end_time - start_time if cumulative runtime is not available.
    """
    run = mlflow.get_run(run_id)
    
    # Try to get cumulative runtime from metrics first
    try:
        client = mlflow.tracking.MlflowClient()
        runtime_history = client.get_metric_history(run_id, 'cumulative_runtime_seconds')
        if runtime_history:
            # Get the latest cumulative runtime value
            cumulative_seconds = runtime_history[-1].value
            return timedelta(seconds=cumulative_seconds)
    except Exception:
        pass  # Fall back to old method
    
    # Fallback: use start_time and end_time (includes downtime)
    start_ms = run.info.start_time      # epoch ms
    end_ms   = run.info.end_time        # epoch ms (None until the run is finished)

    if end_ms is None:
        raise ValueError(f"Run {run_id} is still running or did not record end_time.")

    # Convert epoch-ms → Python datetime
    start_dt = datetime.fromtimestamp(start_ms / 1_000)
    end_dt   = datetime.fromtimestamp(end_ms   / 1_000)
    return end_dt - start_dt

def validate_checkpoint_compatibility(checkpoint, run_args, mode="restart", global_rank=0):
    """
    Validate checkpoint compatibility with current run configuration.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        run_args (dict): Current run arguments
        mode (str): Either "resume" or "restart"
    
    Returns:
        dict: Information about the checkpoint and any compatibility issues
    """
    try:
        info = {
            'mode': mode,
            'step': checkpoint.get('step', 'unknown'),
            'has_model_state': 'model_state_dict' in checkpoint,
            'has_optimizer_state': 'optimizer_state_dict' in checkpoint,
            'has_scheduler_state': 'scheduler_state_dict' in checkpoint,
            'has_rng_state': 'rng_state' in checkpoint,
            'additional_state': checkpoint.get('additional_state', {}),
            'warnings': [],
            'errors': []
        }
        
        # Check for required components
        if not info['has_model_state']:
            info['errors'].append("Checkpoint missing model state")
        
        if not info['has_optimizer_state']:
            info['warnings'].append("Checkpoint missing optimizer state")
        
        # For resume mode, scheduler state is important
        if mode == "resume" and not info['has_scheduler_state']:
            info['warnings'].append("Checkpoint missing scheduler state (may affect learning rate schedule)")
        
        # Check rank information if available
        if 'global_rank' in info['additional_state']:
            info['rank'] = info['additional_state']['global_rank']
        else:
            info['warnings'].append("Checkpoint missing rank information")
        
        # For restart mode, check if learning rate will be updated
        if mode == "restart" and info['has_optimizer_state']:
            try:
                old_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
                new_lr = run_args.get('initial_lr', 'unknown')
                info['lr_change'] = f"{old_lr} -> {new_lr}"
                if old_lr != new_lr:
                    info['warnings'].append(f"Learning rate will be updated: {old_lr} -> {new_lr}")
            except (KeyError, IndexError):
                info['warnings'].append("Could not determine learning rate change")

        # Verify rank information if available
        if 'additional_state' in checkpoint and 'global_rank' in checkpoint['additional_state']:
            checkpoint_rank = checkpoint['additional_state']['global_rank']
            if checkpoint_rank != global_rank:
                if global_rank == 0:
                    print(f"Warning: Checkpoint was saved by rank {checkpoint_rank} but being loaded by rank {global_rank}")
            else:
                if global_rank == 0:
                    print(f"Checkpoint rank verification passed: rank {global_rank}")
        else:
            if global_rank == 0:
                print(f"Note: Checkpoint does not contain rank information")
        
        print_checkpoint_info(info, global_rank)
        return info
        
    except Exception as e:
        return {
            'mode': mode,
            'errors': [f"Failed to load checkpoint: {str(e)}"],
            'warnings': []
        }

def print_checkpoint_info(checkpoint_info, global_rank=0):
    """
    Print checkpoint validation information.
    
    Args:
        checkpoint_info (dict): Output from validate_checkpoint_compatibility
        global_rank (int): Global rank for distributed training
    """
    if global_rank != 0:
        return
    
    print(f"\n=== Checkpoint Validation ({checkpoint_info['mode'].upper()} MODE) ===")
    print(f"Step: {checkpoint_info['step']}")
    
    if 'rank' in checkpoint_info:
        print(f"Rank: {checkpoint_info['rank']}")
    
    if 'lr_change' in checkpoint_info:
        print(f"Learning Rate: {checkpoint_info['lr_change']}")
    
    print(f"Components:")
    print(f"  Model State: {'pass' if checkpoint_info['has_model_state'] else 'fail'}")
    print(f"  Optimizer State: {'pass' if checkpoint_info['has_optimizer_state'] else 'fail'}")
    print(f"  Scheduler State: {'pass' if checkpoint_info['has_scheduler_state'] else 'fail'}")
    print(f"  RNG State: {'pass' if checkpoint_info['has_rng_state'] else 'fail'}")
    
    if checkpoint_info['warnings']:
        print(f"\nWarnings:")
        for warning in checkpoint_info['warnings']:
            print(f"  Warning: {warning}")
    
    if checkpoint_info['errors']:
        print(f"\nErrors:")
        for error in checkpoint_info['errors']:
            print(f"  ERROR: {error}")
    
    if not checkpoint_info['errors']:
        print(f"\nCheckpoint validation passed")
    
    print("=" * 50)

def print_training_state(model, optimizer, step):
    print(f"=== OPTIMIZER STATE AT STEP {step} ===")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Number of parameter groups: {len(optimizer.param_groups)}")
    
    # Print optimizer state for each parameter group
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"\nParameter Group {i}:")
        print(f"  Learning rate: {param_group['lr']}")
        print(f"  Weight decay: {param_group.get('weight_decay', 0)}")
        print(f"  Number of parameters: {len(param_group['params'])}")
        
        # Print state for first few parameters in this group
        for j, param in enumerate(param_group['params'][:3]):  # Show first 3 params
            if param in optimizer.state:
                state = optimizer.state[param]
                print(f"    Param {j} state keys: {list(state.keys())}")
                if 'exp_avg' in state:
                    print(f"      exp_avg shape: {state['exp_avg'].shape}")
                    print(f"      exp_avg mean: {state['exp_avg'].mean().item():.6f}")
                if 'exp_avg_sq' in state:
                    print(f"      exp_avg_sq shape: {state['exp_avg_sq'].shape}")
                    print(f"      exp_avg_sq mean: {state['exp_avg_sq'].mean().item():.6f}")
                if 'step' in state:
                    print(f"      step: {state['step']}")
            else:
                print(f"    Param {j}: No state stored")
    
    # Print model parameter statistics
    print(f"\nMODEL PARAMETER STATISTICS:")
    total_params = 0
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            if total_params <= 1000000:  # Only print for reasonable number of params
                print(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    print(f"Total trainable parameters: {total_params}")
    print("=" * 50)

def get_contour_area(samples, level, *params, global_rank=None, design_type='nominal'):
    """
    Calculate contour areas for parameter pairs.
    
    Args:
        samples: GetDist MCSamples object or list of samples
        level: Contour level (e.g., 0.68 for 68% confidence)
        *params: Variable number of parameters. Calculates areas for all possible pairs.
    
    Returns:
        dict: Dictionary with keys as parameter pairs (e.g., "param1_param2") and values as lists of areas
    """
    # Generate all possible pairs
    param_pairs = list(combinations(params, 2))

    samples = [samples] if type(samples) != list else samples
    areas_list = []

    if len(params) < 2:
        raise ValueError("At least 2 parameters must be provided")
    for sample in samples:
        # Helper function to calculate area for a single parameter pair
        def calculate_area_for_pair(param1, param2):
            """Helper function to calculate area for a single parameter pair"""
            try:
                # Create a temporary figure for this pair
                temp_fig, temp_ax = plt.subplots()
                
                density = sample.get2DDensity(param1, param2)
                if density is None:
                    warnings.warn(f"2D density returned None for {param1}-{param2}. Skipping area calculation.")
                    plt.close(temp_fig)
                    if global_rank is not None:
                        return f"{design_type}_area_{global_rank}_{param1}_{param2}", np.nan
                    else:
                        return f"{design_type}_area_{param1}_{param2}", np.nan
                
                contour_level = density.getContourLevels([level])[0]
                
                # Plot contour on the temporary axes
                cs = temp_ax.contour(density.x, density.y, density.P, 
                                    levels=[contour_level])
                paths = cs.get_paths()
                
                # Check if any contours were found at this level
                if not paths:
                    warnings.warn(f"No contour found for {param1}-{param2} at level {level}. Assigning area NaN.")
                    plt.close(temp_fig)
                    if global_rank is not None:
                        return f"{design_type}_area_{param1}_{param2}_{global_rank}", np.nan
                    else:
                        return f"{design_type}_area_{param1}_{param2}", np.nan
                
                # Calculate total area by summing areas of all paths
                total_area = 0.0
                for path in paths:
                    vertices = path.vertices
                    x, y = vertices[:, 0], vertices[:, 1]
                    
                    # Calculate area using Shoelace formula for this path
                    path_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    total_area += path_area
                
                # Close the temporary figure
                plt.close(temp_fig)
                if global_rank is not None:
                    return f"{design_type}_area_{global_rank}_{param1}_{param2}", total_area
                else:
                    return f"{design_type}_area_{param1}_{param2}", total_area
                    
            except Exception as e:
                warnings.warn(f"Error calculating area for {param1}-{param2}: {str(e)}. Setting area to NaN.")
                if global_rank is not None:
                    return f"{design_type}_area_{global_rank}_{param1}_{param2}", np.nan
                else:
                    return f"{design_type}_area_{param1}_{param2}", np.nan
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks at once
            future_to_pair = {
                executor.submit(calculate_area_for_pair, param1, param2): (param1, param2)
                for param1, param2 in param_pairs
            }
        
        # Collect results as they complete
        areas_dict = {}
        for future in concurrent.futures.as_completed(future_to_pair):
            pair_key, area = future.result()
            areas_dict[pair_key] = area
        areas_list.append(areas_dict)

    return areas_list

def create_gif(run_id, fps=1, add_labels=True, label_position='top-right', text_size=1.0, pause_last_frame=3.0, rank=0):
    """Create GIF using PIL with optional labels for a specific rank
    
    Args:
        run_id: MLflow run ID
        fps: Frames per second for GIF
        add_labels: Whether to add step labels to frames
        label_position: Position of step label ('top-left', 'top-right', 'bottom-left', 'bottom-right', 'center')
        text_size: Size of label text
        pause_last_frame: Duration to pause on last frame (seconds)
        rank: Rank index to create GIF for (default: 0)
    """
    
    client = MlflowClient()
    run = client.get_run(run_id)
    cosmo_exp = run.data.params["cosmo_exp"]
    storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{cosmo_exp}"
    
    # Create GIF for the specified rank only
    dir_path = f"{storage_path}/mlruns/{run.info.experiment_id}/{run.info.run_id}/artifacts/plots/rank_{rank}/posterior"
    output_path = f"{dir_path}/animation.gif"

    # Get all PNG files and sort them numerically
    png_files = glob.glob(os.path.join(dir_path, "*.png"))
    png_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Load images
    images = []
    for idx, file in enumerate(png_files):
        img = Image.open(file)
        
        if add_labels:
            # Extract step number from filename
            step_num = os.path.basename(file).split('.')[0]
            
            # Create a copy to avoid modifying the original
            img_with_label = img.copy()
            
            # Get image dimensions
            width, height = img_with_label.size
            
            # Create drawing context
            draw = ImageDraw.Draw(img_with_label)
            
            # Create scalable font based on text_size
            try:
                # Try to use LiberationSans font (available on your system)
                font = ImageFont.truetype("/usr/share/fonts/truetype/LiberationSans-Regular.ttf", int(20 * text_size))
            except:
                if idx == 0:
                    print("Warning: Could not load LiberationSans font. Falling back to default font.")
                # Fallback to default font
                font = ImageFont.load_default()
            
            # Create label text
            label_text = f"Step: {step_num}"
            
            # Simple text size estimation (scaled by text_size parameter)
            text_width = len(label_text) * int(12 * text_size)  # Approximate width
            text_height = int(20 * text_size)  # Approximate height
            
            # Add padding around text
            padding = 10
            
            # Calculate text position based on label_position with proper margins
            if label_position == 'top-left':
                text_x, text_y = padding, padding
            elif label_position == 'top-right':
                text_x, text_y = width - text_width - padding, padding
            elif label_position == 'bottom-left':
                text_x, text_y = padding, height - text_height - padding
            elif label_position == 'bottom-right':
                text_x, text_y = width - text_width - padding, height - text_height - padding
            else:  # center
                text_x, text_y = (width - text_width) // 2, (height - text_height) // 2
            
            # Ensure text doesn't go outside image bounds
            text_x = max(padding, min(text_x, width - text_width - padding))
            text_y = max(padding, min(text_y, height - text_height - padding))
            
            # Draw simple black text
            draw.text((text_x, text_y), label_text, font=font, fill='black')
            
            images.append(img_with_label)
        else:
            images.append(img)
    
    # Check if any images were found
    if len(images) == 0:
        print(f"Warning: No posterior plots found for rank {rank}. Skipping GIF creation.")
        return
    
    # Create GIF with pause on first frame
    duration = 1000 // fps  # Convert fps to duration in milliseconds
    last_frame_duration = int(pause_last_frame * 1000)  # Convert pause to milliseconds
    
    # Save with custom durations: first frame gets longer duration, others get normal duration
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=[duration] * (len(images) - 1) + [last_frame_duration],
        loop=0
    )
    print(f"GIF created: {output_path}")

def convert_color(c):
    if isinstance(c, np.ndarray):
        # Convert RGBA array to hex color string
        return colors.to_hex(c)
    elif isinstance(c, str):
        return c
    else:
        return str(c)


def _merge_sibling_eig_data(own_eig_data, other_eig_data, step_key):
    """Merge NF and grid eig_data dicts into a single combined dict for plotting.

    The two sides write into disjoint keys within step_key: NF fills
    step_key/variable/eigs_avg etc.; grid fills step_key/variable/grid/*.
    Top-level and section collisions favor own_eig_data.
    """
    combined = {}
    if isinstance(other_eig_data, dict):
        combined.update({k: v for k, v in other_eig_data.items() if k != step_key})
    if isinstance(own_eig_data, dict):
        combined.update({k: v for k, v in own_eig_data.items() if k != step_key})

    own_step = own_eig_data.get(step_key, {}) if isinstance(own_eig_data, dict) else {}
    other_step = other_eig_data.get(step_key, {}) if isinstance(other_eig_data, dict) else {}

    step_combined = {}
    for src in (other_step, own_step):
        if isinstance(src, dict):
            for k, v in src.items():
                if k not in ('variable', 'nominal'):
                    step_combined[k] = v

    for section in ('variable', 'nominal'):
        section_combined = {}
        for src in (other_step, own_step):
            src_section = src.get(section, {}) if isinstance(src, dict) else {}
            if isinstance(src_section, dict):
                section_combined.update(src_section)
        step_combined[section] = section_combined

    combined[step_key] = step_combined
    combined['status'] = 'complete'
    return combined


def render_overlay(
    own_eig_data,
    own_role,
    other_eig_data_path,
    plotter,
    eval_step,
    levels=(0.68,),
    transform_output=True,
    include_nominal=False,
    sort=True,
    plot_prior=False,
    overlay_save_dir=None,
    device=None,
    nf_checkpoint_path=None,
    grid_experiment=None,
):
    """Render NF+grid overlay plots if the sibling eig_data file is complete.

    Called at the end of evaluate.py and grid_calc.py when the user linked the
    two jobs via --other-eig-data. Reads sibling JSON, merges fields into an
    in-memory combined dict, loads grid_samples .npy, and calls the plotter
    methods with the combined data. No merged file is written to disk.

    Args:
        own_eig_data: In-memory eig_data dict produced by the current job.
        own_role: 'nf' or 'grid'.
        other_eig_data_path: Path to sibling JSON (may be None or missing).
        plotter: ``RunPlotter`` (MLflow run, and optionally explicit checkpoint) or
            ``BasePlotter`` together with ``grid_experiment`` for EIG-only / grid-posterior overlays.
        eval_step: Step to render plots for.
        levels: Contour levels.
        transform_output: Whether to transform samples to physical space.
        include_nominal: Passed to eig_designs.
        sort: Passed to eig_designs.
        plot_prior: Passed to generate_posterior overlay figures.
        overlay_save_dir: If set, figures are written here instead of the default
            MLflow artifacts path (used by standalone ``grid_calc`` overlays).
        device: Torch device string for loading the NF checkpoint (default: cuda if
            available, else cpu).
        nf_checkpoint_path: If set, load NF weights from this ``.pt`` file. When the file embeds
            :data:`NF_INIT_CONFIG_KEY`, MLflow ``run_args`` are not required. Otherwise pass
            ``RunPlotter`` so ``run_data['params']`` can be used, or extend the checkpoint via training.
        grid_experiment: Used with ``BasePlotter`` for EIG / grid-only posteriors, and with an explicit
            ``nf_checkpoint_path`` when the checkpoint is self-contained (embedded NF init config).

    Returns True if overlay plots were rendered, False otherwise.
    """
    # Local import avoids circular dependency (plotting imports util at module load).
    from bedcosmo.plotting import BasePlotter, RunPlotter

    if own_role not in ('nf', 'grid'):
        raise ValueError(f"own_role must be 'nf' or 'grid', got {own_role!r}")

    if not other_eig_data_path:
        return False
    if not os.path.exists(other_eig_data_path):
        print(f"Sibling eig_data not found at {other_eig_data_path}; skipping overlay plots.")
        return False

    try:
        with open(other_eig_data_path, 'r') as f:
            other_eig_data = json.load(f)
    except Exception as e:
        print(f"Warning: could not load sibling eig_data {other_eig_data_path}: {e}")
        return False

    if other_eig_data.get('status') != 'complete':
        print(f"Sibling eig_data at {other_eig_data_path} is not complete (status={other_eig_data.get('status')!r}); skipping overlay plots.")
        return False

    step_key = f"step_{eval_step}" if not str(eval_step).startswith('step_') else str(eval_step)
    combined = _merge_sibling_eig_data(own_eig_data, other_eig_data, step_key)

    grid_samples = None
    for src in (own_eig_data, other_eig_data):
        step_data = src.get(step_key, {}) if isinstance(src, dict) else {}
        samples_path = step_data.get('grid_samples_path')
        if samples_path and os.path.exists(samples_path):
            try:
                grid_samples = np.load(samples_path)
                break
            except Exception as e:
                print(f"Warning: could not load grid samples from {samples_path}: {e}")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    posterior_kwargs = {"device": device}
    eig_kwargs = {}
    if overlay_save_dir is not None:
        posterior_kwargs["save_dir"] = overlay_save_dir
        eig_kwargs["save_dir"] = overlay_save_dir

    print(f"Rendering overlay plots (own_role={own_role}) using sibling {other_eig_data_path}...")

    # --- Posterior ---
    try:
        if nf_checkpoint_path:
            if grid_experiment is not None:
                experiment_pf = grid_experiment
                cosmo_exp = grid_experiment.name
                run_params = None
            elif isinstance(plotter, RunPlotter):
                experiment_pf = plotter.get_experiment(device=device)
                cosmo_exp = plotter.cosmo_exp
                run_params = plotter.run_data["params"]
            else:
                raise ValueError(
                    "nf_checkpoint_path requires grid_experiment or a RunPlotter plotter."
                )
            post_flow = load_posterior_flow_from_checkpoint_file(
                experiment_pf, nf_checkpoint_path, run_params, device, global_rank=0
            )
            pk = _posterior_kwds_from_merged_eig(combined, step_key)
            bp = BasePlotter(cosmo_exp=cosmo_exp)
            bp.generate_posterior(
                experiment=experiment_pf,
                posterior_flow=post_flow,
                display=['nominal', 'optimal'],
                guide_samples=50000,
                levels=list(levels),
                plot_prior=plot_prior,
                transform_output=transform_output,
                grid_samples=grid_samples,
                filename='posterior_samples_overlay',
                **posterior_kwargs,
                **pk,
            )
        elif isinstance(plotter, RunPlotter) and nf_checkpoint_path is None:
            plotter.generate_posterior(
                eval_step=eval_step,
                display=['nominal', 'optimal'],
                guide_samples=50000,
                levels=list(levels),
                plot_prior=plot_prior,
                transform_output=transform_output,
                grid_samples=grid_samples,
                eig_data=combined,
                filename='posterior_samples_overlay',
                **posterior_kwargs,
            )
        elif grid_experiment is not None and isinstance(plotter, BasePlotter) and not isinstance(plotter, RunPlotter):
            if grid_samples is None:
                print("No grid_samples available; skipping posterior overlay figure.")
            else:
                plotter.generate_posterior(
                    experiment=grid_experiment,
                    posterior_flow=None,
                    display=(),
                    guide_samples=50000,
                    levels=list(levels),
                    plot_prior=plot_prior,
                    transform_output=transform_output,
                    grid_samples=grid_samples,
                    device=device,
                    filename='posterior_samples_overlay',
                    title="Posterior (grid nominal + prior)",
                    **eig_kwargs,
                )
        else:
            print(
                "Warning: skipped posterior overlay (need RunPlotter for NF posteriors, or "
                "BasePlotter + grid_experiment for grid-only)."
            )
    except Exception as e:
        print(f"Warning: overlay generate_posterior failed: {e}")
        traceback.print_exc()

    # --- EIG designs ---
    try:
        if isinstance(plotter, RunPlotter):
            plotter.eig_designs(
                eval_step=eval_step,
                sort=sort,
                include_nominal=include_nominal,
                eig_data=combined,
                color='black',
                filename='eig_designs_overlay',
                **eig_kwargs,
            )
        elif grid_experiment is not None and isinstance(plotter, BasePlotter) and not isinstance(plotter, RunPlotter):
            ek = _eig_design_kwds_from_merged_eig(
                combined, step_key, grid_experiment, include_nominal=include_nominal
            )
            plotter.eig_designs(
                sort=sort,
                color='black',
                filename='eig_designs_overlay',
                **eig_kwargs,
                **ek,
            )
        else:
            print("Warning: skipped EIG overlay (unsupported plotter configuration).")
    except Exception as e:
        print(f"Warning: overlay eig_designs failed: {e}")
        traceback.print_exc()

    return True