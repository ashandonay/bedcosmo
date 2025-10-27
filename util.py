import sys
import os
import torch
import pyro
import mlflow
import zuko
import random
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
import getdist
from pyro import distributions as dist
from pyro_oed_src import _safe_mean_terms, posterior_loss
import json
import contextlib
import io
import matplotlib.pyplot as plt
from pyro.contrib.util import lexpand
import subprocess
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime, timedelta
import traceback
import functools
import time
import inspect
import concurrent.futures
import warnings
from itertools import combinations
from PIL import Image, ImageDraw, ImageFont
import glob

torch.set_default_dtype(torch.float64)

home_dir = os.environ["HOME"]
if home_dir + "/bed/BED_cosmo" not in sys.path:
    sys.path.insert(0, home_dir + "/bed/BED_cosmo")
sys.path.insert(0, home_dir + "/bed/BED_cosmo/num_tracers")


class Bijector:
    def __init__(self, experiment, priors=None, cdf_bins=1000, cdf_samples=500000):
        self.experiment = experiment
        if priors is not None:
            self.priors = priors
        else:
            self.priors = self.experiment.priors
        self.cdfs = self.create_cdfs(num_bins=int(cdf_bins), num_samples=int(cdf_samples))

    def create_cdfs(self, num_bins, num_samples):
        """
        Create memory-efficient CDF bins from raw samples.
        
        Instead of storing all raw samples, this creates a compact representation
        of the empirical CDF using a small number of bins.
        
        Args:
            samples: Raw samples from the empirical prior
            num_bins: Number of CDF bins to create (default: 1000)
            
        Returns:
            Dictionary with 'bins' and 'cdf_values' keys for fast CDF computation
        """
        with pyro.plate_stack("plate", (num_samples,)):
            empirical_priors = self.experiment.sample_valid_parameters((num_samples,), priors=self.priors)
        cdfs = {}
        for key, samples in empirical_priors.items():
            sorted_samples, _ = torch.sort(samples.flatten())
            n_samples = len(sorted_samples)

            # Create evenly spaced bins across the extended range
            bins = torch.linspace(self.priors[key].low, self.priors[key].high, num_bins, device=samples.device)
            
            # Compute CDF values for each bin
            cdf_values = torch.zeros_like(bins)
            for i, bin_val in enumerate(bins):
                count = (sorted_samples <= bin_val).sum()
                cdf_values[i] = count / n_samples
            
            # Store additional metadata for better boundary handling
            cdfs[key] = {
                'bins': bins,
                'cdf_values': cdf_values
            }
        return cdfs


    def get_state(self):
        """Serialize the bijector's CDF tables to CPU tensors so they can be
        persisted inside a checkpoint without device-specific dependencies.
        """
        state = {}
        for key, tensors in self.cdfs.items():
            state[key] = {
                'bins': tensors['bins'].detach().cpu(),
                'cdf_values': tensors['cdf_values'].detach().cpu()
            }
        return state

    def set_state(self, state, device=None, dtype=None):
        """Restore CDF tables from a serialized state dictionary.

        Args:
            state (dict): Output of `get_state`.
            device (str or torch.device, optional): Target device for tensors.
            dtype (torch.dtype, optional): Target dtype for tensors.
        """
        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)

        sample_cdf = next(iter(self.cdfs.values()), None)
        default_device = sample_cdf['bins'].device if sample_cdf else torch.device('cpu')
        default_dtype = sample_cdf['bins'].dtype if sample_cdf else torch.float64

        target_device = device if device is not None else default_device
        target_dtype = dtype if dtype is not None else default_dtype

        restored = {}
        for key, tensors in state.items():
            bins = tensors['bins'].to(device=target_device, dtype=target_dtype)
            cdf_values = tensors['cdf_values'].to(device=target_device, dtype=target_dtype)
            restored[key] = {'bins': bins, 'cdf_values': cdf_values}
        self.cdfs = restored

    def prior_to_gaussian(self, samples, param_key, target_mean=0.0, target_std=1.0):
        """
        Convert samples from an empirical prior distribution to Gaussian distribution using pre-computed CDF bins.
        
        This method uses pre-computed empirical CDF bins for fast transformation:
        1. Interpolate empirical CDF values using stored bins: u = F_empirical(x)
        2. Apply inverse CDF of standard normal: z = F_normal^(-1)(u)
        3. Scale to target Gaussian: y = target_mean + target_std * z
        
        Args:
            samples: Tensor of samples to transform to Gaussian space
            empirical_cdf: Pre-computed CDF bins dict with 'bins' and 'cdf_values' keys
            target_mean: Mean of the target Gaussian distribution (default: 0.0)
            target_std: Standard deviation of the target Gaussian distribution (default: 1.0)
            
        Returns:
            Tensor of samples transformed to the target Gaussian distribution
        """
        # Extract CDF bins and values
        bins = self.cdfs[param_key]['bins']
        cdf_values = self.cdfs[param_key]['cdf_values']
        
        samples_flat = samples.flatten()
        

        # Use the continuous CDF for smooth transformation
        # Clamp samples to ensure they're within bin bounds to prevent index out of bounds
        samples_flat = torch.clamp(samples_flat, bins[0], bins[-1])
        
        # Find insertion indices for each sample in the sorted bins
        indices = torch.searchsorted(bins, samples_flat, right=False)
        
        # Get CDF values at these indices
        u = cdf_values[indices]
        
                # Handle boundary cases separately with special interpolation
        at_bottom_bin = (indices == 0)
        at_top_bin = (indices == len(bins) - 1)
        
        # Interpolate samples at bottom bin (between min and next bin above)
        if at_bottom_bin.any():
            u[at_bottom_bin] = self._interpolate_cdf(
                samples_flat[at_bottom_bin], 
                bins[0], bins[1], 
                cdf_values[0], cdf_values[1]
            )
        
        # Interpolate samples at top bin (between max and previous bin below)
        if at_top_bin.any():
            u[at_top_bin] = self._interpolate_cdf(
                samples_flat[at_top_bin], 
                bins[-2], bins[-1], 
                cdf_values[-2], cdf_values[-1]
            )
        
        # Handle middle bins (normal interpolation between adjacent bins)
        middle_mask = ~(at_bottom_bin | at_top_bin)
        if middle_mask.any():
            middle_indices = indices[middle_mask]
            u[middle_mask] = self._interpolate_cdf(
                samples_flat[middle_mask],
                bins[middle_indices], 
                bins[middle_indices + 1],
                cdf_values[middle_indices], 
                cdf_values[middle_indices + 1]
            )
        
        # Step 2: Apply inverse CDF of standard normal (quantile function)
        # Add small epsilon to prevent floating-point issues with exactly 0 or 1
        eps = 1e-10
        u_safe = torch.clamp(u, eps, 1.0 - eps)
        
        z = np.sqrt(2.0) * torch.erfinv(2.0 * u_safe - 1.0)
        
        # Step 3: Scale to target Gaussian distribution
        y = target_mean + target_std * z
        
        return y.to(samples.dtype).unsqueeze(-1)

    def gaussian_to_prior(self, gaussian_samples, param_key, source_mean=0.0, source_std=1.0):
        """
        Convert samples from Gaussian distribution back to the original empirical prior distribution space.
        
        This is the inverse of prior_to_gaussian:
        1. Scale from target Gaussian to standard normal: z = (y - source_mean) / source_std
        2. Apply CDF of standard normal: u = F_normal(z)
        3. Apply empirical inverse CDF of empirical prior: x = F_empirical^(-1)(u)
        
        Args:
            gaussian_samples: Tensor of samples from the Gaussian distribution
            empirical_cdf: Pre-computed CDF bins dict with 'bins' and 'cdf_values' keys
            source_mean: Mean of the source Gaussian distribution (default: 0.0)
            source_std: Standard deviation of the source Gaussian distribution (default: 1.0)
            
        Returns:
            Tensor of samples transformed back to the empirical prior distribution space
        """
        # Step 1: Scale from target Gaussian to standard normal N(0,1)
        z = (gaussian_samples - source_mean) / source_std
        
        # Step 2: Apply CDF of standard normal to get uniform [0,1]
        u = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
        
        eps = 1e-4
        u_safe = torch.clamp(u, eps, 1.0 - eps)
        
        # Step 3: Apply empirical inverse CDF using the same boundary-aware interpolation
        bins = self.cdfs[param_key]['bins']
        cdf_values = self.cdfs[param_key]['cdf_values']
        
        # Use searchsorted to find where each CDF value would be inserted in cdf_values
        indices = torch.searchsorted(cdf_values, u_safe.flatten(), right=False)
        indices = torch.clamp(indices, 0, len(cdf_values) - 1)
        
        # Initialize output tensor
        x = torch.zeros_like(u_safe.flatten())
        
        # Handle boundary cases separately with special interpolation
        at_bottom_cdf = (indices == 0)
        at_top_cdf = (indices == len(cdf_values) - 1)
        
        # Interpolate samples at bottom CDF (between min and next CDF above)
        if at_bottom_cdf.any():
            x[at_bottom_cdf] = self._interpolate_cdf(
                u_safe.flatten()[at_bottom_cdf],
                cdf_values[0], cdf_values[1],
                bins[0], bins[1]
            )
        
        # Interpolate samples at top CDF (between max and previous CDF below)
        if at_top_cdf.any():
            x[at_top_cdf] = self._interpolate_cdf(
                u_safe.flatten()[at_top_cdf],
                cdf_values[-2], cdf_values[-1],
                bins[-2], bins[-1]
            )
        
        # Handle middle CDF values (normal interpolation between adjacent CDF values)
        middle_mask = ~(at_bottom_cdf | at_top_cdf)
        if middle_mask.any():
            middle_indices = indices[middle_mask]
            x[middle_mask] = self._interpolate_cdf(
                u_safe.flatten()[middle_mask],
                cdf_values[middle_indices],
                cdf_values[middle_indices + 1],
                bins[middle_indices],
                bins[middle_indices + 1]
            )
        
        return x.to(gaussian_samples.dtype).reshape(gaussian_samples.shape)
    

    # Helper function for linear interpolation
    def _interpolate_cdf(self, samples, bin_low, bin_high, cdf_low, cdf_high):
        """Interpolate CDF values between two bins"""
        distances = (samples - bin_low) / (bin_high - bin_low)
        return cdf_low + distances * (cdf_high - cdf_low)
    
    @staticmethod
    def _logit(x, eps: float = 1e-6):
        """
        Numerically stable logit with clamping.
        x in (0,1) -> R
        """
        x = torch.clamp(x, eps, 1.0 - eps)
        return torch.log(x) - torch.log1p(-x)

    @staticmethod
    def _sigmoid(y: torch.Tensor) -> torch.Tensor:
        """
        Standard logistic sigmoid.
        R -> (0,1)
        """
        return torch.sigmoid(y)

    @staticmethod
    def _to_unit_interval(x: torch.Tensor, a: float, b: float, eps: float = 1e-6) -> torch.Tensor:
        """
        Affine map from (a,b) to (0,1), with clamping for numerical stability.
        """
        u = (x - a) / (b - a)
        return torch.clamp(u, eps, 1.0 - eps)

    @staticmethod
    def _from_unit_interval(u: torch.Tensor, a: float, b: float) -> torch.Tensor:
        """
        Affine map from (0,1) back to (a,b).
        """
        return a + u * (b - a)

    @staticmethod
    def _interval_to_R(x: torch.Tensor, a: float, b: float, eps: float = 1e-6) -> torch.Tensor:
        """
        Map from a bounded open interval (a,b) to R via affine -> (-1,1) then atanh.
        Equivalent to: atanh(u) = 0.5 * [log(1+u) - log(1-u)].
        """
        # map to (0,1)
        u01 = Bijector._to_unit_interval(x, a, b, eps)
        # to (-1,1)
        u = 2.0 * u01 - 1.0
        u = torch.clamp(u, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(u) - torch.log1p(-u))  # atanh(u)

    @staticmethod
    def _R_to_interval(y: torch.Tensor, a: float, b: float) -> torch.Tensor:
        """
        Map from R to (a,b) via tanh then affine.
        """
        u = torch.tanh(y)          # (-1,1)
        u01 = 0.5 * (u + 1.0)      # (0,1)
        return Bijector._from_unit_interval(u01, a, b)

    @staticmethod
    def _log_interval_to_R(x: torch.Tensor, a: float, b: float, H: float = 5.0) -> torch.Tensor:
        """
        Log + affine bijector mapping a positive interval [a, b] with 0 < a < b
        to an approximately symmetric range [-H, H] in R.

        Forward:
            y = s * (log(x) - c)
        where:
            c = 0.5 * (log(a) + log(b))      # center in log-space (geometric mean)
            s = (2H) / (log(b) - log(a))     # scale so that a -> -H and b -> +H
        """
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
        """
        Inverse of `_log_interval_to_R`:
            x = exp( y / s + c )
        with the same c and s definitions.
        """
        a_t = torch.tensor(a, device=y.device, dtype=y.dtype)
        b_t = torch.tensor(b, device=y.device, dtype=y.dtype)
        H_t = torch.tensor(H, device=y.device, dtype=y.dtype)

        loga = torch.log(a_t)
        logb = torch.log(b_t)
        c = 0.5 * (loga + logb)
        s = (2.0 * H_t) / (logb - loga)

        return torch.exp(y / s + c)

def profile_method(func):
    """Decorator to profile method execution time - checks self.profile"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if the instance has profile
        if hasattr(args[0], 'profile') and args[0].profile:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"⏱️  {func.__name__} took {execution_time:.5f} seconds")
            return result
        else:
            # No profiling overhead when disabled
            return func(*args, **kwargs)
    return wrapper

def profile_function(profile=False):
    """Decorator to profile function execution time - only active when profile is True"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if profile:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"⏱️  {func.__name__} took {execution_time:.2f} seconds")
                return result
            else:
                # No profiling overhead when disabled
                return func(*args, **kwargs)
        return wrapper
    return decorator

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

def get_runs_data(mlflow_exp=None, run_ids=None, excluded_runs=[], filter_string=None, parse_params=True):
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


def get_nominal_samples(run_obj, run_args, guide_samples=101, seed=1, device="cuda:0", step='loss_best', cosmo_exp='num_tracers', global_rank=0):
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    mlflow.set_tracking_uri(storage_path + "/mlruns")

    # Pass run_obj, run_args (which should be consistent with run_obj), and classes
    experiment = init_experiment(run_obj, run_args, device, global_rank=global_rank)
    posterior_flow, selected_step = load_model(experiment, step, run_obj, run_args, device, global_rank=global_rank)
    auto_seed(seed)

    nominal_design = torch.tensor(experiment.desi_tracers.groupby('class').sum()['observed'].reindex(experiment.targets).values, device=device, dtype=torch.float64)
    central_vals = experiment.central_val if run_args.get("include_D_M", False) else experiment.central_val[1::2]
    nominal_context = torch.cat([nominal_design, central_vals], dim=-1)

    nominal_samples = experiment.get_guide_samples(posterior_flow, nominal_context, num_samples=guide_samples)

    return nominal_samples, selected_step

def init_experiment(
        run_obj,
        run_args, 
        device,
        checkpoint=None,
        global_rank=0,
        design_args={},
        ):
    """
    Initializes the experiment class with the run arguments.
    Args:
        cosmo_exp (str): The name of the cosmology experiment.
        run_args (dict): The run arguments.
        device (str): The device to use.
        design_args (dict): The design arguments.
        seed (int): The seed to use.
    """
    run_args.update(design_args)
    run_args["priors_path"] = run_obj.info.artifact_uri + "/priors.yaml"
    run_args["device"] = device
    
    cosmo_exp = run_args.get("cosmo_exp")
    
    if cosmo_exp == 'num_tracers':
        from num_tracers import NumTracers
        valid_params = inspect.signature(NumTracers.__init__).parameters.keys()
        valid_params = [k for k in valid_params if k != 'self']
        
        # Filter run_args to only include valid NumTracers parameters
        exp_args = {k: v for k, v in run_args.items() if k in valid_params}
        exp_args['global_rank'] = global_rank
        if checkpoint is not None and 'bijector_state' in checkpoint.keys():
            exp_args['bijector_state'] = checkpoint['bijector_state']
        experiment = NumTracers(**exp_args)
    
    elif cosmo_exp == 'variable_redshift':
        from variable_redshift.variable_redshift import VariableRedshift
        valid_params = inspect.signature(VariableRedshift.__init__).parameters.keys()
        valid_params = [k for k in valid_params if k != 'self']
        
        # Filter run_args to only include valid VariableRedshift parameters
        exp_args = {k: v for k, v in run_args.items() if k in valid_params}
        exp_args['global_rank'] = global_rank
        if checkpoint is not None and 'bijector_state' in checkpoint.keys():
            exp_args['bijector_state'] = checkpoint['bijector_state']
        experiment = VariableRedshift(**exp_args)
    
    else:
        raise ValueError(f"Experiment '{cosmo_exp}' not supported. Supported experiments: 'num_tracers', 'variable_redshift'")
    
    return experiment
    
def eval_eigs(
        experiment, 
        run_args, 
        posterior_flow, 
        n_evals=10,
        n_particles=1000
        ):
    """
    Evaluates the EIG of the posterior flow for an input designs tensor.

    Args:
        designs (torch.Tensor): Designs to evaluate.
        nominal_design (torch.Tensor): Nominal design to evaluate.
        run_args (dict): Run arguments.

    Returns:
        tuple: A tuple containing:
            - avg_eigs (np.ndarray): Average EIGs for each design.
            - optimal_eig (float): Maximum EIG.
            - avg_nominal_eig (float): Average EIG of the nominal design.
            - optimal_design (torch.Tensor): Design with the maximum EIG.
    """
    eigs_batch = []
    nominal_eig_batch = []
    for n in range(n_evals):
        with torch.no_grad():
            agg_loss, eigs = posterior_loss(
                                            experiment=experiment,
                                            guide=posterior_flow,
                                            num_particles=n_particles,
                                            evaluation=True,
                                            nflow=True,
                                            analytic_prior=False,
                                            condition_design=run_args["condition_design"]
                                            )
        eigs_batch.append(eigs.cpu().numpy()/np.log(2))

        with torch.no_grad():
            agg_loss, nominal_eig = posterior_loss(
                                            experiment=experiment,
                                            guide=posterior_flow,
                                            num_particles=n_particles,
                                            evaluation=True,
                                            nflow=True,
                                            analytic_prior=False,
                                            condition_design=run_args["condition_design"],
                                            nominal_design=True
                                            )
        nominal_eig_batch.append(nominal_eig.cpu().numpy()/np.log(2))

    eigs_batch = np.array(eigs_batch)
    nominal_eig_batch = np.array(nominal_eig_batch)
    # avg over the number of evaluations
    avg_eigs = np.mean(eigs_batch, axis=0)
    eigs_std = np.std(eigs_batch, axis=0)
    eigs_se = eigs_std/np.sqrt(n_evals)
    avg_nominal_eig = np.mean(nominal_eig_batch, axis=0).item()

    optimal_design = experiment.designs[np.argmax(avg_eigs)]
    return avg_eigs, np.max(avg_eigs), avg_nominal_eig, optimal_design

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
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{experiment.name}"

    current_run_id = run_obj.info.run_id
    exp_id = run_obj.info.experiment_id
    
    if experiment.name == 'num_tracers':
        input_dim = len(experiment.cosmo_params)
    else:
        raise ValueError(f"{experiment.name} not supported")

    posterior_flow = init_nf(
        run_args, 
        input_dim, 
        experiment.context_dim,
        device,
        seed=None
        )

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
    
    posterior_flow.load_state_dict(checkpoint['model_state_dict'], strict=True)
    posterior_flow.to(device)
    posterior_flow.eval()
    
    return posterior_flow, selected_step

def calc_entropy(design, posterior_flow, experiment, num_samples):
    # sample values of y from experiment.pyro_model
    expanded_design = lexpand(design.unsqueeze(0), num_samples)
    y = experiment.pyro_model(expanded_design)
    nominal_context = torch.cat([expanded_design, y], dim=-1)
    passed_ratio = experiment.calc_passed(expanded_design)
    constrained_parameters = experiment.sample_valid_parameters(passed_ratio.shape[:-1])
    with pyro.plate_stack("plate", passed_ratio.shape[:-1]):
        # register samples in the trace using pyro.sample
        parameters = {}
        for k, v in constrained_parameters.items():
            # use dist.Delta to fix the value of each parameter
            parameters[k] = pyro.sample(k, dist.Delta(v)).unsqueeze(-1)
    evaluate_samples = torch.cat([parameters[k].unsqueeze(dim=-1) for k in experiment.cosmo_params], dim=-1)
    flattened_samples = torch.flatten(evaluate_samples, start_dim=0, end_dim=len(expanded_design.shape[:-1])-1)
    samples = posterior_flow(nominal_context).log_prob(flattened_samples)
    # calculate entropy of samples
    _, entropy = _safe_mean_terms(samples)
    return entropy

def load_desi_samples(cosmo_model):
    desi_samples = np.load(f"{home_dir}/data/desi/mcmc_samples/{cosmo_model}.npy")
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

    return desi_samples, target_labels, latex_labels

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
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
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
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    
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