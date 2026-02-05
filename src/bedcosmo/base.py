"""
Base experiment class for bedcosmo experiments.

This module provides the BaseExperiment abstract base class that defines
the common interface and shared functionality for all experiment classes.
"""

import contextlib
import io
from abc import ABC, abstractmethod

import getdist
import numpy as np
import pyro
from pyro import distributions as dist
from pyro.contrib.util import lexpand
import torch

from bedcosmo.util import profile_method


class BaseExperiment(ABC):
    """
    Abstract base class for all bedcosmo experiments.

    This class defines the common interface that all experiment classes
    must implement, as well as shared functionality for parameter
    transformations, sampling, and evaluation.

    Subclasses must implement:
        - __init__: Initialize experiment with configuration
        - init_designs: Initialize design space
        - init_prior: Initialize prior distributions
        - pyro_model: Define probabilistic model for inference
        - sample_parameters: Sample parameters from prior with constraints

    Subclasses may override:
        - params_to_unconstrained: Transform parameters to unconstrained space
        - params_from_unconstrained: Transform parameters from unconstrained space
        - get_guide_samples: Get samples from trained guide
        - get_nominal_samples: Get nominal/reference samples
        - sample_data: Sample synthetic data
        - sample_params_from_data_samples: Sample parameters given data

    Concrete methods (shared by all subclasses):
        - get_prior_samples: Get samples from prior (uses prior_flow if available)
        - apply_multipliers: Apply parameter multipliers to stacked parameter tensor

    Required attributes (set by subclass):
        - name: Experiment name (str)
        - device: Torch device
        - cosmo_params: List of cosmological parameter names
        - prior: Dictionary of prior distributions
        - latex_labels: List of LaTeX labels for parameters
        - observation_labels: List of observation names
        - context_dim: Dimension of context vector
        - nominal_context: Nominal context for guide sampling
        - transform_input: Whether to transform parameters to unconstrained space
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize experiment.

        Subclasses must set required attributes and call init_prior and init_designs.
        """
        raise NotImplementedError

    @abstractmethod
    def init_designs(self, **kwargs):
        """
        Initialize design space.

        Must set self.designs tensor with shape (n_designs, design_dim).
        """
        raise NotImplementedError

    @abstractmethod
    def init_prior(self, parameters, **kwargs):
        """
        Initialize prior distributions.

        Must return prior dictionary and set related attributes.
        """
        raise NotImplementedError

    @abstractmethod
    def pyro_model(self, design):
        """
        Pyro probabilistic model for inference.

        Args:
            design: Design tensor(s)

        Returns:
            Sampled observations
        """
        raise NotImplementedError

    @abstractmethod
    def sample_parameters(self, sample_shape, prior=None, **kwargs):
        """
        Sample valid parameters from prior.

        Must handle any parameter constraints specific to the experiment.

        Args:
            sample_shape: Shape of samples to draw
            prior: Optional prior dictionary (defaults to self.prior)
            **kwargs: Additional arguments

        Returns:
            Dictionary of parameter tensors
        """
        raise NotImplementedError

    @property
    def _name(self):
        """Experiment name."""
        return getattr(self, "name", self.__class__.__name__.lower())

    # =========================================================================
    # Parameter Transformation Methods
    # =========================================================================

    @profile_method
    def params_to_unconstrained(self, params, bijector_class=None):
        """
        Transform parameters from physical space to unconstrained R^D.

        Default implementation uses bijector for all indexed parameters.
        Subclasses should override if they need custom transformation logic.

        Args:
            params: Parameter tensor with shape (..., D)
            bijector_class: Optional bijector to use (defaults to self.param_bijector)

        Returns:
            Unconstrained parameter tensor with same shape
        """
        if not getattr(self, "transform_input", False):
            return params

        D = len(self.cosmo_params)
        assert params.shape[-1] == D, f"Expected last dim {D}, got {params.shape[-1]}"
        y = params.clone()

        if bijector_class is None:
            bijector_class = self.param_bijector

        # Transform each parameter that has an index
        for param_name in self.cosmo_params:
            idx_attr = f"_idx_{param_name.replace('hrdrag', 'hr')}"
            idx = getattr(self, idx_attr, [])
            if idx:
                x = params[..., idx]
                y[..., idx] = bijector_class.prior_to_gaussian(x, param_name)

        return y

    @profile_method
    def params_from_unconstrained(self, y, bijector_class=None):
        """
        Transform parameters from unconstrained R^D to physical space.

        Default implementation uses bijector for all indexed parameters.
        Subclasses should override if they need custom transformation logic.

        Args:
            y: Unconstrained parameter tensor with shape (..., D)
            bijector_class: Optional bijector to use (defaults to self.param_bijector)

        Returns:
            Physical parameter tensor with same shape
        """
        if not getattr(self, "transform_input", False):
            return y

        D = len(self.cosmo_params)
        assert y.shape[-1] == D, f"Expected last dim {D}, got {y.shape[-1]}"
        x = y.clone()

        if bijector_class is None:
            bijector_class = self.param_bijector

        # Transform each parameter that has an index
        for param_name in self.cosmo_params:
            idx_attr = f"_idx_{param_name.replace('hrdrag', 'hr')}"
            idx = getattr(self, idx_attr, [])
            if idx:
                x[..., idx] = bijector_class.gaussian_to_prior(y[..., idx], param_name)

        return x

    # =========================================================================
    # Sampling Methods
    # =========================================================================

    @profile_method
    def get_guide_samples(
        self,
        guide,
        context=None,
        num_samples=5000,
        params=None,
        transform_output=True,
    ):
        """
        Sample parameters from the guide (variational distribution).

        Args:
            guide: Trained guide/normalizing flow
            context: Context tensor (defaults to self.nominal_context)
            num_samples: Number of samples to draw
            params: Optional list of parameter names to return (defaults to all)
            transform_output: Whether to transform from unconstrained to physical space

        Returns:
            getdist.MCSamples object with parameter samples
        """
        if context is None:
            context = self.nominal_context

        with torch.no_grad():
            param_samples = guide(context.squeeze()).sample((num_samples,))

        # Transform if needed
        if getattr(self, "transform_input", False) and transform_output:
            param_samples = self.params_from_unconstrained(param_samples)

        self.apply_multipliers(param_samples)

        # Filter parameters if requested
        if params is None:
            names = self.cosmo_params
            labels = self.latex_labels
        else:
            param_indices = [self.cosmo_params.index(param) for param in params if param in self.cosmo_params]
            param_samples = param_samples[:, param_indices]
            names = [self.cosmo_params[i] for i in param_indices]
            labels = [self.latex_labels[i] for i in param_indices]

        # Add tiny noise to constant columns to prevent getdist from excluding them
        for i in range(param_samples.shape[1]):
            col = param_samples[:, i]
            if torch.all(col == col[0]):
                if getattr(self, "global_rank", 0) == 0:
                    print(f"Column {i} ({names[i]}) is constant with value {col[0]}, adding tiny noise")
                noise_scale = abs(col[0]) * 1e-10 if col[0] != 0 else 1e-10
                param_samples[:, i] = col + torch.randn_like(col) * noise_scale

        with contextlib.redirect_stdout(io.StringIO()):
            param_samples_gd = getdist.MCSamples(
                samples=param_samples.cpu().numpy(),
                names=names,
                labels=labels,
            )

        return param_samples_gd

    def apply_multipliers(self, param_samples):
        """
        Apply parameter multipliers to a stacked parameter tensor.

        Checks for {param_name}_multiplier attributes (set by init_prior from YAML config).

        Args:
            param_samples: Tensor with shape (..., n_params) ordered by self.cosmo_params

        Returns:
            The same tensor with multipliers applied in-place
        """
        for i, param_name in enumerate(self.cosmo_params):
            multiplier = getattr(self, f"{param_name}_multiplier", None)
            if multiplier is not None:
                param_samples[..., i] *= multiplier
        return param_samples

    @profile_method
    def get_prior_samples(self, num_samples=100000):
        """
        Sample from prior or prior_flow if available; return GetDist MCSamples in physical units.

        Applies parameter multipliers (e.g. hrdrag_multiplier) so output is in display units.

        Args:
            num_samples: Number of samples to draw

        Returns:
            getdist.MCSamples object with parameter samples
        """
        if hasattr(self, "prior_flow") and self.prior_flow is not None:
            param_samples = self._sample_prior_flow_cache(num_samples).to(
                device=self.device, dtype=torch.float64
            )
        else:
            with pyro.plate("plate", num_samples):
                parameters = self.sample_parameters(
                    (num_samples,), use_prior_flow=False
                )
            param_samples = torch.stack(
                [parameters[k].squeeze(-1) for k in self.cosmo_params], dim=-1
            )

        if param_samples.dtype != torch.float64:
            param_samples = param_samples.to(torch.float64)

        self.apply_multipliers(param_samples)

        with contextlib.redirect_stdout(io.StringIO()):
            samples_gd = getdist.MCSamples(
                samples=param_samples.cpu().numpy(),
                names=self.cosmo_params,
                labels=self.latex_labels,
            )
        return samples_gd

    @profile_method
    def get_nominal_samples(self, num_samples=10000, params=None, transform_output=True):
        """
        Get nominal/reference samples (e.g., from MCMC chains).

        Default implementation raises NotImplementedError.
        Subclasses should override if they have reference samples.

        Args:
            num_samples: Number of samples to return
            params: Optional list of parameter names to return
            transform_output: Whether to transform parameters

        Returns:
            getdist.MCSamples object with parameter samples
        """
        raise NotImplementedError(f"get_nominal_samples not implemented for {self._name}")

    @profile_method
    def sample_data(self, design, num_samples=100, central=False):
        """
        Sample synthetic data given design.

        Default implementation uses pyro_model with expanded design.
        Subclasses may override for experiment-specific behavior.

        Args:
            design: Design tensor
            num_samples: Number of data samples to draw
            central: If True, use central values instead of sampling parameters

        Returns:
            Data samples tensor
        """
        expanded_design = lexpand(design, num_samples)
        with torch.no_grad():
            data_samples = self.pyro_model(expanded_design)
        return data_samples

    @profile_method
    def sample_params_from_data_samples(
        self,
        design,
        guide,
        num_data_samples=100,
        num_param_samples=1000,
        central=False,
        transform_output=True,
    ):
        """
        Sample parameters from posterior given data samples.

        Vectorized implementation that batches all sampling operations.

        Args:
            design: Design tensor
            guide: Trained guide for parameter sampling
            num_data_samples: Number of data realizations to sample
            num_param_samples: Number of parameter samples per data realization
            central: If True, use central values for data generation
            transform_output: If True, transform parameters to physical space

        Returns:
            numpy array with shape (num_data_samples, num_param_samples, num_params)
        """
        data_samples = self.sample_data(design, num_data_samples, central)

        # Create context: concatenate design and observations
        if design.dim() == 2:
            expanded_design = design.unsqueeze(0).expand(num_data_samples, -1, -1)
        else:
            expanded_design = design.expand(num_data_samples, -1, -1)

        if data_samples.dim() == 2:
            data_samples = data_samples.unsqueeze(1)

        context = torch.cat([expanded_design, data_samples], dim=-1)

        # Vectorized sampling
        context_squeezed = context.squeeze(1) if context.dim() == 3 else context
        expanded_context = context_squeezed.unsqueeze(1).expand(-1, num_param_samples, -1).contiguous()
        expanded_context = expanded_context.view(-1, expanded_context.shape[-1])

        # Sample all parameters at once
        with torch.no_grad():
            param_samples = guide(expanded_context).sample(())

        # Apply transformations if needed
        if getattr(self, "transform_input", False) and transform_output:
            param_samples = self.params_from_unconstrained(param_samples)

        self.apply_multipliers(param_samples)

        # Reshape to [num_data_samples, num_param_samples, num_params]
        param_samples = param_samples.view(num_data_samples, num_param_samples, -1)

        # Add tiny noise to constant columns
        for i in range(param_samples.shape[2]):
            col = param_samples[:, :, i]
            if torch.all(col == col[0, 0]):
                if getattr(self, "global_rank", 0) == 0:
                    print(f"Column {i} ({self.cosmo_params[i]}) is constant, adding tiny noise")
                noise_scale = abs(col[0, 0]) * 1e-10 if col[0, 0] != 0 else 1e-10
                param_samples[:, :, i] = col + torch.randn_like(col) * noise_scale

        return param_samples.cpu().numpy()

    # =========================================================================
    # Prior Flow Sampling Methods
    # =========================================================================
    #
    # These methods support using a trained normalizing flow as a prior.
    # Subclasses that use prior flows should set the following attributes:
    #
    #   prior_flow: The trained flow model (required if using prior flow)
    #   prior_flow_nominal_context: Context for sampling (defaults to nominal_context)
    #   prior_flow_transform_input: Whether flow outputs unconstrained params (default: False)
    #   prior_flow_batch_size: Batch size for direct sampling (default: 10000)
    #   prior_flow_cache_size: Size of the sample cache (default: 100000)
    #   prior_flow_use_cache: Whether to use caching for speed (default: True)
    #
    # Internal cache state (managed by methods):
    #   _prior_flow_cache: Tensor of cached samples
    #   _prior_flow_cache_idx: Current index into cache

    @profile_method
    def _sample_prior_flow(self, total_samples):
        """
        Sample directly from the prior flow (slow but fresh samples each time).

        This method samples from the prior flow in batches to avoid memory issues.
        It does NOT register samples with pyro - the caller handles registration.

        Args:
            total_samples: Number of samples to generate

        Returns:
            Tensor of shape (total_samples, n_params) with parameter values
        """
        if not hasattr(self, "prior_flow") or self.prior_flow is None:
            raise RuntimeError("prior_flow not set - cannot sample from prior flow")

        # Get configuration with defaults
        nominal_context = getattr(
            self, "prior_flow_nominal_context", getattr(self, "nominal_context", None)
        )
        if nominal_context is None:
            raise RuntimeError("No nominal context available for prior flow sampling")

        nominal_context = nominal_context.to(self.device)
        batch_size = getattr(self, "prior_flow_batch_size", 10000)
        prior_transform_input = getattr(self, "prior_flow_transform_input", False)

        all_samples = []
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                batch_end = min(i + batch_size, total_samples)
                batch_size_actual = batch_end - i

                expanded_context = nominal_context.unsqueeze(0).expand(batch_size_actual, -1)
                posterior_dist = self.prior_flow(expanded_context)
                batch_samples = posterior_dist.sample()

                if prior_transform_input:
                    batch_samples = self.params_from_unconstrained(batch_samples)

                all_samples.append(batch_samples)

        return torch.cat(all_samples, dim=0)

    @profile_method
    def _init_prior_flow_cache(self, cache_size=None):
        """
        Pre-generate a cache of samples from the prior flow for fast access during training.

        NAF (Neural Autoregressive Flow) is inherently slow due to sequential computation.
        By pre-sampling a large cache at initialization, we can draw from it instantly
        during training instead of running expensive forward passes each step.

        Args:
            cache_size: Number of samples to pre-generate (default from prior_flow_cache_size
                        or 100000, ~2MB memory for 5 params)
        """
        if cache_size is None:
            cache_size = getattr(self, "prior_flow_cache_size", 100000)

        if getattr(self, "global_rank", 0) == 0:
            print(f"Pre-generating {cache_size:,} samples from prior_flow for cache...")

        self._prior_flow_cache = self._sample_prior_flow(cache_size)
        self._prior_flow_cache_idx = 0

    @profile_method
    def _sample_prior_flow_cache(self, sample_shape):
        """
        Sample parameter values from a trained flow model used as prior, with caching.

        If prior_flow_use_cache is True, uses a pre-generated cache for fast access.
        If False, samples directly from the prior flow each time (slow but fresh).

        The flow model is conditional on context (design + observations),
        so we sample at the nominal context (nominal design + central values).

        Args:
            sample_shape: int or tuple specifying the desired sample shape

        Returns:
            Tensor of shape (*sample_shape, n_params) with raw parameter values.
            Does NOT register with pyro - caller handles registration.
        """
        total_samples = int(np.prod(sample_shape)) if isinstance(sample_shape, tuple) else int(sample_shape)

        use_cache = getattr(self, "prior_flow_use_cache", True)

        if use_cache:
            # Initialize cache if needed
            cache_size = getattr(self, "prior_flow_cache_size", 100000)
            if not hasattr(self, "_prior_flow_cache") or self._prior_flow_cache is None:
                self._init_prior_flow_cache(cache_size)

            # Check if we need to wrap around - shuffle and reuse
            if self._prior_flow_cache_idx + total_samples > len(self._prior_flow_cache):
                perm = torch.randperm(len(self._prior_flow_cache), device=self.device)
                self._prior_flow_cache = self._prior_flow_cache[perm]
                self._prior_flow_cache_idx = 0

            # Draw samples from cache
            samples = self._prior_flow_cache[
                self._prior_flow_cache_idx : self._prior_flow_cache_idx + total_samples
            ]
            self._prior_flow_cache_idx += total_samples
        else:
            # Sample directly from prior flow (slow)
            samples = self._sample_prior_flow(total_samples)

        # Reshape to match sample_shape
        if isinstance(sample_shape, tuple):
            samples = samples.reshape(sample_shape + (-1,))
        else:
            samples = samples.reshape(sample_shape, -1)

        return samples

    def clear_prior_flow_cache(self):
        """Clear the prior flow cache to free memory or force regeneration."""
        if hasattr(self, "_prior_flow_cache"):
            del self._prior_flow_cache
        self._prior_flow_cache = None
        self._prior_flow_cache_idx = 0

    # =========================================================================
    # Likelihood Methods
    # =========================================================================

    @profile_method
    def unnorm_lfunc(self, params, features, designs):
        """
        Compute unnormalized log-likelihood for BED calculations.

        This method is used by the bayesdesign package for brute force
        EIG calculations.

        Default implementation raises NotImplementedError.
        Subclasses should override if they support brute force calculations.

        Args:
            params: Grid object containing cosmological parameters
            features: Grid object containing observed features
            designs: Grid object containing design variables

        Returns:
            numpy array of unnormalized likelihood values
        """
        raise NotImplementedError(f"unnorm_lfunc not implemented for {self._name}")
