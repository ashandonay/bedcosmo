import os
import sys
import yaml
import json
import mlflow
import pandas as pd
import numpy as np
import pyro
from pyro import distributions as dist
from pyro.contrib.util import lexpand
from astropy.cosmology import Planck18
from astropy import constants
import torch
from torch import trapezoid
from bed.grid import GridStack
from bed.grid import Grid
import matplotlib.pyplot as plt
import contextlib
import io
import getdist
import math
import inspect

from bedcosmo.custom_dist import ConstrainedUniform2D
from bedcosmo.util import Bijector, auto_seed, profile_method, get_experiment_config_path
from bedcosmo.base import BaseExperiment
from bedcosmo.cosmology import CosmologyMixin, _interp1

storage_path = os.environ["SCRATCH"] + "/bedcosmo/variable_redshift"
home_dir = os.environ["HOME"]
mlflow.set_tracking_uri(storage_path + "/mlruns")


class VariableRedshift(BaseExperiment, CosmologyMixin):
    def __init__(
        self, 
        prior_args=None,
        design_args=None,
        cosmo_model="base",
        flow_type="MAF",
        nominal_design=None,
        include_D_M=False,
        error_scale=1.0,
        sigma_D_H=0.2,
        sigma_D_M=0.2,
        bijector_state=None,
        seed=None,
        global_rank=0, 
        transform_input=False,
        device="cuda:0",
        mode='eval',
        verbose=False,
        profile=False
    ):

        self.name = 'variable_redshift'
        self.cosmo_model = cosmo_model
        self.flow_type = flow_type
        self.device = device
        self.mode = mode
        self.global_rank = global_rank  
        self.seed = seed
        self.verbose = verbose
        self.profile = profile
        
        self.n_redshifts = len(design_args.get('labels', ['z_1']))
        self.obs_per_redshift = 2 if include_D_M else 1
        self.observation_dim = self.n_redshifts * self.obs_per_redshift
        if seed is not None:
            auto_seed(self.seed)
        
        self.rdrag = 149.77
        self.H0 = Planck18.H0.value
        self.c = constants.c.to('km/s').value
        self.coeff = self.c / (self.H0 * self.rdrag)
        self.include_D_M = include_D_M

        if os.path.exists(os.path.join(home_dir, "data", "variable_error.csv")):
            self.error_scale = error_scale
            if self.global_rank == 0:
                print(f"Loading variable error from {os.path.join(home_dir, 'data', 'variable_error.csv')}")
            self.variable_error_df = pd.read_csv(os.path.join(home_dir, "data", "variable_error.csv"))
            self.variable_error = {
                'z': torch.tensor(
                    self.variable_error_df['z'].values,
                    device=self.device,
                    dtype=torch.float64
                ),
                'D_H': torch.tensor(
                    self.variable_error_df['DH_errors'].values * self.error_scale,
                    device=self.device,
                    dtype=torch.float64
                )
            }
            if self.include_D_M:
                self.variable_error['D_M'] = torch.tensor(
                    self.variable_error_df['DM_errors'].values * self.error_scale,
                    device=self.device,
                    dtype=torch.float64
                )
        self.sigma_D_H = sigma_D_H
        self.sigma_D_M = sigma_D_M
        
        # Context dimension calculation
        # Context = design (redshift) + observations
        self.context_dim = self.n_redshifts + self.observation_dim
        
        # Initialize the prior
        self.prior_args = prior_args
        if self.prior_args is None:
            raise ValueError("prior_args must be provided. It should be loaded from MLflow artifacts or passed explicitly.")
        self.prior, self.param_constraints, self.latex_labels = self.init_prior(**self.prior_args)
        self.desi_prior = self.prior  # For now, same as prior
        self.cosmo_params = list(self.prior.keys())
        
        self.transform_input = transform_input
        if self.transform_input:
            # Initialize bijector for parameter transformation
            self.param_bijector = Bijector(self, cdf_bins=5000, cdf_samples=1e7)
            if bijector_state is not None:
                if self.global_rank == 0:
                    print(f"Restoring bijector state from checkpoint.")
                self.param_bijector.set_state(bijector_state)
            
            # For consistency with desi bijector (even though we may not have DESI data)
            self.desi_bijector = self.param_bijector
        
        # Store parameter indices for transformations
        self._idx_Om = [self.cosmo_params.index('Om')] if 'Om' in self.cosmo_params else []
        self._idx_Ok = [self.cosmo_params.index('Ok')] if 'Ok' in self.cosmo_params else []
        self._idx_w0 = [self.cosmo_params.index('w0')] if 'w0' in self.cosmo_params else []
        self._idx_wa = [self.cosmo_params.index('wa')] if 'wa' in self.cosmo_params else []
        self._idx_hr = [self.cosmo_params.index('hrdrag')] if 'hrdrag' in self.cosmo_params else []
        
        # Observation labels
        self.observation_labels = ["y"]  # Single observation label for MultivariateNormal
        
        # Extract design parameters from design_args for nominal_design calculation
        if design_args is not None:
            design_lower = design_args.get('lower', 0.0)
            design_upper = design_args.get('upper', 5.0)
        else:
            design_lower = 0.0
            design_upper = 5.0
        
        # Initialize designs
        if design_args is not None:
            self.init_designs(**design_args)
        else:
            self.init_designs()
        
        # Extract labels from design_args if provided, otherwise generate from n_redshifts
        if design_args is not None and 'labels' in design_args:
            self.design_labels = design_args['labels']
        else:
            self.design_labels = [f"z_{i+1}" for i in range(self.n_redshifts)]
        
        if nominal_design is None:
            if self.n_redshifts == 1:
                nominal_values = torch.tensor([(design_upper - design_lower) / 2.0 + design_lower], device=self.device, dtype=torch.float64)
            else:
                nominal_values = torch.linspace(design_lower, design_upper, steps=self.n_redshifts, device=self.device, dtype=torch.float64)
        else:
            nominal_values = torch.as_tensor(nominal_design, device=self.device, dtype=torch.float64)
            if nominal_values.ndim == 0:
                nominal_values = nominal_values.unsqueeze(0)
        if nominal_values.shape[-1] != self.n_redshifts:
            raise ValueError(f"nominal_design must have length {self.n_redshifts}, got {nominal_values.shape[-1]}")
        self.nominal_design = nominal_values.reshape(-1)
        
        # Compute central values using fiducial cosmology
        self.central_val = self._compute_central_values()
        
        # Update nominal context with central values
        if self.include_D_M:
            self.nominal_context = torch.cat([
                self.nominal_design,
                self.central_val
            ], dim=-1)
        else:
            # Only include D_H observation
            self.nominal_context = torch.cat([
                self.nominal_design,
                self.central_val  # Only D_H
            ], dim=-1)
        
        if self.global_rank == 0 and self.verbose:
            print(f"Variable Redshift Experiment Initialized")
            print(f"  Cosmology model: {self.cosmo_model}")
            print(f"  Parameters: {self.cosmo_params}")
            print(f"  Observation labels: {self.observation_labels}")
            print(f"  Context dimension: {self.context_dim}")
            print(f"  Include D_M: {self.include_D_M}")
            print(f"  Number of redshifts: {self.n_redshifts}")
            print(f"  Number of designs: {self.designs.shape[0]}")
            print(f"  Nominal design: {self.nominal_design}")

    @profile_method
    def init_designs(
        self, input_designs=None, input_designs_path=None, 
        step=0.1, lower=0.0, upper=5.0, perm_invar=True, 
        labels=None, input_type="variable"):
        """
        Initialize the redshift design grid.
        
        Args:
            input_designs: Can be:
                - None: Generate design grid (default)
                - "nominal": Use the nominal design as the input design
                - list/array/tensor: Use specific design(s), shape should be (num_designs, n_redshifts)
                  If 1D list with length == n_redshifts, it will be reshaped to (1, n_redshifts)
                  Examples: [2.0] for single redshift design with n_redshifts=1
                           [[2.0], [2.5]] for multiple single-redshift designs
                           [[2.0, 2.5]] for single design with n_redshifts=2
            input_designs_path: Path to JSON file containing designs (overrides input_designs if provided)
            step: Step size for design grid (default: 0.1, ignored if input_design is provided)
            lower: Lower bound for redshift grid (default: 0.0, ignored if input_design is provided)
            upper: Upper bound for redshift grid (default: 5.0, ignored if input_design is provided)
            perm_invar: Enforce permutation invariance by removing duplicate permutations (default: True)
        """
        # If input_designs_path is provided, load from path (assumed to be absolute)
        if input_designs_path is not None:
            if not os.path.isabs(input_designs_path):
                raise ValueError(f"input_designs_path must be an absolute path, got: {input_designs_path}")
            if not os.path.exists(input_designs_path):
                raise FileNotFoundError(f"input_designs_path not found: {input_designs_path}")
            
            if self.global_rank == 0:
                print(f"Loading input designs from numpy file: {input_designs_path}")
            input_designs_array = np.load(input_designs_path)
            # Convert to tensor
            if isinstance(input_designs_array, torch.Tensor):
                input_designs = input_designs_array.to(self.device, dtype=torch.float64)
            elif isinstance(input_designs_array, (list, tuple, np.ndarray)):
                input_designs = torch.as_tensor(input_designs_array, device=self.device, dtype=torch.float64)
            else:
                raise ValueError(f"input_designs must be a list, array, or tensor, got {type(input_designs_array)}")
        
        # Check if input_designs is the special "nominal" keyword
        if input_designs == "nominal":
            # Compute nominal design (same logic as in __init__)
            if self.n_redshifts == 1:
                nominal_values = torch.tensor([(upper - lower) / 2.0 + lower], device=self.device, dtype=torch.float64)
            else:
                nominal_values = torch.linspace(lower, upper, steps=self.n_redshifts, device=self.device, dtype=torch.float64)
            designs = nominal_values.unsqueeze(0)  # Add batch dimension
        elif input_designs is not None:
            if isinstance(input_designs, torch.Tensor):
                designs = input_designs.to(self.device, dtype=torch.float64)
            elif isinstance(input_designs, (list, tuple, np.ndarray)):
                designs = torch.as_tensor(input_designs, device=self.device, dtype=torch.float64)
            else:
                raise ValueError(f"input_designs must be a list, array, or tensor, got {type(input_designs)}")
            
            # Handle 1D input (single design)
            if designs.ndim == 1:
                if len(designs) != self.n_redshifts:
                    raise ValueError(f"Input design must have {self.n_redshifts} values, got {len(designs)}")
                designs = designs.unsqueeze(0)
        else:
            # Create a grid of redshift values (build on CPU for cartesian_prod support)
            z_values = torch.arange(lower, upper + step, step, device='cpu', dtype=torch.float64)
            if self.n_redshifts == 1:
                designs = z_values.unsqueeze(1)
            else:
                grid_inputs = [z_values for _ in range(self.n_redshifts)]
                designs = torch.cartesian_prod(*grid_inputs)
        if designs.ndim == 1:
            designs = designs.unsqueeze(0)
        if designs.shape[-1] != self.n_redshifts:
            raise ValueError(f"Design dimension ({designs.shape[-1]}) must match n_redshifts ({self.n_redshifts})")
        
        # Enforce permutation invariance: remove duplicate permutations
        # Sort each row to normalize order, then find unique combinations
        if perm_invar and self.n_redshifts > 1:
            designs_sorted = torch.sort(designs, dim=-1)[0]
            # Find unique sorted combinations (this gives us the canonical sorted representation)
            unique_sorted, inverse_indices = torch.unique(designs_sorted, dim=0, return_inverse=True)
            # Use the sorted (canonical) version of each unique combination
            designs = unique_sorted
        
        self.designs = designs.to(self.device)
        
        if self.global_rank == 0 and self.verbose:
            print(f"Initialized {self.designs.shape[0]} redshift designs from z={lower} to z={upper}")

    @profile_method
    def init_prior(
        self,
        parameters,
        constraints=None,
        prior_flow=None,
        prior_run_id=None,
        **kwargs
    ):
        """
        Load cosmological prior and constraints from prior arguments.
        
        This function dynamically constructs the prior and latex labels based on the
        specified cosmology model in the YAML file. It supports uniform distributions
        and handles parameter constraints.
        
        Args:
            parameters (dict): Dictionary defining each parameter with distribution type and bounds.
                Each parameter should have:
                - distribution: dict with 'type' ('uniform'), 'lower', 'upper'
                - latex: LaTeX label for the parameter
                - multiplier: optional multiplier for the parameter
            constraints (dict, optional): Dictionary defining parameter constraints.
                Keys are constraint names, values are dicts with 'affected_parameters' and 'bounds'.
            prior_flow (str, optional): Absolute path to prior flow checkpoint file.
                Must be an absolute path. Required if using a trained posterior as prior.
            prior_run_id (str, optional): MLflow run ID for prior flow metadata.
                Required if prior_flow is specified.
            **kwargs: Additional arguments (ignored, for compatibility with YAML structure).
        """
        try:
            models_yaml_path = get_experiment_config_path('variable_redshift', 'models.yaml')
            with open(models_yaml_path, 'r') as file:
                cosmo_models = yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Error parsing models.yaml file: {e}")

        # Get the specified cosmology model
        if self.cosmo_model not in cosmo_models:
            raise ValueError(f"Cosmology model '{self.cosmo_model}' not found in models.yaml. Available models: {list(cosmo_models.keys())}")
        
        # Initialize prior, constraints, and latex labels
        prior = {}
        param_constraints = {}
        latex_labels = []

        model_parameters = cosmo_models[self.cosmo_model]['parameters']
        latex_labels = cosmo_models[self.cosmo_model]['latex_labels']
        
        # Process constraints if provided
        if constraints is not None:
            for constraint in cosmo_models[self.cosmo_model].get('constraints', []):
                if constraint not in constraints:
                    raise ValueError(f"Constraint '{constraint}' required by model '{self.cosmo_model}' not found in constraints")
                param_constraints[constraint] = constraints[constraint]
        
        # Create prior for each parameter in the model
        for param_name in model_parameters:
            if param_name not in parameters:
                raise ValueError(f"Parameter '{param_name}' not found in prior_args.yaml parameters section")
            
            param_config = parameters[param_name]
            
            # Validate parameter configuration
            if 'distribution' not in param_config:
                raise ValueError(f"Parameter '{param_name}' missing 'distribution' section in prior_args.yaml")
            if 'latex' not in param_config:
                raise ValueError(f"Parameter '{param_name}' missing 'latex' section in prior_args.yaml")
            
            dist_config = param_config['distribution']
            
            if dist_config['type'] == 'uniform':
                if 'lower' not in dist_config or 'upper' not in dist_config:
                    raise ValueError(f"Prior distribution for '{param_name}' is missing 'lower' or 'upper' bounds")
                lower = dist_config.get('lower', 0.0)
                upper = dist_config.get('upper', 1.0)
                if lower >= upper:
                    raise ValueError(f"Invalid bounds for '{param_name}': lower ({lower}) must be < upper ({upper})")
                prior[param_name] = dist.Uniform(*torch.tensor([lower, upper], device=self.device))
            else:
                raise ValueError(f"Distribution type '{dist_config['type']}' not supported. Only 'uniform' is currently supported.")
            
            if 'multiplier' in param_config.keys():
                setattr(self, f'{param_name}_multiplier', float(param_config['multiplier']))
        if 'hrdrag' not in model_parameters:
            setattr(self, 'hrdrag_multiplier', 100.0)

        return prior, param_constraints, latex_labels

    def error_func(self, z, D_H, D_M=None):
        if hasattr(self, 'variable_error'):
            z_tensor = torch.as_tensor(z, device=self.device, dtype=torch.float64)
            D_H_scale = _interp1(
                self.variable_error['z'],
                self.variable_error['D_H'],
                z_tensor
            ).to(dtype=D_H.dtype)
            D_H_error = D_H * D_H_scale

            if self.include_D_M:
                D_M_scale = _interp1(
                    self.variable_error['z'],
                    self.variable_error['D_M'],
                    z_tensor
                ).to(dtype=D_M.dtype)
                D_M_error = D_M * D_M_scale
                return D_H_error, D_M_error

            return D_H_error
        else:
            sigma_D_H_tensor = torch.as_tensor(self.sigma_D_H, device=self.device, dtype=D_H.dtype)
            D_H_error = sigma_D_H_tensor.expand_as(D_H)
            if self.include_D_M:
                sigma_D_M_tensor = torch.as_tensor(self.sigma_D_M, device=self.device, dtype=D_M.dtype)
                D_M_error = sigma_D_M_tensor.expand_as(D_M)
                return D_H_error, D_M_error
            return D_H_error

    @profile_method
    def _compute_central_values(self):
        """
        Compute central values (D_H, D_M) at the nominal redshift using fiducial cosmology.
        
        Uses Planck18 fiducial values:
        - Om = 0.3152
        - hrdrag = 99.079 km/s/Mpc
        
        Returns:
            torch.Tensor: Central values for observations as 1D tensor [D_H] or [D_H, D_M]
        """
        # Fiducial cosmology parameters (Planck18)
        fiducial_Om = torch.tensor(0.3152, device=self.device, dtype=torch.float64)
        fiducial_hrdrag = torch.tensor(99.079, device=self.device, dtype=torch.float64)
        
        # Use nominal design redshift (ensure it's properly shaped)
        z_nominal = self.nominal_design.flatten()
        
        # Prepare parameters for D_H_func and D_M_func
        params = {
            'Om': fiducial_Om.unsqueeze(-1),
            'hrdrag': fiducial_hrdrag.unsqueeze(-1)
        }
        
        # Add Ok, w0, wa if they're in the model (use fiducial values)
        if 'Ok' in self.cosmo_params:
            params['Ok'] = torch.tensor(0.0, device=self.device, dtype=torch.float64).unsqueeze(-1)
        if 'w0' in self.cosmo_params:
            params['w0'] = torch.tensor(-1.0, device=self.device, dtype=torch.float64).unsqueeze(-1)
        if 'wa' in self.cosmo_params:
            params['wa'] = torch.tensor(0.0, device=self.device, dtype=torch.float64).unsqueeze(-1)
        
        # Compute D_H at nominal redshift
        D_H_central = self.D_H_func(z_nominal, **params)
        D_H_val = D_H_central.reshape(-1)
        
        if self.include_D_M:
            # Compute D_M at nominal redshift
            D_M_central = self.D_M_func(z_nominal, **params)
            D_M_val = D_M_central.reshape(-1)
            # Interleave [D_H(z_i), D_M(z_i)] for each redshift
            central_vals = torch.stack([D_H_val, D_M_val], dim=-1).reshape(-1)
        else:
            # Just D_H as 1D tensor
            central_vals = D_H_val
        
        if self.global_rank == 0 and self.verbose:
            if z_nominal.numel() == 1:
                print(f"Computed central values at z={z_nominal.item():.2f}:")
            else:
                z_list = [f"{z:.2f}" for z in z_nominal.detach().cpu().tolist()]
                print(f"Computed central values at z={z_list}:")
            print(f"  D_H/r_d = {D_H_val.reshape(-1)[0].item():.4f}")
            if self.include_D_M:
                print(f"  D_M/r_d = {D_M_val.reshape(-1)[0].item():.4f}")
        
        return central_vals

    @profile_method
    def params_to_unconstrained(self, params, bijector_class=None):
        """Vectorized: map PHYSICAL space -> unconstrained R^D."""
        D = len(self.cosmo_params)
        assert params.shape[-1] == D, f"Expected last dim {D}, got {params.shape[-1]}"
        y = params.clone()

        if bijector_class is None:
            bijector_class = self.param_bijector

        # Om in (0,1): use empirical prior transformation
        if self._idx_Om:
            x = params[..., self._idx_Om]
            y[..., self._idx_Om] = bijector_class.prior_to_gaussian(x, 'Om')

        # Ok in (-0.3, 0.3): use empirical prior transformation
        if self._idx_Ok:
            x = params[..., self._idx_Ok]
            y[..., self._idx_Ok] = bijector_class.prior_to_gaussian(x, 'Ok')

        # w0 in (-3, 1): use empirical prior transformation
        if self._idx_w0:
            x = params[..., self._idx_w0]
            y[..., self._idx_w0] = bijector_class.prior_to_gaussian(x, 'w0')

        # wa in (-3, 2): use empirical prior transformation
        if self._idx_wa:
            x = params[..., self._idx_wa]
            y[..., self._idx_wa] = bijector_class.prior_to_gaussian(x, 'wa')

        # hrdrag > 0: use empirical prior transformation
        if self._idx_hr:
            x = params[..., self._idx_hr]
            y[..., self._idx_hr] = bijector_class.prior_to_gaussian(x, 'hrdrag')

        return y

    @profile_method
    def params_from_unconstrained(self, y, bijector_class=None):
        """Vectorized: map unconstrained R^D -> PHYSICAL space."""
        D = len(self.cosmo_params)
        assert y.shape[-1] == D, f"Expected last dim {D}, got {y.shape[-1]}"
        x = y.clone()

        if bijector_class is None:
            bijector_class = self.param_bijector
        
        # Om: use empirical prior inverse transformation
        if self._idx_Om:
            x[..., self._idx_Om] = bijector_class.gaussian_to_prior(y[..., self._idx_Om], 'Om')

        # Ok: use empirical prior inverse transformation
        if self._idx_Ok:
            x[..., self._idx_Ok] = bijector_class.gaussian_to_prior(y[..., self._idx_Ok], 'Ok')

        # w0: use empirical prior inverse transformation
        if self._idx_w0:
            x[..., self._idx_w0] = bijector_class.gaussian_to_prior(y[..., self._idx_w0], 'w0')

        # wa: use empirical prior inverse transformation
        if self._idx_wa:
            x[..., self._idx_wa] = bijector_class.gaussian_to_prior(y[..., self._idx_wa], 'wa')

        # hrdrag: use empirical prior inverse transformation
        if self._idx_hr:
            x[..., self._idx_hr] = bijector_class.gaussian_to_prior(y[..., self._idx_hr], 'hrdrag')

        return x

    # Note: _E_of_z, D_H_func, D_M_func are inherited from CosmologyMixin

    @profile_method
    def sample_valid_parameters(self, sample_shape, prior=None, use_prior_flow=True):
        """Sample parameters from prior with constraints."""
        parameters = {}
        if prior is None:
            prior = self.prior
        
        # Handle constraints based on YAML configuration
        if hasattr(self, 'param_constraints'):
            # Check for valid density constraint
            if 'valid_densities' in self.param_constraints:
                # 0 < Om + Ok < 1
                OmOk_prior = {'Om': prior['Om'], 'Ok': prior['Ok']}
                OmOk_samples = ConstrainedUniform2D(OmOk_prior, **self.param_constraints["valid_densities"]["bounds"]).sample(sample_shape)
                parameters['Om'] = pyro.sample('Om', dist.Delta(OmOk_samples[..., 0])).unsqueeze(-1)
                parameters['Ok'] = pyro.sample('Ok', dist.Delta(OmOk_samples[..., 1])).unsqueeze(-1)
            else:
                # Sample Om, Ok normally if no constraint or Ok not present
                if 'Om' in prior.keys():
                    parameters['Om'] = pyro.sample('Om', prior['Om']).unsqueeze(-1)
                if 'Ok' in prior.keys():
                    parameters['Ok'] = pyro.sample('Ok', prior['Ok']).unsqueeze(-1)
            
            # Check for high z matter domination constraint
            if 'high_z_matter_dom' in self.param_constraints:
                # w0 + wa < 0
                w0wa_prior = {'w0': prior['w0'], 'wa': prior['wa']}
                w0wa_samples = ConstrainedUniform2D(w0wa_prior, **self.param_constraints["high_z_matter_dom"]["bounds"]).sample(sample_shape)
                parameters['w0'] = pyro.sample('w0', dist.Delta(w0wa_samples[..., 0])).unsqueeze(-1)
                parameters['wa'] = pyro.sample('wa', dist.Delta(w0wa_samples[..., 1])).unsqueeze(-1)
            else:
                # Sample w0, wa normally if no constraint or wa not present
                if 'w0' in prior.keys():
                    parameters['w0'] = pyro.sample('w0', prior['w0']).unsqueeze(-1)
                if 'wa' in prior.keys():
                    parameters['wa'] = pyro.sample('wa', prior['wa']).unsqueeze(-1)
        else:
            # If no parameter constraints defined
            if 'Om' in prior.keys():
                parameters['Om'] = pyro.sample('Om', prior['Om']).unsqueeze(-1)
            if 'Ok' in prior.keys():
                parameters['Ok'] = pyro.sample('Ok', prior['Ok']).unsqueeze(-1)
            if 'w0' in prior.keys():
                parameters['w0'] = pyro.sample('w0', prior['w0']).unsqueeze(-1)
            if 'wa' in prior.keys():
                parameters['wa'] = pyro.sample('wa', prior['wa']).unsqueeze(-1)
            
        # Always sample hrdrag if it's in the model
        if 'hrdrag' in prior.keys():
            parameters['hrdrag'] = pyro.sample('hrdrag', prior['hrdrag']).unsqueeze(-1)

        return parameters

    @profile_method
    def pyro_model(self, z):
        """
        Pyro model for generating observations given redshift designs.
        Uses MultivariateNormal with diagonal covariance matrix.
        """
        with pyro.plate_stack("plate", z.shape[:-1]):
            parameters = self.sample_valid_parameters(z.shape[:-1])
            
            # Compute mean predictions for all observations
            D_H_mean = self.D_H_func(z, **parameters)
            if self.include_D_M:
                D_M_mean = self.D_M_func(z, **parameters)
                D_H_error, D_M_error = self.error_func(z, D_H_mean, D_M_mean)
                means_per_z = torch.cat([D_H_mean.unsqueeze(-1), D_M_mean.unsqueeze(-1)], dim=-1)
                sigmas_per_z = torch.cat([D_H_error.unsqueeze(-1), D_M_error.unsqueeze(-1)], dim=-1)
            else:
                D_H_error = self.error_func(z, D_H_mean)
                means_per_z = D_H_mean.unsqueeze(-1)
                sigmas_per_z = D_H_error.unsqueeze(-1)

            means = means_per_z.reshape(means_per_z.shape[:-2] + (-1,))
            sigmas = sigmas_per_z.reshape(sigmas_per_z.shape[:-2] + (-1,))
            covariance_matrix = torch.diag_embed(sigmas ** 2)
            # Sample from MultivariateNormal
            return pyro.sample(self.observation_labels[0], dist.MultivariateNormal(means, covariance_matrix))
                

    @profile_method
    def get_guide_samples(self, guide, context=None, num_samples=5000, params=None, transform_output=True):
        """Sample parameters from the guide (variational distribution)."""
        if context is None:
            context = self.nominal_context
        
        with torch.no_grad():
            param_samples = guide(context.squeeze()).sample((num_samples,))
        
        if self.transform_input and transform_output:
            param_samples = self.params_from_unconstrained(param_samples)
        
        # Apply multiplier for hrdrag if present
        if self._idx_hr and hasattr(self, 'hrdrag_multiplier'):
            param_samples[..., self._idx_hr[0]] *= self.hrdrag_multiplier

        if params is None:
            names = self.cosmo_params
            labels = self.latex_labels
        else:
            param_indices = [self.cosmo_params.index(param) for param in params if param in self.cosmo_params]
            param_samples = param_samples[:, param_indices]
            names = [self.cosmo_params[i] for i in param_indices]
            labels = [self.latex_labels[i] for i in param_indices]
        
        # Check for any constant columns and add tiny noise to prevent getdist from excluding them
        for i in range(param_samples.shape[1]):
            col = param_samples[:, i]
            if torch.all(col == col[0]):
                if self.global_rank == 0:
                    print(f"Column {i} ({names[i]}) is constant with value {col[0]}, adding tiny noise")
                noise_scale = abs(col[0]) * 1e-10 if col[0] != 0 else 1e-10
                param_samples[:, i] = col + torch.randn_like(col) * noise_scale
        
        with contextlib.redirect_stdout(io.StringIO()):
            param_samples_gd = getdist.MCSamples(samples=param_samples.cpu().numpy(), names=names, labels=labels)

        return param_samples_gd
    
    def get_nominal_samples(self, num_samples=100000, params=None, transform_output=False):
        """Get nominal samples for comparison plots."""
        raise NotImplementedError("get_nominal_samples not implemented for variable_redshift")


    @profile_method
    def sample_data(self, designs, num_samples=100, central=False):
        """
        Sample data from the likelihood.
        
        Args:
            designs: redshift values (design variables)
            num_samples: number of data samples to draw
            central: if True, use central values instead of sampling from prior
        
        Returns:
            torch.Tensor: data samples with shape (num_samples, num_observations)
        """
        if central:
            # Use central values with sampling from observation noise only
            # Expand designs for multiple samples
            expanded_designs = lexpand(designs, num_samples)
            
            # Create means equal to central values
            means = self.central_val.unsqueeze(0).expand(num_samples, -1)
            
            # Create diagonal covariance matrix
            if self.include_D_M:
                sigmas = torch.tensor([self.sigma_D_H, self.sigma_D_M], device=self.device, dtype=torch.float64).repeat(self.n_redshifts)
            else:
                sigmas = torch.tensor([self.sigma_D_H], device=self.device, dtype=torch.float64).repeat(self.n_redshifts)
            covariance_matrix = torch.diag(sigmas ** 2)
            
            # Sample from MultivariateNormal centered at central values
            with pyro.plate("data", num_samples):
                data_samples = pyro.sample(self.observation_labels[0], 
                                          dist.MultivariateNormal(means, covariance_matrix))
        else:
            # Sample from full prior (parameters + observations)
            expanded_designs = lexpand(designs, num_samples)
            with torch.no_grad():
                data_samples = self.pyro_model(expanded_designs)
        
        return data_samples
    
    def sample_params_from_data_samples(self, designs, guide, num_data_samples=100, num_param_samples=1000, central=False, transform_output=True):
        """
        Sample parameters from the posterior distribution conditioned on data.
        Vectorized version that batches all sampling operations for improved performance.
        
        Args:
            designs: redshift values (design variables)
            guide: trained guide for sampling parameters
            num_data_samples: number of data realizations to sample
            num_param_samples: number of parameter samples per data realization
            central: if True, use central values for data generation
            transform_output: if True, transform parameters back to physical space
        
        Returns:
            np.ndarray: parameter samples with shape (num_data_samples, num_param_samples, num_params)
        """
        data_samples = self.sample_data(designs, num_data_samples, central)
        
        # Create context: concatenate design and observations
        # data_samples is now always a tensor with shape (num_data_samples, num_observations)
        context = torch.cat([designs.expand(num_data_samples, -1), data_samples], dim=-1)
        
        # Vectorized sampling: expand contexts to (num_data_samples * num_param_samples, context_dim)
        # Each context is repeated num_param_samples times
        # Expand: [num_data_samples, context_dim] -> [num_data_samples, num_param_samples, context_dim] -> [num_data_samples * num_param_samples, context_dim]
        expanded_context = context.unsqueeze(1).expand(-1, num_param_samples, -1).contiguous()
        expanded_context = expanded_context.view(-1, expanded_context.shape[-1])  # [num_data_samples * num_param_samples, context_dim]
        
        # Sample all parameters at once
        with torch.no_grad():
            param_samples = guide(expanded_context).sample(())  # Shape: [num_data_samples * num_param_samples, num_params]
        
        # Apply transformations if needed
        if self.transform_input and transform_output:
            param_samples = self.params_from_unconstrained(param_samples)
        
        # Apply multiplier for hrdrag if present
        if self._idx_hr and hasattr(self, 'hrdrag_multiplier'):
            param_samples[..., self._idx_hr[0]] *= self.hrdrag_multiplier
        
        # Reshape to [num_data_samples, num_param_samples, num_params]
        param_samples = param_samples.view(num_data_samples, num_param_samples, -1)
        
        # Check for any constant columns and add tiny noise to prevent getdist from excluding them
        for i in range(param_samples.shape[2]):
            col = param_samples[:, :, i]
            if torch.all(col == col[0, 0]):
                if self.global_rank == 0:
                    print(f"Column {i} ({self.cosmo_params[i]}) is constant with value {col[0, 0]}, adding tiny noise")
                noise_scale = abs(col[0, 0]) * 1e-10 if col[0, 0] != 0 else 1e-10
                param_samples[:, :, i] = col + torch.randn_like(col) * noise_scale
        
        # Convert to numpy array: [num_data_samples, num_param_samples, num_params]
        param_samples_array = param_samples.cpu().numpy()
        
        return param_samples_array

    def unnorm_lfunc(self, params, features, designs):
        """
        Unnormalized likelihood function for BED bayesdesign package.
        
        This function computes the likelihood without normalization, compatible with
        the BED bayesdesign package's brute force EIG calculations.
        
        Args:
            params (Grid): Grid object containing cosmological parameters (Om, w0, wa, hrdrag, etc.)
                          Must include all parameters required by the cosmology model
            features (Grid): Grid object containing observed features (D_H, optionally D_M)
            designs (Grid): Grid object containing design variables (redshift z)
            
        Returns:
            np.ndarray: Unnormalized likelihood values over the parameter grid
            
        Example:
            For base model (Om, hrdrag):
                params = Grid(Om=np.linspace(0.2, 0.4, 50), 
                             hrdrag=np.linspace(80, 120, 50))
        """
        # Verify required parameters are present
        required_params = self.cosmo_params
        missing_params = [p for p in required_params if p not in params.names]
        if missing_params:
            raise ValueError(
                f"Missing required parameters: {missing_params}\n"
                f"Model '{self.cosmo_model}' requires: {required_params}\n"
                f"Got params with: {list(params.names)}\n"
                f"Please create Grid with all required parameters."
            )
        
        # Extract parameters and convert to tensors with proper shape for D_H_func
        # Grid values need .unsqueeze(-1) to have shape (..., 1) for plate broadcasting
        parameters = {}
        for key in params.names:
            param_array = torch.tensor(getattr(params, key), device=self.device, dtype=torch.float64)
            # Add trailing dimension for plate system
            parameters[key] = param_array.unsqueeze(-1)
        
        # Extract redshift from designs
        print(getattr(designs, designs.names[0]).shape)
        z_array = torch.tensor(getattr(designs, designs.names[0]), device=self.device, dtype=torch.float64)
        # z also needs proper shape for broadcasting
        z = z_array.unsqueeze(-1)
        
        print(z.shape)
        # Compute D_H mean and likelihood
        D_H_mean = self.D_H_func(z, **parameters)
        
        # Extract feature values (convert to numpy for comparison)
        D_H_obs = getattr(features, features.names[0])
        D_H_diff = D_H_obs - D_H_mean.cpu().numpy()
        likelihood = np.exp(-0.5 * (D_H_diff / self.sigma_D_H) ** 2)
        
        # If including D_M, add its contribution
        if self.include_D_M:
            D_M_mean = self.D_M_func(z, **parameters)
            D_M_obs = getattr(features, features.names[1])
            D_M_diff = D_M_obs - D_M_mean.cpu().numpy()
            D_M_likelihood = np.exp(-0.5 * (D_M_diff / self.sigma_D_M) ** 2)
            likelihood = likelihood * D_M_likelihood
        
        return likelihood
