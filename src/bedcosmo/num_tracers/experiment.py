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
from mpl_toolkits.mplot3d import Axes3D
import contextlib
import io
import getdist
import math
import inspect
from bedcosmo.util import (
    load_prior_flow_from_file, Bijector, auto_seed, profile_method,
    load_nominal_samples, get_experiment_config_path,
)
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
from bedcosmo.custom_dist import ConstrainedUniform2D
from bedcosmo.base import BaseExperiment
from bedcosmo.cosmology import CosmologyMixin, _infer_plate_shape

storage_path = os.environ["SCRATCH"] + "/bedcosmo/num_tracers"
home_dir = os.environ["HOME"]
mlflow.set_tracking_uri(storage_path + "/mlruns")


class NumTracers(BaseExperiment, CosmologyMixin):
    def __init__(
        self,
        prior_args=None,
        design_args=None,
        dataset="dr2",
        analysis="bao",
        cosmo_model="base",
        flow_type="MAF",
        nominal_design=None,
        include_D_M=True,
        include_D_V=True,
        bijector_state=None,
        seed=None,
        global_rank=0,
        transform_input=False,
        device="cuda:0",
        mode='train',
        verbose=False,
        profile=False,
        prior_flow_batch_size=10000,
        prior_flow_cache_size=100000,
        prior_flow_use_cache=True,
        fullshape_mocks=None,
        fullshape_covariance=None,
        fullshape_k_bins=None,
        fullshape_z_eff=None,
        fullshape_ells=(0, 2, 4)
    ):

        self.name = 'num_tracers'
        self.dataset = dataset
        self.analysis = analysis
        self.desi_data = pd.read_csv(os.path.join(home_dir, f"data/desi/bao_{self.dataset}", 'desi_data.csv'))
        self.desi_tracers = pd.read_csv(os.path.join(home_dir, f"data/desi/bao_{self.dataset}", 'desi_tracers.csv'))
        self.nominal_cov = np.load(os.path.join(home_dir, f"data/desi/bao_{self.dataset}", 'desi_cov.npy'))
        self.DH_idx = np.where(self.desi_data["quantity"] == "DH_over_rs")[0]
        self.DM_idx = np.where(self.desi_data["quantity"] == "DM_over_rs")[0]
        self.DV_idx = np.where(self.desi_data["quantity"] == "DV_over_rs")[0]
        self.cosmo_model = cosmo_model
        self.flow_type = flow_type
        self.device = device
        self.mode = mode
        self.global_rank = global_rank  
        self.seed = seed
        self.verbose = verbose
        self.profile = profile
        self.prior_flow_batch_size = prior_flow_batch_size
        self.prior_flow_cache_size = prior_flow_cache_size
        self.prior_flow_use_cache = prior_flow_use_cache
        if seed is not None:
            auto_seed(self.seed)
        self.rdrag = 149.77
        self.c = constants.c.to('km/s').value
        self.corr_matrix = torch.tensor(self.nominal_cov/np.sqrt(np.outer(np.diag(self.nominal_cov), np.diag(self.nominal_cov))), device=self.device)
        self.include_D_M = include_D_M
        self.include_D_V = include_D_V
        self.efficiency = torch.tensor(self.desi_data.drop_duplicates(subset=['tracer'])['efficiency'].tolist(), device=self.device)
        self.sigmas = torch.tensor(self.desi_data["std"].tolist(), device=self.device)
        self.central_val = torch.tensor(self.desi_data["value_at_z"].tolist(), device=self.device)
        self.tracer_bins = self.desi_data.drop_duplicates(subset=['tracer'])['tracer'].tolist()
        self.nominal_total_obs = int(self.desi_data.drop_duplicates(subset=['tracer'])['observed'].sum())
        self.nominal_passed_ratio = torch.tensor(self.desi_data['passed'].tolist(), device=self.device)/self.nominal_total_obs
        # Create dictionary with upper limits and lower limit lists for each class
        # Extract labels from design_args if provided, otherwise use default
        if design_args is not None and 'labels' in design_args:
            self.design_labels = design_args['labels']
        else:
            self.design_labels = ["BGS", "LRG", "ELG", "QSO"]
        self.num_targets = self.desi_tracers.groupby('class').sum()['targets'].reindex(self.design_labels)
        if nominal_design is None:
            self.nominal_design = torch.tensor(self.desi_tracers.groupby('class').sum()['observed'].reindex(self.design_labels).values, device=self.device)
        else:
            self.nominal_design = nominal_design
        self.nominal_context = torch.cat([
            self.nominal_design, 
            self.central_val if self.include_D_M else self.central_val[1::2]
            ], dim=-1)
        # Compute context_dim dynamically from the actual nominal_context size
        # This ensures it matches the actual data structure regardless of flags
        self.context_dim = self.nominal_context.shape[-1]
        
        # initialize the prior
        self.prior_args = prior_args
        

        self.prior, self.param_constraints, self.latex_labels, self.prior_flow, self.prior_flow_metadata = self.init_prior(**self.prior_args)
        
        # Extract prior_flow settings for use in _sample_prior_flow
        if self.prior_flow_metadata is not None:
            self.prior_flow_transform_input = self.prior_flow_metadata.get('transform_input', False)
            self.prior_flow_nominal_context = self.prior_flow_metadata.get('nominal_context', self.nominal_context)
        else:
            self.prior_flow_transform_input = False
            self.prior_flow_nominal_context = None
        
        # Load DESI prior from the default location
        desi_prior_path = os.path.join(home_dir, f"data/desi/bao_{self.dataset}", 'prior_args.yaml')
        if os.path.exists(desi_prior_path):
            with open(desi_prior_path, 'r') as file:
                desi_prior_data = yaml.safe_load(file)
            self.desi_prior, _, _, _, _ = self.init_prior(**desi_prior_data)
        else:
            # If DESI prior not found, use the same as the main prior
            self.desi_prior = self.prior
        self.cosmo_params = list(self.prior.keys())
        
        # Determine which bijector_state to use:
        # 1. If bijector_state argument is provided (resume training), use that
        # 2. Else if prior_flow has bijector_state (using prior_flow as prior), use that
        # 3. Else create new bijector by sampling
        effective_bijector_state = bijector_state
        bijector_source = "checkpoint"
        
        if effective_bijector_state is None and self.prior_flow_metadata is not None:
            prior_flow_bijector_state = self.prior_flow_metadata.get('bijector_state')
            if prior_flow_bijector_state is not None:
                effective_bijector_state = prior_flow_bijector_state
                bijector_source = "prior_flow"
        
        skip_bijector_sampling = effective_bijector_state is not None
        self.param_bijector = Bijector(self, cdf_bins=5000, cdf_samples=1e5, skip_sampling=skip_bijector_sampling)
        if effective_bijector_state is not None:
            if self.global_rank == 0:
                print(f"Restoring bijector state from {bijector_source}.")
            self.param_bijector.set_state(effective_bijector_state, device=self.device)
        # if the prior is not the same as the DESI prior, create a new bijector for the DESI samples
        if self.prior.items() != self.desi_prior.items():
            # desi_bijector uses the uniform desi_prior, not the prior_flow
            self.desi_bijector = Bijector(self, prior=self.desi_prior, cdf_bins=5000, cdf_samples=1e6,
                                         use_prior_flow=False)
        else:
            self.desi_bijector = self.param_bijector

        self.transform_input = transform_input
        # store as Python lists of ints; works for advanced indexing
        self._idx_Om = [self.cosmo_params.index('Om')] if 'Om' in self.cosmo_params else []
        self._idx_Ok = [self.cosmo_params.index('Ok')] if 'Ok' in self.cosmo_params else []
        self._idx_w0 = [self.cosmo_params.index('w0')] if 'w0' in self.cosmo_params else []
        self._idx_wa = [self.cosmo_params.index('wa')] if 'wa' in self.cosmo_params else []
        self._idx_hr = [self.cosmo_params.index('hrdrag')] if 'hrdrag' in self.cosmo_params else []
        self.observation_labels = ["y"]
        
        # Pass design_args using ** unpacking if provided, otherwise use defaults
        if design_args is not None:
            self.init_designs(**design_args)
        else:
            self.init_designs()
        
        # Initialize full shape data if provided
        if fullshape_mocks is not None:
            self.fullshape_covariance = self.compute_covariance_from_mocks(fullshape_mocks)
        elif fullshape_covariance is not None:
            if isinstance(fullshape_covariance, np.ndarray):
                self.fullshape_covariance = torch.tensor(fullshape_covariance, device=self.device, dtype=torch.float64)
            else:
                self.fullshape_covariance = fullshape_covariance.to(self.device)
        else:
            self.fullshape_covariance = None
        
        if fullshape_k_bins is not None:
            if isinstance(fullshape_k_bins, np.ndarray):
                self.fullshape_k_bins = torch.tensor(fullshape_k_bins, device=self.device, dtype=torch.float64)
            else:
                self.fullshape_k_bins = fullshape_k_bins.to(self.device)
        else:
            self.fullshape_k_bins = None
        
        if fullshape_z_eff is not None:
            if isinstance(fullshape_z_eff, (int, float)):
                self.fullshape_z_eff = torch.tensor(fullshape_z_eff, device=self.device, dtype=torch.float64)
            elif isinstance(fullshape_z_eff, np.ndarray):
                self.fullshape_z_eff = torch.tensor(fullshape_z_eff, device=self.device, dtype=torch.float64)
            else:
                self.fullshape_z_eff = fullshape_z_eff.to(self.device)
        else:
            self.fullshape_z_eff = None
        
        self.fullshape_ells = fullshape_ells

    @profile_method
    def init_designs(
        self, input_designs_path=None, step=0.05, lower=0.05, upper=None, 
        sum_lower=1.0, sum_upper=1.0, tol=1e-3, labels=None, input_type="variable"):
        """
        Initialize design space.
        
        Args:
            input_designs_path: Path to numpy file containing designs
            step: Step size(s) for design grid (default: 0.05)
            lower: Lower bound(s) for each design variable (default: 0.05)
            upper: Upper bound(s) for each design variable (default: None)
            sum_lower: Lower bound on sum of design variables (default: 1.0)
            sum_upper: Upper bound on sum of design variables (default: 1.0)
            tol: Tolerance for sum constraint (default: 1e-3),
            labels: Labels for design variables (default: None)
            input_type: Type of input designs ("nominal" or "variable")

        """
        # Check if input_type is "nominal"
        if input_type == "nominal":
            designs = self.nominal_design.unsqueeze(0)  # Add batch dimension
        elif input_type == "variable":
            # If input_designs_path is provided, load from path (assumed to be absolute)
            if input_designs_path is not None:
                if not os.path.isabs(input_designs_path):
                    raise ValueError(f"input_designs_path must be an absolute path, got: {input_designs_path}")
                if not os.path.exists(input_designs_path):
                    raise FileNotFoundError(f"input_designs_path not found: {input_designs_path}")
                
                input_designs_array = np.load(input_designs_path)
                # Convert to tensor
                designs = torch.tensor(input_designs_array, device=self.device, dtype=torch.float64)
                # Handle 1D input (single design)
                if designs.ndim == 1:
                    if len(designs) != len(labels):
                        raise ValueError(f"Input design must have {len(labels)} values, got {len(designs)}")
                    designs = designs.reshape(1, -1)
                elif designs.ndim == 2:
                    if designs.shape[1] != len(labels):
                        raise ValueError(f"Input design must have {len(labels)} columns, got {designs.shape[1]}")
                else:
                    raise ValueError(f"Input design must be 1D or 2D, got shape {designs.shape}")
            else:
                # Generate design grid
                if type(step) == float:
                    design_steps = [step]*len(labels)
                elif type(step) == list:
                    design_steps = step
                else:
                    raise ValueError("step must be a float or list")
                
                if type(lower) == float:
                    lower_limits = [lower]*len(labels)
                elif type(lower) == list:
                    lower_limits = lower
                else:
                    raise ValueError("lower must be a float or list")
                
                if upper is None:
                    upper_limits = [self.num_targets[target] / self.nominal_total_obs for target in labels]
                elif type(upper) == float:
                    upper_limits = [upper]*len(labels)
                elif type(upper) == list:
                    upper_limits = upper
                else:
                    raise ValueError("upper must be a float or list")
                    
                designs_dict = {
                    f'N_{target}': np.arange(
                        lower_limits[i],
                        upper_limits[i],
                        design_steps[i]
                    ) for i, target in enumerate(labels)
                }
                
                # Create constrained grid based on sum_lower and sum_upper
                if sum_lower is None and sum_upper is None:
                    # No constraint on sum
                    grid_designs = Grid(**designs_dict)
                elif sum_lower is not None and sum_upper is not None:
                    if abs(sum_lower - sum_upper) < tol:
                        # Equal bounds: sum must equal this value
                        target_sum = sum_lower
                        grid_designs = Grid(
                            **designs_dict,
                            constraint=lambda **kwargs: np.abs(sum(kwargs.values()) - target_sum) < tol
                        )
                    else:
                        # Range: sum must be between lower and upper
                        grid_designs = Grid(
                            **designs_dict,
                            constraint=lambda **kwargs: ((sum(kwargs.values()) >= sum_lower - tol) & 
                                                        (sum(kwargs.values()) <= sum_upper + tol))
                        )
                elif sum_lower is not None:
                    # Only lower bound: sum >= lower
                    grid_designs = Grid(
                        **designs_dict,
                        constraint=lambda **kwargs: sum(kwargs.values()) >= (sum_lower - tol)
                    )
                else:
                    # Only upper bound: sum <= upper
                    grid_designs = Grid(
                        **designs_dict,
                        constraint=lambda **kwargs: sum(kwargs.values()) <= (sum_upper + tol)
                    )

                designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=self.device).unsqueeze(1)
                for name in grid_designs.names[1:]:
                    design_tensor = torch.tensor(getattr(grid_designs, name).squeeze(), device=self.device).unsqueeze(1)
                    designs = torch.cat((designs, design_tensor), dim=1)
                
        self.designs = designs.to(self.device)
    
    @profile_method
    def init_prior(
        self,
        parameters,
        constraints=None,
        prior_flow_path=None,
        prior_run_id=None,
        **kwargs
    ):
        """
        Load cosmological prior and constraints from prior arguments.
        
        This function dynamically constructs the prior and latex labels based on the
        specified cosmology model in the YAML file. It supports uniform distributions
        and handles parameter constraints like Om + Ok < 1 and w0 + wa < 0.
        
        Args:
            parameters (dict): Dictionary defining each parameter with distribution type and bounds.
                Each parameter should have:
                - distribution: dict with 'type' ('uniform'), 'lower', 'upper'
                - latex: LaTeX label for the parameter
                - multiplier: optional multiplier for the parameter
            constraints (dict, optional): Dictionary defining parameter constraints.
                Keys are constraint names, values are dicts with 'affected_parameters' and 'bounds'.
            prior_flow_path (str, optional): Absolute path to prior flow checkpoint file.
                Must be an absolute path. Required if using a trained posterior as prior.
            prior_run_id (str, optional): MLflow run ID for prior flow metadata.
                Required if prior_flow is specified.
            **kwargs: Additional arguments (ignored, for compatibility with YAML structure).
        
        Returns:
            tuple: (prior, param_constraints, latex_labels, prior_flow, prior_flow_metadata)
                - prior: Dictionary of parameter distributions
                - param_constraints: Dictionary of parameter constraints
                - latex_labels: List of LaTeX labels for parameters
                - prior_flow: Loaded prior flow model or None
                - prior_flow_metadata: Metadata dict from prior flow or None
        """
        try:
            models_yaml_path = get_experiment_config_path('num_tracers', 'models.yaml')
            with open(models_yaml_path, 'r') as file:
                cosmo_models = yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Error parsing models.yaml file: {e}")

        # Load prior flow if specified
        if prior_flow_path:
            if not os.path.isabs(prior_flow_path):
                raise ValueError(
                    f"prior_flow path '{prior_flow_path}' must be an absolute path. "
                    "Relative paths are not supported."
                )

            if prior_run_id is None:
                raise ValueError("prior_run_id must be specified when using prior_flow_path")

            prior_flow, prior_flow_metadata = load_prior_flow_from_file(
                prior_flow_path,
                prior_run_id,
                self.device,
                self.global_rank
            )
            prior_flow_metadata['nominal_context'] = self.nominal_context.to(self.device) if prior_flow_metadata['nominal_context'] is None else prior_flow_metadata['nominal_context']

            if self.global_rank == 0:
                print("Using trained posterior model as prior for parameter sampling (loaded from prior_args.yaml)")
                print(f"  Metadata loaded from MLflow run_id: {prior_run_id}")
        else:
            prior_flow = None
            prior_flow_metadata = None

        # Get the specified cosmology model
        if self.cosmo_model not in cosmo_models[self.analysis]:
            raise ValueError(f"Cosmology model '{self.cosmo_model}' not found in cosmo_models.yaml. Available models: {list(cosmo_models.keys())}")
        
        # Initialize prior, constraints, and latex labels
        prior = {}
        param_constraints = {}
        latex_labels = []

        model_parameters = cosmo_models[self.analysis][self.cosmo_model]['parameters']
        latex_labels = cosmo_models[self.analysis][self.cosmo_model]['latex_labels']
        
        # Process constraints if provided
        if constraints is not None:
            for constraint in cosmo_models[self.analysis][self.cosmo_model].get('constraints', []):
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
            elif dist_config['type'] == 'gaussian':
                if 'loc' not in dist_config or 'scale' not in dist_config:
                    raise ValueError(f"Prior distribution for '{param_name}' is missing 'loc' or 'scale' bounds")
                loc = dist_config.get('loc', 0.0)
                scale = dist_config.get('scale', 1.0)
                if scale <= 0:
                    raise ValueError(f"Invalid scale for '{param_name}': scale ({scale}) must be > 0")
                prior[param_name] = dist.Normal(torch.tensor(loc, device=self.device), 
                                                torch.tensor(scale, device=self.device))
            else:
                raise ValueError(f"Distribution type '{dist_config['type']}' not supported. Supported types: 'uniform', 'gaussian'.")
            
            if 'multiplier' in param_config.keys():
                setattr(self, f'{param_name}_multiplier', float(param_config['multiplier']))

        return prior, param_constraints, latex_labels, prior_flow, prior_flow_metadata

    @profile_method
    def params_to_unconstrained(self, params, bijector_class=None):
        """
        Vectorized: map PHYSICAL space -> unconstrained R^D.
        Expected input shape: (..., D) where D == len(self.cosmo_params).

        Returns:
            y (torch.Tensor): The unconstrained parameters.
        """
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
        """
        Vectorized: map unconstrained R^D -> PHYSICAL space.
        Expected input shape: (..., D) where D == len(self.cosmo_params).

        Returns:
            x (torch.Tensor): The physical parameters.
        """
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
            
    @profile_method
    def sigma_scaling_factor(self, passed_ratio, class_ratio, index):
        """
        Calculate the scaling factor for likelihood uncertainties based on the distribution of observations 
        across tracers (passed_ratio) and the total number of observations (encoded in the sum of class_ratio).

        The total_obs_multiplier is inferred from the sum of class_ratio. If class_ratio (design var) sums to 1.4, 
        then the total_obs_multiplier is 1.4.

        Args:
            passed_ratio: Fraction of passed objects in each tracer bin relative to total obs
            class_ratio: Design variables (fraction allocated to each class)
            index: Index for the specific measurement
        
        Returns:
            Scaling factor for sigma: sigma_new = sigma_nominal * scaling_factor
        
        """
        total_obs_multiplier = class_ratio.sum(dim=-1, keepdim=True)  # sum across classes
        return torch.sqrt(self.nominal_passed_ratio[index] / (total_obs_multiplier * passed_ratio[..., index]))

    @profile_method
    def calc_passed(self, class_ratio):
        if type(class_ratio) == torch.Tensor:
            assert class_ratio.shape[-1] == len(self.design_labels), f"class_ratio should have {len(self.design_labels)} columns"
            obs_ratio = torch.zeros((class_ratio.shape[0], class_ratio.shape[1], len(self.desi_data)), device=self.device)

            # multiply each class ratio by the observed fraction in each tracer bin
            BGSs = self.desi_tracers.loc[self.desi_tracers["class"] == "BGS"]["observed"]
            BGS_dist = class_ratio[..., 0].unsqueeze(-1) * torch.tensor((BGSs / BGSs.sum()).values, device=self.device).unsqueeze(0)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "BGS")[0]] = BGS_dist[..., 0].unsqueeze(-1)

            LRGs = self.desi_tracers.loc[self.desi_tracers["class"] == "LRG"]["observed"]
            LRG_dist = class_ratio[..., 1].unsqueeze(-1) * torch.tensor((LRGs / LRGs.sum()).values, device=self.device).unsqueeze(0)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "LRG1")[0]] = LRG_dist[..., 0].unsqueeze(-1)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "LRG2")[0]] = LRG_dist[..., 1].unsqueeze(-1)

            ELGs = self.desi_tracers.loc[self.desi_tracers["class"] == "ELG"]["observed"]
            ELG_dist = class_ratio[..., 2].unsqueeze(-1) * torch.tensor((ELGs / ELGs.sum()).values, device=self.device).unsqueeze(0)
            # add the last value in LRG_dist to the first value in ELG_dist to get LRG3+ELG1
            obs_ratio[..., np.where(self.desi_data["tracer"] == "LRG3+ELG1")[0]] = (LRG_dist[..., 2] + ELG_dist[..., 0]).unsqueeze(-1)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "ELG2")[0]] = ELG_dist[..., 1].unsqueeze(-1)

            QSOs = self.desi_tracers.loc[self.desi_tracers["class"] == "QSO"]["observed"]
            QSO_dist = class_ratio[..., 3].unsqueeze(-1) * torch.tensor((QSOs / QSOs.sum()).values, device=self.device).unsqueeze(0)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "QSO")[0]] = QSO_dist[..., 0].unsqueeze(-1)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "Lya QSO")[0]] = QSO_dist[..., 1].unsqueeze(-1)

            efficiency = torch.zeros((class_ratio.shape[0], class_ratio.shape[1], len(self.desi_data)), device=self.device)
            efficiency[..., np.where(self.desi_data["tracer"] == "BGS")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "BGS", "efficiency"].values[0], device=self.device).expand_as(BGS_dist[..., 0]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "LRG1")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG1", "efficiency"].values[0], device=self.device).expand_as(LRG_dist[..., 0]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "LRG2")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG2", "efficiency"].values[0], device=self.device).expand_as(LRG_dist[..., 1]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "LRG3+ELG1")[0]] = ((LRG_dist[..., 2] / (LRG_dist[..., 2] + ELG_dist[..., 0])) * self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG3", "efficiency"].values[0] + (ELG_dist[..., 0] / (LRG_dist[..., 2] + ELG_dist[..., 0])) * self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG1", "efficiency"].values[0]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "ELG2")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG2", "efficiency"].values[0], device=self.device).expand_as(ELG_dist[..., 1]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "QSO")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "QSO", "efficiency"].values[0], device=self.device).expand_as(QSO_dist[..., 0]).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "Lya QSO")[0]] = torch.tensor(self.desi_tracers.loc[self.desi_tracers["tracer"] == "Lya QSO", "efficiency"].values[0], device=self.device).expand_as(QSO_dist[..., 1]).unsqueeze(-1)

            # scale obs_ratio by efficiency to get the number of passed objects
            passed_ratio = obs_ratio*efficiency

            return passed_ratio
        elif type(class_ratio) == Grid:
            obs_ratio = np.zeros((5,) + len(class_ratio.shape)*(1,))

            LRGs = self.desi_tracers.loc[self.desi_tracers["class"] == "LRG"]["observed"]
            LRG_dist = (class_ratio.N_LRG * np.array((LRGs / LRGs.sum()).values)).squeeze()
            
            obs_ratio[0:2, ...] = LRG_dist[0:2]

            ELGs = self.desi_tracers.loc[self.desi_tracers["class"] == "ELG"]["observed"]
            ELG_dist = (class_ratio.N_ELG * np.array((ELGs / ELGs.sum()).values)).squeeze()
            obs_ratio[..., 2] = (LRG_dist[..., 2] + ELG_dist[..., 0])
            obs_ratio[..., 3] = ELG_dist[..., 1]

            QSOs = self.desi_tracers.loc[self.desi_tracers["class"] == "QSO"]["observed"]
            QSO_dist = (class_ratio.N_QSO * np.array((QSOs / QSOs.sum()).values)).squeeze()
            obs_ratio[..., 4] = QSO_dist[..., 0]

            efficiency = np.stack([
                np.array(self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG1", "efficiency"].values[0]),
                np.array(self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG2", "efficiency"].values[0]),
                (LRG_dist[..., 2] / (LRG_dist[..., 2] + ELG_dist[..., 0])) * self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG3", "efficiency"].values[0] + (ELG_dist[..., 0] / (LRG_dist[..., 2] + ELG_dist[..., 0])) * self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG1", "efficiency"].values[0],
                np.array(self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG2", "efficiency"].values[0]),
                np.array(self.desi_tracers.loc[self.desi_tracers["tracer"] == "Lya QSO", "efficiency"].values[0])
            ], axis=-1)

            passed_ratio = obs_ratio*efficiency

            return passed_ratio

    # Note: _E_of_z, D_H_func, D_M_func, D_V_func are inherited from CosmologyMixin

    @profile_method
    def get_guide_samples(self, guide, context=None, num_samples=5000, params=None, transform_output=True):
        """
        Samples parameters from the guide (variational distribution).
        Args:
            guide (pyro.infer.guide.Guide): The guide to sample from.
            context (torch.Tensor): The context to sample from.
            num_samples (int): The number of parameter samples to draw.
        """
        if context is None:
            context = self.nominal_context
        with torch.no_grad():
            param_samples = guide(context.squeeze()).sample((num_samples,))
        if self.transform_input and transform_output:
            param_samples = self.params_from_unconstrained(param_samples)
        param_samples[..., -1] *= self.hrdrag_multiplier

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
                print(f"Column {i} ({names[i]}) is constant with value {col[0]}, adding tiny noise")
                # Add tiny noise to make it non-constant (1e-10 times the value)
                noise_scale = abs(col[0]) * 1e-10
                param_samples[:, i] = col + torch.randn_like(col) * noise_scale
        
        with contextlib.redirect_stdout(io.StringIO()):
            param_samples_gd = getdist.MCSamples(samples=param_samples.cpu().numpy(), names=names, labels=labels)

        return param_samples_gd
    
    def get_nominal_samples(self, num_samples=100000, params=None, transform_output=False):
        param_samples, target_labels, latex_labels = load_nominal_samples('num_tracers', self.cosmo_model, dataset=self.dataset)
        param_samples = param_samples[:num_samples]
        if transform_output:
            param_samples = torch.tensor(param_samples, device=self.device)
            param_samples[..., -1] /= 100 # to get hrdrag in units of 100 km/s/Mpc
            param_samples = self.params_to_unconstrained(param_samples, bijector_class=self.desi_bijector).cpu().numpy()
        
        if params is None:
            names = target_labels
            labels = latex_labels
        else:
            param_indices = [target_labels.index(param) for param in params if param in target_labels]
            param_samples = param_samples[:, param_indices]
            names = [target_labels[i] for i in param_indices]
            labels = [latex_labels[i] for i in param_indices]

        with contextlib.redirect_stdout(io.StringIO()):
            desi_samples_gd = getdist.MCSamples(samples=param_samples, names=names, labels=labels)

        return desi_samples_gd


    @profile_method
    def sample_data(self, tracer_ratio, num_samples=100, central=True):
        """
        Samples data from the likelihood.
        Args:
            tracer_ratio (torch.Tensor): The tracer ratio (design variables).
            num_samples (int): The number of data samples to draw.
            central (bool): Whether to use the fixed central value of the data samples.
        
        """
        if central:
            # Expand tracer_ratio for batching with num_samples
            expanded_tracer_ratio = lexpand(tracer_ratio, num_samples)
            passed_ratio = self.calc_passed(expanded_tracer_ratio)
            means = torch.zeros(passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)
            # Broadcast central values across batch dimensions
            means[:, :, self.DH_idx] = self.central_val[self.DH_idx]
            rescaled_sigmas = torch.zeros(passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)
            rescaled_sigmas[:, :, self.DH_idx] = self.sigmas[self.DH_idx] * self.sigma_scaling_factor(passed_ratio, expanded_tracer_ratio, self.DH_idx)
            if self.include_D_M:
                means[:, :, self.DM_idx] = self.central_val[self.DM_idx]
                rescaled_sigmas[:, :, self.DM_idx] = self.sigmas[self.DM_idx] * self.sigma_scaling_factor(passed_ratio, expanded_tracer_ratio, self.DM_idx)
            if self.include_D_V:
                means[:, :, self.DV_idx] = self.central_val[self.DV_idx]
                rescaled_sigmas[:, :, self.DV_idx] = self.sigmas[self.DV_idx] * self.sigma_scaling_factor(passed_ratio, expanded_tracer_ratio, self.DV_idx)

            if self.include_D_V and self.include_D_M:
                covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
            else:
                covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-1) * rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-2))

            with pyro.plate("data", num_samples):
                data_samples = pyro.sample(self.observation_labels[0], dist.MultivariateNormal(means.squeeze(), covariance_matrix.squeeze())).unsqueeze(1)
        else:
            data_samples = self.pyro_model(tracer_ratio)
        return data_samples
    
    def sample_params_from_data_samples(self, tracer_ratio, guide, num_data_samples=100, num_param_samples=1000, central=True, transform_output=True):
        """
        Samples parameters from the posterior distribution conditioned on the data sampled from the likelihood.
        Vectorized version that batches all sampling operations for improved performance.
        Args:
            tracer_ratio (torch.Tensor): The tracer ratio (design variables).
            guide (pyro.infer.guide.Guide): The guide to sample from.
            num_data_samples (int): The number of data samples to use.
            num_param_samples (int): The number of parameter samples to draw.
            central (bool): Whether to use the fixed central value of the data samples.
        
        """
        data_samples = self.sample_data(tracer_ratio, num_data_samples, central)
        # Expand tracer_ratio to match data_samples shape for concatenation
        # tracer_ratio is [1, 4], need to expand to [num_data_samples, 1, 4]
        if tracer_ratio.dim() == 2:
            expanded_tracer_ratio = tracer_ratio.unsqueeze(0).expand(num_data_samples, -1, -1)
        else:
            expanded_tracer_ratio = tracer_ratio.expand(num_data_samples, -1, -1)
        context = torch.cat([expanded_tracer_ratio, data_samples], dim=-1)
        
        # Vectorized sampling: expand contexts to (num_data_samples * num_param_samples, context_dim)
        # Each context is repeated num_param_samples times
        context_squeezed = context.squeeze(1) if context.dim() == 3 else context  # Shape: [num_data_samples, context_dim]
        # Expand: [num_data_samples, context_dim] -> [num_data_samples, num_param_samples, context_dim] -> [num_data_samples * num_param_samples, context_dim]
        expanded_context = context_squeezed.unsqueeze(1).expand(-1, num_param_samples, -1).contiguous()
        expanded_context = expanded_context.view(-1, expanded_context.shape[-1])  # [num_data_samples * num_param_samples, context_dim]
        
        # Sample all parameters at once
        with torch.no_grad():
            param_samples = guide(expanded_context).sample(())  # Shape: [num_data_samples * num_param_samples, num_params]
        
        # Apply transformations if needed
        if self.transform_input and transform_output:
            param_samples = self.params_from_unconstrained(param_samples)
        param_samples[..., -1] *= self.hrdrag_multiplier
        
        # Reshape to [num_data_samples, num_param_samples, num_params]
        param_samples = param_samples.view(num_data_samples, num_param_samples, -1)
        
        # Check for any constant columns and add tiny noise to prevent getdist from excluding them
        for i in range(param_samples.shape[2]):
            col = param_samples[:, :, i]
            if torch.all(col == col[0, 0]):
                if self.global_rank == 0:
                    print(f"Column {i} ({self.cosmo_params[i]}) is constant with value {col[0, 0]}, adding tiny noise")
                noise_scale = abs(col[0, 0]) * 1e-10
                param_samples[:, :, i] = col + torch.randn_like(col) * noise_scale
        
        # Convert to numpy array: [num_data_samples, num_param_samples, num_params]
        param_samples_array = param_samples.cpu().numpy()
        
        return param_samples_array

    def sample_brute_force(self, tracer_ratio, grid_designs, grid_features, grid_params, designer, num_data_samples=100, num_param_samples=1000):
        
        rescaled_sigmas = torch.zeros(self.sigmas.shape, device=self.device)
        rescaled_sigmas[self.DH_idx] = self.sigmas[self.DH_idx] * torch.sqrt((self.efficiency[self.DH_idx]*tracer_ratio)/self.nominal_passed_ratio[self.DH_idx])
        if self.include_D_M:
            means = self.central_val
            rescaled_sigmas[self.DM_idx] = self.sigmas[self.DM_idx] * torch.sqrt((self.efficiency[self.DM_idx]*tracer_ratio)/self.nominal_passed_ratio[self.DM_idx])
            covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
        else:
            means = self.central_val[self.DH_idx]
            covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (rescaled_sigmas[self.DH_idx].unsqueeze(-1) * rescaled_sigmas[self.DH_idx].unsqueeze(-2))

        with pyro.plate("data", num_data_samples):
            data_samples = pyro.sample(self.observation_labels[0], dist.MultivariateNormal(means, covariance_matrix))
            
        post_samples = []
        post_input = {}
        for j in range(num_data_samples):
            for i, k in enumerate(grid_designs.names):
                post_input[k] = tracer_ratio[i].item()
            for i, k in enumerate(grid_features.names):
                post_input[k] = data_samples[j, i].item()
            posterior_pdf = designer.get_posterior(**post_input)
            param_samples = []
            flat_pdf = posterior_pdf.flatten()
            indices = np.array(list(np.ndindex(posterior_pdf.shape)))
            sampled_indices = np.random.choice(len(indices), size=num_param_samples, p=flat_pdf)
            indices = indices[sampled_indices]
            param_samples = []
            param_mesh = np.stack(np.meshgrid(*[getattr(grid_params, grid_params.names[i]).squeeze() for i in range(len(grid_params.names))], indexing='ij'), axis=-1)
            for i in range(num_param_samples):
                param_samples.append(param_mesh[tuple(indices[i])])
            post_samples.append(param_samples)
        param_samples = torch.tensor(np.array(post_samples), device=self.device)
        return param_samples.reshape(-1, len(grid_params.names))

    def brute_force_posterior(self, tracer_ratio, designer, grid_params, num_param_samples=1000):
        
        with pyro.plate("plate", num_param_samples):
            parameters = {}
            for i, (k, v) in enumerate(self.prior.items()):
                if isinstance(v, dist.Distribution):
                    parameters[k] = pyro.sample(k, v).unsqueeze(-1)
                else:
                    parameters[k] = v

        rescaled_sigmas = torch.zeros(grid_params.shape + (len(self.sigmas),), device=self.device)
        rescaled_sigmas[..., self.DH_idx] = self.sigmas[self.DH_idx] * torch.sqrt((self.efficiency[self.DH_idx]*tracer_ratio)/self.nominal_passed_ratio[self.DH_idx])
        if self.include_D_M:
            y = self.central_val
            rescaled_sigmas[..., self.DM_idx] = self.sigmas[self.DM_idx] * torch.sqrt((self.efficiency[self.DM_idx]*tracer_ratio)/self.nominal_passed_ratio[self.DM_idx])
            covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
        else:
            y = self.central_val[self.DH_idx]
            covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (rescaled_sigmas[..., self.DH_idx].unsqueeze(-1) * rescaled_sigmas[..., self.DH_idx].unsqueeze(-2))
        with GridStack(grid_params):
            parameters = {k: torch.tensor(getattr(grid_params, k), device=self.device).unsqueeze(-1) for k in grid_params.names}
        mean = torch.zeros(grid_params.shape + (len(self.sigmas),), device=self.device)
        mean[..., self.DH_idx] = self.D_H_func(**parameters)

        if self.include_D_M:
            mean[..., self.DM_idx] = self.D_M_func(**parameters)
        else:
            mean = mean[..., self.DH_idx]

        # evaluate the multivariate normal likelihood
        likelihood = dist.MultivariateNormal(mean, covariance_matrix).log_prob(y).exp()

        # normalize the likelihood to get the posterior
        posterior_pdf = likelihood / likelihood.sum()

        flat_pdf = posterior_pdf.cpu().flatten()
        indices = np.array(list(np.ndindex(posterior_pdf.shape)))
        sampled_indices = np.random.choice(len(indices), size=num_param_samples, p=flat_pdf)
        indices = indices[sampled_indices]
        param_samples = []
        param_mesh = np.stack(np.meshgrid(*[getattr(grid_params, grid_params.names[i]).squeeze() for i in range(len(grid_params.names))], indexing='ij'), axis=-1)
        for i in range(num_param_samples):
            param_samples.append(param_mesh[tuple(indices[i])])
        return torch.tensor(np.array(param_samples), device=self.device)

    @profile_method
    def sample_valid_parameters(self, sample_shape, prior=None, use_prior_flow=True):
        """
        Sample parameters from prior or from a trained posterior model if available.

        If prior_flow is set and use_prior_flow is True, samples are drawn from
        the posterior model at the nominal context. Otherwise, samples are drawn
        from the uniform prior.

        All samples are registered with pyro.sample for proper tracing.

        Args:
            sample_shape: Shape of the samples to draw.
            prior: Optional prior dict to sample from. Defaults to self.prior.
            use_prior_flow: If True (default), use prior_flow when available.
                Set to False to bypass prior_flow and sample from the explicit prior.
        """
        parameters = {}
        
        # Check if we should use a posterior model as prior
        if use_prior_flow and hasattr(self, 'prior_flow') and self.prior_flow is not None:
            # Get raw samples from prior flow (shape: *sample_shape, n_params)
            samples = self._sample_prior_flow(sample_shape)
            
            # Register each parameter with pyro.sample
            for i, param_name in enumerate(self.cosmo_params):
                param_values = samples[..., i]
                print(param_name, param_values[:5])
                parameters[param_name] = pyro.sample(param_name, dist.Delta(param_values)).unsqueeze(-1)
            
            return parameters
        
        # Otherwise sample from explicit prior distributions
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
            # if no parameter constraints defined
            if 'Om' in prior.keys():
                parameters['Om'] = pyro.sample('Om', prior['Om']).unsqueeze(-1)
            if 'w0' in prior.keys():
                parameters['w0'] = pyro.sample('w0', prior['w0']).unsqueeze(-1)
            
        # Always sample hrdrag
        if self.analysis == "bao":
            parameters['hrdrag'] = pyro.sample('hrdrag', prior['hrdrag']).unsqueeze(-1)

        return parameters

    @profile_method
    def pyro_model(self, tracer_ratio):
        passed_ratio = self.calc_passed(tracer_ratio)
        with pyro.plate_stack("plate", passed_ratio.shape[:-1]):
            parameters = self.sample_valid_parameters(passed_ratio.shape[:-1])
            means = torch.zeros(passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)
            rescaled_sigmas = torch.zeros(passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device)
            z_eff = torch.tensor(self.desi_data[self.desi_data["quantity"] == "DH_over_rs"]["z"].to_list(), device=self.device)
            means[:, :, self.DH_idx] = self.D_H_func(z_eff, **parameters)
            rescaled_sigmas[:, :, self.DH_idx] = self.sigmas[self.DH_idx] * self.sigma_scaling_factor(passed_ratio, tracer_ratio, self.DH_idx)
            if self.include_D_M:
                z_eff = torch.tensor(self.desi_data[self.desi_data["quantity"] == "DM_over_rs"]["z"].to_list(), device=self.device)
                means[:, :, self.DM_idx] = self.D_M_func(z_eff, **parameters)
                rescaled_sigmas[:, :, self.DM_idx] = self.sigmas[self.DM_idx] * self.sigma_scaling_factor(passed_ratio, tracer_ratio, self.DM_idx)

            if self.include_D_V:
                z_eff = torch.tensor(self.desi_data[self.desi_data["quantity"] == "DV_over_rs"]["z"].to_list(), device=self.device)
                means[:, :, self.DV_idx] = self.D_V_func(z_eff, **parameters)
                rescaled_sigmas[:, :, self.DV_idx] = self.sigmas[self.DV_idx] * self.sigma_scaling_factor(passed_ratio, tracer_ratio, self.DV_idx)

            # extract correlation matrix from DESI covariance matrix
            if self.include_D_V and self.include_D_M:
                means = means.to(self.device)
                # convert correlation matrix to covariance matrix using rescaled sigmas
                covariance_matrix = self.corr_matrix * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))
            else:
                # only use D_H values for mean and covariance matrix
                means = means[:, :, self.DH_idx].to(self.device)
                covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-1) * rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-2))

            return pyro.sample(self.observation_labels[0], dist.MultivariateNormal(means, covariance_matrix))

    @profile_method
    def Pk_multipoles(self, k_bins, z_eff, omega_cdm, omega_b, h, ln10A_s, n_s,
                      b1_sigma8, b2_sigma8_sq=None, bs_sigma8_sq=None,
                      alpha_0=None, alpha_2=None, SN_0=None, SN_2=None,
                      ells=(0, 2, 4), tau_reio=0.054, use_cosmopower=True):
        """
        Compute power spectrum multipoles P_ell(k) for given cosmological parameters.
        
        This method:
        1. Computes matter power spectrum P_m(k) using cosmopower-jax (or placeholder)
        2. Applies Kaiser formula: P_g(k, mu) = (b1 + f*mu^2)^2 * P_m(k)
        3. Converts to multipoles: P_ell(k) = (2*ell+1)/2 * int_0^1 P_g(k,mu) * L_ell(mu) dmu
        
        Args:
            k_bins (torch.Tensor): k bins for power spectrum evaluation, shape (n_k,)
            z_eff (float or torch.Tensor): Effective redshift(s)
            omega_cdm (torch.Tensor): Cold dark matter density, shape (..., 1)
            omega_b (torch.Tensor): Baryon density, shape (..., 1)
            h (torch.Tensor): Hubble parameter, shape (..., 1)
            ln10A_s (torch.Tensor): Amplitude of primordial power spectrum, shape (..., 1)
            n_s (torch.Tensor): Spectral index, shape (..., 1)
            b1_sigma8 (torch.Tensor): (1 + b_1) * σ_8, shape (..., 1)
            b2_sigma8_sq (torch.Tensor, optional): b_2 * σ_8^2, shape (..., 1)
            bs_sigma8_sq (torch.Tensor, optional): b_s * σ_8^2, shape (..., 1)
            alpha_0 (torch.Tensor, optional): Monopole counterterm, shape (..., 1)
            alpha_2 (torch.Tensor, optional): Quadrupole counterterm, shape (..., 1)
            SN_0 (torch.Tensor, optional): Shot noise monopole, shape (..., 1)
            SN_2 (torch.Tensor, optional): Shot noise quadrupole, shape (..., 1)
            ells (tuple): Multipoles to compute, default (0, 2, 4)
            tau_reio (float): Reionization optical depth, default 0.054
            use_cosmopower (bool): Whether to use cosmopower-jax for P_m(k). If False, uses placeholder.
        
        Returns:
            torch.Tensor: Power spectrum multipoles with shape (..., n_ells, n_k)
                where the ell dimension corresponds to the multipoles in order
        """
        DTYPE = torch.float64
        dev = self.device
        
        # Infer plate shape from required parameters
        plate = _infer_plate_shape(dev, DTYPE, omega_cdm, omega_b, h, ln10A_s, n_s, b1_sigma8)
        
        def to_plate1(x, default=None):
            if x is None: x = default
            t = torch.as_tensor(x, device=dev, dtype=DTYPE)
            if t.ndim == len(plate)+1 and t.shape[-1] == 1 and list(t.shape[:-1]) == list(plate):
                return t
            return t.view(*([1]*len(plate)), 1).expand(plate + (1,))
        
        # Convert parameters to plate shape
        omega_cdm = to_plate1(omega_cdm)
        omega_b = to_plate1(omega_b)
        h = to_plate1(h)
        ln10A_s = to_plate1(ln10A_s)
        n_s = to_plate1(n_s)
        b1_sigma8 = to_plate1(b1_sigma8)
        tau_reio = to_plate1(tau_reio)
        
        # Convert inputs to tensors
        k_bins = torch.as_tensor(k_bins, device=dev, dtype=DTYPE)
        z_eff = torch.as_tensor(z_eff, device=dev, dtype=DTYPE)
        if z_eff.ndim == 0:
            z_eff = z_eff[None]
        
        n_k = k_bins.shape[0]
        n_ells = len(ells)
        
        # z -> (plate, Nz)
        if z_eff.ndim == 1:
            Nz = z_eff.shape[0]
            z = z_eff.reshape(*([1]*len(plate)), Nz).expand(plate + (Nz,))
        else:
            if tuple(z_eff.shape[:-1]) != tuple(plate):
                z = torch.broadcast_to(z_eff, plate + (z_eff.shape[-1],))
            else:
                z = z_eff
        
        # Step 1: Compute matter power spectrum P_m(k)
        if use_cosmopower:
            try:
                # Load emulators (cache them as instance attributes)
                if not hasattr(self, '_cp_lin'):
                    self._cp_lin = CosmoPowerJAX(probe='mpk_lin')
                if not hasattr(self, '_cp_boost'):
                    self._cp_boost = CosmoPowerJAX(probe='mpk_boost')
                
                # cosmopower expects: omega_b, omega_cdm, h, n_s, ln10^10A_s, tau_reio
                # Stack parameters: shape (..., 6)
                param_array = torch.stack([
                    omega_b.squeeze(-1), omega_cdm.squeeze(-1), h.squeeze(-1),
                    n_s.squeeze(-1), ln10A_s.squeeze(-1), tau_reio.squeeze(-1)
                ], dim=-1)
                
                # Convert to numpy for cosmopower (handle batching)
                param_array_np = param_array.cpu().numpy()
                k_bins_np = k_bins.cpu().numpy()
                
                # Reshape for cosmopower: expects (n_samples, n_params)
                original_shape = param_array_np.shape[:-1]
                n_samples = int(np.prod(original_shape))
                param_flat = param_array_np.reshape(n_samples, -1)
                
                # Compute P_m(k) vectorized
                pk_lin = self._cp_lin.predict(param_flat, k=k_bins_np)  # (n_samples, n_k)
                pk_boost = self._cp_boost.predict(param_flat, k=k_bins_np)
                pk_nonlin = pk_lin * pk_boost  # (n_samples, n_k)
                
                # Reshape back to plate shape
                pk_m = torch.tensor(pk_nonlin, device=self.device, dtype=torch.float64)
                pk_m = pk_m.reshape(*original_shape, n_k)
                
            except (ImportError, AttributeError) as e:
                if self.global_rank == 0:
                    print(f"Warning: cosmopower-jax not available or error: {e}. Using placeholder.")
                use_cosmopower = False
        
        if not use_cosmopower:
            # Placeholder: return a simple power law (for testing)
            # P_m(k) ~ k^n with n ~ -2
            pk_m = (k_bins.unsqueeze(0) / 0.1) ** (-2.0)
            # Expand to plate shape
            pk_m = pk_m.expand(*plate, n_k)
        
        # Step 2: Compute growth rate f from cosmology
        # f ≈ Ω_m(z)^0.55 (approximate formula)
        # Compute Om from omega_cdm, omega_b, h
        Om = (omega_cdm.squeeze(-1) + omega_b.squeeze(-1)) / (h.squeeze(-1)**2)
        
        # Compute Om(z) = Om * (1+z)^3 / [Om*(1+z)^3 + (1-Om)]
        zp1 = 1.0 + z
        Om_z = Om.unsqueeze(-1) * zp1**3 / (Om.unsqueeze(-1) * zp1**3 + (1 - Om.unsqueeze(-1)))
        f = Om_z ** 0.55  # shape (plate, Nz)
        
        # Step 3: Extract bias parameter from b1_sigma8
        # b1_sigma8 = (1 + b1) * sigma8
        # For Kaiser formula, we need b1, but we have b1_sigma8
        # This is a simplification - in full implementation, compute sigma8 from cosmology
        # For now, use b1_sigma8 as effective bias (needs proper sigma8 calculation)
        b1_eff = b1_sigma8.squeeze(-1)  # shape (plate,)
        
        # Expand f and b1_eff to match pk_m shape (plate, n_k)
        # pk_m has shape (plate, n_k), f has shape (plate, Nz), need to broadcast
        if z.ndim == len(plate) + 1:
            # f is (plate, Nz), need (plate, n_k) - use first z value or average
            f_tensor = f[..., 0:1].expand(plate + (n_k,))  # Use first redshift
        else:
            f_tensor = f.expand(plate + (n_k,))
        
        b1_tensor = b1_eff.unsqueeze(-1).expand(plate + (n_k,))
        
        # Step 3: Apply Kaiser formula and convert to multipoles
        # P_g(k, mu) = (b1 + f*mu^2)^2 * P_m(k)
        # P_ell(k) = (2*ell+1)/2 * int_0^1 P_g(k,mu) * L_ell(mu) dmu
        # Using analytical formulas for ell=0,2,4:
        # P_0 = (b1^2 + 2/3*b1*f + 1/5*f^2) * P_m
        # P_2 = (4/3*b1*f + 4/7*f^2) * P_m
        # P_4 = (8/35*f^2) * P_m
        
        # Compute multipoles using analytical formulas
        multipoles = []
        for ell in ells:
            if ell == 0:
                # Monopole: P_0 = (b1^2 + 2/3*b1*f + 1/5*f^2) * P_m
                P_ell = (b1_tensor**2 + (2.0/3.0)*b1_tensor*f_tensor + (1.0/5.0)*f_tensor**2) * pk_m
            elif ell == 2:
                # Quadrupole: P_2 = (4/3*b1*f + 4/7*f^2) * P_m
                P_ell = ((4.0/3.0)*b1_tensor*f_tensor + (4.0/7.0)*f_tensor**2) * pk_m
            elif ell == 4:
                # Hexadecapole: P_4 = (8/35*f^2) * P_m
                P_ell = (8.0/35.0) * f_tensor**2 * pk_m
            else:
                # For higher multipoles, use numerical integration
                # This is a simplified version - for production, use proper integration
                raise NotImplementedError(f"Multipole ell={ell} not implemented. Only ell=0,2,4 supported.")
            
            multipoles.append(P_ell)
        
        # Stack multipoles: shape (..., n_ells, n_k)
        P_multipoles = torch.stack(multipoles, dim=-2)
        
        return P_multipoles

    @profile_method
    def compute_covariance_from_mocks(self, mocks):
        """
        Compute covariance matrix from DESI mocks.
        
        Args:
            mocks (list or np.ndarray): List of mock power spectrum measurements or
                array of shape (n_mocks, n_data_points)
        
        Returns:
            torch.Tensor: Covariance matrix with shape (n_data_points, n_data_points)
        """
        # Convert mocks to numpy array if needed
        if isinstance(mocks, list):
            # If mocks are PowerSpectrumStatistics objects, extract the flat power
            try:
                mock_array = np.array([mock.power_nonorm.ravel() for mock in mocks])
            except AttributeError:
                # Assume mocks are already arrays
                mock_array = np.array(mocks)
        else:
            mock_array = np.asarray(mocks)
        
        # Compute covariance: cov = mean((x - mean(x))^T @ (x - mean(x)))
        mock_mean = np.mean(mock_array, axis=0)
        mock_centered = mock_array - mock_mean[None, :]
        covariance = np.cov(mock_centered.T)
        
        return torch.tensor(covariance, device=self.device, dtype=torch.float64)

    @profile_method
    def pyro_model_fs(self, tracer_ratio, mocks=None, covariance=None, 
                             k_bins=None, z_eff=None, ells=None):
        """
        Full shape version of pyro_model using power spectrum multipoles.
        
        This function defines a Gaussian likelihood for power spectrum multipoles:
        - Means come from theory calculated with vectorized approach
        - Covariance comes from DESI mocks
        
        Args:
            tracer_ratio (torch.Tensor): Design variables (tracer ratios)
            mocks (list or np.ndarray, optional): DESI mocks for covariance computation.
                If None, uses self.fullshape_covariance if available.
            covariance (torch.Tensor, optional): Pre-computed covariance matrix.
                If None, computes from mocks.
            k_bins (torch.Tensor, optional): k bins for power spectrum.
                If None, uses self.fullshape_k_bins if available.
            z_eff (float or torch.Tensor, optional): Effective redshift(s).
                If None, uses self.fullshape_z_eff if available.
            ells (tuple, optional): Multipoles to compute (e.g., (0, 2, 4) for monopole, 
                quadrupole, hexadecapole). If None, uses self.fullshape_ells if available,
                otherwise defaults to (0, 2, 4).
        
        Returns:
            torch.Tensor: Sampled power spectrum multipoles
        """
        passed_ratio = self.calc_passed(tracer_ratio)
        
        # Get ells: use provided value, or instance attribute, or default
        if ells is None:
            ells = getattr(self, 'fullshape_ells', (0, 2, 4))
        
        # Get or compute covariance
        if covariance is None:
            if mocks is not None:
                covariance = self.compute_covariance_from_mocks(mocks)
            elif hasattr(self, 'fullshape_covariance'):
                covariance = self.fullshape_covariance
            else:
                raise ValueError("Must provide either mocks or covariance, or set self.fullshape_covariance")
        
        # Get k_bins and z_eff
        if k_bins is None:
            if hasattr(self, 'fullshape_k_bins'):
                k_bins = self.fullshape_k_bins
            else:
                raise ValueError("Must provide k_bins or set self.fullshape_k_bins")
        
        if z_eff is None:
            if hasattr(self, 'fullshape_z_eff'):
                z_eff = self.fullshape_z_eff
            else:
                raise ValueError("Must provide z_eff or set self.fullshape_z_eff")
        
        # Ensure covariance is on correct device and has correct shape
        if isinstance(covariance, np.ndarray):
            covariance = torch.tensor(covariance, device=self.device, dtype=torch.float64)
        else:
            covariance = covariance.to(self.device)
        
        n_data_points = covariance.shape[0]
        
        with pyro.plate_stack("plate", passed_ratio.shape[:-1]):
            # Sample cosmological parameters from prior
            parameters = self.sample_valid_parameters(passed_ratio.shape[:-1])
            
            # Compute theoretical power spectrum multipoles (vectorized)
            # Shape: (..., n_ells, n_k)
            P_multipoles = self.Pk_multipoles(
                k_bins, z_eff,
                **parameters,
                ells=ells
            )
            
            # Flatten: [P_0(k1), P_0(k2), ..., P_0(k_nk), P_2(k1), ..., P_4(k1), ...]
            # Reshape from (..., n_ells, n_k) to (..., n_ells * n_k)
            n_ells, n_k = P_multipoles.shape[-2:]
            means = P_multipoles.reshape(*P_multipoles.shape[:-2], n_ells * n_k)
            
            # Ensure means have correct shape: (..., n_data_points)
            if means.shape[-1] != n_data_points:
                raise ValueError(
                    f"Computed means shape {means.shape[-1]} does not match "
                    f"covariance shape {n_data_points}. "
                    f"Expected n_ells * n_k = {n_ells} * {n_k} = {n_ells * n_k} data points."
                )
            
            # Broadcast covariance to match batch dimensions if needed
            # covariance is (n_data_points, n_data_points)
            # means is (..., n_data_points)
            # We need covariance to be (..., n_data_points, n_data_points)
            batch_shape = means.shape[:-1]
            if len(batch_shape) > 0:
                # Expand covariance to match batch dimensions
                cov_expanded = covariance.unsqueeze(0).expand(*batch_shape, -1, -1)
            else:
                cov_expanded = covariance
            
            # Sample from MultivariateNormal
            return pyro.sample(
                self.observation_labels[0], 
                dist.MultivariateNormal(means, cov_expanded)
            )

    def unnorm_lfunc(self, params, features, designs):
        parameters = { }
        for key in params.names:
            parameters[key] = torch.tensor(getattr(params, key), device=self.device)
        likelihood = 1
        passed_ratio = self.calc_passed(designs)
        for i in range(len(self.tracer_bins)):
            D_H_mean = self.D_H_func(**parameters)
            D_H_diff = getattr(features, features.names[i]) - D_H_mean.cpu().numpy()
            print(getattr(designs, designs.names[i]).shape)
            D_H_sigma = self.sigmas[self.DH_idx].cpu().numpy()[i] * np.sqrt(self.nominal_passed_ratio[i].cpu().numpy()/passed_ratio[i])
            likelihood = np.exp(-0.5 * (D_H_diff / D_H_sigma) ** 2) * likelihood
            
            if self.include_D_M:
                D_M_mean = self.D_M_func(**parameters)
                D_M_diff = getattr(features, features.names[i+len(self.tracer_bins)]) - D_M_mean.cpu().numpy()
                D_M_sigma = self.sigmas[self.DM_idx].cpu().numpy()[i] * np.sqrt(self.nominal_passed_ratio[i].cpu().numpy()/passed_ratio[i])
                likelihood = np.exp(-0.5 * (D_M_diff / D_M_sigma) ** 2) * likelihood

        return likelihood
