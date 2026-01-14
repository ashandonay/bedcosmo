import os
import sys
import yaml
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

# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory ('BED_cosmo/') and add it to the Python path
parent_dir_abs = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, parent_dir_abs)
from custom_dist import ConstrainedUniform2D
from util import Bijector, auto_seed, profile_method, load_nominal_samples

storage_path = os.environ["SCRATCH"] + "/bed/BED_cosmo/num_tracers"
home_dir = os.environ["HOME"]
mlflow.set_tracking_uri(storage_path + "/mlruns")

T_CMB = 2.7255
KB_eV_per_K = 8.617333262e-5
Tnu0_eV = (4.0/11.0)**(1.0/3.0) * T_CMB * KB_eV_per_K

def _interp1(xg, yg, x):
    xg = torch.as_tensor(xg, dtype=torch.float64, device=yg.device)
    yg = torch.as_tensor(yg, dtype=torch.float64, device=yg.device)
    x  = torch.as_tensor(x,  dtype=torch.float64, device=yg.device)

    x = x.clamp(xg[0], xg[-1])
    idx = torch.searchsorted(xg, x, right=True) - 1
    idx = idx.clamp(0, xg.numel()-2)
    idx1 = idx + 1

    x0, x1 = xg[idx], xg[idx1]
    y0, y1 = yg[idx], yg[idx1]
    w = (x - x0) / (x1 - x0)
    return y0 + w*(y1 - y0)

class _NeutrinoTableCache:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.key = None
        self.a = None
        self.lna = None
        self.fnu = None  # (n_massive, n_ag)
        self._built = False
    
    def _build_table(self, mnu_eV, n_massive, n_ag=1200, qmax=40.0, nq=2000, a_min=0.28):
        """
        Build f_nu(a) on an a-grid in (a_min, 1], where
        f_nu(a) = rho_nu(a)/rho_nu(1)
                = [ I(a) / I(1) ] * a^{-4}
        with I(a) = ∫ dq q^2 sqrt(q^2 + (a*y0)^2) / (1+e^q),
        y0 = (m_per / Tnu0), m_per = mnu_eV / n_massive.
        """
        # constants
        T_cmb = torch.tensor(2.7255, device=self.device, dtype=self.dtype)          # K
        kB_eV = torch.tensor(8.61733262e-5, device=self.device, dtype=self.dtype)  # eV/K
        Tnu0 = (torch.tensor(4.0/11.0, device=self.device, dtype=self.dtype))**(1.0/3.0) * T_cmb * kB_eV

        m_per = torch.tensor(float(mnu_eV)/int(n_massive), device=self.device, dtype=self.dtype)
        y0 = m_per / Tnu0

        # ln a grid
        self.lna = torch.linspace(torch.log(torch.as_tensor(a_min, device=self.device, dtype=self.dtype)),
                            torch.tensor(0.0, device=self.device, dtype=self.dtype),
                            int(n_ag), device=self.device, dtype=self.dtype)
        self.a = torch.exp(self.lna)  # (G,)
        # momentum grid
        q = torch.linspace(0.0, float(qmax), int(nq), device=self.device, dtype=self.dtype)  # (nq,)
        fq = 1.0 / (torch.exp(q) + 1.0)
        q2 = q*q

        # I(a) integral
        # E(q,a) = sqrt(q^2 + (a*y0)^2)
        Ey = torch.sqrt(q2[:, None] + (self.a[None, :] * y0)**2)   # (nq, G)
        Ia = torch.trapezoid(q2[:, None] * Ey * fq[:, None], q, dim=0)  # (G,)
        # I(1) at a=1:
        E1 = torch.sqrt(q2 + y0*y0)
        I1 = torch.trapezoid(q2 * E1 * fq, q, dim=0)  # scalar

        # f_nu(a) = (Ia/I1) * a^{-4}
        fnu_1d = (Ia / I1) * (self.a**(-4))
        self.fnu = fnu_1d.unsqueeze(0).expand(int(n_massive), -1)  # (n_massive, G)

        self._built = True
        pass
    
    def get_table(self, mnu_eV, n_massive, n_ag=1200, qmax=40.0, nq=2000, a_min=0.28):
        """Get the neutrino table, building it if not already built."""
        if not self._built:
            self._build_table(mnu_eV, n_massive, n_ag, qmax, nq, a_min)
        return self.a, self.fnu, self.lna

def _infer_plate_shape(dev, dtype, *args):
    """
    Look at inputs (tensors or scalars), treat any final-dim==1 as 'placeholder',
    and broadcast the *leading* dims to a common plate shape.
    """
    lead_shapes = []
    for a in args:
        ta = torch.as_tensor(a, device=dev, dtype=dtype) if a is not None else None
        if ta is None or ta.ndim == 0:
            lead_shapes.append(())  # scalar: no plate dims
        else:
            if ta.shape[-1] == 1:
                lead_shapes.append(ta.shape[:-1])
            else:
                lead_shapes.append(ta.shape)
    # torch.broadcast_shapes(*shapes) is available on recent torch; fall back if needed
    plate = ()
    for s in lead_shapes:
        plate = torch.broadcast_shapes(plate, s)
    return plate

def _cumsimpson(x, y, dim=-1):
    """
    Cumulative composite Simpson's rule along `dim`.
    Requires an even number of intervals; if not, the last interval falls back to trapezoid.
    Returns same shape as y.
    """
    x = torch.as_tensor(x, dtype=y.dtype, device=y.device)
    N = y.shape[dim]
    if N < 2:
        return torch.zeros_like(y)

    # Δx segments
    dx = x.diff()
    # Move dim to last for easier slicing
    yT = y.transpose(dim, -1)  # (..., N)

    out = torch.zeros_like(yT)
    # handle pairs [0:2], [2:4], ...
    # Simpson on each pair of intervals (3 points)
    # We'll accumulate cumulatively.
    acc = torch.zeros_like(yT[..., 0])
    out[..., 0] = 0.0

    # iterate over triplets (0,1,2), (2,3,4), ...
    # vectorized: build weights
    # For simplicity and speed, do blocks of 2 intervals where possible, last interval → trap if needed
    two = (N - 1) // 2 * 2  # largest even number ≤ N-1

    # Simpson over [2k, 2k+2]
    i0 = torch.arange(0, two, 2, device=y.device)
    i1 = i0 + 1
    i2 = i0 + 2
    dx0 = dx[i0]
    dx1 = dx[i1]
    # assume nearly uniform in practice; if strongly non-uniform, use local (dx0+dx1) weighting:
    h = dx0 + dx1
    contrib = h/6.0 * (yT[..., i0] + 4.0*yT[..., i1] + yT[..., i2])  # (..., #pairs)

    # cumulative sum in pairs
    acc_pairs = contrib.cumsum(dim=-1)
    # scatter accumulated values into out at even indices
    out[..., 2:two+1:2] = acc_pairs

    # fill odd indices (use Simpson partials)
    # cumulative at index 2k+1 is previous even + Simpson over [2k, 2k+1] via trapezoid as a mid-step
    # (this makes cumulative monotone and very accurate for nearly uniform grids)
    trap01 = 0.5*dx0*(yT[..., i0] + yT[..., i1])
    out[..., 1:two:2] = (acc_pairs - contrib) + trap01

    # last interval if N-1 is odd: trapezoid from N-2→N-1
    if (N - 1) % 2 == 1:
        last = N - 1
        out[..., last] = out[..., last-1] + 0.5*dx[last-1]*(yT[..., last-1] + yT[..., last])

    # transpose back
    return out.transpose(-1, dim)

class NumTracers:
    def __init__(
        self, 
        dataset="dr2", 
        cosmo_model="base",
        priors_path=None,
        flow_type="MAF",
        input_designs=None,
        design_args=None,
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
        profile=False
    ):

        self.name = 'num_tracers'
        self.dataset = dataset
        self.desi_data = pd.read_csv(os.path.join(home_dir, f"data/desi/bao_{self.dataset}", 'desi_data.csv'))
        self.desi_tracers = pd.read_csv(os.path.join(home_dir, f"data/desi/bao_{self.dataset}", 'desi_tracers.csv'))
        if priors_path is None:
            priors_path = os.path.join(home_dir, f"data/desi/bao_{self.dataset}", 'priors.yaml')
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
        
        # initialize the priors
        with open(priors_path, 'r') as file:
            self.prior_data = yaml.safe_load(file)
        self.priors, self.param_constraints, self.latex_labels = self.get_priors(priors_path)
        self.desi_priors, _, _ = self.get_priors(os.path.join(home_dir, f"data/desi/bao_{self.dataset}", 'priors.yaml'))
        self.cosmo_params = list(self.priors.keys())
        self.param_bijector = Bijector(self, cdf_bins=5000, cdf_samples=1e7)
        if bijector_state is not None:
            if self.global_rank == 0:
                print(f"Restoring bijector state from checkpoint.")
            self.param_bijector.set_state(bijector_state)
        # if the priors are not the same as the DESI priors, create a new bijector for the DESI samples
        if self.priors.items() != self.desi_priors.items():
            self.desi_bijector = Bijector(self, priors=self.desi_priors, cdf_bins=5000, cdf_samples=2e7)
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
            self.init_designs(input_designs=input_designs, **design_args)
        else:
            self.init_designs(input_designs=input_designs)

    @profile_method
    def init_designs(self, input_designs=None, step=0.05, lower=0.05, upper=None, sum_lower=1.0, sum_upper=1.0, tol=1e-3, labels=None):
        """
        Initialize design space.
        
        Args:
            input_designs: Can be:
                - None: Generate design grid using design parameters (default)
                - "nominal": Use the nominal design as the input design
                - list/array: Use specific design(s), shape should be (num_designs, num_targets)
                  If 1D list with length == num_targets, it will be reshaped to (1, num_targets)
                  Examples: [0.2, 0.3, 0.3, 0.2] for single design
                           [[0.2, 0.3, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25]] for multiple designs
            step: Step size(s) for design grid (default: 0.05, only used if input_designs is None)
            lower: Lower bound(s) for each design variable (default: 0.05, only used if input_designs is None)
            upper: Upper bound(s) for each design variable (default: None, only used if input_designs is None)
            sum_lower: Lower bound on sum of design variables (default: 1.0)
            sum_upper: Upper bound on sum of design variables (default: 1.0)
            tol: Tolerance for sum constraint (default: 1e-3)

        """
        # Check if input_designs is the special "nominal" keyword
        if input_designs == "nominal":
            designs = self.nominal_design.unsqueeze(0)  # Add batch dimension
        elif input_designs is not None:
            # User provided specific design(s)
            design_array = np.array(input_designs)
            
            # Handle 1D input (single design)
            if design_array.ndim == 1:
                if len(design_array) != len(labels):
                    raise ValueError(f"Input design must have {len(labels)} values, got {len(design_array)}")
                design_array = design_array.reshape(1, -1)
            elif design_array.ndim == 2:
                if design_array.shape[1] != len(labels):
                    raise ValueError(f"Input design must have {len(labels)} columns, got {design_array.shape[1]}")
            else:
                raise ValueError(f"Input design must be 1D or 2D, got shape {design_array.shape}")
            
            designs = torch.tensor(design_array, device=self.device, dtype=torch.float64)
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

        if self.global_rank == 0 and self.verbose:
            print(
                f"Designs initialized with the following parameters:\n",
                f"step size: {step}\n",
                f"lower range: {lower}\n",
                f"upper range: {upper}\n",
                f"sum lower: {sum_lower}\n",
                f"sum upper: {sum_upper}\n"
                )
            print(f"Designs shape: {self.designs.shape}")
            if input_designs == "nominal":
                print(f"Using nominal design as input design: {self.designs}")
            elif input_designs is not None:
                print(f"Input design(s): {self.designs}")
            print(f"Nominal design: {self.nominal_design}\n")
    def design_plot(self):
        """
        Plot the design variables in 3D with the 4th dimension (QSO) as color
        """
        fig = plt.figure(figsize=(10, 8))
        ax_3d = fig.add_subplot(111, projection='3d')

        # Convert tensors to numpy for plotting
        designs_np = self.designs.cpu().numpy()
        nominal_design_np = self.nominal_design.cpu().numpy()
        
        cbar_min = np.min(designs_np[:, 3])
        cbar_max = np.max(designs_np[:, 3])
        # 3D scatter plot with 4th dimension (QSO) as color
        scatter_3d = ax_3d.scatter(designs_np[:, 1], designs_np[:, 2], designs_np[:, 0],
                    c=designs_np[:, 3],  # 4th dimension (QSO) as color
                    cmap='viridis',
                    s=60,
                    alpha=0.8,
                    marker='o',
                    vmin=cbar_min,
                    vmax=cbar_max)
        
        # Add nominal and optimal markers
        ax_3d.scatter(nominal_design_np[1], nominal_design_np[2], nominal_design_np[0],
                    c=nominal_design_np[3],
                    cmap='viridis',
                    s=100,
                    alpha=0.8,
                    vmin=cbar_min,
                    vmax=cbar_max,
                    marker='*',
                    label='Nominal Design')
        
        # Configure 3D plot
        ax_3d.set_title('Design Variables', fontsize=16)
        ax_3d.set_xlabel("$f_{LRG}$", fontsize=14)
        ax_3d.set_ylabel("$f_{ELG}$", fontsize=14)
        ax_3d.set_zlabel("$f_{BGS}$", fontsize=14)
        ax_3d.grid(True, alpha=0.3)
        ax_3d.legend(fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(scatter_3d, ax=ax_3d, shrink=0.8, aspect=20)
        cbar.set_label("$f_{QSO}$", fontsize=14)
        return fig
    
    @profile_method
    def get_priors(self, prior_path):
        """
        Load cosmological priors and constraints from a YAML configuration file.
        
        This function dynamically constructs the priors and latex labels based on the
        specified cosmology model in the YAML file. It supports uniform distributions
        and handles parameter constraints like Om + Ok < 1 and w0 + wa < 0.
        
        Args:
            prior_path (str): Path to the YAML file containing prior definitions
            
        Raises:
            FileNotFoundError: If the prior file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            ValueError: If required sections or parameters are missing, or if bounds are invalid
            
        The YAML file should have the following structure:
        - parameters: defines each parameter with distribution type and bounds
        - models: defines cosmology models and their parameter lists
        - constraints: defines parameter constraints (optional)
        """
        try:
            with open(prior_path, 'r') as file:
                prior_data = yaml.safe_load(file)
            with open(os.path.join(os.path.dirname(__file__), 'models.yaml'), 'r') as file:
                cosmo_models = yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        # Get the specified cosmology model
        if self.cosmo_model not in cosmo_models:
            raise ValueError(f"Cosmology model '{self.cosmo_model}' not found in cosmo_models.yaml. Available models: {list(cosmo_models.keys())}")
        
        # Initialize priors, constraints, and latex labels
        priors = {}
        param_constraints = {}
        latex_labels = []

        model_parameters = cosmo_models[self.cosmo_model]['parameters']
        latex_labels = cosmo_models[self.cosmo_model]['latex_labels']
        for constraint in cosmo_models[self.cosmo_model].get('constraints', []):
            param_constraints[constraint] = prior_data['constraints'][constraint]
        
        # Create priors for each parameter in the model
        for param_name in model_parameters:
            if param_name not in prior_data['parameters']:
                raise ValueError(f"Parameter '{param_name}' not found in prior.yaml parameters section")
            
            param_config = prior_data['parameters'][param_name]
            
            # Validate parameter configuration
            if 'distribution' not in param_config:
                raise ValueError(f"Parameter '{param_name}' missing 'distribution' section in prior.yaml")
            if 'latex' not in param_config:
                raise ValueError(f"Parameter '{param_name}' missing 'latex' section in prior.yaml")
            
            dist_config = param_config['distribution']
            
            if dist_config['type'] == 'uniform':
                if 'lower' not in dist_config or 'upper' not in dist_config:
                    raise ValueError(f"Prior distribution for '{param_name}' is missing 'lower' or 'upper' bounds")
                lower = dist_config.get('lower', 0.0)
                upper = dist_config.get('upper', 1.0)
                if lower >= upper:
                    raise ValueError(f"Invalid bounds for '{param_name}': lower ({lower}) must be < upper ({upper})")
                priors[param_name] = dist.Uniform(*torch.tensor([lower, upper], device=self.device))
            else:
                raise ValueError(f"Distribution type '{dist_config['type']}' not supported. Only 'uniform' is currently supported.")
            
            if 'multiplier' in param_config.keys():
                setattr(self, f'{param_name}_multiplier', float(param_config['multiplier']))

        return priors, param_constraints, latex_labels

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

    def _E_of_z(self, z, Om, Ok, w0, wa, Or, Onu0, Ode0, n_massive, cache):
        """
        z:      (plate, Nz)
        Om,...: (plate, 1)
        cache:  object with .lna and .fnu (or None if n_massive==0)
        """
        zp1 = 1.0 + z
        # CPL dark energy
        fde = zp1**(3.0*(1.0 + w0 + wa)) * torch.exp(-3.0*wa * z / zp1)

        # massive ν(a) term
        if n_massive > 0 and cache is not None:
            ln_a = -torch.log1p(z)
            fnu_at_a = _interp1(cache.lna, cache.fnu[0], ln_a)
            Onu_a = Onu0 * fnu_at_a
        else:
            Onu_a = torch.zeros_like(z, dtype=z.dtype, device=z.device)

        # split cb from total matter today
        Ocb = (Om - Onu0).clamp_min(0.0)

        E2 = (Or * zp1**4
            + Ocb * zp1**3
            + Ok  * zp1**2
            + Ode0 * fde
            + Onu_a)
        return torch.sqrt(torch.clamp_min(E2, torch.finfo(z.dtype).tiny * 1e6))

    @profile_method
    def D_H_func(self, z_eff, Om, Ok=None, w0=None, wa=None, hrdrag=None,
                h=0.6736, Neff=3.044, mnu=0.06, n_massive=1, T_cmb=2.7255,
                include_radiation=True):

        DTYPE = torch.float64
        dev   = self.device

        plate = _infer_plate_shape(dev, DTYPE, Om, Ok, w0, wa, hrdrag)
        def to_plate1(x, default=None):
            if x is None: x = default
            t = torch.as_tensor(x, device=dev, dtype=DTYPE)
            if t.ndim == len(plate)+1 and t.shape[-1] == 1 and list(t.shape[:-1]) == list(plate):
                return t
            return t.view(*([1]*len(plate)), 1).expand(plate + (1,))

        Om     = to_plate1(Om)
        Ok     = to_plate1(0.0 if Ok is None else Ok)
        w0     = to_plate1(-1.0 if w0 is None else w0)
        wa     = to_plate1( 0.0 if wa is None else wa)
        hrdrag = to_plate1(hrdrag)

        h_t  = None if h is None else torch.as_tensor(h, device=dev, dtype=DTYPE)
        Tcmb = torch.as_tensor(T_cmb, device=dev, dtype=DTYPE)

        # z -> (plate, Nz)
        z_eff = torch.as_tensor(z_eff, device=dev, dtype=DTYPE)
        if z_eff.ndim == 0:
            z_eff = z_eff[None]

        if z_eff.ndim == 1:
            # unbatched z: expand over plate to (plate, Nz)
            Nz = z_eff.shape[0]
            z  = z_eff.reshape(*([1]*len(plate)), Nz).expand(plate + (Nz,))
        else:
            # batched z already: expect (..., Nz); just broadcast to plate if needed
            assert z_eff.shape[-1] > 0, "z must have a last dimension Nz > 0"
            if tuple(z_eff.shape[:-1]) != tuple(plate):
                z = torch.broadcast_to(z_eff, plate + (z_eff.shape[-1],))
            else:
                z = z_eff
        Nz = z.shape[-1]  # from here on, use the last-dim size

        # radiation today
        if include_radiation:
            assert h_t is not None, "Pass h=H0/100 when include_radiation=True"
            Ogam0 = 2.469e-5 * (Tcmb/2.7255)**4 / (h_t*h_t)
            N_massless = max(0.0, float(Neff) - int(n_massive))
            Onur0 = (7.0/8.0) * (4.0/11.0)**(4.0/3.0) * N_massless * Ogam0
            Or = torch.as_tensor(Ogam0 + Onur0, device=dev, dtype=DTYPE).view(*([1]*len(plate)), 1).expand(plate + (1,))
        else:
            Or = torch.zeros(plate + (1,), device=dev, dtype=DTYPE)

        # massive ν today
        if n_massive > 0:
            assert h_t is not None, "Need h to compute Ων0"
            Onu0 = torch.as_tensor((float(mnu)/93.14)/(float(h_t)*float(h_t)), device=dev, dtype=DTYPE)
            Onu0 = Onu0.view(*([1]*len(plate)), 1).expand(plate + (1,))
        else:
            Onu0 = torch.zeros(plate + (1,), device=dev, dtype=DTYPE)

        # Dark-energy density today
        Ode0 = 1.0 - Om - Ok - Or

        # neutrino table cache (if needed)
        if n_massive > 0:
            if not hasattr(self, "_nu_cache"): 
                self._nu_cache = _NeutrinoTableCache(dev, DTYPE)
            self._nu_cache.get_table(mnu, n_massive)
        else:
            self._nu_cache = None

        # common E(z)
        E = self._E_of_z(z, Om, Ok, w0, wa, Or, Onu0, Ode0, n_massive, self._nu_cache)

        prefac = (torch.as_tensor(self.c, device=dev, dtype=DTYPE) / (self.hrdrag_multiplier*hrdrag))
        return prefac / E

    @profile_method
    def D_M_func(self, z_eff, Om, Ok=None, w0=None, wa=None, hrdrag=None,
                h=0.6736, Neff=3.044, mnu=0.06, n_massive=1, T_cmb=2.7255,
                include_radiation=True, n_int=1025):

        DTYPE = torch.float64
        dev   = self.device

        # ---- plate inference & params → plate+(1,)  ----
        plate = _infer_plate_shape(dev, DTYPE, Om, Ok, w0, wa, hrdrag)
        def to_plate1(x, default=None):
            if x is None: x = default
            t = torch.as_tensor(x, device=dev, dtype=DTYPE)
            if t.ndim == len(plate)+1 and t.shape[-1] == 1 and list(t.shape[:-1]) == list(plate):
                return t
            return t.view(*([1]*len(plate)), 1).expand(plate + (1,))

        Om = to_plate1(Om)
        Ok = to_plate1(0.0 if Ok is None else Ok)
        w0 = to_plate1(-1.0 if w0 is None else w0)
        wa = to_plate1( 0.0 if wa is None else wa)
        hrdrag = to_plate1(hrdrag)

        # ---- targets in z / ln a ----
        z_eff = torch.as_tensor(z_eff, device=dev, dtype=DTYPE)
        if z_eff.ndim == 0:
            z_eff = z_eff[None]
        z_eff = z_eff.reshape(-1)
        z_eval = z_eff.reshape(*([1]*len(plate)), -1).expand(plate + (z_eff.shape[-1],))
        ln_a_ev = -torch.log1p(z_eval)

        # ---- base Simpson grid in ln a (odd length) & merge exact eval nodes ----
        zmax = float(z_eff.max())
        a_min = 1.0 / (1.0 + zmax)
        ln_a_base = torch.linspace(math.log(a_min), 0.0, int(n_int)|1, device=dev, dtype=DTYPE)
        ln_a_all  = torch.unique(torch.cat([ln_a_base, ln_a_ev.reshape(-1)])).sort().values
        if ln_a_all[-1].abs() > 0:
            ln_a_all = torch.unique(torch.cat([ln_a_all, torch.zeros(1, device=dev, dtype=DTYPE)])).sort().values
        if (ln_a_all.numel() % 2) == 0:
            mid = 0.5*(ln_a_all[-2] + ln_a_all[-1])
            ln_a_all = torch.unique(torch.cat([ln_a_all, mid[None]])).sort().values

        a_all = torch.exp(ln_a_all)
        z_all = 1.0 / a_all - 1.0
        zB = z_all.reshape(*([1]*len(plate)), -1).expand(plate + (ln_a_all.numel(),))

        DH_over_rd_all = self.D_H_func(
            zB, Om, Ok, w0, wa, hrdrag,
            h=h, Neff=Neff, mnu=mnu, n_massive=n_massive,
            T_cmb=T_cmb, include_radiation=include_radiation
        )

        pref = (torch.as_tensor(self.c, device=dev, dtype=DTYPE) / (self.hrdrag_multiplier*hrdrag))
        pref = pref.expand_as(DH_over_rd_all)
        E_all = (pref / DH_over_rd_all).clamp_min(torch.finfo(DTYPE).tiny*1e6)

        # ---- integrate: ∫ d ln a / (a E(a)) from a(z)→1 ----
        aB = a_all.reshape(*([1]*len(plate)), -1).expand_as(E_all)
        integ = 1.0 / (aB * E_all)
        cum = _cumsimpson(ln_a_all, integ, dim=-1)
        cum_1 = cum[..., -1:]

        # exact pick at ln a(z) (we merged nodes above)
        idx = torch.searchsorted(ln_a_all, ln_a_ev)
        DC_over_DH0 = cum_1 - torch.gather(cum, -1, idx)

        # ---- curvature mapping (with flat-limit series) ----
        absOk = torch.abs(Ok)
        sqrtOk = torch.sqrt(absOk)
        x = sqrtOk * DC_over_DH0

        Skx = torch.where(Ok > 0, torch.sinh(x),
            torch.where(Ok < 0, torch.sin(x), DC_over_DH0))
        den = torch.where(Ok == 0, torch.ones_like(Ok), sqrtOk)

        tiny = (absOk < 1e-8).expand_as(DC_over_DH0)
        series = DC_over_DH0 * (1.0 + (Ok * DC_over_DH0**2)/6.0 + (Ok**2 * DC_over_DH0**4)/120.0)
        geom = torch.where(tiny, series, torch.where(Ok != 0, Skx/den, Skx))

        # (c/H0)/r_d
        prefac = (torch.as_tensor(self.c, device=dev, dtype=DTYPE) / (self.hrdrag_multiplier*hrdrag))
        return prefac * geom

    @profile_method
    def D_V_func(
        self, z_eff, Om, Ok=None, w0=None, wa=None, hrdrag=None,
        h=0.6736, Neff=3.044, mnu=0.06, n_massive=1, T_cmb=2.7255,
        include_radiation=True, n_int=1025,
    ):
        """
        D_V/r_d = [ z * (D_M/r_d)^2 * (D_H/r_d) ]^{1/3}
        Shape: (plate, Nz). Uses identical settings as D_H_func/D_M_func.
        """
        # Compute using the SAME settings/grid as the caller expects
        DM = self.D_M_func(
            z_eff, Om, Ok, w0, wa, hrdrag,
            h=h, Neff=Neff, mnu=mnu, n_massive=n_massive, T_cmb=T_cmb,
            include_radiation=include_radiation, n_int=n_int
        )
        DH = self.D_H_func(
            z_eff, Om, Ok, w0, wa, hrdrag,
            h=h, Neff=Neff, mnu=mnu, n_massive=n_massive, T_cmb=T_cmb,
            include_radiation=include_radiation
        )

        # Build z to the same shape as DM/DH (handle 1D or batched z)
        DTYPE, dev = DM.dtype, DM.device
        z_t = torch.as_tensor(z_eff, dtype=DTYPE, device=dev)
        if z_t.ndim == 0:
            z_t = z_t[None]
        if z_t.ndim == 1:
            zB = z_t.reshape(*([1]*(DM.ndim-1)), -1).expand_as(DM)
        else:
            zB = torch.broadcast_to(z_t, DM.shape)

        # Identity
        return (zB * (DM**2) * DH).pow(1.0/3.0)

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
            for i, (k, v) in enumerate(self.priors.items()):
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

    def sample_valid_parameters(self, sample_shape, priors=None):

        # register samples in the trace using pyro.sample
        parameters = {}
        if priors is None:
            priors = self.priors
        # Handle constraints based on YAML configuration
        if hasattr(self, 'param_constraints'):
            # Check for valid density constraint
            if 'valid_densities' in self.param_constraints:
                # 0 < Om + Ok < 1
                OmOk_priors = {'Om': priors['Om'], 'Ok': priors['Ok']}
                OmOk_samples = ConstrainedUniform2D(OmOk_priors, **self.param_constraints["valid_densities"]["bounds"]).sample(sample_shape)
                parameters['Om'] = pyro.sample('Om', dist.Delta(OmOk_samples[..., 0])).unsqueeze(-1)
                parameters['Ok'] = pyro.sample('Ok', dist.Delta(OmOk_samples[..., 1])).unsqueeze(-1)
            else:
                # Sample Om, Ok normally if no constraint or Ok not present
                if 'Om' in priors.keys():
                    parameters['Om'] = pyro.sample('Om', priors['Om']).unsqueeze(-1)
                if 'Ok' in priors.keys():
                    parameters['Ok'] = pyro.sample('Ok', priors['Ok']).unsqueeze(-1)
            
            # Check for high z matter domination constraint
            if 'high_z_matter_dom' in self.param_constraints:
                # w0 + wa < 0
                w0wa_priors = {'w0': priors['w0'], 'wa': priors['wa']}
                w0wa_samples = ConstrainedUniform2D(w0wa_priors, **self.param_constraints["high_z_matter_dom"]["bounds"]).sample(sample_shape)
                parameters['w0'] = pyro.sample('w0', dist.Delta(w0wa_samples[..., 0])).unsqueeze(-1)
                parameters['wa'] = pyro.sample('wa', dist.Delta(w0wa_samples[..., 1])).unsqueeze(-1)
            else:
                # Sample w0, wa normally if no constraint or wa not present
                if 'w0' in priors.keys():
                    parameters['w0'] = pyro.sample('w0', priors['w0']).unsqueeze(-1)
                if 'wa' in priors.keys():
                    parameters['wa'] = pyro.sample('wa', priors['wa']).unsqueeze(-1)
        else:
            # if no parameter constraints defined
            if 'Om' in priors.keys():
                parameters['Om'] = pyro.sample('Om', priors['Om']).unsqueeze(-1)
            if 'w0' in priors.keys():
                parameters['w0'] = pyro.sample('w0', priors['w0']).unsqueeze(-1)
            
        # Always sample hrdrag
        parameters['hrdrag'] = pyro.sample('hrdrag', priors['hrdrag']).unsqueeze(-1)

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
