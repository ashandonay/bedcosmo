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
from util import Bijector, auto_seed, profile_method, load_desi_samples

storage_path = os.environ["SCRATCH"] + "/bed/BED_cosmo/variable_redshift"
home_dir = os.environ["HOME"]
mlflow.set_tracking_uri(storage_path + "/mlruns")


# Helper functions for high-accuracy cosmology calculations
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


class VariableRedshift:
    def __init__(
        self, 
        cosmo_model="base",
        priors_path=None,
        flow_type="MAF",
        design_step=0.1,
        design_lower=0.0,
        design_upper=5.0,
        fixed_design=False,
        include_D_M=False,
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
        if seed is not None:
            auto_seed(self.seed)
        
        self.rdrag = 149.77
        self.H0 = Planck18.H0.value
        self.c = constants.c.to('km/s').value
        self.coeff = self.c / (self.H0 * self.rdrag)
        
        self.include_D_M = include_D_M
        self.sigma_D_H = sigma_D_H
        self.sigma_D_M = sigma_D_M
        
        # Context dimension calculation
        # Context = design (redshift) + observations
        self.context_dim = 1  # redshift
        if include_D_M:
            self.context_dim += 2  # D_H and D_M
        else:
            self.context_dim += 1  # D_H only
        
        # Initialize the priors
        if priors_path is None:
            priors_path = os.path.join(script_dir, 'priors.yaml')
        with open(priors_path, 'r') as file:
            self.prior_data = yaml.safe_load(file)
        self.priors, self.param_constraints, self.latex_labels = self.get_priors(priors_path)
        self.desi_priors, _, _ = self.get_priors(priors_path)  # For now, same as priors
        self.cosmo_params = list(self.priors.keys())
        
        self.transform_input = transform_input
        if self.transform_input:
            # Initialize bijector for parameter transformation
            self.param_bijector = Bijector(self, cdf_bins=5000, cdf_samples=3e7)
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
        self.observation_labels = ["D_H"]
        if self.include_D_M:
            self.observation_labels.append("D_M")
        
        # Initialize designs
        self.init_designs(
            fixed_design=fixed_design, 
            design_step=design_step, 
            design_lower=design_lower, 
            design_upper=design_upper
        )
        
        # Nominal design (single redshift point for evaluation)
        self.nominal_design = torch.tensor([2.5], device=self.device).unsqueeze(0)
        if self.include_D_M:
            self.nominal_context = torch.cat([
                self.nominal_design,
                torch.zeros(1, 2, device=self.device)  # Placeholder observations
            ], dim=-1)
        else:
            self.nominal_context = torch.cat([
                self.nominal_design,
                torch.zeros(1, 1, device=self.device)  # Placeholder observation
            ], dim=-1)
        
        if self.global_rank == 0 and self.verbose:
            print(f"Variable Redshift Experiment Initialized")
            print(f"  Cosmology model: {self.cosmo_model}")
            print(f"  Parameters: {self.cosmo_params}")
            print(f"  Observation labels: {self.observation_labels}")
            print(f"  Context dimension: {self.context_dim}")
            print(f"  Include D_M: {self.include_D_M}")
            print(f"  Number of designs: {self.designs.shape[0]}")

    @profile_method
    def init_designs(self, fixed_design=False, design_step=0.1, design_lower=0.0, design_upper=5.0):
        """Initialize the redshift design grid."""
        if fixed_design:
            # Use a single fixed redshift
            designs = torch.tensor([[2.5]], device=self.device)
        else:
            # Create a grid of redshift values
            z_values = torch.arange(design_lower, design_upper + design_step, design_step, device=self.device)
            designs = z_values.unsqueeze(1)
        
        self.designs = designs.to(self.device)
        
        if self.global_rank == 0 and self.verbose:
            print(f"Initialized {self.designs.shape[0]} redshift designs from z={design_lower} to z={design_upper}")

    @profile_method
    def get_priors(self, prior_path):
        """Load cosmological priors and constraints from a YAML configuration file."""
        try:
            with open(prior_path, 'r') as file:
                prior_data = yaml.safe_load(file)
            with open(os.path.join(os.path.dirname(__file__), 'models.yaml'), 'r') as file:
                cosmo_models = yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        # Get the specified cosmology model
        if self.cosmo_model not in cosmo_models:
            raise ValueError(f"Cosmology model '{self.cosmo_model}' not found in models.yaml. Available models: {list(cosmo_models.keys())}")
        
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

    def _E_of_z(self, z, Om, Ok, w0, wa, Or, Onu0, Ode0, n_massive, cache):
        """
        Compute E(z) = H(z)/H0 including radiation and massive neutrinos.
        
        Args:
            z:      (plate, Nz) redshift array
            Om,...: (plate, 1) cosmological parameters
            cache:  _NeutrinoTableCache object (or None if n_massive==0)
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
        """
        High-accuracy Hubble distance divided by the sound horizon D_H/r_d.
        Includes radiation and massive neutrino contributions.
        """
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
        """
        High-accuracy transverse comoving distance divided by the sound horizon D_M/r_d.
        Uses Simpson's rule integration in ln(a) coordinates with sophisticated curvature handling.
        """
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
    def sample_valid_parameters(self, sample_shape, priors=None):
        """Sample parameters from priors with constraints."""
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
            # If no parameter constraints defined
            if 'Om' in priors.keys():
                parameters['Om'] = pyro.sample('Om', priors['Om']).unsqueeze(-1)
            if 'Ok' in priors.keys():
                parameters['Ok'] = pyro.sample('Ok', priors['Ok']).unsqueeze(-1)
            if 'w0' in priors.keys():
                parameters['w0'] = pyro.sample('w0', priors['w0']).unsqueeze(-1)
            if 'wa' in priors.keys():
                parameters['wa'] = pyro.sample('wa', priors['wa']).unsqueeze(-1)
            
        # Always sample hrdrag if it's in the model
        if 'hrdrag' in priors.keys():
            parameters['hrdrag'] = pyro.sample('hrdrag', priors['hrdrag']).unsqueeze(-1)

        return parameters

    @profile_method
    def pyro_model(self, z):
        """Pyro model for generating observations given redshift designs."""
        with pyro.plate_stack("plate", z.shape[:-1]):
            parameters = self.sample_valid_parameters(z.shape[:-1])
            
            # Compute mean predictions
            D_H_mean = self.D_H_func(z, **parameters)
            D_H_obs = pyro.sample("D_H", dist.Normal(D_H_mean, self.sigma_D_H).to_event(1))
            
            # Sample observations
            if self.include_D_M:
                D_M_mean = self.D_M_func(z, **parameters)
                D_M_obs = pyro.sample("D_M", dist.Normal(D_M_mean, self.sigma_D_M).to_event(1))
                return D_H_obs, D_M_obs
            else:
                return D_H_obs
                

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
    
    def get_desi_samples(self, num_samples=100000, params=None, transform_output=False):
        """Get DESI samples (or placeholder for variable redshift)."""
        # For variable_redshift, we might not have DESI samples
        # Return samples from the prior as a placeholder
        if self.global_rank == 0:
            print("Warning: get_desi_samples not fully implemented for variable_redshift. Returning prior samples.")
        
        with torch.no_grad():
            with pyro.plate("samples", num_samples):
                param_dict = self.sample_valid_parameters((num_samples,))
            
            # Convert to array
            param_samples = torch.cat([param_dict[k] for k in self.cosmo_params], dim=-1)
        
        if transform_output:
            param_samples = self.params_to_unconstrained(param_samples, bijector_class=self.desi_bijector)
        
        param_samples = param_samples.cpu().numpy()
        
        if params is None:
            names = self.cosmo_params
            labels = self.latex_labels
        else:
            param_indices = [self.cosmo_params.index(param) for param in params if param in self.cosmo_params]
            param_samples = param_samples[:, param_indices]
            names = [self.cosmo_params[i] for i in param_indices]
            labels = [self.latex_labels[i] for i in param_indices]

        with contextlib.redirect_stdout(io.StringIO()):
            samples_gd = getdist.MCSamples(samples=param_samples, names=names, labels=labels)

        return samples_gd

    @profile_method
    def sample_data(self, designs, num_samples=100, central=False):
        """Sample data from the likelihood."""
        # For variable_redshift, designs are redshift values
        # Generate samples from the pyro model
        expanded_designs = lexpand(designs, num_samples)
        
        with torch.no_grad():
            data_samples = self.pyro_model(expanded_designs)
        
        return data_samples
    
    def sample_params_from_data_samples(self, designs, guide, num_data_samples=100, num_param_samples=1000, central=False, transform_output=True):
        """Sample parameters from the posterior distribution conditioned on data."""
        data_samples = self.sample_data(designs, num_data_samples, central)
        
        # Create context: concatenate design and observations
        if isinstance(data_samples, tuple):
            # Multiple observations (D_H and D_M)
            context = torch.cat([designs.expand(num_data_samples, -1), data_samples[0], data_samples[1]], dim=-1)
        else:
            # Single observation (D_H only)
            context = torch.cat([designs.expand(num_data_samples, -1), data_samples], dim=-1)
        
        # Sample parameters conditioned on the data
        param_samples = self.get_guide_samples(guide, context.squeeze(), num_param_samples, transform_output)
        
        return param_samples

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
