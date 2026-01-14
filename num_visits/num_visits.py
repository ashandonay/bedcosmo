import contextlib
import io
import math
import os

from astropy import units as u
from astropy.constants import sigma_sb, h, c, k_B
import galsim  # type: ignore
import numpy as np
import pyro
from pyro import distributions as dist
from pyro.contrib.util import lexpand
import torch
import yaml

from speclite import filters as speclite_filters
from astropy.constants import h, c
from bed.grid import Grid
import getdist
from util import load_prior_flow_from_file

# LSST photometric zeropoints (AB magnitudes that produce 1 count per second)
# From SMTN-002 (v1.9 throughputs): https://smtn-002.lsst.io
# Based on syseng_throughputs v1.9 with triple silver mirror coatings
# and as-measured filter/lens/detector throughputs
s0 = {'u': 26.52,
      'g': 28.51,
      'r': 28.36,
      'i': 28.17,
      'z': 27.78,
      'y': 26.82}

# Sky brightnesses in AB mag / arcsec^2 (zenith, dark sky)
# From SMTN-002: https://smtn-002.lsst.io
# Based on dark sky spectrum from UVES/Gemini/ESO, normalized to match SDSS observations
B = {'u': 23.05,
     'g': 22.25,
     'r': 21.2,
     'i': 20.46,
     'z': 19.61,
     'y': 18.6}

fiducial_nvisits = {'u': 70,
                    'g': 100,
                    'r': 230,
                    'i': 230,
                    'z': 200,
                    'y': 200}
# Sky brightness per arcsec^2 per second
# At sky magnitude B[k]: flux = 10^(-0.4*(B[k] - s0[k])) photons/sec/arcsec^2
sbar = {}
for k in B:
    sbar[k] = 10**(-0.4*(B[k] - s0[k]))
from util import profile_method

# Pre-compute physical constants for blackbody computation
_h_cgs = h.cgs.value
_c_cgs = c.cgs.value
_k_B_cgs = k_B.cgs.value
_hc = _h_cgs * _c_cgs
_two_hc2 = 2 * _h_cgs * _c_cgs**2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


class NumVisits:
    """
    Experiment that models LSST magnitude measurements as a function of redshift
    and per-filter visit allocations.
    """

    def __init__(
        self,
        filters_list=None,
        temperature=5000,
        z_prior_bounds=(0.1, 3.0),
        fixed_design=False,
        design_args=None,
        nominal_design=None,
        priors_path=None,
        pixel_scale=0.2,
        stamp_size=31,
        threshold=0.0,
        exposure_time=15.0,
        n_exp_per_visit=2,
        read_noise=8.8,
        dark_current=0.2,
        device="cuda:0",
        transform_input=False,
        profile=False,
        verbose=False,
        global_rank=0,
    ):
        self.name = "num_visits"
        self.device = device
        self.profile = profile
        self.verbose = verbose
        self.transform_input = transform_input
        self.global_rank = global_rank

        self.filters_list = list(filters_list) if filters_list else ["u", "g", "r", "i", "z", "y"]
        self.num_filters = len(self.filters_list)
        
        # Extract labels from design_args if provided, otherwise generate from filters_list
        if design_args is not None and 'labels' in design_args:
            self.design_labels = design_args['labels']
        else:
            self.design_labels = [f"n_{band}" for band in self.filters_list]
        
        self.observation_labels = ["magnitudes"]
        self.context_dim = len(self.design_labels) + self.num_filters

        if isinstance(temperature, (int, float)):
            self.temperature = float(temperature) * u.K
        else:
            self.temperature = temperature

        z_low, z_high = z_prior_bounds
        if z_low >= z_high:
            raise ValueError("z_prior_bounds must satisfy lower < upper.")
        self.z_prior_bounds = z_prior_bounds
        if priors_path is None:
            priors_path = os.path.join(SCRIPT_DIR, "priors.yaml")
        self.priors, self.latex_labels, self.prior_data = self.get_priors(priors_path)
        self.cosmo_params = list(self.priors.keys())
        self.z_prior = self.priors[self.cosmo_params[0]]
        
        # Check if prior_flow is specified in priors YAML
        self.prior_flow = None
        if 'prior_flow' in self.prior_data and self.prior_data['prior_flow'] is not None and self.prior_data['prior_flow'] != '':
            prior_flow_path = self.prior_data['prior_flow']
            # Handle relative paths - assume relative to priors_path directory
            if not os.path.isabs(prior_flow_path):
                prior_flow_path = os.path.join(os.path.dirname(priors_path), prior_flow_path)
            
            # prior_run_id is required to get metadata from MLflow
            prior_run_id = self.prior_data.get('prior_run_id', None)
            if prior_run_id is None:
                raise ValueError("prior_run_id must be specified in priors.yaml when using prior_flow")
            
            
            self.prior_flow, prior_flow_metadata = load_prior_flow_from_file(
                prior_flow_path,
                prior_run_id,
                self.device,
                self.global_rank
            )
            # Store metadata in experiment class, not on the flow model
            self.prior_flow_transform_input = prior_flow_metadata['transform_input']
            # Set nominal context for num_visits
            # Context = design (nvisits per filter) + observations (magnitudes per filter)
            if prior_flow_metadata['nominal_context'] is not None:
                self.prior_flow_nominal_context = prior_flow_metadata['nominal_context']
            else:
                nominal_observations = torch.zeros(self.num_filters, device=self.device, dtype=torch.float64)
                self.prior_flow_nominal_context = torch.cat([self.nominal_design, nominal_observations], dim=-1)
            if self.global_rank == 0:
                print("Using trained posterior model as prior for parameter sampling (loaded from priors.yaml)")
                if prior_run_id:
                    print(f"  Metadata loaded from MLflow run_id: {prior_run_id}")

        if nominal_design is None:
            self.nominal_design = torch.tensor(
                [fiducial_nvisits[band] for band in self.filters_list], device=self.device, dtype=torch.float64
            )
        else:
            nominal_array = np.asarray(nominal_design, dtype=np.float64)
            if nominal_array.shape != (self.num_filters,):
                raise ValueError(
                    f"nominal_design must have shape ({self.num_filters},), got {nominal_array.shape}"
                )
            self.nominal_design = torch.tensor(nominal_array, device=self.device, dtype=torch.float64)

        self.pixel_scale = pixel_scale
        self.stamp_size = stamp_size
        self.threshold = threshold
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.n_exp_per_visit = n_exp_per_visit
        self.exposure_time = exposure_time
        self.visit_time = exposure_time * n_exp_per_visit
        self._base_img = self._calculate_base_profile()
        self._s0_array = np.array([s0[band] for band in self.filters_list], dtype=np.float64)
        self._sbar_array = np.array([sbar[band] for band in self.filters_list], dtype=np.float64)

        # Cache filter data to avoid reloading on every call
        # Create a common wavelength grid and store all filters as tensors for parallel processing
        hc_erg_angstrom = (h * c).to(u.erg * u.Angstrom).value
        downsample_factor = 5
        
        # First pass: collect all wavelength ranges to create common grid
        all_wlen_min = []
        all_wlen_max = []
        filter_data_list = []
        
        for band in self.filters_list:
            loaded_filter = speclite_filters.load_filter("lsst2023-"+band)
            wlen = loaded_filter.wavelength * u.AA
            transmission_result = loaded_filter(wlen)
            transmission = transmission_result if isinstance(transmission_result, np.ndarray) else transmission_result.value
            
            # Downsample to reduce computation
            if downsample_factor > 1:
                wlen_aa_downsampled = wlen.value[::downsample_factor]
                transmission_downsampled = transmission[::downsample_factor]
            else:
                wlen_aa_downsampled = wlen.value
                transmission_downsampled = transmission
            
            all_wlen_min.append(wlen_aa_downsampled.min())
            all_wlen_max.append(wlen_aa_downsampled.max())
            filter_data_list.append({
                'band': band,
                'wlen_aa': wlen_aa_downsampled,
                'transmission': transmission_downsampled,
            })
        
        # Create common wavelength grid covering all filters
        wlen_min = min(all_wlen_min)
        wlen_max = max(all_wlen_max)
        # Use the maximum number of points from any filter, or a reasonable default
        max_points = max(len(fd['wlen_aa']) for fd in filter_data_list)
        wlen_common_aa = np.linspace(wlen_min, wlen_max, max_points)
        wlen_common_cm = wlen_common_aa * 1e-8  # Convert Angstrom to cm
        wlen_over_hc_common = (wlen_common_aa / hc_erg_angstrom)
        
        # Interpolate all filters to common grid
        from scipy.interpolate import interp1d
        transmission_array = np.zeros((self.num_filters, len(wlen_common_aa)), dtype=np.float64)
        
        for i, filter_data in enumerate(filter_data_list):
            # Interpolate transmission to common grid
            interp_func = interp1d(
                filter_data['wlen_aa'],
                filter_data['transmission'],
                kind='linear',
                bounds_error=False,
                fill_value=0.0  # Outside filter range, transmission is 0
            )
            transmission_array[i, :] = interp_func(wlen_common_aa)
        
        # Store as tensors on device
        self._wlen_aa_tensor = torch.tensor(wlen_common_aa, device=self.device, dtype=torch.float64)  # (n_wlen,)
        self._wlen_cm_tensor = torch.tensor(wlen_common_cm, device=self.device, dtype=torch.float64)  # (n_wlen,)
        self._transmission_tensor = torch.tensor(transmission_array, device=self.device, dtype=torch.float64)  # (n_filters, n_wlen)
        self._wlen_over_hc_tensor = torch.tensor(wlen_over_hc_common, device=self.device, dtype=torch.float64)  # (n_wlen,)

        # Pass design_args using ** unpacking if provided, otherwise use defaults
        if design_args is not None:
            self.init_designs(fixed_design=fixed_design, **design_args)
        else:
            self.init_designs(fixed_design=fixed_design)

        if self.global_rank == 0 and self.verbose:
            print(f"Num Visits Experiment Initialized")
            print(f"  Filters: {self.filters_list}")
            print(f"prior z∈[{z_low}, {z_high}]")
            print(f"  Number of designs: {self.designs.shape[0]}")
            print(f"  Nominal design: {self.nominal_design}")

    @profile_method
    def _expand_to_filters(self, value, label):
        if isinstance(value, (int, float)):
            return [float(value)] * self.num_filters
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != self.num_filters:
                raise ValueError(f"{label} must have length {self.num_filters}, got {len(value)}")
            return [float(v) for v in value]
        raise TypeError(f"{label} must be a float or sequence, got {type(value)}")

    @profile_method
    @profile_method
    def get_priors(self, prior_path):
        with open(prior_path, "r") as file:
            prior_data = yaml.safe_load(file)
        parameters = prior_data.get("parameters", {})
        priors = {}
        latex_labels = []
        for name, cfg in parameters.items():
            dist_cfg = cfg.get("distribution", {})
            if dist_cfg.get("type") != "uniform":
                raise ValueError(f"Only uniform priors supported (got {dist_cfg.get('type')})")
            lower = float(dist_cfg.get("lower", 0.0))
            upper = float(dist_cfg.get("upper", 1.0))
            priors[name] = dist.Uniform(
                torch.tensor(lower, device=self.device, dtype=torch.float64),
                torch.tensor(upper, device=self.device, dtype=torch.float64),
            )
            latex_labels.append(cfg.get("latex", name))
        return priors, latex_labels, prior_data

    @profile_method
    def _calculate_base_profile(self):
        gal = galsim.Gaussian(flux=1.0, sigma=2.0)
        psf = galsim.Gaussian(fwhm=0.67)
        profile = galsim.Convolve([gal, psf])
        image = galsim.ImageD(self.stamp_size, self.stamp_size, scale=self.pixel_scale)
        profile.drawImage(image=image)
        return image.array.copy()

    @profile_method
    def init_designs(self, fixed_design=False, step=20.0, lower=10.0, upper=250.0, sum_lower=None, sum_upper=None, labels=None):
        # Expand parameters to filters if needed
        design_step = self._expand_to_filters(step, "step")
        design_lower = self._expand_to_filters(lower, "lower")
        design_upper = self._expand_to_filters(upper, "upper")
        
        # Set sum_lower and sum_upper defaults if not provided
        if sum_lower is None:
            sum_lower = float(self.nominal_design.sum()) if hasattr(self, 'nominal_design') else None
        if sum_upper is None:
            sum_upper = float(self.nominal_design.sum()) if hasattr(self, 'nominal_design') else None
        
        if isinstance(fixed_design, (list, tuple, np.ndarray, torch.Tensor)):
            designs_np = np.asarray(fixed_design, dtype=np.float64)
            if designs_np.ndim == 1:
                designs_np = designs_np.reshape(1, -1)
            if designs_np.shape[1] != self.num_filters:
                raise ValueError(
                    f"fixed_design must have {self.num_filters} columns, got {designs_np.shape[1]}"
                )
            designs = torch.tensor(designs_np, device=self.device, dtype=torch.float64)
        elif fixed_design:
            midpoint = torch.tensor(
                [[(lo + hi) / 2.0 for lo, hi in zip(design_lower, design_upper)]],
                device=self.device,
                dtype=torch.float64,
            )
            designs = midpoint
        else:
            design_axes = {}
            for idx, label in enumerate(labels):
                design_axes[label] = np.arange(
                    design_lower[idx],
                    design_upper[idx] + 0.5 * design_step[idx],
                    design_step[idx],
                    dtype=np.float64,
                )

            constraint = None
            if sum_lower is not None or sum_upper is not None:
                lower_bound = sum_lower if sum_lower is not None else -np.inf
                upper_bound = sum_upper if sum_upper is not None else np.inf

                def _constraint(**kwargs):
                    total = sum(kwargs.values())
                    within = np.logical_and(total >= lower_bound - 1e-9, total <= upper_bound + 1e-9)
                    return within.astype(int)

                constraint = _constraint

            grid = Grid(**design_axes, constraint=constraint) if constraint else Grid(**design_axes)

            designs = torch.tensor(
                getattr(grid, grid.names[0]).squeeze(), device=self.device, dtype=torch.float64
            ).unsqueeze(1)
            for name in grid.names[1:]:
                values = torch.tensor(
                    getattr(grid, name).squeeze(), device=self.device, dtype=torch.float64
                ).unsqueeze(1)
                designs = torch.cat([designs, values], dim=1)

        self.designs = designs
        if self.designs.numel() == 0:
            if self.verbose:
                print("Design grid empty under current constraint; defaulting to nominal design.")
            self.designs = self.nominal_design.unsqueeze(0).to(self.device)

    @profile_method
    def _luminosity_distance(self, z, n_int=1025):
        """
        Compute luminosity distance using tensor operations (GPU-compatible).
        Uses Planck18 cosmology: H0=67.4 km/s/Mpc, Om=0.315, flat universe.
        
        For a flat universe: D_L(z) = (1+z) * c/H0 * ∫[0 to z] dz'/E(z')
        Where E(z) = sqrt(Om*(1+z)^3 + (1-Om))
        
        Args:
            z: torch.Tensor, redshift values
            n_int: int, number of integration points
            
        Returns:
            torch.Tensor: Luminosity distance in cm (same shape as z)
        """
        DTYPE = torch.float64
        dev = self.device
        
        # Planck18 parameters
        H0_km_s_Mpc = 67.4  # km/s/Mpc
        # 1 Mpc = 3.086e22 m = 3.086e24 cm
        Mpc_to_cm = 3.086e24  # cm
        H0_cm_s = H0_km_s_Mpc * 1e5  # cm/s (1 km = 1e5 cm)
        H0_s_inv = H0_cm_s / Mpc_to_cm  # s^-1
        c_cm_s = 2.998e10  # cm/s
        Om = 0.315
        
        # Convert z to tensor and flatten
        z_t = torch.as_tensor(z, device=dev, dtype=DTYPE)
        original_shape = z_t.shape
        z_flat = z_t.flatten()
        
        if z_flat.numel() == 0:
            return torch.zeros_like(z_t)
        
        z_max = float(z_flat.max())
        if z_max <= 0:
            return torch.zeros_like(z_t)
        
        # Create integration grid from 0 to z_max
        z_grid = torch.linspace(0.0, z_max, n_int, device=dev, dtype=DTYPE)
        
        # Merge evaluation points into grid
        z_all = torch.unique(torch.cat([z_grid, z_flat])).sort().values
        # Ensure we have 0 in the grid
        if z_all[0] > 1e-10:
            z_all = torch.cat([torch.zeros(1, device=dev, dtype=DTYPE), z_all])
        # Ensure odd number of points for Simpson's rule
        if z_all.numel() % 2 == 0:
            mid = 0.5 * (z_all[-2] + z_all[-1])
            z_all = torch.unique(torch.cat([z_all, mid.unsqueeze(0)])).sort().values
        
        # Compute E(z) = sqrt(Om*(1+z)^3 + (1-Om)) for flat universe
        zp1 = 1.0 + z_all
        E_z = torch.sqrt(Om * zp1**3 + (1.0 - Om))
        
        # Integrand: 1/E(z)
        integrand = 1.0 / E_z
        
        # Cumulative integral using Simpson's rule
        cum_int = _cumsimpson(z_all, integrand, dim=-1)
        
        # Interpolate to get values at z_flat
        # Find indices where z_flat should be inserted
        idx = torch.searchsorted(z_all, z_flat)
        # Clamp indices to valid range
        idx = torch.clamp(idx, 0, z_all.numel() - 1)
        # For exact matches, use the value directly
        exact_match = torch.abs(z_all[idx] - z_flat) < 1e-10
        # For non-exact, interpolate linearly
        idx_lower = torch.clamp(idx - 1, 0, z_all.numel() - 1)
        z_lower = z_all[idx_lower]
        z_upper = z_all[idx]
        cum_lower = cum_int[idx_lower]
        cum_upper = cum_int[idx]
        
        # Linear interpolation
        alpha = torch.where(exact_match, torch.zeros_like(z_flat),
                           (z_flat - z_lower) / (z_upper - z_lower + 1e-10))
        dc_over_dh0 = torch.where(exact_match, cum_upper,
                                  cum_lower + alpha * (cum_upper - cum_lower))
        
        # Comoving distance: D_C = (c/H0) * ∫ dz'/E(z')
        dc_cm = (c_cm_s / H0_s_inv) * dc_over_dh0
        
        # Luminosity distance for flat universe: D_L = (1+z) * D_C
        dl_cm = (1.0 + z_flat) * dc_cm
        
        # Reshape to original shape
        return dl_cm.reshape(original_shape)

    def _blackbody_flux(self, wlen_cm, T_K):
        """Compute blackbody flux using PyTorch tensors (GPU-accelerated)."""
        # Use float32 for faster computation (sufficient precision)
        # Physical constants (pre-computed, create tensors once)
        if not hasattr(self, '_hc_tensor_f32'):
            self._hc_tensor_f32 = torch.tensor(_hc, device=self.device, dtype=torch.float32)
            self._k_B_tensor_f32 = torch.tensor(_k_B_cgs, device=self.device, dtype=torch.float32)
            self._two_hc2_tensor_f32 = torch.tensor(_two_hc2, device=self.device, dtype=torch.float32)
            self._pi_tensor_f32 = torch.tensor(np.pi, device=self.device, dtype=torch.float32)
            self._1e8_tensor_f32 = torch.tensor(1e-8, device=self.device, dtype=torch.float32)
        
        # Convert inputs to float32 for speed
        wlen_cm_f32 = wlen_cm.to(torch.float32)
        T_K_f32 = T_K.to(torch.float32)
        
        hc_over_kT = self._hc_tensor_f32 / (self._k_B_tensor_f32 * T_K_f32)
        exponent = hc_over_kT / wlen_cm_f32
        
        # Use expm1 directly - it's optimized in PyTorch
        # For very large negative exponents, exp(x) - 1 ≈ -1, so we can approximate
        # For moderate values, use expm1
        # This avoids expensive exp() for very large negative values
        # Use a threshold to avoid exp() for very negative values
        very_large_neg = exponent < -20.0
        moderate = ~very_large_neg
        
        exp_term = torch.zeros_like(exponent)
        # For very large negative: exp(x) - 1 ≈ -1 (since exp(x) ≈ 0)
        exp_term[very_large_neg] = -1.0
        # For moderate: use expm1 (optimized in PyTorch)
        exp_term[moderate] = torch.expm1(exponent[moderate])
        exp_term = torch.clamp(exp_term, min=torch.finfo(torch.float32).tiny)
        
        wlen_cm_5 = wlen_cm_f32**5
        B = self._two_hc2_tensor_f32 / (wlen_cm_5 * exp_term)
        F = self._pi_tensor_f32 * B
        F_per_AA = F * self._1e8_tensor_f32
        
        # Convert back to float64 for consistency
        return F_per_AA.to(torch.float64)
    
    @profile_method
    def _calculate_magnitudes(self, z_tensor):
        """
        Calculate magnitudes across each filter.
        
        Args:
            z_tensor: Tensor of redshift values, shape (n_z,)
            
        Returns:
            Tensor of magnitudes with shape (n_z, n_filters)
        """
        # Flatten and find unique z values to avoid redundant calculations
        z_flat = z_tensor.flatten()  # (n_z,)
        z_unique, inverse_indices = torch.unique(z_flat, return_inverse=True)
        lum_dist = self._luminosity_distance(z_unique)
        
        # Extract T value (cache it)
        if not hasattr(self, '_T_K_tensor'):
            if hasattr(self.temperature, 'value'):
                self._T_K_tensor = torch.tensor(self.temperature.to(u.K).value, device=self.device, dtype=torch.float64)
            else:
                self._T_K_tensor = torch.tensor(float(self.temperature), device=self.device, dtype=torch.float64)
        T_K = self._T_K_tensor
        
        # Pre-compute luminosity constants (cache them)
        if not hasattr(self, '_R_eff_tensor'):
            L_sun = 3.826e33
            L_bol = 1e9 * L_sun
            sigma_sb_cgs = sigma_sb.to(u.erg / (u.s * u.cm**2 * u.K**4)).value
            T_K_4 = T_K**4
            R_eff_val = torch.sqrt(torch.tensor(L_bol / (4 * np.pi * sigma_sb_cgs), device=self.device, dtype=torch.float64) / T_K_4)
            self._R_eff_tensor = R_eff_val
            self._four_pi_R2_tensor = torch.tensor(4 * np.pi, device=self.device, dtype=torch.float64) * self._R_eff_tensor**2
        
        # Reshape for broadcasting: z is (n_z,), wlen is (n_wlen,)
        z_2d = z_unique.unsqueeze(-1)  # (n_z, 1)
        wlen_2d = self._wlen_aa_tensor.unsqueeze(0)  # (1, n_wlen)
        
        # Rest-frame wavelength
        if not hasattr(self, '_1e8_tensor_f64'):
            self._1e8_tensor_f64 = torch.tensor(1e-8, device=self.device, dtype=torch.float64)
        one_plus_z = 1 + z_2d  # (n_z, 1)
        wlen_rest_aa = wlen_2d / one_plus_z  # (n_z, n_wlen)
        wlen_rest_cm = wlen_rest_aa * self._1e8_tensor_f64  # Convert Angstrom to cm
        
        # Blackbody flux (GPU-accelerated)
        F = self._blackbody_flux(wlen_rest_cm, T_K)  # (n_z, n_wlen)
        L = self._four_pi_R2_tensor * F  # (n_z, n_wlen)
        
        # Observed flux
        lum_dist_2d = lum_dist.unsqueeze(-1)  # (n_z, 1)
        if not hasattr(self, '_four_pi_tensor'):
            self._four_pi_tensor = torch.tensor(4 * np.pi, device=self.device, dtype=torch.float64)
        flux = L / (one_plus_z * self._four_pi_tensor * lum_dist_2d**2)  # (n_z, n_wlen)
        
        # Expand flux for all filters: (n_z, 1, n_wlen)
        flux_expanded = flux.unsqueeze(1)  # (n_z, 1, n_wlen)
        
        # Expand transmission: (1, n_filters, n_wlen)
        transmission_expanded = self._transmission_tensor.unsqueeze(0)  # (1, n_filters, n_wlen)
        
        # Expand wlen_over_hc: (1, 1, n_wlen)
        wlen_over_hc_expanded = self._wlen_over_hc_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, n_wlen)
        
        # Integrand for all filters: (n_z, n_filters, n_wlen)
        integrand = flux_expanded * transmission_expanded * wlen_over_hc_expanded
        
        # Use torch.trapezoid to integrate over wavelength dimension (last dimension)
        photon_flux = torch.trapezoid(integrand, self._wlen_aa_tensor, dim=-1)  # (n_z, n_filters)
        
        # Convert to magnitude for all filters
        A_cm2 = (319/9.6) * 1e4  # cm^2
        photon_flux_pixel = photon_flux * A_cm2  # (n_z, n_filters)
        
        # s0 values for all filters: (n_filters,)
        s0_vals = torch.tensor([s0[band] for band in self.filters_list], device=self.device, dtype=torch.float64)
        s0_vals = s0_vals.unsqueeze(0)  # (1, n_filters) for broadcasting
        
        # Magnitudes: (n_z, n_filters)
        mags_unique = 24.0 - 2.5 * torch.log10(photon_flux_pixel / s0_vals)
        
        # Map back to original z values
        mags_flat = mags_unique[inverse_indices]  # (n_z, n_filters)
        
        # Reshape to match z_tensor shape, then add filter dimension
        mags_reshaped = mags_flat.reshape(*z_tensor.shape, self.num_filters)
        
        return mags_reshaped

    @profile_method
    def _magnitude_errors(self, mags, nvisits):
        mags_np = mags.detach().cpu().numpy()
        nvisits_np = nvisits.detach().cpu().numpy()

        pad_shape = (1,) * (mags_np.ndim - 1) + (self.num_filters,)
        s0 = self._s0_array.reshape(pad_shape)
        sbar = self._sbar_array.reshape(pad_shape)

        fluxes = 10.0 ** (-0.4 * (mags_np - s0)) * nvisits_np * self.visit_time
        pixels = fluxes[..., np.newaxis, np.newaxis] * self._base_img

        sigma_sky = np.sqrt(sbar * nvisits_np * self.visit_time * (self.pixel_scale**2))
        mask = pixels > (self.threshold * sigma_sky[..., np.newaxis, np.newaxis])
        masked_pixels = pixels * mask

        signal = (masked_pixels**2).sum(axis=(-2, -1))
        sky_var = sbar * nvisits_np * self.visit_time * (self.pixel_scale**2)
        dark_var = self.dark_current * nvisits_np * self.visit_time
        read_var = (self.read_noise**2) * nvisits_np * self.n_exp_per_visit
        src_var = masked_pixels

        total_var = (
            sky_var[..., np.newaxis, np.newaxis]
            + src_var
            + dark_var[..., np.newaxis, np.newaxis]
            + read_var[..., np.newaxis, np.newaxis]
        )
        noise = np.sqrt((masked_pixels**2 * total_var).sum(axis=(-2, -1)))
        snr = np.where(noise == 0, 0.0, signal / noise)
        coeff = 2.5 / np.log(10.0)
        mag_err = np.where(
            snr <= 0,
            1e6,
            coeff / snr,
        )

        errors = torch.tensor(mag_err, device=self.device, dtype=torch.float64)
        return torch.clamp(errors, min=1e-6)

    @profile_method
    def pyro_model(self, nvisits):
        nvisits = torch.as_tensor(nvisits, dtype=torch.float64, device=self.device)
        if nvisits.ndim == 2:
            nvisits = nvisits.unsqueeze(1)
        elif nvisits.ndim == 1:
            nvisits = nvisits.view(1, 1, -1)

        batch_shape = nvisits.shape[:-1]
        
        # Check if we should use a posterior model as prior
        if hasattr(self, 'prior_flow') and self.prior_flow is not None:
            z = self._sample_z_from_prior_flow(batch_shape, nvisits)
        else:
            z_dist = self.z_prior.expand(batch_shape).to_event(0)
            z = pyro.sample("z", z_dist)
        
        means = self._calculate_magnitudes(z)
        sigmas = self._magnitude_errors(means, nvisits)
        covariance = torch.diag_embed(sigmas**2)
        return pyro.sample(self.observation_labels[0], dist.MultivariateNormal(means, covariance))
    
    def _sample_z_from_prior_flow(self, batch_shape, nvisits):
        """
        Sample z from a trained posterior model used as prior.
        
        The posterior model is conditional on context (design + observations),
        so we sample at the nominal context (nominal design + zero observations).
        """
        # Get the total number of samples needed
        total_samples = int(np.prod(batch_shape)) if batch_shape else 1
        
        # Get nominal context for sampling
        nominal_context = getattr(self, 'prior_flow_nominal_context', None)
        if nominal_context is None:
            # Fallback: create nominal context
            nominal_observations = torch.zeros(self.num_filters, device=self.device, dtype=torch.float64)
            nominal_context = torch.cat([self.nominal_design, nominal_observations], dim=-1)
        
        # Expand context to match sample shape
        expanded_context = nominal_context.unsqueeze(0).expand(total_samples, -1)
        
        # Sample from the posterior flow
        with torch.no_grad():
            posterior_dist = self.prior_flow(expanded_context)
            samples = posterior_dist.sample()  # Shape: (total_samples, 1) for z
        
        # Transform from unconstrained to constrained space if needed
        prior_transform_input = getattr(self, 'prior_flow_transform_input', False)
        if prior_transform_input:
            # Prior flow outputs unconstrained parameters, transform to constrained space
            samples = self.params_from_unconstrained(samples)
        
        # Reshape to match batch_shape
        if batch_shape:
            z = samples.reshape(batch_shape + (1,)).squeeze(-1)
        else:
            z = samples.squeeze(-1)
        
        # Register with pyro.sample using Delta distribution
        z = pyro.sample("z", dist.Delta(z))
        
        return z

    @profile_method
    def get_priors(self, prior_path):
        """Load uniform priors for the redshift parameter."""
        try:
            with open(prior_path, "r") as file:
                prior_data = yaml.safe_load(file)
            with open(os.path.join(SCRIPT_DIR, "models.yaml"), "r") as file:
                cosmo_models = yaml.safe_load(file)
        except Exception as exc:
            raise ValueError(f"Error parsing YAML file: {exc}") from exc

        model_key = "base"
        if model_key not in cosmo_models:
            raise ValueError(f"Model '{model_key}' not found in models.yaml")

        model_parameters = cosmo_models[model_key]["parameters"]
        latex_labels = cosmo_models[model_key]["latex_labels"]

        priors = {}
        for idx, param_name in enumerate(model_parameters):
            if param_name not in prior_data["parameters"]:
                raise ValueError(f"Parameter '{param_name}' missing from priors.yaml")
            param_config = prior_data["parameters"][param_name]
            dist_config = param_config.get("distribution", {})
            if dist_config.get("type") != "uniform":
                raise ValueError("num_visits only supports uniform priors")
            lower = dist_config.get("lower")
            upper = dist_config.get("upper")
            if lower is None or upper is None or lower >= upper:
                raise ValueError(f"Invalid bounds for '{param_name}'")
            priors[param_name] = dist.Uniform(
                torch.tensor(lower, device=self.device, dtype=torch.float64),
                torch.tensor(upper, device=self.device, dtype=torch.float64),
            )
            latex_labels[idx] = param_config.get("latex", latex_labels[idx])

        return priors, latex_labels, prior_data

    @profile_method
    def params_to_unconstrained(self, params, bijector_class=None):
        return params

    @profile_method
    def params_from_unconstrained(self, y, bijector_class=None):
        return y

    @profile_method
    def get_guide_samples(
        self,
        guide,
        context=None,
        num_samples=5000,
        params=None,
        transform_output=True,
    ):
        if context is None:
            raise ValueError("context is required")
        with torch.no_grad():
            param_samples = guide(context.squeeze()).sample((num_samples,))

        if params is None:
            names = self.cosmo_params
            labels = self.latex_labels
        else:
            indices = [self.cosmo_params.index(p) for p in params if p in self.cosmo_params]
            param_samples = param_samples[:, indices]
            names = [self.cosmo_params[i] for i in indices]
            labels = [self.latex_labels[i] for i in indices]

        for idx in range(param_samples.shape[1]):
            col = param_samples[:, idx]
            if torch.all(col == col[0]):
                noise_scale = 1e-10 if col[0] == 0 else abs(col[0]) * 1e-10
                param_samples[:, idx] = col + torch.randn_like(col) * noise_scale

        with contextlib.redirect_stdout(io.StringIO()):
            samples_gd = getdist.MCSamples(
                samples=param_samples.cpu().numpy(), names=names, labels=labels
            )
        return samples_gd

    @profile_method
    def get_nominal_samples(self, *_, **__):
        raise NotImplementedError("Nominal samples not defined for num_visits.")

    @profile_method
    def sample_data(
        self,
        designs,
        num_samples=100,
        central=False,
    ):
        expanded_designs = lexpand(designs, num_samples)
        with torch.no_grad():
            data_samples = self.pyro_model(expanded_designs)
        return data_samples

    @profile_method
    def sample_params_from_data_samples(
        self,
        designs,
        guide,
        num_data_samples=100,
        num_param_samples=1000,
        central=False,
        transform_output=True,
    ):
        data_samples = self.sample_data(designs, num_data_samples, central)
        context = torch.cat(
            [designs.expand(num_data_samples, -1), data_samples],
            dim=-1,
        )
        param_samples_list = []
        for i in range(num_data_samples):
            context_i = context[i]
            param_samples_i = self.get_guide_samples(
                guide,
                context_i,
                num_samples=num_param_samples,
                transform_output=transform_output,
            )
            param_samples_list.append(param_samples_i.samples)
        return np.stack(param_samples_list, axis=0)

    @profile_method
    def unnorm_lfunc(self, params, features, designs):
        """
        Unnormalized likelihood compatible with bed.bayesdesign brute-force routines.
        """
        z_tensor = torch.as_tensor(getattr(params, "z"), device=self.device, dtype=torch.float64)
        mags_model = self._calculate_magnitudes(z_tensor).cpu().numpy()

        mag_obs = np.stack([getattr(features, f"mag_{band}") for band in self.filters_list], axis=-1)
        nvisits = np.stack([getattr(designs, f"n_{band}") for band in self.filters_list], axis=-1)

        sigmas = self._magnitude_errors(
            torch.as_tensor(mags_model, device=self.device, dtype=torch.float64),
            torch.as_tensor(nvisits, device=self.device, dtype=torch.float64),
        ).cpu().numpy()

        diff = (mag_obs - mags_model) / sigmas
        return np.exp(-0.5 * diff**2).prod(axis=-1)