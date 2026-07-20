import contextlib
import io
import math
import os
from pathlib import Path

from astropy import units as u
from astropy.constants import sigma_sb, h, c, k_B
import galsim  # type: ignore
import jax.numpy as jnp
import numpy as np
import pyro
from pyro import distributions as dist
from pyro.contrib.util import lexpand
import torch
import yaml

from speclite import filters as speclite_filters
from astropy.constants import h, c
import getdist
from bedcosmo.profiling import profile_method
from bedcosmo.util import (
    _central_params_as_dict,
    get_experiment_config_path,
    load_prior_flow_from_file,
)
from bedcosmo.base import BaseExperiment
from bedcosmo.custom_dist import EmpiricalPrior
from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
    mode_central_params_from_artifact,
)
from bedcosmo.num_visits.empirical.sed_prior import (
    PRIOR_SOURCE_FLOW,
    EmpiricalSedPrior,
    normalize_prior_source,
    resolve_runtime_prior_root,
)
from bedcosmo.num_visits.empirical.simplex import (
    PARAMETERIZATION_ILR,
    ilr_to_weights_torch,
)
from bedcosmo.num_visits.empirical.templates import load_eazy_template_bank
from bedcosmo.cosmology import CosmologyMixin, _cumsimpson

# LSST photometric zeropoints (AB magnitudes that produce 1 count per second)
# From SMTN-002 (v1.9 throughputs): https://smtn-002.lsst.io
# Based on syseng_throughputs v1.9 with triple silver mirror coatings
# and as-measured filter/lens/detector throughputs
s0 = {"u": 26.52, "g": 28.51, "r": 28.36, "i": 28.17, "z": 27.78, "y": 26.82}

# Sky brightnesses in AB mag / arcsec^2 (zenith, dark sky)
# From SMTN-002: https://smtn-002.lsst.io
# Based on dark sky spectrum from UVES/Gemini/ESO, normalized to match SDSS observations
B = {"u": 23.05, "g": 22.25, "r": 21.2, "i": 20.46, "z": 19.61, "y": 18.6}

fiducial_nvisits = {"u": 70, "g": 100, "r": 230, "i": 230, "z": 200, "y": 200}
# Sky brightness per arcsec^2 per second
# At sky magnitude B[k]: flux = 10^(-0.4*(B[k] - s0[k])) photons/sec/arcsec^2
sbar = {}
for k in B:
    sbar[k] = 10 ** (-0.4 * (B[k] - s0[k]))

# Pre-compute physical constants for blackbody computation
_h_cgs = h.cgs.value
_c_cgs = c.cgs.value
_k_B_cgs = k_B.cgs.value
_hc = _h_cgs * _c_cgs
_two_hc2 = 2 * _h_cgs * _c_cgs**2
_L_sun_cgs = 3.826e33  # erg/s

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class NumVisits(BaseExperiment, CosmologyMixin):
    """
    Experiment that models LSST magnitude measurements as a function of redshift
    and per-filter visit allocations.
    """

    def __init__(
        self,
        prior_args=None,
        design_args=None,
        cosmo_model=None,
        temperature=10000,
        l_bol=1e9,
        central_params=None,
        nominal_design=None,
        pixel_scale=0.2,
        stamp_size=31,
        threshold=0.0,
        exposure_time=15.0,
        n_exp_per_visit=2,
        read_noise=8.8,
        dark_current=0.2,
        mag_err_cap=10.0,
        device="cuda:0",
        transform_input=False,
        logit_flow_scale=8.0,
        input_transform_type="joint",
        joint_transform_shrinkage=1e-3,
        joint_transform_fit_path=None,
        joint_transform_fit_samples=None,
        flow_squash_params=None,
        bijector_state=None,
        cdf_bins=5000,
        cdf_samples=int(1e7),
        profile=False,
        verbose=False,
        global_rank=0,
        artifacts_dir=None,
        target_params=None,
    ):
        self.name = "num_visits"
        self.device = device
        self.profile = profile
        self.verbose = verbose
        self.transform_input = transform_input
        self.logit_flow_scale = float(logit_flow_scale)
        self.global_rank = global_rank
        self.prior_kde_path = None
        self.artifacts_dir = artifacts_dir
        self.sed_prior: EmpiricalSedPrior | None = None
        self._init_input_transform_options(
            input_transform_type=input_transform_type,
            joint_transform_shrinkage=joint_transform_shrinkage,
            joint_transform_fit_path=joint_transform_fit_path,
            joint_transform_fit_samples=joint_transform_fit_samples,
            flow_squash_params=flow_squash_params,
        )

        self.filters_list = design_args.get("labels", ["u", "g", "r", "i", "z", "y"])
        self.design_labels = self.filters_list
        self.num_filters = len(self.filters_list)
        self.observation_labels = ["magnitudes"]
        # Context = design (nvisits per filter) + observations (magnitudes per filter)
        self.context_dim = 2 * len(self.filters_list)

        if isinstance(temperature, (int, float)):
            self.temperature = float(temperature) * u.K
        else:
            self.temperature = temperature

        # Bolometric luminosity in L_sun, applied to the bb/bb_temp SED. The
        # emitting radius is back-solved from Stefan-Boltzmann to hold this fixed
        # at every T, making the source a zero-scatter standard candle -- so this
        # value sets the whole SNR scale of the experiment, not just a brightness
        # offset. 1e9 (the historical default) is a dwarf galaxy (M_bol=-17.8);
        # LSST photo-z targets are L* ~ 2-3e10.
        self.l_bol = float(l_bol)
        if self.l_bol <= 0:
            raise ValueError(f"l_bol must be positive (L_sun), got {l_bol}")

        self.prior_args = prior_args or {}
        self.cosmo_model = cosmo_model

        self.prior, self.latex_labels, self.cosmo_params = self.init_prior(
            cosmo_model=cosmo_model,
            artifacts_dir=self.artifacts_dir,
            **self.prior_args,
        )

        # Which cosmo_params the guide actually infers (default: all). A strict
        # subset trains a focused guide (e.g. redshift only) while the generative
        # model still samples the full parameter vector (nuisances marginalized
        # by simulation).
        self._init_target_params(target_params)

        self._init_param_bijector(
            bijector_state=bijector_state,
            cdf_bins=cdf_bins,
            cdf_samples=cdf_samples,
        )

        if nominal_design is None:
            self.nominal_design = torch.tensor(
                [fiducial_nvisits[band] for band in self.filters_list],
                device=self.device,
                dtype=torch.float64,
            )
        else:
            nominal_array = np.asarray(nominal_design, dtype=np.float64)
            if nominal_array.shape != (self.num_filters,):
                raise ValueError(
                    f"nominal_design must have shape ({self.num_filters},), got {nominal_array.shape}"
                )
            self.nominal_design = torch.tensor(
                nominal_array, device=self.device, dtype=torch.float64
            )

        self.pixel_scale = pixel_scale
        self.stamp_size = stamp_size
        self.threshold = threshold
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.mag_err_cap = mag_err_cap
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
            loaded_filter = speclite_filters.load_filter("lsst2023-" + band)
            wlen = loaded_filter.wavelength * u.AA
            transmission_result = loaded_filter(wlen)
            transmission = (
                transmission_result
                if isinstance(transmission_result, np.ndarray)
                else transmission_result.value
            )

            # Downsample to reduce computation
            if downsample_factor > 1:
                wlen_aa_downsampled = wlen.value[::downsample_factor]
                transmission_downsampled = transmission[::downsample_factor]
            else:
                wlen_aa_downsampled = wlen.value
                transmission_downsampled = transmission

            all_wlen_min.append(wlen_aa_downsampled.min())
            all_wlen_max.append(wlen_aa_downsampled.max())
            filter_data_list.append(
                {
                    "band": band,
                    "wlen_aa": wlen_aa_downsampled,
                    "transmission": transmission_downsampled,
                }
            )

        # Create common wavelength grid covering all filters
        wlen_min = min(all_wlen_min)
        wlen_max = max(all_wlen_max)
        # Use the maximum number of points from any filter, or a reasonable default
        max_points = max(len(fd["wlen_aa"]) for fd in filter_data_list)
        wlen_common_aa = np.linspace(wlen_min, wlen_max, max_points)
        wlen_common_cm = wlen_common_aa * 1e-8  # Convert Angstrom to cm
        wlen_over_hc_common = wlen_common_aa / hc_erg_angstrom

        # Interpolate all filters to common grid
        from scipy.interpolate import interp1d

        transmission_array = np.zeros((self.num_filters, len(wlen_common_aa)), dtype=np.float64)

        for i, filter_data in enumerate(filter_data_list):
            # Interpolate transmission to common grid
            interp_func = interp1d(
                filter_data["wlen_aa"],
                filter_data["transmission"],
                kind="linear",
                bounds_error=False,
                fill_value=0.0,  # Outside filter range, transmission is 0
            )
            transmission_array[i, :] = interp_func(wlen_common_aa)

        # Store as tensors on device
        self._wlen_aa_tensor = torch.tensor(
            wlen_common_aa, device=self.device, dtype=torch.float64
        )  # (n_wlen,)
        self._wlen_cm_tensor = torch.tensor(
            wlen_common_cm, device=self.device, dtype=torch.float64
        )  # (n_wlen,)
        self._transmission_tensor = torch.tensor(
            transmission_array, device=self.device, dtype=torch.float64
        )  # (n_filters, n_wlen)
        self._wlen_over_hc_tensor = torch.tensor(
            wlen_over_hc_common, device=self.device, dtype=torch.float64
        )  # (n_wlen,)
        defaults = {"z": 1.0}
        if self.cosmo_model == "empirical":
            defaults = mode_central_params_from_artifact(self.sed_prior_artifact)
            if self.global_rank == 0 and self.verbose:
                n_train = len(self.sed_prior_artifact.get("training_x", []))
                print(
                    f"  central_params defaults: KDE-prior marginal modes "
                    f"(training N={n_train}); override via train_args central_params"
                )
        elif "T" in self.cosmo_params:
            defaults["T"] = 10000.0
        self.central_params = dict(defaults)
        self.central_params.update(_central_params_as_dict(central_params))

        if self.cosmo_model == "empirical":
            self.central_val = self._central_magnitudes_from_dict(self.central_params)
        else:
            z_central = torch.tensor(
                [self.central_params["z"]], device=self.device, dtype=torch.float64
            )
            if "T" in self.central_params:
                T_central = torch.tensor(
                    [self.central_params["T"]], device=self.device, dtype=torch.float64
                )
                flux_aa = self._observed_spectral_flux(z_central, T=T_central)
            else:
                flux_aa = self._observed_spectral_flux(z_central)
            self.central_val = self._calculate_magnitudes(flux_aa).squeeze(0)

        self.nominal_context = torch.cat([self.nominal_design, self.central_val], dim=-1)
        if hasattr(self, "prior_flow") and self.prior_flow is not None:
            self.prior_flow_nominal_context = self.nominal_context

        # Pass design_args using ** unpacking if provided, otherwise use defaults
        if design_args is not None:
            self.init_designs(**design_args)
        else:
            self.init_designs()

        if self.global_rank == 0 and self.verbose:
            print(f"Num Visits Experiment Initialized")
            print(f"  Filters: {self.filters_list}")
            print(f"  Number of designs: {self.designs.shape[0]}")
            print(f"  Nominal design: {self.nominal_design}")
            print(f"  Central params: {self.central_params}")

    @property
    def sed_prior_artifact(self):
        return self.sed_prior.artifact if self.sed_prior is not None else None

    @property
    def prior_pool(self):
        return self.sed_prior.pool if self.sed_prior is not None else None

    @property
    def prior_feature_names(self):
        return self.sed_prior.feature_names if self.sed_prior is not None else None

    def _init_param_bijector(
        self,
        bijector_state=None,
        cdf_bins=5000,
        cdf_samples=int(1e7),
        always_build=False,
    ):
        """Load the NF bijector from the KDE artifact for empirical runs."""
        if getattr(self, "cosmo_model", None) == "empirical" and getattr(
            self, "transform_input", False
        ):
            self._init_empirical_param_bijector(
                bijector_state=bijector_state,
                cdf_bins=cdf_bins,
            )
            return
        super()._init_param_bijector(
            bijector_state=bijector_state,
            cdf_bins=cdf_bins,
            cdf_samples=cdf_samples,
            always_build=always_build,
        )

    def _init_empirical_param_bijector(
        self,
        *,
        bijector_state=None,
        cdf_bins=5000,
    ) -> None:
        from bedcosmo.transform import Bijector
        from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
            get_empirical_gaussianizer,
        )

        if bijector_state is not None:
            self.param_bijector = Bijector(
                self,
                cdf_bins=cdf_bins,
                cdf_samples=0,
                skip_sampling=True,
                matrix_columns=list(self.cosmo_params),
            )
            self.param_bijector.set_state(bijector_state, device=self.device)
            if self.global_rank == 0 and self.verbose:
                print("Restoring bijector state from checkpoint.")
            return

        artifact = self.sed_prior_artifact
        if artifact is None or artifact.get("gaussianizer_state") is None:
            raise RuntimeError(
                "empirical NumVisits with transform_input=True requires "
                "gaussianizer_state in the KDE artifact. Rebuild with "
                "`python -m bedcosmo.num_visits.empirical.build_prior` "
                "(do not pass --no-gaussianizer)."
            )

        self.param_bijector = get_empirical_gaussianizer(artifact)
        self.param_bijector.experiment = self
        self.param_bijector.set_state(
            self.param_bijector.get_state(),
            device=self.device,
        )
        if self.global_rank == 0 and self.verbose:
            print(
                "Loaded param_bijector from empirical KDE artifact " "(build_prior gaussianizer)."
            )

    @profile_method
    def _expand_to_filters(self, value, label):
        if isinstance(value, (int, float)):
            return [float(value)] * self.num_filters
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != self.num_filters:
                raise ValueError(f"{label} must have length {self.num_filters}, got {len(value)}")
            return [float(v) for v in value]
        raise TypeError(f"{label} must be a float or sequence, got {type(value)}")

    @staticmethod
    def _strip_math_delimiters(label: str) -> str:
        """GetDist adds $ around labels; strip pre-wrapped delimiters to avoid $$."""
        s = str(label).strip()
        if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
            return s[1:-1].strip()
        return s

    def get_joint_transform_fit_matrix(self, n_max: int) -> np.ndarray | None:
        """Use empirical-prior training rows to fit the joint Gaussianizing transform."""
        if getattr(self, "cosmo_model", None) != "empirical":
            return None
        artifact = getattr(self, "sed_prior_artifact", None)
        if artifact is None:
            return None
        from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
            get_training_matrix,
        )

        x_full = get_training_matrix(artifact)
        names = self._joint_transform_param_names()
        fn = list(self.prior_feature_names or self.cosmo_params)
        cols = [fn.index(n) for n in names]
        x = x_full[:, cols]
        if len(x) > n_max:
            rng = np.random.default_rng(0)
            x = x[rng.choice(len(x), size=n_max, replace=False)]
        return x

    @profile_method
    def init_prior(
        self,
        parameters,
        prior_flow_path=None,
        prior_dir=None,
        artifacts_dir=None,
        cosmo_model=None,
        prior_pool_size=65536,
        prior_pool_seed=7,
        template_dir=None,
        template_param="templates/fsps_full/fsps_QSF_12_v3.param",
        prior_source="kde",
        **kwargs,
    ):
        """
        Load prior from prior_args and models.yaml.

        For ``empirical``, resolves a prior root directory (frozen run artifacts,
        else ``prior_dir``, else the default scratch build) and loads the KDE /
        PriorFlows from there according to ``prior_source``.
        Otherwise uses analytic Pyro distributions and optional ``prior_flow_path``.
        """
        if cosmo_model is None:
            cosmo_model = self.cosmo_model
        if cosmo_model is None:
            raise ValueError("cosmo_model must be set for NumVisits (e.g. bb, bb_temp, empirical)")

        models_yaml_path = get_experiment_config_path("num_visits", "models.yaml")
        with open(models_yaml_path, "r") as f:
            cosmo_models = yaml.safe_load(f)
        if cosmo_model not in cosmo_models:
            raise ValueError(
                f"cosmo_model '{cosmo_model}' not in {models_yaml_path}. "
                f"Available: {list(cosmo_models.keys())}"
            )
        model_cfg = cosmo_models[cosmo_model]

        if cosmo_model == "empirical":
            # Parameter names / latex come from the KDE artifact (+ optional
            # latex_labels in prior_args), so alternate template banks only need
            # a different prior_args file — not a new models.yaml entry.
            prior_root = resolve_runtime_prior_root(
                artifacts_dir=artifacts_dir,
                prior_dir=prior_dir,
            )
            return self._init_prior_empirical(
                parameters,
                prior_root=prior_root,
                prior_pool_size=int(prior_pool_size),
                prior_pool_seed=int(prior_pool_seed),
                template_dir=template_dir,
                template_param=template_param,
                prior_source=str(prior_source),
                latex_labels=kwargs.get("latex_labels"),
            )

        model_parameters = list(model_cfg["parameters"])
        latex_labels = [self._strip_math_delimiters(lbl) for lbl in model_cfg["latex_labels"]]
        if len(model_parameters) != len(latex_labels):
            raise ValueError("models.yaml parameters and latex_labels length mismatch")

        prior = {}

        for name in model_parameters:
            if name not in parameters:
                raise ValueError(
                    f"Parameter '{name}' required by model '{cosmo_model}' "
                    f"missing from prior_args parameters"
                )
            cfg = parameters[name]
            dist_cfg = cfg.get("distribution", {})
            dist_type = dist_cfg.get("type", "uniform")

            if dist_type == "uniform":
                lower = float(dist_cfg.get("lower", 0.0))
                upper = float(dist_cfg.get("upper", 1.0))
                if lower >= upper:
                    raise ValueError(
                        f"Invalid bounds for '{name}': lower ({lower}) must be < upper ({upper})"
                    )
                prior[name] = dist.Uniform(
                    torch.tensor(lower, device=self.device, dtype=torch.float64),
                    torch.tensor(upper, device=self.device, dtype=torch.float64),
                )
            elif dist_type == "gamma":
                # Gamma distribution
                # For LSST redshift distribution: p(z) = (1/(2*z_0)) * (z/z_0)^2 * exp(-z/z_0)
                shape = float(dist_cfg.get("shape", 1.0))
                if "z_0" not in dist_cfg:
                    raise ValueError(f"Gamma distribution for '{name}' requires 'z_0' parameter")
                z_0 = float(dist_cfg["z_0"])
                if z_0 <= 0:
                    raise ValueError(f"Gamma z_0 for '{name}' must be > 0, got {z_0}")
                if shape <= 0:
                    raise ValueError(f"Gamma shape for '{name}' must be > 0, got {shape}")
                # Convert z_0 to rate: rate = 1/z_0
                rate = 1.0 / z_0
                prior[name] = dist.Gamma(
                    torch.tensor(shape, device=self.device, dtype=torch.float64),
                    torch.tensor(rate, device=self.device, dtype=torch.float64),
                )
            elif dist_type == "gaussian":
                # Normal distribution: loc (mean) and scale (std)
                loc = float(dist_cfg.get("loc", 0.0))
                scale = float(dist_cfg.get("scale", 1.0))
                if scale <= 0:
                    raise ValueError(f"Normal scale for '{name}' must be > 0, got {scale}")
                prior[name] = dist.Normal(
                    torch.tensor(loc, device=self.device, dtype=torch.float64),
                    torch.tensor(scale, device=self.device, dtype=torch.float64),
                )

            else:
                raise ValueError(
                    f"Distribution type '{dist_type}' not supported. "
                    "Supported types: uniform, gamma, normal"
                )

        for name in model_parameters:
            if name not in prior:
                raise KeyError(f"Model parameter '{name}' was not added to prior dict")

        # Load prior flow if specified
        # Note: prior_flow_path must be an absolute path
        if prior_flow_path:
            if not os.path.isabs(prior_flow_path):
                raise ValueError(
                    f"prior_flow path '{prior_flow_path}' must be an absolute path. "
                    "Relative paths are not supported."
                )

            self.prior_flow, self.prior_flow_metadata = load_prior_flow_from_file(
                prior_flow_path, self.device, self.global_rank
            )
            # Store metadata in experiment class, not on the flow model
            # If not provided, will be set in __init__ after nominal_design is available
            if self.global_rank == 0:
                print(
                    "Using trained posterior model as prior for parameter sampling (loaded from prior_args.yaml)"
                )
        else:
            self.prior_flow = None

        return prior, latex_labels, model_parameters

    def _init_prior_empirical(
        self,
        parameters: dict,
        *,
        prior_root: Path,
        prior_pool_size: int,
        prior_pool_seed: int,
        template_dir: str | None,
        template_param: str,
        prior_source: str = "kde",
        latex_labels: list[str] | None = None,
    ) -> tuple[dict, list[str], list[str]]:
        from bedcosmo.num_visits.empirical.paths import SED_PRIOR_KDE_NATIVE_FILENAME

        if not template_dir:
            from bedcosmo.num_visits.empirical.paths import get_template_dir

            template_dir = str(get_template_dir())

        prior_root = Path(prior_root)
        kde_path = prior_root / SED_PRIOR_KDE_NATIVE_FILENAME
        if self.global_rank == 0 and self.verbose:
            print(f"Loading empirical prior from {prior_root}")
            print(f"  prior_source={prior_source}; pool n={prior_pool_size}")
        self.sed_prior = EmpiricalSedPrior.from_kde_path(
            kde_path,
            pool_size=int(prior_pool_size),
            pool_seed=int(prior_pool_seed),
            device=self.device,
        )
        self.prior_kde_path = self.sed_prior.path
        self._prior_parameterization = self.sed_prior.parameterization
        self._n_eazy_templates = self.sed_prior.n_templates
        model_parameters = list(self.sed_prior.feature_names)

        # Optional latex overrides in prior_args; otherwise f_i / log s / z.
        if latex_labels is not None:
            if len(latex_labels) != len(model_parameters):
                raise ValueError(
                    f"prior_args latex_labels length {len(latex_labels)} != "
                    f"{len(model_parameters)} KDE features {model_parameters}"
                )
            latex_labels = [self._strip_math_delimiters(lbl) for lbl in latex_labels]
        else:
            latex_labels = []
            for name in model_parameters:
                if name == "log_c_scale":
                    latex_labels.append(r"\log s")
                elif name.startswith("f") and name[1:].isdigit():
                    latex_labels.append(
                        f"f_{{{name[1:]}}}" if len(name) > 2 else f"f_{name[1:]}"
                    )
                else:
                    latex_labels.append(name)

        # ``parameters`` is the inner prior_args ``parameters:`` mapping.
        missing = [n for n in model_parameters if n not in parameters]
        if missing:
            raise ValueError(
                f"prior_args parameters missing entries for KDE features {missing}. "
                f"Expected keys matching artifact feature_names {model_parameters}."
            )

        if normalize_prior_source(prior_source) == PRIOR_SOURCE_FLOW:
            loaded = self.sed_prior.enable_flow_prior(
                prior_pool_size,
                seed=prior_pool_seed,
                device=self.device,
                flow_dir=prior_root,
            )
            if self.global_rank == 0 and self.verbose:
                names = ", ".join(p.name for p in loaded.values())
                print(
                    f"  loaded flows: {names}; "
                    f"rebuilt pool (n={prior_pool_size}) from native flow"
                )

        wave_rest, template_stack, _ = load_eazy_template_bank(
            template_param,
            template_dir=template_dir,
        )
        self._template_wave_rest = torch.tensor(wave_rest, device=self.device, dtype=torch.float64)
        self._template_flux = torch.tensor(template_stack, device=self.device, dtype=torch.float64)
        if int(template_stack.shape[0]) != int(self._n_eazy_templates):
            raise ValueError(
                f"template_param bank has {template_stack.shape[0]} templates but KDE "
                f"n_templates={self._n_eazy_templates}. Set template_param in prior_args to "
                "match the build used for this prior_dir."
            )

        name_to_idx = {n: i for i, n in enumerate(self.prior_feature_names)}
        prior = {}
        for name in model_parameters:
            if name not in name_to_idx:
                raise ValueError(
                    f"Parameter '{name}' not in KDE feature_names {self.prior_feature_names}"
                )
            idx = name_to_idx[name]
            low = float(self.prior_pool.bounds_min[idx].cpu())
            high = float(self.prior_pool.bounds_max[idx].cpu())
            span = max(high - low, 1e-12)
            pad = max(1e-6, 0.02 * span)
            prior[name] = EmpiricalPrior(low - pad, high + pad, device=self.device)

        if self.global_rank == 0 and self.verbose:
            print(
                f"  EAZY templates: {self._n_eazy_templates} on "
                f"{self._template_wave_rest.shape[0]} rest-frame grid points"
            )

        return prior, latex_labels, model_parameters

    @profile_method
    def _calculate_base_profile(self):
        gal = galsim.Gaussian(flux=1.0, sigma=2.0)
        psf = galsim.Gaussian(fwhm=0.67)
        profile = galsim.Convolve([gal, psf])
        image = galsim.ImageD(self.stamp_size, self.stamp_size, scale=self.pixel_scale)
        profile.drawImage(image=image)
        return image.array.copy()

    @profile_method
    def init_designs(
        self,
        input_designs_path=None,
        input_type="variable",
        step=20.0,
        lower=10.0,
        upper=250.0,
        sum_lower=None,
        sum_upper=None,
        labels=None,
    ):
        """
        Initialize design space.

        Args:
            input_designs_path: Path to numpy file containing designs (assumed to be absolute)
            input_type: Type of input designs ("nominal" or "variable")
            step: Step size(s) for design grid. Can be:
                - float: Same step size for all dimensions (default: 20.0)
                - list/array: Different step size for each dimension (must have length num_filters)
            lower: Lower bound(s) for each design variable. Can be:
                - float: Same lower bound for all dimensions (default: 10.0)
                - list/array: Different lower bound for each dimension (must have length num_filters)
            upper: Upper bound(s) for each design variable. Can be:
                - float: Same upper bound for all dimensions (default: 250.0)
                - list/array: Different upper bound for each dimension (must have length num_filters)
            sum_lower: Lower bound on sum of design variables (default: None)
            sum_upper: Upper bound on sum of design variables (default: None)
            labels: Labels for design variables (default: None)
        """
        if labels is None:
            labels = self.filters_list
        grid_labels = [str(label) for label in labels]
        design_pts = None
        design_axes = None
        constraint = None

        # Check if input_type is "nominal"
        if input_type == "nominal":
            design_pts = self.nominal_design.unsqueeze(0)  # Add batch dimension
        elif input_type == "variable":
            # If input_designs_path is provided, load from path (assumed to be absolute)
            if input_designs_path is not None:
                if not os.path.isabs(input_designs_path):
                    raise ValueError(
                        f"input_designs_path must be an absolute path, got: {input_designs_path}"
                    )
                if not os.path.exists(input_designs_path):
                    raise FileNotFoundError(f"input_designs_path not found: {input_designs_path}")

                input_designs_array = np.load(input_designs_path)
                # Convert to tensor
                if isinstance(input_designs_array, torch.Tensor):
                    design_pts = input_designs_array.to(self.device, dtype=torch.float64)
                elif isinstance(input_designs_array, (list, tuple, np.ndarray)):
                    design_pts = torch.as_tensor(
                        input_designs_array, device=self.device, dtype=torch.float64
                    )
                else:
                    raise ValueError(
                        f"input_designs must be a list, array, or tensor, got {type(input_designs_array)}"
                    )

                # Handle 1D input (single design)
                if design_pts.ndim == 1:
                    if len(design_pts) != self.num_filters:
                        raise ValueError(
                            f"Input design must have {self.num_filters} values, got {len(design_pts)}"
                        )
                    design_pts = design_pts.reshape(1, -1)
                elif design_pts.ndim == 2:
                    if design_pts.shape[1] != self.num_filters:
                        raise ValueError(
                            f"Input design must have {self.num_filters} columns, got {design_pts.shape[1]}"
                        )
                else:
                    raise ValueError(f"Input design must be 1D or 2D, got shape {design_pts.shape}")

            else:
                # Expand parameters to filters if needed
                design_step = self._expand_to_filters(step, "step")
                design_lower = self._expand_to_filters(lower, "lower")
                design_upper = self._expand_to_filters(upper, "upper")

                # Set sum_lower and sum_upper defaults if not provided
                if sum_lower is None:
                    sum_lower = (
                        float(self.nominal_design.sum())
                        if hasattr(self, "nominal_design")
                        else None
                    )
                if sum_upper is None:
                    sum_upper = (
                        float(self.nominal_design.sum())
                        if hasattr(self, "nominal_design")
                        else None
                    )

                design_axes = {}
                for idx, grid_label in enumerate(grid_labels):
                    design_axes[grid_label] = np.arange(
                        design_lower[idx],
                        design_upper[idx] + 0.5 * design_step[idx],
                        design_step[idx],
                        dtype=np.float64,
                    )
                if sum_lower is not None or sum_upper is not None:
                    lower_bound = sum_lower if sum_lower is not None else -np.inf
                    upper_bound = sum_upper if sum_upper is not None else np.inf

                    def _constraint(**kwargs):
                        total = sum(kwargs.values())
                        within = np.logical_and(
                            total >= lower_bound - 1e-9, total <= upper_bound + 1e-9
                        )
                        return within.astype(int)

                    constraint = _constraint
        else:
            raise ValueError(
                f"Unknown input_type '{input_type}'. Expected 'nominal' or 'variable'."
            )

        if design_pts is not None:
            # Explicit designs (nominal or input_designs_path): use the points
            # directly. A bed.grid.Grid is a Cartesian product, so building one
            # from scattered points materializes prod(unique values per axis)
            # cells (e.g. ~6e8 for a wide 6-band set -> GPU OOM), and nothing in
            # training/eval consumes designs_grid for explicit points. Leave it
            # None; the grid-EIG subcommand guards against this case explicitly.
            self.designs_grid = None
            self.designs = design_pts.to(self.device)
        else:
            self.designs_grid = self._build_design_grid(
                design_axes=design_axes,
                labels=grid_labels,
                constraint=constraint,
            )
            design_pts = self._designs_from_grid(
                self.designs_grid, device=self.device, dtype=torch.float64
            )
            self.designs = design_pts.to(self.device)

    @staticmethod
    def _interp1d_linear(
        x_src: torch.Tensor, y_src: torch.Tensor, x_q: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation; x_src (N,), y_src (N,), x_q (...) -> same shape as x_q."""
        x_q_flat = x_q.reshape(-1)
        x_q_clamped = torch.clamp(x_q_flat, float(x_src[0]), float(x_src[-1]))
        idx = torch.searchsorted(x_src, x_q_clamped, right=False)
        idx = torch.clamp(idx, 1, x_src.numel() - 1)
        x0 = x_src[idx - 1]
        x1 = x_src[idx]
        y0 = y_src[idx - 1]
        y1 = y_src[idx]
        w = (x_q_clamped - x0) / (x1 - x0 + 1e-30)
        y_out = y0 + w * (y1 - y0)
        return y_out.reshape(x_q.shape)

    def _coeffs_from_a_log_s(self, a: torch.Tensor, log_s: torch.Tensor) -> torch.Tensor:
        """Raw coefficients c_k = exp(log s) * a_k."""
        return torch.exp(log_s).unsqueeze(-1) * a

    @profile_method
    def _calculate_magnitudes(self, flux_aa: torch.Tensor) -> torch.Tensor:
        """
        LSST AB magnitudes from observed-frame spectral flux on the instrument grid.

        Args:
            flux_aa: Observed-frame flux per Angstrom on ``self._wlen_aa_tensor``,
                shape ``(..., n_wlen)``.

        Returns:
            Magnitudes with shape ``flux_aa.shape[:-1] + (n_filters,)``.
        """
        n_wlen = self._wlen_aa_tensor.shape[0]
        if flux_aa.shape[-1] != n_wlen:
            raise ValueError(
                f"spectral flux last dim {flux_aa.shape[-1]} != wavelength grid {n_wlen}"
            )
        batch_shape = flux_aa.shape[:-1]
        flux_flat = flux_aa.reshape(-1, n_wlen)

        flux_expanded = flux_flat.unsqueeze(1)
        transmission_expanded = self._transmission_tensor.unsqueeze(0)
        wlen_over_hc_expanded = self._wlen_over_hc_tensor.unsqueeze(0).unsqueeze(0)
        integrand = flux_expanded * transmission_expanded * wlen_over_hc_expanded
        photon_flux = torch.trapezoid(integrand, self._wlen_aa_tensor, dim=-1)

        if torch.any(photon_flux < 0) and self.global_rank == 0:
            negative_count = (photon_flux < 0).sum().item()
            print(
                f"WARNING: {negative_count} negative photon_flux values found! "
                f"Min: {photon_flux[photon_flux < 0].min().item():.2e}"
            )

        s0_vals = torch.tensor(
            [s0[band] for band in self.filters_list],
            device=self.device,
            dtype=torch.float64,
        ).unsqueeze(0)
        min_flux = torch.finfo(photon_flux.dtype).tiny * 1e10
        A_cm2 = (319 / 9.6) * 1e4
        photon_flux_pixel = torch.clamp(photon_flux * A_cm2, min=min_flux)
        flux_ratio = torch.clamp(photon_flux_pixel / s0_vals, min=min_flux)
        mags_flat = 24.0 - 2.5 * torch.log10(flux_ratio)
        return mags_flat.reshape(*batch_shape, self.num_filters)

    @profile_method
    def _observed_spectral_flux(
        self,
        z: torch.Tensor,
        *,
        T: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        log_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Observed-frame spectral flux per Angstrom on ``self._wlen_aa_tensor``.

        Pass ``a`` and ``log_s`` for the EAZY template mixture (empirical prior).
        Omit them for a blackbody, using ``self.temperature`` or optional ``T``.

        Returns:
            Flux with shape ``z.shape + (n_wlen,)``.
        """
        if a is not None or log_s is not None:
            if a is None or log_s is None:
                raise ValueError("template SED requires both a and log_s")
            z_flat = z.reshape(-1)
            a_flat = a.reshape(-1, self._n_eazy_templates)
            log_s_flat = log_s.reshape(-1)
            n_batch = z_flat.shape[0]
            one_plus_z = (1.0 + z_flat).unsqueeze(-1)
            wlen_rest = self._wlen_aa_tensor.unsqueeze(0) / one_plus_z
            c = self._coeffs_from_a_log_s(a_flat, log_s_flat)

            # All templates share the same source grid (_template_wave_rest) and
            # the same per-particle query points (wlen_rest depends only on z),
            # so the searchsorted/index/weight geometry is identical across
            # templates. Compute it once, then gather + lerp all templates in a
            # single batched op instead of looping (the per-template Python loop
            # recomputed searchsorted 12x redundantly).
            n_wlen = self._wlen_aa_tensor.shape[0]
            x_src = self._template_wave_rest
            x_q = wlen_rest.reshape(-1)
            x_q_clamped = torch.clamp(x_q, float(x_src[0]), float(x_src[-1]))
            idx = torch.searchsorted(x_src, x_q_clamped, right=False)
            idx = torch.clamp(idx, 1, x_src.numel() - 1)
            x0 = x_src[idx - 1]
            x1 = x_src[idx]
            w = (x_q_clamped - x0) / (x1 - x0 + 1e-30)  # (n_batch * n_wlen,)

            # Gather every template at the shared indices: (n_templates, M).
            y0 = self._template_flux[:, idx - 1]
            y1 = self._template_flux[:, idx]
            T_all = y0 + w.unsqueeze(0) * (y1 - y0)
            T_all = T_all.reshape(self._n_eazy_templates, n_batch, n_wlen)

            # Per-particle weighted sum over templates: replaces the loop's accumulation.
            flux_obs = torch.einsum("bk,kbw->bw", c, T_all)
            flux_obs = flux_obs / one_plus_z
            return flux_obs.reshape(*z.shape, n_wlen)

        z_tensor = z
        T_tensor = T
        if T_tensor is None:
            z_flat = z_tensor.flatten()
            z_unique, inverse_indices = torch.unique(z_flat, return_inverse=True)
            lum_dist = self._luminosity_distance(z_unique)

            if not hasattr(self, "_T_K_tensor"):
                if hasattr(self.temperature, "value"):
                    self._T_K_tensor = torch.tensor(
                        self.temperature.to(u.K).value,
                        device=self.device,
                        dtype=torch.float64,
                    )
                else:
                    self._T_K_tensor = torch.tensor(
                        float(self.temperature),
                        device=self.device,
                        dtype=torch.float64,
                    )
            T_K = self._T_K_tensor

            if not hasattr(self, "_four_pi_R2_tensor"):
                L_bol = self.l_bol * _L_sun_cgs
                sigma_sb_cgs = sigma_sb.to(u.erg / (u.s * u.cm**2 * u.K**4)).value
                T_K_4 = T_K**4
                R_eff_val = torch.sqrt(
                    torch.tensor(
                        L_bol / (4 * np.pi * sigma_sb_cgs),
                        device=self.device,
                        dtype=torch.float64,
                    )
                    / T_K_4
                )
                self._four_pi_R2_tensor = (
                    torch.tensor(4 * np.pi, device=self.device, dtype=torch.float64) * R_eff_val**2
                )
            four_pi_R2 = self._four_pi_R2_tensor
            T_K_for_bb = T_K
            four_pi_R2_for_L = four_pi_R2
        else:
            z_t = torch.as_tensor(z_tensor, device=self.device, dtype=torch.float64)
            T_t = torch.as_tensor(T_tensor, device=self.device, dtype=torch.float64)
            z_b, T_b = torch.broadcast_tensors(z_t, T_t)
            joint_shape = z_b.shape
            z_unique = z_b.flatten()
            T_K = T_b.flatten()
            inverse_indices = None
            lum_dist = self._luminosity_distance(z_unique)

            L_bol = self.l_bol * _L_sun_cgs
            sigma_sb_cgs = sigma_sb.to(u.erg / (u.s * u.cm**2 * u.K**4)).value
            L_bol_const = torch.tensor(
                L_bol / (4 * np.pi * sigma_sb_cgs),
                device=self.device,
                dtype=torch.float64,
            )
            R_eff = torch.sqrt(L_bol_const / (T_K**4))
            four_pi_R2 = torch.tensor(4 * np.pi, device=self.device, dtype=torch.float64) * R_eff**2
            T_K_for_bb = T_K.unsqueeze(-1)
            four_pi_R2_for_L = four_pi_R2.unsqueeze(-1)

        z_2d = z_unique.unsqueeze(-1)
        wlen_2d = self._wlen_aa_tensor.unsqueeze(0)

        if not hasattr(self, "_1e8_tensor_f64"):
            self._1e8_tensor_f64 = torch.tensor(1e-8, device=self.device, dtype=torch.float64)
        one_plus_z = 1 + z_2d
        wlen_rest_aa = wlen_2d / one_plus_z
        wlen_rest_cm = wlen_rest_aa * self._1e8_tensor_f64

        F = self._blackbody_flux(wlen_rest_cm, T_K_for_bb)
        L = four_pi_R2_for_L * F

        lum_dist_2d = lum_dist.unsqueeze(-1)
        if not hasattr(self, "_four_pi_tensor"):
            self._four_pi_tensor = torch.tensor(4 * np.pi, device=self.device, dtype=torch.float64)
        flux = L / (one_plus_z * self._four_pi_tensor * lum_dist_2d**2)

        if torch.any(~torch.isfinite(flux)) or torch.any(flux <= 0):
            problematic_flux = (~torch.isfinite(flux)) | (flux <= 0)
            problematic_count = problematic_flux.sum().item()
            if self.global_rank == 0 and problematic_count > 0:
                print(f"WARNING: {problematic_count} problematic flux values found!")
                print(f"  flux range: [{flux.min().item():.2e}, {flux.max().item():.2e}]")
                print(f"  L range: [{L.min().item():.2e}, {L.max().item():.2e}]")
                print(
                    f"  lum_dist range: [{lum_dist.min().item():.2e}, {lum_dist.max().item():.2e}]"
                )
                print(
                    f"  z_unique range: [{z_unique.min().item():.4f}, {z_unique.max().item():.4f}]"
                )

        n_wlen = self._wlen_aa_tensor.shape[0]
        if T_tensor is not None:
            return flux.reshape(*joint_shape, n_wlen)
        flux_flat = flux[inverse_indices]
        return flux_flat.reshape(*z_tensor.shape, n_wlen)

    def _require_parameterization(self) -> str:
        param = getattr(self, "_prior_parameterization", None)
        if param is None:
            raise ValueError(
                "Empirical prior has no parameterization; the KDE artifact must set "
                "'parameterization' ('ilr' or legacy 'clr')."
            )
        return param

    def _empirical_rows_to_physical(
        self,
        rows: torch.Tensor,
        sample_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode empirical-prior feature rows into physical SED quantities.

        Rows are in the artifact feature space:
            ilr: f1..f_{K-1}, log_c_scale, z
            clr: f1..fK, log_c_scale, z (legacy)
        """
        parameterization = self._require_parameterization()
        K = int(self._n_eazy_templates)
        flat = sample_shape if sample_shape else (rows.shape[0],)

        if parameterization == PARAMETERIZATION_ILR:
            ilr = rows[:, : K - 1]
            a = ilr_to_weights_torch(ilr)
            log_s = rows[:, K - 1]
            z = rows[:, K]

        elif parameterization == "clr":
            clr = rows[:, :K]
            shifted = clr - torch.amax(clr, dim=-1, keepdim=True)
            exp = torch.exp(shifted)
            a = exp / exp.sum(dim=-1, keepdim=True).clamp_min(1e-300)
            log_s = rows[:, K]
            z = rows[:, K + 1]

        else:
            raise ValueError(f"Unknown prior parameterization {parameterization!r}")

        return (
            a.reshape(*flat, K),
            log_s.reshape(*flat),
            z.reshape(*flat),
        )

    def _central_magnitudes_from_dict(self, central: dict) -> torch.Tensor:
        parameterization = self._require_parameterization()
        K = int(self._n_eazy_templates)

        if parameterization == PARAMETERIZATION_ILR:
            ilr = torch.tensor(
                [[float(central.get(f"f{k}", 0.0)) for k in range(1, K)]],
                device=self.device,
                dtype=torch.float64,
            )
            a = ilr_to_weights_torch(ilr)

        elif parameterization == "clr":
            clr = torch.tensor(
                [[float(central.get(f"f{k}", 0.0)) for k in range(1, K + 1)]],
                device=self.device,
                dtype=torch.float64,
            )
            shifted = clr - torch.amax(clr, dim=-1, keepdim=True)
            exp = torch.exp(shifted)
            a = exp / exp.sum(dim=-1, keepdim=True).clamp_min(1e-300)

        else:
            raise ValueError(f"Unknown prior parameterization {parameterization!r}")

        log_s = torch.tensor(
            [float(central["log_c_scale"])],
            device=self.device,
            dtype=torch.float64,
        )
        z = torch.tensor(
            [float(central["z"])],
            device=self.device,
            dtype=torch.float64,
        )
        flux_aa = self._observed_spectral_flux(z, a=a, log_s=log_s)
        return self._calculate_magnitudes(flux_aa).squeeze(0)

    def _prior_rows_to_param_dict(
        self,
        rows: torch.Tensor,
        sample_shape: tuple[int, ...],
    ) -> dict[str, torch.Tensor]:
        """Map empirical-prior feature rows to parameter tensors with trailing dim 1."""
        if self.sed_prior is None:
            raise RuntimeError("sed_prior is not initialized")
        return self.sed_prior.rows_to_param_dict(rows, self.cosmo_params, sample_shape)

    @profile_method
    def sample_parameters(self, sample_shape, prior=None, use_prior_flow=True, **kwargs):
        """
        Sample parameters from analytic prior, prior flow, or empirical KDE pool.
        """
        if self.cosmo_model == "empirical" and self.sed_prior is not None:
            n = int(np.prod(sample_shape)) if sample_shape else 1
            rows = self.sed_prior.sample_batch(n)
            return self._prior_rows_to_param_dict(rows, sample_shape)

        if prior is None:
            prior = self.prior

        parameters = {}
        for k, v in prior.items():
            if isinstance(v, dist.Distribution):
                parameters[k] = pyro.sample(k, v).unsqueeze(-1)
            else:
                parameters[k] = v

        return parameters

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
        alpha = torch.where(
            exact_match, torch.zeros_like(z_flat), (z_flat - z_lower) / (z_upper - z_lower + 1e-10)
        )
        dc_over_dh0 = torch.where(
            exact_match, cum_upper, cum_lower + alpha * (cum_upper - cum_lower)
        )

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
        if not hasattr(self, "_hc_tensor_f32"):
            self._hc_tensor_f32 = torch.tensor(_hc, device=self.device, dtype=torch.float32)
            self._k_B_tensor_f32 = torch.tensor(_k_B_cgs, device=self.device, dtype=torch.float32)
            self._two_hc2_tensor_f32 = torch.tensor(
                _two_hc2, device=self.device, dtype=torch.float32
            )
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

        # Calculate magnitude errors: coeff / snr, with optional cap
        if self.mag_err_cap is not None:
            min_snr_for_detection = coeff / self.mag_err_cap
            mag_err = np.where(snr < min_snr_for_detection, self.mag_err_cap, coeff / snr)
        else:
            mag_err = np.where(snr > 0, coeff / snr, coeff / 1e-10)

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

        if self.cosmo_model == "empirical" and self.sed_prior is not None:
            n = int(np.prod(batch_shape))
            rows = self.sed_prior.sample_batch(n)
            params = self._prior_rows_to_param_dict(rows, batch_shape)
            for name, val in params.items():
                pyro.sample(name, dist.Delta(val.squeeze(-1)).to_event(0))
            a, log_s, z = self._empirical_rows_to_physical(rows, batch_shape)
            flux_aa = self._observed_spectral_flux(z, a=a, log_s=log_s)
            means = self._calculate_magnitudes(flux_aa)
        elif hasattr(self, "prior_flow") and self.prior_flow is not None:
            samples = self._sample_prior_flow_cache(batch_shape)
            z = samples.squeeze(-1)
            z = pyro.sample("z", dist.Delta(z))
            flux_aa = self._observed_spectral_flux(z)
            means = self._calculate_magnitudes(flux_aa)
        else:
            z_dist = self.prior["z"].expand(batch_shape).to_event(0)
            z = pyro.sample("z", z_dist)
            if "T" in self.prior:
                T_dist = self.prior["T"].expand(batch_shape).to_event(0)
                T = pyro.sample("T", T_dist)
                flux_aa = self._observed_spectral_flux(z, T=T)
            else:
                flux_aa = self._observed_spectral_flux(z)
            means = self._calculate_magnitudes(flux_aa)
        sigmas = self._magnitude_errors(means, nvisits)
        covariance = torch.diag_embed(sigmas**2)
        return pyro.sample(self.observation_labels[0], dist.MultivariateNormal(means, covariance))

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

        def _grid_array(grid, name):
            """Get axis values without GridStack trailing singleton pad dimensions."""
            if name not in grid.names:
                raise ValueError(f"Axis '{name}' not found in grid names {list(grid.names)}.")
            values = jnp.asarray(getattr(grid, name), dtype=jnp.float64)
            grid_ndim = len(grid.shape)
            if values.ndim > grid_ndim:
                values = values.reshape(values.shape[:grid_ndim])
            return values

        # Parameter grid: z (always present), optionally T. Final P shape is
        # the joint broadcast of all parameter axes.
        z_values = _grid_array(params, "z")
        if "T" in getattr(params, "names", []):
            T_values = _grid_array(params, "T")
            z_b, T_b = jnp.broadcast_arrays(z_values, T_values)
            z_shape = z_b.shape
            z_tensor = torch.as_tensor(np.asarray(z_b), device=self.device, dtype=torch.float64)
            T_tensor = torch.as_tensor(np.asarray(T_b), device=self.device, dtype=torch.float64)
            flux_aa = self._observed_spectral_flux(z_tensor, T=T_tensor)
            mags_model = jnp.asarray(self._calculate_magnitudes(flux_aa).detach().cpu().numpy())
        else:
            z_shape = z_values.shape
            z_tensor = torch.as_tensor(
                np.asarray(z_values), device=self.device, dtype=torch.float64
            )
            flux_aa = self._observed_spectral_flux(z_tensor)
            mags_model = jnp.asarray(self._calculate_magnitudes(flux_aa).detach().cpu().numpy())
        # mags_model shape: P + (num_filters,)

        # Feature grid (magnitudes): broadcast per-filter axes to a common feature shape X
        feature_axes = [_grid_array(features, f"y_{band}") for band in self.design_labels]
        feature_axes = jnp.broadcast_arrays(*feature_axes)
        feature_obs = jnp.stack(feature_axes, axis=-1)
        feature_shape = feature_obs.shape[:-1]
        # features_obs shape: X + (num_filters,)

        # Design grid (visits): broadcast per-filter axes to common design shape D
        design_axes = [_grid_array(designs, str(band)) for band in self.design_labels]
        design_axes = jnp.broadcast_arrays(*design_axes)
        nvisits = jnp.stack(design_axes, axis=-1)
        design_shape = nvisits.shape[:-1]
        # nvisits shape: D + (num_filters,)

        # Magnitude errors depend on model magnitudes and design only (not on observed features).
        # Arrange as D + P + (num_filters,) so final output ordering is X + D + P.
        mags_for_sigma = mags_model.reshape(
            (1,) * len(design_shape) + z_shape + (self.num_filters,)
        )
        nvisits_for_sigma = nvisits.reshape(
            design_shape + (1,) * len(z_shape) + (self.num_filters,)
        )
        sigmas = jnp.asarray(
            self._magnitude_errors(
                torch.as_tensor(
                    np.asarray(mags_for_sigma), device=self.device, dtype=torch.float64
                ),
                torch.as_tensor(
                    np.asarray(nvisits_for_sigma), device=self.device, dtype=torch.float64
                ),
            )
            .detach()
            .cpu()
            .numpy()
        )
        # sigmas shape: D + P + (num_filters,)

        # Build full broadcasted shapes for likelihood: X + D + P + (num_filters,)
        feature_obs_full = feature_obs.reshape(
            feature_shape + (1,) * len(design_shape) + (1,) * len(z_shape) + (self.num_filters,)
        )
        mags_full = mags_model.reshape(
            (1,) * len(feature_shape) + (1,) * len(design_shape) + z_shape + (self.num_filters,)
        )
        sigma_full = sigmas.reshape(
            (1,) * len(feature_shape) + design_shape + z_shape + (self.num_filters,)
        )

        diff = (feature_obs_full - mags_full) / sigma_full
        log_likelihood = -0.5 * jnp.sum(diff**2, axis=-1) - jnp.sum(jnp.log(sigma_full), axis=-1)

        # Stabilize exponentiation by subtracting the global max.
        # This preserves relative P(y|z,d) across all parameters/features.
        log_likelihood = log_likelihood - jnp.max(log_likelihood)

        likelihood = jnp.exp(log_likelihood)
        # likelihood shape: X + D + P
        if (
            getattr(params, "_stack_offset", 0) == 0
            and getattr(features, "_stack_offset", 0) == 0
            and getattr(designs, "_stack_offset", 0) == 0
        ):
            param_shape = tuple(params.shape)
            if likelihood.shape != param_shape and likelihood.size == int(np.prod(param_shape)):
                likelihood = likelihood.reshape(param_shape)
        return likelihood
