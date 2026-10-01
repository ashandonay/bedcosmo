import os
import sys
import yaml
import json
import mlflow
import pandas as pd
import jax.numpy as jnp
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
from bedcosmo.profiling import profile_method
from bedcosmo.util import (
    load_prior_flow_from_file,
    auto_seed,
    load_nominal_samples,
    get_experiment_config_path,
)
from bedcosmo.transform import Bijector
from bedcosmo.custom_dist import ConstrainedUniform2D, PresampledPrior
from bedcosmo.base import BaseExperiment
from bedcosmo.cosmology import CosmologyMixin, FIDUCIAL_PARAMS

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
        mode="train",
        verbose=False,
        profile=False,
        prior_flow_batch_size=10000,
        prior_flow_cache_size=100000,
        prior_flow_use_cache=True,
        likelihood_mode="scaling",
        apply_desi_syst=False,
        vary_z_eff=False,
        vary_lya_qso=False,
        ref_cov=None,
        emulator_sqrtn_ref=None,
        emulator_space=None,
        artifacts_dir=None,
        input_transform_type="marginal",
        joint_transform_shrinkage=1e-3,
        joint_transform_fit_path=None,
        joint_transform_fit_samples=None,
        flow_squash_params=None,
        target_params=None,
    ):

        self.name = "num_tracers"
        self.dataset = dataset
        # Same values as desilike-emulator's --analysis: they select the tracer bins
        # (util.tracers_for), the emulators.yaml subtree and the models.yaml block.
        if analysis not in ("bao", "shapefit"):
            raise ValueError(f"analysis must be 'bao' or 'shapefit'; got {analysis!r}.")
        self.analysis = analysis
        self.desi_data = pd.read_csv(
            os.path.join(home_dir, f"data/desi/bao_{self.dataset}", "desi_data.csv")
        )
        self.desi_tracers = pd.read_csv(
            os.path.join(home_dir, f"data/desi/bao_{self.dataset}", "desi_tracers.csv")
        )
        # Reference covariance (scaling mode scales this; also sets corr_matrix/sigmas).
        # Defaults to the dataset's desi_cov.npy, but can be overridden via ref_cov
        # (train_arg). An absolute/user path is used as-is; a relative path resolves
        # against the dataset data dir (where desi_cov.npy lives), so e.g.
        # ref_cov="ref_cov_alt.npy" drops in a sibling file.
        default_cov_dir = os.path.join(home_dir, f"data/desi/bao_{self.dataset}")
        if ref_cov is None:
            resolved_ref_cov = os.path.join(default_cov_dir, "desi_cov.npy")
        else:
            ref_cov = os.path.expanduser(ref_cov)
            resolved_ref_cov = (
                ref_cov if os.path.isabs(ref_cov)
                else os.path.join(default_cov_dir, ref_cov)
            )
        if not os.path.exists(resolved_ref_cov):
            raise FileNotFoundError(f"Reference covariance file not found: {resolved_ref_cov}")
        self.ref_cov_path = resolved_ref_cov
        self.ref_cov = np.load(resolved_ref_cov)
        self.DH_idx = np.where(self.desi_data["quantity"] == "DH_over_rs")[0]
        self.DM_idx = np.where(self.desi_data["quantity"] == "DM_over_rs")[0]
        self.DV_idx = np.where(self.desi_data["quantity"] == "DV_over_rs")[0]
        # By default the Lya QSO error is held fixed at its nominal value in scaling
        # mode (design-independent), mirroring how the emulator mode leaves Lya_QSO
        # at ref_cov instead of scaling it. Set vary_lya_qso=True to instead let it
        # scale with the design like the other tracers.
        self.vary_lya_qso = vary_lya_qso
        self._lya_qso_rows = np.where(self.desi_data["tracer"] == "Lya QSO")[0]
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
        self.c = constants.c.to("km/s").value
        self.corr_matrix = torch.tensor(
            self.ref_cov
            / np.sqrt(np.outer(np.diag(self.ref_cov), np.diag(self.ref_cov))),
            device=self.device,
        )
        self.include_D_M = include_D_M
        self.include_D_V = include_D_V
        self.efficiency = torch.tensor(
            self.desi_data.drop_duplicates(subset=["tracer"])["efficiency"].tolist(),
            device=self.device,
        )
        self.sigmas = torch.sqrt(
            torch.tensor(np.diag(self.ref_cov), device=self.device, dtype=torch.float32)
        )
        self.central_val = torch.tensor(self.desi_data["value_at_z"].tolist(), device=self.device)
        self.tracer_bins = self.desi_data.drop_duplicates(subset=["tracer"])["tracer"].tolist()
        self.nominal_total_obs = int(
            self.desi_data.drop_duplicates(subset=["tracer"])["observed"].sum()
        )
        self.nominal_passed_ratio = (
            torch.tensor(self.desi_data["passed"].tolist(), device=self.device)
            / self.nominal_total_obs
        )
        # Create dictionary with upper limits and lower limit lists for each class
        # Extract labels from design_args if provided, otherwise use default
        if design_args is not None and "labels" in design_args:
            self.design_labels = design_args["labels"]
        else:
            self.design_labels = ["BGS", "LRG", "ELG", "QSO"]
        self.num_targets = (
            self.desi_tracers.groupby("class").sum()["targets"].reindex(self.design_labels)
        )
        if nominal_design is None:
            self.nominal_design = torch.tensor(
                self.desi_tracers.groupby("class")
                .sum()["observed"]
                .reindex(self.design_labels)
                .values,
                device=self.device,
            )
        else:
            self.nominal_design = nominal_design

        # Initialize emulator-based likelihood mode
        self.likelihood_mode = likelihood_mode
        if self.analysis == "shapefit":
            # ShapeFit has no DESI covariance to rescale and its emulators already
            # derive z_eff(N) internally, so the BAO-only switches have no meaning here.
            if likelihood_mode != "emulator":
                raise ValueError(
                    f"analysis='shapefit' requires likelihood_mode='emulator'; got {likelihood_mode!r}.")
            for name, value in (("vary_z_eff", vary_z_eff), ("apply_desi_syst", apply_desi_syst),
                                ("emulator_sqrtn_ref", emulator_sqrtn_ref)):
                if value:
                    raise ValueError(f"{name} is BAO-only; got {name}={value!r} with analysis='shapefit'.")
        # Optionally inflate the emulator's statistical covariance by DESI's measured
        # per-tracer systematic budget (desilike_emulator desi_reference.apply_desi_syst).
        self.apply_desi_syst = apply_desi_syst
        self._desi_syst_factors_cache = None
        # Evaluate the BAO means at each tracer bin's effective redshift for the
        # design's N_tracers (desilike_emulator util.effective_redshift, DESI 2024 III
        # Eq. 2.1 in the fiducial frame) instead of desi_data.csv's fixed DR1 z. The
        # emulator's sigma labels are defined at z_eff(N) (desilike-emulator shapefit
        # CHANGELOG §159), so this makes mean and covariance describe the same
        # redshift. Emulator mode only: scaling-mode sigmas are DESI's, measured at
        # DESI's z. DR1 only: desilike-emulator ships DR1 n(z) tables. Off by default.
        if vary_z_eff and likelihood_mode != "emulator":
            raise ValueError(
                "vary_z_eff requires likelihood_mode='emulator': scaling-mode sigmas "
                "are DESI's own, measured at desi_data.csv's fixed z.")
        if vary_z_eff and dataset != "dr1":
            raise ValueError(
                f"vary_z_eff is DR1-only (desilike-emulator has no {dataset} n(z) "
                "tables); got dataset={dataset!r}.")
        self.vary_z_eff = bool(vary_z_eff)
        self._effective_redshift = None
        if self.vary_z_eff:
            # Import now so a missing or pre-#37 desilike_emulator fails at
            # construction, not at the first likelihood call of a queued job.
            from desilike_emulator.util import effective_redshift
            self._effective_redshift = effective_redshift
        self._init_mean_z_eff_rows()
        # Diagnostic: replace the emulator's nonlinear N-dependence with pure 1/sqrt(N)
        # scaling anchored to the nominal design, while keeping the emulator's magnitude.
        # None -> off (full emulator); "sampled" -> reference sigma at sampled cosmology,
        # nominal N (cosmology dependence retained, only N-law replaced); "fiducial" ->
        # reference sigma at fiducial cosmology + nominal N, cached once (cosmology-independent,
        # the direct analog of likelihood_mode='scaling').
        if emulator_sqrtn_ref not in (None, "sampled", "fiducial"):
            raise ValueError(
                f"emulator_sqrtn_ref must be None, 'sampled', or 'fiducial'; got {emulator_sqrtn_ref!r}."
            )
        self.emulator_sqrtn_ref = emulator_sqrtn_ref
        # Which forecast space's BAO emulators to load (bao's emulators.yaml entries are
        # keyed by space). Unused by shapefit, which always loads both mean and covar.
        self.emulator_space = emulator_space
        self._sqrtn_n_ref_cache = None  # per-bin nominal N_tracers (design-independent)
        self._sqrtn_ref_cov_cache = None  # frozen fiducial reference covariance
        self._n_extrap_warned = set()  # checkpoint paths already warned (warn once)
        # Prefer checkpoints snapshotted into the run's artifacts at submission time
        # (bao: artifacts/emulators/<tracer_bin>.pt, shapefit:
        # artifacts/emulators/<quantity>/<tracer_bin>.pt). When no artifacts dir is
        # provided (direct/local init), resolve from emulators.yaml against $SCRATCH.
        emu_dir = os.path.join(artifacts_dir, "emulators") if artifacts_dir else None
        if self.analysis == "shapefit":
            self._init_shapefit(emu_dir)
        elif self.likelihood_mode == "emulator":
            # Bins without a .pt in the artifacts are treated as fallback (null) bins.
            if emu_dir is not None and os.path.isdir(emu_dir):
                checkpoints = {}
                for tracer_bin in self._EMULATOR_TRACER_TO_DESI:
                    ckpt_path = os.path.join(emu_dir, f"{tracer_bin}.pt")
                    checkpoints[tracer_bin] = ckpt_path if os.path.exists(ckpt_path) else None
            else:
                checkpoints = self.resolve_emulator_checkpoints(
                    self.analysis, self.cosmo_model, self.dataset, self.emulator_space,
                )
            self._emulator_checkpoints = checkpoints
            self._load_emulators()

        if self.analysis == "shapefit":
            nominal_data = self.central_val
        else:
            nominal_data = self.central_val if self.include_D_M else self.central_val[1::2]
        self.nominal_context = torch.cat([self.nominal_design, nominal_data], dim=-1)
        # Compute context_dim dynamically from the actual nominal_context size
        # This ensures it matches the actual data structure regardless of flags
        self.context_dim = self.nominal_context.shape[-1]

        # initialize the prior
        self.prior_args = prior_args

        (
            self.prior,
            self.param_constraints,
            self.latex_labels,
            self.prior_flow,
            self.prior_flow_metadata,
        ) = self.init_prior(**self.prior_args)
        self.cosmo_params = list(self.prior.keys())
        if self.analysis == "shapefit":
            self._omega_m_log_acceptance = self._omega_m_domain_log_acceptance()

        # Extract prior_flow settings for use in _sample_prior_flow
        if self.prior_flow_metadata is not None:
            self.prior_flow_transform_input = self.prior_flow_metadata.get("transform_input", False)
            self.prior_flow_nominal_context = self.prior_flow_metadata.get(
                "nominal_context", self.nominal_context
            )
        else:
            self.prior_flow_transform_input = False
            self.prior_flow_nominal_context = None

        # Load DESI prior from the default location. It is a BAO prior (Om, hrdrag, ...),
        # so shapefit's omega-basis parameters use their own prior.
        desi_prior_path = os.path.join(home_dir, f"data/desi/bao_{self.dataset}", "prior_args.yaml")
        if self.analysis == "bao" and os.path.exists(desi_prior_path):
            with open(desi_prior_path, "r") as file:
                desi_prior_data = yaml.safe_load(file)
            self.desi_prior, _, _, _, _ = self.init_prior(**desi_prior_data)
        else:
            # If DESI prior not found, use the same as the main prior
            self.desi_prior = self.prior

        self.transform_input = transform_input
        self._init_input_transform_options(
            input_transform_type=input_transform_type,
            joint_transform_shrinkage=joint_transform_shrinkage,
            joint_transform_fit_path=joint_transform_fit_path,
            joint_transform_fit_samples=joint_transform_fit_samples,
            flow_squash_params=flow_squash_params,
        )
        self._init_target_params(target_params)

        # param_bijector is built unconditionally (DESI sampling consumes it
        # even when transform_input is False). State resolution (explicit
        # arg vs prior_flow_metadata fallback) lives in the helper.
        self._init_param_bijector(bijector_state=bijector_state, always_build=True)

        # If the main prior differs from the DESI prior, build a separate
        # bijector against the uniform DESI prior (not the prior_flow).
        if self.prior.items() != self.desi_prior.items():
            self.desi_bijector = Bijector(
                self,
                prior=self.desi_prior,
                cdf_bins=5000,
                cdf_samples=int(1e7),
                use_prior_flow=False,
            )
        else:
            self.desi_bijector = self.param_bijector

        self.observation_labels = ["y"]

        # Pass design_args using ** unpacking if provided, otherwise use defaults
        if design_args is not None:
            self.init_designs(**design_args)
        else:
            self.init_designs()

    @profile_method
    def init_designs(
        self,
        input_designs_path=None,
        step=0.05,
        lower=0.05,
        upper=None,
        sum_lower=1.0,
        sum_upper=1.0,
        tol=1e-3,
        labels=None,
        input_type="variable",
    ):
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
        if labels is None:
            labels = list(self.design_labels)
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
                design_pts = torch.tensor(
                    input_designs_array, device=self.device, dtype=torch.float64
                )
                # Handle 1D input (single design)
                if design_pts.ndim == 1:
                    if len(design_pts) != len(labels):
                        raise ValueError(
                            f"Input design must have {len(labels)} values, got {len(design_pts)}"
                        )
                    design_pts = design_pts.reshape(1, -1)
                elif design_pts.ndim == 2:
                    if design_pts.shape[1] != len(labels):
                        raise ValueError(
                            f"Input design must have {len(labels)} columns, got {design_pts.shape[1]}"
                        )
                else:
                    raise ValueError(f"Input design must be 1D or 2D, got shape {design_pts.shape}")
            else:
                # Generate design grid
                if type(step) == float:
                    design_steps = [step] * len(labels)
                elif type(step) == list:
                    design_steps = step
                else:
                    raise ValueError("step must be a float or list")

                if type(lower) == float:
                    lower_limits = [lower] * len(labels)
                elif type(lower) == list:
                    lower_limits = lower
                else:
                    raise ValueError("lower must be a float or list")

                if upper is None:
                    upper_limits = [
                        self.num_targets[target] / self.nominal_total_obs for target in labels
                    ]
                elif type(upper) == float:
                    upper_limits = [upper] * len(labels)
                elif type(upper) == list:
                    upper_limits = upper
                else:
                    raise ValueError("upper must be a float or list")

                designs_dict = {
                    grid_labels[i]: np.arange(lower_limits[i], upper_limits[i], design_steps[i])
                    for i, target in enumerate(labels)
                }

                # Create constrained grid based on sum_lower and sum_upper
                if sum_lower is None and sum_upper is None:
                    pass
                elif sum_lower is not None and sum_upper is not None:
                    if abs(sum_lower - sum_upper) < tol:
                        # Equal bounds: sum must equal this value
                        target_sum = sum_lower
                        constraint = (
                            lambda **kwargs: np.abs(sum(kwargs.values()) - target_sum) < tol
                        )
                    else:
                        # Range: sum must be between lower and upper
                        constraint = lambda **kwargs: (
                            (sum(kwargs.values()) >= sum_lower - tol)
                            & (sum(kwargs.values()) <= sum_upper + tol)
                        )
                elif sum_lower is not None:
                    # Only lower bound: sum >= lower
                    constraint = lambda **kwargs: sum(kwargs.values()) >= (sum_lower - tol)
                else:
                    # Only upper bound: sum <= upper
                    constraint = lambda **kwargs: sum(kwargs.values()) <= (sum_upper + tol)

                design_axes = designs_dict
        else:
            raise ValueError(
                f"Unknown input_type '{input_type}'. Expected 'nominal' or 'variable'."
            )

        self.designs_grid = self._build_design_grid(
            design_axes=design_axes,
            design_pts=design_pts,
            labels=grid_labels,
            constraint=constraint,
        )
        if design_pts is None:
            design_pts = self._designs_from_grid(
                self.designs_grid, device=self.device, dtype=torch.float64
            )
        self.designs = design_pts.to(self.device)

    @profile_method
    def init_prior(self, parameters, constraints=None, prior_flow_path=None, **kwargs):
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
            models_yaml_path = get_experiment_config_path("num_tracers", "models.yaml")
            with open(models_yaml_path, "r") as file:
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

            prior_flow, prior_flow_metadata = load_prior_flow_from_file(
                prior_flow_path, self.device, self.global_rank
            )
            prior_flow_metadata["nominal_context"] = (
                self.nominal_context.to(self.device)
                if prior_flow_metadata["nominal_context"] is None
                else prior_flow_metadata["nominal_context"]
            )

            if self.global_rank == 0:
                print(
                    "Using trained posterior model as prior for parameter sampling (loaded from prior_args.yaml)"
                )
        else:
            prior_flow = None
            prior_flow_metadata = None

        # Get the specified cosmology model
        if self.cosmo_model not in cosmo_models[self.analysis]:
            raise ValueError(
                f"Cosmology model '{self.cosmo_model}' not found in cosmo_models.yaml. Available models: {list(cosmo_models.keys())}"
            )

        # Initialize prior, constraints, and latex labels
        prior = {}
        param_constraints = {}
        latex_labels = []

        model_parameters = cosmo_models[self.analysis][self.cosmo_model]["parameters"]
        latex_labels = cosmo_models[self.analysis][self.cosmo_model]["latex_labels"]

        # Process constraints if provided
        if constraints is not None:
            for constraint in cosmo_models[self.analysis][self.cosmo_model].get("constraints", []):
                if constraint not in constraints:
                    raise ValueError(
                        f"Constraint '{constraint}' required by model '{self.cosmo_model}' not found in constraints"
                    )
                param_constraints[constraint] = constraints[constraint]

        # Create prior for each parameter in the model
        for param_name in model_parameters:
            if param_name not in parameters:
                raise ValueError(
                    f"Parameter '{param_name}' not found in prior_args.yaml parameters section"
                )

            param_config = parameters[param_name]

            # Validate parameter configuration
            if "distribution" not in param_config:
                raise ValueError(
                    f"Parameter '{param_name}' missing 'distribution' section in prior_args.yaml"
                )
            if "latex" not in param_config:
                raise ValueError(
                    f"Parameter '{param_name}' missing 'latex' section in prior_args.yaml"
                )

            dist_config = param_config["distribution"]

            if dist_config["type"] == "uniform":
                if "lower" not in dist_config or "upper" not in dist_config:
                    raise ValueError(
                        f"Prior distribution for '{param_name}' is missing 'lower' or 'upper' bounds"
                    )
                lower = dist_config.get("lower", 0.0)
                upper = dist_config.get("upper", 1.0)
                if lower >= upper:
                    raise ValueError(
                        f"Invalid bounds for '{param_name}': lower ({lower}) must be < upper ({upper})"
                    )
                prior[param_name] = dist.Uniform(*torch.tensor([lower, upper], device=self.device))
            elif dist_config["type"] == "gaussian":
                if "loc" not in dist_config or "scale" not in dist_config:
                    raise ValueError(
                        f"Prior distribution for '{param_name}' is missing 'loc' or 'scale' bounds"
                    )
                loc = dist_config.get("loc", 0.0)
                scale = dist_config.get("scale", 1.0)
                if scale <= 0:
                    raise ValueError(
                        f"Invalid scale for '{param_name}': scale ({scale}) must be > 0"
                    )
                prior[param_name] = dist.Normal(
                    torch.tensor(loc, device=self.device), torch.tensor(scale, device=self.device)
                )
            else:
                raise ValueError(
                    f"Distribution type '{dist_config['type']}' not supported. Supported types: 'uniform', 'gaussian'."
                )

            if "multiplier" in param_config.keys():
                setattr(self, f"{param_name}_multiplier", float(param_config["multiplier"]))

        return prior, param_constraints, latex_labels, prior_flow, prior_flow_metadata

    @profile_method
    def sigma_scaling_factor(self, passed_ratio, index):
        """
        Calculate the scaling factor for likelihood uncertainties from the passed-tracer counts
        implied by the design.

        ``passed_ratio`` (``calc_passed`` output) is linear in the design ``class_ratio``, so it
        already encodes *both* the split across tracers *and* the total observation budget (the
        design sum): a design scaled uniformly by ``s`` gives ``passed_ratio = s *
        nominal_passed_ratio``. The per-bin count is ``N_i = passed_ratio_i * nominal_total_obs``
        (see ``_passed_ratio_to_n_tracers``), so the shot-noise scaling ``sigma ~ 1/sqrt(N)`` is

            factor = sqrt(nominal_passed_ratio / passed_ratio)

        which reduces to ``1/sqrt(s)`` under uniform scaling.

        Do NOT also divide by the design sum ("total_obs_multiplier"): ``passed_ratio`` already
        carries the budget, so that double-counts it and yields ``sigma ~ 1/s``. It is a no-op at
        the historical ``sum == 1`` designs, which is why it went unnoticed; it only biases
        designs whose sum is free. See tests/test_num_tracers_sigma_scaling.py.

        Note this is scaling mode's shot-noise *idealization*. Emulator mode learns the true BAO
        forecast (including sample-variance saturation, where dense bins improve far more slowly
        than 1/sqrt(N)) and does not use this method.

        Args:
            passed_ratio: Fraction of passed objects in each tracer bin (``calc_passed`` output);
                carries both the split and the budget.
            index: Index for the specific measurement

        Returns:
            Scaling factor for sigma: sigma_new = sigma_nominal * scaling_factor

        """
        factor = torch.sqrt(self.nominal_passed_ratio[index] / passed_ratio[..., index])
        if not self.vary_lya_qso and self._lya_qso_rows.size:
            # By default the Lya QSO error is held fixed. `index` is a positional array
            # into desi_data rows; entries that are Lya QSO rows keep the nominal sigma
            # (factor = 1), making that error design-independent. torch.where keeps this
            # autograd-safe. Set vary_lya_qso=True to let it scale like the other tracers.
            fixed = torch.as_tensor(
                np.isin(np.asarray(index), self._lya_qso_rows), device=factor.device
            )
            factor = torch.where(fixed, torch.ones_like(factor), factor)
        return factor

    @profile_method
    def calc_passed(self, class_ratio):
        label_to_index = {str(label): i for i, label in enumerate(self.design_labels)}

        def _label_index(label):
            label_str = str(label)
            if label_str not in label_to_index:
                raise ValueError(
                    f"Required design label '{label_str}' not found in design_labels={list(self.design_labels)}."
                )
            return label_to_index[label_str]

        def _grid_axis(grid, label):
            label_str = str(label)
            if label_str not in grid.names:
                raise ValueError(
                    f"Could not find design axis '{label_str}'. "
                    f"Available names={list(grid.names)}."
                )
            return np.asarray(getattr(grid, label_str), dtype=np.float64)

        if type(class_ratio) == torch.Tensor:
            assert class_ratio.shape[-1] == len(
                self.design_labels
            ), f"class_ratio should have {len(self.design_labels)} columns"
            obs_ratio = torch.zeros(
                (*class_ratio.shape[:-1], len(self.desi_data)),
                device=self.device,
            )
            idx_BGS = _label_index("BGS")
            idx_LRG = _label_index("LRG")
            idx_ELG = _label_index("ELG")
            idx_QSO = _label_index("QSO")

            # multiply each class ratio by the observed fraction in each tracer bin
            BGSs = self.desi_tracers.loc[self.desi_tracers["class"] == "BGS"]["observed"]
            BGS_dist = class_ratio[..., idx_BGS].unsqueeze(-1) * torch.tensor(
                (BGSs / BGSs.sum()).values, device=self.device
            ).unsqueeze(0)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "BGS")[0]] = BGS_dist[
                ..., 0
            ].unsqueeze(-1)

            LRGs = self.desi_tracers.loc[self.desi_tracers["class"] == "LRG"]["observed"]
            LRG_dist = class_ratio[..., idx_LRG].unsqueeze(-1) * torch.tensor(
                (LRGs / LRGs.sum()).values, device=self.device
            ).unsqueeze(0)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "LRG1")[0]] = LRG_dist[
                ..., 0
            ].unsqueeze(-1)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "LRG2")[0]] = LRG_dist[
                ..., 1
            ].unsqueeze(-1)

            ELGs = self.desi_tracers.loc[self.desi_tracers["class"] == "ELG"]["observed"]
            ELG_dist = class_ratio[..., idx_ELG].unsqueeze(-1) * torch.tensor(
                (ELGs / ELGs.sum()).values, device=self.device
            ).unsqueeze(0)
            # add the last value in LRG_dist to the first value in ELG_dist to get LRG3+ELG1
            obs_ratio[..., np.where(self.desi_data["tracer"] == "LRG3+ELG1")[0]] = (
                LRG_dist[..., 2] + ELG_dist[..., 0]
            ).unsqueeze(-1)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "ELG2")[0]] = ELG_dist[
                ..., 1
            ].unsqueeze(-1)

            QSOs = self.desi_tracers.loc[self.desi_tracers["class"] == "QSO"]["observed"]
            QSO_dist = class_ratio[..., idx_QSO].unsqueeze(-1) * torch.tensor(
                (QSOs / QSOs.sum()).values, device=self.device
            ).unsqueeze(0)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "QSO")[0]] = QSO_dist[
                ..., 0
            ].unsqueeze(-1)
            obs_ratio[..., np.where(self.desi_data["tracer"] == "Lya QSO")[0]] = QSO_dist[
                ..., 1
            ].unsqueeze(-1)

            efficiency = torch.zeros(
                (*class_ratio.shape[:-1], len(self.desi_data)),
                device=self.device,
            )
            efficiency[..., np.where(self.desi_data["tracer"] == "BGS")[0]] = (
                torch.tensor(
                    self.desi_tracers.loc[
                        self.desi_tracers["tracer"] == "BGS", "efficiency"
                    ].values[0],
                    device=self.device,
                )
                .expand_as(BGS_dist[..., 0])
                .unsqueeze(-1)
            )
            efficiency[..., np.where(self.desi_data["tracer"] == "LRG1")[0]] = (
                torch.tensor(
                    self.desi_tracers.loc[
                        self.desi_tracers["tracer"] == "LRG1", "efficiency"
                    ].values[0],
                    device=self.device,
                )
                .expand_as(LRG_dist[..., 0])
                .unsqueeze(-1)
            )
            efficiency[..., np.where(self.desi_data["tracer"] == "LRG2")[0]] = (
                torch.tensor(
                    self.desi_tracers.loc[
                        self.desi_tracers["tracer"] == "LRG2", "efficiency"
                    ].values[0],
                    device=self.device,
                )
                .expand_as(LRG_dist[..., 1])
                .unsqueeze(-1)
            )
            efficiency[..., np.where(self.desi_data["tracer"] == "LRG3+ELG1")[0]] = (
                (LRG_dist[..., 2] / (LRG_dist[..., 2] + ELG_dist[..., 0]))
                * self.desi_tracers.loc[self.desi_tracers["tracer"] == "LRG3", "efficiency"].values[
                    0
                ]
                + (ELG_dist[..., 0] / (LRG_dist[..., 2] + ELG_dist[..., 0]))
                * self.desi_tracers.loc[self.desi_tracers["tracer"] == "ELG1", "efficiency"].values[
                    0
                ]
            ).unsqueeze(-1)
            efficiency[..., np.where(self.desi_data["tracer"] == "ELG2")[0]] = (
                torch.tensor(
                    self.desi_tracers.loc[
                        self.desi_tracers["tracer"] == "ELG2", "efficiency"
                    ].values[0],
                    device=self.device,
                )
                .expand_as(ELG_dist[..., 1])
                .unsqueeze(-1)
            )
            efficiency[..., np.where(self.desi_data["tracer"] == "QSO")[0]] = (
                torch.tensor(
                    self.desi_tracers.loc[
                        self.desi_tracers["tracer"] == "QSO", "efficiency"
                    ].values[0],
                    device=self.device,
                )
                .expand_as(QSO_dist[..., 0])
                .unsqueeze(-1)
            )
            efficiency[..., np.where(self.desi_data["tracer"] == "Lya QSO")[0]] = (
                torch.tensor(
                    self.desi_tracers.loc[
                        self.desi_tracers["tracer"] == "Lya QSO", "efficiency"
                    ].values[0],
                    device=self.device,
                )
                .expand_as(QSO_dist[..., 1])
                .unsqueeze(-1)
            )

            # scale obs_ratio by efficiency to get the number of passed objects
            passed_ratio = obs_ratio * efficiency

            return passed_ratio
        elif type(class_ratio) == Grid:
            obs_ratio = np.zeros((5,) + len(class_ratio.shape) * (1,))

            LRGs = self.desi_tracers.loc[self.desi_tracers["class"] == "LRG"]["observed"]
            LRG_axis = _grid_axis(class_ratio, "LRG")
            LRG_dist = (LRG_axis * np.array((LRGs / LRGs.sum()).values)).squeeze()

            obs_ratio[0:2, ...] = LRG_dist[0:2]

            ELGs = self.desi_tracers.loc[self.desi_tracers["class"] == "ELG"]["observed"]
            ELG_axis = _grid_axis(class_ratio, "ELG")
            ELG_dist = (ELG_axis * np.array((ELGs / ELGs.sum()).values)).squeeze()
            obs_ratio[..., 2] = LRG_dist[..., 2] + ELG_dist[..., 0]
            obs_ratio[..., 3] = ELG_dist[..., 1]

            QSOs = self.desi_tracers.loc[self.desi_tracers["class"] == "QSO"]["observed"]
            QSO_axis = _grid_axis(class_ratio, "QSO")
            QSO_dist = (QSO_axis * np.array((QSOs / QSOs.sum()).values)).squeeze()
            obs_ratio[..., 4] = QSO_dist[..., 0]

            efficiency = np.stack(
                [
                    np.array(
                        self.desi_tracers.loc[
                            self.desi_tracers["tracer"] == "LRG1", "efficiency"
                        ].values[0]
                    ),
                    np.array(
                        self.desi_tracers.loc[
                            self.desi_tracers["tracer"] == "LRG2", "efficiency"
                        ].values[0]
                    ),
                    (LRG_dist[..., 2] / (LRG_dist[..., 2] + ELG_dist[..., 0]))
                    * self.desi_tracers.loc[
                        self.desi_tracers["tracer"] == "LRG3", "efficiency"
                    ].values[0]
                    + (ELG_dist[..., 0] / (LRG_dist[..., 2] + ELG_dist[..., 0]))
                    * self.desi_tracers.loc[
                        self.desi_tracers["tracer"] == "ELG1", "efficiency"
                    ].values[0],
                    np.array(
                        self.desi_tracers.loc[
                            self.desi_tracers["tracer"] == "ELG2", "efficiency"
                        ].values[0]
                    ),
                    np.array(
                        self.desi_tracers.loc[
                            self.desi_tracers["tracer"] == "Lya QSO", "efficiency"
                        ].values[0]
                    ),
                ],
                axis=-1,
            )

            passed_ratio = obs_ratio * efficiency

            return passed_ratio

    # Note: _E_of_z, D_H_func, D_M_func, D_V_func are inherited from CosmologyMixin

    @profile_method
    def get_guide_samples(
        self, guide, context=None, num_samples=5000, params=None, transform_output=True
    ):
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
        self.apply_multipliers(param_samples)

        if params is None:
            names = self.cosmo_params
            labels = self.latex_labels
        else:
            param_indices = [
                self.cosmo_params.index(param) for param in params if param in self.cosmo_params
            ]
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
            param_samples_gd = getdist.MCSamples(
                samples=param_samples.cpu().numpy(), names=names, labels=labels
            )

        return param_samples_gd

    def get_nominal_samples(self, num_samples=100000, params=None, transform_output=False):
        param_samples, target_labels, latex_labels = load_nominal_samples(
            "num_tracers", self.cosmo_model, dataset=self.dataset, analysis=self.analysis
        )
        param_samples = param_samples[:num_samples]
        if transform_output:
            param_samples = torch.tensor(param_samples, device=self.device)
            if self.analysis == "bao":
                param_samples[..., -1] /= 100  # to get hrdrag in units of 100 km/s/Mpc
            param_samples = (
                self.params_to_unconstrained(param_samples, bijector_class=self.desi_bijector)
                .cpu()
                .numpy()
            )

        if params is None:
            names = target_labels
            labels = latex_labels
        else:
            param_indices = [
                target_labels.index(param) for param in params if param in target_labels
            ]
            param_samples = param_samples[:, param_indices]
            names = [target_labels[i] for i in param_indices]
            labels = [latex_labels[i] for i in param_indices]

        # The legend label of every plot that overlays this reference.
        source = {"bao": "BAO", "shapefit": "ShapeFit"}[self.analysis]
        with contextlib.redirect_stdout(io.StringIO()):
            desi_samples_gd = getdist.MCSamples(
                samples=param_samples, names=names, labels=labels,
                label=f"DESI {self.dataset.upper()} {source}",
            )

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
        if central and self.analysis == "shapefit":
            # Fixed fiducial mean (central_val); covariance from the emulators at the
            # fiducial cosmology and this design's N_tracers.
            n_tracers = self._shapefit_n_tracers(lexpand(tracer_ratio, num_samples))
            _, covariance_matrix = self._shapefit_likelihood(
                n_tracers, self._shapefit_fiducial_parameters())
            with pyro.plate("data", num_samples):
                data_samples = pyro.sample(
                    self.observation_labels[0],
                    dist.MultivariateNormal(self.central_val, covariance_matrix.squeeze(-3)),
                ).unsqueeze(1)
        elif central:
            # Expand tracer_ratio for batching with num_samples
            expanded_tracer_ratio = lexpand(tracer_ratio, num_samples)
            passed_ratio = self.calc_passed(expanded_tracer_ratio)
            means = torch.zeros(
                passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device
            )
            # Broadcast central values across batch dimensions
            means[:, :, self.DH_idx] = self.central_val[self.DH_idx]
            rescaled_sigmas = torch.zeros(
                passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device
            )
            rescaled_sigmas[:, :, self.DH_idx] = self.sigmas[
                self.DH_idx
            ] * self.sigma_scaling_factor(passed_ratio, self.DH_idx)
            if self.include_D_M:
                means[:, :, self.DM_idx] = self.central_val[self.DM_idx]
                rescaled_sigmas[:, :, self.DM_idx] = self.sigmas[
                    self.DM_idx
                ] * self.sigma_scaling_factor(passed_ratio, self.DM_idx)
            if self.include_D_V:
                means[:, :, self.DV_idx] = self.central_val[self.DV_idx]
                rescaled_sigmas[:, :, self.DV_idx] = self.sigmas[
                    self.DV_idx
                ] * self.sigma_scaling_factor(passed_ratio, self.DV_idx)

            if self.include_D_V and self.include_D_M:
                covariance_matrix = self.corr_matrix * (
                    rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2)
                )
            else:
                covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (
                    rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-1)
                    * rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-2)
                )

            with pyro.plate("data", num_samples):
                data_samples = pyro.sample(
                    self.observation_labels[0],
                    dist.MultivariateNormal(means.squeeze(), covariance_matrix.squeeze()),
                ).unsqueeze(1)
        else:
            data_samples = self.pyro_model(tracer_ratio)
        return data_samples

    def sample_params_from_data_samples(
        self,
        tracer_ratio,
        guide,
        num_data_samples=100,
        num_param_samples=1000,
        central=True,
        transform_output=True,
    ):
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
        from bedcosmo.artifacts import observations_to_numpy

        data_samples = self.sample_data(tracer_ratio, num_data_samples, central)
        y_np = observations_to_numpy(data_samples, num_data_samples)
        # Expand tracer_ratio to match data_samples shape for concatenation
        # tracer_ratio is [1, 4], need to expand to [num_data_samples, 1, 4]
        if tracer_ratio.dim() == 2:
            expanded_tracer_ratio = tracer_ratio.unsqueeze(0).expand(num_data_samples, -1, -1)
        else:
            expanded_tracer_ratio = tracer_ratio.expand(num_data_samples, -1, -1)
        context = torch.cat([expanded_tracer_ratio, data_samples], dim=-1)

        # Vectorized sampling: expand contexts to (num_data_samples * num_param_samples, context_dim)
        # Each context is repeated num_param_samples times
        context_squeezed = (
            context.squeeze(1) if context.dim() == 3 else context
        )  # Shape: [num_data_samples, context_dim]
        # Expand: [num_data_samples, context_dim] -> [num_data_samples, num_param_samples, context_dim] -> [num_data_samples * num_param_samples, context_dim]
        expanded_context = (
            context_squeezed.unsqueeze(1).expand(-1, num_param_samples, -1).contiguous()
        )
        expanded_context = expanded_context.view(
            -1, expanded_context.shape[-1]
        )  # [num_data_samples * num_param_samples, context_dim]

        # Sample all parameters at once
        with torch.no_grad():
            param_samples = guide(expanded_context).sample(
                ()
            )  # Shape: [num_data_samples * num_param_samples, num_params]

        # Apply transformations if needed
        if self.transform_input and transform_output:
            param_samples = self.params_from_unconstrained(param_samples)
        self.apply_multipliers(param_samples)

        # Reshape to [num_data_samples, num_param_samples, num_params]
        param_samples = param_samples.view(num_data_samples, num_param_samples, -1)

        # Check for any constant columns and add tiny noise to prevent getdist from excluding them
        for i in range(param_samples.shape[2]):
            col = param_samples[:, :, i]
            if torch.all(col == col[0, 0]):
                if self.global_rank == 0:
                    print(
                        f"Column {i} ({self.cosmo_params[i]}) is constant with value {col[0, 0]}, adding tiny noise"
                    )
                noise_scale = abs(col[0, 0]) * 1e-10
                param_samples[:, :, i] = col + torch.randn_like(col) * noise_scale

        # Convert to numpy array: [num_data_samples, num_param_samples, num_params]
        param_samples_array = param_samples.cpu().numpy()

        return param_samples_array, y_np
    def sample_brute_force(
        self,
        tracer_ratio,
        grid_designs,
        grid_features,
        grid_params,
        designer,
        num_data_samples=100,
        num_param_samples=1000,
    ):

        rescaled_sigmas = torch.zeros(self.sigmas.shape, device=self.device)
        rescaled_sigmas[self.DH_idx] = self.sigmas[self.DH_idx] * torch.sqrt(
            (self.efficiency[self.DH_idx] * tracer_ratio) / self.nominal_passed_ratio[self.DH_idx]
        )
        if self.include_D_M:
            means = self.central_val
            rescaled_sigmas[self.DM_idx] = self.sigmas[self.DM_idx] * torch.sqrt(
                (self.efficiency[self.DM_idx] * tracer_ratio)
                / self.nominal_passed_ratio[self.DM_idx]
            )
            covariance_matrix = self.corr_matrix * (
                rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2)
            )
        else:
            means = self.central_val[self.DH_idx]
            covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (
                rescaled_sigmas[self.DH_idx].unsqueeze(-1)
                * rescaled_sigmas[self.DH_idx].unsqueeze(-2)
            )

        with pyro.plate("data", num_data_samples):
            data_samples = pyro.sample(
                self.observation_labels[0], dist.MultivariateNormal(means, covariance_matrix)
            )

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
            param_mesh = np.stack(
                np.meshgrid(
                    *[
                        getattr(grid_params, grid_params.names[i]).squeeze()
                        for i in range(len(grid_params.names))
                    ],
                    indexing="ij",
                ),
                axis=-1,
            )
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
        rescaled_sigmas[..., self.DH_idx] = self.sigmas[self.DH_idx] * torch.sqrt(
            (self.efficiency[self.DH_idx] * tracer_ratio) / self.nominal_passed_ratio[self.DH_idx]
        )
        if self.include_D_M:
            y = self.central_val
            rescaled_sigmas[..., self.DM_idx] = self.sigmas[self.DM_idx] * torch.sqrt(
                (self.efficiency[self.DM_idx] * tracer_ratio)
                / self.nominal_passed_ratio[self.DM_idx]
            )
            covariance_matrix = self.corr_matrix * (
                rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2)
            )
        else:
            y = self.central_val[self.DH_idx]
            covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (
                rescaled_sigmas[..., self.DH_idx].unsqueeze(-1)
                * rescaled_sigmas[..., self.DH_idx].unsqueeze(-2)
            )
        with GridStack(grid_params):
            parameters = {
                k: torch.tensor(getattr(grid_params, k), device=self.device).unsqueeze(-1)
                for k in grid_params.names
            }
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
        param_mesh = np.stack(
            np.meshgrid(
                *[
                    getattr(grid_params, grid_params.names[i]).squeeze()
                    for i in range(len(grid_params.names))
                ],
                indexing="ij",
            ),
            axis=-1,
        )
        for i in range(num_param_samples):
            param_samples.append(param_mesh[tuple(indices[i])])
        return torch.tensor(np.array(param_samples), device=self.device)

    def exact_prior_marginals(self, use_prior_flow=True):
        # Mirrors sample_parameters: a prior flow, or a 2D constraint drawn by
        # ConstrainedUniform2D, reshapes the marginals; the other parameters it
        # draws come straight from their priors.
        if use_prior_flow and self.prior_flow is not None:
            return set()
        drawn = {
            "bao": {"Om", "Ok", "w0", "wa", "hrdrag"},
            "shapefit": {"omega_cdm", "omega_b", "h", "ln10A_s", "n_s"},
        }[self.analysis]
        constrained = {
            p for c in self.param_constraints.values() for p in c["affected_parameters"]
        }
        return (set(self.prior) & drawn) - constrained

    @profile_method
    def sample_parameters(self, sample_shape, prior=None, use_prior_flow=True):
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
        if use_prior_flow and hasattr(self, "prior_flow") and self.prior_flow is not None:
            # Get raw samples from prior flow (shape: *sample_shape, n_params)
            samples = self._sample_prior_flow_cache(sample_shape)

            # Register each parameter with pyro.sample
            for i, param_name in enumerate(self.cosmo_params):
                param_values = samples[..., i]
                parameters[param_name] = pyro.sample(
                    param_name, dist.Delta(param_values)
                ).unsqueeze(-1)

            return parameters

        # Otherwise sample from explicit prior distributions
        if prior is None:
            prior = self.prior

        if self.analysis == "shapefit":
            return self._sample_shapefit_parameters(sample_shape, prior)

        # Handle constraints based on YAML configuration
        if hasattr(self, "param_constraints"):
            # Check for valid density constraint
            if "valid_densities" in self.param_constraints:
                # 0 < Om + Ok < 1
                OmOk_prior = {"Om": prior["Om"], "Ok": prior["Ok"]}
                OmOk = ConstrainedUniform2D(
                    OmOk_prior, **self.param_constraints["valid_densities"]["bounds"]
                )
                OmOk_samples = OmOk.sample(sample_shape)
                # The pair is uniform on the constrained region; record its density once, on Om.
                parameters["Om"] = pyro.sample(
                    "Om", PresampledPrior(OmOk_samples[..., 0], log_density=-math.log(float(OmOk.areas.sum())))
                ).unsqueeze(-1)
                parameters["Ok"] = pyro.sample("Ok", PresampledPrior(OmOk_samples[..., 1])).unsqueeze(-1)
            else:
                # Sample Om, Ok normally if no constraint or Ok not present
                if "Om" in prior.keys():
                    parameters["Om"] = pyro.sample("Om", prior["Om"]).unsqueeze(-1)
                if "Ok" in prior.keys():
                    parameters["Ok"] = pyro.sample("Ok", prior["Ok"]).unsqueeze(-1)

            # Check for high z matter domination constraint
            if "high_z_matter_dom" in self.param_constraints:
                # w0 + wa < 0
                w0wa_prior = {"w0": prior["w0"], "wa": prior["wa"]}
                w0wa = ConstrainedUniform2D(
                    w0wa_prior, **self.param_constraints["high_z_matter_dom"]["bounds"]
                )
                w0wa_samples = w0wa.sample(sample_shape)
                # Uniform on the constrained region; density recorded once, on w0 (see Om/Ok).
                parameters["w0"] = pyro.sample(
                    "w0", PresampledPrior(w0wa_samples[..., 0], log_density=-math.log(float(w0wa.areas.sum())))
                ).unsqueeze(-1)
                parameters["wa"] = pyro.sample("wa", PresampledPrior(w0wa_samples[..., 1])).unsqueeze(-1)
            else:
                # Sample w0, wa normally if no constraint or wa not present
                if "w0" in prior.keys():
                    parameters["w0"] = pyro.sample("w0", prior["w0"]).unsqueeze(-1)
                if "wa" in prior.keys():
                    parameters["wa"] = pyro.sample("wa", prior["wa"]).unsqueeze(-1)
        else:
            # if no parameter constraints defined
            if "Om" in prior.keys():
                parameters["Om"] = pyro.sample("Om", prior["Om"]).unsqueeze(-1)
            if "w0" in prior.keys():
                parameters["w0"] = pyro.sample("w0", prior["w0"]).unsqueeze(-1)

        # Always sample hrdrag
        parameters["hrdrag"] = pyro.sample("hrdrag", prior["hrdrag"]).unsqueeze(-1)

        return parameters

    # Fixed neutrino density of the DESI fiducial (one massive neutrino, 0.06 eV), as in
    # desilike_emulator bao/core.py _OMEGA_NU_FID. Enters the derived Omega_m that bounds
    # the shapefit emulators' training domain.
    _OMEGA_NU_FID = 0.06 / 93.14

    _OMEGA_M_COUPLED = ("omega_cdm", "omega_b", "h")

    def _omega_m_in_domain(self, omega_cdm, omega_b, h):
        bounds = self.param_constraints["omega_m_domain"]["bounds"]
        omega_m = (omega_cdm + omega_b + self._OMEGA_NU_FID) / h ** 2
        return (omega_m >= bounds["lower"]) & (omega_m <= bounds["upper"])

    def _omega_m_domain_log_acceptance(self, n_chunks=4, chunk_size=10_000_000):
        """log P(accept) of the omega_m_domain cut under the independent prior.

        The truncated prior's normalizer. Monte Carlo over n_chunks * chunk_size draws:
        relative error ~2e-4 at the defaults, i.e. a ~2e-4 nat, design-independent shift
        of H_prior. Seeded under a forked RNG, so every rank gets the same value and the
        run's seeded stream is untouched.
        """
        if "omega_m_domain" not in self.param_constraints:
            raise ValueError(
                "analysis='shapefit' needs the 'omega_m_domain' constraint in prior_args: "
                "the emulators are only trained on Omega_m inside it.")
        accepted = 0
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.manual_seed(0)
            for _ in range(n_chunks):
                draws = {n: self.prior[n].sample((chunk_size,)) for n in self._OMEGA_M_COUPLED}
                accepted += int(self._omega_m_in_domain(**draws).sum())
        return math.log(accepted / (n_chunks * chunk_size))

    def _sample_shapefit_parameters(self, sample_shape, prior):
        """Draw the omega-basis cosmology, rejecting Omega_m outside the emulator domain.

        The shapefit training generator rejects samples whose derived
        Omega_m = (omega_cdm + omega_b + omega_nu) / h^2 falls outside
        [lower, upper] (desilike_emulator shapefit/core.py _check_omega_m; ~62% of the
        raw prior box), so the emulators are only valid inside it. The
        ``omega_m_domain`` constraint carries those bounds; rejected rows are redrawn
        jointly from the prior until every row is inside. The other parameters are
        independent.

        Accepted draws have density prior(theta) * 1[in domain] / P(accept). Each coupled
        site records its marginal prior log-density, and omega_cdm also carries
        -log P(accept), so the trace's summed prior log-prob (and hence H_prior) is exact;
        a plain Delta would score 0 (see PresampledPrior).
        """
        if prior is not self.prior:
            raise ValueError(
                "shapefit's Omega_m acceptance is computed for self.prior; sampling another "
                "prior would record the wrong truncated density.")
        shape = torch.Size(sample_shape)
        coupled = self._OMEGA_M_COUPLED
        draws = {name: prior[name].sample(shape) for name in coupled}
        while True:
            reject = ~self._omega_m_in_domain(**draws)
            if not reject.any():
                break
            for name in coupled:
                draws[name] = torch.where(reject, prior[name].sample(shape), draws[name])

        parameters = {}
        for name in prior:
            if name in coupled:
                log_density = prior[name].log_prob(draws[name])
                if name == coupled[0]:
                    log_density = log_density - self._omega_m_log_acceptance
                parameters[name] = pyro.sample(
                    name, PresampledPrior(draws[name], log_density=log_density)).unsqueeze(-1)
            else:
                parameters[name] = pyro.sample(name, prior[name]).unsqueeze(-1)
        return parameters

    @staticmethod
    def resolve_emulator_checkpoints(analysis, cosmo_model, dataset, quantity):
        """Resolve emulator checkpoint paths for one (analysis, cosmo_model, dataset, quantity).

        Emulator checkpoints live in emulators.yaml under
        <analysis>.<dataset>.<cosmo_model>.<quantity>, mirroring the storage path
        models/{dataset}/{cosmo_model}/{quantity}/. ``quantity`` is desilike-emulator's
        ``--quantity``: the forecast space for bao (config | fourier, one per run, set by
        ``emulator_space``), and mean or covar for shapefit (both loaded together).
        Relative paths resolve against
        $SCRATCH/bedcosmo/num_tracers/emulator/{analysis}/models/{dataset}/{cosmo_model}/;
        absolute paths are used verbatim; null -> fall back to fixed DESI nominal covariance
        (bao only).

        The {analysis} segment is required: the emulator repo stores each analysis in its own
        subtree, and 'covar' is a valid quantity for both bao and shapefit, so an analysis-less
        path is ambiguous. Omitting it resolves to a non-existent directory, which this method
        reports as null -> every bin silently falls back to the fixed DESI covariance instead of
        using the emulator at all.

        Returns:
            dict mapping tracer-bin -> absolute checkpoint path (or None for fallback bins).
        """
        emulators_yaml_path = get_experiment_config_path("num_tracers", "emulators.yaml")
        with open(emulators_yaml_path, "r") as f:
            all_emulators = yaml.safe_load(f)
        by_analysis = all_emulators.get(analysis, {})
        if dataset not in by_analysis:
            raise ValueError(
                f"No emulator checkpoints for dataset '{dataset}' under '{analysis}' "
                f"in emulators.yaml (have: {sorted(by_analysis)})"
            )
        by_dataset = by_analysis[dataset]
        if cosmo_model not in by_dataset:
            raise ValueError(
                f"Cosmo model '{cosmo_model}' has no emulator entry under "
                f"{analysis}.{dataset} in emulators.yaml (have: {sorted(by_dataset)})"
            )
        by_quantity = by_dataset[cosmo_model]
        if quantity is None:
            raise ValueError(
                f"emulator_space is required: {analysis}.{dataset}.{cosmo_model} in "
                f"emulators.yaml is keyed by forecast space (have: {sorted(by_quantity)}). "
                f"Set it in train_args.yaml or pass --emulator-space."
            )
        if quantity not in by_quantity:
            raise ValueError(
                f"No emulator checkpoints for '{quantity}' under "
                f"{analysis}.{dataset}.{cosmo_model} (have: {sorted(by_quantity)})"
            )
        entry = by_quantity[quantity]

        base_dir = os.path.join(
            storage_path, "emulator", analysis, "models", dataset, cosmo_model)
        return {
            tb: (None if p is None else (p if os.path.isabs(p) else os.path.join(base_dir, p)))
            for tb, p in entry.items()
        }

    def _load_emulators(self):
        """Load trained NN emulators for each tracer bin from checkpoint files.

        Tracer bins whose checkpoint is null (None) are skipped and recorded in
        ``self._emulator_fallback_bins``; their covariance blocks fall back to the
        fixed DESI nominal covariance in ``_build_emulator_covariance``.
        """
        self._emulators = {}
        self._emulator_fallback_bins = []
        for tracer_bin, ckpt_path in self._emulator_checkpoints.items():
            if ckpt_path is None:
                self._emulator_fallback_bins.append(tracer_bin)
                continue
            self._emulators[tracer_bin] = self._load_emulator(ckpt_path)

        if self.global_rank == 0:
            self._print_emulator_summary(self._emulators)
            if self._emulator_fallback_bins:
                print(
                    f"No emulator checkpoint for tracer bins {self._emulator_fallback_bins}; "
                    f"falling back to fixed DESI nominal covariance for these."
                )

    def _load_emulator(self, ckpt_path):
        """Load one emulator checkpoint into the dict ``_emulator_predict`` consumes."""
        from desilike_emulator.util import DEFAULT_SIGMA_FLOOR, load_model, model_label

        # Sigma clamps passed to decode_and_unscale. The floor is the
        # emulator's own convention (single-sourced from util, so it can't drift
        # from what the model was decoded with); the ceiling is a bedcosmo choice
        # (util has no default -- it defaults OFF) that caps non-detection-tail
        # extrapolation so the covariance stays finite and PD. See _SIGMA_CEILING.
        self._sigma_floor = DEFAULT_SIGMA_FLOOR
        self._sigma_ceiling = self._SIGMA_CEILING

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model = load_model(ckpt).to(self.device)
        model.requires_grad_(False)

        # Recover the trained N_tracers box from the input standardization
        # stats: N was drawn uniform[a, b], so mu = (a+b)/2 and
        # sigma = (b-a)/sqrt(12)  =>  [a, b] = mu -+ sqrt(3)*sigma. Used only
        # to warn (not clamp) on gross N-extrapolation, which signals a
        # wiring bug rather than a valid design excursion.
        pnames = list(ckpt["param_names"])
        n_train_lo = n_train_hi = None
        if "N_tracers" in pnames:
            i_n = pnames.index("N_tracers")
            xmu = ckpt["x_mu"].reshape(-1)[i_n].item()
            xsg = ckpt["x_sigma"].reshape(-1)[i_n].item()
            half = (3.0 ** 0.5) * xsg
            n_train_lo, n_train_hi = xmu - half, xmu + half

        # scale_data.py recipe recorded by train.py. Empty = unscaled (legacy).
        # Must be passed to decode_and_unscale or σ comes back still multiplied
        # by the scale factor (e.g. hrdrag ~ 100) and hits _SIGMA_CEILING.
        scale_expressions = list(ckpt.get("scale_expressions") or [])

        return {
            "model": model,
            "n_train_lo": n_train_lo,
            "n_train_hi": n_train_hi,
            "param_names": list(ckpt["param_names"]),
            "target_names": list(ckpt["target_names"]),
            "x_mu": ckpt["x_mu"].to(self.device),
            "x_sigma": ckpt["x_sigma"].to(self.device),
            "y_mu": ckpt["y_mu"].to(self.device),
            "y_sigma": ckpt["y_sigma"].to(self.device),
            "log_normalize": ckpt.get("log_normalize", False),
            "y_linthresh": (
                ckpt["y_linthresh"].to(self.device)
                if ckpt.get("y_linthresh") is not None
                else None
            ),
            "scale_expressions": scale_expressions,
            # Provenance, for the summary printed below. A forecast is only
            # as good as the emulator behind it, and the run log is the only
            # place that pairing is recorded.
            "ckpt_path": str(ckpt_path),
            "arch": model_label(ckpt.get("architecture", "resnet"), ckpt["model_kwargs"]),
            "n_params": sum(p_.numel() for p_ in model.parameters()),
            "git_commit": str(ckpt.get("git_commit", "") or ""),
        }

    def _print_emulator_summary(self, emulators):
        """Log WHICH emulators were loaded and their defining properties.

        Every forecast this class produces is conditioned on these checkpoints, and
        a run log that records only "loaded 6 emulators" cannot be audited later:
        the same tracer name has meant a 6/4/4 unscaled net and a 64/6/4 net trained
        on hrdrag-scaled targets within one week. Printing the path, architecture,
        target transform and scaling recipe makes a stale or mixed pool obvious at a
        glance instead of after re-deriving it from the numbers.
        """
        if not emulators:
            print("No emulators loaded; every tracer bin falls back to the fixed "
                  "DESI nominal covariance.")
            return
        print(f"Loaded {len(emulators)} emulator(s):")
        w = max(len(tb) for tb in emulators)
        for tb, emu in emulators.items():
            n_lo, n_hi = emu.get("n_train_lo"), emu.get("n_train_hi")
            nbox = (f"N[{n_lo:.3g}, {n_hi:.3g}]" if n_lo is not None else "N box unknown")
            print(f"  {tb:<{w}}  {emu['arch']} ({emu['n_params']:,} params)  "
                  f"{'symlog' if emu['log_normalize'] else 'raw'}  {nbox}")
            print(f"  {'':<{w}}  {emu['ckpt_path']}")
            # An empty recipe is meaningful, not missing: it says the targets are
            # already physical. Say so rather than printing nothing.
            recipe = (" * ".join(emu["scale_expressions"]) if emu["scale_expressions"]
                      else "none (targets already physical)")
            extra = f"  git {emu['git_commit'][:8]}" if emu["git_commit"] else ""
            print(f"  {'':<{w}}  targets {', '.join(emu['target_names'])}"
                  f"  | inputs {', '.join(emu['param_names'])}"
                  f"  | scaling: {recipe}{extra}")

    def _emulator_predict(self, emu, emulator_input):
        """Run differentiable inference through an emulator.

        Args:
            emu: A loaded emulator (``_load_emulator`` output), e.g. self._emulators['LRG1'].
            emulator_input: Tensor of shape (..., in_dim) with columns
                matching the checkpoint's ``param_names`` (raw / physical).

        Returns:
            Tensor of shape (..., n_targets) in physical units (σ ≥ sigma_floor;
            ρ from tanh decode). Target order matches the checkpoint's
            ``target_names``. Checkpoints trained on scale_data.py-rewritten
            labels are unscaled here via ``decode_and_unscale``.
        """
        from desilike_emulator.util import decode_and_unscale

        model_dtype = next(emu["model"].parameters()).dtype
        x = emulator_input.to(model_dtype)
        x_mu = emu["x_mu"].to(model_dtype)
        x_sigma = emu["x_sigma"].to(model_dtype)
        y_mu = emu["y_mu"].to(model_dtype)
        y_sigma = emu["y_sigma"].to(model_dtype)
        x_norm = (x - x_mu) / x_sigma
        y_norm = emu["model"](x_norm)
        # Numerical guard on non-detection / out-of-domain extrapolation: cap
        # sigma to a finite ceiling (inf/nan -> ceiling; encodes "no info") and
        # pull rho strictly inside (-1, 1) so the assembled covariance is always
        # finite and every 2x2 block is strictly positive-definite. Both are
        # applied inside the decode (rho clip is always-on there; the ceiling is
        # opt-in via sigma_ceiling and off for training/eval). Floor/ceiling run
        # AFTER unscaling so they keep physical units (see _SIGMA_CEILING).
        return decode_and_unscale(
            y_norm,
            y_mu,
            y_sigma,
            emu["target_names"],
            log_normalize=emu["log_normalize"],
            y_linthresh=emu["y_linthresh"],
            sigma_floor=self._sigma_floor,
            sigma_ceiling=self._sigma_ceiling,
            scale_expressions=emu["scale_expressions"],
            x_raw=x,
            param_names=emu["param_names"],
        )

    # How far outside the trained N_tracers box counts as "gross" extrapolation
    # worth warning about. A valid design only reaches ~0.70-1.05x the trained
    # box (verified for dr1), where the emulator stays accurate (~0.6%); values
    # far beyond this (e.g. raw counts, a bad hrdrag_multiplier) indicate a
    # wiring bug, so we warn -- but never clamp, to avoid silently biasing the
    # small, benign excursions the design legitimately makes.
    _N_EXTRAP_LO = 0.5   # warn below 0.5 x trained low
    _N_EXTRAP_HI = 2.0   # warn above 2.0 x trained high

    def _warn_n_extrapolation(self, tracer_bin, emu, n_values):
        """Warn once per emulator if fed N_tracers grossly outside training."""
        if emu["ckpt_path"] in self._n_extrap_warned:
            return
        lo, hi = emu["n_train_lo"], emu["n_train_hi"]
        if lo is None or hi is None:
            return
        with torch.no_grad():
            nmin = float(torch.as_tensor(n_values).min())
            nmax = float(torch.as_tensor(n_values).max())
        if nmin < self._N_EXTRAP_LO * lo or nmax > self._N_EXTRAP_HI * hi:
            self._n_extrap_warned.add(emu["ckpt_path"])
            import warnings
            warnings.warn(
                f"[{tracer_bin}] N_tracers fed to emulator "
                f"[{nmin:.3e}, {nmax:.3e}] is grossly outside the trained box "
                f"[{lo:.3e}, {hi:.3e}] (tolerance x{self._N_EXTRAP_LO}/"
                f"x{self._N_EXTRAP_HI}); predictions there are extrapolated and "
                f"may be unreliable. This usually signals an N_tracers wiring "
                f"bug, not a valid design.",
                RuntimeWarning,
            )

    def _passed_ratio_to_n_tracers(self, passed_ratio):
        """Convert a passed *ratio* (``calc_passed`` output) to absolute passed ``N_tracers``.

        ``calc_passed`` is the single source of the passed-tracer computation (shared with
        scaling mode) and returns a passed fraction per desi_data row. The emulator's
        ``N_tracers`` feature is an absolute redshift-confirmed count (desilike_emulator
        ``_get_ntracers`` reads the ``passed`` column), so this helper does just the final
        conversion: scale by ``self.nominal_total_obs`` and remap rows onto emulator-bin keys
        via ``_EMULATOR_TRACER_TO_DESI``.

        Identity: ``passed_ratio * nominal_total_obs == passed count`` for any design, because
        ``passed_ratio = class_ratio * observed_frac * efficiency`` and ``nominal_total_obs`` is
        the observed sum. At the nominal design this reproduces ``desi_data["passed"]`` per bin
        (== the emulator's ``_get_ntracers`` values).

        Args:
            passed_ratio: Tensor of shape (..., n_desi_rows) from ``calc_passed``.

        Returns:
            Dict mapping emulator tracer-bin name to a tensor of passed N_tracers (batch shape).
        """
        n_tracers = {}
        for tracer_bin, desi_name in self._EMULATOR_TRACER_TO_DESI.items():
            rows = self.desi_data.index[self.desi_data["tracer"] == desi_name].tolist()
            if not rows:
                continue  # bin absent from this dataset's data vector (never consumed)
            # All rows of a given tracer carry the same passed value; take the first.
            n_tracers[tracer_bin] = passed_ratio[..., rows[0]] * self.nominal_total_obs
        return n_tracers

    # Sigma ceiling for emulator extrapolation into the non-detection tail. At
    # degenerate cosmologies (e.g. very low Om) the true BAO forecast is itself
    # non-finite -- the training generator discards those rows -- so the emulator
    # has no signal there and its decode's expm1 overflows sigma to +inf, which
    # poisons the assembled covariance and crashes the MultivariateNormal
    # Cholesky. Passing this as ``sigma_ceiling`` to decode_and_unscale
    # caps sigma to a finite value that reproduces the true "no information"
    # verdict (precision -> 0) instead of crashing. Applied after unscaling so
    # a scale_data.py factor (e.g. hrdrag) is not mistaken for a non-detection.
    # Chosen to never touch in-domain values: it sits far above any real
    # forecast (in-domain sigma <~ 1e2, training-tail finite max ~1.5e7). The
    # companion rho clip to strictly inside (-1, 1) -- which keeps every 2x2
    # block PD -- lives in the decode itself (util._RHO_CLIP), always-on,
    # mirroring the forward transform.
    _SIGMA_CEILING = 1.0e8          # cap on decoded sigma (encodes zero precision)

    # Maps emulator tracer-bin keys (models.yaml likelihood_emulator) to the
    # tracer names used in desi_data.csv.
    _EMULATOR_TRACER_TO_DESI = {
        "BGS": "BGS",
        "LRG1": "LRG1",
        "LRG2": "LRG2",
        "LRG3_ELG1": "LRG3+ELG1",
        "ELG2": "ELG2",
        "QSO": "QSO",
        "Lya_QSO": "Lya QSO",
    }

    def _build_emulator_covariance(self, passed_ratio, parameters):
        """Build the block-diagonal covariance from emulator predictions.

        The emulator for each tracer bin predicts per-tracer targets:

        - Isotropic tracers (BGS; BGS+QSO for dr1): ``[sigma_DV_over_rd]`` only.
        - Anisotropic tracers: ``[sigma_DH_over_rd, sigma_DM_over_rd, rho_DH_DM]``
          assembled directly into the 2×2 block (PSD by construction).
        - Tracer bins with a null checkpoint (e.g. Lya_QSO in base): the
          within-tracer block falls back to the fixed DESI nominal covariance.

        Args:
            passed_ratio: Tensor of shape (..., n_desi_rows) from ``calc_passed`` — the passed
                (redshift-confirmed) fraction per desi_data row. The caller already computes this
                (``pyro_model`` needs it for the plate shape), so it is passed in rather than
                recomputed here.
            parameters: Dict of cosmological parameter tensors (from sample_parameters).

        Returns:
            Tensor of shape (..., n_data, n_data) — block-diagonal covariance.
        """
        n_tracers = self._passed_ratio_to_n_tracers(passed_ratio)

        # Extract cosmo params, filling missing ones with FIDUCIAL_PARAMS from cosmology.py
        Om = (
            parameters["Om"].squeeze(-1)
            if "Om" in parameters
            else torch.tensor(FIDUCIAL_PARAMS.get("Om", 0.3), device=self.device)
        )
        Ok = (
            parameters["Ok"].squeeze(-1)
            if "Ok" in parameters
            else torch.tensor(FIDUCIAL_PARAMS["Ok"], device=self.device)
        )
        w0 = (
            parameters["w0"].squeeze(-1)
            if "w0" in parameters
            else torch.tensor(FIDUCIAL_PARAMS["w0"], device=self.device)
        )
        wa = (
            parameters["wa"].squeeze(-1)
            if "wa" in parameters
            else torch.tensor(FIDUCIAL_PARAMS["wa"], device=self.device)
        )
        hrdrag = (
            parameters["hrdrag"].squeeze(-1)
            if "hrdrag" in parameters
            else torch.tensor(FIDUCIAL_PARAMS.get("hrdrag", 1.0), device=self.device)
        )

        # Determine batch shape from the passed-ratio (same leading dims as the design).
        batch_shape = passed_ratio.shape[:-1]

        # Convert bedcosmo's hrdrag parameter to the emulator's hrdrag feature,
        # which is the physical h*r_d (desilike _FID convention, fiducial ~99.08).
        # bedcosmo:  D_H/rd = c / (hrdrag_multiplier * hrdrag * E(z))
        # emulator:  D_H/rd = (c/100) / (h*r_d * E(z))
        # => h*r_d = hrdrag_multiplier * hrdrag / 100  (= 100*hrdrag when mult=1e4).
        hrdrag_multiplier = getattr(self, "hrdrag_multiplier", 100.0)
        hrdrag_phys = hrdrag * hrdrag_multiplier / 100.0

        # Normal path: emulator covariance at the actual design N and cosmology.
        if self.emulator_sqrtn_ref is None:
            return self._maybe_apply_desi_syst(
                self._fill_emulator_blocks(n_tracers, Om, Ok, w0, wa, hrdrag_phys, batch_shape)
            )

        # 1/sqrt(N) diagnostic: freeze the emulator at the nominal-design N (per bin) and
        # impose pure 1/sqrt(N) on the design dependence by scaling each tracer block by
        # N_ref/N (each sigma by sqrt(N_ref/N), rho unchanged). The reference covariance is
        # taken either at the sampled cosmology (N pinned to nominal) or at a cached fiducial
        # point (cosmology-independent, the analog of likelihood_mode='scaling').
        n_ref = self._sqrtn_nominal_n_tracers()
        if self.emulator_sqrtn_ref == "fiducial":
            cov_ref = self._sqrtn_fiducial_cov_ref()
        else:  # "sampled"
            cov_ref = self._fill_emulator_blocks(n_ref, Om, Ok, w0, wa, hrdrag_phys, batch_shape)

        # Per-data-row factor g[row] = sqrt(N_ref[bin] / N[bin]) for the owning tracer bin,
        # so cov_ref * g g^T scales each block by N_ref/N (rho intact). Only bins with a
        # real emulator are scaled; null-checkpoint fallback bins (e.g. Lya_QSO in base)
        # keep g=1 so their block stays at the fixed reference value, matching both the
        # normal emulator path (_fill_emulator_blocks copies ref_cov unscaled) and scaling
        # mode's default (Lya QSO held fixed) -- they are not meant to be design dependent.
        cov_size = len(self.desi_data)
        g = torch.ones(batch_shape + (cov_size,), device=self.device, dtype=torch.float64)
        for tracer_bin in self._emulators:
            desi_name = self._EMULATOR_TRACER_TO_DESI.get(tracer_bin)
            rows = self.desi_data.index[self.desi_data["tracer"] == desi_name].tolist()
            if not rows:
                continue
            factor = torch.sqrt(n_ref[tracer_bin] / n_tracers[tracer_bin])
            for r in rows:
                g[..., r] = factor
        covariance = cov_ref * g.unsqueeze(-1) * g.unsqueeze(-2)
        return self._maybe_apply_desi_syst(covariance)

    def _fill_emulator_blocks(self, n_tracers, Om, Ok, w0, wa, hrdrag_phys, batch_shape):
        """Assemble the block-diagonal emulator covariance for given N_tracers and cosmology.

        Shared by the normal emulator path and the 1/sqrt(N) diagnostic (which calls it with
        N pinned to the nominal design). Returns a (``batch_shape``, n_data, n_data) tensor.
        """
        cov_size = len(self.desi_data)
        covariance = torch.zeros(
            batch_shape + (cov_size, cov_size), device=self.device, dtype=torch.float64
        )

        # Fixed reference covariance, used as fallback for tracer bins whose
        # emulator checkpoint was null (e.g. Lya_QSO in the base config).
        ref_cov = torch.as_tensor(self.ref_cov, device=self.device, dtype=torch.float64)

        def _to_broadcastable(value):
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device, dtype=torch.float64)
            return torch.broadcast_to(value.to(torch.float32), batch_shape)

        def _build_emulator_input(tracer_bin):
            # Build emulator inputs in the exact feature order expected by the
            # checkpoint (e.g. current BAO models use ['N_tracers', 'Om', 'hrdrag']).
            emu = self._emulators[tracer_bin]
            self._warn_n_extrapolation(tracer_bin, emu, n_tracers[tracer_bin])
            feature_values = {
                "N_tracers": n_tracers[tracer_bin],
                "Om": Om,
                "Omega_m": Om,  # alias safety
                "Ok": Ok,
                "w0": w0,
                "wa": wa,
                "hrdrag": hrdrag_phys,
            }

            expanded = []
            for name in emu["param_names"]:
                if name not in feature_values:
                    raise ValueError(
                        f"Unsupported emulator input '{name}' in tracer '{tracer_bin}'. "
                        f"Available features: {sorted(feature_values.keys())}"
                    )
                expanded.append(_to_broadcastable(feature_values[name]))
            return torch.stack(expanded, dim=-1)

        def _predict_sigmas(tracer_bin):
            """Return dict {target_name: tensor} of emulator sigma predictions."""
            emu_input = _build_emulator_input(tracer_bin)
            pred = self._emulator_predict(
                self._emulators[tracer_bin], emu_input).to(torch.float64)
            return {
                name: pred[..., k]
                for k, name in enumerate(self._emulators[tracer_bin]["target_names"])
            }

        # Iterate over every tracer bin declared in the emulator config and fill
        # its covariance block, driven by the actual desi_data row layout.
        for tracer_bin in self._emulator_checkpoints:
            desi_name = self._EMULATOR_TRACER_TO_DESI.get(tracer_bin)
            if desi_name is None:
                raise ValueError(
                    f"No desi_data mapping for emulator tracer bin '{tracer_bin}'. "
                    f"Known: {sorted(self._EMULATOR_TRACER_TO_DESI.keys())}"
                )
            rows = self.desi_data.index[self.desi_data["tracer"] == desi_name].tolist()
            if not rows:
                # Tracer not present in this dataset's data vector — skip.
                continue
            quantities = self.desi_data.loc[rows, "quantity"].tolist()

            # Null checkpoint -> fixed reference covariance within-tracer block.
            if tracer_bin in self._emulator_fallback_bins:
                for i in rows:
                    for j in rows:
                        covariance[..., i, j] = ref_cov[i, j]
                continue

            sig = _predict_sigmas(tracer_bin)
            emu_targets = set(self._emulators[tracer_bin]["target_names"])

            if quantities == ["DV_over_rs"]:
                if "sigma_DV_over_rd" not in sig:
                    raise ValueError(
                        f"Tracer '{tracer_bin}' is isotropic (DV only) but its "
                        f"emulator has no 'sigma_DV_over_rd' target."
                    )
                idx = rows[0]
                covariance[..., idx, idx] = sig["sigma_DV_over_rd"] ** 2
            elif set(quantities) == {"DM_over_rs", "DH_over_rs"}:
                dm_row = rows[quantities.index("DM_over_rs")]
                dh_row = rows[quantities.index("DH_over_rs")]

                required = {"sigma_DH_over_rd", "sigma_DM_over_rd", "rho_DH_DM"}
                if not required.issubset(emu_targets):
                    raise ValueError(
                        f"Tracer '{tracer_bin}' anisotropic layout requires emulator "
                        f"targets {sorted(required)}; got {sorted(emu_targets)}."
                    )
                from desilike_emulator.util import cov_block_from_marginals

                c11, c12, c22 = cov_block_from_marginals(
                    sig["sigma_DH_over_rd"],
                    sig["sigma_DM_over_rd"],
                    sig["rho_DH_DM"],
                )
                covariance[..., dh_row, dh_row] = c11
                covariance[..., dm_row, dm_row] = c22
                covariance[..., dh_row, dm_row] = c12
                covariance[..., dm_row, dh_row] = c12
            else:
                raise ValueError(
                    f"Unexpected quantity layout {quantities} for tracer "
                    f"'{tracer_bin}' ({desi_name}); expected ['DV_over_rs'] or "
                    f"{{'DM_over_rs', 'DH_over_rs'}}."
                )

        return covariance

    def _sqrtn_nominal_n_tracers(self):
        """Per-bin N_tracers at the nominal DESI design, cached.

        Reference anchor for the 1/sqrt(N) diagnostic; design-independent. The nominal
        design is ``self.nominal_design`` (class shares, e.g. [0.037, 0.265, 0.410, 0.288]),
        the same design the scaling-mode nominal and the emulator wiring check use.
        """
        if self._sqrtn_n_ref_cache is None:
            nd = self.nominal_design.to(torch.float64).view(1, len(self.design_labels))
            passed_ratio = self.calc_passed(nd)
            self._sqrtn_n_ref_cache = {
                bin_name: val.squeeze(0)
                for bin_name, val in self._passed_ratio_to_n_tracers(passed_ratio).items()
            }
        return self._sqrtn_n_ref_cache

    # DESI 2024 fiducial cosmology used as the frozen reference point for the
    # 'fiducial' 1/sqrt(N) diagnostic. hrdrag is the PHYSICAL h*r_d (~99.08),
    # i.e. bedcosmo raw 0.9908 x 100 (multiplier-independent at this point).
    _SQRTN_FIDUCIAL_OM = 0.3152
    _SQRTN_FIDUCIAL_HRDRAG_PHYS = 99.08

    def _sqrtn_fiducial_cov_ref(self):
        """Emulator covariance at nominal N and the DESI fiducial cosmology, cached once.

        Used as the frozen reference for ``emulator_sqrtn_ref='fiducial'`` (cosmology-independent
        1/sqrt(N) scaling). Om/hrdrag are the explicit DESI fiducial values (FIDUCIAL_PARAMS only
        carries Ok/w0/wa); hrdrag is passed as the already-physical h*r_d.
        """
        if self._sqrtn_ref_cov_cache is None:
            n_ref = self._sqrtn_nominal_n_tracers()
            Om = torch.tensor(self._SQRTN_FIDUCIAL_OM, device=self.device)
            Ok = torch.tensor(FIDUCIAL_PARAMS["Ok"], device=self.device)
            w0 = torch.tensor(FIDUCIAL_PARAMS["w0"], device=self.device)
            wa = torch.tensor(FIDUCIAL_PARAMS["wa"], device=self.device)
            hrdrag_phys = torch.tensor(self._SQRTN_FIDUCIAL_HRDRAG_PHYS, device=self.device)
            self._sqrtn_ref_cov_cache = self._fill_emulator_blocks(
                n_ref, Om, Ok, w0, wa, hrdrag_phys, batch_shape=()
            )
        return self._sqrtn_ref_cov_cache

    def _maybe_apply_desi_syst(self, covariance):
        # Optionally inflate σ_stat -> σ_tot with DESI's systematic budget. The
        # inflation is a diagonal per-quantity σ scaling R, so on the covariance it
        # acts as cov[i,j] -> R_i·R_j·cov[i,j] (ρ unchanged; cross-tracer zeros stay
        # zero). Fallback bins / unknown tracers scale by 1.0 (no-op).
        if self.apply_desi_syst:
            f = self._desi_syst_factors()
            covariance = covariance * f.unsqueeze(-1) * f.unsqueeze(-2)

        return covariance

    def _desi_syst_factors(self):
        """Per-data-row σ inflation factors R from DESI's systematic budget.

        Uses desilike_emulator's apply_desi_syst to map each
        emulator tracer bin's frozen DESI_SYST_INFLATION factors onto the
        desi_data row layout. Cached after first build. Rows not covered by an
        emulator bin (or absent from the table) get 1.0.
        """
        if self._desi_syst_factors_cache is not None:
            return self._desi_syst_factors_cache

        from desilike_emulator.bao.desi_syst import apply_desi_syst

        factors = torch.ones(len(self.desi_data), device=self.device, dtype=torch.float64)
        unit = {"DH_over_rs": 1.0, "DM_over_rs": 1.0, "DV_over_rs": 1.0}
        for tracer_bin, desi_name in self._EMULATOR_TRACER_TO_DESI.items():
            rows = self.desi_data.index[self.desi_data["tracer"] == desi_name].tolist()
            if not rows:
                continue
            R = apply_desi_syst(unit, tracer_bin)  # {quantity: factor}, no-op if unknown
            for i in rows:
                q = self.desi_data.loc[i, "quantity"]
                factors[i] = float(R.get(q, 1.0))
        self._desi_syst_factors_cache = factors
        return factors

    def _stabilize_covariance(self, covariance_matrix):
        """Make covariance numerically SPD for MultivariateNormal sampling."""
        cov = 0.5 * (covariance_matrix + covariance_matrix.transpose(-1, -2))
        if not torch.isfinite(cov).all():
            # Map a non-finite variance to a LARGE finite one, not zero: +inf
            # variance means "no information", so zeroing it (the old behavior)
            # inverts the physics into "infinite precision". With the upstream
            # sigma cap in _emulator_predict this branch should never trigger
            # for the emulator path; kept as defense in depth. Ceiling^2 matches
            # the sigma cap's variance scale.
            big = self._sigma_ceiling ** 2
            cov = torch.nan_to_num(cov, nan=0.0, posinf=big, neginf=-big)
            cov = 0.5 * (cov + cov.transpose(-1, -2))
        n_dim = cov.shape[-1]
        eye = torch.eye(n_dim, device=cov.device, dtype=cov.dtype)
        eye = eye.view(*([1] * (cov.ndim - 2)), n_dim, n_dim)

        # Ensure strictly positive diagonal entries.
        diag = torch.diagonal(cov, dim1=-2, dim2=-1)
        cov = cov.clone()
        idx = torch.arange(n_dim, device=cov.device)
        cov[..., idx, idx] = torch.clamp(diag, min=1e-12)

        # Try adaptive diagonal jitter first.
        mean_diag = torch.diagonal(cov, dim1=-2, dim2=-1).abs().mean(dim=-1)
        jitter = torch.clamp(mean_diag, min=1.0) * 1e-10
        for _ in range(6):
            info = torch.linalg.cholesky_ex(cov).info
            if torch.all(info == 0):
                return cov
            failing = (info > 0).to(cov.dtype)
            cov = cov + failing[..., None, None] * jitter[..., None, None] * eye
            jitter = jitter * 10.0

        # Final robust fallback: project to SPD by clipping eigenvalues.
        try:
            evals, evecs = torch.linalg.eigh(cov)
        except RuntimeError:
            cov_cpu = cov.detach().cpu()
            evals, evecs = torch.linalg.eigh(cov_cpu)
            evals = evals.to(device=cov.device, dtype=cov.dtype)
            evecs = evecs.to(device=cov.device, dtype=cov.dtype)
        evals = torch.clamp(evals, min=1e-10)
        cov = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        return cov

    # Tracer bins with no n(z) tables: their mean stays at desi_data.csv's z.
    _FIXED_Z_EFF_TRACERS = ("Lya QSO",)
    _MEAN_QUANTITIES = ("DH_over_rs", "DM_over_rs", "DV_over_rs")

    def _init_mean_z_eff_rows(self):
        """Per-quantity row layout for ``_mean_z_eff``, built once.

        For each mean quantity: the desi_data.csv ``z`` of its rows as a tensor on
        ``self.device``, and each row's emulator tracer bin (``None`` for tracers in
        ``_FIXED_Z_EFF_TRACERS``). Row order matches ``DH_idx``/``DM_idx``/``DV_idx``.
        """
        desi_to_bin = {v: k for k, v in self._EMULATOR_TRACER_TO_DESI.items()}
        self._mean_z_eff_rows = {}
        for quantity in self._MEAN_QUANTITIES:
            rows = self.desi_data[self.desi_data["quantity"] == quantity]
            bins = [None if t in self._FIXED_Z_EFF_TRACERS else desi_to_bin[t]
                    for t in rows["tracer"].to_list()]
            z_fixed = torch.tensor(rows["z"].to_list(), device=self.device)
            self._mean_z_eff_rows[quantity] = (z_fixed, bins)

    def _mean_z_eff(self, passed_ratio):
        """Redshifts at which the BAO mean is evaluated, keyed by quantity.

        Without ``vary_z_eff``: desi_data.csv's fixed ``z`` per row, shape
        ``(n_rows,)``. With it: ``util.effective_redshift(bin, N)`` at the design's
        passed ``N_tracers``, computed once per tracer bin and shared by the
        D_H/D_M/D_V rows, shape ``passed_ratio.shape[:-1] + (n_rows,)``; tracers in
        ``_FIXED_Z_EFF_TRACERS`` keep their fixed z. ``D_H_func``/``D_M_func``/
        ``D_V_func`` accept both shapes.
        """
        if not self.vary_z_eff:
            return {q: z for q, (z, _) in self._mean_z_eff_rows.items()}
        if self._effective_redshift is None:  # flag switched on after construction
            from desilike_emulator.util import effective_redshift
            self._effective_redshift = effective_redshift

        n_tracers = self._passed_ratio_to_n_tracers(passed_ratio)
        batch_shape = passed_ratio.shape[:-1]
        z_bin = {}
        out = {}
        for quantity, (z_fixed, bins) in self._mean_z_eff_rows.items():
            cols = []
            for i, tracer_bin in enumerate(bins):
                if tracer_bin is None:
                    cols.append(z_fixed[i].to(passed_ratio.dtype).expand(batch_shape))
                    continue
                if tracer_bin not in z_bin:
                    z_bin[tracer_bin] = self._effective_redshift(
                        tracer_bin, n_tracers[tracer_bin], "dr1")
                cols.append(z_bin[tracer_bin])
            out[quantity] = torch.stack(cols, dim=-1)
        return out

    _SHAPEFIT_QUANTITIES = ("mean", "covar")
    # DESI template fiducial (AbacusSummit c000): desilike_emulator shapefit/core.py
    # FIDUCIAL_SAMPLE, the cosmology the mean labels are measured against. Copied rather
    # than imported because that module needs desilike, which this env does not install.
    _SHAPEFIT_FIDUCIAL = {
        "omega_cdm": 0.1200,
        "omega_b": 0.02237,
        "h": 0.6736,
        "ln10A_s": 3.036394,
        "n_s": 0.9649,
    }
    # Eigenvalue floor for each tracer's 4x4 ShapeFit correlation matrix. The covar
    # emulator predicts the six rho independently, and the tanh decode only keeps each
    # 2x2 block PD, so the 4x4 needs a nearest-PD projection (desilike-emulator
    # shapefit/README "Target contract"). Blocks already above the floor are unchanged.
    _SHAPEFIT_CORR_EIG_FLOOR = 1e-6

    def _init_shapefit(self, emu_dir):
        """Tracer bins, N_tracers weights, both emulator sets and the central data vector.

        Bins are desilike-emulator's ``tracers_for("shapefit")``: the 0.8-1.1 bin is
        LRG3 alone, not BAO's LRG3+ELG1 (ELG1 is excluded from full shape, DESI 2024 V
        Sec 2). A bin's passed count sums its ``components`` (default: the bin itself)
        in desi_tracers.csv, as ``util.ntracers`` does:

            N_b = sum_c design[class_c] * observed_c / observed_{class_c} * efficiency_c
                  * nominal_total_obs

        which is ``calc_passed``'s split and efficiency per sub-sample, stored as one
        (n_bins, n_design_labels) matrix so ``_shapefit_n_tracers`` is a matmul.
        """
        from desilike_emulator.util import get_tracer_config, tracers_for

        self.shapefit_bins = tracers_for("shapefit")
        tracers = self.desi_tracers.set_index("tracer")
        class_observed = tracers.groupby("class")["observed"].sum()
        weights = torch.zeros(len(self.shapefit_bins), len(self.design_labels), dtype=torch.float64)
        for i, tracer_bin in enumerate(self.shapefit_bins):
            cfg = get_tracer_config(tracer_bin, analysis="shapefit", data_release=self.dataset)
            for comp in cfg.get("components", [tracer_bin]):
                row = tracers.loc[comp]
                j = list(self.design_labels).index(row["class"])
                weights[i, j] += row["observed"] / class_observed[row["class"]] * row["efficiency"]
        self._shapefit_n_weights = (weights * self.nominal_total_obs).to(self.device)

        self._shapefit_emulators = {}
        for quantity in self._SHAPEFIT_QUANTITIES:
            if emu_dir is not None and os.path.isdir(emu_dir):
                paths = {tb: os.path.join(emu_dir, quantity, f"{tb}.pt") for tb in self.shapefit_bins}
            else:
                paths = self.resolve_emulator_checkpoints(
                    "shapefit", self.cosmo_model, self.dataset, quantity)
            missing = [tb for tb in self.shapefit_bins if paths.get(tb) is None]
            if missing or set(paths) != set(self.shapefit_bins):
                raise ValueError(
                    f"shapefit {quantity} emulators must cover exactly {self.shapefit_bins} "
                    f"(no null fallback: there is no DESI ShapeFit covariance to fall back to); "
                    f"got {paths}.")
            self._shapefit_emulators[quantity] = {
                tb: self._load_emulator(paths[tb]) for tb in self.shapefit_bins}

        # Target contract: mean -> the physical parameters, covar -> their sigma_* then the
        # upper-triangle rho_*, row-major (desilike_emulator shapefit/core.py TARGET_NAMES).
        self.shapefit_quantities = self._shapefit_emulators["mean"][self.shapefit_bins[0]]["target_names"]
        names = self.shapefit_quantities
        covar_names = [f"sigma_{n}" for n in names] + [
            f"rho_{names[i]}_{names[j]}"
            for i in range(len(names)) for j in range(i + 1, len(names))]
        for tracer_bin in self.shapefit_bins:
            got_mean = self._shapefit_emulators["mean"][tracer_bin]["target_names"]
            got_covar = self._shapefit_emulators["covar"][tracer_bin]["target_names"]
            if got_mean != names or got_covar != covar_names:
                raise ValueError(
                    f"[{tracer_bin}] shapefit emulator targets mean={got_mean}, covar={got_covar}; "
                    f"expected mean={names}, covar={covar_names}.")

        if self.global_rank == 0:
            self._print_emulator_summary({
                f"{quantity}/{tb}": emu
                for quantity, by_bin in self._shapefit_emulators.items()
                for tb, emu in by_bin.items()})

        # The flow's nominal data: the mean emulator at the fiducial and the nominal design.
        # Evaluated as a batch of one: the emulators' (1, in_dim) x_mu adds that dim anyway.
        n_nominal = self._shapefit_n_tracers(self.nominal_design.view(1, -1))
        self.central_val = torch.cat([
            self._shapefit_predict("mean", tb, n_nominal[..., i], self._shapefit_fiducial_parameters())
            for i, tb in enumerate(self.shapefit_bins)], dim=-1)[0]

    def _shapefit_n_tracers(self, tracer_ratio):
        """Passed N_tracers per shapefit bin, shape (..., n_bins) in ``shapefit_bins`` order."""
        return tracer_ratio.to(torch.float64) @ self._shapefit_n_weights.T

    def _shapefit_fiducial_parameters(self):
        return {
            name: torch.tensor([value], device=self.device, dtype=torch.float64)
            for name, value in self._SHAPEFIT_FIDUCIAL.items()}

    def _shapefit_predict(self, quantity, tracer_bin, n_tracers, parameters):
        """One shapefit emulator's physical targets at (N_tracers, cosmology).

        ``parameters`` holds (..., 1) tensors (``sample_parameters`` output); inputs are
        stacked in the checkpoint's ``param_names`` order and broadcast against N.
        """
        emu = self._shapefit_emulators[quantity][tracer_bin]
        self._warn_n_extrapolation(tracer_bin, emu, n_tracers)
        features = {"N_tracers": n_tracers, **{k: v.squeeze(-1) for k, v in parameters.items()}}
        missing = [p for p in emu["param_names"] if p not in features]
        if missing:
            raise ValueError(
                f"[{quantity}/{tracer_bin}] emulator needs inputs {missing}, which cosmo_model "
                f"'{self.cosmo_model}' does not sample (have {sorted(features)}).")
        x = torch.stack(torch.broadcast_tensors(
            *[features[p].to(torch.float64) for p in emu["param_names"]]), dim=-1)
        return self._emulator_predict(emu, x).to(torch.float64)

    def _shapefit_cov_block(self, sigma, rho):
        """4x4 covariance from sigma (..., 4) and the row-major upper-triangle rho (..., 6).

        A correlation matrix with an eigenvalue below ``_SHAPEFIT_CORR_EIG_FLOOR`` is
        projected to its nearest PD neighbour by flooring its eigenvalues there and
        restoring the unit diagonal; the others are used as predicted. Only those blocks
        go through eigh, in chunks: cuSOLVER's batched eigh raises INVALID_VALUE at
        batch >= 32768 (eval's particle batches are 50000), and they are ~5e-5 of draws.
        """
        n = sigma.shape[-1]
        i, j = torch.triu_indices(n, n, offset=1, device=sigma.device)
        eye = torch.eye(n, dtype=sigma.dtype, device=sigma.device)
        corr = eye.expand(sigma.shape + (n,)).clone()
        corr[..., i, j] = rho
        corr[..., j, i] = rho
        # Cholesky of corr - floor*I succeeds iff every eigenvalue exceeds the floor.
        below = torch.linalg.cholesky_ex(corr - self._SHAPEFIT_CORR_EIG_FLOOR * eye).info != 0
        if below.any():
            projected = []
            for chunk in torch.split(corr[below], 16384):
                evals, evecs = torch.linalg.eigh(chunk)
                evals = evals.clamp(min=self._SHAPEFIT_CORR_EIG_FLOOR)
                chunk = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
                d = torch.diagonal(chunk, dim1=-2, dim2=-1).sqrt()
                projected.append(chunk / (d.unsqueeze(-1) * d.unsqueeze(-2)))
            corr[below] = torch.cat(projected)
        cov = corr * sigma.unsqueeze(-1) * sigma.unsqueeze(-2)
        return 0.5 * (cov + cov.transpose(-1, -2))

    def _shapefit_likelihood(self, n_tracers, parameters):
        """ShapeFit Gaussian likelihood: mean (..., 4 * n_bins) and block-diagonal covariance.

        Per bin, the mean emulator gives [qiso, qap, f_sigmar, m] and the covar emulator its
        4x4 covariance, both at the bin's N_tracers and the sampled cosmology. Bins are
        independent (distinct redshift slices), so off-diagonal blocks are zero.
        """
        n_q = len(self.shapefit_quantities)
        means, blocks = [], []
        for i, tracer_bin in enumerate(self.shapefit_bins):
            n = n_tracers[..., i]
            means.append(self._shapefit_predict("mean", tracer_bin, n, parameters))
            pred = self._shapefit_predict("covar", tracer_bin, n, parameters)
            blocks.append(self._shapefit_cov_block(pred[..., :n_q], pred[..., n_q:]))
        mean = torch.cat(means, dim=-1)
        covariance = torch.zeros(
            mean.shape + (mean.shape[-1],), device=mean.device, dtype=torch.float64)
        for i, block in enumerate(blocks):
            covariance[..., i * n_q:(i + 1) * n_q, i * n_q:(i + 1) * n_q] = block
        return mean, covariance

    @profile_method
    def pyro_model(self, tracer_ratio):
        if self.analysis == "shapefit":
            n_tracers = self._shapefit_n_tracers(tracer_ratio)
            with pyro.plate_stack("plate", n_tracers.shape[:-1]):
                parameters = self.sample_parameters(n_tracers.shape[:-1])
                means, covariance_matrix = self._shapefit_likelihood(n_tracers, parameters)
                return pyro.sample(
                    self.observation_labels[0], dist.MultivariateNormal(means, covariance_matrix)
                )

        passed_ratio = self.calc_passed(tracer_ratio)
        with pyro.plate_stack("plate", passed_ratio.shape[:-1]):
            parameters = self.sample_parameters(passed_ratio.shape[:-1])

            if self.likelihood_mode == "emulator":
                # Means: same D_H, D_M, D_V computation as scaling mode
                means = torch.zeros(
                    passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device
                )
                z_eff = self._mean_z_eff(passed_ratio)
                means[:, :, self.DH_idx] = self.D_H_func(z_eff["DH_over_rs"], **parameters)
                if self.include_D_M:
                    means[:, :, self.DM_idx] = self.D_M_func(z_eff["DM_over_rs"], **parameters)
                if self.include_D_V:
                    means[:, :, self.DV_idx] = self.D_V_func(z_eff["DV_over_rs"], **parameters)
                means = means.to(self.device)

                # Covariance: from emulators (reuse the passed_ratio already computed above)
                covariance_matrix = self._build_emulator_covariance(passed_ratio, parameters)
                covariance_matrix = self._stabilize_covariance(covariance_matrix)
                return pyro.sample(
                    self.observation_labels[0], dist.MultivariateNormal(means, covariance_matrix)
                )
            else:
                # Existing scaling code
                means = torch.zeros(
                    passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device
                )
                rescaled_sigmas = torch.zeros(
                    passed_ratio.shape[:-1] + (self.sigmas.shape[-1],), device=self.device
                )
                z_eff = torch.tensor(
                    self.desi_data[self.desi_data["quantity"] == "DH_over_rs"]["z"].to_list(),
                    device=self.device,
                )
                means[:, :, self.DH_idx] = self.D_H_func(z_eff, **parameters)
                rescaled_sigmas[:, :, self.DH_idx] = self.sigmas[
                    self.DH_idx
                ] * self.sigma_scaling_factor(passed_ratio, self.DH_idx)
                if self.include_D_M:
                    z_eff = torch.tensor(
                        self.desi_data[self.desi_data["quantity"] == "DM_over_rs"]["z"].to_list(),
                        device=self.device,
                    )
                    means[:, :, self.DM_idx] = self.D_M_func(z_eff, **parameters)
                    rescaled_sigmas[:, :, self.DM_idx] = self.sigmas[
                        self.DM_idx
                    ] * self.sigma_scaling_factor(passed_ratio, self.DM_idx)

                if self.include_D_V:
                    z_eff = torch.tensor(
                        self.desi_data[self.desi_data["quantity"] == "DV_over_rs"]["z"].to_list(),
                        device=self.device,
                    )
                    means[:, :, self.DV_idx] = self.D_V_func(z_eff, **parameters)
                    rescaled_sigmas[:, :, self.DV_idx] = self.sigmas[
                        self.DV_idx
                    ] * self.sigma_scaling_factor(passed_ratio, self.DV_idx)

                # extract correlation matrix from DESI covariance matrix
                if self.include_D_V and self.include_D_M:
                    means = means.to(self.device)
                    # convert correlation matrix to covariance matrix using rescaled sigmas
                    covariance_matrix = self.corr_matrix * (
                        rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2)
                    )
                else:
                    # only use D_H values for mean and covariance matrix
                    means = means[:, :, self.DH_idx].to(self.device)
                    covariance_matrix = self.corr_matrix[self.DH_idx, self.DH_idx] * (
                        rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-1)
                        * rescaled_sigmas[:, :, self.DH_idx].unsqueeze(-2)
                    )

                return pyro.sample(
                    self.observation_labels[0], dist.MultivariateNormal(means, covariance_matrix)
                )

    def unnorm_lfunc(self, params, features, designs):
        parameters = {}
        for key in params.names:
            parameters[key] = torch.tensor(
                np.asarray(getattr(params, key)), device=self.device, dtype=torch.float64
            )
        likelihood = jnp.asarray(1.0, dtype=jnp.float64)
        passed_ratio = self.calc_passed(designs)
        for i in range(len(self.tracer_bins)):
            D_H_mean = self.D_H_func(**parameters)
            D_H_diff = jnp.asarray(getattr(features, features.names[i])) - jnp.asarray(
                D_H_mean.cpu().numpy()
            )
            D_H_sigma = jnp.asarray(self.sigmas[self.DH_idx].cpu().numpy()[i]) * jnp.sqrt(
                jnp.asarray(self.nominal_passed_ratio[i].cpu().numpy())
                / jnp.asarray(passed_ratio[i])
            )
            likelihood = jnp.exp(-0.5 * (D_H_diff / D_H_sigma) ** 2) * likelihood

            if self.include_D_M:
                D_M_mean = self.D_M_func(**parameters)
                D_M_diff = jnp.asarray(
                    getattr(features, features.names[i + len(self.tracer_bins)])
                ) - jnp.asarray(D_M_mean.cpu().numpy())
                D_M_sigma = jnp.asarray(self.sigmas[self.DM_idx].cpu().numpy()[i]) * jnp.sqrt(
                    jnp.asarray(self.nominal_passed_ratio[i].cpu().numpy())
                    / jnp.asarray(passed_ratio[i])
                )
                likelihood = jnp.exp(-0.5 * (D_M_diff / D_M_sigma) ** 2) * likelihood

        if (
            getattr(params, "_stack_offset", 0) == 0
            and getattr(features, "_stack_offset", 0) == 0
            and getattr(designs, "_stack_offset", 0) == 0
        ):
            param_shape = tuple(params.shape)
            likelihood = jnp.asarray(likelihood, dtype=jnp.float64)
            if likelihood.shape != param_shape and likelihood.size == int(np.prod(param_shape)):
                likelihood = likelihood.reshape(param_shape)

        return likelihood
