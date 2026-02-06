import sys
import os
import contextlib
import io
import json
import gc
import pickle
import re
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from datetime import datetime

import numpy as np
import torch
import getdist
import argparse
from bedcosmo.plotting import RunPlotter
import traceback
from bedcosmo.pyro_oed_src import nf_loss, LikelihoodDataset
from matplotlib.patches import Rectangle
from bedcosmo.util import (
    auto_seed, init_experiment, init_nf, load_model, get_runs_data,
    get_experiment_config_path, profile_method, parse_float_or_list,
    get_rng_state,
)
import mlflow
import inspect
import yaml
from bedcosmo.brute_force import brute_force_from_experiment

class Evaluator:
    def __init__(
            self, run_id, guide_samples=1000, design_chunk_size=None, seed=1, cosmo_exp='num_tracers', 
            levels=[0.68, 0.95], global_rank=0, eig_file_path=None, n_evals=10, n_particles=1000, 
            param_space='physical', display_run=False, verbose=False, device="cuda:0", profile=False, 
            sort=True, include_nominal=False, batch_size=1, particle_batch_size=None, design_args_path=None,
            brute_force=False, brute_force_param_points=75, brute_force_feature_points=35
            ):
        self.cosmo_exp = cosmo_exp
        
        # Store seed parameter for later use (may be overridden by eig_file_path)
        self.seed = seed
        
        # Set MLflow tracking URI based on cosmo_exp
        mlflow.set_tracking_uri("file:" + os.environ["SCRATCH"] + f"/bedcosmo/{self.cosmo_exp}/mlruns")
        
        self.run_id = run_id
        print(f"\nStarting evaluation for run {self.run_id}")
        print(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")
        run_data_list, _, _ = get_runs_data(run_ids=self.run_id, cosmo_exp=self.cosmo_exp)
        if not run_data_list:  # Check for empty list instead of None
            print(f"Run {self.run_id} not found.")
            raise ValueError(f"Run {self.run_id} not found in experiment {self.cosmo_exp}. Please check that the run exists and cosmo_exp is correct.")
        run_data = run_data_list[0]
        self.run_obj = run_data['run_obj']
        self.run_args = run_data['params']
        self.total_steps = self.run_args["total_steps"]
        self.exp_id = run_data['exp_id']
        self.storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{self.cosmo_exp}"
        self.save_path = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts"
        
        self.n_evals = n_evals
        self.n_particles = n_particles
        self.particle_batch_size = particle_batch_size
        self.seed = seed
        # Load eig_file_path early to get seed for reproducibility
        self.eig_file_path = eig_file_path
        self._init_eig_data()
        
        # Set seed before any operations that might use randomness
        auto_seed(self.seed)
        self.guide_samples = guide_samples
        # Normalize levels to always be a list
        if isinstance(levels, (int, float)):
            self.levels = [levels]
        else:
            self.levels = levels
        self.device = device
        self.global_rank = global_rank
        self.display_run = display_run
        self.verbose = verbose
        self.profile = profile
        self.design_chunk_size = design_chunk_size  # Number of designs per chunk (None = use all)
        self.sort = sort
        self.include_nominal = include_nominal
        self.batch_size = batch_size  # Batch size for sample_posterior to reduce memory usage
        self.brute_force = brute_force
        self.brute_force_param_points = brute_force_param_points
        self.brute_force_feature_points = brute_force_feature_points
        self._brute_force_experiment = None
        
        # Load design_args from file
        if design_args_path is not None:
            if not os.path.exists(design_args_path):
                raise FileNotFoundError(f"Design args file not found: {design_args_path}")
            with open(design_args_path, 'r') as f:
                self.design_args = yaml.safe_load(f)
            print(f"Loaded design_args from: {design_args_path}")
        else:
            # Will default to the run's artifacts design_args.yaml
            self.design_args = None

        # Initialize experiment - it will handle input_design and generate designs accordingly 
        # (single design, multiple designs, or grid)
        self.experiment = init_experiment(
            self.run_obj, self.run_args, device=self.device, 
            design_args=self.design_args, global_rank=self.global_rank
        )
        if self.eig_file_path is not None and 'input_designs' in self.eig_data:
            self.input_designs = torch.tensor(self.eig_data['input_designs'], device=self.device, dtype=torch.float64)
        else:
            self.input_designs = self.experiment.designs
            # save to eig_data object
            self.eig_data['input_designs'] = self.input_designs.cpu().numpy().tolist()
            self.eig_data['nominal_design'] = self.experiment.nominal_design.cpu().numpy().tolist()
            
        
        # Initialize plotter for saving figures
        self.plotter = RunPlotter(run_id=self.run_id, cosmo_exp=self.cosmo_exp)
        
        # Cache for EIG calculations to avoid redundant computations
        self._eig_cache = {}

        self.param_space = param_space
        if self.param_space == 'physical':
            self.nominal_transform_output = False
            self.nf_transform_output = True
        elif self.param_space == 'unconstrained':
            self.nominal_transform_output = True
            self.nf_transform_output = False
        else:
            raise ValueError(f"Invalid parameter space: {self.param_space}")
        
        # Initialize timing mechanism
        self.session_start_time = None

    def _compute_brute_force_eig(self, step_key):
        print("Running brute-force EIG calculation with ExperimentDesigner...")
        # Brute-force path is pinned to CPU for stability / compatibility.
        # Keep a separate experiment instance so the main evaluator can still run on GPU.
        if self._brute_force_experiment is None:
            print("Initializing CPU experiment for brute-force EIG...")
            self._brute_force_experiment = init_experiment(
                self.run_obj,
                self.run_args,
                device="cpu",
                design_args=self.design_args,
                global_rank=self.global_rank,
            )

        result = brute_force_from_experiment(
            experiment=self._brute_force_experiment,
            param_points=self.brute_force_param_points,
            feature_points=self.brute_force_feature_points,
        )
        eig = np.asarray(result["eig"], dtype=float)
        best_design = result["best_design"]

        step_dict = self.eig_data.setdefault(step_key, {})
        variable_data = step_dict.setdefault("variable", {})
        nominal_data = step_dict.setdefault("nominal", {})

        eig_flat = eig.reshape(-1)
        max_idx = int(np.argmax(eig_flat))
        optimal_eig = float(eig_flat[max_idx])
        variable_data["brute_force"] = {
            "eigs_avg": eig_flat.tolist(),
            "eigs_std": np.zeros_like(eig_flat).tolist(),
            "optimal_eig": optimal_eig,
            "optimal_eig_std": 0.0,
            "optimal_design": best_design,
            "design_grid_shape": list(result["design_grid_shape"]),
            "feature_grid_shape": list(result["feature_grid_shape"]),
            "parameter_grid_shape": list(result["parameter_grid_shape"]),
        }

        nominal_design = np.asarray(self.experiment.nominal_design.detach().cpu().numpy(), dtype=float).reshape(1, -1)
        input_designs_np = np.asarray(self.input_designs.detach().cpu().numpy(), dtype=float)
        nominal_idx = int(np.argmin(np.linalg.norm(input_designs_np - nominal_design, axis=1)))
        nominal_data["brute_force"] = {
            "eigs_avg": float(eig_flat[nominal_idx]),
            "eigs_std": 0.0,
        }
        print(
            f"Brute-force EIG complete: nominal={nominal_data['brute_force']['eigs_avg']:.4f}, "
            f"optimal={optimal_eig:.4f}"
        )

    def _update_runtime(self):
        """Update and print session runtime."""
        if self.session_start_time is None:
            # First call - initialize session start time
            self.session_start_time = time.time()
            return
        
        # Calculate session runtime
        session_runtime = time.time() - self.session_start_time
        
        # Print current runtime
        hours = int(session_runtime // 3600)
        minutes = int((session_runtime % 3600) // 60)
        seconds = int(session_runtime % 60)
        print(f"Evaluation runtime: {hours}h {minutes}m {seconds}s")

    def _init_eig_data(self):
        if self.eig_file_path is not None:
            with open(self.eig_file_path, 'r') as f:
                self.loaded_eig_data = json.load(f)
            self.eig_data = self.loaded_eig_data
            # Override seed from loaded data to ensure reproducibility
            self.seed = self.eig_data.get('seed', self.seed)
            self.n_particles = self.eig_data.get('n_particles', self.n_particles)
            self.particle_batch_size = self.eig_data.get('particle_batch_size', self.particle_batch_size)
            self.n_evals = self.eig_data.get('n_evals', self.n_evals)
            self.restore_path = self.eig_data.get('rng_state_path', None)
            # Ensure status field exists (default to 'incomplete' for old files without status)
            if 'status' not in self.eig_data:
                self.eig_data['status'] = 'incomplete'
            print(f"Loaded EIG data from {self.eig_file_path} using seed {self.seed} from file")
        else:
            self.eig_data = {}
            self.loaded_eig_data = {}
            self.eig_data['seed'] = self.seed
            self.eig_data['n_evals'] = int(self.n_evals)
            self.eig_data['n_particles'] = int(self.n_particles)
            if self.particle_batch_size is not None:
                self.eig_data['particle_batch_size'] = int(self.particle_batch_size)
            self.eig_data['status'] = 'incomplete'  # Mark as incomplete when first created
            self.restore_path = None

    def _save_rng_state(self):
        """Save RNG state for resuming partial evaluations."""
        rng_state = get_rng_state()
        rng_state_path = f"{self.save_path}/eval_rng_state_{getattr(self, 'timestamp', None)}.pkl"
        if 'rng_state_path' not in self.eig_data and os.path.exists(rng_state_path):
            self.eig_data['rng_state_path'] = rng_state_path

        try:
            with open(rng_state_path, 'wb') as f:
                pickle.dump(rng_state, f)
        except Exception as e:
            print(f"  Warning: Could not save RNG state to {rng_state_path}: {e}")

    @profile_method
    def _sample_rank_step(self, rank, step, design, nominal_design=False):
        """
        Generates samples from a single rank.
        
        Args:
            rank (int): Global rank to generate samples from
            step (int): Step to generate samples from
            design: Design tensor
            nominal_design (bool): Whether this is for nominal design
            
        Returns:
            samples: MCSamples object
        """
        # Load model
        flow_model, _ = load_model(
            self.experiment, step, self.run_obj, 
            self.run_args, self.device, global_rank=rank
        )
        flow_model = flow_model.to(self.device)
        
        # Get samples
        if nominal_design:
            context = self.experiment.nominal_context
        else:
            context = torch.cat([design, self.experiment.central_val], dim=-1)
            
        samples = self.experiment.get_guide_samples(
            flow_model, context, 
            num_samples=self.guide_samples, 
            transform_output=self.nf_transform_output
        )
        
        return samples

    @profile_method
    def _eval_step(self, step, nominal_design=False):
        """
        Generates samples given the nominal context (nominal design + central values).
        Args:
            step (int or str): Step to calculate the nominal samples for. Can be 'last' or an integer.
        """
        # Resolve step to actual step number (handles 'last' and other special values)
        _, selected_step = load_model(
            self.experiment, step, self.run_obj, 
            self.run_args, self.device, 
            global_rank=self.global_rank
        )
        
        # Determine which design to use
        if nominal_design:
            design = self.experiment.nominal_design
        else:
            # Check if we have EIG data for this step in the loaded file
            step_key = f'step_{selected_step}'
            if self.eig_file_path is not None and step_key in self.eig_data:
                step_dict = self.eig_data[step_key]
                step_data = step_dict.get('variable', {})
                if 'eigs_avg' in step_data:
                    print(f"  Using pre-calculated EIG data for step {selected_step} from {self.eig_file_path} to find optimal design")
                    eigs = np.array(step_data['eigs_avg'])
                    design = self.input_designs[np.argmax(eigs)]
                else:
                    # Fall back to calculating EIG
                    eigs, _ = self.get_eig(selected_step, nominal_design=False)
                    design = self.input_designs[np.argmax(eigs)]
            else:
                # No cached data, calculate EIG
                eigs, _ = self.get_eig(selected_step, nominal_design=False)
                design = self.input_designs[np.argmax(eigs)]

        # Generate samples
        samples = self._sample_rank_step(self.global_rank, step, design, nominal_design)
        return samples

    
    @profile_method
    def sample_posterior(self, step, levels, num_data_samples=10, global_rank=0, central=True, batch_size=1):
        # Normalize levels to always be a list
        if isinstance(levels, (int, float)):
            levels = [levels]
        
        print(f"Running sample posterior evaluation...")
        posterior_flow, _ = load_model(
            self.experiment, step, 
            self.run_obj, self.run_args, 
            device=self.device, global_rank=global_rank
            )
        posterior_flow = posterior_flow.to(self.device)

        # Get optimal design based on EIGs across ranks
        eigs, _ = self.get_eig(step, nominal_design=False)
        optimal_design = self.input_designs[np.argmax(eigs)]
        
        # Generate all unique parameter pairs
        params = self.experiment.cosmo_params
        pair_names = [f"{params[i]}_{params[j]}" for i in range(len(params)) for j in range(i+1, len(params))]
        design_areas = {'nominal': {p: [] for p in pair_names}, 'optimal': {p: [] for p in pair_names}}
        all_samples = []
        colors = []
        inputs = (('optimal', 'tab:orange', optimal_design), ('nominal', 'tab:blue', self.experiment.nominal_design))
        for design_type, color, design in inputs:
            data_idxs = np.arange(1, num_data_samples) # sample N data points
            
            # Process in batches to reduce memory usage
            all_batch_samples = []
            num_batches = (num_data_samples + batch_size - 1) // batch_size
            
            print(f"  Processing {num_data_samples} data samples in {num_batches} batch(es) of size {batch_size}...")
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_data_samples)
                batch_size_actual = batch_end - batch_start
                
                print(f"    Batch {batch_idx + 1}/{num_batches}: processing samples {batch_start} to {batch_end - 1}")
                
                # Sample for this batch
                batch_samples_array = self.experiment.sample_params_from_data_samples(
                    design.unsqueeze(0), 
                    posterior_flow, 
                    num_data_samples=batch_size_actual, 
                    num_param_samples=self.guide_samples,
                    central=central,
                    transform_output=self.nf_transform_output
                )
                
                # Convert to numpy if it's a torch tensor
                if isinstance(batch_samples_array, torch.Tensor):
                    batch_samples_array = batch_samples_array.cpu().numpy()
                all_batch_samples.append(batch_samples_array)
                
                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate all batches
            samples_array = np.concatenate(all_batch_samples, axis=0)  # Shape: [num_data_samples, num_param_samples, num_params]

            pair_avg_areas = []
            pair_avg_areas_std = []
            for d in data_idxs:
                # Extract the numpy array from MCSamples and select the d-th data sample
                with contextlib.redirect_stdout(io.StringIO()):
                    samples_gd = getdist.MCSamples(
                        samples=samples_array[d, :, :],
                        names=self.experiment.cosmo_params,
                        labels=self.experiment.latex_labels
                    )
                all_samples.append(samples_gd)
                for p in pair_names:
                    param1, param2 = p.split('_')
                    design_areas[design_type][p].append(get_contour_area(
                        [samples_gd], levels, param1, param2, design_type=design_type)[0][f'{design_type}_area_{p}']
                        )
            # Get the central samples with the nominal design indexed by the input global rank
            central_samples_gd = self._eval_step(step, nominal_design=True)
            all_samples.append(central_samples_gd)
            colors.extend([color]*len(data_idxs) + ['black'])

        plot_width = 10
        g = self.plotter.plot_posterior(all_samples, colors, levels=levels, alpha=0.4, width_inch=plot_width)
        
        # Calculate dynamic font sizes based on plot dimensions and number of parameters
        n_params = len(all_samples[0].paramNames.names)
        # Scale fonts up with more parameters since triangle plot grows
        # Use additive sqrt scaling with reduced coefficients for better balance
        base_fontsize = max(6, min(18, plot_width * (0.2 + 0.42 * np.sqrt(n_params))))
        title_fontsize = base_fontsize * 1.15
        legend_fontsize = base_fontsize * 0.80
        text_fontsize = base_fontsize * 0.75  # For area annotations
        
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()

        def _format_area(value):
            if value is None or np.isnan(value):
                return "nan"
            abs_val = abs(value)
            if abs_val >= 10:
                return f"{value:.2f}"
            elif abs_val >= 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        param_names = g.param_names_for_root(all_samples[0])
        param_name_list = [p.name for p in param_names.names]
        for design_idx, (design_type, color, design) in enumerate(inputs):
            for p in pair_names:
                param1, param2 = p.split('_')
                pair_avg_areas = np.mean(design_areas[design_type][p])
                pair_avg_areas_std = np.std(design_areas[design_type][p])
                # find parameter indices and map to lower triangle
                if param1 in param_name_list and param2 in param_name_list:
                    j = param_name_list.index(param1)  # x-axis
                    i_idx = param_name_list.index(param2)  # y-axis
                    r, c = (max(i_idx, j), min(i_idx, j))
                    ax = g.subplots[r][c]

                    avg_str = _format_area(pair_avg_areas)
                    std_str = _format_area(pair_avg_areas_std)
                    title = f"Avg Area: {avg_str} +/- {std_str}"
                    y_pos = 0.95 - 0.08 * design_idx
                    ax.text(0.05, y_pos, title, transform=ax.transAxes, fontsize=text_fontsize, va='top', color=color)
        
        g.fig.set_constrained_layout(True)
        # Set title with proper positioning
        # add labels for Nominal and Optimal to total legend
        custom_legend = []
        custom_legend.append(
            Line2D([0], [0], color='tab:orange', label=f'Optimal Design', linewidth=1.2)
        )
        custom_legend.append(
            Line2D([0], [0], color='tab:blue', label=f'Nominal Design', linewidth=1.2)
        )
        custom_legend.append(
            Line2D([0], [0], color='black', label=f'Nominal DESI Result', linewidth=1.2)
        )
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96), fontsize=legend_fontsize)
        leg.set_in_layout(False)
        levels_str = ', '.join([f"{int(level*100)}%" for level in levels])
        if self.display_run:
            title = f"Posterior Evaluations for {num_data_samples} Likelihood Samples - Run: {self.run_id[:8]}"
        else:
            title = f"Posterior Evaluations for {num_data_samples} Likelihood Samples"
        g.fig.suptitle(title, fontsize=title_fontsize, weight='bold')
        filename = f"posterior_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.plotter.save_figure(g.fig, filename, run_id=self.run_id, experiment_id=self.exp_id, dpi=400)

    def _compute_eig(self, flow_model, nominal_design=False, designs=None):
        """
        Helper function that evaluates the EIG of a single flow model.

        Args:
            rank (int): Global rank number
            step (int): Step for reference
            flow_model (torch.nn.Module): The flow model to evaluate.
            nominal_design (bool): Whether to evaluate the nominal design.
            designs: Tensor of designs to evaluate. If None, determines from nominal_design or self.input_designs.
        Returns:
            eigs (torch.Tensor): The EIGs for each design.
        """
        device_obj = torch.device(self.device)
        
        # Determine which designs to use
        if designs is None:
            if nominal_design:
                designs = self.experiment.nominal_design.unsqueeze(0)
            else:
                designs = self.input_designs
        
        # Ensure designs are on the correct device
        designs = designs.to(device_obj)
        
        # Clear GPU cache before creating dataset to ensure clean state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Create dataset with explicit device - this creates Pyro trace
        # Each call samples from p(y|θ,d) using Pyro's RNG, which advances state.
        # This means different ranks (or multiple calls) get different likelihood samples,
        # causing Monte Carlo variance in EIG that decreases with n_particles.
        # Process particles in batches to reduce memory usage
        dataset_result = LikelihoodDataset(
            experiment=self.experiment,
            n_particles_per_device=self.n_particles,
            device=self.device,
            evaluation=True,
            designs=designs,
            particle_batch_size=self.particle_batch_size
        )[0]
        
        samples, context, log_probs = dataset_result
        
        # Clear GPU cache after dataset creation to free memory from trace operations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Ensure all tensors are on the correct device
        samples = samples.to(device_obj)
        context = context.to(device_obj)
        if log_probs is not None:
            log_probs = {k: v.to(device_obj) for k, v in log_probs.items()}
        flow_model = flow_model.to(device_obj)

        with torch.no_grad():
            _, eigs = nf_loss(
                samples=samples,
                context=context,
                guide=flow_model,
                experiment=self.experiment,
                rank=0,
                verbose_shapes=False,
                log_probs=log_probs,
                evaluation=True,
                chunk_size=(self.n_particles // 10)
            )
        eigs = eigs.detach().cpu()  # Move to CPU immediately to free GPU memory
        # release temporary tensors to avoid graph retention and free GPU caches after each chunk
        del dataset_result, samples, context, log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations complete before clearing cache

        return eigs

    @profile_method
    def get_eig(self, step, nominal_design=False):
        """
        Calculates EIG for the specified rank.
        Uses caching to avoid redundant calculations.

        Args:
            step (int): Step for caching key and to load the model.
            nominal_design (bool): If True, calculate EIG for nominal design only.
                                 If False, calculate EIGs for all designs in self.experiment.designs.

        Returns:
            tuple: (result, result_std)
                - result: If nominal_design=True, float; else np.ndarray (mean over n_evals)
                - result_std: If nominal_design=True, float; else np.ndarray (std over n_evals)
        """
        cache_key = f"step_{step}_particles_{self.n_particles}_nominal_{nominal_design}_evals_{self.n_evals}"
        
        # Check for cached result
        if cache_key in self._eig_cache:
            print(f"Using cached EIG results for {cache_key}")
            cached_result = self._eig_cache[cache_key]
            # Handle old cache format (2 or 3 values) vs new format (2 values: result, result_std)
            if len(cached_result) == 2:
                # Could be old format (result, individual_ranks) or new format (result, result_std)
                # Check if second element is a dict (old format)
                if isinstance(cached_result[1], dict):
                    # Old format: (result, individual_ranks) - add empty std
                    result = cached_result[0]
                    if nominal_design:
                        result_std = 0.0
                    else:
                        result_std = np.zeros_like(result)
                    return (result, result_std)
                else:
                    # New format: (result, result_std)
                    return cached_result
            elif len(cached_result) == 3:
                # Old format: (result, individual_ranks, result_std)
                return (cached_result[0], cached_result[2])
            else:
                return cached_result

        # Load model to resolve actual checkpoint step
        flow_model, selected_step = load_model(
            self.experiment, step, self.run_obj, 
            self.run_args, self.device, 
            global_rank=self.global_rank
        )
        step_key = f"step_{selected_step}"
        
        # Check if we have precomputed EIG data for this step
        if step_key in self.eig_data:
            step_dict = self.eig_data[step_key]
            if nominal_design:
                step_data = step_dict.get('nominal', {})
                if 'eigs_avg' in step_data and step_data['eigs_avg'] is not None:
                    print(f"Using pre-calculated nominal EIG for {step_key}")
                    result = float(step_data['eigs_avg'])
                    result_std = float(step_data.get('eigs_std', 0.0))
                    self._eig_cache[cache_key] = (result, result_std)
                    return (result, result_std)
            else:
                step_data = step_dict.get('variable', {})
                if 'eigs_avg' in step_data and step_data['eigs_avg'] is not None:
                    print(f"Using pre-calculated EIG values for {step_key}")
                    result = np.array(step_data['eigs_avg'], dtype=float)
                    if 'eigs_std' in step_data and step_data['eigs_std'] is not None:
                        result_std = np.array(step_data['eigs_std'], dtype=float)
                    else:
                        result_std = np.zeros_like(result)
                    self._eig_cache[cache_key] = (result, result_std)
                    return (result, result_std)
        
        flow_model = flow_model.to(self.device)
        flow_model.eval()  # Ensure model is in eval mode
        
        # Clear memory before starting evaluation to ensure clean state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Determine which designs to evaluate
        if nominal_design:
            all_designs = self.experiment.nominal_design.unsqueeze(0)
            num_designs = 1
        else:
            all_designs = self.input_designs
            num_designs = len(all_designs)
        
        # Chunk design space if chunk size is specified (for memory constraints)
        # Chunking is implied by design_chunk_size: None means no chunking, otherwise chunk
        if num_designs == 1 or self.design_chunk_size is None:
            # Single design or no chunking - use all designs
            design_chunks = [list(range(num_designs))]
        else:
            # Chunk the design space according to specified chunk size
            chunk_size = self.design_chunk_size
            design_chunks = []
            for i in range(0, num_designs, chunk_size):
                chunk_indices = list(range(i, min(i + chunk_size, num_designs)))
                design_chunks.append(chunk_indices)
        
        # Get target n_evals from eig_data (may have been loaded from file)
        target_n_evals = int(self.eig_data.get('n_evals', self.n_evals))
        
        if self.verbose:
            print(f"  Evaluating {num_designs} design(s) in {len(design_chunks)} chunk(s) with {target_n_evals} evaluation(s)")
        
        # Persist results in eig_data for future reuse
        # Use nested structure: step_key -> 'nominal' or 'optimal' -> data
        step_dict = self.eig_data.setdefault(step_key, {})
        if nominal_design:
            step_data = step_dict.setdefault('nominal', {})
        else:
            step_data = step_dict.setdefault('variable', {})
        
        # Check if we have existing partial results to resume from
        all_eval_results = []
        
        if 'eigs' in step_data and step_data['eigs'] is not None:
            existing_results = step_data['eigs']
            if isinstance(existing_results, list) and len(existing_results) > 0:
                # Convert to numpy arrays (JSON loads as lists, but we need arrays for computation)
                # Handle both list-of-lists (from JSON) and list-of-arrays (in-memory) cases
                all_eval_results = []
                expected_len = 1 if nominal_design else num_designs
                for idx, r in enumerate(existing_results):
                    if isinstance(r, np.ndarray):
                        arr = r
                    elif isinstance(r, list):
                        arr = np.array(r)
                    else:
                        # Single scalar case - for nominal design, this is expected
                        # Convert scalar to array of length 1
                        arr = np.array([r])
                    
                    # Validate that each entry has the correct length
                    # Ensure array is 1D
                    arr = np.atleast_1d(arr)
                    if len(arr) != expected_len:
                        print(f"  Warning: Evaluation {idx+1} has length {len(arr)}, expected {expected_len}. Skipping this entry.")
                        continue
                    
                    all_eval_results.append(arr)
                
                num_existing = len(all_eval_results)
                if num_existing < target_n_evals:
                    print(f"  Found {num_existing} existing evaluations, continuing from evaluation {num_existing + 1} to {target_n_evals}")
                elif num_existing >= target_n_evals:
                    print(f"  Found {num_existing} existing evaluations (>= {target_n_evals} required), using existing results")
                    # Use only the first target_n_evals results
                    all_eval_results = all_eval_results[:target_n_evals]
                    step_data['eigs'] = [r.tolist() if isinstance(r, np.ndarray) else r for r in all_eval_results]
                    # Recompute averages and return
                    result = np.mean(np.array(all_eval_results), axis=0)
                    result_std = np.std(np.array(all_eval_results), axis=0)
                    if nominal_design:
                        result = result[0]
                        result_std = result_std[0]
                    self._eig_cache[cache_key] = (result, result_std)
                    if nominal_design:
                        step_data['eigs_avg'] = float(result)
                        step_data['eigs_std'] = float(result_std)
                    else:
                        result_array = np.atleast_1d(np.array(result, dtype=float))
                        result_std_array = np.atleast_1d(np.array(result_std, dtype=float))
                        sorted_idx = np.argsort(result_array)[::-1]
                        eigs_avg = result_array.tolist()
                        eigs_std = result_std_array.tolist()
                        optimal_idx = int(sorted_idx[0]) if len(sorted_idx) > 0 else 0
                        designs_source = self.input_designs
                        optimal_design = designs_source[optimal_idx].detach().cpu().numpy().tolist()
                        step_data['eigs_avg'] = eigs_avg
                        step_data['eigs_std'] = eigs_std
                        step_data['optimal_eig'] = float(result_array[optimal_idx])
                        step_data['optimal_eig_std'] = float(result_std_array[optimal_idx]) if len(result_std_array) > optimal_idx else 0.0
                        step_data['optimal_design'] = optimal_design
                    timestamp = getattr(self, 'timestamp', None) or datetime.now().strftime('%Y%m%d_%H%M')
                    self.eig_data['status'] = 'incomplete'  # Mark as incomplete during evaluation
                    eig_data_save_path = f"{self.save_path}/eig_data_{timestamp}.json"
                    with open(eig_data_save_path, "w") as f:
                        json.dump(self.eig_data, f, indent=2)
                    return (result, result_std)
        
        # Loop over remaining n_evals (starting from where we left off)
        start_idx = len(all_eval_results)
        
        # Restore RNG state from the last completed evaluation if resuming
        if start_idx > 0:
            # Try to restore RNG state from the last completed evaluation
            if self.restore_path is not None and os.path.exists(self.restore_path):
                try:
                    with open(self.restore_path, 'rb') as f:
                        rng_state = pickle.load(f)
                    # Create a checkpoint-like dictionary for restore_rng_state
                    checkpoint = {'rng_state': rng_state}
                    # Use the restore_rng_state function from util.py
                    restore_rng_state(checkpoint, self.global_rank)
                    print(f"  Restored RNG state from last completed evaluation (evaluation {start_idx})")
                except Exception as e:
                    print(f"  Warning: Could not restore RNG state from {self.restore_path}: {e}")
                    print(f"  Continuing with current RNG state (results may not be exactly reproducible)")
        
        for eval_idx in range(start_idx, target_n_evals):
            if self.verbose:
                print(f"  Evaluation {eval_idx+1}/{target_n_evals}")
            
            # Clear memory before each evaluation to prevent accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Evaluate each chunk and combine results for this evaluation
            full_eigs = np.zeros(num_designs)
            
            for chunk_idx, design_indices in enumerate(design_chunks):
                # Get designs for this chunk
                chunk_designs = all_designs[design_indices].to(self.device)
                
                # Evaluate this chunk
                eig_result = self._compute_eig(
                    flow_model,
                    nominal_design=nominal_design,
                    designs=chunk_designs
                )
                
                # Store results for this chunk (eig_result is already on CPU)
                eig_array = eig_result.numpy() / np.log(2)
                if len(design_indices) == 1:
                    # Single design case
                    if nominal_design or num_designs == 1:
                        full_eigs[design_indices[0]] = eig_array.item()
                    else:
                        full_eigs[design_indices[0]] = eig_array if isinstance(eig_array, (int, float, np.number)) else eig_array[0]
                else:
                    # Multiple designs in chunk
                    for idx, eig_val in zip(design_indices, eig_array.flatten()):
                        full_eigs[idx] = eig_val
            
            all_eval_results.append(full_eigs)
            # Save to step_data, converting numpy arrays to lists for JSON serialization
            step_data['eigs'] = [r.tolist() if isinstance(r, np.ndarray) else r for r in all_eval_results]

            # Save EIG data to file
            timestamp = getattr(self, 'timestamp', None) or datetime.now().strftime('%Y%m%d_%H%M')
            self.eig_data['status'] = 'incomplete'  # Mark as incomplete during evaluation
            eig_data_save_path = f"{self.save_path}/eig_data_{timestamp}.json"
            with open(eig_data_save_path, "w") as f:
                json.dump(self.eig_data, f, indent=2)
            
            # Save RNG state for resuming
            self._save_rng_state()

            # Clear memory between evaluations
            del eig_result, eig_array, chunk_designs, full_eigs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete
            gc.collect()  # Force Python garbage collection
        
        # Average over evaluations and compute std
        all_eval_results = np.array(all_eval_results)
        result = np.mean(all_eval_results, axis=0)  # Shape: (num_designs,)
        result_std = np.std(all_eval_results, axis=0)  # Shape: (num_designs,)
        
        # Handle scalar case for nominal design
        if nominal_design:
            result = result[0]  # Scalar
            result_std = result_std[0]  # Scalar
        
        # Cache result and std
        self._eig_cache[cache_key] = (result, result_std)

        if nominal_design:
            step_data['eigs_avg'] = float(result)
            step_data['eigs_std'] = float(result_std)
        else:
            result_array = np.atleast_1d(np.array(result, dtype=float))
            result_std_array = np.atleast_1d(np.array(result_std, dtype=float))
            sorted_idx = np.argsort(result_array)[::-1]
            eigs_avg = result_array.tolist()
            eigs_std = result_std_array.tolist()
            optimal_idx = int(sorted_idx[0]) if len(sorted_idx) > 0 else 0

            designs_source = self.input_designs
            optimal_design = designs_source[optimal_idx].detach().cpu().numpy().tolist()

            step_data['eigs_avg'] = eigs_avg
            step_data['eigs_std'] = eigs_std
            step_data['optimal_eig'] = float(result_array[optimal_idx])
            step_data['optimal_eig_std'] = float(result_std_array[optimal_idx]) if len(result_std_array) > optimal_idx else 0.0
            step_data['optimal_design'] = optimal_design

        timestamp = getattr(self, 'timestamp', None) or datetime.now().strftime('%Y%m%d_%H%M')
        self.eig_data['status'] = 'incomplete'  # Mark as incomplete during evaluation
        eig_data_save_path = f"{self.save_path}/eig_data_{timestamp}.json"
        with open(eig_data_save_path, "w") as f:
            json.dump(self.eig_data, f, indent=2)
        print(f"Saved EIG data to {eig_data_save_path}")

        return (result, result_std)

    @profile_method
    def eig_grid(self, step):
        """
        Plots the EIG on a 2D grid with subplot layout:
        - Top plot: colors points by the 3rd design variable (f_QSO)
        - Bottom plot: colors points by EIG values
        
        Args:
            step: The step to evaluate
        """
        print(f"Running EIG grid evaluation...")
        # Only generate grid plot if we have multiple designs (grid case)
        if len(self.input_designs) <= 1:
            print("Warning: Single or no designs available, skipping EIG grid plot.")
            return

        # Get EIGs, nominal EIG, and optimal design (handles all ranks internally)
        eigs, _ = self.get_eig(step, nominal_design=False)
        optimal_design_tensor = self.input_designs[np.argmax(eigs)]
        optimal_design = optimal_design_tensor.cpu().numpy()
        
        # Determine label suffix
        label_suffix = f'Rank {self.global_rank}'
        
        # Create figure with subplots
        fig = plt.figure(figsize=(8, 12))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.08], height_ratios=[1, 1], wspace=0.1, hspace=0.2)
        
        # Create subplots
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bottom = fig.add_subplot(gs[1, 0])
        cbar_ax_top = fig.add_subplot(gs[0, 1])
        cbar_ax_bottom = fig.add_subplot(gs[1, 1])

        # Get design variables
        designs = self.input_designs.cpu().numpy()
        nominal_design = self.experiment.nominal_design.cpu().numpy()
        # optimal_design is already a numpy array from the combination step above

        # Top plot: scatter with 3rd design variable as color
        scatter_top = ax_top.scatter(designs[:, 1], designs[:, 2],
                    c=designs[:, 2],
                    cmap='viridis',
                    s=60,
                    alpha=0.8,
                    marker='o',
                    vmin=np.min(designs[:, 2]),
                    vmax=np.max(designs[:, 2]))
        
        # Add nominal and optimal markers to top plot
        ax_top.scatter(nominal_design[1], nominal_design[2],
                    c=nominal_design[2],
                    cmap='viridis',
                    s=75,
                    alpha=0.8,
                    vmin=np.min(designs[:, 2]),
                    vmax=np.max(designs[:, 2]),
                    edgecolor='black',
                    marker='*',
                    label='Nominal Design')
        
        ax_top.scatter(optimal_design[1], optimal_design[2],
                    c='tab:blue',
                    s=75,
                    alpha=0.8,
                    edgecolor='tab:blue',
                    marker='*',
                    label='Optimal Design')
        
        # Configure top plot
        ax_top.set_title(f'Design Variables', weight='bold')
        ax_top.set_xlabel("$f_{LRG}$", fontsize=14)
        ax_top.set_ylabel("$f_{ELG}$", fontsize=14)
        ax_top.grid(True, alpha=0.3)
        ax_top.set_aspect('equal')
        ax_top.set_box_aspect(1)
        ax_top.legend(fontsize=12)

        # Bottom plot: scatter with EIG as color
        scatter_bottom = ax_bottom.scatter(designs[:, 1], designs[:, 2],
                    c=eigs,
                    cmap='viridis',
                    s=60,
                    alpha=0.8,
                    marker='o',
                    vmin=np.min(eigs),
                    vmax=np.max(eigs))
        
        # Add nominal and optimal markers to bottom plot
        ax_bottom.scatter(nominal_design[1], nominal_design[2],
                    c='black',
                    s=75,
                    marker='*',
                    label='Nominal Design')
        
        ax_bottom.scatter(optimal_design[1], optimal_design[2],
                    c='tab:blue',
                    s=75,
                    edgecolor='tab:blue',
                    marker='*',
                    label='Optimal Design')

        # Configure bottom plot
        ax_bottom.set_title('Expected Information Gain', weight='bold')
        ax_bottom.set_xlabel("$f_{LRG}$", fontsize=14)
        ax_bottom.set_ylabel("$f_{ELG}$", fontsize=14)
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.set_aspect('equal')
        ax_bottom.set_box_aspect(1)
        ax_bottom.legend(fontsize=12)

        # Add colorbars with tick values
        cbar_top = plt.colorbar(scatter_top, cax=cbar_ax_top, label='$f_{QSO}$')
        cbar_bottom = plt.colorbar(scatter_bottom, cax=cbar_ax_bottom, label='EIG [bits]')
        
        # Set tick values for colorbars with intermediate ticks
        # Top colorbar (f_QSO) - 5 evenly spaced ticks
        f_qso_ticks = np.linspace(np.min(designs[:, 2]), np.max(designs[:, 2]), 5)
        cbar_top.set_ticks(f_qso_ticks)
        cbar_top.set_ticklabels([f'{tick:.2f}' for tick in f_qso_ticks])
        
        # Bottom colorbar (EIG) - 5 evenly spaced ticks
        eig_ticks = np.linspace(np.min(eigs), np.max(eigs), 5)
        cbar_bottom.set_ticks(eig_ticks)
        cbar_bottom.set_ticklabels([f'{tick:.2f}' for tick in eig_ticks])

        # Add overall title with rank information
        if self.display_run:
            title = f'EIG Grid ({label_suffix}) - Run: {self.run_id[:8]}'
        else:
            title = f'EIG Grid ({label_suffix})'
        fig.suptitle(title, fontsize=16, y=0.995, weight='bold')
        
        plt.tight_layout()
        filename = f"eig_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.plotter.save_figure(fig, filename, run_id=self.run_id, experiment_id=self.exp_id, dpi=400)

    def run(self, eval_step=None):
        # Determine eval_step
        if eval_step is None or eval_step == 'last':
            eval_step = self.total_steps
        else:
            eval_step = int(eval_step)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        # Initialize timing at start of run
        self._update_runtime()

        step_key = f"step_{eval_step}"
        step_dict = self.eig_data.get(step_key, {})
        has_step_data = 'eigs_avg' in step_dict.get('variable', {})
        if has_step_data:
            print(f"Using existing EIG input designs at step {eval_step}...")
        else:
            print(f"Computing EIG input designs at step {eval_step}...")

        nominal_eig, nominal_eig_std = self.get_eig(step=eval_step, nominal_design=True)
        eigs, eigs_std = self.get_eig(step=eval_step, nominal_design=False)
        # Determine label based on whether we have multiple designs or a single input design
        has_multiple_designs = len(self.input_designs) > 1
        eig_label = 'Optimal' if has_multiple_designs else 'Input'

        step_data = step_dict.get('variable', {})
        max_idx = int(np.argmax(eigs)) if len(np.atleast_1d(eigs)) > 0 else 0
        optimal_eig_value = float(step_data.get('optimal_eig', np.atleast_1d(eigs)[max_idx]))
        if isinstance(eigs_std, np.ndarray):
            optimal_eig_std_value = float(step_data.get('optimal_eig_std', eigs_std[max_idx]))
        else:
            optimal_eig_std_value = float(step_data.get('optimal_eig_std', eigs_std))
        optimal_design = step_data.get('optimal_design')
        if optimal_design is None:
            # If we have multiple designs, use the one with max EIG; otherwise use the single design
            if len(self.input_designs) > 1:
                optimal_design_tensor = self.input_designs[max_idx].cpu().numpy()
            else:
                optimal_design_tensor = self.input_designs[0].cpu().numpy()
            optimal_design = optimal_design_tensor.tolist()

        print(f"Nominal EIG: {nominal_eig}, {eig_label} EIG: {optimal_eig_value}, {eig_label} Design: {optimal_design}")

        self.eig_data["eval_step"] = int(eval_step)
        self.eig_data["nominal_eig"] = float(nominal_eig)
        self.eig_data["nominal_eig_std"] = float(nominal_eig_std) if isinstance(nominal_eig_std, (int, float, np.number)) else 0.0
        self.eig_data["nominal_design"] = self.experiment.nominal_design.cpu().numpy().tolist()
        self.eig_data["optimal_eig"] = optimal_eig_value
        self.eig_data["optimal_eig_std"] = optimal_eig_std_value
        self.eig_data["optimal_design"] = optimal_design
        self.eig_data["design_type"] = eig_label.lower()  # 'optimal' or 'fixed'
        self.eig_data['status'] = 'incomplete'  # Mark as incomplete during evaluation

        if self.brute_force:
            try:
                self._compute_brute_force_eig(step_key=step_key)
            except Exception as e:
                print(f"Warning: brute-force EIG calculation failed: {e}")
                traceback.print_exc()

        eig_data_save_path = f"{self.save_path}/eig_data_{self.timestamp}.json"
        with open(eig_data_save_path, "w") as f:
            json.dump(self.eig_data, f, indent=2)
        print(f"Saved EIG data to {eig_data_save_path}")

        # Update timing after EIG computation
        self._update_runtime()
        
        # Check if EIG data is already complete (from loaded file)
        is_complete = self.eig_data.get('status', 'incomplete') == 'complete'
        
        # If not complete, mark as complete and save
        if not is_complete:
            self.eig_data['status'] = 'complete'  # Mark as complete when evaluation finishes
            eig_data_save_path = f"{self.save_path}/eig_data_{self.timestamp}.json"
            with open(eig_data_save_path, "w") as f:
                json.dump(self.eig_data, f, indent=2)
            print(f"Saved EIG data to {eig_data_save_path}")
        
        # Make some evaluation plots
        try:
            self.plotter.generate_posterior(
                step_key=eval_step, display=['nominal', 'optimal'],  guide_samples=50000,
                levels=self.levels, plot_prior=True, transform_output=self.nf_transform_output
                )
            self._update_runtime()
        except Exception as e:
            print(f"Warning: generate_posterior failed: {e}")
            traceback.print_exc()
        
        try:
            if self.cosmo_exp == 'num_tracers':
                self.plotter.design_comparison(step_key=eval_step, log_scale=True, use_fractional=False)
                self._update_runtime()
        except Exception as e:
            print(f"Warning: design_comparison failed: {e}")
            traceback.print_exc()
        
        try:
            self.plotter.eig_designs(step_key=eval_step, sort=self.sort, include_nominal=self.include_nominal)
            self._update_runtime()
        except Exception as e:
            print(f"Warning: eig_designs failed: {e}")
            traceback.print_exc()
        
        try:
            # Plot posterior at different training steps
            steps_to_plot = [self.total_steps//4, self.total_steps//2, self.total_steps*3//4, 'last']
            self.plotter.posterior_steps(steps=steps_to_plot, levels=self.levels)
            self._update_runtime()
        except Exception as e:
            print(f"Warning: posterior_steps failed: {e}")
            traceback.print_exc()
        
        print(f"Evaluation completed successfully!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Cosmo Experiment Evaluation')
    parser.add_argument('--run_id', type=str, default=None, help='MLflow run ID to resume training from (continues existing run with same parameters)')
    parser.add_argument('--eval_step', type=str, default='last', help='Step to evaluate (can be integer or "last")')
    parser.add_argument('--cosmo_exp', type=str, default='num_tracers', help='Cosmological experiment name')
    parser.add_argument('--levels', type=parse_float_or_list, default=0.68, help='Levels for contour plot (can be a single float or JSON list of floats)')
    parser.add_argument('--global_rank', type=int, default=0, help='Global rank to evaluate (default: 0)')
    parser.add_argument('--n_particles', type=int, default=1000, help='Number of particles to use for evaluation')
    parser.add_argument('--guide_samples', type=int, default=10000, help='Number of samples to generate from the posterior')
    parser.add_argument('--design_chunk_size', type=int, default=None, help='Number of designs per chunk (None = use all designs)')
    parser.add_argument('--n_evals', type=int, default=5, help='Number of evaluations to average over')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Seed for evaluation')
    parser.add_argument('--param_space', type=str, default='physical', help='Parameter space to use for evaluation')
    parser.add_argument('--profile', action='store_true', help='Enable profiling of methods')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-sort', dest='sort', action='store_false', help='Disable sorting of designs by EIG (default: sorting enabled)')
    parser.add_argument('--include_nominal', action='store_true', default=False, help='Include nominal EIG in eig_designs plot (default: False)')
    parser.add_argument('--eig_file_path', type=str, default=None, help='Path to previously calculated eig_data JSON file to load instead of recalculating')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for sample_posterior to reduce memory usage (default: 1)')
    parser.add_argument('--particle_batch_size', type=int, default=None, help='Batch size for processing particles in LikelihoodDataset to reduce memory usage (default: None to use all particles)')
    parser.add_argument('--design_args_path', type=str, default=None, help='Path to design_args.yaml file. If None, defaults to the run\'s artifacts/design_args.yaml')
    parser.add_argument('--brute_force', action='store_true', default=False, help='Run brute-force EIG using bayesdesign ExperimentDesigner')
    parser.add_argument('--brute_force_param_points', type=int, default=75, help='Number of points per parameter axis for brute-force parameter grid')
    parser.add_argument('--brute_force_feature_points', type=int, default=35, help='Number of points per feature axis for brute-force feature grid')

    args = parser.parse_args()

    valid_params = inspect.signature(Evaluator.__init__).parameters.keys()
    valid_params = [k for k in valid_params if k != 'self']
    eval_args = {k: v for k, v in vars(args).items() if k in valid_params}
    
    print(f"Evaluating with parameters:")
    print(json.dumps(eval_args, indent=2))
    evaluator = Evaluator(**eval_args)
    evaluator.run(eval_step=args.eval_step)
