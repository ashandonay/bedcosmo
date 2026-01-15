import sys
import os
import contextlib
import io
import json
import gc
import pickle
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

base_dir = os.environ["HOME"] + '/bed/BED_cosmo'
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
    
import numpy as np
import torch
import getdist
import argparse
from plotting import *
import traceback
from pyro_oed_src import nf_loss, LikelihoodDataset
from matplotlib.patches import Rectangle
from util import *
import mlflow
import inspect

class Evaluator:
    def __init__(
            self, run_id, guide_samples=1000, design_step=0.05, design_lower=0.05, design_upper=None,
            design_chunk_size=None, seed=1, cosmo_exp='num_tracers', levels=[0.68, 0.95], global_rank=0, eig_file_path=None,
            n_evals=10, n_particles=1000, param_space='physical', input_designs=None, design_sum_lower=1.0, design_sum_upper=1.0,
            display_run=False, verbose=False, device="cuda:0", profile=False, sort=True, include_nominal=False, batch_size=1,
            particle_batch_size=None
            ):
        self.cosmo_exp = cosmo_exp
        
        # Store seed parameter for later use (may be overridden by eig_file_path)
        self.seed = seed
        
        # Set MLflow tracking URI based on cosmo_exp
        mlflow.set_tracking_uri("file:" + os.environ["SCRATCH"] + f"/bed/BED_cosmo/{self.cosmo_exp}/mlruns")
        
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
        self.storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{self.cosmo_exp}"
        self.save_path = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts"
        
        self.n_evals = n_evals
        self.n_particles = n_particles
        self.particle_batch_size = particle_batch_size
        self.seed = seed
        # Load eig_file_path early if provided, to get the seed that was used previously
        # This ensures we use the same seed for reproducibility
        self.eig_file_path = eig_file_path
        self._init_eig_data()
        
        # Set seed NOW before any operations that might use randomness (like init_experiment)
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
        
        # Store design arguments
        self.design_args = {
            "design_step": design_step,
            "design_lower": design_lower,
            "design_upper": design_upper,
            "design_sum_lower": design_sum_lower,
            "design_sum_upper": design_sum_upper,
        }
        
        # Add input_design if provided (otherwise it will be None, which triggers grid generation)
        if input_designs is not None:
            self.design_args["input_designs"] = input_designs
        
        # Initialize experiment - it will handle input_design and generate designs accordingly 
        # (single design, multiple designs, or grid)
        self.experiment = init_experiment(
            self.run_obj, self.run_args, self.device, 
            design_args=self.design_args, global_rank=self.global_rank
        )
        if self.eig_file_path is not None and 'input_designs' in self.eig_data:
            self.input_designs = torch.tensor(self.eig_data['input_designs'], device=self.device, dtype=torch.float64)
        else:
            self.input_designs = self.experiment.designs
            # save to eig_data object
            self.eig_data['input_designs'] = self.input_designs.cpu().numpy().tolist()
            self.eig_data['nominal_design'] = self.experiment.nominal_design.cpu().numpy().tolist()
            
        
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
        # Save RNG state after each evaluation for resuming capability
        # This allows us to restore the exact random state when continuing from a partial evaluation
        # Save only the latest RNG state (overwrite previous) to avoid accumulating too many files
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
    def generate_posterior(self, step='last', display=['nominal', 'optimal'], levels=[0.68]):
        """
        Generates the posterior for given type(s) of design input.

        Args:
            step (int): The checkpoint step to plot the posterior for.
            display (list): The designs to display.
            levels (float or list): Contour level(s) to plot.

        """
        # Normalize levels to always be a list
        if isinstance(levels, (int, float)):
            levels = [levels]
        
        print(f"Generating posterior plot...")
        all_samples = []
        all_colors = []
        all_alphas = []
        
        if 'nominal' in display:
            print(f"Generating posterior samples with nominal design...")
            nominal_samples = self._eval_step(step, nominal_design=True)
            all_samples.append(nominal_samples)
            all_colors.append('tab:blue')
            all_alphas.append(1.0)
        
        # Check if we have multiple designs (grid) or single/multiple input designs
        # Only show "optimal" if we have a grid (multiple designs to choose from)
        has_multiple_designs = len(self.input_designs) > 1
        
        if 'optimal' in display and has_multiple_designs:
            print(f"Generating posterior samples with optimal design...")
            optimal_samples = self._eval_step(step)
            all_samples.append(optimal_samples)
            all_colors.append('tab:orange')
            all_alphas.append(1.0)

        # Get the DESI MCMC samples
        try:
            nominal_samples_gd = self.experiment.get_nominal_samples(transform_output=self.nominal_transform_output)
            all_samples.append(nominal_samples_gd)
            all_colors.append('black')
            all_alphas.append(1.0)
            ref_contour = True
        except NotImplementedError:
            print(f"Warning: get_nominal_samples not implemented for {self.cosmo_exp}, skipping nominal design plot.")
            ref_contour = False
            pass
        
        plot_width = 10
        g = plot_posterior(all_samples, all_colors, levels=levels, width_inch=plot_width, alpha=all_alphas)
        
        # Calculate dynamic font sizes based on plot dimensions and number of parameters
        n_params = len(all_samples[0].paramNames.names)
        # Base font size scales with plot width AND number of parameters
        # More parameters = larger triangle plot = need larger fonts
        # Use additive sqrt scaling with reduced coefficients for better balance
        base_fontsize = max(6, min(18, plot_width * (0.2 + 0.42 * np.sqrt(n_params))))
        title_fontsize = base_fontsize * 1.15  # Title slightly larger than base
        legend_fontsize = base_fontsize * 0.65  # Legend smaller than base
        
        if self.display_run:
            title = f"Posterior Comparison - Run: {self.run_id[:8]}"
        else:
            title = f"Posterior Comparison"
        g.fig.suptitle(title, fontsize=title_fontsize, weight='bold')

        # Get EIG values for legend
        nominal_eig = None
        optimal_eig = None
        if 'nominal' in display:
            try:
                nominal_eig, _ = self.get_eig(step, nominal_design=True)
            except Exception as e:
                print(f"Warning: Could not get EIG for nominal design: {e}")
        if 'optimal' in display:
            try:
                eigs, _ = self.get_eig(step, nominal_design=False)
                if len(self.input_designs) > 1:
                    optimal_eig = np.max(eigs)
                else:
                    optimal_eig = eigs[0] if isinstance(eigs, np.ndarray) else eigs
            except Exception as e:
                print(f"Warning: Could not get EIG for optimal design: {e}")

        # Create custom legend
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()
        custom_legend = []
        if 'nominal' in display:
            eig_str = f", EIG: {nominal_eig:.3f} bits" if nominal_eig is not None else ""
            custom_legend.append(
                Line2D([0], [0], color='tab:blue', label=f'Nominal Design (NF){eig_str}', linewidth=1.2)
            )
        # Check if we have multiple designs (grid) - only show optimal if we have multiple
        has_multiple_designs = len(self.input_designs) > 1
        if 'optimal' in display and has_multiple_designs:
            eig_str = f", EIG: {optimal_eig:.3f} bits" if optimal_eig is not None else ""
            custom_legend.append(
                Line2D([0], [0], color='tab:orange', label=f'Optimal Design (NF){eig_str}', linewidth=1.2)
            )
        elif 'optimal' in display and not has_multiple_designs:
            eig_str = f", EIG: {optimal_eig:.3f} bits" if optimal_eig is not None else ""
            custom_legend.append(
                Line2D([0], [0], color='tab:orange', label=f'Input Design (NF){eig_str}', linewidth=1.2)
            )
        if ref_contour:
            custom_legend.append(
                Line2D([0], [0], color='black', label=f'Nominal Design (MCMC)', linewidth=1.2)
            )
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96), fontsize=legend_fontsize)
        leg.set_in_layout(False)
        save_figure(f"{self.save_path}/plots/posterior_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.param_space}.png", fig=g.fig, dpi=400)
    
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
        g = plot_posterior(all_samples, colors, levels=levels, alpha=0.4, width_inch=plot_width)
        
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
        save_figure(f"{self.save_path}/plots/posterior_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=g.fig, dpi=400)

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
            
            # Clear memory between evaluations to prevent accumulation
            # This is important because log_prob computations can create large computation graphs
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
    def design_comparison(self, step, width=0.2, log_scale=True, use_fractional=False):
        """
        Plots a bar chart the nominal and optimal design.
        
        Args:
            step: The step to evaluate
            width: Width of the bars in the bar chart
            log_scale: Whether to use log scale for y-axis (default: True)
            use_fractional: Whether to plot fractional values or absolute quantities (default: False)
        """
        print(f"Generating design comparison plot...")
        
        # Determine which design to use
        # If we have multiple designs, find optimal; otherwise use the first (and only) design
        optimal_eig = None
        nominal_eig = None
        if len(self.input_designs) > 1:
            # Get EIGs and optimal design
            eigs, _ = self.get_eig(step, nominal_design=False)
            optimal_eig = np.max(eigs)
            optimal_design = self.input_designs[np.argmax(eigs)]
            design = optimal_design.cpu().numpy()
        else:
            # Single design case - use the only available design
            try:
                eigs, _ = self.get_eig(step, nominal_design=False)
                optimal_eig = eigs[0] if isinstance(eigs, np.ndarray) else eigs
            except Exception as e:
                print(f"Warning: Could not get EIG for input design: {e}")
            design = self.input_designs[0].cpu().numpy()
        
        # Get EIG for nominal design
        try:
            nominal_eig, _ = self.get_eig(step, nominal_design=True)
        except Exception as e:
            print(f"Warning: Could not get EIG for nominal design: {e}")
        
        # Get nominal design and total observations
        nominal_design_cpu = self.experiment.nominal_design.cpu().numpy()
        nominal_total_obs = self.experiment.nominal_total_obs
        
        # Determine whether to use fractional or absolute values
        if use_fractional:
            # Use fractional values directly
            nominal_design_plot = nominal_design_cpu
            design_plot = design
            # Get maximum possible tracers as fractions
            max_tracers = np.array([self.experiment.num_targets[target] for target in self.experiment.design_labels])
            max_tracers = max_tracers / nominal_total_obs
            ylabel = 'Fraction of Total Tracers'
        else:
            # Convert fractional values to actual number of tracers
            nominal_design_plot = nominal_design_cpu * nominal_total_obs
            design_plot = design * nominal_total_obs
            # Get maximum possible tracers for each class
            max_tracers = np.array([self.experiment.num_targets[target] for target in self.experiment.design_labels])
            ylabel = 'Number of Tracers'
        
        # Set the positions for the bars
        x = np.arange(len(self.experiment.design_labels))  # the label locations

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Add gray bars with dotted edges for maximum possible tracers
        # Use Rectangle patches for the filled bars, then manually draw dotted edges
        for i, max_val in enumerate(max_tracers):
            # Draw filled gray rectangle
            rect = Rectangle((x[i] - width, 0), width*2, max_val,
                           facecolor='gray', alpha=0.3, edgecolor='none')
            ax.add_patch(rect)
            # Manually draw dotted edges
            x_left = x[i] - width
            x_right = x[i] + width
            y_top = max_val
            # Top edge
            ax.plot([x_left, x_right], [y_top, y_top], 'k:', linewidth=1.5, alpha=0.7)
            # Left edge
            ax.plot([x_left, x_left], [0, y_top], 'k:', linewidth=1.5, alpha=0.7)
            # Right edge
            ax.plot([x_right, x_right], [0, y_top], 'k:', linewidth=1.5, alpha=0.7)
        
        # Determine label based on whether we have multiple designs or a single input design
        has_multiple_designs = len(self.input_designs) > 1
        design_label = 'Optimal Design' if has_multiple_designs else 'Input Design'
        # Add EIG values to labels
        nominal_label = 'Nominal Design'
        if nominal_eig is not None:
            nominal_label += f', EIG: {nominal_eig:.3f} bits'
        design_label_with_eig = design_label
        if optimal_eig is not None:
            design_label_with_eig += f', EIG: {optimal_eig:.3f} bits'
        
        bars1 = ax.bar(x - width/2, nominal_design_plot, width, label=nominal_label, color='tab:blue')
        bars2 = ax.bar(x + width/2, design_plot, width, label=design_label_with_eig, color='tab:orange')
        
        # Add label for max possible tracers (manually since we used patches)
        max_label = 'Max Possible Tracers' if not use_fractional else 'Max Possible Fraction'
        ax.plot([], [], 'k:', linewidth=1.5, alpha=0.7, label=max_label)
        
        ax.set_xlabel('Tracer Class', fontsize=12, weight='bold')
        ax.set_ylabel(ylabel, fontsize=12, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.experiment.design_labels)
        ax.legend(fontsize=14)
        if log_scale:
            ax.set_yscale('log')
        if self.display_run:
            title = f"Design Comparison - Run: {self.run_id[:8]}"
        else:
            title = f"Design Comparison"
        plt.suptitle(title, fontsize=16, weight='bold')
        plt.tight_layout()
        save_path = f"{self.save_path}/plots/design_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_figure(save_path, fig=fig, dpi=400)
        plt.close(fig)


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
        save_figure(f"{self.save_path}/plots/eig_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=fig, dpi=400)

    @profile_method
    def posterior_steps(self, steps, levels=[0.68]):
        """
        Plots posterior distributions at different training steps for a single run.
        
        Args:
            steps (list): List of steps to plot. Can include 'last' or 'loss_best' as special values.
            levels (float or list): Contour level(s) to plot.
        """
        # Normalize levels to always be a list
        if isinstance(levels, (int, float)):
            levels = [levels]
        
        print(f"Running posterior steps evaluation...")
        colors = plt.cm.viridis_r(np.linspace(0, 1, len(steps)))
        
        all_samples = []
        all_colors = []
        custom_legend = []
        checkpoint_dir = f'{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/checkpoints'
        if not os.path.isdir(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found for run {self.run_id}, skipping. Path: {checkpoint_dir}")
            return
        for i, step in enumerate(steps):
            samples = self._eval_step(step, nominal_design=True)
            # Convert RGBA color to hex string before extending
            color_hex = matplotlib.colors.to_hex(colors[i % len(colors)])
            # samples is a single MCSamples object
            all_samples.append(samples)
            all_colors.append(color_hex)
            if step == 'last':
                step_label = self.run_args["total_steps"]
            elif step == 'loss_best':
                step_label = 'Best Loss'
            else:
                step_label = step
            custom_legend.append(
                Line2D([0], [0], color=color_hex, 
                        label=f'Step {step_label}', linewidth=1.2)
            )

        try:
            # Get nominal samples using reference experiment
            nominal_samples_gd = self.experiment.get_nominal_samples(transform_output=self.nominal_transform_output)
            all_samples.append(nominal_samples_gd)
            all_colors.append('black')  
        except NotImplementedError:
            print(f"Warning: get_nominal_samples not implemented for {self.cosmo_exp}, skipping nominal design plot.")
            pass

        plot_width = 12
        g = plot_posterior(all_samples, all_colors, levels=levels, width_inch=plot_width)
        
        # Calculate dynamic font sizes based on plot dimensions and number of parameters
        n_params = len(all_samples[0].paramNames.names)
        # Scale fonts up with more parameters since triangle plot grows
        # Use additive sqrt scaling with reduced coefficients for better balance
        base_fontsize = max(6, min(18, plot_width * (0.2 + 0.42 * np.sqrt(n_params))))
        title_fontsize = base_fontsize * 1.15
        legend_fontsize = base_fontsize * 0.80
        
        # Remove existing legends if any
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()

        custom_legend.append(
            Line2D([0], [0], color='black', label=f'DESI', linewidth=1.2)
        )
        
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96), fontsize=legend_fontsize)
        leg.set_in_layout(False)
        levels_str = ', '.join([f"{int(level*100)}%" for level in levels])
        if self.display_run:
            title = f"Posterior Steps - Run: {self.run_id[:8]}"
        else:
            title = f"Posterior Steps"
        g.fig.suptitle(title, fontsize=title_fontsize, weight='bold')
        
        save_figure(f"{self.save_path}/plots/posterior_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=g.fig, dpi=400)

    @profile_method
    def eig_designs(self, steps='last', sort=True, sort_step=None, include_nominal=True):
        """
        Plots sorted EIG values and corresponding designs using the evaluator's EIG calculations.
        Can plot single or multiple training steps.
        
        Args:
            steps (int, str, or list): Step(s) to evaluate. Can be:
                                      - Single int/str (e.g., 5000 or 'last')
                                      - List of steps (e.g., [2500, 5000, 7500, 'last'])
            sort_step (int, str, or None): Which step to use for sorting designs. Must be in `steps`.
                                          If None, uses 'last' if present, otherwise the largest step.
            include_nominal (bool): Whether to include the nominal EIG in the plot.
        """
        # Convert single step to list for unified handling
        if not isinstance(steps, list):
            steps = [steps]
        
        print(f"Generating EIG designs plot for step(s) {steps}...")
        
        # Only generate sorted EIG plot if we have multiple designs (grid case)
        if len(self.input_designs) <= 1:
            print("Warning: Single or no designs available, skipping sorted EIG plot.")
            return
        
        # Determine which step to use for sorting
        if sort_step is None:
            # Default: use 'last' if present, otherwise the largest numeric step
            if 'last' in steps:
                sort_step = 'last'
            else:
                # Find the largest numeric step
                numeric_steps = [s for s in steps if isinstance(s, int)]
                if numeric_steps:
                    sort_step = max(numeric_steps)
                else:
                    sort_step = steps[-1]  # Fall back to last in list
        else:
            # Validate that sort_step is in the steps list
            if sort_step not in steps:
                raise ValueError(f"sort_step={sort_step} must be one of the steps to plot: {steps}")
        
        print(f"  Sorting designs by step: {sort_step}")
        
        # Create color gradient for multiple steps (light gray to black)
        if len(steps) > 1:
            from matplotlib.colors import LinearSegmentedColormap
            light_gray = np.array([0.7, 0.7, 0.7])  # RGB for light gray
            black = np.array([0.0, 0.0, 0.0])        # RGB for black
            
            n_steps = len(steps)
            colors = np.array([light_gray * (1 - i/(n_steps-1)) + black * (i/(n_steps-1)) 
                              for i in range(n_steps)])
        else:
            colors = [np.array([0.0, 0.0, 0.0])]  # Just black for single step
        
        # Store EIG data for all steps
        all_steps_data = []
        sorted_eigs_idx = None
        
        for step_idx, s in enumerate(steps):
            # Calculate/get EIG data for this step (get_eig handles caching and storage)
            print(f"  Getting EIG data for step {s}...")
            eigs_avg, eigs_std = self.get_eig(s, nominal_design=False)
            eigs_avg = np.atleast_1d(eigs_avg)
            eigs_std = np.atleast_1d(eigs_std)
            
            # Get the actual step number that was resolved (needed for step_key)
            _, selected_step = load_model(
                self.experiment, s, self.run_obj, 
                self.run_args, self.device, 
                global_rank=self.global_rank
            )
            
            # Store data for this step
            data_dict = {
                'step': selected_step,
                'step_label': s,
                'eigs_avg': eigs_avg,
                'eigs_std': eigs_std,
                'color': colors[step_idx]
            }
            
            if include_nominal:
                # Get nominal EIG (get_eig handles caching and storage)
                nominal_eig, _ = self.get_eig(s, nominal_design=True)
                data_dict['nominal'] = float(nominal_eig)
                
            all_steps_data.append(data_dict)
            
            # Use the specified sort_step to determine sorting order
            if s == sort_step:
                if sort:
                    sorted_eigs_idx = np.argsort(eigs_avg)[::-1]
                else:
                    # Keep original design order
                    sorted_eigs_idx = np.arange(len(eigs_avg))
        
        # Get designs and order them according to sorted_eigs_idx (use reference experiment)
        designs = self.input_designs.cpu().numpy()
        if sorted_eigs_idx is None:
            sorted_eigs_idx = np.arange(len(designs))
        sorted_designs = designs[sorted_eigs_idx]

        nominal_sorted_pos = None
        nominal_sorted_value = None

        metrics_step_data = next((d for d in all_steps_data if d['step_label'] == sort_step), None)
        
        # Check if designs are 1D or multi-dimensional
        is_1d_design = (sorted_designs.shape[1] == 1)
        
        if include_nominal:
            nominal_design = self.experiment.nominal_design.cpu().numpy()
            try:
                nominal_idx = int(np.argmin(np.linalg.norm(designs - nominal_design, axis=1)))
            except ValueError:
                nominal_idx = None

            if nominal_idx is not None:
                if is_1d_design and not sort:
                    nominal_sorted_value = float(designs[nominal_idx, 0])
                else:
                    matches = np.where(sorted_eigs_idx == nominal_idx)[0]
                    if matches.size > 0:
                        nominal_sorted_pos = int(matches[0])

        # Create figure with subplots and space for colorbar
        if is_1d_design and not sort:
            # For 1D designs without sorting, plot EIG directly against design variable (no second subplot needed)
            fig, ax0 = plt.subplots(figsize=(16, 6))
            ax1 = None
        else:
            # For sorted 1D or multi-dimensional designs, align heatmap beneath line plot and add a vertical colorbar
            if is_1d_design and sort:
                height_ratios = [0.65, 0.15]
            else:
                height_ratios = [0.6, 0.2]
            fig = plt.figure(figsize=(16, 6))
            gs = gridspec.GridSpec(
                2, 1,
                height_ratios=height_ratios,
                hspace=0.0
            )
            ax0 = fig.add_subplot(gs[0, 0])  # Top plot for EIG
            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)  # Bottom plot for heatmap shares x-axis
            # Reserve a dedicated colorbar axis on the right that spans both subplots
            fig.subplots_adjust(left=0.06, right=0.92)
            bbox = ax1.get_position()
            cbar_width = 0.015
            cbar_gap = 0.02  # maintain a small gap between plots and colorbar
            cbar_left = min(0.985 - cbar_width, bbox.x1 + cbar_gap)
            cbar_height = 0.6
            cbar_bottom = 0.5 - cbar_height / 2  # vertically center the colorbar
            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            plt.setp(ax0.get_xticklabels(), visible=False)
            ax0.tick_params(axis='x', which='both', length=0)
        
        # Plot all steps
        for step_data in all_steps_data:
            sorted_eigs_avg = step_data['eigs_avg'][sorted_eigs_idx]
            
            # For 1D designs without sorting, use design variable values as x-axis; otherwise use index
            if is_1d_design and not sort:
                x_vals = sorted_designs[:, 0]
            else:
                x_vals = np.arange(len(sorted_eigs_avg))
            
            # Plot error bars (gray fill) only for the sorted step
            if step_data['step_label'] == sort_step and 'eigs_std' in step_data:
                sorted_eigs_std = step_data['eigs_std'][sorted_eigs_idx]
                ax0.fill_between(
                    x_vals,
                    sorted_eigs_avg - sorted_eigs_std,
                    sorted_eigs_avg + sorted_eigs_std,
                    color='gray', alpha=0.3, zorder=1
                )
            
            # Plot EIG (main line)
            if len(all_steps_data) > 1:
                # Multiple steps: show step number in label
                line_label = f"Step {step_data['step']}"
            else:
                # Single step: show rank
                sort_label = "Sorted" if sort else None
                if sort_label is not None:
                    line_label = f"{sort_label} EIG"
                else:
                    line_label = f"EIG"
            
            # Convert color array to tuple for matplotlib
            plot_color = tuple(step_data['color']) if isinstance(step_data['color'], np.ndarray) else step_data['color']
            ax0.plot(x_vals, sorted_eigs_avg, label=line_label, 
                    color=plot_color, linestyle='-', linewidth=2.5, alpha=1.0 if step_data['step_label'] == sort_step else 0.6, zorder=5)
            
            # Plot nominal EIG for the sorting step
            if step_data['step_label'] == sort_step and include_nominal:
                ax0.axhline(y=step_data['nominal'], color='tab:blue', linestyle='--', 
                           label='Nominal EIG', linewidth=2, zorder=10)
        
        # Add vertical orange dotted line and orange dot at optimal EIG
        # Get the data for the sort_step to find optimal EIG
        sort_step_data = next((d for d in all_steps_data if d['step_label'] == sort_step), None)
        if sort_step_data is not None:
            sorted_eigs_avg = sort_step_data['eigs_avg'][sorted_eigs_idx]
            # Determine x_vals for the sort_step
            if is_1d_design and not sort:
                x_vals_optimal = sorted_designs[:, 0]
            else:
                x_vals_optimal = np.arange(len(sorted_eigs_avg))
            
            # Find optimal EIG position
            optimal_idx = np.argmax(sorted_eigs_avg)
            optimal_x = x_vals_optimal[optimal_idx]
            optimal_y = sorted_eigs_avg[optimal_idx]
            
            # Plot vertical orange dotted line
            ax0.axvline(optimal_x, color='tab:orange', linestyle=':', linewidth=2, 
                       zorder=8)
            # Plot orange dot at optimal position
            ax0.plot(optimal_x, optimal_y, 'o', color='tab:orange', markersize=8, zorder=9, 
                    label='Optimal Design')
        
        # Set x-axis label and limits based on design dimensionality and sorting
        if is_1d_design and not sort:
            ax0.set_xlabel(f'${self.experiment.design_labels[0]}$', fontsize=12, weight='bold')
            ax0.set_xlim(x_vals.min(), x_vals.max())
        else:
            ax0.set_xlim(-0.5, len(sorted_eigs_avg) - 0.5)
        
        ax0.set_ylabel("Expected Information Gain [bits]", fontsize=12, weight='bold')
        '''
        if nominal_sorted_pos is not None:
            if is_1d_design and not sort:
                ax0.axvline(nominal_sorted_value, color='black', linestyle=':', linewidth=1.5, label='Nearest Nominal Design')
            elif nominal_sorted_pos is not None:
                ax0.axvline(nominal_sorted_pos, color='black', linestyle=':', linewidth=1.5, label='Nearest Nominal Design')
        '''
        ax0.legend(loc='lower left', fontsize=9, framealpha=0.9)
        
        # Plot sorted designs
        if is_1d_design and sort:
            # For 1D sorted designs, visualize as a single-row heatmap to highlight values
            im = ax1.imshow(sorted_designs.T, aspect='auto', cmap='viridis')
            ax1.set_ylabel('')
            ax1.set_yticks([0])
            ax1.set_yticklabels([f'${self.experiment.design_labels[0]}$'])
            ax1.tick_params(axis='y', length=0)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(True)
            ax1.spines['left'].set_visible(True)
            ax1.set_xlim(-0.5, len(sorted_designs) - 0.5)
            if len(all_steps_data) > 1:
                sort_step_number = next((step_data['step'] for step_data in all_steps_data 
                                        if step_data['step_label'] == sort_step), sort_step)
                xlabel = f"Design Index (sorted by reference step {sort_step_number} EIG)"
            else:
                xlabel = "Design Index (sorted by EIG)"
            ax1.set_xlabel(xlabel, fontsize=12, weight='bold')
            ax1.margins(x=0)
            plt.setp(ax1.get_xticklabels(), rotation=0)
            if nominal_sorted_pos is not None:
                ax1.axvline(nominal_sorted_pos, color='black', linestyle=':', linewidth=1.5)
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Design Value', labelpad=10, fontsize=12, weight='bold')
            ax0.spines['bottom'].set_visible(False)
            ax1.spines['bottom'].set_visible(True)
        elif not is_1d_design:
            # For multi-dimensional designs, use heatmap with colorbar
            # Use ratio to nominal design colormapping (same as compare_best_designs)
            nominal_design = self.experiment.nominal_design.cpu().numpy()
            
            # Check for zeros in nominal_design to avoid division by zero
            if np.any(nominal_design == 0):
                # Fallback to absolute values if nominal has zeros
                plot_data = sorted_designs.T
                use_relative_colors = False
                cmap = 'viridis'
            else:
                # Simple ratio: designs / nominal_design
                # Ratio > 1.0 means larger than nominal, Ratio < 1.0 means smaller than nominal
                # Ensure proper broadcasting: sorted_designs is (n_designs, n_dims), nominal_design is (n_dims,)
                plot_data = sorted_designs / nominal_design[np.newaxis, :]
                plot_data = plot_data.T  # Transpose for imshow
                use_relative_colors = True
                cmap = 'RdBu'  # Red for larger ratios (>1.0), Blue for smaller ratios (<1.0)
            
            if use_relative_colors:
                # Use TwoSlopeNorm to center the colormap at 1.0
                from matplotlib.colors import TwoSlopeNorm
                vmin = plot_data.min()
                vmax = plot_data.max()
                norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
                
                im = ax1.imshow(plot_data, aspect='auto', cmap=cmap, norm=norm)
            else:
                im = ax1.imshow(plot_data, aspect='auto', cmap=cmap)
            
            if len(all_steps_data) > 1:
                # Find the actual step number used for sorting
                sort_step_number = next((step_data['step'] for step_data in all_steps_data 
                                        if step_data['step_label'] == sort_step), sort_step)
                if sort:
                    xlabel = f"Design Index (sorted by reference step {sort_step_number} EIG)"
                else:
                    xlabel = f"Design Index"
            else:
                if sort:
                    xlabel = "Design Index (sorted by EIG)"
                else:
                    xlabel = "Design Index"
            ax1.set_xlabel(xlabel, fontsize=12, weight='bold')
            ax1.set_yticks(np.arange(len(self.experiment.design_labels)), [f'${label}$' for label in self.experiment.design_labels])
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(True)
            ax1.spines['left'].set_visible(True)
            ax1.tick_params(axis='y', length=0)
            ax1.set_xlim(-0.5, len(sorted_designs) - 0.5)
            ax1.set_ylabel('')
            ax1.margins(x=0)
            # Restore x tick labels only on the heatmap axis
            plt.setp(ax1.get_xticklabels(), rotation=0)

            # Add colorbar spanning the full height
            cbar = fig.colorbar(im, cax=cbar_ax)
            if use_relative_colors:
                cbar.set_label('Ratio to Nominal Design', labelpad=10, fontsize=12, weight='bold')
                # Set symmetric tick marks around 1.0 for consistent display on both red and blue sides
                # Calculate the maximum deviation from 1.0 to ensure symmetric range
                max_deviation = max(abs(vmin - 1.0), abs(vmax - 1.0))
                
                # Create symmetric tick marks with consistent spacing on both sides
                # Number of major ticks per side (excluding 1.0)
                n_ticks_per_side = 4
                
                # Generate ticks symmetrically around 1.0
                # Create evenly spaced ticks from 1.0 to the symmetric bounds
                symmetric_min = max(vmin, 1.0 - max_deviation)
                symmetric_max = min(vmax, 1.0 + max_deviation)
                
                # Generate ticks above 1.0
                if symmetric_max > 1.0:
                    positive_ticks = np.linspace(1.0, symmetric_max, n_ticks_per_side + 1)[1:]  # Exclude 1.0
                    positive_ticks = positive_ticks[positive_ticks <= vmax]
                else:
                    positive_ticks = np.array([])
                
                # Generate ticks below 1.0 with the same spacing (mirrored)
                if symmetric_min < 1.0:
                    if len(positive_ticks) > 0:
                        # Mirror the spacing: use the same interval size as positive side
                        spacing = positive_ticks[0] - 1.0
                        # Create the same number of negative ticks as positive ticks
                        n_negative = len(positive_ticks)
                        negative_ticks = 1.0 - np.arange(1, n_negative + 1) * spacing
                        negative_ticks = negative_ticks[negative_ticks >= vmin]
                    else:
                        # If no positive ticks, create evenly spaced ticks below 1.0
                        negative_ticks = np.linspace(symmetric_min, 1.0, n_ticks_per_side + 1)[:-1]  # Exclude 1.0
                        negative_ticks = negative_ticks[negative_ticks >= vmin]
                else:
                    negative_ticks = np.array([])
                
                # Combine all ticks: negative, 1.0, and positive
                all_ticks = np.concatenate([negative_ticks, [1.0], positive_ticks])
                all_ticks = np.unique(all_ticks)  # Remove duplicates
                all_ticks = np.clip(all_ticks, vmin, vmax)  # Ensure all ticks are within bounds
                all_ticks = np.sort(all_ticks)  # Final sort
                
                # Format tick labels with appropriate precision
                cbar.set_ticks(all_ticks)
                cbar.set_ticklabels([f'{tick:.2f}' for tick in all_ticks])
            else:
                cbar.set_label('Design Value', labelpad=10, fontsize=12, weight='bold')
            ax0.spines['bottom'].set_visible(False)
            ax1.spines['bottom'].set_visible(True)
        
        # Add overall title
        sort_title = "Sorted" if sort else None
        if sort_title is not None:
            title = f'{sort_title} EIG per Design (N={self.n_evals} Evaluations)'
        else:
            title = f'EIG per Design (N={self.n_evals} Evaluations)'
        if self.display_run:
            title += f' - Run: {self.run_id[:8]}'
        if not (is_1d_design and not sort):
            # Subplots already adjusted earlier
            pass
        else:
            fig.subplots_adjust(left=0.06, right=0.98)
        fig.suptitle(title, fontsize=16, y=0.95, weight='bold')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_figure(f"{self.save_path}/plots/eig_designs_{timestamp}.png", fig=fig, dpi=400)
        
        plt.show()

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

        eig_data_save_path = f"{self.save_path}/eig_data_{self.timestamp}.json"
        with open(eig_data_save_path, "w") as f:
            json.dump(self.eig_data, f, indent=2)
        print(f"Saved EIG data to {eig_data_save_path}")

        # Update timing after EIG computation
        self._update_runtime()

        try:
            if self.cosmo_exp == 'num_tracers':
                self.generate_posterior(step=eval_step, levels=self.levels)
                self._update_runtime()
            self._update_runtime()
        except Exception as e:
            traceback.print_exc()

        try:
            #self.posterior_steps(steps=[self.total_steps//4, self.total_steps//2, self.total_steps*3//4, 'last'])
            self._update_runtime()
        except Exception as e:
            traceback.print_exc()
        
        try:
            self.eig_designs(steps=['last'], sort=self.sort, include_nominal=self.include_nominal)
            self._update_runtime()
        except Exception as e:
            traceback.print_exc()

        try:
            if self.cosmo_exp == 'num_tracers':
                self.design_comparison(step=eval_step, log_scale=True, use_fractional=False)
                self._update_runtime()
        except Exception as e:
            traceback.print_exc()

        try:
            if self.cosmo_exp == 'num_tracers':
                self.sample_posterior(step=eval_step, levels=[0.68], num_data_samples=20, central=True, batch_size=self.batch_size)
                self._update_runtime()
        except Exception as e:
            traceback.print_exc()
        
        if self.eig_file_path is None:
            # Save combined eig_data at the end of run
            self.eig_data['status'] = 'complete'  # Mark as complete when evaluation finishes
            eig_data_save_path = f"{self.save_path}/eig_data_{self.timestamp}.json"
            with open(eig_data_save_path, "w") as f:
                json.dump(self.eig_data, f, indent=2)
            print(f"Saved EIG data to {eig_data_save_path}")
            
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
    parser.add_argument('--design_lower', type=parse_float_or_list, default=0.05, help='Lowest design value (float or JSON list)')
    parser.add_argument('--design_upper', type=parse_float_or_list, default=None, help='Highest design value (float or JSON list)')
    parser.add_argument('--design_step', type=parse_float_or_list, default=0.05, help='Step size for design grid (float or JSON list)')
    parser.add_argument('--design_sum_lower', type=float, default=1.0, help='Lower bound for design sum')
    parser.add_argument('--design_sum_upper', type=float, default=1.0, help='Upper bound for design sum')
    parser.add_argument('--design_chunk_size', type=int, default=None, help='Number of designs per chunk (None = use all designs)')
    parser.add_argument('--n_evals', type=int, default=5, help='Number of evaluations to average over')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Seed for evaluation')
    parser.add_argument('--param_space', type=str, default='physical', help='Parameter space to use for evaluation')
    parser.add_argument('--input_designs', type=str, default=None, help='Evaluate specific design(s). Can be: 1) A JSON file path (e.g., designs.json), 2) A JSON string with single design [x1,...,xn] or multiple designs [[x1,...,xn], [y1,...,yn], ...]. If provided, overrides grid generation parameters (default: None)')
    parser.add_argument('--profile', action='store_true', help='Enable profiling of methods')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-sort', dest='sort', action='store_false', help='Disable sorting of designs by EIG (default: sorting enabled)')
    parser.add_argument('--include_nominal', action='store_true', default=False, help='Include nominal EIG in eig_designs plot (default: False)')
    parser.add_argument('--eig_file_path', type=str, default=None, help='Path to previously calculated eig_data JSON file to load instead of recalculating')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for sample_posterior to reduce memory usage (default: 1)')
    parser.add_argument('--particle_batch_size', type=int, default=None, help='Batch size for processing particles in LikelihoodDataset to reduce memory usage (default: None to use all particles)')

    args = parser.parse_args()

    valid_params = inspect.signature(Evaluator.__init__).parameters.keys()
    valid_params = [k for k in valid_params if k != 'self']
    eval_args = {k: v for k, v in vars(args).items() if k in valid_params}
    if eval_args['input_designs'] is not None:
        # Handle input_design - can be a JSON file path, JSON string, or already parsed
        if isinstance(eval_args['input_designs'], str):
            input_design_str = eval_args['input_designs'].strip()
            
            # Check if it's the special "nominal" keyword
            if input_design_str.lower() == 'nominal':
                eval_args['input_designs'] = 'nominal'
            # Check if it's a file path (ends with .json and file exists, or just a valid file path)
            elif (input_design_str.endswith('.json') or input_design_str.endswith('.JSON')) and os.path.isfile(input_design_str):
                # Read from JSON file
                try:
                    with open(input_design_str, 'r') as f:
                        parsed_value = json.load(f)
                    eval_args['input_designs'] = parsed_value
                    print(f"Loaded input_designs from file: {input_design_str}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON file {input_design_str}: {e}")
                except Exception as e:
                    raise ValueError(f"Failed to read file {input_design_str}: {e}")
            else:
                # Try to parse as JSON string
                # Replace newlines and multiple spaces with single spaces for better JSON parsing
                input_design_str = re.sub(r'\s+', ' ', input_design_str)
                try:
                    parsed_value = json.loads(input_design_str)
                    eval_args['input_designs'] = parsed_value
                except json.JSONDecodeError as e:
                    # Show more context in error message
                    error_pos = e.pos if hasattr(e, 'pos') else 'unknown'
                    context_start = max(0, error_pos - 50)
                    context_end = min(len(input_design_str), error_pos + 50)
                    context = input_design_str[context_start:context_end]
                    raise ValueError(
                        f"Failed to parse input_designs as JSON at position {error_pos}.\n"
                        f"Error: {e.msg}\n"
                        f"Context: ...{context}...\n"
                        f"Full input (first 500 chars): {input_design_str[:500]}\n"
                        f"Note: If you intended to pass a file path, make sure the file exists and ends with .json"
                    )
        # If it's already a list/dict, use it as-is
        elif isinstance(eval_args['input_designs'], (list, dict)):
            pass  # Already parsed, use as-is
        else:
            raise ValueError(f"input_designs must be a JSON file path, JSON string, list, or dict, got {type(eval_args['input_designs'])}")
    
    print(f"Evaluating with parameters:")
    print(json.dumps(eval_args, indent=2))
    evaluator = Evaluator(**eval_args)
    evaluator.run(eval_step=args.eval_step)