import sys
import os
import contextlib
import io
import json
import gc
import pickle
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
from bedcosmo.entropy import (
    bits_to_nats,
    knn_entropy,
    nats_to_bits,
    plugin_entropy_from_log_probs,
)
from matplotlib.patches import Rectangle
from bedcosmo.profiling import (
    profile_method,
    profile_loop,
    profile_section,
    ProfileTimerGroup,
)
from bedcosmo.util import (
    auto_seed, init_experiment, load_model, get_runs_data,
    parse_float_or_list,
    parse_param_subsets,
    get_rng_state, parse_extra_args, render_overlay,
    get_checkpoint,
)
import mlflow
import inspect
import yaml

# Sentinel for "not computed yet", so a legitimately-None result is still cached.
_UNSET = object()


class Evaluator:
    def __init__(
            self, run_id, guide_samples=1000, design_chunk_size=None, seed=1, cosmo_exp='num_tracers',
            levels=[0.68, 0.95], global_rank=0, n_evals=10, n_particles=1000,
            param_space='physical', display_run=False, verbose=False, device="cuda:0", profile=False,
            sort=True, include_nominal=False, plot_prior=False, batch_size=1, particle_batch_size=None,
            design_args_path=None,
            experiment_args=None, nf_eig_data=None, grid_eig_data=None,
            marginal_eig_subsets=None, marginal_outer_y=8, marginal_inner_samples=200,
            marginal_knn_k=3, step_diagnostics=False,
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
        # --nf-eig-data is the unified own-role path: write target, and also load
        # target if the file already exists (resume-from-file behavior).
        self.output_path = nf_eig_data
        self.eig_file_path = nf_eig_data if (nf_eig_data and os.path.exists(nf_eig_data)) else None
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
        self.plot_prior = plot_prior
        self.batch_size = batch_size  # Batch size for sample_posterior to reduce memory usage
        self.other_eig_data = grid_eig_data

        # Marginal EIG over parameter subsets (validated against cosmo_params below,
        # once the experiment is initialized).
        self._raw_marginal_eig_subsets = marginal_eig_subsets
        self.marginal_outer_y = int(marginal_outer_y)
        self.marginal_inner_samples = int(marginal_inner_samples)
        self.marginal_knn_k = int(marginal_knn_k)
        # Design-independent H(target); computed on first use, see _target_prior_entropy.
        self._target_prior_entropy_cache = _UNSET
        self.step_diagnostics = bool(step_diagnostics)

        # Parse experiment_args override
        if isinstance(experiment_args, str):
            self.experiment_args = json.loads(experiment_args)
        elif experiment_args is None:
            self.experiment_args = {}
        else:
            self.experiment_args = dict(experiment_args)
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

        # If the run was trained with transform_input=True, load the bijector
        # state from the latest checkpoint so the evaluator's bijector matches
        # the one the flow was trained against. Without transform_input there
        # is no bijector to seed, so we skip the checkpoint peek entirely.
        eval_checkpoint = None
        if self.run_args.get("transform_input", False):
            checkpoint_dir = f"{self.save_path}/checkpoints"
            eval_checkpoint, _ = get_checkpoint(
                'last', checkpoint_dir, self.device, self.global_rank,
                self.total_steps,
            )
            if 'bijector_state' not in eval_checkpoint:
                raise RuntimeError(
                    f"Run {self.run_id} has transform_input=True but its "
                    f"checkpoint does not contain a bijector_state. The flow "
                    f"cannot be evaluated against a bijector it was not "
                    f"trained against -- retrain on current HEAD."
                )

        # Initialize experiment - it will handle input_design and generate designs accordingly
        # (single design, multiple designs, or grid)
        self.experiment = init_experiment(
            self.run_obj, self.run_args, device=self.device,
            design_args=self.design_args, global_rank=self.global_rank,
            checkpoint=eval_checkpoint,
            **self.experiment_args
        )
        if self.eig_file_path is not None and 'input_designs' in self.eig_data:
            self.input_designs = torch.tensor(self.eig_data['input_designs'], device=self.device, dtype=torch.float64)
        else:
            self.input_designs = self.experiment.designs
            # save to eig_data object
            self.eig_data['input_designs'] = self.input_designs.cpu().numpy().tolist()
            self.eig_data['nominal_design'] = self.experiment.nominal_design.cpu().numpy().tolist()

        # Validate marginal-EIG subsets against the experiment's parameter names.
        self.marginal_eig_subsets = self._normalize_marginal_subsets(self._raw_marginal_eig_subsets)
        if self.marginal_eig_subsets:
            print(f"Marginal EIG subsets: {self.marginal_eig_subsets}")

        
        # Initialize plotter for saving figures
        self.plotter = RunPlotter(run_id=self.run_id, cosmo_exp=self.cosmo_exp, experiment_args=self.experiment_args)
        # Plotting must reuse this experiment so param_bijector comes from the checkpoint,
        # not a second init_experiment() that refits the joint Gaussianizer.
        self.plotter._experiment = self.experiment
        # Plot from this run's own (live) eig_data rather than rediscovering the
        # "newest complete file" on disk, which is wrong when multiple evals share
        # the run's artifacts dir. Mutated in place by get_eig/get_marginal_eig, so
        # the reference stays current as results accumulate.
        self.plotter._eig_data_override = self.eig_data
        
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

    def _eig_data_save_path(self, timestamp):
        if self.output_path:
            return self.output_path
        return f"{self.save_path}/eig_data_{timestamp}.json"

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
        self.plotter.save_figure(g.fig, filename="posterior_samples", dpi=400, run_id=self.run_id, experiment_id=self.exp_id)

    def _compute_eig(self, flow_model, nominal_design=False, designs=None, timers=None):
        """
        Helper function that evaluates the EIG of a single flow model.

        Args:
            rank (int): Global rank number
            step (int): Step for reference
            flow_model (torch.nn.Module): The flow model to evaluate.
            nominal_design (bool): Whether to evaluate the nominal design.
            designs: Tensor of designs to evaluate. If None, determines from nominal_design or self.input_designs.
        Returns:
            tuple (eigs, entropy_terms) where eigs is per-design EIG in BITS and
            entropy_terms maps prior_entropy / posterior_entropy (also bits), as
            numpy arrays. This method is the nats -> bits boundary: nf_loss and
            the prior plug-in work in nats, and nothing downstream of here does.
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

        dataset_cm = (
            timers.track("LikelihoodDataset") if timers is not None else contextlib.nullcontext()
        )
        with dataset_cm:
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

        samples, context, prior_log_probs = dataset_result

        # Clear GPU cache after dataset creation to free memory from trace operations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Ensure all tensors are on the correct device
        samples = samples.to(device_obj)
        context = context.to(device_obj)
        if prior_log_probs is not None:
            prior_log_probs = {k: v.to(device_obj) for k, v in prior_log_probs.items()}
        flow_model = flow_model.to(device_obj)

        nfloss_cm = (
            timers.track("nf_loss") if timers is not None else contextlib.nullcontext()
        )
        with torch.no_grad():
            with nfloss_cm:
                _, posterior_entropy = nf_loss(
                    samples=samples,
                    context=context,
                    guide=flow_model,
                    experiment=self.experiment,
                    rank=0,
                    verbose_shapes=False,
                    chunk_size=(self.n_particles // 10)
                )
        # EIG = H_prior - H_post (nats). nf_loss owns only the posterior term;
        # picking the right prior is this method's job.
        prior_entropy = self._prior_entropy(prior_log_probs, posterior_entropy)
        # This is the nats -> bits seam: nf_loss must stay in nats (its agg_loss
        # is the training objective that gets backprop'd, and rescaling it would
        # silently change the effective LR and break comparability of every
        # logged loss curve). Everything the Evaluator hands out downstream is bits.
        eigs = nats_to_bits((prior_entropy - posterior_entropy).detach().cpu())
        entropy_terms = {
            "prior_entropy": nats_to_bits(prior_entropy.detach().cpu()),
            "posterior_entropy": nats_to_bits(posterior_entropy.detach().cpu()),
        }
        # release temporary tensors to avoid graph retention and free GPU caches after each chunk
        del dataset_result, samples, context, prior_log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations complete before clearing cache

        return eigs, entropy_terms

    # =========================================================================
    # Marginal EIG over parameter subsets
    # =========================================================================

    @staticmethod
    def _subset_id(subset):
        """Stable, filesystem-friendly id for a parameter subset."""
        return "+".join(subset)

    def _normalize_marginal_subsets(self, subsets):
        """Coerce the requested subsets into a validated list-of-lists of param names.

        Accepts None, a flat list of names (single subset), or a list of lists.
        Drops names not present in experiment.cosmo_params (with a warning) and
        discards empty subsets.
        """
        if not subsets:
            return []
        # A flat list of strings is treated as a single subset.
        if all(isinstance(s, str) for s in subsets):
            subsets = [subsets]

        cosmo_params = self.experiment.cosmo_params
        normalized = []
        for subset in subsets:
            valid = [p for p in subset if p in cosmo_params]
            missing = [p for p in subset if p not in cosmo_params]
            if missing:
                print(
                    f"  Warning: marginal-EIG subset {subset} references unknown "
                    f"parameters {missing}; ignoring them. Available: {cosmo_params}"
                )
            if valid:
                normalized.append(valid)
        return normalized

    def _marginal_knn_space(self) -> str:
        """Coordinate system for k-NN marginal entropy.

        Per-parameter (marginal) bijectors (bb, bbt): unconstrained flow space,
        matching joint nf_loss. Joint empirical Gaussianizer (empirical): physical
        space so subset marginals are in interpretable units.
        """
        exp = self.experiment
        if getattr(exp, "transform_input", False) and not exp._uses_joint_transform():
            return "unconstrained"
        return "physical"

    def _marginal_knn_coords_from_physical(self, physical: np.ndarray) -> np.ndarray:
        if self._marginal_knn_space() == "physical":
            return physical
        t = torch.tensor(physical, dtype=torch.float64, device=self.experiment.device)
        u = self.experiment.params_to_unconstrained(t)
        return u.detach().cpu().numpy()

    def _marginal_flow_samples_to_knn_coords(self, samples: torch.Tensor) -> torch.Tensor:
        """Guide samples -> k-NN coordinate space, clamped to the prior bounds.

        For plotting/diagnostic callers, where clamping is harmless. Estimator
        paths must use :meth:`_marginal_knn_coords_and_mask` instead -- see the
        note on :meth:`BaseExperiment._sanitize_physical_samples`.
        """
        if self._marginal_knn_space() == "unconstrained":
            return samples
        return self._marginal_inner_to_physical(samples)

    def _marginal_knn_coords_and_mask(self, samples: torch.Tensor):
        """Guide samples -> ``(coords, valid)`` for k-NN, *without* clamping.

        ``valid`` is an element-wise mask shaped like ``coords``; rows failing it
        (non-finite, or outside an ``EmpiricalPrior``'s support) must be dropped
        before neighbor-based estimation rather than clamped, which would stack
        them all onto one identical value and collapse the k-NN estimate.
        """
        if self._marginal_knn_space() == "unconstrained":
            return samples, torch.isfinite(samples)
        physical = self._marginal_inner_to_physical(samples, sanitize=False)
        return physical, self.experiment._physical_samples_valid_mask(physical)

    def _marginal_inner_to_physical(self, samples, *, sanitize: bool = True):
        """Transform guide samples to the same (physical) space as the prior samples.

        Mirrors BaseExperiment.get_guide_samples so the marginal posterior and
        marginal prior entropies are estimated in a common coordinate system.
        Operates on tensors of shape (..., n_params).

        Args:
            sanitize: Clamp empirical-prior params to their support. Leave True
                for presentation; pass False in estimator paths and drop invalid
                rows via :meth:`_marginal_knn_coords_and_mask`.
        """
        if getattr(self.experiment, "transform_input", False) and self.nf_transform_output:
            samples = self.experiment.params_from_unconstrained(samples)
            if sanitize:
                samples = self.experiment._sanitize_physical_samples(samples)
        self.experiment.apply_multipliers(samples)
        return samples

    def _marginal_prior_physical_samples(self, num_samples: int) -> np.ndarray:
        """Physical cosmo-parameter matrix for marginal prior entropy estimation.

        Empirical GPU prior pools are subsampled *without replacement* so k-NN
        entropy is not biased by duplicate rows (with-replacement pool draws
        collapse neighbor distances at large ``num_samples``). Other experiments
        use :meth:`BaseExperiment.get_prior_samples`.

        Clamped to the prior bounds; estimator paths want
        :meth:`_marginal_prior_physical_and_mask` instead.
        """
        return self._marginal_prior_physical_and_mask(num_samples, sanitize=True)[0]

    def _marginal_prior_physical_and_mask(
        self, num_samples: int, *, sanitize: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prior samples plus an element-wise validity mask, *without* clamping.

        The prior counterpart to :meth:`_marginal_knn_coords_and_mask`. KDE
        smoothing kernels do not respect the catalog bounds, so a small fraction
        of pool draws land outside an ``EmpiricalPrior``'s support. Clamping
        them stacks every offender onto the identical bound value, which is
        exactly the tie pattern that collapses k-NN (see
        ``BaseExperiment._sanitize_physical_samples``). Callers reduce the mask
        over their own columns and drop the failing rows.
        """
        exp = self.experiment
        pool = getattr(exp, "prior_pool", None)
        sed_prior = getattr(exp, "sed_prior", None)
        n_want = int(num_samples)

        param_samples = None
        if sed_prior is not None or pool is not None:
            src = sed_prior.pool if sed_prior is not None else pool
            n_pool = int(src.pool.shape[0])
            n_draw = min(n_want, n_pool)
            gen = torch.Generator(device=src.pool.device)
            gen.manual_seed(int(self.seed))
            if sed_prior is not None:
                rows = sed_prior.sample_unique(n_draw, generator=gen)
            else:
                from bedcosmo.num_visits.empirical.sed_prior import sample_prior_pool_unique

                rows = sample_prior_pool_unique(pool, n_draw, generator=gen)
            params = exp._prior_rows_to_param_dict(rows, (n_draw,))
            param_samples = torch.stack(
                [params[k].squeeze(-1) for k in exp.cosmo_params], dim=-1
            )
            exp.apply_multipliers(param_samples)

        if param_samples is None:
            phys = exp.get_prior_samples(num_samples=n_want).samples
            param_samples = torch.as_tensor(phys, dtype=torch.float64)

        valid = exp._physical_samples_valid_mask(param_samples)
        if sanitize:
            param_samples = exp._sanitize_physical_samples(param_samples)
        return (
            param_samples.detach().cpu().numpy(),
            valid.detach().cpu().numpy(),
        )

    @staticmethod
    def _marginal_prior_sample_count(marginal_inner_samples: int, marginal_outer_y: int) -> int:
        """Prior rows for k-NN entropy: match posterior MC depth, modest floor."""
        return max(int(marginal_inner_samples) * int(marginal_outer_y), 4096)

    def _is_focused_target(self) -> bool:
        """True when the guide models a strict subset of ``cosmo_params``."""
        n_targets = getattr(self.experiment, "n_targets", None)
        return n_targets is not None and n_targets != len(self.experiment.cosmo_params)

    def _prior_entropy(self, prior_log_probs, posterior_entropy):
        """Per-design prior entropy H_prior for the EIG, in NATS.

        Two cases, chosen explicitly rather than by a silent fallback:

        * **Focused-target guide** -- the plug-in prior scorer is over the full
          joint, but ``EIG_target = H_prior(target) - H_post(target)``, so the
          joint prior would pair a 13-D prior entropy with a 1-D posterior
          entropy and report a large, plausible, meaningless number. Use the
          design-independent target-marginal prior entropy instead, broadcast
          across designs.
        * **Default (all-params) guide** -- the plug-in prior entropy from the
          sampled log-probs.

        Raises:
            ValueError: if the inputs for the selected case are missing. This is
                deliberately loud: every silent fallback here produces a wrong
                EIG that still looks reasonable.
        """
        if self._is_focused_target():
            h_bits = self._target_prior_entropy()
            if h_bits is None:
                raise ValueError(
                    "Focused-target run needs a target-marginal prior entropy, but "
                    "_target_prior_entropy() returned None. Pairing the full-joint "
                    "prior with a target-marginal posterior yields a meaningless EIG."
                )
            return torch.full_like(posterior_entropy, float(bits_to_nats(h_bits)))
        if prior_log_probs is None:
            raise ValueError(
                "Prior log-probs are required to compute the EIG prior entropy "
                "(LikelihoodDataset must be built with evaluation=True)."
            )
        return plugin_entropy_from_log_probs(prior_log_probs, self.experiment.cosmo_params)

    def _target_prior_entropy(self) -> float | None:
        """Design-independent marginal prior entropy H(target), in BITS (memoized).

        Memoized because it is design-independent by construction but costs a
        50k-sample prior draw plus a k-NN build, and ``_prior_entropy`` asks for
        it on every EIG evaluation.

        For a focused-target guide, EIG_target = H_prior(target) - H_post(target).
        The prior scorer is over the full joint, so we estimate the
        target-marginal prior entropy once from a large prior draw, sliced to the
        target columns in the same knn-coord space the posterior entropy uses
        (so H_prior and H_post are commensurable).

        Estimator: k-NN. On the num_visits prior z-marginal this reads 0.484 bits
        against a KDE plug-in's 0.501 -- agreement to ~0.02 bits, which cannot
        move a design ranking because H(target) is a design-independent constant
        that cancels from every comparison. A parametric flow is deliberately not
        used: it underfits a skewed, bounded 1-D marginal and *overestimates* H
        (0.58-0.97) unless trained to convergence; the "exact density" win only
        pays off for the hard-trained posterior guide.

        Returns ``None`` for default (all-params) runs, which have no target
        marginal to estimate.
        """
        if self._target_prior_entropy_cache is not _UNSET:
            return self._target_prior_entropy_cache

        target_indices = getattr(self.experiment, "target_indices", None)
        if not self._is_focused_target() or target_indices is None:
            self._target_prior_entropy_cache = None
            return None
        n = max(
            self._marginal_prior_sample_count(
                self.marginal_inner_samples, self.marginal_outer_y
            ),
            int(getattr(self, "target_prior_samples", 50000)),
        )
        coords, n_drop, n_total = self._prior_marginal_coords(n, target_indices)
        if n_drop and self.verbose:
            print(
                f"  H(target) prior: dropped {n_drop}/{n_total} "
                f"({100.0 * n_drop / n_total:.2f}%) out-of-support prior draws",
                flush=True,
            )
        h_bits = knn_entropy(coords, k=self.marginal_knn_k)
        if self.verbose:
            names = [self.experiment.cosmo_params[i] for i in target_indices]
            print(
                f"  H(target={names}) prior via k-NN = {h_bits:.3f} bits "
                f"(dim={coords.shape[1]}, N={coords.shape[0]})",
                flush=True,
            )
        self._target_prior_entropy_cache = h_bits
        return h_bits

    def _prior_marginal_coords(self, n_samples, indices):
        """Prior draws in k-NN coord space, sliced to ``indices``, invalid rows dropped.

        Shared by the focused-target prior entropy and the marginal-EIG prior
        entropy. Rows are dropped on the subset's *own* columns: a draw that is
        invalid in some nuisance column is still a good draw for the z marginal.

        Returns:
            ``(coords, n_dropped, n_total)`` with ``coords`` shaped ``(n_ok, |indices|)``.
        """
        prior_phys, prior_valid = self._marginal_prior_physical_and_mask(n_samples)
        coords = np.asarray(
            self._marginal_knn_coords_from_physical(prior_phys)[:, indices],
            dtype=np.float64,
        )
        if coords.ndim == 1:
            coords = coords[:, None]
        row_ok = prior_valid[:, indices].all(axis=-1)
        n_total = int(coords.shape[0])
        return coords[row_ok], int(n_total - row_ok.sum()), n_total

    def _find_nominal_design_index(self, designs=None, atol=1e-5):
        """Index of nominal design in ``designs``, or None if not present."""
        if designs is None:
            designs = self.input_designs
        nominal = self.experiment.nominal_design.detach().cpu().numpy()
        for i, row in enumerate(designs.detach().cpu().numpy()):
            if np.allclose(row, nominal, rtol=0.0, atol=atol):
                return i
        return None

    def _store_entropy_eval_results(
        self, step_data, all_prior_eval, all_post_eval, nominal_design
    ):
        """Persist per-eval and mean/std entropy arrays into step_data (bits)."""
        step_data["prior_entropies"] = [
            r.tolist() if isinstance(r, np.ndarray) else r for r in all_prior_eval
        ]
        step_data["posterior_entropies"] = [
            r.tolist() if isinstance(r, np.ndarray) else r for r in all_post_eval
        ]
        prior_mean = np.mean(np.array(all_prior_eval), axis=0)
        prior_std = np.std(np.array(all_prior_eval), axis=0)
        post_mean = np.mean(np.array(all_post_eval), axis=0)
        post_std = np.std(np.array(all_post_eval), axis=0)
        if nominal_design:
            step_data["prior_entropy_avg"] = float(prior_mean.reshape(-1)[0])
            step_data["prior_entropy_std"] = float(prior_std.reshape(-1)[0])
            step_data["posterior_entropy_avg"] = float(post_mean.reshape(-1)[0])
            step_data["posterior_entropy_std"] = float(post_std.reshape(-1)[0])
        else:
            step_data["prior_entropy_avg"] = np.atleast_1d(prior_mean).tolist()
            step_data["prior_entropy_std"] = np.atleast_1d(prior_std).tolist()
            step_data["posterior_entropy_avg"] = np.atleast_1d(post_mean).tolist()
            step_data["posterior_entropy_std"] = np.atleast_1d(post_std).tolist()

    @staticmethod
    def _first_scalar(x):
        """First element of ``x`` as a float, whether it is a scalar or an array.

        The nominal-design callers are inconsistent: one passes 0-d values
        (``full_eigs[0]``), another passes shape-``(1,)`` means over evals
        (``np.mean(..., axis=0)`` with a single design). Bare ``float()`` on the
        latter is a NumPy DeprecationWarning today and an error in a future
        release, so normalize instead of relying on the caller.
        """
        return float(np.ravel(x)[0])

    def _log_entropy_summary(
        self, eig_bits, eig_std_bits, prior_bits, prior_std, post_bits, post_std, nominal_design
    ):
        if nominal_design:
            s = self._first_scalar
            print(
                f"  Entropy (bits): H_prior={s(prior_bits):.3f}"
                f"±{s(prior_std):.3f}, "
                f"H_post={s(post_bits):.3f}±{s(post_std):.3f}, "
                f"EIG={s(eig_bits):.3f}±{s(eig_std_bits):.3f}"
            )
        else:
            print(
                f"  Mean entropy (bits): H_prior={float(np.mean(prior_bits)):.3f}"
                f"±{float(np.mean(prior_std)):.3f}, "
                f"H_post={float(np.mean(post_bits)):.3f}"
                f"±{float(np.mean(post_std)):.3f}, "
                f"mean EIG={float(np.mean(eig_bits)):.3f}"
                f"±{float(np.mean(eig_std_bits)):.3f}"
            )

    @profile_loop("per-design flow sample", total_from="n_designs")
    def _marginal_design_indices(self, n_designs):
        yield from range(n_designs)

    @profile_loop(
        "_compute_eig chunk",
        total_from="num_designs",
        on_interim_from="eig_timers",
    )
    def _eig_design_chunk_indices(self, design_chunks, num_designs, eig_timers):
        yield from design_chunks

    @profile_method
    def _marginal_posterior_entropy(self, flow_model, designs, subset_ids, subset_idx):
        """E_y[ H(theta_S | y, d) ] in bits, for each subset and each design.

        Draws M_outer outer samples y ~ p(y|d) (reusing LikelihoodDataset to build
        the [y, design] contexts), draws K_inner posterior samples from the guide
        for each context, and estimates E_y[H(q(theta_S | y, d))] by applying k-NN
        to each outer ``y`` (on ``K`` inner samples) and averaging over ``M``.

        Same estimator for every subset S, including S = all cosmo_params (full
        joint); full-joint marginal EIG should then be comparable to joint nf_loss.

        Returns:
            dict {subset_id: np.ndarray of shape (n_designs,)} in bits.
        """
        device_obj = torch.device(self.device)
        designs = designs.to(device_obj)
        n_designs = designs.shape[0]
        M = self.marginal_outer_y
        K = self.marginal_inner_samples
        k = self.marginal_knn_k

        # Outer y ~ p(y|d): context rows are [y, design], shape (M, n_designs, ctx_dim).
        with profile_section(self, "LikelihoodDataset (outer y)"):
            _, context = LikelihoodDataset(
                experiment=self.experiment,
                n_particles_per_device=M,
                device=self.device,
                evaluation=False,
                designs=designs,
            )[0]
        context = context.to(device_obj)

        out = {sid: np.zeros(n_designs) for sid in subset_ids}
        n_dropped = 0
        n_total = 0
        n_degenerate = 0
        n_ties = 0
        with torch.inference_mode():
            for j in self._marginal_design_indices(n_designs):
                ctx_j = context[:, j, :]  # (M, ctx_dim)
                coords, valid = self._marginal_knn_coords_and_mask(flow_model(ctx_j).sample((K,)))
                inner_np = coords.detach().cpu().numpy()  # (K, M, n_params)
                valid_np = valid.detach().cpu().numpy()  # (K, M, n_params)
                if self.verbose and (
                    j == 0 or (j + 1) % 20 == 0 or j == n_designs - 1
                ):
                    print(f"    Marginal posterior entropy: design {j + 1}/{n_designs}")
                for sid, idx in zip(subset_ids, subset_idx):
                    cols = inner_np[..., idx]  # (K, M, |S|)
                    # Drop draws that fall outside the subset's own support; a
                    # clamp here would stack them on one value and collapse k-NN.
                    row_ok = valid_np[..., idx].all(axis=-1)  # (K, M)
                    H_per_y = []
                    for m in range(M):
                        rows = cols[row_ok[:, m], m, :]
                        n_total += K
                        n_dropped += K - rows.shape[0]
                        if rows.shape[0] <= k + 1:
                            n_degenerate += 1
                            H_per_y.append(np.nan)
                            continue
                        # Count ties here and warn once in aggregate below: this
                        # runs n_designs * M times, so a per-call warning spams.
                        n_ties += rows.shape[0] - len(np.unique(rows, axis=0))
                        H_per_y.append(knn_entropy(rows, k=k, warn_duplicates=False))
                    out[sid][j] = (
                        float(np.nanmean(H_per_y)) if not np.all(np.isnan(H_per_y)) else np.nan
                    )
        if n_dropped:
            msg = (
                f"  Marginal posterior entropy: dropped {n_dropped}/{n_total} "
                f"({100.0 * n_dropped / n_total:.2f}%) out-of-support guide draws"
            )
            if n_degenerate:
                msg += f"; {n_degenerate} (y, subset) estimates had too few valid draws -> NaN"
            print(msg)
        if n_ties:
            # Expected at ~0.05%: Bijector._icdf_lookup clamps u to
            # [cdf_eps, 1-cdf_eps], so draws past ~3.09 sigma (cdf_eps=1e-3) all
            # map onto the CDF grid endpoint. That value is inside the prior
            # support, so the valid-mask above cannot catch it. A rate much above
            # ~0.1% means something else is stacking samples onto one value.
            print(
                f"  Marginal posterior entropy: dropped {n_ties}/{n_total} "
                f"({100.0 * n_ties / n_total:.3f}%) duplicate rows (icdf-clamp ties)"
            )
        del context
        return out

    @profile_method
    def get_marginal_eig(self, step, subsets=None, n_evals=None):
        """Compute marginal EIG over parameter subsets for all input designs.

        For a subset S:  EIG_S(d) = H[p(theta_S)] - E_y[ H[q(theta_S | y, d)] ],
        with both entropies estimated from samples (k-NN Kozachenko-Leonenko) in
        physical parameter space. When S is all cosmo_params, this is the full
        joint EIG and should be compared to joint nf_loss. Results are stored under
        ``eig_data[step_{N}]["marginal"][subset_id]`` and the JSON is re-saved.

        Subsets that already have a complete marginal block for this step (e.g.
        loaded via --nf-eig-data) are reused; only missing subsets are computed,
        so a re-run that just needs the plots skips the posterior-entropy loop.

        Args:
            step: Training step (int or "last") whose guide is evaluated.
            subsets: List-of-lists of parameter names. Defaults to
                self.marginal_eig_subsets.
            n_evals: Number of repeats to average over (defaults to self.n_evals).

        Returns:
            tuple (marginal_dict, selected_step). marginal_dict is keyed by
            subset_id, or {} if there is nothing to compute.
        """
        if subsets is None:
            subsets = self.marginal_eig_subsets
        if not subsets:
            return {}, None
        if n_evals is None:
            n_evals = self.n_evals

        # The kNN marginal path samples and column-slices the FULL joint guide.
        # A focused-target guide already yields exact per-target EIG through the
        # main eval path, and its guide output is not the full param vector, so
        # this path does not apply.
        n_targets = getattr(self.experiment, "n_targets", len(self.experiment.cosmo_params))
        if n_targets < len(self.experiment.cosmo_params):
            print(
                "Skipping marginal EIG: this is a focused-target guide "
                f"(target_params={getattr(self.experiment, 'target_params', None)}); "
                "the main eval already reports exact per-target EIG."
            )
            return {}, None

        flow_model, selected_step = load_model(
            self.experiment, step, self.run_obj,
            self.run_args, self.device, global_rank=self.global_rank,
        )
        flow_model = flow_model.to(self.device)
        flow_model.eval()

        cosmo_params = self.experiment.cosmo_params
        all_subset_ids = [self._subset_id(s) for s in subsets]
        n_designs = len(self.input_designs)
        nominal_idx = self._find_nominal_design_index()

        # Reuse any complete per-subset marginal results already present for this
        # step (e.g. loaded via --nf-eig-data), recomputing only the missing
        # subsets. Mirrors get_eig's pre-calculated-value reuse so a re-run that
        # just needs the plots does not redo the (expensive) posterior-entropy loop.
        step_key = f"step_{selected_step}"
        existing_marginal = self.eig_data.get(step_key, {}).get("marginal", {})
        marginal = {}
        subsets_to_compute = []
        for subset, sid in zip(subsets, all_subset_ids):
            prev = existing_marginal.get(sid)
            if prev is not None and prev.get("eigs_avg"):
                print(f"  Using pre-calculated marginal EIG for [{sid}] at {step_key}")
                marginal[sid] = prev
            else:
                subsets_to_compute.append(subset)

        if subsets_to_compute:
            subset_ids = [self._subset_id(s) for s in subsets_to_compute]
            subset_idx = [[cosmo_params.index(p) for p in s] for s in subsets_to_compute]
            n_prior = self._marginal_prior_sample_count(
                self.marginal_inner_samples, self.marginal_outer_y
            )

            # Marginal prior entropy (design-independent), k-NN coordinate space, bits.
            if self.verbose:
                print(f"  Marginal k-NN entropy space: {self._marginal_knn_space()}")
            prior_H = {}
            for sid, idx in zip(subset_ids, subset_idx):
                coords, n_drop, n_total = self._prior_marginal_coords(n_prior, idx)
                if n_drop and self.verbose:
                    print(
                        f"  Marginal prior entropy [{sid}]: dropped {n_drop}/{n_total} "
                        "out-of-support prior draws"
                    )
                prior_H[sid] = knn_entropy(coords, k=self.marginal_knn_k)

            prior_H_acc = {sid: [] for sid in subset_ids}
            var_eig_acc = {sid: [] for sid in subset_ids}
            nom_eig_acc = {sid: [] for sid in subset_ids}

            for eval_idx in range(int(n_evals)):
                if self.verbose:
                    print(f"  Marginal EIG evaluation {eval_idx + 1}/{int(n_evals)}")

                var_post_H = self._marginal_posterior_entropy(
                    flow_model, self.input_designs, subset_ids, subset_idx
                )
                if nominal_idx is not None:
                    nom_post_H = {
                        sid: np.array([var_post_H[sid][nominal_idx]]) for sid in subset_ids
                    }
                else:
                    nom_post_H = self._marginal_posterior_entropy(
                        flow_model,
                        self.experiment.nominal_design.unsqueeze(0),
                        subset_ids,
                        subset_idx,
                    )
                for sid in subset_ids:
                    prior_H_acc[sid].append(prior_H[sid])
                    var_eig_acc[sid].append(prior_H[sid] - var_post_H[sid])
                    nom_eig_acc[sid].append(prior_H[sid] - nom_post_H[sid][0])

            for sid, subset in zip(subset_ids, subsets_to_compute):
                var = np.array(var_eig_acc[sid])  # (n_evals, n_designs)
                eigs_avg = var.mean(axis=0)
                eigs_std = var.std(axis=0)
                nom = np.array(nom_eig_acc[sid])  # (n_evals,)
                optimal_idx = int(np.argmax(eigs_avg)) if n_designs > 0 else 0
                optimal_design = self.input_designs[optimal_idx].detach().cpu().numpy().tolist()
                marginal[sid] = {
                    "params": list(subset),
                    "prior_entropy": float(np.mean(prior_H_acc[sid])),
                    "eigs_avg": eigs_avg.tolist(),
                    "eigs_std": eigs_std.tolist(),
                    "nominal": {"eigs_avg": float(nom.mean()), "eigs_std": float(nom.std())},
                    "optimal_eig": float(eigs_avg[optimal_idx]),
                    "optimal_eig_std": float(eigs_std[optimal_idx]),
                    "optimal_design": optimal_design,
                }
                print(
                    f"  Marginal EIG [{sid}]: nominal {marginal[sid]['nominal']['eigs_avg']:.3f} bits, "
                    f"optimal {marginal[sid]['optimal_eig']:.3f} bits"
                )
        else:
            print("  All requested marginal subsets already computed; skipping recompute.")

        # Persist into the same eig_data JSON under the resolved step, merging so
        # any previously-cached subsets for this step are preserved.
        step_dict = self.eig_data.setdefault(step_key, {})
        step_dict.setdefault("marginal", {}).update(marginal)
        self.eig_data["marginal_subsets"] = [list(s) for s in subsets]
        timestamp = getattr(self, 'timestamp', None) or datetime.now().strftime('%Y%m%d_%H%M')
        eig_data_save_path = self._eig_data_save_path(timestamp)
        with open(eig_data_save_path, "w") as f:
            json.dump(self.eig_data, f, indent=2)
        print(f"Saved marginal EIG data to {eig_data_save_path}")

        # Return only the requested subsets (cached + newly computed).
        marginal = {sid: step_dict["marginal"][sid] for sid in all_subset_ids if sid in step_dict["marginal"]}

        return marginal, selected_step

    @profile_method
    def get_eig(self, step, nominal_design=False, n_evals=None):
        """
        Calculates EIG for the specified rank.
        Uses caching to avoid redundant calculations.

        Args:
            step (int): Step for caching key and to load the model.
            nominal_design (bool): If True, calculate EIG for nominal design only.
                                 If False, calculate EIGs for all designs in self.experiment.designs.
            n_evals (int, optional): Number of evaluations to average over. If None, uses self.n_evals.

        Returns:
            tuple: (result, result_std)
                - result: If nominal_design=True, float; else np.ndarray (mean over n_evals)
                - result_std: If nominal_design=True, float; else np.ndarray (std over n_evals)
        """
        if n_evals is None:
            n_evals = self.n_evals
        cache_key = f"step_{step}_particles_{self.n_particles}_nominal_{nominal_design}_evals_{n_evals}"
        
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
        
        target_n_evals = int(n_evals)
        
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
        all_prior_entropy_eval = []
        all_posterior_entropy_eval = []
        
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
                
                if "prior_entropies" in step_data and step_data["prior_entropies"]:
                    for r in step_data["prior_entropies"][: len(all_eval_results)]:
                        all_prior_entropy_eval.append(
                            np.array(r) if not isinstance(r, np.ndarray) else r
                        )
                if "posterior_entropies" in step_data and step_data["posterior_entropies"]:
                    for r in step_data["posterior_entropies"][: len(all_eval_results)]:
                        all_posterior_entropy_eval.append(
                            np.array(r) if not isinstance(r, np.ndarray) else r
                        )

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
                    if all_prior_entropy_eval and all_posterior_entropy_eval:
                        self._store_entropy_eval_results(
                            step_data,
                            all_prior_entropy_eval,
                            all_posterior_entropy_eval,
                            nominal_design,
                        )
                        prior_mean = np.mean(np.array(all_prior_entropy_eval), axis=0)
                        prior_std = np.std(np.array(all_prior_entropy_eval), axis=0)
                        post_mean = np.mean(np.array(all_posterior_entropy_eval), axis=0)
                        post_std = np.std(np.array(all_posterior_entropy_eval), axis=0)
                        if nominal_design:
                            self._log_entropy_summary(
                                result, result_std, prior_mean, prior_std, post_mean, post_std, True
                            )
                    timestamp = getattr(self, 'timestamp', None) or datetime.now().strftime('%Y%m%d_%H%M')
                    self.eig_data['status'] = 'incomplete'  # Mark as incomplete during evaluation
                    eig_data_save_path = self._eig_data_save_path(timestamp)
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
            full_prior_entropy = np.zeros(num_designs)
            full_posterior_entropy = np.zeros(num_designs)
            eig_timers = ProfileTimerGroup(self)

            for design_indices in self._eig_design_chunk_indices(
                design_chunks, num_designs, eig_timers
            ):
                # Get designs for this chunk
                chunk_designs = all_designs[design_indices].to(self.device)

                # Evaluate this chunk
                eig_result, entropy_terms = self._compute_eig(
                    flow_model,
                    nominal_design=nominal_design,
                    designs=chunk_designs,
                    timers=eig_timers,
                )
                
                # Store results for this chunk (_compute_eig already returns bits on CPU)
                eig_array = eig_result
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
                if entropy_terms is not None:
                    prior_bits = entropy_terms["prior_entropy"]
                    post_bits = entropy_terms["posterior_entropy"]
                    if len(design_indices) == 1:
                        full_prior_entropy[design_indices[0]] = float(prior_bits.reshape(-1)[0])
                        full_posterior_entropy[design_indices[0]] = float(post_bits.reshape(-1)[0])
                    else:
                        for idx, p_val, q_val in zip(design_indices, prior_bits, post_bits):
                            full_prior_entropy[idx] = float(p_val)
                            full_posterior_entropy[idx] = float(q_val)

            eig_timers.report_all()

            all_eval_results.append(full_eigs)
            all_prior_entropy_eval.append(full_prior_entropy.copy())
            all_posterior_entropy_eval.append(full_posterior_entropy.copy())
            # Save to step_data, converting numpy arrays to lists for JSON serialization
            step_data['eigs'] = [r.tolist() if isinstance(r, np.ndarray) else r for r in all_eval_results]
            self._store_entropy_eval_results(
                step_data,
                all_prior_entropy_eval,
                all_posterior_entropy_eval,
                nominal_design,
            )
            eval_prior_mean = np.mean(full_prior_entropy)
            eval_prior_std = np.std(full_prior_entropy) if num_designs > 1 else 0.0
            eval_post_mean = np.mean(full_posterior_entropy)
            eval_post_std = np.std(full_posterior_entropy) if num_designs > 1 else 0.0
            eval_eig_mean = float(np.mean(full_eigs))
            eval_eig_std = float(np.std(full_eigs)) if num_designs > 1 else 0.0
            if nominal_design:
                self._log_entropy_summary(
                    full_eigs[0],
                    eval_eig_std,
                    full_prior_entropy[0],
                    eval_prior_std,
                    full_posterior_entropy[0],
                    eval_post_std,
                    True,
                )
            else:
                self._log_entropy_summary(
                    eval_eig_mean,
                    eval_eig_std,
                    eval_prior_mean,
                    eval_prior_std,
                    eval_post_mean,
                    eval_post_std,
                    False,
                )

            # Save EIG data to file
            timestamp = getattr(self, 'timestamp', None) or datetime.now().strftime('%Y%m%d_%H%M')
            self.eig_data['status'] = 'incomplete'  # Mark as incomplete during evaluation
            eig_data_save_path = self._eig_data_save_path(timestamp)
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

        if all_prior_entropy_eval and all_posterior_entropy_eval:
            self._store_entropy_eval_results(
                step_data,
                all_prior_entropy_eval,
                all_posterior_entropy_eval,
                nominal_design,
            )
            prior_mean = np.mean(np.array(all_prior_entropy_eval), axis=0)
            prior_std = np.std(np.array(all_prior_entropy_eval), axis=0)
            post_mean = np.mean(np.array(all_posterior_entropy_eval), axis=0)
            post_std = np.std(np.array(all_posterior_entropy_eval), axis=0)
            eig_bits = result if nominal_design else np.atleast_1d(result)
            eig_std_bits = result_std if nominal_design else np.atleast_1d(result_std)
            self._log_entropy_summary(
                eig_bits, eig_std_bits, prior_mean, prior_std, post_mean, post_std, nominal_design
            )

        timestamp = getattr(self, 'timestamp', None) or datetime.now().strftime('%Y%m%d_%H%M')
        self.eig_data['status'] = 'incomplete'  # Mark as incomplete during evaluation
        eig_data_save_path = self._eig_data_save_path(timestamp)
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
        self.plotter.save_figure(fig, filename="eig_grid", dpi=400, run_id=self.run_id, experiment_id=self.exp_id)

    def compute_eig_steps(self, steps, n_evals=1):
        """
        Compute EIG at multiple training steps and store results in eig_data.

        Uses a reduced n_evals (default 1) since EIG computation is expensive
        and this is primarily for visualising how the EIG landscape evolves
        during training.

        Args:
            steps (list): Training steps to evaluate (ints or 'last').
            n_evals (int): Number of evaluations per step (default: 1).
        """
        for step in steps:
            # Resolve 'last' to actual step number
            _, selected_step = load_model(
                self.experiment, step, self.run_obj,
                self.run_args, self.device,
                global_rank=self.global_rank,
            )
            step_key = f"step_{selected_step}"

            # Skip if this step already has variable EIG data
            existing = self.eig_data.get(step_key, {}).get('variable', {})
            if 'eigs_avg' in existing and existing['eigs_avg'] is not None:
                print(f"  Step {selected_step}: using existing EIG data")
                continue

            print(f"  Step {selected_step}: computing EIG (n_evals={n_evals})...")
            self.get_eig(step=selected_step, nominal_design=False, n_evals=n_evals)

        # Save updated eig_data (restore 'complete' status since get_eig marks it 'incomplete' during evaluation)
        self.eig_data['status'] = 'complete'
        timestamp = getattr(self, 'timestamp', None) or datetime.now().strftime('%Y%m%d_%H%M')
        eig_data_save_path = self._eig_data_save_path(timestamp)
        with open(eig_data_save_path, "w") as f:
            json.dump(self.eig_data, f, indent=2)
        print(f"Saved EIG steps data to {eig_data_save_path}")

    def run(self, eval_step=None):
        # Determine eval_step
        if eval_step is None or eval_step == 'last':
            eval_step = self.total_steps
        else:
            eval_step = int(eval_step)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        # Initialize timing at start of run
        self._update_runtime()

        # Focused-target guide: report the design-independent target-marginal prior
        # entropy up front. _prior_entropy computes it on demand (and memoizes);
        # this is purely so it appears early in the log. No-op for all-param runs.
        target_prior_bits = self._target_prior_entropy()
        if target_prior_bits is not None:
            print(
                f"Focused target {self.experiment.target_params}: "
                f"H_prior = {target_prior_bits:.3f} bits"
            )

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

        eig_data_save_path = self._eig_data_save_path(self.timestamp)
        with open(eig_data_save_path, "w") as f:
            json.dump(self.eig_data, f, indent=2)
        print(f"Saved EIG data to {eig_data_save_path}")

        # Update timing after joint EIG computation
        self._update_runtime()

        # Marginal EIG over requested parameter subsets (stored in the same JSON).
        marginal_ok = True
        if self.marginal_eig_subsets:
            try:
                self.get_marginal_eig(step=eval_step, subsets=self.marginal_eig_subsets)
                self._update_runtime()
            except Exception as e:
                marginal_ok = False
                print(f"Warning: marginal EIG computation failed: {e}")
                traceback.print_exc()

        # Mark complete only after joint EIG and any requested marginal EIG succeed.
        is_complete = self.eig_data.get('status', 'incomplete') == 'complete'
        if not is_complete and marginal_ok:
            self.eig_data['status'] = 'complete'
            eig_data_save_path = self._eig_data_save_path(self.timestamp)
            with open(eig_data_save_path, "w") as f:
                json.dump(self.eig_data, f, indent=2)
            print(f"Saved EIG data to {eig_data_save_path}")

        # Make some evaluation plots
        try:
            self.plotter.generate_posterior(
                eval_step=eval_step,
                display=['nominal', 'optimal'],
                guide_samples=self.guide_samples,
                levels=self.levels,
                plot_prior=self.plot_prior,
                transform_output=self.nf_transform_output,
            )
            self._update_runtime()
        except Exception as e:
            print(f"Warning: generate_posterior failed: {e}")
            traceback.print_exc()

        # Marginal posterior triangles + marginal EIG-vs-design plots per subset.
        for subset in self.marginal_eig_subsets:
            subset_id = self._subset_id(subset)
            try:
                self.plotter.generate_posterior(
                    eval_step=eval_step,
                    display=['nominal', 'optimal'],
                    guide_samples=self.guide_samples,
                    levels=self.levels,
                    plot_prior=self.plot_prior,
                    transform_output=self.nf_transform_output,
                    params=subset,
                    filename=f"posterior_marginal_{subset_id}",
                )
                self._update_runtime()
            except Exception as e:
                print(f"Warning: posterior_marginal ({subset_id}) failed: {e}")
                traceback.print_exc()
            step_key = f"step_{eval_step}"
            has_marginal = (
                subset_id
                in self.eig_data.get(step_key, {}).get("marginal", {})
            )
            if not has_marginal:
                print(
                    f"Warning: skipping eig_designs_marginal ({subset_id}): "
                    f"marginal EIG not in {step_key}"
                )
                continue
            try:
                self.plotter.eig_designs_marginal(
                    eval_step=eval_step,
                    subset=subset,
                    sort=self.sort,
                    include_nominal=self.include_nominal,
                )
                self._update_runtime()
            except Exception as e:
                print(f"Warning: eig_designs_marginal ({subset_id}) failed: {e}")
                traceback.print_exc()
        
        try:
            if self.cosmo_exp == 'num_tracers':
                self.plotter.design_comparison(
                    eval_step=eval_step, log_scale=True, 
                    use_fractional=False, filename="design_comparison")
                self._update_runtime()
        except Exception as e:
            print(f"Warning: design_comparison failed: {e}")
            traceback.print_exc()
        
        try:
            self.plotter.eig_designs(eval_step=eval_step, sort=self.sort, include_nominal=self.include_nominal)
            self._update_runtime()
        except Exception as e:
            print(f"Warning: eig_designs failed: {e}")
            traceback.print_exc()
        
        if self.step_diagnostics:
            try:
                # Plot posterior at different training steps
                steps_to_plot = [self.total_steps//4, self.total_steps//2, self.total_steps*3//4, 'last']
                self.plotter.posterior_steps(
                    steps=steps_to_plot,
                    levels=self.levels,
                    guide_samples=self.guide_samples,
                    filename='posterior_steps',
                    transform_output=self.nf_transform_output
                )
                self._update_runtime()
            except Exception as e:
                print(f"Warning: posterior_steps failed: {e}")
                traceback.print_exc()

            try:
                # Compute variable EIG at intermediate training steps (cheap: n_evals=1)
                eig_steps_to_compute = [self.total_steps//4, self.total_steps//2, self.total_steps*3//4]
                self.compute_eig_steps(steps=eig_steps_to_compute, n_evals=1)
                # Plot EIG across training steps (intermediate + final eval_step)
                all_eig_steps = eig_steps_to_compute + [eval_step]
                self.plotter.eig_designs(eval_step=all_eig_steps, sort=self.sort, include_nominal=self.include_nominal, filename="eig_designs_steps")
                self._update_runtime()
            except Exception as e:
                print(f"Warning: eig_designs (multi-step) failed: {e}")
                traceback.print_exc()

        try:
            render_overlay(
                own_eig_data=self.eig_data,
                own_role='nf',
                other_eig_data_path=self.other_eig_data,
                plotter=self.plotter,
                eval_step=eval_step,
                levels=self.levels,
                transform_output=self.nf_transform_output,
                include_nominal=self.include_nominal,
                sort=self.sort,
                plot_prior=self.plot_prior,
            )
        except Exception as e:
            print(f"Warning: overlay rendering failed: {e}")
            traceback.print_exc()

        print(f"Evaluation completed successfully!")

    def run_marginal(self, eval_step=None):
        """Run only the marginal EIG evaluation loop (and its per-subset plots).

        Standalone counterpart to :meth:`run` that skips the full joint EIG
        pipeline (nominal/variable EIG, posterior triangles, EIG-vs-design,
        overlays, etc.) and computes only the marginal EIG over
        ``self.marginal_eig_subsets``, persisting it into the same eig_data JSON.
        Triggered by ``./submit.sh eval <exp> <run_id> --marginal``.
        """
        # Determine eval_step
        if eval_step is None or eval_step == 'last':
            eval_step = self.total_steps
        else:
            eval_step = int(eval_step)

        if not self.marginal_eig_subsets:
            raise ValueError(
                "--marginal requires at least one parameter subset; pass "
                "--marginal-eig-subsets (e.g. '[[\"log_c_scale\",\"z\"]]') or set "
                "marginal_eig_subsets in eval_args.yaml."
            )

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        self._update_runtime()

        # Marginal EIG over requested parameter subsets (stored in the same JSON).
        self.get_marginal_eig(step=eval_step, subsets=self.marginal_eig_subsets)
        self._update_runtime()

        self.eig_data["eval_step"] = int(eval_step)
        self.eig_data['status'] = 'complete'
        eig_data_save_path = self._eig_data_save_path(self.timestamp)
        with open(eig_data_save_path, "w") as f:
            json.dump(self.eig_data, f, indent=2)
        print(f"Saved marginal EIG data to {eig_data_save_path}")

        # Marginal posterior triangles + marginal EIG-vs-design plots per subset.
        for subset in self.marginal_eig_subsets:
            subset_id = self._subset_id(subset)
            try:
                self.plotter.generate_posterior(
                    eval_step=eval_step,
                    display=['nominal', 'optimal'],
                    guide_samples=self.guide_samples,
                    levels=self.levels,
                    plot_prior=self.plot_prior,
                    transform_output=self.nf_transform_output,
                    params=subset,
                    filename=f"posterior_marginal_{subset_id}",
                )
                self._update_runtime()
            except Exception as e:
                print(f"Warning: posterior_marginal ({subset_id}) failed: {e}")
                traceback.print_exc()
            try:
                self.plotter.eig_designs_marginal(
                    eval_step=eval_step,
                    subset=subset,
                    sort=self.sort,
                    include_nominal=self.include_nominal,
                )
                self._update_runtime()
            except Exception as e:
                print(f"Warning: eig_designs_marginal ({subset_id}) failed: {e}")
                traceback.print_exc()

        print(f"Marginal EIG evaluation completed successfully!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Cosmo Experiment Evaluation')
    parser.add_argument('--run-id', type=str, default=None, help='MLflow run ID to resume training from (continues existing run with same parameters)')
    parser.add_argument('--eval-step', type=str, default='last', help='Step to evaluate (can be integer or "last")')
    parser.add_argument('--cosmo-exp', type=str, default='num_tracers', help='Cosmological experiment name')
    parser.add_argument('--levels', type=parse_float_or_list, default=0.68, help='Levels for contour plot (can be a single float or JSON list of floats)')
    parser.add_argument('--global-rank', type=int, default=0, help='Global rank to evaluate (default: 0)')
    parser.add_argument('--n-particles', type=int, default=1000, help='Number of particles to use for evaluation')
    parser.add_argument('--guide-samples', type=int, default=10000, help='Number of samples to generate from the posterior')
    parser.add_argument('--design-chunk-size', type=int, default=None, help='Number of designs per chunk (None = use all designs)')
    parser.add_argument('--n-evals', type=int, default=5, help='Number of evaluations to average over')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Seed for evaluation')
    parser.add_argument('--param-space', type=str, default='physical', help='Parameter space to use for evaluation')
    parser.add_argument('--profile', action='store_true', help='Enable profiling of methods')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-sort', dest='sort', action='store_false', help='Disable sorting of designs by EIG (default: sorting enabled)')
    parser.add_argument('--include-nominal', action='store_true', default=False, help='Include nominal EIG in eig_designs plot (default: False)')
    parser.add_argument('--plot-prior', action='store_true', default=False, help='Include prior contours on posterior triangle plots (default: False)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for sample_posterior to reduce memory usage (default: 1)')
    parser.add_argument('--particle-batch-size', type=int, default=None, help='Batch size for processing particles in LikelihoodDataset to reduce memory usage (default: None to use all particles)')
    parser.add_argument('--design-args-path', type=str, default=None, help='Path to design_args.yaml file. If None, defaults to the run\'s artifacts/design_args.yaml')
    parser.add_argument('--nf-eig-data', type=str, default=None, help='Path for this NF job to write its eig_data JSON. If the file already exists, it is loaded first (inheriting seed, n_particles, particle_batch_size, n_evals, rng_state, input_designs, and cached per-step EIGs) and then overwritten at end of run.')
    parser.add_argument('--grid-eig-data', type=str, default=None, help='Path to the sibling grid job eig_data JSON. If it is complete at end of run, overlay plots are generated.')
    parser.add_argument('--marginal-eig-subsets', type=parse_param_subsets, default=None, help='Parameter subsets for marginal EIG. JSON list-of-lists (e.g. \'[["log_c_scale","z"]]\') or semicolon/comma string (e.g. "log_c_scale,z; f1,f2").')
    parser.add_argument('--marginal-outer-y', type=int, default=8, help='Outer y ~ p(y|d) samples per design for marginal posterior entropy (default: 8)')
    parser.add_argument('--marginal-inner-samples', type=int, default=200, help='Guide samples per outer y for marginal posterior entropy (default: 200)')
    parser.add_argument('--marginal-knn-k', type=int, default=3, help='Neighbor rank k for the k-NN entropy estimator (default: 3)')
    parser.add_argument('--marginal', action='store_true', help='Run only the marginal EIG evaluation loop (and its per-subset plots), skipping the full joint EIG pipeline. Requires --marginal-eig-subsets (or marginal_eig_subsets in eval_args.yaml).')
    parser.add_argument('--step-diagnostics', action='store_true', help='Run intermediate-step diagnostics (posterior_steps and eig_designs_steps). Disabled by default.')

    args, extra_args = parser.parse_known_args()

    experiment_args = parse_extra_args(extra_args)
    if experiment_args:
        print(f"Experiment args overrides: {experiment_args}")

    valid_params = inspect.signature(Evaluator.__init__).parameters.keys()
    valid_params = [k for k in valid_params if k != 'self']
    eval_args = {k: v for k, v in vars(args).items() if k in valid_params}
    eval_args['experiment_args'] = experiment_args

    print(f"Evaluating with parameters:")
    print(json.dumps(eval_args, indent=2))
    evaluator = Evaluator(**eval_args)
    if args.marginal:
        evaluator.run_marginal(eval_step=args.eval_step)
    else:
        evaluator.run(eval_step=args.eval_step)
