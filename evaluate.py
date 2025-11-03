import sys
import os
import contextlib
import io
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

base_dir = os.environ["HOME"] + '/bed/BED_cosmo'
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
    
from pyro import distributions as dist
import numpy as np
import torch
import pandas as pd
import getdist
from getdist import plots
from bed.grid import Grid
import argparse
from plotting import *
import traceback
from pyro_oed_src import nf_loss, LikelihoodDataset
from matplotlib.patches import Rectangle
from util import *
import mlflow
import inspect
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("file:" + os.environ["SCRATCH"] + "/bed/BED_cosmo/num_tracers/mlruns")

class Evaluator:
    def __init__(
            self, run_id, guide_samples=1000, design_step=0.05, design_lower=0.05, design_upper=None,
            seed=1, cosmo_exp='num_tracers', levels=[0.68, 0.95], global_rank=0, 
            n_evals=20, n_particles=1000, param_space='physical', fixed_design=None,
            plot_ranks=True, combine_ranks='mean', verbose=False, device="cuda:0", profile=False,
            ):
        self.run_id = run_id
        print(f"\nStarting evaluation for run {self.run_id}")
        run_data_list, _, _ = get_runs_data(run_ids=self.run_id)
        if run_data_list is None:
            print(f"Run {self.run_id} not found.")
            return
        run_data = run_data_list[0]
        self.run_obj = run_data['run_obj']
        self.run_args = run_data['params']
        self.total_steps = self.run_args["total_steps"]
        self.exp_id = run_data['exp_id']
        self.cosmo_exp = cosmo_exp
        self.storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{self.cosmo_exp}"
        self.save_path = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts"
        self.guide_samples = guide_samples
        self.seed = seed
        auto_seed(self.seed) # fix random seed
        self.device = device
        self.levels = levels if isinstance(levels, list) else [levels]
        self.global_ranks = global_rank if isinstance(global_rank, list) else [global_rank]
        self.plot_ranks = plot_ranks
        self.combine_ranks = combine_ranks
        print(f"Handling multi-rank evaluation with {self.combine_ranks} method.")
        self.n_evals = n_evals
        self.n_particles = n_particles
        self.verbose = verbose
        self.profile = profile
        design_args = {
            "design_step": design_step,
            "design_lower": design_lower,
            "design_upper": design_upper,
            "fixed_design": self.run_args["fixed_design"]
        }
        self.experiment = init_experiment(self.run_obj, self.run_args, self.device, design_args=design_args)
        if fixed_design is not None:
            self.fixed_design = torch.tensor(fixed_design, device=self.device)
        elif self.run_args['fixed_design']:
            if type(self.run_args['fixed_design']) == list:
                self.fixed_design = torch.tensor(self.run_args['fixed_design'], device=self.device)
            elif self.run_args['fixed_design'] == True:
                self.fixed_design = self.experiment.nominal_design
        else:
            self.fixed_design = False
        # Cache for EIG calculations to avoid redundant computations
        self._eig_cache = {}

        self.param_space = param_space
        if self.param_space == 'physical':
            self.desi_transform_output = False
            self.nf_transform_output = True
        elif self.param_space == 'unconstrained':
            self.desi_transform_output = True
            self.nf_transform_output = False
        else:
            raise ValueError(f"Invalid parameter space: {self.param_space}")

    @profile_method
    def _eval_step(self, step, nominal_design=False, global_rank=None):
        """
        Generates samples given the nominal context (nominal design + central values).
        Args:
            step (int): Step to calculate the nominal samples for.
        """
        if global_rank is None:
            # If combine_ranks='first', only use the first rank
            if self.combine_ranks == 'first':
                global_ranks_list = [self.global_ranks[0]]
            else:
                global_ranks_list = self.global_ranks
        else:
            global_ranks_list = [global_rank]
        
        rank_samples = []
        
        if self.fixed_design is not False:
            design = self.fixed_design
        elif nominal_design:
            design = self.experiment.nominal_design
        else:
            eigs = self.get_eig(step, nominal_design=False)
            design = self.experiment.designs[np.argmax(eigs)]

        # Generate samples for each rank using the same design
        for rank in global_ranks_list:
            flow_model, _ = load_model(
                self.experiment, step, self.run_obj, 
                self.run_args, self.device, global_rank=rank
            )
            rank_samples.append(self._get_samples(design, flow_model))
        
        # Handle combining/returning samples based on combine_ranks setting
        if global_rank is not None:
            # If a specific rank was requested, return just that rank's samples
            return rank_samples[0]
        elif self.combine_ranks in ['mean', 'median'] and len(rank_samples) > 1:
            # Combine samples from all ranks by concatenating
            combined_samples_array = np.concatenate([s.samples for s in rank_samples], axis=0)
            with contextlib.redirect_stdout(io.StringIO()):
                combined_samples = getdist.MCSamples(
                    samples=combined_samples_array,
                    names=self.experiment.cosmo_params,
                    labels=self.experiment.latex_labels
                )
            return combined_samples
        elif self.combine_ranks and len(rank_samples) == 1:
            # Single rank (either only one rank total, or combine_ranks='first')
            return rank_samples[0]
        else:
            # Return list of samples (when combine_ranks is False/None)
            return rank_samples

    def _get_samples(self, design, flow_model):
        context = torch.cat([design, self.experiment.central_val], dim=-1)
        return self.experiment.get_guide_samples(flow_model, context, num_samples=self.guide_samples, transform_output=self.nf_transform_output)
    
    @profile_method
    def generate_posterior(self, step='last', display=['nominal', 'optimal']):
        """
        Generates the posterior for given type(s) of design input.

        Args:
            step (int): The checkpoint step to plot the posterior for.
            display (list): The designs to display.

        """
        print(f"Running posterior evaluation...")
        all_samples = []
        all_colors = []
        all_alphas = []
        
        # If plot_ranks=True and multiple ranks, plot individual ranks first (faded)
        if self.plot_ranks and len(self.global_ranks) > 1 and self.combine_ranks in ['mean', 'median']:
            for rank in self.global_ranks:
                if 'nominal' in display:
                    rank_nominal_samples = self._eval_step(step, nominal_design=True, global_rank=rank)
                    all_samples.append(rank_nominal_samples)
                    all_colors.append('tab:blue')
                    all_alphas.append(0.25)  # Faded for individual ranks
                
                if 'optimal' in display and self.fixed_design is False:
                    rank_optimal_samples = self._eval_step(step, global_rank=rank)
                    all_samples.append(rank_optimal_samples)
                    all_colors.append('tab:orange')
                    all_alphas.append(0.25)  # Faded for individual ranks
        
        # Now add combined samples with full alpha
        if 'nominal' in display:
            print(f"Generating posterior samples with nominal design...")
            # Sample with nominal design
            nominal_samples = self._eval_step(step, nominal_design=True)
            if self.combine_ranks:
                # nominal_samples is a single MCSamples object
                all_samples.append(nominal_samples)
                all_colors.append('tab:blue')
                all_alphas.append(1.0)
            else:
                # nominal_samples is a list of MCSamples objects (one per rank)
                all_samples.extend(nominal_samples)
                all_colors.extend(['tab:blue'] * len(nominal_samples))
                all_alphas.extend([1.0] * len(nominal_samples))
        
        if 'optimal' in display and self.fixed_design is False:
            print(f"Generating posterior samples with optimal design...")
            # Sample with optimal design
            optimal_samples = self._eval_step(step)
            if self.combine_ranks:
                # optimal_samples is a single MCSamples object
                all_samples.append(optimal_samples)
                all_colors.append('tab:orange')
                all_alphas.append(1.0)
            else:
                # optimal_samples is a list of MCSamples objects (one per rank)
                all_samples.extend(optimal_samples)
                all_colors.extend(['tab:orange'] * len(optimal_samples))
                all_alphas.extend([1.0] * len(optimal_samples))

        # Get the DESI MCMC samples
        desi_samples_gd = self.experiment.get_desi_samples(transform_output=self.desi_transform_output)
        all_samples.append(desi_samples_gd)
        all_colors.append('black')
        all_alphas.append(1.0)
        
        g = plot_posterior(all_samples, all_colors, levels=self.levels, width_inch=10, alpha=all_alphas)
        ranks_label = f"Combined {len(self.global_ranks)} Ranks" if self.combine_ranks and len(self.global_ranks) > 1 else f"{len(self.global_ranks)} Rank{'s' if len(self.global_ranks) > 1 else ''}"
        g.fig.suptitle(f"Posterior Evaluation - Run: {self.run_id[:8]} ({ranks_label}, {self.param_space.capitalize()} Space)", fontsize=16)

        # Create custom legend
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()
        custom_legend = []
        if 'nominal' in display:
            custom_legend.append(
                Line2D([0], [0], color='tab:blue', label=f'Nominal Design (NF)')
            )
        if 'optimal' in display and self.fixed_design is False:
            custom_legend.append(
                Line2D([0], [0], color='tab:orange', label=f'Optimal Design (NF)')
            )
        custom_legend.append(
            Line2D([0], [0], color='black', label=f'DESI Nominal Design (MCMC)')
        )
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96))
        leg.set_in_layout(False)
        save_figure(f"{self.save_path}/plots/posterior_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.param_space}.png", fig=g.fig, dpi=400)
    
    @profile_method
    def sample_posterior(self, step, level, num_data_samples=10, global_rank=0, central=True):
        print(f"Running sample posterior evaluation...")
        posterior_flow, _ = load_model(
            self.experiment, step, 
            self.run_obj, self.run_args, 
            device=self.device, global_rank=global_rank
            )

        # Get optimal design based on combined EIGs across ranks
        eigs = self.get_eig(step, nominal_design=False)
        optimal_design = self.experiment.designs[np.argmax(eigs)]
        pair_keys = [k for k in self.run_obj.data.metrics.keys() if k.startswith(f'nominal_area_{global_rank}_')]
        pair_names = [p.replace('nominal_area_0_', '') for p in pair_keys]
        design_areas = {'nominal': {p: [] for p in pair_names}, 'optimal': {p: [] for p in pair_names}}
        all_samples = []
        colors = []
        inputs = (('optimal', 'tab:orange', optimal_design), ('nominal', 'tab:blue', self.experiment.nominal_design))
        for design_type, color, design in inputs:
            data_idxs = np.arange(1, num_data_samples) # sample N data points
            samples_array = self.experiment.sample_params_from_data_samples(
                lexpand(design.unsqueeze(0), num_data_samples), 
                posterior_flow, 
                num_data_samples=num_data_samples, 
                num_param_samples=self.guide_samples,
                central=central,
                transform_output=self.nf_transform_output
            )

            pair_avg_areas = []
            pair_avg_areas_std = []
            for d in data_idxs:
                # Extract the numpy array from MCSamples and select the d-th data sample
                with contextlib.redirect_stdout(io.StringIO()):
                    samples_gd = getdist.MCSamples(
                        samples=samples_array[:, d, :],
                        names=self.experiment.cosmo_params,
                        labels=self.experiment.latex_labels
                    )
                all_samples.append(samples_gd)
                for p in pair_names:
                    param1, param2 = p.split('_')
                    design_areas[design_type][p].append(get_contour_area(
                        [samples_gd], level, param1, param2, design_type=design_type)[0][f'{design_type}_area_{p}']
                        )
            # Get the central samples with the nominal design indexed by the input global rank
            central_samples_gd = self._eval_step(step, nominal_design=True, global_rank=global_rank)
            all_samples.append(central_samples_gd[0])
            colors.extend([color]*len(data_idxs) + ['black'])

        g = plot_posterior(all_samples, colors, levels=[level], alpha=0.7, width_inch=10)
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()

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

                    title = f"Avg Area: {pair_avg_areas:.3f} +/- {pair_avg_areas_std:.3f}"
                    y_pos = 0.95 - 0.08 * design_idx
                    ax.text(0.05, y_pos, title, transform=ax.transAxes, fontsize=10, va='top', color=color)
        
        g.fig.set_constrained_layout(True)
        # Set title with proper positioning
        # add labels for Nominal and Optimal to total legend
        custom_legend = []
        custom_legend.append(
            Line2D([0], [0], color='tab:orange', label=f'Optimal Design')
        )
        custom_legend.append(
            Line2D([0], [0], color='tab:blue', label=f'Nominal Design')
        )
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96))
        leg.set_in_layout(False)
        g.fig.suptitle(f"Posterior (Rank {global_rank}) for {num_data_samples} Data Samples, {int(level*100)}% Credible Regions", fontsize=12)
        save_figure(f"{self.save_path}/plots/posterior_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=g.fig, dpi=400)

    def _compute_eig(self, flow_model, nominal_design=False):
        """
        Helper function that evaluates the EIG of a single flow model for the evaluation designs tensor.

        Args:
            flow_model (torch.nn.Module): The flow model to evaluate.
            nominal_design (bool): Whether to evaluate the nominal design.
        Returns:
            eigs (torch.Tensor): The EIGs for each design.
        """
        if nominal_design:
            designs = self.experiment.nominal_design.unsqueeze(0)
        elif self.fixed_design is not False:
            designs = self.fixed_design.unsqueeze(0)
        else:
            designs = self.experiment.designs
        
        samples, context, log_probs = LikelihoodDataset(
            experiment=self.experiment,
            n_particles_per_device=self.n_particles,
            device=self.device,
            evaluation=True,
            designs=designs
        )[0]

        with torch.no_grad():
            _, eigs = nf_loss(
                samples=samples,
                context=context,
                guide=flow_model,
                experiment=self.experiment,
                rank=0,
                verbose_shapes=False,
                log_probs=log_probs,
                evaluation=True
            )

        return eigs

    @profile_method
    def get_eig(self, step, nominal_design=False):
        """
        Calculates the average EIG over a batch of evaluations and multiple ranks.
        Uses caching to avoid redundant calculations.
        Loops over all global ranks and n_evals internally.

        Args:
            step (int): Step for caching key and to load the model.
            nominal_design (bool): If True, calculate EIG for nominal design only.
                                 If False, calculate EIGs for all designs (or fixed_design if set).

        Returns:
            If nominal_design=True: float - Combined nominal EIG value.
            If nominal_design=False: np.ndarray - Combined EIGs for each design (after averaging over evals and ranks).
        """
        
        # Create cache key based on model parameters and evaluation settings
        ranks_str = f"{self.global_ranks}" if len(self.global_ranks) > 1 else str(self.global_ranks[0])
        cache_key = f"step_{step}_ranks_{ranks_str}_combine_{self.combine_ranks}_evals_{self.n_evals}_particles_{self.n_particles}_nominal_{nominal_design}"
        
        if cache_key in self._eig_cache:
            print(f"Using cached EIG results for {cache_key}")
            return self._eig_cache[cache_key]

        # Collect EIG results for each rank
        rank_avg_eigs_list = []
        
        for rank in self.global_ranks:
            # Load model for this rank
            flow_model, _ = load_model(
                self.experiment, step, self.run_obj, 
                self.run_args, self.device, 
                global_rank=rank
            )
            
            # Average over n_evals for this rank
            eigs_batch = []
            
            for n in range(self.n_evals):
                if self.verbose and n % 5 == 0:
                    print(f"  EIG evaluation {n+1}/{self.n_evals} for rank {rank}")
                
                eig_result = self._compute_eig(flow_model, nominal_design=nominal_design)
                eigs_batch.append(eig_result.cpu().numpy()/np.log(2))
            
            # Average over evaluations for this rank
            eigs_batch = np.array(eigs_batch)
            if nominal_design:
                # For nominal design, result is a scalar
                avg_eig = np.mean(eigs_batch, axis=0).item()
                rank_avg_eigs_list.append(avg_eig)
            else:
                # For multiple designs, result is an array
                avg_eigs = np.mean(eigs_batch, axis=0)
                rank_avg_eigs_list.append(avg_eigs)
        
        # Combine across ranks
        if len(self.global_ranks) > 1:
            if nominal_design:
                # For nominal design, combine scalar values
                if self.combine_ranks == 'mean':
                    result = np.mean(rank_avg_eigs_list)
                elif self.combine_ranks == 'median':
                    result = np.median(rank_avg_eigs_list)
                elif self.combine_ranks == 'first':
                    result = rank_avg_eigs_list[0]
                else:
                    raise ValueError(f"Invalid combine_ranks value: {self.combine_ranks}")
            else:
                # For multiple designs, combine arrays
                rank_eigs_array = np.array(rank_avg_eigs_list)
                if self.combine_ranks == 'mean':
                    result = np.mean(rank_eigs_array, axis=0)
                elif self.combine_ranks == 'median':
                    result = np.median(rank_eigs_array, axis=0)
                elif self.combine_ranks == 'first':
                    result = rank_avg_eigs_list[0]
                else:
                    raise ValueError(f"Invalid combine_ranks value: {self.combine_ranks}")
        else:
            result = rank_avg_eigs_list[0]
        
        # Cache the results
        self._eig_cache[cache_key] = result
        
        return result

    @profile_method
    def design_comparison(self, step, width=0.2):
        """
        Plots a comparison of the nominal and optimal design.
        
        Args:
            step: The step to evaluate
            width: Width of the bars in the bar chart
        """
        print(f"Running design comparison...")
        if self.fixed_design is not False:
            design = self.fixed_design.cpu().numpy()
        else:
            # Get combined EIGs and optimal design (handles all ranks internally)
            eigs = self.get_eig(step, nominal_design=False)
            optimal_design = self.experiment.designs[np.argmax(eigs)]
            design = optimal_design.cpu().numpy()
        
        # Convert fractional values to actual number of tracers
        nominal_design_cpu = self.experiment.nominal_design.cpu().numpy()
        nominal_total_obs = self.experiment.nominal_total_obs
        
        # Both nominal_design and design are fractional values
        nominal_design_actual = nominal_design_cpu * nominal_total_obs
        design_actual = design * nominal_total_obs
        
        # Get maximum possible tracers for each class
        max_tracers = np.array([self.experiment.num_targets[target] for target in self.experiment.targets])
        
        # Set the positions for the bars
        x = np.arange(len(self.experiment.targets))  # the label locations

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
        
        bars1 = ax.bar(x - width/2, nominal_design_actual, width, label='Nominal Design', color='tab:blue')
        bars2 = ax.bar(x + width/2, design_actual, width, label=f'Optimal Design' if self.fixed_design is False else 'Input Design', color='tab:orange')
        
        # Add label for max possible tracers (manually since we used patches)
        ax.plot([], [], 'k:', linewidth=1.5, alpha=0.7, label='Max Possible Tracers')
        
        ax.set_xlabel('Tracer Class')
        ax.set_ylabel('Number of Tracers')
        ax.set_xticks(x)
        ax.set_xticklabels(self.experiment.targets)
        ax.legend()
        ax.set_yscale('log')
        plt.suptitle(f"{self.experiment.name} Design Variables", fontsize=16)
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
        if self.fixed_design is not False:
            print("Warning: Fixed design was used for training, skipping EIG grid plot.")
            return

        # Get combined EIGs, nominal EIG, and optimal design (handles all ranks internally)
        eigs = self.get_eig(step, nominal_design=False)
        optimal_design_tensor = self.experiment.designs[np.argmax(eigs)]
        optimal_design = optimal_design_tensor.cpu().numpy()
        
        # Determine label suffix based on combine_ranks
        if len(self.global_ranks) > 1:
            if self.combine_ranks == 'mean':
                label_suffix = 'Mean'
            elif self.combine_ranks == 'median':
                label_suffix = 'Median'
            elif self.combine_ranks == 'first':
                label_suffix = f'Rank {self.global_ranks[0]}'
            else:
                label_suffix = 'Combined'
        else:
            label_suffix = f'Rank {self.global_ranks[0]}'
        
        # Create figure with subplots
        fig = plt.figure(figsize=(8, 12))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.08], height_ratios=[1, 1], wspace=0.1, hspace=0.2)
        
        # Create subplots
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bottom = fig.add_subplot(gs[1, 0])
        cbar_ax_top = fig.add_subplot(gs[0, 1])
        cbar_ax_bottom = fig.add_subplot(gs[1, 1])

        # Get design variables
        designs = self.experiment.designs.cpu().numpy()
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
        ax_top.set_title(f'Design Variables')
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
        ax_bottom.set_title('Expected Information Gain')
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
        fig.suptitle(f'EIG Grid ({label_suffix}, {len(self.global_ranks)} Rank{"s" if len(self.global_ranks) > 1 else ""})', 
                    fontsize=14, y=0.995)
        
        plt.tight_layout()
        save_figure(f"{self.save_path}/plots/eig_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=fig, dpi=400)

    @profile_method
    def posterior_steps(self, steps, level=0.68):
        """
        Plots posterior distributions at different training steps for a single run.
        
        Args:
            steps (list): List of steps to plot. Can include 'last' or 'loss_best' as special values.
            level (float): Contour level to plot.
        """
        print(f"Running posterior steps evaluation...")
        colors = plt.cm.viridis_r(np.linspace(0, 1, len(steps)))
        
        all_samples = []
        all_colors = []
        custom_legend = []
        # Use first global_rank for pair_keys (all ranks should have the same structure)
        first_rank = self.global_ranks[0] if isinstance(self.global_ranks, list) else self.global_ranks
        pair_keys = [k for k in self.run_obj.data.metrics.keys() if k.startswith(f'nominal_area_{first_rank}_')]
        pair_names = [p.replace(f'nominal_area_{first_rank}_', '') for p in pair_keys]
        checkpoint_dir = f'{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/checkpoints'
        if not os.path.isdir(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found for run {self.run_id}, skipping. Path: {checkpoint_dir}")
            return
        for i, step in enumerate(steps):
            samples = self._eval_step(step, nominal_design=True)
            # Convert RGBA color to hex string before extending
            color_hex = matplotlib.colors.to_hex(colors[i % len(colors)])
            if self.combine_ranks:
                # samples is a single MCSamples object
                all_samples.append(samples)
                all_colors.append(color_hex)
            else:
                # samples is a list of MCSamples objects
                all_samples.extend(samples)
                all_colors.extend([color_hex] * len(samples))
            if step == 'last':
                step_label = self.run_args["total_steps"]
            elif step == 'loss_best':
                step_label = 'Best Loss'
            else:
                step_label = step
            custom_legend.append(
                Line2D([0], [0], color=color_hex, 
                        label=f'Step {step_label}')
            )
        desi_samples_gd = self.experiment.get_desi_samples(transform_output=self.desi_transform_output)
        all_samples.append(desi_samples_gd)
        all_colors.append('black')  
        g = plot_posterior(all_samples, all_colors, levels=[level], width_inch=12)
        # Remove existing legends if any
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()

        custom_legend.append(
            Line2D([0], [0], color='black', 
                label=f'DESI')
        )
        
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96), title=f'{int(level*100)}% Level')
        leg.set_in_layout(False)
        g.fig.suptitle(f"Posterior Steps for Run: {self.run_id[:8]}", 
                      fontsize=12)
        
        save_figure(f"{self.save_path}/plots/posterior_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=g.fig, dpi=400)

    @profile_method
    def sorted_eig_designs(self, steps='last', sort_step=None):
        """
        Plots sorted EIG values and corresponding designs using the evaluator's EIG calculations.
        Can plot single or multiple training steps.
        
        Args:
            steps (int, str, or list): Step(s) to evaluate. Can be:
                                      - Single int/str (e.g., 5000 or 'last')
                                      - List of steps (e.g., [2500, 5000, 7500, 'last'])
            sort_step (int, str, or None): Which step to use for sorting designs. Must be in `steps`.
                                          If None, uses 'last' if present, otherwise the largest step.
        """
        # Convert single step to list for unified handling
        if not isinstance(steps, list):
            steps = [steps]
        
        print(f"Running sorted EIG and designs plot for step(s) {steps}...")

        if self.fixed_design is not False:
            print("Warning: Fixed design was used for training, skipping sorted EIG plot.")
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
        
        # Create color gradient for multiple steps (yellow to orange)
        if len(steps) > 1:
            from matplotlib.colors import LinearSegmentedColormap
            tab_orange = np.array([1.0, 0.498, 0.055])  # RGB for tab:orange
            yellow = np.array([1.0, 0.933, 0.0])         # RGB for bright yellow
            
            n_steps = len(steps)
            colors = np.array([yellow * (1 - i/(n_steps-1)) + tab_orange * (i/(n_steps-1)) 
                              for i in range(n_steps)])
        else:
            colors = [np.array([1.0, 0.498, 0.055])]  # Just tab:orange for single step
        
        # Store EIG data for all steps
        all_steps_data = []
        sorted_eigs_idx = None
        
        for step_idx, s in enumerate(steps):
            flow_model, selected_step = load_model(
                self.experiment, s, self.run_obj, 
                self.run_args, self.device, 
                global_rank=self.global_ranks[0]  # Just need model for one rank to get selected_step
            )
            
            # Get combined EIG result (handles all ranks internally)
            combined_eigs = self.get_eig(selected_step, nominal_design=False)
            combined_nominal = self.get_eig(selected_step, nominal_design=True)
            
            # Get individual rank EIGs for plotting (if needed)
            rank_eigs_list = []
            if self.plot_ranks and len(self.global_ranks) > 1 and self.combine_ranks != 'first':
                # Need individual rank EIGs for faded plotting
                # Calculate individually per rank (simplified - just one eval per rank for speed)
                for rank in self.global_ranks:
                    rank_flow_model, _ = load_model(
                        self.experiment, selected_step, self.run_obj, 
                        self.run_args, self.device, global_rank=rank
                    )
                    # Single eval for individual rank (not averaged)
                    eigs = self._compute_eig(rank_flow_model, nominal_design=False).cpu().numpy() / np.log(2)
                    rank_eigs_list.append(eigs)
            else:
                # If not plotting individual ranks, just use combined result
                rank_eigs_list = [combined_eigs]
            
            # Store data for this step
            all_steps_data.append({
                'step': selected_step,
                'step_label': s,
                'combined_eigs': combined_eigs,
                'rank_eigs_list': rank_eigs_list,
                'nominal': combined_nominal,
                'color': colors[step_idx]
            })
            
            # Use the specified sort_step to determine sorting order
            if s == sort_step:
                sorted_eigs_idx = np.argsort(combined_eigs)[::-1]
        
        # Get designs and sort them according to sorted EIG indices
        designs = self.experiment.designs.cpu().numpy()
        sorted_designs = designs[sorted_eigs_idx]
        
        # Create figure with subplots and space for colorbar
        fig = plt.figure(figsize=(22, 6))
        gs = gridspec.GridSpec(2, 2, height_ratios=[0.5, 0.1], width_ratios=[1, 0.02], hspace=0.2, wspace=0.1)
        
        # Create subplots
        ax0 = fig.add_subplot(gs[0, 0])  # Top plot for EIG
        ax1 = fig.add_subplot(gs[1, 0])  # Bottom plot for heatmap
        cbar_ax = fig.add_subplot(gs[:, 1])  # Colorbar spanning both plots
        
        # Plot all steps with sorted x-axis
        for step_data in all_steps_data:
            sorted_combined_eigs = step_data['combined_eigs'][sorted_eigs_idx]
            
            # Plot individual ranks if requested (faded)
            if self.plot_ranks and len(self.global_ranks) > 1 and self.combine_ranks != 'first':
                for rank_eigs in step_data['rank_eigs_list']:
                    sorted_rank_eigs = rank_eigs[sorted_eigs_idx]
                    ax0.plot(sorted_rank_eigs, color=step_data['color'], 
                            linestyle='-', alpha=0.25, linewidth=1.0)
            
            # Plot combined EIG (main line)
            if len(all_steps_data) > 1:
                # Multiple steps: show step number in label
                line_label = f"Step {step_data['step']}"
            else:
                # Single step: show rank aggregation method
                if len(self.global_ranks) > 1:
                    if self.combine_ranks == 'first':
                        line_label = f"Sorted EIG (Rank {self.global_ranks[0]})"
                    else:
                        line_label = f"Sorted EIG ({self.combine_ranks.capitalize()})"
                else:
                    line_label = "Sorted EIG"
            
            ax0.plot(sorted_combined_eigs, label=line_label, 
                    color=step_data['color'], linestyle='-', linewidth=2.5, alpha=1.0, zorder=5)
            
            # Plot nominal EIG for the sorting step
            if step_data['step_label'] == sort_step:
                ax0.axhline(y=step_data['nominal'], color='black', linestyle='--', 
                           label='Nominal EIG', linewidth=2, zorder=10)
        
        ax0.set_xlim(0, len(sorted_combined_eigs))
        ax0.set_ylabel("Expected Information Gain [bits]", fontsize=11)
        ax0.legend(loc='lower left', fontsize=9, framealpha=0.9)
        
        # Plot sorted designs
        im = ax1.imshow(sorted_designs.T, aspect='auto', cmap='viridis')
        if len(all_steps_data) > 1:
            # Find the actual step number used for sorting
            sort_step_number = next((step_data['step'] for step_data in all_steps_data 
                                    if step_data['step_label'] == sort_step), sort_step)
            xlabel = f"Design Index (sorted by reference step {sort_step_number})"
        else:
            xlabel = "Design Index"
        ax1.set_xlabel(xlabel, fontsize=11)
        ax1.set_yticks(np.arange(len(self.experiment.targets)), self.experiment.targets)
        
        # Add colorbar spanning the full height
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Design Value')
        
        # Add overall title
        title = f'Sorted EIG Evaluation - Run: {self.run_id[:8]}'
        fig.suptitle(title, fontsize=14, y=0.98)
        
        save_figure(f"{self.save_path}/plots/sorted_eig_designs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=fig, dpi=400)
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Number Tracers Training')
    parser.add_argument('--run_id', type=str, default=None, help='MLflow run ID to resume training from (continues existing run with same parameters)')
    parser.add_argument('--eval_step', type=str, default='last', help='Step to evaluate (can be integer or "last")')
    parser.add_argument('--cosmo_exp', type=str, default='num_tracers', help='Cosmological model set to use from run_args.json')
    parser.add_argument('--levels', type=float, default=0.68, help='Levels for contour plot')
    parser.add_argument('--global_rank', type=str, default='0', help='List of global ranks to evaluate')
    parser.add_argument('--n_particles', type=int, default=1000, help='Number of particles to use for evaluation')
    parser.add_argument('--guide_samples', type=int, default=10000, help='Number of samples to generate from the posterior')
    parser.add_argument('--design_lower', type=parse_float_or_list, default=0.05, help='Lowest design fraction (float or JSON list)')
    parser.add_argument('--design_step', type=parse_float_or_list, default=0.05, help='Step size for design grid (float or JSON list)')
    parser.add_argument('--n_evals', type=int, default=20, help='Number of evaluations to average over')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--eval_seed', type=int, default=1, help='Seed for evaluation')
    parser.add_argument('--param_space', type=str, default='physical', help='Parameter space to use for evaluation')
    parser.add_argument('--fixed_design', type=str, default=None, help='Evaluate a fixed design as a JSON list (default: None)')
    parser.add_argument('--profile', action='store_true', help='Enable profiling of methods')

    args = parser.parse_args()
    
    # Convert global_rank string to list of integers using json.loads
    if isinstance(args.global_rank, str):
        args.global_rank = json.loads(args.global_rank)

    valid_params = inspect.signature(Evaluator.__init__).parameters.keys()
    valid_params = [k for k in valid_params if k != 'self']
    eval_args = {k: v for k, v in vars(args).items() if k in valid_params}
    if eval_args['fixed_design'] is not None:
        parsed_value = json.loads(eval_args['fixed_design'])
        eval_args['fixed_design'] = parsed_value
    
    print(f"Evaluating with parameters:")
    print(json.dumps(eval_args, indent=2))
    evaluator = Evaluator(**eval_args)

    if args.eval_step == 'last':
        eval_step = evaluator.total_steps
    elif args.eval_step is not None:
        eval_step = int(args.eval_step)

    nominal_eig = evaluator.get_eig(step=eval_step, nominal_design=True)
    if evaluator.fixed_design is not False:
        eig = evaluator.get_eig(step=eval_step, nominal_design=False)[0]
        print(f"Nominal EIG: {nominal_eig}, Fixed Design EIG: {eig}, Fixed Design: {evaluator.fixed_design.tolist()}")
    else:
        eigs = evaluator.get_eig(step=eval_step, nominal_design=False)
        optimal_design = evaluator.experiment.designs[np.argmax(eigs)].cpu().numpy()
        print(f"Nominal EIG: {nominal_eig}, Optimal EIG: {np.max(eigs)}, Optimal Design: {optimal_design.tolist()}")

    try:
        pass
        #evaluator.generate_posterior(step=eval_step)
    except Exception as e:
        traceback.print_exc()
    #evaluator.eig_grid(step=eval_step)
    #evaluator.posterior_steps(steps=[30000, 100000, 200000, 'last'])
    
    # Plot EIG evolution across multiple training steps
    try:
        evaluator.sorted_eig_designs(steps=[eval_step//6, eval_step//4, eval_step//2, 'last'])
    except Exception as e:
        traceback.print_exc()

    try:
        evaluator.design_comparison(step=eval_step)
    except Exception as e:
        traceback.print_exc()
    
    #try:
    #    evaluator.sample_posterior(step=eval_step, level=0.68, central=True)
    #except Exception as e:
    #     traceback.print_exc()

    #evaluator.sample_posterior(step=eval_step, level=0.68, central=True)
    print(f"Evaluation completed successfully!")