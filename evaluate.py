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
from util import *
import mlflow
import inspect
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("file:" + os.environ["SCRATCH"] + "/bed/BED_cosmo/num_tracers/mlruns")

class Evaluator:
    def __init__(
            self, run_id, guide_samples=1000, design_step=0.05, design_lower=0.05, design_upper=None,
            seed=1, cosmo_exp='num_tracers', levels=[0.68, 0.95], global_rank=0, 
            n_evals=20, n_particles=1000, param_space='physical',
            verbose=False, device="cuda:0", profile=False,
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
    def _eval_step(self, step, design_type='nominal', global_rank=None, combine_ranks=False):
        """
        Generates samples given the nominal context (nominal design + central values).
        Args:
            step (int): Step to calculate the nominal samples for.
            combine_ranks (bool): If True, combines samples from all ranks into a single MCSamples object.
        Returns: 
            If combine_ranks=True: a single GetDist MCSamples object with combined samples from all ranks
            If combine_ranks=False: a list of GetDist MCSamples objects, one per rank
        """
        if global_rank is None:
            global_ranks_list = self.global_ranks
        else:
            global_ranks_list = [global_rank]
        rank_samples = []
        rank_eigs = []
        for rank in global_ranks_list:
            flow_model, _ = load_model(
                self.experiment, step, self.run_obj, 
                self.run_args, self.device, 
                global_rank=rank
                )
            if design_type == 'optimal':
                _, eig, _, design = self.calc_eig_batch(flow_model, step, rank)
                rank_eigs.append(eig)
            elif design_type == 'nominal':
                design = self.experiment.nominal_design
                _, _, eig, _ = self.calc_eig_batch(flow_model, step, rank)
                rank_eigs.append(eig)
            else:
                raise ValueError(f"Invalid design type: {design_type}")
            
            rank_samples.append(self._get_samples(design, flow_model))
        
        # If combine_ranks is True, combine all rank samples into a single MCSamples object
        if combine_ranks and len(rank_samples) > 1:
            # Extract sample arrays from all ranks and concatenate
            combined_samples_array = np.concatenate([s.samples for s in rank_samples], axis=0)
            with contextlib.redirect_stdout(io.StringIO()):
                combined_samples = getdist.MCSamples(
                    samples=combined_samples_array,
                    names=self.experiment.cosmo_params,
                    labels=self.experiment.latex_labels
                )
            return combined_samples, np.mean(rank_eigs)
        elif combine_ranks and len(rank_samples) == 1:
            # Only one rank, just return it directly (not in a list)
            return rank_samples[0], np.mean(rank_eigs)
        else:
            # Return list of samples (original behavior)
            return rank_samples, np.mean(rank_eigs)

    @profile_method
    def _get_samples(self, design, flow_model):
        context = torch.cat([design, self.experiment.central_val], dim=-1)
        return self.experiment.get_guide_samples(flow_model, context, num_samples=self.guide_samples, transform_output=self.nf_transform_output)
    
    @profile_method
    def generate_posterior(self, step='last', display=['nominal', 'optimal'], combine_ranks=True):
        """
        Generates the posterior for given type(s) of design input.

        Args:
            step (int): The checkpoint step to plot the posterior for.
            display (list): The designs to display.
            combine_ranks (bool): If True, combines samples from all ranks into a single posterior distribution.
                                If False, plots each rank's samples separately.

        """
        print(f"Running posterior evaluation...")
        all_samples = []
        all_colors = []
        if 'nominal' in display:
            print(f"Generating posterior samples with nominal design...")
            # Sample with nominal design
            nominal_samples, nominal_eig = self._eval_step(step, design_type='nominal', combine_ranks=combine_ranks)
            if combine_ranks:
                # nominal_samples is a single MCSamples object
                all_samples.append(nominal_samples)
                all_colors.append('tab:blue')
            else:
                # nominal_samples is a list of MCSamples objects (one per rank)
                all_samples.extend(nominal_samples)
                all_colors.extend(['tab:blue'] * len(nominal_samples))
        if 'optimal' in display and not self.run_args['fixed_design']:
            print(f"Generating posterior samples with optimal design...")
            # Sample with optimal design
            optimal_samples, optimal_eig = self._eval_step(step, design_type='optimal', combine_ranks=combine_ranks)
            if combine_ranks:
                # optimal_samples is a single MCSamples object
                all_samples.append(optimal_samples)
                all_colors.append('tab:orange')
            else:
                # optimal_samples is a list of MCSamples objects (one per rank)
                all_samples.extend(optimal_samples)
                all_colors.extend(['tab:orange'] * len(optimal_samples))

        # Get the DESI MCMC samples
        desi_samples_gd = self.experiment.get_desi_samples(transform_output=self.desi_transform_output)
        all_samples.append(desi_samples_gd)
        all_colors.append('black')
        g = plot_posterior(all_samples, all_colors, levels=self.levels, width_inch=10)
        ranks_label = f"Combined {len(self.global_ranks)} Ranks" if combine_ranks and len(self.global_ranks) > 1 else f"{len(self.global_ranks)} Rank{'s' if len(self.global_ranks) > 1 else ''}"
        g.fig.suptitle(f"Posterior Evaluation - Run: {self.run_id[:8]} ({ranks_label}, {self.param_space.capitalize()} Space)", fontsize=16)

        # Create custom legend
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()
        custom_legend = []
        if 'optimal' in display:
            custom_legend.append(
                Line2D([0], [0], color='tab:orange', label=f'Optimal Design (NF)')
            )
        if 'nominal' in display:
            custom_legend.append(
                Line2D([0], [0], color='tab:blue', label=f'Nominal Design (NF)')
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

        _, _, _, optimal_design = self.calc_eig_batch(posterior_flow, step, global_rank)
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
            central_samples_gd, _ = self._eval_step(step, design_type='nominal', global_rank=global_rank)
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

    @profile_method
    def calc_eig_batch(self, flow_model, step=None, global_rank=None):
        """
        Calculates the average + std of the EIG over a batch of evaluations.
        Uses caching to avoid redundant calculations.

        Args:
            flow_model (torch.nn.Module): The flow model to evaluate.
            step (int): Step for caching key (optional).
            global_rank (int): Global rank for caching key (optional).

        Returns:
            tuple: A tuple containing:
                - avg_eigs (np.ndarray): Average EIGs for each design.
                - optimal_avg_eig (float): Maximum of average EIGs.
                - nominal_avg_eig (float): Average EIG of the nominal design.
                - optimal_design (torch.Tensor): Design with the maximum EIG.
        """
        
        # Create cache key based on model parameters and evaluation settings
        cache_key = f"step_{step}_rank_{global_rank}_evals_{self.n_evals}_particles_{self.n_particles}"
        
        if cache_key in self._eig_cache:
            print(f"Using cached EIG results for {cache_key}")
            return self._eig_cache[cache_key]

        eigs_batch = []
        nominal_eig_batch = []
        
        for n in range(self.n_evals):
            if self.verbose and n % 5 == 0:
                print(f"  EIG evaluation {n+1}/{self.n_evals}")
            
            eigs = self.calc_eig(flow_model, nominal_design=False)
            eigs_batch.append(eigs.cpu().numpy()/np.log(2))
            
            nominal_eig = self.calc_eig(flow_model, nominal_design=True)
            nominal_eig_batch.append(nominal_eig.cpu().numpy()/np.log(2))
        
        eigs_batch = np.array(eigs_batch)
        nominal_eig_batch = np.array(nominal_eig_batch)
        # avg over the number of evaluations
        avg_eigs = np.mean(eigs_batch, axis=0)
        eigs_std = np.std(eigs_batch, axis=0)
        eigs_se = eigs_std/np.sqrt(self.n_evals)
        nominal_avg_eig = np.mean(nominal_eig_batch, axis=0).item()

        optimal_design = self.experiment.designs[np.argmax(avg_eigs)]
        
        # Cache the results
        result = (avg_eigs, np.max(avg_eigs), nominal_avg_eig, optimal_design)
        self._eig_cache[cache_key] = result
        
        return result

    def calc_eig(self, flow_model, nominal_design=False):
        """
        Evaluates the EIG of the posterior flow for the evaluation designs tensor.

        Args:
            flow_model (torch.nn.Module): The flow model to evaluate.
            nominal_design (bool): Whether to evaluate the nominal design.
        Returns:
            eigs (torch.Tensor): The EIGs for each design.
        """
        if nominal_design:
            designs = self.experiment.nominal_design.unsqueeze(0)
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
    def design_comparison(self, step, width=0.2, combine_ranks='mean'):
        """
        Plots a comparison of the nominal and optimal design.
        
        Args:
            step: The step to evaluate
            width: Width of the bars in the bar chart
            combine_ranks (str): How to combine optimal designs from multiple ranks:
                                'mean' - average the optimal designs across ranks
                                'median' - median of optimal designs across ranks
                                'first' - use only the first rank
        """
        print(f"Running design comparison...")
        if self.run_args['fixed_design']:
            print("Warning: Fixed design was used for training, skipping design comparison plot.")
            return
        
        # Calculate optimal design for each rank
        rank_optimal_designs = []
        for rank in self.global_ranks:
            flow_model, _ = load_model(
                self.experiment, step, self.run_obj, 
                self.run_args, self.device, 
                global_rank=rank
            )
            _, _, _, optimal_design = self.calc_eig_batch(flow_model, step, rank)
            rank_optimal_designs.append(optimal_design.cpu().numpy())
        
        # Combine optimal designs
        if len(self.global_ranks) > 1:
            rank_optimal_designs_array = np.array(rank_optimal_designs)
            if combine_ranks == 'mean':
                optimal_design_cpu = np.mean(rank_optimal_designs_array, axis=0)
                label_suffix = 'Mean'
            elif combine_ranks == 'median':
                optimal_design_cpu = np.median(rank_optimal_designs_array, axis=0)
                label_suffix = 'Median'
            elif combine_ranks == 'first':
                optimal_design_cpu = rank_optimal_designs[0]
                label_suffix = f'Rank {self.global_ranks[0]}'
            else:
                raise ValueError(f"Invalid combine_ranks value: {combine_ranks}")
        else:
            optimal_design_cpu = rank_optimal_designs[0]
            label_suffix = f'Rank {self.global_ranks[0]}'
        
        # Set the positions for the bars
        x = np.arange(len(self.experiment.targets))  # the label locations

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(14, 7))
        # Convert tensors to CPU numpy arrays if they're on GPU
        nominal_design_cpu = self.experiment.nominal_design.cpu().numpy()
        
        bars1 = ax.bar(x - width/2, nominal_design_cpu, width, label='Nominal Design', color='tab:blue')
        bars2 = ax.bar(x + width/2, optimal_design_cpu, width, label=f'Optimal Design ({label_suffix})', color='tab:orange')
        ax.set_xlabel('Tracers')
        ax.set_ylabel('Num Tracers')
        ax.set_xticks(x)
        ax.set_xticklabels(self.experiment.targets)
        ax.legend()
        plt.suptitle(f"{self.experiment.name} Design Variables ({', '.join(self.experiment.targets)}, {len(self.global_ranks)} Rank{'s' if len(self.global_ranks) > 1 else ''})", fontsize=16)
        plt.tight_layout()
        save_path = f"{self.save_path}/plots/design_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_figure(save_path, fig=fig, dpi=400)
        plt.close(fig)


    @profile_method
    def eig_grid(self, step, combine_ranks='mean'):
        """
        Plots the EIG on a 2D grid with subplot layout:
        - Top plot: colors points by the 3rd design variable (f_QSO)
        - Bottom plot: colors points by EIG values
        
        Args:
            step: The step to evaluate
            combine_ranks (str): How to combine EIG results from multiple ranks:
                                'mean' - average the EIGs across ranks
                                'median' - median of EIGs across ranks
                                'first' - use only the first rank
        """
        print(f"Running EIG grid evaluation...")
        if self.run_args['fixed_design']:
            print("Warning: Fixed design was used for training, skipping EIG grid plot.")
            return

        # Calculate EIG for each rank
        rank_eigs_list = []
        rank_optimal_designs = []
        rank_nominal_eigs = []
        
        for rank in self.global_ranks:
            flow_model, _ = load_model(
                self.experiment, step, self.run_obj, 
                self.run_args, self.device, 
                global_rank=rank
            )
            eigs, _, nominal_avg_eig, optimal_design = self.calc_eig_batch(flow_model, step, rank)
            rank_eigs_list.append(eigs)
            rank_optimal_designs.append(optimal_design.cpu().numpy())
            rank_nominal_eigs.append(nominal_avg_eig)
        
        # Combine EIG results
        if len(self.global_ranks) > 1:
            rank_eigs_array = np.array(rank_eigs_list)
            if combine_ranks == 'mean':
                eigs = np.mean(rank_eigs_array, axis=0)
                optimal_design = np.mean(rank_optimal_designs, axis=0)
                nominal_avg_eig = np.mean(rank_nominal_eigs)
                label_suffix = 'Mean'
            elif combine_ranks == 'median':
                eigs = np.median(rank_eigs_array, axis=0)
                optimal_design = np.median(rank_optimal_designs, axis=0)
                nominal_avg_eig = np.median(rank_nominal_eigs)
                label_suffix = 'Median'
            elif combine_ranks == 'first':
                eigs = rank_eigs_list[0]
                optimal_design = rank_optimal_designs[0]
                nominal_avg_eig = rank_nominal_eigs[0]
                label_suffix = f'Rank {self.global_ranks[0]}'
            else:
                raise ValueError(f"Invalid combine_ranks value: {combine_ranks}")
        else:
            eigs = rank_eigs_list[0]
            optimal_design = rank_optimal_designs[0]
            nominal_avg_eig = rank_nominal_eigs[0]
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
    def posterior_steps(self, steps, level=0.68, combine_ranks=True):
        """
        Plots posterior distributions at different training steps for a single run.
        
        Args:
            steps (list): List of steps to plot. Can include 'last' or 'loss_best' as special values.
            level (float): Contour level to plot.
            combine_ranks (bool): If True, combines samples from all ranks into a single posterior distribution.
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
            samples, _ = self._eval_step(step, design_type='nominal', combine_ranks=combine_ranks)
            # Convert RGBA color to hex string before extending
            color_hex = matplotlib.colors.to_hex(colors[i % len(colors)])
            if combine_ranks:
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
    def eig_steps(self, steps=None, plot_ranks=True, combine_ranks='mean'):
        """
        Plots the EIG in 1D at various steps.
        Args:
            steps (list): List of steps to plot. Can include 'last' or 'best' as special values.
            plot_ranks (bool): If True, plots each rank separately with different linestyles.
                             If False, only plots combined results.
            combine_ranks (str or None): How to combine EIG results from multiple ranks:
                                        'mean' - average the EIGs across ranks
                                        'median' - median of EIGs across ranks
                                        None - don't show combined results
        """
        print(f"Running EIG steps evaluation...")
        checkpoint_dir = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/checkpoints"
        if not os.path.isdir(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found for run {self.run_id}, skipping. Path: {checkpoint_dir}")
            return
        if self.run_args['fixed_design']:
            print("Warning: Fixed design was used for training, skipping EIG steps plot.")
            return
        
        plt.figure(figsize=(14, 6))
        if 'last' not in steps and self.run_args["total_steps"] not in steps:
            steps.append('last')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(steps)))
        
        for step_idx, s in enumerate(steps):
            step_color = colors[step_idx]
            rank_eigs_list = []
            rank_nominal_eigs = []
            
            # Calculate EIG for each rank
            for rank_idx, rank in enumerate(self.global_ranks):
                flow_model, selected_step = load_model(
                    self.experiment, s, 
                    self.run_obj, self.run_args, 
                    self.device, global_rank=rank
                )
                eigs, _, nominal_avg_eig, _ = self.calc_eig_batch(flow_model, selected_step, rank)
                rank_eigs_list.append(eigs)
                rank_nominal_eigs.append(nominal_avg_eig)
                
                # Plot individual rank if requested
                if plot_ranks and len(self.global_ranks) > 1:
                    alpha = 0.4 if combine_ranks else 0.7
                    plt.plot(eigs, label=f'Step {selected_step}, Rank {rank}', 
                            color=step_color, linestyle='-', alpha=alpha, linewidth=1.0)
            
            # Plot combined EIG if requested
            if combine_ranks and len(self.global_ranks) > 1:
                rank_eigs_array = np.array(rank_eigs_list)
                if combine_ranks == 'mean':
                    combined_eigs = np.mean(rank_eigs_array, axis=0)
                    label_suffix = 'Mean'
                elif combine_ranks == 'median':
                    combined_eigs = np.median(rank_eigs_array, axis=0)
                    label_suffix = 'Median'
                else:
                    raise ValueError(f"Invalid combine_ranks value: {combine_ranks}")
                
                plt.plot(combined_eigs, label=f'Step {selected_step} ({label_suffix})', 
                        color=step_color, linestyle='-', linewidth=2.5, alpha=1.0)
                
                # Plot nominal EIG for the last step (combined)
                if s == 'last':
                    combined_nominal = np.mean(rank_nominal_eigs) if combine_ranks == 'mean' else np.median(rank_nominal_eigs)
                    plt.axhline(y=combined_nominal, color='black', linestyle='--', 
                               label=f'Nominal EIG ({label_suffix})', linewidth=2)
            elif len(self.global_ranks) == 1:
                # Single rank, just plot it normally
                plt.plot(rank_eigs_list[0], label=f'Step {selected_step}', 
                        color=step_color, linestyle='-', linewidth=2)
                if s == 'last':
                    plt.axhline(y=rank_nominal_eigs[0], color='black', linestyle='--', 
                               label='Nominal EIG', linewidth=2)

        plt.xlabel("Design Index", fontsize=12)
        plt.ylabel("EIG [bits]", fontsize=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, framealpha=0.9)
        ranks_label = f"{len(self.global_ranks)} Ranks" if len(self.global_ranks) > 1 else "1 Rank"
        plt.suptitle(f"EIG Steps for Run: {self.run_obj.info.run_name} ({self.run_id[:8]}, {ranks_label})", fontsize=14)
        plt.tight_layout()
        save_figure(f"{self.save_path}/plots/eig_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=plt.gcf(), dpi=400)

    @profile_method
    def sorted_eig_designs(self, step, combine_ranks='mean'):
        """
        Plots sorted EIG values and corresponding designs using the evaluator's EIG calculations.
        
        Args:
            step (int): Step to evaluate.
            combine_ranks (str): How to combine EIG results from multiple ranks:
                                'mean' - average the EIGs across ranks
                                'median' - median of EIGs across ranks
                                'first' - use only the first rank
        """
        print(f"Running sorted EIG and designs plot for step {step}...")

        if self.run_args['fixed_design']:
            print("Warning: Fixed design was used for training, skipping sorted EIG plot.")
            return

        # Calculate EIG for each rank
        rank_eigs_list = []
        rank_nominal_eigs = []
        
        for rank in self.global_ranks:
            flow_model, _ = load_model(
                self.experiment, step, self.run_obj, 
                self.run_args, self.device, 
                global_rank=rank
            )
            eigs, _, nominal_avg_eig, _ = self.calc_eig_batch(flow_model, step, rank)
            rank_eigs_list.append(eigs)
            rank_nominal_eigs.append(nominal_avg_eig)
        
        # Combine EIG results
        if len(self.global_ranks) > 1:
            rank_eigs_array = np.array(rank_eigs_list)
            if combine_ranks == 'mean':
                eigs = np.mean(rank_eigs_array, axis=0)
                nominal_avg_eig = np.mean(rank_nominal_eigs)
                label_suffix = 'Mean'
            elif combine_ranks == 'median':
                eigs = np.median(rank_eigs_array, axis=0)
                nominal_avg_eig = np.median(rank_nominal_eigs)
                label_suffix = 'Median'
            elif combine_ranks == 'first':
                eigs = rank_eigs_list[0]
                nominal_avg_eig = rank_nominal_eigs[0]
                label_suffix = f'Rank {self.global_ranks[0]}'
            else:
                raise ValueError(f"Invalid combine_ranks value: {combine_ranks}")
        else:
            eigs = rank_eigs_list[0]
            nominal_avg_eig = rank_nominal_eigs[0]
            label_suffix = f'Rank {self.global_ranks[0]}'
        
        # Sort EIG values in descending order (based on combined/mean EIG)
        sorted_eigs_idx = np.argsort(eigs)[::-1]
        sorted_eigs = eigs[sorted_eigs_idx]
        
        # Get designs and sort them according to sorted EIG indices
        designs = self.experiment.designs.cpu().numpy()
        sorted_designs = designs[sorted_eigs_idx]
        
        # Sort individual rank EIGs using the same ordering as the combined EIG
        sorted_rank_eigs = []
        if len(self.global_ranks) > 1:
            for rank_eigs in rank_eigs_list:
                sorted_rank_eigs.append(rank_eigs[sorted_eigs_idx])
        
        # Create figure with subplots and space for colorbar
        fig = plt.figure(figsize=(22, 6))  # Increased width to accommodate colorbar
        gs = gridspec.GridSpec(2, 2, height_ratios=[0.5, 0.1], width_ratios=[1, 0.02], hspace=0.2, wspace=0.1)
        
        # Create subplots
        ax0 = fig.add_subplot(gs[0, 0])  # Top plot for EIG
        ax1 = fig.add_subplot(gs[1, 0])  # Bottom plot for heatmap
        cbar_ax = fig.add_subplot(gs[:, 1])  # Colorbar spanning both plots
        
        # Plot individual rank EIGs (faded) if multiple ranks
        if len(self.global_ranks) > 1 and combine_ranks != 'first':
            for rank_idx, sorted_rank_eig in enumerate(sorted_rank_eigs):
                ax0.plot(sorted_rank_eig, color="tab:orange", alpha=0.25, linewidth=1.0, 
                        label=f"Rank {self.global_ranks[rank_idx]}" if rank_idx < 3 else "")
        
        # Plot nominal EIG line
        ax0.axhline(nominal_avg_eig, color='tab:blue', linestyle='--', label="Nominal EIG", linewidth=2, zorder=10)
        
        # Plot sorted combined EIG (main line)
        line_label = f"Sorted EIG ({label_suffix})" if len(self.global_ranks) > 1 else "Sorted EIG"
        ax0.plot(sorted_eigs, color="tab:orange", label=line_label, linewidth=2.5, alpha=1.0, zorder=5)
        
        ax0.set_xlim(0, len(sorted_eigs))
        ax0.set_ylabel("Expected Information Gain [bits]", fontsize=11)
        ax0.legend(loc='upper right', fontsize=9)
        
        # Plot sorted designs
        im = ax1.imshow(sorted_designs.T, aspect='auto', cmap='viridis')
        ax1.set_xlabel("Design Index")
        ax1.set_yticks(np.arange(len(self.experiment.targets)), self.experiment.targets)
        
        # Add colorbar spanning the full height
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Design Value')
        
        # Add overall title with rank information
        fig.suptitle(f'Sorted EIG and Designs ({label_suffix}, {len(self.global_ranks)} Rank{"s" if len(self.global_ranks) > 1 else ""})', 
                    fontsize=14, y=0.98)
        
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
    parser.add_argument('--profile', action='store_true', help='Enable profiling of methods')

    args = parser.parse_args()
    
    # Convert global_rank string to list of integers using json.loads
    if isinstance(args.global_rank, str):
        args.global_rank = json.loads(args.global_rank)

    valid_params = inspect.signature(Evaluator.__init__).parameters.keys()
    valid_params = [k for k in valid_params if k != 'self']
    eval_args = {k: v for k, v in vars(args).items() if k in valid_params}

    evaluator = Evaluator(**eval_args)

    if args.eval_step == 'last':
        eval_step = evaluator.total_steps
    elif args.eval_step is not None:
        eval_step = int(args.eval_step)

    try:
        evaluator.generate_posterior(step=eval_step)
    except Exception as e:
        traceback.print_exc()
    #evaluator.eig_grid(step=eval_step)
    #evaluator.posterior_steps(steps=[30000, 100000, 200000, 'last'])
    try:
        evaluator.eig_steps(steps=[eval_step//4, eval_step//2, 3*eval_step//4, 'last'])
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

    try:
        evaluator.sorted_eig_designs(step=eval_step)
    except Exception as e:
        traceback.print_exc()
    #evaluator.sample_posterior(step=eval_step, level=0.68, central=True)
    print(f"Evaluation completed successfully!")