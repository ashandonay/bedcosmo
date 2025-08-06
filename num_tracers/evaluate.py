import sys
import os
import contextlib
import io
import json
import matplotlib
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

base_dir = os.environ["HOME"] + '/bed/BED_cosmo'
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
    
from num_tracers import NumTracers
from pyro import distributions as dist
import numpy as np
import torch
import pandas as pd
from getdist import plots
from bed.grid import Grid
import argparse
from plotting import *
from util import *
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("file:" + os.environ["SCRATCH"] + "/bed/BED_cosmo/num_tracers/mlruns")

class Evaluation:
    def __init__(
            self, run_id, guide_samples, seed=1, design_step=0.05, design_lower=0.05, design_upper=None,
            cosmo_exp='num_tracers', levels=[0.68, 0.95], global_rank=0, 
            n_evals=20, device="cuda:0", n_particles=1000, verbose=False
            ):
        self.run_id = run_id
        run_data_list, _, _ = get_runs_data(run_ids=self.run_id)
        if run_data_list is None:
            print(f"Run {self.run_id} not found.")
            return
        run_data = run_data_list[0]
        self.run_obj = run_data['run_obj']
        self.run_args = run_data['params']
        self.exp_id = run_data['exp_id']
        self.cosmo_exp = cosmo_exp
        self.storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{self.cosmo_exp}"
        self.save_path = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts"
        self.guide_samples = guide_samples
        self.seed = seed
        auto_seed(self.seed) # fix random seed
        self.device = device
        self.levels = levels
        self.global_ranks = global_rank if isinstance(global_rank, list) else [global_rank]
        self.n_evals = n_evals
        self.n_particles = n_particles
        self.verbose = verbose
        design_args = {
            "design_step": design_step,
            "design_lower": design_lower,
            "design_upper": design_upper,
            "fixed_design": self.run_args["fixed_design"]
        }
        self.experiment = init_experiment(self.cosmo_exp, self.run_args, device=self.device, design_args=design_args, seed=self.seed)

    def _eval_step(self, step, design_type='nominal', global_rank=None):
        """
        Generates samples given the nominal context (nominal design + central values).
        Args:
            step (int): Step to calculate the nominal samples for.
        Returns: a GetDist MCSamples object with the nominal samples
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
                _, eig, _, design = self.calc_eig_batch(flow_model)
                rank_eigs.append(eig)
            elif design_type == 'nominal':
                design = self.experiment.nominal_design
                _, _, eig, _ = self.calc_eig_batch(flow_model)
                rank_eigs.append(eig)
            else:
                raise ValueError(f"Invalid design type: {design_type}")
            
            rank_samples.append(self._get_samples(design, flow_model))
        return rank_samples, np.mean(rank_eigs)

    def _get_samples(self, design, flow_model):
        context = torch.cat([design, self.experiment.central_val], dim=-1)
        samples = flow_model(context).sample((self.guide_samples,)).cpu().numpy()
        samples[:, -1] *= 100000 # Rescale the hrdrag samples 
        with contextlib.redirect_stdout(io.StringIO()):
            samples_gd = getdist.MCSamples(samples=samples, names=self.experiment.cosmo_params, labels=self.experiment.latex_labels)
        return samples_gd
    
    def posterior(self, step):
        """
        Evaluates the posterior of the optimal and nominal designs for a given run and returns the samples.
        """
        all_samples = []
        all_colors = []
        # Sample with optimal design
        optimal_samples, optimal_eig = self._eval_step(step, design_type='optimal')
        all_samples.extend(optimal_samples)
        # Set the optimal design samples to blue
        all_colors.extend(['tab:blue'] * len(optimal_samples))
        # Sample with nominal design
        nominal_samples, nominal_eig = self._eval_step(step, design_type='nominal')
        all_samples.extend(nominal_samples)
        # Set the nominal design samples to gray
        all_colors.extend(['gray'] * len(nominal_samples))
        # Get the DESI MCMC samples
        desi_samples_gd = get_desi_samples(self.run_args['cosmo_model'])
        all_samples.append(desi_samples_gd)
        all_colors.append('black')
        g = plot_posterior(all_samples, all_colors, levels=self.levels, width_inch=10)

        # Create custom legend
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()
        custom_legend = []
        custom_legend.append(
            Line2D([0], [0], color='tab:blue', label=f'Optimal Design (NF), EIG: {optimal_eig:.2f}')
        )
        custom_legend.append(
            Line2D([0], [0], color='gray', label=f'Nominal Design (NF), EIG: {nominal_eig:.2f}')
        )
        custom_legend.append(
            Line2D([0], [0], color='black', label=f'DESI Nominal Design (MCMC)')
        )
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96))
        leg.set_in_layout(False)
        save_figure(f"{self.save_path}/plots/posterior_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=g.fig, dpi=400)
    
    def sample_posterior(self, step, level, num_data_samples, design_type='optimal', global_rank=0, central=True):
        posterior_flow, _ = load_model(
            self.experiment, step, 
            self.run_obj, self.run_args, 
            device=self.device, global_rank=global_rank
            )

        _, _, _, optimal_design = self.calc_eig_batch(posterior_flow)

        if design_type == 'optimal':
            design = optimal_design
        elif design_type == 'nominal':
            design = self.experiment.nominal_design

        samples = self.experiment.sample_guide(
            lexpand(design.unsqueeze(0), num_data_samples), 
            posterior_flow, 
            num_data_samples=num_data_samples, 
            num_param_samples=self.guide_samples,
            central=central
            )

        data_idxs = np.arange(1, num_data_samples) # sample 10 data points
        all_samples = []
        areas = []
        for d in data_idxs:
            with contextlib.redirect_stdout(io.StringIO()):
                samples_gd = getdist.MCSamples(
                    samples=samples[:, d, :].cpu().numpy(),
                    names=self.experiment.cosmo_params,
                    labels=self.experiment.latex_labels
                )
            all_samples.append(samples_gd)
            areas.append(get_contour_area([samples_gd], 'Om', 'hrdrag', level)[0])
        # Get the central samples with the nominal design indexed by the input global rank
        central_samples_gd, _ = self._eval_step(step, design_type='nominal', global_rank=global_rank)
        central_area = get_contour_area(central_samples_gd, 'Om', 'hrdrag', level)[0]
        all_samples.append(central_samples_gd[0])
        g = plot_posterior(all_samples, ['gray']*len(data_idxs) + ['black'], levels=[level], alpha=0.7, width_inch=10)
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()
        
        g.fig.set_constrained_layout(True)
        # Set title with proper positioning
        g.fig.suptitle(f"{design_type.capitalize()} Design {int(level*100)}% Credible Region, Avg areas: {np.mean(areas):.3f} +/- {np.std(areas):.3f}, Central val: {central_area:.3f}", 
                      fontsize=12)
        
        save_figure(f"{self.save_path}/plots/posterior_samples_{design_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=g.fig, dpi=400)

    def calc_eig_batch(self, flow_model):
        """
        Calculates the average + std of the EIG over a batch of evaluations.

        Args:
            flow_model (torch.nn.Module): The flow model to evaluate.

        Returns:
            tuple: A tuple containing:
                - avg_eigs (np.ndarray): Average EIGs for each design.
                - optimal_avg_eig (float): Maximum of average EIGs.
                - nominal_avg_eig (float): Average EIG of the nominal design.
                - optimal_design (torch.Tensor): Design with the maximum EIG.
        """

        eigs_batch = []
        nominal_eig_batch = []
        for n in range(self.n_evals):
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
        return avg_eigs, np.max(avg_eigs), nominal_avg_eig, optimal_design

    def calc_eig(self, flow_model, nominal_design=False):
        """
        Evaluates the EIG of the posterior flow for the evaluation designs tensor.

        Args:
            flow_model (torch.nn.Module): The flow model to evaluate.
            nominal_design (bool): Whether to evaluate the nominal design.
        Returns:
            eigs (torch.Tensor): The EIGs for the evaluation designs tensor.
        """

        with torch.no_grad():
            _, eigs = posterior_loss(
                            experiment=self.experiment,
                            guide=flow_model,
                            num_particles=self.n_particles,
                            evaluation=True,
                            nflow=True,
                            analytic_prior=False,
                            condition_design=self.run_args["condition_design"],
                            nominal_design=nominal_design
                            )

        return eigs

    def design_comparison(self, step, width=0.2, global_rank=0):
        """
        Plots a comparison of the nominal and optimal design.
        """
        flow_model, _ = load_model(
            self.experiment, step, self.run_obj, 
            self.run_args, self.device, 
            global_rank=global_rank
            )
        _, _, _, optimal_design = self.calc_eig_batch(flow_model)
        # Set the positions for the bars
        x = np.arange(len(self.experiment.targets))  # the label locations

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(14, 7))
        # Convert tensors to CPU numpy arrays if they're on GPU
        nominal_design_cpu = self.experiment.nominal_design.cpu().numpy()
        optimal_design_cpu = optimal_design.cpu().numpy()
        
        bars1 = ax.bar(x - width/2, nominal_design_cpu, width, label='Nominal Design', color='black')
        bars2 = ax.bar(x + width/2, optimal_design_cpu, width, label='Optimal Design', color='tab:blue')
        ax.set_xlabel('Tracers')
        ax.set_ylabel('Num Tracers')
        ax.set_xticks(x)
        ax.set_xticklabels(self.experiment.targets)
        ax.legend()
        plt.suptitle(f"{self.experiment.name} Design Variables ({', '.join(self.experiment.targets)})", fontsize=16)
        plt.tight_layout()
        save_path = f"{self.save_path}/plots/design_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_figure(save_path, fig=fig, dpi=400)
        plt.close(fig)


    def eig_grid(self, step, global_rank=0):
        """
        Plots the EIG on a 2D grid with subplot layout:
        - Top plot: colors points by the 3rd design variable (f_QSO)
        - Bottom plot: colors points by EIG values
        """
        flow_model, _ = load_model(
            self.experiment, step, self.run_obj, 
            self.run_args, self.device, 
            global_rank=global_rank
            )
        eigs, optimal_avg_eig, nominal_avg_eig, optimal_design = self.calc_eig_batch(flow_model)
        
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
        optimal_design = optimal_design.cpu().numpy()

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

        plt.tight_layout()
        save_figure(f"{self.save_path}/plots/eig_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=fig, dpi=400)

    def posterior_steps(self, steps, level=0.68):
        """
        Plots posterior distributions at different training steps for a single run.
        
        Args:
            steps (list): List of steps to plot. Can include 'last' or 'loss_best' as special values.
            level (float): Contour level to plot.
            type (str): Type of steps to plot. Can be 'all', 'area', or 'loss'.
        """
        colors = plt.cm.viridis_r(np.linspace(0, 1, len(steps)))
        
        all_samples = []
        all_colors = []
        areas = []
        custom_legend = []

        checkpoint_dir = f'{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/checkpoints'
        if not os.path.isdir(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found for run {self.run_id}, skipping. Path: {checkpoint_dir}")
            return
        for i, step in enumerate(steps):
            samples, _ = self._eval_step(step, design_type='nominal')
            all_samples.extend(samples)
            # Convert RGBA color to hex string before extending
            color_hex = matplotlib.colors.to_hex(colors[i % len(colors)])
            all_colors.extend([color_hex] * len(samples))
            areas.append(np.mean(get_contour_area(samples, 'Om', 'hrdrag', level), axis=0))
            if step == 'last':
                step_label = self.run_args["total_steps"]
            elif step == 'loss_best':
                step_label = 'Best Loss'
            else:
                step_label = step
            custom_legend.append(
                Line2D([0], [0], color=color_hex, 
                        label=f'Step {step_label}, {int(level*100)}% Area: {areas[i]:.2f}')
            )
        desi_samples_gd = get_desi_samples(self.run_args['cosmo_model'])
        desi_area = get_contour_area([desi_samples_gd], 'Om', 'hrdrag', level)[0]
        all_samples.append(desi_samples_gd)
        all_colors.append('black')  
        areas.append(desi_area)
        g = plot_posterior(all_samples, all_colors, levels=[level], width_inch=12)
        # Remove existing legends if any
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()

        custom_legend.append(
            Line2D([0], [0], color='black', 
                label=f'DESI, Area ({int(level*100)}% Contour): {desi_area:.3f}')
        )
        
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96))
        leg.set_in_layout(False)
        g.fig.suptitle(f"Posterior Steps for Run: {self.run_id[:8]}", 
                      fontsize=12)
        
        save_figure(f"{self.save_path}/plots/posterior_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=g.fig, dpi=400)

    def eig_steps(self, steps=None, global_rank=0):
        """
        Plots the EIG in 1D at various steps.
        Args:
            steps (list): List of steps to plot. Can include 'last' or 'best' as special values.
        """
        checkpoint_dir = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/checkpoints"
        if not os.path.isdir(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found for run {self.run_id}, skipping. Path: {checkpoint_dir}")
            return
        plt.figure(figsize=(10, 6))
        if 'last' not in steps and self.run_args["total_steps"] not in steps:
            steps.append('last')
        for s in steps:
            flow_model, selected_step = load_model(
                self.experiment, s, 
                self.run_obj, self.run_args, 
                self.device, global_rank=global_rank
                )
            eigs, _, nominal_avg_eig, _ = self.calc_eig_batch(flow_model)
            plt.plot(eigs, label=f'Step {selected_step}')
            if s == 'last':
                plt.axhline(y=nominal_avg_eig, color='black', linestyle='--', label='Nominal EIG')

        plt.xlabel("Design Index")
        plt.ylabel("EIG")
        plt.legend()
        plt.suptitle(f"EIG Steps for Run: {self.run_obj.info.run_name} ({self.run_id[:8]})")
        plt.tight_layout()
        save_figure(f"{self.save_path}/plots/eig_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=plt.gcf(), dpi=400)

def run_eval(
        design_lower, 
        design_step, 
        run_id, 
        eval_step, 
        guide_samples, 
        device, 
        eval_seed, 
        cosmo_exp, 
        levels, 
        global_rank, 
        n_particles,
        n_evals
        ):
    """
    Runs the evaluation routine for a given run ID.

    Args:
        design_lower (float): Lowest design fraction.
        design_step (float): Step size for design grid.
        run_id (str): MLflow run ID to evaluate.
        eval_step (int): Step to evaluate.
        guide_samples (int): Number of samples to generate from the posterior.
        device (str): Device to use for evaluation.
    """
    # default to run design parameters if not specified
    client = MlflowClient()
    run_info = client.get_run(run_id)
    run_args = parse_mlflow_params(run_info.data.params)
    if design_lower is None:
        design_lower = run_args["design_lower"]
    if design_step is None:
        design_step = run_args["design_step"]
    
    if eval_step == 'last':
        eval_step = run_args["total_steps"]
    elif eval_step is not None:
        eval_step = int(eval_step)

    # Create evaluation plots
    evaluate = Evaluation(
        run_id=run_id,
        guide_samples=guide_samples,
        seed=eval_seed,
        device=device,
        design_lower=design_lower,
        design_step=design_step,
        cosmo_exp=cosmo_exp,
        levels=levels,
        global_rank=global_rank,
        n_evals=n_evals,
        n_particles=n_particles
    )
    evaluate.posterior(step=eval_step)
    evaluate.eig_grid(step=eval_step)
    evaluate.posterior_steps(steps=[1000, 5000, 20000, 'last'])
    evaluate.eig_steps(steps=[eval_step//4, eval_step//2, 3*eval_step//4, 'last'])
    evaluate.design_comparison(step=eval_step)

    evaluate.sample_posterior(step=eval_step, level=0.68, num_data_samples=15, design_type='optimal', central=True)
    evaluate.sample_posterior(step=eval_step, level=0.68, num_data_samples=15, design_type='nominal', central=True)
    
    compare_posterior(
        run_ids=[run_id],
        var=None, 
        guide_samples=guide_samples,
        seed=1,
        device=device,
        show_scatter=False,
        step=eval_step,
        global_rank=list(range(int(run_args["n_devices"])))
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Number Tracers Training')
    parser.add_argument('--run_id', type=str, default=None, help='MLflow run ID to resume training from (continues existing run with same parameters)')
    parser.add_argument('--eval_step', type=str, default=None, help='Step to evaluate (can be integer or "last")')
    parser.add_argument('--cosmo_exp', type=str, default='num_tracers', help='Cosmological model set to use from run_args.json')
    parser.add_argument('--levels', type=float, default=[0.68, 0.95], help='Levels for contour plot')
    parser.add_argument('--global_rank', type=str, default='0', help='List of global ranks to evaluate')
    parser.add_argument('--n_particles', type=int, default=1000, help='Number of particles to use for evaluation')
    parser.add_argument('--guide_samples', type=int, default=10000, help='Number of samples to generate from the posterior')
    parser.add_argument('--design_lower', type=float, default=None, help='Lowest design fraction')
    parser.add_argument('--design_step', type=float, default=None, help='Step size for design grid')
    parser.add_argument('--n_evals', type=int, default=20, help='Number of evaluations to average over')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for evaluation')
    parser.add_argument('--eval_seed', type=int, default=1, help='Seed for evaluation')

    args = parser.parse_args()
    
    # Convert global_rank string to list of integers using json.loads
    if isinstance(args.global_rank, str):
        args.global_rank = json.loads(args.global_rank)

    run_eval(**vars(args))