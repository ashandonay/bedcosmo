import sys
import os
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
            self, run_id, guide_samples, seed=1, design_lower=0.05, design_step=0.05,
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
        self.save_path = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/"
        self.guide_samples = guide_samples
        self.seed = seed
        auto_seed(self.seed) # fix random seed
        self.device = device
        self.levels = levels
        self.global_rank = global_rank
        self.n_evals = n_evals
        self.n_particles = n_particles
        self.verbose = verbose
        design_args = {
            "design_lower": design_lower,
            "design_step": design_step,
            "fixed_design": self.run_args["fixed_design"]
        }
        with open(mlflow.artifacts.download_artifacts(run_id=self.run_id, artifact_path="classes.json")) as f:
            self.classes = json.load(f)

        self.experiment = init_experiment(self.cosmo_exp, self.run_args, device=self.device, design_args=design_args, seed=self.seed)


    def posterior(self, step):
        """
        Evaluates the posterior of the optimal and nominal designs for a given run and returns the samples.
        """
        flow_model, _ = load_model(
            self.experiment, step, self.run_obj, 
            self.run_args, self.device, 
            global_rank=self.global_rank
            )
        eigs, optimal_eig, nominal_eig, optimal_design = self.calc_eig_batch(flow_model)
        optimal_samples_gd = self.get_samples(optimal_design, flow_model)
        nominal_samples_gd = self.get_samples(self.experiment.nominal_design, flow_model)
        desi_samples = np.load(f"{home_dir}/data/mcmc_samples/{self.run_args['cosmo_model']}.npy")
        with contextlib.redirect_stdout(io.StringIO()):
            desi_samples_gd = getdist.MCSamples(samples=desi_samples, names=self.experiment.cosmo_params, labels=self.experiment.latex_labels)
        color_list = ['tab:blue', 'black', 'black']
        line_style = ['-', '-', '--']
        all_samples = [optimal_samples_gd, nominal_samples_gd, desi_samples_gd]
        legend_labels = ['Optimal Design (NF)', 'DESI Nominal Design (NF)', 'DESI Nominal Design (MCMC)']
        g = plot_posterior(all_samples, color_list, levels=self.levels, line_style=line_style)

        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()

        # Create custom legend
        custom_legend = []
        custom_legend.append(
            Line2D([0], [0], color=color_list[0], linestyle=line_style[0],
                    label=f'{legend_labels[0]}, EIG: {optimal_eig:.2f}')
        )
        custom_legend.append(
            Line2D([0], [0], color=color_list[1], linestyle=line_style[1],
                    label=f'{legend_labels[1]}, EIG: {nominal_eig:.2f}')
        )
        custom_legend.append(
            Line2D([0], [0], color=color_list[2], linestyle=line_style[2],
                    label=f'{legend_labels[2]}')
        )
        g.fig.set_constrained_layout(True)
        leg = g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(0.99, 0.96))
        leg.set_in_layout(False)
        show_figure(f"{self.save_path}/plots/posterior_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=g.fig)
    
    def sample_posterior(self, step, level, num_data_samples, design_type='optimal', central=True):
        posterior_flow, _ = load_model(
            self.experiment, step, 
            self.run_obj, self.run_args, 
            device=self.device, global_rank=self.global_rank
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
        central_samples_gd = self.get_samples(design, posterior_flow)
        central_area = get_contour_area([central_samples_gd], 'Om', 'hrdrag', level)[0]
        all_samples.append(central_samples_gd)
        g = plot_posterior(all_samples, ['tab:blue']*len(data_idxs), levels=[level], alpha=0.7)
        if g.fig.legends:
            for legend in g.fig.legends:
                legend.remove()
        
        g.fig.set_constrained_layout(True)
        # Set title with proper positioning
        g.fig.suptitle(f"{design_type.capitalize()} Design {int(level*100)}% Credible Region, Avg areas: {np.mean(areas):.3f} +/- {np.std(areas):.3f}, Central val: {central_area:.3f}", 
                      fontsize=12)
        
        show_figure(f"{self.save_path}/plots/posterior_samples_{design_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=g.fig)

    def calc_eig_batch(self, flow_model):
        """
        Calculates the average + std of the EIG over a batch of evaluations.

        Args:
            flow_model (torch.nn.Module): The flow model to evaluate.

        Returns:
            tuple: A tuple containing:
                - avg_eigs (np.ndarray): Average EIGs for each design.
                - optimal_eig (float): Maximum EIG.
                - avg_nominal_eig (float): Average EIG of the nominal design.
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
        avg_nominal_eig = np.mean(nominal_eig_batch, axis=0).item()

        optimal_design = self.experiment.designs[np.argmax(avg_eigs)]
        return avg_eigs, np.max(avg_eigs), avg_nominal_eig, optimal_design

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

    def get_samples(self, design, flow_model):
        """
        Generates samples given the nominal context (nominal design + central values).
        Args:
            step (int): Step to calculate the nominal samples for.
        Returns: a GetDist MCSamples object with the nominal samples
        """
        context = torch.cat([design, self.experiment.central_val], dim=-1)
        samples = flow_model(context).sample((self.guide_samples,)).cpu().numpy()
        samples[:, -1] *= 100000 # Rescale the hrdrag samples 
        with contextlib.redirect_stdout(io.StringIO()):
            samples_gd = getdist.MCSamples(samples=samples, names=self.experiment.cosmo_params, labels=self.experiment.latex_labels)
        return samples_gd

    def eig_grid(self, step):
        """
        Plots the EIG on a 2D grid for a given step.
        """
        flow_model, _ = load_model(
            self.experiment, step, self.run_obj, 
            self.run_args, self.device, 
            global_rank=self.global_rank
            )
        eigs, optimal_eig, avg_nominal_eig, optimal_design = self.calc_eig_batch(flow_model)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.12, 0.1, 0.68, 0.8])
        sc = ax.scatter(
            self.experiment.designs[:, 0].cpu().numpy(), self.experiment.designs[:, 1].cpu().numpy(),
            c=eigs, cmap='viridis', s=60, alpha=0.8, marker='o',
            vmin=np.min(eigs), vmax=np.max(eigs)
        )
        # Add optimal and nominal markers
        ax.scatter(optimal_design[0].cpu().numpy(), optimal_design[1].cpu().numpy(),
                c='red', s=70, alpha=0.8, marker='*', label='Optimal Design')
        ax.scatter(self.experiment.nominal_design[0].cpu().numpy(), self.experiment.nominal_design[1].cpu().numpy(),
                c='black', s=70, alpha=0.8, marker='*', label='Nominal Design')
        ax.set_xlabel("$f_{LRG}$", fontsize=14)
        ax.set_ylabel("$f_{ELG}$", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_box_aspect(1)
        ax.legend(fontsize=14)

        # Dynamically match colorbar height and position to the plot
        plot_pos = ax.get_position().bounds  # (left, bottom, width, height)
        gap = 0.03
        cbar_width = 0.03
        cax = fig.add_axes([
            plot_pos[0] + plot_pos[2] + gap,  # left
            plot_pos[1],                      # bottom
            cbar_width,                       # width
            plot_pos[3]                       # height
        ])
        fig.colorbar(sc, cax=cax, label='EIG [bits]')

        show_figure(f"{self.save_path}/plots/eig_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=fig)

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
        all_areas = []
        color_list = []
        custom_legend = []

        checkpoint_dir = f'{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/checkpoints/'
        if not os.path.isdir(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found for run {self.run_id}, skipping. Path: {checkpoint_dir}")
            return
        for i, step in enumerate(steps):
            flow_model, _ = load_model(
                self.experiment, step, 
                self.run_obj, self.run_args, 
                self.device, global_rank=self.global_rank
                )
            samples = self.get_samples(self.experiment.nominal_design, flow_model)
            all_samples.append(samples)
            color_list.append(colors[i % len(colors)])
            area = get_contour_area(samples, 'Om', 'hrdrag', level)[0]
            all_areas.append(area)
            if step == 'last':
                step_label = self.run_args["total_steps"]
            elif step == 'loss_best':
                step_label = 'Best Loss'
            else:
                step_label = step
            custom_legend.append(
                Line2D([0], [0], color=colors[i % len(colors)], 
                        label=f'Step {step_label}, {int(level*100)}% Area: {area:.2f}')
            )
        desi_samples_gd = get_desi_samples(self.run_args['cosmo_model'])
        desi_area = get_contour_area([desi_samples_gd], 'Om', 'hrdrag', level)[0]
        all_samples.append(desi_samples_gd)
        color_list.append('black')  
        all_areas.append(desi_area)
        g = plot_posterior(all_samples, color_list, levels=[level])
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
        
        show_figure(f"{self.save_path}/plots/posterior_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=g.fig)

    def eig_steps(self, steps=None):
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
                self.device, global_rank=self.global_rank
                )
            eigs, _, nominal_eig, _ = self.calc_eig_batch(flow_model)
            plt.plot(eigs, label=f'Step {selected_step}')
            if s == 'last':
                plt.axhline(y=nominal_eig, color='black', linestyle='--', label='Nominal EIG')

        plt.xlabel("Design Index")
        plt.ylabel("EIG")
        plt.legend()
        plt.suptitle(f"EIG Steps for Run: {self.run_obj.info.run_name} ({self.run_id[:8]})")
        plt.tight_layout()
        show_figure(f"{self.save_path}/plots/eig_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", fig=plt.gcf())

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
    evaluate.posterior_steps(steps=[eval_step//4, eval_step//2, 3*eval_step//4, 'last'])
    evaluate.eig_steps(steps=[eval_step//4, eval_step//2, 3*eval_step//4, 'last'])

    evaluate.sample_posterior(step=eval_step, level=0.68, num_data_samples=15, design_type='optimal', central=True)
    evaluate.sample_posterior(step=eval_step, level=0.68, num_data_samples=15, design_type='nominal', central=True)
    
    compare_posterior(
        run_ids=[run_id],
        var=None, 
        guide_samples=guide_samples,
        seed=1,
        device=device,
        show_scatter=False,
        step=200000,
        global_rank=[0,1,2,3,4,5,6,7]
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Number Tracers Training")
    parser.add_argument('--run_id', type=str, default=None, help='MLflow run ID to resume training from (continues existing run with same parameters)')
    parser.add_argument('--eval_step', type=int, default=None, help='Step to resume training from (required when using --resume_id)')
    parser.add_argument('--cosmo_exp', type=str, default='num_tracers', help='Cosmological model set to use from run_args.json')
    parser.add_argument('--levels', type=float, default=[0.68, 0.95], help='Levels for contour plot')
    parser.add_argument('--global_rank', type=int, default=0, help='Global rank')
    parser.add_argument('--n_particles', type=int, default=1000, help='Number of particles to use for evaluation')
    parser.add_argument('--guide_samples', type=int, default=10000, help='Number of samples to generate from the posterior')
    parser.add_argument('--design_lower', type=float, default=0.05, help='Lowest design fraction')
    parser.add_argument('--design_step', type=float, default=0.05, help='Step size for design grid')
    parser.add_argument('--n_evals', type=int, default=20, help='Number of evaluations to average over')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for evaluation')
    parser.add_argument('--eval_seed', type=int, default=1, help='Seed for evaluation')

    args = parser.parse_args()

    run_eval(**vars(args))