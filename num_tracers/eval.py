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

class Evaluate:
    def __init__(
            self, designs, run_id, eval_args, 
            cosmo_exp='num_tracers', levels=[0.68, 0.95], global_rank=0, 
            n_evals=20, eval_particles=1000, verbose=False
            ):
        self.designs = designs
        self.run_id = run_id
        self.eval_args = eval_args
        self.cosmo_exp = cosmo_exp
        self.levels = levels
        self.global_rank = global_rank
        self.n_evals = n_evals
        self.eval_particles = eval_particles
        self.verbose = verbose

        run_data_list, _, _ = get_runs_data(
            run_ids=self.run_id,
            parse_params=True,
        )

        if not run_data_list:
            print(f"Run {self.run_id} not found.")
            return

        run_data = run_data_list[0]
        self.run_args = run_data['params']
        self.run_obj = run_data['run_obj']
        self.exp_id = run_data['exp_id']
        self.storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{self.cosmo_exp}"
        with open(mlflow.artifacts.download_artifacts(run_id=self.run_id, artifact_path="classes.json")) as f:
            self.classes = json.load(f)

        self.likelihood = init_likelihood(self.run_args, self.eval_args, self.cosmo_exp)

        self.nominal_design = torch.tensor(
            self.likelihood.desi_tracers.groupby('class').sum()['observed'].reindex(self.classes.keys()).values, 
            device=self.eval_args["device"]
            )

    def posterior(self, step):
        """
        Evaluates the posterior of the optimal and nominal designs for a given run and returns the samples.
        """
        flow_model = load_model(
            self.likelihood, step, self.run_obj, 
            self.run_args, self.classes, self.eval_args, 
            self.cosmo_exp, global_rank=self.global_rank
            )
        eigs, optimal_eig, nominal_eig, optimal_design = self.calc_eig(flow_model)
        central_vals = self.likelihood.central_val if self.run_args["include_D_M"] else self.likelihood.central_val[1::2]
        optimal_context = torch.cat([optimal_design, central_vals], dim=-1)
        nominal_context = torch.cat([self.nominal_design, central_vals], dim=-1)
        print(f"Optimal context: {optimal_context} \nNominal context: {nominal_context}")
        optimal_samples = flow_model(optimal_context).sample((self.eval_args["n_samples"],)).cpu().numpy()
        nominal_samples = flow_model(nominal_context).sample((self.eval_args["n_samples"],)).cpu().numpy()
        # Rescale the hrdrag samples 
        optimal_samples[:, -1] *= 100000
        nominal_samples[:, -1] *= 100000
        optimal_samples_gd = getdist.MCSamples(samples=optimal_samples, names=self.likelihood.cosmo_params, labels=self.likelihood.latex_labels)
        nominal_samples_gd = getdist.MCSamples(samples=nominal_samples, names=self.likelihood.cosmo_params, labels=self.likelihood.latex_labels)
        desi_samples = np.load(f"{home_dir}/data/mcmc_samples/{self.run_args['cosmo_model']}.npy")
        desi_samples_gd = getdist.MCSamples(samples=desi_samples, names=self.likelihood.cosmo_params, labels=self.likelihood.latex_labels)
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
        g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(1, 0.99))
        save_path = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/plots/posterior_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        show_figure(save_path)

    def calc_eig(self, flow_model):
        eigs, optimal_eig, nominal_eig, optimal_design = eval_designs(
            self.designs, self.nominal_design, self.run_args, 
            flow_model, self.likelihood, self.likelihood.cosmo_params, 
            n_evals=self.n_evals, eval_particles=self.eval_particles
            )
        return eigs, optimal_eig, nominal_eig, optimal_design
    
    def eig_grid(self, step):
        """
        Plots the EIG on a 2D grid for a given step.
        """
        flow_model = load_model(
            self.likelihood, step, self.run_obj, 
            self.run_args, self.classes, self.eval_args, 
            self.cosmo_exp, global_rank=self.global_rank
            )
        eigs, _, _, optimal_design = self.calc_eig(flow_model)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.12, 0.1, 0.68, 0.8])
        sc = ax.scatter(
            self.designs[:, 0].cpu().numpy(), self.designs[:, 1].cpu().numpy(),
            c=eigs, cmap='viridis', s=60, alpha=0.8, marker='o',
            vmin=np.min(eigs), vmax=np.max(eigs)
        )
        # Add optimal and nominal markers
        ax.scatter(optimal_design[0].cpu().numpy(), optimal_design[1].cpu().numpy(),
                c='red', s=70, alpha=0.8, marker='*', label='Optimal Design')
        ax.scatter(self.nominal_design[0].cpu().numpy(), self.nominal_design[1].cpu().numpy(),
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

        save_path = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/plots/eig_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        show_figure(save_path)

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

        checkpoint_dir = f'{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/checkpoints/'
        if not os.path.isdir(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found for run {self.run_id}, skipping. Path: {checkpoint_dir}")
            return
        for i, step in enumerate(steps):
            _, selected_step = get_checkpoint(step, checkpoint_dir, self.eval_args["device"], self.global_rank, self.run_args["total_steps"])
            samples = run_eval(self.run_obj, self.run_args, self.eval_args, step=selected_step, cosmo_exp=self.cosmo_exp, global_rank=self.global_rank)
            all_samples.append(samples)
            color_list.append(colors[i % len(colors)])
            area = get_contour_area(samples, 'Om', 'hrdrag', level)[0]
            all_areas.append(area)

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

        # Create custom legend
        custom_legend = []
        for step in steps:
            if step == 'last':
                step_label = self.run_args["total_steps"]
            elif step == 'loss_best':
                step_label = 'Best Loss'
            else:
                step_label = step
            
            # Get the area for this step (first run's area from the tuple)
            area = all_areas[i]
            
            custom_legend.append(
                Line2D([0], [0], color=colors[i % len(colors)], 
                        label=f'Step {step_label}, {int(level*100)}% Area: {area:.2f}')
            )
        custom_legend.append(
            Line2D([0], [0], color='black', 
                label=f'DESI, Area ({int(level*100)}% Contour): {desi_area:.3f}')
        )
        g.fig.legend(handles=custom_legend, loc='upper right', bbox_to_anchor=(1, 0.99))
        g.fig.suptitle(f"Posterior Steps for Run: {self.run_obj.info.run_name} ({self.run_id[:8]})")
        save_path = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/plots/posterior_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        show_figure(save_path)

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
        for s in steps:
            _, selected_step = get_checkpoint(s, checkpoint_dir, self.eval_args["device"], self.global_rank, self.run_args["total_steps"])
            flow_model = load_model(self.likelihood, selected_step, self.run_obj, self.run_args, self.classes, self.eval_args, self.cosmo_exp, global_rank=self.global_rank)
            eigs, _, nominal_eig, _ = self.calc_eig(flow_model)
            plt.plot(eigs, label=f'Step {selected_step}')
        plt.axhline(y=nominal_eig, color='black', linestyle='--', label='Nominal EIG')

        plt.xlabel("Design Index")
        plt.ylabel("EIG")
        plt.legend()
        plt.suptitle(f"EIG Steps for Run: {self.run_obj.info.run_name} ({self.run_id[:8]})")
        plt.tight_layout()
        save_path = f"{self.storage_path}/mlruns/{self.exp_id}/{self.run_id}/artifacts/plots/eig_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        show_figure(save_path)

def eval_design(
        design_low, 
        design_step, 
        run_id, 
        eval_step, 
        n_samples, 
        device, 
        eval_seed, 
        cosmo_exp, 
        levels, 
        global_rank, 
        eval_particles,
        n_evals
        ):
    
    total_observations = 6565626
    desi_tracers = pd.read_csv(os.environ["HOME"] + "/data/tracers_v1/desi_tracers.csv")
    classes = (desi_tracers.groupby('class').sum()['targets'].reindex(["LRG", "ELG", "QSO"]) / total_observations).to_dict()
    # enforce lows:
    classes["LRG"] = (0.0, classes["LRG"])
    classes["ELG"] = (0.0, classes["ELG"])
    classes["QSO"] = (0.0, classes["QSO"])

    # Create design grid with specified step size
    designs_dict = {
        f'N_{class_name}': np.arange(
            max(class_frac[0], design_low),
            min(class_frac[1] + design_step, 1.0), 
            design_step
        ) for class_name, class_frac in classes.items()
    }

    # Create constrained grid ensuring designs sum to 1
    tol = 1e-3
    grid_designs = Grid(
        **designs_dict, 
        constraint=lambda **kwargs: abs(sum(kwargs.values()) - 1.0) < tol
    )

    # Convert to tensor format
    designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device="cuda:0").unsqueeze(1)
    for name in grid_designs.names[1:]:
        design_tensor = torch.tensor(getattr(grid_designs, name).squeeze(), device="cuda:0").unsqueeze(1)
        designs = torch.cat((designs, design_tensor), dim=1)

    eval_args = {"n_samples": n_samples, "device": device, "eval_seed": eval_seed}

    # Create evaluation plots
    eval_plots = Evaluate(
        designs=designs,
        run_id=run_id,
        eval_args=eval_args,
        cosmo_exp=cosmo_exp,
        levels=levels,
        global_rank=global_rank,
        n_evals=n_evals,
        eval_particles=eval_particles
    )

    eval_plots.posterior(step=eval_step)
    eval_plots.eig_grid(step=eval_step)
    eval_plots.posterior_steps(steps=[eval_step//4, eval_step//2, eval_step, 'last'])
    eval_plots.eig_steps(steps=[eval_step//4, eval_step//2, eval_step, 'last'])
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Number Tracers Training")
    parser.add_argument('--run_id', type=str, default=None, help='MLflow run ID to resume training from (continues existing run with same parameters)')
    parser.add_argument('--eval_step', type=int, default=None, help='Step to resume training from (required when using --resume_id)')
    parser.add_argument('--cosmo_exp', type=str, default='num_tracers', help='Cosmological model set to use from run_args.json')
    parser.add_argument('--levels', type=float, default=[0.68, 0.95], help='Levels for contour plot')
    parser.add_argument('--global_rank', type=int, default=0, help='Global rank')
    parser.add_argument('--eval_particles', type=int, default=1000, help='Number of particles to use for evaluation')
    parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples to generate from the posterior')
    parser.add_argument('--design_low', type=float, default=0.05, help='Lowest design fraction')
    parser.add_argument('--design_step', type=float, default=0.05, help='Step size for design grid')
    parser.add_argument('--n_evals', type=int, default=20, help='Number of evaluations to average over')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for evaluation')
    parser.add_argument('--eval_seed', type=int, default=1, help='Seed for evaluation')

    args = parser.parse_args()

    eval_design(**vars(args))