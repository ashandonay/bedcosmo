import os
import sys
# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory ('BED_cosmo/') and add it to the Python path
parent_dir_abs = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, parent_dir_abs)

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from scipy.stats import norm
from urllib.request import urlopen

import pyro
from pyro import poutine
from pyro import distributions as dist
from pyro_oed_src import posterior_loss
from pyro.contrib.util import lexpand, rexpand

import matplotlib.pyplot as plt
import seaborn as sns
import corner
import getdist

from nflows.transforms import made as made_module
from bed.grid import Grid

from astropy.cosmology import Planck18
from astropy import constants

import mlflow
import mlflow.pytorch
home_dir = os.environ["HOME"]
mlflow.set_tracking_uri(home_dir + "/bed/BED_cosmo/num_tracers/mlruns")
from tqdm import trange

from bed.grid import Grid

import psutil
import os
import gc
from num_tracers import NumTracers
from util import *
import json
import argparse
from plotting import get_contour_area

def single_run(
    cosmo_model,
    run_args,
    mlflow_experiment_name,
    device="cuda:0",
    checkpoint_path=None,
    **kwargs,
):
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Trigger garbage collection for CPU
    gc.collect()
    print(f"Memory before run: {process.memory_info().rss / 1024**2} MB")
    pyro.clear_param_store()
    print(json.dumps(run_args, indent=2))
    data_path = home_dir + run_args["data_path"]

    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.log_param("cosmo_model", cosmo_model)
    # log params in run_args
    for key, value in run_args.items():
        mlflow.log_param(key, value)
    # log params in kwargs
    for key, value in kwargs.items():
        mlflow.log_param(key, value)
    ml_info = mlflow.active_run().info

    desi_df = pd.read_csv(data_path + 'desi_data.csv')
    desi_tracers = pd.read_csv(data_path + 'desi_tracers.csv')
    nominal_cov = np.load(data_path + 'desi_cov.npy')
    # select only the rows corresponding to the tracers
    

    #desi_data = desi_df[desi_df['tracer'].isin(run_args["tracers"])]
    #nominal_cov = desi_cov[np.ix_(desi_data.index, desi_data.index)]

    ############################################### Priors ###############################################

    total_observations = 6565626
    #classes = kwargs['classes']
    classes = (desi_tracers.groupby('class').sum()['targets'].reindex(["LRG", "ELG", "QSO"]) / total_observations).to_dict()
    mlflow.log_dict(classes, "classes.json")

    num_tracers = NumTracers(
        desi_df,
        desi_tracers,
        cosmo_model,
        nominal_cov,
        include_D_M=run_args["include_D_M"], 
        device=device,
        verbose=True
        )
    
    target_labels = num_tracers.cosmo_params
    print(f"Classes: {classes}\n"
        f"Cosmology: {cosmo_model}\n"
        f"Target labels: {target_labels}")

    ############################################### Designs ###############################################
    # if fixed design:
    if kwargs["fixed_design"]:
        # Get nominal design from observed tracer counts
        nominal_design = torch.tensor(desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=device)
        
        # Create grid with nominal design values
        grid_designs = Grid(
            N_LRG=nominal_design[0].cpu().numpy(), 
            N_ELG=nominal_design[1].cpu().numpy(), 
            N_QSO=nominal_design[2].cpu().numpy()
        )

        # Convert grid to tensor format
        designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=device).unsqueeze(0)
        for name in grid_designs.names[1:]:
            design_tensor = torch.tensor(getattr(grid_designs, name).squeeze(), device=device).unsqueeze(0)
            designs = torch.cat((designs, design_tensor), dim=0)
        designs = designs.unsqueeze(0)

    else:
        # Create design grid with specified step size
        designs_dict = {
            f'N_{class_name}': np.arange(
                run_args["design_low"],
                class_frac + run_args["design_step"], 
                run_args["design_step"]
            ) for class_name, class_frac in classes.items()
        }

        # Create constrained grid ensuring designs sum to 1
        tol = 1e-3
        grid_designs = Grid(
            **designs_dict, 
            constraint=lambda **kwargs: abs(sum(kwargs.values()) - 1.0) < tol
        )

        # Convert to tensor format
        designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=device).unsqueeze(1)
        for name in grid_designs.names[1:]:
            design_tensor = torch.tensor(getattr(grid_designs, name).squeeze(), device=device).unsqueeze(1)
            designs = torch.cat((designs, design_tensor), dim=1)
    print("Designs shape:", designs.shape)
    np.save(f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/designs.npy", designs.squeeze().cpu().detach().numpy())

    fig, axs = plt.subplots(ncols=len(target_labels), nrows=1, figsize=(5*len(target_labels), 5))
    if len(target_labels) == 1:
        axs = [axs]
    for i, p in enumerate(target_labels):
        support = num_tracers.priors[p].support
        eval_pts = torch.linspace(support.lower_bound, support.upper_bound, 200, device=device)
        prob = torch.exp(num_tracers.priors[p].log_prob(eval_pts))[:-1]
        prob_norm = prob/torch.sum(prob)
        axs[i].plot(eval_pts.cpu().numpy()[:-1], prob_norm.cpu().numpy(), label="Prior", color="tab:blue", alpha=0.5)
        axs[i].set_title(p)
    plt.tight_layout()
    plt.savefig(f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/prior.png")

    print("Calculating normalizing flow EIG...")
    input_dim = len(target_labels)
    context_dim = len(classes.keys()) + 10 if run_args["include_D_M"] else len(classes.keys()) + 5
    print(f'Input dim: {input_dim}, Context dim: {context_dim}')

    posterior_flow = init_nf(
        run_args["flow_type"],
        input_dim, 
        context_dim,
        run_args,
        device,
        seed=run_args["nf_seed"],
        verbose=True
        )
    nominal_design = torch.tensor(desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=device)
    central_vals = num_tracers.central_val if run_args["include_D_M"] else num_tracers.central_val[1::2]
    nominal_context = torch.cat([nominal_design, central_vals], dim=-1)
    # test sample from the flow
    with torch.no_grad():
        samples = posterior_flow(nominal_context).sample((100,)).cpu().numpy()
        plt.figure()
        plt.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.5)
        plt.savefig(f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/init_samples.png")

    seed = auto_seed(run_args["pyro_seed"])
    print(f"Seed: {seed}")
    optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=run_args["lr"])
    if checkpoint_path is not None:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        posterior_flow.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=run_args["lr"])
        print(f"Checkpoint loaded from {checkpoint_path}")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=run_args["gamma"])

    mlflow.log_param("lr", run_args["lr"])
    mlflow.log_param("gamma", run_args["gamma"])
    mlflow.log_param("gamma_freq", run_args["gamma_freq"])
    verbose_shapes = run_args["verbose"]
    history = []
    print("MLFlow Run Info:", ml_info.experiment_id + "/" + ml_info.run_id)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    total_steps = run_args["steps"]
    plot_steps = [int(0.25*total_steps), int(0.5*total_steps), int(0.75*total_steps)]
    # Disable tqdm progress bar if output is not a TTY
    is_tty = sys.stdout.isatty()
    num_steps_range = trange(0, run_args["steps"], desc="Loss: 0.000 ", disable=not is_tty)
    best_loss = float('inf')
    best_area = float('inf')
    os.makedirs(f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints", exist_ok=True)
    for step in num_steps_range:
        optimizer.zero_grad() #  clear gradients from previous step
        agg_loss, loss = posterior_loss(design=designs,
                                        model=num_tracers.pyro_model,
                                        guide=posterior_flow,
                                        num_particles=run_args["n_particles"],
                                        observation_labels=["y"],
                                        target_labels=target_labels,
                                        evaluation=False,
                                        nflow=True,
                                        condition_design=run_args["condition_design"],
                                        verbose_shapes=verbose_shapes)
        agg_loss.backward()
        # Clip gradients to prevent large jumps
        torch.nn.utils.clip_grad_norm_(posterior_flow.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 0:
            verbose_shapes = False
        if step in plot_steps:
            with torch.no_grad():
                agg_loss, eigs = posterior_loss(design=designs,
                                                model=num_tracers.pyro_model,
                                                guide=posterior_flow,
                                                num_particles=run_args["eval_particles"],
                                                observation_labels=["y"],
                                                target_labels=target_labels,
                                                evaluation=True,
                                                nflow=True,
                                                analytic_prior=False,
                                                condition_design=run_args["condition_design"])
            eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
            ax2.plot(eigs_bits, label=f'step {step}')
        history.append(loss.cpu().detach().item())
        if loss.cpu().detach().item() < best_loss and step > 999:
            best_loss = loss.cpu().detach().item()
            # save the checkpoint
            checkpoint_path = f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/nf_loss_checkpoint_{step}.pt"
            save_checkpoint(posterior_flow, optimizer, checkpoint_path, artifact_path="checkpoints")
            checkpoint_path = f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/nf_checkpoint_best_loss.pt"
            save_checkpoint(posterior_flow, optimizer, checkpoint_path, artifact_path="checkpoints")
            mlflow.log_metric("best_loss", best_loss, step=step)

        # log learning rate and loss
        for param_group in optimizer.param_groups:
            mlflow.log_metric("lr", param_group['lr'], step=step)
        mlflow.log_metric("loss", loss.mean().item(), step=step)
        mlflow.log_metric("agg_loss", agg_loss.item(), step=step)
        if step % 100 == 0:
            if step > 999:
                nominal_samples = posterior_flow(nominal_context).sample((30000,)).cpu().numpy()
                nominal_samples[:, -1] *= 100000
                with contextlib.redirect_stdout(io.StringIO()):
                    nominal_samples_gd = getdist.MCSamples(samples=nominal_samples, names=target_labels, labels=num_tracers.latex_labels)
                nominal_area = get_contour_area(nominal_samples_gd, 'Om', 'hrdrag', 0.68)[0]
                mlflow.log_metric("nominal_area", nominal_area, step=step)
                if nominal_area < best_area:
                    best_area = nominal_area
                    mlflow.log_metric("best_nominal_area", best_area, step=step)
                    # save the checkpoint
                    checkpoint_path = f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/nf_area_checkpoint_{step}.pt"
                    save_checkpoint(posterior_flow, optimizer, checkpoint_path, artifact_path="checkpoints")
                    checkpoint_path = f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/nf_checkpoint_best_area.pt"
                    save_checkpoint(posterior_flow, optimizer, checkpoint_path, artifact_path="checkpoints")
            else:
                nominal_area = np.nan
            # Only update description if running in a TTY
            if is_tty:
                num_steps_range.set_description("Loss: {:.3f}, Area: {:.3f}".format(loss.mean().item(), nominal_area))
            else:
                print(f"Step {step}, Loss: {loss.mean().item()}")
        if step % run_args["gamma_freq"] == 0 and step > 0:
            scheduler.step()
        if step % 5000 == 0 and step > 0:
            checkpoint_path = f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/nf_checkpoint_{step}.pt"
            save_checkpoint(posterior_flow, optimizer, checkpoint_path, artifact_path="checkpoints")

        # Track memory usage
        cpu_memory = process.memory_info().rss / 1024**2  # Convert to MB
        mlflow.log_metric("cpu_memory_usage", cpu_memory, step=step)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            mlflow.log_metric("gpu_memory_usage", gpu_memory, step=step)

    stacked_history = torch.tensor(history)
    ax1.plot(stacked_history - stacked_history.min())
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_yscale('log')

    nominal_design = torch.tensor(desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=device)
    print("Nominal design:", nominal_design)

    with torch.no_grad():
        agg_loss, eigs = posterior_loss(design=designs,
                                        model=num_tracers.pyro_model,
                                        guide=posterior_flow,
                                        num_particles=run_args["eval_particles"],
                                        observation_labels=["y"],
                                        target_labels=target_labels,
                                        evaluation=True,
                                        nflow=True,
                                        analytic_prior=False,
                                        condition_design=run_args["condition_design"])
    eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
    ax2.plot(eigs_bits, label=f'NF step {run_args["steps"]}')

    with torch.no_grad():
        agg_loss, nominal_eig = posterior_loss(design=nominal_design.unsqueeze(0),
                                        model=num_tracers.pyro_model,
                                        guide=posterior_flow,
                                        num_particles=run_args["eval_particles"],
                                        observation_labels=["y"],
                                        target_labels=target_labels,
                                        evaluation=True,
                                        nflow=True,
                                        analytic_prior=False,
                                        condition_design=run_args["condition_design"])
    nominal_eig_bits = nominal_eig.cpu().detach().numpy()/np.log(2)
    ax2.axhline(y=nominal_eig_bits, color='black', linestyle='--', label='Nominal EIG')
    ax2.set_xlabel("Design Index")
    ax2.set_ylabel("EIG")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/training.png")
    checkpoint_path = f"{home_dir}/bed/BED_cosmo/num_tracers/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/nf_checkpoint_last.pt"
    save_checkpoint(posterior_flow, optimizer, checkpoint_path)

    plt.close('all')
    print("Run", ml_info.experiment_id + "/" + ml_info.run_id, "completed.")



if __name__ == '__main__':

    #set default dtype
    torch.set_default_dtype(torch.float64)

    # --- Load Default Config --- 
    config_path = os.path.join(os.path.dirname(__file__), 'run_args.json')
    with open(config_path, 'r') as f:
        run_args_dict = json.load(f)

    # --- Argument Parsing --- 
    cosmo_model_default = 'base' 
    default_args = run_args_dict[cosmo_model_default]
    default_exp_name = f"{cosmo_model_default}_{default_args['flow_type']}"

    parser = argparse.ArgumentParser(description="Run Number Tracers Training")

    # Add arguments dynamically based on the default config file
    parser.add_argument('--cosmo_model', type=str, default=cosmo_model_default, help='Cosmological model set to use from run_args.json')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--exp_name', type=str, default=default_exp_name, help='Experiment name')
    for key, value in default_args.items():
        arg_type = type(value)
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true', help=f'Enable {key}')
            # Set default explicitly for bools, action handles the logic
            parser.set_defaults(**{key: value})
        elif isinstance(value, (int, float, str)):
             parser.add_argument(f'--{key}', type=arg_type, default=None, help=f'Override {key} (default: {value})')
        else:
            print(f"Warning: Argument type for key '{key}' not explicitly handled ({arg_type}). Treating as string.")
            parser.add_argument(f'--{key}', type=str, default=None, help=f'Override {key} (default: {value})')

    args = parser.parse_args()
    cosmo_model = args.cosmo_model
    device = args.device

    # --- Prepare Final Config --- 
    run_args = run_args_dict[cosmo_model].copy() # Start with defaults for the chosen model

    # Override defaults with any provided command-line arguments
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key != 'cosmo_model' and value is not None and key in run_args:
            if isinstance(run_args[key], bool) and isinstance(value, bool):
                run_args[key] = value
            elif not isinstance(run_args[key], bool):
                print(f"Overriding '{key}': {run_args[key]} -> {value}")
                run_args[key] = value

    # --- Setup & Run --- 
    process = psutil.Process(os.getpid())
    device = torch.device(device) if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')
    print(f"Running with parameters for cosmo_model='{cosmo_model}':")

    single_run(
        cosmo_model=cosmo_model,
        run_args=run_args,
        mlflow_experiment_name=args.exp_name,
        device=device,
        fixed_design=True,
    )
    mlflow.end_run()


