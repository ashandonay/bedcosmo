
import os
import sys
parent_dir_abs = os.path.abspath(os.pardir)
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

from nflows.transforms import made as made_module
from bed.grid import Grid

from astropy.cosmology import Planck18
from astropy import constants

import mlflow
import mlflow.pytorch
from tqdm import trange

import zuko

from bed.grid import Grid, TopHat, CosineBump, Gaussian
from bed.design import ExperimentDesigner
from bed.plot import cornerPlot

import psutil
import os
import gc
from num_tracers import NumTracers
from util import *
import json

def single_run(
    cosmo_model,
    train_args,
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

    data_path = train_args["data_path"]

    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.log_param("cosmo_model", cosmo_model)
    # log params in train_args
    for key, value in train_args.items():
        mlflow.log_param(key, value)
    # log params in kwargs
    for key, value in kwargs.items():
        mlflow.log_param(key, value)
    ml_info = mlflow.active_run().info

    desi_df = pd.read_csv(data_path + 'desi_data.csv')
    desi_tracers = pd.read_csv(data_path + 'desi_tracers.csv')
    nominal_cov = np.load(data_path + 'desi_cov.npy')
    # select only the rows corresponding to the tracers
    

    #desi_data = desi_df[desi_df['tracer'].isin(train_args["tracers"])]
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
        eff=kwargs["eff"], 
        include_D_M=train_args["include_D_M"], 
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
                train_args["design_low"],
                class_frac + train_args["design_step"], 
                train_args["design_step"]
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
    np.save(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/designs.npy", designs.squeeze().cpu().detach().numpy())

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
    plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/prior.png")

    print("Calculating normalizing flow EIG...")
    input_dim = len(target_labels)
    context_dim = len(classes.keys()) + 10 if train_args["include_D_M"] else len(classes.keys()) + 5
    print(f'Input dim: {input_dim}, Context dim: {context_dim}')
    posterior_flow = init_nf(
        train_args["flow_type"],
        input_dim, 
        context_dim,
        train_args["n_transforms"],
        train_args["hidden"],
        train_args["num_layers"],
        device,
        seed=train_args["nf_seed"],
        verbose=True
        )
    # test sample from the flow
    with torch.no_grad():
        nominal_design = torch.tensor(desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=device)
        central_vals = num_tracers.central_val if train_args["include_D_M"] else num_tracers.central_val[1::2]
        nominal_context = torch.cat([nominal_design, central_vals], dim=-1)
        samples = posterior_flow(nominal_context).sample((100,)).cpu().numpy()
        plt.figure()
        plt.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.5)
        plt.savefig("init_samples.png")

    seed = auto_seed(train_args["pyro_seed"])
    print(f"Seed: {seed}")
    optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=train_args["lr"])
    if checkpoint_path is not None:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        posterior_flow.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=train_args["lr"])
        print(f"Checkpoint loaded from {checkpoint_path}")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_args["gamma"])

    mlflow.log_param("lr", train_args["lr"])
    mlflow.log_param("gamma", train_args["gamma"])
    mlflow.log_param("gamma_freq", train_args["gamma_freq"])
    verbose_shapes = train_args["verbose"]
    history = []
    print("MLFlow Run Info:", ml_info.experiment_id + "/" + ml_info.run_id)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    total_steps = train_args["steps"]
    plot_steps = [int(0.25*total_steps), int(0.5*total_steps), int(0.75*total_steps)]
    num_steps_range = trange(0, train_args["steps"], desc="Loss: 0.000 ")
    best_loss = float('inf')
    os.makedirs(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/best_loss", exist_ok=True)
    os.makedirs(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints", exist_ok=True)
    for step in num_steps_range:
        optimizer.zero_grad() #  clear gradients from previous step
        agg_loss, loss = posterior_loss(design=designs,
                                        model=num_tracers.pyro_model,
                                        guide=posterior_flow,
                                        num_particles=train_args["n_particles"],
                                        observation_labels=["y"],
                                        target_labels=target_labels,
                                        evaluation=False,
                                        nflow=True,
                                        condition_design=train_args["condition_design"],
                                        verbose_shapes=verbose_shapes)
        agg_loss.backward()
        optimizer.step()

        if step == 0:
            verbose_shapes = False
        if step in plot_steps:
            with torch.no_grad():
                agg_loss, eigs = posterior_loss(design=designs,
                                                model=num_tracers.pyro_model,
                                                guide=posterior_flow,
                                                num_particles=train_args["eval_particles"],
                                                observation_labels=["y"],
                                                target_labels=target_labels,
                                                evaluation=True,
                                                nflow=True,
                                                analytic_prior=False,
                                                condition_design=train_args["condition_design"])
            eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
            ax2.plot(eigs_bits, label=f'step {step}')
        history.append(loss.cpu().detach().item())
        if loss.cpu().detach().item() < best_loss:
            best_loss = loss.cpu().detach().item()
            # save the checkpoint
            checkpoint_path = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/best_loss/nf_checkpoint_{step}.pt"
            save_checkpoint(posterior_flow, optimizer, checkpoint_path, artifact_path="best_loss")
            checkpoint_path = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/nf_checkpoint_best.pt"
            save_checkpoint(posterior_flow, optimizer, checkpoint_path, artifact_path="checkpoints")
        mlflow.log_metric("best_loss", best_loss, step=step)

        if step % 25 == 0:
            # log the current learning rate
            for param_group in optimizer.param_groups:
                mlflow.log_metric("lr", param_group['lr'], step=step)
            mlflow.log_metric("loss", loss.mean().item(), step=step)
            mlflow.log_metric("agg_loss", agg_loss.item(), step=step)
            num_steps_range.set_description("Loss: {:.3f} ".format(loss.mean().item()))
        if step % train_args["gamma_freq"] == 0 and step > 0:
            scheduler.step()
        if step % 5000 == 0 and step > 0:
            checkpoint_path = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/nf_checkpoint_{step}.pt"
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
                                        num_particles=train_args["eval_particles"],
                                        observation_labels=["y"],
                                        target_labels=target_labels,
                                        evaluation=True,
                                        nflow=True,
                                        analytic_prior=False,
                                        condition_design=train_args["condition_design"])
    eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
    ax2.plot(eigs_bits, label=f'NF step {train_args["steps"]}')

    with torch.no_grad():
        agg_loss, nominal_eig = posterior_loss(design=nominal_design.unsqueeze(0),
                                        model=num_tracers.pyro_model,
                                        guide=posterior_flow,
                                        num_particles=train_args["eval_particles"],
                                        observation_labels=["y"],
                                        target_labels=target_labels,
                                        evaluation=True,
                                        nflow=True,
                                        analytic_prior=False,
                                        condition_design=train_args["condition_design"])
    nominal_eig_bits = nominal_eig.cpu().detach().numpy()/np.log(2)
    ax2.axhline(y=nominal_eig_bits, color='black', linestyle='--', label='Nominal EIG')
    ax2.set_xlabel("Design Index")
    ax2.set_ylabel("EIG")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/training.png")
    checkpoint_path = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/nf_checkpoint_last.pt"
    save_checkpoint(posterior_flow, optimizer, checkpoint_path)

    plt.close('all')
    print("Run", ml_info.experiment_id + "/" + ml_info.run_id, "completed.")



if __name__ == '__main__':

    #set default dtype
    torch.set_default_dtype(torch.float64)
    with open('train_args.json', 'r') as f:
        train_args_dict = json.load(f)
    process = psutil.Process(os.getpid())

    device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"

    for s in [1,2,3,4]:
        cosmo_model = 'base'
        train_args = train_args_dict[cosmo_model]
        train_args['pyro_seed'] = s
        train_args['n_particles'] = 50000
        print(f'Using device: {device}.')
        single_run(
            cosmo_model,
            train_args,
            f"{cosmo_model}_{train_args['flow_type']}_pyro1_fixed",
            device=device,
            signal=16,
            fixed_design=True,
            eff=True,
            )
        mlflow.end_run()


