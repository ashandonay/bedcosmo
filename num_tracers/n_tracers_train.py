
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

def single_run(
    train_args,
    mlflow_experiment_name,
    checkpoint_path=None,
    **kwargs,
):
    pyro.clear_param_store()
    seed = auto_seed(train_args["seed"])
    data_path = train_args["data_path"]

    mlflow.set_experiment(mlflow_experiment_name)
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
    cosmo_model = train_args["cosmo_model"]
    # select only the rows corresponding to the tracers
    

    #desi_data = desi_df[desi_df['tracer'].isin(train_args["tracers"])]
    #nominal_cov = desi_cov[np.ix_(desi_data.index, desi_data.index)]

    ############################################### Priors ###############################################
    
    Om_mean = torch.tensor(Planck18.Om0).to(train_args["device"])
    Om_std = torch.tensor(0.01).to(train_args["device"])
    w0_mean = torch.tensor(-1.0).to(train_args["device"])
    w0_std = torch.tensor(0.2).to(train_args["device"])
    wa_mean = torch.tensor(0.0).to(train_args["device"])
    wa_std = torch.tensor(0.2).to(train_args["device"])

    Om_range = torch.tensor([0.01, 0.99], device=train_args["device"])
    Ok_range = torch.tensor([-0.3, 0.3], device=train_args["device"])
    w0_range = torch.tensor([-3.0, 1.0], device=train_args["device"])
    wa_range = torch.tensor([-3.0, 2.0], device=train_args["device"])
    hrdrag_range = torch.tensor([0.01, 1.0], device=train_args["device"])

    if cosmo_model == 'base':
        priors = {'Om': dist.Uniform(*Om_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
    elif cosmo_model == 'base_omegak':
        priors = {'Om': dist.Uniform(*Om_range), 'Ok': dist.Uniform(*Ok_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
    elif cosmo_model == 'base_w':
        priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
    elif cosmo_model == 'base_w_wa':
        priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range), 'hrdrag': dist.Uniform(*hrdrag_range)}

    observation_labels = ["y"]
    total_observations = 6565626
    #classes = kwargs['classes']
    classes = (desi_tracers.groupby('class').sum()['targets'].reindex(["LRG", "ELG", "QSO"]) / total_observations).to_dict()
    mlflow.log_dict(classes, "classes.json")
    target_labels = list(priors.keys())
    print(f"Classes: {classes}\n"
        f"Cosmology: {cosmo_model}\n"
        f"Observation labels: {observation_labels}\n"
        f"Target labels: {target_labels}")

    num_tracers = NumTracers(
        desi_df,
        desi_tracers,
        priors,
        cosmo_model,
        observation_labels,
        nominal_cov,
        eff=kwargs["eff"], 
        include_D_M=train_args["include_D_M"], 
        device=train_args["device"],
        verbose=True
        )

    ############################################### Designs ###############################################
    designs_dict = {}
    for c, u in classes.items():
        designs_dict['N_' + c] = np.arange(
            train_args["design_low"], 
            u + train_args["design_step"], 
            train_args["design_step"]
            )
    tol = 1e-3
    grid_designs = Grid(**designs_dict, constraint=lambda **kwargs: abs(sum(kwargs.values()) - 1.0) < tol)
    del designs_dict
    
    designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=device).unsqueeze(1)
    for n in grid_designs.names[1:]:
        designs = torch.cat((designs, torch.tensor(getattr(grid_designs, n).squeeze(), device=device).unsqueeze(1)), dim=1)
    print("Designs shape:", designs.shape)
    np.save(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/designs.npy", designs.squeeze().cpu().detach().numpy())

    fig, axs = plt.subplots(ncols=len(priors.keys()), nrows=1, figsize=(5*len(priors.keys()), 5))
    if len(priors.keys()) == 1:
        axs = [axs]
    for i, p in enumerate(priors.keys()):
        support = priors[p].support
        eval_pts = torch.linspace(support.lower_bound, support.upper_bound, 200, device=device)
        prob = torch.exp(priors[p].log_prob(eval_pts))[:-1]
        prob_norm = prob/torch.sum(prob)
        axs[i].plot(eval_pts.cpu().numpy()[:-1], prob_norm.cpu().numpy(), label="Prior", color="tab:blue", alpha=0.5)
        axs[i].set_title(p)
    plt.tight_layout()
    plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/prior.png")

    print("Calculating normalizing flow EIG...")
    input_dim = len(target_labels)
    context_dim = len(classes.keys()) + 10 if train_args["include_D_M"] else len(classes.keys()) + 5
    n_transforms = train_args["n_transforms"]
    hidden_features = ((train_args["hidden"],) + (2*train_args["hidden"],) * (train_args["num_layers"] - 1))
    print(f'Input dim: {input_dim}, Context dim: {context_dim}')
    print(f'Hidden features: {hidden_features}')
    posterior_flow = init_nf(
        train_args["flow_type"],
        input_dim, 
        context_dim, 
        n_transforms, 
        train_args["device"], 
        hidden_features=hidden_features
        )
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

    pyro.set_rng_seed(0)
    verbose_shapes = train_args["verbose"]
    history = []
    print("MLFlow Run Info:", ml_info.experiment_id + "/" + ml_info.run_id)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    total_steps = train_args["steps"]
    plot_steps = [int(0.25*total_steps), int(0.5*total_steps), int(0.75*total_steps)]
    num_eval_samples = 5000 # number of samples to evaluate the EIG
    num_steps_range = trange(0, train_args["steps"], desc="Loss: 0.000 ")
    for step in num_steps_range:
        optimizer.zero_grad() #  clear gradients from previous step
        agg_loss, loss = posterior_loss(design=designs,
                                        model=num_tracers.pyro_model,
                                        guide=posterior_flow,
                                        num_particles=train_args["n_particles"],
                                        observation_labels=observation_labels,
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
                                                observation_labels=observation_labels,
                                                target_labels=target_labels,
                                                evaluation=True,
                                                nflow=True,
                                                analytic_prior=False,
                                                condition_design=train_args["condition_design"])
            eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
            ax2.plot(eigs_bits, label=f'step {step}')
        history.append(loss)
        if step % 25 == 0:
            mlflow.log_metric("loss", loss.mean().item(), step=step)
            mlflow.log_metric("agg_loss", agg_loss.item(), step=step)
            num_steps_range.set_description("Loss: {:.3f} ".format(loss.mean().item()))
        if step % train_args["gamma_freq"] == 0 and step > 0:
            scheduler.step()
        if step % 5000 == 0 and step > 0:
            checkpoint_path = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nf_checkpoint_{step}.pt"
            save_checkpoint(posterior_flow, optimizer, checkpoint_path)

    stacked_history = torch.stack(history)
    history_array = stacked_history.cpu().detach().numpy()
    loss_history = history_array.sum(axis=1)/history_array.shape[-1]
    min_loss = loss_history.min()
    ax1.plot(loss_history+abs(min_loss))
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
                                        observation_labels=observation_labels,
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
                                        observation_labels=observation_labels,
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
    checkpoint_path = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nf_checkpoint_last.pt"
    save_checkpoint(posterior_flow, optimizer, checkpoint_path)

    plt.close('all')
    print("Run", ml_info.experiment_id + "/" + ml_info.run_id, "completed.")



if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')
    #set default dtype
    torch.set_default_dtype(torch.float64)

    process = psutil.Process(os.getpid())
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Trigger garbage collection for CPU
    gc.collect()
    print(f"Memory before run: {process.memory_info().rss / 1024**2} MB")
    train_args = {
        "seed": 1,
        "data_path": '/home/ashandonay/data/tracers_v1/',
        "cosmo_model": 'base_omegak',
        "include_D_M": True,
        "flow_type": 'NAF',
        "design_low": 0.05,
        "design_step": 0.05,
        "n_transforms": 2,
        "num_layers": 4,
        "hidden": 16,
        "steps": 25000,
        "lr": 1e-2,
        "gamma": 0.85,
        "gamma_freq": 500,
        "n_particles": 100,
        "eval_particles": 200,
        "verbose": False, 
        "condition_design": True,
        "device": device
        }   
    single_run(
        train_args,
        f"{train_args['cosmo_model']}_{train_args['flow_type']}",
        signal=16,
        eff=True,
        )
    mlflow.end_run()