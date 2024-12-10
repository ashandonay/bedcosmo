
import os
import sys
parent_dir_abs = os.path.abspath(os.pardir)
sys.path.insert(0, parent_dir_abs) 

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad, cumulative_trapezoid, trapezoid
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

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}.')

#set default dtype
torch.set_default_dtype(torch.float64)

def single_run(
    train_args,
    mlflow_experiment_name,
    checkpoint_path=None,
    **kwargs,
):

    desi_data = pd.read_csv('/home/ashandonay/bed/desi_data.csv')
    sigma_D_H = []
    sigma_D_M = []
    z_eff = []
    true_n_vals = []
    for t in train_args["tracers"]:
        D_H_std = desi_data.loc[(desi_data['tracer'] == t) & (desi_data['quantity'] == 'DH_over_rs'), "std"].values[0]
        D_M_std = desi_data.loc[(desi_data['tracer'] == t) & (desi_data['quantity'] == 'DM_over_rs'), "std"].values[0]
        z = desi_data.loc[desi_data['tracer'] == t, "z"].values[0]
        sigma_D_H.append(D_H_std * kwargs["multiplier"])
        sigma_D_M.append(D_M_std * kwargs["multiplier"])
        z_eff.append(z)
        true_n_vals.append(desi_data.loc[desi_data['tracer'] == t, "num"].values[0])
    
    sigma_D_H = torch.tensor(sigma_D_H, device=train_args["device"])
    sigma_D_M = torch.tensor(sigma_D_M, device=train_args["device"])
    z_eff = torch.tensor(z_eff, device=train_args["device"])
    true_n_vals = torch.tensor(true_n_vals, device=train_args["device"])
    true_n_ratios = true_n_vals/true_n_vals.sum()

    if not train_args["include_D_M"]:
        sigma_D_M = None
        print(f'tracers: {train_args["tracers"]}\n'
            f'z_eff: {z_eff}\n'
            f'sigma_D_H: {sigma_D_H}\n'
            f'true_n_ratios: {true_n_ratios}')
    else:
        print(f'tracers: {train_args["tracers"]}\n'
            f'z_eff: {z_eff}\n'
            f'sigma_D_H: {sigma_D_H}\n'
            f'sigma_D_M: {sigma_D_M}\n'
            f'true_n_ratios: {true_n_ratios}')

    Om_mean = torch.tensor(Planck18.Om0).to(train_args["device"])
    Om_std = torch.tensor(kwargs["Om_sigma"]).to(train_args["device"])
    w0_mean = torch.tensor(-1.0).to(train_args["device"])
    w0_std = torch.tensor(0.2).to(train_args["device"])
    wa_mean = torch.tensor(0.0).to(train_args["device"])
    wa_std = torch.tensor(0.2).to(train_args["device"])

    low = Planck18.Om0 - 5*Om_std
    high = Planck18.Om0 + 5*Om_std
    Om_range = torch.tensor([low, high], device=train_args["device"])
    print(f'Om range: {Om_range}')
    w0_range = torch.tensor([-3.0, 1.0], device=train_args["device"])
    wa_range = torch.tensor([-3.0, 2.0], device=train_args["device"])

    ############################################### Priors ###############################################
    if train_args["cosmology"] == 'Om':
        params = ["Om"]
        grid_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), kwargs["params_grid"]))
        #priors = {'Om': dist.Uniform(*Om_range)}
        #grid_prior = TopHat(grid_params.Om)
        priors = {'Om': dist.Normal(Om_mean, Om_std)}
        grid_prior = Gaussian(grid_params.Om, Planck18.Om0, kwargs["Om_sigma"])
    elif train_args["cosmology"] == 'w':
        params = ["Om", "w0"]
        priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range)}
        #priors = {'Om': dist.Normal(Om_mean, Om_std), 'w0': dist.Normal(w0_mean, w0_std)}
        grid_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), kwargs["params_grid"]), 
            w0=np.linspace(-3, 1, kwargs["params_grid"]))
        grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.w0)
        #grid_prior = Gaussian(grid_params.Om, Planck18.Om0, kwargs["Om_sigma"]) * TopHat(grid_params.w0)
    elif train_args["cosmology"] == 'w0wa':
        params = ["Om", "w0", "wa"]
        priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range)}
        #priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range)}
        grid_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), kwargs["params_grid"]), 
            w0=np.linspace(-3, 1, kwargs["params_grid"]), 
            wa=np.linspace(-3, 2, kwargs["params_grid"]))
        grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.w0) * TopHat(grid_params.wa)
        #grid_prior = Gaussian(grid_params.Om, Planck18.Om0, kwargs["Om_sigma"]) * TopHat(grid_params.w0) * TopHat(grid_params.wa)
    grid_params.normalize(grid_prior);

    observation_labels = ["y"]
    target_labels = list(priors.keys())
    print(f'Cosmology: {train_args["cosmology"]}')
    print(f'Observation Labels: {observation_labels}')
    print(f'Target Labels: {target_labels}')

    num_tracers = NumTracers(priors, true_n_ratios, observation_labels, z_eff, sigma_D_H, sigma_D_M, device=train_args["device"])


    ############################################### Features ###############################################
    if train_args["cosmology"] == 'Om':
        range_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), 1000))
    elif train_args["cosmology"] == 'w':
        range_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), 500), 
            w0=np.linspace(-3, 1, 500))
    elif train_args["cosmology"] == 'w0wa':
        range_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), 300), 
            w0=np.linspace(-3, 1, 300), 
            wa=np.linspace(-3, 2, 300))
    parameters = { }
    for key in range_params.names:
        parameters[key] = torch.tensor(getattr(range_params, key), device=train_args["device"])
    if train_args["include_D_M"]:
        features_dict = {}
        for i in range(len(train_args["tracers"])):
            D_H_mean = num_tracers.D_H_func(z_eff[i], **parameters)
            print('D_H_' + train_args["tracers"][i].split()[0] + ' range:', D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2)
            features_dict['D_H_' + train_args["tracers"][i].split()[0]] = np.linspace(D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2, kwargs["features_grid"])
        for i in range(len(train_args["tracers"])):
            z_array = z_eff[i].unsqueeze(-1) * torch.linspace(0, 1, 100, device=train_args["device"]).view(1, -1)
            z = z_array.expand((len(range_params.names)-1)*[1] + [-1, -1])
            D_M_mean = num_tracers.D_M_func(z, **parameters)
            print('D_M_' + train_args["tracers"][i].split()[0] + ' range:', D_M_mean.cpu().numpy().min(), D_M_mean.cpu().numpy().max() + 2)
            features_dict['D_M_' + train_args["tracers"][i].split()[0]] = np.linspace(D_M_mean.cpu().numpy().min(), D_M_mean.cpu().numpy().max() + 2, kwargs["features_grid"])
    else:
        features_dict = {}
        for i in range(len(train_args["tracers"])):
            D_H_mean = num_tracers.D_H_func(z_eff[i], **parameters)
            print('D_H_' + train_args["tracers"][i].split()[0] + ' range:', D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2)
            features_dict['D_H_' + train_args["tracers"][i].split()[0]] = np.linspace(D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2, kwargs["features_grid"])
    grid_features = Grid(**features_dict)
    del range_params, parameters, features_dict

    ############################################### Designs ###############################################
    base = 5.0
    designs_dict = {}
    for i in range(len(train_args["tracers"])):
        #designs_dict['N_' + train_args["tracers"][i].split()[0]] = np.arange(0.0, 0.3, 0.05)
        designs_dict['N_' + train_args["tracers"][i].split()[0]] = np.arange(
            np.round(true_n_ratios[i].cpu().numpy() - train_args["design_scope"], train_args["design_precision"]), 
            np.round(true_n_ratios[i].cpu().numpy() + train_args["design_scope"], train_args["design_precision"]), 
            np.power(base, -train_args["design_precision"])
            )
    grid_designs = Grid(**designs_dict, constraint=lambda **kwargs: np.abs(sum(kwargs.values()) - 1.0) < np.power(base, -(train_args["design_precision"] + 1)))
    del designs_dict
    print("Grid designs shape:", grid_designs.shape)
    
    designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=device).unsqueeze(1)
    for n in grid_designs.names[1:]:
        designs = torch.cat((designs, torch.tensor(getattr(grid_designs, n).squeeze(), device=device).unsqueeze(1)), dim=1)

    pyro.clear_param_store()
    seed = auto_seed(train_args["seed"])

    mlflow.set_experiment(mlflow_experiment_name)
    # log params in train_args
    for key, value in train_args.items():
        mlflow.log_param(key, value)

    # log params in kwargs
    for key, value in kwargs.items():
        mlflow.log_param(key, value)
    ml_info = mlflow.active_run().info
    print("MLFlow Run Info:", ml_info.experiment_id + "/" + ml_info.run_id)

    # plot priors
    #plt.figure()
    #cornerPlot(grid_prior, grid_params);
    #plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/grid_prior.png")

    fig, axs = plt.subplots(ncols=len(params), nrows=1, figsize=(5*len(params), 5))
    if len(params) == 1:
        axs = [axs]
    for i, p in enumerate(params):
        prior_samples = priors[p].sample((10000,))
        axs[i].hist(prior_samples.cpu().numpy(), bins=50, density=True)
        axs[i].set_title(p)
    plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/pyro_prior.png")

    if kwargs["brute_force"]:
        print("Calculating brute force EIG...")
        designer = ExperimentDesigner(grid_params, grid_features, grid_designs, num_tracers.unnorm_lfunc, mem=190000)
        designer.calculateEIG(grid_prior)
        brute_force_EIG = designer.EIG
        designer.describe()
        np.save(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/designs.npy", np.array([getattr(grid_designs, name) for name in grid_designs.names]).squeeze())
        np.save(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/brute_force_eigs.npy", brute_force_EIG)

        plt.figure()
        if len(grid_designs.expand(designer.EIG).shape) > 2:
            for i in range(grid_designs.expand(designer.EIG).shape[-1]):
                plt.imshow(grid_designs.expand(designer.EIG)[:,:,i].T, origin='lower',
                extent=grid_designs.extent(grid_designs.names[0])+grid_designs.extent(grid_designs.names[1]))
            plt.xlabel(grid_designs.names[0])
            plt.ylabel(grid_designs.names[1])
            plt.colorbar(label="EIG [bits]")
            plt.tight_layout()
            plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/brute_force.png")
        else:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            ax[0].plot(getattr(grid_designs, grid_designs.names[0]).squeeze(), brute_force_EIG.squeeze())
            ax[0].set_xlabel(grid_designs.names[0])
            ax[0].set_ylabel("EIG [bits]")
            I = ax[1].imshow(grid_designs.expand(designer.EIG).T, origin='lower', 
            extent=grid_designs.extent(grid_designs.names[0])+grid_designs.extent(grid_designs.names[1]))
            ax[1].set_xlabel(grid_designs.names[0])
            ax[1].set_ylabel(grid_designs.names[1])
            # set colorbar
            plt.colorbar(I, label="EIG [bits]", ax=ax[1])
            plt.tight_layout()
            plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/brute_force.png")

    if kwargs["nf_model"]:
        print("Calculating normalizing flow EIG...")
        input_dim = len(target_labels)
        context_dim = 3 * len(train_args["tracers"]) if train_args["include_D_M"] else 2 * len(train_args["tracers"])
        print(f'Input dim: {input_dim}, Context dim: {context_dim}')
        if train_args["flow_type"] == "NSF":
            posterior_flow = zuko.flows.NSF(
                features=input_dim, 
                context=context_dim, 
                transforms=train_args["n_transforms"], 
                bins=10,
                hidden_features=(kwargs["hidden"], 2*kwargs["hidden"], 2*kwargs["hidden"])
            ).to(train_args["device"])
            optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=train_args["lr"])
        elif train_args["flow_type"] == "NAF":
            posterior_flow = zuko.flows.NAF(
                features=input_dim, 
                context=context_dim, 
                transforms=train_args["n_transforms"],
                signal=kwargs["signal"],
                hidden_features=(kwargs["hidden"], 2*kwargs["hidden"], 2*kwargs["hidden"])
            ).to(train_args["device"])
            optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=train_args["lr"])
        elif train_args["flow_type"] == "MAF":
            posterior_flow = zuko.flows.MAF(
                features=input_dim, 
                context=context_dim, 
                transforms=train_args["n_transforms"],
                hidden_features=(kwargs["hidden"], 2*kwargs["hidden"], 2*kwargs["hidden"])
            ).to(train_args["device"])
            optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=train_args["lr"])
        elif train_args["flow_type"] == "MAF_Affine":
            posterior_flow = zuko.flows.MAF(
                features=input_dim, 
                context=context_dim, 
                transforms=train_args["n_transforms"],
                univariate=zuko.transforms.MonotonicAffineTransform,
                hidden_features=(kwargs["hidden"], 2*kwargs["hidden"], 2*kwargs["hidden"])  
            ).to(train_args["device"])
            optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=train_args["lr"])
        elif train_args["flow_type"] == "MAF_RQS":
            posterior_flow = zuko.flows.MAF(
                features=input_dim, 
                context=context_dim, 
                transforms=train_args["n_transforms"],
                univariate=zuko.transforms.MonotonicRQSTransform,
                shapes = ([kwargs["shape"]], [kwargs["shape"]], [kwargs["shape"]-1]),
                hidden_features=(kwargs["hidden"], 2*kwargs["hidden"], 2*kwargs["hidden"]),  
            ).to(train_args["device"])
            optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=train_args["lr"])
        elif train_args["flow_type"] == "affine_coupling":
            posterior_flow, transforms = pyro_flows.affine_coupling_flow(train_args["n_transforms"], input_dim, context_dim, [32, 32], train_args["device"])
            modules = torch.nn.ModuleList(transforms)
            optimizer = torch.optim.Adam(modules.parameters(), lr=train_args["lr"])
        elif train_args["flow_type"] == "GF":
            posterior_flow = zuko.flows.GF(
                features=input_dim, 
                context=context_dim, 
                transforms=train_args["n_transforms"]
            ).to(train_args["device"])
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
        verbose_shapes = train_args["train_verbose"]
        history = []
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
                                                    condition_design=train_args["condition_design"])
                eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
                ax2.plot(eigs_bits, label=f'step {step}')
            history.append(loss)
            if step % 25 == 0:
                mlflow.log_metric("loss", loss.mean().item(), step=step)
                mlflow.log_metric("agg_loss", agg_loss.item(), step=step)
                num_steps_range.set_description("Loss: {:.3f} ".format(loss.mean().item()))
                # Clear unused variables and cache
                del agg_loss, loss
                gc.collect()
                torch.cuda.empty_cache()
            if step % 1000 == 0:
                scheduler.step()

        stacked_history = torch.stack(history)
        history_array = stacked_history.cpu().detach().numpy()
        loss_history = history_array.sum(axis=1)/history_array.shape[-1]
        min_loss = loss_history.min()
        ax1.plot(loss_history+abs(min_loss))
        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Loss")
        ax1.set_yscale('log')

        with torch.no_grad():
            agg_loss, eigs = posterior_loss(design=designs,
                                            model=num_tracers.pyro_model,
                                            guide=posterior_flow,
                                            num_particles=train_args["eval_particles"],
                                            observation_labels=observation_labels,
                                            target_labels=target_labels,
                                            evaluation=True,
                                            nflow=True,
                                            condition_design=train_args["condition_design"])
        eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
        ax2.plot(eigs_bits, label=f'NF step {train_args["steps"]}')
        if kwargs["brute_force"]:
            ax2.plot(brute_force_EIG.squeeze(), label='Brute Force', color='black')
        ax2.set_xlabel("Design Index")
        ax2.set_ylabel("EIG")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/training.png")

        # make a corner plot of the posterior
        # assuming default params associated with data to condition the posterior
        if train_args["condition_design"]:
            if context_dim == 4:
                context = torch.tensor([500, 500, 19.993, 8.620], device=train_args["device"])
            elif context_dim == 6:
                context = torch.tensor([500, 500, 19.993, 8.620, 15.0, 38.0], device=train_args["device"])
        else:
            context = torch.tensor([19.993, 8.620], device=train_args["device"])
        #posterior_samples = posterior_flow(context).sample((20000,)).squeeze().cpu().detach().numpy()
        #figure = corner.corner(posterior_samples, labels=target_labels, show_titles=True, title_fmt=".2f", 
        #                   title_kwargs={"fontsize": 12}, ranges=[0.99]*len(target_labels))
        #plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/posterior.png")

        # save eigs at the end
        np.save(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/design_eigs.npy", designs.squeeze().cpu().detach().numpy())
        np.save(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/eigs.npy", eigs_bits)
        mlflow.log_artifact(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/eigs.npy")

        checkpoint_path = f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nf_checkpoint.pt"
        save_checkpoint(posterior_flow, optimizer, checkpoint_path)

        plt.close('all')
        print("Run", ml_info.experiment_id + "/" + ml_info.run_id, "completed.")



if __name__ == '__main__':

    #for s in np.array([0.01, 0.03, 0.05])*np.sqrt(12)/2:
    for l in [0.01, 0.03, 0.05]:
        process = psutil.Process(os.getpid())
        # Clear GPU cache
        torch.cuda.empty_cache()
        # Trigger garbage collection for CPU
        gc.collect()
        print(f"Memory before run: {process.memory_info().rss / 1024**2} MB")
        train_args = {
            "seed": 1,
            "cosmology": "Om", # Om, w, w0wa
            "tracers": ["LRG2", "Lya QSO"], # "LRG1, "LRG2", "LRG3 ELG1", "ELG2", "Lya QSO"
            "include_D_M": False,
            "flow_type": 'NAF',
            "design_precision": 2,
            "design_scope": 0.2,
            "n_transforms": 5,
            "steps": 7000,
            "lr": 5e-3,
            "gamma": 0.8,
            "n_particles": 101,
            "eval_particles": 10000,
            "train_verbose": True, 
            "condition_design": True,
            "device": device
            }   
        single_run(
            train_args,
            str(len(train_args["tracers"])) + "_tracers_sigma",
            signal=8,
            hidden=32,
            Om_low=0.0,
            Om_sigma=l,
            params_grid=500,
            features_grid=400,
            multiplier=1.0,
            brute_force=True,
            nf_model=True
            )
        mlflow.end_run()
                