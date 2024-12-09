
import os
import sys
parent_dir_abs = os.path.abspath(os.pardir)
sys.path.insert(0, parent_dir_abs) 

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad, cumulative_trapezoid
from urllib.request import urlopen

import pyro
from pyro import poutine
from pyro import distributions as dist
from pyro_oed_src import posterior_loss
from pyro.contrib.util import lexpand, rexpand

import matplotlib.pyplot as plt
import seaborn as sns

from nflows.transforms import made as made_module
import neural_nets
import flows
import pyro_flows
from bed.grid import Grid

from astropy.cosmology import Planck18
from astropy.cosmology import FLRW, w0waCDM, LambdaCDM, FlatLambdaCDM
from astropy import constants

import mlflow
import mlflow.pytorch
from tqdm import trange

import zuko

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(f'Using device: {device}.')

#set default dtype
torch.set_default_dtype(torch.float64)

class bed_cosmology:

    def __init__(self, cosmo_params, priors, sigma_D_H=0.2, sigma_D_M=0.2, include_D_M=False, device=device):
        self.cosmo_params = set(cosmo_params.names)
        self.priors = priors
        self.sigma_D_H = sigma_D_H
        self.sigma_D_M = sigma_D_M
        self.r_drag = 149.77
        self.H0 = Planck18.H0.value
        self.coeff = constants.c.to('km/s').value/(self.H0*self.r_drag)
        self.include_D_M = include_D_M
        self.device = device
    
    def D_H_func(self, z, Om, w0=None, wa=None):
        if self.cosmo_params == {'Om'}:
            return self.coeff / torch.sqrt(Om * (1+z)**3 + (1-Om))

        elif self.cosmo_params == {'Om', 'w0'}:
            return self.coeff / torch.sqrt(Om * (1+z)**3 + (1-Om) * (1+z)**(3*(1+w0)))

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            return self.coeff / torch.sqrt(Om * (1+z)**3 + (1-Om) * (1+z)**(3*(1+(w0+wa*(z/(1+z))))))
            
        else:
            raise ValueError(f"Unsupported cosmology model: {self.cosmo.name}")

    def D_M_func(self, z, Om, w0=None, wa=None):
        if self.cosmo_params == {'Om'}:
            result = self.coeff * cumulative_trapezoid((1 / torch.sqrt(Om * (1 + z)**3 + (1 - Om))).cpu(), z.cpu(), initial=0, axis=0)
            return torch.tensor(result).to(self.device)

        elif self.cosmo_params == {'Om', 'w0'}:
            result = self.coeff * cumulative_trapezoid((1 / torch.sqrt(Om * (1 + z)**3 + (1 - Om) * (1 + z)**(3 * (1 + w0)))).cpu(), z.cpu(), initial=0, axis=0)
            return torch.tensor(result).to(self.device)

        elif self.cosmo_params == {'Om', 'w0', 'wa'}:
            result = self.coeff * cumulative_trapezoid((1 / torch.sqrt(Om * (1 + z)**3 + (1 - Om) * (1 + z)**(3 * (1 + (w0 + wa * (z / (1 + z))))))).cpu(), z.cpu(), initial=0, axis=0)
            return torch.tensor(result).to(self.device)
            
        else:
            raise ValueError(f"Unsupported cosmology model: {self.cosmo.name}")

    def likelihood(self, params, features, designs):
        with GridStack(features, designs, params):
            # create a dictionary of the parameters
            kwargs = { }
            for key in params.names:
                kwargs[key] = getattr(params, key)

            D_H_mean = self.D_H_func(designs.z, **kwargs)
            D_H_diff = features.D_H - D_H_mean
            D_H_likelihood = np.exp(-0.5 * (D_H_diff / self.sigma_D_H) ** 2) 

            if self.include_D_M:
                D_M_mean = self.D_M_func(designs.z, **kwargs)
                D_M_diff = features.D_M - D_M_mean
                D_M_likelihood = np.exp(-0.5 * (D_M_diff / self.sigma_D_M) ** 2)
                likelihood = D_H_likelihood * D_M_likelihood
            else:
                likelihood = D_H_likelihood
            features.normalize(likelihood)
        return likelihood
    
    def pyro_model(self, z):
        with pyro.plate_stack("plate", z.shape[:-1]):
            kwargs = {}
            for i, (k, v) in enumerate(self.priors.items()):
                if isinstance(v, dist.Distribution):
                    kwargs[k] = pyro.sample(k, v).unsqueeze(-1)
                else:
                    kwargs[k] = v
            D_H_mean = self.D_H_func(z, **kwargs)
            if self.include_D_M:
                D_M_mean = self.D_M_func(z, **kwargs)
                return pyro.sample("D_H", dist.Normal(D_H_mean, self.sigma_D_H).to_event(1)), pyro.sample("D_M", dist.Normal(D_M_mean, self.sigma_D_M).to_event(1))
            else:
                return pyro.sample("D_H", dist.Normal(D_H_mean, self.sigma_D_H).to_event(1))

def auto_seed(seed):
    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        pyro.set_rng_seed(seed)
    return seed

def single_run(
    train_args,
    mlflow_experiment_name,
    **kwargs,
):

    designs = torch.linspace(0, 5, train_args["design_grid"], device=train_args["device"]).unsqueeze(-1)
    Om_range = torch.tensor([0.2, 0.4], device=train_args["device"])
    w0_range = torch.tensor([-3.0, 1.0], device=train_args["device"])
    wa_range = torch.tensor([-3.0, 2.0], device=train_args["device"])

    if train_args["cosmology"] == 'Om':
        priors = {'Om': dist.Uniform(*Om_range)}
        params = Grid(Om=np.linspace(0.2, 0.4, 120))
    elif train_args["cosmology"] == 'w':
        priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range)}
        params = Grid(Om=np.linspace(0.2, 0.4, 120), w0=np.linspace(-3, 1, 101))
    elif train_args["cosmology"] == 'w0wa':
        priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range)}
        params = Grid(Om=np.linspace(0.2, 0.4, 120), w0=np.linspace(-3, 1, 51), wa=np.linspace(-3, 2, 51))

    if not train_args["include_D_M"]:
        bed_cosmo = bed_cosmology(
            params,
            priors=priors,
            sigma_D_H=train_args["D_H_error"])
    else:
        bed_cosmo = bed_cosmology(
            params,
            priors=priors,
            sigma_D_H=train_args["D_H_error"],
            sigma_D_M=train_args["D_M_error"],
            include_D_M=True)

    target_labels = list(priors.keys())
    if train_args["include_D_M"]:
        observation_labels = ["D_H", "D_M"]
    else:
        observation_labels = ["D_H"]
    print(f'Cosmology: {train_args["cosmology"]}')
    print(f'Observation Labels: {observation_labels}')
    print(f'Target Labels: {target_labels}')

    pyro.clear_param_store()
    seed = auto_seed(train_args["seed"])

    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.log_param("seed", seed)
    mlflow.log_param("cosmology", train_args["cosmology"])
    mlflow.log_param("num_steps", train_args["steps"])
    mlflow.log_param("n_particles", train_args["n_particles"])
    mlflow.log_param("flow_type", train_args["flow_type"])
    mlflow.log_param("design_grid", train_args["design_grid"])
    mlflow.log_param("n_transforms", train_args["n_transforms"])
    mlflow.log_param("include_D_M", train_args["include_D_M"])
    mlflow.log_param("D_H_error", train_args["D_H_error"])
    if train_args["include_D_M"]:
        mlflow.log_param("D_M_error", train_args["D_M_error"])

    # log params in kwargs
    for key, value in kwargs.items():
        mlflow.log_param(key, value)


    input_dim = len(target_labels)
    context_dim = len(observation_labels) + 1
    if train_args["flow_type"] == "NSF":
        #posterior_flow, transforms = pyro_flows.spline_flow(train_args["n_transforms"], input_dim, context_dim, segments, train_args["device"])
        posterior_flow = zuko.flows.NSF(
            features=input_dim, 
            context=context_dim, 
            transforms=train_args["n_transforms"], 
            bins=10
        ).to(train_args["device"])
        optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=train_args["lr"])
    elif train_args["flow_type"] == "NAF":
        #posterior_flow, transforms = pyro_flows.affine_autoregressive_flow(train_args["n_transforms"], input_dim, context_dim, [32, 32], train_args["device"])
        #modules = torch.nn.ModuleList(transforms)
        #optimizer = torch.optim.Adam(modules.parameters(), lr=train_args["lr"])
        posterior_flow = zuko.flows.NAF(
            features=input_dim, 
            context=context_dim, 
            transforms=train_args["n_transforms"]
        ).to(train_args["device"])
        optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=train_args["lr"])
    elif train_args["flow_type"] == "MAF":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=train_args["n_transforms"],
            hidden_features=(64, 128, 128)
        ).to(train_args["device"])
        optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=train_args["lr"])
    elif train_args["flow_type"] == "MAF_Affine":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=train_args["n_transforms"],
            univariate=zuko.transforms.MonotonicAffineTransform,
            hidden_features=(64, 128, 128),  
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
    #optimizer = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': learning_rate}, 'gamma': 0.9})
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_args["gamma"])

    mlflow.log_param("lr", train_args["lr"])
    mlflow.log_param("gamma", train_args["gamma"])

    pyro.set_rng_seed(0)
    verbose_shapes = False
    history = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    total_steps = train_args["steps"]
    plot_steps = [1000, int(0.25*total_steps), int(0.5*total_steps), int(0.75*total_steps)]
    num_eval_samples = 5000 # number of samples to evaluate the EIG
    num_steps_range = trange(0, train_args["steps"], desc="Loss: 0.000 ")
    for step in num_steps_range:
        optimizer.zero_grad() #  clear gradients from previous step
        agg_loss, loss = posterior_loss(design=designs,
                                        model=bed_cosmo.pyro_model,
                                        guide = posterior_flow,
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
                                                model=bed_cosmo.pyro_model,
                                                guide=posterior_flow,
                                                num_particles=train_args["eval_particles"],
                                                observation_labels=observation_labels,
                                                target_labels=target_labels,
                                                evaluation=True,
                                                nflow=True,
                                                condition_design=train_args["condition_design"])
            eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
            ax2.plot(designs.cpu().detach().numpy(), eigs_bits, label=f'step {step}')
        history.append(loss)
        if step % 25 == 0:
            mlflow.log_metric("loss", loss.mean().item(), step=step)
            mlflow.log_metric("agg_loss", agg_loss.item(), step=step)
            num_steps_range.set_description("Loss: {:.3f} ".format(loss.mean().item()))
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
                                        model=bed_cosmo.pyro_model,
                                        guide=posterior_flow,
                                        num_particles=train_args["eval_particles"],
                                        observation_labels=observation_labels,
                                        target_labels=target_labels,
                                        evaluation=True,
                                        nflow=True,
                                        condition_design=train_args["condition_design"])
    eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
    ax2.plot(designs.cpu().detach().numpy(), eigs_bits, label=f'NF step {train_args["steps"]}')
    if train_args["cosmology"] == 'Om':
        grid_eigs = np.load(f"grid_Om_eigs_{train_args['D_H_error']}.npy")
    elif train_args["cosmology"] == 'w':
        grid_eigs = np.load("grid_w_eigs.npy")
    elif train_args["cosmology"] == 'w0wa':
        grid_eigs = np.load("grid_w0wa_eigs.npy")
    ax2.plot(grid_eigs[:, 0], grid_eigs[:, 1], label="Brute Force", color='black')
    ax2.set_xlabel("Redshift of Observation")
    ax2.set_ylabel("EIG")
    ax2.legend()
    plt.tight_layout()
    ml_info = mlflow.active_run().info
    plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/training.png")

    # save eigs at the end
    design_eigs = np.stack((designs.squeeze().cpu().detach().numpy(), eigs_bits), axis=1)
    np.save(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/eigs.npy", design_eigs)
    mlflow.log_artifact(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/eigs.npy")

    #checkpoint = {"guide": posterior_flow,
    #            "transforms": transforms,
    #            "optimizer": optimizer.state_dict()}
    #torch.save(checkpoint, f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nf_checkpoint.pt")
    #mlflow.log_artifact(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nf_checkpoint.pt")
    print("Run", ml_info.experiment_id + "/" + ml_info.run_id, "completed.")



if __name__ == '__main__':
    train_args = {
        "seed": 1,
        "cosmology": "w0wa", # Om, w, w0wa
        "D_H_error": 0.2,
        "D_M_error": 0.2,
        "include_D_M": True,
        "flow_type": 'MAF_RQS',
        "design_grid": 50,
        "n_transforms": 10,
        "steps": 25000,
        "lr": 1e-4,
        "gamma": 0.8,
        "n_particles": 101,
        "eval_particles": 2000,
        "train_verbose": True, 
        "condition_design": True,
        "device": device
        }   
    single_run(
        train_args,
        "w0wa_Cosmology",
        shape=16,
        hidden=250
        )
    mlflow.end_run()