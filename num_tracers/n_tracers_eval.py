
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
from mlflow.tracking import MlflowClient
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

client = MlflowClient()

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}.')

#set default dtype
torch.set_default_dtype(torch.float64)

def marginalize_posterior(num_tracers, posterior_flow, ratios, num_param_samples=1000, num_data_samples=100):

    marginal_samples = []
    data_samples = num_tracers.pyro_model(lexpand(ratios.unsqueeze(0), num_data_samples))
    for i in range(len(data_samples)):
        # Sample a single data point
        data_i = data_samples[i, :, :]
        # Combine data with designs for conditioning the posterior
        context = torch.cat([ratios, data_i.squeeze()], dim=-1)
        # Sample parameters conditioned on the data batch
        param_samples = posterior_flow(context).sample((num_param_samples,))  # Shape: (samples_per_data, param_dim)
        # Append to the list
        marginal_samples.append(param_samples)

    # Concatenate all samples
    marginal_samples = torch.cat(marginal_samples, dim=0)  # Shape: (num_data_samples * samples_per_data, param_dim)

    return marginal_samples

def single_run(run_id, **kwargs):

    run = client.get_run(run_id)
    exp_id = run.info.experiment_id
    mlflow.set_experiment(client.get_experiment(run.info.experiment_id).name)
    tracers = eval(run.data.params["tracers"])
    desi_df = pd.read_csv('/home/ashandonay/bed/desi_data.csv')
    desi_cov = np.load('/home/ashandonay/bed/desi_cov.npy')
    # select only the rows corresponding to the tracers
    desi_data = desi_df[desi_df['tracer'].isin(tracers)]
    nominal_cov = desi_cov[np.ix_(desi_data.index, desi_data.index)]
    with mlflow.start_run(run_id=run_id, nested=True):
        ml_info = mlflow.active_run().info
        print("MLFlow Run Info:", ml_info.experiment_id + "/" + ml_info.run_id)


        w0_mean = torch.tensor(-1.0).to(run.data.params["device"])
        w0_std = torch.tensor(0.2).to(run.data.params["device"])
        wa_mean = torch.tensor(0.0).to(run.data.params["device"])
        wa_std = torch.tensor(0.2).to(run.data.params["device"])

        Om_range = torch.tensor([0.0, 1.0], device=run.data.params["device"])
        w0_range = torch.tensor([-3.0, 1.0], device=run.data.params["device"])
        wa_range = torch.tensor([-3.0, 2.0], device=run.data.params["device"])

        ############################################### Priors ###############################################
        if run.data.params["cosmology"] == 'Om':
            params = {"Om": Om_range}
            grid_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), eval(run.data.params["params_grid"]))[:-1])
            priors = {'Om': dist.Uniform(*Om_range)}
            grid_prior = TopHat(grid_params.Om)
            #priors = {'Om': dist.Normal(Om_mean, Om_std)}
            #grid_prior = Gaussian(grid_params.Om, Planck18.Om0, run.data.params["Om_sigma"])
        elif run.data.params["cosmology"] == 'w':
            params = {"Om": Om_range, "w0": w0_range}
            priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range)}
            #priors = {'Om': dist.Normal(Om_mean, Om_std), 'w0': dist.Normal(w0_mean, w0_std)}
            grid_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), eval(run.data.params["params_grid"])), 
                w0=np.linspace(-3, 1, eval(run.data.params["params_grid"])))
            grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.w0)
            #grid_prior = Gaussian(grid_params.Om, Planck18.Om0, run.data.params["Om_sigma"]) * TopHat(grid_params.w0)
        elif run.data.params["cosmology"] == 'w0wa':
            params = {"Om": Om_range, "w0": w0_range, "wa": wa_range}
            priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range)}
            #priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range)}
            grid_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), eval(run.data.params["params_grid"])), 
                w0=np.linspace(-3, 1, eval(run.data.params["params_grid"])), 
                wa=np.linspace(-3, 2, eval(run.data.params["params_grid"])))
            grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.w0) * TopHat(grid_params.wa)
            #grid_prior = Gaussian(grid_params.Om, Planck18.Om0, run.data.params["Om_sigma"]) * TopHat(grid_params.w0) * TopHat(grid_params.wa)
        grid_params.normalize(grid_prior);

        observation_labels = ["y"]
        target_labels = list(priors.keys())
        print(f'Cosmology: {run.data.params["cosmology"]}')
        print(f'Tracers: {run.data.params["tracers"]}')
        print(f'Observation Labels: {observation_labels}')
        print(f'Target Labels: {target_labels}')

        num_tracers = NumTracers(
            desi_data, 
            nominal_cov,
            priors, 
            observation_labels, 
            eff=eval(run.data.params["eff"]), 
            include_D_M=eval(run.data.params["include_D_M"]), 
            device=run.data.params["device"]
            )

        ############################################### Designs ###############################################

        designs_dict = {}
        for i in range(len(tracers)):
            designs_dict['N_' + tracers[i].split()[0]] = np.arange(
                eval(run.data.params["design_low"]), 
                eval(run.data.params["design_high"]) + eval(run.data.params["design_step"]), 
                eval(run.data.params["design_step"])
                )

        grid_designs = Grid(**designs_dict, constraint=lambda **kwargs: sum(kwargs.values()) == 1.0)
        del designs_dict
        print("Grid designs shape:", grid_designs.shape)
        
        designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=device).unsqueeze(1)
        for n in grid_designs.names[1:]:
            designs = torch.cat((designs, torch.tensor(getattr(grid_designs, n).squeeze(), device=device).unsqueeze(1)), dim=1)

        bins = [50]*len(target_labels)
        levels = (0.68, 0.95)
        hist_range = [0.997]*len(target_labels)
        if eval(run.data.params["brute_force"]):
            ############################################### Features ###############################################
            if run.data.params["cosmology"] == 'Om':
                range_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), 1000))
            elif run.data.params["cosmology"] == 'w':
                range_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), 500), 
                    w0=np.linspace(-3, 1, 500))
            elif run.data.params["cosmology"] == 'w0wa':
                range_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), 200), 
                    w0=np.linspace(-3, 1, 200), 
                    wa=np.linspace(-3, 2, 200))
            parameters = { }
            for key in range_params.names:
                parameters[key] = torch.tensor(getattr(range_params, key), device=run.data.params["device"])
            if eval(run.data.params["include_D_M"]):
                features_dict = {}
                for i in range(len(tracers)):
                    D_H_mean = num_tracers.D_H_func(z_eff[i], **parameters)
                    print('D_H_' + tracers[i].split()[0] + ' range:', D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2)
                    features_dict['D_H_' + tracers[i].split()[0]] = np.linspace(D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2, eval(run.data.params["features_grid"]))
                for i in range(len(run.data.params["tracers"])):
                    z_array = z_eff[i].unsqueeze(-1) * torch.linspace(0, 1, 100, device=run.data.params["device"]).view(1, -1)
                    z = z_array.expand((len(range_params.names)-1)*[1] + [-1, -1])
                    D_M_mean = num_tracers.D_M_func(z, **parameters)
                    print('D_M_' + tracers[i].split()[0] + ' range:', D_M_mean.cpu().numpy().min(), D_M_mean.cpu().numpy().max() + 2)
                    features_dict['D_M_' + tracers[i].split()[0]] = np.linspace(D_M_mean.cpu().numpy().min(), D_M_mean.cpu().numpy().max() + 2, eval(run.data.params["features_grid"]))
            else:
                features_dict = {}
                for i in range(len(tracers)):
                    D_H_mean = num_tracers.D_H_func(z_eff[i], **parameters)
                    print('D_H_' + tracers[i].split()[0] + ' range:', D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2)
                    features_dict['D_H_' + tracers[i].split()[0]] = np.linspace(D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2, eval(run.data.params["features_grid"]))
            grid_features = Grid(**features_dict)
            del range_params, parameters, features_dict

            print("Calculating brute force EIG...")
            designer = ExperimentDesigner(grid_params, grid_features, grid_designs, num_tracers.unnorm_lfunc, mem=190000)
            designer.calculateEIG(grid_prior)
            brute_force_EIG = designer.EIG

            posterior = designer.get_posterior()
            print("posterior shape:", posterior.shape)
            # sum over all features
            marginalized_post = np.sum(posterior, axis=(0, 1))
            # normalize
            marginalized_post /= np.sum(marginalized_post)
            plt.figure()
            plt.plot(grid_params.Om, marginalized_post[brute_force_EIG.argmax()].squeeze())
            plt.xlabel("Om")
            plt.ylabel("p(Om)")
            plt.savefig(f"mlruns/{exp_id}/{run_id}/brute_force_posterior.png")

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

            pyro.set_rng_seed(0)
            verbose_shapes = run.data.params["train_verbose"]
            history = []

        if eval(run.data.params["nf_model"]):

            input_dim = len(target_labels)
            context_dim = 3 * len(tracers) if eval(run.data.params["include_D_M"]) else 2 * len(tracers)
            hidden = int(run.data.params["hidden"])
            print(f'Input Dim: {input_dim}\n'
                f'Context Dim: {context_dim}\n')

            posterior_flow = zuko.flows.NAF(
                        features=input_dim, 
                        context=context_dim, 
                        transforms=int(run.data.params["n_transforms"]),
                        signal=int(run.data.params["signal"]),
                        hidden_features=(hidden, 2*hidden, 2*hidden),
                    ).to("cuda:0")

            checkpoint = torch.load(f'mlruns/{exp_id}/{run_id}/artifacts/nf_checkpoint.pt', map_location="cuda:0")
            posterior_flow.load_state_dict(checkpoint['model_state_dict'], strict=True)
            posterior_flow.to("cuda:0")
            posterior_flow.eval()

            plt.figure()
            with torch.no_grad():
                agg_loss, eigs = posterior_loss(design=designs,
                                                model=num_tracers.pyro_model,
                                                guide=posterior_flow,
                                                num_particles=kwargs["eval_particles"],
                                                observation_labels=observation_labels,
                                                target_labels=target_labels,
                                                evaluation=True,
                                                nflow=True,
                                                condition_design=eval(run.data.params["condition_design"]))
            eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
            plt.plot(eigs_bits, label=f'NF step {run.data.params["steps"]}')

            with torch.no_grad():
                agg_loss, nominal_eig = posterior_loss(design=num_tracers.nominal_n_ratios.unsqueeze(0),
                                                model=num_tracers.pyro_model,
                                                guide=posterior_flow,
                                                num_particles=kwargs["eval_particles"],
                                                observation_labels=observation_labels,
                                                target_labels=target_labels,
                                                evaluation=True,
                                                nflow=True,
                                                condition_design=eval(run.data.params["condition_design"]))
            nominal_eig_bits = nominal_eig.cpu().detach().numpy()/np.log(2)
            plt.axhline(y=nominal_eig_bits, color='black', linestyle='--', label='Nominal EIG')
            if eval(run.data.params["brute_force"]):
                plt.plot(brute_force_EIG.squeeze(), label='Brute Force', color='black')
            plt.xlabel("Design Index")
            plt.ylabel("EIG")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/evaluation.png")

            # save eigs at the end
            np.save(f"mlruns/{exp_id}/{run_id}/artifacts/eigs.npy", eigs_bits)
            np.save(f"mlruns/{exp_id}/{run_id}/artifacts/nominal_eig.npy", nominal_eig_bits)
            mlflow.log_artifact(f"mlruns/{exp_id}/{run_id}/artifacts/eigs.npy")

            optimal_n_ratios = designs[torch.argmax(eigs)]
            param_samples = 100
            data_samples = 100
            nominal_samples = marginalize_posterior(
                num_tracers,
                posterior_flow, 
                num_tracers.nominal_n_ratios,
                num_param_samples=param_samples,
                num_data_samples=data_samples
                ).cpu().detach().numpy()
            optimal_samples = marginalize_posterior(
                num_tracers,
                posterior_flow, 
                optimal_n_ratios,
                num_param_samples=param_samples,
                num_data_samples=data_samples
            ).cpu().detach().numpy()

            fig = corner.corner(nominal_samples, labels=target_labels, show_titles=True, title_fmt=".2f", 
                                title_kwargs={"fontsize": 12}, levels=levels, 
                                bins=bins, plot_datapoints=False, plot_density=False, smooth=1.0, 
                                range=hist_range, label='Nominal')
            plt.savefig(f"mlruns/{exp_id}/{run_id}/nominal_posterior.png")

            figure = corner.corner(optimal_samples, labels=target_labels, show_titles=True, title_fmt=".2f", 
                                title_kwargs={"fontsize": 12}, levels=levels,
                                bins=bins, plot_datapoints=False, plot_density=False, smooth=1.0, 
                                range=hist_range, label='Nominal')
            plt.savefig(f"mlruns/{exp_id}/{run_id}/optimal_posterior.png")

            plt.close('all')



if __name__ == '__main__':

    process = psutil.Process(os.getpid())
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Trigger garbage collection for CPU
    gc.collect()
    print(f"Memory before run: {process.memory_info().rss / 1024**2} MB")
    single_run(
        '371080c86b804889b0bb8cd13c81770b',
        eval_particles=2000,
        )
    mlflow.end_run()
            