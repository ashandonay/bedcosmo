
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

def run_eval(eval_args, run_id, exp, **kwargs):

    if exp is not None:
        exp_id = client.get_experiment_by_name(exp).experiment_id
        run_ids = [run.info.run_id for run in client.search_runs(exp_id)]
    else:
        run = client.get_run(run_id)
        exp_id = run.info.experiment_id
        exp = client.get_experiment(run.info.experiment_id).name
        run_ids = [run_id]
    print(f"Evaluating runs from experiment: {exp}")
    mlflow.set_experiment(exp)
    for run_id in run_ids:
        with mlflow.start_run(run_id=run_id, nested=True):
            run = client.get_run(run_id)
            ml_info = mlflow.active_run().info
            print("MLFlow Run Info:", ml_info.experiment_id + "/" + ml_info.run_id)

            tracers = eval(run.data.params["tracers"])
            desi_df = pd.read_csv('/home/ashandonay/bed/desi_data.csv')
            desi_cov = np.load('/home/ashandonay/bed/desi_cov.npy')
            # select only the rows corresponding to the tracers
            desi_data = desi_df[desi_df['tracer'].isin(tracers)]
            nominal_cov = desi_cov[np.ix_(desi_data.index, desi_data.index)]

            w0_mean = torch.tensor(-1.0).to(run.data.params["device"])
            w0_std = torch.tensor(0.2).to(run.data.params["device"])
            wa_mean = torch.tensor(0.0).to(run.data.params["device"])
            wa_std = torch.tensor(0.2).to(run.data.params["device"])

            Om_range = torch.tensor([0.01, 0.99], device=run.data.params["device"])
            w0_range = torch.tensor([-3.0, 1.0], device=run.data.params["device"])
            wa_range = torch.tensor([-3.0, 2.0], device=run.data.params["device"])

            ############################################### Priors ###############################################
            if run.data.params["cosmology"] == 'Om':
                priors = {'Om': dist.Uniform(*Om_range)}
                if eval_args["brute_force"]:
                    grid_params = Grid(
                        Om=np.linspace(*Om_range.cpu().numpy(), eval_args["params_grid"])[:-1]
                        )
                    grid_prior = TopHat(grid_params.Om)
                #grid_prior = Gaussian(grid_params.Om, Planck18.Om0, run.data.params["Om_sigma"])
            elif run.data.params["cosmology"] == 'w':
                priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range)}
                if eval_args["brute_force"]:
                    grid_params = Grid(
                        Om=np.linspace(*Om_range.cpu().numpy(), eval_args["params_grid"]), 
                        w0=np.linspace(-3, 1, eval_args["params_grid"])
                        )
                    grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.w0)
                #grid_prior = Gaussian(grid_params.Om, Planck18.Om0, run.data.params["Om_sigma"]) * TopHat(grid_params.w0)
            elif run.data.params["cosmology"] == 'w0wa':
                priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range)}
                if eval_args["brute_force"]:
                    grid_params = Grid(
                        Om=np.linspace(*Om_range.cpu().numpy(), eval_args["params_grid"]), 
                        w0=np.linspace(-3, 1, eval_args["params_grid"]), 
                        wa=np.linspace(-3, 2, eval_args["params_grid"])
                        )
                    grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.w0) * TopHat(grid_params.wa)
                #grid_prior = Gaussian(grid_params.Om, Planck18.Om0, run.data.params["Om_sigma"]) * TopHat(grid_params.w0) * TopHat(grid_params.wa)

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

            levels = (0.68, 0.95)
            hist_range = [0.997]*len(target_labels)

            if eval_args["brute_force"]:
                grid_params.normalize(grid_prior);
                print("Calculating brute force EIG...")
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
                        D_H_mean = num_tracers.D_H_func(num_tracers.z_eff[i], **parameters)
                        print('D_H_' + tracers[i].split()[0] + ' range:', D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2)
                        features_dict['D_H_' + tracers[i].split()[0]] = np.linspace(D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2, eval_args["features_grid"])
                    for i in range(len(run.data.params["tracers"])):
                        z_array = num_tracers.z_eff[i].unsqueeze(-1) * torch.linspace(0, 1, 100, device=run.data.params["device"]).view(1, -1)
                        z = z_array.expand((len(range_params.names)-1)*[1] + [-1, -1])
                        D_M_mean = num_tracers.D_M_func(z, **parameters)
                        print('D_M_' + tracers[i].split()[0] + ' range:', D_M_mean.cpu().numpy().min(), D_M_mean.cpu().numpy().max() + 2)
                        features_dict['D_M_' + tracers[i].split()[0]] = np.linspace(D_M_mean.cpu().numpy().min(), D_M_mean.cpu().numpy().max() + 2, eval_args["features_grid"])
                else:
                    features_dict = {}
                    for i in range(len(tracers)):
                        D_H_mean = num_tracers.D_H_func(num_tracers.z_eff[i], **parameters)
                        print('D_H_' + tracers[i].split()[0] + ' range:', D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2)
                        features_dict['D_H_' + tracers[i].split()[0]] = np.linspace(D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2, eval_args["features_grid"])
                grid_features = Grid(**features_dict)
                del range_params, parameters, features_dict

                print("Calculating brute force EIG...")
                designer = ExperimentDesigner(grid_params, grid_features, grid_designs, num_tracers.unnorm_lfunc, mem=190000)
                optimal_brute_force = designer.calculateEIG(grid_prior)
                optimal_brute_force = torch.tensor(list(optimal_brute_force.values()), device=device)
                brute_force_EIG = designer.EIG
                print("Optimal brute force design:", optimal_brute_force)
                np.save(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/brute_force_designs.npy", np.array([getattr(grid_designs, name) for name in grid_designs.names]).squeeze())
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

            if eval_args["nf_model"]:
                input_dim = len(target_labels)
                context_dim = 3 * len(tracers) if eval(run.data.params["include_D_M"]) else 2 * len(tracers)
                hidden = int(run.data.params["hidden"])

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

                nominal_n_ratios = num_tracers.obs_n_ratios/num_tracers.efficiency[::2]
                print("Nominal ratios:", nominal_n_ratios)
                plt.figure()
                colors = sns.color_palette("Set1", eval_args["num_evals"])
                for n in range(eval_args["num_evals"]):
                    with torch.no_grad():
                        agg_loss, eigs = posterior_loss(design=designs,
                                                        model=num_tracers.pyro_model,
                                                        guide=posterior_flow,
                                                        num_particles=eval_args["eval_particles"],
                                                        observation_labels=observation_labels,
                                                        target_labels=target_labels,
                                                        evaluation=True,
                                                        nflow=True,
                                                        condition_design=eval(run.data.params["condition_design"]))
                    eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
                    plt.plot(eigs_bits, label=f'NF step {run.data.params["steps"]}', color=colors[n])

                    with torch.no_grad():
                        agg_loss, nominal_eig = posterior_loss(design=nominal_n_ratios.unsqueeze(0),
                                                        model=num_tracers.pyro_model,
                                                        guide=posterior_flow,
                                                        num_particles=eval_args["eval_particles"],
                                                        observation_labels=observation_labels,
                                                        target_labels=target_labels,
                                                        evaluation=True,
                                                        nflow=True,
                                                        condition_design=eval(run.data.params["condition_design"]))
                    nominal_eig_bits = nominal_eig.cpu().detach().numpy()/np.log(2)
                    plt.axhline(y=nominal_eig_bits, linestyle='--', label='Nominal EIG', color=colors[n])
                if eval_args["brute_force"]:
                    plt.plot(brute_force_EIG.squeeze(), label='Brute Force', color='black')
                plt.xlabel("Design Index")
                plt.ylabel("EIG")
                # Get the existing handles and labels
                handles, labels = plt.gca().get_legend_handles_labels()
                # Choose which items you want to include (e.g., the first and third)
                handles_to_show = [handles[0], handles[1]]
                labels_to_show = [labels[0], labels[1]]
                # Create a legend with only the selected items
                plt.legend(handles_to_show, labels_to_show)
                plt.title(f"{eval_args['num_evals']} EIG eval(s) with {eval_args['eval_particles']} samples")  
                plt.tight_layout()
                plt.savefig(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/evaluation.png")

                # save eigs at the end
                np.save(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/eigs.npy", eigs_bits)
                np.save(f"mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nominal_eig.npy", nominal_eig_bits)
                mlflow.log_artifact(f"mlruns/{exp_id}/{run_id}/artifacts/eigs.npy")

                optimal_n_ratios = designs[torch.argmax(eigs)]
                central_vals = num_tracers.central_val if eval(run.data.params["include_D_M"]) else num_tracers.central_val[1::2]
                optimal_context = torch.cat([optimal_n_ratios, central_vals], dim=-1)
                optimal_samples = posterior_flow(optimal_context).sample((eval_args["post_samples"],)).cpu().numpy()

                nominal_context = torch.cat([nominal_n_ratios, central_vals], dim=-1)
                nominal_samples = posterior_flow(nominal_context).sample((eval_args["post_samples"],)).cpu().numpy()

                #nominal_samples = num_tracers.sample_flow(nominal_n_ratios, posterior_flow).cpu().numpy()
                #optimal_samples = num_tracers.sample_flow(optimal_n_ratios, posterior_flow).cpu().numpy()
                if len(target_labels) == 1:
                    plt.figure()
                    # get min and max of the samples
                    min_val = min([nominal_samples.min(), optimal_samples.min()])
                    max_val = max([nominal_samples.max(), optimal_samples.max()])
                    if eval_args["brute_force"]:
                        min_val = min([min_val, brute_force_samples.min()])
                        max_val = max([max_val, brute_force_samples.max()])
                    bins = np.linspace(min_val, max_val, 50)
                    plt.hist(nominal_samples, bins=bins, color='tab:blue', alpha=0.5, label='Nominal', density=True)
                    plt.hist(optimal_samples, bins=bins, color='tab:orange', alpha=0.5, label='Optimal', density=True)
                    if eval_args["brute_force"]:
                        brute_force_nominal_samples = num_tracers.brute_force_posterior(nominal_n_ratios, designer, grid_params, num_param_samples=eval_args["post_samples"]).cpu().numpy()
                        brute_force_optimal_samples = num_tracers.brute_force_posterior(optimal_n_ratios, designer, grid_params, num_param_samples=eval_args["post_samples"]).cpu().numpy()
                        plt.hist(brute_force_nominal_samples, bins=bins, color='tab:green', alpha=0.5, label='Brute Force', density=True)
                    plt.xlabel(target_labels[0])
                    plt.ylabel("Frequency")
                    plt.legend()
                else:
                    legend_labels = []
                    bins = [30]*len(target_labels)
                    post_fig = corner.corner(nominal_samples, labels=target_labels, show_titles=False, title_fmt=".2f", 
                                        title_kwargs={"fontsize": 12}, levels=levels, 
                                        bins=bins, plot_datapoints=False, plot_density=False, smooth=1.0, 
                                        fill_contours=False, color='tab:blue', range=hist_range)
                    legend_labels.append(plt.Line2D([0], [0], color="tab:blue", lw=1, label="NF (Nominal)"))
                    corner.corner(optimal_samples, labels=target_labels, show_titles=False, title_fmt=".2f", 
                                    title_kwargs={"fontsize": 12}, levels=levels, fig=post_fig,
                                    bins=bins, plot_datapoints=False, plot_density=False, smooth=1.0, 
                                    fill_contours=False, color='tab:orange', range=hist_range)
                    legend_labels.append(plt.Line2D([0], [0], color="tab:orange", lw=1, label="NF (Optimal)"))
                    if eval_args["brute_force"]:
                        brute_force_nominal_samples = num_tracers.brute_force_posterior(nominal_n_ratios, designer, grid_params, num_param_samples=eval_args["post_samples"]).cpu().numpy()
                        brute_force_optimal_samples = num_tracers.brute_force_posterior(optimal_n_ratios, designer, grid_params, num_param_samples=eval_args["post_samples"]).cpu().numpy()
                        #brute_force_samples = num_tracers.sample_brute_force(optimal_brute_force, grid_designs, grid_features, grid_params, designer).cpu().numpy()
                        corner.corner(brute_force_nominal_samples, labels=target_labels, show_titles=False, title_fmt=".2f", 
                                        title_kwargs={"fontsize": 12}, levels=levels, fig=post_fig,
                                        bins=bins, plot_datapoints=False, plot_density=False, smooth=1.0, 
                                        fill_contours=False, color='tab:blue', range=hist_range, hist_kwargs={"linestyle": 'dashed'}, contour_kwargs={"linestyles": 'dashed'})
                        legend_labels.append(plt.Line2D([0], [0], color="tab:blue", lw=1, label="Brute Force (Nominal)", linestyle='dashed'))
                        corner.corner(brute_force_optimal_samples, labels=target_labels, show_titles=False, title_fmt=".2f", 
                                        title_kwargs={"fontsize": 12}, levels=levels, fig=post_fig,
                                        bins=bins, plot_datapoints=False, plot_density=False, smooth=1.0, 
                                        fill_contours=False, color='tab:orange', range=hist_range, hist_kwargs={"linestyle": 'dashed'}, contour_kwargs={"linestyles": 'dashed'})
                        legend_labels.append(plt.Line2D([0], [0], color="tab:orange", lw=1, label="Brute Force (Optimal)", linestyle='dashed'))

                plt.legend(handles=legend_labels)
                plt.savefig(f"mlruns/{exp_id}/{run_id}/posterior.png")

        plt.close('all')

if __name__ == '__main__':

    process = psutil.Process(os.getpid())
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Trigger garbage collection for CPU
    gc.collect()
    print(f"Memory before run: {process.memory_info().rss / 1024**2} MB")
    eval_args = {
        "nf_model": True,
        "brute_force": False,
        "post_samples": 50000,
        "eval_particles": 4000,
        "num_evals": 3,
        "params_grid": 100,
        "features_grid": 100,
        "mem": 200000,
    }
    run_eval(
        eval_args,
        run_id='c57a5d01fd5e49189b95a09d6e215d16',
        exp=None,
        )
    mlflow.end_run()
            