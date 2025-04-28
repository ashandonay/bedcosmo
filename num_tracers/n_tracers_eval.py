
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
import json

import pyro
from pyro import poutine
from pyro import distributions as dist
from pyro_oed_src import posterior_loss
from pyro.contrib.util import lexpand, rexpand

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import corner
import getdist
from getdist import plots

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
from plotting import plot_posterior
from util import *


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

def run_eval(eval_args, run_id, exp, device, **kwargs):
    storage_path = os.environ["SCRATCH"] + "/bed/BED_cosmo/num_tracers"
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    client = MlflowClient()
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

            run_args = parse_mlflow_params(run.data.params)
            home_dir = os.environ["HOME"]
            desi_df = pd.read_csv(home_dir + run_args["data_path"] + 'desi_data.csv')
            desi_tracers = pd.read_csv(home_dir + run_args["data_path"] + 'desi_tracers.csv')
            nominal_cov = np.load(home_dir + run_args["data_path"] + 'desi_cov.npy')
            cosmo_model = run_args["cosmo_model"]

            Om_range = torch.tensor([0.01, 0.99], device=device)
            Ok_range = torch.tensor([-0.3, 0.3], device=device)
            w0_range = torch.tensor([-3.0, 1.0], device=device)
            wa_range = torch.tensor([-3.0, 2.0], device=device)
            hrdrag_range = torch.tensor([0.01, 1.0], device=device)

            ############################################### Priors ###############################################
            if cosmo_model == 'base':
                latex_labels = ['\Omega_m', 'H_0r_d']
                priors = {'Om': dist.Uniform(*Om_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
                if eval_args["brute_force"]:
                    grid_params = Grid(
                        Om=np.linspace(*Om_range.cpu().numpy(), eval_args["params_grid"]), 
                        hrdrag=np.linspace(*hrdrag_range.cpu().numpy(), eval_args["params_grid"])
                        )
                    grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.hrdrag)
            elif cosmo_model == 'base_omegak':
                latex_labels = ['\Omega_m', '\Omega_k', 'H_0r_d']
                priors = {'Om': dist.Uniform(*Om_range), 'Ok': dist.Uniform(*Ok_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
                if eval_args["brute_force"]:
                    grid_params = Grid(
                        Om=np.linspace(*Om_range.cpu().numpy(), eval_args["params_grid"]), 
                        Ok=np.linspace(*Ok_range.cpu().numpy(), eval_args["params_grid"]), 
                        hrdrag=np.linspace(*hrdrag_range.cpu().numpy(), eval_args["params_grid"])
                        )
                    grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.Ok) * TopHat(grid_params.hrdrag)
            elif cosmo_model == 'base_w':
                latex_labels = ['\Omega_m', 'w_0', 'H_0r_d']
                priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
                if eval_args["brute_force"]:
                    grid_params = Grid(
                        Om=np.linspace(*Om_range.cpu().numpy(), eval_args["params_grid"]), 
                        w0=np.linspace(*w0_range.cpu().numpy(), eval_args["params_grid"]), 
                        hrdrag=np.linspace(*hrdrag_range.cpu().numpy(), eval_args["params_grid"])
                        )
                    grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.w0) * TopHat(grid_params.hrdrag)
            elif cosmo_model == 'base_w_wa':
                latex_labels = ['\Omega_m', 'w_0', 'w_a', 'H_0r_d']
                priors = {'Om': dist.Uniform(*Om_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
                if eval_args["brute_force"]:
                    grid_params = Grid(
                        Om=np.linspace(*Om_range.cpu().numpy(), eval_args["params_grid"]), 
                        w0=np.linspace(*w0_range.cpu().numpy(), eval_args["params_grid"]), 
                        wa=np.linspace(*wa_range.cpu().numpy(), eval_args["params_grid"]),
                        hrdrag=np.linspace(*hrdrag_range.cpu().numpy(), eval_args["params_grid"])
                        )
                    grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.w0) * TopHat(grid_params.wa) * TopHat(grid_params.hrdrag)
            elif cosmo_model == 'base_omegak_w_wa':
                latex_labels = ['\Omega_m', '\Omega_k', 'w_0', 'w_a', 'H_0r_d']
                priors = {'Om': dist.Uniform(*Om_range), 'Ok': dist.Uniform(*Ok_range), 'w0': dist.Uniform(*w0_range), 'wa': dist.Uniform(*wa_range), 'hrdrag': dist.Uniform(*hrdrag_range)}
                if eval_args["brute_force"]:
                    grid_params = Grid(
                        Om=np.linspace(*Om_range.cpu().numpy(), eval_args["params_grid"]), 
                        Ok=np.linspace(*Ok_range.cpu().numpy(), eval_args["params_grid"]), 
                        w0=np.linspace(*w0_range.cpu().numpy(), eval_args["params_grid"]), 
                        wa=np.linspace(*wa_range.cpu().numpy(), eval_args["params_grid"]), 
                        hrdrag=np.linspace(*hrdrag_range.cpu().numpy(), eval_args["params_grid"])
                        )
                    grid_prior = TopHat(grid_params.Om) * TopHat(grid_params.Ok) * TopHat(grid_params.w0) * TopHat(grid_params.wa) * TopHat(grid_params.hrdrag)

            with open(mlflow.artifacts.download_artifacts(run_id=ml_info.run_id, artifact_path="classes.json")) as f:
                classes = json.load(f)

            target_labels = list(priors.keys())
            print(f"Classes: {classes}")
            print(f'Cosmology: {cosmo_model}')
            print(f'Target Labels: {target_labels}')

            num_tracers = NumTracers(
                desi_df, 
                desi_tracers,
                cosmo_model,
                nominal_cov,
                device=device, 
                include_D_M=run_args["include_D_M"],
                verbose=True
                )

            ############################################### Designs ###############################################

            designs_dict = {}
            for c, u in classes.items():
                designs_dict['N_' + c] = np.arange(
                    run_args["design_low"], 
                    u + run_args["design_step"], 
                    run_args["design_step"]
                    )
            tol = 1e-3
            grid_designs = Grid(**designs_dict, constraint=lambda **kwargs: abs(sum(kwargs.values()) - 1.0) < tol)
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
                if cosmo_model == 'Om':
                    range_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), 1000))
                elif cosmo_model == 'w':
                    range_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), 500), 
                        w0=np.linspace(-3, 1, 500))
                elif cosmo_model == 'w0wa':
                    range_params = Grid(Om=np.linspace(*Om_range.cpu().numpy(), 200), 
                        w0=np.linspace(-3, 1, 200), 
                        wa=np.linspace(-3, 2, 200))
                parameters = { }
                for key in range_params.names:
                    parameters[key] = torch.tensor(getattr(range_params, key), device=device)
                tracers = desi_df.tracer[::2].to_list()
                if run_args["include_D_M"]:
                    features_dict = {}
                    for i in range(len(tracers)):
                        D_H_mean = num_tracers.D_H_func(num_tracers.z_eff[i], **parameters)
                        print('D_H_' + tracers[i].split()[0] + ' range:', D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2)
                        features_dict['D_H_' + tracers[i].split()[0]] = np.linspace(D_H_mean.cpu().numpy().min(), D_H_mean.cpu().numpy().max() + 2, eval_args["features_grid"])
                    for i in range(len(tracers)):
                        z_array = num_tracers.z_eff[i].unsqueeze(-1) * torch.linspace(0, 1, 100, device=device).view(1, -1)
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

                designer = ExperimentDesigner(grid_params, grid_features, grid_designs, num_tracers.unnorm_lfunc, mem=190000)
                optimal_brute_force = designer.calculateEIG(grid_prior)
                optimal_brute_force = torch.tensor(list(optimal_brute_force.values()), device=device)
                brute_force_EIG = designer.EIG
                print("Optimal brute force design:", optimal_brute_force)
                np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/brute_force_designs.npy", np.array([getattr(grid_designs, name) for name in grid_designs.names]).squeeze())
                np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/brute_force_eigs.npy", brute_force_EIG)
                mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/brute_force_designs.npy")
                mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/brute_force_eigs.npy")

                plt.figure()
                if len(grid_designs.expand(designer.EIG).shape) > 2:
                    for i in range(grid_designs.expand(designer.EIG).shape[-1]):
                        plt.imshow(grid_designs.expand(designer.EIG)[:,:,i].T, origin='lower',
                        extent=grid_designs.extent(grid_designs.names[0])+grid_designs.extent(grid_designs.names[1]))
                    plt.xlabel(grid_designs.names[0])
                    plt.ylabel(grid_designs.names[1])
                    plt.colorbar(label="EIG [bits]")
                    plt.tight_layout()
                    plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/brute_force.png")
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
                    plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/brute_force.png")

            if eval_args["nf_model"]:
                input_dim = len(target_labels)
                context_dim = len(classes.keys()) + 10 if run_args["include_D_M"] else len(classes.keys()) + 5

                posterior_flow = init_nf(
                    run_args["flow_type"],
                    input_dim, 
                    context_dim, 
                    run_args,
                    device,
                    seed=run_args["nf_seed"],
                    verbose=True
                    )
                print(f"Loading NF model from {run_id}...")
                checkpoint = torch.load(f'{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/nf_checkpoint_best_loss.pt', map_location=device)
                posterior_flow.load_state_dict(checkpoint['model_state_dict'], strict=True)
                posterior_flow.to(device)
                posterior_flow.eval()
                
                # fix the seed for reproducibility
                seed = auto_seed(run_args["pyro_seed"])

                # get the nominal design by the observed amount of tracers per class assuming default numbers from desi_df
                nominal_design = torch.tensor(desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=device)
                print("Nominal design:", nominal_design)
                np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nominal_design.npy", nominal_design.cpu().numpy())
                mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nominal_design.npy")

                plt.figure()
                eigs_batch = []
                nominal_eig_batch = []
                for n in range(eval_args["num_evals"]):
                    with torch.no_grad():
                        agg_loss, eigs = posterior_loss(design=designs,
                                                        model=num_tracers.pyro_model,
                                                        guide=posterior_flow,
                                                        num_particles=eval_args["eval_particles"],
                                                        observation_labels=["y"],
                                                        target_labels=target_labels,
                                                        evaluation=True,
                                                        nflow=True,
                                                        analytic_prior=False,
                                                        condition_design=run_args["condition_design"])
                    eigs_bits = eigs.cpu().detach().numpy()/np.log(2)
                    eigs_batch.append(eigs_bits)
                    plt.plot(eigs_bits, label=f'NF step {run_args["steps"]}', color="tab:blue", alpha=0.4)

                    with torch.no_grad():
                        agg_loss, nominal_eig = posterior_loss(design=nominal_design.unsqueeze(0),
                                                        model=num_tracers.pyro_model,
                                                        guide=posterior_flow,
                                                        num_particles=eval_args["eval_particles"],
                                                        observation_labels=["y"],
                                                        target_labels=target_labels,
                                                        evaluation=True,
                                                        nflow=True,
                                                        analytic_prior=False,
                                                        condition_design=run_args["condition_design"])
                    nominal_eig_bits = nominal_eig.cpu().detach().numpy()/np.log(2)
                    nominal_eig_batch.append(nominal_eig_bits)
                    plt.axhline(y=nominal_eig_bits, linestyle='--', label='Nominal EIG', color="black", alpha=0.4)
                if eval_args["brute_force"]:
                    plt.plot(brute_force_EIG.squeeze(), label='Brute Force', color='black')
                plt.xlabel("Design Index")
                plt.ylabel("EIG")
                eigs_batch = np.array(eigs_batch)
                nominal_eig_batch = np.array(nominal_eig_batch)
                # avg over the number of evaluations
                eig_avg = torch.tensor(np.mean(eigs_batch, axis=0), device=device)
                eig_std = torch.tensor(np.std(eigs_batch, axis=0), device=device)
                eig_se = eig_std/np.sqrt(eval_args["num_evals"])
                avg_nominal_eig = np.mean(nominal_eig_batch, axis=0)
                plt.plot(eig_avg.cpu().numpy(), label='Avg EIG', color='tab:blue', lw=2)
                plt.axhline(y=avg_nominal_eig, linestyle='--', label='Avg Nominal EIG', color='black', lw=2)
                # Get the existing handles and labels
                handles, labels = plt.gca().get_legend_handles_labels()
                # Choose which items you want to include (e.g., the first and third)
                handles_to_show = [handles[-2], handles[-1]]
                labels_to_show = [labels[-2], labels[-1]]
                # Create a legend with only the selected items
                plt.legend(handles_to_show, labels_to_show)
                plt.title(f"{eval_args['num_evals']} EIG eval(s) with {eval_args['eval_particles']} samples")  
                plt.tight_layout()
                plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/eigs.png")
                # save eigs at the end
                np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/eigs.npy", eig_avg.cpu().numpy())
                np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nominal_eig.npy", avg_nominal_eig)
                mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/eigs.npy")
                if eval_args["brute_force"]:
                    bf_optimal_design = designs[np.argmax(brute_force_EIG)]
                    print("Optimal design (BF):", bf_optimal_design)
                    np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/bf_optimal_design.npy", bf_optimal_design.cpu().numpy())
                    mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/bf_optimal_design.npy")
                nf_optimal_design = designs[torch.argmax(eig_avg)]
                print("Optimal design (NF):", nf_optimal_design)
                np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nf_optimal_design.npy", nf_optimal_design.cpu().numpy())
                mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nf_optimal_design.npy")

                sorted_eig_idx = torch.argsort(eig_avg, descending=True)
                sorted_eig = eig_avg[sorted_eig_idx]
                sorted_se = eig_se[sorted_eig_idx]
                sorted_design = designs[sorted_eig_idx]
                fig = plt.figure(figsize=(20, 6))  # Increase figure height as needed
                gs = gridspec.GridSpec(3, 1, height_ratios=[0.6, 0.2, 0.4], width_ratios=[1])
                ax0 = fig.add_subplot(gs[0, 0])
                ax1 = fig.add_subplot(gs[1, 0])
                ax2 = fig.add_subplot(gs[2, 0])
                ax0.fill_between(
                    range(len(sorted_eig)), 
                    (sorted_eig-sorted_se).cpu().numpy(), 
                    (sorted_eig+sorted_se).cpu().numpy(), 
                    color="tab:blue", 
                    alpha=0.2)
                ax0.plot(sorted_eig.cpu().numpy(), color="tab:blue", label="NF Model")
                if eval_args["brute_force"]:
                    plt.plot(brute_force_EIG.squeeze()[sorted_eig_idx.cpu().numpy()], label='Brute Force', color='tab:blue', linestyle='--')
                ax0.set_xlim(0, len(sorted_eig))
                ax0.axhline(avg_nominal_eig, color='black', linestyle='--', label="Nominal EIG")
                ax0.set_ylabel("Expected Information Gain [bits]")
                ax0.set_xticks([])
                ax0.legend()
                im = ax1.imshow(sorted_design.T.cpu().numpy(), aspect='auto', cmap='viridis')
                ax1.set_xlabel("Design Index")
                ax1.set_yticks(np.arange(len(classes.keys())), classes.keys())

                correlations = np.array([np.corrcoef(designs[:, i].cpu().numpy(), eig_avg.cpu().numpy())[0, 1] for i in range(designs.shape[1])])
                im2 = ax2.imshow(correlations.reshape(1, -1))
                ax2.set_xticks(np.arange(len(correlations)))
                ax2.set_xticklabels(classes.keys())
                ax2.set_yticks([])
                # Create divider for existing axes instance
                divider = make_axes_locatable(ax2)
                # Add axes below for colorbar with 20% height of main plot and padding
                cax = divider.append_axes("bottom", size="10%", pad=0.3)
                # Create colorbar in the new axes
                plt.colorbar(im2, cax=cax, orientation='horizontal', label='Correlation Coefficient')
                plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/evaluation.png")

                #tot = (desi_df[::2]["observed"]).sum()
                values1 = nominal_design.squeeze().cpu().numpy()  # Values for the first set of bars
                values2 = nf_optimal_design.squeeze().cpu().numpy()  # Values for the second set of bars

                # Set the positions for the bars
                x = np.arange(len(classes.keys()))  # the label locations
                width = 0.2 # the width of the bars

                # Create the bar chart
                fig, ax = plt.subplots(figsize=(14, 7))
                bars1 = ax.bar(x - width/2, values1, width, label='Nominal Design', color="black")
                bars2 = ax.bar(x + width/2, values2, width, label='Optimal Design', color='tab:blue')
                ax.set_xlabel('Tracers')
                ax.set_ylabel('Num Tracers')
                ax.set_xticks(x)
                ax.set_xticklabels(classes.keys())
                ax.legend()
                plt.suptitle("Five Tracer Design (LRG1, LRG2, LRG3+ELG1, ELG2, and Lya QSO)", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/design_comparison.png")

                ############################################### Plot Posterior ###############################################
                central_vals = num_tracers.central_val if run_args["include_D_M"] else num_tracers.central_val[1::2]
                optimal_context = torch.cat([nf_optimal_design, central_vals], dim=-1)
                nominal_context = torch.cat([nominal_design, central_vals], dim=-1)
                print(f"Optimal context: {optimal_context} \nNominal context: {nominal_context}")
                optimal_samples = posterior_flow(optimal_context).sample((eval_args["post_samples"],)).cpu().numpy()
                nominal_samples = posterior_flow(nominal_context).sample((eval_args["post_samples"],)).cpu().numpy()
                # Rescale the hrdrag samples 
                optimal_samples[:, -1] *= 100000
                nominal_samples[:, -1] *= 100000
                optimal_samples_gd = getdist.MCSamples(samples=optimal_samples, names=target_labels, labels=latex_labels)
                nominal_samples_gd = getdist.MCSamples(samples=nominal_samples, names=target_labels, labels=latex_labels)
                desi_samples = np.load(f"{home_dir}/data/mcmc_samples/{run_args['cosmo_model']}.npy")
                desi_samples_gd = getdist.MCSamples(samples=desi_samples, names=target_labels, labels=latex_labels)
                np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/optimal_samples.npy", optimal_samples)
                np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nominal_samples.npy", nominal_samples)
                np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/desi_samples.npy", desi_samples)
                mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/optimal_samples.npy")
                mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/nominal_samples.npy")
                mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/desi_samples.npy")

                g = plots.getSubplotPlotter()
                if not eval_args["brute_force"]:
                    g = plot_posterior(
                        [desi_samples_gd, nominal_samples_gd, optimal_samples_gd],
                        ['black', 'tab:blue', 'tab:orange'],
                        legend_labels=['DESI', 'Nominal', 'Optimal'],
                        show_scatter=True
                    )
                else:
                    brute_force_nominal_samples = num_tracers.brute_force_posterior(nominal_design, designer, grid_params, num_param_samples=eval_args["post_samples"]).cpu().numpy()
                    brute_force_optimal_samples = num_tracers.brute_force_posterior(nf_optimal_design, designer, grid_params, num_param_samples=eval_args["post_samples"]).cpu().numpy()
                    brute_force_nominal_samples_gd = getdist.MCSamples(samples=brute_force_nominal_samples, names=target_labels, labels=latex_labels)
                    brute_force_optimal_samples_gd = getdist.MCSamples(samples=brute_force_optimal_samples, names=target_labels, labels=latex_labels)

                    g.triangle_plot(
                        [nominal_samples_gd, optimal_samples_gd, brute_force_nominal_samples_gd, brute_force_optimal_samples_gd],
                        filled=False, 
                        colors=['black', 'tab:blue', 'black', 'tab:blue'], 
                        legend_labels=['Nominal', 'Optimal', 'Brute Force (Nominal)', 'Brute Force (Optimal)'], 
                        legend_loc='upper right',
                        diag1d_kwargs={
                            'colors': ['black', 'tab:blue', 'black', 'tab:blue'],
                            'normalized': True,
                            'linestyle': ['-', '-', '--', '--']
                        },
                        contour_args={
                            'ls': ['-', '-', '--', '--']
                            }
                        )
                    #brute_force_samples = num_tracers.sample_brute_force(optimal_brute_force, grid_designs, grid_features, grid_params, designer).cpu().numpy()
                    corner.corner(brute_force_nominal_samples, labels=latex_labels, show_titles=False, title_fmt=".2f", 
                                    title_kwargs={"fontsize": 12}, levels=levels, fig=post_fig,
                                    bins=bins, plot_datapoints=False, plot_density=False, smooth=1.0, 
                                    fill_contours=False, color='black', range=hist_range, hist_kwargs={"linestyle": 'dashed'}, contour_kwargs={"linestyles": 'dashed'})
                    legend_labels.append(plt.Line2D([0], [0], color="black", lw=1, label="Brute Force (Nominal)", linestyle='dashed'))
                    corner.corner(brute_force_optimal_samples, labels=latex_labels, show_titles=False, title_fmt=".2f", 
                                    title_kwargs={"fontsize": 12}, levels=levels, fig=post_fig,
                                    bins=bins, plot_datapoints=False, plot_density=False, smooth=1.0, 
                                    fill_contours=False, color='tab:blue', range=hist_range, hist_kwargs={"linestyle": 'dashed'}, contour_kwargs={"linestyles": 'dashed'})
                    legend_labels.append(plt.Line2D([0], [0], color="tab:blue", lw=1, label="Brute Force (Optimal)", linestyle='dashed'))

                g.export(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/posterior.png")

        plt.close('all')
        print("Eval", ml_info.experiment_id + "/" + ml_info.run_id, "completed.")

if __name__ == '__main__':

    device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    #set default dtype
    torch.set_default_dtype(torch.float64)

    process = psutil.Process(os.getpid())
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Trigger garbage collection for CPU
    gc.collect()
    print(f"Memory before run: {process.memory_info().rss / 1024**2} MB")
    eval_args = {
        "nf_model": True,
        "brute_force": False,
        "post_samples": 200000,
        "eval_particles": 3000,
        "num_evals": 8,
        "params_grid": 100,
        "features_grid": 30,
        "mem": 200000
    }
    run_eval(
        eval_args,
        run_id=None,
        exp='base_NAF_particles4_fixed',
        device=device
        )
    mlflow.end_run()