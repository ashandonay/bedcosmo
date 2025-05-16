import os
import sys
# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory ('BED_cosmo/') and add it to the Python path
parent_dir_abs = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, parent_dir_abs)

import torch
import torch.nn as nn
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.stats import norm
from urllib.request import urlopen

import pyro
from pyro import poutine
from pyro_oed_src import nf_loss, _create_condition_input
from pyro.contrib.util import lexpand, rexpand

import matplotlib.pyplot as plt

from nflows.transforms import made as made_module
from bed.grid import Grid

from astropy.cosmology import Planck18
from astropy import constants

import mlflow
import mlflow.pytorch
from tqdm import tqdm

from bed.grid import Grid

import psutil
import os
import gc
from num_tracers import NumTracers
from util import *
import json
import argparse
import io
import contextlib
from plotting import get_contour_area, plot_training, plot_eig_steps

class FlowLikelihoodDataset(Dataset):
    def __init__(self, num_tracers, designs, n_particles_per_gpu, observation_labels, target_labels, device="cuda"):
        self.num_tracers = num_tracers
        # Move designs to GPU for faster sampling
        self.designs = designs.to(device)
        self.n_particles_per_gpu = n_particles_per_gpu
        self.device = device
        self.observation_labels = observation_labels
        self.target_labels = target_labels

    def __len__(self):
        # We only have one "batch" of designs, so return 1
        return 1

    def __getitem__(self, idx):
        # Dynamically expand the designs on each access
        expanded_design = lexpand(self.designs, self.n_particles_per_gpu).to(self.device)

        # Generate samples from the NumTracers pyro model
        with torch.no_grad():
            # Generate the samples directly on the GPU
            trace = poutine.trace(self.num_tracers.pyro_model).get_trace(expanded_design)
            y_dict = {l: trace.nodes[l]["value"].to(self.device) for l in self.observation_labels}
            theta_dict = {l: trace.nodes[l]["value"].to(self.device) for l in self.target_labels}

            # Extract the target samples (theta)
            samples = torch.cat([theta_dict[k].unsqueeze(dim=-1) for k in self.target_labels], dim=-1)

            # Create the condition input
            condition_input = _create_condition_input(
                design=expanded_design,
                y_dict=y_dict,
                observation_labels=self.observation_labels,
                condition_design=True
            )

        # Return everything on the appropriate device (GPU)
        return condition_input, samples

# Helper function for DataLoader with DistributedSampler
def get_dataloader(num_tracers, designs, n_particles_per_gpu, observation_labels, target_labels, batch_size, local_rank, num_workers=0, world_size=None):
    # Create dataset with designs on GPU
    dataset = FlowLikelihoodDataset(
        num_tracers=num_tracers,
        designs=designs.to(f"cuda:{local_rank}"),
        n_particles_per_gpu=n_particles_per_gpu,
        observation_labels=observation_labels,
        target_labels=target_labels,
        device=f"cuda:{local_rank}"
    )
    
    # Use regular DataLoader without DistributedSampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return dataloader

def single_run(
    cosmo_model,
    run_args,
    mlflow_experiment_name,
    device="cuda:0",
    resume_id=None,
    resume_step=None,
    add_steps=0,
    **kwargs,
):
    print(mlflow_experiment_name)
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Trigger garbage collection for CPU
    gc.collect()
    process = psutil.Process(os.getpid())
    print(f"Memory before run: {process.memory_info().rss / 1024**2} MB")
    pyro.clear_param_store()
    print(f"Running with parameters for cosmo_model='{cosmo_model}':")
    storage_path = os.environ["SCRATCH"] + "/bed/BED_cosmo/num_tracers"
    home_dir = os.environ["HOME"]
    mlflow.set_tracking_uri(storage_path + "/mlruns")

    # DDP initialization
    if "LOCAL_RANK" in os.environ:
        tdist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    # Initialize MLflow for all ranks
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    
    # Initialize MLflow experiment and run info on rank 0
    if local_rank == 0:
        if resume_id:
            # First get the experiment info from the run we want to resume
            client = mlflow.MlflowClient()
            run_info = client.get_run(resume_id)
            exp_id = run_info.info.experiment_id
            exp_name = mlflow.get_experiment(exp_id).name
            
            # Set the experiment before starting the run
            mlflow.set_experiment(experiment_id=exp_id)
            mlflow.start_run(run_id=resume_id)
            
            if resume_step is None:
                raise ValueError("resume_step must be provided when resuming a run")
            
            # get the exp name from the run id
            mlflow_experiment_name = exp_name
            cosmo_model = run_info.data.params["cosmo_model"]
            run_args = parse_mlflow_params(run_info.data.params)
            print(json.dumps(run_args, indent=2))
            if add_steps:
                run_args["steps"] += add_steps
            best_nominal_areas = client.get_metric_history(resume_id, 'best_nominal_area')
            best_nominal_area_steps = np.array([metric.step for metric in best_nominal_areas])
            # get the best area from a step prior to the resume step
            closest_idx = np.argmin(np.abs(best_nominal_area_steps - resume_step))
            best_nominal_area = best_nominal_areas[closest_idx].value if best_nominal_area_steps[closest_idx] < resume_step else best_nominal_areas[closest_idx - 1].value
            print(f"Starting at best nominal area: {best_nominal_area} at step {best_nominal_area_steps[closest_idx]}")
            best_losses = client.get_metric_history(resume_id, 'best_loss')
            best_loss_steps = np.array([metric.step for metric in best_losses])
            # get the best loss from a step prior to the resume step
            closest_idx = np.argmin(np.abs(best_loss_steps - resume_step))
            best_loss = best_losses[closest_idx].value if best_loss_steps[closest_idx] < resume_step else best_losses[closest_idx - 1].value
            print(f"Starting at best loss: {best_loss} at step {best_loss_steps[closest_idx]}")
            checkpoint_files = os.listdir(f"{storage_path}/mlruns/{exp_id}/{resume_id}/artifacts/checkpoints")
            checkpoint_steps = sorted([
                int(f.split('_')[-1].split('.')[0]) 
                for f in checkpoint_files 
                if f.startswith('checkpoint_') 
                and f.endswith('.pt') 
                and not f.endswith('_loss_best.pt') 
                and not f.endswith('_nominal_area_best.pt') 
                and not f.endswith('_last.pt')
            ])
            # get the checkpoint prior to the resume step
            closest_idx = np.argmin(np.abs(np.array(checkpoint_steps) - resume_step))
            start_step = checkpoint_steps[closest_idx] if checkpoint_steps[closest_idx] < resume_step else checkpoint_steps[closest_idx - 1]
            history = client.get_metric_history(resume_id, 'loss')
            history = [metric.value for metric in history]
            history = history[:start_step]
            # get the checkpoint file name for the start step
            checkpoint_dir = f"{storage_path}/mlruns/{exp_id}/{resume_id}/artifacts/checkpoints"
            checkpoint_file = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pt') and f.split('_')[-1].split('.')[0] == str(start_step)][0]
            start_step += 1
            checkpoint_path = f"{checkpoint_dir}/{checkpoint_file}"
            print(f"Resuming MLflow run: {resume_id}")
            
            # Convert numeric values to tensors for broadcasting
            exp_id_tensor = torch.tensor([int(exp_id)], dtype=torch.long, device=f"cuda:{local_rank}")
            start_step_tensor = torch.tensor([start_step], dtype=torch.long, device=f"cuda:{local_rank}")
            best_loss_tensor = torch.tensor([best_loss], dtype=torch.float64, device=f"cuda:{local_rank}")
            best_nominal_area_tensor = torch.tensor([best_nominal_area], dtype=torch.float64, device=f"cuda:{local_rank}")
            
            # Store run_id as a string in a list for broadcasting
            run_id_list = [resume_id]
            exp_name_list = [exp_name]
        else:
            print(json.dumps(run_args, indent=2))
            mlflow.set_experiment(mlflow_experiment_name)
            mlflow.start_run()
            # Log parameters for a new run
            mlflow.log_param("cosmo_model", cosmo_model)
            # log params in run_args
            for key, value in run_args.items():
                mlflow.log_param(key, value)
            # log params in kwargs
            for key, value in kwargs.items():
                mlflow.log_param(key, value)
            start_step = 0
            best_loss = float('inf')
            best_nominal_area = float('inf')
            history = []
            
            # Create directories for artifacts
            os.makedirs(f"{storage_path}/mlruns/{mlflow.active_run().info.experiment_id}/{mlflow.active_run().info.run_id}/artifacts/checkpoints", exist_ok=True)
            os.makedirs(f"{storage_path}/mlruns/{mlflow.active_run().info.experiment_id}/{mlflow.active_run().info.run_id}/artifacts/plots", exist_ok=True)
            
            # Convert numeric values to tensors for broadcasting
            exp_id_tensor = torch.tensor([int(mlflow.active_run().info.experiment_id)], dtype=torch.long, device=f"cuda:{local_rank}")
            start_step_tensor = torch.tensor([start_step], dtype=torch.long, device=f"cuda:{local_rank}")
            best_loss_tensor = torch.tensor([best_loss], dtype=torch.float64, device=f"cuda:{local_rank}")
            best_nominal_area_tensor = torch.tensor([best_nominal_area], dtype=torch.float64, device=f"cuda:{local_rank}")
            
            # Store run_id as a string in a list for broadcasting
            run_id_list = [mlflow.active_run().info.run_id]
            exp_name_list = [mlflow_experiment_name]
    else:
        # Initialize tensors on other ranks
        exp_id_tensor = torch.zeros(1, dtype=torch.long, device=f"cuda:{local_rank}")
        start_step_tensor = torch.zeros(1, dtype=torch.long, device=f"cuda:{local_rank}")
        best_loss_tensor = torch.zeros(1, dtype=torch.float64, device=f"cuda:{local_rank}")
        best_nominal_area_tensor = torch.zeros(1, dtype=torch.float64, device=f"cuda:{local_rank}")
        run_id_list = [None]  # Initialize empty list for run_id
        exp_name_list = [None]  # Initialize empty list for exp_name

    # Broadcast tensors to all ranks
    tdist.broadcast(exp_id_tensor, src=0)
    tdist.broadcast(start_step_tensor, src=0)
    tdist.broadcast(best_loss_tensor, src=0)
    tdist.broadcast(best_nominal_area_tensor, src=0)
    
    # Broadcast run_id and exp_name lists
    tdist.broadcast_object_list(run_id_list, src=0)
    tdist.broadcast_object_list(exp_name_list, src=0)
    
    # Set up MLflow for all ranks
    if local_rank != 0:
        mlflow.set_experiment(experiment_id=str(exp_id_tensor.item()))
        mlflow.start_run(run_id=run_id_list[0], nested=True)
    
    # Create ml_info object on all ranks with the same values
    ml_info = type('mlinfo', (), {})()
    ml_info.experiment_id = str(exp_id_tensor.item())
    ml_info.run_id = run_id_list[0]
    
    # Set resume-related variables on all ranks
    start_step = start_step_tensor.item()
    best_loss = best_loss_tensor.item()
    best_nominal_area = best_nominal_area_tensor.item()
    history = []  # History is not critical for other ranks

    # All ranks join the same run
    if resume_id:
        mlflow.start_run(run_id=resume_id, nested=True)
    else:
        mlflow.start_run(run_id=ml_info.run_id, nested=True)

    desi_df = pd.read_csv(home_dir + run_args["data_path"] + 'desi_data.csv')
    desi_tracers = pd.read_csv(home_dir + run_args["data_path"] + 'desi_tracers.csv')
    nominal_cov = np.load(home_dir + run_args["data_path"] + 'desi_cov.npy')
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
        device=f"cuda:{local_rank}",
        verbose=True
        )
    
    target_labels = num_tracers.cosmo_params
    print(f"Classes: {classes}\n"
        f"Cosmology: {cosmo_model}\n"
        f"Target labels: {target_labels}")

    ############################################### Designs ###############################################
    # if fixed design:
    if run_args["nominal_design"]:
        # Get nominal design from observed tracer counts
        nominal_design = torch.tensor(desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=f"cuda:{local_rank}")
        
        # Create grid with nominal design values
        grid_designs = Grid(
            N_LRG=nominal_design[0].cpu().numpy(), 
            N_ELG=nominal_design[1].cpu().numpy(), 
            N_QSO=nominal_design[2].cpu().numpy()
        )

        # Convert grid to tensor format
        designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=f"cuda:{local_rank}").unsqueeze(0)
        for name in grid_designs.names[1:]:
            design_tensor = torch.tensor(getattr(grid_designs, name).squeeze(), device=f"cuda:{local_rank}").unsqueeze(0)
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
        designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=f"cuda:{local_rank}").unsqueeze(1)
        for name in grid_designs.names[1:]:
            design_tensor = torch.tensor(getattr(grid_designs, name).squeeze(), device=f"cuda:{local_rank}").unsqueeze(1)
            designs = torch.cat((designs, design_tensor), dim=1)

    # Only create prior plot if not resuming and on rank 0
    if not resume_id and local_rank == 0:
        fig, axs = plt.subplots(ncols=len(target_labels), nrows=1, figsize=(5*len(target_labels), 5))
        if len(target_labels) == 1:
            axs = [axs]
        for i, p in enumerate(target_labels):
            support = num_tracers.priors[p].support
            eval_pts = torch.linspace(support.lower_bound, support.upper_bound, 200, device=f"cuda:{local_rank}")
            prob = torch.exp(num_tracers.priors[p].log_prob(eval_pts))[:-1]
            prob_norm = prob/torch.sum(prob)
            axs[i].plot(eval_pts.cpu().numpy()[:-1], prob_norm.cpu().numpy(), label="Prior", color="tab:blue", alpha=0.5)
            axs[i].set_title(p)
        plt.tight_layout()
        plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/prior.png")

    input_dim = len(target_labels)
    context_dim = len(classes.keys()) + 10 if run_args["include_D_M"] else len(classes.keys()) + 5
    if local_rank == 0:
        print("MLFlow Run Info:", ml_info.experiment_id + "/" + ml_info.run_id)
        np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/designs.npy", designs.cpu().detach().numpy())
        mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/designs.npy")
        print("Designs shape:", designs.shape)
        print("Calculating normalizing flow EIG...")
        print(f'Input dim: {input_dim}, Context dim: {context_dim}')

    posterior_flow = init_nf(
        run_args["flow_type"],
        input_dim,
        context_dim,
        run_args,
        device=f"cuda:{local_rank}" if torch.cuda.is_available() else device,
        seed=run_args["nf_seed"],
        verbose=True,
        local_rank=local_rank
    )
    posterior_flow = DDP(posterior_flow, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    nominal_design = torch.tensor(desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=f"cuda:{local_rank}")
    central_vals = num_tracers.central_val if run_args["include_D_M"] else num_tracers.central_val[1::2]
    nominal_context = torch.cat([nominal_design, central_vals], dim=-1)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=run_args["initial_lr"])
    scheduler = init_scheduler(optimizer, run_args)
    
    # Load checkpoint if specified
    if resume_id:
        # Broadcast checkpoint path to all ranks
        if local_rank == 0:
            # Convert checkpoint path to bytes and create tensor
            checkpoint_path_bytes = checkpoint_path.encode('utf-8')
            checkpoint_path_length = len(checkpoint_path_bytes)
            checkpoint_path_tensor = torch.tensor([checkpoint_path_length], dtype=torch.long, device=f"cuda:{local_rank}")
            checkpoint_path_bytes_tensor = torch.tensor(list(checkpoint_path_bytes), dtype=torch.uint8, device=f"cuda:{local_rank}")
        else:
            checkpoint_path_tensor = torch.zeros(1, dtype=torch.long, device=f"cuda:{local_rank}")
            checkpoint_path_bytes_tensor = None

        # Broadcast checkpoint path length
        tdist.broadcast(checkpoint_path_tensor, src=0)
        
        # Create tensor for receiving bytes on non-zero ranks
        if local_rank != 0:
            checkpoint_path_bytes_tensor = torch.zeros(checkpoint_path_tensor.item(), dtype=torch.uint8, device=f"cuda:{local_rank}")
        
        # Broadcast checkpoint path bytes
        tdist.broadcast(checkpoint_path_bytes_tensor, src=0)
        
        # Convert bytes back to string
        if local_rank != 0:
            checkpoint_path = bytes(checkpoint_path_bytes_tensor.cpu().numpy()).decode('utf-8')

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{local_rank}")
        
        # Load model state dict
        state_dict = checkpoint['model_state_dict']
        # Remove module prefixes if they exist
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.module.'):
                k = k[7:]  # Remove one module prefix
            new_state_dict[k] = v
        
        # Load the cleaned state dict
        posterior_flow.module.load_state_dict(new_state_dict, strict=True)
        
        if local_rank == 0:
            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"Resuming from step {start_step}")

        # Restore RNG states if they exist in the checkpoint
        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'].cpu())
            pyro.get_param_store().set_state(rng_state['pyro'])
            if torch.cuda.is_available() and rng_state['cuda'] is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng_state['cuda']])
            if local_rank == 0:
                print("RNG states restored from checkpoint")

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if it exists
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Synchronize all ranks after loading checkpoint
        tdist.barrier()
    else:
        seed = auto_seed(base_seed=run_args["pyro_seed"], local_rank=local_rank)
        # test sample from the flow (only on rank 0)
        if local_rank == 0:
            with torch.no_grad():
                samples = posterior_flow.module(nominal_context).sample((1000,)).cpu().numpy()
                plt.figure()
                plt.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.5)
                plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/init_samples.png")
            print(f"Seed: {seed}")

    verbose_shapes = run_args["verbose"]
    # Disable tqdm progress bar if output is not a TTY
    is_tty = sys.stdout.isatty()
    # Set n_particles per GPU, which will also be the batch size
    n_particles_per_gpu = run_args.get("n_particles", 1024)

    dataloader = get_dataloader(
        num_tracers=num_tracers,
        designs=designs.cpu(),
        n_particles_per_gpu=n_particles_per_gpu,
        observation_labels=["y"],
        target_labels=target_labels,
        batch_size=1,
        local_rank=local_rank,
        world_size=tdist.get_world_size()
    )

    steps = run_args.get("steps", 1000)

    step = start_step
    best_loss = float('inf')
    best_nominal_area = float('inf')
    global_nominal_area = np.nan

    # Initialize pbar as None by default
    if is_tty and local_rank == 0:
        pbar = tqdm(total=steps - step, desc="Training Progress", position=0, leave=True)

    while step < steps:
        for context, samples in dataloader:
            optimizer.zero_grad()

            # Move data to GPU
            context = context.to(f"cuda:{local_rank}")
            samples = samples.to(f"cuda:{local_rank}")

            if step > 0:
                verbose_shapes = False

            # Compute the loss using nf_loss
            agg_loss, loss = nf_loss(context, posterior_flow.module, samples, verbose_shapes=verbose_shapes)

            # Aggregate global loss and agg_loss across all ranks
            global_loss_tensor = torch.tensor(loss.mean().item(), device=f"cuda:{local_rank}")
            global_agg_loss_tensor = torch.tensor(agg_loss.item(), device=f"cuda:{local_rank}")
            tdist.all_reduce(global_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            tdist.all_reduce(global_agg_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            global_loss = global_loss_tensor.item() / tdist.get_world_size()
            global_agg_loss = global_agg_loss_tensor.item() / tdist.get_world_size()

            # Backpropagation
            agg_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(posterior_flow.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Before optimizer step, verify model consistency across ranks
            if tdist.get_world_size() > 1:
                # Gather all parameter tensors from each rank
                for name, param in posterior_flow.module.named_parameters():
                    # Only check parameters that require gradients
                    if param.requires_grad:
                        # Sum the parameters across all ranks
                        param_tensor = param.data.clone()
                        tdist.all_reduce(param_tensor, op=torch.distributed.ReduceOp.SUM)
                        
                        # Compute the average by dividing by the world size
                        param_avg = param_tensor / tdist.get_world_size()
                        # Check for consistency
                        if not torch.allclose(param.data, param_avg, rtol=1e-5, atol=1e-8):
                            print(f"Rank {local_rank} | Parameter '{name}' is not consistent across ranks!")
                            print(f"Local value: {param.data.mean().item()}, Global average: {param_avg.mean().item()}")

            # Synchronize gradients across processes
            tdist.barrier()

            # Log memory usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024**2

            # Log metrics for each rank
            mlflow.log_metric(f"loss_rank_{local_rank}", loss.mean().item(), step=step)
            mlflow.log_metric(f"agg_loss_rank_{local_rank}", agg_loss.item(), step=step)
            mlflow.log_metric(f"memory_usage_MB_rank_{local_rank}", memory_usage, step=step)

            # Save the checkpoint and log the best loss on rank 0
            if local_rank == 0 and step > 99:
                if global_loss < best_loss:
                    best_loss = global_loss
                checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_loss_{step}.pt"
                save_checkpoint(posterior_flow, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler)
                checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_loss_best.pt"
                save_checkpoint(posterior_flow, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler)
                mlflow.log_metric("best_loss", best_loss, step=step)

            # Update progress bar or print status
            if local_rank == 0:
                mlflow.log_metric("loss", global_loss, step=step)
                mlflow.log_metric("agg_loss", global_agg_loss, step=step)
                if is_tty:
                    pbar.update(1)
                    if kwargs["log_nominal_area"]:
                        pbar.set_description(f"Loss: {loss.mean().item():.3f}, Area: {global_nominal_area:.3f}")
                    else:
                        pbar.set_description(f"Loss: {loss.mean().item():.3f}")
                else:
                    print(f"Step {step}, Loss: {loss.mean().item():.3f}") if not kwargs["log_nominal_area"] else print(f"Step {step}, Loss: {loss.mean().item():.3f}, Area: {global_nominal_area:.3f}")

            if step % 100 == 0:
                if step > 99 and kwargs["log_nominal_area"]:
                    nominal_samples = posterior_flow(nominal_context).sample((3000,)).cpu().numpy()
                    nominal_samples[:, -1] *= 100000
                    with contextlib.redirect_stdout(io.StringIO()):
                        nominal_samples_gd = getdist.MCSamples(samples=nominal_samples, names=target_labels, labels=num_tracers.latex_labels)

                    # Calculate the nominal area
                    local_nominal_area = get_contour_area(nominal_samples_gd, 'Om', 'hrdrag', 0.68)[0]

                    # Aggregate the nominal areas across all ranks
                    nominal_area_tensor = torch.tensor(local_nominal_area, device=f"cuda:{local_rank}")
                    tdist.all_reduce(nominal_area_tensor, op=tdist.ReduceOp.SUM)
                    global_nominal_area = nominal_area_tensor.item() / tdist.get_world_size()

                # Log the global nominal area on rank 0
                if local_rank == 0:
                    mlflow.log_metric("nominal_area", global_nominal_area, step=step)

                    # Save checkpoint if the global nominal area is the best so far
                    if global_nominal_area < best_nominal_area:
                        best_nominal_area = global_nominal_area
                        mlflow.log_metric("best_nominal_area", best_nominal_area, step=step)

                        # Save the best nominal area checkpoint
                        checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_nominal_area_{step}.pt"
                        save_checkpoint(posterior_flow, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler)

                        # Save the final best area checkpoint
                        checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_nominal_area_best.pt"
                        save_checkpoint(posterior_flow, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler)

            # Synchronize across ranks before scheduler step
            tdist.barrier()
            # Update scheduler every step_freq steps
            if steps % run_args.get("step_freq", 1) == 0:
                scheduler.step()

            if local_rank == 0:
                for param_group in optimizer.param_groups:
                    mlflow.log_metric("lr", param_group['lr'], step=step)

            step += 1
            if step >= steps:
                break

    # Ensure progress bar closes cleanly
    if local_rank == 0 and pbar is not None:
        pbar.close()

    checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_last.pt"
    # Only save checkpoint and plot on rank 0
    if local_rank == 0:
        save_checkpoint(posterior_flow.module, optimizer, checkpoint_path, step=run_args.get("steps", 0), artifact_path="checkpoints", scheduler=scheduler)
        plot_training(
            run_id=ml_info.run_id,
            var=None,
            log_scale=True,
            loss_step_freq=10,
            area_step_freq=100,
            show_best=False
        )
        if not run_args["nominal_design"]:
            eval_args = {"n_samples": 3000, "device": f"cuda:{local_rank}", "eval_seed": 1}
            plot_eig_steps(
                run_id=ml_info.run_id,
                steps=[500, 'last'],
                eval_args=eval_args,
                cosmo_exp='num_tracers'
            )

        plt.close('all')
        print("Run", ml_info.experiment_id + "/" + ml_info.run_id, "completed.")
    # Final memory logging
    if "LOCAL_RANK" in os.environ:
        if local_rank == 0:
            print(f"Rank {local_rank} | Final Memory: {process.memory_info().rss / 1024**2:.2f} MB")
        # Synchronize all ranks before ending MLflow runs
        tdist.barrier()
        # End MLflow run for all ranks
        mlflow.end_run()
        tdist.destroy_process_group()

if __name__ == '__main__':

    mp.set_start_method("spawn", force=True)
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

    # Add arguments for resuming from a checkpoint
    parser.add_argument('--resume_id', type=str, default=None, help='MLflow run ID to resume training from')
    parser.add_argument('--resume_step', type=int, default=None, help='Step to resume training from')
    parser.add_argument('--add_steps', type=int, default=0, help='Number of steps to add to the training')

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
        if key not in ['cosmo_model', 'resume_id', 'resume_step', 'add_steps'] and value is not None and key in run_args:
            if isinstance(run_args[key], bool) and isinstance(value, bool):
                run_args[key] = value
            elif not isinstance(run_args[key], bool):
                print(f"Overriding '{key}': {run_args[key]} -> {value}")
                run_args[key] = value

    # --- Setup & Run --- 
    device = torch.device(device) if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    single_run(
        cosmo_model=cosmo_model,
        run_args=run_args,
        mlflow_experiment_name=args.exp_name,
        device=device,
        log_nominal_area=False,
        resume_id=args.resume_id,
        resume_step=args.resume_step,
        add_steps=args.add_steps
    )


