import os
import sys
import time
import datetime

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
from bedcosmo.pyro_oed_src import nf_loss, _create_condition_input
from pyro.contrib.util import lexpand, rexpand

import matplotlib.pyplot as plt

import mlflow
import mlflow.pytorch
from tqdm import tqdm

import psutil
import gc
from bedcosmo.util import auto_seed, init_nf
import json
import argparse
import io
import contextlib
from bedcosmo.plotting import plot_training
import getdist.mcsamples
from bedcosmo.num_tracers.cmb_likelihood import CMBLikelihood

class FlowLikelihoodDataset(Dataset):
    def __init__(self, likelihood, designs, n_particles_per_device, observation_labels, target_labels, device="cuda"):
        self.likelihood = likelihood
        # Move designs to GPU for faster sampling
        self.designs = designs.to(device)
        self.n_particles_per_device = n_particles_per_device
        self.device = device
        self.observation_labels = observation_labels
        self.target_labels = target_labels

    def __len__(self):
        # We only have one "batch" of designs, so return 1
        return 1

    def __getitem__(self, idx):
        # Dynamically expand the designs on each access
        expanded_design = lexpand(self.designs, self.n_particles_per_device).to(self.device)

        # Generate samples from the likelihood's pyro model
        with torch.no_grad():
            # Generate the samples directly on the GPU
            trace = poutine.trace(self.likelihood.pyro_model).get_trace(expanded_design)
            
            # Assuming trace values are already on the correct device from the model
            y_dict = {l: trace.nodes[l]["value"] for l in self.observation_labels}
            theta_dict = {l: trace.nodes[l]["value"] for l in self.target_labels}

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
def get_dataloader(likelihood, designs, n_particles_per_device, observation_labels, target_labels, batch_size, pytorch_device_idx_for_ddp, num_workers=0, world_size=None):
    # Create dataset with designs on GPU
    dataset = FlowLikelihoodDataset(
        likelihood=likelihood,
        designs=designs.to(f"cuda:{pytorch_device_idx_for_ddp}"),
        n_particles_per_device=n_particles_per_device,
        observation_labels=observation_labels,
        target_labels=target_labels,
        device=f"cuda:{pytorch_device_idx_for_ddp}"
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
    run_args,
    mlflow_experiment_name,
    device="cuda:0",
    resume_id=None,
    resume_step=None,
    add_steps=0,
    profile=False,
    restart_path=None,
    **kwargs,
):

    global_rank, local_rank, effective_device_id, pytorch_device_idx = init_training_env(tdist, device)

    # Clear GPU cache
    torch.cuda.empty_cache()
    process = psutil.Process()
    # Trigger garbage collection for CPU
    gc.collect()
    pyro.clear_param_store()
    storage_path = os.environ["SCRATCH"] + "/bedcosmo/num_tracers"
    home_dir = os.environ["HOME"]
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    
    # Initialize MLflow experiment and run info on rank 0
    current_pytorch_device = f"cuda:{pytorch_device_idx}" if "LOCAL_RANK" in os.environ and torch.cuda.is_available() else (f"cuda:{effective_device_id}" if effective_device_id != -1 else "cpu")

    ml_info, run_args, start_step, best_loss, _ = init_run(
        tdist, 
        global_rank, 
        current_pytorch_device, 
        storage_path, 
        mlflow_experiment_name, 
        'cmb', 
        run_args, 
        kwargs, 
        resume_id=resume_id, 
        resume_step=resume_step, 
        add_steps=add_steps
        )

    # Broadcast run_args from rank 0 to ensure consistency
    if global_rank == 0:
        run_args_list_to_broadcast = [run_args]
    else:
        run_args_list_to_broadcast = [None]

    if tdist.is_initialized():
        tdist.barrier()
    
    tdist.broadcast_object_list(run_args_list_to_broadcast, src=0)
    run_args = run_args_list_to_broadcast[0]
    if global_rank != 0 and resume_id:
        print(f"Rank {global_rank} received run_args after broadcast: steps = {run_args.get('steps')}")

    # All ranks join the same run
    if resume_id:
        mlflow.start_run(run_id=resume_id, nested=True)
    else:
        mlflow.start_run(run_id=ml_info.run_id, nested=True)

    # Instantiate the CMB Likelihood model
    likelihood_model = CMBLikelihood(device=current_pytorch_device)
    target_labels = likelihood_model.target_labels
    observation_labels = likelihood_model.observation_labels

    # For CMB, 'design' is a placeholder, as the experiment is fixed.
    # It's used to control the batch size of samples from the simulator.
    designs = torch.ones(1, 1, device=current_pytorch_device)
    
    input_dim = len(target_labels)
    # Context for the flow is the data vector (y_cmb) and the (dummy) design.
    context_dim = likelihood_model.covariance.shape[0] + designs.shape[-1]

    # Create a nominal context for logging/evaluation purposes
    nominal_design = designs[0]
    # The fiducial C_l vector serves as the central value for the data
    central_vals = likelihood_model.mean_fiducial
    nominal_context = torch.cat([nominal_design, central_vals], dim=-1)

    # Only create prior plot if not resuming and on rank 0
    if not resume_id and global_rank == 0:
        fig, axs = plt.subplots(ncols=len(target_labels), nrows=1, figsize=(5*len(target_labels), 5))
        if len(target_labels) == 1:
            axs = [axs]
        for i, p in enumerate(target_labels):
            support = likelihood_model.priors[p].support
            eval_pts = torch.linspace(support.lower_bound, support.upper_bound, 200, device=current_pytorch_device)
            prob = torch.exp(likelihood_model.priors[p].log_prob(eval_pts))
            axs[i].plot(eval_pts.cpu().numpy(), prob.cpu().numpy(), label="Prior", color="tab:blue", alpha=0.5)
            axs[i].set_title(p)
        plt.tight_layout()
        plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/prior.png")

    if global_rank == 0:
        print("MLFlow Run Info:", ml_info.experiment_id + "/" + ml_info.run_id)
        print(f"Using {run_args['n_devices']} devices with {run_args['n_particles']} total particles.")
        print("Designs shape:", designs.shape)
        print("Calculating normalizing flow for CMB posterior...")
        print(f'Input dim: {input_dim}, Context dim: {context_dim}')
        print(f"Cosmology: CMB\n"
              f"Target labels: {target_labels}")
        np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/designs.npy", designs.cpu().detach().numpy())
        mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/designs.npy")

    posterior_flow = init_nf(
        run_args["flow_type"],
        input_dim,
        context_dim,
        run_args,
        device=current_pytorch_device,
        seed=run_args["nf_seed"],
        verbose=True,
        local_rank=local_rank
    )

    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        ddp_device_spec = [pytorch_device_idx] 
        ddp_output_device_spec = pytorch_device_idx
    else:
        ddp_device_spec = [effective_device_id] if effective_device_id != -1 else None
        ddp_output_device_spec = effective_device_id if effective_device_id != -1 else None

    if torch.cuda.is_available():
        model_device = next(posterior_flow.parameters()).device
        if model_device.type == 'cpu' and ddp_device_spec is not None and ddp_device_spec[0] is not None and isinstance(ddp_device_spec[0], int) :
            posterior_flow.to(f"cuda:{ddp_device_spec[0]}")
    
    posterior_flow = DDP(posterior_flow, device_ids=ddp_device_spec, output_device=ddp_output_device_spec, find_unused_parameters=False)
    
    optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=run_args["initial_lr"])
    scheduler = init_scheduler(optimizer, run_args)
    
    # Load checkpoint if specified
    if resume_id:
        # Checkpoint loading logic adapted from original script
        checkpoint_dir = f"{storage_path}/mlruns/{ml_info.experiment_id}/{resume_id}/artifacts/checkpoints"
        checkpoint_file = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pt') and f.split('_')[-1].split('.')[0] == str(start_step)][0]
        checkpoint_path = f"{checkpoint_dir}/{checkpoint_file}"
        if global_rank == 0:
            checkpoint_path_bytes = checkpoint_path.encode('utf-8')
            checkpoint_path_length = len(checkpoint_path_bytes)
            checkpoint_path_tensor = torch.tensor([checkpoint_path_length], dtype=torch.long, device=current_pytorch_device)
            checkpoint_path_bytes_tensor = torch.tensor(list(checkpoint_path_bytes), dtype=torch.uint8, device=current_pytorch_device)
        else:
            checkpoint_path_tensor = torch.zeros(1, dtype=torch.long, device=current_pytorch_device)
            checkpoint_path_bytes_tensor = None

        tdist.broadcast(checkpoint_path_tensor, src=0)
        
        if global_rank != 0:
            checkpoint_path_bytes_tensor = torch.zeros(checkpoint_path_tensor.item(), dtype=torch.uint8, device=current_pytorch_device)
        
        tdist.broadcast(checkpoint_path_bytes_tensor, src=0)
        
        if global_rank != 0:
            checkpoint_path = bytes(checkpoint_path_bytes_tensor.cpu().numpy()).decode('utf-8')

        checkpoint = torch.load(checkpoint_path, map_location=current_pytorch_device, weights_only=False)
        posterior_flow.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if global_rank == 0:
            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"Resuming from step {start_step}")
        tdist.barrier()

    elif restart_path:
        checkpoint = torch.load(restart_path, map_location=current_pytorch_device, weights_only=False)
        posterior_flow.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if global_rank == 0:
            print(f"Checkpoint loaded from {restart_path}")
        tdist.barrier()
    
    # Set n_particles per GPU, which will also be the batch size
    n_particles_per_device = run_args.get("n_particles_per_device", 1024)

    dataloader = get_dataloader(
        likelihood=likelihood_model,
        designs=designs.cpu(),
        n_particles_per_device=n_particles_per_device,
        observation_labels=observation_labels,
        target_labels=target_labels,
        batch_size=1,
        pytorch_device_idx_for_ddp=pytorch_device_idx if "LOCAL_RANK" in os.environ else effective_device_id,
        world_size=tdist.get_world_size() if "LOCAL_RANK" in os.environ else 1
    )

    steps = run_args.get("steps", 1000)
    step = start_step
    best_loss = float('inf')

    if sys.stdout.isatty() and global_rank == 0:
        pbar = tqdm(total=steps - step, desc="Training Progress", position=0, leave=True)

    while step < steps:
        for context, samples in dataloader:
            optimizer.zero_grad()
            
            agg_loss, loss = nf_loss(context, posterior_flow.module, samples, rank=global_rank, verbose_shapes=(step==0))

            global_loss_tensor = loss.mean().detach()
            tdist.all_reduce(global_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            global_loss = global_loss_tensor.item() / tdist.get_world_size()

            agg_loss.backward()
            torch.nn.utils.clip_grad_norm_(posterior_flow.parameters(), max_norm=1.0)
            optimizer.step()
            tdist.barrier()

            if global_rank == 0:
                mlflow.log_metric("loss", global_loss, step=step)
                if sys.stdout.isatty():
                    pbar.update(1)
                    pbar.set_description(f"Loss: {global_loss:.3f}")
                else:
                    print(f"Step {step}, Loss: {global_loss:.3f}")

            if step % 100 == 0 and step != 0:
                if global_rank == 0:
                    if global_loss < best_loss:
                        best_loss = global_loss
                        checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_loss_best.pt"
                        save_checkpoint(posterior_flow.module, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler)
                        mlflow.log_metric("best_loss", best_loss, step=step)

            tdist.barrier()

            if global_rank == 0:
                mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=step)

            if step % run_args.get("step_freq", 1) == 0:
                scheduler.step()

            log_usage_metrics(current_pytorch_device, process, step, global_rank)
            step += 1
            if step >= steps:
                break

    if global_rank == 0 and sys.stdout.isatty():
        pbar.close()

    if global_rank == 0:
        checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_last.pt"
        save_checkpoint(posterior_flow.module, optimizer, checkpoint_path, step=steps, artifact_path="checkpoints", scheduler=scheduler)
        plot_training(run_id=ml_info.run_id, log_scale=True, loss_step_freq=10)
        
        # Plot posterior samples
        with torch.no_grad():
            samples = posterior_flow.module(nominal_context).sample((5000,)).cpu().numpy()
        
        g = getdist.mcsamples.MCSamples(samples=samples, names=target_labels, labels=likelihood_model.latex_labels)
        g.save_publish_table(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/posterior_table.txt")
        
        plt.close('all')
        print("Run", ml_info.experiment_id + "/" + ml_info.run_id, "completed.")

    if "LOCAL_RANK" in os.environ:
        tdist.barrier()
        mlflow.end_run()
        tdist.destroy_process_group()
        if global_rank == 0:
            print("Runtime:", get_runtime(ml_info.run_id))

if __name__ == '__main__':

    mp.set_start_method("spawn", force=True)
    torch.set_default_dtype(torch.float64)

    config_path = os.path.join(os.path.dirname(__file__), 'run_args.json')
    with open(config_path, 'r') as f:
        run_args_dict = json.load(f)

    # Use a specific section of the config for cmb, or a default
    default_args = run_args_dict.get('cmb', run_args_dict.get('base', {}))
    default_exp_name = f"cmb_{default_args.get('flow_type', 'nsf')}"

    parser = argparse.ArgumentParser(description="Run CMB Posterior Inference Training")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--exp_name', type=str, default=default_exp_name, help='Experiment name')
    parser.add_argument('--resume_id', type=str, default=None, help='MLflow run ID to resume training from')
    parser.add_argument('--resume_step', type=int, default=None, help='Step to resume training from')
    parser.add_argument('--add_steps', type=int, default=0, help='Number of steps to add to the training')
    parser.add_argument('--profile', action='store_true', help='Enable profiling for a few steps and then exit.')
    parser.add_argument('--restart_path', type=str, default=None, help='Path to checkpoint for restarting training')

    for key, value in default_args.items():
        arg_type = type(value)
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true', help=f'Enable {key}')
            parser.set_defaults(**{key: value})
        else:
            parser.add_argument(f'--{key}', type=arg_type, default=None, help=f'Override {key} (default: {value})')

    args = parser.parse_args()
    device = args.device

    run_args = default_args.copy()
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None and key in run_args:
            if not isinstance(run_args[key], bool) or value:
                if os.environ.get('RANK') == '0':
                    print(f"Overriding '{key}': {run_args[key]} -> {value}")
                run_args[key] = value
    
    initial_device_check_cuda = False
    if torch.cuda.is_available():
        try:
            if args.device.startswith("cuda:") and ":" in args.device.split(":"):
                parsed_id = int(args.device.split(':')[1])
                if 0 <= parsed_id < torch.cuda.device_count():
                    script_level_device_str = args.device
                else:
                    script_level_device_str = "cuda:0"
            else:
                 script_level_device_str = "cuda:0"
            torch.device(script_level_device_str)
            initial_device_check_cuda = True
        except Exception as e:
            print(f"Initial CUDA check failed: {e}. Defaulting to CPU.")
            script_level_device_str = "cpu"
    else:
        script_level_device_str = "cpu"

    print(f'Main script context: Using device string: {script_level_device_str}.')
    print(f"Rank {os.environ.get('RANK')} (Global) / {os.environ.get('LOCAL_RANK')} (Env Var): In __main__, about to call single_run.")

    single_run(
        run_args=run_args,
        mlflow_experiment_name=args.exp_name,
        device=script_level_device_str,
        resume_id=args.resume_id,
        resume_step=args.resume_step,
        add_steps=args.add_steps,
        profile=args.profile,
        restart_path=args.restart_path
    ) 