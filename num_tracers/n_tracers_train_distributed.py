import os
import sys
import time
import datetime
import random
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

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
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
from plotting import get_contour_area, plot_training, save_figure
import getdist.mcsamples

class FlowLikelihoodDataset(Dataset):
    def __init__(self, experiment, n_particles_per_device, device="cuda"):
        self.experiment = experiment
        self.n_particles_per_device = n_particles_per_device
        self.device = device

    def __len__(self):
        # We only have one "batch" of designs, so return 1
        return 1

    def __getitem__(self, idx):
        # Dynamically expand the designs on each access
        expanded_design = lexpand(self.experiment.designs, self.n_particles_per_device).to(self.device)

        # Generate samples from the NumTracers pyro model
        with torch.no_grad():
            # Generate the samples directly on the GPU
            trace = poutine.trace(self.experiment.pyro_model).get_trace(expanded_design)
            
            # Assuming trace values are already on the correct device from the model
            y_dict = {l: trace.nodes[l]["value"] for l in self.experiment.observation_labels}
            theta_dict = {l: trace.nodes[l]["value"] for l in self.experiment.cosmo_params}

            # Extract the target samples (theta)
            samples = torch.cat([theta_dict[k].unsqueeze(dim=-1) for k in self.experiment.cosmo_params], dim=-1)

            # Create the condition input
            condition_input = _create_condition_input(
                design=expanded_design,
                y_dict=y_dict,
                observation_labels=self.experiment.observation_labels,
                condition_design=True
            )

        # Return everything on the appropriate device (GPU)
        return condition_input, samples

# Helper function for DataLoader with DistributedSampler
def get_dataloader(experiment, n_particles_per_device, batch_size, pytorch_device_idx_for_ddp, num_workers=0, world_size=None):
    # Create dataset with designs on GPU
    dataset = FlowLikelihoodDataset(
        experiment=experiment,
        n_particles_per_device=n_particles_per_device,
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
    cosmo_exp,
    cosmo_model,
    run_args,
    mlflow_exp,
    device="cuda:0",
    profile=False,
    **kwargs
):

    global_rank, local_rank, effective_device_id, pytorch_device_idx = init_training_env(tdist, device)

    # Clear GPU cache
    torch.cuda.empty_cache()
    process = psutil.Process()
    # Trigger garbage collection for CPU
    gc.collect()
    pyro.clear_param_store()
    storage_path = os.environ["SCRATCH"] + "/bed/BED_cosmo/num_tracers"
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    
    # Initialize MLflow experiment and run info on rank 0
    current_pytorch_device = f"cuda:{pytorch_device_idx}" if "LOCAL_RANK" in os.environ and torch.cuda.is_available() else (f"cuda:{effective_device_id}" if effective_device_id != -1 else "cpu")

    ml_info, run_args, checkpoint, start_step, best_loss, best_nominal_area = init_run(
        mlflow_exp, 
        cosmo_model, 
        run_args, 
        tdist, 
        global_rank, 
        current_pytorch_device, 
        storage_path, 
        **kwargs
        )

    if cosmo_exp == "num_tracers":
        experiment = NumTracers(
            data_path=run_args["data_path"],
            cosmo_model=run_args["cosmo_model"],
            design_step=run_args["design_step"],
            design_lower=run_args["design_lower"],
            design_upper=run_args["design_upper"],
            fixed_design=run_args["fixed_design"],
            include_D_M=run_args["include_D_M"],
            global_rank=global_rank,
            device=current_pytorch_device,
            mode='train',
            verbose=run_args["verbose"]
            )
        if global_rank == 0:
            fig = experiment.design_plot()
            save_figure(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/designs.png", fig=fig, close_fig=True, display_fig=False)
    # Only create prior plot if not resuming and on rank 0
    if not kwargs.get("resume_id", None) and global_rank == 0: # global_rank check
        fig, axs = plt.subplots(ncols=len(experiment.cosmo_params), nrows=1, figsize=(5*len(experiment.cosmo_params), 5))
        if len(experiment.cosmo_params) == 1:
            axs = [axs]
        for i, p in enumerate(experiment.cosmo_params):
            support = experiment.priors[p].support
            eval_pts = torch.linspace(support.lower_bound, support.upper_bound, 200, device=current_pytorch_device)
            prob = torch.exp(experiment.priors[p].log_prob(eval_pts))[:-1]
            prob_norm = prob/torch.sum(prob)
            axs[i].plot(eval_pts.cpu().numpy()[:-1], prob_norm.cpu().numpy(), label="Prior", color="tab:blue", alpha=0.5)
            axs[i].set_title(p)
        plt.tight_layout()
        plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/prior.png")

    posterior_flow = init_nf(
        run_args,
        len(experiment.cosmo_params),
        experiment.context_dim,
        device=current_pytorch_device,
        seed=run_args["nf_seed"]
        )
    if global_rank == 0:
        print("MLFlow Run Info:", ml_info.experiment_id + "/" + ml_info.run_id)
        print(f"Using {run_args['n_devices']} devices with {run_args['n_particles']} total particles.")
        print("Designs shape:", experiment.designs.shape)
        print("Calculating normalizing flow EIG...")
        print(f'Input dim: {len(experiment.cosmo_params)}, Context dim: {experiment.context_dim}')
        print(f"Cosmology: {cosmo_model}")
        print(f"Target labels: {experiment.cosmo_params}")
        print("Flow model initialized: \n", posterior_flow)
        np.save(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/designs.npy", experiment.designs.cpu().detach().numpy())
        mlflow.log_artifact(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/designs.npy")

    # For DDP, ensure device_ids and output_device use the pytorch_device_idx (which is 0)
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        ddp_device_spec = [pytorch_device_idx] 
        ddp_output_device_spec = pytorch_device_idx
    else:
        ddp_device_spec = [effective_device_id] if effective_device_id != -1 else None
        ddp_output_device_spec = effective_device_id if effective_device_id != -1 else None

    # Check if model is already on a CUDA device if CUDA is available
    if torch.cuda.is_available():
        model_device = next(posterior_flow.parameters()).device
        if model_device.type == 'cpu' and ddp_device_spec is not None and ddp_device_spec[0] is not None and isinstance(ddp_device_spec[0], int) :
            posterior_flow.to(f"cuda:{ddp_device_spec[0]}")
    
    posterior_flow = DDP(posterior_flow, device_ids=ddp_device_spec, output_device=ddp_output_device_spec, find_unused_parameters=False)
    
    nominal_context = torch.cat([
        experiment.nominal_design, 
        experiment.central_val if run_args["include_D_M"] else experiment.central_val[1::2]
        ], dim=-1)
    
    # Initialize optimizer
    if run_args["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=run_args["initial_lr"])
    elif run_args["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(posterior_flow.parameters(), lr=run_args["initial_lr"])
    else:
        raise ValueError(f"Invalid optimizer: {run_args['optimizer']}")
    
    if kwargs.get("resume_id", None):
        # Load model state dict
        posterior_flow.module.load_state_dict(checkpoint['model_state_dict'], strict=True)

        # Load optimizer state (preserve original optimizer state for resume)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Create scheduler with original run parameters (already loaded into run_args during resume)
        scheduler = init_scheduler(optimizer, run_args)
        
        # Load scheduler state if it exists (preserve original scheduler state for resume)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if global_rank == 0:
                print("Scheduler state restored from checkpoint")
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr}")
        else:
            if global_rank == 0:
                print("Warning: No scheduler state found in checkpoint")
                print("  Learning rate may not be correct for the resume step")
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Current learning rate: {current_lr}")
                
                # If no scheduler state, we need to manually step the scheduler to the correct point
                print(f"  Manually stepping scheduler to step {kwargs['resume_step']}")
                step_freq = run_args.get("step_freq", 1)
                scheduler_steps_needed = kwargs['resume_step'] // step_freq
                print(f"  Step frequency: {step_freq}, scheduler steps needed: {scheduler_steps_needed}")
                
                for _ in range(scheduler_steps_needed):
                    scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                if global_rank == 0:
                    print(f"  Learning rate after manual stepping: {current_lr}")
        
        if global_rank == 0:
            print(f"Optimizer LR after loading: {optimizer.param_groups[0]['lr']}")
            print("Resume checkpoint loading completed")
            
        # Synchronize all ranks after loading checkpoint
        tdist.barrier()

    elif kwargs.get("restart_id", None):
        # Load model weights and optimizer state
        posterior_flow.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        if kwargs.get("restart_optimizer", None):
            if run_args["optimizer"] != checkpoint["optimizer_type"] if "optimizer_type" in checkpoint else False:
                raise ValueError(f"Optimizer type mismatch: {run_args['optimizer']} != {checkpoint['optimizer_type']}")
            # Update the learning rate in the loaded optimizer state to match current run_args
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] = run_args['initial_lr']
                param_group['initial_lr'] = run_args['initial_lr']
                if global_rank == 0:
                    print(f"Restarted optimizer with initial learning rate: {old_lr} -> {run_args['initial_lr']}")
        else:
            if global_rank == 0:
                print(f"Created fresh optimizer with initial learning rate: {run_args['initial_lr']}")
        
        # Create fresh scheduler with new learning rate bounds
        scheduler = init_scheduler(optimizer, run_args)
        
        if global_rank == 0:
            print(f"Using existing optimizer statistics with new learning rate schedule")
            print(f"Initial lr: {run_args['initial_lr']}, Final lr: {run_args.get('final_lr', run_args['initial_lr'])}")
            print("Fresh scheduler created with current parameters")
        
        # Synchronize all ranks after loading checkpoint
        tdist.barrier()
        
        # test sample from the flow (only on rank 0)
        with torch.no_grad():
            samples = posterior_flow.module(nominal_context).sample((1000,)).cpu().numpy()
            plt.figure()
            plt.plot(samples.squeeze()[:, 0], samples.squeeze()[:, 1], 'o', alpha=0.5)
            plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/init_samples_rank_{global_rank}.png")
    else:
        scheduler = init_scheduler(optimizer, run_args)
        seed = auto_seed(base_seed=run_args["pyro_seed"], rank=global_rank)
        # test sample from the flow (only on rank 0)
        with torch.no_grad():
            samples = posterior_flow.module(nominal_context).sample((1000,)).cpu().numpy()
            plt.figure()
            plt.plot(samples.squeeze()[:, 0], samples.squeeze()[:, 1], 'o', alpha=0.5)
            plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/init_samples_rank_{global_rank}.png")

    verbose_shapes = run_args["verbose"]
    # Disable tqdm progress bar if output is not a TTY
    is_tty = sys.stdout.isatty()
    # Set n_particles per GPU, which will also be the batch size
    n_particles_per_device = run_args.get("n_particles_per_device", 500)

    dataloader = get_dataloader(
        experiment=experiment,
        n_particles_per_device=n_particles_per_device,
        batch_size=1,
        pytorch_device_idx_for_ddp=pytorch_device_idx if "LOCAL_RANK" in os.environ else effective_device_id,
        world_size=tdist.get_world_size() if "LOCAL_RANK" in os.environ else 1
    )

    total_steps = run_args.get("total_steps", 1000)
    step = start_step
    global_nominal_area = None

    # Initialize pbar as None by default
    if is_tty and global_rank == 0:
        pbar = tqdm(total=total_steps - step, desc="Training Progress", position=0, leave=True)

    # Profiling block
    if profile:
        if global_rank == 0:
            print("Starting DDP profiling...")
            profiler_log_dir = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/profiler_log_rank0"
            os.makedirs(profiler_log_dir, exist_ok=True)
            trace_handler = torch.profiler.tensorboard_trace_handler(profiler_log_dir)
        else:
            trace_handler = None # Only rank 0 writes traces

        # Define a schedule for profiling
        prof_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=prof_schedule,
            on_trace_ready=trace_handler,
            record_shapes=False, # Keep False unless shapes are critical for analysis
            profile_memory=False, # Keep False unless memory is critical
            with_stack=False, # Keep False, can be very verbose
            with_flops=False
        ) as prof:
            for profile_step in range(1 + 1 + 3): # wait + warmup + active
                if global_rank == 0 and profile_step >= 2: # Start describing active steps
                    print(f"Profiling active step: {profile_step - 2}")
                
                # --- Simulate one training step --- 
                for context, samples in dataloader: # This will iterate once due to batch_size=1 for the dataset
                    optimizer.zero_grad()
                    context = context.to(current_pytorch_device)
                    samples = samples.to(current_pytorch_device)
                    
                    temp_verbose_shapes = run_args["verbose"] if profile_step == 0 else False
                    agg_loss, loss = nf_loss(context, posterior_flow.module, samples, rank=global_rank, verbose_shapes=temp_verbose_shapes)
                    
                    # No need for global loss aggregation during profiling, focus on per-rank
                    agg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(posterior_flow.parameters(), max_norm=1.0)


                    # Save checkpoints BEFORE optimizer step (so loss values match exactly)
                    if step % 100 == 0 and step != 0:
                        # Save rank-specific checkpoint (default behavior)
                        checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_rank_{global_rank}_{step}.pt"
                        save_checkpoint(posterior_flow.module, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler, global_rank=global_rank, additional_state={
                            'rank': global_rank,
                            'world_size': tdist.get_world_size(),
                            'local_loss': loss.mean().detach().item(),
                            'local_agg_loss': agg_loss.detach().item()
                        })
                        
                        if global_rank == 0:
                            print(f"Saved rank-specific checkpoints for step {step}")

                    # Optimizer step
                    optimizer.step()
                    break # Exit after one batch from dataloader for this profile step
                # --- End of simulated training step ---
                
                prof.step() # Signal profiler that a step is complete
                tdist.barrier() # Synchronize all ranks after each profile step

            if global_rank == 0:
                print("\n=== Profiler Results (Rank 0) ===\n")
                # Sort by self_cuda_time_total for GPU-bound tasks
                print(prof.key_averages().table(
                    sort_by="self_cuda_time_total", 
                    row_limit=20
                ))
                summary_path = f"{profiler_log_dir}/profiler_summary_rank0.txt"
                with open(summary_path, 'w') as f:
                    f.write("=== Profiler Results (Rank 0) ===\n\n")
                    f.write(prof.key_averages().table(
                        sort_by="self_cuda_time_total", 
                        row_limit=20
                    ))
                mlflow.log_artifact(summary_path, "profiler_log_rank0")
                print(f"Profiling complete. Rank 0 data saved to TensorBoard format in {profiler_log_dir}")            
            else:
                # Other ranks can print their summary too if desired, or just confirm completion
                print(f"\n=== Profiler Results (Rank {global_rank}) ===\n")
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
                print(f"Rank {global_rank} profiling step complete.")

            tdist.barrier() # Ensure all ranks finish printing before exiting
            if "LOCAL_RANK" in os.environ: # Clean up DDP
                mlflow.end_run()
                tdist.destroy_process_group()
            print("Exiting after profiling.")
            return # Exit single_run function

    while step < total_steps:
        # Restore RNG state right before the first training step if resuming from checkpoint
        if step == start_step and checkpoint is not None:
            restore_state(checkpoint, start_step, global_rank)
            
        for context, samples in dataloader:
            optimizer.zero_grad()

            if step > 0:
                verbose_shapes = False
            # Compute the loss using nf_loss
            agg_loss, loss = nf_loss(context, posterior_flow.module, samples, rank=global_rank, verbose_shapes=verbose_shapes)

            # Aggregate global loss and agg_loss across all ranks
            loss_tensor = loss.mean().detach()
            agg_loss_tensor = agg_loss.detach()
            tdist.all_reduce(loss_tensor, op=tdist.ReduceOp.SUM)
            tdist.all_reduce(agg_loss_tensor, op=tdist.ReduceOp.SUM)
            global_loss = loss_tensor.item() / tdist.get_world_size()
            global_agg_loss = agg_loss_tensor.item() / tdist.get_world_size()

            # Backpropagation
            agg_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(posterior_flow.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            step += 1
        
            # Synchronize across ranks before scheduler step
            tdist.barrier()

            # Update scheduler every step_freq steps
            if step % run_args.get("step_freq", 1) == 0:
                scheduler.step()

            if step % 1000 == 0:
                log_usage_metrics(current_pytorch_device, process, step, global_rank)
                # Save rank-specific checkpoint at regular interval
                checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_rank_{global_rank}_{step}.pt"
                save_checkpoint(posterior_flow.module, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler, global_rank=global_rank, additional_state={
                    'rank': global_rank,
                    'world_size': tdist.get_world_size(),
                    'local_loss': loss.mean().detach().item(),
                    'local_agg_loss': agg_loss.detach().item()
                })
                if run_args["log_nominal_area"]:
                    with torch.no_grad():
                        nominal_samples = posterior_flow(nominal_context).sample((5000,)).cpu().numpy()
                    nominal_samples[:, -1] *= 100000
                    with contextlib.redirect_stdout(io.StringIO()):
                        nominal_samples_gd = getdist.mcsamples.MCSamples(samples=nominal_samples, names=experiment.cosmo_params, labels=experiment.latex_labels)
                    local_nominal_area = get_contour_area(nominal_samples_gd, 'Om', 'hrdrag', 0.68)[0]
                    mlflow.log_metric(f"nominal_area_rank_{global_rank}", local_nominal_area, step=step)
                    
                    # Aggregate the nominal areas across all ranks
                    local_nominal_area_tensor = torch.tensor(local_nominal_area, device=current_pytorch_device)
                    tdist.all_reduce(local_nominal_area_tensor, op=tdist.ReduceOp.SUM)
                    global_nominal_area = local_nominal_area_tensor.item() / tdist.get_world_size()

                    if global_rank == 0:
                        mlflow.log_metric("nominal_area", global_nominal_area, step=step)

                        if global_nominal_area < best_nominal_area:
                            # Save checkpoint if the global nominal area is the best so far
                            best_nominal_area = global_nominal_area
                            mlflow.log_metric("best_nominal_area", best_nominal_area, step=step)
                            # Save the best nominal area checkpoint
                            checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_nominal_area_{step}.pt"
                            save_checkpoint(posterior_flow.module, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler, global_rank=global_rank)

                            # Save the last best area checkpoint
                            checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_nominal_area_best.pt"
                            save_checkpoint(posterior_flow.module, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler, global_rank=global_rank)

                # Log the global nominal area on rank 0
                if global_rank == 0:
                    print_training_state(posterior_flow, optimizer, step)
                    if global_loss < best_loss:
                        best_loss = global_loss
                        checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_loss_{step}.pt"
                        save_checkpoint(posterior_flow.module, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler, global_rank=global_rank)
                        checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_loss_best.pt"
                        save_checkpoint(posterior_flow.module, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler, global_rank=global_rank)
                        mlflow.log_metric("best_loss", best_loss, step=step)

            # Update progress bar or print status
            if step % 10 == 0:
                mlflow.log_metric(f"loss_rank_{global_rank}", loss.mean().detach().item(), step=step)
                mlflow.log_metric(f"agg_loss_rank_{global_rank}", agg_loss.detach().item(), step=step)
                if global_rank == 0:
                    mlflow.log_metric("loss", global_loss, step=step)
                    mlflow.log_metric("agg_loss", global_agg_loss, step=step)
                    for param_group in optimizer.param_groups:
                        mlflow.log_metric("lr", param_group['lr'], step=step)
                    if is_tty:
                        pbar.update(10)
                        if run_args["log_nominal_area"] and global_nominal_area is not None:
                            pbar.set_description(f"Loss: {global_loss:.3f}, Area: {global_nominal_area:.3f}")
                        else:
                            pbar.set_description(f"Loss: {global_loss:.3f}")
                    else:
                        if run_args["log_nominal_area"] and global_nominal_area is not None:
                            print(f"Step {step}, Loss: {global_loss:.3f}, Area: {global_nominal_area:.3f}")
                        else:
                            print(f"Step {step}, Loss: {global_loss:.3f}")

            tdist.barrier()

            if step >= total_steps:
                break

    # Ensure progress bar closes cleanly
    if global_rank == 0 and is_tty:
        pbar.close()

    # Save last rank-specific checkpoint for all ranks (default behavior)
    last_checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_rank_{global_rank}_last.pt"
    save_checkpoint(posterior_flow.module, optimizer, last_checkpoint_path, step=run_args.get("total_steps", 0), artifact_path="checkpoints", scheduler=scheduler, global_rank=global_rank, additional_state={
        'rank': global_rank,
        'world_size': tdist.get_world_size(),
        'last_step': step,
        'training_completed': True
    })
    if global_rank == 0:
        print(f"Saved last rank-specific checkpoints for all ranks")

    ########################################## Final Plots ##########################################
    if global_rank == 0:
        plot_training(
            run_id=ml_info.run_id,
            var=None,
            log_scale=True,
            loss_step_freq=10,
            area_step_freq=100,
            show_best=False
        )
        plt.close('all')
        print("Run", ml_info.experiment_id + "/" + ml_info.run_id, "completed.")
    
    # Final memory logging
    if "LOCAL_RANK" in os.environ:
        # Synchronize all ranks before ending MLflow runs
        tdist.barrier()
        # End MLflow run for all ranks
        mlflow.end_run()
        tdist.destroy_process_group()
        if global_rank == 0:
            print("Runtime:", get_runtime(ml_info.run_id))

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
    default_mlflow_exp_name = f"{cosmo_model_default}_{default_args['flow_type']}"

    parser = argparse.ArgumentParser(description="Run Number Tracers Training")

    # Add arguments dynamically based on the default config file
    parser.add_argument('--cosmo_exp', type=str, default='num_tracers', help='Cosmological experiment name')
    parser.add_argument('--mlflow_exp', type=str, default=default_mlflow_exp_name, help='MLflow experiment name')
    parser.add_argument('--cosmo_model', type=str, default=cosmo_model_default, help='Cosmological model set to use from run_args.json')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')

    # Add arguments for resuming from a checkpoint
    parser.add_argument('--resume_id', type=str, default=None, help='MLflow run ID to resume training from (continues existing run with same parameters)')
    parser.add_argument('--resume_step', type=int, default=None, help='Step to resume training from (required when using --resume_id)')
    parser.add_argument('--add_steps', type=int, default=0, help='Number of steps to add to the training (only used with --resume_id)')
    parser.add_argument('--profile', action='store_true', help='Enable profiling for a few steps and then exit.')
    parser.add_argument('--restart_id', type=str, default=None, help='MLflow run ID to restart training from (creates new run with current parameters)')
    parser.add_argument('--restart_step', type=lambda x: int(x) if x.isdigit() else x, default=None, help='Step to restart training from (optional when using --restart_id, defaults to latest). Can be an integer or string like "last"')
    parser.add_argument('--restart_checkpoint', type=str, default=None, help='Specific checkpoint file name to use for all ranks during restart (alternative to --restart_step)')
    parser.add_argument('--restart_optimizer', action='store_true', help='Restart optimizer from checkpoint')
    
    # Note: Rank-specific checkpoint saving/loading is now the default behavior
    # No additional arguments needed for this functionality

    for key, value in default_args.items():
        if value is None:
            parser.add_argument(f'--{key}', type=str, default=None, 
                              help=f'Override {key} (default: None)')
            continue
            
        arg_type = type(value)
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true', help=f'Enable {key}')
            # Set default explicitly for bools, action handles the logic
            parser.set_defaults(**{key: value})
        elif isinstance(value, (int, float, str)):
            parser.add_argument(f'--{key}', type=arg_type, default=None, help=f'Override {key} (default: {value})')
        elif isinstance(value, list):
            # For lists, we'll accept them as JSON strings and parse them
            parser.add_argument(f'--{key}', type=str, default=None, 
                              help=f'Override {key} as JSON string (default: {value})')
        else:
            print(f"Warning: Argument type for key '{key}' not explicitly handled ({arg_type}). Treating as string.")
            parser.add_argument(f'--{key}', type=str, default=None, help=f'Override {key} (default: {value})')

    args = parser.parse_args()

    # Validate checkpoint loading arguments
    if args.resume_id and args.restart_id:
        raise ValueError("Cannot use --resume_id with --restart_id. Choose one:\n"
                        "  --resume_id: Continue existing run with same parameters\n"
                        "  --restart_id: Start new run with potentially different parameters")
    
    if args.resume_id and args.resume_step is None:
        raise ValueError("--resume_step is required when using --resume_id")
    
    if args.resume_step is not None and args.resume_id is None:
        raise ValueError("--resume_step can only be used with --resume_id")
    
    if args.restart_step is not None and args.restart_id is None:
        raise ValueError("--restart_step can only be used with --restart_id")
    
    if args.restart_checkpoint is not None and args.restart_id is None:
        raise ValueError("--restart_checkpoint can only be used with --restart_id")
    
    if args.restart_checkpoint is not None and args.restart_step is not None:
        raise ValueError("Cannot use --restart_checkpoint with --restart_step. Choose one:\n"
                        "  --restart_step: Find checkpoint by step number in MLflow run\n"
                        "  --restart_checkpoint: Use specific checkpoint file path")
    
    if args.add_steps > 0 and args.resume_id is None:
        raise ValueError("--add_steps can only be used with --resume_id")

    # --- Prepare Final Config --- 
    run_args = run_args_dict[args.cosmo_model].copy() # Start with defaults for the chosen model

    # Override defaults with any provided command-line arguments
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key not in ['cosmo_model', 'resume_id', 'resume_step', 'add_steps', 'profile', 'checkpoint_path'] and value is not None and key in run_args:
            # Handle the case where the original value was None
            if run_args[key] is None:
                # Try to parse as JSON first (for lists), then as other types
                try:
                    parsed_value = json.loads(value)
                    if os.environ.get('RANK') == 0:
                        print(f"Setting '{key}' from None to: {parsed_value}")
                    run_args[key] = parsed_value
                except json.JSONDecodeError:
                    # If not JSON, try to parse as other types
                    try:
                        # Try to parse as float first, then int, then keep as string
                        if '.' in value:
                            parsed_value = float(value)
                        else:
                            parsed_value = int(value)
                        if os.environ.get('RANK') == 0:
                            print(f"Setting '{key}' from None to: {parsed_value}")
                        run_args[key] = parsed_value
                    except ValueError:
                        # Keep as string
                        if os.environ.get('RANK') == 0:
                            print(f"Setting '{key}' from None to: {value}")
                        run_args[key] = value
            elif isinstance(run_args[key], bool) and isinstance(value, bool):
                run_args[key] = value
            elif isinstance(run_args[key], list):
                # Parse JSON string back to list
                try:
                    parsed_value = json.loads(value)
                    if os.environ.get('RANK') == 0:
                        print(f"Overriding '{key}': {run_args[key]} -> {parsed_value}")
                    run_args[key] = parsed_value
                except json.JSONDecodeError as e:
                    if os.environ.get('RANK') == 0:
                        print(f"Warning: Could not parse '{key}' as JSON: {e}. Keeping default value.")
            elif not isinstance(run_args[key], bool):
                if os.environ.get('RANK') == 0:
                    print(f"Overriding '{key}': {run_args[key]} -> {value}")
                run_args[key] = value
    
    # --- Setup & Run --- 
    # Determine device for the main script part, before DDP spawns processes
    # This initial `device_check` is for the process launching srun, not for the DDP ranks themselves.
    initial_device_check_cuda = False
    if torch.cuda.is_available():
        try:
            # Check if the device specified in args is valid, otherwise default to cuda:0
            # The `args.device` is like "cuda:0"
            if args.device.startswith("cuda:") and ":" in args.device.split(":"):
                parsed_id = int(args.device.split(':')[1])
                if 0 <= parsed_id < torch.cuda.device_count():
                    torch.device(args.device) # Try to create device object
                    initial_device_check_cuda = True
                    script_level_device_str = args.device
                else: # Invalid specific cuda device
                    script_level_device_str = "cuda:0" # Default to cuda:0
                    torch.device(script_level_device_str)
                    initial_device_check_cuda = True
            else: # Non-specific "cuda" or other
                 script_level_device_str = "cuda:0" # Default to cuda:0
                 torch.device(script_level_device_str)
                 initial_device_check_cuda = True
        except Exception as e: # Fallback to CPU if any error with CUDA init
            print(f"Initial CUDA check failed: {e}. Defaulting to CPU for main script context.")
            initial_device_check_cuda = False
            script_level_device_str = "cpu"
    else:
        script_level_device_str = "cpu"

    if args.resume_id:
        kwargs = {
            "resume_id": args.resume_id,
            "resume_step": args.resume_step,
            "add_steps": args.add_steps
        }
    elif args.restart_id:
        kwargs = {
            "restart_id": args.restart_id,
            "restart_step": args.restart_step,
            "restart_checkpoint": args.restart_checkpoint,
            "restart_optimizer": args.restart_optimizer
        }
    else:
        kwargs = {}

    single_run(
        cosmo_exp=args.cosmo_exp,
        cosmo_model=args.cosmo_model,
        run_args=run_args,
        mlflow_exp=args.mlflow_exp,
        device=script_level_device_str,
        profile=args.profile,
        **kwargs
    )