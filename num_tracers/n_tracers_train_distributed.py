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
from bed.grid import Grid

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
from plotting import get_contour_area, plot_training, eig_steps
import getdist.mcsamples

class FlowLikelihoodDataset(Dataset):
    def __init__(self, num_tracers, designs, n_particles_per_device, observation_labels, target_labels, device="cuda"):
        self.num_tracers = num_tracers
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

        # Generate samples from the NumTracers pyro model
        with torch.no_grad():
            # Generate the samples directly on the GPU
            trace = poutine.trace(self.num_tracers.pyro_model).get_trace(expanded_design)
            
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
def get_dataloader(num_tracers, designs, n_particles_per_device, observation_labels, target_labels, batch_size, pytorch_device_idx_for_ddp, num_workers=0, world_size=None):
    # Create dataset with designs on GPU
    dataset = FlowLikelihoodDataset(
        num_tracers=num_tracers,
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
    cosmo_model,
    run_args,
    mlflow_experiment_name,
    device="cuda:0",
    resume_id=None,
    resume_step=None,
    add_steps=0,
    profile=False,
    restart_id=None,
    restart_step=None,
    restart_checkpoint=None,
    **kwargs,
):

    global_rank, local_rank, effective_device_id, pytorch_device_idx = init_training_env(tdist, device)

    # Clear GPU cache
    torch.cuda.empty_cache()
    process = psutil.Process()
    # Trigger garbage collection for CPU
    gc.collect()
    pyro.clear_param_store()
    storage_path = os.environ["SCRATCH"] + "/bed/BED_cosmo/num_tracers"
    home_dir = os.environ["HOME"]
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    
    # Initialize MLflow experiment and run info on rank 0
    current_pytorch_device = f"cuda:{pytorch_device_idx}" if "LOCAL_RANK" in os.environ and torch.cuda.is_available() else (f"cuda:{effective_device_id}" if effective_device_id != -1 else "cpu")

    ml_info, run_args, checkpoint, start_step, best_loss, best_nominal_area = init_run(
        tdist, 
        global_rank, 
        current_pytorch_device, 
        storage_path, 
        mlflow_experiment_name, 
        cosmo_model, 
        run_args, 
        kwargs, 
        resume_id=resume_id, 
        resume_step=resume_step, 
        add_steps=add_steps,
        restart_id=restart_id,
        restart_step=restart_step,
        restart_checkpoint=restart_checkpoint
        )

    desi_df = pd.read_csv(home_dir + run_args["data_path"] + 'desi_data.csv')
    desi_tracers = pd.read_csv(home_dir + run_args["data_path"] + 'desi_tracers.csv')
    nominal_cov = np.load(home_dir + run_args["data_path"] + 'desi_cov.npy')
    
    ############################################### Priors ###############################################

    total_observations = 6565626
    #classes = kwargs['classes']
    classes = (desi_tracers.groupby('class').sum()['targets'].reindex(["LRG", "ELG", "QSO"]) / total_observations).to_dict()
    # enforce lows:
    classes["LRG"] = (0.0, classes["LRG"])
    classes["ELG"] = (0.0, classes["ELG"])
    classes["QSO"] = (0.0, classes["QSO"])
    mlflow.log_dict(classes, "classes.json")
    # Get nominal design from observed tracer counts
    nominal_design = torch.tensor(desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=current_pytorch_device)

    num_tracers = NumTracers(
        desi_df,
        desi_tracers,
        cosmo_model,
        nominal_cov,
        rank=global_rank,
        include_D_M=run_args["include_D_M"], 
        device=current_pytorch_device,
        verbose=True
        )
    
    target_labels = num_tracers.cosmo_params

    ############################################### Designs ###############################################
    # if fixed design:
    if run_args["nominal_design"]:   
        # Create grid with nominal design values
        grid_designs = Grid(
            N_LRG=nominal_design[0].cpu().numpy(), 
            N_ELG=nominal_design[1].cpu().numpy(), 
            N_QSO=nominal_design[2].cpu().numpy()
        )

        # Convert grid to tensor format
        designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=current_pytorch_device).unsqueeze(0)
        for name in grid_designs.names[1:]:
            design_tensor = torch.tensor(getattr(grid_designs, name).squeeze(), device=current_pytorch_device).unsqueeze(0)
            designs = torch.cat((designs, design_tensor), dim=0)
        designs = designs.unsqueeze(0)

    else:
        # Create design grid with specified step size
        designs_dict = {
            f'N_{class_name}': np.arange(
                max(class_frac[0], run_args["design_low"]),
                min(class_frac[1] + run_args["design_step"], 1.0), 
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
        designs = torch.tensor(getattr(grid_designs, grid_designs.names[0]).squeeze(), device=current_pytorch_device).unsqueeze(1)
        for name in grid_designs.names[1:]:
            design_tensor = torch.tensor(getattr(grid_designs, name).squeeze(), device=current_pytorch_device).unsqueeze(1)
            designs = torch.cat((designs, design_tensor), dim=1)

    # Only create prior plot if not resuming and on rank 0
    if not resume_id and global_rank == 0: # global_rank check
        fig, axs = plt.subplots(ncols=len(target_labels), nrows=1, figsize=(5*len(target_labels), 5))
        if len(target_labels) == 1:
            axs = [axs]
        for i, p in enumerate(target_labels):
            support = num_tracers.priors[p].support
            eval_pts = torch.linspace(support.lower_bound, support.upper_bound, 200, device=current_pytorch_device)
            prob = torch.exp(num_tracers.priors[p].log_prob(eval_pts))[:-1]
            prob_norm = prob/torch.sum(prob)
            axs[i].plot(eval_pts.cpu().numpy()[:-1], prob_norm.cpu().numpy(), label="Prior", color="tab:blue", alpha=0.5)
            axs[i].set_title(p)
        plt.tight_layout()
        plt.savefig(f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/plots/prior.png")

    input_dim = len(target_labels)
    context_dim = len(classes.keys()) + 10 if run_args["include_D_M"] else len(classes.keys()) + 5
    if global_rank == 0:
        print("MLFlow Run Info:", ml_info.experiment_id + "/" + ml_info.run_id)
        print(f"Using {run_args['n_devices']} devices with {run_args['n_particles']} total particles.")
        print("Designs shape:", designs.shape)
        print("Calculating normalizing flow EIG...")
        print(f'Input dim: {input_dim}, Context dim: {context_dim}')
        print(f"Classes: {classes}\n"
            f"Cosmology: {cosmo_model}\n"
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
    
    central_vals = num_tracers.central_val if run_args["include_D_M"] else num_tracers.central_val[1::2]
    nominal_context = torch.cat([nominal_design, central_vals], dim=-1)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(posterior_flow.parameters(), lr=run_args["initial_lr"])
    
    if resume_id:
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
                print(f"  Manually stepping scheduler to step {resume_step}")
                step_freq = run_args.get("step_freq", 1)
                scheduler_steps_needed = resume_step // step_freq
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

    elif restart_id:
        # Load model weights and optimizer state
        posterior_flow.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        # Update the learning rate in the loaded optimizer state to match current run_args
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = run_args['initial_lr']
            if global_rank == 0:
                print(f"Updated optimizer learning rate: {old_lr} -> {run_args['initial_lr']}")
        
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
        num_tracers=num_tracers,
        designs=designs.cpu(),
        n_particles_per_device=n_particles_per_device,
        observation_labels=["y"],
        target_labels=target_labels,
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
            global_loss_tensor = loss.mean().detach()
            global_agg_loss_tensor = agg_loss.detach()
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
            step += 1
        
            # Synchronize across ranks before scheduler step
            tdist.barrier()

            # Update scheduler every step_freq steps
            if step % run_args.get("step_freq", 1) == 0:
                scheduler.step()

            if step % 1000 == 0 and step != total_steps:
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
                    # Capture getdist output only on rank 0 to avoid clutter, ensure getdist is available
                    if global_rank == 0: # Check global_rank
                        try:
                            with contextlib.redirect_stdout(io.StringIO()):
                                nominal_samples_gd = getdist.mcsamples.MCSamples(samples=nominal_samples, names=target_labels, labels=num_tracers.latex_labels)
                            local_nominal_area = get_contour_area(nominal_samples_gd, 'Om', 'hrdrag', 0.68)[0]
                        except Exception as e:
                            print(f"Rank 0: Error during getdist processing: {e}")
                            local_nominal_area = float('nan')
                    else: # Other ranks don't compute area directly but need a value for aggregation
                        local_nominal_area = 0.0 # Or float('nan') if that's better for averaging

                    # Aggregate the nominal areas across all ranks
                    nominal_area_tensor = torch.tensor(local_nominal_area, device=current_pytorch_device)
                    tdist.all_reduce(nominal_area_tensor, op=tdist.ReduceOp.SUM)
                    # For area, if only rank 0 computes it, broadcasting might be better than summing zeros.
                    # Or ensure all ranks compute it if data is available, or rank 0 broadcasts its result.
                    # Assuming rank 0 computes and then we want that value:
                    if global_rank == 0:
                        global_nominal_area_val = local_nominal_area # Rank 0 has the actual value
                    else:
                        global_nominal_area_val = nominal_area_tensor.item() # Should be 0 if others sent 0

                    # Broadcast the actual area from rank 0 to all other ranks
                    area_to_broadcast = torch.tensor([global_nominal_area_val if global_rank == 0 else 0.0], device=current_pytorch_device)
                    tdist.broadcast(area_to_broadcast, src=0)
                    global_nominal_area = area_to_broadcast.item()
                    if global_rank == 0:
                        mlflow.log_metric("nominal_area", global_nominal_area, step=step)

                        if global_nominal_area < best_nominal_area:
                            # Save checkpoint if the global nominal area is the best so far
                            best_nominal_area = global_nominal_area
                            mlflow.log_metric("best_nominal_area", best_nominal_area, step=step)
                            # Save the best nominal area checkpoint
                            checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_nominal_area_{step}.pt"
                            save_checkpoint(posterior_flow.module, optimizer, checkpoint_path, step=step, artifact_path="checkpoints", scheduler=scheduler, global_rank=global_rank)

                            # Save the final best area checkpoint
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
            if global_rank == 0 and step % 10 == 0:
                mlflow.log_metric(f"loss_rank_{global_rank}", loss.mean().detach().item(), step=step)
                mlflow.log_metric(f"agg_loss_rank_{global_rank}", agg_loss.detach().item(), step=step)
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

    # Save final rank-specific checkpoint for all ranks (default behavior)
    final_checkpoint_path = f"{storage_path}/mlruns/{ml_info.experiment_id}/{ml_info.run_id}/artifacts/checkpoints/checkpoint_rank_{global_rank}_final.pt"
    save_checkpoint(posterior_flow.module, optimizer, final_checkpoint_path, step=run_args.get("total_steps", 0), artifact_path="checkpoints", scheduler=scheduler, global_rank=global_rank, additional_state={
        'rank': global_rank,
        'world_size': tdist.get_world_size(),
        'final_step': step,
        'training_completed': True
    })
    if global_rank == 0:
        print(f"Saved final rank-specific checkpoints for all ranks")

    ########################################## Final Evaluation ##########################################
    if global_rank == 0:
        if run_args["log_nominal_area"]:
            with torch.no_grad():
                nominal_samples = posterior_flow(nominal_context).sample((5000,)).cpu().numpy()
            nominal_samples[:, -1] *= 100000
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    nominal_samples_gd = getdist.mcsamples.MCSamples(samples=nominal_samples, names=target_labels, labels=num_tracers.latex_labels)
                final_nominal_area = get_contour_area(nominal_samples_gd, 'Om', 'hrdrag', 0.68)[0]
            except Exception as e:
                print(f"Rank 0: Error during getdist processing: {e}")
                final_nominal_area = float('nan')
            mlflow.log_metric("nominal_area", final_nominal_area, step=step)
        plot_training(
            run_id=ml_info.run_id,
            var=None,
            log_scale=True,
            loss_step_freq=10,
            area_step_freq=100,
            show_best=False
        )
        if not run_args["nominal_design"]:
            eval_args = {"n_samples": 3000, "device": current_pytorch_device, "eval_seed": 1}
            eig_steps(
                run_id=ml_info.run_id,
                steps=[500, 1000, 'final'],
                eval_args=eval_args,
                cosmo_exp='num_tracers'
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
    default_exp_name = f"{cosmo_model_default}_{default_args['flow_type']}"

    parser = argparse.ArgumentParser(description="Run Number Tracers Training")

    # Add arguments dynamically based on the default config file
    parser.add_argument('--cosmo_model', type=str, default=cosmo_model_default, help='Cosmological model set to use from run_args.json')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--exp_name', type=str, default=default_exp_name, help='Experiment name')

    # Add arguments for resuming from a checkpoint
    parser.add_argument('--resume_id', type=str, default=None, help='MLflow run ID to resume training from (continues existing run with same parameters)')
    parser.add_argument('--resume_step', type=int, default=None, help='Step to resume training from (required when using --resume_id)')
    parser.add_argument('--add_steps', type=int, default=0, help='Number of steps to add to the training (only used with --resume_id)')
    parser.add_argument('--profile', action='store_true', help='Enable profiling for a few steps and then exit.')
    parser.add_argument('--restart_id', type=str, default=None, help='MLflow run ID to restart training from (creates new run with current parameters)')
    parser.add_argument('--restart_step', type=lambda x: int(x) if x.isdigit() else x, default=None, help='Step to restart training from (optional when using --restart_id, defaults to latest). Can be an integer or string like "final"')
    parser.add_argument('--restart_checkpoint', type=str, default=None, help='Specific checkpoint file name to use for all ranks during restart (alternative to --restart_step)')
    
    # Note: Rank-specific checkpoint saving/loading is now the default behavior
    # No additional arguments needed for this functionality

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
    run_args = run_args_dict[cosmo_model].copy() # Start with defaults for the chosen model

    # Override defaults with any provided command-line arguments
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key not in ['cosmo_model', 'resume_id', 'resume_step', 'add_steps', 'profile', 'checkpoint_path'] and value is not None and key in run_args:
            if isinstance(run_args[key], bool) and isinstance(value, bool):
                run_args[key] = value
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

    single_run(
        cosmo_model=cosmo_model,
        run_args=run_args,
        mlflow_experiment_name=args.exp_name,
        device=script_level_device_str, # Pass the determined device string
        resume_id=args.resume_id,
        resume_step=args.resume_step,
        add_steps=args.add_steps,
        profile=args.profile,
        restart_id=args.restart_id,
        restart_step=args.restart_step,
        restart_checkpoint=args.restart_checkpoint
    )