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
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.stats import norm
from urllib.request import urlopen

import pyro
from pyro import poutine
from pyro_oed_src import nf_loss, LikelihoodDataset
from pyro.contrib.util import lexpand, rexpand

import matplotlib.pyplot as plt

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from tqdm import tqdm

import shutil
import psutil
import os
import gc
from util import *
import json
import yaml
import argparse
import traceback
from plotting import get_contour_area, plot_training, save_figure, plot_posterior
import getdist.mcsamples

class Trainer:
    def __init__(self, cosmo_exp, mlflow_exp, run_args, device=None, profile=False, verbose=False):
        self.cosmo_exp = cosmo_exp
        self.mlflow_exp = mlflow_exp
        self.run_args = run_args
        self.profile = profile
        self.storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
        self.is_tty = sys.stdout.isatty()
        self.verbose = verbose
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        self.process = psutil.Process()
        # Trigger garbage collection for CPU
        gc.collect()
        pyro.clear_param_store()
        mlflow.set_tracking_uri(self.storage_path + "/mlruns")
        
        # Create MLflow client AFTER setting the tracking URI
        self.client = MlflowClient()

        self._init_device_settings(device)
        self._init_run()
        self.experiment = init_experiment(self.run_obj, self.run_args, checkpoint=self.checkpoint, device=self.device, global_rank=self.global_rank)

        self.posterior_flow = init_nf(
            self.run_args,
            len(self.experiment.cosmo_params),
            self.experiment.context_dim,
            device=self.device,
            seed=self.run_args["nf_seed"]
        )
        if self.global_rank == 0:
            print("MLFlow Run Info:", self.run_obj.info.experiment_id + "/" + self.run_obj.info.run_id)
            print(f"Using {self.run_args['n_devices']} devices with {self.run_args['n_particles']} total particles.")
            print("Designs shape:", self.experiment.designs.shape)
            print(f'Input dim: {len(self.experiment.cosmo_params)}, Context dim: {self.experiment.context_dim}')
            print(f"Cosmology: {self.run_args['cosmo_model']}")
            print(f"Target labels: {self.experiment.cosmo_params}")
            print("Flow model initialized: \n", self.posterior_flow)
            np.save(f"{self.run_path}/artifacts/designs.npy", self.experiment.designs.cpu().detach().numpy())
            mlflow.log_artifact(f"{self.run_path}/artifacts/designs.npy")

        # For DDP, ensure device_ids and output_device use the pytorch_device_idx (which is 0)
        if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
            ddp_device_spec = [self.pytorch_device_idx] 
            ddp_output_device_spec = self.pytorch_device_idx
        else:
            ddp_device_spec = [self.effective_device_id] if self.effective_device_id != -1 else None
            ddp_output_device_spec = self.effective_device_id if self.effective_device_id != -1 else None

        # Check if model is already on a CUDA device if CUDA is available
        if torch.cuda.is_available():
            model_device = next(self.posterior_flow.parameters()).device
            if model_device.type == 'cpu' and ddp_device_spec is not None and ddp_device_spec[0] is not None and isinstance(ddp_device_spec[0], int) :
                self.posterior_flow.to(f"cuda:{ddp_device_spec[0]}")
        
        self.posterior_flow = DDP(self.posterior_flow, device_ids=ddp_device_spec, output_device=ddp_output_device_spec, find_unused_parameters=False)

        # Load model checkpoint before creating optimizer (for resume)
        if self.run_args.get("resume_id", None):
            self.posterior_flow.module.load_state_dict(self.checkpoint['model_state_dict'], strict=True)
            self.posterior_flow.train()

        self._init_optimizer()
        self._init_scheduler()
        self._init_dataloader()
        
        # Only plot initial samples for fresh runs
        if self.checkpoint is None:
            with torch.no_grad():
                samples = self.posterior_flow.module(self.experiment.nominal_context).sample((1000,)).cpu().numpy()
                plt.figure()
                plt.plot(samples.squeeze()[:, 0], samples.squeeze()[:, 1], 'o', alpha=0.5)
                plt.savefig(f"{self.run_path}/artifacts/plots/rank_{self.global_rank}/init_samples.png")

    @profile_method
    def run(self):

        step = self.start_step
        # Initialize pbar as None by default
        if self.is_tty and self.global_rank == 0 and not self.profile:
            pbar = tqdm(total=self.run_args["total_steps"] - self.start_step, desc="Training Progress", position=0, leave=True)

        # Initialize session start time for runtime tracking
        self._update_cumulative_runtime(step)
        
        # Restore RNG state on resume
        if self.checkpoint is not None:
            self._restore_state()

        while step < self.run_args["total_steps"]:
            for samples, context in self.dataloader:         
                self.optimizer.zero_grad()

                if step > 0:
                    self.verbose = False  
                # Compute the loss using nf_loss
                agg_loss, loss = nf_loss(
                    samples, context, self.posterior_flow, self.experiment, 
                    rank=self.global_rank, verbose_shapes=self.verbose, evaluation=False
                    )
                # Aggregate global loss and agg_loss across all ranks
                loss_tensor = loss.mean().detach()
                agg_loss_tensor = agg_loss.detach()
                tdist.all_reduce(loss_tensor, op=tdist.ReduceOp.SUM)
                tdist.all_reduce(agg_loss_tensor, op=tdist.ReduceOp.SUM)
                global_loss = loss_tensor.item() / tdist.get_world_size()
                global_agg_loss = agg_loss_tensor.item() / tdist.get_world_size()

                # Backpropagation
                agg_loss.backward()

                # Optional gradient clipping for stability
                grad_clip = self.run_args.get("grad_clip", 0.0)
                if grad_clip > 0:
                    print(f"Clipping gradients to {grad_clip}")
                    torch.nn.utils.clip_grad_norm_(self.posterior_flow.parameters(), max_norm=float(grad_clip))

                # Optimizer step
                self.optimizer.step()
                step += 1
            
                # Synchronize across ranks before scheduler step
                tdist.barrier()

                # Update scheduler every step_freq steps
                if step % self.run_args.get("step_freq", 1) == 0:
                    self.scheduler.step()

                if step % self.run_args.get("checkpoint_step_freq", 1000) == 0:
                    # Update cumulative runtime at each checkpoint
                    self._update_cumulative_runtime(step)
                    
                    if self.run_args.get("log_usage", False):
                        log_usage_metrics(self.device, self.process, step, self.global_rank)
                    
                    # Perform all RNG operations (plotting, etc.) BEFORE saving checkpoint
                    # This ensures the saved RNG state is ready for the next training step
                    if self.run_args.get("log_nominal_area", False):
                        plot_samples, plot_colors, plot_labels, plot_scatter = [], [], [], []
                        nf_samples = self.experiment.get_guide_samples(self.posterior_flow, self.experiment.nominal_context, num_samples=10000)
                        plot_samples.append(nf_samples)
                        plot_colors.append('tab:blue')
                        plot_labels.append('NF')
                        plot_scatter.append(True)
                        try:
                            nominal_samples = self.experiment.get_nominal_samples(num_samples=10000, params=self.experiment.cosmo_params, transform_output=False)
                            plot_samples.append(nominal_samples)
                            plot_colors.append('black')
                            plot_labels.append('MCMC')
                            plot_scatter.append(False)
                        except NotImplementedError:
                            pass
                        ranges = {param: (self.experiment.prior_data['parameters'][param]['plot']['lower'], self.experiment.prior_data['parameters'][param]['plot']['upper']) for param in self.experiment.cosmo_params}
                        plt.figure()
                        plot_posterior(
                            plot_samples, 
                            plot_colors, 
                            legend_labels=plot_labels, 
                            levels=[0.68], 
                            width_inch=12, 
                            show_scatter=plot_scatter,
                            ranges=ranges
                            )
                        plt.savefig(f"{self.run_path}/artifacts/plots/rank_{self.global_rank}/posterior/{step}.png", dpi=400)
                        plt.close('all')
                        local_nominal_areas = get_contour_area(nf_samples, 0.68, *self.experiment.cosmo_params, global_rank=self.global_rank, design_type='nominal')[0]
                        
                        # Log metrics per rank with tags instead of rank labels
                        mlflow.log_metrics(local_nominal_areas, step=step)
                        
                        # Aggregate the nominal areas across all ranks
                        # Sort keys consistently across ranks to ensure proper aggregation
                        sorted_keys = sorted(local_nominal_areas.keys())
                        local_nominal_areas_tensor = torch.tensor([local_nominal_areas[key] for key in sorted_keys], device=self.device)
                        
                        tdist.all_reduce(local_nominal_areas_tensor, op=tdist.ReduceOp.SUM)
                        global_nominal_areas_tensor = local_nominal_areas_tensor / tdist.get_world_size()
                        
                        # Log each individual parameter pair area separately
                        if self.global_rank == 0:
                            for i, pair_key in enumerate(sorted_keys):
                                pair_name = pair_key.replace('nominal_area_0_', '')  # Remove rank prefix for metric name
                                global_value = global_nominal_areas_tensor[i].item()
                                mlflow.log_metric(f"nominal_area_avg_{pair_name}", global_value, step=step)
                                if global_value < self.best_nominal_area:
                                    # Save checkpoint if the mean area is the best so far
                                    self.best_nominal_area = global_value
                                    mlflow.log_metric("best_avg_nominal_area", self.best_nominal_area, step=step)
                                    mlflow.log_metric(f"best_nominal_area_{pair_name}", self.best_nominal_area, step=step)
                                    # Save the best nominal area checkpoint
                                    checkpoint_path = f"{self.run_path}/artifacts/checkpoints/checkpoint_nominal_area_{step}.pt"
                                    self.save_checkpoint(checkpoint_path, step=step, artifact_path="checkpoints", scheduler=self.scheduler, global_rank=self.global_rank)

                                    # Save the last best area checkpoint
                                    checkpoint_path = f"{self.run_path}/artifacts/checkpoints/checkpoint_nominal_area_best.pt"
                                    self.save_checkpoint(checkpoint_path, step=step, artifact_path="checkpoints", scheduler=self.scheduler, global_rank=self.global_rank)

                    # Save rank-specific checkpoint after all RNG operations complete
                    self.save_checkpoint(
                        f"{self.run_path}/artifacts/checkpoints/checkpoint_rank_{self.global_rank}_{step}.pt", 
                        step=step, 
                        artifact_path="checkpoints", 
                        scheduler=self.scheduler, 
                        global_rank=self.global_rank, 
                        additional_state={
                        'rank': self.global_rank,
                        'world_size': tdist.get_world_size(),
                        'local_loss': loss.mean().detach().item(),
                        'local_agg_loss': agg_loss.detach().item()
                        }
                    )
                    
                    # Log the global nominal area on rank 0
                    if self.global_rank == 0:
                        if self.verbose:
                            print_training_state(self.posterior_flow, self.optimizer, step)
                        if global_loss < self.best_loss:
                            self.best_loss = global_loss
                            checkpoint_path = f"{self.run_path}/artifacts/checkpoints/checkpoint_loss_{step}.pt"
                            self.save_checkpoint(checkpoint_path, step=step, artifact_path="checkpoints", scheduler=self.scheduler, global_rank=self.global_rank)
                            checkpoint_path = f"{self.run_path}/artifacts/checkpoints/checkpoint_loss_best.pt"
                            self.save_checkpoint(checkpoint_path, step=step, artifact_path="checkpoints", scheduler=self.scheduler, global_rank=self.global_rank)
                            mlflow.log_metric("best_loss", self.best_loss, step=step)

                # Update progress bar or print status
                if step % 10 == 0:
                    mlflow.log_metric(f"loss_rank_{self.global_rank}", loss.mean().detach().item(), step=step)
                    mlflow.log_metric(f"agg_loss_rank_{self.global_rank}", agg_loss.detach().item(), step=step)
                    if self.global_rank == 0:
                        mlflow.log_metric("loss", global_loss, step=step)
                        mlflow.log_metric("agg_loss", global_agg_loss, step=step)
                        for param_group in self.optimizer.param_groups:
                            mlflow.log_metric("lr", param_group['lr'], step=step)
                        if self.is_tty and not self.profile:
                            pbar.update(10)
                            pbar.set_description(f"Loss: {global_loss:.3f}")
                        else:
                            print(f"Step {step}, Loss: {global_loss:.3f}")

                tdist.barrier()

                if step >= self.run_args["total_steps"]:
                    break

        # Ensure progress bar closes cleanly
        if self.global_rank == 0 and self.is_tty and not self.profile:
            pbar.close()

        # Save last rank-specific checkpoint for all ranks (default behavior)
        last_checkpoint_path = f"{self.run_path}/artifacts/checkpoints/checkpoint_rank_{self.global_rank}_last.pt"
        self.save_checkpoint(last_checkpoint_path, step=self.run_args.get("total_steps", 0), artifact_path="checkpoints", scheduler=self.scheduler, global_rank=self.global_rank, additional_state={
            'rank': self.global_rank,
            'world_size': tdist.get_world_size(),
            'last_step': step,
            'training_completed': True
        })
        if self.global_rank == 0:
            print(f"Saved last rank-specific checkpoints for all ranks")

        # Synchronize all ranks before expensive plotting
        if "LOCAL_RANK" in os.environ:
            tdist.barrier()
        
        if self.global_rank == 0:
            plot_training(
                run_id=self.run_obj.info.run_id,
                cosmo_exp=self.cosmo_exp,
                var=None,
                log_scale=True,
                loss_step_freq=10,
                area_step_freq=100,
                area_limits=[0.5, 2.0]
            )
            plt.close('all')
        
        # Create GIF for all ranks (only if we logged posterior plots)
        if self.run_args.get("log_nominal_area", False):
            create_gif(self.run_obj.info.run_id, fps=5, add_labels=True, label_position='top-right', text_size=1.0, pause_last_frame=3.0, rank=self.global_rank)
        
        if self.global_rank == 0:
            print("Run", self.run_obj.info.experiment_id + "/" + self.run_obj.info.run_id, "completed.")
        
        # Final cumulative runtime update
        final_step = self.run_args.get("total_steps", step)
        self._update_cumulative_runtime(final_step)
        
        # Final memory logging
        if "LOCAL_RANK" in os.environ:
            # Synchronize all ranks before ending MLflow runs
            tdist.barrier()
            # End MLflow run for all ranks
            mlflow.end_run()
            tdist.destroy_process_group()
            if self.global_rank == 0:
                runtime = get_runtime(self.run_obj.info.run_id)
                hours = int(runtime.total_seconds() // 3600)
                minutes = int((runtime.total_seconds() % 3600) // 60)
                seconds = int(runtime.total_seconds() % 60)
                print(f"Total active training time: {hours}h {minutes}m {seconds}s")
            
    @profile_method
    def save_checkpoint(self, filepath, step=None, artifact_path=None, scheduler=None, additional_state=None, global_rank=None):
        """
        Saves the training checkpoint with comprehensive state information.

        Args:
            filepath (str): Path to save the checkpoint.
            step (int, optional): Current training step.
            artifact_path (str, optional): Path to log the artifact to in MLflow.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            additional_state (dict, optional): Additional state to save (e.g., training metrics, configuration).
            global_rank (int, optional): Global rank in distributed training.
        """
        model_state_dict = self.posterior_flow.module.state_dict()

        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_type': self.optimizer.__class__.__name__
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if step is not None:
            checkpoint['step'] = step

        # Save RNG states correctly
        checkpoint['rng_state'] = get_rng_state()
        
        # Also save Pyro's param store state (in case there are any parameters)
        checkpoint['rng_state']['pyro_param_state'] = pyro.get_param_store().get_state()

        if hasattr(self, 'experiment') and hasattr(self.experiment, 'param_bijector'):
            try:
                checkpoint['bijector_state'] = self.experiment.param_bijector.get_state()
            except Exception as exc:
                if self.global_rank == 0:
                    print(f'Warning: Failed to serialize bijector state: {exc}')

        # Always include global rank information
        if global_rank is not None:
            if additional_state is None:
                additional_state = {}
            additional_state['global_rank'] = global_rank
        
        # Include additional state if provided
        if additional_state is not None:
            checkpoint['additional_state'] = additional_state
        
        # Log the checkpoint to mlflow
        torch.save(checkpoint, filepath)
        if artifact_path:
            mlflow.log_artifact(filepath, artifact_path)

    @profile_method
    def _init_device_settings(self, device):
        # Set PyTorch CPU threads at the beginning of DDP environment initialization.
        # This helps manage CPU resources used by PyTorch's own operations on CPU.
        # It's separate from OMP_NUM_THREADS which affects OpenMP-enabled libraries like NumPy/SciPy.
        try:
            # Logic based on OMP_NUM_THREADS or a default, with a cap.
            omp_threads_str = os.environ.get("OMP_NUM_THREADS")
            num_pytorch_cpu_threads = 0 # Initialize
            if omp_threads_str:
                num_pytorch_cpu_threads = int(omp_threads_str)
                # Cap PyTorch threads if OMP_NUM_THREADS is very large (e.g. > 8)
                # Adjust this cap as needed. If OMP_NUM_THREADS is already a per-process target for all libs, this might not be needed.
                if num_pytorch_cpu_threads > 32: 
                    num_pytorch_cpu_threads = 32
                elif num_pytorch_cpu_threads <= 0:
                    num_pytorch_cpu_threads = 1 # Must be at least 1
            else:
                num_pytorch_cpu_threads = 4 # Default if OMP_NUM_THREADS is not set
            
            torch.set_num_threads(num_pytorch_cpu_threads)
            # Logging will be done after rank is determined.
        except Exception as e:
            # Using a generic print here as rank might not be available yet for prefixed logging.
            print(f"Warning: Could not set PyTorch CPU threads during init_training_env: {e}")

        # DDP initialization
        if device is None:
            if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
                local_rank = int(os.environ["LOCAL_RANK"]) # SLURM's local rank
                self.global_rank = int(os.environ["RANK"])
                
                # When CUDA_VISIBLE_DEVICES isolates one GPU, PyTorch sees it as device 0.
                self.pytorch_device_idx = int(os.environ["LOCAL_RANK"])  # The only GPU visible to this process
                self.effective_device_id = self.pytorch_device_idx
                
                # Initialize CUDA context before DDP setup
                torch.cuda.init()
                torch.cuda.set_device(self.pytorch_device_idx)
                
                # Ensure CUDA is available and device is set before DDP init
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available but LOCAL_RANK is set")
                
                # Create torch.device object for device_id
                device_obj = torch.device(f"cuda:{self.pytorch_device_idx}")
                
                # Initialize process group with explicit device ID
                tdist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    world_size=int(os.environ["WORLD_SIZE"]),
                    rank=self.global_rank,
                    timeout=timedelta(seconds=360),  # Increased timeout
                    device_id=device_obj  # Pass torch.device object
                )

            else: # Not using DDP
                local_rank = 0 # Placeholder, not a DDP local rank
                self.global_rank = 0
                print("Running without DDP (single-node, single-GPU/CPU)")
                if torch.cuda.is_available():
                    parsed_id_from_arg = 0 # Default to 0 for non-DDP CUDA
                    if isinstance(self.device, str) and self.device.startswith("cuda:") and ":" in self.device.split(":"):
                        try:
                            parsed_id_from_arg = int(self.device.split(':')[1])
                            if not (0 <= parsed_id_from_arg < torch.cuda.device_count()):
                                print(f"Warning: Parsed device ID {parsed_id_from_arg} from '{self.device}' is invalid. Defaulting to 0.")
                                parsed_id_from_arg = 0
                        except (IndexError, ValueError):
                            print(f"Warning: Could not parse device ID from '{self.device}'. Defaulting to 0.")
                            parsed_id_from_arg = 0
                    elif isinstance(self.device, int):
                        if 0 <= self.device < torch.cuda.device_count():
                            parsed_id_from_arg = self.device
                        else:
                            print(f"Warning: Integer device ID {self.device} is invalid. Defaulting to 0.")
                            parsed_id_from_arg = 0
                    
                    self.effective_device_id = parsed_id_from_arg
                    torch.cuda.set_device(self.effective_device_id)
                else:
                    self.effective_device_id = -1 # Indicates CPU
        else:
            self.global_rank = 0
            self.pytorch_device_idx = 0
            self.effective_device_id = 0
            
        # Log the PyTorch thread count after rank is known
        if self.global_rank == 0:
            log_rank_prefix = ""
            if "RANK" in os.environ:
                log_rank_prefix = f"[Rank {os.environ.get('RANK')}] "
            elif "SLURM_PROCID" in os.environ:
                log_rank_prefix = f"[SlurmPROCID {os.environ.get('SLURM_PROCID')}] "
            try:
                print(f"Process group initialized for rank {self.global_rank}")
                print(f"{log_rank_prefix}PyTorch CPU threads set to: {torch.get_num_threads()} (within init_training_env)")
                print(f"{log_rank_prefix}os.cpu_count() (cores available to this process): {os.cpu_count()}")
                slurm_cpus_task = os.environ.get("SLURM_CPUS_PER_TASK")
                if slurm_cpus_task:
                    print(f"{log_rank_prefix}SLURM_CPUS_PER_TASK (inherited by worker): {slurm_cpus_task}")
            except Exception as e:
                print(f"{log_rank_prefix}Warning: Could not get PyTorch CPU thread count or os.cpu_count(): {e}")

        # Initialize MLflow experiment and run info on rank 0
        self.device = f"cuda:{self.pytorch_device_idx}" if "LOCAL_RANK" in os.environ and torch.cuda.is_available() else (f"cuda:{self.effective_device_id}" if self.effective_device_id != -1 else "cpu")

    def _init_dataloader(self, batch_size=1, num_workers=0):
        # Create dataset with designs on GPU
        dataset = LikelihoodDataset(
            experiment=self.experiment,
            n_particles_per_device=self.run_args["n_particles_per_device"],
            device=f"cuda:{self.pytorch_device_idx}",
            evaluation=False
        )
        
        # Use regular DataLoader without DistributedSampler
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    @profile_method
    def _init_run(self):
        """Initialize MLflow run settings and broadcast to all ranks."""
        if not self.run_args.get("resume_id", None):
            if self.global_rank == 0:
                mlflow.set_experiment(self.mlflow_exp)
                mlflow.start_run()
                self.run_obj = mlflow.active_run()
                # Set n_devices and n_particles using the world size and n_particles_per_device
                self.run_args["n_devices"] = tdist.get_world_size() if "LOCAL_RANK" in os.environ else 1
                self.run_args["n_particles"] = self.run_args["n_devices"] * self.run_args["n_particles_per_device"]

            if not self.run_args.get("restart_id", None):
                # Start new run
                if self.global_rank == 0:
                    print(f"=== NEW RUN MODE ===")
                    print("Starting fresh training run")
                self.checkpoint = None
                
                # For fresh run, priors_run_path is set to the priors_path.yaml file
                self.priors_run_path = self.run_args.get("priors_path", None)
                
            else: 
                # Restart from existing run
                if self.global_rank == 0:
                    print(f"=== RESTART MODE ===")
                    print(f"Restart ID: {self.run_args['restart_id']}")
                    if self.run_args.get("restart_checkpoint", None) is not None:
                        print(f"Restarting from specific checkpoint file: {self.run_args['restart_checkpoint']}")
                    elif self.run_args.get("restart_step", None) is not None:
                        print(f"Restarting from checkpoint at step {self.run_args['restart_step']}")
                    else:
                        print("Restarting from latest checkpoint")
                    print("Will use MLflow run ID to find checkpoint directory")
                    print("Will create new MLflow run with current parameters")

                ref_run = self.client.get_run(self.run_args["restart_id"])
                # fix the model parameters for the restart run
                self._fix_model_args(ref_run)

                checkpoint_dir = f"{self.storage_path}/mlruns/{ref_run.info.experiment_id}/{self.run_args['restart_id']}/artifacts/checkpoints"
                
                # Use saved priors file from artifacts for restart
                self.priors_run_path = f"{self.storage_path}/mlruns/{ref_run.info.experiment_id}/{self.run_args['restart_id']}/artifacts/priors.yaml"

                if self.run_args.get("restart_checkpoint", None) is not None:
                    # Load specific checkpoint file for all ranks
                    print(f"Loading checkpoint from file: {checkpoint_dir}/{self.run_args['restart_checkpoint']}")
                    self.checkpoint = torch.load(f"{checkpoint_dir}/{self.run_args['restart_checkpoint']}", map_location=self.device, weights_only=False)
                else:
                    # Use existing logic to find checkpoint by step
                    # Load the checkpoint
                    self.checkpoint, _ = get_checkpoint(
                        self.run_args["restart_step"], 
                        checkpoint_dir, 
                        self.device, 
                        self.global_rank, 
                        total_steps=int(ref_run.data.params["total_steps"])
                        )
                
                # Validate checkpoint compatibility for restart mode
                validate_checkpoint_compatibility(self.checkpoint, self.run_args, mode="restart", global_rank=self.global_rank)

            # Initialize start step
            self.start_step = 0
            self.best_loss = float('inf')
            self.best_nominal_area = float('inf')
            
            # Initialize runtime tracking (only on rank 0)
            self.previous_cumulative_runtime = 0.0  # seconds

            if self.global_rank == 0:
                if self.priors_run_path is not None:
                    self._save_priors_file()
                
                # Log parameters
                for key, value in self.run_args.items():
                    mlflow.log_param(key, value)

                # Prepare tensors for broadcasting
                tensors = self._prepare_broadcast_tensors()
                print(f"Running with parameters for cosmo_model='{self.run_args['cosmo_model']}':")
                print(json.dumps(self.run_args, indent=2))
            else:
                # Initialize tensors on other ranks
                tensors = {
                    'exp_id': torch.zeros(1, dtype=torch.long, device=self.device),
                    'run_id': [None],
                    'mlflow_exp': [None]
                    }
        else:
            # Resume existing run
            self.run_obj = self.client.get_run(self.run_args["resume_id"])
            checkpoint_dir = f"{self.storage_path}/mlruns/{self.run_obj.info.experiment_id}/{self.run_args['resume_id']}/artifacts/checkpoints"
            
            # Use saved priors file from artifacts for resume
            self.priors_run_path = f"{self.storage_path}/mlruns/{self.run_obj.info.experiment_id}/{self.run_args['resume_id']}/artifacts/priors.yaml"

            # Load the checkpoint
            self.checkpoint, self.start_step = get_checkpoint(
                self.run_args["resume_step"], 
                checkpoint_dir, 
                self.device, 
                self.global_rank, 
                total_steps=self.run_args.get("total_steps")
                )
            
            # Validate that the loaded checkpoint step matches the requested resume step
            if self.start_step != self.run_args["resume_step"]:
                if self.global_rank == 0:
                    print(f"Warning: Requested resume step {self.run_args['resume_step']} not found.")
                    print(f"  Loaded checkpoint from step {self.start_step} instead.")
                    print(f"  This may cause loss discontinuity.")
                # Update the resume_step to match the actual loaded step
                self.run_args["resume_step"] = self.start_step
            
            # Validate checkpoint compatibility for resume mode
            validate_checkpoint_compatibility(self.checkpoint, self.run_args, mode="resume", global_rank=self.global_rank)
            if self.global_rank == 0:
                mlflow.set_experiment(experiment_id=self.run_obj.info.experiment_id)
                mlflow.start_run(run_id=self.run_args["resume_id"])
                
                if self.run_args.get("resume_step", None) is None:
                    raise ValueError("resume_step must be provided when resuming a run")
                print(f"=== RESUME MODE ===")
                print(f"Resume ID: {self.run_args['resume_id']}")
                print(f"Resume Step: {self.start_step}")
                print(f"Add Steps: {self.run_args['add_steps']}")
                print("Will continue existing MLflow run with original parameters")
                resume_args = {
                    "resume_id": self.run_args["resume_id"],
                    "resume_step": self.run_args["resume_step"],
                    "add_steps": self.run_args.get("add_steps", None)
                }
                # Overwrite run parameters with original parameters
                self.run_args = parse_mlflow_params(self.run_obj.data.params)
                self.run_args.update(resume_args)
                if self.run_args.get("add_steps", None):
                    self.run_args["total_steps"] += self.run_args["add_steps"]
                
                n_devices = tdist.get_world_size() if "LOCAL_RANK" in os.environ else 1
                if self.run_args["n_particles"] != n_devices * self.run_args["n_particles_per_device"]:
                    raise ValueError(f"n_particles ({self.run_args['n_particles']}) must be equal to n_devices * n_particles_per_device ({n_devices * self.run_args['n_particles_per_device']})")

                # Get metrics from previous run
                self._get_resume_metrics()
                
                # Load previous cumulative runtime for resume
                self._load_cumulative_runtime()

                # Prepare tensors for broadcasting
                tensors = self._prepare_broadcast_tensors()
            else:
                # Initialize tensors on other ranks
                tensors = {
                    'exp_id': torch.zeros(1, dtype=torch.long, device=self.device),
                    'run_id': [None],
                    'mlflow_exp': [None]
                    }
        self._broadcast_variables(tensors)

        # Ensure rank 0 has fully initialized the MLflow run before other ranks join
        if tdist.is_initialized():
            tdist.barrier()
        
        # Set up MLflow for non-zero ranks
        if self.global_rank != 0:
            mlflow.set_experiment(experiment_id=str(tensors['exp_id'].item()))
            mlflow.start_run(run_id=tensors['run_id'][0], nested=True)
            self.run_obj = mlflow.active_run()
        # Define the MLflow run path
        self.run_path = f"{self.storage_path}/mlruns/{self.run_obj.info.experiment_id}/{self.run_obj.info.run_id}"
        os.makedirs(f"{self.run_path}/artifacts/checkpoints", exist_ok=True)
        os.makedirs(f"{self.run_path}/artifacts/plots/rank_{self.global_rank}/posterior", exist_ok=True)
        # Broadcast run_args from rank 0 to ensure consistency, especially for 'steps' when resuming with add_steps
        if self.global_rank == 0:
            self.session_start_time = None
            run_args_list_to_broadcast = [self.run_args]
        else:
            run_args_list_to_broadcast = [None]

        if tdist.is_initialized():
            tdist.barrier() # Ensure all ranks are ready before broadcasting run_args
        
        tdist.broadcast_object_list(run_args_list_to_broadcast, src=0)
        self.run_args = run_args_list_to_broadcast[0] # All ranks now have the definitive run_args

    def _restore_state(self):
        # Restore RNG state at the beginning of each step if resuming from checkpoint
        
        # Handle Pyro's RNG state restoration (includes torch, random, and numpy)
        try:
            rng_state = self.checkpoint['rng_state']
            pyro_state = rng_state['pyro']
            if pyro_state is None:
                if self.global_rank == 0:
                    print("Warning: Pyro RNG state is None, skipping restoration")
                return
            
            # Ensure the torch state in Pyro's state has the correct dtype and is on CPU
            if isinstance(pyro_state, dict) and 'torch' in pyro_state:
                torch_state = pyro_state['torch']
                if isinstance(torch_state, torch.Tensor):
                    # Move to CPU first, then ensure correct dtype
                    torch_state = torch_state.cpu()
                    if torch_state.dtype != torch.uint8:
                        # Convert to uint8 if it's not already
                        torch_state = torch_state.to(torch.uint8)
                elif isinstance(torch_state, (list, np.ndarray)):
                    # Convert from list/array to uint8 tensor on CPU
                    torch_state = torch.tensor(torch_state, dtype=torch.uint8, device='cpu')
                else:
                    # If we can't convert, skip Pyro RNG restoration
                    raise ValueError(f"Cannot convert torch state of type {type(torch_state)} to uint8 tensor")
                pyro_state['torch'] = torch_state
            
            pyro.util.set_rng_state(pyro_state)  # Pyro's RNG state for deterministic sampling
            
            # Restore CUDA RNG states separately (not handled by Pyro)
            if torch.cuda.is_available() and 'cuda' in rng_state and rng_state['cuda'] is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng_state['cuda']])
            
            # Also restore Pyro's param store state if it exists in the checkpoint
            if 'pyro_param_state' in rng_state:
                pyro.get_param_store().set_state(rng_state['pyro_param_state'])

            if self.global_rank == 0:
                print(f"RNG state restored at step {self.start_step}")
                
        except Exception as e:
            if self.global_rank == 0:
                print(f"Warning: Could not restore RNG state: {e}. Continuing with current state.")
                traceback.print_exc()

    def _save_priors_file(self):
        """Save priors file to artifacts."""
        if self.priors_run_path is None:
            raise ValueError("Priors path is None")
        
        if not os.path.exists(self.priors_run_path):
            raise RuntimeError(f"Priors file not found at {self.priors_run_path}")
        
        try:
            priors_artifact_path = f"{self.storage_path}/mlruns/{mlflow.active_run().info.experiment_id}/{mlflow.active_run().info.run_id}/artifacts/priors.yaml"
            shutil.copy2(self.priors_run_path, priors_artifact_path)
            mlflow.log_artifact(priors_artifact_path)
            print(f"Saved priors file to artifacts: {priors_artifact_path}")
                
            # Verify the file was saved correctly
            if not os.path.exists(priors_artifact_path):
                raise RuntimeError(f"Failed to save priors file to artifacts: {priors_artifact_path}")
                
            return priors_artifact_path
        except Exception as e:
            raise RuntimeError(f"Failed to save priors file: {e}")

    def _fix_model_args(self, ref_run):
        """
        Update run_args associated with the model to match the reference run.
        
        """
        changed_params = {}
        
        # Common parameters that apply to all flow types
        common_params = ["cosmo_model", "flow_type", "n_transforms", "activation", "cond_hidden_size", "cond_n_layers"]
        
        # Store original values and update common parameters
        for param in common_params:
            original_value = self.run_args.get(param)
            new_value = ref_run.data.params[param]
            self.run_args[param] = new_value
            
            if original_value != new_value:
                changed_params[param] = (original_value, new_value)
        
        # Flow-specific parameters
        flow_type = ref_run.data.params["flow_type"]
        flow_specific_params = {
            "MAF": ["nf_transform"],
            "NAF": ["mnn_signal", "mnn_hidden_size", "mnn_n_layers"],
            "NSF": ["spline_bins"]
        }
        
        if flow_type in flow_specific_params:
            for param in flow_specific_params[flow_type]:
                original_value = self.run_args.get(param)
                new_value = ref_run.data.params[param]
                self.run_args[param] = new_value
                
                if original_value != new_value:
                    changed_params[param] = (original_value, new_value)
        
        # Handle condition_design parameter
        original_condition_design = self.run_args.get("condition_design")
        self.run_args["condition_design"] = ref_run.data.params["condition_design"]
        if original_condition_design != run_args["condition_design"]:
            changed_params["condition_design"] = (original_condition_design, self.run_args["condition_design"])
        
        # Log changes if any parameters were modified
        if changed_params and self.global_rank == 0:
            print("=== MODEL PARAMETERS OVERWRITTEN BY REFERENCE RUN ===")
            for param_name, (old_value, new_value) in changed_params.items():
                print(f"  {param_name}: {old_value} -> {new_value}")

    def _broadcast_variables(self, tensors):

        # Broadcast tensors from rank 0 to all ranks
        tdist.barrier()
        tdist.broadcast(tensors['exp_id'], src=0)
        tdist.broadcast_object_list(tensors['run_id'], src=0)
        tdist.broadcast_object_list(tensors['mlflow_exp'], src=0)

    def _get_resume_metrics(self):
        """Get metrics from previous run for resuming."""
        best_nominal_areas = self.client.get_metric_history(self.run_args["resume_id"], 'best_avg_nominal_area')
        best_nominal_area_steps = np.array([metric.step for metric in best_nominal_areas])
        if len(best_nominal_area_steps) > 0:
            closest_idx = np.argmin(np.abs(best_nominal_area_steps - self.run_args["resume_step"]))
            self.best_nominal_area = best_nominal_areas[closest_idx].value if best_nominal_area_steps[closest_idx] <= self.run_args["resume_step"] else best_nominal_areas[closest_idx - 1].value
        else:
            self.best_nominal_area = np.nan
        
        best_losses = self.client.get_metric_history(self.run_args["resume_id"], 'best_loss')
        best_loss_steps = np.array([metric.step for metric in best_losses])
        closest_idx = np.argmin(np.abs(best_loss_steps - self.run_args["resume_step"]))
        self.best_loss = best_losses[closest_idx].value if best_loss_steps[closest_idx] <= self.run_args["resume_step"] else best_losses[closest_idx - 1].value
    
    def _load_cumulative_runtime(self):
        """Load cumulative runtime from previous training sessions."""
        try:
            runtime_history = self.client.get_metric_history(self.run_args["resume_id"], 'cumulative_runtime_seconds')
            if runtime_history:
                # Get the latest cumulative runtime
                self.previous_cumulative_runtime = runtime_history[-1].value
                if self.global_rank == 0:
                    hours = int(self.previous_cumulative_runtime // 3600)
                    minutes = int((self.previous_cumulative_runtime % 3600) // 60)
                    seconds = int(self.previous_cumulative_runtime % 60)
                    print(f"Loaded previous cumulative runtime: {hours}h {minutes}m {seconds}s")
            else:
                self.previous_cumulative_runtime = 0.0
        except Exception as e:
            if self.global_rank == 0:
                print(f"Warning: Could not load cumulative runtime: {e}. Starting from 0.")
            self.previous_cumulative_runtime = 0.0
    
    def _update_cumulative_runtime(self, step):
        """Update cumulative runtime metric (only rank 0)."""
        if self.global_rank != 0:
            return
            
        if self.session_start_time is None:
            # First call - initialize session start time
            self.session_start_time = time.time()
            return
        
        # Calculate session runtime and add to previous cumulative
        session_runtime = time.time() - self.session_start_time
        cumulative_runtime = self.previous_cumulative_runtime + session_runtime
        
        # Log runtime to MLflow
        mlflow.log_metric("cumulative_runtime_seconds", cumulative_runtime, step=step)
        
    def _prepare_broadcast_tensors(self):
        """Prepare tensors for broadcasting to all ranks."""
        return {
            'exp_id': torch.tensor([int(self.run_obj.info.experiment_id)], dtype=torch.long, device=self.device),
            'run_id': [self.run_obj.info.run_id],
            'mlflow_exp': [mlflow.get_experiment(self.run_obj.info.experiment_id).name]
        }

    def _init_optimizer(self):
        """
        Initialize the optimizer.
        """
        if self.run_args["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(self.posterior_flow.parameters(), lr=self.run_args["initial_lr"])
        elif self.run_args["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(self.posterior_flow.parameters(), lr=self.run_args["initial_lr"])
        else:
            raise ValueError(f"Invalid optimizer: {self.run_args['optimizer']}")

    @profile_method
    def _init_scheduler(self):
        """
        Initialize the scheduler.
        """
        if self.run_args.get("resume_id", None):
            # Load optimizer state
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            
            # Create scheduler with original run parameters (already loaded into run_args during resume)
            self.scheduler = create_scheduler(self.optimizer, self.run_args)
            
            # Load scheduler state if it exists (preserve original scheduler state for resume)
            if 'scheduler_state_dict' in self.checkpoint:            
                self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
                
                # Update optimizer learning rate to match scheduler's _last_lr
                if hasattr(self.scheduler, '_last_lr') and self.scheduler._last_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.scheduler._last_lr[0]
                
                if self.global_rank == 0:
                    print("Scheduler state restored from checkpoint")
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Current learning rate: {current_lr}")
            else:
                if self.global_rank == 0:
                    print("Warning: No scheduler state found in checkpoint")
                    print("  Learning rate may not be correct for the resume step")
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Current learning rate: {current_lr}")
                    
                # If no scheduler state, we need to manually step the scheduler to the correct point
                if self.global_rank == 0:
                    print(f"  Manually stepping scheduler to step {self.run_args['resume_step']}")
                step_freq = self.run_args.get("step_freq", 1)
                scheduler_steps_needed = self.run_args['resume_step'] // step_freq
                if self.global_rank == 0:
                    print(f"  Step frequency: {step_freq}, scheduler steps needed: {scheduler_steps_needed}")
                
                # All ranks must step the scheduler to stay synchronized
                for _ in range(scheduler_steps_needed):
                    self.scheduler.step()
                
                if self.global_rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Learning rate after manual stepping: {current_lr}")
            
            if self.global_rank == 0:
                print(f"Optimizer LR after loading: {self.optimizer.param_groups[0]['lr']}")
                print("Resume checkpoint loading completed")
                
            # Synchronize all ranks after loading checkpoint
            tdist.barrier()

        elif self.run_args.get("restart_id", None):
            # Load model weights and optimizer state
            self.posterior_flow.module.load_state_dict(self.checkpoint['model_state_dict'], strict=True)
            
            if self.run_args.get("restart_optimizer", None):
                if self.run_args["optimizer"] != self.checkpoint["optimizer_type"] if "optimizer_type" in self.checkpoint else False:
                    raise ValueError(f"Optimizer type mismatch: {self.run_args['optimizer']} != {self.checkpoint['optimizer_type']}")
                # Update the learning rate in the loaded optimizer state to match current run_args
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = self.run_args['initial_lr']
                    param_group['initial_lr'] = self.run_args['initial_lr']
                    if self.global_rank == 0:
                        print(f"Restarted optimizer with initial learning rate: {old_lr} -> {self.run_args['initial_lr']}")
            else:
                if self.global_rank == 0:
                    print(f"Created fresh optimizer with initial learning rate: {self.run_args['initial_lr']}")
            
            # Create fresh scheduler with new learning rate bounds
            self.scheduler = create_scheduler(self.optimizer, self.run_args)
            
            if self.global_rank == 0:
                print(f"Using existing optimizer statistics with new learning rate schedule")
                print(f"Initial lr: {run_args['initial_lr']}, Final lr: {run_args.get('final_lr', run_args['initial_lr'])}")
                print("Fresh scheduler created with current parameters")
            
            # Synchronize all ranks after loading checkpoint
            tdist.barrier()
            
        else:
            self.scheduler = create_scheduler(self.optimizer, self.run_args)
            seed = auto_seed(base_seed=self.run_args["pyro_seed"], rank=self.global_rank)

if __name__ == '__main__':

    mp.set_start_method("spawn", force=True)
    #set default dtype
    torch.set_default_dtype(torch.float64)
    
    # --- Argument Parsing --- 
    parser = argparse.ArgumentParser(description="Run Number Tracers Training")

    # Add arguments dynamically based on the default config file
    parser.add_argument('--cosmo_exp', type=str, default='num_tracers', help='Cosmological experiment name')
    parser.add_argument('--cosmo_model', type=str, default=None, help='Cosmological model set to use from train_args.yaml (not needed for resume)')
    parser.add_argument('--mlflow_exp', type=str, default='base', help='MLflow experiment name')
    parser.add_argument('--device', type=str, default=None, help='Device to use for training')
    
    # Add resume/restart arguments early so we can check them
    parser.add_argument('--resume_id', type=str, default=None, help='MLflow run ID to resume training from')
    parser.add_argument('--resume_step', type=int, default=None, help='Step to resume training from')
    parser.add_argument('--add_steps', type=int, default=0, help='Number of steps to add to the training (only used with --resume_id)')
    parser.add_argument('--restart_id', type=str, default=None, help='MLflow run ID to restart training from (creates new run with current parameters)')
    parser.add_argument('--restart_step', type=lambda x: int(x) if x.isdigit() else x, default=None, help='Step to restart training from (optional when using --restart_id, defaults to latest). Can be an integer or string like "last"')
    parser.add_argument('--restart_checkpoint', type=str, default=None, help='Specific checkpoint file name to use for all ranks during restart (alternative to --restart_step)')
    parser.add_argument('--restart_optimizer', action='store_true', help='Restart optimizer from checkpoint')
    parser.add_argument('--log_usage', action='store_true', help='Log compute usage of the training script')
    parser.add_argument('--profile', action='store_true', help='Enable profiling for a few steps and then exit.')

    # --- Load Default Config --- 
    # Parse just these essential arguments
    args, unknown = parser.parse_known_args()

    # For resume or restart, get cosmo_model from MLflow run metadata
    if args.resume_id:
        mlflow.set_tracking_uri(os.environ["SCRATCH"] + f"/bed/BED_cosmo/{args.cosmo_exp}/mlruns")
        client = MlflowClient()
        resume_run = client.get_run(args.resume_id)
        cosmo_model = resume_run.data.params.get('cosmo_model', 'base')
        if os.environ.get('RANK', '0') == '0':
            print(f"Resume mode: Loading cosmo_model '{cosmo_model}' from run {args.resume_id}")
    elif args.restart_id:
        mlflow.set_tracking_uri(os.environ["SCRATCH"] + f"/bed/BED_cosmo/{args.cosmo_exp}/mlruns")
        client = MlflowClient()
        restart_run = client.get_run(args.restart_id)
        cosmo_model = restart_run.data.params.get('cosmo_model', 'base')
        if os.environ.get('RANK', '0') == '0':
            print(f"Restart mode: Loading cosmo_model '{cosmo_model}' from run {args.restart_id}")
    else:
        # For new runs, cosmo_model must be specified
        if args.cosmo_model is None:
            raise ValueError("--cosmo_model is required for new training runs (not needed for resume/restart)")
        cosmo_model = args.cosmo_model

    # Load config from train_args.yaml based on cosmo_model
    config_path = os.path.join(os.path.dirname(__file__), args.cosmo_exp, 'train_args.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    # Select configuration for the specified cosmo_model
    if cosmo_model not in all_configs:
        available_models = ', '.join(all_configs.keys())
        raise ValueError(f"Model '{cosmo_model}' not found in {config_path}. Available models: {available_models}")
    
    run_args_dict = all_configs[cosmo_model]
    
    if os.environ.get('RANK', '0') == '0':
        print(f"Loaded configuration for model '{cosmo_model}' from {config_path}")

    # Skip keys that are already defined as parser arguments
    skip_keys = {
        'cosmo_exp', 'cosmo_model', 'mlflow_exp', 'device', 'resume_id', 
        'resume_step', 'restart_id', 'restart_step', 'restart_checkpoint', 
        'restart_optimizer', 'add_steps', 'log_usage', 'profile'
        }
    
    for key, value in run_args_dict.items():
        if key in skip_keys:
            continue
        if value is None:
            parser.add_argument(f'--{key}', type=str, default=None, 
                              help=f'Override {key} (default: None)')
            continue
            
        arg_type = type(value)
        # Special handling for fixed_design to allow --fixed_design (→ True) or --fixed_design [list]
        if key == 'fixed_design':
            parser.add_argument(f'--{key}', nargs='?', const=True, default=None,
                              help=f'Use fixed design. Pass no value for nominal design, or a JSON list for specific design(s) (default: {value})')
        elif isinstance(value, bool):
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


    # Override default run_args with any provided command-line arguments
    run_args = vars(args)
    for key, value in run_args.items():
        if key in run_args_dict.keys():
            if value is not None:
                # Special handling for fixed_design
                if key == 'fixed_design':
                    if value is True:
                        # Flag passed with no value → use nominal design
                        run_args[key] = True
                    elif isinstance(value, str):
                        # JSON string passed → parse as list
                        try:
                            parsed_value = json.loads(value)
                            run_args[key] = parsed_value
                        except json.JSONDecodeError as e:
                            if os.environ.get('RANK', '0') == '0':
                                print(f"Warning: Could not parse 'fixed_design' as JSON: {e}. Using as-is.")
                            run_args[key] = value
                    else:
                        run_args[key] = value
                elif isinstance(run_args_dict[key], bool) and isinstance(value, bool):
                    run_args[key] = value
                elif isinstance(run_args_dict[key], list):
                    # Parse JSON string back to list
                    try:
                        parsed_value = json.loads(value)
                        run_args[key] = parsed_value
                    except json.JSONDecodeError as e:
                        if os.environ.get('RANK', '0') == '0':
                            print(f"Warning: Could not parse '{key}' as JSON: {e}. Keeping default value.")
                elif not isinstance(run_args_dict[key], float):
                    run_args[key] = value
            else:
                run_args[key] = run_args_dict[key]

    trainer = Trainer(
        cosmo_exp=args.cosmo_exp,
        mlflow_exp=args.mlflow_exp,
        run_args=run_args,
        profile=args.profile,
    )
    trainer.run()
