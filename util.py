import sys
import os
import torch
import pyro
import mlflow
import zuko
import random
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
import getdist
from pyro import distributions as dist
from pyro_oed_src import _safe_mean_terms, posterior_loss
import json
from num_tracers import NumTracers
import contextlib
import io
import matplotlib.pyplot as plt
from pyro.contrib.util import lexpand
import subprocess
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime, timedelta
import traceback

torch.set_default_dtype(torch.float64)

home_dir = os.environ["HOME"]
if home_dir + "/bed/BED_cosmo" not in sys.path:
    sys.path.insert(0, home_dir + "/bed/BED_cosmo")
sys.path.insert(0, home_dir + "/bed/BED_cosmo/num_tracers")

def auto_seed(base_seed=0, rank=0):
    if base_seed < 0:
        base_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    global_seed = base_seed + rank
    # Set all relevant seeds
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)
        torch.cuda.manual_seed_all(global_seed)
        # For completely deterministic results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    pyro.set_rng_seed(global_seed)
    return global_seed

def restore_state(checkpoint, step, global_rank):
    # Restore RNG state at the beginning of each step if resuming from checkpoint
    
    # Handle Pyro's RNG state restoration (includes torch, random, and numpy)
    try:
        rng_state = checkpoint['rng_state']
        pyro_state = rng_state['pyro']
        if pyro_state is None:
            if global_rank == 0:
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
    except Exception as e:
        if global_rank == 0:
            print(f"Warning: Could not restore Pyro RNG state: {e}. Continuing with current state.")
            traceback.print_exc()
    
    # Restore CUDA RNG states separately (not handled by Pyro)
    if torch.cuda.is_available() and 'cuda' in rng_state and rng_state['cuda'] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng_state['cuda']])
    
    # Also restore Pyro's param store state if it exists in the checkpoint
    if 'pyro_param_state' in rng_state:
        pyro.get_param_store().set_state(rng_state['pyro_param_state'])
    
    if global_rank == 0:
        print(f"RNG state restored for step {step}")

def print_memory_usage(process, step):
    mem_info = process.memory_info()
    print(f"Step {step}: Memory Usage: {mem_info.rss / 1024**2:.2f} MB")

def save_checkpoint(model, optimizer, filepath, step=None, artifact_path=None, scheduler=None, additional_state=None, global_rank=None):
    """
    Saves the training checkpoint with comprehensive state information.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        filepath (str): Path to save the checkpoint.
        step (int, optional): Current training step.
        artifact_path (str, optional): Path to log the artifact to in MLflow.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        additional_state (dict, optional): Additional state to save (e.g., training metrics, configuration).
        global_rank (int, optional): Global rank in distributed training.
    """
    # Get the state dict from the model, handling DDP wrapper if present
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_type': optimizer.__class__.__name__
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if step is not None:
        checkpoint['step'] = step

    # Save RNG states correctly
    checkpoint['rng_state'] = get_rng_state()
    
    # Also save Pyro's param store state (in case there are any parameters)
    checkpoint['rng_state']['pyro_param_state'] = pyro.get_param_store().get_state()
    
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

def get_rng_state():
    return {
        'pyro': pyro.util.get_rng_state(),  # This includes torch, random, and numpy states
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

def init_training_env(tdist, device):
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
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"]) # SLURM's local rank
        global_rank = int(os.environ["RANK"])
        
        # When CUDA_VISIBLE_DEVICES isolates one GPU, PyTorch sees it as device 0.
        pytorch_device_idx = int(os.environ["LOCAL_RANK"])  # The only GPU visible to this process
        effective_device_id = pytorch_device_idx
        
        # Initialize CUDA context before DDP setup
        torch.cuda.init()
        torch.cuda.set_device(pytorch_device_idx)
        
        # Ensure CUDA is available and device is set before DDP init
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but LOCAL_RANK is set")
        
        # Create torch.device object for device_id
        device_obj = torch.device(f"cuda:{pytorch_device_idx}")
        
        # Initialize process group with explicit device ID
        tdist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=global_rank,
            timeout=timedelta(seconds=180),  # Increased timeout
            device_id=device_obj  # Pass torch.device object
        )
        print(f"Process group initialized for rank {global_rank}")

    else: # Not DDP
        local_rank = 0 # Placeholder, not a DDP local rank
        global_rank = 0
        print("Running without DDP (single-node, single-GPU/CPU)")
        if torch.cuda.is_available():
            parsed_id_from_arg = 0 # Default to 0 for non-DDP CUDA
            if isinstance(device, str) and device.startswith("cuda:") and ":" in device.split(":"):
                try:
                    parsed_id_from_arg = int(device.split(':')[1])
                    if not (0 <= parsed_id_from_arg < torch.cuda.device_count()):
                        print(f"Warning: Parsed device ID {parsed_id_from_arg} from '{device}' is invalid. Defaulting to 0.")
                        parsed_id_from_arg = 0
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse device ID from '{device}'. Defaulting to 0.")
                    parsed_id_from_arg = 0
            elif isinstance(device, int):
                 if 0 <= device < torch.cuda.device_count():
                     parsed_id_from_arg = device
                 else:
                     print(f"Warning: Integer device ID {device} is invalid. Defaulting to 0.")
                     parsed_id_from_arg = 0
            
            effective_device_id = parsed_id_from_arg
            torch.cuda.set_device(effective_device_id)
        else:
            effective_device_id = -1 # Indicates CPU

    # Log the PyTorch thread count after rank is known
    # Determine a preliminary rank for logging even before full DDP init, if possible
    log_rank_prefix = ""
    if "RANK" in os.environ:
        log_rank_prefix = f"[Rank {os.environ.get('RANK')}] "
    elif "SLURM_PROCID" in os.environ:
        log_rank_prefix = f"[SlurmPROCID {os.environ.get('SLURM_PROCID')}] "
    
    try:
        print(f"{log_rank_prefix}PyTorch CPU threads set to: {torch.get_num_threads()} (within init_training_env)")
        # Print available CPU cores as seen by this Python process
        print(f"{log_rank_prefix}os.cpu_count() (cores available to this process): {os.cpu_count()}")
        # You can also check SLURM_CPUS_PER_TASK if it's set and inherited by the DDP worker
        slurm_cpus_task = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus_task:
            print(f"{log_rank_prefix}SLURM_CPUS_PER_TASK (inherited by worker): {slurm_cpus_task}")
    except Exception as e:
        print(f"{log_rank_prefix}Warning: Could not get PyTorch CPU thread count or os.cpu_count(): {e}")

    return global_rank, local_rank, effective_device_id, pytorch_device_idx

def init_nf(flow_type, input_dim, context_dim, run_args, device="cuda:0", seed=0, verbose=False, **kwargs):
    # Set seeds first, before any model initialization
    if seed is not None:
        # Ensure seed is an integer for all random libraries
        torch.manual_seed(int(seed))
        random.seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    n_layers = int(run_args.get("n_layers", 1))
    hidden_size = int(run_args.get("hidden_size", 64))
    n_transforms = int(run_args.get("n_transforms", 5))

    # Initialize the flow model
    if flow_type == "NSF":
        bins_val = int(run_args.get("bins"))
        posterior_flow = zuko.flows.NSF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms, 
            bins=bins_val,
            hidden_features=((hidden_size,) * n_layers),
            **kwargs
        )
    elif flow_type == "NAF":
        signal_val = int(run_args.get("signal"))
        posterior_flow = zuko.flows.NAF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            signal=signal_val,
            network={"hidden_features": ((hidden_size,) * n_layers)}
        )
    elif flow_type == "MAF":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            hidden_features=((hidden_size,) * n_layers),
            **kwargs
        )
    elif flow_type == "MAF_Affine":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            univariate=zuko.transforms.MonotonicAffineTransform,
            **kwargs
        )
    elif flow_type == "MAF_RQS":
        shape_val = int(run_args.get("shape"))
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            univariate=zuko.transforms.MonotonicRQSTransform,
            shapes=([shape_val], [shape_val], [shape_val-1]),
            **kwargs
        )
    elif flow_type == "NICE":
        posterior_flow = zuko.flows.NICE(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            **kwargs
        )
    elif flow_type == "GF":
        posterior_flow = zuko.flows.GF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms
        )
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")

    # Move to the correct device
    posterior_flow.to(device)

    return posterior_flow

def init_scheduler(optimizer, run_args):
    # Setup
    steps_per_cycle = run_args["total_steps"] // run_args["n_cycles"]
    initial_lr = run_args["initial_lr"]
    final_lr = run_args["final_lr"]

    if run_args["scheduler_type"] == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, 
            factor=1.0
            )
    elif run_args["scheduler_type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=run_args["total_steps"],
            eta_min=final_lr
            )
    elif run_args["scheduler_type"] == "linear":
        # factor is the number we multiply the initial lr by to get the final lr at each step
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=final_lr / initial_lr, 
            total_iters=run_args["total_steps"] - 1
            )
    elif run_args["scheduler_type"] == "exponential":
        # calculate gamma from initial and final lr
        gamma = (final_lr / initial_lr) ** (1 / run_args["total_steps"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=gamma
            )
    elif run_args["scheduler_type"] == "lambda":
        def lr_lambda(step):
            cycle = step // steps_per_cycle
            cycle_progress = (step % steps_per_cycle) / steps_per_cycle
            # Decaying peak
            peak = initial_lr * (gamma ** cycle)
            # Cosine decay within cycle
            cosine = 0.5 * (1 + np.cos(np.pi * cycle_progress))
            lr = final_lr + (peak - final_lr) * cosine
            return lr / initial_lr  # LambdaLR expects a multiplier of the initial LR
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda
            )
    else:
        raise ValueError(f"Unknown scheduler_type: {run_args['scheduler_type']}")
    return scheduler

def init_run(
        tdist, 
        global_rank, 
        current_pytorch_device, 
        storage_path, 
        mlflow_experiment_name, 
        cosmo_model, 
        run_args, 
        **kwargs
        ):
    """Initialize MLflow run settings and broadcast to all ranks."""
    if not kwargs.get("resume_id", None):
        if not kwargs.get("restart_id", None):
            # Start new run
            if global_rank == 0:
                print(f"=== NEW RUN MODE ===")
                print("Starting fresh training run")
            checkpoint = None
        else:
            if global_rank == 0:
                print(f"=== RESTART MODE ===")
                print(f"Restart ID: {kwargs['restart_id']}")
                if kwargs.get("restart_checkpoint", None) is not None:
                    print(f"Restarting from specific checkpoint file: {kwargs['restart_checkpoint']}")
                elif kwargs.get("restart_step", None) is not None:
                    print(f"Restarting from checkpoint at step {kwargs['restart_step']}")
                else:
                    print("Restarting from latest checkpoint")
                print("Will use MLflow run ID to find checkpoint directory")
                print("Will create new MLflow run with current parameters")
            client = mlflow.MlflowClient()
            ref_run = client.get_run(kwargs["restart_id"])
            checkpoint_dir = f"{storage_path}/mlruns/{ref_run.info.experiment_id}/{kwargs['restart_id']}/artifacts/checkpoints"
            if kwargs.get("restart_checkpoint", None) is not None:
                # Load specific checkpoint file for all ranks
                print(f"Loading checkpoint from file: {checkpoint_dir}/{kwargs['restart_checkpoint']}")
                checkpoint = torch.load(f"{checkpoint_dir}/{kwargs['restart_checkpoint']}", map_location=current_pytorch_device, weights_only=False)
            else:
                # Use existing logic to find checkpoint by step
                # Load the checkpoint
                checkpoint, _ = get_checkpoint(
                    kwargs["restart_step"], 
                    checkpoint_dir, 
                    current_pytorch_device, 
                    global_rank, 
                    total_steps=int(ref_run.data.params["total_steps"])
                    )
            
            # Validate checkpoint compatibility for restart mode
            validate_checkpoint_compatibility(checkpoint, run_args, mode="restart", global_rank=global_rank)

        if global_rank == 0:
            mlflow.set_experiment(mlflow_experiment_name)
            mlflow.start_run()
            # Set n_devices and n_particles using the world size and n_particles_per_device
            run_args["n_devices"] = tdist.get_world_size() if "LOCAL_RANK" in os.environ else 1
            run_args["n_particles"] = run_args["n_devices"] * run_args["n_particles_per_device"]

            # Log parameters
            mlflow.log_param("cosmo_model", cosmo_model)
            for key, value in run_args.items():
                mlflow.log_param(key, value)
            for key, value in kwargs.items():
                mlflow.log_param(key, value)
            
            # Initialize metrics
            start_step = 0
            best_loss = float('inf')
            best_nominal_area = float('inf')
            
            # Create artifact directories
            os.makedirs(f"{storage_path}/mlruns/{mlflow.active_run().info.experiment_id}/{mlflow.active_run().info.run_id}/artifacts/checkpoints", exist_ok=True)
            os.makedirs(f"{storage_path}/mlruns/{mlflow.active_run().info.experiment_id}/{mlflow.active_run().info.run_id}/artifacts/plots", exist_ok=True)
            
            # Prepare tensors for broadcasting
            tensors = _prepare_broadcast_tensors(
                mlflow.active_run().info.experiment_id, start_step, 
                best_loss, best_nominal_area, current_pytorch_device,
                mlflow.active_run().info.run_id, mlflow_experiment_name
            )
            print(f"Running with parameters for cosmo_model='{cosmo_model}':")
            print(json.dumps(run_args, indent=2))
        else:
            # Initialize tensors on other ranks
            tensors = {
                'exp_id': torch.zeros(1, dtype=torch.long, device=current_pytorch_device),
                'start_step': torch.zeros(1, dtype=torch.long, device=current_pytorch_device),
                'best_loss': torch.zeros(1, dtype=torch.float64, device=current_pytorch_device),
                'best_nominal_area': torch.zeros(1, dtype=torch.float64, device=current_pytorch_device),
                'run_id': [None],
                'exp_name': [None]
                }
    else:
        client = mlflow.MlflowClient()
        # Resume existing run
        run_info = client.get_run(kwargs["resume_id"])
        exp_id = run_info.info.experiment_id
        exp_name = mlflow.get_experiment(exp_id).name
        checkpoint_dir = f"{storage_path}/mlruns/{exp_id}/{kwargs['resume_id']}/artifacts/checkpoints"
        # Load the checkpoint
        checkpoint, start_step = get_checkpoint(
            kwargs["resume_step"], 
            checkpoint_dir, 
            current_pytorch_device, 
            global_rank, 
            total_steps=run_args.get("total_steps")
            )
        # Validate checkpoint compatibility for resume mode
        validate_checkpoint_compatibility(checkpoint, run_args, mode="resume", global_rank=global_rank)
        if global_rank == 0:
            mlflow.set_experiment(experiment_id=exp_id)
            mlflow.start_run(run_id=kwargs["resume_id"])
            
            if kwargs.get("resume_step", None) is None:
                raise ValueError("resume_step must be provided when resuming a run")
            print(f"=== RESUME MODE ===")
            print(f"Resume ID: {kwargs['resume_id']}")
            print(f"Resume Step: {start_step}")
            print(f"Add Steps: {kwargs['add_steps']}")
            print("Will continue existing MLflow run with original parameters")
            # Update run parameters
            mlflow_experiment_name = exp_name
            cosmo_model = run_info.data.params["cosmo_model"]
            run_args = parse_mlflow_params(run_info.data.params)
            if kwargs.get("add_steps", None):
                run_args["total_steps"] += kwargs["add_steps"]
            
            n_devices = tdist.get_world_size() if "LOCAL_RANK" in os.environ else 1
            if run_args["n_particles"] != n_devices * run_args["n_particles_per_device"]:
                raise ValueError(f"n_particles ({run_args['n_particles']}) must be equal to n_devices * n_particles_per_device ({n_devices * run_args['n_particles_per_device']})")

            # Get metrics from previous run
            best_nominal_area, best_loss = _get_resume_metrics(client, kwargs["resume_id"], kwargs["resume_step"])

            # Prepare tensors for broadcasting
            tensors = _prepare_broadcast_tensors(
                exp_id, start_step, best_loss, best_nominal_area, 
                current_pytorch_device, kwargs["resume_id"], exp_name
            )
        else:
            # Initialize tensors on other ranks
            tensors = {
                'exp_id': torch.zeros(1, dtype=torch.long, device=current_pytorch_device),
                'start_step': torch.zeros(1, dtype=torch.long, device=current_pytorch_device),
                'best_loss': torch.zeros(1, dtype=torch.float64, device=current_pytorch_device),
                'best_nominal_area': torch.zeros(1, dtype=torch.float64, device=current_pytorch_device),
                'run_id': [None],
                'exp_name': [None]
                }
    _broadcast_variables(tensors, global_rank, run_args, tdist)

    # Ensure rank 0 has fully initialized the MLflow run before other ranks join
    if tdist.is_initialized():
        tdist.barrier()
    
    # Set up MLflow for non-zero ranks
    if global_rank != 0:
        mlflow.set_experiment(experiment_id=str(tensors['exp_id'].item()))
        mlflow.start_run(run_id=tensors['run_id'][0], nested=True)
    
    # Create ml_info object
    ml_info = type('mlinfo', (), {})()
    ml_info.experiment_id = str(tensors['exp_id'].item())
    ml_info.run_id = tensors['run_id'][0]

    # Broadcast run_args from rank 0 to ensure consistency, especially for 'steps' when resuming with add_steps
    if global_rank == 0:
        # run_args was already prepared correctly on rank 0 (either new or resumed with add_steps)
        run_args_list_to_broadcast = [run_args]
    else:
        run_args_list_to_broadcast = [None]  # Placeholder for other ranks

    if tdist.is_initialized():
        tdist.barrier() # Ensure all ranks are ready before broadcasting run_args
    
    tdist.broadcast_object_list(run_args_list_to_broadcast, src=0)
    run_args = run_args_list_to_broadcast[0] # All ranks now have the definitive run_args

    # MLflow runs are already properly initialized above - no need for additional start_run calls
    # This prevents race conditions that can corrupt meta.yaml files

    return ml_info, run_args, checkpoint, tensors['start_step'].item(), tensors['best_loss'].item(), tensors['best_nominal_area'].item()

def _broadcast_variables(tensors, global_rank, run_args,tdist):

    # Broadcast tensors from rank 0 to all ranks
    tdist.barrier()
    tdist.broadcast(tensors['exp_id'], src=0)
    tdist.broadcast(tensors['start_step'], src=0)
    tdist.broadcast(tensors['best_loss'], src=0)
    tdist.broadcast(tensors['best_nominal_area'], src=0)
    tdist.broadcast_object_list(tensors['run_id'], src=0)
    tdist.broadcast_object_list(tensors['exp_name'], src=0)

def _get_resume_metrics(client, resume_id, resume_step):
    """Get metrics from previous run for resuming."""
    best_nominal_areas = client.get_metric_history(resume_id, 'best_nominal_area')
    best_nominal_area_steps = np.array([metric.step for metric in best_nominal_areas])
    if len(best_nominal_area_steps) > 0:
        closest_idx = np.argmin(np.abs(best_nominal_area_steps - resume_step))
        best_nominal_area = best_nominal_areas[closest_idx].value if best_nominal_area_steps[closest_idx] <= resume_step else best_nominal_areas[closest_idx - 1].value
    else:
        best_nominal_area = np.nan
    
    best_losses = client.get_metric_history(resume_id, 'best_loss')
    best_loss_steps = np.array([metric.step for metric in best_losses])
    closest_idx = np.argmin(np.abs(best_loss_steps - resume_step))
    best_loss = best_losses[closest_idx].value if best_loss_steps[closest_idx] <= resume_step else best_losses[closest_idx - 1].value
    
    return best_nominal_area, best_loss

def _prepare_broadcast_tensors(exp_id, start_step, best_loss, best_nominal_area, device, run_id, exp_name):
    """Prepare tensors for broadcasting."""
    return {
        'exp_id': torch.tensor([int(exp_id)], dtype=torch.long, device=device),
        'start_step': torch.tensor([start_step], dtype=torch.long, device=device),
        'best_loss': torch.tensor([best_loss], dtype=torch.float64, device=device),
        'best_nominal_area': torch.tensor([best_nominal_area], dtype=torch.float64, device=device),
        'run_id': [run_id],
        'exp_name': [exp_name]
    }

def get_runs_data(exp_name=None, run_ids=None, excluded_runs=[], filter_string=None, parse_params=False):
    """
    Fetches run data from MLflow based on experiment name or run IDs.

    Args:
        exp_name (str, optional): Name of the MLflow experiment.
        run_ids (str or list, optional): A single run ID or a list of run IDs.
        excluded_runs (list, optional): A list of run IDs to exclude from the result.
        filter_string (str, optional): An MLflow filter string to apply when searching for runs.
        parse_params (bool): Whether to parse MLflow parameters using `parse_mlflow_params`.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries, where each dictionary represents a run and contains
                    'run_id', 'params', 'run_obj', 'name', and 'exp_id'. Returns empty list on failure.
            - str or None: The experiment ID for the fetched runs.
            - str or None: The experiment name for the fetched runs.
    """
    client = MlflowClient()
    run_data_list = []
    experiment_id = None
    actual_exp_name = exp_name

    if exp_name is not None:
        try:
            exp = client.get_experiment_by_name(exp_name)
            if exp is None:
                print(f"Error: Experiment '{exp_name}' not found.")
                return [], None, None
            experiment_id = exp.experiment_id
            actual_exp_name = exp.name
            
            run_infos = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string if filter_string else "",
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
            )
            
            for run_obj in run_infos:
                if run_obj.info.run_id not in excluded_runs:
                    params = parse_mlflow_params(run_obj.data.params) if parse_params else run_obj.data.params
                    run_data_list.append({
                        'run_id': run_obj.info.run_id,
                        'params': params,
                        'run_obj': run_obj,
                        'name': run_obj.info.run_name,
                        'exp_id': run_obj.info.experiment_id
                    })
        except Exception as e:
            print(f"Error fetching runs for experiment '{exp_name}': {e}")
            return [], None, None
            
    elif run_ids is not None:
        run_ids_list = [run_ids] if isinstance(run_ids, str) else run_ids
        processed_run_ids = [rid for rid in run_ids_list if rid not in excluded_runs]
        
        if not processed_run_ids:
            print("No valid runs to process after exclusion.")
            return [], None, None

        for rid in processed_run_ids:
            try:
                run_obj = client.get_run(rid)
                if experiment_id is None:
                    experiment_id = run_obj.info.experiment_id
                    if actual_exp_name is None:
                        try:
                            exp = client.get_experiment(experiment_id)
                            actual_exp_name = exp.name if exp else "Selected Runs"
                        except Exception:
                            actual_exp_name = "Selected Runs (from run_id)"

                params = parse_mlflow_params(run_obj.data.params) if parse_params else run_obj.data.params
                run_data_list.append({
                    'run_id': rid,
                    'params': params,
                    'run_obj': run_obj,
                    'name': run_obj.info.run_name,
                    'exp_id': run_obj.info.experiment_id
                })
            except Exception as e:
                print(f"Warning: Could not fetch data for run {rid}: {e}. Skipping this run.")
    else:
        print("Either exp_name or run_ids must be provided.")
        return [], None, None

    if not run_data_list:
        print("No runs found to process.")
        return [], None, None

    return run_data_list, experiment_id, actual_exp_name


def run_eval(run_obj, parsed_run_params, eval_args, step='loss_best', cosmo_exp='num_tracers', run_id_for_fallback_only=None, global_rank=0):
    # run_id_for_fallback_only is if run_obj and parsed_run_params are somehow not available from caller
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    mlflow.set_tracking_uri(storage_path + "/mlruns")

    current_run_id = None

    if run_obj:
        current_run_id = run_obj.info.run_id
    elif run_id_for_fallback_only:
        client = MlflowClient() # Initialize client only if needed for fallback
        run_obj_fallback = client.get_run(run_id_for_fallback_only)
        current_run_id = run_obj_fallback.info.run_id
        parsed_run_params = parse_mlflow_params(run_obj_fallback.data.params) 
        run_obj = run_obj_fallback 
    else:
        raise ValueError("run_eval requires either run_obj and parsed_run_params, or a run_id_for_fallback_only")

    with open(mlflow.artifacts.download_artifacts(run_id=current_run_id, artifact_path="classes.json")) as f:
        classes = json.load(f)
    # Pass run_obj, parsed_run_params (which should be consistent with run_obj), and classes
    likelihood = init_likelihood(parsed_run_params, eval_args, cosmo_exp)
    posterior_flow = load_model(likelihood, step, run_obj, parsed_run_params, classes, eval_args, cosmo_exp, global_rank=global_rank)

    nominal_design = torch.tensor(likelihood.desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=eval_args["device"], dtype=torch.float64)
    central_vals = likelihood.central_val if parsed_run_params.get("include_D_M", False) else likelihood.central_val[1::2]
    nominal_context = torch.cat([nominal_design, central_vals], dim=-1)

    np.random.seed(eval_args["eval_seed"])
    torch.manual_seed(eval_args["eval_seed"])
    torch.cuda.manual_seed(eval_args["eval_seed"])

    nominal_samples = posterior_flow(nominal_context).sample((eval_args["n_samples"],)).cpu().numpy()
    nominal_samples[:, -1] *= 100000

    with contextlib.redirect_stdout(io.StringIO()):
        samples = getdist.MCSamples(samples=nominal_samples, names=likelihood.cosmo_params, labels=likelihood.latex_labels, settings={'ignore_rows': 0.0})
    return samples

def init_likelihood(run_args, eval_args, cosmo_exp='num_tracers'):
    data_path_param = run_args.get("data_path", "") 
    data_path = home_dir + data_path_param
    
    if cosmo_exp == 'num_tracers':
        desi_df = pd.read_csv(data_path + 'desi_data.csv')
        desi_tracers = pd.read_csv(data_path + 'desi_tracers.csv')
        nominal_cov = np.load(data_path + 'desi_cov.npy')

        cosmo_model = run_args.get("cosmo_model", None)
        if cosmo_model is None:
            raise ValueError("cosmo_model is required for num_tracers")
        
        likelihood = NumTracers(
            desi_df, 
            desi_tracers,
            cosmo_model,
            nominal_cov,
            device=eval_args["device"],
            include_D_M=run_args.get("include_D_M", False) 
            )
    else:
        raise ValueError(f"{cosmo_exp} not supported")
    return likelihood
    
def eval_designs(
        designs, 
        nominal_design, 
        run_args, 
        posterior_flow, 
        likelihood, 
        target_labels, 
        n_evals=10,
        eval_particles=1000
        ):
    """
    Evaluates the EIG of the posterior flow for an input designs tensor.

    Args:
        designs (torch.Tensor): Designs to evaluate.
        nominal_design (torch.Tensor): Nominal design to evaluate.
        run_args (dict): Run arguments.

    Returns:
        tuple: A tuple containing:
            - avg_eigs (np.ndarray): Average EIGs for each design.
            - optimal_eig (float): Maximum EIG.
            - avg_nominal_eig (float): Average EIG of the nominal design.
            - optimal_design (torch.Tensor): Design with the maximum EIG.
    """
    eigs_batch = []
    nominal_eig_batch = []
    for n in range(n_evals):
        with torch.no_grad():
            agg_loss, eigs = posterior_loss(design=designs,
                                            model=likelihood.pyro_model,
                                            guide=posterior_flow,
                                            num_particles=eval_particles,
                                            observation_labels=["y"],
                                            target_labels=target_labels,
                                            evaluation=True,
                                            nflow=True,
                                            analytic_prior=False,
                                            condition_design=run_args["condition_design"])
        eigs_batch.append(eigs.cpu().numpy()/np.log(2))

        with torch.no_grad():
            agg_loss, nominal_eig = posterior_loss(design=nominal_design.unsqueeze(0),
                                            model=likelihood.pyro_model,
                                            guide=posterior_flow,
                                            num_particles=eval_particles,
                                            observation_labels=["y"],
                                            target_labels=target_labels,
                                            evaluation=True,
                                            nflow=True,
                                            analytic_prior=False,
                                            condition_design=run_args["condition_design"])
        nominal_eig_batch.append(nominal_eig.cpu().numpy()/np.log(2))

    eigs_batch = np.array(eigs_batch)
    nominal_eig_batch = np.array(nominal_eig_batch)
    # avg over the number of evaluations
    avg_eigs = np.mean(eigs_batch, axis=0)
    eigs_std = np.std(eigs_batch, axis=0)
    eigs_se = eigs_std/np.sqrt(n_evals)
    avg_nominal_eig = np.mean(nominal_eig_batch, axis=0).item()

    optimal_design = designs[np.argmax(avg_eigs)]
    print("Optimal design (NF):", optimal_design, "Optimal EIG:", np.max(avg_eigs))
    return avg_eigs, np.max(avg_eigs), avg_nominal_eig, optimal_design

def load_model(likelihood, step, run_obj, parsed_run_params, classes, eval_args, cosmo_exp='num_tracers', global_rank=0):
    # Assumes run_obj is the MLflow Run object and parsed_run_params is the output of parse_mlflow_params(run_obj.data.params)
    # classes is the already loaded content of classes.json
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"

    current_run_id = run_obj.info.run_id
    exp_id = run_obj.info.experiment_id
    
    if cosmo_exp == 'num_tracers':
        input_dim = len(likelihood.cosmo_params)
        context_dim = len(classes.keys()) + 10 if parsed_run_params.get("include_D_M", False) else len(classes.keys()) + 5
    else:
        raise ValueError(f"{cosmo_exp} not supported")
    
    nf_seed_from_params = parsed_run_params.get("nf_seed")
    nf_seed_for_init = None
    if isinstance(nf_seed_from_params, str):
        nf_seed_for_init = int(nf_seed_from_params)
    elif isinstance(nf_seed_from_params, int):
        nf_seed_for_init = nf_seed_from_params
    elif nf_seed_from_params is None:
        pass 

    posterior_flow = init_nf(
        parsed_run_params.get("flow_type"), 
        input_dim, 
        context_dim,
        parsed_run_params, 
        eval_args["device"],
        seed=nf_seed_for_init 
        )

    effective_step = step
    if step == parsed_run_params.get("total_steps"): 
        effective_step = 'last'
    
    checkpoint_dir = f'{storage_path}/mlruns/{exp_id}/{current_run_id}/artifacts/checkpoints/'
    if not os.path.isdir(checkpoint_dir):
         print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
         raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}. Ensure artifacts are downloaded or path is correct.")

    checkpoint, selected_step = get_checkpoint(effective_step, checkpoint_dir, eval_args["device"], global_rank, parsed_run_params["total_steps"])
    if selected_step != effective_step:
        print(f"Warning: Step {effective_step} not found in checkpoints. Loading checkpoint for step {selected_step} instead.")
        effective_step = selected_step
    
    posterior_flow.load_state_dict(checkpoint['model_state_dict'], strict=True)
    posterior_flow.to(eval_args["device"])
    posterior_flow.eval()
    
    return posterior_flow

def calc_entropy(design, posterior_flow, likelihood, num_samples):
    # sample values of y from likelihood.pyro_model
    expanded_design = lexpand(design.unsqueeze(0), num_samples)
    y = likelihood.pyro_model(expanded_design)
    nominal_context = torch.cat([expanded_design, y], dim=-1)
    passed_ratio = likelihood.calc_passed(expanded_design)
    constrained_parameters = likelihood.sample_valid_parameters(passed_ratio.shape[:-1])
    with pyro.plate_stack("plate", passed_ratio.shape[:-1]):
        # register samples in the trace using pyro.sample
        parameters = {}
        for k, v in constrained_parameters.items():
            # use dist.Delta to fix the value of each parameter
            parameters[k] = pyro.sample(k, dist.Delta(v)).unsqueeze(-1)
    evaluate_samples = torch.cat([parameters[k].unsqueeze(dim=-1) for k in likelihood.cosmo_params], dim=-1)
    flattened_samples = torch.flatten(evaluate_samples, start_dim=0, end_dim=len(expanded_design.shape[:-1])-1)
    samples = posterior_flow(nominal_context).log_prob(flattened_samples)
    # calculate entropy of samples
    _, entropy = _safe_mean_terms(samples)
    return entropy

def get_desi_samples(cosmo_model):
    desi_samples = np.load(f"{home_dir}/data/mcmc_samples/{cosmo_model}.npy")
    if cosmo_model == 'base':
        target_labels = ['Om', 'hrdrag']
        latex_labels = ['\Omega_m', 'H_0r_d']
    elif cosmo_model == 'base_omegak':
        target_labels = ['Om', 'Ok', 'hrdrag']
        latex_labels = ['\Omega_m', '\Omega_k', 'H_0r_d']
    elif cosmo_model == 'base_w':
        target_labels = ['Om', 'w0', 'hrdrag']
        latex_labels = ['\Omega_m', 'w_0', 'H_0r_d']
    elif cosmo_model == 'base_w_wa':
        target_labels = ['Om', 'w0', 'wa', 'hrdrag']
        latex_labels = ['\Omega_m', 'w_0', 'w_a', 'H_0r_d']
    elif cosmo_model == 'base_omegak_w_wa':
        target_labels = ['Om', 'Ok', 'w0', 'wa', 'hrdrag']
        latex_labels = ['\Omega_m', '\Omega_k', 'w_0', 'w_a', 'H_0r_d']
    with contextlib.redirect_stdout(io.StringIO()):
        desi_samples_gd = getdist.MCSamples(samples=desi_samples, names=target_labels, labels=latex_labels)
    return desi_samples_gd

def parse_mlflow_params(params_dict):
    """
    Safely parses a dictionary of MLflow parameters (string values)
    into their likely Python types (int, float, bool, str).
    """
    parsed_params = {}
    for key, value_str in params_dict.items():
        # 1. Try Boolean
        if value_str.lower() == 'true':
            parsed_params[key] = True
            continue
        if value_str.lower() == 'false':
            parsed_params[key] = False
            continue

        # 2. Try Integer
        try:
            parsed_params[key] = int(value_str)
            continue
        except ValueError:
            pass # Not an integer

        # 3. Try Float
        try:
            parsed_params[key] = float(value_str)
            continue
        except ValueError:
            pass # Not a float

        # 4. Try JSON object/list (optional, if you store complex params)
        # This handles cases where a param might be a stringified list/dict
        try:
            # Be cautious if params could contain very complex/nested JSON
            parsed_value = json.loads(value_str)
            # Only accept if it results in a list or dict (or other simple JSON types)
            if isinstance(parsed_value, (dict, list)):
                 parsed_params[key] = parsed_value
                 continue
            # else: fall through to treat as plain string if it's just a JSON string like "\"hello\""
        except json.JSONDecodeError:
            pass # Not valid JSON

        # 5. Keep as String (Fallback)
        parsed_params[key] = value_str # Keep original string if no conversion worked

    return parsed_params

def get_checkpoints(run_id, steps, checkpoint_files, type='all', cosmo_exp='num_tracers', verbose=False):
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    client = MlflowClient()
    run = client.get_run(run_id)
    run_args = parse_mlflow_params(run.data.params)
    steps = [step if step != run_args["total_steps"] else 'last' for step in steps]
    if type == 'all':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.endswith('.pt') and not f.endswith('last.pt') and not f.endswith('best.pt')
        ])
    elif type == 'area':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.startswith('checkpoint_nominal_area_')
        ])
    elif type == 'loss':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.startswith('checkpoint_loss_')
        ])
    # print stpes not found in checkpoints
    steps_not_found = [step for step in steps if step not in checkpoints]
    if verbose:
        print(f"Steps not found in checkpoints: {steps_not_found}")
    # get the checkpoints closest to the steps
    plot_checkpoints = [
        checkpoints[np.argmin(np.abs(np.array(checkpoints) - step))]
        for step in steps if step not in ['loss_best', 'nominal_area_best', 'last']
    ]

    # check for duplicates in plot_checkpoints and replace them with the next checkpoint index 
    for i, step in enumerate(plot_checkpoints):
        if step in plot_checkpoints[:i]:
            if np.argmin(np.abs(np.array(checkpoints) - steps[i])) + 1 < len(checkpoints):
                plot_checkpoints[i] = checkpoints[np.argmin(np.abs(np.array(checkpoints) - steps[i])) + 1]
            else:
                print(f"Warning: No next checkpoint found for step {steps[i]}")
                plot_checkpoints.pop(i)

    # add loss_best and nominal_area_best to the plot_checkpoints if requested
    if 'loss_best' in steps:
        plot_checkpoints.append('loss_best')
    if 'nominal_area_best' in steps:
        plot_checkpoints.append('nominal_area_best')
    if 'last' in steps:
        plot_checkpoints.append('last')

    return plot_checkpoints

def get_gpu_utilization(gpu_index=0):
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        lines = output.strip().splitlines()

        if gpu_index < len(lines):
            util_str, mem_str = lines[gpu_index].strip().split(", ")
            return int(util_str), int(mem_str)
        else:
            raise IndexError(f"GPU index {gpu_index} out of range (only {len(lines)} GPUs found).")
    except Exception as e:
        print(f"GPU stat collection failed: {e}")
        return None, None
    
def log_usage_metrics(device_str, process, step, global_rank=0):
    cpu_memory = process.memory_info().rss / 1024**2  # MB
    mlflow.log_metric("cpu_memory_usage_" + str(global_rank), cpu_memory, step=step)

    # CPU utilization (percent)
    cpu_percent = process.cpu_percent(interval=0)  # short interval for up-to-date value
    mlflow.log_metric("cpu_percent_" + str(global_rank), cpu_percent, step=step)

    # Disk I/O (per process)
    try:
        io_counters = process.io_counters()
        # Convert to MB and update metric name
        io_read_mb = io_counters.read_bytes / (1024**2)
        mlflow.log_metric("io_read_" + str(global_rank), io_read_mb, step=step)
        
        # Convert to MB and update metric name
        io_write_mb = io_counters.write_bytes / (1024**2)
        mlflow.log_metric("io_write_" + str(global_rank), io_write_mb, step=step)
    except (NotImplementedError, AttributeError) as e: # Handle systems where io_counters might not be available
        print(f"Rank {global_rank}: Could not log process I/O: {e}")

    # GPU metrics from nvidia-smi
    if torch.cuda.is_available():
        physical_gpu_id_for_nvidia_smi = None
        if isinstance(device_str, str) and device_str.startswith("cuda:"):
            try:
                logical_idx = int(device_str.split(':')[1])

                cuda_visible_devices_env = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cuda_visible_devices_env:
                    # Filter out non-integer values and handle empty strings if any
                    visible_physical_ids_str = [s.strip() for s in cuda_visible_devices_env.split(',') if s.strip()]
                    visible_physical_ids = []
                    for s_id in visible_physical_ids_str:
                        try:
                            visible_physical_ids.append(int(s_id))
                        except ValueError:
                            print(f"Rank {global_rank}: Warning: Non-integer value '{s_id}' in CUDA_VISIBLE_DEVICES='{cuda_visible_devices_env}'. Skipping.")
                    
                    if not visible_physical_ids and cuda_visible_devices_env: # If parsing resulted in empty list but env var was set
                         print(f"Rank {global_rank}: Warning: CUDA_VISIBLE_DEVICES='{cuda_visible_devices_env}' parsed to no valid IDs. Assuming logical index {logical_idx} as physical.")
                         physical_gpu_id_for_nvidia_smi = logical_idx
                    elif 0 <= logical_idx < len(visible_physical_ids):
                        physical_gpu_id_for_nvidia_smi = visible_physical_ids[logical_idx]
                    else:
                        print(f"Rank {global_rank}: Logical index {logical_idx} is out of bounds for parsed CUDA_VISIBLE_DEVICES physical IDs {visible_physical_ids}. Defaulting to logical index as physical.")
                        physical_gpu_id_for_nvidia_smi = logical_idx # Fallback
                else:
                    # CUDA_VISIBLE_DEVICES not set, logical index is the physical index
                    physical_gpu_id_for_nvidia_smi = logical_idx
            except Exception as e:
                print(f"Rank {global_rank}: Error determining physical GPU ID for nvidia-smi from device string '{device_str}': {e}. Attempting with 0.")
                physical_gpu_id_for_nvidia_smi = 0 # Fallback to 0
        
        elif device_str == "cpu":
            pass # No GPU to query with nvidia-smi
        else:
            # This case implies device_str is not "cuda:X" or "cpu"
            print(f"Rank {global_rank}: Unexpected device string '{device_str}' for nvidia-smi. If CUDA is available, attempting with GPU 0.")
            physical_gpu_id_for_nvidia_smi = 0 # Default to 0 if CUDA is available but device_str is unusual

        if physical_gpu_id_for_nvidia_smi is not None:
            gpu_util, gpu_mem_mb = get_gpu_utilization(gpu_index=physical_gpu_id_for_nvidia_smi)
            if gpu_util is not None:
                mlflow.log_metric("gpu_util_nvidia_smi_percent_" + str(global_rank), gpu_util, step=step)
            if gpu_mem_mb is not None: # gpu_mem_mb is in MiB
                mlflow.log_metric("gpu_memory_nvidia_smi_used_" + str(global_rank), gpu_mem_mb, step=step)

    # GPU metrics from PyTorch (uses the device_str like "cuda:0" directly)
    if torch.cuda.is_available() and isinstance(device_str, str) and device_str.startswith("cuda:"):
        try:
            # Log current allocated memory
            gpu_memory_allocated_mb = torch.cuda.memory_allocated(device_str) / 1024**2
            mlflow.log_metric("gpu_memory_torch_allocated_" + str(global_rank), gpu_memory_allocated_mb, step=step)
            
            # Log current reserved memory
            gpu_memory_reserved_mb = torch.cuda.memory_reserved(device_str) / 1024**2
            mlflow.log_metric("gpu_memory_torch_reserved_" + str(global_rank), gpu_memory_reserved_mb, step=step)
            
            # Log max memory allocated since last reset
            gpu_max_memory_allocated_mb = torch.cuda.max_memory_allocated(device_str) / 1024**2
            mlflow.log_metric("gpu_memory_torch_max_allocated_" + str(global_rank), gpu_max_memory_allocated_mb, step=step)

            # Calculate and log GPU memory capacity utilization (memory occupancy)
            total_gpu_memory_bytes = torch.cuda.get_device_properties(device_str).total_memory
            if total_gpu_memory_bytes > 0:
                max_allocated_bytes = torch.cuda.max_memory_allocated(device_str) 
                gpu_memory_capacity_utilization = (max_allocated_bytes / total_gpu_memory_bytes) * 100
                mlflow.log_metric("gpu_memory_capacity_utilization_" + str(global_rank), gpu_memory_capacity_utilization, step=step)
            
            torch.cuda.reset_peak_memory_stats(device_str) 

        except Exception as e:
            print(f"Rank {global_rank}: Error logging PyTorch CUDA memory metrics for device '{device_str}': {e}")
    elif device_str == "cpu" and global_rank == 0: # Log only once for CPU info or if relevant
        print(f"Rank {global_rank}: Process is on CPU. No PyTorch GPU memory metrics to log.")
        pass

def get_checkpoint(target_step, checkpoint_dir, current_pytorch_device, global_rank, total_steps):

    if target_step == 'loss_best' or target_step == 'nominal_area_best':
        checkpoint = torch.load(f"{checkpoint_dir}/checkpoint_{target_step}.pt", map_location=current_pytorch_device, weights_only=False)
        return checkpoint, target_step
    elif target_step == 'last':
        target_step = total_steps
        
    # Load rank-specific checkpoints (default behavior)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    rank_checkpoint_pattern = f"checkpoint_rank_{global_rank}_"
    rank_checkpoint_files = [f for f in checkpoint_files if f.startswith(rank_checkpoint_pattern)]

    if rank_checkpoint_files:
        # Find the checkpoint closest to target_step
        rank_checkpoint_steps = []
        for f in rank_checkpoint_files:
            try:
                if 'last' in f:
                    step = total_steps
                else:
                    step = int(f.split('_')[-1].split('.')[0])
                rank_checkpoint_steps.append((step, f))
            except ValueError:
                continue
        
        rank_checkpoint_steps.sort()
        
        # Find checkpoint closest to target_step
        closest_idx = 0
        if len(rank_checkpoint_steps) > 1:
            if target_step == float('inf'):
                # For 'latest', take the highest step
                closest_idx = len(rank_checkpoint_steps) - 1
            else:
                closest_idx = np.argmin(np.abs(np.array([s[0] for s in rank_checkpoint_steps]) - target_step))
        
        selected_step, selected_file = rank_checkpoint_steps[closest_idx]
        checkpoint_path = f"{checkpoint_dir}/{selected_file}"
        
        if global_rank == 0:
            print(f"Loading rank-specific checkpoints:")
            print(f"  Requested step: {target_step}")
            print(f"  Available steps: {[s[0] for s in rank_checkpoint_steps]}")
            print(f"  Selected step: {selected_step}")

    else:
        # Fallback to shared checkpoint if no rank-specific ones found
        # Find the checkpoint closest to target_step
        shared_checkpoint_steps = []
        for f in checkpoint_files:
            if f.startswith('checkpoint_') and f.endswith('.pt') and not f.startswith('checkpoint_rank_'):
                try:
                    step = int(f.split('_')[-1].split('.')[0])
                    shared_checkpoint_steps.append((step, f))
                except ValueError:
                    continue
        
        shared_checkpoint_steps.sort()
        
        # Find checkpoint closest to target_step
        closest_idx = 0
        if len(shared_checkpoint_steps) > 1:
            if target_step == float('inf'):
                # For 'latest', take the highest step
                closest_idx = len(shared_checkpoint_steps) - 1
            else:
                closest_idx = np.argmin(np.abs(np.array([s[0] for s in shared_checkpoint_steps]) - target_step))
        
        selected_step, selected_file = shared_checkpoint_steps[closest_idx]
        checkpoint_path = f"{checkpoint_dir}/{selected_file}"
        
        if global_rank == 0:
            print(f"No rank-specific checkpoints found, using shared checkpoint")
            print(f"  Requested step: {target_step}")
            print(f"  Available steps: {[s[0] for s in shared_checkpoint_steps]}")
            print(f"  Selected step: {selected_step}")
            print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=current_pytorch_device, weights_only=False)
    return checkpoint, selected_step

def get_runtime(run_id):
    run = mlflow.get_run(run_id)

    start_ms = run.info.start_time      # epoch ms
    end_ms   = run.info.end_time        # epoch ms (None until the run is finished)

    if end_ms is None:
        raise ValueError(f"Run {run_id} is still running or did not record end_time.")

    # Convert epoch-ms → Python datetime
    start_dt = datetime.fromtimestamp(start_ms / 1_000)
    end_dt   = datetime.fromtimestamp(end_ms   / 1_000)
    return end_dt - start_dt

def validate_checkpoint_compatibility(checkpoint, run_args, mode="restart", global_rank=0):
    """
    Validate checkpoint compatibility with current run configuration.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        run_args (dict): Current run arguments
        mode (str): Either "resume" or "restart"
    
    Returns:
        dict: Information about the checkpoint and any compatibility issues
    """
    try:
        info = {
            'mode': mode,
            'step': checkpoint.get('step', 'unknown'),
            'has_model_state': 'model_state_dict' in checkpoint,
            'has_optimizer_state': 'optimizer_state_dict' in checkpoint,
            'has_scheduler_state': 'scheduler_state_dict' in checkpoint,
            'has_rng_state': 'rng_state' in checkpoint,
            'additional_state': checkpoint.get('additional_state', {}),
            'warnings': [],
            'errors': []
        }
        
        # Check for required components
        if not info['has_model_state']:
            info['errors'].append("Checkpoint missing model state")
        
        if not info['has_optimizer_state']:
            info['warnings'].append("Checkpoint missing optimizer state")
        
        # For resume mode, scheduler state is important
        if mode == "resume" and not info['has_scheduler_state']:
            info['warnings'].append("Checkpoint missing scheduler state (may affect learning rate schedule)")
        
        # Check rank information if available
        if 'global_rank' in info['additional_state']:
            info['rank'] = info['additional_state']['global_rank']
        else:
            info['warnings'].append("Checkpoint missing rank information")
        
        # For restart mode, check if learning rate will be updated
        if mode == "restart" and info['has_optimizer_state']:
            try:
                old_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
                new_lr = run_args.get('initial_lr', 'unknown')
                info['lr_change'] = f"{old_lr} -> {new_lr}"
                if old_lr != new_lr:
                    info['warnings'].append(f"Learning rate will be updated: {old_lr} -> {new_lr}")
            except (KeyError, IndexError):
                info['warnings'].append("Could not determine learning rate change")

        # Verify rank information if available
        if 'additional_state' in checkpoint and 'global_rank' in checkpoint['additional_state']:
            checkpoint_rank = checkpoint['additional_state']['global_rank']
            if checkpoint_rank != global_rank:
                if global_rank == 0:
                    print(f"Warning: Checkpoint was saved by rank {checkpoint_rank} but being loaded by rank {global_rank}")
            else:
                if global_rank == 0:
                    print(f"Checkpoint rank verification passed: rank {global_rank}")
        else:
            if global_rank == 0:
                print(f"Note: Checkpoint does not contain rank information")
        
        print_checkpoint_info(info, global_rank)
        return info
        
    except Exception as e:
        return {
            'mode': mode,
            'errors': [f"Failed to load checkpoint: {str(e)}"],
            'warnings': []
        }

def print_checkpoint_info(checkpoint_info, global_rank=0):
    """
    Print checkpoint validation information.
    
    Args:
        checkpoint_info (dict): Output from validate_checkpoint_compatibility
        global_rank (int): Global rank for distributed training
    """
    if global_rank != 0:
        return
    
    print(f"\n=== Checkpoint Validation ({checkpoint_info['mode'].upper()} MODE) ===")
    print(f"Step: {checkpoint_info['step']}")
    
    if 'rank' in checkpoint_info:
        print(f"Rank: {checkpoint_info['rank']}")
    
    if 'lr_change' in checkpoint_info:
        print(f"Learning Rate: {checkpoint_info['lr_change']}")
    
    print(f"Components:")
    print(f"  Model State: {'pass' if checkpoint_info['has_model_state'] else 'fail'}")
    print(f"  Optimizer State: {'pass' if checkpoint_info['has_optimizer_state'] else 'fail'}")
    print(f"  Scheduler State: {'pass' if checkpoint_info['has_scheduler_state'] else 'fail'}")
    print(f"  RNG State: {'pass' if checkpoint_info['has_rng_state'] else 'fail'}")
    
    if checkpoint_info['warnings']:
        print(f"\nWarnings:")
        for warning in checkpoint_info['warnings']:
            print(f"  Warning: {warning}")
    
    if checkpoint_info['errors']:
        print(f"\nErrors:")
        for error in checkpoint_info['errors']:
            print(f"  ERROR: {error}")
    
    if not checkpoint_info['errors']:
        print(f"\nCheckpoint validation passed")
    
    print("=" * 50)

def print_training_state(model, optimizer, step):
    print(f"=== OPTIMIZER STATE AT STEP {step} ===")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Number of parameter groups: {len(optimizer.param_groups)}")
    
    # Print optimizer state for each parameter group
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"\nParameter Group {i}:")
        print(f"  Learning rate: {param_group['lr']}")
        print(f"  Weight decay: {param_group.get('weight_decay', 0)}")
        print(f"  Number of parameters: {len(param_group['params'])}")
        
        # Print state for first few parameters in this group
        for j, param in enumerate(param_group['params'][:3]):  # Show first 3 params
            if param in optimizer.state:
                state = optimizer.state[param]
                print(f"    Param {j} state keys: {list(state.keys())}")
                if 'exp_avg' in state:
                    print(f"      exp_avg shape: {state['exp_avg'].shape}")
                    print(f"      exp_avg mean: {state['exp_avg'].mean().item():.6f}")
                if 'exp_avg_sq' in state:
                    print(f"      exp_avg_sq shape: {state['exp_avg_sq'].shape}")
                    print(f"      exp_avg_sq mean: {state['exp_avg_sq'].mean().item():.6f}")
                if 'step' in state:
                    print(f"      step: {state['step']}")
            else:
                print(f"    Param {j}: No state stored")
    
    # Print model parameter statistics
    print(f"\nMODEL PARAMETER STATISTICS:")
    total_params = 0
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            if total_params <= 1000000:  # Only print for reasonable number of params
                print(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    print(f"Total trainable parameters: {total_params}")
    print("=" * 50)
