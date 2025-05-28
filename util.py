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
import datetime

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

def print_memory_usage(process, step):
    mem_info = process.memory_info()
    print(f"Step {step}: Memory Usage: {mem_info.rss / 1024**2:.2f} MB")

def save_checkpoint(model, optimizer, filepath, step=None, artifact_path=None, scheduler=None):
    """
    Saves the training checkpoint.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        filepath (str): Path to save the checkpoint.
        step (int, optional): Current training step.
        loss_best (float, optional): Best loss value seen so far.
        nominal_area_best (float, optional): Best area value seen so far.
        history (list, optional): Training loss history.
        artifact_path (str, optional): Path to log the artifact to in MLflow.
    """
    # Get the state dict from the model, handling DDP wrapper if present
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict()
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if step is not None:
        checkpoint['step'] = step

    # Save RNG states correctly
    checkpoint['rng_state'] = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'pyro': pyro.get_param_store().get_state(),
        'cuda': [state for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    }
    
    # Log the checkpoint to mlflow
    torch.save(checkpoint, filepath)
    mlflow.log_artifact(filepath, artifact_path=artifact_path)

def init_training_env(tdist, device):
        # DDP initialization
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"]) # SLURM's local rank
        global_rank = int(os.environ["RANK"])
        
        # When CUDA_VISIBLE_DEVICES isolates one GPU, PyTorch sees it as device 0.
        pytorch_device_idx = int(os.environ["LOCAL_RANK"])  # The only GPU visible to this process
        effective_device_id = pytorch_device_idx
        torch.cuda.set_device(pytorch_device_idx)
        
        tdist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=global_rank,
            timeout=datetime.timedelta(seconds=180)  # Increased timeout
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
    return global_rank, local_rank, effective_device_id, pytorch_device_idx

def init_nf(flow_type, input_dim, context_dim, run_args, device="cuda:0", seed=None, verbose=False, local_rank=None, **kwargs):
    # Set seeds first, before any model initialization
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For completely deterministic results
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if run_args["hidden_size"] is not None and local_rank is not None and local_rank == 0 and verbose:
        print(f'Hidden features: {(run_args["hidden_size"],) * run_args["n_layers"]}')

    # Initialize the flow model
    if flow_type == "NSF":
        posterior_flow = zuko.flows.NSF(
            features=input_dim, 
            context=context_dim, 
            transforms=run_args["n_transforms"], 
            bins=run_args["bins"],
            hidden_features=((run_args["hidden_size"],) * run_args["n_layers"]),
            **kwargs
        )
    elif flow_type == "NAF":
        posterior_flow = zuko.flows.NAF(
            features=input_dim, 
            context=context_dim, 
            transforms=run_args["n_transforms"],
            signal=run_args["signal"],
            network={"hidden_features": ((run_args["hidden_size"],) * run_args["n_layers"])}
        )
    elif flow_type == "MAF":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=run_args["n_transforms"],
            hidden_features=((run_args["hidden_size"],) * run_args["n_layers"]),
            **kwargs
        )
    elif flow_type == "MAF_Affine":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=run_args["n_transforms"],
            univariate=zuko.transforms.MonotonicAffineTransform,
            **kwargs
        )
    elif flow_type == "MAF_RQS":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=run_args["n_transforms"],
            univariate=zuko.transforms.MonotonicRQSTransform,
            shapes=([run_args["shape"]], [run_args["shape"]], [run_args["shape"]-1]),
            **kwargs
        )
    elif flow_type == "NICE":
        posterior_flow = zuko.flows.NICE(
            features=input_dim, 
            context=context_dim, 
            transforms=run_args["n_transforms"],
            **kwargs
        )
    elif flow_type == "GF":
        posterior_flow = zuko.flows.GF(
            features=input_dim, 
            context=context_dim, 
            transforms=run_args["n_transforms"]
        )
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")

    # Move to the correct device
    posterior_flow.to(device)

    # Wrap in DDP if using distributed training
    if local_rank is not None:
        posterior_flow = DDP(posterior_flow, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    return posterior_flow

def init_scheduler(optimizer, run_args):
    # Setup
    steps_per_cycle = run_args["steps"] // run_args["n_cycles"]
    initial_lr = run_args["initial_lr"]
    final_lr = run_args["final_lr"]

    if run_args["scheduler_type"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=run_args["steps"],
            eta_min=final_lr
            )
    elif run_args["scheduler_type"] == "linear":
        # factor is the number we multiply the initial lr by to get the final lr at each step
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=final_lr / initial_lr, 
            total_iters=run_args["steps"] - 1
            )
    elif run_args["scheduler_type"] == "exponential":
        # calculate gamma from initial and final lr
        gamma = (final_lr / initial_lr) ** (1 / run_args["steps"])
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

def init_run(tdist, global_rank, current_pytorch_device, storage_path, mlflow_experiment_name, cosmo_model, run_args, kwargs, resume_id=None, resume_step=None, add_steps=0):
        """Initialize MLflow run settings and broadcast to all ranks."""
        
        if global_rank == 0:
            if resume_id:
                # Resume existing run
                client = mlflow.MlflowClient()
                run_info = client.get_run(resume_id)
                exp_id = run_info.info.experiment_id
                exp_name = mlflow.get_experiment(exp_id).name
                
                mlflow.set_experiment(experiment_id=exp_id)
                mlflow.start_run(run_id=resume_id)
                
                if resume_step is None:
                    raise ValueError("resume_step must be provided when resuming a run")
                
                # Update run parameters
                mlflow_experiment_name = exp_name
                cosmo_model = run_info.data.params["cosmo_model"]
                run_args = parse_mlflow_params(run_info.data.params)
                if add_steps:
                    run_args["steps"] += add_steps
                
                n_devices = tdist.get_world_size() if "LOCAL_RANK" in os.environ else 1
                if run_args["n_particles"] != n_devices * run_args["n_particles_per_device"]:
                    raise ValueError(f"n_particles ({run_args['n_particles']}) must be equal to n_devices * n_particles_per_device ({n_devices * run_args['n_particles_per_device']})")
                
                # Get metrics from previous run
                best_nominal_area, best_loss, start_step = _get_resume_metrics(
                    client, resume_id, resume_step, storage_path, exp_id
                )

                # Prepare tensors for broadcasting
                tensors = _prepare_broadcast_tensors(
                    exp_id, start_step, best_loss, best_nominal_area, 
                    current_pytorch_device, resume_id, exp_name
                )
            else:
                # Start new run
                mlflow.set_experiment(mlflow_experiment_name)
                mlflow.start_run()
                
                n_devices = tdist.get_world_size() if "LOCAL_RANK" in os.environ else 1
                # Log parameters
                mlflow.log_param("cosmo_model", cosmo_model)
                for key, value in run_args.items():
                    mlflow.log_param(key, value)
                for key, value in kwargs.items():
                    mlflow.log_param(key, value)
                mlflow.log_param("n_devices", n_devices)
                mlflow.log_param("n_particles", n_devices * run_args["n_particles_per_device"])
                
                # Initialize metrics
                start_step = 0
                best_loss = np.nan
                best_nominal_area = np.nan
                
                
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
        
        _broadcast_variables(tensors, global_rank, run_args, tdist)

        # Set up MLflow for non-zero ranks
        if global_rank != 0:
            mlflow.set_experiment(experiment_id=str(tensors['exp_id'].item()))
            mlflow.start_run(run_id=tensors['run_id'][0], nested=True)
        
        # Create ml_info object
        ml_info = type('mlinfo', (), {})()
        ml_info.experiment_id = str(tensors['exp_id'].item())
        ml_info.run_id = tensors['run_id'][0]

        # All ranks join the same run
        if resume_id:
            mlflow.start_run(run_id=resume_id, nested=True)
        else:
            mlflow.start_run(run_id=ml_info.run_id, nested=True)

        return ml_info, tensors['start_step'].item(), tensors['best_loss'].item(), tensors['best_nominal_area'].item()

def _broadcast_variables(tensors, global_rank, run_args,tdist):

    # Broadcast tensors from rank 0 to all ranks
    tdist.barrier()
    tdist.broadcast(tensors['exp_id'], src=0)
    tdist.broadcast(tensors['start_step'], src=0)
    tdist.broadcast(tensors['best_loss'], src=0)
    tdist.broadcast(tensors['best_nominal_area'], src=0)
    tdist.broadcast_object_list(tensors['run_id'], src=0)
    tdist.broadcast_object_list(tensors['exp_name'], src=0)

def _get_resume_metrics(client, resume_id, resume_step, storage_path, exp_id):
    """Get metrics from previous run for resuming."""
    best_nominal_areas = client.get_metric_history(resume_id, 'best_nominal_area')
    best_nominal_area_steps = np.array([metric.step for metric in best_nominal_areas])
    if len(best_nominal_area_steps) > 0:
        closest_idx = np.argmin(np.abs(best_nominal_area_steps - resume_step))
        best_nominal_area = best_nominal_areas[closest_idx].value if best_nominal_area_steps[closest_idx] < resume_step else best_nominal_areas[closest_idx - 1].value
    else:
        best_nominal_area = np.nan
    
    best_losses = client.get_metric_history(resume_id, 'best_loss')
    best_loss_steps = np.array([metric.step for metric in best_losses])
    closest_idx = np.argmin(np.abs(best_loss_steps - resume_step))
    best_loss = best_losses[closest_idx].value if best_loss_steps[closest_idx] < resume_step else best_losses[closest_idx - 1].value
    
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
    closest_idx = np.argmin(np.abs(np.array(checkpoint_steps) - resume_step))
    start_step = checkpoint_steps[closest_idx] if checkpoint_steps[closest_idx] < resume_step else checkpoint_steps[closest_idx - 1]
    
    return best_nominal_area, best_loss, start_step

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

def run_eval(run_id, eval_args, step='loss_best', cosmo_exp='num_tracers'):
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    client = MlflowClient()
    run = client.get_run(run_id)
    num_tracers, posterior_flow = load_model(run_id, step, eval_args, cosmo_exp)

    with open(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="classes.json")) as f:
        classes = json.load(f)

    nominal_design = torch.tensor(num_tracers.desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=eval_args["device"], dtype=torch.float64)
    central_vals = num_tracers.central_val if eval(run.data.params["include_D_M"]) else num_tracers.central_val[1::2]
    nominal_context = torch.cat([nominal_design, central_vals], dim=-1)

    # set random seed
    np.random.seed(eval_args["eval_seed"])
    torch.manual_seed(eval_args["eval_seed"])
    torch.cuda.manual_seed(eval_args["eval_seed"])

    nominal_samples = posterior_flow(nominal_context).sample((eval_args["n_samples"],)).cpu().numpy()
    nominal_samples[:, -1] *= 100000

    #entropy = calc_entropy(nominal_design, posterior_flow, num_tracers, eval_args["n_samples"])
    """
    _, loss = posterior_loss(
        nominal_design.unsqueeze(0),
        num_tracers.pyro_model,
        posterior_flow,
        eval_args["n_samples"],
        observation_labels=["y"],
        target_labels=num_tracers.cosmo_params,
        evaluation=False,
        nflow=True,
        condition_design=True,
        verbose_shapes=False
        )
    print(f"Loss: {loss.cpu().item()}")
    """

    # Temporarily redirect stdout to suppress GetDist messages
    with contextlib.redirect_stdout(io.StringIO()):
        samples = getdist.MCSamples(samples=nominal_samples, names=num_tracers.cosmo_params, labels=num_tracers.latex_labels, settings={'ignore_rows': 0.0})
    return samples

def load_model(run_id, step, eval_args, cosmo_exp='num_tracers'):
    storage_path = os.environ["SCRATCH"] + f"/bed/BED_cosmo/{cosmo_exp}"
    mlflow.set_tracking_uri(storage_path + "/mlruns")
    client = MlflowClient()
    run = client.get_run(run_id)
    exp_id = run.info.experiment_id
    with mlflow.start_run(run_id=run_id, nested=True):
        run = client.get_run(run_id)
        ml_info = mlflow.active_run().info

        #tracers = eval(run.data.params["tracers"])
        data_path = home_dir + run.data.params["data_path"]
        desi_df = pd.read_csv(data_path + 'desi_data.csv')
        desi_tracers = pd.read_csv(data_path + 'desi_tracers.csv')
        nominal_cov = np.load(data_path + 'desi_cov.npy')
        cosmo_model = run.data.params["cosmo_model"]
        
        with open(mlflow.artifacts.download_artifacts(run_id=ml_info.run_id, artifact_path="classes.json")) as f:
            classes = json.load(f)

        num_tracers = NumTracers(
            desi_df, 
            desi_tracers,
            cosmo_model,
            nominal_cov,
            device=eval_args["device"],
            include_D_M=eval(run.data.params["include_D_M"])
            )

        input_dim = len(num_tracers.cosmo_params)
        context_dim = len(classes.keys()) + 10 if eval(run.data.params["include_D_M"]) else len(classes.keys()) + 5
        # Initialize model without seed (since we're loading state dict)
        run_args = parse_mlflow_params(run.data.params)
        posterior_flow = init_nf(
            run.data.params["flow_type"],
            input_dim, 
            context_dim,
            run_args,
            eval_args["device"],
            seed=eval(run.data.params["nf_seed"])
            )
        if step == run_args["steps"]:
            step = 'last'
        area_checkpoints = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/') if f.startswith('checkpoint_nominal_area_') and f.endswith('.pt') and 'best' not in f]
        loss_checkpoints = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/') if f.startswith('checkpoint_loss_') and f.endswith('.pt') and 'best' not in f]
        regular_checkpoints = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/') if f.startswith('checkpoint_') and f.endswith('.pt') and not f.endswith('last.pt') and 'loss' not in f and 'area' not in f]
        if step not in area_checkpoints and step not in loss_checkpoints and step not in regular_checkpoints and step != 'loss_best' and step != 'nominal_area_best' and step != 'last':
            raise ValueError(f"Step {step} not found in checkpoints")
        if step in area_checkpoints:
            checkpoint = torch.load(f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/checkpoint_nominal_area_{step}.pt', map_location=eval_args["device"])
        elif step in loss_checkpoints:
            checkpoint = torch.load(f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/checkpoint_loss_{step}.pt', map_location=eval_args["device"])
        else:
            checkpoint = torch.load(f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/checkpoint_{step}.pt', map_location=eval_args["device"])
        
        # Get the state dict and handle module prefixes
        state_dict = checkpoint['model_state_dict']
        
        # Remove module prefixes if they exist
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefixes
            if k.startswith('module.'):
                k = k[7:]  # Remove 'module.'
            if k.startswith('module.'):  # Handle double module prefix
                k = k[7:]  # Remove second 'module.'
            new_state_dict[k] = v
            
        # Load the cleaned state dict
        posterior_flow.load_state_dict(new_state_dict, strict=True)
        posterior_flow.to(eval_args["device"])
        posterior_flow.eval()
        return num_tracers, posterior_flow

def calc_entropy(design, posterior_flow, num_tracers, num_samples):
    # sample values of y from num_tracers.pyro_model
    expanded_design = lexpand(design.unsqueeze(0), num_samples)
    y = num_tracers.pyro_model(expanded_design)
    nominal_context = torch.cat([expanded_design, y], dim=-1)
    passed_ratio = num_tracers.calc_passed(expanded_design)
    constrained_parameters = num_tracers.sample_valid_parameters(passed_ratio.shape[:-1])
    with pyro.plate_stack("plate", passed_ratio.shape[:-1]):
        # register samples in the trace using pyro.sample
        parameters = {}
        for k, v in constrained_parameters.items():
            # use dist.Delta to fix the value of each parameter
            parameters[k] = pyro.sample(k, dist.Delta(v)).unsqueeze(-1)
    evaluate_samples = torch.cat([parameters[k].unsqueeze(dim=-1) for k in num_tracers.cosmo_params], dim=-1)
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
    steps = [step if step != run_args["steps"] else 'last' for step in steps]
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
    
def log_usage_metrics(device_str, process, step, global_rank=0): # Renamed device to device_str for clarity
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