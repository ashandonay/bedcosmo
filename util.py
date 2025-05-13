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

torch.set_default_dtype(torch.float64)

home_dir = os.environ["HOME"]
if home_dir + "/bed/BED_cosmo" not in sys.path:
    sys.path.insert(0, home_dir + "/bed/BED_cosmo")
sys.path.insert(0, home_dir + "/bed/BED_cosmo/num_tracers")

def auto_seed(seed):
    if seed < 0:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    # Set all relevant seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For completely deterministic results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    pyro.set_rng_seed(seed)
    return seed

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

def run_eval(run_id, eval_args, step='best_loss', cosmo_exp='num_tracers'):
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
        area_checkpoints = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/') if f.startswith('checkpoint_nominal_area_') and f.endswith('.pt') and 'best' not in f]
        loss_checkpoints = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/') if f.startswith('checkpoint_loss_') and f.endswith('.pt') and 'best' not in f]
        regular_checkpoints = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/') if f.startswith('checkpoint_') and f.endswith('.pt') and not f.endswith('last.pt') and 'loss' not in f and 'area' not in f]
        if step not in area_checkpoints and step not in loss_checkpoints and step not in regular_checkpoints and step != 'loss_best' and step != 'best_nominal_area' and step != 'last':
            raise ValueError(f"Step {step} not found in checkpoints")
        if step in area_checkpoints:
            checkpoint = torch.load(f'{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/checkpoint_area_{step}.pt', map_location=eval_args["device"])
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

def get_checkpoints(steps, checkpoint_files, type='all', cosmo_exp='num_tracers', verbose=False):
    if type == 'all':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.startswith('nf_') and f.endswith('.pt') and not f.endswith('last.pt') and not f.endswith('loss.pt') and not f.endswith('area.pt')
        ])
    elif type == 'area':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.startswith('nf_area')
        ])
    elif type == 'loss':
        checkpoints = sorted([
            int(f.split('_')[-1].split('.')[0]) 
            for f in checkpoint_files 
            if f.startswith('nf_loss')
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
    
def log_usage_metrics(device, process, step):
    cpu_memory = process.memory_info().rss / 1024**2  # MB
    mlflow.log_metric("cpu_memory_usage", cpu_memory, step=step)

    # CPU utilization (percent)
    cpu_percent = process.cpu_percent(interval=0)  # short interval for up-to-date value
    mlflow.log_metric("cpu_percent", cpu_percent, step=step)

    # Disk I/O (per process)
    try:
        io_counters = process.io_counters()
        mlflow.log_metric("io_read_bytes", io_counters.read_bytes, step=step)
        mlflow.log_metric("io_write_bytes", io_counters.write_bytes, step=step)
    except Exception as e:
        print(f"Could not log process I/O: {e}")

    gpu_index = device.index if hasattr(device, 'index') and device.index is not None else 0
    gpu_util, gpu_mem = get_gpu_utilization(gpu_index)
    if gpu_util is not None:
        mlflow.log_metric("gpu_util", gpu_util, step=step)
        mlflow.log_metric("gpu_memory_total_usage", gpu_mem * 1024**2 / 1000000, step=step)

    if torch.cuda.is_available():
        with torch.cuda.device(device):
            gpu_memory = torch.cuda.memory_allocated(device) / 1024**2
            mlflow.log_metric("gpu_memory_torch_usage", gpu_memory, step=step)
            gpu_reserved = torch.cuda.memory_reserved(device) / 1024**2
            mlflow.log_metric("gpu_memory_torch_reserved", gpu_reserved, step=step)
            gpu_peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
            mlflow.log_metric("gpu_memory_torch_peak", gpu_peak_memory, step=step)