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

torch.set_default_dtype(torch.float64)

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

def save_checkpoint(model, optimizer, filepath):
    """
    Saves the training checkpoint.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        filepath (str): Path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)
    mlflow.log_artifact(filepath)  # Logs the checkpoint to mlflow

def init_nf(flow_type, input_dim, context_dim, n_transforms, device, seed=None, **kwargs):
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

    # Initialize the flow model
    if flow_type == "NSF":
        posterior_flow = zuko.flows.NSF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms, 
            bins=10,
            **kwargs
        ).to(device)
    elif flow_type == "NAF":
        posterior_flow = zuko.flows.NAF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            network={**kwargs}
        ).to(device)
    elif flow_type == "MAF":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            **kwargs
        ).to(device)
    elif flow_type == "MAF_Affine":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            univariate=zuko.transforms.MonotonicAffineTransform,
            **kwargs
        ).to(device)
    elif flow_type == "MAF_RQS":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            univariate=zuko.transforms.MonotonicRQSTransform,
            shapes = ([kwargs["shape"]], [kwargs["shape"]], [kwargs["shape"]-1]),
            **kwargs
        ).to(device)
    elif flow_type == "NICE":
        posterior_flow = zuko.flows.NICE(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            **kwargs
        ).to(device)
    elif flow_type == "GF":
        posterior_flow = zuko.flows.GF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms
        ).to(device)
    return posterior_flow

def run_eval(run_id, eval_args, step='last', device='cuda:0'):
    client = MlflowClient()
    run = client.get_run(run_id)
    exp_id = run.info.experiment_id
    with mlflow.start_run(run_id=run_id, nested=True):
        run = client.get_run(run_id)
        ml_info = mlflow.active_run().info

        #tracers = eval(run.data.params["tracers"])
        data_path = run.data.params["data_path"]
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
            eff=eval(run.data.params["eff"]), 
            include_D_M=eval(run.data.params["include_D_M"])
            )

        input_dim = len(num_tracers.cosmo_params)
        context_dim = len(classes.keys()) + 10 if eval(run.data.params["include_D_M"]) else len(classes.keys()) + 5
        n_transforms = eval(run.data.params["n_transforms"])
        hidden_features = (eval(run.data.params["hidden"]),) + (2*eval(run.data.params["hidden"]),) * (eval(run.data.params["num_layers"]) - 1)
        n_transforms = eval(run.data.params["n_transforms"])

        # Initialize model without seed (since we're loading state dict)
        posterior_flow = init_nf(
            run.data.params["flow_type"],
            input_dim, 
            context_dim, 
            n_transforms, 
            device, 
            hidden_features=hidden_features
            )
        
        checkpoint = torch.load(f'../mlruns/{exp_id}/{run_id}/artifacts/nf_checkpoint_{step}.pt', map_location=eval_args["device"])
        posterior_flow.load_state_dict(checkpoint['model_state_dict'], strict=True)
        posterior_flow.to(eval_args["device"])
        posterior_flow.eval()

        nominal_design = torch.tensor(desi_tracers.groupby('class').sum()['observed'].reindex(classes.keys()).values, device=device, dtype=torch.float64)
        central_vals = num_tracers.central_val if eval(run.data.params["include_D_M"]) else num_tracers.central_val[1::2]
        nominal_context = torch.cat([nominal_design, central_vals], dim=-1)

        # set random seed
        np.random.seed(eval_args["eval_seed"])
        torch.manual_seed(eval_args["eval_seed"])
        torch.cuda.manual_seed(eval_args["eval_seed"])

        nominal_samples = posterior_flow(nominal_context).sample((eval_args["post_samples"],)).cpu().numpy()
        nominal_samples[:, -1] *= 100000
        
        # Temporarily redirect stdout to suppress GetDist messages
        with contextlib.redirect_stdout(io.StringIO()):
            samples = getdist.MCSamples(samples=nominal_samples, names=num_tracers.cosmo_params, labels=num_tracers.latex_labels, settings={'ignore_rows': 0.0})
    return samples