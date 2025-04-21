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

def save_checkpoint(model, optimizer, filepath, artifact_path=None):
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
    mlflow.log_artifact(filepath, artifact_path=artifact_path)  # Logs the checkpoint to mlflow

def init_nf(flow_type, input_dim, context_dim, n_transforms, hidden_size=None, n_layers=None, device="cuda:0", seed=None, verbose=False, **kwargs):
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

    if hidden_size is not None and verbose:
        print(f'Hidden features: {(hidden_size,) * n_layers}')

    # Initialize the flow model
    if flow_type == "NSF":
        posterior_flow = zuko.flows.NSF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms, 
            bins=10,
            hidden_features=((hidden_size,) * n_layers),
            **kwargs
        ).to(device)
    elif flow_type == "NAF":
        posterior_flow = zuko.flows.NAF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            network={"hidden_features": ((hidden_size,) * n_layers)} # (hidden_size,) + (2*hidden_size,) * (n_layers - 1)} 
        ).to(device)
    elif flow_type == "MAF":
        posterior_flow = zuko.flows.MAF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms,
            hidden_features=((hidden_size,) * n_layers),
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

def run_eval(run_id, eval_args, step='best_loss', device='cuda:0', cosmo_exp='num_tracers', best=False):
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
            eff=eval(run.data.params["eff"]), 
            include_D_M=eval(run.data.params["include_D_M"])
            )

        input_dim = len(num_tracers.cosmo_params)
        context_dim = len(classes.keys()) + 10 if eval(run.data.params["include_D_M"]) else len(classes.keys()) + 5
        # Initialize model without seed (since we're loading state dict)
        posterior_flow = init_nf(
            run.data.params["flow_type"],
            input_dim, 
            context_dim, 
            eval(run.data.params["n_transforms"]),
            eval(run.data.params["hidden"]),
            eval(run.data.params["num_layers"]),
            device,
            seed=eval(run.data.params["nf_seed"])
            )

        if best:
            checkpoint = torch.load(f'{home_dir}/bed/BED_cosmo/{cosmo_exp}/mlruns/{exp_id}/{run_id}/artifacts/best_loss/nf_checkpoint_{step}.pt', map_location=eval_args["device"])
        else:
            checkpoint = torch.load(f'{home_dir}/bed/BED_cosmo/{cosmo_exp}/mlruns/{exp_id}/{run_id}/artifacts/checkpoints/nf_checkpoint_{step}.pt', map_location=eval_args["device"])
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

        #entropy = calc_entropy(nominal_design, posterior_flow, num_tracers, eval_args["post_samples"])
        """
        _, loss = posterior_loss(
            nominal_design.unsqueeze(0),
            num_tracers.pyro_model,
            posterior_flow,
            eval_args["post_samples"],
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