import torch
import pyro
import mlflow
import zuko

def auto_seed(seed):
    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
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

def init_nf(flow_type, input_dim, context_dim, n_transforms, device, **kwargs):
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
    elif flow_type == "affine_coupling":
        posterior_flow, transforms = pyro_flows.affine_coupling_flow(n_transforms, input_dim, context_dim, [32, 32], device)
        modules = torch.nn.ModuleList(transforms)
    elif flow_type == "GF":
        posterior_flow = zuko.flows.GF(
            features=input_dim, 
            context=context_dim, 
            transforms=n_transforms
        ).to(device)
    return posterior_flow