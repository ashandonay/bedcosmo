import torch
import pyro
import mlflow


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
    print(f"Checkpoint saved at {filepath}")