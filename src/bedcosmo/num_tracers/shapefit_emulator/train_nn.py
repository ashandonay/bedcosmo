import argparse
import os
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import mlflow
import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from model import NNRegressor
from prep_shapefit_data import get_default_save_path, _next_version
from eval_nn import run_eval

def load_data(data_path: str) -> Dict[str, np.ndarray]:
    train = np.load(f"{data_path}/train.npz", allow_pickle=True)
    test = np.load(f"{data_path}/test.npz", allow_pickle=True)
    data = {
        "x_train": train["x"].astype(np.float32),
        "y_train": train["y"].astype(np.float32),
        "x_test": test["x"].astype(np.float32),
        "y_test": test["y"].astype(np.float32),
        "param_names": train["param_names"],
        "target_names": train["target_names"],
    }
    _drop_nonfinite_rows(data)
    return data

def create_scheduler(optimizer, run_args):
    # Setup
    steps_per_cycle = run_args["total_steps"] // run_args["n_cycles"]
    initial_lr = run_args["initial_lr"]
    final_lr = run_args["final_lr"]
    
    # Get warmup fraction, defaulting to 0.0 (no warmup)
    warmup_fraction = run_args.get("warmup_fraction", 0.0)
    warmup_steps = int(warmup_fraction * run_args["total_steps"])

    if run_args["scheduler_type"] == "constant":
        if warmup_steps > 0:
            # Create a warmup + constant schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to initial_lr
                    return step / warmup_steps
                else:
                    # Constant at initial_lr
                    return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, 
                factor=1.0
                )
    elif run_args["scheduler_type"] == "cosine":
        if warmup_steps > 0:
            # Create a warmup + cosine schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to initial_lr
                    return step / warmup_steps
                else:
                    # Cosine decay from initial_lr to final_lr
                    adjusted_step = step - warmup_steps
                    adjusted_total_steps = run_args["total_steps"] - warmup_steps
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * adjusted_step / adjusted_total_steps))
                    return (final_lr + (initial_lr - final_lr) * cosine_factor) / initial_lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=run_args["total_steps"],
                eta_min=final_lr
                )
    elif run_args["scheduler_type"] == "linear":
        if warmup_steps > 0:
            # Create a warmup + linear schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to initial_lr
                    return step / warmup_steps
                else:
                    # Linear decay from initial_lr to final_lr
                    adjusted_step = step - warmup_steps
                    adjusted_total_steps = run_args["total_steps"] - warmup_steps
                    if adjusted_total_steps <= 1:
                        return final_lr / initial_lr
                    
                    progress = adjusted_step / (adjusted_total_steps - 1)
                    end_factor = final_lr / initial_lr
                    return 1.0 + (end_factor - 1.0) * progress
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # This provides a linear ramp from initial_lr to final_lr.
            # It can handle both increasing and decreasing LR by using LambdaLR.
            def lr_lambda(step):
                total_steps = run_args["total_steps"]
                if initial_lr == 0:
                    if final_lr != 0:
                        raise ValueError("Cannot use 'linear' scheduler for warmup from initial_lr=0, as the optimizer's base LR is 0.")
                    return 1.0  # LR is 0 and stays 0.
                
                if total_steps <= 1:
                    return final_lr / initial_lr
                
                progress = step / (total_steps - 1)
                end_factor = final_lr / initial_lr
                
                # Linear interpolation of the multiplicative factor
                return 1.0 + (end_factor - 1.0) * progress
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif run_args["scheduler_type"] == "exponential":
        if warmup_steps > 0:
            # Create a warmup + exponential schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to initial_lr
                    return step / warmup_steps
                else:
                    # Exponential decay from initial_lr to final_lr
                    adjusted_step = step - warmup_steps
                    adjusted_total_steps = run_args["total_steps"] - warmup_steps
                    gamma = (final_lr / initial_lr) ** (1 / adjusted_total_steps)
                    return (initial_lr * (gamma ** adjusted_step)) / initial_lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # calculate gamma from initial and final lr
            gamma = (final_lr / initial_lr) ** (1 / run_args["total_steps"])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=gamma
                )
    elif run_args["scheduler_type"] == "lambda":
        # Get gamma from run_args for lambda scheduler
        gamma = run_args.get("gamma", 0.1)
        if warmup_steps > 0:
            # Create a warmup + lambda schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup from 0 to initial_lr
                    return step / warmup_steps
                else:
                    # Original lambda schedule logic
                    adjusted_step = step - warmup_steps
                    adjusted_total_steps = run_args["total_steps"] - warmup_steps
                    steps_per_cycle = adjusted_total_steps // run_args["n_cycles"]
                    cycle = adjusted_step // steps_per_cycle
                    cycle_progress = (adjusted_step % steps_per_cycle) / steps_per_cycle
                    # Decaying peak
                    peak = initial_lr * (gamma ** cycle)
                    # Cosine decay within cycle
                    cosine = 0.5 * (1 + np.cos(np.pi * cycle_progress))
                    lr = final_lr + (peak - final_lr) * cosine
                    return lr / initial_lr  # LambdaLR expects a multiplier of the initial LR
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
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

def _drop_nonfinite_rows(data: Dict[str, np.ndarray]) -> None:
    """Drop rows with any NaN/Inf in x or y; update arrays in place and warn."""
    param_names = data["param_names"].tolist()
    target_names = data["target_names"].tolist()
    for split, x_key, y_key in (
        ("train", "x_train", "y_train"),
        ("test", "x_test", "y_test"),
    ):
        x = data[x_key]
        y = data[y_key]
        bad_x = (~np.isfinite(x)).any(axis=1)
        bad_y = (~np.isfinite(y)).any(axis=1)
        bad = bad_x | bad_y
        n_bad = bad.sum()
        if n_bad > 0:
            keep = ~bad
            data[x_key] = x[keep]
            data[y_key] = y[keep]
            bad_cols_x = [param_names[j] for j in range(x.shape[1]) if np.any(~np.isfinite(x[:, j]))]
            bad_cols_y = [target_names[j] for j in range(y.shape[1]) if np.any(~np.isfinite(y[:, j]))]
            parts = [f"inputs: {bad_cols_x}" if bad_cols_x else "", f"targets: {bad_cols_y}" if bad_cols_y else ""]
            print(f"Dropped {n_bad} {split} rows with non-finite values ({', '.join(p for p in parts if p)}).")
    n_train = len(data["x_train"])
    n_test = len(data["x_test"])
    if n_train == 0 or n_test == 0:
        raise ValueError(
            "After dropping non-finite rows, train or test set is empty. "
            "Regenerate data with prep_shapefit_data.py."
        )
    print(f"Using {n_train} train samples, {n_test} test samples.")


def standardize(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    # Use minimum sigma to avoid huge normalized values (e.g. constant columns)
    x_mu = x_train.mean(axis=0, keepdims=True)
    x_sigma = np.maximum(
        x_train.std(axis=0, keepdims=True),
        1e-6 * (x_train.max(axis=0, keepdims=True) - x_train.min(axis=0, keepdims=True)),
    ) + 1e-8
    y_mu = y_train.mean(axis=0, keepdims=True)
    y_sigma = np.maximum(
        y_train.std(axis=0, keepdims=True),
        1e-6 * (y_train.max(axis=0, keepdims=True) - y_train.min(axis=0, keepdims=True)),
    ) + 1e-8

    x_train_n = (x_train - x_mu) / x_sigma
    x_test_n = (x_test - x_mu) / x_sigma
    y_train_n = (y_train - y_mu) / y_sigma
    y_test_n = (y_test - y_mu) / y_sigma

    # Ensure we didn't introduce NaNs or overflow (e.g. 0/0 or extreme test values)
    for name, arr in [("x_train", x_train_n), ("y_train", y_train_n), ("x_test", x_test_n), ("y_test", y_test_n)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"Standardization produced non-finite values in {name}. "
                "Check for NaN/Inf in data or extreme outliers."
            )

    stats = {"x_mu": x_mu, "x_sigma": x_sigma, "y_mu": y_mu, "y_sigma": y_sigma}
    return x_train_n, y_train_n, x_test_n, y_test_n, stats


def compare_losses(
        run_ids: list,
        labels: list | None = None,
        log_scale: bool = True,
        y_lim: tuple | None = None,
        per_step: bool = False,
        ) -> None:
    """Compare train/test loss curves across multiple MLflow runs.

    Args:
        run_ids: List of MLflow run IDs to compare.
        labels: Optional display labels for each run. Defaults to run IDs.
        log_scale: Use log scale for y-axis.
        y_lim: Tuple of (min, max) y-axis limits.
        per_step: If True, plot per-batch losses instead of epoch-averaged.
    """
    from mlflow.tracking import MlflowClient

    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    mlflow.set_tracking_uri(f"file:{scratch}/bedcosmo/shapefit_emulator/mlruns")
    client = MlflowClient()

    if labels is None:
        labels = run_ids

    if per_step:
        train_metric, test_metric = "batch_train_loss", "batch_test_loss"
        x_label = "Step"
    else:
        train_metric, test_metric = "epoch_train_loss", "epoch_test_loss"
        x_label = "Epoch"

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(12, 4))

    for run_id, label in zip(run_ids, labels):
        train_hist = client.get_metric_history(run_id, train_metric)
        test_hist = client.get_metric_history(run_id, test_metric)

        if train_hist:
            steps, vals = zip(*[(m.step, m.value) for m in train_hist if np.isfinite(m.value)])
            ax_train.plot(steps, vals, label=label, alpha=0.7)
        if test_hist:
            steps, vals = zip(*[(m.step, m.value) for m in test_hist if np.isfinite(m.value)])
            ax_test.plot(steps, vals, label=label, alpha=0.7)

    for ax, title in [(ax_train, "Train Loss"), (ax_test, "Test Loss")]:
        ax.set_xlabel(x_label)
        ax.set_ylabel("MSE Loss")
        ax.set_title(title)
        ax.legend()
        if log_scale:
            ax.set_yscale("log")
        if y_lim:
            ax.set_ylim(y_lim)

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ShapeFit emulator NN.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=get_default_save_path(),
        help="Base data directory (contains training_data/v{N} subdirs).",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="latest",
        help="Data version to use (e.g. '1', '2', or 'latest' for most recent).",
    )
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduler-type", type=str, default="constant",
                        choices=["constant", "cosine", "linear", "exponential", "lambda"],
                        help="Learning rate scheduler type.")
    parser.add_argument("--final-lr", type=float, default=1e-6,
                        help="Final learning rate for scheduler decay.")
    parser.add_argument("--n-cycles", type=int, default=1,
                        help="Number of scheduler cycles.")
    parser.add_argument("--warmup-fraction", type=float, default=0.0,
                        help="Fraction of total steps for linear warmup.")
    parser.add_argument(
        "--mlflow-exp",
        type=str,
        default="default",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name for easier identification.",
    )
    args = parser.parse_args()

    # Resolve versioned data path
    if args.data_version == "latest":
        version = _next_version(args.data_path) - 1
        if version < 1:
            raise FileNotFoundError(
                f"No training data versions found in {args.data_path}/training_data/. "
                "Run prep_shapefit_data.py first."
            )
    else:
        version = int(args.data_version)
    data_path = os.path.join(args.data_path, "training_data", f"v{version}")
    print(f"Using training data: {data_path}")

    # Set up MLflow tracking
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
    mlflow_tracking_uri = f"file:{scratch}/bedcosmo/shapefit_emulator/mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_exp)
    mlflow.start_run(run_name=args.run_name)
    active_run = mlflow.active_run()
    mlflow_run_id = active_run.info.run_id
    artifacts_dir = os.path.join(
        scratch, "bedcosmo", "shapefit_emulator", "mlruns",
        active_run.info.experiment_id, active_run.info.run_id, "artifacts"
    )
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    print(f"MLflow run ID: {mlflow_run_id}")
    print(f"Artifacts: {artifacts_dir}")

    # Log parameters immediately after starting run (same pattern as train.py)
    run_params = vars(args)
    for key, value in run_params.items():
        if key == "data_path":
            continue  # log resolved versioned path instead
        mlflow.log_param(key, value)
    mlflow.log_param("data_path", data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_data(data_path)
    x_train, y_train, x_test, y_test, stats = standardize(
        data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = x_train.shape[1]
    out_dim = y_train.shape[1]
    model = NNRegressor(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    batches_per_epoch = len(train_loader)
    total_steps = args.epochs * batches_per_epoch
    scheduler = create_scheduler(opt, {
        "total_steps": total_steps,
        "n_cycles": args.n_cycles,
        "initial_lr": args.lr,
        "final_lr": args.final_lr,
        "scheduler_type": args.scheduler_type,
        "warmup_fraction": args.warmup_fraction,
    })
    print(f"Scheduler: {args.scheduler_type}, total_steps={total_steps}, "
          f"initial_lr={args.lr}, final_lr={args.final_lr}")

    train_losses = []
    test_losses = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        max_batch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            batch_loss = loss.item()
            if not np.isfinite(batch_loss):
                raise RuntimeError(
                    f"NaN/Inf loss at epoch {epoch}. "
                    "Check that prepped data has no NaN/Inf (run with --data-path and inspect or regenerate with prep_data.py)."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            global_step += 1
            mlflow.log_metric("batch_train_loss", batch_loss, step=global_step)
            train_loss += batch_loss * xb.size(0)
            if batch_loss > max_batch_loss:
                max_batch_loss = batch_loss
        epoch_train_loss = train_loss / len(train_ds)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                batch_test = loss_fn(pred, yb).item()
                global_step += 1
                mlflow.log_metric("batch_test_loss", batch_test, step=global_step)
                test_loss += batch_test * xb.size(0)
        epoch_test_loss = test_loss / len(test_ds)

        if not np.isfinite(train_loss) or not np.isfinite(test_loss):
            raise RuntimeError(
                f"NaN/Inf MSE at epoch {epoch}. "
                "Data may contain non-finite values or learning rate may be too high; try --lr 1e-3 or smaller."
            )
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        mlflow.log_metric("epoch_train_loss", epoch_train_loss, step=epoch)
        mlflow.log_metric("epoch_test_loss", epoch_test_loss, step=epoch)
        mlflow.log_metric("max_batch_loss", max_batch_loss, step=epoch)
        mlflow.log_metric("lr", opt.param_groups[0]["lr"], step=epoch)
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:4d} train_mse={epoch_train_loss:.6e} test_mse={epoch_test_loss:.6e}")

    with torch.no_grad():
        yhat_n = model(torch.from_numpy(x_test).to(device)).cpu().numpy()
    yhat = yhat_n * stats["y_sigma"] + stats["y_mu"]
    mae = np.mean(np.abs(yhat - data["y_test"]), axis=0)
    rms = np.sqrt(np.mean((yhat - data["y_test"]) ** 2, axis=0))
    target_names = data["target_names"].tolist()

    print("Per-target metrics on test set:")
    for i, name in enumerate(target_names):
        print(f"  {name:10s}  MAE={mae[i]:.6e}  RMSE={rms[i]:.6e}")
        mlflow.log_metric(f"mae_{name}", mae[i])
        mlflow.log_metric(f"rmse_{name}", rms[i])

    fig, ax = plt.subplots()
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="tab:blue", label="Train")
    ax.plot(epochs, test_losses, color="tab:orange", label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("ShapeFit NN Training")
    fig.tight_layout()
    loss_plot_path = os.path.join(artifacts_dir, "loss.png")
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss plot to: {loss_plot_path}")

    model_path = os.path.join(artifacts_dir, "model.pt")
    payload = {
        "state_dict": model.state_dict(),
        "x_mu": torch.from_numpy(stats["x_mu"]),
        "x_sigma": torch.from_numpy(stats["x_sigma"]),
        "y_mu": torch.from_numpy(stats["y_mu"]),
        "y_sigma": torch.from_numpy(stats["y_sigma"]),
        "param_names": data["param_names"].tolist(),
        "target_names": target_names,
        "hidden_dim": args.hidden_dim,
        "n_hidden": args.n_hidden,
    }
    torch.save(payload, model_path)
    print(f"Saved model checkpoint to: {model_path}")

    train_params: Dict[str, Any] = {
        "data_path": data_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "n_hidden": args.n_hidden,
        "seed": args.seed,
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "param_names": data["param_names"].tolist(),
        "target_names": target_names,
    }
    # Log data-dependent params that weren't available at startup
    mlflow.log_param("in_dim", int(in_dim))
    mlflow.log_param("out_dim", int(out_dim))
    mlflow.log_param("param_names", ", ".join(data["param_names"].tolist()))
    mlflow.log_param("target_names", ", ".join(target_names))

    train_params_path = os.path.join(artifacts_dir, "train_params.yaml")
    with open(train_params_path, "w") as f:
        yaml.safe_dump(train_params, f, default_flow_style=False, sort_keys=False)
    print(f"Saved training params to: {train_params_path}")

    # Run evaluation
    print("\nRunning evaluation...")
    run_eval(model_path, save_path=artifacts_dir, n_samples=3000)

    mlflow.end_run()
    print(f"MLflow run completed: {mlflow_run_id}")
    print(f"Artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()

