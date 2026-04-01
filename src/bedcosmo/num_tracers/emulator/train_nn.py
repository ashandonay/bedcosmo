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
from util import get_default_save_path, _next_version, TRACER_TYPE_CHOICES, build_model
from eval_nn import run_eval

def load_data(data_path: str, tracer: str | None = None) -> Dict[str, np.ndarray]:
    if tracer and os.path.isfile(f"{data_path}/{tracer}_train.npz"):
        train = np.load(f"{data_path}/{tracer}_train.npz", allow_pickle=True)
        test = np.load(f"{data_path}/{tracer}_test.npz", allow_pickle=True)
    else:
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


def _symlog_transform(y: np.ndarray, linthresh: np.ndarray) -> np.ndarray:
    """Per-column symlog: sign(y) * log1p(|y| / linthresh)."""
    return np.sign(y) * np.log1p(np.abs(y) / linthresh)


def _symlog_inverse(y_log: np.ndarray, linthresh: np.ndarray) -> np.ndarray:
    """Inverse of _symlog_transform."""
    return np.sign(y_log) * linthresh * np.expm1(np.abs(y_log))


def standardize(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    log_normalize: bool = False,
):
    # Optional log normalization of targets before z-score
    y_linthresh = None
    if log_normalize:
        # Per-column linthresh = min absolute nonzero value
        y_linthresh = np.empty((1, y_train.shape[1]), dtype=np.float32)
        for col in range(y_train.shape[1]):
            abs_nz = np.abs(y_train[:, col][y_train[:, col] != 0])
            y_linthresh[0, col] = float(np.min(abs_nz)) if len(abs_nz) > 0 else 1e-8
        y_train = _symlog_transform(y_train, y_linthresh)
        y_test = _symlog_transform(y_test, y_linthresh)

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

    stats = {"x_mu": x_mu, "x_sigma": x_sigma, "y_mu": y_mu, "y_sigma": y_sigma,
             "log_normalize": log_normalize, "y_linthresh": y_linthresh}
    return x_train_n, y_train_n, x_test_n, y_test_n, stats

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ShapeFit emulator NN.")
    parser.add_argument(
        "--analysis",
        type=str,
        default="shapefit",
        help="Analysis type: 'shapefit' or 'bao'.",
    )
    parser.add_argument(
        "--quantity",
        type=str,
        default="mean",
        help="Quantity: 'mean' or 'covar'.",
    )
    parser.add_argument(
        "--cosmo-model",
        type=str,
        default="base",
        choices=["base", "base_w", "base_w_wa", "base_omegak", "base_omegak_w_wa"],
        help="Cosmology model defining the parameter set (affects data path). Default: 'base'.",
    )
    parser.add_argument(
        "--nn-model",
        type=str,
        default=None,
        help=(
            "YAML top-level key in {analysis}/model_config.yaml for NN hyperparameters and "
            "architecture. If omitted, uses --cosmo-model as the key; if that key is missing "
            "but a 'default' entry exists, uses 'default' (see shapefit/model_config.yaml)."
        ),
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="latest",
        help="Data version to use (e.g. '1', '2', or 'latest' for most recent).",
    )
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (epochs without test loss improvement). 0 = disabled.")
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--log-normalize", action="store_true",
                        help="Apply symlog transform to targets before z-score standardization.")
    parser.add_argument("--eval-atol", type=float, default=2e-3,
                        help="Absolute tolerance for evaluation.")
    parser.add_argument("--eval-rtol", type=float, default=2e-3,
                        help="Relative tolerance for evaluation.")
    parser.add_argument("--eval-rescale-by", type=str, default=None,
                        help="Divide predictions by this input param before eval comparison (e.g. 'N_tracers').")
    parser.add_argument("--tracer", type=str, default=None,
                        choices=TRACER_TYPE_CHOICES,
                        help="Tracer type (e.g. BGS, LRG1). Sets zrange, z_eff, and N_tracers bounds for evaluation.")
    args = parser.parse_args()

    # Load training config from {analysis}/model_config.yaml. Keys are arbitrary (e.g. scaled_base);
    # use --nn-model to pick one. If --nn-model is omitted, try --cosmo-model, then optional "default".
    _here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(_here, args.analysis, "model_config.yaml")
    with open(config_path) as f:
        all_model_configs = yaml.safe_load(f)
    requested = args.nn_model if args.nn_model is not None else args.cosmo_model
    if requested in all_model_configs:
        config_key = requested
    elif args.nn_model is None and "default" in all_model_configs:
        config_key = "default"
        print(
            f"No YAML entry '{requested}' in {config_path}; using 'default' for NN hyperparameters."
        )
    else:
        raise ValueError(
            f"No training config for nn model key '{requested}' in {config_path}. "
            f"Available: {list(all_model_configs.keys())}"
        )
    model_cfg = all_model_configs[config_key]
    architecture = model_cfg.get("architecture", "resnet")
    print(f"Loaded training config for {args.analysis}/{config_key} (architecture={architecture}): {model_cfg}")

    root_data_dir = get_default_save_path(analysis=args.analysis, quantity=args.quantity, cosmo_model=args.cosmo_model)

    # Resolve versioned data path
    if args.data_version == "latest":
        version = _next_version(root_data_dir) - 1
        if version < 1:
            raise FileNotFoundError(
                f"No training data versions found in {root_data_dir}. "
                "Run the appropriate prep script first."
            )
    else:
        version = args.data_version
    data_path = os.path.join(root_data_dir, f"v{version}")
    print(f"Using {args.analysis}/{args.cosmo_model}/{args.quantity} training data: {data_path}")

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
    for key, value in vars(args).items():
        mlflow.log_param(key, value)
    for key, value in model_cfg.items():
        mlflow.log_param(key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_data(data_path, tracer=args.tracer)
    x_train, y_train, x_test, y_test, stats = standardize(
        data["x_train"], data["y_train"], data["x_test"], data["y_test"],
        log_normalize=args.log_normalize,
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=model_cfg["batch_size"], shuffle=False)

    in_dim = x_train.shape[1]
    out_dim = y_train.shape[1]
    model = build_model(
        analysis=args.analysis,
        architecture=architecture,
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=model_cfg["hidden_dim"],
        n_hidden=model_cfg["n_hidden"],
        dropout=model_cfg.get("dropout", 0.0),
        expand=model_cfg.get("expand", 4),
    ).to(device)

    lr = model_cfg["lr"]
    final_lr = lr * model_cfg.get("final_lr_frac", 0.01)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=model_cfg.get("weight_decay", 1e-5))
    loss_fn = nn.MSELoss()

    batches_per_epoch = len(train_loader)
    total_steps = args.epochs * batches_per_epoch
    scheduler = create_scheduler(opt, {
        "total_steps": total_steps,
        "n_cycles": model_cfg.get("lr_restarts", 1),
        "initial_lr": lr,
        "final_lr": final_lr,
        "scheduler_type": model_cfg.get("scheduler_type", "lambda"),
        "warmup_fraction": model_cfg.get("warmup_fraction", 0.0),
        "gamma": model_cfg.get("lr_gamma", 1.0),
    })
    print(f"Scheduler: {model_cfg.get('scheduler_type', 'lambda')}, total_steps={total_steps}, "
          f"initial_lr={lr}, final_lr={final_lr}")

    train_losses = []
    test_losses = []
    global_step = 0
    best_test_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=model_cfg.get("grad_clip", 1.0))
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

        # Track best model and early stopping
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs). "
                  f"Best test loss: {best_test_loss:.6e}")
            break

    # Restore best model weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Restored best model (test loss: {best_test_loss:.6e})")

    model.eval()
    with torch.no_grad():
        yhat_n = model(torch.from_numpy(x_test).to(device)).cpu().numpy()
    yhat = yhat_n * stats["y_sigma"] + stats["y_mu"]
    if stats["log_normalize"]:
        yhat = _symlog_inverse(yhat, stats["y_linthresh"])
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
        "log_normalize": stats["log_normalize"],
        "y_linthresh": torch.from_numpy(stats["y_linthresh"]) if stats["y_linthresh"] is not None else None,
        "param_names": data["param_names"].tolist(),
        "target_names": target_names,
        "hidden_dim": model_cfg["hidden_dim"],
        "n_hidden": model_cfg["n_hidden"],
        "dropout": model_cfg.get("dropout", 0.0),
        "cosmo_model": args.cosmo_model,
        "nn_model": config_key,
        "architecture": architecture,
        "analysis": args.analysis,
        "expand": model_cfg.get("expand", 4),
    }
    torch.save(payload, model_path)
    print(f"Saved model checkpoint to: {model_path}")

    train_params: Dict[str, Any] = {
        "data_path": data_path,
        "epochs": args.epochs,
        "seed": args.seed,
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "param_names": data["param_names"].tolist(),
        "target_names": target_names,
        **model_cfg,
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
    run_eval(model_path, save_path=artifacts_dir, analysis=args.analysis, quantity=args.quantity, n_samples=1000, atol=args.eval_atol, rtol=args.eval_rtol, log_scale=True, rescale_by=args.eval_rescale_by, tracer=args.tracer)

    mlflow.end_run()
    print(f"MLflow run completed: {mlflow_run_id}")
    print(f"Artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()

