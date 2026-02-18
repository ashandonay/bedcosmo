import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
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
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Base directory for run outputs. Default: <data-path>/runs",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
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

    runs_base = args.runs_dir or os.path.join(args.data_path, "runs")
    run_dir = os.path.join(
        runs_base,
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

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

    train_losses = []
    test_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not np.isfinite(loss.item()):
                raise RuntimeError(
                    f"NaN/Inf loss at epoch {epoch}. "
                    "Check that prepped data has no NaN/Inf (run with --data-path and inspect or regenerate with prep_data.py)."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                test_loss += loss_fn(pred, yb).item() * xb.size(0)
        test_loss /= len(test_ds)

        if not np.isfinite(train_loss) or not np.isfinite(test_loss):
            raise RuntimeError(
                f"NaN/Inf MSE at epoch {epoch}. "
                "Data may contain non-finite values or learning rate may be too high; try --lr 1e-3 or smaller."
            )
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:4d} train_mse={train_loss:.6e} test_mse={test_loss:.6e}")

    with torch.no_grad():
        yhat_n = model(torch.from_numpy(x_test).to(device)).cpu().numpy()
    yhat = yhat_n * stats["y_sigma"] + stats["y_mu"]
    mae = np.mean(np.abs(yhat - data["y_test"]), axis=0)
    rms = np.sqrt(np.mean((yhat - data["y_test"]) ** 2, axis=0))
    target_names = data["target_names"].tolist()

    print("Per-target metrics on test set:")
    for i, name in enumerate(target_names):
        print(f"  {name:10s}  MAE={mae[i]:.6e}  RMSE={rms[i]:.6e}")

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
    loss_plot_path = os.path.join(run_dir, "loss.png")
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss plot to: {loss_plot_path}")

    model_path = os.path.join(run_dir, "model.pt")
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
    if args.runs_dir is not None:
        train_params["runs_dir"] = args.runs_dir
    train_params_path = os.path.join(run_dir, "train_params.yaml")
    with open(train_params_path, "w") as f:
        yaml.safe_dump(train_params, f, default_flow_style=False, sort_keys=False)
    print(f"Saved training params to: {train_params_path}")

    # Run evaluation
    print("\nRunning evaluation...")
    run_eval(model_path, save_path=run_dir, n_samples=3000)


if __name__ == "__main__":
    main()

