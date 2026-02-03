import argparse
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_data(data_path: str) -> Dict[str, np.ndarray]:
    train = np.load(f"{data_path}/shapefit_train.npz", allow_pickle=True)
    test = np.load(f"{data_path}/shapefit_test.npz", allow_pickle=True)
    return {
        "x_train": train["x"].astype(np.float32),
        "y_train": train["y"].astype(np.float32),
        "x_test": test["x"].astype(np.float32),
        "y_test": test["y"].astype(np.float32),
        "param_names": train["param_names"],
        "target_names": train["target_names"],
    }


def standardize(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    x_mu = x_train.mean(axis=0, keepdims=True)
    x_sigma = x_train.std(axis=0, keepdims=True) + 1e-8
    y_mu = y_train.mean(axis=0, keepdims=True)
    y_sigma = y_train.std(axis=0, keepdims=True) + 1e-8

    x_train_n = (x_train - x_mu) / x_sigma
    x_test_n = (x_test - x_mu) / x_sigma
    y_train_n = (y_train - y_mu) / y_sigma
    y_test_n = (y_test - y_mu) / y_sigma

    stats = {"x_mu": x_mu, "x_sigma": x_sigma, "y_mu": y_mu, "y_sigma": y_sigma}
    return x_train_n, y_train_n, x_test_n, y_test_n, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ShapeFit emulator MLP.")
    parser.add_argument("--data-path", type=str, default=".")
    parser.add_argument("--model-path", type=str, default="./shapefit_mlp.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_data(args.data_path)
    x_train, y_train, x_test, y_test, stats = standardize(
        data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = x_train.shape[1]
    out_dim = y_train.shape[1]
    model = MLPRegressor(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb)
                test_loss += loss_fn(pred, yb).item() * xb.size(0)
        test_loss /= len(test_ds)

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:4d} train_mse={train_loss:.6e} test_mse={test_loss:.6e}")

    with torch.no_grad():
        yhat_n = model(torch.from_numpy(x_test)).numpy()
    yhat = yhat_n * stats["y_sigma"] + stats["y_mu"]
    mae = np.mean(np.abs(yhat - data["y_test"]), axis=0)
    rms = np.sqrt(np.mean((yhat - data["y_test"]) ** 2, axis=0))
    target_names = data["target_names"].tolist()

    print("Per-target metrics on test set:")
    for i, name in enumerate(target_names):
        print(f"  {name:10s}  MAE={mae[i]:.6e}  RMSE={rms[i]:.6e}")

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
    torch.save(payload, args.model_path)
    print(f"Saved model checkpoint to: {args.model_path}")


if __name__ == "__main__":
    main()

