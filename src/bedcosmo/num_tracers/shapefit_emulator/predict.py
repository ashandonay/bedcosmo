from typing import Dict

import numpy as np
import torch

from .train_nn import MLPRegressor


def load_emulator(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    model = MLPRegressor(
        in_dim=len(ckpt["param_names"]),
        out_dim=len(ckpt["target_names"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        n_hidden=int(ckpt["n_hidden"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ckpt, model


def predict_shapefit(model_path: str, params: Dict[str, float]) -> Dict[str, float]:
    ckpt, model = load_emulator(model_path)
    x = np.array([[params[name] for name in ckpt["param_names"]]], dtype=np.float32)
    x_n = (x - ckpt["x_mu"].numpy()) / ckpt["x_sigma"].numpy()

    with torch.no_grad():
        y_n = model(torch.from_numpy(x_n)).numpy()
    y = y_n * ckpt["y_sigma"].numpy() + ckpt["y_mu"].numpy()

    return {name: float(y[0, i]) for i, name in enumerate(ckpt["target_names"])}

