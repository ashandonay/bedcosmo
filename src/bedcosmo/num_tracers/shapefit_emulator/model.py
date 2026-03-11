import torch
import torch.nn as nn

class NNRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)