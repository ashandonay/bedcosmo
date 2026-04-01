import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, expand: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class ResNetRegressor(nn.Module):
    """SiLU + residual MLP; width/depth/dropout/expand come from model_config.yaml."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 24,
        n_hidden: int = 6,
        dropout: float = 0.0,
        expand: int = 4,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResBlock(hidden_dim, dropout, expand) for _ in range(n_hidden)]
        )
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.proj_in(x))
        for block in self.blocks:
            x = block(x)
        return self.proj_out(x)
