"""Shared output location for the num_visits diagnostic scripts.

Figures go to ``experiments/num_visits/plots/`` -- a sibling of ``scripts/``, so
generated PNGs never mix with either the source that makes them or the YAML
configs in the experiment root. Anchored to ``__file__`` rather than the working
directory, so a script writes to the same place wherever it is invoked from.
"""

from __future__ import annotations

from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"


def plot_path(name: str) -> Path:
    """Absolute path for a figure named ``name``, creating ``plots/`` if needed."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR / name
