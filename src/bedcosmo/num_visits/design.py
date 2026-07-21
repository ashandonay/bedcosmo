"""Generate explicit num_visits design points with a fixed total-visit budget.

A Cartesian grid over six bands with an exact sum constraint is sparse or
enormous depending on step size. This module builds explicit design arrays
(``input_designs_path``) by combining:

1. All ratio-grid points that satisfy the budget exactly.
2. Single-band floor/cap corners with proportional fill on the rest.
3. Random compositions within per-band ratio bounds until ``n_target`` is reached.

Every band uses the same fractional range relative to the LSST nominal visit
counts, so all six filters appear on comparable footing in ratio-to-nominal plots.

Example::

    python -m bedcosmo.num_visits.design

Writes ``$SCRATCH/bedcosmo/num_visits/designs/designs_<n>_<YYYYMMDD_HHMMSS>.npy``
and prints the absolute path for ``input_designs_path``.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Iterable

import numpy as np

BANDS = ["u", "g", "r", "i", "z", "y"]
# Keep in sync with bedcosmo.num_visits.experiment.fiducial_nvisits
NOMINAL = np.array([70, 100, 230, 230, 200, 200], dtype=np.int64)
BUDGET = int(NOMINAL.sum())
UNIT = 10
T = BUDGET // UNIT


def _round_to_unit(value: float) -> int:
    return int(max(UNIT, round(value / UNIT) * UNIT))


def ratio_bounds(ratio_min: float, ratio_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-band visit floors and caps from uniform ratio limits."""
    lower = np.array([_round_to_unit(n * ratio_min) for n in NOMINAL], dtype=np.int64)
    upper = np.array([_round_to_unit(n * ratio_max) for n in NOMINAL], dtype=np.int64)
    return lower, upper


def ratio_axes(ratio_min: float, ratio_max: float, n_levels: int = 3) -> list[np.ndarray]:
    """Discrete visit levels per band from a shared ratio grid."""
    if n_levels == 3:
        ratios = [ratio_min, 1.0, ratio_max]
    else:
        ratios = np.linspace(ratio_min, ratio_max, n_levels).tolist()
    axes = []
    for nom in NOMINAL:
        levels = sorted({max(UNIT, _round_to_unit(nom * ratio)) for ratio in ratios})
        axes.append(np.array(levels, dtype=np.int64))
    return axes


def to_units(visits: np.ndarray) -> np.ndarray:
    return np.asarray(visits, dtype=np.int64) // UNIT


def to_visits(units: Iterable[int] | np.ndarray) -> np.ndarray:
    return np.asarray(units, dtype=np.int64) * UNIT


def proportional_fill(
    total_units: int,
    floor_u: np.ndarray,
    cap_u: np.ndarray,
    exclude: int | None = None,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Distribute ``total_units`` across bands ~ nominal, honoring floors and caps."""
    mask = np.ones(len(BANDS), dtype=bool)
    if exclude is not None:
        mask[exclude] = False
    if exclude_mask is not None:
        mask &= ~exclude_mask

    floors = floor_u.copy()
    caps = cap_u.copy()
    floors[~mask] = 0
    caps[~mask] = 0

    if total_units < int(floors.sum()) or total_units > int(caps.sum()):
        return None

    out = floors.copy()
    remaining = total_units - int(out.sum())
    weights = NOMINAL.astype(float) * mask
    order = np.argsort(-weights)

    while remaining > 0:
        progressed = False
        for band in order:
            if remaining == 0:
                break
            if out[band] < caps[band]:
                out[band] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            return None
    return out


def enumerate_ratio_grid(
    ratio_min: float,
    ratio_max: float,
    n_levels: int = 3,
) -> np.ndarray:
    """All grid points whose visit counts sum exactly to ``BUDGET``."""
    axes = ratio_axes(ratio_min, ratio_max, n_levels=n_levels)
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = np.stack([grid.ravel() for grid in mesh], axis=1)
    keep = flat.sum(axis=1) == BUDGET
    return flat[keep]


def _check_bounds_feasible(floor_u: np.ndarray, cap_u: np.ndarray) -> None:
    if int(floor_u.sum()) > T:
        raise ValueError(
            f"Per-band lower bounds sum to {int(floor_u.sum() * UNIT)} visits, "
            f"above budget {BUDGET}."
        )
    if int(cap_u.sum()) < T:
        raise ValueError(
            f"Per-band upper bounds sum to {int(cap_u.sum() * UNIT)} visits, "
            f"below budget {BUDGET}."
        )


def _random_feasible_units(
    floor_u: np.ndarray,
    cap_u: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Sample one composition uniformly-ish over feasible integer allocations."""
    out = floor_u.copy()
    remaining = T - int(out.sum())
    if remaining < 0 or remaining > int((cap_u - out).sum()):
        return None

    weights = NOMINAL.astype(float)
    while remaining > 0:
        candidates = np.flatnonzero(out < cap_u)
        if candidates.size == 0:
            return None
        probs = weights[candidates]
        probs = probs / probs.sum()
        band = int(rng.choice(candidates, p=probs))
        out[band] += 1
        remaining -= 1
    return out


def generate_designs(
    n_target: int = 100,
    ratio_min: float = 0.75,
    ratio_max: float = 1.25,
    n_levels: int = 3,
    seed: int = 0,
    include_corners: bool = True,
) -> np.ndarray:
    """Build ``(n_designs, 6)`` visit-count designs summing to the nominal budget."""
    lower, upper = ratio_bounds(ratio_min, ratio_max)
    floor_u = to_units(lower)
    cap_u = to_units(upper)
    _check_bounds_feasible(floor_u, cap_u)
    designs: set[tuple[int, ...]] = set()

    def add(units: np.ndarray | Iterable[int]) -> None:
        u = np.asarray(units, dtype=np.int64)
        if u.shape != (len(BANDS),):
            return
        if int(u.sum()) != T:
            return
        if np.any(u < floor_u) or np.any(u > cap_u):
            return
        designs.add(tuple(int(x) for x in u))

    add(to_units(NOMINAL))

    for row in enumerate_ratio_grid(ratio_min, ratio_max, n_levels=n_levels):
        add(to_units(row))

    if include_corners:
        for band in range(len(BANDS)):
            rest = T - int(floor_u[band])
            if rest >= 0:
                filled = proportional_fill(rest, floor_u, cap_u, exclude=band)
                if filled is not None:
                    filled = filled.copy()
                    filled[band] = floor_u[band]
                    add(filled)

            rest = T - int(cap_u[band])
            if rest >= 0:
                filled = proportional_fill(rest, floor_u, cap_u, exclude=band)
                if filled is not None:
                    filled = filled.copy()
                    filled[band] = cap_u[band]
                    add(filled)

    rng = np.random.default_rng(seed)
    max_tries = max(500, 50 * n_target)
    tries = 0
    while len(designs) < n_target and tries < max_tries:
        tries += 1
        draw = _random_feasible_units(floor_u, cap_u, rng)
        if draw is None:
            continue
        add(draw)

    if len(designs) < n_target:
        raise RuntimeError(
            f"Only generated {len(designs)} unique designs (target {n_target}) "
            f"after {tries} tries. Try widening ratio_min/ratio_max or lowering n_target."
        )

    arr = np.array(sorted(designs), dtype=np.int64)
    if arr.shape[0] > n_target:
        rng = np.random.default_rng(seed)
        keep_nominal = np.all(to_visits(arr) == NOMINAL, axis=1)
        nominal_rows = arr[keep_nominal]
        other_rows = arr[~keep_nominal]
        n_other = n_target - nominal_rows.shape[0]
        if n_other <= 0:
            arr = nominal_rows[:n_target]
        else:
            pick = rng.choice(other_rows.shape[0], size=n_other, replace=False)
            arr = np.vstack([nominal_rows, other_rows[pick]])
            arr = arr[np.lexsort(arr.T[::-1])]

    visits = to_visits(arr).astype(np.float64)
    _validate_designs(visits, lower, upper)
    return visits


def _validate_designs(visits: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
    if not np.all(visits.sum(axis=1) == BUDGET):
        raise AssertionError("some designs do not sum to the nominal budget")
    if np.any(visits < lower[None, :]) or np.any(visits > upper[None, :]):
        raise AssertionError("design outside ratio bounds")
    if not np.all(visits % UNIT == 0):
        raise AssertionError("design visit counts are not multiples of 10")
    if not any(np.all(visits == NOMINAL, axis=1)):
        raise AssertionError("nominal design missing from output")


def plot_designs(
    visits: np.ndarray,
    out_path: str,
    ratio_min: float = 0.75,
    ratio_max: float = 1.25,
    cmap: str = "viridis",
    color_dim: int = 0,
) -> None:
    """Parallel-coordinates view of the design space."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_designs, n_dims = visits.shape
    x = np.arange(n_dims)
    lower, cap = ratio_bounds(ratio_min, ratio_max)
    cap = cap.astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    norm = visits / cap[None, :]

    color_vals = visits[:, color_dim]
    cmin, cmax = color_vals.min(), color_vals.max()
    colormap = plt.get_cmap(cmap)
    colors = colormap((color_vals - cmin) / (cmax - cmin + 1e-10))

    for i in range(n_designs):
        ax.plot(x, norm[i], color=colors[i], alpha=0.45, linewidth=0.8, zorder=1)

    ax.plot(x, NOMINAL / cap, color="black", linewidth=2.5, marker="o", label="Nominal", zorder=3)
    ax.plot(x, np.ones(n_dims), color="0.6", linewidth=1.0, linestyle="--", label=f"{ratio_max:.2f}x nominal", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"${b}$" for b in BANDS], fontsize=13)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel(f"visits / ({ratio_max:.2f}x nominal)", fontsize=12)
    for xi in range(n_dims):
        ax.axvline(xi, color="0.85", linewidth=0.8, zorder=0)
        ax.text(xi, 1.045, f"{int(cap[xi])}", ha="center", va="bottom", fontsize=8, color="0.4")
        ax.text(xi, -0.055, f"{int(lower[xi])}", ha="center", va="top", fontsize=8, color="0.4")

    ax.set_title(
        f"NumVisits design space: {n_designs} designs, all summing to {BUDGET} visits",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=cmin, vmax=cmax))
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(f"${BANDS[color_dim]}$ visits", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _designs_dir() -> str:
    scratch = os.environ.get("SCRATCH", "/pscratch/sd/a/ashandon")
    return os.path.join(scratch, "bedcosmo", "num_visits", "designs")


def _dated_output_path(n_target: int, when: datetime | None = None) -> str:
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    filename = f"designs_{n_target}_{stamp}.npy"
    return os.path.join(_designs_dir(), filename)


def main(argv: list[str] | None = None) -> np.ndarray:
    parser = argparse.ArgumentParser(description="Generate num_visits design arrays.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npy path (default: $SCRATCH/bedcosmo/num_visits/designs/designs_<n>_<date>.npy)",
    )
    parser.add_argument("--plot", default=None, help="Optional parallel-coordinates .png path")
    parser.add_argument("--n-target", type=int, default=100, help="Target number of designs")
    parser.add_argument("--ratio-min", type=float, default=0.75, help="Lower ratio limit (all bands)")
    parser.add_argument("--ratio-max", type=float, default=1.25, help="Upper ratio limit (all bands)")
    parser.add_argument("--n-levels", type=int, default=3, help="Ratio grid levels per band")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random fill / subsample")
    parser.add_argument("--no-corners", action="store_true", help="Skip single-band floor/cap corners")
    args = parser.parse_args(argv)

    out_path = os.path.abspath(args.output or _dated_output_path(args.n_target))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    visits = generate_designs(
        n_target=args.n_target,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        n_levels=args.n_levels,
        seed=args.seed,
        include_corners=not args.no_corners,
    )

    np.save(out_path, visits)

    lower, upper = ratio_bounds(args.ratio_min, args.ratio_max)
    print(f"wrote {visits.shape[0]} designs -> {out_path}")
    print(f"all sum to {BUDGET}: {bool(np.all(visits.sum(1) == BUDGET))}")
    print(f"ratio range [{args.ratio_min}, {args.ratio_max}]")
    print("per-band min:", visits.min(0).astype(int).tolist())
    print("per-band max:", visits.max(0).astype(int).tolist())
    print("per-band lower bound:", lower.astype(int).tolist())
    print("per-band upper bound:", upper.astype(int).tolist())
    print("nominal present:", any(np.all(visits == NOMINAL, axis=1)))

    plot_path = args.plot
    if plot_path is None:
        root, _ = os.path.splitext(out_path)
        plot_path = root + ".png"
    plot_designs(visits, plot_path, ratio_min=args.ratio_min, ratio_max=args.ratio_max)
    print(f"wrote design-space plot -> {plot_path}")
    print(f"shape: {visits.shape}")
    print(f"\ninput_designs_path: {out_path}")
    return visits


if __name__ == "__main__":
    main()
