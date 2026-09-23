"""Generate explicit num_visits design points with a fixed total-visit budget.

A Cartesian grid over the bands with an exact sum constraint is sparse or
enormous depending on step size. This module builds explicit design arrays
(``input_designs_path``) by combining:

1. All ratio-grid points that satisfy the budget exactly.
2. Single-band floor/cap corners with proportional fill on the rest.
3. Random compositions within per-band ratio bounds until ``n_target`` is reached.

``--bands`` picks the filters (default all six). The budget is the nominal total
of those bands, and every band uses the same fractional range relative to its
LSST nominal visit count, so all filters appear on comparable footing in
ratio-to-nominal plots.

Example::

    python -m bedcosmo.num_visits.design --bands gri --n-target 100

Outputs land in two places: the design array (and its plot) goes to
``$SCRATCH/bedcosmo/num_visits/designs/<name>.npy`` (``--designs-dir``), while the
``design_args_<name>.yaml`` that points at it goes to the experiment config dir
(``--out-dir``) so ``--design-args-path`` can find it. The YAML's ``labels`` list the
bands in column order of the array.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import yaml

from bedcosmo.num_visits.experiment import fiducial_nvisits

BANDS = list(fiducial_nvisits)
UNIT = 10


def _round_to_unit(value: float) -> int:
    return int(max(UNIT, round(value / UNIT) * UNIT))


def nominal_visits(bands: Sequence[str]) -> np.ndarray:
    """LSST nominal visit counts for ``bands``, in the given order."""
    unknown = [b for b in bands if b not in fiducial_nvisits]
    if unknown:
        raise ValueError(f"Unknown bands {unknown}; choose from {BANDS}")
    if len(set(bands)) != len(bands):
        raise ValueError(f"Duplicate bands in {list(bands)}")
    if not bands:
        raise ValueError("At least one band is required")
    return np.array([fiducial_nvisits[b] for b in bands], dtype=np.int64)


def ratio_bounds(
    nominal: np.ndarray, ratio_min: float, ratio_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-band visit floors and caps from uniform ratio limits."""
    lower = np.array([_round_to_unit(n * ratio_min) for n in nominal], dtype=np.int64)
    upper = np.array([_round_to_unit(n * ratio_max) for n in nominal], dtype=np.int64)
    return lower, upper


def ratio_axes(
    nominal: np.ndarray, ratio_min: float, ratio_max: float, n_levels: int = 3
) -> list[np.ndarray]:
    """Discrete visit levels per band from a shared ratio grid."""
    if n_levels == 3:
        ratios = [ratio_min, 1.0, ratio_max]
    else:
        ratios = np.linspace(ratio_min, ratio_max, n_levels).tolist()
    axes = []
    for nom in nominal:
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
    weights: np.ndarray,
    exclude: int | None = None,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Distribute ``total_units`` across bands ~ ``weights``, honoring floors and caps."""
    mask = np.ones(len(floor_u), dtype=bool)
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
    order = np.argsort(-(weights.astype(float) * mask))

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
    nominal: np.ndarray,
    ratio_min: float,
    ratio_max: float,
    n_levels: int = 3,
) -> np.ndarray:
    """All grid points whose visit counts sum exactly to the nominal budget."""
    axes = ratio_axes(nominal, ratio_min, ratio_max, n_levels=n_levels)
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = np.stack([grid.ravel() for grid in mesh], axis=1)
    keep = flat.sum(axis=1) == nominal.sum()
    return flat[keep]


def _check_bounds_feasible(floor_u: np.ndarray, cap_u: np.ndarray, t: int) -> None:
    if int(floor_u.sum()) > t:
        raise ValueError(
            f"Per-band lower bounds sum to {int(floor_u.sum() * UNIT)} visits, "
            f"above budget {t * UNIT}."
        )
    if int(cap_u.sum()) < t:
        raise ValueError(
            f"Per-band upper bounds sum to {int(cap_u.sum() * UNIT)} visits, "
            f"below budget {t * UNIT}."
        )


def _random_feasible_units(
    floor_u: np.ndarray,
    cap_u: np.ndarray,
    t: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Sample one composition uniformly-ish over feasible integer allocations."""
    out = floor_u.copy()
    remaining = t - int(out.sum())
    if remaining < 0 or remaining > int((cap_u - out).sum()):
        return None

    weights = weights.astype(float)
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
    bands: Sequence[str] = BANDS,
    n_target: int = 100,
    ratio_min: float = 0.75,
    ratio_max: float = 1.25,
    n_levels: int = 3,
    seed: int = 0,
    include_corners: bool = True,
) -> np.ndarray:
    """Build ``(n_designs, len(bands))`` visit-count designs summing to the nominal budget."""
    nominal = nominal_visits(bands)
    t = int(nominal.sum()) // UNIT
    lower, upper = ratio_bounds(nominal, ratio_min, ratio_max)
    floor_u = to_units(lower)
    cap_u = to_units(upper)
    _check_bounds_feasible(floor_u, cap_u, t)
    designs: set[tuple[int, ...]] = set()

    def add(units: np.ndarray | Iterable[int]) -> None:
        u = np.asarray(units, dtype=np.int64)
        if u.shape != (len(bands),):
            return
        if int(u.sum()) != t:
            return
        if np.any(u < floor_u) or np.any(u > cap_u):
            return
        designs.add(tuple(int(x) for x in u))

    add(to_units(nominal))

    for row in enumerate_ratio_grid(nominal, ratio_min, ratio_max, n_levels=n_levels):
        add(to_units(row))

    if include_corners:
        for band in range(len(bands)):
            rest = t - int(floor_u[band])
            if rest >= 0:
                filled = proportional_fill(rest, floor_u, cap_u, nominal, exclude=band)
                if filled is not None:
                    filled = filled.copy()
                    filled[band] = floor_u[band]
                    add(filled)

            rest = t - int(cap_u[band])
            if rest >= 0:
                filled = proportional_fill(rest, floor_u, cap_u, nominal, exclude=band)
                if filled is not None:
                    filled = filled.copy()
                    filled[band] = cap_u[band]
                    add(filled)

    rng = np.random.default_rng(seed)
    max_tries = max(500, 50 * n_target)
    tries = 0
    while len(designs) < n_target and tries < max_tries:
        tries += 1
        draw = _random_feasible_units(floor_u, cap_u, t, nominal, rng)
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
        keep_nominal = np.all(to_visits(arr) == nominal, axis=1)
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
    _validate_designs(visits, nominal, lower, upper)
    return visits


def _validate_designs(
    visits: np.ndarray, nominal: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> None:
    if not np.all(visits.sum(axis=1) == nominal.sum()):
        raise AssertionError("some designs do not sum to the nominal budget")
    if np.any(visits < lower[None, :]) or np.any(visits > upper[None, :]):
        raise AssertionError("design outside ratio bounds")
    if not np.all(visits % UNIT == 0):
        raise AssertionError("design visit counts are not multiples of 10")
    if not any(np.all(visits == nominal, axis=1)):
        raise AssertionError("nominal design missing from output")


def write_design_args(
    visits: np.ndarray,
    bands: Sequence[str],
    name: str,
    out_dir: str,
    designs_dir: str,
    header: str,
    command: str,
) -> tuple[str, str]:
    """Write the design ``.npy`` and its ``design_args_<name>.yaml``; return both paths.

    The YAML stores an absolute ``input_designs_path``, so moving the ``.npy``
    afterwards breaks the YAML unless it is rewritten.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(designs_dir, exist_ok=True)
    npy_path = os.path.abspath(os.path.join(designs_dir, f"{name}.npy"))
    yaml_path = os.path.abspath(os.path.join(out_dir, f"design_args_{name}.yaml"))

    np.save(npy_path, visits)
    design_args = {
        "labels": list(bands),
        # "variable" + an explicit path bypasses step/lower/upper/sum entirely;
        # the file *is* the design pool.
        "input_type": "variable",
        "input_designs_path": npy_path,  # absolute path required by the loader
    }
    with open(yaml_path, "w") as f:
        f.write(f"# {header}\n")
        f.write(f"# Regenerate with:\n#   {command}\n")
        yaml.safe_dump(design_args, f, sort_keys=False, default_flow_style=None)
    return npy_path, yaml_path


def plot_designs(
    visits: np.ndarray,
    bands: Sequence[str],
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
    nominal = nominal_visits(bands)
    lower, cap = ratio_bounds(nominal, ratio_min, ratio_max)
    cap = cap.astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    norm = visits / cap[None, :]

    color_vals = visits[:, color_dim]
    cmin, cmax = color_vals.min(), color_vals.max()
    colormap = plt.get_cmap(cmap)
    colors = colormap((color_vals - cmin) / (cmax - cmin + 1e-10))

    for i in range(n_designs):
        ax.plot(x, norm[i], color=colors[i], alpha=0.45, linewidth=0.8, zorder=1)

    ax.plot(x, nominal / cap, color="black", linewidth=2.5, marker="o", label="Nominal", zorder=3)
    ax.plot(x, np.ones(n_dims), color="0.6", linewidth=1.0, linestyle="--", label=f"{ratio_max:.2f}x nominal", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"${b}$" for b in bands], fontsize=13)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel(f"visits / ({ratio_max:.2f}x nominal)", fontsize=12)
    for xi in range(n_dims):
        ax.axvline(xi, color="0.85", linewidth=0.8, zorder=0)
        ax.text(xi, 1.045, f"{int(cap[xi])}", ha="center", va="bottom", fontsize=8, color="0.4")
        ax.text(xi, -0.055, f"{int(lower[xi])}", ha="center", va="top", fontsize=8, color="0.4")

    ax.set_title(
        f"NumVisits design space: {n_designs} designs, all summing to {int(nominal.sum())} visits",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=cmin, vmax=cmax))
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(f"${bands[color_dim]}$ visits", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _default_out_dir() -> str:
    """Experiment config dir, so generated design_args sit beside the hand-written ones."""
    from bedcosmo.util import get_experiment_config_path

    return os.path.dirname(str(get_experiment_config_path("num_visits", "train_args.yaml")))


def _designs_dir() -> str:
    scratch = os.environ.get("SCRATCH", "/pscratch/sd/a/ashandon")
    return os.path.join(scratch, "bedcosmo", "num_visits", "designs")


def main(argv: list[str] | None = None) -> np.ndarray:
    parser = argparse.ArgumentParser(description="Generate num_visits design arrays.")
    parser.add_argument(
        "--bands", default="".join(BANDS),
        help=f"Filters to vary, in design-column order (default: {''.join(BANDS)}; e.g. gri)",
    )
    parser.add_argument(
        "--name", default=None,
        help="Names <name>.npy and design_args_<name>.yaml "
             "(default: <bands>_<n>_<YYYYMMDD_HHMMSS>)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Dir for the design_args_*.yaml (default: experiment config dir)",
    )
    parser.add_argument(
        "--designs-dir", default=None,
        help="Dir for the design .npy (default: $SCRATCH/bedcosmo/num_visits/designs)",
    )
    parser.add_argument("--plot", default=None, help="Parallel-coordinates .png path (default: beside the .npy)")
    parser.add_argument("--n-target", type=int, default=100, help="Target number of designs")
    parser.add_argument("--ratio-min", type=float, default=0.75, help="Lower ratio limit (all bands)")
    parser.add_argument("--ratio-max", type=float, default=1.25, help="Upper ratio limit (all bands)")
    parser.add_argument("--n-levels", type=int, default=3, help="Ratio grid levels per band")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random fill / subsample")
    parser.add_argument("--no-corners", action="store_true", help="Skip single-band floor/cap corners")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    bands = list(args.bands)
    nominal = nominal_visits(bands)
    budget = int(nominal.sum())

    visits = generate_designs(
        bands=bands,
        n_target=args.n_target,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        n_levels=args.n_levels,
        seed=args.seed,
        include_corners=not args.no_corners,
    )

    name = args.name or (
        f"{''.join(bands)}_{visits.shape[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    npy_path, yaml_path = write_design_args(
        visits,
        bands,
        name,
        out_dir=args.out_dir or _default_out_dir(),
        designs_dir=args.designs_dir or _designs_dir(),
        header=(
            f"{visits.shape[0]} designs over {bands}, each summing to exactly {budget} "
            f"visits, every band within {args.ratio_min:g}-{args.ratio_max:g}x nominal."
        ),
        command=f"python -m bedcosmo.num_visits.design {shlex.join(argv)}".rstrip(),
    )

    lower, upper = ratio_bounds(nominal, args.ratio_min, args.ratio_max)
    print(f"bands: {bands}  nominal: {nominal.tolist()}")
    print(f"wrote {visits.shape[0]} designs -> {npy_path}")
    print(f"all sum to {budget}: {bool(np.all(visits.sum(1) == budget))}")
    print(f"ratio range [{args.ratio_min}, {args.ratio_max}]")
    print("per-band min:", visits.min(0).astype(int).tolist())
    print("per-band max:", visits.max(0).astype(int).tolist())
    print("per-band lower bound:", lower.astype(int).tolist())
    print("per-band upper bound:", upper.astype(int).tolist())
    print("nominal present:", any(np.all(visits == nominal, axis=1)))

    plot_path = os.path.abspath(args.plot or os.path.splitext(npy_path)[0] + ".png")
    plot_designs(visits, bands, plot_path, ratio_min=args.ratio_min, ratio_max=args.ratio_max)
    print(f"wrote design-space plot -> {plot_path}")
    print(f"\nwrote design_args -> {yaml_path}")
    print(f"train with: --design-args-path {os.path.basename(yaml_path)}")
    return visits


if __name__ == "__main__":
    main()
