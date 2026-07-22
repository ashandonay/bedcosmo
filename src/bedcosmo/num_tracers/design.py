"""Generate explicit num_tracers design arrays (``input_designs_path``).

The num_tracers design vector is a per-class observation split ``[BGS, LRG, ELG, QSO]``
whose *sum is the total-observation budget* (``total_obs_multiplier``; see
``NumTracers.sigma_scaling_factor``). Sum 1.0 is the nominal DESI footprint, 1.2 means
"20% more observations". That makes the budget a design axis, not a separate knob.

Two build modes, mirroring the two ways to measure a budget trend:

``scaled``
    One single-design pool per scale factor: the nominal split times ``s``. Each run
    trains its own flow on one design, so each contributes one point to
    ``plotting.compare_increasing_design``. Simple, but see the warning below.

``pool``
    One multi-design pool spanning a range of budgets, so a single amortized flow
    measures the whole axis. Preferred: the training batch is
    ``n_particles_per_device * n_designs``, so a one-row pool is data-starved relative
    to a grid run, and EIG is a variational *lower* bound -- an underfit flow reads low.
    Comparing separately-trained flows conflates fit quality with the design effect;
    comparing designs inside one flow does not.

Example::

    python -m bedcosmo.num_tracers.design scaled --scales 1.00 1.05 1.10 1.15 1.20 1.25
    python -m bedcosmo.num_tracers.design pool --sum-lower 1.0 --sum-upper 1.25

Outputs land in two places: the design array goes to
``$SCRATCH/bedcosmo/num_tracers/designs/<name>.npy`` (``--designs-dir``), matching
``bedcosmo.num_visits.design``, while the ``design_args_<name>.yaml`` that points at it
goes to the experiment config dir (``--out-dir``) so ``--design-args-path`` can find it.
The YAML stores an absolute ``input_designs_path``, so relocating the ``.npy`` afterwards
requires rewriting the YAML.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import yaml

LABELS = ["BGS", "LRG", "ELG", "QSO"]
# Mirror NumTracers: DESI CSVs live under $HOME/data/desi/bao_<dataset>/
_DATA_ROOT = os.path.join(os.environ.get("HOME", ""), "data", "desi")
# Default per-class grid from design_args_dr1.yaml (hand-set caps, not target-availability
# limits -- the physical target caps are ~10^6 tracers and never bind at these budgets).
DEFAULT_STEP = [0.01, 0.02, 0.02, 0.01]
DEFAULT_LOWER = [0.02, 0.1, 0.1, 0.1]
DEFAULT_UPPER = [0.0585, 0.3820, 0.5, 0.3291]
TOL = 1e-9


def nominal_vector(dataset: str = "dr1") -> np.ndarray:
    """Per-class nominal design vector (sums to 1.0), reproducing ``NumTracers.__init__``."""
    import pandas as pd

    csv = os.path.join(_DATA_ROOT, f"bao_{dataset}", "desi_tracers.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(f"DESI tracer table not found: {csv}")
    df = pd.read_csv(csv)
    v0 = df.groupby("class").sum(numeric_only=True)["observed"].reindex(LABELS).values
    return np.asarray(v0, dtype=np.float64)


def scale_tag(s: float) -> str:
    """1.05 -> '05', 1.10 -> '10', 1.00 -> '00', 1.5 -> '50'."""
    return f"{round((s - 1.0) * 100):02d}"


def uniform_scaled(v0: np.ndarray, scale: float) -> np.ndarray:
    """The nominal split scaled uniformly by ``scale`` (so the sum == ``scale``)."""
    return np.asarray(v0, dtype=np.float64) * float(scale)


def budget_pool(
    sum_lower: float = 1.0,
    sum_upper: float = 1.2,
    step: Sequence[float] = DEFAULT_STEP,
    lower: Sequence[float] = DEFAULT_LOWER,
    upper: Sequence[float] = DEFAULT_UPPER,
    dataset: str = "dr1",
    include_nominal_scales: Iterable[float] | None = None,
    n_target: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Cartesian per-class grid keeping rows whose sum lies in ``[sum_lower, sum_upper]``.

    Reproduces the grid ``NumTracers`` builds internally from ``step``/``lower``/``upper``
    plus the sum constraint, but materializes it so the pool can be inspected, plotted and
    pinned to a file. Verified against the experiment's own grid: the defaults give 287
    designs at sum == 1.0 and 2447 over [1.0, 1.2], matching ``design_args_dr1.yaml`` and
    ``design_args_dr1_budget.yaml`` respectively.

    ``include_nominal_scales`` force-adds the uniformly scaled nominal designs (which
    generally fall between grid nodes) so the pool contains the exact points the ``scaled``
    mode would have produced -- useful as within-pool reference designs.

    ``n_target`` caps the pool size by random subsample. Since the training batch is
    ``n_particles_per_device * n_designs``, pool size sets the memory/data footprint;
    this is the lever for hitting a target footprint without coarsening ``step``. Pinned
    rows (nominal and any ``include_nominal_scales``) always survive the subsample.

    Note ``np.arange`` excludes the endpoint, so ``upper`` is exclusive -- matching
    ``NumTracers``, which builds its axes the same way.
    """
    _validate_axis_spec(step, lower, upper)
    axes = [
        np.arange(lower[i], upper[i], step[i])
        for i in range(len(LABELS))
    ]
    for i, ax in enumerate(axes):
        if ax.size == 0:
            raise ValueError(
                f"{LABELS[i]}: empty axis -- lower={lower[i]} upper={upper[i]} "
                f"step={step[i]} yields no grid points"
            )
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = np.stack([g.ravel() for g in mesh], axis=1)
    total = flat.sum(axis=1)
    keep = (total >= sum_lower - TOL) & (total <= sum_upper + TOL)
    pool = flat[keep]

    pinned = np.empty((0, len(LABELS)), dtype=np.float64)
    if include_nominal_scales:
        v0 = nominal_vector(dataset)
        pinned = np.array([uniform_scaled(v0, s) for s in include_nominal_scales])
        pool = np.vstack([pool, pinned])

    # Stable, de-duplicated ordering so a rebuild is reproducible.
    pool = np.unique(np.round(pool, 12), axis=0)

    if n_target is not None and pool.shape[0] > n_target:
        pool = _subsample(pool, np.round(pinned, 12), n_target, seed)

    _validate_pool(pool, sum_lower, sum_upper)
    return pool


def _validate_axis_spec(
    step: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> None:
    for name, seq in (("step", step), ("lower", lower), ("upper", upper)):
        if len(seq) != len(LABELS):
            raise ValueError(
                f"{name} must have {len(LABELS)} entries (one per {LABELS}), got {len(seq)}"
            )
    for i in range(len(LABELS)):
        if step[i] <= 0:
            raise ValueError(f"{LABELS[i]}: step must be positive, got {step[i]}")
        if lower[i] >= upper[i]:
            raise ValueError(
                f"{LABELS[i]}: lower ({lower[i]}) must be below upper ({upper[i]})"
            )


def _subsample(
    pool: np.ndarray, pinned: np.ndarray, n_target: int, seed: int
) -> np.ndarray:
    """Randomly thin ``pool`` to ``n_target`` rows, always keeping ``pinned`` rows."""
    if pinned.size:
        is_pinned = (pool[:, None, :] == pinned[None, :, :]).all(-1).any(1)
    else:
        is_pinned = np.zeros(pool.shape[0], dtype=bool)
    keep_rows = pool[is_pinned]
    if keep_rows.shape[0] >= n_target:
        raise ValueError(
            f"n_target ({n_target}) is below the {keep_rows.shape[0]} pinned designs; "
            "raise n_target or drop some include_nominal_scales"
        )
    other = pool[~is_pinned]
    rng = np.random.default_rng(seed)
    pick = rng.choice(other.shape[0], size=n_target - keep_rows.shape[0], replace=False)
    out = np.vstack([keep_rows, other[pick]])
    return out[np.lexsort(out.T[::-1])]


def _validate_pool(pool: np.ndarray, sum_lower: float, sum_upper: float) -> None:
    if pool.ndim != 2 or pool.shape[1] != len(LABELS):
        raise AssertionError(f"pool must be (n, {len(LABELS)}), got {pool.shape}")
    if pool.shape[0] == 0:
        raise AssertionError("pool is empty -- widen the sum range or per-class bounds")
    total = pool.sum(axis=1)
    if total.min() < sum_lower - 1e-6 or total.max() > sum_upper + 1e-6:
        raise AssertionError(
            f"pool sums [{total.min():.6f}, {total.max():.6f}] escape "
            f"[{sum_lower}, {sum_upper}]"
        )
    if np.any(pool <= 0):
        raise AssertionError("non-positive class fraction in pool")


def write_design_args(
    designs: np.ndarray,
    name: str,
    out_dir: str,
    header: str = "",
    yaml_name: str | None = None,
    designs_dir: str | None = None,
) -> tuple[str, str]:
    """Write the design ``.npy`` and its ``design_args_*.yaml``; return both paths.

    The two land in different places by default, because they play different roles:
    the ``.npy`` is bulk data (``designs_dir``, default ``$SCRATCH/.../designs``, mirroring
    ``bedcosmo.num_visits.design``), while the YAML is config that ``--design-args-path``
    resolves against the experiment config dir (``out_dir``). The YAML always stores an
    absolute ``input_designs_path``, so the two may live anywhere relative to each other --
    but moving the ``.npy`` afterwards breaks the YAML unless it is rewritten.

    ``name`` names the ``.npy`` (an explicit ``.npy`` suffix is accepted and stripped);
    ``yaml_name`` names the YAML when it should differ from ``name``.
    """
    designs_dir = designs_dir or _designs_dir()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(designs_dir, exist_ok=True)
    stem = name[:-4] if name.endswith(".npy") else name
    npy_path = os.path.abspath(os.path.join(designs_dir, f"{stem}.npy"))
    yaml_path = os.path.abspath(
        os.path.join(out_dir, f"design_args_{yaml_name or stem}.yaml")
    )

    np.save(npy_path, designs)
    design_args = {
        "labels": LABELS,
        # "variable" + an explicit path bypasses step/lower/upper/sum entirely;
        # the file *is* the design pool.
        "input_type": "variable",
        "input_designs_path": npy_path,  # absolute path required by the loader
    }
    with open(yaml_path, "w") as f:
        if header:
            f.write(f"# {header}\n")
        f.write("# generated by bedcosmo.num_tracers.design\n")
        yaml.safe_dump(design_args, f, sort_keys=False)
    return npy_path, yaml_path


def plot_designs(
    designs: np.ndarray, out_path: str, cmap: str = "viridis", title: str = ""
) -> None:
    """Parallel-coordinates view of the design pool, colored by total budget.

    Handles a single-row ``designs`` (one plot per scaled design), in which case the
    budget colorbar is dropped and each class is annotated with its ratio to nominal.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_designs, n_dims = designs.shape
    x = np.arange(n_dims)

    # Normalize per class by nominal, so y reads directly as "x nominal" and all four
    # classes sit on comparable footing. A uniformly scaled design is then a flat line
    # at its budget -- the shape shows how a design departs from the nominal split.
    v0 = None
    try:
        v0 = nominal_vector()
    except FileNotFoundError:
        pass
    ref = v0 if v0 is not None else designs.mean(axis=0)
    norm = designs / ref[None, :]

    fig, ax = plt.subplots(figsize=(10, 6))

    budget = designs.sum(axis=1)
    cmin, cmax = float(budget.min()), float(budget.max())
    colormap = plt.get_cmap(cmap)
    # A single design (or an all-equal-budget pool) has no spread to color by; place it
    # mid-map rather than dividing by ~0.
    spread = cmax - cmin
    if spread > 1e-12:
        colors = colormap((budget - cmin) / spread)
    else:
        colors = colormap(np.full(n_designs, 0.5))

    # Mark each design point. Dots carry the detail for a small pool; for a large one
    # they would smear into a band, so taper size/opacity as the pool grows.
    if n_designs <= 12:
        msize, lw, alpha = 7.0, 1.8, 0.95
    elif n_designs <= 200:
        msize, lw, alpha = 3.0, 1.0, 0.55
    else:
        msize, lw, alpha = 1.5, 0.8, 0.35

    for i in range(n_designs):
        ax.plot(
            x, norm[i], color=colors[i], alpha=alpha, linewidth=lw,
            marker="o", markersize=msize, zorder=1,
        )

    if v0 is not None:
        ax.axhline(
            1.0, color="black", linewidth=2.0, linestyle="--",
            label="Nominal split (1.0x)", zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=13)
    ax.set_ylabel("class fraction / nominal", fontsize=12)
    for xi in range(n_dims):
        ax.axvline(xi, color="0.85", linewidth=0.8, zorder=0)

    if spread > 1e-12:
        ax.set_title(
            f"{title or 'NumTracers design pool'}: {n_designs} designs, "
            f"budget {cmin:.3f}-{cmax:.3f}x nominal",
            fontsize=13,
        )
    else:
        # Single design (or one shared budget): annotate the value instead of a range,
        # and pad the y-range so a flat line is not drawn on top of the axis.
        ax.set_title(
            f"{title or 'NumTracers design'}: {n_designs} design"
            f"{'s' if n_designs != 1 else ''}, budget {cmin:.3f}x nominal",
            fontsize=13,
        )
        lo, hi = min(1.0, norm.min()), max(1.0, norm.max())
        pad = max(0.05, 0.35 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        for xi in range(n_dims):
            ax.annotate(
                f"{norm[0, xi]:.3f}x", (xi, norm[0, xi]), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=9, color="0.25",
            )

    if v0 is not None:
        ax.legend(loc="upper right", fontsize=10)

    if spread > 1e-12:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=cmin, vmax=cmax))
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("total observation budget (design sum)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _default_out_dir() -> str:
    """Experiment config dir, so generated design_args sit beside the hand-written ones."""
    from bedcosmo.util import get_experiment_config_path

    return os.path.dirname(str(get_experiment_config_path("num_tracers", "train_args.yaml")))


def _designs_dir() -> str:
    """Scratch dir for design arrays, mirroring ``bedcosmo.num_visits.design``."""
    scratch = os.environ.get("SCRATCH", "/pscratch/sd/a/ashandon")
    return os.path.join(scratch, "bedcosmo", "num_tracers", "designs")


def _timestamp(when: datetime | None = None) -> str:
    return (when or datetime.now()).strftime("%Y%m%d_%H%M%S")


def _cmd_scaled(args: argparse.Namespace) -> np.ndarray:
    v0 = nominal_vector(args.dataset)
    print(f"dataset={args.dataset}  labels={LABELS}")
    print(f"nominal v0={np.round(v0, 4).tolist()}  sum={v0.sum():.4f}\n")

    rows = []
    for s in args.scales:
        tag = scale_tag(s)
        design = uniform_scaled(v0, s)
        rows.append(design)
        npy_path, yaml_path = write_design_args(
            design,
            f"nominal_scaled_p{tag}",
            args.out_dir,
            header=f"Uniform nominal scaling x{s:g} (design sum == {s:g})",
            # Existing convention: nominal_scaled_pNN.npy <-> design_args_nominal_pNN.yaml
            yaml_name=f"nominal_p{tag}",
            designs_dir=args.designs_dir,
        )
        print(f"s={s:<5g} tag={tag}  sum={design.sum():.4f}  "
              f"design={np.round(design, 4).tolist()}")
        print(f"           -> {os.path.basename(npy_path)}, {os.path.basename(yaml_path)}")
        # One plot per design, beside its own .npy.
        args._plots.append(
            (os.path.splitext(npy_path)[0], design.reshape(1, -1), f"Nominal x{s:g}")
        )
    return np.array(rows)


def _cmd_pool(args: argparse.Namespace) -> np.ndarray:
    scales = args.include_scales if args.include_scales else None
    pool = budget_pool(
        sum_lower=args.sum_lower,
        sum_upper=args.sum_upper,
        step=args.step,
        lower=args.lower,
        upper=args.upper,
        dataset=args.dataset,
        include_nominal_scales=scales,
        n_target=args.n_target,
        seed=args.seed,
    )
    # Date-stamp the default so repeated builds accumulate instead of overwriting
    # (mirrors bedcosmo.num_visits.design). An explicit --name opts out.
    if args.name:
        name = args.name
    else:
        span = f"{args.sum_lower:g}_{args.sum_upper:g}".replace(".", "p")
        name = f"pool_sum{span}_{pool.shape[0]}_{_timestamp()}"
    npy_path, yaml_path = write_design_args(
        pool,
        name,
        args.out_dir,
        header=(
            f"Budget pool: sum in [{args.sum_lower:g}, {args.sum_upper:g}], "
            f"{pool.shape[0]} designs"
        ),
        designs_dir=args.designs_dir,
    )
    total = pool.sum(axis=1)
    print(f"wrote {pool.shape[0]} designs -> {npy_path}")
    print(f"design_args -> {yaml_path}")
    print(f"grid step:  {list(args.step)}")
    print(f"grid lower: {list(args.lower)}  (inclusive)")
    print(f"grid upper: {list(args.upper)}  (exclusive)")
    if args.n_target is not None:
        print(f"subsampled to n_target={args.n_target} with seed={args.seed}")
    print(f"budget range [{total.min():.4f}, {total.max():.4f}]")
    print("per-class min:", np.round(pool.min(0), 4).tolist())
    print("per-class max:", np.round(pool.max(0), 4).tolist())
    if scales:
        print(f"forced nominal scales present: {list(scales)}")
    # Training batch = n_particles_per_device * n_designs; match a proven footprint.
    print(
        f"\nsizing: batch = npd x {pool.shape[0]}; for the 28,700/device footprint "
        f"use --n-particles-per-device {max(1, round(28700 / pool.shape[0]))}"
    )
    args._plots.append((os.path.splitext(npy_path)[0], pool, ""))
    return pool


def main(argv: list[str] | None = None) -> np.ndarray:
    parser = argparse.ArgumentParser(
        description="Generate num_tracers design arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="dr1", help="DESI dataset suffix (default: dr1)")
    parser.add_argument(
        "--out-dir", default=None,
        help="Dir for the design_args_*.yaml (default: experiment config dir)",
    )
    parser.add_argument(
        "--designs-dir", default=None,
        help="Dir for the design .npy (default: $SCRATCH/bedcosmo/num_tracers/designs)",
    )
    parser.add_argument(
        "--plot", default=None,
        help="Override the parallel-coordinates .png path (default: beside the .npy)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the design-space plot")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_scaled = sub.add_parser("scaled", help="One single-design pool per uniform scale factor")
    p_scaled.add_argument(
        "--scales", type=float, nargs="+",
        default=[1.00, 1.05, 1.10, 1.15, 1.20, 1.25],
        help="Uniform scale factors (design sum == scale). Default: 1.00..1.25 by 0.05",
    )
    p_scaled.set_defaults(func=_cmd_scaled)

    p_pool = sub.add_parser("pool", help="One multi-design pool spanning a budget range")
    p_pool.add_argument("--sum-lower", type=float, default=1.0, help="Min design sum (budget)")
    p_pool.add_argument("--sum-upper", type=float, default=1.25, help="Max design sum (budget)")
    p_pool.add_argument(
        "--name", default=None,
        help="Filename for the .npy, with or without the .npy suffix; also names the "
             "design_args_<name>.yaml. Default is date-stamped "
             "(pool_sum<lo>_<hi>_<n>_<YYYYMMDD_HHMMSS>) so builds accumulate; "
             "pass --name for a stable, overwritable filename",
    )
    p_pool.add_argument(
        "--include-scales", type=float, nargs="*", default=None,
        help="Force-add uniformly scaled nominal designs as within-pool reference points",
    )
    p_pool.add_argument(
        "--step", type=float, nargs=4, default=DEFAULT_STEP,
        metavar=tuple(LABELS),
        help=f"Per-class grid step (default: {DEFAULT_STEP})",
    )
    p_pool.add_argument(
        "--lower", type=float, nargs=4, default=DEFAULT_LOWER,
        metavar=tuple(LABELS),
        help=f"Per-class lower bound, inclusive (default: {DEFAULT_LOWER})",
    )
    p_pool.add_argument(
        "--upper", type=float, nargs=4, default=DEFAULT_UPPER,
        metavar=tuple(LABELS),
        help=f"Per-class upper bound, EXCLUSIVE (default: {DEFAULT_UPPER})",
    )
    p_pool.add_argument(
        "--n-target", type=int, default=None,
        help="Cap pool size by random subsample (pinned rows always kept)",
    )
    p_pool.add_argument("--seed", type=int, default=0, help="RNG seed for --n-target subsample")
    p_pool.set_defaults(func=_cmd_pool)

    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = _default_out_dir()
    if args.designs_dir is None:
        args.designs_dir = _designs_dir()

    args._plots: list[tuple[str, np.ndarray, str]] = []
    designs = args.func(args)

    # Always render a design plot beside each .npy (as num_visits.design does), unless
    # suppressed. --plot overrides the location, and so only applies to a single plot.
    if not args.no_plot:
        if args.plot and len(args._plots) > 1:
            parser.error(
                f"--plot names one file but this run produced {len(args._plots)} plots; "
                "drop --plot to write each beside its .npy, or pass a single --scales value"
            )
        for stem, arr, title in args._plots:
            plot_path = os.path.abspath(args.plot or f"{stem}.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plot_designs(arr, plot_path, title=title)
            print(f"wrote design plot -> {plot_path}")
    return designs


if __name__ == "__main__":
    main()
