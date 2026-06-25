#!/usr/bin/env python
"""
Compare empirical prior coordinates (a_k, log s, z) across DESI HEALPix patches.

Loads per-healpix fit CSVs (from fit_eazy_weights_to_desi.py), optionally subsamples
to equal N for fair overlays, and writes summary tables + comparison plots.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fit_eazy_weights_to_desi import (
    build_prior_parameter_samples,
    prior_a_column_names,
    prior_quality_mask,
    save_triangle_plot,
)
from .paths import (
    DEFAULT_EMPIRICAL_PRIOR_DIR,
    DEFAULT_HEALPIX,
    find_healpix_weights_csv,
    get_prior_build_dir,
)


def resolve_weights_csv(
    healpix: int,
    outdir: Path | None,
    prior_dir: Path,
) -> Path:
    if outdir is not None:
        p = outdir / "desi_eazy_empirical_weights.csv"
        if p.exists():
            return p
        raise FileNotFoundError(f"Missing weights CSV for healpix {healpix}: {p}")

    found = find_healpix_weights_csv(healpix, prior_dir=prior_dir)
    if found is not None:
        return found
    raise FileNotFoundError(
        f"No weights CSV for healpix {healpix} under {prior_dir} "
        "(expected healpix/hp{id}/ or legacy desi_eazy_hp{id}/)"
    )


def load_prior_sample(
    csv_path: Path,
    healpix: int,
    *,
    n_subsample: int | None,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["healpix"] = healpix
    mask = prior_quality_mask(df)
    good = df.loc[mask].copy()
    if len(good) == 0:
        return good
    if n_subsample is not None and len(good) > n_subsample:
        rng = np.random.default_rng(seed + healpix)
        idx = rng.choice(good.index.to_numpy(), size=n_subsample, replace=False)
        good = good.loc[idx]
    return good


def a_columns(df: pd.DataFrame) -> list[str]:
    return prior_a_column_names(df)


def summary_row(df: pd.DataFrame, healpix: int) -> dict:
    cols = a_columns(df)
    a = df[cols].to_numpy(dtype=float) if len(df) else np.empty((0, len(cols)))
    z = df["z"].to_numpy(dtype=float) if len(df) else np.array([])
    log_s = df["log_c_scale"].to_numpy(dtype=float) if len(df) else np.array([])
    active = (a > 1e-8).sum(axis=1) if a.size else np.array([])
    return {
        "healpix": healpix,
        "n_quality_pass": len(df),
        "z_median": float(np.median(z)) if z.size else np.nan,
        "z_p16": float(np.percentile(z, 16)) if z.size else np.nan,
        "z_p84": float(np.percentile(z, 84)) if z.size else np.nan,
        "log_s_median": float(np.median(log_s)) if log_s.size else np.nan,
        "n_active_templates_median": float(np.median(active)) if active.size else np.nan,
        "frac_a4_zero": float((a[:, 3] <= 1e-8).mean()) if a.shape[0] else np.nan,
        "frac_a10_zero": float((a[:, 9] <= 1e-8).mean()) if a.shape[0] else np.nan,
    }


def plot_marginal_overlays(
    samples: dict[int, pd.DataFrame],
    outdir: Path,
    *,
    params: list[tuple[str, str]],
) -> None:
    """Overlaid 1D histograms per healpix."""
    healpix_ids = sorted(samples.keys())
    cmap = plt.cm.tab10(np.linspace(0, 1, len(healpix_ids)))

    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 3.5))
    if len(params) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, params):
        for color, hp in zip(cmap, healpix_ids):
            df = samples[hp]
            if len(df) == 0 or col not in df.columns:
                continue
            vals = df[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            ax.hist(
                vals,
                bins=35,
                histtype="step",
                density=True,
                label=str(hp),
                color=color,
                linewidth=1.4,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Prior-quality fits by HEALPix (equal subsample when --n-subsample set)")
    fig.tight_layout()
    fig.savefig(outdir / "marginals_by_healpix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compact_triangle(
    samples: dict[int, pd.DataFrame],
    outdir: Path,
    *,
    param_indices: tuple[int, ...] = (3, 9, 12, 13),
) -> None:
    """Small corner plot: a4, a10, log s, z with per-healpix colors."""
    healpix_ids = sorted(samples.keys())
    labels_all = [
        r"$a_4$",
        r"$a_{10}$",
        r"$\log s$",
        r"$z$",
    ]
    n = len(param_indices)
    cmap = {hp: plt.cm.tab10(i % 10) for i, hp in enumerate(healpix_ids)}

    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n), squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            for hp in healpix_ids:
                df = samples[hp]
                if len(df) == 0:
                    continue
                cols = a_columns(df)
                a_arr = df[cols].to_numpy(dtype=float)
                log_s = df["log_c_scale"].to_numpy(dtype=float)
                z = df["z"].to_numpy(dtype=float)
                joint = np.column_stack([a_arr, log_s, z])
                xi = joint[:, param_indices[j]]
                yi = joint[:, param_indices[i]]
                ok = np.isfinite(xi) & np.isfinite(yi)
                if i == j:
                    ax.hist(
                        xi[ok],
                        bins=30,
                        histtype="step",
                        density=True,
                        color=cmap[hp],
                        linewidth=1.2,
                        label=str(hp),
                    )
                else:
                    ax.scatter(
                        xi[ok],
                        yi[ok],
                        s=6,
                        alpha=0.35,
                        color=cmap[hp],
                        edgecolors="none",
                        label=str(hp),
                    )
            if i == n - 1:
                ax.set_xlabel(labels_all[j])
            if j == 0:
                ax.set_ylabel(labels_all[i])
            ax.tick_params(labelsize=7)

    handles = [
        plt.Line2D([0], [0], color=cmap[hp], lw=2, label=str(hp)) for hp in healpix_ids
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    fig.suptitle(r"Overlay: $a_4$, $a_{10}$, $\log s$, $z$", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "compact_triangle_by_healpix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_healpix_triangles(
    samples: dict[int, pd.DataFrame],
    outdir: Path,
) -> None:
    """One small triangle per healpix (same 4 params as compact overlay)."""
    sub = outdir / "per_healpix"
    sub.mkdir(parents=True, exist_ok=True)
    idx = (3, 9, 12, 13)
    labels = [r"$a_4$", r"$a_{10}$", r"$\log s$", r"$z$"]

    for hp, df in sorted(samples.items()):
        if len(df) < 3:
            continue
        cols = a_columns(df)
        a_arr = df[cols].to_numpy(dtype=float)
        joint_full, _ = build_prior_parameter_samples(
            a_arr,
            df["log_c_scale"].to_numpy(dtype=float),
            df["z"].to_numpy(dtype=float),
        )
        joint = joint_full[:, idx]
        save_triangle_plot(
            sub,
            joint,
            labels,
            filename=f"triangle_hp{hp}.png",
            title=f"HEALPix {hp} (n={len(df)})",
            panel_size=1.8,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--healpix",
        type=int,
        nargs="+",
        default=DEFAULT_HEALPIX,
    )
    parser.add_argument(
        "--build-name",
        default=DEFAULT_EMPIRICAL_PRIOR_DIR,
        help="Prior build directory under num_visits (parent of healpix/hp* subdirs).",
    )
    parser.add_argument(
        "--prior-dir",
        type=Path,
        default=None,
        help="Override prior build root (default: $SCRATCH/bedcosmo/num_visits/<build-name>).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Comparison plot output directory (default: <prior-dir>/healpix_prior_comparison).",
    )
    parser.add_argument(
        "--n-subsample",
        type=int,
        default=500,
        help="Subsample each patch to this many quality-pass rows (fair overlay).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--weights-dir",
        action="append",
        default=[],
        metavar="HEALPIX=PATH",
        help="Override CSV path, e.g. 23040=/path/to/weights.csv",
    )
    args = parser.parse_args()

    prior_dir = (
        Path(os.path.expanduser(args.prior_dir))
        if args.prior_dir is not None
        else get_prior_build_dir(args.build_name)
    )
    outdir = (
        Path(os.path.expanduser(args.outdir))
        if args.outdir is not None
        else prior_dir / "healpix_prior_comparison"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    overrides: dict[int, Path] = {}
    for item in args.weights_dir:
        hp_s, _, path = item.partition("=")
        overrides[int(hp_s)] = Path(os.path.expanduser(path))

    samples: dict[int, pd.DataFrame] = {}
    summaries = []

    for hp in args.healpix:
        try:
            if hp in overrides:
                csv_path = overrides[hp]
            else:
                csv_path = resolve_weights_csv(hp, None, prior_dir)
            df = load_prior_sample(
                csv_path,
                hp,
                n_subsample=args.n_subsample,
                seed=args.seed,
            )
            samples[hp] = df
            summaries.append(summary_row(df, hp))
            print(f"HEALPIX {hp}: {len(df)} rows from {csv_path}")
        except FileNotFoundError as e:
            print(f"SKIP HEALPIX {hp}: {e}")

    if not samples:
        raise SystemExit("No healpix CSVs loaded.")

    summary_df = pd.DataFrame(summaries)
    summary_path = outdir / "healpix_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print(summary_df.to_string(index=False))

    plot_marginal_overlays(
        samples,
        outdir,
        params=[
            ("z", r"$z$"),
            ("log_c_scale", r"$\log s$"),
            ("a4", r"$a_4$"),
            ("a10", r"$a_{10}$"),
        ],
    )
    plot_compact_triangle(samples, outdir)
    plot_per_healpix_triangles(samples, outdir)
    print(f"\nPlots written to {outdir}")


if __name__ == "__main__":
    main()
