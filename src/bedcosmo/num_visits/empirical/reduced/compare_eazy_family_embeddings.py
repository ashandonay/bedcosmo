#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Compare PCA cutoff and whitening choices for EAZY family discovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .discover_eazy_spectral_families import (  # noqa: E402
    decode_family_bases,
    fit_population_embedding,
    load_fit_population,
    load_subset_searches,
)

GRID = "#D9DDE3"
COLORS = {"standardized": "#3366CC", "raw": "#DC3912"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights-csv", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variance-thresholds",
        type=float,
        nargs="+",
        default=(0.85, 0.90, 0.95, 1.00),
    )
    parser.add_argument("--min-cluster-size", type=int, default=300)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--required-family-coverage", type=float, default=0.80)
    parser.add_argument("--max-chi2-dof", type=float, default=1.2)
    parser.add_argument("--max-delta-chi2-dof", type=float, default=0.05)
    parser.add_argument("--max-color-rms", type=float, default=0.02)
    return parser.parse_args()


def run_comparison(args: argparse.Namespace) -> pd.DataFrame:
    table, weights = load_fit_population(args.weights_csv)
    searches = load_subset_searches(
        args.cohort_root,
        table["targetid"].to_numpy(np.int64),
        max_chi2_dof=args.max_chi2_dof,
        max_delta_chi2_dof=args.max_delta_chi2_dof,
        max_color_rms=args.max_color_rms,
    )
    results: dict[tuple[str, float], dict[str, object]] = {}
    for scaling in ("standardized", "raw"):
        for threshold in args.variance_thresholds:
            pca, scores, labels, _ = fit_population_embedding(
                weights,
                variance_threshold=threshold,
                clr_eps=1e-5,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                pca_scaling=scaling,
            )
            family_summary, _ = decode_family_bases(
                labels,
                searches,
                required_coverage=args.required_family_coverage,
            )
            resolved = family_summary["meets_required_coverage"].astype(bool)
            passing_count = np.rint(
                family_summary.loc[resolved, "member_count"]
                * family_summary.loc[resolved, "selected_coverage_fraction"]
            ).sum()
            retained = scores.shape[1]
            results[(scaling, float(threshold))] = {
                "labels": labels,
                "row": {
                    "pca_scaling": scaling,
                    "variance_threshold": float(threshold),
                    "n_retained_pcs": retained,
                    "actual_variance_retained": float(
                        pca.explained_variance_ratio_[:retained].sum()
                    ),
                    "n_families": int(labels.max()),
                    "clustered_count": int(np.sum(labels > 0)),
                    "clustered_fraction": float(np.mean(labels > 0)),
                    "resolved_family_count": int(resolved.sum()),
                    "resolved_family_member_count": int(
                        family_summary.loc[resolved, "member_count"].sum()
                    ),
                    "selected_basis_passing_count": int(passing_count),
                },
            }

    baseline_key = ("standardized", 0.90)
    if baseline_key not in results:
        raise ValueError("The comparison must include the 0.90 baseline threshold")
    baseline = np.asarray(results[baseline_key]["labels"], dtype=int)
    rows = []
    for result in results.values():
        labels = np.asarray(result["labels"], dtype=int)
        shared = (baseline > 0) & (labels > 0)
        row = dict(result["row"])
        row["shared_with_baseline_fraction"] = float(np.mean(shared))
        row["ari_including_unclustered"] = float(adjusted_rand_score(baseline, labels))
        row["ari_on_shared_clustered_core"] = (
            float(adjusted_rand_score(baseline[shared], labels[shared]))
            if np.any(shared)
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pca_scaling", "variance_threshold"])


def make_figure(results: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    panels = (
        (axes[0, 0], "n_families", "HDBSCAN families"),
        (axes[0, 1], "clustered_fraction", "Fraction assigned to a family"),
        (axes[1, 0], "ari_including_unclustered", "Agreement with 90% standardized baseline"),
        (
            axes[1, 1],
            "selected_basis_passing_count",
            "Spectra passing selected family bases",
        ),
    )
    for ax, column, title in panels:
        for scaling in ("standardized", "raw"):
            data = results.loc[results["pca_scaling"] == scaling]
            label = "Standardized (whitened)" if scaling == "standardized" else "Raw PC scores"
            ax.plot(
                100 * data["variance_threshold"],
                data[column],
                marker="o",
                lw=2,
                color=COLORS[scaling],
                label=label,
            )
            for row in data.itertuples(index=False):
                ax.annotate(
                    f"{row.n_retained_pcs} PCs",
                    (100 * row.variance_threshold, getattr(row, column)),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=COLORS[scaling],
                )
        ax.set_xlabel("Requested cumulative variance [%]")
        ax.set_title(title, loc="left")
        ax.grid(True, color=GRID, alpha=0.7, linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0, 1].set_ylim(0, 1)
    axes[1, 0].set_ylim(0, 1.03)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Sensitivity of EAZY12 spectral families to PCA compression and whitening")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = run_comparison(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "embedding_choice_comparison.csv"
    figure_path = args.output_dir / "embedding_choice_comparison.png"
    results.to_csv(csv_path, index=False)
    make_figure(results, figure_path)
    print(results.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
