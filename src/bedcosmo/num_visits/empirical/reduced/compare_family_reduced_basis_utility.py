#!/usr/bin/env python
"""Compare PCA/HDBSCAN choices by downstream reduced-template utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from ..paths import DEFAULT_EMPIRICAL_PRIOR_DIR, get_prior_build_dir  # noqa: E402
from ..simplex import DEFAULT_CLR_EPS, weights_to_ilr  # noqa: E402
from .bootstrap_hdbscan_family_stability import (  # noqa: E402
    COLORS,
    ClusterConfig,
    fit_ordered_embedding,
    parse_config,
)
from .discover_eazy_spectral_families import (  # noqa: E402
    DEFAULT_FAMILY_WEIGHT_COVERAGE,
    decode_family_bases,
    load_fit_population,
    load_subset_searches,
    select_balanced_family_bases,
)

GRID = "#D9DDE3"


def summarize_family_weights(labels: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for family in sorted(value for value in np.unique(labels) if value > 0):
        members = labels == family
        for index, mean_weight in enumerate(weights[members].mean(axis=0)):
            rows.append(
                {
                    "family": f"F{family:02d}",
                    "template": f"T{index + 1}",
                    "mean_weight": float(mean_weight),
                }
            )
    return pd.DataFrame(rows)


def summarize_selected_bases(
    selected: pd.DataFrame,
    member_counts: pd.Series,
    *,
    total_population: int,
) -> dict[str, float | int]:
    selected = selected.set_index("family").loc[member_counts.index]
    assigned_count = int(member_counts.sum())
    passing_count = int(selected["balanced_passing_count"].sum())
    all_set_passing_count = int(selected["balanced_subset_passing_count"].sum())
    completeness = passing_count / assigned_count if assigned_count else np.nan
    purity = passing_count / all_set_passing_count if all_set_passing_count else np.nan
    balanced_f1 = (
        2 * completeness * purity / (completeness + purity)
        if completeness + purity > 0
        else np.nan
    )
    weights = member_counts / assigned_count
    return {
        "n_families": len(member_counts),
        "assigned_count": assigned_count,
        "assigned_fraction": assigned_count / total_population,
        "recovered_count": passing_count,
        "recovered_population_fraction": passing_count / total_population,
        "micro_completeness": completeness,
        "micro_purity": purity,
        "micro_balanced_f1": balanced_f1,
        "member_weighted_family_f1": float(
            np.sum(selected["balanced_f1"].to_numpy(float) * weights.to_numpy(float))
        ),
        "median_family_f1": float(selected["balanced_f1"].median()),
        "median_selected_n": float(selected["balanced_n"].median()),
        "mean_selected_n": float(selected["balanced_n"].mean()),
    }


def evaluate_configs(
    weights: np.ndarray,
    searches: dict[int, dict[str, np.ndarray]],
    configs: list[ClusterConfig],
    *,
    scaling: str,
    clr_eps: float,
    family_weight_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ilr = weights_to_ilr(weights, eps=clr_eps)
    summaries: list[dict[str, float | int | str]] = []
    family_rows: list[pd.DataFrame] = []
    for config in configs:
        labels = fit_ordered_embedding(ilr, config, scaling)
        family_summary, candidates = decode_family_bases(
            labels, searches, required_coverage=1.0
        )
        family_weights = summarize_family_weights(labels, weights)
        selected = select_balanced_family_bases(
            candidates,
            family_weights,
            family_weight_coverage=family_weight_coverage,
        )
        members = family_summary.set_index("family")["member_count"].astype(int)
        summary: dict[str, float | int | str] = {
            "config": config.slug,
            "n_components": config.n_components,
            "min_cluster_size": config.min_cluster_size,
            "min_samples": config.min_samples,
            "selection_method": config.selection_method,
        }
        summary.update(
            summarize_selected_bases(selected, members, total_population=len(weights))
        )
        summaries.append(summary)
        details = selected.merge(
            members.rename("member_count"),
            left_on="family",
            right_index=True,
            validate="one_to_one",
        )
        details.insert(0, "config", config.slug)
        details.insert(1, "n_components", config.n_components)
        family_rows.append(details)
        print(
            f"{config.slug}: {summary['n_families']} families, "
            f"{summary['recovered_count']}/{len(weights)} DESI recovered, "
            f"micro completeness={summary['micro_completeness']:.3f}, "
            f"purity={summary['micro_purity']:.3f}",
            flush=True,
        )
    return pd.DataFrame(summaries), pd.concat(family_rows, ignore_index=True)


def make_figure(
    summary: pd.DataFrame,
    families: pd.DataFrame,
    configs: list[ClusterConfig],
    output: Path,
) -> None:
    order = [config.slug for config in configs]
    labels = [f"{config.n_components} PCs" for config in configs]
    indexed = summary.set_index("config").loc[order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)

    width = 0.34
    axes[0, 0].bar(
        x - width / 2,
        indexed["assigned_fraction"],
        width,
        color="#AFC6E9",
        label="Assigned to a family",
    )
    axes[0, 0].bar(
        x + width / 2,
        indexed["recovered_population_fraction"],
        width,
        color="#3366CC",
        label="Passes selected family basis",
    )
    axes[0, 0].set_title("End-to-end DESI population coverage", loc="left")
    axes[0, 0].legend(frameon=False)

    metric_columns = ("micro_completeness", "micro_purity", "micro_balanced_f1")
    metric_labels = ("Completeness", "Purity", "F1")
    metric_colors = ("#3366CC", "#DC3912", "#109618")
    metric_width = 0.24
    for offset, column, label, color in zip(
        (-metric_width, 0, metric_width), metric_columns, metric_labels, metric_colors
    ):
        axes[0, 1].bar(x + offset, indexed[column], metric_width, color=color, label=label)
    axes[0, 1].set_title("Selected family-basis performance", loc="left")
    axes[0, 1].legend(frameon=False)

    box_data = [
        families.loc[families["config"] == config, "balanced_f1"].to_numpy(float)
        for config in order
    ]
    box = axes[1, 0].boxplot(box_data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    axes[1, 0].scatter(
        x + 1,
        indexed["member_weighted_family_f1"],
        marker="D",
        color="#222222",
        label="Member-weighted mean",
        zorder=3,
    )
    axes[1, 0].set_title("Distribution across individual families", loc="left")
    axes[1, 0].legend(frameon=False)

    for index, (config, color) in enumerate(zip(order, COLORS)):
        selected = families.loc[families["config"] == config]
        axes[1, 1].scatter(
            selected["balanced_completeness"],
            selected["balanced_purity"],
            s=np.sqrt(selected["member_count"]) * 5,
            alpha=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=labels[index],
        )
    axes[1, 1].set_xlabel("Family completeness")
    axes[1, 1].set_ylabel("Subset purity")
    axes[1, 1].set_title("Each selected reduced-template set", loc="left")
    axes[1, 1].legend(frameon=False)

    for ax in axes.flat:
        ax.grid(True, axis="y", color=GRID, alpha=0.8, linewidth=0.7)
        ax.set_axisbelow(True)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(x if ax is not axes[1, 0] else x + 1, labels)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, 1.03)
    axes[1, 1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].set_xlim(0, 1.03)
    axes[1, 1].set_ylim(0, 1.03)
    fig.suptitle("Reduced-template utility of PCA/HDBSCAN spectral families")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--build-name", default=DEFAULT_EMPIRICAL_PRIOR_DIR)
    parser.add_argument("--weights-csv", type=Path, default=None)
    parser.add_argument("--cohort-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--configs",
        type=parse_config,
        nargs="+",
        default=[parse_config(f"{count}:300:30:eom") for count in (6, 7, 8, 9)],
        metavar="PC:MCS:MS:METHOD",
    )
    parser.add_argument("--pca-scaling", choices=("standardized", "raw"), default="standardized")
    parser.add_argument("--clr-eps", type=float, default=DEFAULT_CLR_EPS)
    parser.add_argument(
        "--family-weight-coverage", type=float, default=DEFAULT_FAMILY_WEIGHT_COVERAGE
    )
    parser.add_argument("--max-chi2-dof", type=float, default=1.2)
    parser.add_argument("--max-delta-chi2-dof", type=float, default=0.05)
    parser.add_argument("--max-color-rms", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prior_dir = get_prior_build_dir(args.build_name)
    weights_csv = args.weights_csv or prior_dir / "desi_eazy_empirical_weights.csv"
    cohort_root = args.cohort_root or prior_dir / "reduced_template_cohorts"
    output_dir = args.output_dir or cohort_root / "spectral_families/downstream_utility"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table, weights = load_fit_population(Path(weights_csv))
    searches = load_subset_searches(
        Path(cohort_root),
        table["targetid"].to_numpy(np.int64),
        max_chi2_dof=args.max_chi2_dof,
        max_delta_chi2_dof=args.max_delta_chi2_dof,
        max_color_rms=args.max_color_rms,
    )
    summary, families = evaluate_configs(
        weights,
        searches,
        args.configs,
        scaling=args.pca_scaling,
        clr_eps=args.clr_eps,
        family_weight_coverage=args.family_weight_coverage,
    )
    summary.to_csv(output_dir / "configuration_utility.csv", index=False)
    families.to_csv(output_dir / "family_selected_bases.csv", index=False)
    make_figure(summary, families, args.configs, output_dir / "family_basis_utility.png")
    parameters = {
        "weights_csv": str(Path(weights_csv).expanduser().resolve()),
        "cohort_root": str(Path(cohort_root).expanduser().resolve()),
        "output_dir": str(output_dir.expanduser().resolve()),
        "n_spectra": len(table),
        "configs": [config.__dict__ for config in args.configs],
        "pca_scaling": args.pca_scaling,
        "clr_eps": args.clr_eps,
        "family_weight_coverage": args.family_weight_coverage,
        "quality_thresholds": {
            "max_chi2_dof": args.max_chi2_dof,
            "max_delta_chi2_dof": args.max_delta_chi2_dof,
            "max_color_rms": args.max_color_rms,
        },
        "searched_subset_sizes": sorted(searches),
    }
    (output_dir / "utility_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n"
    )
    print("\nConfiguration summary:")
    print(summary.to_string(index=False))
    print(f"Wrote downstream utility outputs to {output_dir}")


if __name__ == "__main__":
    main()
