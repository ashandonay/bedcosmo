#!/usr/bin/env python
"""Summarize how each EAZY template contributes within reduced-fit cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--top-sets", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def summarize(
    memberships: pd.DataFrame,
    subset_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    position_columns = sorted(
        (
            int(name.removeprefix("template_"))
            for name in memberships.columns
            if name.startswith("template_") and name.removeprefix("template_").isdigit()
        )
    )
    if not position_columns:
        raise ValueError("No template_<position> columns found in subset_memberships.csv")
    n_components = len(position_columns)
    contribution_rows: list[dict[str, float | int | str]] = []
    composition_rows: list[dict[str, float | int | str]] = []

    indexed_summary = subset_summary.set_index("templates", drop=False)
    for label, group in memberships.groupby("templates", sort=False):
        template_numbers = np.asarray(
            [int(group[f"template_{position}"].iloc[0]) for position in position_columns]
        )
        weights = group[[f"a_{position}" for position in position_columns]].to_numpy(float)
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"Non-finite normalized weights in {label}")
        weight_sums = weights.sum(axis=1)
        if not np.allclose(weight_sums, 1.0, atol=1e-8):
            raise ValueError(f"Normalized weights do not sum to one in {label}")

        entropy = -(weights * np.log(np.clip(weights, 1e-300, None))).sum(axis=1)
        effective_n = np.exp(entropy)
        dominant_position = np.argmax(weights, axis=1)
        base = indexed_summary.loc[label]
        composition_rows.append(
            {
                "templates": label,
                "coverage_count": int(base["coverage_count"]),
                "exclusive_count": int(base["exclusive_count"]),
                "assigned_count": int(base["assigned_count"]),
                "effective_n_mean": float(effective_n.mean()),
                "effective_n_p10": float(np.percentile(effective_n, 10)),
                "effective_n_median": float(np.median(effective_n)),
                "effective_n_p90": float(np.percentile(effective_n, 90)),
                "nominal_n": n_components,
            }
        )
        for position, template_number in enumerate(template_numbers):
            values = weights[:, position]
            contribution_rows.append(
                {
                    "templates": label,
                    "template": f"T{template_number}",
                    "template_number": int(template_number),
                    "coverage_count": len(group),
                    "mean_weight": float(values.mean()),
                    "weight_p10": float(np.percentile(values, 10)),
                    "weight_median": float(np.median(values)),
                    "weight_p90": float(np.percentile(values, 90)),
                    "dominant_fraction": float(np.mean(dominant_position == position)),
                    "majority_fraction": float(np.mean(values > 0.5)),
                    "substantial_fraction": float(np.mean(values >= 0.1)),
                }
            )

    composition = pd.DataFrame(composition_rows).sort_values(
        ["coverage_count", "assigned_count"], ascending=False
    )
    contributions = pd.DataFrame(contribution_rows)
    return composition, contributions


def plot_summary(
    composition: pd.DataFrame,
    contributions: pd.DataFrame,
    *,
    output: Path,
    top_sets: int,
) -> None:
    shown = composition.head(top_sets).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(shown))
    template_numbers = sorted(contributions["template_number"].unique())
    template_palette = (
        "#1F77B4",  # T1  blue
        "#FF7F0E",  # T2  orange
        "#2CA02C",  # T3  green
        "#D62728",  # T4  red
        "#9467BD",  # T5  purple
        "#8C564B",  # T6  brown
        "#E377C2",  # T7  pink
        "#7F7F7F",  # T8  gray
        "#BCBD22",  # T9  olive
        "#17BECF",  # T10 cyan
        "#003F5C",  # T11 navy
        "#E6AB02",  # T12 gold
    )
    colors = {number: template_palette[number - 1] for number in template_numbers}

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(19, max(7.0, 0.42 * len(shown) + 2.2)),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.45, 1.45, 0.85]},
    )
    ax_coverage, ax_weight, ax_dominant, ax_neff = axes

    ax_coverage.barh(y, shown["coverage_count"], color="#9DB7DF", edgecolor="#3366CC")
    ax_coverage.barh(y, shown["exclusive_count"], color="#3366CC")
    ax_coverage.set_yticks(y, shown["templates"])
    ax_coverage.set_xlabel("DESI spectra")
    ax_coverage.set_title("Coverage: light=total, dark=unique", loc="left")

    for ax, value_column, title in (
        (ax_weight, "mean_weight", "Mean integrated-flux share"),
        (ax_dominant, "dominant_fraction", "Fraction where template dominates"),
    ):
        left = np.zeros(len(shown))
        for template_number in template_numbers:
            values = []
            for label in shown["templates"]:
                row = contributions.loc[
                    (contributions["templates"] == label)
                    & (contributions["template_number"] == template_number),
                    value_column,
                ]
                values.append(float(row.iloc[0]) if len(row) else 0.0)
            values = np.asarray(values)
            ax.barh(
                y,
                values,
                left=left,
                color=colors[template_number],
                edgecolor="white",
                linewidth=0.35,
                label=f"T{template_number}",
            )
            left += values
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction")
        ax.set_title(title, loc="left")

    ax_neff.hlines(
        y,
        shown["effective_n_p10"],
        shown["effective_n_p90"],
        color="#25272B",
        lw=1.2,
    )
    ax_neff.scatter(
        shown["effective_n_median"],
        y,
        color="#3366CC",
        edgecolor="#25272B",
        s=30,
        zorder=3,
    )
    nominal_n = int(shown["nominal_n"].iloc[0])
    ax_neff.set_xlim(1, nominal_n + 0.08)
    ax_neff.set_xlabel(r"$N_{\rm eff}$")
    ax_neff.set_title("Effective components", loc="left")

    handles, labels = ax_weight.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(12, len(labels)), frameon=False)
    for ax in axes:
        ax.grid(True, color="#D9DDE3", alpha=0.65, linewidth=0.7)
        ax.set_axisbelow(True)
        if ax is not ax_coverage:
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle(
        f"EAZY12 N={nominal_n} reduced-cohort template composition",
        fontsize=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    memberships = pd.read_csv(args.cohort_dir / "subset_memberships.csv")
    subset_summary = pd.read_csv(args.cohort_dir / "subset_summary.csv")
    output_dir = args.output_dir or args.cohort_dir / "composition"
    output_dir.mkdir(parents=True, exist_ok=True)

    composition, contributions = summarize(memberships, subset_summary)
    composition.to_csv(output_dir / "subset_composition_summary.csv", index=False)
    contributions.to_csv(output_dir / "template_contributions.csv", index=False)
    figure_path = output_dir / "template_composition.png"
    plot_summary(composition, contributions, output=figure_path, top_sets=args.top_sets)
    print(f"Wrote {output_dir / 'subset_composition_summary.csv'}")
    print(f"Wrote {output_dir / 'template_contributions.csv'}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
