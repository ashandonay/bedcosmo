#!/usr/bin/env python
"""Scan PCA dimension and primary HDBSCAN spectral-family hyperparameters."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from ..paths import DEFAULT_EMPIRICAL_PRIOR_DIR, get_prior_build_dir  # noqa: E402
from ..simplex import DEFAULT_CLR_EPS, weights_to_ilr  # noqa: E402
from .discover_eazy_spectral_families import load_fit_population  # noqa: E402

GRID = "#D9DDE3"
BASELINE = (8, 300, 30, "eom")


def fit_hdbscan(
    scores: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    selection_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=selection_method,
    )
    labels = clusterer.fit_predict(scores)
    return labels, np.asarray(clusterer.probabilities_, dtype=float)


def summarize_clustering(
    labels: np.ndarray,
    probabilities: np.ndarray,
    baseline_labels: np.ndarray,
) -> dict[str, float | int]:
    clustered = labels >= 0
    sizes = np.bincount(labels[clustered]) if np.any(clustered) else np.array([], dtype=int)
    shared = (baseline_labels >= 0) & clustered
    return {
        "n_families": int(len(sizes)),
        "clustered_count": int(clustered.sum()),
        "clustered_fraction": float(clustered.mean()),
        "min_family_size": int(sizes.min()) if len(sizes) else 0,
        "median_family_size": float(np.median(sizes)) if len(sizes) else 0.0,
        "max_family_size": int(sizes.max()) if len(sizes) else 0,
        "mean_clustered_probability": (
            float(probabilities[clustered].mean()) if np.any(clustered) else np.nan
        ),
        "shared_with_baseline_fraction": float(shared.mean()),
        "ari_including_unclustered": float(adjusted_rand_score(baseline_labels, labels)),
        "ari_on_shared_clustered_core": (
            float(adjusted_rand_score(baseline_labels[shared], labels[shared]))
            if shared.sum() > 1
            else np.nan
        ),
    }


def run_scan(
    scores_by_component: dict[int, np.ndarray],
    explained_variance_ratio: np.ndarray,
    *,
    min_cluster_sizes: list[int],
    min_samples_values: list[int],
    selection_methods: list[str],
    baseline: tuple[int, int, int, str] = BASELINE,
) -> pd.DataFrame:
    baseline_components, baseline_size, baseline_samples, baseline_method = baseline
    if baseline_components not in scores_by_component:
        raise ValueError(f"Baseline requires {baseline_components} retained PCs")
    baseline_labels, _ = fit_hdbscan(
        scores_by_component[baseline_components],
        min_cluster_size=baseline_size,
        min_samples=baseline_samples,
        selection_method=baseline_method,
    )
    rows: list[dict[str, float | int | str]] = []
    total = (
        len(scores_by_component)
        * len(min_cluster_sizes)
        * len(min_samples_values)
        * len(selection_methods)
    )
    completed = 0
    for n_components, scores in sorted(scores_by_component.items()):
        retained_variance = float(explained_variance_ratio[:n_components].sum())
        for method in selection_methods:
            for min_cluster_size in min_cluster_sizes:
                for min_samples in min_samples_values:
                    started = time.perf_counter()
                    labels, probabilities = fit_hdbscan(
                        scores,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        selection_method=method,
                    )
                    row = {
                        "n_components": n_components,
                        "variance_retained": retained_variance,
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        "selection_method": method,
                        "runtime_seconds": time.perf_counter() - started,
                    }
                    row.update(summarize_clustering(labels, probabilities, baseline_labels))
                    rows.append(row)
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(f"Evaluated {completed}/{total} HDBSCAN configurations", flush=True)
    return pd.DataFrame(rows)


def make_component_summary(results: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    panels = (
        ("n_families", "Number of families", None),
        ("clustered_fraction", "Fraction assigned to a family", "percent"),
        ("ari_including_unclustered", "Agreement with 8-PC 300/30 EOM baseline", None),
        ("mean_clustered_probability", "Mean HDBSCAN membership strength", None),
    )
    for ax, (column, title, formatting) in zip(axes.flat, panels):
        for method, color in (("eom", "#3366CC"), ("leaf", "#E07A3F")):
            grouped = results.loc[results["selection_method"] == method].groupby("n_components")[
                column
            ]
            x = np.asarray(sorted(grouped.groups))
            median = grouped.median().reindex(x).to_numpy()
            low = grouped.quantile(0.1).reindex(x).to_numpy()
            high = grouped.quantile(0.9).reindex(x).to_numpy()
            ax.plot(x, median, marker="o", color=color, label=method.upper())
            ax.fill_between(x, low, high, color=color, alpha=0.18)
        ax.set_xlabel("Retained principal components")
        ax.set_title(title, loc="left")
        ax.grid(True, color=GRID, alpha=0.7, linewidth=0.7)
        ax.set_axisbelow(True)
        if formatting == "percent":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("HDBSCAN sensitivity across independently varied density settings")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_heatmap_grid(results: pd.DataFrame, column: str, output: Path) -> None:
    component_counts = sorted(results["n_components"].unique())
    methods = [method for method in ("eom", "leaf") if method in results["selection_method"].unique()]
    sizes = sorted(results["min_cluster_size"].unique())
    samples = sorted(results["min_samples"].unique())
    values = results[column].to_numpy(float)
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    fig, axes = plt.subplots(
        len(component_counts),
        len(methods),
        figsize=(5.8 * len(methods), 3.4 * len(component_counts)),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row_index, n_components in enumerate(component_counts):
        for column_index, method in enumerate(methods):
            ax = axes[row_index, column_index]
            selected = results.loc[
                (results["n_components"] == n_components)
                & (results["selection_method"] == method)
            ]
            matrix = (
                selected.pivot(index="min_samples", columns="min_cluster_size", values=column)
                .reindex(index=samples, columns=sizes)
                .to_numpy(float)
            )
            image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(sizes)), sizes)
            ax.set_yticks(np.arange(len(samples)), samples)
            ax.set_title(f"{n_components} PCs · {method.upper()}", loc="left")
            if row_index == len(component_counts) - 1:
                ax.set_xlabel("Minimum cluster size")
            if column_index == 0:
                ax.set_ylabel("Minimum samples")
    if image is not None:
        fig.colorbar(image, ax=axes, label=column.replace("_", " "))
    fig.suptitle(f"HDBSCAN scan: {column.replace('_', ' ')}", fontsize=15)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--build-name", default=DEFAULT_EMPIRICAL_PRIOR_DIR)
    parser.add_argument("--weights-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-components", type=int, nargs="+", default=(6, 7, 8, 9, 10, 11))
    parser.add_argument(
        "--min-cluster-sizes", type=int, nargs="+", default=(150, 200, 300, 400, 600)
    )
    parser.add_argument(
        "--min-samples-values", type=int, nargs="+", default=(10, 20, 30, 50, 80)
    )
    parser.add_argument(
        "--selection-methods", nargs="+", choices=("eom", "leaf"), default=("eom", "leaf")
    )
    parser.add_argument("--pca-scaling", choices=("standardized", "raw"), default="standardized")
    parser.add_argument("--clr-eps", type=float, default=DEFAULT_CLR_EPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prior_dir = get_prior_build_dir(args.build_name)
    weights_csv = args.weights_csv or prior_dir / "desi_eazy_empirical_weights.csv"
    output_dir = args.output_dir or prior_dir / "reduced_template_cohorts/spectral_families/hdbscan_scan"
    output_dir = Path(output_dir)
    table, weights = load_fit_population(Path(weights_csv))
    ilr = weights_to_ilr(weights, eps=args.clr_eps)
    pca = PCA().fit(ilr)
    raw_scores = pca.transform(ilr)
    component_counts = sorted(set(args.n_components))
    if component_counts[0] < 2 or component_counts[-1] > raw_scores.shape[1]:
        raise ValueError(f"--n-components must lie in [2, {raw_scores.shape[1]}]")
    if args.pca_scaling == "standardized":
        raw_scores = raw_scores / np.sqrt(pca.explained_variance_)[None, :]
    scores_by_component = {count: raw_scores[:, :count] for count in component_counts}
    results = run_scan(
        scores_by_component,
        pca.explained_variance_ratio_,
        min_cluster_sizes=sorted(set(args.min_cluster_sizes)),
        min_samples_values=sorted(set(args.min_samples_values)),
        selection_methods=list(dict.fromkeys(args.selection_methods)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "hdbscan_hyperparameter_scan.csv", index=False)
    make_component_summary(results, output_dir / "hdbscan_component_sensitivity.png")
    for column in ("n_families", "clustered_fraction", "ari_including_unclustered"):
        make_heatmap_grid(results, column, output_dir / f"hdbscan_{column}_heatmaps.png")
    parameters = vars(args).copy()
    parameters.update(
        {
            "weights_csv": str(Path(weights_csv).expanduser().resolve()),
            "output_dir": str(output_dir.expanduser().resolve()),
            "n_spectra": len(table),
            "baseline": {
                "n_components": BASELINE[0],
                "min_cluster_size": BASELINE[1],
                "min_samples": BASELINE[2],
                "selection_method": BASELINE[3],
            },
        }
    )
    for key, value in list(parameters.items()):
        if isinstance(value, Path):
            parameters[key] = str(value)
    (output_dir / "hdbscan_scan_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(results)} configurations to {output_dir}")


if __name__ == "__main__":
    main()
