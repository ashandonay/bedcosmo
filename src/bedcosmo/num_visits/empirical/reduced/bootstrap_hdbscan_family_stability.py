#!/usr/bin/env python
"""Measure PCA/HDBSCAN spectral-family stability under DESI subsampling."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
COLORS = ("#3366CC", "#DC3912", "#109618", "#990099", "#FF9900", "#0099C6")


@dataclass(frozen=True)
class ClusterConfig:
    n_components: int
    min_cluster_size: int = 300
    min_samples: int = 30
    selection_method: str = "eom"

    @property
    def label(self) -> str:
        return (
            f"{self.n_components} PCs\n"
            f"{self.min_cluster_size}/{self.min_samples} {self.selection_method.upper()}"
        )

    @property
    def slug(self) -> str:
        return (
            f"pc{self.n_components}_mcs{self.min_cluster_size}_"
            f"ms{self.min_samples}_{self.selection_method}"
        )

    def scaled_counts(self, fraction: float) -> ClusterConfig:
        """Preserve density thresholds as fractions of the sampled population."""
        return ClusterConfig(
            self.n_components,
            max(2, int(round(self.min_cluster_size * fraction))),
            max(1, int(round(self.min_samples * fraction))),
            self.selection_method,
        )


def parse_config(value: str) -> ClusterConfig:
    """Parse ``PCs:min_cluster_size:min_samples:method``."""
    fields = value.split(":")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError(
            "config must be PCs:min_cluster_size:min_samples:method"
        )
    try:
        n_components, min_cluster_size, min_samples = map(int, fields[:3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("the first three config fields must be integers") from exc
    method = fields[3].lower()
    if method not in {"eom", "leaf"}:
        raise argparse.ArgumentTypeError("selection method must be eom or leaf")
    return ClusterConfig(n_components, min_cluster_size, min_samples, method)


def fit_embedding(ilr: np.ndarray, config: ClusterConfig, scaling: str) -> np.ndarray:
    pca = PCA(n_components=config.n_components).fit(ilr)
    scores = pca.transform(ilr)
    if scaling == "standardized":
        scores = scores / np.sqrt(pca.explained_variance_)[None, :]
    elif scaling != "raw":
        raise ValueError(f"Unknown PCA scaling {scaling!r}")
    return HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_method=config.selection_method,
    ).fit_predict(scores)


def _pair_count(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    return float(np.sum(counts * (counts - 1) // 2))


def pairwise_cluster_scores(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Score same-family pairs, excluding noise from positive pair definitions."""
    reference = np.asarray(reference, dtype=int)
    candidate = np.asarray(candidate, dtype=int)
    reference_clusters = np.unique(reference[reference >= 0])
    candidate_clusters = np.unique(candidate[candidate >= 0])
    reference_pairs = _pair_count(
        np.array([(reference == label).sum() for label in reference_clusters])
    )
    candidate_pairs = _pair_count(
        np.array([(candidate == label).sum() for label in candidate_clusters])
    )
    shared_pairs = 0.0
    for reference_label in reference_clusters:
        selected = reference == reference_label
        shared_pairs += _pair_count(
            np.array([(candidate[selected] == label).sum() for label in candidate_clusters])
        )
    precision = shared_pairs / candidate_pairs if candidate_pairs else np.nan
    recall = shared_pairs / reference_pairs if reference_pairs else np.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else np.nan
    )
    return {
        "pair_precision": float(precision),
        "pair_recall": float(recall),
        "pair_f1": float(f1),
    }


def match_reference_families(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    repetition: int,
    config: ClusterConfig,
) -> list[dict[str, float | int | str]]:
    """Match each reference family to the candidate family with maximum F1 overlap."""
    rows: list[dict[str, float | int | str]] = []
    candidate_clusters = np.unique(candidate[candidate >= 0])
    for reference_label in np.unique(reference[reference >= 0]):
        reference_members = reference == reference_label
        reference_count = int(reference_members.sum())
        assigned_fraction = float(np.mean(candidate[reference_members] >= 0))
        best = (np.nan, np.nan, np.nan, -1, 0)
        for candidate_label in candidate_clusters:
            candidate_members = candidate == candidate_label
            overlap = int(np.sum(reference_members & candidate_members))
            recall = overlap / reference_count
            precision = overlap / int(candidate_members.sum())
            f1 = 2.0 * precision * recall / (precision + recall) if overlap else 0.0
            if np.isnan(best[2]) or f1 > best[2]:
                best = (precision, recall, f1, int(candidate_label), overlap)
        rows.append(
            {
                "config": config.slug,
                "repetition": repetition,
                "reference_family": int(reference_label),
                "reference_sample_count": reference_count,
                "assigned_fraction": assigned_fraction,
                "matched_candidate_family": best[3],
                "overlap_count": best[4],
                "match_precision": best[0],
                "match_recall": best[1],
                "match_f1": best[2],
            }
        )
    return rows


def run_bootstrap_stability(
    ilr: np.ndarray,
    configs: list[ClusterConfig],
    *,
    repetitions: int,
    sample_fraction: float,
    seed: int,
    scaling: str = "standardized",
    scale_density_counts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must lie in (0, 1]")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    sample_size = int(round(sample_fraction * len(ilr)))
    rng = np.random.default_rng(seed)
    subsamples = [
        np.sort(rng.choice(len(ilr), size=sample_size, replace=False))
        for _ in range(repetitions)
    ]
    summary_rows: list[dict[str, float | int | str]] = []
    family_rows: list[dict[str, float | int | str]] = []
    reference_rows: list[dict[str, float | int | str]] = []
    total = len(configs) * repetitions
    completed = 0
    for config in configs:
        reference = fit_embedding(ilr, config, scaling)
        resample_config = config.scaled_counts(sample_fraction) if scale_density_counts else config
        reference_clustered = reference >= 0
        reference_rows.append(
            {
                "config": config.slug,
                "n_components": config.n_components,
                "min_cluster_size": config.min_cluster_size,
                "min_samples": config.min_samples,
                "selection_method": config.selection_method,
                "n_families": int(len(np.unique(reference[reference_clustered]))),
                "clustered_count": int(reference_clustered.sum()),
                "clustered_fraction": float(reference_clustered.mean()),
            }
        )
        for repetition, indices in enumerate(subsamples):
            candidate = fit_embedding(ilr[indices], resample_config, scaling)
            reference_sample = reference[indices]
            reference_assigned = reference_sample >= 0
            candidate_assigned = candidate >= 0
            union = reference_assigned | candidate_assigned
            assignment_jaccard = (
                float(np.sum(reference_assigned & candidate_assigned) / np.sum(union))
                if np.any(union)
                else np.nan
            )
            row = {
                "config": config.slug,
                "repetition": repetition,
                "sample_size": sample_size,
                "effective_min_cluster_size": resample_config.min_cluster_size,
                "effective_min_samples": resample_config.min_samples,
                "n_families": int(len(np.unique(candidate[candidate_assigned]))),
                "clustered_fraction": float(candidate_assigned.mean()),
                "assignment_jaccard": assignment_jaccard,
                "ari_including_unclustered": float(
                    adjusted_rand_score(reference_sample, candidate)
                ),
            }
            row.update(pairwise_cluster_scores(reference_sample, candidate))
            summary_rows.append(row)
            family_rows.extend(
                match_reference_families(
                    reference_sample,
                    candidate,
                    repetition=repetition,
                    config=config,
                )
            )
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"Completed {completed}/{total} stability fits", flush=True)
    return pd.DataFrame(summary_rows), pd.DataFrame(family_rows), pd.DataFrame(reference_rows)


def make_summary_figure(
    results: pd.DataFrame, references: pd.DataFrame, configs: list[ClusterConfig], output: Path
) -> None:
    order = [config.slug for config in configs]
    labels = [config.label for config in configs]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    panels = (
        ("n_families", "Families recovered per resample", None),
        ("clustered_fraction", "DESI fraction assigned per resample", "percent"),
        ("ari_including_unclustered", "Agreement with full-data partition", None),
        ("pair_f1", "Same-family pair stability", None),
    )
    for ax, (column, title, formatting) in zip(axes.flat, panels):
        data = [results.loc[results["config"] == config, column].dropna() for config in order]
        box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        if column in references.columns:
            ax.scatter(
                np.arange(1, len(order) + 1),
                references.set_index("config").loc[order, column],
                marker="D",
                color="#222222",
                s=28,
                zorder=3,
                label="Full-data value",
            )
        ax.set_title(title, loc="left")
        ax.grid(True, axis="y", color=GRID, alpha=0.8, linewidth=0.7)
        ax.set_axisbelow(True)
        if formatting == "percent":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("PCA + HDBSCAN stability under repeated DESI subsampling")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_family_figure(
    family_results: pd.DataFrame, configs: list[ClusterConfig], output: Path
) -> None:
    order = [config.slug for config in configs]
    summaries = family_results.groupby(["config", "reference_family"])[
        ["match_precision", "match_recall", "match_f1"]
    ].median()
    summaries = summaries.reset_index()
    max_families = int(summaries["reference_family"].max()) + 1
    panels = (
        ("match_precision", "Purity of matched cluster\n(low indicates merging)"),
        ("match_recall", "Recovery of reference family\n(low indicates splitting or loss)"),
        ("match_f1", "Balanced family match F1"),
    )
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.1 * len(panels), max(5.5, 0.38 * max_families)),
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for ax, (metric, title) in zip(axes, panels):
        matrix = np.full((max_families, len(order)), np.nan)
        for column, config in enumerate(order):
            selected = summaries.loc[summaries["config"] == config]
            matrix[selected["reference_family"].astype(int), column] = selected[metric]
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(
            np.arange(len(order)), [f"{config.n_components} PCs" for config in configs]
        )
        ax.set_yticks(
            np.arange(max_families), [f"F{index + 1:02d}" for index in range(max_families)]
        )
        ax.set_xlabel("Full-data solution")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Reference family")
    if image is not None:
        fig.colorbar(image, ax=axes, label="Median best-match score")
    fig.suptitle("Family-level stability across DESI subsamples (300/30 EOM solutions)")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--build-name", default=DEFAULT_EMPIRICAL_PRIOR_DIR)
    parser.add_argument("--weights-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--configs",
        type=parse_config,
        nargs="+",
        default=[parse_config(f"{count}:300:30:eom") for count in (6, 7, 8, 9)],
        metavar="PC:MCS:MS:METHOD",
    )
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--pca-scaling", choices=("standardized", "raw"), default="standardized")
    parser.add_argument(
        "--scale-density-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale min_cluster_size and min_samples by the sampled population fraction",
    )
    parser.add_argument("--clr-eps", type=float, default=DEFAULT_CLR_EPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prior_dir = get_prior_build_dir(args.build_name)
    weights_csv = args.weights_csv or prior_dir / "desi_eazy_empirical_weights.csv"
    output_dir = args.output_dir or (
        prior_dir / "reduced_template_cohorts/spectral_families/hdbscan_bootstrap"
    )
    output_dir = Path(output_dir)
    # Fail before the expensive resampling loop if the destination is invalid.
    output_dir.mkdir(parents=True, exist_ok=True)
    table, weights = load_fit_population(Path(weights_csv))
    ilr = weights_to_ilr(weights, eps=args.clr_eps)
    results, family_results, references = run_bootstrap_stability(
        ilr,
        args.configs,
        repetitions=args.repetitions,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        scaling=args.pca_scaling,
        scale_density_counts=args.scale_density_counts,
    )
    results.to_csv(output_dir / "bootstrap_stability.csv", index=False)
    family_results.to_csv(output_dir / "bootstrap_family_matches.csv", index=False)
    references.to_csv(output_dir / "full_data_solutions.csv", index=False)
    make_summary_figure(results, references, args.configs, output_dir / "bootstrap_stability.png")
    make_family_figure(
        family_results, args.configs, output_dir / "bootstrap_family_stability.png"
    )
    parameters = {
        "weights_csv": str(Path(weights_csv).expanduser().resolve()),
        "output_dir": str(output_dir.expanduser().resolve()),
        "n_spectra": len(table),
        "configs": [config.__dict__ for config in args.configs],
        "repetitions": args.repetitions,
        "sample_fraction": args.sample_fraction,
        "seed": args.seed,
        "pca_scaling": args.pca_scaling,
        "scale_density_counts": args.scale_density_counts,
        "clr_eps": args.clr_eps,
        "resampling": "without-replacement random subsampling",
    }
    (output_dir / "bootstrap_parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n"
    )
    print(references.to_string(index=False))
    print("\nMedian resample metrics:")
    print(
        results.groupby("config")[
            [
                "n_families",
                "clustered_fraction",
                "assignment_jaccard",
                "ari_including_unclustered",
                "pair_precision",
                "pair_recall",
                "pair_f1",
            ]
        ]
        .median()
        .to_string()
    )
    print(f"Wrote bootstrap stability outputs to {output_dir}")


if __name__ == "__main__":
    main()
