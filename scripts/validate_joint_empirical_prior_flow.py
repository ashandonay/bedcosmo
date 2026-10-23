"""Validate a saved mixed-pattern NF against fresh joint-KDE draws and DESI fits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joint_empirical_prior_flow import JointPriorFlow
from prototype_joint_empirical_mixture import JointPatternMixture
from scipy.stats import kstest, norm
from validate_empirical_coefficient_prior import averaged_metrics, support_metrics


def validate_gaussianization(flow, datasets, output_dir):
    """Check held-out KDE Gaussianity separately from the NF's own base law."""
    report, arrays = {}, {}
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout="constrained")
    for label, rows in datasets.items():
        ids, groups, jac = flow.gaussianize(rows)
        restored = flow.degaussianize(ids, groups)
        if not np.array_equal(restored[:, :-2] == 0, rows[:, :-2] == 0):
            raise ValueError("Gaussian round trip changed an activation mask")
        base_score = np.log(flow.probabilities[ids])
        continuous = ~flow.metadata["atomic"][ids]
        stats = {}
        arrays[f"{label}_pattern_ids"] = ids
        arrays[f"{label}_log_jacobian"] = jac
        for ax, m in zip(axes.flat, sorted(flow.groups)):
            if m not in groups:
                continue
            values = groups[m]
            selected = continuous & (flow.patterns[ids].sum(axis=1) == m)
            base_score[selected] += -0.5 * (values**2 + np.log(2 * np.pi)).sum(axis=1)
            arrays[f"{label}_active_{m}"] = values
            stats[str(m)] = {
                "n": len(values),
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0).tolist(),
                "ks_vs_standard_normal": [
                    float(kstest(column, "norm").statistic) for column in values.T
                ],
            }
            if len(values) > 1:
                stats[str(m)]["max_covariance_error"] = float(
                    np.abs(np.cov(values, rowvar=False) - np.eye(m + 1)).max()
                )
            quantiles = np.linspace(0.01, 0.99, 99)
            color = "#168556" if label == "kde" else "#b14988"
            for j, column in enumerate(values.T):
                ax.plot(
                    norm.ppf(quantiles),
                    np.quantile(column, quantiles),
                    color=color,
                    alpha=0.5,
                    label=label if j == 0 else None,
                )
        report[label] = {
            "max_roundtrip_error_by_feature": np.abs(restored - rows).max(axis=0).tolist(),
            "max_log_density_identity_error": float(
                np.abs(flow.log_prob(rows) - base_score - jac).max()
            ),
            "groups": stats,
        }
    for ax, m in zip(axes.flat, sorted(flow.groups)):
        ax.plot([-3, 3], [-3, 3], "k:", label="Standard Gaussian")
        ax.set(
            title=f"{m} active templates",
            xlabel="Standard Gaussian quantile",
            ylabel="Transformed sample quantile",
        )
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(flow.groups) :]:
        ax.set_visible(False)
    fig.suptitle(
        "NF Gaussianization: one Q–Q curve per continuous coordinate\nFresh joint-KDE target (green); fresh NF draws (pink). Patterns and atoms stay discrete."
    )
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"gaussianization_qq.{ext}", dpi=170)
    plt.close(fig)
    np.savez_compressed(output_dir / "gaussian_coordinates.npz", **arrays)
    (output_dir / "gaussianization.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def validate(flow_path, output_dir, *, n_samples=20000, seed=991):
    flow_path, output_dir = Path(flow_path).resolve(), Path(output_dir).resolve()
    if n_samples < 100:
        raise ValueError("Use at least 100 validation draws")
    output_dir.mkdir(parents=True, exist_ok=False)
    flow = JointPriorFlow.load(flow_path)
    with np.load(flow_path.parent / "joint_kde_fit.npz") as fitted:
        target = JointPatternMixture(fitted["training_rows"], bandwidth=float(fitted["bandwidth"]))
    start = time.perf_counter()
    kde = target.sample(n_samples, seed=seed)
    kde_seconds = time.perf_counter() - start
    start = time.perf_counter()
    nf = flow.sample(n_samples, seed=seed + 1)
    nf_seconds = time.perf_counter() - start
    start = time.perf_counter()
    scores = flow.log_prob(kde)
    score_seconds = time.perf_counter() - start
    if not np.isfinite(scores).all():
        raise ValueError("Trained NF cannot score all fresh target draws")
    if not np.isfinite(nf).all() or not np.allclose(nf[:, :-2].sum(axis=1), 1):
        raise ValueError("Trained NF generated invalid simplex rows")
    if any(tuple(row[:-2] > 0) not in flow.lookup for row in nf):
        raise ValueError("NF invented an activation pattern")
    names = flow.metadata["feature_names"]
    k = len(names) - 2
    metrics = {
        "nf_vs_joint_kde": averaged_metrics(kde, nf, target.rows),
        "joint_kde_vs_fitted_desi": averaged_metrics(target.rows, kde, target.rows),
        "nf_vs_fitted_desi": averaged_metrics(target.rows, nf, target.rows),
    }
    report = {
        "flow_path": str(flow_path),
        "n_samples": n_samples,
        "seed": seed,
        "metrics": metrics,
        "support_nf_vs_kde": support_metrics(kde, nf),
        "seconds": {
            "joint_kde_sampling": kde_seconds,
            "nf_sampling": nf_seconds,
            "nf_scoring_fresh_kde": score_seconds,
        },
        "finite_score_fraction": float(np.isfinite(scores).mean()),
        "atomic_mass": float(flow.probabilities[flow.metadata["atomic"]].sum()),
        "number_of_patterns": len(flow.patterns),
        "number_of_flow_groups": len(flow.groups),
        "gaussianization": validate_gaussianization(flow, {"kde": kde, "nf": nf}, output_dir),
        "notes": [
            "Fresh target and NF samples use different seeds and were not used for NF checkpoint selection.",
            "Scores include pattern probabilities and the stick/aux Jacobian on continuous faces, and singleton probability masses for atomic patterns.",
            "Fitted DESI rows built the joint-KDE target; comparison to that population is descriptive rather than a new galaxy holdout.",
            "No posterior training, end-to-end EIG or production configuration changes.",
        ],
    }
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(
        output_dir / "fresh_samples.npz",
        joint_kde=kde,
        nf=nf,
        source=target.rows,
        feature_names=names,
        log_prob_fresh_kde=scores,
    )
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), layout="constrained")
    for i, ax in enumerate(axes.flat):
        if i >= len(names):
            ax.set_visible(False)
            continue
        bins = (
            np.linspace(0, 1, 51)
            if i < k
            else np.linspace(target.rows[:, i].min(), target.rows[:, i].max(), 51)
        )
        for data, label, color, style in [
            (target.rows, "Fitted DESI coefficients", "black", ":"),
            (kde, "Joint-KDE target", "#168556", "-"),
            (nf, "Trained pattern NF", "#b14988", "-"),
        ]:
            count, _ = np.histogram(data[:, i], bins)
            ax.stairs(count / len(data) * 100, bins, color=color, ls=style, label=label)
        ax.set(xlabel=names[i], ylabel="Population per bin (%)")
        ax.grid(alpha=0.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle(
        "Joint-pattern NF prior: fresh draws versus its KDE target and fitted DESI population\nExact inactive templates remain zero; no CLR/ILR transformation"
    )
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"nf_prior_marginals.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    for ax, (m, group) in zip(axes.flat, sorted(flow.groups.items())):
        values = np.asarray(group["history"])
        ax.plot(values, color="#b14988")
        ax.plot(values.argmin(), values.min(), "o", color="black")
        ax.set(
            title=f"{m} active templates",
            xlabel="Epoch",
            ylabel="Validation NLL (standardized logits)",
        )
        ax.grid(alpha=0.15)
    for ax in axes.flat[len(flow.groups) :]:
        ax.set_visible(False)
    fig.suptitle("NF checkpoint selection on independent joint-KDE validation draws")
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"nf_training.{ext}", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=991)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    validate(**vars(args))
