"""Nested bandwidth selection against held-out DESI coefficient distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from prototype_joint_empirical_mixture import JointPatternMixture
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance
from validate_joint_empirical_mixture import fit_ilr_draws

BANDWIDTHS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.6]
METRIC_SEEDS = [304, 401, 509]


def distribution_metrics(reference, samples, training, *, seed, max_rows=700):
    """Compare physical coordinates jointly using training-only metric settings."""
    rng = np.random.default_rng(seed)
    std = training.std(axis=0)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("Training features must have finite nonzero spread")
    center = training.mean(axis=0)
    standardized_train = (training - center) / std
    count = min(len(reference), len(samples), max_rows)
    if count < 2:
        raise ValueError("At least two rows are required")
    x = (reference[rng.choice(len(reference), count, replace=False)] - center) / std
    y = (samples[rng.choice(len(samples), count, replace=False)] - center) / std
    pairs = rng.integers(len(training), size=(3000, 2))
    distances = np.sum(
        (standardized_train[pairs[:, 0]] - standardized_train[pairs[:, 1]]) ** 2, axis=1
    )
    length2 = np.median(distances[distances > 0])
    mmd2 = np.exp(-cdist(x, x, "sqeuclidean") / (2 * length2)).mean()
    mmd2 += np.exp(-cdist(y, y, "sqeuclidean") / (2 * length2)).mean()
    mmd2 -= 2 * np.exp(-cdist(x, y, "sqeuclidean") / (2 * length2)).mean()
    directions = rng.normal(size=(128, training.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projected_x, projected_y = x @ directions.T, y @ directions.T
    return {
        "mmd2": float(mmd2),
        "sliced_wasserstein": float(
            np.mean([wasserstein_distance(a, b) for a, b in zip(projected_x.T, projected_y.T)])
        ),
        "correlation_error": float(
            np.linalg.norm(np.corrcoef(reference.T) - np.corrcoef(samples.T))
        ),
        "max_marginal_ks": float(
            max(ks_2samp(a, b).statistic for a, b in zip(reference.T, samples.T))
        ),
        "rows_per_population": count,
    }


def averaged_metrics(reference, samples, training):
    values = [
        distribution_metrics(reference, samples, training, seed=seed) for seed in METRIC_SEEDS
    ]
    return {key: float(np.mean([value[key] for value in values])) for key in values[0]}


def raw_coefficients(rows):
    return np.column_stack([rows[:, :-2] * np.exp(rows[:, -2, None]), rows[:, -1]])


def support_metrics(reference, samples):
    k = reference.shape[1] - 2
    patterns_ref, counts_ref = np.unique(reference[:, :k] > 0, axis=0, return_counts=True)
    patterns_model, counts_model = np.unique(samples[:, :k] > 0, axis=0, return_counts=True)
    p = {tuple(a): n / len(reference) for a, n in zip(patterns_ref, counts_ref)}
    q = {tuple(a): n / len(samples) for a, n in zip(patterns_model, counts_model)}
    return {
        "pattern_total_variation": float(
            0.5 * sum(abs(p.get(key, 0) - q.get(key, 0)) for key in p.keys() | q.keys())
        ),
        "max_absolute_zero_probability_error": float(
            np.max(abs((reference[:, :k] == 0).mean(axis=0) - (samples[:, :k] == 0).mean(axis=0)))
        ),
    }


def nested_indices(n_rows, seed, *, heldout=None):
    """Return fit/validation/test indices without using test rows in selection."""
    rng = np.random.default_rng(seed)
    if heldout is None:
        order = rng.permutation(n_rows)
        test = order[: int(n_rows * 0.2)]
        development = order[int(n_rows * 0.2) :]
    else:
        test = np.asarray(heldout, dtype=int)
        development = rng.permutation(np.setdiff1d(np.arange(n_rows), test))
    count = max(1, int(len(development) * 0.2))
    validation, fit = development[:count], development[count:]
    if min(len(test), len(validation), len(fit)) < 2:
        raise ValueError("Each nested split must contain at least two rows")
    return fit, validation, test


def audit_run_snapshots(root):
    from datetime import datetime

    import joblib
    import yaml
    from zoneinfo import ZoneInfo

    evidence = []
    for path in Path(root).glob("*/*/artifacts/empirical/sed_prior_kde_native.joblib"):
        run = path.parents[2]
        prior_path = run / "artifacts/prior_args.yaml"
        if not prior_path.exists():
            continue
        prior = yaml.safe_load(prior_path.read_text())
        metadata = yaml.safe_load((run / "meta.yaml").read_text())
        artifact = joblib.load(path)
        lines = []
        for log in (run / "artifacts/logs").glob("*.log"):
            lines.extend(
                line.strip()
                for line in log.read_text(errors="replace").splitlines()
                if "density_type=" in line or "rebuilt pool" in line
            )
        evidence.append(
            {
                "run_id": run.name,
                "start_time_ms": metadata["start_time"],
                "start_time_pacific": datetime.fromtimestamp(
                    metadata["start_time"] / 1000, ZoneInfo("America/Los_Angeles")
                ).isoformat(),
                "template_source": prior.get("template_source"),
                "density_type": prior.get("density_type"),
                "parameterization": artifact["parameterization"],
                "bandwidth": artifact["kde"].bandwidth,
                "kernel": artifact["kde"].kernel,
                "n_training": len(artifact["training_x"]),
                "prior_snapshot": str(prior_path),
                "kde_snapshot": str(path),
                "kde_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "runtime_evidence": lines[:8],
                "frozen_flow_paths": [str(p) for p in path.parent.glob("sed_prior_flow_*.pt")],
            }
        )
    return sorted(evidence, key=lambda x: x["start_time_ms"], reverse=True)


def run(prior_dir, output_dir, mlruns_dir=None, *, n_samples=4000):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
        load_prior_training_table,
        load_sed_prior_kde,
    )

    prior_dir, output_dir = Path(prior_dir).resolve(), Path(output_dir).resolve()
    if output_dir == prior_dir or prior_dir in output_dir.parents:
        raise ValueError("Output must be outside the source prior build")
    if n_samples < 700:
        raise ValueError("Use at least 700 generated rows for the fixed metric protocol")
    output_dir.mkdir(parents=True, exist_ok=False)
    if mlruns_dir is not None:
        evidence = audit_run_snapshots(mlruns_dir)
        (output_dir / "production_audit.json").write_text(json.dumps(evidence, indent=2))
    artifact_path = prior_dir / "sed_prior_kde_native.joblib"
    artifact = load_sed_prior_kde(artifact_path)
    metadata = artifact["metadata"]
    csv_path = prior_dir / "desi_eazy_empirical_weights.csv"
    table = load_prior_training_table(
        csv_path, z_min=metadata["z_min"], max_chi2_dof=metadata["max_chi2_dof"]
    )
    k = artifact["n_templates"]
    names = [f"a{i+1}" for i in range(k)] + ["log_c_scale", "z"]
    rows = table[names].to_numpy()
    splits = []
    for seed in [503, 811, 1297]:
        splits.append((f"random_{seed}", *nested_indices(len(rows), seed), seed))
    regions, counts = np.unique(table["healpix"], return_counts=True)
    for region in regions[np.argsort(counts)[-3:]]:
        test = np.flatnonzero(table["healpix"].to_numpy() == region)
        splits.append((f"region_{region}", *nested_indices(len(rows), 1709, heldout=test), 1709))
    records, selections = [], []
    for split, fit_idx, val_idx, test_idx, seed in splits:
        print(
            f"{split}: {len(fit_idx)} fit / {len(val_idx)} validation / {len(test_idx)} test",
            flush=True,
        )
        fit, validation = rows[fit_idx], rows[val_idx]
        validation_results = []
        for family in ["joint", "ilr"]:
            for bandwidth in BANDWIDTHS:
                started = time.perf_counter()
                generated = (
                    JointPatternMixture(fit, bandwidth=bandwidth).sample(n_samples, seed=seed)
                    if family == "joint"
                    else fit_ilr_draws(fit, names, bandwidth, n_samples, seed)
                )
                fit_and_sample_seconds = time.perf_counter() - started
                metrics = averaged_metrics(validation, generated, fit)
                validation_results.append(
                    {
                        "family": family,
                        "bandwidth": bandwidth,
                        "fit_and_sample_seconds": fit_and_sample_seconds,
                        **metrics,
                    }
                )
                print(
                    f"  validation {family} h={bandwidth}: MMD²={metrics['mmd2']:.6g}", flush=True
                )
        selected = {
            family: min(
                (r for r in validation_results if r["family"] == family), key=lambda r: r["mmd2"]
            )["bandwidth"]
            for family in ["joint", "ilr"]
        }
        selections.append(
            {"split": split, "selected": selected, "validation_sweep": validation_results}
        )
        development_idx = np.r_[fit_idx, val_idx]
        development, test = rows[development_idx], rows[test_idx]
        rng = np.random.default_rng(seed + 1)
        candidates, timings = {}, {}
        started = time.perf_counter()
        candidates["bootstrap"] = development[rng.integers(len(development), size=n_samples)]
        timings["bootstrap"] = time.perf_counter() - started
        started = time.perf_counter()
        candidates["joint_selected"] = JointPatternMixture(
            development, bandwidth=selected["joint"]
        ).sample(n_samples, seed=seed + 1)
        timings["joint_selected"] = time.perf_counter() - started
        for name, bandwidth in [("ilr_selected", selected["ilr"]), ("ilr_default_0.3", 0.3)]:
            started = time.perf_counter()
            candidates[name] = fit_ilr_draws(development, names, bandwidth, n_samples, seed + 1)
            timings[name] = time.perf_counter() - started
        for name, generated in candidates.items():
            record = {
                "split": split,
                "model": name,
                "selected_bandwidth": (
                    selected.get(name.split("_")[0])
                    if name.endswith("_selected")
                    else (0.3 if name == "ilr_default_0.3" else None)
                ),
                "fit_and_sample_seconds": timings[name],
                "physical_joint": averaged_metrics(test, generated, development),
                "raw_coefficient_joint": averaged_metrics(
                    raw_coefficients(test),
                    raw_coefficients(generated),
                    raw_coefficients(development),
                ),
                "support": support_metrics(test, generated),
            }
            records.append(record)
            print(f"  TEST {name}: MMD²={record['physical_joint']['mmd2']:.6g}", flush=True)
        np.savez_compressed(
            output_dir / f"{split}.npz",
            fit_indices=fit_idx,
            validation_indices=val_idx,
            test_indices=test_idx,
            test=test,
            **candidates,
            feature_names=names,
        )
        (output_dir / f"{split}.json").write_text(
            json.dumps(
                {"selection": selections[-1], "test_records": records[-4:]},
                indent=2,
                allow_nan=False,
            )
        )
    report = {
        "source_prior": str(prior_dir),
        "n_fitted": len(rows),
        "feature_names": names,
        "bandwidth_grid": BANDWIDTHS,
        "n_generated": n_samples,
        "benchmark_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "timing_scope": "Elapsed CPU wall time for fitting and drawing n_generated rows, including coordinate preprocessing and bound handling; excludes input reads, diagnostics, plotting, prior-flow training and density scoring. Outer timings are measured after validation has warmed imports.",
        "selection_objective": "Mean validation RBF MMD² in standardized physical (a1..aK, log_c_scale, z) coordinates across three fixed metric subsamples.",
        "selections": selections,
        "test_results": records,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [csv_path, artifact_path, prior_dir / "build_provenance.json"]
        },
        "limitations": [
            "Fixed template bank learned before this test; held-out spectra can have contributed to NMF basis learning. This tests the prior density fit conditional on that bank.",
            "Bandwidth selection uses validation galaxies only. Outer test coordinates are not used for scaling, kernel length, or bandwidth choice.",
            "Primary metric uses normalized template fractions, natural-log scale and redshift with equal per-coordinate training standardization. Secondary metrics use raw coefficients c=exp(log_scale)*a and z.",
            "RBF MMD² is a biased nonnegative sample discrepancy with a finite-sample floor; bootstrap is a matched control, not a density model. Differences are descriptive, not significance claims.",
            "Exact activation-pattern and zero-probability discrepancies are reported separately; ILR has no exact zeros by construction.",
            "Three subsamples share generated populations and overlapping outer splits; they are variance checks, not independent experimental replicates.",
            "Current production flow approximation is verified from saved snapshots/logs but is not evaluated as an additional sampler here.",
        ],
    }
    (output_dir / "coefficient_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    summary = [
        "# Joint-pattern versus ILR prior comparison",
        "",
        "Bandwidths are selected on validation galaxies; these are outer held-out averages. Lower discrepancy is better. Small differences near the bootstrap control are not significance claims.",
        "",
        "| Method | Joint MMD² | Raw-coefficient MMD² | Pattern TV | Fit + draw seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ["bootstrap", "joint_selected", "ilr_selected", "ilr_default_0.3"]:
        selected_records = [r for r in records if r["model"] == model]
        values = [
            np.mean([r["physical_joint"]["mmd2"] for r in selected_records]),
            np.mean([r["raw_coefficient_joint"]["mmd2"] for r in selected_records]),
            np.mean([r["support"]["pattern_total_variation"] for r in selected_records]),
            np.mean([r["fit_and_sample_seconds"] for r in selected_records]),
        ]
        summary.append(
            f"| {model} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.4f} | {values[3]:.4f} |"
        )
    summary += [
        "",
        f"Timing covers fitting plus {n_samples} generated rows in this environment, excluding diagnostics, file reads, density evaluation and any NF training. It is not a production BED throughput benchmark.",
        "",
        "## Selected bandwidths",
        "",
    ]
    summary += [
        f"- {s['split']}: joint={s['selected']['joint']}, ILR={s['selected']['ilr']}"
        for s in selections
    ]
    summary += [
        "",
        "Joint patterns preserve exact zeros. ILR has no exact-zero atoms, so its pattern-TV score must be interpreted separately from agreement of small coefficient values.",
        "",
        "The fixed NMF bank predates these splits. This evaluates the prior conditional on that bank; A separate experimental pattern NF is available; BED posterior/EIG integration remains unfinished.",
        "",
        "[Full metrics and protocol](coefficient_validation.json) · [Coefficient comparison figure](coefficient_quality.png)",
    ]
    (output_dir / "comparison.md").write_text("\n".join(summary) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    models = ["bootstrap", "joint_selected", "ilr_selected", "ilr_default_0.3"]
    colors = ["0.5", "#168556", "#a35db4", "#246bb2"]
    for ax, (space, metric, label) in zip(
        axes,
        [
            ("physical_joint", "mmd2", "Joint fractions + log scale + z: MMD²"),
            ("physical_joint", "sliced_wasserstein", "Joint fractions + log scale + z: distance"),
            ("raw_coefficient_joint", "mmd2", "Joint raw coefficients + z: MMD²"),
        ],
    ):
        for i, (model, color) in enumerate(zip(models, colors)):
            values = [r[space][metric] for r in records if r["model"] == model]
            ax.scatter(
                np.full(len(values), i) + np.linspace(-0.12, 0.12, len(values)),
                values,
                color=color,
                s=35,
            )
            ax.plot([i - 0.2, i + 0.2], [np.mean(values)] * 2, color=color, lw=3)
        ax.set(
            xticks=range(4),
            xticklabels=["Bootstrap", "Joint\nselected h", "ILR\nselected h", "ILR\nh=0.3"],
            ylabel=label,
            yscale="log",
        )
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Recovering the DESI coefficient population on untouched outer test galaxies\nBandwidth selected using validation coefficients only; dots = six outer splits, bars = means; lower is better"
    )
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"coefficient_quality.{ext}", dpi=170)
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mlruns-dir", type=Path, help="Optional read-only audit of saved production runs"
    )
    parser.add_argument("--n-samples", type=int, default=4000)
    run(**vars(parser.parse_args()))
