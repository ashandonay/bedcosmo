"""Repeated held-out tests of joint LSST colors for empirical mixture samplers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from prototype_empirical_mixture import diagnostics
from prototype_joint_empirical_mixture import JointPatternMixture, lsst_magnitudes
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance


def joint_color_metrics(reference, samples, training, *, seed=19, max_rows=500):
    """Training-scaled, equal-size RBF MMD and sliced Wasserstein distances.

    Kernel length and scales use only training data. Biased MMD squared has a
    finite-sample floor, assessed with the matched empirical-bootstrap control.
    """
    rng = np.random.default_rng(seed)
    scale = training.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("Training color coordinates must have nonzero variance")
    center = training.mean(axis=0)
    train = (training - center) / scale
    count = min(max_rows, len(reference), len(samples))
    if count < 2:
        raise ValueError("Joint color diagnostics require at least two samples")
    x = (reference[rng.choice(len(reference), count, replace=False)] - center) / scale
    y = (samples[rng.choice(len(samples), count, replace=False)] - center) / scale
    pairs = rng.integers(len(train), size=(2000, 2))
    distances = np.sum((train[pairs[:, 0]] - train[pairs[:, 1]]) ** 2, axis=1)
    length2 = np.median(distances[distances > 0])
    xx = np.exp(-cdist(x, x, "sqeuclidean") / (2 * length2))
    yy = np.exp(-cdist(y, y, "sqeuclidean") / (2 * length2))
    xy = np.exp(-cdist(x, y, "sqeuclidean") / (2 * length2))
    directions = rng.normal(size=(64, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projected_x = x @ directions.T
    projected_y = y @ directions.T
    return {
        "rbf_mmd2_biased": float(xx.mean() + yy.mean() - 2 * xy.mean()),
        "sliced_wasserstein": float(
            np.mean([wasserstein_distance(a, b) for a, b in zip(projected_x.T, projected_y.T)])
        ),
        "color_correlation_error": float(
            np.linalg.norm(np.corrcoef(reference.T) - np.corrcoef(samples.T))
        ),
        "max_marginal_color_ks": float(
            max(ks_2samp(a, b).statistic for a, b in zip(reference.T, samples.T))
        ),
        "metric_rows_per_population": count,
        "kernel_length_squared": float(length2),
    }


def fit_ilr_draws(train, names, bandwidth, n_samples, seed):
    import pandas as pd

    from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
        build_feature_matrix,
        fit_sed_prior_kde,
        pack_kde_artifact,
        sample_sed_prior,
        samples_to_coeffs,
    )

    k = train.shape[1] - 2
    table = pd.DataFrame(train, columns=names)
    for i in range(k):
        table[f"c{i+1}"] = train[:, i] * np.exp(train[:, -2])
    features, feature_names, _ = build_feature_matrix(table, simplex_smoothing_eps=1e-5)
    kde, scaler = fit_sed_prior_kde(features, bandwidth=bandwidth)
    artifact = pack_kde_artifact(kde, scaler, feature_names, features, metadata={"n_templates": k})
    a, log_s, z = samples_to_coeffs(sample_sed_prior(artifact, n_samples, seed=seed), k)
    return np.column_stack([a, log_s, z])


def run(samples_path, prior_dir, output_dir, *, n_samples=1000):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
        load_prior_training_table,
        load_sed_prior_kde,
    )
    from bedcosmo.num_visits.empirical.templates import (
        load_two_column_template,
        read_template_param,
    )

    samples_path, prior_dir, output_dir = [
        Path(p).resolve() for p in [samples_path, prior_dir, output_dir]
    ]
    if output_dir == prior_dir or prior_dir in output_dir.parents:
        raise ValueError("Output must be outside the source prior build")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(samples_path) as saved:
        rows, names = saved["training"], saved["feature_names"].tolist()
    metadata = load_sed_prior_kde(prior_dir / "sed_prior_kde_native.joblib")["metadata"]
    csv_path = prior_dir / "desi_eazy_empirical_weights.csv"
    table = load_prior_training_table(
        csv_path, z_min=metadata["z_min"], max_chi2_dof=metadata["max_chi2_dof"]
    )
    if rows.shape != table[names].shape or not np.allclose(rows, table[names].to_numpy()):
        raise ValueError("Saved fitted rows do not match the source build CSV selection")
    config = yaml.safe_load((prior_dir / "prior_args.yaml").read_text())
    if config["flux_unit_scale"] != 1e-17:
        raise ValueError("DESI photometry requires flux_unit_scale=1e-17")
    param = prior_dir / "templates" / config["template_param"]
    paths = [param.parent / p for p in read_template_param(param)]
    bank = [load_two_column_template(p) for p in paths]
    wave = bank[0][0]
    if len(bank) != rows.shape[1] - 2 or any(not np.array_equal(wave, w) for w, _ in bank):
        raise ValueError("Template grids or component count do not match")
    basis = np.stack([b for _, b in bank])
    print("Synthesizing reference photometry", flush=True)
    reference_mags, bands = lsst_magnitudes(rows, wave, basis)
    reference_colors = -np.diff(reference_mags, axis=1)
    np.savez_compressed(
        output_dir / "reference.npz", rows=rows, magnitudes=reference_mags, bands=bands
    )
    splits = []
    for seed in [37, 71, 109]:
        indices = np.random.default_rng(seed).permutation(len(rows))
        count = int(len(rows) * 0.2)
        splits.append((f"random_{seed}", indices[count:], indices[:count], seed))
    regions = table["healpix"].to_numpy()
    region_ids, counts = np.unique(regions, return_counts=True)
    if len(region_ids) < 2:
        raise ValueError("Region validation needs at least two HEALPix regions")
    for region in region_ids[np.argsort(counts)[-3:]]:
        splits.append(
            (
                f"healpix_{region}",
                np.flatnonzero(regions != region),
                np.flatnonzero(regions == region),
                211,
            )
        )
    bandwidths = [0.05, 0.1, 0.3, 0.6]
    records = []
    for split_name, train_idx, test_idx, seed in splits:
        print(f"Evaluating {split_name}: {len(train_idx)} train, {len(test_idx)} test", flush=True)
        train, test = rows[train_idx], rows[test_idx]
        rng = np.random.default_rng(seed)
        bootstrap_idx = train_idx[rng.integers(len(train), size=n_samples)]
        candidates = {"bootstrap": rows[bootstrap_idx]}
        candidate_colors = {"bootstrap": reference_colors[bootstrap_idx]}
        for bandwidth in bandwidths:
            candidates[f"joint_{bandwidth}"] = JointPatternMixture(
                train, bandwidth=bandwidth
            ).sample(n_samples, seed=seed)
            candidates[f"ilr_{bandwidth}"] = fit_ilr_draws(train, names, bandwidth, n_samples, seed)
        split_records = []
        pattern_summary = JointPatternMixture(train).pattern_summary(test)
        for model_name, generated in candidates.items():
            if model_name not in candidate_colors:
                mags, _ = lsst_magnitudes(generated, wave, basis)
                candidate_colors[model_name] = -np.diff(mags, axis=1)
            result = {
                "split": split_name,
                "model": model_name,
                "n_train": len(train),
                "n_test": len(test),
                "unseen_test_pattern_mass": pattern_summary["heldout_mass_in_unseen_patterns"],
                **joint_color_metrics(
                    reference_colors[test_idx],
                    candidate_colors[model_name],
                    reference_colors[train_idx],
                    seed=seed,
                ),
                "coefficient_correlation_error": diagnostics(test, generated)[
                    "correlation_frobenius"
                ],
            }
            records.append(result)
            split_records.append(result)
            print(
                f"  {model_name}: MMD²={result['rbf_mmd2_biased']:.5g}, SW={result['sliced_wasserstein']:.4g}",
                flush=True,
            )
        (output_dir / f"{split_name}.json").write_text(
            json.dumps(split_records, indent=2, allow_nan=False)
        )
        np.savez_compressed(
            output_dir / f"{split_name}.npz",
            train_indices=train_idx,
            test_indices=test_idx,
            **candidate_colors,
        )
    report = {
        "records": records,
        "bandwidths": bandwidths,
        "n_draws": n_samples,
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [samples_path, csv_path, param, *paths]
        },
        "protocol": [
            "Three new random 80/20 splits plus leave-one-region-out for the three most populated HEALPix regions.",
            "Every model is refitted using only its split's training galaxies; fixed bandwidth sweep is exploratory, with no winner selected before testing.",
            "RBF MMD and sliced Wasserstein operate jointly on all five colors, standardized using training color moments only.",
            "Matched sample counts, same reference subsample, training-only kernel scale and fixed random projection directions within each split.",
            "MMD squared uses the biased nonnegative estimator; compare against bootstrap sampling floor rather than zero.",
            "Regions share a previously learned template bank. Galaxy holdout tests the coefficient prior, not an independent end-to-end NMF training pipeline.",
            "Bootstrap controls sampling fluctuations but is not a smooth density; tests do not establish entropy or EIG correctness.",
        ],
    }
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for col, split_prefix in enumerate(["random", "healpix"]):
        for row, metric in enumerate(["rbf_mmd2_biased", "sliced_wasserstein"]):
            ax = axes[row, col]
            for family, color in [("joint", "#249866"), ("ilr", "#246bb2")]:
                for split_name, *_ in splits:
                    if not split_name.startswith(split_prefix):
                        continue
                    values = [
                        next(
                            r[metric]
                            for r in records
                            if r["split"] == split_name and r["model"] == f"{family}_{b}"
                        )
                        for b in bandwidths
                    ]
                    ax.plot(bandwidths, values, color=color, marker="o", alpha=0.6)
                ax.plot([], [], color=color, label=family)
            controls = [
                r[metric]
                for r in records
                if r["split"].startswith(split_prefix) and r["model"] == "bootstrap"
            ]
            ax.axhspan(
                min(controls),
                max(controls),
                color="gray",
                alpha=0.2,
                label="Bootstrap control range",
            )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Bandwidth (within each model's own coordinates)",
                ylabel=metric,
                title=f"{split_prefix}: separate line per split",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    fig.suptitle(
        "Joint five-color agreement on fresh random and region holdouts\nLower is better; bandwidth units differ between model families"
    )
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"robustness.{ext}", dpi=170)
    plt.close(fig)
    # A pairwise view complements, rather than replaces, the five-dimensional metrics.
    split_name, _, test_idx, _ = splits[0]
    with np.load(output_dir / f"{split_name}.npz") as saved:
        panels = [reference_colors[test_idx], saved["joint_0.1"], saved["ilr_0.1"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    ranges = [np.quantile(reference_colors[:, i], [0.005, 0.995]) for i in [0, 1]]
    for ax, colors, label in zip(
        axes, panels, ["Held-out DESI fits", "Joint mixture, h=0.1", "ILR KDE, h=0.1"]
    ):
        ax.scatter(colors[:, 0], colors[:, 1], s=3, alpha=0.15)
        ax.set(
            xlabel="u−g (AB mag)",
            ylabel="g−r (AB mag)",
            xlim=ranges[0],
            ylim=ranges[1],
            title=label,
        )
    fig.suptitle(
        "Example color–color structure, fresh split 37\nDisplay clips to population 0.5–99.5 percentiles; metrics retain all finite samples"
    )
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"color_color.{ext}", dpi=170)
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-path", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    run(**vars(parser.parse_args()))
