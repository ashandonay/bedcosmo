"""Offline zero-mass + reflected-KDE coefficient experiment; see empirical README."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp


class CoefficientMixture:
    """Independent hurdle marginals on fractions, followed by simplex closure.

    Positive draws use a Gaussian KDE reflected repeatedly into [threshold, 1].
    Independent Bernoulli masks are conditioned on at least one active component.
    Closure preserves exact zeros but changes the positive marginal densities.
    """

    def __init__(self, weights, *, bandwidth=0.03, zero_threshold=0.0):
        weights = np.asarray(weights, dtype=float)
        if (
            weights.ndim != 2
            or min(weights.shape) < 1
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=1), 1)
        ):
            raise ValueError("weights must be finite nonnegative simplex rows")
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be positive and finite")
        if not np.isfinite(zero_threshold) or not 0 <= zero_threshold < 1:
            raise ValueError("zero_threshold must be in [0, 1)")
        active = weights > zero_threshold
        if not active.any(axis=1).all():
            raise ValueError("zero_threshold removes every component of a training row")
        self.p0 = (~active).mean(axis=0)
        self.positive = [weights[active[:, k], k].copy() for k in range(weights.shape[1])]
        self.bandwidth = bandwidth
        self.zero_threshold = zero_threshold

    def sample(self, n, *, seed=7):
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        rng = np.random.default_rng(seed)
        # Sampling one guaranteed-active index from the conditional distribution
        # avoids potentially unbounded rejection when almost all marginals are zero.
        p_active = 1 - self.p0
        first_prob = p_active * np.cumprod(np.r_[1.0, self.p0[:-1]])
        first = rng.choice(len(self.p0), n, p=first_prob / first_prob.sum())
        indices = np.arange(len(self.p0))
        active = (indices[None, :] > first[:, None]) & (rng.random((n, len(self.p0))) < p_active)
        active[np.arange(n), first] = True
        draws = np.zeros(active.shape)
        lo = self.zero_threshold
        width = 1 - lo
        for k, centers in enumerate(self.positive):
            count = int(active[:, k].sum())
            if not count:
                continue
            values = rng.choice(centers, count) + rng.normal(0, self.bandwidth, count)
            folded = np.remainder(values - lo, 2 * width)
            draws[active[:, k], k] = lo + width - np.abs(folded - width)
        return draws / draws.sum(axis=1, keepdims=True)


def diagnostics(reference, samples):
    """Descriptive checks on fitted fractions, scale and redshift (no holdout)."""
    k = reference.shape[1] - 2
    ref_support = {tuple(row) for row in reference[:, :k] > 0}

    # Constant components have zero standardized residuals, hence zero correlations.
    def correlation(x):
        std = x.std(axis=0)
        centered = (x - x.mean(axis=0)) / np.where(std > 0, std, 1)
        return centered.T @ centered / len(x)

    return {
        "p0": (samples[:, :k] == 0).mean(axis=0).tolist(),
        "ks_per_feature": [float(ks_2samp(a, b).statistic) for a, b in zip(reference.T, samples.T)],
        "correlation_frobenius": float(
            np.linalg.norm(correlation(reference) - correlation(samples))
        ),
        "mean_active_components": float((samples[:, :k] > 0).sum(axis=1).mean()),
        "unseen_support_fraction": float(
            np.mean([tuple(row) not in ref_support for row in samples[:, :k] > 0])
        ),
    }


def run(prior_dir, output_dir, *, n_samples=10000, seed=7, bandwidth=0.03, zero_threshold=0.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
        build_feature_matrix,
        load_prior_training_table,
        load_sed_prior_kde,
        sample_sed_prior,
        samples_to_coeffs,
    )

    prior_dir = Path(prior_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == prior_dir or prior_dir in output_dir.parents:
        raise ValueError("output_dir must be outside the source prior build")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    artifact_path = prior_dir / "sed_prior_kde_native.joblib"
    csv_path = prior_dir / "desi_eazy_empirical_weights.csv"
    artifact = load_sed_prior_kde(artifact_path)
    metadata = artifact["metadata"]
    table = load_prior_training_table(
        csv_path, z_min=metadata["z_min"], max_chi2_dof=metadata["max_chi2_dof"]
    )
    k = artifact["n_templates"]
    features, _, _ = build_feature_matrix(
        table,
        parameterization=artifact["parameterization"],
        simplex_smoothing_eps=metadata["simplex_smoothing_eps"],
    )
    if features.shape != artifact["training_x"].shape or not np.allclose(
        features, artifact["training_x"]
    ):
        raise ValueError("CSV selection does not reproduce baseline KDE training rows")
    weights = table[[f"a{i + 1}" for i in range(k)]].to_numpy()
    auxiliary = table[["log_c_scale", "z"]].to_numpy()
    model = CoefficientMixture(weights, bandwidth=bandwidth, zero_threshold=zero_threshold)
    sampled = model.sample(n_samples, seed=seed)
    # Preserve scale-redshift dependence by resampling their paired empirical rows;
    # independence from coefficients is a deliberate limitation of this baseline.
    rng = np.random.default_rng(seed + 1)
    mixture = np.column_stack([sampled, auxiliary[rng.integers(len(table), size=n_samples)]])
    baseline_features = sample_sed_prior(artifact, n_samples, seed=seed)
    a, log_s, z = samples_to_coeffs(
        baseline_features, k, parameterization=artifact["parameterization"]
    )
    baseline = np.column_stack([a, log_s, z])
    reference = np.column_stack([weights, auxiliary])
    names = [f"a{i + 1}" for i in range(k)] + ["log_c_scale", "z"]
    report = {
        "source_prior": str(prior_dir),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [csv_path, artifact_path]
        },
        "n_training": len(table),
        "n_samples": n_samples,
        "seed": seed,
        "bandwidth_fraction_units": bandwidth,
        "zero_threshold": zero_threshold,
        "feature_names": names,
        "fitted_p0": model.p0.tolist(),
        "independent_all_zero_probability": float(np.prod(model.p0)),
        "conditional_p0_before_closure": (
            (model.p0 - np.prod(model.p0)) / (1 - np.prod(model.p0))
        ).tolist(),
        "limitations": [
            "Descriptive in-sample comparison; no held-out predictive assessment.",
            "Independent coefficient marginals lose joint activation and amplitude correlations.",
            "Conditioning away all-zero rows changes P0; normalization changes positive marginals.",
            "Positive KDE uses repeated reflection at threshold and one; bandwidth is absolute fraction units.",
            "Scale and redshift are jointly bootstrapped, independently of coefficients.",
            "Exact-zero sparse mixture has atoms on simplex faces; not a continuous ILR/flow density or BED entropy estimator.",
            "Threshold defines training inactivity; positive normalized draws may fall below it.",
        ],
        "training": diagnostics(reference, reference),
        "mixture": diagnostics(reference, mixture),
        "ilr_kde": diagnostics(reference, baseline),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an earlier experiment.
    with (output_dir / "comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    np.savez_compressed(
        output_dir / "samples.npz",
        training=reference,
        mixture=mixture,
        ilr_kde=baseline,
        feature_names=names,
    )
    np.savez_compressed(
        output_dir / "mixture_fit.npz",
        training_weights=weights,
        p0=model.p0,
        bandwidth=bandwidth,
        zero_threshold=zero_threshold,
    )
    fig, axes = plt.subplots(3, int(np.ceil(len(names) / 3)), figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        if i >= len(names):
            ax.set_visible(False)
            continue
        for data, label, color in [
            (reference, "DESI fits", "black"),
            (mixture, "P0 + KDE", "tab:orange"),
            (baseline, "ILR KDE", "tab:blue"),
        ]:
            bins = (
                np.linspace(0, 1, 51)
                if i < k
                else np.linspace(
                    min(reference[:, i].min(), baseline[:, i].min()),
                    max(reference[:, i].max(), baseline[:, i].max()),
                    51,
                )
            )
            ax.hist(data[:, i], bins=bins, density=True, histtype="step", label=label, color=color)
        ax.set(xlabel=names[i], ylabel="density", yscale="symlog")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "marginals.png", dpi=150)
    plt.close(fig)
    # Joint structure is central to assessing this deliberately simple model.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, data, title in zip(
        axes, [reference, mixture, baseline], ["DESI fits", "P0 + KDE", "ILR KDE"]
    ):
        std = data.std(axis=0)
        centered = (data - data.mean(axis=0)) / np.where(std > 0, std, 1)
        image = ax.imshow(centered.T @ centered / len(data), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set(
            xticks=range(len(names)),
            yticks=range(len(names)),
            xticklabels=names,
            yticklabels=names,
            title=title,
        )
        ax.tick_params(axis="x", rotation=90)
    fig.colorbar(image, ax=axes.tolist(), shrink=0.8)
    fig.savefig(output_dir / "correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bandwidth", type=float, default=0.03)
    parser.add_argument("--zero-threshold", type=float, default=0.0)
    args = parser.parse_args()
    run(**vars(args))
