"""Test observed support patterns and joint physical-coordinate KDE kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from prototype_empirical_mixture import CoefficientMixture, diagnostics
from scipy.stats import ks_2samp


class JointPatternMixture:
    """Empirical pattern masses, with shared-donor product Gaussian KDE kernels.

    Each donor contains all coefficients, log scale and z. Kernel widths are
    bandwidth times within-pattern feature standard deviations. Reflection keeps
    coefficients in [0, 1] and auxiliary coordinates within global training bounds;
    closure then enforces the simplex. Singleton/constant coordinates have zero
    smoothing, explicitly preserving their empirical atoms.
    """

    def __init__(self, rows, *, bandwidth=0.3):
        rows = np.asarray(rows, dtype=float)
        if rows.ndim != 2 or rows.shape[1] < 3 or len(rows) < 2:
            raise ValueError("rows must contain coefficients, log scale and z")
        if (
            not np.isfinite(rows).all()
            or np.any(rows[:, :-2] < 0)
            or not np.allclose(rows[:, :-2].sum(axis=1), 1)
        ):
            raise ValueError("rows must have finite simplex coefficients and auxiliary values")
        if np.any(rows[:, -1] < 0):
            raise ValueError("redshift must be nonnegative")
        if not np.isfinite(bandwidth) or bandwidth < 0:
            raise ValueError("bandwidth must be finite and nonnegative")
        self.rows = rows.copy()
        self.bandwidth = bandwidth
        self.patterns, self.group, self.counts = np.unique(
            rows[:, :-2] > 0, axis=0, return_inverse=True, return_counts=True
        )
        self.members = [np.flatnonzero(self.group == i) for i in range(len(self.patterns))]
        self.widths = np.stack([rows[index].std(axis=0) * bandwidth for index in self.members])
        self.lower = np.r_[np.zeros(rows.shape[1] - 2), rows[:, -2:].min(axis=0)]
        self.upper = np.r_[np.ones(rows.shape[1] - 2), rows[:, -2:].max(axis=0)]

    def sample(self, n, *, seed=7):
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        rng = np.random.default_rng(seed)
        # Uniform donor selection is exactly pattern probability n_pattern / N,
        # followed by uniform donor selection conditional on the pattern.
        donor = rng.integers(len(self.rows), size=n)
        active = self.rows[donor, :-2] > 0
        draws = (
            self.rows[donor]
            + rng.normal(size=(n, self.rows.shape[1])) * self.widths[self.group[donor]]
        )
        width = self.upper - self.lower
        varying = width > 0
        folded = np.remainder(draws[:, varying] - self.lower[varying], 2 * width[varying])
        reflected = self.lower[varying] + width[varying] - np.abs(folded - width[varying])
        outside = (draws[:, varying] < self.lower[varying]) | (
            draws[:, varying] > self.upper[varying]
        )
        draws[:, varying] = np.where(outside, reflected, draws[:, varying])
        draws[:, ~varying] = self.lower[~varying]
        # Reflecting a subnormal positive donor can round to zero; retain an open
        # positive support for active components without adding mass to inactive ones.
        draws[:, :-2] = np.where(active, np.maximum(draws[:, :-2], np.finfo(float).tiny), 0)
        draws[:, :-2] /= draws[:, :-2].sum(axis=1, keepdims=True)
        return draws

    def pattern_summary(self, heldout):
        counts = self.counts
        known = {tuple(p) for p in self.patterns}
        return {
            "n_patterns": len(counts),
            "pattern_counts": counts.tolist(),
            "patterns": self.patterns.astype(int).tolist(),
            "singleton_patterns": int((counts == 1).sum()),
            "mass_in_singleton_patterns": float(counts[counts == 1].sum() / counts.sum()),
            "patterns_below_20_rows": int((counts < 20).sum()),
            "mass_in_patterns_below_20_rows": float(counts[counts < 20].sum() / counts.sum()),
            "heldout_mass_in_unseen_patterns": float(
                np.mean([tuple(p) not in known for p in heldout[:, :-2] > 0])
            ),
        }


def lsst_magnitudes(rows, wave, basis):
    """Full native LSST filter curves; f_lambda = exp(log_s) a.B / (1+z)."""
    from speclite.filters import load_filters

    filters = load_filters("lsst2023-*")
    result = np.empty((len(rows), len(filters)))
    # Interpolate the SED once on the union of native filter grids. The LSST
    # tables share a grid; repeating that expensive interpolation six times is
    # unnecessary. No filter response or wavelength samples are discarded.
    obs = np.unique(np.concatenate([filt.wavelength for filt in filters]))
    for start in range(0, len(rows), 256):
        chunk = rows[start : start + 256]
        query = obs[None, :] / (1 + chunk[:, -1, None])
        if query.min() < wave[0] or query.max() > wave[-1]:
            raise ValueError(
                "Template bank does not cover the full LSST range for sampled redshifts"
            )
        flux = np.zeros(query.shape)
        for i, component in enumerate(basis):
            flux += chunk[:, i, None] * np.interp(query, wave, component)
        flux *= np.exp(chunk[:, -2, None]) * 1e-17 / (1 + chunk[:, -1, None])
        for j, filt in enumerate(filters):
            maggies = filt.get_ab_maggies(flux, obs, axis=1)
            if np.any(~np.isfinite(maggies)) or np.any(maggies <= 0):
                raise ValueError("Photometry produced nonpositive or nonfinite flux")
            result[start : start + len(chunk), j] = -2.5 * np.log10(maggies)
    return result, filters.names


def run(samples_path, prior_dir, output_dir, *, bandwidth=0.3, seed=17, n_samples=10000):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
        build_feature_matrix,
        fit_sed_prior_kde,
        pack_kde_artifact,
        sample_sed_prior,
        samples_to_coeffs,
    )

    samples_path, prior_dir, output_dir = (
        Path(p).resolve() for p in [samples_path, prior_dir, output_dir]
    )
    if output_dir == prior_dir or prior_dir in output_dir.parents:
        raise ValueError("Output must be outside the source prior build")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(samples_path) as saved:
        reference = saved["training"]
        names = saved["feature_names"].tolist()
    k = reference.shape[1] - 2
    if names != [f"a{i+1}" for i in range(k)] + ["log_c_scale", "z"]:
        raise ValueError("Expected named coefficient, scale and redshift columns")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(reference))
    n_holdout = max(1, int(0.2 * len(reference)))
    holdout_idx, train_idx = order[:n_holdout], order[n_holdout:]
    train, heldout = reference[train_idx], reference[holdout_idx]
    start = time.perf_counter()
    model = JointPatternMixture(train, bandwidth=bandwidth)
    joint = model.sample(n_samples, seed=seed)
    joint_seconds = time.perf_counter() - start
    independent_weights = CoefficientMixture(train[:, :k]).sample(n_samples, seed=seed)
    independent = np.column_stack(
        [independent_weights, train[rng.integers(len(train), size=n_samples), -2:]]
    )
    # Refit the ILR comparison on exactly the same split; the old artifact has
    # already seen the held-out rows and would give it an unfair advantage.
    table = pd.DataFrame(train, columns=names)
    for i in range(k):
        table[f"c{i + 1}"] = train[:, i] * np.exp(train[:, -2])
    features, feature_names, _ = build_feature_matrix(table, simplex_smoothing_eps=1e-5)
    kde, scaler = fit_sed_prior_kde(features, bandwidth=0.3)
    artifact = pack_kde_artifact(kde, scaler, feature_names, features, metadata={"n_templates": k})
    a, log_s, z = samples_to_coeffs(sample_sed_prior(artifact, n_samples, seed=seed), k)
    ilr = np.column_stack([a, log_s, z])
    bootstrap = train[rng.integers(len(train), size=n_samples)]
    populations = {
        "heldout": heldout,
        "bootstrap": bootstrap,
        "independent": independent,
        "joint": joint,
        "ilr": ilr,
    }
    # Load the actual exported bank, not a freshly normalized or refitted basis.
    import yaml

    from bedcosmo.num_visits.empirical.templates import (
        load_two_column_template,
        read_template_param,
    )

    config = yaml.safe_load((prior_dir / "prior_args.yaml").read_text())
    if config["flux_unit_scale"] != 1e-17:
        raise ValueError("This DESI photometry experiment expects flux_unit_scale=1e-17")
    param = prior_dir / "templates" / config["template_param"]
    paths = [param.parent / p for p in read_template_param(param)]
    bank = [load_two_column_template(path) for path in paths]
    wave = bank[0][0]
    if len(bank) != k or any(not np.array_equal(wave, item[0]) for item in bank):
        raise ValueError("Expected a shared exported wavelength grid and matching template rank")
    basis = np.stack([item[1] for item in bank])
    photometry, bands = {}, None
    for key, rows in populations.items():
        photometry[key], bands = lsst_magnitudes(rows, wave, basis)
    colors = {key: np.diff(-mags, axis=1) for key, mags in photometry.items()}
    color_names = [a.split("-")[-1] + "−" + b.split("-")[-1] for a, b in zip(bands[:-1], bands[1:])]
    report = {
        "n_train": len(train),
        "n_heldout": len(heldout),
        "n_draws": n_samples,
        "split_seed": seed,
        "joint_bandwidth_within_pattern_std_units": bandwidth,
        "joint_fit_and_sampling_seconds": joint_seconds,
        "pattern_summary": model.pattern_summary(heldout),
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [samples_path, prior_dir / "prior_args.yaml", param, *paths]
        },
        "metrics_vs_heldout": {
            key: diagnostics(heldout, rows) for key, rows in populations.items() if key != "heldout"
        },
        "lsst_color_names": color_names,
        "lsst_color_ks_vs_heldout": {
            key: [
                float(ks_2samp(colors["heldout"][:, i], values[:, i]).statistic)
                for i in range(len(bands) - 1)
            ]
            for key, values in colors.items()
            if key != "heldout"
        },
        "limitations": [
            "Joint model uses observed activation patterns only; unseen patterns have zero probability.",
            "Gaussian product kernels share a joint donor; feature noise is independent within each kernel.",
            "Reflection and simplex closure alter the raw Gaussian KDE density; this is a generative sampler, not a scored simplex density.",
            "Within-pattern zero variance/singletons receive zero smoothing; sparse pattern fits are weakly constrained.",
            "All three models use the same 80/20 random galaxy split, not independent sky-region validation.",
            "Bandwidths fixed in advance: joint 0.3 within-pattern std by default; ILR 0.3 global standardized units; independent 0.03 absolute fractions.",
            "Photometry uses the saved extrapolated templates, not independent observed LSST truth; no measurement noise.",
            "Bootstrap is a no-smoothing empirical control, not an independent model.",
            "No continuous density, BED integration or entropy/EIG assessment implemented.",
        ],
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(
        output_dir / "samples.npz",
        **populations,
        feature_names=names,
        train_indices=train_idx,
        heldout_indices=holdout_idx,
    )
    np.savez_compressed(output_dir / "joint_fit.npz", training_rows=train, bandwidth=bandwidth)
    np.savez_compressed(output_dir / "photometry.npz", **photometry, bands=bands)
    labels = {
        "heldout": "Held-out DESI fits",
        "bootstrap": "Empirical bootstrap",
        "independent": "Independent mixture",
        "joint": "Joint pattern mixture",
        "ilr": "ILR KDE (same split)",
    }
    palette = {
        "heldout": "black",
        "bootstrap": "0.6",
        "independent": "#d66a12",
        "joint": "#249866",
        "ilr": "#246bb2",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, (xname, yname) in zip(axes, [("a4", "a6"), ("z", "a6"), ("z", "a3")]):
        ix, iy = names.index(xname), names.index(yname)
        edges = np.linspace(reference[:, ix].min(), reference[:, ix].max(), 16)
        for key in ["heldout", "independent", "joint", "ilr"]:
            data = populations[key]
            counts, _ = np.histogram(data[:, ix], edges)
            total, _ = np.histogram(data[:, ix], edges, weights=data[:, iy])
            means = np.divide(total, counts, out=np.full(len(counts), np.nan), where=counts >= 20)
            r = np.corrcoef(data[:, ix], data[:, iy])[0, 1]
            ax.plot(
                (edges[1:] + edges[:-1]) / 2,
                means,
                color=palette[key],
                label=f"{labels[key]} (r={r:+.2f})",
            )
        ax.set(xlabel=xname, ylabel=f"Mean {yname} in bin", ylim=(0, 0.8))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Does the simpler joint mixture reproduce held-out DESI relationships?\nAll models fitted on the same 80% of galaxies; bins need ≥20 objects"
    )
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"relationships.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), layout="constrained")
    for i, ax in enumerate(axes):
        for key in ["heldout", "independent", "joint", "ilr"]:
            values = np.sort(colors[key][:, i])
            ax.plot(
                values,
                np.arange(1, len(values) + 1) / len(values),
                color=palette[key],
                label=labels[key],
            )
        ax.set(xlabel=f"LSST {color_names[i]} (AB mag)", ylabel="Cumulative fraction", ylim=(0, 1))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "LSST colors from the same exported template bank\nHeld-out fitted population is the reference; these are synthetic, noiseless colors"
    )
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"lsst_colors.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].bar(np.arange(len(model.counts)), np.sort(model.counts)[::-1], color="#249866")
    axes[0].axhline(20, color="black", ls="--", label="20 training rows")
    axes[0].set(
        xlabel="Activation pattern ranked by training count",
        ylabel="Training galaxies",
        yscale="log",
    )
    axes[0].legend()
    order_counts = np.sort(model.counts)
    axes[1].step(order_counts, np.cumsum(order_counts) / len(train), where="post", color="#249866")
    axes[1].set(
        xlabel="Training galaxies per pattern",
        ylabel="Cumulative population fraction",
        xscale="log",
        ylim=(0, 1),
    )
    fig.suptitle("How much population mass lives in sparse activation patterns?")
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"pattern_coverage.{ext}", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-path", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bandwidth", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-samples", type=int, default=10000)
    run(**vars(parser.parse_args()))
