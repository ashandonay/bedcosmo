"""Compare reflected, Dirichlet, and within-mask log-ratio KDEs on held-out DESI fits."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prototype_joint_empirical_mixture import JointPatternMixture
from scipy.linalg import helmert
from sklearn.covariance import oas


def dirichlet_draws(model, donors, gaussian, kappa, rng):
    """Replace fractions while preserving the donor and the auxiliary draw."""
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    a = model.rows[donors, :-2]
    smooth = (model.counts[model.group[donors]] > 1) & ((a > 0).sum(axis=1) > 1)
    a = a[smooth]
    active = a > 0
    alpha = kappa * (a / a.sum(axis=1, keepdims=True))[active]
    # Gamma(alpha+1) * Uniform**(1/alpha) has shape alpha. Normalize in
    # log space so very small shape parameters cannot zero out a whole draw.
    log_gamma = np.full_like(a, -np.inf)
    log_gamma[active] = np.log(rng.gamma(alpha + 1)) - rng.exponential(size=len(alpha)) / alpha
    relative = log_gamma - log_gamma.max(axis=1, keepdims=True)
    floor = np.log(np.finfo(float).tiny)
    floored = int(np.count_nonzero(relative[active] < floor))
    weights = np.where(active, np.exp(np.maximum(relative, floor)), 0)
    samples = gaussian.copy()
    samples[smooth, :-2] = weights / weights.sum(axis=1, keepdims=True)
    return samples, floored


def fit_logratio_kernels(model):
    """Fit one shrunk full covariance in orthonormal log-ratio coordinates per mask."""
    kernels = {}
    for group, (mask, members) in enumerate(zip(model.patterns, model.members)):
        if mask.sum() < 2 or len(members) < 2:
            continue
        basis = helmert(mask.sum(), full=False)
        coords = np.log(model.rows[members, :-2][:, mask]) @ basis.T
        covariance, _ = oas(coords)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        kernels[group] = (basis, root)
    return kernels


def logratio_draws(model, kernels, donors, gaussian, h, rng):
    """Smooth on each mask's simplex, sharing the reflected KDE's auxiliary draws."""
    if not np.isfinite(h) or h < 0:
        raise ValueError("h must be finite and nonnegative")
    samples = gaussian.copy()
    groups = model.group[donors]
    floor = np.log(np.finfo(float).tiny)
    floored = 0
    for group, (basis, root) in kernels.items():
        selected = np.flatnonzero(groups == group)
        cols = np.flatnonzero(model.patterns[group])
        fractions = model.rows[np.ix_(donors[selected], cols)]
        if h == 0:
            samples[np.ix_(selected, cols)] = fractions
            continue
        coord = np.log(fractions) @ basis.T
        noise = rng.normal(size=coord.shape) @ root.T
        logits = (coord + h * noise) @ basis
        relative = logits - logits.max(axis=1, keepdims=True)
        floored += int(np.count_nonzero(relative < floor))
        weights = np.exp(np.maximum(relative, floor))
        samples[np.ix_(selected, cols)] = weights / weights.sum(axis=1, keepdims=True)
    return samples, floored


def audit(fit, out, seed):
    with np.load(fit) as data:
        rows = data["training_rows"]
        bandwidth = float(data["bandwidth"])
    rng = np.random.default_rng(seed)
    train, val, test = np.split(rng.permutation(rows), [int(0.6 * len(rows)), int(0.8 * len(rows))])
    model = JointPatternMixture(train, bandwidth=bandwidth)
    kernels = fit_logratio_kernels(model)
    known = {tuple(mask) for mask in model.patterns}
    directions = rng.normal(size=(rows.shape[1], 64))
    scale = train.std(axis=0)
    scale[scale == 0] = 1
    projection = directions / (np.linalg.norm(directions, axis=0) * scale[:, None])
    q = np.linspace(0.001, 0.999, 401)
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]

    def boundary_cdf(sample):
        a = sample[:, :-2]
        return np.stack(
            [((a > 0) & (a <= t)).sum(axis=0) / (a > 0).sum(axis=0) for t in thresholds]
        )

    def evaluate(reference, sample_seed, concentrations, ilr_bandwidths):
        keep = np.array([tuple(mask) in known for mask in reference[:, :-2] > 0])
        target = reference[keep]
        target_q = np.quantile(target @ projection, q, axis=0)
        target_cdf = boundary_cdf(target)
        target_corr = np.corrcoef(target, rowvar=False)[:-2, -2:]
        n = 24000
        gaussian = model.sample(n, seed=sample_seed)
        # Replay the sampler's uniform donor draw before it generates Gaussian noise.
        donors = np.random.default_rng(sample_seed).integers(len(train), size=n)
        np.testing.assert_array_equal(gaussian[:, :-2] > 0, train[donors, :-2] > 0)
        candidates = {"Bootstrap": (train[donors], 0), "Reflected Gaussian": (gaussian, 0)}
        for kappa in concentrations:
            candidates[f"Dirichlet {kappa}"] = dirichlet_draws(
                model, donors, gaussian, kappa, np.random.default_rng([sample_seed, 1])
            )
        for h in ilr_bandwidths:
            candidates[f"Within-mask ILR (h={h:g})"] = logratio_draws(
                model, kernels, donors, gaussian, h, np.random.default_rng([sample_seed, 2])
            )
        scores = {}
        for label, (sample, floored) in candidates.items():
            scores[label] = {
                "sliced_wasserstein": float(
                    np.abs(np.quantile(sample @ projection, q, axis=0) - target_q).mean()
                ),
                "near_zero_mae_pp": float(100 * np.abs(boundary_cdf(sample) - target_cdf).mean()),
                "fraction_aux_correlation_rmse": float(
                    np.sqrt(
                        np.mean((np.corrcoef(sample, rowvar=False)[:-2, -2:] - target_corr) ** 2)
                    )
                ),
                "floored_active_values": floored,
            }
        return scores, target, candidates, int((~keep).sum())

    grid = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    ilr_grid = [0, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5]
    val_scores, _, _, val_unseen = evaluate(val, seed + 1, grid, ilr_grid)
    best_h = min(
        ilr_grid, key=lambda h: val_scores[f"Within-mask ILR (h={h:g})"]["sliced_wasserstein"]
    )
    best = min(grid, key=lambda k: val_scores[f"Dirichlet {k}"]["sliced_wasserstein"])
    scores, target, candidates, test_unseen = evaluate(test, seed + 2, [best], [best_h])
    report = {
        "seed": seed,
        "bandwidth": bandwidth,
        "kappa": best,
        "ilr_bandwidth": best_h,
        "split_sizes": [len(train), len(val), len(test)],
        "unseen_validation": val_unseen,
        "unseen_test": test_unseen,
        "thresholds": thresholds,
        "validation": val_scores,
        "test": scores,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    mask = model.patterns[model.counts.argmax()]
    active = np.flatnonzero(mask)
    x = np.geomspace(0.0001, 0.1, 160)
    fig, axes = plt.subplots(
        2,
        int(np.ceil(len(active) / 2)),
        figsize=(12, 6),
        squeeze=False,
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    populations = {"Held-out DESI": target, **{k: v[0] for k, v in candidates.items()}}
    for (label, sample), color, style in zip(
        populations.items(),
        ["#20252c", "#858790", "#a95225", "#167a78", "#6c4da2"],
        ["-", ":", "--", "-", "-."],
    ):
        sample = sample[np.all((sample[:, :-2] > 0) == mask, axis=1)]
        for ax, j in zip(axes.flat, active):
            values = np.sort(sample[:, j])
            ax.plot(
                100 * x,
                100 * np.searchsorted(values, x, side="right") / len(values),
                label=label,
                color=color,
                ls=style,
                lw=1.8,
            )
            ax.set_title(f"Template {j+1}")
            ax.set_xscale("log")
    for ax in list(axes.flat)[: len(active)]:
        ax.axvline(1, color="#cccccc", lw=0.8)
        ax.grid(axis="y", alpha=0.2)
    for ax in list(axes.flat)[len(active) :]:
        ax.set_visible(False)
    bits = "".join(str(int(bit)) for bit in mask)
    count = np.all((target[:, :-2] > 0) == mask, axis=1).sum()
    fig.suptitle(f"Near-zero fractions in mask {bits} | {count:,} held-out DESI galaxies")
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Fraction threshold (%)")
    fig.supylabel("Share at or below threshold (%)")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncols=5)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"near_zero_comparison.{ext}", dpi=175)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "validation"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    audit(args.fit, args.output_dir, args.seed)
