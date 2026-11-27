"""Fit the same joint-pattern architecture with and without KDE training draws."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prototype_joint_empirical_mixture import JointPatternMixture

from bedcosmo.num_visits.empirical.joint_prior_flow import JointPriorFlow, train


def compare(fit, out, *, seed=23, epochs=100, draws=20000):
    with np.load(fit) as saved:
        rows = saved["training_rows"]
        bandwidth = float(saved["bandwidth"])
        names = saved["feature_names"].tolist()
    indices = np.random.default_rng(seed).permutation(len(rows))
    fit_ids, val_ids, test_ids = np.split(indices, [int(0.6 * len(rows)), int(0.8 * len(rows))])
    fitted, validation, test = rows[fit_ids], rows[val_ids], rows[test_ids]
    target = JointPatternMixture(fitted, bandwidth=bandwidth)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "splits.npz", fit=fit_ids, validation=val_ids, test=test_ids)
    models = {}
    samples = {
        "Donor bootstrap": fitted[
            np.random.default_rng(seed + 210).integers(len(fitted), size=draws)
        ]
    }
    report = {
        "seed": seed,
        "source": str(fit),
        "source_sha256": hashlib.sha256(fit.read_bytes()).hexdigest(),
        "split_sizes": [len(fitted), len(validation), len(test)],
        "epochs_requested": epochs,
        "kde_bandwidth": bandwidth,
        "models": {},
    }
    for label, direct in [("Direct flow", True), ("KDE-trained flow", False)]:
        print(label, flush=True)
        model = train(
            target,
            n_samples=len(fitted),
            epochs=epochs,
            batch_size=256,
            patience=18,
            seed=seed,
            direct=direct,
            validation_rows=validation,
        )
        model.metadata["feature_names"] = names
        path = out / ("direct_prior_flow.pt" if direct else "kde_prior_flow.pt")
        model.save(path)
        model = JointPriorFlow.load(path)
        sample = model.sample(draws, seed=seed + 101)
        np.testing.assert_allclose(sample[:, :-2].sum(axis=1), 1)
        if not np.isfinite(model.log_prob(sample)).all():
            raise ValueError(f"{label} generated unscorable rows")
        samples[label] = sample
        models[label] = model
        report["models"][label] = model.metadata["training_groups"]

    log_scores = {k: model.log_prob(test) for k, model in models.items()}
    finite = np.logical_and.reduce([np.isfinite(s) for s in log_scores.values()])
    for score in log_scores.values():
        np.testing.assert_array_equal(np.isfinite(score), finite)
    best_mask = target.patterns[target.counts.argmax()]
    masks = test[:, :-2] > 0
    report["test_coverage"] = {
        "finite_log_density_rows": int(finite.sum()),
        "unseen_mask_rows": sum(tuple(mask) not in models["Direct flow"].lookup for mask in masks),
        "all_active_rows": int(np.all(masks == best_mask, axis=1).sum()),
    }
    dimensions = np.random.default_rng(seed + 310).normal(size=(rows.shape[1], 64))
    dimensions /= np.linalg.norm(dimensions, axis=0)
    projection = dimensions / fitted.std(axis=0)[:, None]
    q = np.linspace(0.001, 0.999, 401)
    target_q = np.quantile(test @ projection, q, axis=0)
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]

    def positive_cdf(data):
        a = data[:, :-2]
        return np.array([((a > 0) & (a <= x)).sum(0) / (a > 0).sum(0) for x in thresholds])

    cdf = positive_cdf(test)
    scores = {}
    for label, sample in samples.items():
        entry = {
            "sliced_wasserstein": float(
                np.abs(np.quantile(sample @ projection, q, axis=0) - target_q).mean()
            ),
            "near_zero_mae_pp": float(100 * np.abs(positive_cdf(sample) - cdf).mean()),
        }
        if label in log_scores:
            entry["mean_test_log_density_nats"] = float(log_scores[label][finite].mean())
        scores[label] = entry
    per_row_diff = log_scores["Direct flow"][finite] - log_scores["KDE-trained flow"][finite]
    boot = np.random.default_rng(seed + 410)
    means = np.array(
        [
            boot.choice(per_row_diff, size=len(per_row_diff), replace=True).mean()
            for _ in range(2000)
        ]
    )
    report["paired_log_density_direct_minus_kde_95pct_row_bootstrap"] = np.quantile(
        means, [0.025, 0.975]
    ).tolist()
    report["test"] = scores
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    active = np.flatnonzero(best_mask)
    x = np.geomspace(0.0001, 0.1, 180)
    fig, axes = plt.subplots(
        2,
        int(np.ceil(len(active) / 2)),
        figsize=(12, 6),
        squeeze=False,
        sharey=True,
        sharex=True,
        layout="constrained",
    )
    for (label, sample), color, style in zip(
        {"Held-out DESI": test, **samples}.items(),
        ["#252e3c", "#888888", "#177774", "#aa5734"],
        ["-", ":", "-", "--"],
    ):
        selected = np.all((sample[:, :-2] > 0) == best_mask, axis=1)
        for ax, j in zip(axes.flat, active):
            sorted_values = np.sort(sample[selected, j])
            ax.plot(
                100 * x,
                100 * np.searchsorted(sorted_values, x, side="right") / len(sorted_values),
                label=label,
                color=color,
                ls=style,
                lw=1.8,
            )
            ax.set_title(f"Template {j+1}")
            ax.set_xscale("log")
            ax.axvline(1, color="#dddddd", lw=0.7, zorder=0)
    for ax in list(axes.flat)[len(active) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Active fraction threshold (%)")
    fig.supylabel("Share at or below threshold (%)")
    fig.suptitle(
        f"Direct versus KDE-trained flows | mask {''.join(str(int(bit)) for bit in best_mask)}"
    )
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncols=4)
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"heldout_near_zero.{ext}", dpi=175)
    plt.close(fig)
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    compare(args.fit, args.output_dir, seed=args.seed, epochs=args.epochs, draws=args.draws)
