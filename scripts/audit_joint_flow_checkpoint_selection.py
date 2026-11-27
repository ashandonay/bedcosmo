"""Compare direct-flow checkpoint selection by likelihood and sample distance."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from prototype_joint_empirical_mixture import JointPatternMixture

from bedcosmo.num_visits.empirical.joint_prior_flow import JointPriorFlow, train


def audit(fit, baseline_dir, out, seed):
    with np.load(fit) as data, np.load(baseline_dir / "splits.npz") as split:
        rows = data["training_rows"]
        names = data["feature_names"].tolist()
        bandwidth = float(data["bandwidth"])
        fitted, val, test = [rows[split[key]] for key in ("fit", "validation", "test")]
    target = JointPatternMixture(fitted, bandwidth=bandwidth)
    baseline = JointPriorFlow.load(baseline_dir / "direct_prior_flow.pt")
    alternative = train(
        target,
        n_samples=len(fitted),
        epochs=100,
        batch_size=256,
        patience=18,
        seed=seed,
        direct=True,
        validation_rows=val,
        select_by_samples=True,
    )
    for m, group in alternative.groups.items():
        # Check that we compared candidates from the very same optimization run.
        np.testing.assert_allclose(
            group["history"], baseline.groups[m]["history"], rtol=1e-5, atol=1e-6
        )
    alternative.metadata["feature_names"] = names
    out.mkdir(parents=True, exist_ok=False)
    artifact = out / "sample_selected_prior_flow.pt"
    alternative.save(artifact)
    alternative = JointPriorFlow.load(artifact)
    n = 20000
    samples = {
        "raw_kde": target.sample(n, seed=seed + 101),
        "likelihood_selected": baseline.sample(n, seed=seed + 101),
        "sample_selected": alternative.sample(n, seed=seed + 101),
    }
    for sample in samples.values():
        np.testing.assert_allclose(sample[:, :-2].sum(axis=1), 1)
    assert np.isfinite(alternative.log_prob(samples["sample_selected"])).all()
    directions = np.random.default_rng(seed + 310).normal(size=(rows.shape[1], 64))
    projection = directions / (np.linalg.norm(directions, axis=0) * fitted.std(axis=0)[:, None])
    q = np.linspace(0.001, 0.999, 401)
    generated_q = {
        label: np.quantile(sample @ projection, q, axis=0) for label, sample in samples.items()
    }

    def boundary_cdf(values):
        a = values[:, :-2]
        return np.stack(
            [((a > 0) & (a <= t)).sum(0) / (a > 0).sum(0) for t in [0.001, 0.005, 0.01, 0.02, 0.05]]
        )

    report = {"seed": seed, "groups": alternative.metadata["training_groups"]}
    for stage, reference in [("validation", val), ("test", test)]:
        ref_q = np.quantile(reference @ projection, q, axis=0)
        ref_cdf = boundary_cdf(reference)
        report[stage] = scores = {}
        for label, sample in samples.items():
            scores[label] = {
                "sliced_wasserstein": float(np.abs(generated_q[label] - ref_q).mean()),
                "near_zero_mae_pp": float(100 * np.abs(boundary_cdf(sample) - ref_cdf).mean()),
            }
        log_a, log_b = baseline.log_prob(reference), alternative.log_prob(reference)
        np.testing.assert_array_equal(np.isfinite(log_a), np.isfinite(log_b))
        for label, scores_for_rows in [("likelihood_selected", log_a), ("sample_selected", log_b)]:
            scores[label]["mean_log_density_nats"] = float(
                scores_for_rows[np.isfinite(scores_for_rows)].mean()
            )
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "groups"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.fit, args.baseline_dir, args.output_dir, args.seed)
