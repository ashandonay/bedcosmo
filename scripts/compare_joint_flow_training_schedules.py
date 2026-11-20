"""Compare direct spline training schedules at matched CPU time."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from prototype_joint_empirical_mixture import JointPatternMixture
from tune_joint_prior_sample_loss import evaluate, finetune

from bedcosmo.num_visits.empirical.joint_prior_flow import JointPriorFlow, build_flow, train


def reset_weights(model, seed):
    """Reuse only training geometry and empirical masses, with fresh network weights."""
    cold = copy.deepcopy(model)
    for m, group in cold.groups.items():
        torch.manual_seed(seed + m)
        group["flow"] = build_flow(m + 1, cold.patterns.shape[1], cold.metadata["flow_config"])
        group["history"] = []
    cold.metadata.pop("training_groups")
    cold.metadata["training_source"] = "observed_rows_combined_from_start"
    return cold


def compare(fit, split_path, out, *, seed, warm_steps, cold_lr=2e-4):
    if warm_steps < 1 or not np.isfinite(cold_lr) or cold_lr <= 0:
        raise ValueError("warm_steps and the cold learning rate must be positive")
    with np.load(fit) as saved, np.load(split_path) as split:
        ids = {key: split[key] for key in ("fit", "validation", "test")}
        fitted, val, test = [saved["training_rows"][i] for i in ids.values()]
        names = saved["feature_names"].tolist()
        kde = JointPatternMixture(fitted, bandwidth=float(saved["bandwidth"]))
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "splits.npz", **ids)
    settings = {"seed": seed, "nll_weight": 0.08, "batch_size": 2048, "full_reference": True}
    start = time.process_time()
    likelihood = train(
        kde,
        n_samples=len(fitted),
        epochs=100,
        batch_size=256,
        patience=18,
        seed=seed,
        direct=True,
        validation_rows=val,
    )
    likelihood_seconds = time.process_time() - start
    likelihood.metadata["feature_names"] = names
    cold_init = reset_weights(likelihood, seed)
    warm = finetune(copy.deepcopy(likelihood), fitted, val, steps=warm_steps, **settings)
    warm_stats = warm.metadata["sample_finetuning"]
    budget = likelihood_seconds + warm_stats["cpu_seconds"]
    print(
        f"Switch schedule finished in {budget:.1f} CPU seconds. Starting combined-only.", flush=True
    )
    cold = finetune(
        cold_init,
        fitted,
        val,
        steps=warm_steps,
        cpu_budget=budget,
        lr=cold_lr,
        minimum_log_density=warm_stats["minimum_validation_log_density"],
        **settings,
    )
    models = {
        "likelihood_only": likelihood,
        "likelihood_then_combined": warm,
        "combined_from_start": cold,
    }
    for label, model in models.items():
        path = out / f"{label}.pt"
        model.save(path)
        models[label] = JointPriorFlow.load(path)
    distances = {
        "seed": seed,
        "fit_sha256": hashlib.sha256(fit.read_bytes()).hexdigest(),
        "settings": settings,
        "likelihood_training_cpu_seconds": likelihood_seconds,
        "cpu_budget": budget,
        "likelihood_then_combined": warm_stats,
        "combined_from_start": cold.metadata["sample_finetuning"],
    }
    models["raw_kde"] = kde.sample(24000, seed=seed + 101)
    for name, data in [("validation", val), ("test", test)]:
        support = [np.isfinite(model.log_prob(data)) for model in [likelihood, warm, cold]]
        for supported in support[1:]:
            np.testing.assert_array_equal(supported, support[0])
        distances[name] = evaluate(models, fitted, data, seed)
        distances[name]["density_rows"] = int(support[0].sum())
    (out / "comparison.json").write_text(json.dumps(distances, indent=2, allow_nan=False) + "\n")
    print(json.dumps(distances["test"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warm-steps", type=int, default=180)
    parser.add_argument("--cold-learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    compare(
        args.fit,
        args.splits,
        args.output_dir,
        seed=args.seed,
        warm_steps=args.warm_steps,
        cold_lr=args.cold_learning_rate,
    )
