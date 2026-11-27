"""Build a direct spline prior with an automatic loss schedule."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from prototype_joint_empirical_mixture import JointPatternMixture
from tune_joint_prior_sample_loss import evaluate, finetune

from bedcosmo.num_visits.empirical.joint_prior_flow import (
    JointPriorFlow,
    initialize_direct_prior,
    train,
)


def run(
    fit,
    split_path,
    out,
    *,
    seed,
    cpu_budget,
    baseline_dir=None,
    group_warmup=False,
    schedule="ramp"
):
    started = time.process_time()
    previous = None
    if baseline_dir is not None:
        previous = json.loads((baseline_dir / "comparison.json").read_text())
        seed = previous["seed"]
        cpu_budget = previous["cpu_budget"]
        split_path = baseline_dir / "splits.npz"

    with np.load(fit) as data:
        rows = data["training_rows"]
        names = data["feature_names"].tolist()
        h = float(data["bandwidth"])
    if split_path is None:
        indices = np.random.default_rng(seed).permutation(len(rows))
        fit_ids, val_ids, test_ids = np.split(indices, [int(0.6 * len(rows)), int(0.8 * len(rows))])
    else:
        with np.load(split_path) as split:
            fit_ids, val_ids, test_ids = [split[k] for k in ("fit", "validation", "test")]
    fitted, val, test = [rows[ids] for ids in [fit_ids, val_ids, test_ids]]
    target = JointPatternMixture(fitted, bandwidth=h)
    if schedule == "switch":
        initial = train(
            target,
            n_samples=len(fitted),
            epochs=100,
            batch_size=256,
            patience=18,
            seed=seed,
            direct=True,
            validation_rows=val,
        )
    else:
        initial = initialize_direct_prior(target, seed=seed)
    floor = (
        previous["likelihood_then_combined"]["minimum_validation_log_density"]
        if previous is not None
        else None
    )
    model = finetune(
        initial,
        fitted,
        val,
        seed=seed,
        steps=180,
        nll_weight=0.08,
        batch_size=2048,
        full_reference=True,
        ramp_sample_loss=schedule == "ramp",
        cpu_budget=cpu_budget,
        minimum_log_density=floor,
        group_warmup=group_warmup,
    )
    training_seconds = time.process_time() - started
    model.metadata.update(feature_names=names, loss_schedule=schedule)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "splits.npz", fit=fit_ids, validation=val_ids, test=test_ids)
    path = out / "scheduled_prior_flow.pt"
    model.save(path)
    restored = JointPriorFlow.load(path)
    report = {
        "seed": seed,
        "cpu_budget": cpu_budget,
        "training_cpu_seconds": training_seconds,
        "schedule": schedule,
        "training": restored.metadata["sample_finetuning"],
        "validation": evaluate({"single_run": restored}, fitted, val, seed)["single_run"],
        "test": evaluate({"single_run": restored}, fitted, test, seed)["single_run"],
    }
    if previous is not None:
        report["two_stage_test"] = previous["test"]["likelihood_then_combined"]
        report["raw_kde_test"] = previous["test"]["raw_kde"]
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["test"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--group-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--schedule", choices=["ramp", "switch"], default="ramp")
    parser.add_argument("--cpu-seconds", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu_seconds is not None and (
        not np.isfinite(args.cpu_seconds) or args.cpu_seconds <= 0
    ):
        parser.error("--cpu-seconds must be positive")
    if args.schedule == "switch" and (
        args.baseline_dir is not None or args.group_warmup or args.cpu_seconds is not None
    ):
        parser.error("--schedule switch uses the fixed likelihood and 180-update combined schedule")
    if args.baseline_dir is not None and (args.cpu_seconds is not None or args.splits is not None):
        parser.error("--baseline-dir already supplies the CPU budget and split indices")
    torch.set_num_threads(2)
    run(
        args.fit,
        args.splits,
        args.output_dir,
        seed=args.seed,
        cpu_budget=(args.cpu_seconds or 250) if args.schedule == "ramp" else None,
        baseline_dir=args.baseline_dir,
        group_warmup=args.group_warmup,
        schedule=args.schedule,
    )
