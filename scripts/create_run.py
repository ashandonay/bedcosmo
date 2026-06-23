#!/usr/bin/env python3
"""
Pre-create an MLflow run at submission time and snapshot the referenced config into its artifacts.

On SLURM a train job can sit queued for a long time. The values in train_args.yaml are already
frozen at submission (submit.sh flattens them into argv), but the *referenced files*
(prior_args.yaml, design_args.yaml, and the emulator checkpoints) were historically read at job
start, so edits during the queue window leaked in. This script closes that gap: it creates the
MLflow run now, copies the referenced config into the run's artifacts, tags the run
``submit_status=queued``, and prints the run_id. submit.sh then passes ``--attach-run-id <id>`` to
the job, which attaches to this run (see bedcosmo.train._init_run) instead of creating a new one.

Snapshotted artifacts:
  - prior_args.yaml   (from --prior-args-path, or from --prior-flow-path's run when set)
  - design_args.yaml  (from --design-args-path)
  - emulators/<tracer_bin>.pt  (when --likelihood-mode == emulator)

Only the train flow uses this. resume/restart/eval/grid attach to runs derived from MLflow state.

Usage (args mirror the resolved train argv; extras are ignored via parse_known_args):
    python create_run.py --cosmo-exp num_tracers --mlflow-exp base --cosmo-model base \
        --dataset dr1 --analysis bao --likelihood-mode emulator \
        --prior-args-path prior_args_hrdrag.yaml --design-args-path design_args_dr1.yaml
"""

import argparse
import datetime
import os
import shutil
import sys

import mlflow
from mlflow.tracking import MlflowClient

from bedcosmo.util import get_experiment_config_path, extract_run_info_from_checkpoint_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Pre-create an MLflow run and snapshot config")
    parser.add_argument("--cosmo-exp", type=str, default="num_tracers")
    parser.add_argument("--mlflow-exp", type=str, default=None)
    parser.add_argument("--cosmo-model", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dr2")
    parser.add_argument("--analysis", type=str, default="bao")
    parser.add_argument("--likelihood-mode", type=str, default="scaling")
    parser.add_argument("--prior-args-path", type=str, default=None)
    parser.add_argument("--design-args-path", type=str, default=None)
    parser.add_argument("--prior-flow-path", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


def _resolve_config_path(cosmo_exp, path):
    """Resolve a config path: absolute as-is, else relative to the experiment config dir."""
    if os.path.isabs(path):
        return path
    return str(get_experiment_config_path(cosmo_exp, path))


def _snapshot_prior_args(args, storage_path, artifacts_dir):
    dest = os.path.join(artifacts_dir, "prior_args.yaml")
    if args.prior_flow_path:
        # prior_args come from the prior-flow run's artifacts (mirrors train.py:852-863).
        run_id, exp_id, _ = extract_run_info_from_checkpoint_path(args.prior_flow_path)
        src = f"{storage_path}/mlruns/{exp_id}/{run_id}/artifacts/prior_args.yaml"
        if not os.path.exists(src):
            raise FileNotFoundError(f"prior_args.yaml not found for prior-flow run {run_id}: {src}")
        shutil.copy2(src, dest)
    elif args.prior_args_path:
        src = _resolve_config_path(args.cosmo_exp, args.prior_args_path)
        if not os.path.exists(src):
            raise FileNotFoundError(f"prior_args file not found: {src}")
        shutil.copy2(src, dest)
    else:
        print("Warning: no prior_args_path/prior_flow_path; skipping prior_args.yaml snapshot", file=sys.stderr)


def _snapshot_design_args(args, artifacts_dir):
    if not args.design_args_path:
        print("Warning: no design_args_path; skipping design_args.yaml snapshot", file=sys.stderr)
        return
    src = _resolve_config_path(args.cosmo_exp, args.design_args_path)
    if not os.path.exists(src):
        raise FileNotFoundError(f"design_args file not found: {src}")
    shutil.copy2(src, os.path.join(artifacts_dir, "design_args.yaml"))


def _snapshot_emulators(args, artifacts_dir):
    # Only num_tracers has emulator checkpoints; resolve and copy each non-null .pt.
    from bedcosmo.num_tracers.experiment import NumTracers

    checkpoints = NumTracers.resolve_emulator_checkpoints(args.analysis, args.cosmo_model, args.dataset)
    emu_dir = os.path.join(artifacts_dir, "emulators")
    os.makedirs(emu_dir, exist_ok=True)
    copied = []
    for tracer_bin, ckpt_path in checkpoints.items():
        if ckpt_path is None:
            continue
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Emulator checkpoint for '{tracer_bin}' not found: {ckpt_path}"
            )
        shutil.copy2(ckpt_path, os.path.join(emu_dir, f"{tracer_bin}.pt"))
        copied.append(tracer_bin)
    print(f"Snapshotted emulator checkpoints: {copied}", file=sys.stderr)


def main():
    args = _parse_args()
    mlflow_exp = args.mlflow_exp or args.cosmo_model

    storage_path = os.environ["SCRATCH"] + f"/bedcosmo/{args.cosmo_exp}"
    mlflow.set_tracking_uri(f"file:{storage_path}/mlruns")
    experiment = mlflow.set_experiment(mlflow_exp)

    client = MlflowClient()
    run = client.create_run(experiment.experiment_id, run_name=args.cosmo_model)
    run_id = run.info.run_id

    artifact_uri = run.info.artifact_uri
    artifacts_dir = artifact_uri[7:] if artifact_uri.startswith("file://") else artifact_uri
    os.makedirs(artifacts_dir, exist_ok=True)

    _snapshot_prior_args(args, storage_path, artifacts_dir)
    _snapshot_design_args(args, artifacts_dir)
    if args.likelihood_mode == "emulator":
        _snapshot_emulators(args, artifacts_dir)

    client.set_tag(run_id, "submit_status", "queued")
    client.set_tag(run_id, "submit_time", datetime.datetime.now().isoformat(timespec="seconds"))

    # The MLflow run directory (parent of artifacts/), then run_id on the last stdout
    # line for submit.sh to capture.
    print(f"RUN_PATH={os.path.dirname(artifacts_dir)}")
    print(run_id)


if __name__ == "__main__":
    main()
