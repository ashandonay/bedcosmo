"""
Shapefit pipeline runner: chains prep_shapefit_data -> train_nn (which runs eval internally).

Usage:
    python -m bedcosmo.num_tracers.shapefit.run_pipeline \
        --prep-n-samples 50000 \
        --train-epochs 500 --train-hidden-dim 128 --train-lr 0.001
"""

import argparse
import os
import subprocess
import sys

_SHAPEFIT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full shapefit pipeline: prep -> train (includes eval)."
    )

    # Common args
    parser.add_argument("--seed", type=int, default=0, help="Random seed passed to both stages.")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Base output directory. If not set, each stage uses its own default ($SCRATCH/bedcosmo/num_tracers/shapefit).",
    )

    # Prep args
    prep = parser.add_argument_group("prep", "Arguments forwarded to prep_shapefit_data.py")
    prep.add_argument("--prep-n-samples", type=int, default=10000)
    prep.add_argument("--prep-batch-size", type=int, default=256)
    prep.add_argument("--prep-sigma-clip", type=float, default=4.0)
    prep.add_argument("--prep-priors-json", type=str, default="")

    # Train args
    train = parser.add_argument_group("train", "Arguments forwarded to train_nn.py")
    train.add_argument("--train-epochs", type=int, default=1000)
    train.add_argument("--train-batch-size", type=int, default=256)
    train.add_argument("--train-lr", type=float, default=1e-3)
    train.add_argument("--train-weight-decay", type=float, default=1e-6)
    train.add_argument("--train-hidden-dim", type=int, default=256)
    train.add_argument("--train-n-hidden", type=int, default=4)

    args = parser.parse_args()

    # --- Build prep command ---
    prep_cmd = [
        sys.executable,
        os.path.join(_SHAPEFIT_DIR, "prep_shapefit_data.py"),
        "--n-samples", str(args.prep_n_samples),
        "--batch-size", str(args.prep_batch_size),
        "--sigma-clip", str(args.prep_sigma_clip),
        "--seed", str(args.seed),
    ]
    if args.save_path is not None:
        prep_cmd += ["--save-path", args.save_path]
    if args.prep_priors_json:
        prep_cmd += ["--priors-json", args.prep_priors_json]

    # --- Build train command ---
    train_cmd = [
        sys.executable,
        os.path.join(_SHAPEFIT_DIR, "train_nn.py"),
        "--epochs", str(args.train_epochs),
        "--batch-size", str(args.train_batch_size),
        "--lr", str(args.train_lr),
        "--weight-decay", str(args.train_weight_decay),
        "--hidden-dim", str(args.train_hidden_dim),
        "--n-hidden", str(args.train_n_hidden),
        "--seed", str(args.seed),
        "--data-version", "latest",
    ]
    if args.save_path is not None:
        train_cmd += ["--data-path", args.save_path]

    # --- Run pipeline ---
    print("=" * 60)
    print("STAGE 1: Preparing training data")
    print("=" * 60)
    print(f"Command: {' '.join(prep_cmd)}\n")
    subprocess.run(prep_cmd, check=True)

    print()
    print("=" * 60)
    print("STAGE 2: Training (+ eval)")
    print("=" * 60)
    print(f"Command: {' '.join(train_cmd)}\n")
    subprocess.run(train_cmd, check=True)

    print()
    print("=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
