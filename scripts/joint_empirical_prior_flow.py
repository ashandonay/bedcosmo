"""Train a pattern-preserving conditional NF prior from the joint KDE sampler."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from prototype_joint_empirical_mixture import JointPatternMixture

from bedcosmo.num_visits.empirical.joint_prior_flow import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bandwidth", type=float, default=0.1)
    parser.add_argument("--n-samples", type=int, default=120000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--transforms", type=int, default=3)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    import torch

    from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (
        load_prior_training_table,
        load_sed_prior_kde,
    )

    torch.set_num_threads(args.threads)
    prior_dir, out = args.prior_dir.resolve(), args.output_dir.resolve()
    if out == prior_dir or prior_dir in out.parents:
        raise ValueError("Write the flow to a new directory outside the source prior build")
    if args.bandwidth <= 0:
        raise ValueError("NF target needs positive KDE bandwidth")
    out.mkdir(parents=True, exist_ok=False)
    artifact_path = prior_dir / "sed_prior_kde_native.joblib"
    metadata = load_sed_prior_kde(artifact_path)["metadata"]
    csv_path = prior_dir / "desi_eazy_empirical_weights.csv"
    table = load_prior_training_table(
        csv_path, z_min=metadata["z_min"], max_chi2_dof=metadata["max_chi2_dof"]
    )
    k = sum(name.startswith("a") and name[1:].isdigit() for name in table.columns)
    names = [f"a{i+1}" for i in range(k)] + ["log_c_scale", "z"]
    rows = table[names].to_numpy()
    target = JointPatternMixture(rows, bandwidth=args.bandwidth)
    request = {
        **vars(args),
        "prior_dir": str(prior_dir),
        "output_dir": str(out),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [artifact_path, csv_path]
        },
    }
    (out / "request.json").write_text(json.dumps(request, indent=2, default=str))
    model = train(
        target,
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        transforms=args.transforms,
        bins=args.bins,
        patience=args.patience,
        seed=args.seed,
        output_dir=out,
    )
    model.metadata["source_sha256"] = request["source_sha256"]
    model.metadata["feature_names"] = names
    model.save(out / "joint_prior_flow.pt")
    np.savez_compressed(out / "joint_kde_fit.npz", training_rows=rows, bandwidth=args.bandwidth)
    (out / "training.json").write_text(json.dumps(model.metadata["training_groups"], indent=2))
    print(f"Saved {out/'joint_prior_flow.pt'}", flush=True)


if __name__ == "__main__":
    main()
