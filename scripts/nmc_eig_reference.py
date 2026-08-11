"""Nested-Monte-Carlo reference EIG for num_visits runs.

Why this exists
---------------
``bedcosmo.grid_calc`` brute-forces the EIG on an explicit (parameter x feature)
grid.  That requires the feature (magnitude) axis to resolve the *narrowest*
per-parameter photometric error, because ``bed`` normalizes the likelihood over
the feature grid.  For the ``bbt`` runs the g-band error reaches ~1e-4 mag over
a ~25 mag span, which needs O(1e5-1e6) feature points per band -- two to three
orders of magnitude past what fits in node memory.

The likelihood here is exactly Gaussian given ``(z, T, nvisits)``, so we do not
need a feature grid at all: NMC draws ``y`` from the model and evaluates the
marginal ``p(y|d)`` as a Monte-Carlo mixture over fresh prior draws.  It is
grid-free, exact in the ``M -> inf`` limit, and directly comparable to the
neural-flow EIG (both are differential-entropy differences in bits).

Usage
-----
    python -m scripts.nmc_eig_reference <run_id> [<run_id> ...] \
        --n-outer 4000 --m-inner 4000 --repeats 5 --device cuda:0

Designs always come from the run's artifacts; there is no design-args override.

Pass ``--m-ladder 500,2000,8000`` to check the O(1/M) NMC bias instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import mlflow
import numpy as np
import torch

from bedcosmo.pyro_oed_src import nmc_eig
from bedcosmo.util import auto_seed, load_experiment

LOG2 = math.log(2.0)


def nmc_bits(experiment, designs, n_outer: int, m_inner: int, design_chunk: int) -> np.ndarray:
    """EIG in bits for each row of ``designs``, chunked over designs to cap memory."""
    out = []
    for start in range(0, designs.shape[0], design_chunk):
        chunk = designs[start : start + design_chunk]
        eig_nats = nmc_eig(
            experiment.pyro_model,
            chunk,
            observation_labels=experiment.observation_labels,
            target_labels=list(experiment.cosmo_params),
            N=n_outer,
            M=m_inner,
        )
        out.append(eig_nats.detach().cpu().numpy().reshape(-1) / LOG2)
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--cosmo-exp", default="num_visits")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-outer", type=int, default=4000, help="N: outer y ~ p(y|d) draws")
    p.add_argument("--m-inner", type=int, default=4000, help="M: inner theta draws for p(y|d)")
    p.add_argument(
        "--m-ladder",
        default=None,
        help="Comma-separated M values; runs a bias ladder instead of --m-inner",
    )
    p.add_argument("--repeats", type=int, default=5, help="Independent replicas for the MC error")
    p.add_argument("--design-chunk-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None, help="Write results JSON here")
    args = p.parse_args()

    if "SCRATCH" not in os.environ:
        raise OSError("SCRATCH environment variable is required.")
    mlflow.set_tracking_uri(
        "file:" + os.environ["SCRATCH"] + f"/bedcosmo/{args.cosmo_exp}/mlruns"
    )

    m_values = (
        [int(v) for v in args.m_ladder.split(",")] if args.m_ladder else [args.m_inner]
    )
    results: dict = {"args": vars(args), "runs": {}}

    for run_id in args.run_ids:
        experiment = load_experiment(run_id, args.cosmo_exp, args.device)
        designs = experiment.designs.to(torch.float64)
        designs_np = designs.detach().cpu().numpy()
        nominal_np = experiment.nominal_design.detach().cpu().numpy().reshape(-1)
        nominal_idx = int(np.abs(designs_np - nominal_np).sum(axis=1).argmin())

        tag = f"{run_id[:8]}  norm_mode={experiment.norm_mode}"
        print("\n" + "=" * 78)
        print(f"{tag}   cosmo_params={list(experiment.cosmo_params)}  "
              f"n_designs={designs_np.shape[0]}")
        print("=" * 78)

        run_res: dict = {"norm_mode": experiment.norm_mode,
                         "designs": designs_np.tolist(),
                         "nominal_design": nominal_np.tolist(),
                         "nominal_idx": nominal_idx,
                         "by_m": {}}

        for m in m_values:
            reps = []
            t0 = time.time()
            for r in range(args.repeats):
                auto_seed(args.seed + 1000 * r)
                reps.append(nmc_bits(experiment, designs, args.n_outer, m,
                                     args.design_chunk_size))
            reps_arr = np.stack(reps)  # (repeats, n_designs)
            mean = reps_arr.mean(axis=0)
            sem = reps_arr.std(axis=0, ddof=1) / math.sqrt(args.repeats) if args.repeats > 1 \
                else np.zeros_like(mean)
            best = int(mean.argmax())
            print(f"\n  N={args.n_outer} M={m}  ({time.time() - t0:.0f}s, "
                  f"{args.repeats} replicas)")
            for i, d in enumerate(designs_np):
                mark = ""
                if i == nominal_idx:
                    mark += " <- nominal"
                if i == best:
                    mark += " <- optimal"
                print(f"    {d.tolist()}  EIG = {mean[i]:8.4f} +/- {sem[i]:.4f} bits{mark}")
            print(f"    nominal EIG = {mean[nominal_idx]:.4f} bits, "
                  f"optimal EIG = {mean[best]:.4f} bits @ {designs_np[best].tolist()}")
            run_res["by_m"][str(m)] = {
                "eig_mean_bits": mean.tolist(),
                "eig_sem_bits": sem.tolist(),
                "nominal_eig_bits": float(mean[nominal_idx]),
                "optimal_eig_bits": float(mean[best]),
                "optimal_design": designs_np[best].tolist(),
            }

        results["runs"][run_id] = run_res

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
