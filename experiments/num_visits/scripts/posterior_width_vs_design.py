#!/usr/bin/env python
"""Robust posterior-width-vs-design diagnostic for the num_visits redshift marginal.

Motivation
----------
The built-in marginal EIG (``--marginal --marginal-eig-subsets z``) estimates
``E_y[H(q(z|y,d))]`` with a 1-D k-NN (Kozachenko-Leonenko) differential-entropy
estimator. On the *near-delta* redshift posteriors this experiment produces, that
estimator is high-variance, and reporting ``optimal_eig = max`` over ~100 designs
selects the upper tail of that noise (a "winner's curse"): it reported a 13.9-bit
"optimal" design whose posterior is visually identical to the 4.5-bit nominal one.

This script sidesteps both problems by measuring a *width* instead of an entropy.
For each design d it computes the redshift posterior standard deviation ``sigma_z(d)``
(and an outlier-robust 16-84 percentile half-width), averaged over outer draws
``y ~ p(y|d)``. Width is a low-variance statistic that does not blow up on sharp
posteriors, and we read the *whole curve* across designs to look for a systematic
trend rather than a single noisy maximum.

It reuses the trained guide and the exact context-building / physical-space mapping
from :class:`bedcosmo.evaluate.Evaluator` (so numbers are on the same footing as the
marginal EIG), only swapping the final ``knn_entropy`` call for ``np.std`` / percentile.

For a Gaussian posterior ``H = 0.5*log2(2*pi*e*sigma^2)``, so we also report a
Gaussian-entropy-plugin ``EIG_z`` that is directly comparable to the k-NN marginal
EIG but free of its tail pathology.

Usage
-----
    # full 100-design sweep on the completed full-design run
    python experiments/num_visits/scripts/posterior_width_vs_design.py \
        --run-id a53146b078284919a6ea2225299b342c --device cuda:0

    # quick look: fewer designs, fewer MC draws
    python experiments/num_visits/scripts/posterior_width_vs_design.py \
        --run-id a53146b078284919a6ea2225299b342c --max-designs 12 \
        --outer-y 16 --inner-samples 4000 --device cuda:0

Requires a GPU allocation (the guide flow lives on ``--device``); ``--device cpu``
works for small ``--max-designs`` sweeps. Writes a CSV + PNG next to the run's
artifacts (``artifacts/plots/``) and prints a summary.
"""

import argparse
import os

# Headless plotting.
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bedcosmo.evaluate import Evaluator
from bedcosmo.pyro_oed_src import LikelihoodDataset
from bedcosmo.util import load_model

# 0.5*log2(2*pi*e): Gaussian differential entropy in bits is this + log2(sigma).
_HALF_LOG2_2PIE = 0.5 * np.log2(2.0 * np.pi * np.e)


def gaussian_entropy_bits(sigma: np.ndarray) -> np.ndarray:
    """Differential entropy (bits) of a 1-D Gaussian with std ``sigma``."""
    return _HALF_LOG2_2PIE + np.log2(np.clip(sigma, 1e-12, None))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default="a53146b078284919a6ea2225299b342c", help="MLflow run id of the trained guide.")
    p.add_argument("--cosmo-exp", default="num_visits")
    p.add_argument("--eval-step", default="last")
    p.add_argument("--param", default="z", help="Physical parameter whose posterior width is measured (default: z).")
    p.add_argument("--design-args-path", default=None, help="Optional design_args yaml; default uses the run's own designs.")
    p.add_argument("--max-designs", type=int, default=None, help="Evaluate only the first N designs (speed).")
    p.add_argument("--outer-y", type=int, default=16, help="Outer y ~ p(y|d) draws per design.")
    p.add_argument("--inner-samples", type=int, default=4000, help="Guide z-samples per outer y (width is stable, so use many).")
    p.add_argument("--n-evals", type=int, default=3, help="Independent repeats for error bars.")
    p.add_argument("--prior-samples", type=int, default=100000, help="Prior draws for sigma_z(prior) / H_prior(z).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=None, help="Output dir (default: run's artifacts/plots).")
    return p.parse_args()


def main():
    args = parse_args()

    # Build the Evaluator exactly as the eval CLI does, so experiment init,
    # input_designs, the physical-space mapping, and model loading all match the
    # marginal-EIG pipeline. marginal_eig_subsets=[[param]] just validates the name.
    evaluator = Evaluator(
        run_id=args.run_id,
        cosmo_exp=args.cosmo_exp,
        device=args.device,
        seed=args.seed,
        design_args_path=args.design_args_path,
        marginal_eig_subsets=[[args.param]],
        marginal_outer_y=args.outer_y,
        marginal_inner_samples=args.inner_samples,
        n_evals=args.n_evals,
        verbose=True,
    )

    exp = evaluator.experiment
    device = torch.device(args.device)
    if args.param not in exp.cosmo_params:
        raise SystemExit(f"--param {args.param!r} not in cosmo_params {list(exp.cosmo_params)}")
    p_idx = list(exp.cosmo_params).index(args.param)

    designs = evaluator.input_designs.to(device)
    if args.max_designs is not None:
        designs = designs[: args.max_designs]
    n_designs = designs.shape[0]
    nominal_idx = evaluator._find_nominal_design_index(designs=designs)

    # Load the trained guide (same call get_marginal_eig uses).
    flow_model, sel_step = load_model(
        exp, args.eval_step, evaluator.run_obj, evaluator.run_args, args.device, global_rank=0
    )
    flow_model = flow_model.to(device)
    flow_model.eval()

    # --- Prior width / entropy (design-independent) --------------------------
    prior_phys = evaluator._marginal_prior_physical_samples(args.prior_samples)
    sigma_prior = float(np.std(prior_phys[:, p_idx]))
    lo, hi = np.percentile(prior_phys[:, p_idx], [16, 84])
    hw_prior = float(0.5 * (hi - lo))
    H_prior_bits = float(gaussian_entropy_bits(np.array([sigma_prior]))[0])
    print(
        f"\nPrior {args.param}: sigma={sigma_prior:.4f}  68%-halfwidth={hw_prior:.4f}  "
        f"H_gauss={H_prior_bits:.3f} bits  (n={len(prior_phys)})"
    )

    M, K = args.outer_y, args.inner_samples
    # Per-eval accumulators, shape (n_evals, n_designs).
    sig_acc, hw_acc, eig_acc = [], [], []

    for e in range(args.n_evals):
        print(f"\n[eval {e + 1}/{args.n_evals}] outer_y={M} inner={K} designs={n_designs}")
        # Outer y ~ p(y|d): contexts [y, design], shape (M, n_designs, ctx_dim).
        _, context = LikelihoodDataset(
            experiment=exp,
            n_particles_per_device=M,
            device=args.device,
            evaluation=False,
            designs=designs,
        )[0]
        context = context.to(device)

        sig_d = np.zeros(n_designs)
        hw_d = np.zeros(n_designs)
        eig_d = np.zeros(n_designs)
        with torch.inference_mode():
            for j in range(n_designs):
                ctx_j = context[:, j, :]  # (M, ctx_dim)
                samples = flow_model(ctx_j).sample((K,))  # (K, M, n_params) flow space
                phys = evaluator._marginal_flow_samples_to_knn_coords(samples)
                zc = phys[..., p_idx].detach().cpu().numpy()  # (K, M)
                # Per outer-y width, then average over the M outer draws.
                sig_y = zc.std(axis=0)  # (M,)
                q = np.percentile(zc, [16, 84], axis=0)  # (2, M)
                hw_y = 0.5 * (q[1] - q[0])  # (M,)
                sig_d[j] = float(sig_y.mean())
                hw_d[j] = float(hw_y.mean())
                # Gaussian-entropy-plugin EIG: H_prior - E_y[H(q(z|y,d))].
                eig_d[j] = H_prior_bits - float(gaussian_entropy_bits(sig_y).mean())
                if j == 0 or (j + 1) % 20 == 0 or j == n_designs - 1:
                    print(f"    design {j + 1}/{n_designs}: sigma_z={sig_d[j]:.4f}  EIG_z~{eig_d[j]:.3f} bits")
        del context
        sig_acc.append(sig_d)
        hw_acc.append(hw_d)
        eig_acc.append(eig_d)

    sig = np.array(sig_acc)   # (n_evals, n_designs)
    hw = np.array(hw_acc)
    eig = np.array(eig_acc)
    sig_mean, sig_std = sig.mean(0), sig.std(0)
    hw_mean = hw.mean(0)
    eig_mean, eig_std = eig.mean(0), eig.std(0)

    # --- Summary -------------------------------------------------------------
    def spread(a):
        return a.max() - a.min()

    nom_txt = ""
    if nominal_idx is not None:
        nom_txt = (
            f"  nominal(idx {nominal_idx}): sigma_z={sig_mean[nominal_idx]:.4f} "
            f"EIG_z={eig_mean[nominal_idx]:.3f}"
        )
    best = int(np.argmin(sig_mean))  # tightest redshift => most informative
    designs_np = designs.detach().cpu().numpy()
    print("\n================ posterior width vs design ================")
    print(f"param                : {args.param}")
    print(f"designs              : {n_designs}")
    print(f"sigma_z  min/med/max : {sig_mean.min():.4f} / {np.median(sig_mean):.4f} / {sig_mean.max():.4f}")
    print(f"sigma_z  spread      : {spread(sig_mean):.4f}  (per-design noise ~{sig_std.mean():.4f})")
    print(f"sigma_z  spread/noise: {spread(sig_mean) / max(sig_std.mean(), 1e-9):.1f}x")
    print(f"EIG_z    min/med/max : {eig_mean.min():.3f} / {np.median(eig_mean):.3f} / {eig_mean.max():.3f} bits")
    print(f"EIG_z    spread      : {spread(eig_mean):.3f} bits  (per-design noise ~{eig_std.mean():.3f})")
    print(nom_txt)
    print(f"tightest design(idx {best}): sigma_z={sig_mean[best]:.4f}  design={designs_np[best].tolist()}")
    print("===========================================================")

    # --- Save CSV ------------------------------------------------------------
    out_dir = args.out_dir or os.path.join(evaluator.save_path, "plots")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.run_id[:8]}_{args.param}_step{sel_step}"
    csv_path = os.path.join(out_dir, f"posterior_width_vs_design_{tag}.csv")
    header = "idx," + ",".join(exp.design_labels) + ",sigma_z,sigma_z_std,halfwidth_1684,eig_z_bits,eig_z_std,is_nominal"
    rows = []
    for j in range(n_designs):
        d = ",".join(f"{v:g}" for v in designs_np[j].tolist())
        rows.append(
            f"{j},{d},{sig_mean[j]:.6f},{sig_std[j]:.6f},{hw_mean[j]:.6f},"
            f"{eig_mean[j]:.6f},{eig_std[j]:.6f},{int(j == nominal_idx)}"
        )
    with open(csv_path, "w") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"\nSaved CSV : {csv_path}")

    # --- Plot ----------------------------------------------------------------
    order = np.argsort(sig_mean)  # rank designs by redshift tightness
    x = np.arange(n_designs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax1.errorbar(x, sig_mean[order], yerr=sig_std[order], fmt="o", ms=3, lw=0.8, capsize=2, color="#2b6cb0")
    ax1.axhline(sigma_prior, ls="--", color="k", lw=1, label=f"prior sigma_z = {sigma_prior:.3f}")
    if nominal_idx is not None:
        rank = int(np.where(order == nominal_idx)[0][0])
        ax1.scatter([rank], [sig_mean[nominal_idx]], color="#e53e3e", zorder=5, s=55, label="nominal design")
    ax1.set_ylabel(r"posterior $\sigma_z$  (smaller = more informative)")
    ax1.set_title(f"Redshift posterior width vs design  (run {args.run_id[:8]}, {n_designs} designs, sorted)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.errorbar(x, eig_mean[order], yerr=eig_std[order], fmt="o", ms=3, lw=0.8, capsize=2, color="#2f855a")
    if nominal_idx is not None:
        rank = int(np.where(order == nominal_idx)[0][0])
        ax2.scatter([rank], [eig_mean[nominal_idx]], color="#e53e3e", zorder=5, s=55, label="nominal design")
        ax2.legend()
    ax2.set_ylabel("EIG$_z$ (Gaussian-plugin, bits)")
    ax2.set_xlabel("design (ranked by $\\sigma_z$)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(out_dir, f"posterior_width_vs_design_{tag}.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
