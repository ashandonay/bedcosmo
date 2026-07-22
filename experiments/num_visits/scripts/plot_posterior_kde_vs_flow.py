#!/usr/bin/env python
"""Sanity-check the KDE plug-in: overlay the flow's marginal z-posterior samples
against a 1-D KDE fit and the Gaussian plug-in, for a few outer-y draws.

For a couple of designs (default: the focused-guide *optimal* and *worst* designs)
we draw a few y ~ p(y|d), sample the trained joint guide q(theta|y,d), slice to the
target column z, and plot per (design, y):

  * histogram of the raw flow z-samples (ground truth the plug-ins approximate)
  * scipy gaussian_kde density (the KDE plug-in, Scott bandwidth)
  * Gaussian plug-in N(mean, std) with the same samples

Each panel is annotated with H_kde and H_gauss (bits) so you can see where the
Gaussian assumption over/under-states the entropy. This tells us whether the
num_visits photo-z posteriors are Gaussian, skewed, or multimodal, and whether the
KDE bandwidth is sane -- i.e. whether the KDE plug-in is the right low-variance
estimator to replace the 13-D k-NN marginal.

Usage (GPU allocation required for the guide flow):
    python experiments/num_visits/scripts/plot_posterior_kde_vs_flow.py \
        --n-samples 8000 --n-y 4 --device cuda:0
"""

import argparse
import json
import os

import numpy as np
import torch

FOCUSED_EIG_JSON = (
    "/pscratch/sd/a/ashandon/bedcosmo/num_visits/mlruns/"
    "334848290134960244/cb82ee76b6164d1a8ca3662488d987ac/artifacts/"
    "eig_data_20260714_0932.json"
)
_HALF_LOG2_2PIE = 0.5 * np.log2(2.0 * np.pi * np.e)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", default="a53146b078284919a6ea2225299b342c",
                   help="Joint (all-param) guide run id.")
    p.add_argument("--cosmo-exp", default="num_visits")
    p.add_argument("--eval-step", default="last")
    p.add_argument("--param", default="z")
    p.add_argument("--n-samples", type=int, default=8000, help="Flow z-samples per (design,y).")
    p.add_argument("--n-y", type=int, default=4, help="Outer y ~ p(y|d) draws per design.")
    p.add_argument("--designs", default="auto",
                   help="'auto' = focused optimal+worst; else comma list of design indices.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from scipy.stats import gaussian_kde, norm

    from bedcosmo.evaluate import Evaluator
    from bedcosmo.pyro_oed_src import LikelihoodDataset
    from bedcosmo.util import load_model

    evaluator = Evaluator(
        run_id=args.run_id, cosmo_exp=args.cosmo_exp, device=args.device,
        seed=args.seed, marginal_eig_subsets=[[args.param]],
        marginal_outer_y=args.n_y, marginal_inner_samples=args.n_samples,
        marginal_knn_k=3, n_evals=1, verbose=False,
    )
    exp = evaluator.experiment
    device = torch.device(args.device)
    if args.param not in exp.cosmo_params:
        raise SystemExit(f"--param {args.param!r} not in {list(exp.cosmo_params)}")
    p_idx = list(exp.cosmo_params).index(args.param)

    designs_all = evaluator.input_designs.to(device)
    n_all = designs_all.shape[0]

    # Pick which designs to visualise.
    focused = None
    if os.path.exists(FOCUSED_EIG_JSON):
        fj = json.load(open(FOCUSED_EIG_JSON))
        focused = np.array(fj["step_50000"]["variable"]["eigs_avg"], dtype=float)[:n_all]
    if args.designs == "auto":
        if focused is None:
            raise SystemExit("No focused JSON; pass explicit --designs indices.")
        d_idx = [int(np.argmax(focused)), int(np.argmin(focused))]
        d_label = {d_idx[0]: f"optimal (EIG_z={focused[d_idx[0]]:.3f})",
                   d_idx[1]: f"worst (EIG_z={focused[d_idx[1]]:.3f})"}
    else:
        d_idx = [int(x) for x in args.designs.split(",") if x.strip()]
        d_label = {i: f"design {i}" for i in d_idx}
    designs = designs_all[d_idx]

    flow_model, sel_step = load_model(
        exp, args.eval_step, evaluator.run_obj, evaluator.run_args, args.device, global_rank=0
    )
    flow_model = flow_model.to(device).eval()

    # z-samples: (n_designs, n_y, n_samples) in physical coords.
    M = args.n_y
    _, context = LikelihoodDataset(
        experiment=exp, n_particles_per_device=M, device=args.device,
        evaluation=False, designs=designs,
    )[0]
    context = context.to(device)  # (M, n_designs, ctx)
    nd = designs.shape[0]
    Z = np.empty((nd, M, args.n_samples), dtype=np.float32)
    with torch.inference_mode():
        ctx_flat = context.reshape(M * nd, -1)
        samp = flow_model(ctx_flat).sample((args.n_samples,))  # (n_samp, M*nd, P)
        coords = evaluator._marginal_flow_samples_to_knn_coords(samp)
        zc = coords[..., p_idx].reshape(args.n_samples, M, nd)  # (n_samp, M, nd)
        Z = zc.permute(2, 1, 0).detach().cpu().numpy()  # (nd, M, n_samp)
    del context, samp, coords, zc

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nd, M, figsize=(3.4 * M, 3.0 * nd), squeeze=False)
    for r in range(nd):
        for c in range(M):
            ax = axes[r][c]
            z = Z[r, c]
            mu, sd = float(z.mean()), float(z.std())
            # KDE plug-in.
            kde = gaussian_kde(z)  # Scott bandwidth
            H_kde = float(-np.mean(np.log2(np.clip(kde(z), 1e-300, None))))
            H_g = _HALF_LOG2_2PIE + np.log2(max(sd, 1e-12))
            lo, hi = z.min(), z.max()
            pad = 0.05 * (hi - lo + 1e-6)
            grid = np.linspace(lo - pad, hi + pad, 400)
            ax.hist(z, bins=60, density=True, color="0.8", edgecolor="none", label="flow samples")
            ax.plot(grid, kde(grid), color="C0", lw=1.8, label="KDE plug-in")
            ax.plot(grid, norm.pdf(grid, mu, sd), color="C3", lw=1.4, ls="--",
                    label="Gaussian plug-in")
            ax.axvline(mu, color="0.4", lw=0.8, ls=":")
            ax.set_title(f"{d_label[d_idx[r]]}\ny#{c}: H_kde={H_kde:.3f}  H_gauss={H_g:.3f} bits",
                         fontsize=8)
            ax.set_xlabel(args.param)
            if c == 0:
                ax.set_ylabel("density")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()

    out_dir = args.out_dir or os.path.join(evaluator.save_path, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"posterior_kde_vs_flow_{args.run_id[:8]}_{args.param}.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
