#!/usr/bin/env python
"""Does the k-NN marginal EIG_z converge to the focused-guide result as we add samples?

Consistency check between two independent estimators of the SAME quantity
``I(z; y | d)`` on the num_visits empirical model:

  1. **k-NN marginal EIG** on the trained *joint* 13-D guide (run a53146b0):
     ``EIG_z(d) = H_knn(z) - E_y[ H_knn(z | y, d) ]`` where both entropies are
     Kozachenko-Leonenko k-NN estimates on samples (the production
     ``Evaluator._marginal_posterior_entropy`` path). This is high-variance on
     sharp posteriors; ``optimal = max`` over designs is a winner's-curse tail.

  2. **Focused (marginalized-params-at-training) guide** (run cb82ee76): an
     explicit 1-D ``q(z|y,d)`` flow whose entropy is exact. Reference numbers:
     mean EIG_z = 2.359 bits, spread (max-min over 100 designs) = 0.221 bits,
     per-design MC std ~ 0.006 bits.

The two runs share the *identical* 100 designs, so we can compare per-design.
This script sweeps the k-NN estimator's inner-sample count K (and optionally the
outer-y count M), driving the real production method, and reports for each K:
mean EIG, the max-min spread, the per-design noise, the winner's-curse "optimal",
and the Pearson/Spearman rank correlation of the per-design EIG_z against the
focused guide. If the k-NN estimator is merely noisy (not structurally wrong),
its spread should shrink toward ~0.2 bits and its ranking should correlate with
the focused guide as K grows.

Usage (GPU allocation required for the guide flow):
    python experiments/num_visits/scripts/knn_marginal_convergence.py \
        --k-sweep 200,800,3200,12800 --outer-y 16 --n-evals 3 --device cuda:0
"""

import argparse
import json
import os

import numpy as np
import torch

from bedcosmo.entropy import gaussian_entropy, kde_entropy, knn_entropy
from bedcosmo.evaluate import Evaluator
from bedcosmo.util import load_model

FOCUSED_EIG_JSON = (
    "/pscratch/sd/a/ashandon/bedcosmo/num_visits/mlruns/"
    "334848290134960244/cb82ee76b6164d1a8ca3662488d987ac/artifacts/"
    "eig_data_20260714_0932.json"
)


def _mean_over_outer_y(est, Z, OK, K, M, n_designs, min_n):
    """Per-design mean over outer-y of ``est`` on the in-support draws.

    ``Z``/``OK`` are ``(n_designs, K_max, M)``: the sample buffer and its
    validity mask. Rows failing the mask are dropped rather than clamped (a
    clamp stacks them on one bound value and collapses k-NN). A (design, y)
    cell with ``<= min_n`` surviving draws yields NaN instead of a bogus value.

    Returns:
        ``(n_designs,)`` array of mean entropies, NaN where no y-cell was usable.
    """
    out = np.empty(n_designs)
    for j in range(n_designs):
        per_y = []
        for m in range(M):
            rows = Z[j][:K, m][OK[j][:K, m]]
            per_y.append(est(rows) if rows.size > min_n else np.nan)
        out[j] = np.nan if np.all(np.isnan(per_y)) else np.nanmean(per_y)
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", default="a53146b078284919a6ea2225299b342c",
                   help="Joint (all-param) guide run id.")
    p.add_argument("--cosmo-exp", default="num_visits")
    p.add_argument("--eval-step", default="last")
    p.add_argument("--param", default="z")
    p.add_argument("--k-sweep", default="200,800,3200,12800",
                   help="Comma list of inner-sample counts K to sweep.")
    p.add_argument("--outer-y", type=int, default=16, help="Outer y ~ p(y|d) draws per design (M).")
    p.add_argument("--knn-k", type=int, default=3, help="k in the k-NN entropy estimator.")
    p.add_argument("--n-evals", type=int, default=3, help="Independent repeats for error bars.")
    p.add_argument("--prior-samples", type=int, default=100000, help="Prior draws for H_knn(z).")
    p.add_argument("--max-designs", type=int, default=None)
    p.add_argument("--design-chunk", type=int, default=8,
                   help="Designs per batched flow-sample call (bounds GPU memory).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    k_sweep = [int(x) for x in args.k_sweep.split(",") if x.strip()]

    evaluator = Evaluator(
        run_id=args.run_id,
        cosmo_exp=args.cosmo_exp,
        device=args.device,
        seed=args.seed,
        marginal_eig_subsets=[[args.param]],
        marginal_outer_y=args.outer_y,
        marginal_inner_samples=k_sweep[0],
        marginal_knn_k=args.knn_k,
        n_evals=args.n_evals,
        verbose=False,
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

    # Focused-guide per-design EIG_z reference (identical designs).
    focused = None
    if os.path.exists(FOCUSED_EIG_JSON):
        fj = json.load(open(FOCUSED_EIG_JSON))
        focused = np.array(fj["step_50000"]["variable"]["eigs_avg"], dtype=float)[:n_designs]
        print(f"Focused EIG_z: mean={focused.mean():.3f}  spread={focused.ptp():.3f}  "
              f"min/max={focused.min():.3f}/{focused.max():.3f} bits (n={len(focused)})")

    flow_model, sel_step = load_model(
        exp, args.eval_step, evaluator.run_obj, evaluator.run_args, args.device, global_rank=0
    )
    flow_model = flow_model.to(device).eval()

    # Design-independent prior entropy H_knn(z) (bits), large fixed N.
    # _prior_marginal_coords drops out-of-support draws instead of clamping them:
    # a clamp stacks every offender onto one bound value, which is exactly the tie
    # pattern that collapses k-NN toward -inf. Same helper the production
    # get_marginal_eig uses, so this script measures the production estimator.
    prior_z, n_drop, n_total = evaluator._prior_marginal_coords(args.prior_samples, [p_idx])
    if n_drop:
        print(f"  prior: dropped {n_drop}/{n_total} out-of-support draws")
    H_prior = float(knn_entropy(prior_z, k=args.knn_k))
    print(f"H_knn(prior z) = {H_prior:.3f} bits  (N={len(prior_z)}, k={args.knn_k})\n")

    try:
        from scipy.stats import pearsonr, spearmanr
        have_scipy = True
    except Exception:
        have_scipy = False

    # Efficient + fair sweep: draw K_max inner samples ONCE per (eval, design,
    # outer-y), then subsample prefixes for each K. The expensive part (flow
    # sampling + transform to knn coords) happens once; only the cheap knn_entropy
    # repeats per K. Same estimator as _marginal_posterior_entropy (knn per outer-y,
    # mean over M), just reusing draws across K so noise differences come purely
    # from sample count, not independent randomness.
    import time

    from bedcosmo.pyro_oed_src import LikelihoodDataset

    M = args.outer_y
    K_max = max(k_sweep)
    eig_by_K = {K: [] for K in k_sweep}   # kNN per-eval per-design EIG arrays
    geig_by_K = {K: [] for K in k_sweep}  # Gaussian-plugin per-eval per-design EIG
    keig_by_K = {K: [] for K in k_sweep}  # 1-D KDE-plugin per-eval per-design EIG

    # Straightforward alternatives to kNN for a 1-D target, on the same z-samples
    # (same coord space as the kNN estimator / _marginal_knn_coords_and_mask):
    #   * Gaussian plug-in  H = 0.5*log2(2*pi*e*var)          -- shape-blind
    #   * 1-D KDE plug-in   H = -mean log2 p_hat(z)           -- nonparametric.
    # The KDE plug-in keeps the Gaussian's low 1-D variance but drops the Gaussian
    # shape assumption -- the num_visits z-posteriors are skewed/shouldered (see
    # plot_posterior_kde_vs_flow.py), where Gaussian-sigma is biased high.
    def gauss_H(x):
        x = np.asarray(x, dtype=np.float64)
        return -np.inf if x.std() < 1e-12 else gaussian_entropy(x)

    def kde_H(x):
        x = np.asarray(x, dtype=np.float64)
        return -np.inf if x.std() < 1e-12 else kde_entropy(x)

    H_prior_g = gauss_H(prior_z[:, 0])
    H_prior_kde = kde_H(prior_z[:, 0])
    print(f"H_gauss(prior z) = {H_prior_g:.3f} bits   "
          f"H_kde(prior z) = {H_prior_kde:.3f} bits\n", flush=True)

    chunk = args.design_chunk
    for e in range(args.n_evals):
        t0 = time.time()
        _, context = LikelihoodDataset(
            experiment=exp, n_particles_per_device=M,
            device=args.device, evaluation=False, designs=designs,
        )[0]
        context = context.to(device)              # (M, n_designs, ctx)
        # float64: a float32 buffer quantises sharp posteriors onto a ~1.2e-7 grid
        # and manufactures duplicate rows that k-NN then has to drop -- ties the
        # production path (torch default dtype float64) never produces. Measured:
        # ~12 dup rows per 1000 draws at sigma_z=1e-3 in float32, exactly 0 in float64.
        Z = np.empty((n_designs, K_max, M), dtype=np.float64)  # physical z coords
        # Validity mask rather than a clamp. The buffer must stay rectangular for
        # the prefix-subsampling below, so out-of-support draws are marked here
        # and filtered at estimation time -- never mapped onto a bound, which
        # would stack them on one value and collapse k-NN.
        OK = np.empty((n_designs, K_max, M), dtype=bool)
        with torch.inference_mode():
            # Batch designs in chunks so K_max * (M*chunk) intermediates fit in GPU mem.
            for c0 in range(0, n_designs, chunk):
                c1 = min(c0 + chunk, n_designs)
                nc = c1 - c0
                ctx_c = context[:, c0:c1, :].reshape(M * nc, -1)     # (M*nc, ctx)
                samp = flow_model(ctx_c).sample((K_max,))           # (K_max, M*nc, P)
                coords, valid = evaluator._marginal_knn_coords_and_mask(samp)
                zc = coords[..., p_idx].reshape(K_max, M, nc)        # (K_max, M, nc)
                vc = valid[..., p_idx].reshape(K_max, M, nc)
                Z[c0:c1] = zc.permute(2, 0, 1).detach().cpu().numpy()  # (nc, K_max, M)
                OK[c0:c1] = vc.permute(2, 0, 1).detach().cpu().numpy()
                del samp, coords, valid, zc, vc
        del context
        n_drop_post = int(OK.size - OK.sum())
        if n_drop_post:
            print(f"  [eval {e + 1}] dropped {n_drop_post}/{OK.size} "
                  f"({100.0 * n_drop_post / OK.size:.2f}%) out-of-support guide draws",
                  flush=True)
        print(f"  [eval {e + 1}/{args.n_evals}] sampled K_max={K_max} in "
              f"{time.time() - t0:.0f}s", flush=True)
        min_n = args.knn_k + 1
        for K in k_sweep:
            # warn_duplicates=False: this runs n_designs*M*len(k_sweep) times, and
            # residual ties are the known ~0.05% icdf-clamp endpoint (cdf_eps=1e-3),
            # reported in aggregate rather than per call.
            H_post = _mean_over_outer_y(
                lambda x: knn_entropy(x, k=args.knn_k, warn_duplicates=False),
                Z, OK, K, M, n_designs, min_n,
            )
            eig_by_K[K].append(H_prior - H_post)  # kNN EIG_z per design
            H_post_g = _mean_over_outer_y(gauss_H, Z, OK, K, M, n_designs, min_n)
            geig_by_K[K].append(H_prior_g - H_post_g)  # Gaussian-plugin EIG_z
            H_post_kde = _mean_over_outer_y(kde_H, Z, OK, K, M, n_designs, min_n)
            keig_by_K[K].append(H_prior_kde - H_post_kde)  # KDE-plugin EIG_z
            # Per-eval quick summary so a timeout never loses a completed eval.
            fr = ""
            if focused is not None and have_scipy:
                fr = (f"  spearman[knn/gauss/kde]="
                      f"{spearmanr(eig_by_K[K][-1], focused)[0]:.3f}/"
                      f"{spearmanr(geig_by_K[K][-1], focused)[0]:.3f}/"
                      f"{spearmanr(keig_by_K[K][-1], focused)[0]:.3f}")
            print(f"    K={K} spread[knn/gauss/kde]="
                  f"{eig_by_K[K][-1].ptp():.3f}/{geig_by_K[K][-1].ptp():.3f}/"
                  f"{keig_by_K[K][-1].ptp():.3f} bits{fr}", flush=True)
        print(f"  [eval {e + 1}/{args.n_evals}] done in {time.time() - t0:.0f}s", flush=True)

    rows = []
    header = (f"{'K':>7} {'meanEIG':>8} {'min':>7} {'max':>7} {'spread':>7} "
              f"{'noise':>7} {'spr/noise':>9} {'optimal':>8}")
    if focused is not None:
        header += f" {'pearson':>8} {'spearman':>9}"
    print("\n" + header)
    print("-" * len(header))

    for K in k_sweep:
        eig = np.array(eig_by_K[K])              # (n_evals, n_designs)
        eig_mean = eig.mean(0)
        eig_noise = eig.std(0).mean()
        spread = float(eig_mean.ptp())
        line = (f"{K:>7} {eig_mean.mean():>8.3f} {eig_mean.min():>7.3f} {eig_mean.max():>7.3f} "
                f"{spread:>7.3f} {eig_noise:>7.3f} {spread/max(eig_noise,1e-9):>9.1f} "
                f"{eig_mean.max():>8.3f}")
        rec = dict(K=K, mean=float(eig_mean.mean()), min=float(eig_mean.min()),
                   max=float(eig_mean.max()), spread=spread, noise=float(eig_noise))
        if focused is not None and have_scipy:
            pr = pearsonr(eig_mean, focused)[0]
            sr = spearmanr(eig_mean, focused)[0]
            line += f" {pr:>8.3f} {sr:>9.3f}"
            rec.update(pearson=float(pr), spearman=float(sr))
        print(line)
        rows.append(rec)

    # Parallel Gaussian-plugin table on the identical z-samples (no kNN).
    grows = []
    print("\n" + "=" * len(header))
    print("Gaussian-plugin EIG_z on the SAME joint z-samples (no kNN):")
    print(header)
    print("-" * len(header))
    for K in k_sweep:
        geig = np.array(geig_by_K[K])
        gm = geig.mean(0)
        gn = geig.std(0).mean()
        gs = float(gm.ptp())
        line = (f"{K:>7} {gm.mean():>8.3f} {gm.min():>7.3f} {gm.max():>7.3f} "
                f"{gs:>7.3f} {gn:>7.3f} {gs/max(gn,1e-9):>9.1f} {gm.max():>8.3f}")
        grec = dict(K=K, mean=float(gm.mean()), min=float(gm.min()), max=float(gm.max()),
                    spread=gs, noise=float(gn))
        if focused is not None and have_scipy:
            gpr = pearsonr(gm, focused)[0]
            gsr = spearmanr(gm, focused)[0]
            line += f" {gpr:>8.3f} {gsr:>9.3f}"
            grec.update(pearson=float(gpr), spearman=float(gsr))
        print(line)
        grows.append(grec)

    # Parallel KDE-plugin table on the identical z-samples (shape-correct, low-var).
    krows = []
    print("\n" + "=" * len(header))
    print("KDE-plugin EIG_z on the SAME joint z-samples (no kNN, no Gaussian assumption):")
    print(header)
    print("-" * len(header))
    for K in k_sweep:
        keig = np.array(keig_by_K[K])
        km = keig.mean(0)
        kn = keig.std(0).mean()
        ks = float(km.ptp())
        line = (f"{K:>7} {km.mean():>8.3f} {km.min():>7.3f} {km.max():>7.3f} "
                f"{ks:>7.3f} {kn:>7.3f} {ks/max(kn,1e-9):>9.1f} {km.max():>8.3f}")
        krec = dict(K=K, mean=float(km.mean()), min=float(km.min()), max=float(km.max()),
                    spread=ks, noise=float(kn))
        if focused is not None and have_scipy:
            kpr = pearsonr(km, focused)[0]
            ksr = spearmanr(km, focused)[0]
            line += f" {kpr:>8.3f} {ksr:>9.3f}"
            krec.update(pearson=float(kpr), spearman=float(ksr))
        print(line)
        krows.append(krec)

    print("\nReference (focused guide): mean=2.359  spread=0.221  noise~0.006 bits")

    out_dir = args.out_dir or os.path.join(evaluator.save_path, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"knn_marginal_convergence_{args.run_id[:8]}_{args.param}.json")
    with open(out_json, "w") as f:
        json.dump(dict(run_id=args.run_id, param=args.param, outer_y=args.outer_y,
                       knn_k=args.knn_k, n_evals=args.n_evals,
                       H_prior_knn=H_prior, H_prior_gauss=H_prior_g,
                       H_prior_kde=H_prior_kde,
                       focused_mean=None if focused is None else float(focused.mean()),
                       focused_spread=None if focused is None else float(focused.ptp()),
                       knn_sweep=rows, gauss_sweep=grows, kde_sweep=krows), f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
