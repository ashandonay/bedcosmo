#!/usr/bin/env python
"""Where do the residual k-NN duplicate rows come from after the clamp fixes?

Context. Dropping out-of-support guide draws (instead of clamping them onto a
prior bound) removed the catastrophic tie pile-up from the marginal-EIG path,
but ~49 ``knn_entropy`` duplicate warnings still fire per eval, at 5-9 ties per
1000 draws. Two candidate mechanisms:

  1. float32 quantisation. Ruled out for the production eval: ``util.py`` sets
     the default dtype to float64 at import, and float64 yields zero ties at
     every posterior width tested. (It IS live in knn_marginal_convergence.py,
     which explicitly downcasts its sample buffer to float32.)

  2. ``Bijector._icdf_lookup`` (transform.py:396) clamps ``u`` to
     ``[cdf_eps, 1 - cdf_eps]``, so every draw past |z| ~ 4.75 in unconstrained
     space (cdf_eps=1e-6) maps to the *identical* physical value -- the CDF grid
     endpoint. ``erf`` independently saturates to exactly 1.0 at |z| ~ 8.25.
     Crucially the clamped output lies INSIDE the prior support, so
     ``_physical_samples_valid_mask`` passes it and the row-dropping fixes
     cannot catch it.

Mechanism (2) is confirmed to exist; what is NOT confirmed is whether it
explains the observed RATE. For a standard normal P(|z| > 4.75) ~ 2e-6, which
over K=1000 predicts ~0.002 ties -- a thousand times too few. It only works if
the trained flow puts real mass in the far tail of unconstrained space.

This script settles it by driving the real production path and, for every tied
value found, asking: is it equal to the CDF grid endpoint?

Usage (GPU allocation required):
    python experiments/num_visits/scripts/diagnose_knn_ties.py --device cuda:0
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bedcosmo.evaluate import Evaluator  # noqa: E402
from bedcosmo.pyro_oed_src import LikelihoodDataset  # noqa: E402
from bedcosmo.util import load_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", default="a53146b078284919a6ea2225299b342c")
    p.add_argument("--cosmo-exp", default="num_visits")
    p.add_argument("--eval-step", default="last")
    p.add_argument("--param", default="z")
    p.add_argument("--inner-k", type=int, default=1000, help="K inner draws (eval default).")
    p.add_argument("--outer-y", type=int, default=8, help="M outer y (eval default).")
    p.add_argument("--n-designs", type=int, default=12, help="Designs to scan.")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    evaluator = Evaluator(
        run_id=args.run_id,
        cosmo_exp=args.cosmo_exp,
        device=args.device,
        marginal_eig_subsets=[[args.param]],
        marginal_outer_y=args.outer_y,
        marginal_inner_samples=args.inner_k,
        verbose=False,
    )
    exp = evaluator.experiment
    p_idx = list(exp.cosmo_params).index(args.param)
    designs = evaluator.input_designs.to(device)
    n_designs = min(args.n_designs, designs.shape[0])
    designs = designs[:n_designs]

    flow_model, _ = load_model(
        exp, args.eval_step, evaluator.run_obj, evaluator.run_args, args.device, global_rank=0
    )
    flow_model = flow_model.to(device).eval()

    print(f"knn coord space         : {evaluator._marginal_knn_space()}")
    print(f"default torch dtype     : {torch.get_default_dtype()}")

    # --- Candidate endpoints: what value would an icdf clamp produce? ----------
    bij = exp.param_bijector
    endpoints = []
    cdfs = getattr(bij, "cdfs", {}) or {}
    if args.param in cdfs:
        bins = cdfs[args.param]["bins"]
        endpoints = [float(bins[0]), float(bins[-1])]
        print(f"cdf_eps                 : {getattr(bij, 'cdf_eps', None)}")
        print(f"CDF grid endpoints for {args.param}: {endpoints[0]:.12g} .. {endpoints[1]:.12g}")
        print(f"  (u clamp saturates at |z_unconstrained| >~ "
              f"{abs(float(np.sqrt(2) * torch.erfinv(torch.tensor(2 * (1 - bij.cdf_eps) - 1)))):.2f})")
    else:
        print(f"'{args.param}' has no per-param CDF (joint gaussianizer block?); "
              "endpoint check limited to prior bounds.")
    prior_dist = exp.prior.get(args.param)
    lo, hi = float(prior_dist.low), float(prior_dist.high)
    print(f"prior support           : [{lo:.12g}, {hi:.12g}]")

    # --- Drive the real production path --------------------------------------
    M, K = args.outer_y, args.inner_k
    _, context = LikelihoodDataset(
        experiment=exp, n_particles_per_device=M, device=args.device,
        evaluation=False, designs=designs,
    )[0]
    context = context.to(device)

    tie_values = Counter()
    n_tied_rows = n_rows = n_groups = 0
    unc_tail = []

    with torch.inference_mode():
        for j in range(n_designs):
            ctx_j = context[:, j, :]
            raw = flow_model(ctx_j).sample((K,))          # unconstrained, (K, M, P)
            unc_tail.append(raw[..., p_idx].abs().flatten().cpu().numpy())
            coords, valid = evaluator._marginal_knn_coords_and_mask(raw)
            z = coords[..., p_idx].detach().cpu().numpy()  # (K, M)
            ok = valid[..., p_idx].detach().cpu().numpy()

            for m in range(M):
                col = z[ok[:, m], m]
                n_rows += col.size
                vals, counts = np.unique(col, return_counts=True)
                dup = vals[counts > 1]
                if dup.size:
                    n_groups += int(dup.size)
                    n_tied_rows += int((counts[counts > 1]).sum() - dup.size)
                    # Count EXCESS rows (c - 1 per group), matching n_tied_rows,
                    # so the endpoint share below is a like-for-like fraction.
                    for v, c in zip(dup, counts[counts > 1]):
                        tie_values[float(v)] += int(c) - 1

    unc = np.concatenate(unc_tail)
    print("\n--- unconstrained-space tail mass (the icdf-clamp trigger) ---")
    for q in (0.5, 0.9, 0.99, 0.999, 1.0):
        print(f"  |z_unc| quantile {q:<6}: {np.quantile(unc, q):.3f}")
    for thr in (4.75, 6.36, 8.25):
        frac = float((unc > thr).mean())
        print(f"  P(|z_unc| > {thr:<5}) = {frac:.3e}   (~{frac * K:.2f} of K={K} draws)")

    print("\n--- ties on the production (float64, unclamped) posterior path ---")
    print(f"  rows scanned          : {n_rows}")
    print(f"  tied rows             : {n_tied_rows} "
          f"({100.0 * n_tied_rows / max(n_rows, 1):.3f}%)")
    print(f"  distinct tied values  : {len(tie_values)}  in {n_groups} groups")

    if not tie_values:
        print("\n  No ties reproduced. The icdf-clamp hypothesis does NOT explain the "
              "eval warnings at this design/sample budget.")
        return

    print("\n  top tied values (value, #rows, matches endpoint?):")
    for v, c in tie_values.most_common(10):
        tags = []
        for name, ep in ([("cdf_lo", endpoints[0]), ("cdf_hi", endpoints[1])] if endpoints else []):
            if v == ep:
                tags.append(name)
        if v == lo:
            tags.append("prior_low")
        if v == hi:
            tags.append("prior_high")
        print(f"    {v:.17g}  n={c:<5} {'<- ' + ','.join(tags) if tags else ''}")

    on_ep = sum(c for v, c in tie_values.items() if endpoints and v in endpoints)
    print(f"\n  tied rows sitting exactly on a CDF grid endpoint: {on_ep}/{n_tied_rows} "
          f"({100.0 * on_ep / max(n_tied_rows, 1):.1f}%)")
    if on_ep == 0:
        print("  => VERDICT: icdf-clamp REFUTED as the source; ties are elsewhere.")
    elif on_ep >= 0.9 * n_tied_rows:
        print("  => VERDICT: icdf-clamp CONFIRMED as the dominant source.")
    else:
        print("  => VERDICT: icdf-clamp explains only part; another source remains.")


if __name__ == "__main__":
    main()
