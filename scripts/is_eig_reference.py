"""Importance-sampled reference EIG for num_visits runs.

Relation to the other two references
------------------------------------
``scripts/nmc_eig_reference.py`` estimates the marginal ``log p(y|d)`` by drawing
inner ``theta'`` from the *prior*.  Here the per-parameter information gain runs
to >30 bits in the bright tail, so the posterior is ~1e-9 of the prior and
essentially no prior draw lands near ``y``: the NMC bias needs ``M ~ 1e7-1e9`` to
clear, and below that it fakes a plateau.

This script keeps the outer loop and the final average identical, and replaces
only the marginal estimate.  Inner draws come from a defensive mixture

    q(theta') = alpha * prior(theta')
              + (1 - alpha) * (1/N) sum_j N(theta'; theta_j, s^2 F(theta_j)^-1)

whose Gaussian components sit on the Laplace posteriors of the outer samples --
i.e. exactly where ``p(y|theta')`` has its mass -- while the prior component
keeps the weights bounded so the estimator stays consistent.  Then

    log p_hat(y_n|d) = logsumexp_s[ log p(y_n|theta'_s) + log prior(theta'_s)
                                    - log q(theta'_s) ] - log S

The pool of ``S`` draws is shared across all outer samples.  That matters for
cost: ``_magnitude_errors`` renders a 31x31 stamp per (sample, design, band) in
numpy on the CPU, so sigma-evaluation count is the binding budget, and pooling
makes it ``S`` per design instead of ``N x K``.

Every reported EIG comes with the inner-weight effective sample size, which is
the convergence diagnostic: ESS ~ O(pool draws per component) means the marginal
is resolved, ESS ~ 1 means it is not and the number is still biased high.

Usage
-----
    python -m scripts.is_eig_reference <run_id> [<run_id> ...] \
        --n-outer 2000 --pool 100000 --alpha 0.2 --inflate 2.0
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

from bedcosmo.util import auto_seed, load_experiment
from scripts.laplace_eig_reference import _mags, prior_entropy_bits

LOG2 = math.log(2.0)


# --------------------------------------------------------------------------
# prior / model helpers
# --------------------------------------------------------------------------
def prior_log_prob(experiment, theta: torch.Tensor) -> torch.Tensor:
    """log prior(theta), -inf outside the support. theta: (n, d_par)."""
    total = torch.zeros(theta.shape[0], dtype=torch.float64, device=theta.device)
    for k, name in enumerate(experiment.cosmo_params):
        dist = experiment.prior[name]
        x = theta[:, k]
        try:
            support_ok = dist.support.check(x)
        except (AttributeError, NotImplementedError):
            support_ok = torch.ones_like(x, dtype=torch.bool)
        lp = torch.full_like(x, -float("inf"))
        if support_ok.any():
            lp[support_ok] = dist.log_prob(x[support_ok]).to(torch.float64)
        total = total + lp
    return total


def prior_sample(experiment, n: int) -> torch.Tensor:
    return torch.stack(
        [experiment.prior[name].sample((n,)).to(torch.float64) for name in experiment.cosmo_params],
        dim=-1,
    ).to(experiment.device)


def model_mags_sigma(experiment, theta: torch.Tensor, design: torch.Tensor):
    """Model magnitudes and photometric errors. theta: (n, d_par)."""
    d_par = theta.shape[-1]
    mags = _mags(experiment, theta[:, 0], theta[:, 1] if d_par > 1 else None)
    sigma = experiment._magnitude_errors(mags, design)
    return mags, sigma


def gauss_loglik(y: torch.Tensor, mags: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """log N(y; mags, diag(sigma^2)) broadcast over leading dims, summed over bands."""
    z = (y - mags) / sigma
    return (-0.5 * z**2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(dim=-1)


# --------------------------------------------------------------------------
# Fisher information (mean + sigma channels), reused for the proposal covariance
# --------------------------------------------------------------------------
def fisher(experiment, theta: torch.Tensor, design: torch.Tensor, rel_step: float):
    d_par = theta.shape[-1]
    base = _mags(experiment, theta[:, 0], theta[:, 1] if d_par > 1 else None)
    n_filt = base.shape[-1]
    sigma = experiment._magnitude_errors(base, design)

    J = torch.zeros((theta.shape[0], n_filt, d_par), dtype=torch.float64, device=theta.device)
    G = torch.zeros_like(J)
    for k in range(d_par):
        h = rel_step * theta[:, k].abs().clamp(min=1e-8)
        tp, tm = theta.clone(), theta.clone()
        tp[:, k] += h
        tm[:, k] -= h
        mp = _mags(experiment, tp[:, 0], tp[:, 1] if d_par > 1 else None)
        mm = _mags(experiment, tm[:, 0], tm[:, 1] if d_par > 1 else None)
        J[:, :, k] = (mp - mm) / (2 * h).unsqueeze(-1)
        sp = experiment._magnitude_errors(mp, design)
        sm = experiment._magnitude_errors(mm, design)
        G[:, :, k] = (torch.log(sp) - torch.log(sm)) / (2 * h).unsqueeze(-1)

    w = (1.0 / sigma**2).unsqueeze(-1).unsqueeze(-1)
    F = (w * (J.unsqueeze(-1) * J.unsqueeze(-2))).sum(dim=1)
    F = F + 2.0 * (G.unsqueeze(-1) * G.unsqueeze(-2)).sum(dim=1)
    return F, base, sigma


def proposal_covariances(experiment, F: torch.Tensor, inflate: float):
    """s^2 F^-1, capped at the prior covariance and floored away from singular.

    Returns (cov, valid) where invalid rows get no Gaussian component -- the
    alpha*prior term of the mixture covers those parameters instead.
    """
    d_par = F.shape[-1]
    prior_var = torch.tensor(
        [float(experiment.prior[n].variance) for n in experiment.cosmo_params],
        dtype=torch.float64,
        device=F.device,
    )
    eye = torch.eye(d_par, dtype=torch.float64, device=F.device)

    sign, _ = torch.linalg.slogdet(F)
    valid = sign > 0
    cov = (inflate**2) * torch.linalg.pinv(F)
    # Symmetrize, then cap each marginal variance at the prior's (never propose
    # broader than the prior -- that is what the prior component is for).
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    scale = torch.clamp(prior_var / diag.clamp(min=1e-300), max=1.0).sqrt()
    cov = cov * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    cov = cov + 1e-12 * eye * torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)[:, None, None]

    # Drop components that are still not usable.
    chol_ok = torch.zeros_like(valid)
    L = torch.zeros_like(cov)
    for i in range(cov.shape[0]):
        if not bool(valid[i]):
            continue
        try:
            L[i] = torch.linalg.cholesky(cov[i])
            chol_ok[i] = True
        except Exception:
            chol_ok[i] = False
    return cov, L, chol_ok


def mixture_log_prob(theta: torch.Tensor, mu: torch.Tensor, L: torch.Tensor, chunk: int):
    """log (1/J) sum_j N(theta; mu_j, L_j L_j^T).  theta (S,d), mu (J,d), L (J,d,d)."""
    S, d_par = theta.shape
    J = mu.shape[0]
    logdet = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (J,)
    const = -0.5 * d_par * math.log(2 * math.pi)
    out = torch.full((S,), -float("inf"), dtype=torch.float64, device=theta.device)
    for start in range(0, S, chunk):
        th = theta[start : start + chunk]  # (c,d)
        diff = th.unsqueeze(1) - mu.unsqueeze(0)  # (c,J,d)
        sol = torch.linalg.solve_triangular(
            L.unsqueeze(0).expand(diff.shape[0], -1, -1, -1),
            diff.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)  # (c,J,d)
        lp = const - logdet.unsqueeze(0) - 0.5 * (sol**2).sum(-1)  # (c,J)
        out[start : start + chunk] = torch.logsumexp(lp, dim=1) - math.log(J)
    return out


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------
def is_eig_bits(
    experiment,
    designs: torch.Tensor,
    n_outer: int,
    pool: int,
    alpha: float,
    inflate: float,
    rel_step: float,
    sigma_chunk: int,
    weight_chunk: int,
    verbose: bool = False,
):
    """Return per-design (eig_bits, ess_median, ess_p05, frac_ess_below_5)."""
    d_par = len(experiment.cosmo_params)
    n_d = designs.shape[0]
    eig = np.zeros(n_d)
    eig_sem = np.zeros(n_d)
    ess_med = np.zeros(n_d)
    ess_p05 = np.zeros(n_d)
    ess_bad = np.zeros(n_d)

    for di in range(n_d):
        d = designs[di]
        t0 = time.time()

        # ---- step 1: outer draw (identical to the NMC path) ----------------
        theta = prior_sample(experiment, n_outer)
        F, mags_out, sigma_out = fisher(experiment, theta, d, rel_step)
        y = mags_out + sigma_out * torch.randn_like(mags_out)
        log_p_y_given_theta = gauss_loglik(y, mags_out, sigma_out)

        # ---- step 2: pooled defensive-mixture marginal ---------------------
        _, L, ok = proposal_covariances(experiment, F, inflate)
        mu, Lok = theta[ok], L[ok]
        if mu.shape[0] == 0:
            raise RuntimeError("No usable Laplace components; try --inflate or --mean-only path")

        n_pri = int(round(alpha * pool))
        n_gau = pool - n_pri
        pool_pri = prior_sample(experiment, n_pri)
        pick = torch.randint(0, mu.shape[0], (n_gau,), device=mu.device)
        eps = torch.randn((n_gau, d_par), dtype=torch.float64, device=mu.device)
        pool_gau = mu[pick] + torch.einsum("nij,nj->ni", Lok[pick], eps)
        theta_p = torch.cat([pool_pri, pool_gau], dim=0)

        log_pri_p = prior_log_prob(experiment, theta_p)
        in_supp = torch.isfinite(log_pri_p)
        log_mix_p = torch.full_like(log_pri_p, -float("inf"))
        log_mix_p[in_supp] = mixture_log_prob(
            theta_p[in_supp], mu, Lok, chunk=max(1, weight_chunk // max(1, mu.shape[0]))
        )
        log_q_p = torch.logaddexp(
            math.log(alpha) + log_pri_p, math.log(1.0 - alpha) + log_mix_p
        )

        # sigma/mags for the pool: the expensive part, so chunk it
        # Out-of-support rows are never evaluated, so give them a harmless
        # sigma=1 placeholder: a 0 would make gauss_loglik NaN (-inf + inf) and
        # NaN survives the -inf importance weight that is meant to kill the row.
        mags_p = torch.zeros((theta_p.shape[0], mags_out.shape[-1]), dtype=torch.float64)
        sigma_p = torch.ones_like(mags_p)
        idx = torch.nonzero(in_supp).reshape(-1)
        for s in range(0, idx.numel(), sigma_chunk):
            sl = idx[s : s + sigma_chunk]
            m_s, s_s = model_mags_sigma(experiment, theta_p[sl], d)
            mags_p[sl] = m_s
            sigma_p[sl] = s_s

        # ---- weights, per outer sample ------------------------------------
        base_lw = log_pri_p - log_q_p  # (S,)
        base_lw = torch.where(in_supp, base_lw, torch.full_like(base_lw, -float("inf")))
        log_marg = torch.empty(n_outer, dtype=torch.float64)
        ess = torch.empty(n_outer, dtype=torch.float64)
        rows = max(1, weight_chunk // max(1, theta_p.shape[0]))
        for s in range(0, n_outer, rows):
            yy = y[s : s + rows].unsqueeze(1)  # (r,1,B)
            ll = gauss_loglik(yy, mags_p.unsqueeze(0), sigma_p.unsqueeze(0))  # (r,S)
            lw = ll + base_lw.unsqueeze(0)
            lse = torch.logsumexp(lw, dim=1)
            log_marg[s : s + rows] = lse - math.log(theta_p.shape[0])
            ess[s : s + rows] = torch.exp(2 * lse - torch.logsumexp(2 * lw, dim=1))

        terms = (log_p_y_given_theta - log_marg) / LOG2
        eig[di] = float(terms.mean())
        eig_sem[di] = float(terms.std(unbiased=True) / math.sqrt(n_outer))
        ess_med[di] = float(ess.median())
        ess_p05[di] = float(torch.quantile(ess, 0.05))
        ess_bad[di] = float((ess < 5).to(torch.float64).mean())
        if verbose:
            print(f"    design {designs[di].tolist()}  EIG={eig[di]:8.4f}+/-{eig_sem[di]:.4f}  "
                  f"ESS med={ess_med[di]:8.1f} p05={ess_p05[di]:7.1f}  "
                  f"frac(ESS<5)={ess_bad[di]*100:4.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    return eig, eig_sem, ess_med, ess_p05, ess_bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--cosmo-exp", default="num_visits")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-outer", type=int, default=2000, help="N: outer (theta, y) draws")
    p.add_argument("--pool", type=int, default=100000, help="S: shared inner proposal draws")
    p.add_argument("--alpha", type=float, default=0.2, help="Prior weight in the mixture proposal")
    p.add_argument("--inflate", type=float, default=2.0, help="Scale on the Laplace sqrt-cov")
    p.add_argument("--rel-step", type=float, default=1e-4)
    p.add_argument("--sigma-chunk", type=int, default=50000)
    p.add_argument("--weight-chunk", type=int, default=40_000_000)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if "SCRATCH" not in os.environ:
        raise OSError("SCRATCH environment variable is required.")
    mlflow.set_tracking_uri("file:" + os.environ["SCRATCH"] + f"/bedcosmo/{args.cosmo_exp}/mlruns")

    results = {"args": vars(args), "runs": {}}
    for run_id in args.run_ids:
        experiment = load_experiment(run_id, args.cosmo_exp, args.device)
        designs = experiment.designs.to(torch.float64)
        designs_np = designs.detach().cpu().numpy()
        nominal_np = experiment.nominal_design.detach().cpu().numpy().reshape(-1)
        ni = int(np.abs(designs_np - nominal_np).sum(axis=1).argmin())
        h_prior = prior_entropy_bits(experiment)

        print("\n" + "=" * 78)
        print(f"{run_id[:8]}  norm_mode={experiment.norm_mode}  "
              f"H_prior={h_prior:.4f} bits  N={args.n_outer} pool={args.pool} "
              f"alpha={args.alpha} inflate={args.inflate}")
        print("=" * 78)

        reps = []
        for r in range(args.repeats):
            auto_seed(args.seed + 1000 * r)
            if args.repeats > 1:
                print(f"  replica {r + 1}/{args.repeats}")
            out = is_eig_bits(
                experiment, designs, args.n_outer, args.pool, args.alpha, args.inflate,
                args.rel_step, args.sigma_chunk, args.weight_chunk, verbose=not args.quiet,
            )
            reps.append(out)

        eig = np.mean([o[0] for o in reps], axis=0)
        sem = np.sqrt(np.mean([o[1] ** 2 for o in reps], axis=0) / args.repeats)
        ess_med = np.mean([o[2] for o in reps], axis=0)
        ess_p05 = np.mean([o[3] for o in reps], axis=0)
        ess_bad = np.mean([o[4] for o in reps], axis=0)
        best = int(eig.argmax())

        print()
        for i, dd in enumerate(designs_np):
            mark = (" <- nominal" if i == ni else "") + (" <- optimal" if i == best else "")
            print(f"  {dd.tolist()}  EIG = {eig[i]:8.4f} +/- {sem[i]:.4f} bits   "
                  f"ESS med={ess_med[i]:8.1f} p05={ess_p05[i]:7.1f}{mark}")
        print(f"  nominal EIG = {eig[ni]:.4f} +/- {sem[ni]:.4f} bits, "
              f"optimal EIG = {eig[best]:.4f} bits @ {designs_np[best].tolist()}")
        worst = float(ess_bad.max())
        print(f"  diagnostic: max frac(ESS<5) over designs = {worst*100:.2f}%  "
              f"-> {'marginal resolved' if worst < 0.01 else 'STILL BIASED HIGH, raise --pool'}")

        results["runs"][run_id] = {
            "norm_mode": experiment.norm_mode,
            "h_prior_bits": h_prior,
            "designs": designs_np.tolist(),
            "nominal_design": nominal_np.tolist(),
            "nominal_idx": ni,
            "eig_bits": eig.tolist(),
            "eig_sem_bits": sem.tolist(),
            "ess_median": ess_med.tolist(),
            "ess_p05": ess_p05.tolist(),
            "frac_ess_below_5": ess_bad.tolist(),
            "nominal_eig_bits": float(eig[ni]),
            "optimal_eig_bits": float(eig[best]),
            "optimal_design": designs_np[best].tolist(),
        }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
