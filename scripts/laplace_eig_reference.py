"""Laplace/Fisher reference EIG for num_visits runs.

Companion to ``scripts/nmc_eig_reference.py``.  NMC is grid-free but converges to
the true EIG only from above (the ``log p(y)`` mixture estimate is biased low),
and for this likelihood -- photometric errors reaching ~1e-4 mag -- that bias
dies slowly in ``M``.  This script gives an independent estimate with no Monte
Carlo bias at all, so the two brackets the answer.

Method
------
The likelihood is exactly ``y | theta ~ N(m(theta), diag(sigma(theta, d)^2))``.
Where the posterior is narrow compared with the prior (which is exactly the
regime that dominates the EIG here) it is Gaussian to good accuracy with
covariance ``F^-1``, the inverse Fisher information

    F(theta, d) = J^T diag(1 / sigma^2) J,    J = d m / d theta,

so the per-theta information gain is

    IG(theta) = H_prior - 0.5 * log2 det(2 pi e F^-1)

and ``EIG = E_theta[IG]``.

``sigma`` itself depends on ``theta`` (fainter model magnitudes -> larger
photometric error), which is a second information channel.  For a Gaussian with
parameter-dependent variance the full Fisher information is

    F = J^T diag(1/sigma^2) J + 2 * sum_b grad(log sigma_b) grad(log sigma_b)^T

and the second term dominates wherever ``sigma`` is large -- i.e. exactly the
faint region where the mean channel carries nothing.  ``--mean-only`` drops it
to recover the conservative mean-channel-only bound.  Inside the
``mag_err_cap`` plateau ``sigma`` is exactly constant, so its gradient
vanishes and those parameters are correctly counted as uninformative.

``IG`` is clamped at zero where ``F`` is singular or the Fisher-implied
posterior is no narrower than the prior (the Laplace form knows nothing about
the prior, so it cannot be trusted there).

Usage
-----
    python scripts/laplace_eig_reference.py <run_id> [<run_id> ...] \
        --n-samples 200000
"""

from __future__ import annotations

import argparse
import json
import math
import os

import mlflow
import numpy as np
import torch

from bedcosmo.util import auto_seed, load_experiment

LOG2 = math.log(2.0)


def _mags(experiment, z, T):
    kwargs = {"T": T} if "T" in experiment.prior else {}
    return experiment._calculate_magnitudes(experiment._observed_spectral_flux(z, **kwargs))


def prior_entropy_bits(experiment) -> float:
    """Differential entropy of the (independent) parameter prior, in bits."""
    total = 0.0
    for name in experiment.cosmo_params:
        total += float(experiment.prior[name].entropy()) / LOG2
    return total


def laplace_eig_bits(
    experiment, designs, n_samples: int, rel_step: float, chunk: int, mean_only: bool = False
):
    """Return (eig_bits per design, mean clamped fraction per design)."""
    names = list(experiment.cosmo_params)
    h_prior = prior_entropy_bits(experiment)
    d_par = len(names)
    const = 0.5 * d_par * math.log2(2 * math.pi * math.e)

    # Prior draws, shared across designs so design differences are not MC noise.
    theta = torch.stack(
        [experiment.prior[n].sample((n_samples,)).to(torch.float64) for n in names], dim=-1
    ).to(experiment.device)

    eig = np.zeros(designs.shape[0])
    clamped = np.zeros(designs.shape[0])

    for start in range(0, n_samples, chunk):
        th = theta[start : start + chunk]
        nb = th.shape[0]

        # Central-difference Jacobian J[:, band, param] = d mag_band / d param.
        # mags_pert keeps the perturbed magnitudes so d log sigma / d theta can be
        # formed per design below (sigma depends on the design, J does not).
        base = _mags(experiment, th[:, 0], th[:, 1] if d_par > 1 else None)
        n_filt = base.shape[-1]
        J = torch.zeros((nb, n_filt, d_par), dtype=torch.float64, device=experiment.device)
        steps = []
        mags_pert = []
        for k in range(d_par):
            h = rel_step * th[:, k].abs().clamp(min=1e-8)
            tp, tm = th.clone(), th.clone()
            tp[:, k] += h
            tm[:, k] -= h
            mp = _mags(experiment, tp[:, 0], tp[:, 1] if d_par > 1 else None)
            mm = _mags(experiment, tm[:, 0], tm[:, 1] if d_par > 1 else None)
            J[:, :, k] = (mp - mm) / (2 * h).unsqueeze(-1)
            steps.append(h)
            mags_pert.append((mp, mm))

        for di in range(designs.shape[0]):
            d = designs[di]
            sigma = experiment._magnitude_errors(base, d)
            w = (1.0 / sigma**2).unsqueeze(-1).unsqueeze(-1)  # (nb, n_filt, 1, 1)
            outer = J.unsqueeze(-1) * J.unsqueeze(-2)  # (nb, n_filt, d, d)
            F = (w * outer).sum(dim=1)  # (nb, d, d)

            if not mean_only:
                # G[:, band, param] = d log sigma_band / d param
                G = torch.zeros_like(J)
                for k in range(d_par):
                    mp, mm = mags_pert[k]
                    sp = experiment._magnitude_errors(mp, d)
                    sm = experiment._magnitude_errors(mm, d)
                    G[:, :, k] = (torch.log(sp) - torch.log(sm)) / (2 * steps[k]).unsqueeze(-1)
                F = F + 2.0 * (G.unsqueeze(-1) * G.unsqueeze(-2)).sum(dim=1)

            sign, logabsdet = torch.linalg.slogdet(F)
            log2det = torch.where(
                sign > 0, logabsdet / LOG2, torch.full_like(logabsdet, -float("inf"))
            )
            h_post = const - 0.5 * log2det
            ig = torch.clamp(h_prior - h_post, min=0.0)
            eig[di] += float(ig.sum())
            clamped[di] += float((h_prior - h_post <= 0).sum())

    return eig / n_samples, clamped / n_samples, h_prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_ids", nargs="+")
    p.add_argument("--cosmo-exp", default="num_visits")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=200000)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--rel-step", type=float, default=1e-4, help="Relative FD step for J")
    p.add_argument("--mean-only", action="store_true",
                   help="Drop the d log sigma / d theta Fisher channel (conservative bound)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if "SCRATCH" not in os.environ:
        raise OSError("SCRATCH environment variable is required.")
    mlflow.set_tracking_uri("file:" + os.environ["SCRATCH"] + f"/bedcosmo/{args.cosmo_exp}/mlruns")

    results = {"args": vars(args), "runs": {}}
    for run_id in args.run_ids:
        auto_seed(args.seed)
        experiment = load_experiment(run_id, args.cosmo_exp, args.device)
        designs = experiment.designs.to(torch.float64)
        designs_np = designs.detach().cpu().numpy()
        nominal_np = experiment.nominal_design.detach().cpu().numpy().reshape(-1)
        ni = int(np.abs(designs_np - nominal_np).sum(axis=1).argmin())

        eig, clamped, h_prior = laplace_eig_bits(
            experiment, designs, args.n_samples, args.rel_step, args.chunk, args.mean_only
        )
        best = int(eig.argmax())

        print("\n" + "=" * 78)
        print(f"{run_id[:8]}  norm_mode={experiment.norm_mode}  "
              f"H_prior={h_prior:.4f} bits  N={args.n_samples}  "
              f"channel={'mean-only' if args.mean_only else 'mean+sigma'}")
        print("=" * 78)
        for i, dd in enumerate(designs_np):
            mark = (" <- nominal" if i == ni else "") + (" <- optimal" if i == best else "")
            print(f"  {dd.tolist()}  EIG = {eig[i]:8.4f} bits   "
                  f"(uninformative prior mass {clamped[i]*100:5.1f}%){mark}")
        print(f"  nominal EIG = {eig[ni]:.4f} bits, optimal EIG = {eig[best]:.4f} bits "
              f"@ {designs_np[best].tolist()}")

        results["runs"][run_id] = {
            "norm_mode": experiment.norm_mode,
            "h_prior_bits": h_prior,
            "designs": designs_np.tolist(),
            "eig_bits": eig.tolist(),
            "clamped_fraction": clamped.tolist(),
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
