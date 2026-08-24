#!/usr/bin/env python3
"""One-off cross-check: nf_loss joint EIG vs sample-based k-NN EIG.

Avoids importing bedcosmo.evaluate (and plotting/IPython) so it runs in minimal envs.

Checks on the nominal design:
  1. nf_loss joint EIG vs explicit E[log q - log p] — must match exactly
  2. Full joint EIG via entropy identity (primary sample-based cross-check):
       I(theta; y|d) = H[p(theta)] - E_y[H(q(theta|y,d))]
     Uses -E[log p] on prior samples and -E[log q] on guide samples (same
     densities as nf_loss). This should track nf_loss closely.
  3. Optional k-NN full joint (diagnostic only; biased in 14D)
  4. Optional low-D marginal subsets (--extra-subsets) via k-NN in physical space

Example:
  export SCRATCH=/path/to/scratch
  python scripts/compare_marginal_eig.py \\
      --run-id <run_id> --cosmo-exp num_visits --n-particles 500 --pool-joint-particles
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Literal

import mlflow
import numpy as np
import torch

from bedcosmo.entropy import knn_entropy
from bedcosmo.pyro_oed_src import LikelihoodDataset, nf_loss
from bedcosmo.util import (
    auto_seed,
    get_checkpoint,
    get_runs_data,
    init_experiment,
    load_model,
    parse_param_subsets,
)

KnnSpace = Literal["physical", "unconstrained"]
PosteriorMode = Literal["per_y", "pooled"]


def subset_id(subset):
    return "+".join(subset)


def normalize_marginal_subsets(experiment, subsets):
    if not subsets:
        return []
    if all(isinstance(s, str) for s in subsets):
        subsets = [subsets]
    cosmo_params = experiment.cosmo_params
    normalized = []
    for subset in subsets:
        valid = [p for p in subset if p in cosmo_params]
        missing = [p for p in subset if p not in cosmo_params]
        if missing:
            print(
                f"  Warning: subset {subset} references unknown parameters {missing}; "
                f"ignoring them. Available: {cosmo_params}"
            )
        if valid:
            normalized.append(valid)
    return normalized


@dataclass
class CrossCheckContext:
    experiment: object
    run_obj: object
    run_args: dict
    device: str
    seed: int
    marginal_outer_y: int = 8
    marginal_inner_samples: int = 200
    marginal_knn_k: int = 3
    nf_transform_output: bool = True
    particle_batch_size: int | None = None


def knn_space_for_full_joint(ctx: CrossCheckContext) -> KnnSpace:
    """Flow/nf_loss live in unconstrained space when transform_input is enabled."""
    if getattr(ctx.experiment, "transform_input", False):
        return "unconstrained"
    return "physical"


def load_cross_check_context(
    run_id: str,
    cosmo_exp: str,
    *,
    device: str,
    seed: int,
    marginal_outer_y: int,
    marginal_inner_samples: int,
    marginal_knn_k: int,
) -> CrossCheckContext:
    mlflow.set_tracking_uri("file:" + os.environ["SCRATCH"] + f"/bedcosmo/{cosmo_exp}/mlruns")
    run_data_list, _, _ = get_runs_data(run_ids=run_id, cosmo_exp=cosmo_exp)
    if not run_data_list:
        raise ValueError(f"Run {run_id} not found in experiment {cosmo_exp}")
    run_data = run_data_list[0]
    run_obj = run_data["run_obj"]
    run_args = run_data["params"]
    save_path = (
        f"{os.environ['SCRATCH']}/bedcosmo/{cosmo_exp}/mlruns/"
        f"{run_data['exp_id']}/{run_id}/artifacts"
    )

    eval_checkpoint = None
    if run_args.get("transform_input", False):
        eval_checkpoint, _ = get_checkpoint(
            "last",
            f"{save_path}/checkpoints",
            device,
            0,
            run_args["total_steps"],
        )
        if "bijector_state" not in eval_checkpoint:
            raise RuntimeError(
                f"Run {run_id} has transform_input=True but checkpoint lacks bijector_state"
            )

    experiment = init_experiment(
        run_obj, run_args, device=device, global_rank=0, checkpoint=eval_checkpoint
    )
    return CrossCheckContext(
        experiment=experiment,
        run_obj=run_obj,
        run_args=run_args,
        device=device,
        seed=seed,
        marginal_outer_y=marginal_outer_y,
        marginal_inner_samples=marginal_inner_samples,
        marginal_knn_k=marginal_knn_k,
    )


def marginal_prior_sample_count(marginal_inner_samples: int, marginal_outer_y: int) -> int:
    return max(int(marginal_inner_samples) * int(marginal_outer_y), 4096)


def marginal_prior_physical_samples(ctx: CrossCheckContext, num_samples: int) -> np.ndarray:
    exp = ctx.experiment
    pool = getattr(exp, "prior_pool", None)
    n_want = int(num_samples)
    if pool is not None:
        from bedcosmo.num_visits.sed_prior.prior_sampler import sample_prior_pool_unique

        n_draw = min(n_want, int(pool.pool.shape[0]))
        gen = torch.Generator(device=pool.pool.device)
        gen.manual_seed(int(ctx.seed))
        rows = sample_prior_pool_unique(pool, n_draw, generator=gen)
        params = exp._prior_rows_to_param_dict(rows, (n_draw,))
        param_samples = torch.stack(
            [params[k].squeeze(-1) for k in exp.cosmo_params], dim=-1
        )
        exp.apply_multipliers(param_samples)
        param_samples = exp._sanitize_physical_samples(param_samples)
        return param_samples.detach().cpu().numpy()
    return exp.get_prior_samples(num_samples=n_want).samples


def physical_to_knn_coords(
    ctx: CrossCheckContext, physical: np.ndarray, space: KnnSpace
) -> np.ndarray:
    if space == "physical":
        return physical
    exp = ctx.experiment
    if not getattr(exp, "transform_input", False):
        return physical
    t = torch.tensor(physical, dtype=torch.float64, device=exp.device)
    return exp.params_to_unconstrained(t).detach().cpu().numpy()


def flow_samples_to_knn_coords(
    ctx: CrossCheckContext, samples: torch.Tensor, space: KnnSpace
) -> np.ndarray:
    """Map flow samples to the coordinate system used for k-NN."""
    if space == "unconstrained":
        return samples.detach().cpu().numpy()
    exp = ctx.experiment
    if getattr(exp, "transform_input", False) and ctx.nf_transform_output:
        samples = exp.params_from_unconstrained(samples)
        samples = exp._sanitize_physical_samples(samples)
    exp.apply_multipliers(samples)
    return samples.detach().cpu().numpy()


def marginal_posterior_entropy(
    ctx: CrossCheckContext,
    flow_model,
    designs,
    subset_ids,
    subset_idx,
    *,
    n_outer_y: int | None = None,
    n_inner_samples: int | None = None,
    space: KnnSpace = "physical",
    mode: PosteriorMode = "per_y",
    subsample_to: int | None = None,
):
    exp = ctx.experiment
    device_obj = torch.device(ctx.device)
    designs = designs.to(device_obj)
    n_designs = designs.shape[0]
    M = n_outer_y if n_outer_y is not None else ctx.marginal_outer_y
    K = n_inner_samples if n_inner_samples is not None else ctx.marginal_inner_samples
    k = ctx.marginal_knn_k

    _, context = LikelihoodDataset(
        experiment=exp,
        n_particles_per_device=M,
        device=ctx.device,
        evaluation=False,
        designs=designs,
    )[0]
    context = context.to(device_obj)

    out = {sid: np.zeros(n_designs) for sid in subset_ids}
    with torch.inference_mode():
        for j in range(n_designs):
            ctx_j = context[:, j, :]
            inner_np = flow_samples_to_knn_coords(
                ctx, flow_model(ctx_j).sample((K,)), space=space
            )
            for sid, idx in zip(subset_ids, subset_idx):
                cols = inner_np[..., idx]
                if mode == "pooled":
                    pooled = cols.reshape(-1, cols.shape[-1])
                    if subsample_to is not None and pooled.shape[0] > subsample_to:
                        rng = np.random.default_rng(int(ctx.seed) + j)
                        idx_rows = rng.choice(pooled.shape[0], subsample_to, replace=False)
                        pooled = pooled[idx_rows]
                    out[sid][j] = knn_entropy(pooled, k=k)
                else:
                    H_per_y = [knn_entropy(cols[:, m, :], k=k) for m in range(M)]
                    out[sid][j] = float(np.mean(H_per_y))
    return out


def knn_eig_bits(
    ctx: CrossCheckContext,
    flow_model,
    designs,
    subsets,
    *,
    n_outer_y: int | None = None,
    n_inner_samples: int | None = None,
    space: KnnSpace = "physical",
    mode: PosteriorMode = "per_y",
    n_prior: int | None = None,
    match_prior_post_count: bool = False,
):
    cosmo_params = ctx.experiment.cosmo_params
    subset_ids = [subset_id(s) for s in subsets]
    subset_idx = [[cosmo_params.index(p) for p in s] for s in subsets]
    outer = n_outer_y if n_outer_y is not None else ctx.marginal_outer_y
    inner = n_inner_samples if n_inner_samples is not None else ctx.marginal_inner_samples
    if n_prior is None:
        n_prior = marginal_prior_sample_count(inner, outer)
    prior_phys = marginal_prior_physical_samples(ctx, n_prior)
    n_prior_eff = int(prior_phys.shape[0])
    prior_coords = physical_to_knn_coords(ctx, prior_phys, space)
    prior_H = {
        sid: knn_entropy(prior_coords[:, idx], k=ctx.marginal_knn_k)
        for sid, idx in zip(subset_ids, subset_idx)
    }
    post_H = marginal_posterior_entropy(
        ctx,
        flow_model,
        designs,
        subset_ids,
        subset_idx,
        n_outer_y=outer,
        n_inner_samples=inner,
        space=space,
        mode=mode,
        subsample_to=n_prior_eff if match_prior_post_count else None,
    )
    eig = {sid: prior_H[sid] - post_H[sid] for sid in subset_ids}
    return eig, prior_H, post_H, n_prior_eff, outer * inner


def joint_eig_path_nats(experiment, flow_model, samples, context, log_probs):
    batch_shape = samples.shape[:-1]
    flat_theta = samples.reshape(-1, samples.shape[-1])
    flat_ctx = context.reshape(-1, context.shape[-1])
    if experiment.transform_input:
        y_flat = experiment.params_to_unconstrained(flat_theta)
    else:
        y_flat = flat_theta
    with torch.no_grad():
        log_q = flow_model(flat_ctx).log_prob(y_flat).reshape(batch_shape)
    if log_probs is not None and "joint" in log_probs:
        log_p = log_probs["joint"]
    else:
        log_p = sum(log_probs[l] for l in experiment.cosmo_params)
    return (log_q - log_p).mean(dim=0)


def prior_log_prob_joint(ctx: CrossCheckContext, theta_physical: torch.Tensor) -> torch.Tensor:
    """log p(theta) in NF input coordinates; theta shape (..., n_params)."""
    exp = ctx.experiment
    batch_shape = theta_physical.shape[:-1]
    flat = theta_physical.reshape(-1, theta_physical.shape[-1]).to(
        device=theta_physical.device, dtype=torch.float64
    )

    # prior_flow and joint-Gaussianizer paths live in LikelihoodDataset.
    if getattr(exp, "prior_flow", None) is not None:
        ds = _eval_likelihood_dataset(ctx)
        return ds._compute_prior_log_probs(theta_physical, trace=None)["joint"]

    if (
        getattr(exp, "transform_input", False)
        and getattr(exp, "param_bijector", None) is not None
        and exp.param_bijector.uses_joint_gaussianizer()
    ):
        ds = _eval_likelihood_dataset(ctx)
        return ds._compute_prior_log_probs(theta_physical, trace=None)["joint"]

    # Analytic Pyro priors (bb, bbt): evaluate at arbitrary theta.
    prior = getattr(exp, "prior", None)
    if isinstance(prior, dict) and prior:
        log_joint = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.float64)
        for i, name in enumerate(exp.cosmo_params):
            log_joint = log_joint + prior[name].log_prob(flat[:, i])
        if getattr(exp, "transform_input", False):
            log_det = exp._transform_log_abs_det_flat(exp.param_bijector, flat)
            log_joint = log_joint - log_det
        return log_joint.reshape(batch_shape)

    raise ValueError(
        "prior_log_prob_joint: unsupported prior configuration "
        f"(cosmo_model={getattr(exp, 'cosmo_model', None)})"
    )


def _eval_likelihood_dataset(ctx: CrossCheckContext) -> LikelihoodDataset:
    return LikelihoodDataset(
        experiment=ctx.experiment,
        n_particles_per_device=1,
        device=ctx.device,
        evaluation=True,
        designs=ctx.experiment.nominal_design.unsqueeze(0).to(ctx.device),
    )


def flow_log_prob_samples(flow_model, cond: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """log q(samples | cond). Zuko flows need grad-enabled inputs for Jacobian terms."""
    with torch.enable_grad():
        # clone() so samples are not inference-mode tensors (from no_grad sampling)
        x = samples.clone().requires_grad_(True)
        return flow_model(cond).log_prob(x)


def full_joint_eig_density_bits(
    ctx: CrossCheckContext,
    flow_model,
    designs,
    *,
    n_outer_y: int,
    n_inner_k: int,
    design_idx: int = 0,
):
    """H[p] - E_y[H(q|y)] via resubstitution using flow/prior log densities (bits)."""
    device = torch.device(ctx.device)
    n_prior = marginal_prior_sample_count(n_inner_k, n_outer_y)
    prior_phys = marginal_prior_physical_samples(ctx, n_prior)
    prior_theta = torch.tensor(prior_phys, dtype=torch.float64, device=device)
    log_p = prior_log_prob_joint(ctx, prior_theta)
    H_p_nats = -log_p.mean()

    _, context = LikelihoodDataset(
        experiment=ctx.experiment,
        n_particles_per_device=n_outer_y,
        device=ctx.device,
        evaluation=False,
        designs=designs,
    )[0]
    context = context.to(device)
    ctx_y = context[:, design_idx, :]

    H_q_per_y = []
    for m in range(n_outer_y):
        cond = ctx_y[m : m + 1]
        with torch.no_grad():
            u = flow_model(cond).sample((n_inner_k,))
        log_q = flow_log_prob_samples(flow_model, cond, u)
        H_q_per_y.append(-log_q.detach().mean())
    H_q_avg_nats = torch.stack(H_q_per_y).mean()
    eig_nats = H_p_nats - H_q_avg_nats
    ln2 = float(np.log(2))
    return (
        float(eig_nats / ln2),
        float(H_p_nats / ln2),
        float(H_q_avg_nats / ln2),
        int(prior_theta.shape[0]),
        n_outer_y,
        n_inner_k,
    )


def compare_marginal_eig_estimators(
    ctx: CrossCheckContext,
    flow_model,
    *,
    n_particles: int,
    designs=None,
    subsets=None,
    pool_joint_particles: bool = False,
):
    if designs is None:
        designs = ctx.experiment.nominal_design.unsqueeze(0)
    designs = designs.to(ctx.device)
    cosmo_params = ctx.experiment.cosmo_params
    full_subset = [list(cosmo_params)]
    extra_subsets = normalize_marginal_subsets(ctx.experiment, subsets) if subsets else []

    samples, context, log_probs = LikelihoodDataset(
        experiment=ctx.experiment,
        n_particles_per_device=n_particles,
        device=ctx.device,
        evaluation=True,
        designs=designs,
        particle_batch_size=ctx.particle_batch_size,
    )[0]
    device = torch.device(ctx.device)
    samples = samples.to(device)
    context = context.to(device)
    if log_probs is not None:
        log_probs = {k: v.to(device) for k, v in log_probs.items()}

    with torch.no_grad():
        _, nf_nats = nf_loss(
            samples=samples,
            context=context,
            guide=flow_model,
            experiment=ctx.experiment,
            rank=0,
            log_probs=log_probs,
            evaluation=True,
            chunk_size=max(n_particles // 10, 1),
        )
        path_nats = joint_eig_path_nats(
            ctx.experiment, flow_model, samples, context, log_probs
        )

    nf_bits = (nf_nats.detach().cpu().numpy() / np.log(2)).reshape(-1)
    path_bits = (path_nats.detach().cpu().numpy() / np.log(2)).reshape(-1)
    max_path_diff = float(np.max(np.abs(nf_bits - path_bits)))

    outer_y = n_particles if pool_joint_particles else ctx.marginal_outer_y
    inner_k = ctx.marginal_inner_samples
    n_post = outer_y * inner_k
    full_id = subset_id(cosmo_params)

    density_eig, density_Hp, density_Hq, n_prior_d, _, _ = full_joint_eig_density_bits(
        ctx,
        flow_model,
        designs,
        n_outer_y=outer_y,
        n_inner_k=inner_k,
    )

    full_space = knn_space_for_full_joint(ctx)
    full_eig, full_prior_H, full_post_H, n_prior_eff, _ = knn_eig_bits(
        ctx,
        flow_model,
        designs,
        full_subset,
        n_outer_y=outer_y,
        n_inner_samples=inner_k,
        space=full_space,
        mode="pooled",
        n_prior=n_post,
        match_prior_post_count=True,
    )

    subset_eig = {}
    subset_prior_H = {}
    subset_post_H = {}
    if extra_subsets:
        subset_eig, subset_prior_H, subset_post_H, _, _ = knn_eig_bits(
            ctx,
            flow_model,
            designs,
            extra_subsets,
            n_outer_y=outer_y,
            n_inner_samples=inner_k,
            space="physical",
            mode="per_y",
        )

    print(
        f"\nJoint EIG estimator cross-check "
        f"({n_particles} likelihood particles, {designs.shape[0]} design(s)):"
    )
    print(f"  nf_loss vs path log-prob: max |Δ| = {max_path_diff:.3e} bits")
    for j in range(designs.shape[0]):
        label = "nominal" if designs.shape[0] == 1 else f"design {j}"
        print(
            f"  [{label}] joint nf_loss = {nf_bits[j]:.4f} bits, "
            f"path = {path_bits[j]:.4f} bits"
        )
        print(
            f"           density joint = {density_eig:.4f} bits "
            f"(H[p]={density_Hp:.3f}, E_y[H(q|y)]={density_Hq:.3f}, "
            f"n_prior={n_prior_d}, M={outer_y}, K={inner_k})"
        )
        print(
            f"           k-NN full joint = {float(full_eig[full_id][j]):.4f} bits "
            f"({full_space}, pooled, n={n_prior_eff}; "
            f"high-D k-NN bias — diagnostic only)"
        )
        for sid in subset_eig:
            print(
                f"           k-NN marginal [{sid}] = {float(subset_eig[sid][j]):.4f} bits "
                f"(physical, per-y avg, M={outer_y}, K={inner_k})"
            )

    return {
        "nf_bits": nf_bits,
        "path_bits": path_bits,
        "density_joint_bits": density_eig,
        "full_knn_bits": full_eig,
        "subset_knn_bits": subset_eig,
        "max_path_diff_bits": max_path_diff,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-check nf_loss joint EIG vs k-NN (full joint + optional marginals)"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cosmo-exp", default="num_visits")
    parser.add_argument("--eval-step", default="last")
    parser.add_argument("--n-particles", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--extra-subsets",
        type=parse_param_subsets,
        default=None,
        help="Optional low-D marginals in physical space (e.g. 'log_c_scale,z')",
    )
    parser.add_argument("--marginal-outer-y", type=int, default=8)
    parser.add_argument("--marginal-inner-samples", type=int, default=200)
    parser.add_argument("--marginal-knn-k", type=int, default=3)
    parser.add_argument(
        "--pool-joint-particles",
        action="store_true",
        help="Use n_particles as outer-y count for k-NN MC (else marginal-outer-y)",
    )
    args = parser.parse_args()

    if "SCRATCH" not in os.environ:
        raise SystemExit("SCRATCH must be set")

    auto_seed(args.seed)
    ctx = load_cross_check_context(
        args.run_id,
        args.cosmo_exp,
        device=args.device,
        seed=args.seed,
        marginal_outer_y=args.marginal_outer_y,
        marginal_inner_samples=args.marginal_inner_samples,
        marginal_knn_k=args.marginal_knn_k,
    )
    flow, _ = load_model(
        ctx.experiment, args.eval_step, ctx.run_obj, ctx.run_args, ctx.device, 0
    )
    compare_marginal_eig_estimators(
        ctx,
        flow,
        n_particles=args.n_particles,
        subsets=args.extra_subsets,
        pool_joint_particles=args.pool_joint_particles,
    )


if __name__ == "__main__":
    main()
