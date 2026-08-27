#!/usr/bin/env python
"""Read-only audit of design sensitivity for a trained num_visits run.

The audit uses common prior rows and Gaussian noise draws across designs.  It
reports how strongly the likelihood errors and the trained guide respond to the
visit allocation, without changing any run artifact or production code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bedcosmo.util import get_run, load_experiment, load_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outer", type=int, default=48)
    parser.add_argument("--inner", type=int, default=64)
    parser.add_argument(
        "--max-designs",
        type=int,
        default=None,
        help="Evenly subsample the frozen design array for a faster CPU audit.",
    )
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def quantiles(x):
    return np.percentile(np.asarray(x), [0, 5, 16, 50, 84, 95, 100]).tolist()


def main():
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_obj, run_args, _ = get_run(args.run_id, cosmo_exp="num_visits")
    exp = load_experiment(args.run_id, "num_visits", device=args.device)
    designs = exp.designs.detach()
    if args.max_designs is not None and args.max_designs < len(designs):
        keep = np.linspace(0, len(designs) - 1, args.max_designs, dtype=int)
        designs = designs[keep]
    n_designs = len(designs)

    generator = torch.Generator(device=exp.sed_prior.pool.pool.device)
    generator.manual_seed(args.seed)
    rows = exp.sed_prior.sample_unique(args.outer, generator=generator)
    a, log_s, z = exp._empirical_rows_to_physical(rows, (args.outer,))
    with torch.no_grad():
        flux = exp._observed_spectral_flux(z, a=a, log_s=log_s)
        means = exp._calculate_magnitudes(flux)

    # The same galaxy sample is used for every design.  Only the likelihood
    # error model changes with visit allocation.
    errors = []
    for design in designs:
        errors.append(exp._magnitude_errors(means, design).detach().cpu().numpy())
    errors = np.stack(errors, axis=1)  # outer, design, band
    means_np = means.detach().cpu().numpy()
    designs_np = designs.detach().cpu().numpy()

    # Counterfactual for the former bug: undo the empirical forward model's DESI
    # coadd FLUX unit conversion while retaining the corrected zeropoint equation.
    with torch.no_grad():
        missing_desi_unit_means = exp._calculate_magnitudes(flux / 1e-17)
    missing_desi_unit_errors = []
    for design in designs:
        missing_desi_unit_errors.append(
            exp._magnitude_errors(missing_desi_unit_means, design).detach().cpu().numpy()
        )
    missing_desi_unit_errors = np.stack(missing_desi_unit_errors, axis=1)
    missing_desi_unit_means_np = missing_desi_unit_means.detach().cpu().numpy()

    cap = float(exp.mag_err_cap) if exp.mag_err_cap is not None else np.nan
    cap_fraction = np.mean(np.isclose(errors, cap), axis=0)
    median_error = np.median(errors, axis=0)
    log_error_range = np.ptp(np.log(np.clip(errors, 1e-12, None)), axis=1)

    # Build common-random-number observations for every design: the same prior
    # object and standard-normal deviate, scaled by each design's own errors.
    eps = np.random.default_rng(args.seed).standard_normal((args.outer, 1, exp.num_filters))
    observations = means_np[:, None, :] + eps * errors
    design_context = np.broadcast_to(
        designs_np[None, :, :], (args.outer, n_designs, exp.num_filters)
    )
    context = np.concatenate([design_context, observations], axis=-1)
    context_t = torch.as_tensor(context, device=args.device, dtype=torch.float64)

    flow, selected_step = load_model(
        exp,
        "last",
        run_obj,
        run_args,
        args.device,
        global_rank=0,
    )
    flow.eval()
    flat_context = context_t.reshape(-1, context_t.shape[-1])
    target_rows = rows[:, exp.target_indices]
    repeated_targets = target_rows[:, None, :].expand(
        args.outer, n_designs, len(exp.target_indices)
    )
    flat_targets = repeated_targets.reshape(-1, len(exp.target_indices)).clone()
    with torch.enable_grad():
        cross_entropy = -flow(flat_context).log_prob(flat_targets.requires_grad_(True))
    cross_entropy = (
        cross_entropy.detach().reshape(args.outer, n_designs).cpu().numpy() / np.log(2.0)
    )
    with torch.no_grad():
        posterior = flow(flat_context)
        samples = posterior.sample((args.inner,))
        post_mean = samples.mean(0)
        post_std = samples.std(0)
    # Zuko NAF evaluates its scalar Jacobian with autograd even during inference.
    with torch.enable_grad():
        entropy = -posterior.log_prob(samples.detach().requires_grad_(True)).mean(0)
    entropy = entropy.detach().reshape(args.outer, n_designs).cpu().numpy() / np.log(2.0)
    post_mean = post_mean.reshape(args.outer, n_designs, -1).cpu().numpy()
    post_std = post_std.reshape(args.outer, n_designs, -1).cpu().numpy()

    # Conditioner-only intervention: hold the observed magnitudes fixed while
    # changing the design. This directly detects a guide that ignores design.
    fixed_obs = np.broadcast_to(means_np[:, None, :], observations.shape)
    fixed_context = np.concatenate([design_context, fixed_obs], axis=-1)
    fixed_context_t = torch.as_tensor(
        fixed_context.reshape(-1, fixed_context.shape[-1]),
        device=args.device,
        dtype=torch.float64,
    )
    with torch.no_grad():
        fixed_post = flow(fixed_context_t)
        fixed_samples = fixed_post.sample((args.inner,))
        fixed_mean = fixed_samples.mean(0)
    with torch.enable_grad():
        fixed_entropy = -fixed_post.log_prob(
            fixed_samples.detach().requires_grad_(True)
        ).mean(0)
    fixed_entropy = (
        fixed_entropy.detach().reshape(args.outer, n_designs).cpu().numpy() / np.log(2.0)
    )
    fixed_mean = fixed_mean.reshape(args.outer, n_designs, -1).cpu().numpy()

    result = {
        "run_id": args.run_id,
        "step": int(selected_step),
        "cosmo_params": list(exp.cosmo_params),
        "target_params": list(exp.target_params),
        "prior_source": exp.sed_prior.prior_source,
        "n_templates": int(exp.sed_prior.n_templates),
        "n_designs": int(n_designs),
        "outer": args.outer,
        "inner": args.inner,
        "design_min": designs_np.min(0).tolist(),
        "design_max": designs_np.max(0).tolist(),
        "magnitude_quantiles_by_band": {
            band: quantiles(means_np[:, i]) for i, band in enumerate(exp.filters_list)
        },
        "cap_fraction_range_by_band": {
            band: [float(cap_fraction[:, i].min()), float(cap_fraction[:, i].max())]
            for i, band in enumerate(exp.filters_list)
        },
        "one_micro_mag_floor_fraction_range_by_band": {
            band: [
                float(np.isclose(errors[..., i], 1e-6).mean(0).min()),
                float(np.isclose(errors[..., i], 1e-6).mean(0).max()),
            ]
            for i, band in enumerate(exp.filters_list)
        },
        "median_error_range_by_band": {
            band: [float(median_error[:, i].min()), float(median_error[:, i].max())]
            for i, band in enumerate(exp.filters_list)
        },
        "per_object_log_error_design_range_by_band": {
            band: quantiles(log_error_range[:, i]) for i, band in enumerate(exp.filters_list)
        },
        "legacy_missing_desi_unit_counterfactual": {
            "magnitude_quantiles_by_band": {
                band: quantiles(missing_desi_unit_means_np[:, i])
                for i, band in enumerate(exp.filters_list)
            },
            "cap_fraction_range_by_band": {
                band: [
                    float(np.isclose(missing_desi_unit_errors[..., i], cap).mean(0).min()),
                    float(np.isclose(missing_desi_unit_errors[..., i], cap).mean(0).max()),
                ]
                for i, band in enumerate(exp.filters_list)
            },
            "median_error_range_by_band": {
                band: [
                    float(np.median(missing_desi_unit_errors[..., i], axis=0).min()),
                    float(np.median(missing_desi_unit_errors[..., i], axis=0).max()),
                ]
                for i, band in enumerate(exp.filters_list)
            },
            "per_object_log_error_design_range_by_band": {
                band: quantiles(
                    np.ptp(
                        np.log(np.clip(missing_desi_unit_errors[..., i], 1e-12, None)),
                        axis=1,
                    )
                )
                for i, band in enumerate(exp.filters_list)
            },
        },
        "expected_posterior_entropy_bits": {
            "min": float(entropy.mean(0).min()),
            "median": float(np.median(entropy.mean(0))),
            "max": float(entropy.mean(0).max()),
            "range": float(np.ptp(entropy.mean(0))),
            "median_outer_se": float(np.median(entropy.std(0, ddof=1) / np.sqrt(args.outer))),
        },
        "common_random_cross_entropy_bits": {
            "min": float(cross_entropy.mean(0).min()),
            "median": float(np.median(cross_entropy.mean(0))),
            "max": float(cross_entropy.mean(0).max()),
            "range": float(np.ptp(cross_entropy.mean(0))),
            "median_outer_se": float(
                np.median(cross_entropy.std(0, ddof=1) / np.sqrt(args.outer))
            ),
        },
        "posterior_std_design_range": {
            exp.target_params[i]: quantiles(np.ptp(post_std[..., i], axis=1))
            for i in range(post_std.shape[-1])
        },
        "fixed_observation_conditioner_response": {
            "entropy_range_bits": quantiles(np.ptp(fixed_entropy, axis=1)),
            "posterior_mean_range": {
                exp.target_params[i]: quantiles(np.ptp(fixed_mean[..., i], axis=1))
                for i in range(fixed_mean.shape[-1])
            },
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


if __name__ == "__main__":
    main()
