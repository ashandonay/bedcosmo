#!/usr/bin/env python
"""Quantify the current versus corrected LSST zeropoint conversion.

This is a read-only diagnostic.  It holds the DESI physical-unit correction
fixed and changes only the count-rate-to-magnitude equation.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from bedcosmo.num_visits.experiment import s0
from bedcosmo.util import load_experiment


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def photon_count_rate(exp, flux_aa):
    """Reproduce the filter integration in NumVisits._calculate_magnitudes."""
    n_wlen = exp._wlen_aa_tensor.shape[0]
    batch_shape = flux_aa.shape[:-1]
    flux_flat = flux_aa.reshape(-1, n_wlen)
    integrand = (
        flux_flat.unsqueeze(1)
        * exp._transmission_tensor.unsqueeze(0)
        * exp._wlen_over_hc_tensor.unsqueeze(0).unsqueeze(0)
    )
    photon_flux = torch.trapezoid(integrand, exp._wlen_aa_tensor, dim=-1)
    collecting_area_cm2 = (319 / 9.6) * 1e4
    minimum = torch.finfo(photon_flux.dtype).tiny * 1e10
    return torch.clamp(photon_flux * collecting_area_cm2, min=minimum).reshape(
        *batch_shape, exp.num_filters
    )


def q(values):
    return {
        name: float(value)
        for name, value in zip(
            ("p05", "p16", "median", "p84", "p95"),
            np.percentile(np.asarray(values), (5, 16, 50, 84, 95)),
        )
    }


def main():
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    exp = load_experiment(args.run_id, "num_visits", device=args.device)

    generator = torch.Generator(device=exp.sed_prior.pool.pool.device)
    generator.manual_seed(args.seed)
    rows = exp.sed_prior.sample_unique(args.samples, generator=generator)
    a, log_s, z = exp._empirical_rows_to_physical(rows, (args.samples,))

    with torch.no_grad():
        # The empirical forward model now includes the DESI FLUX unit conversion.
        # Reconstruct the legacy zeropoint equation explicitly for comparison.
        flux = exp._observed_spectral_flux(z, a=a, log_s=log_s)
        counts = photon_count_rate(exp, flux)
        zeropoints = torch.as_tensor(
            [s0[band] for band in exp.filters_list],
            device=counts.device,
            dtype=counts.dtype,
        )
        old_mags = 24.0 - 2.5 * torch.log10(counts / zeropoints)
        corrected_mags = exp._calculate_magnitudes(flux)

    old_np = old_mags.cpu().numpy()
    corrected_np = corrected_mags.cpu().numpy()
    magnitude_bias = old_np - corrected_np

    adjacent_colors = [
        f"{left}-{right}"
        for left, right in zip(exp.filters_list[:-1], exp.filters_list[1:])
    ]
    old_colors = old_np[:, :-1] - old_np[:, 1:]
    corrected_colors = corrected_np[:, :-1] - corrected_np[:, 1:]
    color_bias = old_colors - corrected_colors

    nominal = exp.nominal_design
    old_nominal_error = exp._magnitude_errors(old_mags, nominal).cpu().numpy()
    corrected_nominal_error = exp._magnitude_errors(corrected_mags, nominal).cpu().numpy()
    nominal_ratio = old_nominal_error / corrected_nominal_error

    old_all = []
    corrected_all = []
    for design in exp.designs:
        old_all.append(exp._magnitude_errors(old_mags, design).cpu().numpy())
        corrected_all.append(exp._magnitude_errors(corrected_mags, design).cpu().numpy())
    old_all = np.stack(old_all, axis=1)
    corrected_all = np.stack(corrected_all, axis=1)
    all_ratio = old_all / corrected_all

    result = {
        "run_id": args.run_id,
        "samples": args.samples,
        "n_designs": len(exp.designs),
        "comparison": "old versus corrected zeropoint; both include DESI flux x 1e-17",
        "old_minus_corrected_magnitude_by_band": {
            band: float(np.median(magnitude_bias[:, index]))
            for index, band in enumerate(exp.filters_list)
        },
        "old_minus_corrected_adjacent_color": {
            color: float(np.median(color_bias[:, index]))
            for index, color in enumerate(adjacent_colors)
        },
        "nominal_design_magnitude_error": {
            band: {
                "old": q(old_nominal_error[:, index]),
                "corrected": q(corrected_nominal_error[:, index]),
                "old_over_corrected": q(nominal_ratio[:, index]),
            }
            for index, band in enumerate(exp.filters_list)
        },
        "all_designs_old_over_corrected_error": {
            band: q(all_ratio[..., index].reshape(-1))
            for index, band in enumerate(exp.filters_list)
        },
        "all_designs_error_cap_fraction": {
            band: {
                "old": float(np.isclose(old_all[..., index], exp.mag_err_cap).mean()),
                "corrected": float(
                    np.isclose(corrected_all[..., index], exp.mag_err_cap).mean()
                ),
            }
            for index, band in enumerate(exp.filters_list)
        },
        "all_designs_one_micro_mag_floor_fraction": {
            band: {
                "old": float(np.isclose(old_all[..., index], 1e-6).mean()),
                "corrected": float(
                    np.isclose(corrected_all[..., index], 1e-6).mean()
                ),
            }
            for index, band in enumerate(exp.filters_list)
        },
    }

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()
