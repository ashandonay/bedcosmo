#!/usr/bin/env python3
"""
Plot the empirical num_visits *central_params* SED via EAZY templates.

Uses NumVisits._observed_spectral_flux (same path as _central_magnitudes_from_dict).

Default output (same tree as ``diagnostic_plots``)::

  $SCRATCH/bedcosmo/num_visits/empirical_prior/diagnostics/central_sed/central_params_sed.png

Example:
  python experiments/num_visits/scripts/visualize_central_sed.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from bedcosmo.num_visits import NumVisits
from bedcosmo.num_visits.sed_prior import PARAMETERIZATION_LOGITS, logits_to_weights_torch
from bedcosmo.num_visits.sed_prior.paths import get_prior_build_dir
from bedcosmo.util import get_experiment_config_path

DIAGNOSTICS_SUBDIR = "central_sed"
OUTPUT_FILENAME = "central_params_sed.png"


def default_output_path(prior_dir: Path | None = None) -> Path:
    """Match ``diagnostic_plots``: ``<prior-dir>/diagnostics/central_sed/``."""
    root = Path(prior_dir).expanduser() if prior_dir is not None else get_prior_build_dir()
    return root / "diagnostics" / DIAGNOSTICS_SUBDIR / OUTPUT_FILENAME


def _load_yaml(name: str) -> dict:
    path = get_experiment_config_path("num_visits", name)
    with open(path) as f:
        return yaml.safe_load(f)


def _central_tensors(experiment: NumVisits, central: dict):
    """Mirror _central_magnitudes_from_dict decoding."""
    K = int(experiment._n_eazy_templates)
    parameterization = getattr(experiment, "_prior_parameterization", "weights")

    if parameterization == "clr":
        clr = torch.tensor(
            [[float(central.get(f"f{k}", 0.0)) for k in range(1, K + 1)]],
            device=experiment.device,
            dtype=torch.float64,
        )
        shifted = clr - torch.amax(clr, dim=-1, keepdim=True)
        exp = torch.exp(shifted)
        a = exp / exp.sum(dim=-1, keepdim=True).clamp_min(1e-300)
    elif parameterization == PARAMETERIZATION_LOGITS:
        eta = torch.tensor(
            [[float(central.get(f"f{k}", 0.0)) for k in range(1, K)]],
            device=experiment.device,
            dtype=torch.float64,
        )
        a = logits_to_weights_torch(eta)
    else:
        a = torch.tensor(
            [[float(central.get(f"a{k}", 0.0)) for k in range(1, K + 1)]],
            device=experiment.device,
            dtype=torch.float64,
        )
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-300)

    log_s = torch.tensor([float(central["log_c_scale"])], device=experiment.device, dtype=torch.float64)
    z = torch.tensor([float(central["z"])], device=experiment.device, dtype=torch.float64)
    return a.squeeze(0), log_s.squeeze(0), z.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior-dir",
        type=Path,
        default=None,
        help=(
            "Prior build directory (default: $SCRATCH/bedcosmo/num_visits/empirical_prior). "
            "Used with --out default."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PNG path (default: <prior-dir>/diagnostics/central_sed/central_params_sed.png)."
        ),
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--prior-pool-size",
        type=int,
        default=4096,
        help="Smaller than training default for faster init (SED plot does not use the pool).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    train = _load_yaml("train_args.yaml")["empirical"]
    central_params = train.get("central_params")
    prior_args = _load_yaml("prior_args_empirical.yaml")
    prior_args["prior_pool_size"] = int(args.prior_pool_size)
    design_args = _load_yaml("design_args.yaml")

    print("central_params override:", central_params or "(KDE prior marginal modes)")
    print("Initializing NumVisits (empirical, transform_input=False)...")
    experiment = NumVisits(
        cosmo_model="empirical",
        prior_args=prior_args,
        design_args=design_args,
        central_params=central_params,
        transform_input=False,
        device=args.device,
        verbose=True,
        global_rank=0,
    )

    # Use merged central_params on the experiment (includes all cosmo_params keys).
    a, log_s, z = _central_tensors(experiment, experiment.central_params)
    flux_aa = experiment._observed_spectral_flux(z.unsqueeze(0), a=a.unsqueeze(0), log_s=log_s.unsqueeze(0))
    flux_aa = flux_aa.squeeze(0).detach().cpu().numpy()
    mags = experiment._calculate_magnitudes(
        experiment._observed_spectral_flux(z.unsqueeze(0), a=a.unsqueeze(0), log_s=log_s.unsqueeze(0))
    ).squeeze(0).detach().cpu().numpy()

    wlen_aa = experiment._wlen_aa_tensor.detach().cpu().numpy()
    a_np = a.detach().cpu().numpy()
    c = (torch.exp(log_s) * a).detach().cpu().numpy()
    z_val = float(z.detach().cpu())
    s_val = float(torch.exp(log_s).detach().cpu())

    # Rest-frame weighted template sum (before 1/(1+z) factor in _observed_spectral_flux)
    wlen_rest = experiment._template_wave_rest.detach().cpu().numpy()
    templates = experiment._template_flux.detach().cpu().numpy()
    one_plus_z = 1.0 + z_val
    wlen_obs_from_rest = wlen_rest * one_plus_z
    flux_rest_mix = np.zeros_like(wlen_rest, dtype=np.float64)
    for k in range(len(a_np)):
        flux_rest_mix += c[k] * templates[k]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    k_idx = np.arange(1, len(a_np) + 1)
    ax.bar(k_idx, a_np, color="steelblue", edgecolor="k", linewidth=0.4)
    ax.set_xlabel("EAZY template index $k$")
    ax.set_ylabel(r"Weight $a_k$")
    param = getattr(experiment, "_prior_parameterization", "?")
    ax.set_title(rf"Template weights $a_k$ ({param}; all $f_k=0$ $\Rightarrow$ uniform)")
    ax.set_xticks(k_idx)

    ax = axes[0, 1]
    ax.bar(["$\\log s$", "$s=\\exp(\\log s)$"], [float(log_s.cpu()), s_val], color=["#c44e52", "#55a868"])
    ax.set_ylabel("value")
    ax.set_title(f"Scale at $z={z_val:.2f}$")

    ax = axes[1, 0]
    ax.plot(wlen_aa, flux_aa, color="k", lw=1.5, label="observed $F_\\lambda$")
    ax.set_xlabel(r"Observed $\lambda$ [$\mathrm{\AA}$]")
    ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
    ax.set_title("Central SED (NumVisits._observed_spectral_flux)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(wlen_aa.min(), wlen_aa.max())

    ax = axes[1, 1]
    ax.plot(wlen_rest, flux_rest_mix, color="#4c72b0", lw=1.2, label=r"$\sum_k c_k T_k(\lambda/(1+z))$")
    ax.set_xlabel(r"Rest-frame $\lambda$ [$\mathrm{\AA}$]")
    ax.set_ylabel(r"linear flux (arb. units)")
    ax.set_title(rf"Rest-frame mixture ($c_k = e^{{\log s}} a_k$)")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "num_visits empirical central_params SED\n"
        f"z={z_val}, log_c_scale={float(log_s.cpu()):.1f}, "
        f"central $f_k=0$ (uniform $a_k$)",
        fontsize=12,
        y=1.02,
    )

    # LSST mags annotation
    mag_text = ", ".join(f"{b}={m:.2f}" for b, m in zip(experiment.filters_list, mags))
    fig.text(0.5, -0.02, f"LSST AB mags (@ central): {mag_text}", ha="center", fontsize=9)

    out = Path(args.out).expanduser() if args.out is not None else default_output_path(args.prior_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
