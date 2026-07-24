"""What L_lambda(lambda_rest) actually looks like, and how z enters.

The rest-frame spectral luminosity the bb / bbt model builds is

    L_lambda(lambda_rest) = 4 pi R_eff^2 * pi * B_lambda(lambda_rest, T)
                          = 4 pi R_eff^2 * pi * (2hc^2/lambda_rest^5)
                                                / (exp(hc/(lambda_rest k T)) - 1)

with T the REST-FRAME temperature (the sampled parameter) and R_eff back-solved
so every curve encloses the same area, L_bol = 1e9 L_sun.

The point of the figure: this curve lives entirely in the rest frame and knows
nothing about redshift. Redshift does not deform it. It only decides WHERE the
instrument samples it -- LSST's fixed 3199-10989 AA window maps back to rest
wavelengths 3199/(1+z) .. 10989/(1+z), so a higher z slides the sampling window
blueward across a curve that never moved.

  (a) L_lambda(lambda_rest) for three temperatures, all enclosing 1e9 L_sun.
      Wien peaks at 2.898e7/T AA. Shaded bands are the rest-frame slice LSST
      actually sees at z = 0, 1, 2.
  (b) The same three curves after transport to the observer at z=1 -- the code's
      f_lambda -- against the real filter curves. This is what the likelihood
      integrates over.

Everything is computed with the live forward model's own functions.

Usage:
  python experiments/num_visits/scripts/plot_rest_frame_sed.py [--out <png>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from astropy import units as u
from astropy.constants import sigma_sb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _paths import plot_path  # noqa: E402

from bedcosmo.num_visits import NumVisits  # noqa: E402
from bedcosmo.util import get_experiment_config_path  # noqa: E402

C_COOL, C_MID, C_WARM = "#3B7DD8", "#B0640F", "#C2334D"
INK, MUTED, GRID = "#22252A", "#6B7280", "#D8DBE0"
L_SUN = 3.826e33
L_BOL = 1e9 * L_SUN
TEMPS = [(4000.0, C_COOL), (5000.0, C_MID), (10000.0, C_WARM)]


def build_experiment(device: str = "cpu") -> NumVisits:
    design_args = yaml.safe_load(open(get_experiment_config_path("num_visits", "design_args.yaml")))
    prior_args = yaml.safe_load(
        open(get_experiment_config_path("num_visits", "prior_args_bbt.yaml"))
    )
    exp = NumVisits(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model="bbt",
        norm_mode="bolometric",
        device=device,
        verbose=False,
    )
    exp.init_prior(parameters=prior_args["parameters"], cosmo_model="bbt")
    keys = ("input_type", "step", "lower", "upper", "sum_lower", "sum_upper", "labels")
    exp.init_designs(**{k: v for k, v in design_args.items() if k in keys})
    return exp


def rest_frame_luminosity(exp: NumVisits, wl_aa: np.ndarray, T: float) -> tuple[np.ndarray, float]:
    """L_lambda(lambda_rest) [erg/s/AA] and its enclosed area [L_sun]."""
    sig = sigma_sb.to(u.erg / (u.s * u.cm**2 * u.K**4)).value
    wl_cm = torch.tensor(wl_aa * 1e-8, dtype=torch.float64)
    F = exp._blackbody_flux(wl_cm, torch.tensor(T, dtype=torch.float64)).numpy()  # pi*B per AA
    R_eff = np.sqrt((L_BOL / (4 * np.pi * sig)) / T**4)
    L_lam = 4 * np.pi * R_eff**2 * F
    return L_lam, float(np.trapz(L_lam, wl_aa) / L_SUN)


def panel_a(exp, ax) -> None:
    wl = np.logspace(2.7, 5.0, 4000)  # 500 AA .. 100000 AA, past both tails

    for T, color in TEMPS:
        L_lam, _ = rest_frame_luminosity(exp, wl, T)
        peak = 2.898e7 / T
        ax.plot(wl, L_lam, color=color, lw=2, label=f"T={T:.0f} K — peak {peak:.0f}$\\,\\AA$")
        ax.plot(peak, L_lam[np.argmin(np.abs(wl - peak))], "v", color=color, ms=8)

    # The rest-frame slice LSST samples, for three redshifts, drawn as a rug of
    # bars in the empty lower region -- overlapping shaded spans are unreadable.
    # The curves are fixed; only this window moves.
    for i, z in enumerate((0.0, 1.0, 2.0)):
        lo, hi = 3199.0 / (1 + z), 10989.0 / (1 + z)
        y = 2.2e34 * (3.0**i)
        ax.plot([lo, hi], [y, y], color=INK, lw=4, solid_capstyle="butt", alpha=0.75)
        ax.text(
            hi * 1.2,
            y,
            f"LSST @ z={z:.0f}  ({lo:.0f}–{hi:.0f}$\\,\\AA$)",
            fontsize=7.5,
            color=INK,
            va="center",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(8e33, 3e39)
    ax.set_xlabel(r"REST wavelength $\lambda_{\rm rest}$ [$\AA$]")
    ax.set_ylabel(r"$L_\lambda(\lambda_{\rm rest})$ [erg/s/$\AA$]")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(
        "(a) $L_\\lambda = 4\\pi R_{\\rm eff}^2\\,\\pi B_\\lambda(\\lambda_{\\rm rest}, T)$\n"
        "each curve encloses $10^9 L_\\odot$; z only slides the sampling window",
        fontsize=10,
        color=INK,
    )


def panel_b(exp, ax) -> None:
    wl = exp._wlen_aa_tensor.numpy()
    trans = exp._transmission_tensor.numpy()

    ax2 = ax.twinx()
    for b in range(trans.shape[0]):
        ax2.fill_between(wl, 0, trans[b], color=MUTED, alpha=0.10, lw=0)
    ax2.set_ylim(0, trans.max() * 4.0)
    ax2.set_yticks([])
    for b, name in enumerate(exp.filters_list):
        ax2.text(wl[trans[b].argmax()], trans[b].max() * 1.12, name, ha="center",
                 fontsize=8, color=MUTED)

    for T, color in TEMPS:
        flux = exp._observed_spectral_flux(
            torch.tensor([1.0], dtype=torch.float64),
            T=torch.tensor([T], dtype=torch.float64),
        ).numpy().ravel()
        T_obs = T / 2.0  # z=1
        ax.plot(wl, flux, color=color, lw=2, label=f"T={T:.0f} K → $T_{{\\rm obs}}$={T_obs:.0f} K")

    ax.set_yscale("log")
    ax.set_xlim(3100, 11100)
    ax.set_xlabel(r"OBSERVED wavelength $\lambda_{\rm obs}$ [$\AA$]")
    ax.set_ylabel(r"$f_\lambda(\lambda_{\rm obs})$ [erg/s/cm$^2$/$\AA$]")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title(
        "(b) transported to the observer at z=1:\n"
        r"$f_\lambda = L_\lambda(\lambda_{\rm obs}/(1+z))\,/\,[4\pi D_L^2 (1+z)]$",
        fontsize=10,
        color=INK,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=plot_path("rest_frame_sed.png"))
    args = ap.parse_args()

    exp = build_experiment()
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax in axes:
        ax.grid(color=GRID, lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)

    panel_a(exp, axes[0])
    panel_b(exp, axes[1])
    fig.suptitle(
        r"The rest-frame SED $L_\lambda(\lambda_{\rm rest})$ and how redshift enters", fontsize=12.5
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    wl = np.logspace(2.7, 5.0, 4000)
    print("\nenclosed area of each rest-frame curve (should be 1e9 L_sun):")
    for T, _ in TEMPS:
        _, area = rest_frame_luminosity(exp, wl, T)
        print(f"  T={T:6.0f} K -> {area:.4e} L_sun   peak at {2.898e7 / T:6.0f} AA")
    print("\nrest-frame slice LSST samples (3199-10989 AA observed):")
    for z in (0.0, 1.0, 2.0):
        print(f"  z={z:.0f} -> {3199 / (1 + z):6.0f} - {10989 / (1 + z):6.0f} AA rest")


if __name__ == "__main__":
    main()
