"""How the blackbody temperature sets which LSST bands carry the signal.

Answers the question "should the num_visits blackbody use a much lower T?" by
scanning the live forward + noise model (no trained run is read) over a grid of
temperatures and redshifts at fixed bolometric luminosity, then reading off the
per-band SNR at the nominal visit allocation and the color-z leverage.

The take-aways the four panels encode (fixed L = 3e10 L_sun = ~L*):

  (A) Per-band SNR vs T at z=1. T slides the informative bands across the
      spectrum: 10000 K (the current default) puts the SNR in r/i/g, 4500 K
      shifts it to i/z, and 3000 K is essentially undetected -- so retuning T
      moves the optimal visit allocation.
  (B) Per-band SNR vs z at T=10000 K (current): bright and detectable out to
      z~3 across the blue-optical bands.
  (C) Per-band SNR vs z at T=4500 K (cooler): at FIXED L_bol a cooler blackbody
      radiates into the rest-IR and drops out of the ugrizy window, so the same
      L* source goes undetected past z~2. This is a toy artifact (bolometric vs
      optical brightness are conflated), not real red-galaxy behavior.
  (D) g-i color vs z per T: cooler sources have MORE color-z leverage (a larger,
      steeper g-i swing), i.e. a bright-but-color-flat (hot) vs faint-but-
      colorful (cool) trade-off.

None of this makes a blackbody effective radius physical -- a galaxy-scale radius
would need T ~ 5 K (far-IR), not the optical regime LSST measures.

Usage:
  python experiments/num_visits/scripts/plot_temp_band_leverage.py [--out <png>]
      [--l-bol 3.0e10]
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _paths import plot_path  # noqa: E402

from bedcosmo.num_visits import NumVisits  # noqa: E402
from bedcosmo.util import get_experiment_config_path  # noqa: E402

BANDS = ["u", "g", "r", "i", "z", "y"]
COEFF = 2.5 / np.log(10.0)  # sigma_mag = COEFF / SNR
INK, MUTED = "#22252A", "#6B7280"
# Bands colored by wavelength order (sequential, meaningful); T by a hot ramp.
BAND_COLS = plt.cm.viridis(np.linspace(0.05, 0.9, len(BANDS)))


def build_experiment(l_bol: float, device: str = "cpu") -> NumVisits:
    """Instantiate a bolometric blackbody NumVisits at fixed l_bol."""
    design_args = yaml.safe_load(open(get_experiment_config_path("num_visits", "design_args.yaml")))
    design_args["input_type"] = "nominal"
    prior_args = yaml.safe_load(
        open(get_experiment_config_path("num_visits", "prior_args_gamma.yaml"))
    )
    return NumVisits(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model="bb",
        norm_mode="bolometric",
        l_bol=l_bol,
        device=device,
        verbose=False,
    )


def mags_snr(exp, nominal, z_arr, T_arr):
    """(mags, snr) each shape (..., 6) at the nominal design, fixed l_bol."""
    z = torch.as_tensor(np.asarray(z_arr, dtype=float), dtype=torch.float64)
    T = torch.as_tensor(np.asarray(T_arr, dtype=float), dtype=torch.float64)
    z_b, T_b = torch.broadcast_tensors(z, T)
    flux = exp._observed_spectral_flux(z_b, T=T_b)
    mags = exp._calculate_magnitudes(flux)
    nvisits = torch.as_tensor(np.broadcast_to(nominal, mags.shape).copy(), dtype=torch.float64)
    sigma = exp._magnitude_errors(mags, nvisits)
    return mags.cpu().numpy(), COEFF / sigma.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None, help="output PNG path")
    parser.add_argument(
        "--l-bol",
        type=float,
        default=3.0e10,
        help="fixed bolometric luminosity in L_sun (default 3e10 ~ L*)",
    )
    args = parser.parse_args()

    exp = build_experiment(args.l_bol)
    nominal = exp.nominal_design.cpu().numpy()
    print("nominal visits:", dict(zip(BANDS, nominal.astype(int))))

    T_lines = np.array([3000.0, 4500.0, 6000.0, 8000.0, 10000.0])
    z_curve = np.linspace(0.2, 3.0, 25)
    T_cols = plt.cm.plasma(np.linspace(0.15, 0.85, len(T_lines)))

    _, snrA = mags_snr(exp, nominal, np.full_like(T_lines, 1.0), T_lines)
    _, snr_hot = mags_snr(exp, nominal, z_curve, np.full_like(z_curve, 10000.0))
    _, snr_cool = mags_snr(exp, nominal, z_curve, np.full_like(z_curve, 4500.0))

    gi = np.zeros((len(T_lines), len(z_curve)))
    for i, T in enumerate(T_lines):
        m, _ = mags_snr(exp, nominal, z_curve, np.full_like(z_curve, T))
        gi[i] = m[:, BANDS.index("g")] - m[:, BANDS.index("i")]

    # ---- numeric summary -------------------------------------------------
    print(f"\n=== Per-band SNR at z=1.0, L={args.l_bol:.1e} (nominal design) ===")
    print("T[K] ", "  ".join(f"{b:>6}" for b in BANDS))
    for T, row in zip(T_lines, snrA):
        print(f"{int(T):5d}", "  ".join(f"{v:6.1f}" for v in row))
    print("\n=== g-i color span over z in [0.2, 3.0] (photo-z leverage) ===")
    for T, row in zip(T_lines, gi):
        print(f"{int(T):5d} K  g-i span {row.max() - row.min():.2f} mag")

    # ---- figure ----------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    ax = axs[0, 0]
    for j, b in enumerate(BANDS):
        ax.plot(T_lines, snrA[:, j], "-o", color=BAND_COLS[j], label=b, lw=2)
    ax.axvline(10000, color=MUTED, ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("blackbody T [K]")
    ax.set_ylabel("per-band SNR (nominal design)")
    ax.set_title("(A) SNR by band vs T   (z=1)")
    ax.legend(title="band", ncol=2, fontsize=8)

    ax = axs[0, 1]
    for j, b in enumerate(BANDS):
        ax.plot(z_curve, snr_hot[:, j], color=BAND_COLS[j], lw=2, label=b)
    ax.set_yscale("log")
    ax.set_xlabel("redshift z")
    ax.set_ylabel("per-band SNR")
    ax.set_title("(B) SNR by band vs z   (T=10000 K, current)")
    ax.legend(title="band", ncol=2, fontsize=8)

    ax = axs[1, 0]
    for j, b in enumerate(BANDS):
        ax.plot(z_curve, snr_cool[:, j], color=BAND_COLS[j], lw=2, label=b)
    ax.set_yscale("log")
    ax.set_xlabel("redshift z")
    ax.set_ylabel("per-band SNR")
    ax.set_title("(C) SNR by band vs z   (T=4500 K, cooler)")
    ax.legend(title="band", ncol=2, fontsize=8)

    ax = axs[1, 1]
    for i, T in enumerate(T_lines):
        ax.plot(z_curve, gi[i], color=T_cols[i], lw=2, label=f"{int(T)} K")
    ax.set_xlabel("redshift z")
    ax.set_ylabel("g - i  color [mag]")
    ax.set_title("(D) color-z relation = photo-z leverage")
    ax.legend(title="T", fontsize=8)

    fig.suptitle(
        f"LSST blackbody toy (L={args.l_bol:.1e} L_sun): "
        "informative bands & color-z leverage vs temperature",
        fontsize=13,
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or str(plot_path("temp_band_leverage.png"))
    fig.savefig(out, dpi=130)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
