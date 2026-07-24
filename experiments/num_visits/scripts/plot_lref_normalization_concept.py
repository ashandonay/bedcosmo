"""Concept figures: what in-band (L_ref @ lambda_ref) normalization means.

Explains the ``norm_mode: monochromatic`` mode for the num_visits blackbody, where the
SED amplitude is pinned to a reference luminosity ``L_ref`` at a rest-frame
reference wavelength ``lambda_ref`` instead of to the bolometric integral.

No data or trained run is involved -- these are pure Planck curves.

Two figures are written to ``experiments/num_visits/plots/``:

  lref_normalization_concept.png -- the one-axis-at-a-time explainer:
    (A) Vary L_ref at fixed T. L_ref is a pure amplitude / brightness knob:
        every curve has the same shape and reads off exactly L_ref where it
        crosses the dashed lambda_ref line. A prior on L_ref is a prior on
        where the SED sits at lambda_ref (an absolute-magnitude prior).
    (B) Vary T at fixed L_ref. All curves pass through the SAME anchor point
        (lambda_ref, L_ref) but fan out in shape -- brightness and color are
        decoupled.

  lref_normalization_ensemble.png -- the joint (T, L_ref) prior, i.e. what the
    (z, T, L_ref) model would actually generate before redshifting:
    (A) individual SED draws colored by T. Each L_ref level is a pinch point at
        lambda_ref (value there = L_ref, independent of T); the curves fan out
        by T away from it. So at lambda_ref the vertical spread is the L_ref
        (brightness) prior alone, and the shape spread everywhere else is T.
    (B) the full prior-predictive SED envelope (shaded) over the joint prior --
        the space of SEDs the model spans, with its waist at lambda_ref.

Usage:
  python experiments/num_visits/scripts/plot_lref_normalization_concept.py \
      [--lambda-ref 5000] [--fixed-T 6000]
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from _paths import plot_path  # noqa: E402

# cgs constants
_H = 6.62607015e-27
_C = 2.99792458e10
_K = 1.380649e-16
INK, MUTED, GRID = "#22252A", "#6B7280", "#D8DBE0"

# Joint prior ranges for the ensemble figure.
T_LO, T_HI = 4000.0, 10000.0
M_LO, M_HI = -22.0, -20.0  # brighter .. fainter absolute magnitude at lambda_ref


def planck_lambda(lam_aa: np.ndarray, T: float) -> np.ndarray:
    """Planck B_lambda(T) in arbitrary (shape-only) units; lam in Angstrom."""
    lam_cm = lam_aa * 1e-8
    x = _H * _C / (lam_cm * _K * T)
    return (2 * _H * _C**2 / lam_cm**5) / np.expm1(x)


def anchored_sed(lam_aa, T, lam_ref, l_ref):
    """Planck SED at temperature T scaled so L_lambda(lam_ref) == l_ref."""
    shape = planck_lambda(lam_aa, T)
    amp = l_ref / planck_lambda(np.array([lam_ref]), T)[0]
    return amp * shape


def _l_ref_from_mag(M):
    return 10 ** (-0.4 * (np.asarray(M) + 21.0))  # M_ref = -21 -> 1.0


def _style_axes(ax, lam_ref):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rest-frame wavelength [Å]")
    ax.axvspan(3800, 7500, color=GRID, alpha=0.35, zorder=0)
    ax.axvline(lam_ref, color=MUTED, ls="--", lw=1.3)
    ax.grid(True, which="major", color=GRID, alpha=0.5, lw=0.6)


def concept_figure(args) -> None:
    lam = np.logspace(np.log10(1200), np.log10(30000), 600)
    lam_ref = args.lambda_ref

    M_ref = np.array([-20.0, -21.0, -22.0])
    L_ref = _l_ref_from_mag(M_ref)
    L_cols = plt.cm.Blues(np.linspace(0.45, 0.9, len(M_ref)))
    T_list = np.array([4000.0, 6000.0, 8000.0, 10000.0])
    T_cols = plt.cm.plasma(np.linspace(0.15, 0.82, len(T_list)))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)

    for M, Lr, col in zip(M_ref, L_ref, L_cols):
        axA.plot(
            lam,
            anchored_sed(lam, args.fixed_T, lam_ref, Lr),
            color=col,
            lw=2.2,
            label=f"$M_\\mathrm{{ref}}={M:.0f}$",
        )
        axA.plot(lam_ref, Lr, "o", color=col, ms=9, mec="white", mew=1.2, zorder=5)
    axA.text(
        lam_ref * 1.1, 42, f"$\\lambda_\\mathrm{{ref}}={lam_ref:.0f}$ Å", color=MUTED, fontsize=9
    )
    axA.set_title(f"(A) vary $L_\\mathrm{{ref}}$ at fixed T = {args.fixed_T:.0f} K", color=INK)
    axA.text(
        0.03,
        0.05,
        "same shape, scaled amplitude;\nSED at $\\lambda_\\mathrm{ref}$ = $L_\\mathrm{ref}$",
        transform=axA.transAxes,
        fontsize=9,
        color=MUTED,
        va="bottom",
    )

    for T, col in zip(T_list, T_cols):
        axB.plot(lam, anchored_sed(lam, T, lam_ref, 1.0), color=col, lw=2.2, label=f"{T:.0f} K")
    axB.plot(lam_ref, 1.0, "o", color=INK, ms=10, mec="white", mew=1.4, zorder=6)
    axB.annotate(
        "all pinned to the\nsame $L_\\mathrm{ref}$ here",
        xy=(lam_ref, 1.0),
        xytext=(lam_ref * 1.25, 6.0),
        color=INK,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.1),
    )
    axB.set_title(
        "(B) vary T at fixed $L_\\mathrm{ref}$  →  brightness & color decoupled", color=INK
    )
    axB.text(
        0.03,
        0.05,
        "shapes fan out (color varies);\narea under curve ∝ $L_\\mathrm{bol}$ now differs",
        transform=axB.transAxes,
        fontsize=9,
        color=MUTED,
        va="bottom",
    )

    for ax in (axA, axB):
        _style_axes(ax, lam_ref)
        ax.set_ylim(3e-3, 60)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    axA.set_ylabel("relative $L_\\lambda$ (SED)")
    axA.text(5200, 4e-3, "optical", color=MUTED, fontsize=8)

    fig.suptitle(
        "In-band normalization: $L_\\mathrm{ref}$ sets brightness at $\\lambda_\\mathrm{ref}$, T sets color",
        fontsize=13,
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.out or str(plot_path("lref_normalization_concept.png"))
    fig.savefig(out, dpi=130)
    print("saved:", out)


def ensemble_figure(args) -> None:
    """Vary T AND L_ref together: the joint-prior SED ensemble + envelope."""
    lam = np.logspace(np.log10(1200), np.log10(30000), 600)
    lam_ref = args.lambda_ref

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    norm = Normalize(vmin=T_LO, vmax=T_HI)
    cmap = plt.cm.plasma

    # ---- (A) individual draws colored by T, pinch points per L_ref level ----
    M_levels = np.array([-20.0, -21.0, -22.0])
    T_draws = np.linspace(T_LO, T_HI, 6)
    for M in M_levels:
        Lr = _l_ref_from_mag(M)
        for T in T_draws:
            axA.plot(
                lam, anchored_sed(lam, T, lam_ref, Lr), color=cmap(norm(T)), lw=1.4, alpha=0.85
            )
        axA.plot(lam_ref, Lr, "o", color=INK, ms=9, mec="white", mew=1.3, zorder=6)
        axA.text(
            lam_ref * 0.62, Lr, f"$M_\\mathrm{{ref}}={M:.0f}$", color=INK, fontsize=8.5, va="center"
        )
    axA.set_title(
        "(A) joint (T, $L_\\mathrm{ref}$) draws — pinch at $\\lambda_\\mathrm{ref}$, fan by T",
        color=INK,
    )
    axA.set_ylabel("relative $L_\\lambda$ (SED)")
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axA, pad=0.01)
    cbar.set_label("T [K]")

    # ---- (B) full prior-predictive envelope over the joint prior ----
    T_grid = np.linspace(T_LO, T_HI, 30)
    M_grid = np.linspace(M_LO, M_HI, 30)
    stack = np.array(
        [anchored_sed(lam, T, lam_ref, _l_ref_from_mag(M)) for T in T_grid for M in M_grid]
    )
    lo, hi = stack.min(axis=0), stack.max(axis=0)
    axB.fill_between(
        lam, lo, hi, color="#3B7DD8", alpha=0.28, lw=0, label="prior-predictive SED range"
    )
    axB.plot(
        lam,
        anchored_sed(lam, 0.5 * (T_LO + T_HI), lam_ref, 1.0),
        color="#1A4B8C",
        lw=2.2,
        label=f"median draw (T={0.5*(T_LO+T_HI):.0f} K, $M_\\mathrm{{ref}}$=-21)",
    )
    axB.annotate(
        "waist at $\\lambda_\\mathrm{ref}$:\nwidth = $L_\\mathrm{ref}$ prior only",
        xy=(lam_ref, 1.0),
        xytext=(lam_ref * 1.35, 12.0),
        color=INK,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.1),
    )
    axB.set_title("(B) prior-predictive SED envelope over the joint prior", color=INK)
    axB.legend(fontsize=8.5, loc="lower center", framealpha=0.9)

    for ax in (axA, axB):
        _style_axes(ax, lam_ref)
        ax.set_ylim(1e-3, 1.2e2)
    axA.text(5200, 1.4e-3, "optical", color=MUTED, fontsize=8)

    fig.suptitle(
        f"Joint (T, $L_\\mathrm{{ref}}$) prior at $\\lambda_\\mathrm{{ref}}$={lam_ref:.0f} Å  "
        f"(T∈[{T_LO:.0f},{T_HI:.0f}] K, $M_\\mathrm{{ref}}$∈[{M_HI:.0f},{M_LO:.0f}])",
        fontsize=13,
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = str(plot_path("lref_normalization_ensemble.png"))
    fig.savefig(out, dpi=130)
    print("saved:", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default=None, help="override path for the concept figure")
    p.add_argument("--lambda-ref", type=float, default=5000.0, help="rest-frame Angstrom")
    p.add_argument("--fixed-T", type=float, default=6000.0, help="T for concept panel A [K]")
    args = p.parse_args()
    concept_figure(args)
    ensemble_figure(args)


if __name__ == "__main__":
    main()
