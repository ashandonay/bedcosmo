"""R-vs-L_bol as the flat-T explanation -- TESTED AND RULED OUT.

SUPERSEDED as an explanation, still valid as mechanism. The figure below is a
correct account of what swapping the luminosity normalization (fix R instead of
L_bol) does to the *redshift* constraint. It was produced while testing whether
that choice explains the flat bb_temp T posterior. It does not.

The actual driver is the *value* of ``l_bol``, not R-vs-L_bol: the T signal is a
fixed ~0.213 mag while only the photometric sigma scales with luminosity, so the
signal is ~0.5 sigma at ``l_bol=1e9`` (a dwarf) and ~15 sigma at L* ~ 3e10. Note
too that no single ``l_bol`` is physically realistic across 0.1 < z < 3 -- a
standard candle has no luminosity spread. See ``plot_bb_degeneracy.py`` for the
companion ruled-out hypothesis.

Colors depend only on T_obs = T/(1+z), so every (z, T) on the "color ridge"
T = T_obs*(1+z) produces byte-identical colors. Along that line the shape channel
is blind by construction, and ONLY the bolometric flux can say where you are.
This figure asks how well it can.

  (d) The (z,T) plane. The color ridge (dashed) is the direction colors cannot
      see. Contours are exact 2D grid posteriors (no flow) for the two
      luminosity normalizations. Fixing L_bol produces a tight blob; fixing R
      smears along the ridge -- because its amplitude has gone blind in the same
      direction the colors already were.

  (c) r magnitude ALONG the ridge, with the +/-1 sigma photometric band. This is
      the whole argument in one panel. With L_bol fixed the source dims purely
      as D_L^-2, so the curve stays steep and the amplitude pins your position --
      but sigma blows up at high z as the fixed-wattage source recedes. With R
      fixed, L ~ T^4 ~ (1+z)^4 brightens the source almost exactly as fast as
      D_L^2 dims it: the source stays bright (small sigma) but the curve goes
      FLAT, and even turns over slightly, so a wide range of (z,T) share one
      brightness. Two different failure modes: faintness vs flatness.

      Absolute swing/sigma ratios depend on the design and on how far along the
      ridge you walk -- the robust statement is the RATIO between the two models
      (~17-19x across samplings), not any single sigma figure.

  (a,b) The observed SEDs at the ridge points. Identical shape in every curve
      (that is what "on the ridge" means); only the amplitude differs. With
      L_bol fixed the curves fan out and are trivially distinguishable. With R
      fixed they pile on top of each other -- there is nothing left to measure.

Everything is computed from the live forward model; no trained run is read.

Usage:
  python experiments/num_visits/scripts/plot_ridge_discrimination.py [--out <png>] [--t-obs 5000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _paths import plot_path  # noqa: E402

from bedcosmo.num_visits import NumVisits  # noqa: E402
from bedcosmo.util import get_experiment_config_path  # noqa: E402

C_CANDLE, C_FIXEDR = "#3B7DD8", "#C2334D"
INK, MUTED, GRID = "#22252A", "#6B7280", "#D8DBE0"
T_PIN = 10000.0  # radius pinned at R(T_PIN) in the fixed-R variant
# z=0 is a singularity (D_L=0 -> infinite flux, -inf mags), so grids start just
# above it; the axes still show 0.
Z_MIN, Z_MAX = 0.02, 2.5
# NumVisits defaults to l_bol = 1e9 L_sun (a constructor arg). Flux scales linearly
# with it, so any other luminosity is a constant magnitude offset -- applied to BOTH
# variants, since the fixed-R radius is itself back-solved from L_bol at T_PIN.
L_BOL_REF = 1e9
L_BOL = 1e9  # overridden by --l-bol
Z_RIDGE = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
MODELS = [(False, "$L_{\\rm bol}$ fixed (candle)", C_CANDLE), (True, "$R$ fixed ($L\\propto T^4$)", C_FIXEDR)]


def build_experiment(device: str = "cpu") -> NumVisits:
    design_args = yaml.safe_load(open(get_experiment_config_path("num_visits", "design_args.yaml")))
    prior_args = yaml.safe_load(
        open(get_experiment_config_path("num_visits", "prior_args_gamma_temp.yaml"))
    )
    exp = NumVisits(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model="bb_temp",
        device=device,
        verbose=False,
    )
    exp.init_prior(parameters=prior_args["parameters"], cosmo_model="bb_temp")
    keys = ("input_type", "step", "lower", "upper", "sum_lower", "sum_upper", "labels")
    exp.init_designs(**{k: v for k, v in design_args.items() if k in keys})
    return exp


def mags(exp, z, T, fixed_R: bool) -> torch.Tensor:
    """Model magnitudes. fixed_R undoes the back-solve, restoring L propto T^4."""
    m = exp._calculate_magnitudes(
        exp._observed_spectral_flux(
            torch.as_tensor(np.asarray(z, dtype=float)),
            T=torch.as_tensor(np.asarray(T, dtype=float)),
        )
    )
    if fixed_R:
        offset = 10.0 * torch.log10(torch.as_tensor(np.asarray(T, dtype=float) / T_PIN))
        m = m - offset.unsqueeze(-1)
    return m - 2.5 * np.log10(L_BOL / L_BOL_REF)


def flux(exp, z, T, fixed_R: bool) -> np.ndarray:
    """Observed f_lambda on the instrument grid, with the same fixed-R rescaling."""
    f = exp._observed_spectral_flux(
        torch.tensor([z], dtype=torch.float64), T=torch.tensor([T], dtype=torch.float64)
    ).numpy().ravel()
    if fixed_R:
        f = f * (T / T_PIN) ** 4  # L propto T^4 instead of L = const
    return f * (L_BOL / L_BOL_REF)


def sigmas(exp, m: torch.Tensor, nv: torch.Tensor) -> torch.Tensor:
    return exp._magnitude_errors(m[None, :, :], nv.view(1, 1, -1)).reshape(-1, exp.num_filters)


def grid_posterior(exp, nv, fixed_R: bool, n: int, z_true=1.0, T_true=T_PIN):
    zg = np.linspace(Z_MIN, Z_MAX, n)
    Tg = np.linspace(4000.0, 16000.0, n)
    ZZ, TT = np.meshgrid(zg, Tg, indexing="ij")
    zf, Tf = ZZ.ravel(), TT.ravel()
    data = mags(exp, [z_true], [T_true], fixed_R)
    lls = []
    for k in range(0, len(zf), 2000):
        m = mags(exp, zf[k : k + 2000], Tf[k : k + 2000], fixed_R)
        s = sigmas(exp, m, nv)
        lls.append((-0.5 * ((data.reshape(1, 6) - m) / s) ** 2 - torch.log(s)).sum(-1))
    ll = torch.cat(lls).reshape(n, n).numpy()
    lp = 2 * np.log(zg) - zg / 0.3  # gamma(3, 0.3) on z; uniform on T
    post = np.exp((ll + lp[:, None]) - (ll + lp[:, None]).max())
    post /= post.sum()
    return zg, Tg, post


def hpd_levels(post: np.ndarray, masses=(0.68, 0.95)) -> list[float]:
    """HPD contour levels. Returns [] if the posterior is unresolved on this grid.

    At high L_bol the posterior can be far narrower than a grid cell, collapsing
    all levels onto one value; matplotlib then rejects them as non-increasing.
    Signal that explicitly rather than crashing or drawing something misleading.
    """
    flat = np.sort(post.ravel())[::-1]
    cum = np.cumsum(flat)
    levels = sorted({float(flat[np.searchsorted(cum, m)]) for m in masses})
    return levels if len(levels) == len(masses) else []


def panel_posterior(exp, ax, nv, n_grid, t_obs, letter) -> None:
    for fixed_R, label, color in MODELS:
        zg, Tg, post = grid_posterior(exp, nv, fixed_R, n_grid)
        levels = hpd_levels(post)
        if levels:
            ax.contour(zg, Tg, post.T, levels=levels, colors=color,
                       linewidths=[1.0, 2.0], zorder=3)
        else:
            # Posterior narrower than a grid cell: mark it and say so.
            i, j = np.unravel_index(post.argmax(), post.shape)
            ax.plot(zg[i], Tg[j], "x", color=color, ms=10, mew=2, zorder=3)
            print(f"  WARNING: {label} posterior unresolved on this grid "
                  f"(peak z={zg[i]:.3f}, T={Tg[j]:.0f}); shown as a cross.")
        ax.plot([], [], color=color, lw=2, label=label)  # legend proxy

    zl = np.linspace(Z_MIN, Z_MAX, 100)
    ax.plot(zl, t_obs * (1 + zl), color=INK, lw=1.5, ls="--",
            label=f"color ridge $T={t_obs:.0f}(1+z)$")
    # Not inference -- just markers for where panels (a,b) evaluate the SEDs.
    ax.plot(Z_RIDGE, t_obs * (1 + Z_RIDGE), "o", color=INK, ms=5, zorder=4, mfc="none",
            label="SED samples (panels a,b)")
    ax.plot(1.0, T_PIN, "*", color=INK, ms=14, zorder=5,
            label="truth (data generated here)")
    ax.set_xlim(0.0, Z_MAX)
    ax.set_ylim(4000, 16000)
    ax.set_xlabel("z")
    ax.set_ylabel("T [K]")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.set_title(f"({letter}) exact 2D posteriors (68/95%, no flow)",
                 fontsize=10, color=INK)


def panel_ridge_brightness(exp, ax, nv, t_obs, letter) -> dict:
    zs = np.linspace(0.1, Z_MAX, 60)
    Ts = t_obs * (1 + zs)
    out = {}
    for fixed_R, label, color in MODELS:
        m = mags(exp, zs, Ts, fixed_R)
        s = sigmas(exp, m, nv)
        r, sr = m[:, 2].numpy(), s[:, 2].numpy()
        ax.fill_between(zs, r - sr, r + sr, color=color, alpha=0.18, lw=0)
        ax.plot(zs, r, color=color, lw=2.5, label=label)
        # The swing/sigma ratio depends on how far you walk, so report both the
        # span and the range it was measured over rather than a bare "N sigma".
        out[label] = (r.max() - r.min()) / sr.min()
    ax.invert_yaxis()  # astronomical convention: brighter is up
    ax.set_xlabel("z   (walking the ridge, $T=%.0f(1+z)$)" % t_obs)
    ax.set_ylabel("r magnitude")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(
        f"({letter}) brightness along the ridge, $\\pm1\\sigma$ band ($z$={zs[0]:.1f}–{zs[-1]:.1f})",
        fontsize=10,
        color=INK,
    )
    return out


def panel_sed(exp, ax, fixed_R, label, t_obs, letter) -> None:
    wl = exp._wlen_aa_tensor.numpy()
    cmap = plt.get_cmap("viridis")
    for i, z in enumerate(Z_RIDGE):
        T = t_obs * (1 + z)
        f = flux(exp, z, T, fixed_R)
        ax.plot(wl, f, color=cmap(i / (len(Z_RIDGE) - 1)), lw=2,
                label=f"z={z:.2f}, T={T:.0f} K")
    ax.set_yscale("log")
    ax.set_ylim(3e-21 * (L_BOL / L_BOL_REF), 1e-18 * (L_BOL / L_BOL_REF))
    ax.set_xlim(3100, 11100)
    ax.set_xlabel(r"observed wavelength [$\AA$]")
    ax.set_ylabel(r"$f_\lambda$ [erg/s/cm$^2$/$\AA$]")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_title(f"({letter}) SEDs on the ridge — {label}\n"
                 "identical SHAPE; only amplitude differs", fontsize=10, color=INK)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=plot_path("ridge_discrimination.png"))
    ap.add_argument("--nvisits", type=float, default=None,
                    help="flat visits/band; default is the nominal LSST design from the code")
    ap.add_argument("--n-grid", type=int, default=200)
    ap.add_argument("--t-obs", type=float, default=5000.0)
    ap.add_argument("--l-bol", type=float, default=1e9,
                    help="bolometric luminosity in L_sun (code default 1e9 = a dwarf; L* ~ 3e10)")
    args = ap.parse_args()

    global L_BOL
    L_BOL = args.l_bol
    exp = build_experiment()
    if args.nvisits is None:
        nv = exp.nominal_design.clone()  # fiducial_nvisits: u=70 g=100 r=230 i=230 z=200 y=200
        design_tag = "nominal LSST: " + ", ".join(
            f"{b}={int(v)}" for b, v in zip(exp.filters_list, nv.tolist())
        ) + f"  (sum {int(nv.sum())})"
    else:
        nv = torch.full((exp.num_filters,), args.nvisits, dtype=torch.float64)
        design_tag = f"{args.nvisits:.0f} visits/band (flat)"

    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    for ax in axes.ravel():
        ax.grid(color=GRID, lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)

    # Narrative order: the data first (SEDs), then what the amplitude can do with
    # it along the ridge, then the posterior that results.
    panel_sed(exp, axes[0, 0], False, "$L_{\\rm bol}$ fixed", args.t_obs, "a")
    panel_sed(exp, axes[0, 1], True, "$R$ fixed", args.t_obs, "b")
    power = panel_ridge_brightness(exp, axes[1, 0], nv, args.t_obs, "c")
    panel_posterior(exp, axes[1, 1], nv, args.n_grid, args.t_obs, "d")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    # A swing/sigma ratio over the whole path is meaningless here: as z->0 the
    # source diverges in brightness (D_L->0) and sigma collapses, so the ratio is
    # dominated by the endpoint. Report the LOCAL precision instead, which is
    # what the panel-(c) slope and band actually encode.
    M_bol = 4.74 - 2.5 * np.log10(L_BOL)
    print(f"\n{design_tag}")
    print(f"L_bol = {L_BOL:.1e} L_sun  (M_bol = {M_bol:+.1f})")
    print("\nlocal precision on z from brightness alone:  dz ~ sigma_r / |dr/dz|")
    print(f"{'z':>5} | {'candle dr/dz':>12} {'sig_r':>6} {'-> dz':>7} |"
          f" {'fixedR dr/dz':>12} {'sig_r':>6} {'-> dz':>7}")
    for z in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5):
        row = f"{z:5.2f} |"
        for fixed_R, _, _ in MODELS:
            h = 0.01
            m0 = mags(exp, [z - h], [args.t_obs * (1 + z - h)], fixed_R)[0, 2].item()
            m1 = mags(exp, [z + h], [args.t_obs * (1 + z + h)], fixed_R)[0, 2].item()
            slope = (m1 - m0) / (2 * h)
            s = sigmas(exp, mags(exp, [z], [args.t_obs * (1 + z)], fixed_R), nv)[0, 2].item()
            row += f" {slope:12.3f} {s:6.3f} {s / abs(slope):7.3f} |"
        print(row)


if __name__ == "__main__":
    main()
