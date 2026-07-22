"""Two candidate explanations for the flat T posterior -- BOTH RULED OUT.

SUPERSEDED. This figure records a negative result. It was built to argue that the
flat bb_temp T posterior is structural, forced by the two mechanisms below. Both
were subsequently tested and are NOT the cause. The actual driver is the *value*
of ``l_bol``: the T signal is a fixed ~0.213 mag, while only the photometric sigma
scales with luminosity, so at ``l_bol=1e9`` (a dwarf galaxy) the signal is ~0.5
sigma, and at L* ~ 3e10 it is ~15 sigma. Raising ``l_bol`` is the lever. See
``plot_ridge_discrimination.py`` for the companion ruled-out hypothesis.

Read the panels as a correct description of the *geometry* (the ridge is real),
but not of *why* T is unconstrained.

Note also that ``l_bol`` is no longer hardcoded: it is a ``NumVisits`` constructor
argument (``experiment.py``, default ``1e9`` L_sun). The text below predates that.

The two mechanisms this figure explores:

  1. FIXED BOLOMETRIC LUMINOSITY. ``L_bol`` is held constant and the
     emitting radius is back-solved from Stefan-Boltzmann,
     ``R_eff = sqrt(L_bol / (4 pi sigma T^4))``, so the total luminosity is the
     same constant at every temperature. The observed flux
     ``L / ((1+z) 4 pi D_L^2)`` is then a zero-scatter function of z alone: a
     perfect standard candle. Brightness -> z, no spectral features needed.

  2. EXACT WIEN DEGENERACY. Planck's exponent hc/(lambda_rest k T) is invariant
     under lambda_rest -> s lambda_rest, T -> T/s. Since lambda_rest =
     lambda_obs/(1+z), the observed SED *shape* -- hence every color -- depends
     only on the combination T/(1+z), never on T and z separately.

The argument was: colors measure T/(1+z) (weakly), amplitude measures z (sharply),
and T is only ever recovered as (T/(1+z))*(1+z), so a flat T posterior is the
model being correct rather than the flow failing. The conclusion that the flow is
not at fault still holds; the attribution to these two mechanisms does not.

The four panels:

  (a) The standard candle. R_eff is back-solved so that 4 pi R^2 sigma T^4 is
      pinned to 1e9 L_sun across the whole prior -- the radius swings ~7x while
      the luminosity does not move in the 6th decimal.
  (b) Why the assumed T matters in `bb`. Observed-frame SEDs at z=1 for three
      fixed temperatures against the LSST bands. L_bol is identical for all
      three; what differs is how much of it lands in the 3200-11000 AA window.
      T=10000 K puts the peak mid-band; T=4000 K pushes it past y, leaving only
      the faint blue tail, ~3.5 mag down in r.
  (c) The degeneracy ridge. Exact 2D grid posterior over (z, T) -- no flow
      involved. Colors alone would pin T/(1+z), forcing the steep dashed line;
      the standard-candle amplitude pulls against it, so the ridge the model
      actually has is shallower. Every T along it fits the data nearly as well
      (max logL varies ~3 nats over a factor 2.5 in T), which is why the
      marginal T sd is barely below the prior's. Horizontal lines are the `bb`
      slices: fixing T picks one cut through this ridge.
  (d) The cost of the assumption. Fitting with T pinned to 10000 K when the true
      temperature differs slides z along the ridge, biasing z by ~1 sigma for a
      20% error in T.

Everything is computed from the live forward model; no trained run is read.

Usage:
  python experiments/num_visits/scripts/plot_bb_degeneracy.py [--out <png>] [--nvisits 100]
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

# Categorical hues in fixed order (identity, never cycled); ink stays neutral.
C_COOL, C_MID, C_WARM = "#3B7DD8", "#B0640F", "#C2334D"
INK, MUTED, GRID = "#22252A", "#6B7280", "#D8DBE0"
T_FIXED_BB = 10000.0  # train_args.yaml: bb.temperature
Z_TRUE = 1.0
L_SUN = 3.826e33
R_SUN = 6.957e10


def build_experiment(device: str = "cpu") -> NumVisits:
    """Instantiate NumVisits exactly as the bb_temp config does."""
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
    design_keys = ("input_type", "step", "lower", "upper", "sum_lower", "sum_upper", "labels")
    exp.init_designs(**{k: v for k, v in design_args.items() if k in design_keys})
    return exp


def mags(exp: NumVisits, z, T) -> torch.Tensor:
    z_t = torch.as_tensor(np.asarray(z, dtype=float), dtype=torch.float64)
    T_t = torch.as_tensor(np.asarray(T, dtype=float), dtype=torch.float64)
    return exp._calculate_magnitudes(exp._observed_spectral_flux(z_t, T=T_t))


def sigmas(exp: NumVisits, m: torch.Tensor, nv: torch.Tensor) -> torch.Tensor:
    return exp._magnitude_errors(m[None, :, :], nv.view(1, 1, -1)).reshape(-1, exp.num_filters)


def log_like(exp, z, T, data, nv, chunk=2000) -> np.ndarray:
    """Diagonal-Gaussian log-likelihood of `data` over flat (z, T) arrays."""
    out = []
    for k in range(0, len(z), chunk):
        m = mags(exp, z[k : k + chunk], T[k : k + chunk])
        s = sigmas(exp, m, nv)
        out.append((-0.5 * ((data.reshape(1, -1) - m) / s) ** 2 - torch.log(s)).sum(-1))
    return torch.cat(out).numpy()


def log_prior_z(z: np.ndarray) -> np.ndarray:
    """Gamma(shape=3, z_0=0.3) from prior_args_gamma_temp.yaml, up to a constant."""
    return 2 * np.log(z) - z / 0.3


def moments(grid: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    mean = float((grid * p).sum())
    return mean, float(np.sqrt((p * (grid - mean) ** 2).sum()))


def panel_a(ax) -> None:
    """The back-solved radius pins L_bol to a constant."""
    from astropy import units as u
    from astropy.constants import sigma_sb

    sig = sigma_sb.to(u.erg / (u.s * u.cm**2 * u.K**4)).value
    T = np.linspace(4000.0, 10500.0, 300)
    R = np.sqrt((1e9 * L_SUN / (4 * np.pi * sig)) / T**4)
    L = 4 * np.pi * R**2 * sig * T**4 / L_SUN

    ax.plot(T, R / R_SUN, color=C_COOL, lw=2, label=r"$R_{\rm eff}$ (back-solved)")
    ax.set_yscale("log")
    ax.set_xlabel("temperature [K]")
    ax.set_ylabel(r"$R_{\rm eff}$ [$R_\odot$]", color=C_COOL)
    ax.tick_params(axis="y", colors=C_COOL)
    ax.annotate(
        f"radius swings {R[0] / R[-1]:.0f}$\\times$\n"
        f"luminosity constant to\n{abs(L / 1e9 - 1).max():.0e} relative",
        xy=(0.5, 0.62),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    ax.text(
        0.5,
        0.22,
        r"$4\pi R_{\rm eff}^2\,\sigma T^4 \equiv 10^9\,L_\odot$"
        "\n"
        "(zero scatter $\\Rightarrow$ standard candle)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=INK,
        bbox=dict(fc="#F3F4F6", ec=GRID, boxstyle="round,pad=0.4"),
    )
    ax.set_title("(a) fixed $L_{\\rm bol}$: brightness alone gives $z$", fontsize=10, color=INK)


def panel_b(exp, ax) -> None:
    """Observed-frame SEDs at z=1 vs the LSST bands, for three fixed T."""
    wl = exp._wlen_aa_tensor.numpy()
    trans = exp._transmission_tensor.numpy()

    ax2 = ax.twinx()  # band shapes are context, not a second measure of the data
    for b in range(trans.shape[0]):
        ax2.fill_between(wl, 0, trans[b], color=MUTED, alpha=0.10, lw=0)
    ax2.set_ylim(0, trans.max() * 4.0)
    ax2.set_yticks([])
    for b, name in enumerate(exp.filters_list):
        peak = wl[trans[b].argmax()]
        ax2.text(peak, trans[b].max() * 1.12, name, ha="center", fontsize=8, color=MUTED)

    for T, color in [(4000.0, C_COOL), (5000.0, C_MID), (T_FIXED_BB, C_WARM)]:
        z_t = torch.tensor([Z_TRUE], dtype=torch.float64)
        T_t = torch.tensor([T], dtype=torch.float64)
        flux = exp._observed_spectral_flux(z_t, T=T_t).numpy().ravel()
        r_mag = float(mags(exp, [Z_TRUE], [T])[0, 2])
        # Wien peak in the observed frame; the SED grid stops at the instrument
        # cutoff, so cool peaks fall off the red end entirely. Carry that in the
        # legend -- edge annotations collide here.
        peak = 2.898e7 / T * (1 + Z_TRUE)
        where = f"peak {peak:.0f}$\\,\\AA$" + ("" if peak <= wl.max() else ", off-band")
        ax.plot(wl, flux, color=color, lw=2, label=f"T={T:.0f} K — r={r_mag:.1f}, {where}")
        if peak <= wl.max():
            ax.plot(peak, flux[np.argmin(np.abs(wl - peak))], "v", color=color, ms=8)

    ax.set_yscale("log")
    ax.set_xlim(3100, 11100)
    ax.set_xlabel(r"observed wavelength [$\AA$]")
    ax.set_ylabel(r"$f_\lambda$ [erg/s/cm$^2$/$\AA$]")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title(
        "(b) same $L_{\\rm bol}$, $z{=}1$: assumed T sets what LSST catches",
        fontsize=10,
        color=INK,
    )


def panel_c(exp, ax, nv, n_grid: int) -> dict:
    """Exact 2D grid posterior over (z, T) -- the degeneracy ridge."""
    data = mags(exp, [Z_TRUE], [T_FIXED_BB])
    zg = np.linspace(0.1, 3.0, n_grid)
    Tg = np.linspace(4000.0, 10000.0, n_grid)
    ZZ, TT = np.meshgrid(zg, Tg, indexing="ij")

    ll = log_like(exp, ZZ.ravel(), TT.ravel(), data, nv).reshape(n_grid, n_grid)
    lpost = ll + log_prior_z(zg)[:, None]
    post = np.exp(lpost - lpost.max())
    post /= post.sum()

    ax.pcolormesh(zg, Tg, post.T, cmap="Blues", shading="auto", rasterized=True)

    # Pure Wien: what colors ALONE would demand. Drawn for contrast -- the true
    # ridge is measured below, and sits shallower because the standard-candle
    # amplitude pulls against the color degeneracy.
    T_line = np.linspace(4000.0, 10000.0, 100)
    ax.plot(
        T_line / (T_FIXED_BB / (1 + Z_TRUE)) - 1,
        T_line,
        color=MUTED,
        lw=1.5,
        ls="--",
        label=r"pure Wien $T/(1+z)$=const (colors only)",
    )

    # The ridge the model actually has: argmax of the conditional posterior at
    # each T, i.e. the z you infer if you assert that temperature.
    ridge = []
    for T_slice in Tg[::8]:
        cond = post[:, int(np.argmin(np.abs(Tg - T_slice)))]
        ridge.append(zg[cond.argmax()])
    ax.plot(ridge, Tg[::8], color=C_WARM, lw=2.5, label="actual ridge (colors + amplitude)")

    for T_slice in (5000.0, 8000.0, T_FIXED_BB):
        ax.axhline(T_slice, color=MUTED, lw=1, ls=":")
    ax.text(1.97, T_FIXED_BB - 320, "`bb` slices", fontsize=8, color=MUTED, ha="right")
    ax.plot(Z_TRUE, T_FIXED_BB, "*", color=INK, ms=14, label="truth", zorder=5)
    ax.set_xlim(0.1, 2.0)
    ax.set_ylim(4000, 10000)
    ax.set_xlabel("z")
    ax.set_ylabel("T [K]")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    pz, pT = post.sum(1), post.sum(0)
    z_mean, z_sd = moments(zg, pz)
    T_mean, T_sd = moments(Tg, pT)
    T_prior_sd = (Tg[-1] - Tg[0]) / np.sqrt(12)
    ax.set_title(
        f"(c) exact grid posterior (no flow): z sd={z_sd:.3f},\n"
        f"T sd={T_sd:.0f} vs prior sd={T_prior_sd:.0f} — T barely informed",
        fontsize=10,
        color=INK,
    )
    return dict(z_mean=z_mean, z_sd=z_sd, T_mean=T_mean, T_sd=T_sd, T_prior_sd=T_prior_sd)


def panel_d(exp, ax, nv) -> list:
    """z inferred with T pinned at 10000 K, as the true T varies."""
    zg = np.linspace(0.1, 3.0, 600)
    lp = log_prior_z(zg)
    T_trues = np.linspace(8000.0, 12000.0, 9)

    rows = []
    for T_true in T_trues:
        data = mags(exp, [Z_TRUE], [T_true])
        ll = log_like(exp, zg, np.full_like(zg, T_FIXED_BB), data, nv)
        q = np.exp(ll + lp - (ll + lp).max())
        q /= q.sum()
        mean, sd = moments(zg, q)
        rows.append((T_true, mean, sd))

    Tt = np.array([r[0] for r in rows])
    zh = np.array([r[1] for r in rows])
    sd = np.array([r[2] for r in rows])

    ax.fill_between(
        Tt, Z_TRUE - sd, Z_TRUE + sd, color=C_COOL, alpha=0.15, lw=0, label=r"$\pm1\sigma$ on z"
    )
    ax.axhline(Z_TRUE, color=MUTED, lw=1, ls="--")
    ax.plot(Tt, zh, "o-", color=C_WARM, lw=2, ms=6, label=r"$\hat{z}$ (T pinned at 10000 K)")
    ax.axvline(T_FIXED_BB, color=MUTED, lw=1, ls=":")
    ax.text(T_FIXED_BB + 60, ax.get_ylim()[0], " assumption correct", fontsize=8, color=MUTED)
    ax.set_xlabel("true temperature [K]")
    ax.set_ylabel(r"inferred $z$   (truth $z=1$)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(
        "(d) wrong assumed T slides z along the ridge:\n~1$\\sigma$ bias for a 20% T error",
        fontsize=10,
        color=INK,
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=plot_path("bb_degeneracy.png"))
    ap.add_argument("--nvisits", type=float, default=100.0)
    ap.add_argument("--n-grid", type=int, default=200)
    args = ap.parse_args()

    exp = build_experiment()
    nv = torch.full((exp.num_filters,), args.nvisits, dtype=torch.float64)

    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    for ax in axes.ravel():
        ax.grid(color=GRID, lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)

    panel_a(axes[0, 0])
    panel_b(exp, axes[0, 1])
    stats = panel_c(exp, axes[1, 0], nv, args.n_grid)
    rows = panel_d(exp, axes[1, 1], nv)

    fig.suptitle(
        f"num_visits `bb`: z is a standard candle, not a photo-z  "
        f"({args.nvisits:.0f} visits/band, noiseless data at z=1)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    print(f"\n(c) grid posterior: z sd={stats['z_sd']:.4f}  T sd={stats['T_sd']:.0f} "
          f"(prior sd={stats['T_prior_sd']:.0f})")
    print("(d) T_true -> z_hat +/- sd  [T assumed 10000 K]")
    for T_true, mean, sd in rows:
        print(f"    {T_true:7.0f} -> {mean:.3f} +/- {sd:.3f}  (bias {mean - Z_TRUE:+.3f})")


if __name__ == "__main__":
    main()
