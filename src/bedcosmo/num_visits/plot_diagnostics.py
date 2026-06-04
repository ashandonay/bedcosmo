"""
Unified plotting script for NumVisits experiment diagnostics.

Subcommands:
    filter-flux         Photon flux in each LSST filter vs redshift
    sigma-decomposition Decompose sigma_mag into sky/source/dark/read contributions
    degeneracy          High-z degeneracy: residuals and log-likelihood vs candidate z
    blackbody-sed       Blackbody SED with LSST filter transmissions annotated
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "src")
from bedcosmo.num_visits import NumVisits
from bedcosmo.num_visits.experiment import s0

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def _save(fig, fname):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


FILTER_COLORS = {
    "u": "#56B4E9", "g": "#009E73", "r": "#E69F00",
    "i": "#D55E00", "z": "#CC79A7", "y": "#8B0000",
}

COMPONENT_COLORS = {
    "Sky": "#1f77b4",
    "Source (Poisson)": "#ff7f0e",
    "Dark current": "#2ca02c",
    "Read noise": "#d62728",
}


def make_experiment(
    temperature,
    filters_list,
    mag_err_cap=None,
    device="cpu",
    cosmo_model="bb",
):
    prior_args = {
        "parameters": {
            "z": {
                "distribution": {"type": "uniform", "lower": 0.1, "upper": 3.0},
                "plot": {"lower": 0.1, "upper": 3.0},
                "latex": r"$z$",
            }
        },
        "constraints": {},
    }
    if cosmo_model == "bb_temp":
        prior_args["parameters"]["T"] = {
            "distribution": {"type": "uniform", "lower": 2000.0, "upper": 30000.0},
            "plot": {"lower": 2000.0, "upper": 30000.0},
            "latex": r"$T\,\mathrm{[K]}$",
        }
    design_args = {"input_type": "nominal", "labels": filters_list}
    kwargs = dict(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model=cosmo_model,
        temperature=temperature,
        device=device,
        verbose=False,
        cdf_samples=2000,
        transform_input=False,
    )
    if mag_err_cap is not None:
        kwargs["mag_err_cap"] = mag_err_cap
    return NumVisits(**kwargs)


# ---------------------------------------------------------------------------
# filter-flux
# ---------------------------------------------------------------------------

def plot_filter_flux(temperature, filters_list):
    exp = make_experiment(temperature, filters_list)
    z_grid = np.linspace(0.01, 3.0, 500)
    z_tensor = torch.tensor(z_grid, device=exp.device, dtype=torch.float64)
    flux_aa = exp._observed_spectral_flux(z_tensor)
    mags = exp._calculate_magnitudes(flux_aa).detach().cpu().numpy()

    s0_arr = np.array([s0[band] for band in filters_list])
    photon_flux = 10.0 ** (-0.4 * (mags - s0_arr[np.newaxis, :]))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, band in enumerate(filters_list):
        ax.plot(z_grid, photon_flux[:, i], color=FILTER_COLORS[band], lw=2, label=band)

    ax.set_ylabel(r"Photon flux [photons/s]", fontsize=12)
    ax.set_xlabel("Redshift", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=10, ncol=len(filters_list))
    ax.set_title(f"Blackbody T = {temperature:.0f} K, filter-integrated photon flux", fontsize=13)

    fig.tight_layout()
    _save(fig, f"filter_flux_T{temperature:.0f}.png")
    plt.show()


# ---------------------------------------------------------------------------
# sigma-decomposition
# ---------------------------------------------------------------------------

def decompose_errors(exp, z_grid):
    """
    Reproduce _magnitude_errors but return each variance component separately.

    noise_total^2 = noise_sky^2 + noise_src^2 + noise_dark^2 + noise_read^2
    where each component is sum(p_i^2 * var_component_i) under the optimal
    pixel-weighted SNR.
    """
    z_tensor = torch.tensor(z_grid, device=exp.device, dtype=torch.float64)
    flux_aa = exp._observed_spectral_flux(z_tensor)
    mags = exp._calculate_magnitudes(flux_aa).detach().cpu().numpy()
    nvisits_np = exp.nominal_design.unsqueeze(0).expand(len(z_grid), -1).cpu().numpy()

    pad_shape = (1,) * (mags.ndim - 1) + (exp.num_filters,)
    s0_arr = exp._s0_array.reshape(pad_shape)
    sbar = exp._sbar_array.reshape(pad_shape)

    fluxes = 10.0 ** (-0.4 * (mags - s0_arr)) * nvisits_np * exp.visit_time
    pixels = fluxes[..., np.newaxis, np.newaxis] * exp._base_img

    sigma_sky_threshold = np.sqrt(sbar * nvisits_np * exp.visit_time * (exp.pixel_scale**2))
    mask = pixels > (exp.threshold * sigma_sky_threshold[..., np.newaxis, np.newaxis])
    masked_pixels = pixels * mask

    signal = (masked_pixels**2).sum(axis=(-2, -1))

    sky_var_pp = sbar * nvisits_np * exp.visit_time * (exp.pixel_scale**2)
    dark_var_pp = exp.dark_current * nvisits_np * exp.visit_time
    read_var_pp = (exp.read_noise**2) * nvisits_np * exp.n_exp_per_visit
    src_var_pp = masked_pixels

    noise2_sky = (masked_pixels**2 * sky_var_pp[..., np.newaxis, np.newaxis]).sum(axis=(-2, -1))
    noise2_src = (masked_pixels**2 * src_var_pp).sum(axis=(-2, -1))
    noise2_dark = (masked_pixels**2 * dark_var_pp[..., np.newaxis, np.newaxis]).sum(axis=(-2, -1))
    noise2_read = (masked_pixels**2 * read_var_pp[..., np.newaxis, np.newaxis]).sum(axis=(-2, -1))

    noise2_total = noise2_sky + noise2_src + noise2_dark + noise2_read
    noise_total = np.sqrt(noise2_total)
    snr = np.where(noise_total == 0, 0.0, signal / noise_total)
    coeff = 2.5 / np.log(10.0)
    sigma_mag = np.where(snr > 0, coeff / snr, exp.mag_err_cap)

    safe_total = np.where(noise2_total > 0, noise2_total, 1.0)
    frac_sky = noise2_sky / safe_total
    frac_src = noise2_src / safe_total
    frac_dark = noise2_dark / safe_total
    frac_read = noise2_read / safe_total

    n_masked = np.maximum(mask.sum(axis=(-2, -1)), 1)
    src_var_avg = masked_pixels.sum(axis=(-2, -1)) / n_masked

    return {
        "sigma_mag": sigma_mag,
        "mags": mags,
        "frac_sky": frac_sky, "frac_src": frac_src,
        "frac_dark": frac_dark, "frac_read": frac_read,
        "pp_sky": sky_var_pp, "pp_src": src_var_avg,
        "pp_dark": dark_var_pp, "pp_read": read_var_pp,
    }


def plot_sigma_decomposition(temperature, filters_list):
    exp = make_experiment(temperature, filters_list)
    z_grid = np.linspace(0.01, 3.0, 500)
    result = decompose_errors(exp, z_grid)

    n_filters = len(filters_list)
    fig, axes = plt.subplots(3, n_filters, figsize=(7 * n_filters, 12), sharex=True)
    if n_filters == 1:
        axes = axes[:, np.newaxis]

    for i, band in enumerate(filters_list):
        fc = FILTER_COLORS[band]

        # Row 1: relative error
        ax = axes[0, i]
        mags_band = result["mags"][:, i]
        rel_err = result["sigma_mag"][:, i] / np.abs(mags_band)
        rel_cap = exp.mag_err_cap / np.abs(mags_band)
        ax.plot(z_grid, rel_err, color=fc, lw=2, label=r"$\sigma_{\rm mag} / |m|$")
        ax.plot(z_grid, rel_cap, color="k", ls=":", lw=1, alpha=0.4, label="cap / |m|")
        ax.set_yscale("log")
        if i == 0:
            ax.set_ylabel(r"$\sigma_{\rm mag}\,/\,|m|$", fontsize=11)
        ax.set_title(f"{band}-band", fontsize=13, fontweight="bold", color=fc)
        ax.legend(fontsize=9)

        # Row 2: per-pixel variance components
        ax = axes[1, i]
        pp_total = result["pp_sky"][:, i] + result["pp_src"][:, i] + result["pp_dark"][:, i] + result["pp_read"][:, i]
        ax.plot(z_grid, result["pp_sky"][:, i], color=COMPONENT_COLORS["Sky"], lw=2, label="Sky")
        ax.plot(z_grid, result["pp_src"][:, i], color=COMPONENT_COLORS["Source (Poisson)"], lw=2, label="Source (Poisson)")
        ax.plot(z_grid, result["pp_dark"][:, i], color=COMPONENT_COLORS["Dark current"], lw=2, label="Dark current")
        ax.plot(z_grid, result["pp_read"][:, i], color=COMPONENT_COLORS["Read noise"], lw=2, label="Read noise")
        ax.plot(z_grid, pp_total, color="k", lw=2.5, ls="--", label="Total")
        ax.set_yscale("log")
        if i == 0:
            ax.set_ylabel("Per-pixel variance\n" + r"[$e^2$ / pixel]", fontsize=10)
        ax.legend(fontsize=8, loc="best")

        # Row 3: fractional contribution (stacked)
        ax = axes[2, i]
        ax.fill_between(z_grid, 0, result["frac_sky"][:, i],
                        color=COMPONENT_COLORS["Sky"], alpha=0.7, label="Sky")
        bottom = result["frac_sky"][:, i]
        ax.fill_between(z_grid, bottom, bottom + result["frac_src"][:, i],
                        color=COMPONENT_COLORS["Source (Poisson)"], alpha=0.7, label="Source (Poisson)")
        bottom = bottom + result["frac_src"][:, i]
        ax.fill_between(z_grid, bottom, bottom + result["frac_dark"][:, i],
                        color=COMPONENT_COLORS["Dark current"], alpha=0.7, label="Dark current")
        bottom = bottom + result["frac_dark"][:, i]
        ax.fill_between(z_grid, bottom, bottom + result["frac_read"][:, i],
                        color=COMPONENT_COLORS["Read noise"], alpha=0.7, label="Read noise")
        ax.set_ylim(0, 1.05)
        if i == 0:
            ax.set_ylabel("Fractional contribution\nto noise variance", fontsize=10)
        ax.set_xlabel("Redshift", fontsize=11)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Magnitude Error Decomposition: T = {temperature:.0f} K, nominal visits",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    _save(fig, f"sigma_decomposition_T{temperature:.0f}_{''.join(filters_list)}.png")
    plt.show()


# ---------------------------------------------------------------------------
# degeneracy
# ---------------------------------------------------------------------------

def compute_likelihood_components(exp, z_grid, m_obs, nvisits):
    z_tensor = torch.tensor(z_grid, device=exp.device, dtype=torch.float64)
    flux_aa = exp._observed_spectral_flux(z_tensor)
    m_model = exp._calculate_magnitudes(flux_aa).detach().cpu().numpy()
    nvisits_exp = nvisits.unsqueeze(0).expand(len(z_grid), -1)
    sigmas = exp._magnitude_errors(
        torch.tensor(m_model, device=exp.device, dtype=torch.float64),
        nvisits_exp,
    ).detach().cpu().numpy()

    raw_residual_sq = np.sum((m_obs - m_model) ** 2, axis=-1)
    diff = (m_obs - m_model) / sigmas
    chi2_term = -0.5 * np.sum(diff**2, axis=-1)
    norm_term = -np.sum(np.log(sigmas), axis=-1)
    log_like = chi2_term + norm_term
    return log_like, raw_residual_sq


def plot_degeneracy(filters_list, temperatures, mag_err_cap, z_decomp):
    temp_cmap = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(temperatures)))
    z_grid = np.linspace(0.01, 3.0, 1000)

    experiments = {
        T: make_experiment(float(T), filters_list, mag_err_cap=mag_err_cap)
        for T in temperatures
    }

    fig, axes = plt.subplots(2, len(z_decomp), figsize=(22, 9), sharex=True)
    if len(z_decomp) == 1:
        axes = axes[:, np.newaxis]

    for col, z_true in enumerate(z_decomp):
        ax_resid = axes[0, col]
        ax_loglik = axes[1, col]

        for T, tc in zip(temperatures, temp_cmap):
            exp = experiments[T]
            z_t = torch.tensor([z_true], device=exp.device, dtype=torch.float64)
            flux_aa = exp._observed_spectral_flux(z_t)
            m_obs = exp._calculate_magnitudes(flux_aa).detach().cpu().numpy().squeeze()
            log_like, raw_resid_sq = compute_likelihood_components(
                exp, z_grid, m_obs, exp.nominal_design
            )

            label = f"T = {T} K"
            ax_resid.plot(z_grid, raw_resid_sq, color=tc, lw=2, label=label)
            ax_loglik.plot(z_grid, log_like - log_like.max(), color=tc, lw=2, label=label)

        ax_resid.axvline(z_true, color="gray", ls="--", lw=1.5)
        ax_loglik.axvline(z_true, color="gray", ls="--", lw=1.5)

        ax_resid.set_title(rf"$z_{{\rm true}}$ = {z_true}", fontsize=13, fontweight="bold")
        ax_resid.set_ylim(0, 200)
        ax_loglik.set_yscale("symlog", linthresh=1.0)
        ax_loglik.set_ylim(-1e4, 5)
        ax_resid.set_xlim(0, 3.0)
        ax_loglik.set_xlim(0, 3.0)
        ax_loglik.set_xlabel("Candidate redshift", fontsize=11)

        if col == 0:
            ax_resid.set_ylabel("Magnitude residual (obs - model)^2", fontsize=10)
            ax_loglik.set_ylabel("Log-likelihood", fontsize=10)

        ax_resid.legend(fontsize=9, loc="upper right")
        ax_loglik.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    cap_str = f"cap{mag_err_cap}" if mag_err_cap is not None else "nocap"
    _save(fig, f"degeneracy_decomposition_multiT_{''.join(filters_list)}_{cap_str}.png")
    plt.show()


# ---------------------------------------------------------------------------
# blackbody-sed
# ---------------------------------------------------------------------------

def plot_blackbody_sed(temperatures, redshifts):
    from astropy.constants import h, c, k_B, sigma_sb
    from astropy.cosmology import Planck18
    from astropy import units as u
    from speclite import filters as speclite_filters

    _h = h.cgs.value
    _c = c.cgs.value
    _k_B = k_B.cgs.value
    _sigma_sb = sigma_sb.cgs.value

    def bb_surface_flux(wlen_aa, T_K):
        wlen_cm = wlen_aa * 1e-8
        exponent = _h * _c / (wlen_cm * _k_B * T_K)
        B = (2 * _h * _c**2) / (wlen_cm**5 * np.expm1(exponent))
        return np.pi * B * 1e-8

    def observed_flux(wlen_obs_aa, T_K, z):
        if z == 0.0:
            d_L_cm = (10.0 * u.pc).to(u.cm).value
        else:
            d_L_cm = Planck18.luminosity_distance(z).to(u.cm).value
        L_sun = 3.826e33
        L_bol = 1e9 * L_sun
        R_eff = np.sqrt(L_bol / (4 * np.pi * _sigma_sb * T_K**4))
        wlen_rest_aa = wlen_obs_aa / (1.0 + z)
        F_surface = bb_surface_flux(wlen_rest_aa, T_K)
        L_lambda = 4 * np.pi * R_eff**2 * F_surface
        return L_lambda / (4 * np.pi * d_L_cm**2 * (1.0 + z))

    linestyles = ["-", "--", "-.", ":"]
    temp_linestyles = {T: linestyles[i % len(linestyles)] for i, T in enumerate(temperatures)}
    z_cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(redshifts)))

    filters_data = {}
    for band in FILTER_COLORS:
        filt = speclite_filters.load_filter(f"lsst2023-{band}")
        wlen_raw = filt.wavelength
        wlen = wlen_raw.to(u.AA).value if hasattr(wlen_raw, "to") else np.asarray(wlen_raw, dtype=float)
        trans = filt(wlen_raw * u.AA if hasattr(wlen_raw, "unit") else wlen_raw)
        if not isinstance(trans, np.ndarray):
            trans = np.asarray(trans)
        filters_data[band] = (wlen, trans)

    wlen_min, wlen_max = 3000, 11500
    wlen_aa = np.linspace(wlen_min, wlen_max, 500)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for T in temperatures:
        ls_T = temp_linestyles[T]
        for i, z in enumerate(redshifts):
            flux_obs = observed_flux(wlen_aa, T, z)
            ax1.plot(wlen_aa, flux_obs, ls=ls_T, lw=1.8, color=z_cmap[i],
                     label=f"z = {z:.1f}, {T:.0f} K")

    ax1.set_ylabel(r"$F_\lambda$ [erg/s/cm$^2$/Å]", fontsize=12)
    ax1.set_yscale("log")
    ax1.set_title("Blackbody SED", fontsize=13)
    ax1.legend(fontsize=12, loc="best", ncol=2)
    ax1.set_xlim(wlen_min, wlen_max)

    for band, (wlen_f, trans_f) in filters_data.items():
        ax2.fill_between(wlen_f, trans_f / trans_f.max(), alpha=0.3, color=FILTER_COLORS[band])
        ax2.plot(wlen_f, trans_f / trans_f.max(), color=FILTER_COLORS[band], lw=1.2)
        center = np.average(wlen_f, weights=trans_f)
        ax2.text(center, 1.05, band, ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color=FILTER_COLORS[band])

    ax2.set_xlabel("Observed wavelength [Å]", fontsize=12)
    ax2.set_ylabel("Filter\ntransmission", fontsize=10)
    ax2.set_ylim(0, 1.25)
    ax2.set_xlim(wlen_min, wlen_max)

    plt.tight_layout()
    _save(fig, "blackbody_sed_filters.png")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_cap(s):
    return None if s.lower() == "none" else float(s)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="plot", required=True)

    p = sub.add_parser("filter-flux", help="Photon flux per filter vs redshift")
    p.add_argument("--temperature", type=float, default=5000.0)
    p.add_argument("--filters", type=str, default="u,g,r,i,z,y")

    p = sub.add_parser("sigma-decomposition", help="Magnitude-error noise breakdown")
    p.add_argument("--temperature", type=float, default=5000.0)
    p.add_argument("--filters", type=str, default="u,g")

    p = sub.add_parser("degeneracy", help="High-z degeneracy residual + log-likelihood")
    p.add_argument("--filters", type=str, default="u,g")
    p.add_argument("--temperatures", type=float, nargs="+", default=[2000, 5000, 10000, 30000])
    p.add_argument("--mag-err-cap", type=str, default="10.0",
                   help="Magnitude error cap (float or 'none')")
    p.add_argument("--z-decomp", type=float, nargs="+", default=[0.3, 0.6, 1.2, 2.0])

    p = sub.add_parser("blackbody-sed", help="Redshifted blackbody SED with LSST filters")
    p.add_argument("--temperatures", type=float, nargs="+", default=[5000.0, 10000.0])
    p.add_argument("--redshifts", type=float, nargs="+", default=[0.1, 0.6, 1.2])

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.plot == "filter-flux":
        plot_filter_flux(args.temperature, args.filters.split(","))
    elif args.plot == "sigma-decomposition":
        plot_sigma_decomposition(args.temperature, args.filters.split(","))
    elif args.plot == "degeneracy":
        plot_degeneracy(
            filters_list=args.filters.split(","),
            temperatures=[int(t) for t in args.temperatures],
            mag_err_cap=_parse_cap(args.mag_err_cap),
            z_decomp=args.z_decomp,
        )
    elif args.plot == "blackbody-sed":
        plot_blackbody_sed(args.temperatures, args.redshifts)


if __name__ == "__main__":
    main()
