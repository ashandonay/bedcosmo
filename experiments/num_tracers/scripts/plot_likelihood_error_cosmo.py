"""Visualize per-tracer BAO likelihood errors vs N_tracers at fiducial cosmology.

For each tracer bin, sweep N_tracers (others held at nominal) with cosmology
fixed at the DESI fiducial (Om, hrdrag). Default comparison is scaling vs
emulator; use --compare-mode scaling-refcov to compare two scaling ref_covs.

Run (in the `bedcosmo` env):
    python experiments/num_tracers/scripts/plot_likelihood_error_cosmo.py \\
        --ref-cov /pscratch/.../ref_cov_emulator_dr1_base.npy

    python experiments/num_tracers/scripts/plot_likelihood_error_cosmo.py \\
        --compare-mode scaling-refcov \\
        --ref-cov /pscratch/.../ref_cov_emulator_dr1_base.npy \\
        --ref-cov-b desi_cov.npy
"""
import argparse
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from bedcosmo.util import init_experiment

# DESI 2024 fiducial cosmology (see verify_emulator_likelihood.py).
OM_FID = 0.3152
HRD_RAW_FID = 99.08 / 100.0

DR1_PLOT_ORDER = ["BGS", "LRG1", "LRG2", "LRG3+ELG1", "ELG2", "QSO"]

_DESI_TO_EMU_BIN = {
    "BGS": "BGS",
    "LRG1": "LRG1",
    "LRG2": "LRG2",
    "LRG3+ELG1": "LRG3_ELG1",
    "ELG2": "ELG2",
    "QSO": "QSO",
}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Per-tracer likelihood error plots vs N_tracers "
        "(scaling vs emulator at fiducial cosmology)."
    )
    p.add_argument("--dataset", default="dr1")
    p.add_argument("--cosmo-model", default="base")
    p.add_argument("--analysis", default="bao")
    p.add_argument("--prior-args-path", default="prior_args_hrdrag.yaml")
    p.add_argument("--design-args-path", default=None,
                   help="Defaults to design_args_<dataset>.yaml")
    p.add_argument("--ref-cov", default=None,
                   help="Reference covariance for scaling mode (arm A / circles).")
    p.add_argument("--ref-cov-b", default=None,
                   help="Second ref_cov for --compare-mode scaling-refcov (arm B / squares). "
                   "Defaults to dataset desi_cov.npy.")
    p.add_argument("--compare-mode", default="scaling-emulator",
                   choices=["scaling-emulator", "scaling-refcov"],
                   help="scaling-emulator: circle=scaling, square=emulator (default). "
                   "scaling-refcov: both scaling with different ref_cov.")
    p.add_argument("--emulator-sqrtn-ref", default="none",
                   choices=["none", "sampled", "fiducial"])
    p.add_argument("--n-points", type=int, default=20,
                   help="Number of N_tracers sweep points per tracer (default ~4× sparser than before).")
    p.add_argument("--n-frac-min", type=float, default=0.5,
                   help="Sweep from n_frac_min * N_nominal.")
    p.add_argument("--n-frac-max", type=float, default=1.5,
                   help="Sweep to n_frac_max * N_nominal.")
    p.add_argument("--apply-desi-syst", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None, help="Output PNG path")
    return p.parse_args()


def _parse_emulator_sqrtn_ref(value):
    return None if value == "none" else value


def _init_scaling_experiment(common, ref_cov):
    kwargs = dict(common, likelihood_mode="scaling")
    if ref_cov is not None:
        kwargs["ref_cov"] = ref_cov
    return init_experiment(**kwargs)


def _ref_cov_basename(exp):
    return os.path.basename(exp.ref_cov_path)


def _init_experiments(args):
    design_args_path = args.design_args_path or f"design_args_{args.dataset}.yaml"
    common = dict(
        cosmo_exp="num_tracers",
        prior_args_path=args.prior_args_path,
        design_args_path=design_args_path,
        dataset=args.dataset,
        analysis=args.analysis,
        cosmo_model=args.cosmo_model,
        include_D_M=True,
        include_D_V=True,
        device=args.device,
        mode="eval",
        apply_desi_syst=args.apply_desi_syst,
    )

    if args.compare_mode == "scaling-refcov":
        exp_a = _init_scaling_experiment(common, args.ref_cov)
        exp_b = _init_scaling_experiment(common, args.ref_cov_b)
        label_a = _ref_cov_basename(exp_a)
        label_b = _ref_cov_basename(exp_b)
        return exp_a, exp_b, label_a, label_b

    exp_a = _init_scaling_experiment(common, args.ref_cov)
    emu_kwargs = dict(
        common,
        likelihood_mode="emulator",
        emulator_sqrtn_ref=_parse_emulator_sqrtn_ref(args.emulator_sqrtn_ref),
    )
    if args.ref_cov is not None:
        emu_kwargs["ref_cov"] = args.ref_cov
    exp_b = init_experiment(**emu_kwargs)
    return exp_a, exp_b, "scaling", "emulator"


def _fiducial_params_batched(n, device):
    return {
        "Om": torch.full((n, 1), OM_FID, device=device, dtype=torch.float64),
        "hrdrag": torch.full((n, 1), HRD_RAW_FID, device=device, dtype=torch.float64),
    }


def _nominal_n_tracers(exp):
    cr = exp.nominal_design.to(torch.float64).view(1, -1)
    passed = exp.calc_passed(cr)
    return {
        bin_name: float(val)
        for bin_name, val in exp._passed_ratio_to_n_tracers(passed).items()
    }


def _passed_ratio_for_n_sweep(exp, emu_bin, n_values):
    """Sweep one tracer's N_tracers; all other bins stay at nominal."""
    device = exp.device
    nom_cr = exp.nominal_design.to(torch.float64).view(1, -1)
    nom_passed = exp.calc_passed(nom_cr)  # (1, n_data)

    desi_name = exp._EMULATOR_TRACER_TO_DESI[emu_bin]
    rows = exp.desi_data.index[exp.desi_data["tracer"] == desi_name].tolist()

    n_pts = len(n_values)
    passed = nom_passed.expand(n_pts, -1).clone()
    ratio = torch.as_tensor(n_values, device=device, dtype=torch.float64) / exp.nominal_total_obs
    for r in rows:
        passed[:, r] = ratio

    class_ratio = nom_cr.expand(n_pts, -1)
    return passed, class_ratio


def _build_scaling_covariance(exp, passed_ratio, class_ratio):
    n_data = len(exp.desi_data)
    rescaled_sigmas = torch.zeros(
        passed_ratio.shape[:-1] + (n_data,), device=exp.device, dtype=torch.float64
    )
    dh_idx = torch.as_tensor(exp.DH_idx, device=exp.device)
    rescaled_sigmas[..., dh_idx] = (
        exp.sigmas[dh_idx].to(torch.float64)
        * exp.sigma_scaling_factor(passed_ratio, exp.DH_idx)
    )
    if exp.include_D_M:
        dm_idx = torch.as_tensor(exp.DM_idx, device=exp.device)
        rescaled_sigmas[..., dm_idx] = (
            exp.sigmas[dm_idx].to(torch.float64)
            * exp.sigma_scaling_factor(passed_ratio, exp.DM_idx)
        )
    if exp.include_D_V:
        dv_idx = torch.as_tensor(exp.DV_idx, device=exp.device)
        rescaled_sigmas[..., dv_idx] = (
            exp.sigmas[dv_idx].to(torch.float64)
            * exp.sigma_scaling_factor(passed_ratio, exp.DV_idx)
        )
    corr = exp.corr_matrix.to(torch.float64)
    return corr * (rescaled_sigmas.unsqueeze(-1) * rescaled_sigmas.unsqueeze(-2))


def _extract_one_tracer(cov, exp, desi_name):
    """Extract errors for a single tracer from batched cov (N, n_data, n_data)."""
    dd = exp.desi_data
    rows = dd.index[dd["tracer"] == desi_name].tolist()
    if not rows:
        return None
    quantities = dd.loc[rows, "quantity"].tolist()
    emu_bin = _DESI_TO_EMU_BIN[desi_name]
    out = {"desi_name": desi_name, "emu_bin": emu_bin}

    if quantities == ["DV_over_rs"]:
        idx = rows[0]
        out["kind"] = "isotropic"
        out["sigma_DV"] = torch.sqrt(cov[:, idx, idx]).detach().cpu().numpy()
    elif set(quantities) == {"DM_over_rs", "DH_over_rs"}:
        dm_row = rows[quantities.index("DM_over_rs")]
        dh_row = rows[quantities.index("DH_over_rs")]
        c11 = cov[:, dh_row, dh_row]
        c22 = cov[:, dm_row, dm_row]
        c12 = cov[:, dh_row, dm_row]
        sigma_dh = torch.sqrt(c11)
        sigma_dm = torch.sqrt(c22)
        out["kind"] = "anisotropic"
        out["sigma_DH"] = sigma_dh.detach().cpu().numpy()
        out["sigma_DM"] = sigma_dm.detach().cpu().numpy()
        out["rho"] = (c12 / (sigma_dh * sigma_dm)).detach().cpu().numpy()
    else:
        return None
    return out


def _sweep_tracer(exp_a, exp_b, desi_name, n_values, device, compare_mode):
    """Compute per-tracer errors for both comparison arms over an N_tracers sweep."""
    emu_bin = _DESI_TO_EMU_BIN[desi_name]
    n_arr = np.asarray(n_values, dtype=np.float64)
    passed, class_ratio = _passed_ratio_for_n_sweep(exp_a, emu_bin, n_values)

    cov_a = _build_scaling_covariance(exp_a, passed, class_ratio)
    if compare_mode == "scaling-refcov":
        cov_b = _build_scaling_covariance(exp_b, passed, class_ratio)
    else:
        params = _fiducial_params_batched(len(n_values), device)
        cov_b = exp_b._build_emulator_covariance(passed, params)

    data_a = _extract_one_tracer(cov_a, exp_a, desi_name)
    data_b = _extract_one_tracer(cov_b, exp_b, desi_name)
    data_a["N_tracers"] = n_arr
    data_b["N_tracers"] = n_arr
    return data_a, data_b


def _nominal_marker(exp_a, exp_b, desi_name, n_nom, compare_mode):
    """Error values at nominal N_tracers and fiducial cosmology."""
    passed, class_ratio = _passed_ratio_for_n_sweep(
        exp_a, _DESI_TO_EMU_BIN[desi_name], [n_nom],
    )
    cov_a = _build_scaling_covariance(exp_a, passed, class_ratio)
    if compare_mode == "scaling-refcov":
        cov_b = _build_scaling_covariance(exp_b, passed, class_ratio)
    else:
        params = _fiducial_params_batched(1, exp_a.device)
        cov_b = exp_b._build_emulator_covariance(passed, params)
    fa = _extract_one_tracer(cov_a, exp_a, desi_name)
    fb = _extract_one_tracer(cov_b, exp_b, desi_name)
    return (
        {k: (v[0] if k != "kind" else v) for k, v in fa.items()},
        {k: (v[0] if k != "kind" else v) for k, v in fb.items()},
    )


def _spread(arr):
    return float(np.max(arr) - np.min(arr))


def _print_summary(all_a, all_b, plot_order, label_a, label_b):
    print("\nPer-tracer spread over N_tracers sweep:")
    print(f"{'tracer':<14}{'mode':<22}{'N_trac':>10}{'sig_DH':>10}{'sig_DM':>10}"
          f"{'sig_DV':>10}{'rho':>10}")
    print("-" * 86)
    for name in plot_order:
        if name not in all_a:
            continue
        for label, err in ((label_a, all_a[name]), (label_b, all_b[name])):
            n_sp = _spread(err["N_tracers"])
            if err["kind"] == "anisotropic":
                print(f"{name:<14}{label:<22}{n_sp:>10.2e}"
                      f"{_spread(err['sigma_DH']):>10.2e}"
                      f"{_spread(err['sigma_DM']):>10.2e}"
                      f"{'':>10}{_spread(err['rho']):>10.2e}")
            else:
                print(f"{name:<14}{label:<22}{n_sp:>10.2e}"
                      f"{'':>10}{'':>10}"
                      f"{_spread(err['sigma_DV']):>10.2e}{'':>10}")


def _pad_limits(lo, hi, frac=0.05):
    pad = frac * (hi - lo) if hi > lo else 0.01 * max(abs(hi), 1e-6)
    return lo - pad, hi + pad


def _fmt_3sig(x, _pos):
    if not np.isfinite(x):
        return ""
    if x == 0:
        return "0"
    return f"{x:.3g}"


_TICK_FMT_3SIG = FuncFormatter(_fmt_3sig)


def _apply_3sig_ticks(ax):
    ax.xaxis.set_major_formatter(_TICK_FMT_3SIG)
    ax.yaxis.set_major_formatter(_TICK_FMT_3SIG)


def _shared_sigma_limits(scale_data, emu_data, nom_scale, nom_emu):
    out = {}
    for key in ("sigma_DH", "sigma_DM"):
        vals = np.concatenate([
            scale_data[key], emu_data[key],
            np.array([nom_scale[key], nom_emu[key]]),
        ])
        lo, hi = float(np.min(vals)), float(np.max(vals))
        out[key] = _pad_limits(lo, hi)
    return out


def _shared_rho_limits(scale_data, emu_data, nom_scale, nom_emu):
    vals = np.concatenate([
        scale_data["rho"], emu_data["rho"],
        np.array([nom_scale["rho"], nom_emu["rho"]]),
    ])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    return _pad_limits(lo, hi)


def _shared_limits_1d(scale_data, emu_data, nom_emu):
    vals = np.concatenate([
        scale_data["sigma_DV"], emu_data["sigma_DV"],
        np.array([nom_emu["sigma_DV"]]),
    ])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    return _pad_limits(lo, hi)


def _mode_legend(ax, label_a, label_b):
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.45",
               markeredgecolor="k", markersize=8, label=label_a),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="0.45",
               markeredgecolor="k", markersize=8, label=label_b),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="k", markersize=12, label="nominal $N$"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.85)


def _plot_sigma_dm(ax, se, ee, ns, ne, n_nom, c_vmin, c_vmax, xlim, ylim,
                   label_a, label_b, show_legend=False, show_ylabel=True):
    n_frac = se["N_tracers"] / n_nom
    sc = ax.scatter(
        se["sigma_DH"], se["sigma_DM"], c=n_frac,
        cmap="viridis", vmin=c_vmin, vmax=c_vmax,
        marker="o", s=28, alpha=0.85, linewidths=0.3, edgecolors="k",
    )
    ax.scatter(
        ee["sigma_DH"], ee["sigma_DM"], c=ee["N_tracers"] / n_nom,
        cmap="viridis", vmin=c_vmin, vmax=c_vmax,
        marker="s", s=26, alpha=0.85, linewidths=0.3, edgecolors="k",
    )
    ax.scatter(ns["sigma_DH"], ns["sigma_DM"], marker="*", s=90,
               c="gold", edgecolors="k", linewidths=0.5, zorder=10)
    ax.scatter(ne["sigma_DH"], ne["sigma_DM"], marker="*", s=90,
               c="gold", edgecolors="k", linewidths=0.5, zorder=10)
    ax.set_xlabel(r"$\sigma(D_H/r_d)$", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(r"$\sigma(D_M/r_d)$", fontsize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)
    _apply_3sig_ticks(ax)
    ax.grid(alpha=0.3)
    ax.set_box_aspect(1)
    if show_legend:
        _mode_legend(ax, label_a, label_b)
    return sc


def _plot_rho_vs_nfrac(ax, se, ee, ns, ne, n_nom, c_vmin, c_vmax, ylim, show_ylabel=True):
    n_frac_s = se["N_tracers"] / n_nom
    n_frac_e = ee["N_tracers"] / n_nom
    sc = ax.scatter(
        n_frac_s, se["rho"], c=n_frac_s,
        cmap="viridis", vmin=c_vmin, vmax=c_vmax,
        marker="o", s=28, alpha=0.85, linewidths=0.3, edgecolors="k",
    )
    ax.scatter(
        n_frac_e, ee["rho"], c=n_frac_e,
        cmap="viridis", vmin=c_vmin, vmax=c_vmax,
        marker="s", s=26, alpha=0.85, linewidths=0.3, edgecolors="k",
    )
    ax.scatter([1.0], [ns["rho"]], marker="*", s=90,
               c="gold", edgecolors="k", linewidths=0.5, zorder=10)
    ax.scatter([1.0], [ne["rho"]], marker="*", s=90,
               c="gold", edgecolors="k", linewidths=0.5, zorder=10)
    ax.set_xlabel(r"$N_{\rm tracers}\,/\,N_{\rm nominal}$", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(r"$\rho(D_H, D_M)$", fontsize=8)
    ax.set_xlim(c_vmin, c_vmax)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)
    _apply_3sig_ticks(ax)
    ax.grid(alpha=0.3)
    ax.set_box_aspect(1)
    return sc


def _plot_isotropic_combined(ax, se, ee, ns, ne, n_nom, c_vmin, c_vmax,
                             ylim, label_a, label_b,
                             show_legend=False, show_ylabel=True):
    n_frac_s = se["N_tracers"] / n_nom
    n_frac_e = ee["N_tracers"] / n_nom
    sc = ax.scatter(
        n_frac_s, se["sigma_DV"], c=n_frac_s,
        cmap="viridis", vmin=c_vmin, vmax=c_vmax,
        marker="o", s=28, alpha=0.85, linewidths=0.3, edgecolors="k",
    )
    ax.scatter(
        n_frac_e, ee["sigma_DV"], c=n_frac_e,
        cmap="viridis", vmin=c_vmin, vmax=c_vmax,
        marker="s", s=26, alpha=0.85, linewidths=0.3, edgecolors="k",
    )
    ax.scatter([1.0], [ns["sigma_DV"]], marker="*", s=90,
               c="gold", edgecolors="k", linewidths=0.5, zorder=10)
    ax.scatter([1.0], [ne["sigma_DV"]], marker="*", s=90,
               c="gold", edgecolors="k", linewidths=0.5, zorder=10)
    ax.set_xlabel(r"$N_{\rm tracers}\,/\,N_{\rm nominal}$", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(r"$\sigma(D_V/r_d)$", fontsize=8)
    ax.set_xlim(c_vmin, c_vmax)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)
    _apply_3sig_ticks(ax)
    ax.grid(alpha=0.3)
    ax.set_box_aspect(1)
    if show_legend:
        _mode_legend(ax, label_a, label_b)
    return sc


def _cosmo_param_label(args):
    """Human-readable fiducial cosmology inputs used in the sweep."""
    if args.cosmo_model == "base":
        return rf"$\Omega_m={OM_FID:.4f}$, $h r_d={HRD_RAW_FID * 100:.2f}$"
    return rf"$\Omega_m={OM_FID:.4f}$, $h r_d={HRD_RAW_FID * 100:.2f}$"


def plot_results(all_a, all_b, all_nom_a, all_nom_b, n_nominal, args, label_a, label_b):
    tracers = [n for n in DR1_PLOT_ORDER if n in all_a]
    n_cols = len(tracers)
    col_w = 2.6
    fig = plt.figure(figsize=(col_w * n_cols + 1.0, 5.8))

    gs = gridspec.GridSpec(
        2, n_cols, figure=fig,
        hspace=0.38, wspace=0.22,
        left=0.06, right=0.88, top=0.84, bottom=0.10,
    )

    first_rho_col = next(
        i for i, n in enumerate(tracers) if all_a[n]["kind"] == "anisotropic"
    )
    c_vmin, c_vmax = args.n_frac_min, args.n_frac_max

    last_sc = None
    for col, name in enumerate(tracers):
        da, db = all_a[name], all_b[name]
        na, nb = all_nom_a[name], all_nom_b[name]
        emu_bin = _DESI_TO_EMU_BIN[name]
        n_nom = n_nominal[emu_bin]
        show_leg = col == 0
        show_ylab_top = col == 0
        show_ylab_bot = col == first_rho_col

        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[1, col])
        ax_top.set_title(name, fontsize=10, fontweight="bold", pad=3)

        if da["kind"] == "anisotropic":
            sig_lims = _shared_sigma_limits(da, db, na, nb)
            rho_ylim = _shared_rho_limits(da, db, na, nb)
            last_sc = _plot_sigma_dm(
                ax_top, da, db, na, nb, n_nom, c_vmin, c_vmax,
                sig_lims["sigma_DH"], sig_lims["sigma_DM"],
                label_a, label_b,
                show_legend=show_leg, show_ylabel=show_ylab_top,
            )
            _plot_rho_vs_nfrac(
                ax_bot, da, db, na, nb, n_nom, c_vmin, c_vmax, rho_ylim,
                show_ylabel=show_ylab_bot,
            )
        else:
            ylim = _shared_limits_1d(da, db, nb)
            last_sc = _plot_isotropic_combined(
                ax_top, da, db, na, nb, n_nom, c_vmin, c_vmax, ylim,
                label_a, label_b,
                show_legend=show_leg, show_ylabel=show_ylab_top,
            )
            ax_bot.axis("off")

    if last_sc is not None:
        cax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
        cb = fig.colorbar(last_sc, cax=cax)
        cb.set_label(r"$N_{\rm tracers}\,/\,N_{\rm nominal}$", fontsize=9)
        cb.ax.yaxis.set_major_formatter(_TICK_FMT_3SIG)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Likelihood errors vs $N_{{\\rm tracers}}$: {args.cosmo_model} cosmo "
        f"({_cosmo_param_label(args)}), {args.dataset.upper()}\n"
        f"circle = {label_a}, square = {label_b}",
        fontsize=11,
    )

    if args.output:
        out = os.path.expanduser(args.output)
    else:
        scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
        out_dir = os.path.join(scratch, "bedcosmo", "num_tracers")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(out_dir, f"likelihood_error_cosmo_{args.dataset}_{ts}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {out}")
    plt.close(fig)
    return out


def main():
    args = _parse_args()
    exp_a, exp_b, label_a, label_b = _init_experiments(args)
    device = exp_a.device

    if args.compare_mode == "scaling-refcov":
        print("Compare mode: scaling-refcov")
        print(f"  circle: {exp_a.ref_cov_path}")
        print(f"  square: {exp_b.ref_cov_path}")
    else:
        print(f"Compare mode: scaling-emulator (ref_cov: {exp_a.ref_cov_path})")

    n_nominal = _nominal_n_tracers(exp_a)
    print("Nominal N_tracers per bin:")
    for name in DR1_PLOT_ORDER:
        emu_bin = _DESI_TO_EMU_BIN.get(name)
        if emu_bin in n_nominal:
            print(f"  {name:<14} {n_nominal[emu_bin]:,.0f}")

    all_a, all_b = {}, {}
    all_nom_a, all_nom_b = {}, {}

    for desi_name in DR1_PLOT_ORDER:
        emu_bin = _DESI_TO_EMU_BIN.get(desi_name)
        if emu_bin not in n_nominal:
            continue
        n_nom = n_nominal[emu_bin]
        n_values = np.geomspace(
            n_nom * args.n_frac_min, n_nom * args.n_frac_max, args.n_points,
        )
        print(f"Sweeping {desi_name} ({args.n_points} points, "
              f"N=[{n_values[0]:,.0f}, {n_values[-1]:,.0f}])...")
        all_a[desi_name], all_b[desi_name] = _sweep_tracer(
            exp_a, exp_b, desi_name, n_values, device, args.compare_mode,
        )
        na, nb = _nominal_marker(
            exp_a, exp_b, desi_name, n_nom, args.compare_mode,
        )
        all_nom_a[desi_name] = na
        all_nom_b[desi_name] = nb

    _print_summary(all_a, all_b, DR1_PLOT_ORDER, label_a, label_b)
    plot_results(
        all_a, all_b, all_nom_a, all_nom_b, n_nominal, args, label_a, label_b,
    )


if __name__ == "__main__":
    main()
