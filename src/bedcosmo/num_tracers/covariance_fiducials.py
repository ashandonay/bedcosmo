"""Plot the emulator covariance at the nominal design for a few fixed-covariance cosmologies.

Shows how ``emulator_covariance: {param: value}`` changes the covariance the likelihood
uses, for BAO or ShapeFit, against DESI DR1's published covariance:

- BAO (vary Omega_m by default; H0*r_d stays at the DESI fiducial): 1-sigma D_M-D_H
  ellipses for the anisotropic bins, the D_V-only bins (BGS, QSO) as bars, every row's
  sigma as a fraction of the DESI central values (``central_val``), and every row's
  sigma relative to the emulator at the fiducial.
- ShapeFit (vary omega_cdm by default; the rest stay at the template fiducial): a
  q_iso-q_ap ellipse per bin, every row's absolute sigma (m's central value is ~0, so
  a fraction would mean nothing; q ~ 1, so for q_iso and q_ap it is the fraction), and
  the same relative panel. DESI's published covariance is converted to the emulators'
  (q_iso, q_ap, f_sigmar, m) basis the way ``desi_shapefit_to_targets`` converts the data.

The ShapeFit defaults omega_cdm = [0.0904, 0.12, 0.1585] are Omega_m = [0.25, 0.3152, 0.40]
at the template's h and omega_b, so the two figures compare like for like.

Example::

    python -m bedcosmo.num_tracers.covariance_fiducials --out cov_bao.png
    python -m bedcosmo.num_tracers.covariance_fiducials --analysis shapefit --out cov_shapefit.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.linalg import block_diag

from bedcosmo.num_tracers.experiment import NumTracers
from bedcosmo.util import init_experiment

ANALYSES = {
    "bao": {"prior_args_path": "prior_args_hrdrag.yaml", "fiducial": NumTracers._BAO_FIDUCIAL,
            "param": "Om", "values": [0.25, NumTracers._BAO_FIDUCIAL["Om"], 0.40]},
    "shapefit": {"prior_args_path": "prior_args_shapefit.yaml", "fiducial": NumTracers._SHAPEFIT_FIDUCIAL,
                 "param": "omega_cdm", "values": [0.0904, NumTracers._SHAPEFIT_FIDUCIAL["omega_cdm"], 0.1585]},
}
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
MARKERS = ["o", "s", "^", "D", "v"]
QUANTITY_TEX = {"DV_over_rs": "D_V", "DM_over_rs": "D_M", "DH_over_rs": "D_H",
                "qiso": r"q_{\rm iso}", "qap": r"q_{\rm ap}", "f_sigmar": r"f\sigma_r", "m": "m"}


def nominal_covariances(analysis, param, values, device="cpu"):
    """Emulator covariance at the nominal design with ``param`` fixed at each value.

    Returns ``(experiment, {value: (n_data, n_data) array})``; the experiment (the last
    one built) supplies the data layout, ``central_val`` and DESI's reference covariance.
    """
    covs = {}
    for value in values:
        experiment = init_experiment(
            cosmo_exp="num_tracers", prior_args_path=ANALYSES[analysis]["prior_args_path"],
            design_args_path="design_args_dr1.yaml", dataset="dr1", analysis=analysis,
            cosmo_model="base", likelihood_mode="emulator", emulator_space="fourier",
            emulator_covariance={param: value}, device=device, mode="eval")
        nominal = experiment.nominal_design.double().view(1, -1)
        if analysis == "bao":
            cov = experiment._build_emulator_covariance(experiment.calc_passed(nominal), {})
        else:
            # The mean's cosmology is irrelevant here; the covariance uses the fixed one.
            _, cov = experiment._shapefit_likelihood(
                experiment._shapefit_n_tracers(nominal),
                experiment._shapefit_parameters(experiment._SHAPEFIT_FIDUCIAL))
        covs[value] = cov[0].cpu().numpy()
    return experiment, covs


def desi_covariance(experiment):
    """DESI DR1's published covariance in the experiment's data basis."""
    if experiment.analysis == "bao":
        return experiment.ref_cov
    from desilike_emulator.shapefit import desi_reference

    # desi_shapefit_to_targets rescales each of DESI's (D_V/r_d, D_H/D_M, f sigma_s8, m)
    # and keeps m, so the covariance transforms by that diagonal Jacobian. f_sigmar's
    # factor (our fiducial f_sigmar over Table 11's f sigma_s8) is central / measured.
    central = experiment.central_val.cpu().numpy().reshape(len(experiment.shapefit_bins), -1)
    blocks = []
    for b, tracer_bin in enumerate(experiment.shapefit_bins):
        _, measured, cov = desi_reference.datavector(tracer_bin)
        fid = desi_reference.published_fiducial(tracer_bin)
        jac = np.array([1 / fid["DV_over_rd"], 1 / fid["DH_over_DM"], central[b, 2] / measured[2], 1.0])
        blocks.append(cov * np.outer(jac, jac))
    return block_diag(*blocks)


def _rows(experiment):
    """(bin, quantity, z) per data row, in data-vector order."""
    if experiment.analysis == "bao":
        dd = experiment.desi_data
        return list(zip(dd["tracer"].values, dd["quantity"].values, dd["z"].values))
    from desilike_emulator.shapefit import desi_reference

    return [(tb, q, desi_reference.datavector(tb)[0])
            for tb in experiment.shapefit_bins for q in experiment.shapefit_quantities]


def plot_covariance_fiducials(experiment, covs, param, path):
    """Compare ``covs`` (keyed by ``param`` value, including its fiducial) on one figure."""
    analysis = experiment.analysis
    fid_value = ANALYSES[analysis]["fiducial"][param]
    if fid_value not in covs:
        raise ValueError(f"covs must include the fiducial {param}={fid_value}; got {sorted(covs)}.")
    values = sorted(covs)
    if len(values) > len(COLORS):
        raise ValueError(f"at most {len(COLORS)} values; got {len(values)}.")
    tex = dict(zip(experiment.cosmo_params, experiment.latex_labels)).get(param, param)
    style = {v: (COLORS[k], MARKERS[k]) for k, v in enumerate(values)}
    label = {v: (rf"cov at DESI fid (${tex}={v:g}$)" if v == fid_value else rf"cov at ${tex}={v:g}$")
             for v in values}

    rows = _rows(experiment)
    bins = [r[0] for r in rows]
    quantity = [r[1] for r in rows]
    desi = desi_covariance(experiment)
    central = experiment.central_val.cpu().numpy()
    if analysis == "bao":
        # Fractional errors [%]: sigma over the DESI central value.
        err_scale = 100 / central
        err_label = "fractional $\\sigma$ [%]\n(relative to DESI central values)"
        pairs = [i for i in range(len(rows) - 1)
                 if bins[i] == bins[i + 1] and not bins[i].startswith("Lya")]
        ell_label = (r"$\delta D_M/D_M$ [%]", r"$\delta D_H/D_H$ [%]")
        bar_rows = [i for i, q in enumerate(quantity) if q == "DV_over_rs"]
        emulated = np.array([not b.startswith("Lya") for b in bins])   # Lya has no emulator
        held = rf"$H_0r_d$ = {NumTracers._BAO_FIDUCIAL['hrdrag']:g} km/s"
    else:
        # Absolute errors: m's central value is ~0. q is a ratio to the fiducial, so 100 dq is in %.
        err_scale = np.ones(len(rows))
        err_label = r"$\sigma$ (absolute)"
        pairs = [i for i, q in enumerate(quantity) if q == "qiso"]
        ell_label = (r"$\delta q_{\rm iso}$ [%]", r"$\delta q_{\rm ap}$ [%]")
        bar_rows = []
        emulated = np.ones(len(rows), dtype=bool)
        held = "other parameters at the template fiducial"
    ell_scale = 100 / central if analysis == "bao" else 100 * np.ones(len(rows))

    def err(M):
        return err_scale * np.sqrt(np.diag(M))

    tname = {b: b.replace("+", "+\n").replace("Lya QSO", r"Ly$\alpha$") for b in set(bins)}
    row_labels = [tname[b] + "\n$" + QUANTITY_TEX[q] + "$" for b, q in zip(bins, quantity)]

    n_top = len(pairs) + bool(bar_rows)
    side_by_side = len(rows) <= 12
    if side_by_side:
        fig = plt.figure(figsize=(15, 9.5), constrained_layout=True)
        gs = fig.add_gridspec(2, n_top, height_ratios=[1, 1.15])
        ax_err, ax_ratio = fig.add_subplot(gs[1, :3]), fig.add_subplot(gs[1, 3:])
    else:
        fig = plt.figure(figsize=(17, 14), constrained_layout=True)
        gs = fig.add_gridspec(3, n_top, height_ratios=[1, 1, 1])
        ax_err, ax_ratio = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])

    # Top: 1-sigma ellipses for each (row i, row i+1) pair.
    ell_axes, lim = [], 0
    for k, i in enumerate(pairs):
        ax = fig.add_subplot(gs[0, k])
        ell_axes.append(ax)
        for M, kw in [(desi, {"color": "k", "ls": "--", "lw": 1.5, "zorder": 1})] + [
                (covs[v], {"color": style[v][0], "lw": 2, "zorder": 3}) for v in values]:
            s = ell_scale[[i, i + 1]]
            sub = M[np.ix_([i, i + 1], [i, i + 1])] * np.outer(s, s)
            w, vec = np.linalg.eigh(sub)                       # axes = eigenvectors, half-lengths = sqrt(w)
            ang = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
            ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang, fill=False, **kw))
            lim = max(lim, np.sqrt(np.diag(sub)).max())
        ax.set_title(f"{bins[i]}  (z = {rows[i][2]:.2f})", fontsize=11)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlabel(ell_label[0])
        if k == 0:
            ax.set_ylabel(ell_label[1])
        ax.set_aspect("equal")
    for ax in ell_axes:
        ax.set_xlim(-1.1 * lim, 1.1 * lim)
        ax.set_ylim(-1.1 * lim, 1.1 * lim)

    # Top right (BAO): the D_V-only bins, one bar per value.
    if bar_rows:
        ax = fig.add_subplot(gs[0, len(pairs)])
        width = 0.8 / len(values)
        for j, v in enumerate(values):
            xs = np.arange(len(bar_rows)) + (j - (len(values) - 1) / 2) * width
            heights = err(covs[v])[bar_rows]
            ax.bar(xs, heights, width=width * 0.92, color=style[v][0], zorder=2)
            for xb, hb in zip(xs, heights):
                ax.text(xb, hb - 0.1, f"{hb:.1f}", ha="center", va="top", fontsize=8, color="white", weight="bold")
        for g, i in enumerate(bar_rows):
            ax.hlines(err(desi)[i], g - 0.4, g + 0.4, color="k", ls="--", lw=1.5, zorder=3)
        ax.set_xticks(np.arange(len(bar_rows)), [f"{bins[i]}\n(z = {rows[i][2]:.2f})" for i in bar_rows])
        ax.set_ylabel(r"$\sigma(D_V)/D_V$ [%]")
        ax.set_ylim(0, 1.15 * max(err(covs[v])[bar_rows].max() for v in values))
        ax.set_title(r"$D_V$-only bins", fontsize=11)
        ax.set_box_aspect(1)   # same panel height as the ellipses
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)

    # Per-row error.
    x = np.arange(len(rows))
    ax = ax_err
    ax.scatter(x, err(desi), marker="_", s=500 if side_by_side else 250, color="k", lw=2, zorder=2,
               label="DESI DR1 published")
    for v in values:
        ax.scatter(x, err(covs[v]), color=style[v][0], marker=style[v][1], s=55, zorder=3, label=label[v])
    ax.set_xticks(x, row_labels, fontsize=9)
    ax.set_ylabel(err_label)
    ax.set_ylim(0, None)
    ax.set_title("Per-row error at the nominal design", fontsize=11)
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()

    # Sigma relative to the emulator at the fiducial (log scale, symmetric about 1).
    ax = ax_ratio
    s_fid = np.sqrt(np.diag(covs[fid_value]))
    ratios = {v: np.sqrt(np.diag(covs[v])) / s_fid for v in values if v != fid_value}
    for v, ratio in ratios.items():
        mean = np.exp(np.log(ratio[emulated]).mean())
        ax.axhline(mean, color=style[v][0], lw=1, ls=":", zorder=1)
        ax.text(-0.4, mean, f"emulated-row mean ×{mean:.2f}", color="0.3", fontsize=8, va="bottom", ha="left")
        ax.scatter(x, ratio, color=style[v][0], marker=style[v][1], s=55, zorder=3)
    ax.axhline(1, color=style[fid_value][0], lw=1.2, zorder=1)
    ax.set_yscale("log")
    lo = min([1.0] + [r.min() for r in ratios.values()]) / 1.1
    hi = max([1.0] + [r.max() for r in ratios.values()]) * 1.1
    ticks = [t for t in (0.25, 0.33, 0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.6, 2, 2.5, 3, 4) if lo <= t <= hi]
    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xticks(x, row_labels, fontsize=9)
    ax.set_ylabel(r"$\sigma\,/\,\sigma_{\rm em,\,fid}$")
    ax.set_title("Relative to emulator at fiducial", fontsize=11)
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)

    value_list = ", ".join(f"{v:g}" for v in values)
    name = {"bao": "BAO", "shapefit": "ShapeFit"}[analysis]
    fig.legend(handles, labels, loc="outside lower center", ncol=len(values) + 1, fontsize=10, frameon=False)
    fig.suptitle(rf"{name} emulator covariance at the nominal design, evaluated at ${tex}$ = [{value_list}] "
                 f"({held})", fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", choices=sorted(ANALYSES), default="bao")
    parser.add_argument("--param", default=None,
                        help="Parameter to vary (default: Om for bao, omega_cdm for shapefit)")
    parser.add_argument("--values", type=float, nargs="+", default=None,
                        help="Values of --param, including its fiducial (default: see module docstring)")
    parser.add_argument("--out", required=True, help="Output figure path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    cfg = ANALYSES[args.analysis]
    param = args.param or cfg["param"]
    if param not in cfg["fiducial"]:
        parser.error(f"--param for {args.analysis} must be one of {sorted(cfg['fiducial'])}; got {param!r}")
    if args.values is None and param != cfg["param"]:
        parser.error(f"--values is required with --param {param}")
    values = args.values or cfg["values"]
    if cfg["fiducial"][param] not in values:
        parser.error(f"--values must include the fiducial {param}={cfg['fiducial'][param]}")

    experiment, covs = nominal_covariances(args.analysis, param, values, device=args.device)
    plot_covariance_fiducials(experiment, covs, param, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
