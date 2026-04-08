"""Plot scaled Fisher covariance vs N_tracers, varying one cosmo param (BAO or ShapeFit).

BAO: 2×2 covariance in (DH/rd, DM/rd) from qpar/qper + Jacobian.
ShapeFit: 4×4 covariance in (qiso, qap, m, f·σ_r) from compressed params + Jacobian.

Usage (from this directory):
    python test_cov_scaling.py Om --analysis bao --scale N_tracers
    python test_cov_scaling.py Om --analysis shapefit --scale N_tracers log(Om)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from desilike import Fisher
from desilike.likelihoods.galaxy_clustering import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import (
    CutskyFootprint,
    ObservablesCovarianceMatrix,
    TracerPowerSpectrumMultipolesObservable,
)
from desilike.theories.galaxy_clustering import (
    BAOPowerSpectrumTemplate,
    KaiserTracerPowerSpectrumMultipoles,
    ShapeFitPowerSpectrumTemplate,
    SimpleBAOWigglesTracerPowerSpectrumMultipoles,
)
from desilike.theories.primordial_cosmology import get_cosmo

try:
    from bedcosmo.num_tracers.emulator.scale_data import (
        eval_scale_expression,
        scale_expression_latex,
        scale_expression_suffix,
        variables_in_scale_expression,
    )
except ImportError:
    from scale_data import (  # type: ignore[import-not-found]
        eval_scale_expression,
        scale_expression_latex,
        scale_expression_suffix,
        variables_in_scale_expression,
    )

warnings.filterwarnings("ignore", message=".*EisensteinHu.*")

# --- Fixed shape parameters (matching prep_covar.py / DESI Y1 BAO) ---
_OMEGA_B_FID = 0.02237
_MNU_FID = 0.06
_OMEGA_NU_FID = _MNU_FID / 93.14
_N_S_FID = 0.9649
_LN10A_S_FID = 3.044
_H_FID = 0.6736

# DESI DR2 tracer bins: name -> (z_min, z_max, z_eff, ntracers_low, ntracers_high)
TRACER_BINS = {
    "BGS":       (0.1, 0.4, 0.295, 6e5, 1.8e6),
    "LRG1":      (0.4, 0.6, 0.510, 5e5, 1.6e6),
    "LRG2":      (0.6, 0.8, 0.706, 8e5, 2.4e6),
    "LRG3_ELG1": (0.8, 1.1, 0.934, 2.3e6, 6.8e6),
    "ELG2":      (1.1, 1.6, 1.321, 1.9e6, 5.7e6),
    "QSO":       (0.8, 2.1, 1.484, 7e5, 2.2e6),
    "Lya_QSO":   (1.8, 4.2, 2.330, 6.5e5, 1.9e6),
}

COSMO_MODELS = {
    "base":              ["Om", "hrdrag"],
    "base_w":            ["Om", "w0", "hrdrag"],
    "base_w_wa":         ["Om", "w0", "wa", "hrdrag"],
    "base_omegak":       ["Om", "Ok", "hrdrag"],
    "base_omegak_w_wa":  ["Om", "Ok", "w0", "wa", "hrdrag"],
}

PARAM_DEFAULTS = {"Ok": 0.0, "w0": -1.0, "wa": 0.0}
HRDRAG_FID = 100.0

PRIOR_RANGES = {
    "Om": (0.01, 0.99),
    "Ok": (-0.3, 0.3),
    "w0": (-3.0, 1.0),
    "wa": (-3.0, 2.0),
    "hrdrag": (10.0, 1000.0),
}

PARAM_LATEX = {
    "N_tracers": r"$N_{\mathrm{tracers}}$",
    "Om": r"$\Omega_m$",
    "Ok": r"$\Omega_k$",
    "w0": r"$w_0$",
    "wa": r"$w_a$",
    "hrdrag": r"$h \, r_{\mathrm{drag}}$",
}

# ShapeFit physical block (after J transform); matches shapefit/run_single_fisher.py
SHAPEFIT_PARAM_LABELS = [r"$q_{\mathrm{iso}}$", r"$q_{\mathrm{ap}}$", r"$m$", r"$f\sigma_r$"]
SHAPEFIT_LIKELIHOOD_NAMES = ["qiso", "qap", "dm", "df"]

area = 14000.0
b0 = 0.84
resolution = 3

BAO_PARAM_LABELS = [r"$D_H/r_d$", r"$D_M/r_d$"]


def _get_fiducial():
    fid = {"Om": 0.3111, "hrdrag": HRDRAG_FID}
    fid.update(PARAM_DEFAULTS)
    return fid


def _to_cosmo_dict(Om, Ok, w0, wa, hrdrag=None):
    omega_cdm = Om * _H_FID**2 - _OMEGA_NU_FID - _OMEGA_B_FID
    if omega_cdm <= 0:
        raise ValueError(f"Om={Om:.4f}: omega_cdm={omega_cdm:.4f} < 0")
    theta_cosmo = {
        "Omega_m": Om,
        "Omega_k": Ok,
        "w0_fld": w0,
        "wa_fld": wa,
        "h": _H_FID,
        "omega_b": _OMEGA_B_FID,
        "n_s": _N_S_FID,
        "logA": _LN10A_S_FID,
    }
    return theta_cosmo, hrdrag if hrdrag is not None else HRDRAG_FID


def sample_cosmologies(vary_param, n=20):
    fiducial = _get_fiducial()
    lo, hi = PRIOR_RANGES[vary_param]
    if vary_param == "Om":
        lo = max(lo, (_OMEGA_B_FID + _OMEGA_NU_FID) / _H_FID**2 + 0.01)
    if vary_param == "w0":
        hi = min(hi, -fiducial["wa"] - 0.01)
    if vary_param == "wa":
        hi = min(hi, -fiducial["w0"] - 0.01)
    values = np.linspace(lo, hi, n)
    samples = []
    for v in values:
        p = dict(fiducial)
        p[vary_param] = v
        samples.append(p)
    return samples, values


def run_sweep_bao(cosmo_params, N_tracers_values, z, zrange):
    """BAO: return (n_N, 2, 2) covariance in (DH/rd, DM/rd)."""
    theta_cosmo, hrdrag = _to_cosmo_dict(**cosmo_params)
    cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
    fo = cosmo.get_fourier()

    r = 0.5
    sigmaper = r * 12.4 * 0.758 * fo.sigma8_z(z, of="delta_cb") / 0.9
    f = fo.sigma8_z(z, of="theta_cb") / fo.sigma8_z(z, of="delta_cb")
    b1 = b0 * fo.sigma8_cb / fo.sigma8_z(z, of="delta_cb")
    params = {"b1": b1, "sigmapar": (1.0 + f) * sigmaper, "sigmaper": sigmaper}

    template = BAOPowerSpectrumTemplate(
        z=z, fiducial=("DESI", dict(theta_cosmo)), apmode="qparqper"
    )
    theory = SimpleBAOWigglesTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(
        data=params,
        klim={0: [0.01, 0.5, 0.01], 2: [0.01, 0.5, 0.01]},
        theory=theory,
    )

    rd = hrdrag / _H_FID
    DH_over_rd_fid = float(template.DH_fid) / rd
    DM_over_rd_fid = float(template.DM_fid) / rd
    J = np.diag([DH_over_rd_fid, DM_over_rd_fid])

    results = []
    for N_gal in N_tracers_values:
        nbar = N_gal / area
        footprint = CutskyFootprint(area=area, zrange=zrange, nbar=nbar, cosmo=cosmo)
        covariance = ObservablesCovarianceMatrix(
            observable, footprints=footprint, resolution=resolution
        )
        likelihood = ObservablesGaussianLikelihood(
            observables=observable, covariance=covariance(**params)
        )
        likelihood.all_params["sigmas"].update(fixed=True)

        fisher = Fisher(likelihood)
        fisher_result = fisher(**params)

        param_names = [str(p) for p in fisher_result.names()]
        F_matrix = -np.array(fisher_result._hessian)
        cov_full = np.linalg.inv(F_matrix)

        bao_idx = [param_names.index(p) for p in ["qpar", "qper"]]
        cov_q = cov_full[np.ix_(bao_idx, bao_idx)]
        cov_dist = J @ cov_q @ J.T
        results.append(cov_dist)

    return np.array(results)


def run_sweep_shapefit(cosmo_params, N_tracers_values, z, zrange):
    """ShapeFit: return (n_N, 4, 4) covariance in (qiso, qap, m, f·σ_r)."""
    theta_cosmo, _hrdrag = _to_cosmo_dict(**cosmo_params)
    cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
    fo = cosmo.get_fourier()

    r = 0.5
    sigmaper = r * 12.4 * 0.758 * fo.sigma8_z(z, of="delta_cb") / 0.9
    f = fo.sigma8_z(z, of="theta_cb") / fo.sigma8_z(z, of="delta_cb")
    b1 = b0 * fo.sigma8_cb / fo.sigma8_z(z, of="delta_cb")
    params = {"b1": b1, "sigmapar": (1.0 + f) * sigmaper, "sigmaper": sigmaper}

    template = ShapeFitPowerSpectrumTemplate(
        z=z, fiducial=("DESI", dict(theta_cosmo)), apmode="qisoqap"
    )
    theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(
        data=params,
        klim={0: [0.01, 0.5, 0.01], 2: [0.01, 0.5, 0.01]},
        theory=theory,
    )

    f_sigmar_fid = float(template.f_sigmar_fid)
    J = np.diag([1.0, 1.0, 1.0, f_sigmar_fid])

    dz = zrange[1] - zrange[0]
    results = []
    for N_gal in N_tracers_values:
        nbar_per_dz = N_gal / (area * dz)
        footprint = CutskyFootprint(
            area=area, zrange=zrange, nbar=nbar_per_dz * dz, cosmo=cosmo
        )
        covariance = ObservablesCovarianceMatrix(
            observable, footprints=footprint, resolution=resolution
        )
        likelihood = ObservablesGaussianLikelihood(
            observables=observable, covariance=covariance(**params)
        )

        fisher = Fisher(likelihood)
        fisher_result = fisher(**params)

        param_names = [str(p) for p in fisher_result.names()]
        F_matrix = -np.array(fisher_result._hessian)
        cov_full = np.linalg.inv(F_matrix)

        sf_idx = [param_names.index(p) for p in SHAPEFIT_LIKELIHOOD_NAMES]
        cov_sf = cov_full[np.ix_(sf_idx, sf_idx)]
        cov_phys = J @ cov_sf @ J.T
        results.append(cov_phys)

    return np.array(results)


def get_scale_factor(scale_exprs, cosmo_params, N_tracers_values):
    """Compute product of scale expressions for each N_tracers value."""
    factor = np.ones_like(N_tracers_values, dtype=float)
    for expr in scale_exprs:
        for v in variables_in_scale_expression(expr):
            if v == "N_tracers":
                continue
            if v not in cosmo_params:
                raise ValueError(
                    f"Unknown scale variable '{v}' in {expr!r}. "
                    "Use one of: N_tracers, Om, Ok, w0, wa, hrdrag."
                )
        env = dict(cosmo_params)
        env["N_tracers"] = N_tracers_values
        factor *= np.asarray(eval_scale_expression(expr, env), dtype=float)
    return factor


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot scaled Fisher covariance vs N_tracers, varying one cosmo param."
        )
    )
    parser.add_argument(
        "--analysis",
        choices=("bao", "shapefit"),
        default="bao",
        help="Observable / theory pipeline (default: bao).",
    )
    parser.add_argument("vary_param", help="Cosmo param to vary (colorbar)")
    parser.add_argument(
        "--scale",
        nargs="+",
        default=["N_tracers"],
        help=(
            "Scale expressions for Cov: infix + - * / ^ ** ( ) exp() log(). "
            "Example: --scale log(N_tracers) 1/exp(Om) hrdrag^2 "
            "(use quotes if the shell would split, e.g. 'hrdrag/log(hrdrag)')."
        ),
    )
    parser.add_argument(
        "--tracer", default="LRG2", choices=list(TRACER_BINS.keys())
    )
    parser.add_argument(
        "--cosmo-model", default="base", choices=list(COSMO_MODELS.keys())
    )
    parser.add_argument("--n-cosmo", type=int, default=20)
    args = parser.parse_args()

    analysis = args.analysis
    vary_param = args.vary_param
    tracer_name = args.tracer
    cosmo_model = args.cosmo_model
    scale_vars = args.scale

    if analysis == "bao":
        run_sweep = run_sweep_bao
        param_labels = BAO_PARAM_LABELS
    else:
        run_sweep = run_sweep_shapefit
        param_labels = SHAPEFIT_PARAM_LABELS

    model_params = COSMO_MODELS[cosmo_model]
    assert vary_param in model_params, (
        f"Param '{vary_param}' not in model '{cosmo_model}' "
        f"(available: {model_params})"
    )

    z_min, z_max, z_eff, nt_low, nt_high = TRACER_BINS[tracer_name]
    zrange = (z_min, z_max)
    z = np.mean(zrange)
    N_tracers_values = np.logspace(np.log10(nt_low), np.log10(nt_high), 15)

    scale_label = r" $\times$ ".join(
        scale_expression_latex(v, PARAM_LATEX) for v in scale_vars
    )
    print(
        f"Analysis: {analysis}, model: {cosmo_model}, varying: {vary_param}, "
        f"tracer: {tracer_name}"
    )
    print(f"Scaling: Cov × {' × '.join(scale_vars)}")

    cosmo_samples, vary_values = sample_cosmologies(vary_param, n=args.n_cosmo)

    all_results = []
    all_scale_factors = []
    used_values = []
    for k, cosmo_params in enumerate(cosmo_samples):
        print(
            f"[{k+1}/{args.n_cosmo}] {vary_param}={cosmo_params[vary_param]:.3f}",
            end="",
        )
        try:
            cov_results = run_sweep(cosmo_params, N_tracers_values, z, zrange)
            scale_factor = get_scale_factor(
                scale_vars, cosmo_params, N_tracers_values
            )
            all_results.append(cov_results)
            all_scale_factors.append(scale_factor)
            used_values.append(cosmo_params[vary_param])
            print("  OK")
        except Exception as e:
            print(f"  SKIP ({e.__class__.__name__})")

    used_values = np.array(used_values)
    print(f"\nGot {len(all_results)}/{args.n_cosmo} successful")

    n_p = len(param_labels)
    norm = mcolors.Normalize(vmin=used_values.min(), vmax=used_values.max())
    cmap = cm.viridis
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    fig_w, fig_h = (9, 7) if n_p <= 2 else (11, 9)
    fig, axes = plt.subplots(n_p, n_p, figsize=(fig_w, fig_h), constrained_layout=True)
    fig.suptitle(
        f"{analysis.upper()} | {cosmo_model} {tracer_name} z=[{zrange[0]}, {zrange[1]}], "
        f"{len(all_results)} cosmologies "
        f"(varying {PARAM_LATEX[vary_param]} only)",
        fontsize=10,
    )

    for i in range(n_p):
        for j in range(n_p):
            ax = axes[i, j]
            if j < i:
                ax.set_visible(False)
                continue

            for k, cov_results in enumerate(all_results):
                vals = cov_results[:, i, j]
                scaled = all_scale_factors[k] * vals
                ax.plot(
                    N_tracers_values,
                    scaled,
                    color=cmap(norm(used_values[k])),
                    lw=1.0,
                )

            ax.set_xscale("symlog", linthresh=1e6)
            ax.set_yscale("symlog")

            has_visible_below = any(j >= i2 for i2 in range(i + 1, n_p))
            if not has_visible_below:
                ax.set_xlabel(PARAM_LATEX["N_tracers"])
            else:
                ax.set_xticklabels([])
            if j == i:
                ax.set_ylabel(f"{scale_label}" + r" $\times$ Cov")

            ax.set_title(f"({param_labels[i]}, {param_labels[j]})", fontsize=9)

    fig.colorbar(sm, ax=axes, shrink=0.6, label=PARAM_LATEX[vary_param], pad=0.02)

    scale_suffix = "_".join(scale_expression_suffix(v) for v in scale_vars)
    out_dir = Path(__file__).resolve().parent / analysis
    out_dir.mkdir(parents=True, exist_ok=True)
    savename = out_dir / (
        f"cov_scaling_{cosmo_model}_{tracer_name}_"
        f"scale_{scale_suffix}_vary_{vary_param}.png"
    )
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {savename}")
    plt.show()


if __name__ == "__main__":
    main()
