#!/usr/bin/env python3
"""
Triangle plots for NumVisits ``transform_input`` (marginal or joint Gaussianizer).

Draws KDE prior samples, applies ``params_to_unconstrained``, and writes:

- **Physical (interpretable):** simplex weights $a_k$, $\\log s$, $z$
- **NF input coords:** log-ratios $f_k$ (can look multimodal; clipped at training bounds)
- **After transform:** Gaussianized space used by the flow

Example:

  python -m bedcosmo.num_visits.empirical.diagnose_transform_input \\
    --kde-path ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/sed_prior_kde.joblib \\
    --outdir ~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/transform_diagnostic \\
    --n-samples 8000 --cdf-samples 200000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from .fit_sed_prior_kde import load_sed_prior_kde
from .fit_eazy_weights_to_desi import save_triangle_plot
from .simplex import PARAMETERIZATION_ILR, split_feature_matrix


def _get_experiment_config_path(cosmo_exp: str, config_name: str) -> Path:
    """Resolve experiment YAML without importing bedcosmo package root eagerly."""
    if "BED_COSMO_EXPERIMENTS" in os.environ:
        return Path(os.environ["BED_COSMO_EXPERIMENTS"]) / cosmo_exp / config_name
    repo_experiments = Path(__file__).resolve().parents[4] / "experiments" / cosmo_exp
    if repo_experiments.exists():
        return repo_experiments / config_name
    from bedcosmo.util import get_experiment_config_path

    return get_experiment_config_path(cosmo_exp, config_name)


def _load_prior_args_from_yaml(prior_args_path: Path) -> dict:
    with open(prior_args_path) as f:
        return yaml.safe_load(f)


def _merge_prior_parameters(prior_args: dict, cosmo_model: str) -> dict:
    """Ensure every models.yaml parameter has a prior_args entry."""
    models_path = _get_experiment_config_path("num_visits", "models.yaml")
    with open(models_path) as f:
        models = yaml.safe_load(f)
    params = list(models[cosmo_model]["parameters"])
    out = dict(prior_args)
    block = dict(out.get("parameters") or {})
    for name in params:
        if name not in block:
            block[name] = {
                "distribution": {"type": "empirical"},
                "plot": {"lower": -8.0, "upper": 8.0},
            }
    if "log_c_scale" in block:
        block["log_c_scale"].setdefault("plot", {"lower": 4.0, "upper": 10.5})
    if "z" in block:
        block["z"].setdefault("plot", {"lower": 0.0, "upper": 1.75})
    out["parameters"] = block
    return out


def _sample_physical_tensor(experiment, n_samples: int) -> torch.Tensor:
    with torch.no_grad():
        drawn = experiment.sample_parameters((n_samples,), use_prior_flow=False)
    cols = [drawn[k].squeeze(-1) for k in experiment.cosmo_params]
    return torch.stack(cols, dim=-1)


def _labels_physical(experiment) -> list[str]:
    labels = []
    for name, latex in zip(experiment.cosmo_params, experiment.latex_labels):
        if latex:
            labels.append(latex)
        else:
            labels.append(name)
    return labels


def _labels_gaussian(experiment) -> list[str]:
    """Matplotlib-safe labels (avoid underscores inside math mode)."""
    out = []
    for name, latex in zip(experiment.cosmo_params, experiment.latex_labels):
        if name == "log_c_scale":
            out.append(r"$\log s$ (gauss)")
        elif name == "z":
            out.append(r"$z$ (gauss)")
        elif name.startswith("f") and latex:
            out.append(f"{latex} (gauss)")
        else:
            out.append(f"{name} (gauss)")
    return out


def _print_summary(name: str, x: np.ndarray, names: list[str]) -> None:
    print(f"\n{name} (shape {x.shape}):")
    for j, pname in enumerate(names):
        col = x[:, j]
        ok = np.isfinite(col)
        frac_bad = 1.0 - ok.mean()
        if ok.sum() == 0:
            print(f"  {pname:14s}  all non-finite")
            continue
        c = col[ok]
        print(
            f"  {pname:14s}  mean={c.mean():8.4f}  std={c.std():8.4f}  "
            f"min={c.min():8.4f}  max={c.max():8.4f}  non-finite={frac_bad:.4f}"
        )


def _roundtrip_error(experiment, physical: torch.Tensor) -> float:
    with torch.no_grad():
        z = experiment.params_to_unconstrained(physical)
        x_back = experiment.params_from_unconstrained(z)
    return float((x_back - physical).abs().max().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triangle plots: physical prior samples vs transform_input space."
    )
    parser.add_argument(
        "--kde-path",
        type=Path,
        default=Path.home()
        / "scratch/bedcosmo/num_visits/empirical_prior/sed_prior_kde.joblib",
    )
    parser.add_argument(
        "--prior-args",
        type=Path,
        default=None,
        help="prior_args YAML (default: experiments/num_visits/prior_args_empirical.yaml).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: <kde-dir>/transform_diagnostic).",
    )
    parser.add_argument("--n-samples", type=int, default=8000)
    parser.add_argument(
        "--cdf-samples",
        type=int,
        default=200_000,
        help="Samples used to build empirical CDFs for the bijector.",
    )
    parser.add_argument("--cdf-bins", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cosmo-model", default="empirical")
    parser.add_argument(
        "--input-transform-type",
        choices=("marginal", "joint"),
        default="marginal",
        help="marginal: per-param CDF (+ tanh on f*). joint: empirical joint Gaussianizer.",
    )
    parser.add_argument("--joint-transform-shrinkage", type=float, default=1e-3)
    parser.add_argument("--logit-flow-scale", type=float, default=8.0)
    parser.add_argument("--panel-size", type=float, default=1.25)
    parser.add_argument(
        "--also-logits-triangle",
        action="store_true",
        help="Also save triangle_logits_before_transform.png (f_k marginals; often multimodal).",
    )
    args = parser.parse_args()

    kde_path = Path(args.kde_path).expanduser().resolve()
    if not kde_path.is_file():
        raise FileNotFoundError(kde_path)

    artifact = load_sed_prior_kde(kde_path)
    param = artifact.get("parameterization", "weights")
    print(f"KDE: {kde_path}")
    print(f"  version={artifact.get('version')}  parameterization={param}")
    print(f"  features={artifact['feature_names']}")

    prior_args_path = (
        Path(args.prior_args).expanduser()
        if args.prior_args
        else _get_experiment_config_path("num_visits", "prior_args_empirical.yaml")
    )
    prior_args = _merge_prior_parameters(
        _load_prior_args_from_yaml(prior_args_path),
        args.cosmo_model,
    )
    prior_args["prior_kde_source"] = str(kde_path)

    design_path = _get_experiment_config_path("num_visits", "design_args.yaml")
    with open(design_path) as f:
        design_args = yaml.safe_load(f)
    design_args["input_type"] = "nominal"

    from bedcosmo.num_visits.experiment import NumVisits

    print(
        f"\nInitializing NumVisits (transform_input=True, input_transform_type={args.input_transform_type}, "
        f"cdf_samples={args.cdf_samples})..."
    )
    experiment = NumVisits(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model=args.cosmo_model,
        device=args.device,
        cdf_samples=args.cdf_samples,
        cdf_bins=args.cdf_bins,
        transform_input=True,
        input_transform_type=args.input_transform_type,
        joint_transform_shrinkage=args.joint_transform_shrinkage,
        logit_flow_scale=args.logit_flow_scale,
        verbose=True,
    )
    print(f"  input_transform_type: {experiment.input_transform_type}")
    if experiment._uses_joint_transform():
        print(f"  Joint block ({len(experiment._joint_transform_param_names())} params): "
              f"{experiment._joint_transform_param_names()}")
        if experiment.param_bijector.uses_joint_gaussianizer():
            print("  Joint gaussianizer fitted: yes")
    else:
        bj_names = sorted(experiment._bijector_param_names())
        squash = sorted(experiment._flow_squash_param_names())
        print(f"  Marginal bijector on: {bj_names}")
        print(f"  Flow squash (tanh) on: {squash}")
        print(f"  Logit tanh scale: {experiment.logit_flow_scale}")
    if not experiment.use_eazy_sed:
        raise RuntimeError("Expected empirical KDE prior.")

    n = args.n_samples
    physical = _sample_physical_tensor(experiment, n)
    physical = experiment._sanitize_physical_samples(physical)
    with torch.no_grad():
        gaussian = experiment.params_to_unconstrained(physical)
        normal_scores = None
        bj = experiment.param_bijector
        if bj.uses_joint_gaussianizer():
            cols = []
            for name, idx in zip(
                bj.joint_state["param_names"], bj.joint_state["param_indices"]
            ):
                cols.append(
                    bj.prior_to_gaussian(physical[..., idx : idx + 1], name).reshape(-1)
                )
            normal_scores = torch.stack(cols, dim=1)

    names = experiment.cosmo_params
    x_phys = physical.cpu().numpy()
    x_gauss = gaussian.cpu().numpy()

    n_tpl = int(artifact["n_templates"])
    if param == PARAMETERIZATION_ILR:
        a, log_s, z = split_feature_matrix(x_phys, n_tpl, parameterization=param)
        joint_weights = np.column_stack([a, log_s, z])
    else:
        a, log_s, z = split_feature_matrix(x_phys, n_tpl, parameterization=param)
        joint_weights = x_phys

    _print_summary("NF coords f_k, log_c_scale, z (before transform)", x_phys, names)
    _print_summary(
        "Simplex weights a_k (decoded)", joint_weights, [f"a{k}" for k in range(1, n_tpl + 1)] + ["log_c_scale", "z"]
    )
    _print_summary("NF input (after transform_input)", x_gauss, names)
    if normal_scores is not None:
        y_np = normal_scores.cpu().numpy()
        _print_summary(
            "Normal scores Phi^-1(F(x)) before L^-1 (should be ~N(0,1) per dim)",
            y_np,
            names,
        )
    clamp = 6.3613
    print(f"\nFraction at |NF| > {clamp - 0.01:.1f} (bijector saturation):")
    for j, pname in enumerate(names):
        frac = float((np.abs(x_gauss[:, j]) > clamp - 0.01).mean())
        if frac > 0.001:
            print(f"  {pname:14s}  {frac:.3f}")
    rt = _roundtrip_error(experiment, physical)
    print(f"\nMax round-trip |x - inv(T(x))| per parameter: {rt:.3e}")

    outdir = (
        Path(args.outdir).expanduser()
        if args.outdir is not None
        else kde_path.parent / "transform_diagnostic"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    labels_g = _labels_gaussian(experiment)
    wlabels = [rf"$a_{{{k + 1}}}$" for k in range(n_tpl)] + [r"$\log s$", r"$z$"]

    save_triangle_plot(
        outdir,
        joint_weights,
        wlabels,
        filename="triangle_physical_before_transform.png",
        title=rf"Prior on simplex weights ($N={n}$, before transform_input)",
        panel_size=args.panel_size,
    )
    tag = args.input_transform_type
    gauss_name = f"triangle_gaussian_after_transform_{tag}.png"
    save_triangle_plot(
        outdir,
        x_gauss,
        labels_g,
        filename=gauss_name,
        title=rf"NF input after {tag} transform ($N={n}$, $D={len(names)}$)",
        panel_size=args.panel_size,
    )
    print(f"\nSaved {outdir / 'triangle_physical_before_transform.png'}  (a_k, log s, z)")
    print(f"Saved {outdir / gauss_name}")
    if args.input_transform_type == "joint" and normal_scores is not None:
        save_triangle_plot(
            outdir,
            y_np,
            labels_g,
            filename="triangle_normal_scores_before_Linv.png",
            title=rf"Normal scores $y=\Phi^{{-1}}(F(x))$ ($N={n}$, before $L^{{-1}}$)",
            panel_size=args.panel_size,
        )
        print(f"Saved {outdir / 'triangle_normal_scores_before_Linv.png'}")
        print(
            "\nNote: Marginal Gaussianization is in y-space (triangle_normal_scores_before_Linv). "
            "NF input z = L^{-1} y mixes coordinates, so z marginals need not be N(0,1). "
            "Joint whitening targets an approximately factorized Gaussian in z; one-hot f_k "
            "multimodality in x still appears in y and z."
        )

    if args.also_logits_triangle and param == PARAMETERIZATION_ILR:
        labels_phys = _labels_physical(experiment)
        save_triangle_plot(
            outdir,
            x_phys,
            labels_phys,
            filename="triangle_ilr_before_transform.png",
            title=rf"ILR coords f_k ($N={n}$; expect smooth unimodal 1D)",
            panel_size=args.panel_size,
        )
        print(f"Saved {outdir / 'triangle_ilr_before_transform.png'}")
        print(
            "  Note: f_k = log(a_k/a_12) piles up at KDE clip bounds and at "
            "'inactive template' values; use the a_k triangle for physical shape."
        )


if __name__ == "__main__":
    main()
