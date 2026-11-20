"""Fit and validate a PriorFlow on tempered DR1 BAO samples (``num_tracers.prior``).

The flow is fit in the raw units of a target ``prior_args`` YAML, which may differ
from the sampler's (``prior_args_hrdrag.yaml`` keeps hrdrag in 100 Mpc, multiplier
10000, where ``prior_args.yaml`` keeps Mpc, multiplier 100). Part of the samples is
held out of the fit for validation:

* marginals: KS statistic between flow draws and held-out samples, per parameter;
* calibration: importance weights ``w = p_target / q_flow`` on flow draws, with
  ``p_target = pi_box * L^(1/T)`` from ``TemperedDR1Posterior``: the KL divergence
  from flow to target, the mass with ``|log w| > 1``, and the effective sample size
  fraction (1 for a perfect flow, but set by a few far-tail draws);
* coverage: ``log(p_target / q_flow)`` on held-out samples, against its median. A
  large tail means the flow misses target mass, which flow draws cannot reveal;
* leakage: flow mass outside the box and constraints (``NumTracers`` truncates it);
* a triangle plot of flow draws over held-out samples.

Example::

    python -m bedcosmo.num_tracers.fit_prior_flow \\
        $SCRATCH/bedcosmo/num_tracers/priors/dr1_tempered/base_T16.npz \\
        --prior-args prior_args_hrdrag.yaml
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import ks_2samp

from bedcosmo.num_tracers.prior import TemperedDR1Posterior
from bedcosmo.prior_flow import SPACE_NATIVE, train_prior_flow
from bedcosmo.util import get_experiment_config_path


def _hrdrag_scale(sampler_prior_args, target_prior_args):
    """Factor taking the sampler's raw hrdrag to the target YAML's raw hrdrag."""
    multipliers = []
    for name in (sampler_prior_args, target_prior_args):
        with open(get_experiment_config_path("num_tracers", name)) as f:
            multipliers.append(float(yaml.safe_load(f)["parameters"]["hrdrag"]["multiplier"]))
    return multipliers[0] / multipliers[1]


def validate(pf, target, held_out, scale, n_flow=200_000, seed=1):
    """Validation metrics of flow ``pf`` (target units) against ``target`` (sampler units)."""
    names = target.param_names
    i_hr = names.index("hrdrag")
    to_sampler = np.ones(len(names))
    to_sampler[i_hr] = 1.0 / scale

    draws = pf.sample(n_flow, seed=seed) * to_sampler
    theta = torch.as_tensor(draws, dtype=torch.float64)
    inside = target.in_support(theta).numpy()
    leak = 1.0 - inside.mean()
    draws = draws[inside]

    # log q in sampler units: the flow's density in target units plus log|d x_target / d x_sampler|.
    log_q = pf.log_prob(draws / to_sampler) + math.log(scale) - math.log(inside.mean())
    log_p = target.log_prob(torch.as_tensor(draws, dtype=torch.float64)).numpy()
    log_w = log_p - log_q
    w = np.exp(log_w - log_w.max())
    ess_fraction = w.sum() ** 2 / (w**2).sum() / len(w)
    # log of the target's normalizer relative to q; with it, log_w - log_z = log(p_normalized / q).
    log_z = float(np.log(w.mean()) + log_w.max())

    held = torch.as_tensor(held_out, dtype=torch.float64)
    coverage = target.log_prob(held).numpy() - (
        pf.log_prob(held_out / to_sampler) + math.log(scale) - math.log(inside.mean())
    )
    coverage -= np.median(coverage)

    ks = {n: float(ks_2samp(draws[:, i], held_out[:, i]).statistic) for i, n in enumerate(names)}
    return {
        "ks": ks,
        "ks_max": max(ks.values()),
        "importance_ess_fraction": float(ess_fraction),
        "log_weight_std": float(log_w.std()),
        "kl_flow_to_target_nats": float(log_z - log_w.mean()),
        "frac_abs_log_ratio_gt_1": float(np.mean(np.abs(log_w - log_z) > 1)),
        "coverage_log_ratio_q999": float(np.quantile(coverage, 0.999)),
        "coverage_frac_gt_3": float(np.mean(coverage > 3)),
        "leak_fraction": float(leak),
        "n_flow_draws": n_flow,
    }, draws


def plot(held_out, draws, names, out, title):
    import matplotlib

    matplotlib.use("Agg")
    from getdist import MCSamples, plots

    labels = {
        "Om": r"\Omega_m",
        "Ok": r"\Omega_k",
        "w0": "w_0",
        "wa": "w_a",
        "hrdrag": r"h r_d\,[\mathrm{Mpc}]",
    }
    sets = [
        MCSamples(
            samples=x,
            names=names,
            labels=[labels[n] for n in names],
            label=label,
            settings={"smooth_scale_2D": 0.4, "smooth_scale_1D": 0.4},
        )
        for x, label in ((held_out, "held-out samples"), (draws, "flow (truncated)"))
    ]
    g = plots.get_subplot_plotter(width_inch=2.4 * len(names) + 1)
    g.triangle_plot(
        sets,
        filled=[True, False],
        colors=["#1f77b4", "#d62728"],
        contour_colors=["#1f77b4", "#d62728"],
        legend_loc="upper right",
    )
    g.fig.legends[0].set_title(title, prop={"size": 12})
    g.export(str(out), dpi=110)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("samples", help="Sampler .npz from bedcosmo.num_tracers.prior.")
    ap.add_argument(
        "--prior-args",
        default="prior_args_hrdrag.yaml",
        help="YAML whose raw units the flow is fit in.",
    )
    ap.add_argument(
        "--out", default=None, help="Flow .pt (default: <samples>_flow.pt beside the samples)."
    )
    ap.add_argument("--held-out", type=float, default=0.2)
    ap.add_argument(
        "--early-stop-fraction",
        type=float,
        default=0.1,
        help="Share of the fit samples train_prior_flow keeps for early stopping.",
    )
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--transforms", type=int, default=6)
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    path = Path(args.samples).expanduser()
    f = np.load(path)
    samples, meta = f["samples"], json.loads(str(f["metadata"]))
    names = meta["param_names"]
    target = TemperedDR1Posterior(
        meta["cosmo_model"], meta["temperature"], Path(meta["prior_args_path"]).name
    )
    scale = _hrdrag_scale(Path(meta["prior_args_path"]).name, args.prior_args)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(samples))
    n_held = int(args.held_out * len(samples))
    held_out, fit = samples[perm[:n_held]], samples[perm[n_held:]]
    to_target = np.ones(len(names))
    to_target[names.index("hrdrag")] = scale

    out = Path(args.out).expanduser() if args.out else path.with_name(path.stem + "_flow.pt")
    pf = train_prior_flow(
        fit * to_target,
        n_train=int((1 - args.early_stop_fraction) * len(fit)),
        space=SPACE_NATIVE,
        feature_names=names,
        epochs=args.epochs,
        transforms=args.transforms,
        bins=args.bins,
        hidden_features=tuple(args.hidden),
        seed=args.seed,
        device=args.device,
        log_path=out.with_suffix(".train.log"),
    )
    metrics, draws = validate(pf, target, held_out, scale)
    pf.meta.update(
        cosmo_model=meta["cosmo_model"],
        temperature=meta["temperature"],
        prior_args=args.prior_args,
        source_samples=str(path),
        sampler_metadata=meta,
        validation=metrics,
    )
    pf.save(out)
    plot(
        held_out,
        draws,
        names,
        out.with_suffix(".png"),
        f"{meta['cosmo_model']} T={meta['temperature']:g}",
    )
    print(f"flow -> {out}")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
