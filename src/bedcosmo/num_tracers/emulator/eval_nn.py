import argparse
import json
import math
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from util import latin_hypercube_samples, get_default_save_path, get_pipeline, TRACER_TYPE_CHOICES, build_model

def _log_bins(vals: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """Return histogram bin edges appropriate for log/symlog data."""
    if np.all(vals > 0):
        return np.geomspace(vals.min(), vals.max(), n_bins + 1)
    # Symlog: evenly space bins in transformed coordinates
    abs_nonzero = np.abs(vals[vals != 0])
    linthresh = float(np.min(abs_nonzero)) if len(abs_nonzero) > 0 else 1e-6
    def fwd(x):
        return np.sign(x) * np.log10(1 + np.abs(x) / linthresh)
    def inv(y):
        return np.sign(y) * linthresh * (10 ** np.abs(y) - 1)
    t_min, t_max = fwd(vals.min()), fwd(vals.max())
    return inv(np.linspace(t_min, t_max, n_bins + 1))


def _set_plain_formatter(ax) -> None:
    """Ensure tick labels use compact text, not LaTeX scientific notation."""
    for axis in (ax.xaxis, ax.yaxis):
        scale = axis.get_scale()
        if scale == "linear":
            axis.set_major_locator(mticker.MaxNLocator(nbins=5, min_n_ticks=3))
        elif scale == "log":
            vmin, vmax = axis.get_view_interval()
            if vmin > 0 and vmax > 0 and vmax / vmin < 30:
                axis.set_major_locator(mticker.LogLocator(subs=[1, 2, 5], numticks=8))
        # Use plain '1e-3' style, not '$1 \times 10^{-3}$'
        axis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: "" if v == 0 else f"{v:.3g}"
        ))
        axis.set_minor_formatter(mticker.NullFormatter())
        axis.get_offset_text().set_visible(False)


def _set_log_or_symlog(ax, axis: str, vals: np.ndarray) -> None:
    """Set log scale for all-positive data, symlog for mixed-sign data."""
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    if np.all(vals > 0):
        setter("log")
    else:
        abs_nonzero = np.abs(vals[vals != 0])
        linthresh = float(np.min(abs_nonzero)) if len(abs_nonzero) > 0 else 1e-6
        setter("symlog", linthresh=linthresh)




def run_eval(model_path: str, save_path: str, analysis: str = "shapefit", quantity: str = "covar", n_samples: int = 500, seed: int = 42, hist_xlims: dict[str, tuple[float, float]] | None = None, rtol: float = 5e-3, atol: float = 1e-4, log_scale: bool = False, rescale_by: str | None = None, tracer: str | None = None) -> None:
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.random.seed(seed)

    # Load trained model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(
        analysis=ckpt.get("analysis", "bao"),
        architecture=ckpt.get("architecture", "resnet"),
        in_dim=len(ckpt["param_names"]),
        out_dim=len(ckpt["target_names"]),
        hidden_dim=ckpt["hidden_dim"],
        n_hidden=ckpt["n_hidden"],
        dropout=ckpt.get("dropout", 0.0),
        expand=ckpt.get("expand", 4),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_mu = ckpt["x_mu"].cpu().numpy()
    x_sigma = ckpt["x_sigma"].cpu().numpy()
    y_mu = ckpt["y_mu"].cpu().numpy()
    y_sigma = ckpt["y_sigma"].cpu().numpy()
    log_normalize = ckpt.get("log_normalize", False)
    y_linthresh = ckpt["y_linthresh"].cpu().numpy() if ckpt.get("y_linthresh") is not None else None
    param_names = ckpt["param_names"]
    ckpt_target_names = ckpt["target_names"]

    default_priors, target_names, ground_truth_fn, setup = get_pipeline(analysis, quantity, tracer=tracer)

    true_rows = []
    param_rows = []
    skipped = 0
    batch_seed = seed
    while len(true_rows) < n_samples:
        remaining = n_samples - len(true_rows)
        draws = latin_hypercube_samples(default_priors, n_samples=remaining, seed=batch_seed)
        batch_seed += 1
        for sample in draws:
            try:
                targets = ground_truth_fn(setup, sample)
                vals = [targets[t] for t in target_names]
                if not all(np.isfinite(vals)):
                    skipped += 1
                    continue
                true_rows.append(vals)
                param_rows.append([sample[p] for p in param_names])
            except Exception:
                skipped += 1
                continue
            if len(true_rows) >= n_samples:
                break

    print(f"Collected {len(true_rows)} accepted samples ({skipped} rejected).")

    y_true = np.array(true_rows, dtype=np.float32)
    x_raw = np.array(param_rows, dtype=np.float32)

    # NN predictions (standardize -> predict -> unstandardize)
    x_norm = (x_raw - x_mu) / x_sigma
    with torch.no_grad():
        y_pred_norm = model(torch.from_numpy(x_norm).to(device)).cpu().numpy()
    y_pred = y_pred_norm * y_sigma + y_mu
    if log_normalize and y_linthresh is not None:
        # Invert symlog: sign(z) * linthresh * expm1(|z|)
        y_pred = np.sign(y_pred) * y_linthresh * np.expm1(np.abs(y_pred))

    # Undo input-parameter scaling applied during training (e.g. targets were
    # multiplied by N_tracers before training).  Divide predictions by that
    # input to recover the original physical scale for comparison with truth.
    if rescale_by is not None:
        idx = param_names.index(rescale_by)
        y_pred = y_pred / x_raw[:, idx : idx + 1]

    # Absolute error for covariance elements (off-diagonal can be near zero),
    # percentage error for other targets.
    is_cov = target_names[0].startswith("cov_")
    if is_cov:
        deltas = y_pred - y_true
        delta_label = "absolute error"
        delta_unit = ""
    else:
        deltas = (y_pred - y_true) / y_true * 100.0
        delta_label = "% error"
        delta_unit = " [%]"

    # Plot histograms (dynamic grid)
    n_targets = len(target_names)
    ncols = min(n_targets, 4)
    nrows = math.ceil(n_targets / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i, name in enumerate(target_names):
        ax = axes[i]
        if hist_xlims and name in hist_xlims:
            lo, hi = hist_xlims[name]
            bins = np.linspace(lo, hi, 41)
        else:
            bins = 40
        ax.hist(deltas[:, i], bins=bins, range=(lo, hi) if hist_xlims and name in hist_xlims else None, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        mean = np.mean(deltas[:, i])
        std = np.std(deltas[:, i])
        ax.set_xlabel(f"$\\Delta$ {name}{delta_unit}")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\nmean={mean:.2e}, std={std:.2e}")
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(f"NN prediction {delta_label}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"eval_nn_{timestamp}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation plot to: {os.path.join(save_path, f'eval_nn_{timestamp}.png')}")

    # --- Triangle plot of cosmo inputs, coloured by outlier status ---
    # A sample is an "outlier" if ANY target fails np.isclose
    is_close = np.isclose(y_pred, y_true, rtol=rtol, atol=atol)
    is_outlier = ~np.all(is_close, axis=1)
    n_params = len(param_names)

    panel_size = 4 if log_scale else 3
    fig2, axes2 = plt.subplots(n_params, n_params, figsize=(panel_size * n_params, panel_size * n_params))
    inlier = ~is_outlier

    for i in range(n_params):
        for j in range(n_params):
            ax = axes2[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                all_vals = x_raw[:, i]
                bins = _log_bins(all_vals) if log_scale else np.linspace(all_vals.min(), all_vals.max(), 31)
                ax.hist(x_raw[inlier, i], bins=bins, color="tab:blue", alpha=0.6, label="pass")
                ax.hist(x_raw[is_outlier, i], bins=bins, color="tab:red", alpha=0.6, label="fail")
            else:
                ax.scatter(x_raw[inlier, j], x_raw[inlier, i], s=4, alpha=0.4, color="tab:blue", label="pass")
                ax.scatter(x_raw[is_outlier, j], x_raw[is_outlier, i], s=4, alpha=0.6, color="tab:red", label="fail")
                if log_scale:
                    _set_log_or_symlog(ax, "x", x_raw[:, j])
                    _set_log_or_symlog(ax, "y", x_raw[:, i])
            if log_scale and i == j:
                _set_log_or_symlog(ax, "x", x_raw[:, i])
            _set_plain_formatter(ax)
            ax.tick_params(axis="both", labelsize=8)
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(param_names[i])
            else:
                ax.tick_params(labelleft=False)

    # Single shared legend
    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper right", fontsize=12)
    n_outlier = int(is_outlier.sum())
    fig2.suptitle(f"Cosmo inputs — {n_outlier}/{len(is_outlier)} fail (rtol={rtol}, atol={atol})",
                  fontsize=14, y=1.01)
    fig2.tight_layout()
    # Re-apply formatter and label visibility after tight_layout
    for i in range(n_params):
        for j in range(n_params):
            ax = axes2[i, j]
            if not ax.get_visible():
                continue
            _set_plain_formatter(ax)
            if i != n_params - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)
    fig2.savefig(os.path.join(save_path, f"eval_nn_triangle_{timestamp}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved triangle plot to: {os.path.join(save_path, f'eval_nn_triangle_{timestamp}.png')}")

    # --- Triangle plot of target outputs, coloured by outlier status ---
    target_names = list(ckpt_target_names)
    n_tgt = len(target_names)
    fig3, axes3 = plt.subplots(n_tgt, n_tgt, figsize=(panel_size * n_tgt, panel_size * n_tgt))

    for i in range(n_tgt):
        for j in range(n_tgt):
            ax = axes3[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                all_vals = y_true[:, i]
                bins = _log_bins(all_vals) if log_scale else np.linspace(all_vals.min(), all_vals.max(), 31)
                ax.hist(y_true[inlier, i], bins=bins, color="tab:blue", alpha=0.6, label="pass")
                ax.hist(y_true[is_outlier, i], bins=bins, color="tab:red", alpha=0.6, label="fail")
            else:
                ax.scatter(y_true[inlier, j], y_true[inlier, i], s=4, alpha=0.4, color="tab:blue", label="pass")
                ax.scatter(y_true[is_outlier, j], y_true[is_outlier, i], s=4, alpha=0.6, color="tab:red", label="fail")
                if log_scale:
                    _set_log_or_symlog(ax, "x", y_true[:, j])
                    _set_log_or_symlog(ax, "y", y_true[:, i])
            if log_scale and i == j:
                _set_log_or_symlog(ax, "x", y_true[:, i])
            _set_plain_formatter(ax)
            ax.tick_params(axis="both", labelsize=8)
            if i == n_tgt - 1:
                ax.set_xlabel(target_names[j])
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(target_names[i])
            else:
                ax.tick_params(labelleft=False)

    handles, labels = axes3[0, 0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc="upper right", fontsize=12)
    n_outlier = int(is_outlier.sum())
    fig3.suptitle(f"Target outputs — {n_outlier}/{len(is_outlier)} fail (rtol={rtol}, atol={atol})",
                  fontsize=14, y=1.01)
    fig3.tight_layout()
    for i in range(n_tgt):
        for j in range(n_tgt):
            ax = axes3[i, j]
            if not ax.get_visible():
                continue
            _set_plain_formatter(ax)
            if i != n_tgt - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)
    fig3.savefig(os.path.join(save_path, f"eval_nn_triangle_targets_{timestamp}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved target triangle plot to: {os.path.join(save_path, f'eval_nn_triangle_targets_{timestamp}.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ShapeFit NN against the extractor.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID. Resolves model.pt and save path from the run's artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory from a training run (contains model.pt). Plots are saved here.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model.pt (ignored if --run-id or --run-dir is set).",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hist-xlims",
        type=str,
        default=None,
        help='JSON dict mapping target name to [lo, hi], e.g. \'{"qpar": [-1, 1], "qper": [-2, 2]}\'',
    )
    parser.add_argument("--rtol", type=float, default=2e-3, help="Relative tolerance for allclose outlier check (default: 2e-3).")
    parser.add_argument("--atol", type=float, default=2e-3, help="Absolute tolerance for allclose outlier check (default: 2e-3).")
    parser.add_argument("--log-scale", action="store_true", help="Use log scale on triangle plot axes.")
    parser.add_argument("--rescale-by", type=str, default=None,
                        help="Divide predictions by this input parameter before comparing to truth (e.g. 'N_tracers').")
    parser.add_argument(
        "--analysis",
        type=str,
        default="shapefit",
        help="Analysis type: 'shapefit' or 'bao'.",
    )
    parser.add_argument(
        "--quantity",
        type=str,
        default="covar",
        help="Quantity: 'covar' or 'mean'.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where to save plots (default: resolved from run-id/run-dir, else get_default_save_path()).",
    )
    parser.add_argument(
        "--tracer",
        type=str,
        default=None,
        choices=TRACER_TYPE_CHOICES,
        help="Tracer type (e.g. BGS, LRG1). Sets zrange, z_eff, and N_tracers bounds for evaluation.",
    )
    args = parser.parse_args()

    if args.run_id is not None:
        import mlflow
        scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
        mlflow.set_tracking_uri(f"file:{scratch}/bedcosmo/shapefit_emulator/mlruns")
        run = mlflow.get_run(args.run_id)
        artifact_uri = run.info.artifact_uri
        if artifact_uri.startswith("file://"):
            artifact_uri = artifact_uri[7:]
        model_path = os.path.join(artifact_uri, "model.pt")
        save_path = args.save_path or artifact_uri
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
    elif args.run_dir is not None:
        model_path = os.path.join(args.run_dir, "model.pt")
        save_path = args.save_path or args.run_dir
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
    else:
        if args.model_path is None:
            raise ValueError("Either --run-id, --run-dir, or --model-path must be set.")
        model_path = args.model_path
        save_path = args.save_path or get_default_save_path(analysis=args.analysis, quantity=args.quantity)

    hist_xlims = None
    if args.hist_xlims is not None:
        raw = json.loads(args.hist_xlims)
        hist_xlims = {k: tuple(v) for k, v in raw.items()}

    run_eval(model_path, save_path, analysis=args.analysis, quantity=args.quantity, n_samples=args.n_samples, seed=args.seed, hist_xlims=hist_xlims, rtol=args.rtol, atol=args.atol, log_scale=args.log_scale, rescale_by=args.rescale_by, tracer=args.tracer)


if __name__ == "__main__":
    main()
