import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumExtractor
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from prep_shapefit_data import DEFAULT_PRIORS, TARGET_NAMES, latin_hypercube_samples, run_extractor, get_default_save_path
from model import NNRegressor

def run_eval(model_path: str, save_path: str, n_samples: int = 500, seed: int = 42, hist_xlims: dict[str, tuple[float, float]] | None = None) -> None:
    os.makedirs(save_path, exist_ok=True)
    np.random.seed(seed)

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = NNRegressor(
        in_dim=len(ckpt["param_names"]),
        out_dim=len(ckpt["target_names"]),
        hidden_dim=ckpt["hidden_dim"],
        n_hidden=ckpt["n_hidden"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_mu = ckpt["x_mu"].cpu().numpy()
    x_sigma = ckpt["x_sigma"].cpu().numpy()
    y_mu = ckpt["y_mu"].cpu().numpy()
    y_sigma = ckpt["y_sigma"].cpu().numpy()
    param_names = ckpt["param_names"]

    # Sample fresh cosmological parameters from the priors, collecting
    # exactly n_samples accepted (valid) samples.
    extractor = ShapeFitPowerSpectrumExtractor()
    extractor()
    extractor.get()

    true_rows = []
    param_rows = []
    skipped = 0
    batch_seed = seed
    while len(true_rows) < n_samples:
        remaining = n_samples - len(true_rows)
        draws = latin_hypercube_samples(DEFAULT_PRIORS, n_samples=remaining, seed=batch_seed)
        batch_seed += 1
        for sample in draws:
            try:
                targets = run_extractor(extractor, sample)
                vals = [targets[t] for t in TARGET_NAMES]
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

    deltas = (y_pred - y_true) / y_true * 100.0

    # Plot histograms (2x2 grid)
    n_targets = len(TARGET_NAMES)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, name in enumerate(TARGET_NAMES):
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
        ax.set_xlabel(f"$\\Delta$ {name} [%]")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\nmean={mean:.2f}%, std={std:.2f}%")
        if hist_xlims and name in hist_xlims:
            ax.set_xlim(hist_xlims[name])
    fig.suptitle("NN prediction error (% of true value)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "eval_nn.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation plot to: {os.path.join(save_path, 'eval_nn.png')}")

    # --- Triangle plot of cosmo inputs, coloured by outlier status ---
    # A sample is an "outlier" if ANY of its target percentage errors exceed 0.5%
    is_outlier = np.any(np.abs(deltas) > 0.5, axis=1)
    n_params = len(param_names)

    fig2, axes2 = plt.subplots(n_params, n_params, figsize=(3 * n_params, 3 * n_params))
    inlier = ~is_outlier

    for i in range(n_params):
        for j in range(n_params):
            ax = axes2[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(x_raw[inlier, i], bins=30, color="tab:blue", alpha=0.6, label="$\\leq 0.5\\%$")
                ax.hist(x_raw[is_outlier, i], bins=30, color="tab:red", alpha=0.6, label="$> 0.5\\%$")
            else:
                ax.scatter(x_raw[inlier, j], x_raw[inlier, i], s=4, alpha=0.4, color="tab:blue", label="$\\leq 0.5\\%$")
                ax.scatter(x_raw[is_outlier, j], x_raw[is_outlier, i], s=4, alpha=0.6, color="tab:red", label="$> 0.5\\%$")
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(param_names[i])
            else:
                ax.set_yticklabels([])

    # Single shared legend
    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper right", fontsize=12)
    n_outlier = int(is_outlier.sum())
    fig2.suptitle(f"Cosmo inputs — {n_outlier}/{len(is_outlier)} samples with any $|\\mathrm{{error}}| > 0.5\\%$",
                  fontsize=14, y=1.01)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_path, "eval_nn_triangle.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved triangle plot to: {os.path.join(save_path, 'eval_nn_triangle.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ShapeFit NN against the extractor.")
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
        help="Path to model.pt (ignored if --run-dir is set).",
    )
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hist-xlims",
        type=str,
        default=None,
        help='JSON dict mapping target name to [lo, hi], e.g. \'{"qpar": [-1, 1], "qper": [-2, 2]}\'',
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where to save plots (default: run-dir if set, else get_default_save_path()).",
    )
    args = parser.parse_args()

    if args.run_dir is not None:
        model_path = os.path.join(args.run_dir, "model.pt")
        save_path = args.run_dir
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
    else:
        if args.model_path is None:
            raise ValueError("Either --run-dir or --model-path must be set.")
        model_path = args.model_path
        save_path = args.save_path or get_default_save_path()

    #hist_xlims = None
    #if args.hist_xlims is not None:
    #    raw = json.loads(args.hist_xlims)
    hist_xlims = {"qiso": (-0.5, 0.5), "qap": (-0.5, 0.5), "f_sigmar": (-10, 10), "m": (-10, 10)}

    run_eval(model_path, save_path, n_samples=args.n_samples, seed=args.seed, hist_xlims=hist_xlims)


if __name__ == "__main__":
    main()
