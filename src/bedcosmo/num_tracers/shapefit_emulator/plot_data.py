import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ShapeFit train/test dataset diagnostics.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(
            os.environ.get("SCRATCH", ""),
            "bedcosmo",
            "num_tracers",
            "shapefit",
        ),
        help="Directory containing shapefit_train.npz and shapefit_test.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Where to save plots (defaults to --data-path)",
    )
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_path)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else data_path
    os.makedirs(output_dir, exist_ok=True)

    train = np.load(os.path.join(data_path, "shapefit_train.npz"), allow_pickle=True)
    test = np.load(os.path.join(data_path, "shapefit_test.npz"), allow_pickle=True)

    x_train, y_train = train["x"], train["y"]
    x_test, y_test = test["x"], test["y"]
    param_names = train["param_names"].tolist()
    target_names = train["target_names"].tolist()

    print(f"Loaded train X/Y: {x_train.shape} {y_train.shape}")
    print(f"Loaded test  X/Y: {x_test.shape} {y_test.shape}")
    print(f"Saving plots to: {output_dir}")

    # Inputs: train vs test histograms
    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 3))
    if len(param_names) == 1:
        axes = [axes]
    for i, name in enumerate(param_names):
        axes[i].hist(x_train[:, i], bins=args.bins, alpha=0.6, density=True, label="train")
        axes[i].hist(x_test[:, i], bins=args.bins, alpha=0.6, density=True, label="test")
        axes[i].set_title(name)
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "inputs_train_test_hist.png"), dpi=150)
    plt.close(fig)

    # Targets: train vs test histograms
    fig, axes = plt.subplots(1, len(target_names), figsize=(4 * len(target_names), 3))
    if len(target_names) == 1:
        axes = [axes]
    for i, name in enumerate(target_names):
        axes[i].hist(y_train[:, i], bins=args.bins, alpha=0.6, density=True, label="train")
        axes[i].hist(y_test[:, i], bins=args.bins, alpha=0.6, density=True, label="test")
        axes[i].set_title(name)
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "targets_train_test_hist.png"), dpi=150)
    plt.close(fig)

    # Derived Omega_m vs qiso scatter
    omega_cdm_idx = param_names.index("omega_cdm")
    omega_b_idx = param_names.index("omega_b")
    h_idx = param_names.index("h")
    qiso_idx = target_names.index("qiso")
    omega_m = (x_train[:, omega_cdm_idx] + x_train[:, omega_b_idx]) / (x_train[:, h_idx] ** 2)
    qiso = y_train[:, qiso_idx]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(omega_m, qiso, s=4, alpha=0.3)
    ax.set_xlabel("Omega_m (derived)")
    ax.set_ylabel("qiso")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "omega_m_vs_qiso.png"), dpi=150)
    plt.close(fig)

    print("Saved:")
    print(f"  {os.path.join(output_dir, 'inputs_train_test_hist.png')}")
    print(f"  {os.path.join(output_dir, 'targets_train_test_hist.png')}")
    print(f"  {os.path.join(output_dir, 'omega_m_vs_qiso.png')}")


if __name__ == "__main__":
    main()

