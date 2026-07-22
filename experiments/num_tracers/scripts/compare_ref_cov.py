"""Visualize the difference between two NumTracers reference covariances.

Compares a generated reference covariance (e.g. the emulator-based one from
``generate_ref_cov.py``) against the dataset default ``desi_cov.npy``. Both are
(n_data, n_data) matrices in the same desi_data row order, so they are compared
element-wise. Produces one figure:

  - per-row sigma = sqrt(diag): emulator vs DESI, with the emulator/DESI ratio
  - DESI correlation matrix  and  emulator correlation matrix (block-diagonal)
  - covariance ratio  ref / desi  within each populated block (cross-tracer zeros
    are masked), so tighter (<1) vs looser (>1) is visible per element

Run (in the `bedcosmo` env):
    python experiments/num_tracers/scripts/compare_ref_cov.py \
        --ref /pscratch/.../ref_cov_emulator_dr1_base.npy --dataset dr1
"""
import argparse
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito colorblind-safe pair for the two series (identity, not magnitude).
C_EMU = "#0072B2"   # blue  -> generated (emulator) reference
C_DESI = "#E69F00"  # orange -> DESI published default


def _corr(cov):
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def _parse_args():
    p = argparse.ArgumentParser(description="Compare two reference covariances")
    p.add_argument("--ref", required=True, help="Path to the generated reference cov .npy")
    p.add_argument("--dataset", default="dr1", help="DESI dataset for the default desi_cov.npy")
    p.add_argument("--default", default=None,
                   help="Override path to the baseline cov (default: dataset desi_cov.npy)")
    p.add_argument("--output", default=None, help="Output .png path")
    return p.parse_args()


def main():
    args = _parse_args()
    data_dir = os.path.join(os.environ["HOME"], f"data/desi/bao_{args.dataset}")
    default_path = args.default or os.path.join(data_dir, "desi_cov.npy")

    ref = np.load(os.path.expanduser(args.ref)).astype(np.float64)
    desi = np.load(default_path).astype(np.float64)
    if ref.shape != desi.shape:
        raise ValueError(f"shape mismatch: ref {ref.shape} vs default {desi.shape}")
    dd = pd.read_csv(os.path.join(data_dir, "desi_data.csv"))
    labels = [f"{t}\n{q.replace('_over_rs', '')}" for t, q in zip(dd["tracer"], dd["quantity"])]
    n = ref.shape[0]

    sig_ref, sig_desi = np.sqrt(np.diag(ref)), np.sqrt(np.diag(desi))
    ratio = sig_ref / sig_desi
    corr_ref, corr_desi = _corr(ref), _corr(desi)

    # Covariance ratio within populated blocks; mask cross-tracer zeros (0/0).
    with np.errstate(divide="ignore", invalid="ignore"):
        cov_ratio = np.where((desi != 0) | (ref != 0), ref / desi, np.nan)
    cov_ratio = np.ma.masked_invalid(cov_ratio)

    # ---- console summary ----
    print(f"\nref     : {args.ref}")
    print(f"default : {default_path}   shape {ref.shape}")
    print(f"\n{'row':>3} {'tracer':<11}{'quantity':<10}{'emu_sigma':>11}{'DESI_sigma':>12}{'ratio':>8}")
    for i in range(n):
        print(f"{i:>3} {dd['tracer'].iloc[i]:<11}{dd['quantity'].iloc[i]:<10}"
              f"{sig_ref[i]:>11.5f}{sig_desi[i]:>12.5f}{ratio[i]:>8.3f}")
    print(f"\nsigma ratio (emu/DESI): min={ratio.min():.3f} max={ratio.max():.3f} "
          f"median={np.median(ratio):.3f}")

    # ---- figure ----
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.42, wspace=0.28)

    # (A) sigma comparison bars, spanning the top row.
    axA = fig.add_subplot(gs[0, :])
    x = np.arange(n)
    w = 0.4
    axA.bar(x - w / 2, sig_ref, w, label="generated (emulator) ref", color=C_EMU)
    axA.bar(x + w / 2, sig_desi, w, label="DESI published (default)", color=C_DESI, alpha=0.95)
    for i in range(n):
        axA.annotate(f"{ratio[i]:.2f}", (x[i], max(sig_ref[i], sig_desi[i])),
                     textcoords="offset points", xytext=(0, 3), ha="center",
                     fontsize=7, color="#333333")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=7)
    axA.set_ylabel(r"$\sigma = \sqrt{\mathrm{diag}}$  (D/$r_d$)")
    axA.set_title("Per-row error: generated emulator reference vs DESI default "
                  "(number = emulator/DESI ratio)")
    axA.legend(frameon=False)
    axA.grid(axis="y", alpha=0.3)

    # (B, C) correlation heatmaps, diverging RdBu_r centered at 0.
    for ax, mat, title in ((fig.add_subplot(gs[1, 0]), corr_desi, "DESI correlation"),
                           (fig.add_subplot(gs[1, 1]), corr_ref, "emulator correlation")):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(dd["tracer"], rotation=90, fontsize=5)
        ax.set_yticks(x); ax.set_yticklabels(dd["tracer"], fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (D) covariance ratio ref/desi, diverging around 1 (tighter<1 / looser>1).
    axD = fig.add_subplot(gs[1, 2])
    vmax = float(np.nanmax(np.abs(np.log2(cov_ratio.filled(1.0)))))
    vmax = max(vmax, 0.1)
    cmap = plt.get_cmap("PuOr").copy()
    cmap.set_bad("#dddddd")
    imD = axD.imshow(np.log2(cov_ratio), cmap=cmap, vmin=-vmax, vmax=vmax)
    axD.set_title(r"$\log_2(\mathrm{cov}_{\rm emu}/\mathrm{cov}_{\rm DESI})$"
                  "\n(<0 tighter, gray = both zero)", fontsize=10)
    axD.set_xticks(x); axD.set_xticklabels(dd["tracer"], rotation=90, fontsize=5)
    axD.set_yticks(x); axD.set_yticklabels(dd["tracer"], fontsize=5)
    fig.colorbar(imD, ax=axD, fraction=0.046, pad=0.04)

    out = args.output
    if out is None:
        scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
        out_dir = os.path.join(scratch, "bedcosmo", "num_tracers", "ref_cov")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(out_dir, f"compare_ref_cov_{args.dataset}_{ts}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
