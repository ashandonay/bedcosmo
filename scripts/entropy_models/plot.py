"""Plots for the toy entropy validation (companion to validation.py).

Two separate stories, kept in separate figures so they don't get conflated:

  A. FULL-RANK estimator accuracy (the "Gaussian + transforms" story):
       1. toy_distributions.png -- each well-posed toy as a 2x2 corner plot in
          copula coordinates: every margin is pushed to N(0,1) (diagonal bells,
          red reference), so the off-diagonal joint IS the copula -- round
          (independent), tilted ellipse (Gaussian copula), curved (non-Gaussian
          copula). Makes "marginally normal != jointly normal" explicit.
       1b. copula_lens.png -- the before/after companion, as corner plots so the
          margins show too: top row the raw joint + raw margins (the warp/skew is
          visible on the diagonal), bottom row after normal-scores (margins -> N(0,1)
          bells). Shows the marginal transform erasing the margins and exposing the
          copula (the warp vanishes; the banana stays curved).
       2. toy_entropy_bias.png  -- grouped bars of each estimator's bias (bits)
          vs the known H on those three toys. The point: a Gaussian-copula
          estimator handles warped margins but fails on nonlinear dependence
          (banana), where the flow recovers H. y = 0 is truth.

  B. RANK-DEFICIENCY / parameterization (the "degenerate" story, the known-truth
     twin of ilr_fix.png -- this is NOT an estimator benchmark):
       3. toy_degenerate.png -- the degenerate toy as its ambient 3D flat sheet
          (H=-inf), its ILR projection (a healthy 2D blob, H finite), and the
          ambient covariance eigenvalue cliff. A caption states the truth: ambient
          H=-inf, ILR H = the known finite value recovered exactly (the projection
          is an orthonormal isometry), and that no estimator can rescue the ambient
          case -- only dropping to the intrinsic dimension can.

Estimator colors are the Okabe-Ito colorblind-safe categorical palette, one fixed
hue per estimator identity (never cycled).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.stats import norm, rankdata  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root, for the default output dir
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from estimators import (  # noqa: E402
    flow_plugin_entropy,
    gaussian_entropy,
    make_correlated_nongaussian,
)
from validation import (  # noqa: E402
    copula_plugin_entropy,
    knn_plugin_entropy,
    make_banana,
    make_degenerate,
    make_gaussian,
)

BITS = 1.0 / math.log(2.0)

# Okabe-Ito colorblind-safe categorical palette, fixed order by estimator identity.
EST_ORDER = ["gaussian(cov)", "copula", "knn", "flow"]
EST_COLORS = {
    "gaussian(cov)": "#999999",  # neutral gray -- the naive baseline
    "copula": "#E69F00",  # orange
    "knn": "#56B4E9",  # sky blue
    "flow": "#009E73",  # bluish green
}
DENSITY_CMAP = "viridis"  # sequential, perceptually uniform, CVD-safe


# ----------------------------------------------------------------------
# Entropy estimates + shared bar drawing
# ----------------------------------------------------------------------


def _safe(fn):
    try:
        return fn()
    except ValueError:
        return None  # rank-deficient input: estimator undefined


def estimate_all(x, *, epochs, seed=0):
    """All five estimators (nats) on rows ``x``; None where undefined."""
    return {
        "gaussian(cov)": _safe(lambda: gaussian_entropy(np.cov(x, rowvar=False))),
        "copula": _safe(lambda: copula_plugin_entropy(x, seed=seed)),
        "knn": knn_plugin_entropy(x, seed=seed),
        "flow": flow_plugin_entropy(
            x, epochs=epochs, hidden_features=(128, 128), batch_size=4096, seed=seed
        ),
    }


def compute_biases(cases, *, epochs, seed):
    """[(label, {estimator: bias_bits or None})] for (label, rows, h_true_nats)."""
    out = []
    for label, x, h_true in cases:
        est = estimate_all(x, epochs=epochs, seed=seed)
        out.append(
            (label, {k: (None if v is None else (v - h_true) * BITS) for k, v in est.items()})
        )
    return out


def draw_bias_bars(ax, data, title):
    """Grouped estimator-bias bars; y=0 is truth, undefined estimators marked n/a."""
    nE = len(EST_ORDER)
    width = 0.8 / nE
    labels = [d[0] for d in data]
    xpos = np.arange(len(labels))
    for j, est in enumerate(EST_ORDER):
        vals = [d[1][est] for d in data]
        offs = xpos + (j - (nE - 1) / 2) * width
        heights = [0.0 if v is None else v for v in vals]
        bars = ax.bar(offs, heights, width * 0.92, color=EST_COLORS[est], label=est, zorder=3)
        for rect, v in zip(bars, vals):
            if v is None:  # undefined estimator -> mark n/a at baseline
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    0,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    color="#555",
                )
    ax.axhline(0, color="#333", lw=1.2, zorder=2)  # y=0 == truth
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color="#ddd", lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def _normal_scores(x):
    """Per-margin empirical PIT to a standard normal (rank -> Phi^{-1}).

    The copula lens: standardizes every marginal to N(0,1) without touching the
    dependence, so the joint that remains *is* the copula.
    """
    z = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        u = rankdata(x[:, j]) / (len(x) + 1.0)
        z[:, j] = norm.ppf(u)
    return z


def _corner(fig, subgs, data, title, *, xlabel, ylabel, ref_normal, box, title_fs=10):
    """2x2 corner: the two margins on the diagonal, the joint off-diagonal.

    ``ref_normal`` overlays a red N(0,1) curve on the margins (use in copula
    coordinates, where they should be bells). ``box=(lo, hi)`` fixes a square,
    equal-aspect joint window; ``box=None`` autoscales (for raw data whose scale
    varies, e.g. the banana).
    """
    ax_j = fig.add_subplot(subgs[1, 0])
    ax_x = fig.add_subplot(subgs[0, 0], sharex=ax_j)
    ax_y = fig.add_subplot(subgs[1, 1], sharey=ax_j)

    if box is not None:
        ax_j.hexbin(
            data[:, 0],
            data[:, 1],
            gridsize=40,
            cmap=DENSITY_CMAP,
            mincnt=1,
            extent=(box[0], box[1], box[0], box[1]),
        )
        ax_j.set_xlim(*box)
        ax_j.set_ylim(*box)
        ax_j.set_aspect("equal", adjustable="box")
    else:
        ax_j.hexbin(data[:, 0], data[:, 1], gridsize=40, cmap=DENSITY_CMAP, mincnt=1)
    ax_j.set_xlabel(xlabel)
    ax_j.set_ylabel(ylabel)

    ax_x.hist(data[:, 0], bins=60, density=True, color="#9ecae1")
    ax_y.hist(data[:, 1], bins=60, density=True, orientation="horizontal", color="#9ecae1")
    if ref_normal:
        grid = np.linspace(-4, 4, 200)
        ref = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
        ax_x.plot(grid, ref, color="tab:red", lw=1.0)
        ax_y.plot(ref, grid, color="tab:red", lw=1.0)

    for a in (ax_x, ax_y):
        a.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    ax_x.set_yticks([])
    ax_y.set_xticks([])
    ax_x.set_title(title, fontsize=title_fs)


def _scatter_ellipses(ax, data, title, xlabel, ylabel, n_show=2500):
    """Scatter + covariance sigma-ellipses -- clean way to show a full-rank Gaussian blob."""
    sub = data[:: max(1, len(data) // n_show)]
    ax.scatter(sub[:, 0], sub[:, 1], s=6, alpha=0.25, color="#0072B2", edgecolors="none")
    mu = data.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(data, rowvar=False))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for k in (1, 2):  # 1/2-sigma contours
        ax.add_patch(
            Ellipse(
                mu,
                *(2 * k * np.sqrt(vals)),
                angle=angle,
                fill=False,
                edgecolor="tab:blue",
                lw=1.4,
                alpha=0.9,
            )
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")


# ----------------------------------------------------------------------
# Figure 1 (A): full-rank toy distributions
# ----------------------------------------------------------------------


def plot_distributions(n, seed, out_path):
    toys = [
        ("gaussian toy", make_gaussian(2, n, seed=seed)[0], "Gaussian copula (a plain blob)"),
        (
            "warped-margins toy",
            make_correlated_nongaussian(2, n, seed=seed)[0],
            "same Gaussian copula\n(the warp was only marginal)",
        ),
        (
            "banana toy",
            make_banana(2, n, seed=seed)[0],
            "non-Gaussian copula\n(curved -- survives)",
        ),
    ]

    fig = plt.figure(figsize=(14, 5.6))
    outer = fig.add_gridspec(1, 3, wspace=0.30)
    for k, (name, x, note) in enumerate(toys):
        z = _normal_scores(x)  # copula coordinates: every margin now N(0,1)
        sub = outer[0, k].subgridspec(
            2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05
        )
        _corner(
            fig,
            sub,
            z,
            f"{name}\n$\\rightarrow$ {note}",
            xlabel="$z_0$",
            ylabel="$z_1$",
            ref_normal=True,
            box=(-4, 4),
        )

    fig.suptitle(
        "Margins forced to $N(0,1)$: the off-diagonal joint IS the copula.  First two are "
        "Gaussian copulas (plain blobs); only the banana's dependence stays non-Gaussian.",
        fontsize=11.5,
    )
    fig.subplots_adjust(left=0.05, right=0.97, top=0.83, bottom=0.09)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_copula_lens(n, seed, out_path):
    """Raw joint (top) vs the copula (bottom): the marginal transform, shown at work.

    The before/after companion to toy_distributions.png. Top row is each raw toy;
    the bottom row applies normal-scores so every margin becomes N(0,1) and only the
    copula (dependence) remains. The warped toy's raw non-Gaussianity (boxy margins)
    vanishes into a plain Gaussian-copula blob; the banana stays curved.
    """
    toys = [
        ("gaussian", make_gaussian(2, n, seed=seed)[0]),
        ("warped-margins", make_correlated_nongaussian(2, n, seed=seed)[0]),
        ("banana", make_banana(2, n, seed=seed)[0]),
    ]

    fig = plt.figure(figsize=(14, 9.6))
    outer = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.34)
    for k, (name, x) in enumerate(toys):
        # top: raw joint + raw margins -- the warp/skew is visible on the diagonal
        sub_top = outer[0, k].subgridspec(
            2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05
        )
        _corner(
            fig,
            sub_top,
            x,
            f"raw: {name}",
            xlabel="$x_0$",
            ylabel="$x_1$",
            ref_normal=False,
            box=None,
        )
        # bottom: margins forced to N(0,1) (bells) -> only the copula (joint) remains
        z = _normal_scores(x)
        sub_bot = outer[1, k].subgridspec(
            2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05
        )
        _corner(
            fig,
            sub_bot,
            z,
            "after marginal gaussianization:\n2D copula + $N(0,1)$ marginal",
            xlabel="$z_0$",
            ylabel="$z_1$",
            ref_normal=True,
            box=(-4, 4),
            title_fs=9,
        )

    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ----------------------------------------------------------------------
# Figure 2 (A): full-rank estimator bias
# ----------------------------------------------------------------------


def plot_bias(n, seed, epochs, out_path):
    xg, hg = make_gaussian(2, n, seed=seed)
    xw, hw = make_correlated_nongaussian(2, n, seed=seed)
    xb, hb = make_banana(2, n, seed=seed)
    cases = [("gaussian", xg, hg), ("warped-\nmargins", xw, hw), ("banana", xb, hb)]

    data = compute_biases(cases, epochs=epochs, seed=seed)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    draw_bias_bars(ax, data, "")
    ax.set_ylabel("entropy bias vs known $H$  (bits)")
    ax.legend(title="estimator", fontsize=9, ncol=1, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ----------------------------------------------------------------------
# Figure 3 (B): the degenerate / rank-deficiency story (toy analog of ilr_fix.png)
# ----------------------------------------------------------------------


def plot_degenerate(n, seed, out_path):
    """A well-posedness (CLR-vs-ILR) demo, NOT an estimator benchmark.

    The degenerate toy has a KNOWN intrinsic entropy, so it is the controlled
    twin of ilr_fix.png: the ambient chart makes H ill-posed (-inf), the
    orthonormal ILR chart makes it finite and recovers the known value exactly.
    No estimator bars -- the point is the coordinate system, not estimator skill.
    """
    xd, ilr, hd = make_degenerate(2, n, seed=seed)

    # The ILR distribution is (by construction) an exact Gaussian, so gaussian(cov)
    # on it IS the exact intrinsic entropy -- the isometry recovers the known truth
    # with no "estimator" caveat. Ambient truth is -inf (rank-deficient).
    h_ilr = gaussian_entropy(np.cov(ilr, rowvar=False))
    evals = np.sort(np.linalg.eigvalsh(np.cov(xd, rowvar=False)))[::-1]
    dead = float(np.clip(np.abs(evals[-1]), 1e-300, None))

    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.22], hspace=0.05, wspace=0.28)

    # (1) ambient flat sheet: plot in (in-plane_0, in-plane_1, out-of-plane) coords
    # -- the two ILR axes span the plane; the third is the constraint normal
    # x.(1,1,1)/sqrt(3) ~ 0 everywhere. zlim matched to in-plane range -> flat disc.
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    sub = xd[:: max(1, xd.shape[0] // 4000)]
    isub = ilr[:: max(1, xd.shape[0] // 4000)]
    normal = sub @ (np.ones(3) / np.sqrt(3.0))  # out-of-plane coord ~ 1e-16
    ax1.scatter(isub[:, 0], isub[:, 1], normal, s=4, alpha=0.25, color="#0072B2")
    ax1.set_title("ambient $\\mathbb{R}^3$: flat sheet\n(normal $=0$, $H=-\\infty$)", fontsize=11)
    ax1.set_xlabel("in-plane$_0$")
    ax1.set_ylabel("in-plane$_1$")
    ax1.set_zlabel("out-of-plane")
    ax1.set_zlim(-3, 3)  # match in-plane scale -> zero thickness is obvious
    ax1.view_init(elev=22, azim=45)

    # (2) ILR projection: a healthy full-rank 2D Gaussian blob (scatter + sigma-ellipses).
    ax2 = fig.add_subplot(gs[0, 1])
    _scatter_ellipses(ax2, ilr, "ILR projection: full-rank 2D\n(H recovered)", "ILR$_0$", "ILR$_1$")

    # (3) the rank cliff: ambient covariance eigenvalue spectrum.
    ax3 = fig.add_subplot(gs[0, 2])
    ev_plot = np.clip(np.abs(evals), 1e-18, None)  # |.| so the ~0 dir plots on log
    ax3.bar(range(len(ev_plot)), ev_plot, color=["#0072B2", "#0072B2", "tab:red"], zorder=3)
    ax3.set_yscale("log")
    ax3.set_ylim(1e-18, 10)
    ax3.set_title("ambient cov eigenvalues\n(dead direction = rank cliff)", fontsize=11)
    ax3.set_xlabel("eigenvalue index")
    ax3.set_ylabel("eigenvalue (log)")
    ax3.set_xticks(range(len(ev_plot)))
    ax3.grid(axis="y", color="#ddd", lw=0.7, zorder=0)
    ax3.set_axisbelow(True)

    # caption band -- the truth (no estimator bars: this is about the chart, not skill)
    axc = fig.add_subplot(gs[1, :])
    axc.axis("off")
    caption = (
        f"True differential entropy:   ambient $\\mathbb{{R}}^3 = -\\infty$ "
        f"(rank-deficient; dead eigenvalue {dead:.0e})      |      "
        f"ILR $\\mathbb{{R}}^2 = {hd * BITS:.2f}$ bits (finite).\n"
        f"The ILR projection is an orthonormal isometry, so it preserves the intrinsic entropy "
        f"exactly: gaussian(cov) on ILR $= {h_ilr * BITS:.2f}$ bits, matching the known truth "
        f"(${hd * BITS:.2f}$ bits).\n"
        f"No entropy estimator can rescue the ambient case (even the flow fails) -- the only fix "
        f"is dropping to the intrinsic dimension."
    )
    axc.text(
        0.5,
        0.55,
        caption,
        ha="center",
        va="center",
        fontsize=10,
        color="#222",
        bbox={"boxstyle": "round,pad=0.6", "fc": "#f4f4f4", "ec": "#cccccc"},
    )

    fig.suptitle(
        "Degenerate (rank-deficient) toy: ambient $H=-\\infty$, ILR makes it finite "
        "-- the known-truth twin of ilr_fix.png",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.02)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "experiments" / "num_visits" / "plots"),
        help="directory for the PNGs (default: the num_visits plots dir, "
        "beside ilr_fix.png)",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_distributions(args.n, args.seed, out / "toy_distributions.png")
    plot_copula_lens(args.n, args.seed, out / "copula_lens.png")
    plot_bias(args.n, args.seed, args.epochs, out / "toy_entropy_bias.png")
    plot_degenerate(args.n, args.seed, out / "toy_degenerate.png")


if __name__ == "__main__":
    import torch

    torch.set_num_threads(8)
    main()
