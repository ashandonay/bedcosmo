"""Empirical convergence study: does a 1-D normalizing flow converge to the KDE?

Claim under test (from the target_params H(z) discussion): in 1-D the flow-plugin
entropy ``H = -E_eval[log q]`` is a *biased-high* estimator that only approaches the
truth (and the KDE/kNN plug-ins) as the flow's **capacity** grows -- while KDE/kNN,
being nonparametric, sit at the truth already. A second, honesty-check sweep shows
that growing the **sample count** at *fixed* capacity does NOT close the gap: it
plateaus above truth at the architecture's approximation floor.

Two known-truth 1-D targets (analytic differential entropy in nats):
  * ``warp``  : z + a*sin(z), a monotone smooth warp of a Gaussian (unbounded,
                mildly non-Gaussian). Reference H from the change-of-variables MC.
  * ``gamma`` : Gamma(shape=3), bounded below and right-skewed -- a stand-in for the
                DESI z-marginal (hard floor + skew). Reference H = scipy gamma.entropy().

For each target we report the nonparametric plug-ins (KDE, kNN, Gaussian) and the
flow plug-in across (A) a capacity ladder at fixed N, and (B) a sample ladder at
fixed capacity. Output: a 2x2 PNG + a JSON dump.

CPU-only, self-contained. Run under SLURM (see the scratch wrapper) or locally:

    OMP_NUM_THREADS=8 python experiments/num_visits/scripts/flow_kde_convergence.py \
        --out-dir /pscratch/sd/a/ashandon/bedcosmo/num_visits/flow_kde_convergence
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# The entropy estimators live in scripts/entropy_models; make them importable
# when this file is run as a script.
_ENTROPY_MODELS = Path(__file__).resolve().parents[3] / "scripts" / "entropy_models"
if str(_ENTROPY_MODELS) not in sys.path:
    sys.path.insert(0, str(_ENTROPY_MODELS))

# The reusable flow-plugin entropy estimator (held-out cross-entropy, nats).
from estimators import flow_plugin_entropy  # noqa: E402

LOG2 = math.log(2.0)


# ----------------------------------------------------------------------
# Known-truth 1-D targets
# ----------------------------------------------------------------------
def sample_warp(n: int, *, a: float = 0.6, seed: int = 0) -> tuple[np.ndarray, float]:
    """y = z + a*sin(z), z ~ N(0,1). Monotone for |a|<1 -> closed-form H via MC."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    y = z + a * np.sin(z)
    # H(y) = H(z) + E_z[log(1 + a cos z)], H(z) = 0.5 log(2 pi e).
    z_ref = np.random.default_rng(seed + 10_007).standard_normal(2_000_000)
    jac = float(np.mean(np.log1p(a * np.cos(z_ref))))
    h_true = 0.5 * math.log(2.0 * math.pi * math.e) + jac
    return y[:, None].astype(np.float64), h_true


def sample_gamma(n: int, *, shape: float = 3.0, seed: int = 0) -> tuple[np.ndarray, float]:
    """Gamma(shape), scale=1: bounded below, right-skewed. Closed-form differential H."""
    from scipy.stats import gamma

    rng = np.random.default_rng(seed)
    x = gamma.rvs(shape, size=n, random_state=rng)
    h_true = float(gamma.entropy(shape))  # nats
    return x[:, None].astype(np.float64), h_true


TARGETS = {
    "warp": sample_warp,
    "gamma": sample_gamma,
}


# ----------------------------------------------------------------------
# Nonparametric plug-in estimators (nats)
# ----------------------------------------------------------------------
def kde_entropy(x: np.ndarray) -> float:
    """Resubstitution KDE plug-in H = -mean(log p_hat), Scott bandwidth (nats)."""
    from scipy.stats import gaussian_kde

    xr = np.asarray(x, dtype=np.float64).ravel()
    kde = gaussian_kde(xr)  # Scott's rule
    return float(-np.mean(kde.logpdf(xr)))


def knn_entropy_nats(x: np.ndarray, k: int = 5) -> float:
    """Kozachenko-Leonenko kNN differential entropy in nats (1-D, c_1 = 2)."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    xr = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    n = xr.shape[0]
    tree = cKDTree(xr)
    # k+1 because the first neighbour is the point itself (distance 0).
    dist, _ = tree.query(xr, k=k + 1)
    rho = dist[:, k]
    rho = np.where(rho > 0, rho, np.finfo(float).tiny)
    # H = -psi(k) + psi(n) + log(c_d) + (d/n) sum log(rho), d=1, c_1 = 2.
    return float(-digamma(k) + digamma(n) + math.log(2.0) + np.mean(np.log(rho)))


def gauss_entropy_nats(x: np.ndarray) -> float:
    """Gaussian plug-in H = 0.5 log(2 pi e sigma^2) (nats)."""
    var = float(np.var(np.asarray(x, dtype=np.float64).ravel()))
    return 0.5 * math.log(2.0 * math.pi * math.e * var)


# Capacity ladder: increasing spline flexibility + training budget.
CAPACITY_LADDER = [
    dict(transforms=1, bins=4, hidden_features=(32,), epochs=150),
    dict(transforms=2, bins=6, hidden_features=(64,), epochs=250),
    dict(transforms=3, bins=8, hidden_features=(64, 64), epochs=400),
    dict(transforms=4, bins=12, hidden_features=(128, 128), epochs=600),
    dict(transforms=6, bins=20, hidden_features=(128, 128), epochs=800),
]
# Capacity index -> a rough scalar "flexibility" for plotting (bins*transforms).
def _cap_flex(cfg: dict) -> int:
    return int(cfg["transforms"]) * int(cfg["bins"])


def run(out_dir: Path, *, n_fixed: int, seeds: list[int]) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"n_fixed": n_fixed, "seeds": seeds, "targets": {}}

    sample_ladder = [2_000, 5_000, 10_000, 20_000, n_fixed]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    for col, (name, sampler) in enumerate(TARGETS.items()):
        # Reference draw (big) for the nonparametric truth anchors + analytic H.
        x_ref, h_true = sampler(n_fixed, seed=999)
        h_kde_ref = kde_entropy(x_ref)
        h_knn_ref = knn_entropy_nats(x_ref)
        h_gauss_ref = gauss_entropy_nats(x_ref)

        tgt = {
            "h_true_nats": h_true,
            "h_true_bits": h_true / LOG2,
            "h_kde_ref_bits": h_kde_ref / LOG2,
            "h_knn_ref_bits": h_knn_ref / LOG2,
            "h_gauss_ref_bits": h_gauss_ref / LOG2,
            "capacity_sweep": [],
            "sample_sweep": [],
        }

        # ---- (A) capacity sweep at fixed N -------------------------------
        cap_flex, cap_mean, cap_std = [], [], []
        for cfg in CAPACITY_LADDER:
            hs = []
            for s in seeds:
                x, _ = sampler(n_fixed, seed=s)
                h = flow_plugin_entropy(x, seed=s, **cfg)
                hs.append(h / LOG2)  # bits
            m, sd = float(np.mean(hs)), float(np.std(hs))
            cap_flex.append(_cap_flex(cfg))
            cap_mean.append(m)
            cap_std.append(sd)
            tgt["capacity_sweep"].append(
                {"flex": _cap_flex(cfg), "cfg": cfg, "h_flow_bits_mean": m, "h_flow_bits_std": sd}
            )
            print(f"[{name}] cap flex={_cap_flex(cfg):3d}  H_flow={m:.3f}+/-{sd:.3f} bits", flush=True)

        # ---- (B) sample sweep at fixed (mid) capacity --------------------
        fixed_cfg = CAPACITY_LADDER[2]  # t3/b8, a "reasonable but finite" flow
        smp_n, smp_mean, smp_std, smp_kde = [], [], [], []
        for n in sample_ladder:
            hs, kdes = [], []
            for s in seeds:
                x, _ = sampler(n, seed=1000 + s)
                hs.append(flow_plugin_entropy(x, seed=s, **fixed_cfg) / LOG2)
                kdes.append(kde_entropy(x) / LOG2)
            smp_n.append(n)
            smp_mean.append(float(np.mean(hs)))
            smp_std.append(float(np.std(hs)))
            smp_kde.append(float(np.mean(kdes)))
            tgt["sample_sweep"].append(
                {
                    "n": n,
                    "h_flow_bits_mean": float(np.mean(hs)),
                    "h_flow_bits_std": float(np.std(hs)),
                    "h_kde_bits_mean": float(np.mean(kdes)),
                }
            )
            print(f"[{name}] N={n:6d}  H_flow={np.mean(hs):.3f}  H_kde={np.mean(kdes):.3f} bits", flush=True)

        results["targets"][name] = tgt

        # ---- plots -------------------------------------------------------
        h_true_b = h_true / LOG2
        # Top row: capacity sweep
        axtop = axes[0, col]
        axtop.errorbar(cap_flex, cap_mean, yerr=cap_std, marker="o", color="C0",
                       lw=1.6, capsize=3, label="flow plug-in (bias-high)")
        axtop.axhline(h_true_b, color="k", ls="-", lw=1.6, label=f"truth = {h_true_b:.3f}")
        axtop.axhline(h_kde_ref / LOG2, color="C2", ls="--", lw=1.4,
                      label=f"KDE = {h_kde_ref/LOG2:.3f}")
        axtop.axhline(h_knn_ref / LOG2, color="C4", ls=":", lw=1.4,
                      label=f"kNN = {h_knn_ref/LOG2:.3f}")
        axtop.axhline(h_gauss_ref / LOG2, color="C3", ls="-.", lw=1.2, alpha=0.8,
                      label=f"Gaussian = {h_gauss_ref/LOG2:.3f}")
        axtop.set_title(f"[{name}] capacity sweep  (N={n_fixed}, {len(seeds)} seeds)")
        axtop.set_xlabel("flow flexibility  (transforms x bins)")
        axtop.set_ylabel("H  (bits)")
        axtop.set_xscale("log")
        axtop.legend(fontsize=8)
        axtop.grid(alpha=0.25)

        # Bottom row: sample sweep at fixed capacity
        axbot = axes[1, col]
        axbot.errorbar(smp_n, smp_mean, yerr=smp_std, marker="s", color="C0",
                       lw=1.6, capsize=3, label=f"flow @ fixed cap (t{fixed_cfg['transforms']}/b{fixed_cfg['bins']})")
        axbot.plot(smp_n, smp_kde, marker="^", color="C2", lw=1.6, label="KDE @ same N")
        axbot.axhline(h_true_b, color="k", ls="-", lw=1.6, label=f"truth = {h_true_b:.3f}")
        axbot.set_title(f"[{name}] sample sweep  (fixed capacity)")
        axbot.set_xlabel("training samples N")
        axbot.set_ylabel("H  (bits)")
        axbot.set_xscale("log")
        axbot.legend(fontsize=8)
        axbot.grid(alpha=0.25)

    fig.suptitle(
        "Flow-plugin entropy converges to KDE/truth via CAPACITY (top), not via SAMPLES alone (bottom)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = out_dir / "flow_kde_convergence.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)

    (out_dir / "flow_kde_convergence.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {png}", flush=True)
    print(f"[done] wrote {out_dir / 'flow_kde_convergence.json'}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-fixed", type=int, default=40_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(int(args.threads))
    run(Path(args.out_dir).expanduser(), n_fixed=int(args.n_fixed), seeds=list(args.seeds))


if __name__ == "__main__":
    main()
