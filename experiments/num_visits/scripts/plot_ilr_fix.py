"""Before/after: the CLR rank cliff makes the 14D prior entropy ill-posed; ILR fixes it.

Two panels, from a single CLR build artifact (no mlflow run). ILR-13D is the same
prior expressed in its intrinsic isometric chart -- the CLR sample projected onto
the orthonormal basis of the sum-zero hyperplane -- so the *only* difference
between the two is the dead direction.

  (a) covariance eigenvalue spectrum of the physical prior: CLR has an exact
      rank cliff (dead sum-zero direction ~1e-15); ILR has none (full-rank 13D).
  (b) cumulative Gaussian entropy H_m = sum_{i<=m} 1/2 log(2*pi*e*lambda_i) as
      eigen-directions are added largest-first. The 13 real directions give an
      IDENTICAL finite value in both charts; CLR then adds a 14th (dead) direction
      whose contribution 1/2 log(2*pi*e*lambda_dead) -> -inf (floored here only by
      float64 at lambda~6e-15). So the 14D entropy is ill-posed, while 13D is finite.

This is deliberately a covariance-RANK statement, not a true-entropy or
estimator-accuracy claim: the -inf comes from the rank alone, independent of the
prior's non-Gaussianity. (For the separate non-Gaussianity / negentropy question,
the honest estimator is a normalizing flow in the well-posed 13D space -- kNN is
dimension-biased there; see scripts/entropy_models/validation.py.)

Usage:
  python experiments/num_visits/scripts/plot_ilr_fix.py [--artifact <clr.joblib>] [--out <png>]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _paths import plot_path  # noqa: E402

from bedcosmo.num_visits.empirical.fit_sed_prior_kde import (  # noqa: E402
    get_parameterization,
    load_sed_prior_kde,
    sample_sed_prior,
)
from bedcosmo.num_visits.empirical.simplex import ilr_basis  # noqa: E402

LOG2 = math.log(2.0)
BITS = 1.0 / LOG2
GAUSS_C = 0.5 * math.log(2.0 * math.pi * math.e)  # per-direction Gaussian H offset
PRIOR_DIR = "/pscratch/sd/a/ashandon/bedcosmo/num_visits/empirical_prior"
DEFAULT_ARTIFACT = f"{PRIOR_DIR}/sed_prior_kde_clr_backup.joblib"


def spectrum_and_cumH(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending eigenvalues and cumulative Gaussian entropy (nats)."""
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    h_contrib = GAUSS_C + 0.5 * np.log(np.clip(ev, 1e-300, None))
    return ev, np.cumsum(h_contrib)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="CLR build artifact")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default=str(plot_path("ilr_fix.png")))
    args = ap.parse_args()

    A = load_sed_prior_kde(Path(args.artifact))
    par = get_parameterization(A)
    nt = int(A["n_templates"])
    if par != "clr":
        print(f"WARNING: expected a CLR artifact (14D, degenerate); got '{par}'. "
              "The rank cliff this figure documents only exists in CLR.")

    X = sample_sed_prior(A, args.n, seed=0)  # 14D physical CLR (exact sum-zero)
    V = ilr_basis(nt)
    ilr = np.concatenate([X[:, :nt] @ V, X[:, nt:]], axis=1)  # same draw, 13D ILR

    covI = np.cov(ilr, rowvar=False)
    evC, cumC = spectrum_and_cumH(np.cov(X, rowvar=False))
    evI, cumI = spectrum_and_cumH(covI)

    # Which ILR eigen-directions are a *single* physical variable? Only the two
    # tightly-constrained params (log_c_scale, z, the last two ILR columns) are
    # nearly axis-aligned; the 11 template log-ratio directions are correlation
    # mixtures, so labelling them with one coord would be misleading.
    phys = {nt - 1: "log_c_scale", nt: "z"}  # column index -> name
    wI, QI = np.linalg.eigh(covI)
    order = np.argsort(wI)[::-1]  # descending, matches evI
    dir_labels = {}  # 1-based eigenvalue index -> variable name
    for rank, col in enumerate(order, start=1):
        loadings2 = QI[:, col] ** 2
        top = int(loadings2.argmax())
        if loadings2[top] > 0.9 and top in phys:
            dir_labels[rank] = phys[top]
    dead_contrib = (GAUSS_C + 0.5 * math.log(max(evC[-1], 1e-300))) * BITS
    print(f"CLR dim={X.shape[1]} min-eig={evC[-1]:.2e}  ILR dim={ilr.shape[1]} "
          f"min-eig={evI[-1]:.2e}")
    print(f"cumulative Gaussian H (bits): CLR-13-real={cumC[len(evI) - 1] * BITS:.2f}  "
          f"ILR-full={cumI[-1] * BITS:.2f}  (should match)  |  "
          f"CLR-14D(with dead)={cumC[-1] * BITS:.2f}  dead-term={dead_contrib:.2f} -> -inf")

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 9), sharex=True)

    # (a) eigenvalue spectra -- the cliff
    a0 = ax[0]
    a0.semilogy(np.arange(1, len(evC) + 1), np.clip(evC, 1e-18, None), "o-",
                color="C3", label=f"CLR physical ({len(evC)}D)")
    a0.semilogy(np.arange(1, len(evI) + 1), np.clip(evI, 1e-18, None), "s-",
                color="C2", label=f"ILR physical ({len(evI)}D)")
    a0.axhline(1.0, ls=":", color="gray", lw=1)
    # label only the single-variable (axis-aligned) ILR directions, with a short
    # leader into the empty band below so each points unambiguously at its marker
    label_pos = {"log_c_scale": (10.2, 6e-3), "z": (12.2, 6e-3)}
    for rank, name in dir_labels.items():
        a0.annotate(name, xy=(rank, evI[rank - 1]),
                    xytext=label_pos.get(name, (rank, evI[rank - 1] * 0.05)),
                    ha="center", va="center", fontsize=8, color="C2",
                    arrowprops={"arrowstyle": "->", "color": "C2", "lw": 0.8})
    # the rest of the ILR block is a mixture of template log-ratios
    n_mix = len(evI) - len(dir_labels)
    a0.text(n_mix / 2 + 0.5, 2.2, f"{n_mix} template log-ratio dirs (mixed)",
            ha="center", va="center", fontsize=8, color="C2")
    # the CLR-only 14th direction is the single dead one: the sum-zero constraint
    a0.annotate(r"dead dir: $\sum_i f_i = 0$", xy=(len(evC), max(evC[-1], 1e-18)),
                xytext=(len(evC) - 3.5, 1e-11), ha="center", va="center",
                fontsize=8, color="C3",
                arrowprops={"arrowstyle": "->", "color": "C3", "lw": 0.8})
    a0.set_ylabel(r"eigenvalue $\lambda_i$ (variance)")
    a0.set_title(r"(a) Eigenvalue cliff: CLR is rank-deficient ($\sum_i f_i = 0$)")
    a0.legend(fontsize=9)
    a0.grid(alpha=0.3, which="both")

    # (b) cumulative Gaussian entropy -- the cliff *is* an entropy of -inf
    a1 = ax[1]
    a1.plot(np.arange(1, len(cumC) + 1), cumC * BITS, "o-", color="C3",
            label=f"CLR ({len(evC)}D): dips to $-\\infty$")
    a1.plot(np.arange(1, len(cumI) + 1), cumI * BITS, "s-", color="C2",
            label=f"ILR ({len(evI)}D): finite plateau")
    a1.axhline(cumI[-1] * BITS, ls=":", color="C2", lw=1)
    a1.set_xlabel("eigenvalue index (directions included largest-first)")
    a1.set_ylabel("cumulative Gaussian entropy (bits)")
    a1.set_title(r"(b) Gaussian entropy $H=\sum_i \frac{1}{2}\log(2\pi e\,\lambda_i)$")
    a1.legend(fontsize=8.5, loc="lower left")
    a1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
