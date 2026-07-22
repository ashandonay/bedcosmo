"""Validate the entropy estimators against KNOWN closed-form entropies in low D.

Motivation: before trusting any transformed-space H(z) number, check that the
estimators recover the *right answer* on distributions whose differential entropy
we know exactly. This is also the archival record of why the y-prior Gaussian
copula was removed from bedcosmo: it is exact only when the dependence is
Gaussian, and biased high under nonlinear dependence (the ``banana`` toy). All
estimators use the same formula H = -E[log p]:

  * copula  -- a Gaussian-copula y-prior estimator (REMOVED from bedcosmo; kept
               here to document its failure mode): fit the real
               ``Bijector.fit_from_matrix(..., input_transform_type="joint")``
               (marginal normal-scores + Cholesky whitening) and score held-out
               samples: log p(x) = log N(0,I)(z) + joint_log_abs_det_jacobian(x).
  * knn     -- the PRODUCTION marginal-EIG estimator (Kozachenko-Leonenko).
  * flow    -- the DIAGNOSTIC normalizing-flow plug-in (zuko NSF).
  * cumulant-- Edgeworth negentropy cross-check.

Four toys with exact / known-reference H. The first three are FULL-RANK and
isolate where the Gaussian copula wins vs. loses on a *well-posed* entropy; the
fourth is RANK-DEFICIENT and isolates a different axis of failure entirely.

  1. gaussian        -- correlated N(0, Sigma). Copula is exact by construction.
  2. warped-margins  -- elementwise monotone warp of a correlated Gaussian:
                        NON-Gaussian margins but a GAUSSIAN copula. Normal-scores
                        undoes the margins, so the production copula should still
                        be ~exact. (Reuses make_correlated_nongaussian.)
  3. banana          -- Rosenbrock map with unit Jacobian: NONLINEAR dependence,
                        non-Gaussian copula. H = H(base) exactly. The Gaussian
                        copula CANNOT capture this and is biased high; only the
                        flow should recover the truth.
  4. degenerate      -- intrinsic d-dim Gaussian embedded isometrically on the
                        sum-zero hyperplane of R^{d+1} (exact linear constraint;
                        the toy analog of the empirical prior's CLR sum f = 0).
                        The ambient (d+1)-D density is a delta in the collapsed
                        direction, so its differential entropy is -inf: ILL-POSED.
                        This is a DIFFERENT failure than the banana -- not a bias
                        on a well-posed quantity, but a well-posed-LOOKING finite
                        answer to an ill-posed question. EVERY ambient estimator
                        (copula, knn, flow, cumulant) returns a finite convention
                        value set by how it fills the dead direction, not H. The
                        flow does NOT fix this; projecting onto the plane's own
                        orthonormal ILR basis (full-rank d-D) recovers the honest
                        intrinsic H. Run as two rows -- CLR/ambient vs ILR/projected
                        -- against the same intrinsic-d reference entropy.

READ-ONLY w.r.t. src/bedcosmo: imports the production Bijector / knn_entropy and
uses them, but modifies nothing.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

torch.set_num_threads(8)

from estimators import (  # noqa: E402
    cumulant_negentropy_entropy,
    flow_plugin_entropy,
    gaussian_entropy,
    make_correlated_nongaussian,
)

BITS = 1.0 / math.log(2.0)


# ----------------------------------------------------------------------
# Production estimators as thin wrappers
# ----------------------------------------------------------------------


def copula_plugin_entropy(
    samples: np.ndarray,
    *,
    n_train: int | None = None,
    seed: int = 0,
    shrinkage: float = 1e-3,
    cdf_bins: int | None = None,
    per_bin: int = 400,
    cdf_eps: float = 1e-6,
) -> float:
    """Production joint-EIG estimator: -E_eval[log p] under the fitted Gaussian copula.

    Fits the real production Bijector (marginal normal-scores + Cholesky whitening)
    on a train split and scores a disjoint eval split, so the copula is not scored
    on its own fit points. This is exactly ``log_prob_y_copula`` / the ``joint``
    prior-entropy path, just fit on toy samples.

    The marginal Jacobian is a histogram density estimate (dCDF/dbin), so its bias
    is set by samples-per-bin: the production default ``cdf_bins=5000`` is only
    unbiased with 1000s of samples per bin. We size ``cdf_bins`` to ~``per_bin``
    samples/bin so we test the copula *model* bias (non-Gaussian dependence), not
    histogram noise. Pass ``cdf_bins`` explicitly to override.
    """
    from bedcosmo.transform import Bijector

    x = np.asarray(samples, dtype=np.float64)
    n, d = x.shape
    names = [f"x{i}" for i in range(d)]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = n // 2 if n_train is None else n_train
    x_tr, x_ev = x[perm[:ntr]], x[perm[ntr:]]

    if cdf_bins is None:
        cdf_bins = int(np.clip(ntr // per_bin, 50, 3000))

    bij = Bijector.fit_from_matrix(
        x_tr,
        names,
        input_transform_type="joint",
        shrinkage=shrinkage,
        cdf_bins=cdf_bins,
        cdf_eps=cdf_eps,
        seed=seed,
    )
    xe = torch.as_tensor(x_ev, dtype=torch.float64)
    with torch.no_grad():
        z = bij.matrix_to_gaussian(xe)  # apply_joint=None -> joint (production)
        log_pz = -0.5 * (z.pow(2).sum(dim=-1) + d * math.log(2.0 * math.pi))
        log_det = bij.joint_log_abs_det_jacobian(xe)
        logp = log_pz + log_det
    return float(-logp.mean().item())


def knn_plugin_entropy(
    samples: np.ndarray, *, k: int = 3, max_n: int = 20_000, seed: int = 0
) -> float | None:
    """Production marginal-EIG estimator: Kozachenko-Leonenko k-NN entropy (nats)."""
    try:
        from bedcosmo.entropy import knn_entropy
    except Exception as exc:  # pragma: no cover
        print(f"  (knn_entropy unavailable: {exc})")
        return None
    x = np.asarray(samples, dtype=np.float64)
    if x.shape[0] > max_n:
        rng = np.random.default_rng(seed)
        x = x[rng.permutation(x.shape[0])[:max_n]]
    # knn_entropy returns BITS; convert to nats to match the other estimators.
    return float(knn_entropy(x, k=k) * math.log(2.0))


# ----------------------------------------------------------------------
# Toys with exact / near-exact known entropy
# ----------------------------------------------------------------------


def make_gaussian(d, n, *, seed=0, cov_jitter=0.5):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    cov = a @ a.T / d + np.eye(d) * cov_jitter
    chol = np.linalg.cholesky(cov)
    x = rng.standard_normal((n, d)) @ chol.T
    return x, gaussian_entropy(cov)


def make_banana(d, n, *, seed=0, b=1.0):
    """Rosenbrock banana: unit-Jacobian nonlinear dependence -> H = H(N(0,I_d)) exactly.

    x0 = u0;  x1 = u1 + b*(u0^2 - 1);  remaining dims identity. The map is lower
    triangular with unit diagonal, so |det J| = 1 and H(x) = (d/2) log(2*pi*e).
    Strongly non-Gaussian *dependence* with a non-Gaussian copula.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, d))
    x = u.copy()
    x[:, 1] = u[:, 1] + b * (u[:, 0] ** 2 - 1.0)
    h_true = 0.5 * d * (1.0 + math.log(2.0 * math.pi))
    return x, h_true


def ilr_basis(K: int) -> np.ndarray:
    """Orthonormal basis (K, K-1) of the sum-zero hyperplane in R^K.

    Columns are orthonormal and orthogonal to the ones-vector, so mapping a
    (K-1)-vector through it lands on the exact Sigma x = 0 plane, and the reverse
    projection (x @ V) is the isometric ILR chart.

    Kept as a local copy (rather than importing the identical canonical
    ``bedcosmo.num_visits.empirical.simplex.ilr_basis``) so this diagnostic stays
    standalone and doesn't pull the full bedcosmo package init for a 4-line helper.
    """
    m = np.eye(K) - np.ones((K, K)) / K
    _, V = np.linalg.eigh(m)
    return V[:, 1:]  # drop the ~zero eigenvector (the ones direction)


def make_degenerate(d, n, *, seed=0):
    """Rank-deficient toy: intrinsic d-D Gaussian on the sum-zero plane of R^{d+1}.

    Returns ``(x_ambient (n, d+1), ilr (n, d), h_true_intrinsic)``. The ambient
    coords carry an EXACT linear constraint (rows sum to 0), so their (d+1)-D
    differential entropy is -inf. ``h_true`` is the honest intrinsic-d entropy the
    ILR projection recovers; ``ilr = x @ V`` equals the intrinsic draw exactly
    (V has orthonormal columns), so the ILR row should recover ``h_true``.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    cov = a @ a.T / d + np.eye(d) * 0.5
    z = rng.standard_normal((n, d)) @ np.linalg.cholesky(cov).T
    h_true = gaussian_entropy(cov)
    V = ilr_basis(d + 1)  # (d+1, d)
    x = z @ V.T  # ambient, rows sum to 0 exactly
    ilr = x @ V  # isometric projection back to intrinsic d-D (== z)
    return x, ilr, h_true


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


def run_case(name, x, h_true, *, flow_epochs=300, seed=0):
    n, d = x.shape
    try:
        h_gauss = gaussian_entropy(np.cov(x, rowvar=False))
    except ValueError:
        # Rank-deficient cov (degenerate toy): the ambient Gaussian H is ill-posed.
        h_gauss = None
    h_cop = copula_plugin_entropy(x, seed=seed)
    h_knn = knn_plugin_entropy(x, seed=seed)
    h_flow = flow_plugin_entropy(
        x, epochs=flow_epochs, hidden_features=(128, 128), batch_size=4096, seed=seed
    )
    try:
        h_cum = cumulant_negentropy_entropy(x, seed=seed)
    except ValueError:
        # Degenerate toy: whitening a rank-deficient cov is undefined (the dead
        # eigenvalue tips negative under float noise) -- another way the ambient
        # coords refuse to yield a real entropy.
        h_cum = None

    def bias(h):
        return None if h is None else (h - h_true) * BITS

    print(f"\n--- {name}  (d={d}, N={n}) ---")
    print(f"  H_true            = {h_true:8.4f} nats  ({h_true*BITS:7.4f} bits)")
    rows = [
        ("gaussian(cov)", h_gauss),
        ("copula [PROD joint]", h_cop),
        ("knn [PROD marginal]", h_knn),
        ("flow [diagnostic]", h_flow),
        ("cumulant", h_cum),
    ]
    print(f"  {'estimator':<22} {'nats':>9} {'bits':>9} {'bias (bits)':>12}")
    for label, h in rows:
        if h is None:
            print(f"  {label:<22} {'n/a':>9}")
            continue
        print(f"  {label:<22} {h:>9.4f} {h*BITS:>9.4f} {bias(h):>+12.4f}")
    return {"copula": bias(h_cop), "flow": bias(h_flow), "knn": bias(h_knn)}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--banana-b", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print("ENTROPY ESTIMATOR VALIDATION vs KNOWN CLOSED-FORM H")
    print("=" * 70)
    print("Expect: copula ~exact on gaussian & warped-margins (Gaussian copula),")
    print("        BIASED HIGH on banana (nonlinear dependence); flow ~exact on all.")
    print("        degenerate: ambient H is -inf (exact constraint) -- ALL ambient")
    print("        estimators return finite convention values; only ILR recovers truth.")

    summary = []
    for d in args.dims:
        xg, hg = make_gaussian(d, args.n, seed=args.seed)
        summary.append(
            ("gaussian", d, run_case("gaussian", xg, hg, flow_epochs=args.epochs, seed=args.seed))
        )

        xw, hw = make_correlated_nongaussian(d, args.n, seed=args.seed)
        summary.append(
            (
                "warped-margins",
                d,
                run_case("warped-margins", xw, hw, flow_epochs=args.epochs, seed=args.seed),
            )
        )

        xb, hb = make_banana(d, args.n, seed=args.seed, b=args.banana_b)
        summary.append(
            (
                "banana",
                d,
                run_case(
                    f"banana (b={args.banana_b})", xb, hb, flow_epochs=args.epochs, seed=args.seed
                ),
            )
        )

        # Degenerate: same intrinsic-d reference H, run on the ambient (CLR-like,
        # ill-posed) coords and again on the ILR projection. Ambient should miss H
        # every way; ILR should recover it.
        xd, ilr, hd = make_degenerate(d, args.n, seed=args.seed)
        summary.append(
            (
                "degenerate[CLR]",
                d + 1,
                run_case(
                    f"degenerate CLR/ambient (d={d} -> R^{d + 1}, H_ambient=-inf)",
                    xd,
                    hd,
                    flow_epochs=args.epochs,
                    seed=args.seed,
                ),
            )
        )
        summary.append(
            (
                "degenerate[ILR]",
                d,
                run_case(
                    f"degenerate ILR/projected (d={d})",
                    ilr,
                    hd,
                    flow_epochs=args.epochs,
                    seed=args.seed,
                ),
            )
        )

    print("\n" + "=" * 70)
    print("SUMMARY  (bias vs truth, bits; ~0 = accurate)")
    print("=" * 70)
    print(f"  {'toy':<16} {'d':>2} {'copula':>10} {'flow':>10} {'knn':>10}")
    for toy, d, b in summary:

        def fmt(v):
            return "n/a" if v is None else f"{v:+.3f}"

        print(f"  {toy:<16} {d:>2} {fmt(b['copula']):>10} {fmt(b['flow']):>10} {fmt(b['knn']):>10}")


if __name__ == "__main__":
    main()
