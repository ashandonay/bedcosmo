"""Validate the empirical prior flow(s) against the KDE ground truth.

The gate before trusting ``prior_source: flow``. The KDE is the validated ground
truth; a trained :class:`PriorFlow` must reproduce it before it may act as the
prior. Run-free (loads only the build artifact + the ``.pt`` flows, no mlflow run)
and READ-ONLY w.r.t. src/bedcosmo state.

Two spaces, two views -- pick with ``--space`` and ``--plot``:

* ``native``       -- native/ILR prior coords. Flow draws (``PriorFlow.sample``)
                      vs KDE draws (``sample_sed_prior``).
* ``gaussianized`` -- the whitened ``y = T(x)`` space that ``transform_input=True``
                      EIG lives in. Flow draws (sampled directly in y-space) vs KDE
                      draws pushed through the production gaussianizer.

* ``panel``    -- per-feature marginal histograms + a template-weight/variance bar
                  + an entropy/density-fidelity summary with a PASS/REVIEW verdict.
* ``triangle`` -- an overlaid getdist contour corner plot (KDE filled vs flow line),
                  with fixed smoothing + auto boundary-range detection so sharp ILR
                  edges (e.g. ``f10``) do not produce blotchy contours.

Entropy is measured from each flow's EXACT normalized log-density -- the same NF
plug-in the posterior ``H_post`` uses -- so ``H_flow = -E_flow[log p_flow]`` is the
flow's true entropy up to MC error (no biased nonparametric estimator at 13D). The
NLL gap ``H_cross - H_flow`` (flow scoring real KDE draws minus its own) -> 0 iff the
flow density matches the KDE, a reliable fidelity check from the exact density alone.

NSF scoring is CPU-heavy; cap threads on a login node (``--threads 8``). Example::

    python -m bedcosmo.num_visits.empirical.validate_prior_flow \\
        --space both --plot both --n 50000 --threads 8
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import ks_2samp, norm, wasserstein_distance  # noqa: E402

from bedcosmo.entropy import knn_entropy  # noqa: E402
from bedcosmo.transform import _whitening_to_apply_joint  # noqa: E402

from .fit_sed_prior_kde import (  # noqa: E402
    get_empirical_gaussianizer,
    get_gaussianizer_whitening,
    get_parameterization,
    load_sed_prior_kde,
    sample_sed_prior,
)
from .paths import get_prior_kde_path  # noqa: E402
from .prior_flow import (  # noqa: E402
    SED_PRIOR_FLOW_FILENAMES,
    SPACE_GAUSSIANIZED,
    SPACE_NATIVE,
    PriorFlow,
)
from .sed_prior import score_kde_artifact  # noqa: E402
from .simplex import split_feature_matrix  # noqa: E402

LOG2 = math.log(2.0)

# Heuristic pass thresholds (the flow must reproduce the KDE / KDE pushforward).
KS_MAX_PASS = 0.05  # max per-feature (and template-weight) KS statistic
COV_FROB_PASS = 0.10  # relative Frobenius of cov difference
NLL_GAP_PASS = 0.50  # |H_cross - H_flow| nats, from the exact flow density


# --------------------------------------------------------------------------- #
# sampling helpers
# --------------------------------------------------------------------------- #
def _resolve_flow_path(arg: str | None, artifact_path: Path, space: str) -> Path | None:
    if arg:
        p = Path(arg).expanduser()
        return p if p.is_file() else None
    p = artifact_path.parent / SED_PRIOR_FLOW_FILENAMES[space]
    return p if p.is_file() else None


def _load_flow(path: Path, space: str, dim: int) -> PriorFlow:
    flow = PriorFlow.load(path)
    if flow.space != space:
        raise ValueError(f"expected a {space!r} flow, got space={flow.space!r}")
    if flow.dim != dim:
        raise ValueError(f"flow dim {flow.dim} != {dim} KDE features; retrain on this artifact.")
    return flow


def _sample_gaussianized_flow(gflow: PriorFlow, n: int, seed: int) -> np.ndarray:
    """Draw n rows from a gaussianized-space flow (bypasses the native-only guard).

    Sampling *inside* y-space is exact: only recovering physical x from y needs the
    tail-sensitive inverse, which we never do here.
    """
    import torch

    mu = gflow.meta["standardize"]["mu"]
    std = gflow.meta["standardize"]["std"]
    torch.manual_seed(int(seed))
    gflow.flow.eval()
    with torch.no_grad():
        z = gflow.flow().sample((int(n),)).cpu().numpy().astype(np.float64)
    return z * std + mu


def _kde_pushforward(artifact, xk: np.ndarray) -> np.ndarray:
    """Push native KDE draws through the production gaussianizer: y = T(x)."""
    import torch

    bij = get_empirical_gaussianizer(artifact)
    apply_joint = _whitening_to_apply_joint(get_gaussianizer_whitening(artifact))
    y = bij.matrix_to_gaussian(torch.as_tensor(xk, dtype=torch.float64), apply_joint=apply_joint)
    return y.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# entropy
# --------------------------------------------------------------------------- #
def _native_bridge(artifact, nflow: PriorFlow, xk: np.ndarray) -> tuple[float, float, float]:
    """Native Jacobian bridge H(y) = H_native + E_native[log|det dT/dx|].

    Returns (H_native, E[log|det|], H_bridge). This is the value the runtime uses
    for transform_input=True; a gaussianized-flow direct H(y) should agree with it.
    """
    import torch

    bij = get_empirical_gaussianizer(artifact)
    H_native = float(-nflow.log_prob(xk).mean())
    e_logdet = float(bij.joint_log_abs_det_jacobian(torch.as_tensor(xk, dtype=torch.float64)).mean())
    return H_native, e_logdet, H_native + e_logdet


# --------------------------------------------------------------------------- #
# triangle (getdist)
# --------------------------------------------------------------------------- #
def _detect_boundary_ranges(rows, names, *, edge_frac=0.03, edge_width=0.02):
    """Declare a hard limit where samples pile up against an edge.

    getdist's auto-bandwidth collapses on a sharp boundary (e.g. ILR ``f10`` has
    ~7% of mass within 2% of its max), producing blotchy contours and spurious
    islands as the KDE leaks past the edge. Declaring the boundary lets getdist
    apply its linear boundary-correction kernel there instead. A limit is set only
    on the side that actually piles up; the other side stays open (``None``).
    """
    ranges: dict[str, list] = {}
    for j, nm in enumerate(names):
        v = rows[:, j]
        lo, hi = float(v.min()), float(v.max())
        span = hi - lo
        if span <= 0:
            continue
        rng = [None, None]
        if float(np.mean(v < lo + edge_width * span)) > edge_frac:
            rng[0] = lo
        if float(np.mean(v > hi - edge_width * span)) > edge_frac:
            rng[1] = hi
        if rng[0] is not None or rng[1] is not None:
            ranges[nm] = rng
    return ranges


def _triangle(kde_rows, flow_rows, names, labels, title, out_path, *, settings, ranges):
    """Overlaid filled-contour triangle: KDE (ground truth, filled) vs flow (line)."""
    from getdist import MCSamples, plots

    kde_s = MCSamples(
        samples=kde_rows, names=names, labels=labels, label="KDE", ranges=ranges, settings=settings
    )
    flow_s = MCSamples(
        samples=flow_rows, names=names, labels=labels, label="flow", ranges=ranges, settings=settings
    )
    g = plots.get_subplot_plotter(width_inch=1.6 * len(names))
    g.settings.alpha_filled_add = 0.6
    g.triangle_plot(
        [kde_s, flow_s],
        names,
        filled=[True, False],
        contour_colors=["black", "C3"],
        legend_labels=["KDE", "flow"],
        legend_loc="upper right",
    )
    g.fig.suptitle(title, fontsize=13)
    g.export(str(out_path))
    print(f"[save] {out_path}", flush=True)


# --------------------------------------------------------------------------- #
# panel plots
# --------------------------------------------------------------------------- #
def _panel_native(args, out_path, names, xk, xf, ak, af, ks, weight_ks, metrics, checks, verdict):
    D = len(names)
    ncols = 4
    nrows = int(np.ceil((D + 2) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = axes.ravel()

    for j in range(D):
        ax = axes[j]
        edges = np.linspace(min(xk[:, j].min(), xf[:, j].min()), max(xk[:, j].max(), xf[:, j].max()), args.bins)
        ax.hist(xk[:, j], bins=edges, density=True, histtype="step", color="k", label="KDE")
        ax.hist(xf[:, j], bins=edges, density=True, histtype="step", color="C3", label="flow")
        ax.set_title(f"{names[j]}  (KS={ks[j]:.3f})", fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    ax = axes[D]
    t = np.arange(ak.shape[1])
    ax.bar(t - 0.2, ak.mean(axis=0), width=0.4, color="k", label="KDE")
    ax.bar(t + 0.2, af.mean(axis=0), width=0.4, color="C3", label="flow")
    ax.set_title(f"template weight means (KS max {weight_ks.max():.3f})", fontsize=9)
    ax.set_xlabel("template", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    ax = axes[D + 1]
    ax.axis("off")
    lines = [
        f"VERDICT: {verdict}",
        "",
        f"max KS      : {ks.max():.4f}",
        f"cov Frob    : {metrics['cov_frob']:.4f}",
        f"H_flow      : {metrics['H_flow']:.3f} nats  (= H_prior)",
        f"H_cross     : {metrics['H_cross']:.3f} nats",
        f"NLL gap     : {metrics['nll_gap']:+.4f} nats",
        "",
        "unreliable refs (13D nonparametric):",
        f"  H_kde plugin  : {metrics['H_kde']:.3f}",
        f"  H_knn (kde)   : {metrics['H_knn_kde']:.3f}",
        f"  H_knn (flow)  : {metrics['H_knn_flow']:.3f}",
        "",
        *metrics.get("gauss_lines", []),
        "",
        *[f"[{'OK' if ok else 'XX'}] {k}" for k, ok in checks.items()],
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)

    for k in range(D + 2, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"Empirical prior A/B (native): flow vs KDE  [{verdict}]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    print(f"[save] {out_path}", flush=True)


def _panel_gaussianized(args, out_path, names, yk, yf, ks, metrics, checks, verdict):
    D = len(names)
    ncols = 4
    nrows = int(np.ceil((D + 2) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = axes.ravel()

    for j in range(D):
        ax = axes[j]
        lo = min(yk[:, j].min(), yf[:, j].min())
        hi = max(yk[:, j].max(), yf[:, j].max())
        edges = np.linspace(lo, hi, args.bins)
        ax.hist(yk[:, j], bins=edges, density=True, histtype="step", color="k", label="KDE push")
        ax.hist(yf[:, j], bins=edges, density=True, histtype="step", color="C0", label="gauss flow")
        grid = np.linspace(lo, hi, 200)
        ax.plot(grid, norm.pdf(grid), color="C1", lw=0.8, ls="--", label="N(0,1)")
        ax.set_title(f"y[{names[j]}]  (KS={ks[j]:.3f})", fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    ax = axes[D]
    t = np.arange(D)
    ax.bar(t - 0.2, np.diag(np.cov(yk, rowvar=False)), width=0.4, color="k", label="KDE push")
    ax.bar(t + 0.2, np.diag(np.cov(yf, rowvar=False)), width=0.4, color="C0", label="gauss flow")
    ax.axhline(1.0, color="C1", ls="--", lw=0.8)
    ax.set_title("per-feature variance (target 1)", fontsize=9)
    ax.set_xlabel("y feature", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    ax = axes[D + 1]
    ax.axis("off")
    lines = [
        f"VERDICT: {verdict}",
        "",
        f"max KS          : {ks.max():.4f}",
        f"cov Frob (f-k)  : {metrics['cov_frob']:.4f}",
        f"cov(kde) vs I   : {metrics['cov_kde_from_I']:.4f}",
        f"cov(flow) vs I  : {metrics['cov_flow_from_I']:.4f}",
        "",
        f"H_gflow         : {metrics['H_flow']:.3f} nats ({metrics['H_flow'] / LOG2:.2f} bits)",
        f"H_cross         : {metrics['H_cross']:.3f} nats",
        f"NLL gap         : {metrics['nll_gap']:+.4f} nats",
        f"N(0,I) shortcut : {metrics['H_iso']:.3f} nats ({metrics['H_iso'] / LOG2:.2f} bits)",
        "",
        *metrics.get("bridge_lines", []),
        "",
        *[f"[{'OK' if ok else 'XX'}] {k}" for k, ok in checks.items()],
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)

    for k in range(D + 2, len(axes)):
        axes[k].axis("off")
    fig.suptitle(
        f"Empirical prior A/B (gaussianized y): flow vs KDE pushforward  [{verdict}]", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    print(f"[save] {out_path}", flush=True)


# --------------------------------------------------------------------------- #
# per-space drivers
# --------------------------------------------------------------------------- #
def _run_native(args, out_dir, artifact, names, param, xk, nflow, gflow, tri_settings):
    D = len(names)
    xf = nflow.sample(int(args.n), seed=int(args.seed) + 1)

    ks = np.array([ks_2samp(xk[:, j], xf[:, j]).statistic for j in range(D)])
    ak, _, _ = split_feature_matrix(xk, int(artifact["n_templates"]), parameterization=param)
    af, _, _ = split_feature_matrix(xf, int(artifact["n_templates"]), parameterization=param)
    weight_ks = np.array([ks_2samp(ak[:, t], af[:, t]).statistic for t in range(ak.shape[1])])
    Ck, Cf = np.cov(xk, rowvar=False), np.cov(xf, rowvar=False)
    cov_frob = float(np.linalg.norm(Cf - Ck) / max(np.linalg.norm(Ck), 1e-30))

    H_flow = float(-nflow.log_prob(xf).mean())
    H_cross = float(-nflow.log_prob(xk).mean())
    metrics = {
        "cov_frob": cov_frob,
        "H_flow": H_flow,
        "H_cross": H_cross,
        "nll_gap": H_cross - H_flow,
        "H_kde": float(-score_kde_artifact(artifact, xk).mean()),
        "H_knn_kde": float(knn_entropy(xk[: min(int(args.knn_n), xk.shape[0])])),
        "H_knn_flow": float(knn_entropy(xf[: min(int(args.knn_n), xf.shape[0])])),
    }
    if gflow is not None:
        _, _, H_bridge = _native_bridge(artifact, nflow, xk)
        yk = _kde_pushforward(artifact, xk)
        H_gd = float(-gflow.log_prob(yk).mean())
        H_iso = 0.5 * D * math.log(2.0 * math.pi * math.e)
        metrics["gauss_lines"] = [
            "gaussianized H(y):",
            f"  bridge (native+Jac): {H_bridge:7.3f} nats ({H_bridge / LOG2:6.2f} bits)",
            f"  gaussianized flow  : {H_gd:7.3f} nats  (d {H_gd - H_bridge:+.3f})",
            f"  N(0,I) shortcut    : {H_iso:7.3f} nats  (d {H_iso - H_bridge:+.3f})",
        ]

    checks = {
        f"max feature KS < {KS_MAX_PASS}": float(ks.max()) < KS_MAX_PASS,
        f"max template-weight KS < {KS_MAX_PASS}": float(weight_ks.max()) < KS_MAX_PASS,
        f"cov rel-Frobenius < {COV_FROB_PASS}": cov_frob < COV_FROB_PASS,
        f"flow NLL gap < {NLL_GAP_PASS} nats": abs(metrics["nll_gap"]) < NLL_GAP_PASS,
    }
    verdict = "PASS" if all(checks.values()) else "REVIEW"
    _report(SPACE_NATIVE, names, ks, metrics, checks, verdict, wasser=(xk, xf))

    if args.plot in ("panel", "both"):
        _panel_native(
            args, out_dir / "validate_prior_flow_native.png", names, xk, xf, ak, af,
            ks, weight_ks, metrics, checks, verdict,
        )
    if args.plot in ("triangle", "both"):
        _emit_triangle(args, out_dir, xk, xf, names, "native/ILR", "native", tri_settings)


def _run_gaussianized(args, out_dir, artifact, names, xk, nflow, gflow, tri_settings):
    D = len(names)
    yk = _kde_pushforward(artifact, xk)
    yf = _sample_gaussianized_flow(gflow, int(args.n), int(args.seed) + 1)

    ks = np.array([ks_2samp(yk[:, j], yf[:, j]).statistic for j in range(D)])
    Ck, Cf = np.cov(yk, rowvar=False), np.cov(yf, rowvar=False)
    eye = np.eye(D)
    metrics = {
        "cov_frob": float(np.linalg.norm(Cf - Ck) / max(np.linalg.norm(Ck), 1e-30)),
        "cov_kde_from_I": float(np.linalg.norm(Ck - eye) / np.linalg.norm(eye)),
        "cov_flow_from_I": float(np.linalg.norm(Cf - eye) / np.linalg.norm(eye)),
        "H_flow": float(-gflow.log_prob(yf).mean()),
        "H_cross": float(-gflow.log_prob(yk).mean()),
        "H_iso": 0.5 * D * math.log(2.0 * math.pi * math.e),
    }
    metrics["nll_gap"] = metrics["H_cross"] - metrics["H_flow"]
    if nflow is not None:
        H_native, e_logdet, H_bridge = _native_bridge(artifact, nflow, xk)
        metrics["bridge_lines"] = [
            f"H_native        : {H_native:.3f} nats",
            f"E[log|det|]     : {e_logdet:.3f} nats",
            f"H_bridge        : {H_bridge:.3f} nats  (native+Jac)",
            f"  gflow - bridge: {metrics['H_flow'] - H_bridge:+.3f} nats",
        ]

    checks = {
        f"max feature KS < {KS_MAX_PASS}": float(ks.max()) < KS_MAX_PASS,
        f"cov rel-Frobenius < {COV_FROB_PASS}": metrics["cov_frob"] < COV_FROB_PASS,
        f"flow NLL gap < {NLL_GAP_PASS} nats": abs(metrics["nll_gap"]) < NLL_GAP_PASS,
    }
    verdict = "PASS" if all(checks.values()) else "REVIEW"
    _report(SPACE_GAUSSIANIZED, names, ks, metrics, checks, verdict)

    if args.plot in ("panel", "both"):
        _panel_gaussianized(
            args, out_dir / "validate_prior_flow_gaussianized.png", names, yk, yf,
            ks, metrics, checks, verdict,
        )
    if args.plot in ("triangle", "both"):
        _emit_triangle(
            args, out_dir, yk, yf, names, "gaussianized y", "gaussianized", tri_settings,
            label_wrap="y[{}]",
        )


def _emit_triangle(
    args, out_dir, kde_rows, flow_rows, names, space_label, tag, settings, *, label_wrap="{}"
):
    # getdist-safe names; pretty labels for display. Escape underscores BEFORE
    # wrapping so mathtext doesn't read e.g. y[log_c_scale] as a double subscript.
    gd_names = [f"p{j}" for j in range(kde_rows.shape[1])]
    disp = [label_wrap.format(n.replace("_", r"\_")) for n in names]
    ranges = {} if args.no_boundary else _detect_boundary_ranges(kde_rows, gd_names)
    if ranges:
        print(f"[boundary/{tag}] {[names[int(k[1:])] for k in ranges]}", flush=True)
    _triangle(
        kde_rows, flow_rows, gd_names, disp,
        f"Empirical prior triangle ({space_label}): flow vs KDE",
        out_dir / f"validate_prior_flow_{tag}_triangle.png",
        settings=settings, ranges=ranges,
    )


def _report(space, names, ks, metrics, checks, verdict, *, wasser=None):
    print(f"\n===== {space} flow vs KDE A/B =====", flush=True)
    print(f"D={len(names)}  worst-KS feature: {names[int(ks.argmax())]} ({ks.max():.4f})")
    for j in range(len(names)):
        extra = ""
        if wasser is not None:
            xk, xf = wasser
            w = wasserstein_distance(xk[:, j], xf[:, j]) / max(xk[:, j].std(), 1e-12)
            extra = f"  W/std={w:.4f}"
        print(f"  {names[j]:>12s}: KS={ks[j]:.4f}{extra}")
    print(f"  H_flow={metrics['H_flow']:.3f}  H_cross={metrics['H_cross']:.3f}  "
          f"NLL_gap={metrics['nll_gap']:+.4f} nats")
    for line in metrics.get("gauss_lines", []) + metrics.get("bridge_lines", []):
        print("  " + line)
    for name, ok in checks.items():
        print(f"    [{'OK' if ok else 'XX'}] {name}")
    print(f"  VERDICT: {verdict}", flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--artifact", default=None, help="KDE joblib (default: production artifact).")
    ap.add_argument("--flow-path", default=None, help="Native flow .pt (default: beside KDE).")
    ap.add_argument("--gauss-flow-path", default=None, help="Gaussianized flow .pt (default: beside KDE).")
    ap.add_argument("--space", choices=[SPACE_NATIVE, SPACE_GAUSSIANIZED, "both"], default="both")
    ap.add_argument("--plot", choices=["panel", "triangle", "both"], default="both")
    ap.add_argument("--n", type=int, default=50_000, help="Samples drawn from each prior.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=60, help="Panel histogram bins.")
    ap.add_argument("--knn-n", type=int, default=8000, help="Subsample for the kNN entropy referee.")
    ap.add_argument("--threads", type=int, default=0, help="Cap torch threads (0 = default).")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: beside the KDE artifact).")
    # triangle de-blotch knobs (getdist auto-bandwidth collapses on sharp ILR edges).
    ap.add_argument("--smooth-2d", type=float, default=0.5, help="smooth_scale_2D (frac of std).")
    ap.add_argument("--smooth-1d", type=float, default=0.3, help="smooth_scale_1D (frac of std).")
    ap.add_argument("--mult-bias", type=int, default=1, help="mult_bias_correction_order (0 = off).")
    ap.add_argument("--fine-bins-2d", type=int, default=256, help="fine_bins_2D density grid.")
    ap.add_argument("--no-boundary", action="store_true", help="Disable auto boundary ranges.")
    args = ap.parse_args(argv)

    if args.threads > 0:
        import torch

        torch.set_num_threads(int(args.threads))

    artifact_path = Path(args.artifact).expanduser() if args.artifact else get_prior_kde_path()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else artifact_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] KDE artifact: {artifact_path}", flush=True)
    artifact = load_sed_prior_kde(artifact_path)
    names = list(artifact["feature_names"])
    param = get_parameterization(artifact)
    D = len(names)
    print(f"[load] D={D}  parameterization={param}", flush=True)

    tri_settings = {
        "smooth_scale_2D": args.smooth_2d,
        "smooth_scale_1D": args.smooth_1d,
        "mult_bias_correction_order": args.mult_bias,
        "fine_bins_2D": args.fine_bins_2d,
        "boundary_correction_order": 1,
    }

    # Load whichever flows the requested spaces need. The native flow is also the
    # bridge reference for the gaussianized panel, and vice-versa, so load both when
    # available even for a single space (missing counterpart just drops that section).
    def _try_load(space):
        p = _resolve_flow_path(
            args.flow_path if space == SPACE_NATIVE else args.gauss_flow_path, artifact_path, space
        )
        return _load_flow(p, space, D) if p is not None else None

    nflow = _try_load(SPACE_NATIVE)
    gflow = _try_load(SPACE_GAUSSIANIZED)

    xk = sample_sed_prior(artifact, int(args.n), seed=int(args.seed))

    if args.space in (SPACE_NATIVE, "both"):
        if nflow is None:
            raise FileNotFoundError(
                "native flow not found (pass --flow-path or train it with "
                "python -m bedcosmo.num_visits.empirical.prior_flow --space both)."
            )
        _run_native(args, out_dir, artifact, names, param, xk, nflow, gflow, tri_settings)

    if args.space in (SPACE_GAUSSIANIZED, "both"):
        if gflow is None:
            raise FileNotFoundError(
                "gaussianized flow not found (pass --gauss-flow-path or train --space both)."
            )
        _run_gaussianized(args, out_dir, artifact, names, xk, nflow, gflow, tri_settings)


if __name__ == "__main__":
    main()
