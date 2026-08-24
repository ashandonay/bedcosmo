"""Toy analog of production *marginal* entropy via slice-from-joint.

Mirrors ``Evaluator.get_marginal_eig`` / ``_marginal_posterior_entropy``:

  Production                              This toy
  ------------------------------------    ------------------------------------
  theta = (targets S, nuisances)          theta = (S, nuisances)
  y ~ p(y|d)                              y ~ marginal of joint Gaussian
  prior: sample p(theta), slice to S      sample p(theta), slice to S
  post: sample q(theta|y,d), slice to S   sample p(theta|y), slice to S
  knn on the sliced cloud                 same (production knn_entropy)

Joint is zero-mean Gaussian on (theta, y) with
``dim = n_targets + n_nuisance + 1``. Every entropy has a closed form.
Posterior covariance does not depend on the value of y.

Plots (``--plot``) write timestamped files under
``experiments/num_visits/plots/`` (``*_YYYYMMDD_HHMMSS.png``):
  * sweep   -- bias vs k, vs N, vs M=outer y (horizontal; any dim)
  * clouds  -- joint (x0, x_n) + sliced 1-D densities (horizontal;
               only when ``--n-targets 1 --n-nuisance 1``)

READ-ONLY w.r.t. src/bedcosmo: imports production ``knn_entropy`` only.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from estimators import gaussian_entropy  # noqa: E402

BITS = 1.0 / math.log(2.0)
LN2 = math.log(2.0)

PLOTS_DIR = HERE.parents[1] / "experiments" / "num_visits" / "plots"


def _default_plot_path(stem: str) -> Path:
    """``experiments/num_visits/plots/<stem>_YYYYMMDD_HHMMSS.png``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PLOTS_DIR / f"{stem}_{stamp}.png"


def make_joint_cov(
    *,
    n_targets: int = 1,
    n_nuisance: int = 1,
    rho_ss: float | None = None,
    rho_0n: float = 0.5,
    rho_nn: float | None = None,
    rho_0y: float = 0.7,
    rho_ny: float = 0.4,
) -> np.ndarray:
    """Unit-diagonal SPD correlation on (S, nuisances, y).

    Layout: indices ``0 .. n_targets-1`` are targets S, then nuisances, then y.
    """
    if n_targets < 1:
        raise ValueError(f"n_targets must be >= 1, got {n_targets}")
    if n_nuisance < 0:
        raise ValueError(f"n_nuisance must be >= 0, got {n_nuisance}")
    if rho_ss is None:
        rho_ss = rho_0n
    if rho_nn is None:
        rho_nn = rho_0n

    n_targets = int(n_targets)
    n_nuisance = int(n_nuisance)
    d_theta = n_targets + n_nuisance
    d = d_theta + 1
    cov = np.eye(d, dtype=np.float64)

    # Among targets.
    for i in range(n_targets):
        for j in range(i + 1, n_targets):
            cov[i, j] = cov[j, i] = rho_ss
    # Target–nuisance.
    for i in range(n_targets):
        for j in range(n_targets, d_theta):
            cov[i, j] = cov[j, i] = rho_0n
    # Among nuisances.
    for i in range(n_targets, d_theta):
        for j in range(i + 1, d_theta):
            cov[i, j] = cov[j, i] = rho_nn
    # Observation couplings.
    iy = d_theta
    for i in range(n_targets):
        cov[i, iy] = cov[iy, i] = rho_0y
    for i in range(n_targets, d_theta):
        cov[i, iy] = cov[iy, i] = rho_ny

    evals = np.linalg.eigvalsh(cov)
    if evals.min() <= 1e-10:
        raise ValueError(
            f"joint cov not SPD (min eig={evals.min():.3e}); "
            f"n_targets={n_targets}, n_nuisance={n_nuisance}, "
            f"rho_ss={rho_ss}, rho_0n={rho_0n}, rho_nn={rho_nn}, "
            f"rho_0y={rho_0y}, rho_ny={rho_ny}"
        )
    return cov


def _d_theta(cov: np.ndarray) -> int:
    return int(cov.shape[0]) - 1


def _iy(cov: np.ndarray) -> int:
    return int(cov.shape[0]) - 1


def analytic_targets(cov: np.ndarray, *, n_targets: int) -> dict:
    """Closed-form prior H(S) and posterior H(S|y) in bits."""
    n_targets = int(n_targets)
    d_theta = _d_theta(cov)
    if not (1 <= n_targets <= d_theta):
        raise ValueError(
            f"n_targets={n_targets} incompatible with d_theta={d_theta}"
        )
    iy = _iy(cov)
    sig_ss = cov[:n_targets, :n_targets]
    sig_tt = cov[:d_theta, :d_theta]
    sig_ty = cov[:d_theta, iy : iy + 1]
    sig_yy = float(cov[iy, iy])
    sig_tt_y = sig_tt - (sig_ty @ sig_ty.T) / sig_yy
    sig_ss_y = sig_tt_y[:n_targets, :n_targets]
    return {
        "h_prior_S": float(gaussian_entropy(sig_ss) * BITS),
        "h_post_S_given_y": float(gaussian_entropy(sig_ss_y) * BITS),
        "sig_ss": sig_ss,
        "sig_ss_y": sig_ss_y,
        "sig_tt_y": sig_tt_y,
        "n_targets": n_targets,
        "n_nuisance": d_theta - n_targets,
        "d_theta": d_theta,
        "joint_dim": int(cov.shape[0]),
    }


def sample_prior_theta(n: int, cov: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Draw ``(n, d_theta)`` from the prior p(theta)."""
    d_theta = _d_theta(cov)
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(cov[:d_theta, :d_theta])
    return rng.standard_normal((n, d_theta)) @ chol.T


def sample_posterior_theta_given_y(
    ys: np.ndarray,
    n_samples: int,
    cov: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``(K, M, d_theta)`` from p(theta | y_m) for each outer y."""
    rng = np.random.default_rng(seed)
    ys = np.asarray(ys, dtype=np.float64)
    m = int(ys.shape[0])
    d_theta = _d_theta(cov)
    iy = _iy(cov)
    sig_tt = cov[:d_theta, :d_theta]
    sig_ty = cov[:d_theta, iy]
    sig_yy = float(cov[iy, iy])
    sig_tt_y = sig_tt - np.outer(sig_ty, sig_ty) / sig_yy
    chol = np.linalg.cholesky(sig_tt_y)
    mean_scale = sig_ty / sig_yy
    out = np.empty((n_samples, m, d_theta), dtype=np.float64)
    for j, y in enumerate(ys):
        noise = rng.standard_normal((n_samples, d_theta)) @ chol.T
        out[:, j, :] = noise + mean_scale * float(y)
    return out


def knn_bits(samples: np.ndarray, *, k: int = 3, warn_duplicates: bool = True) -> float:
    from bedcosmo.entropy import knn_entropy

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return float(knn_entropy(x, k=k, warn_duplicates=warn_duplicates))


def prior_sliced_entropy(
    cov: np.ndarray,
    *,
    n_targets: int,
    n_samples: int,
    k: int,
    seed: int = 0,
) -> float:
    """Production prior path: sample theta, slice to S, k-NN."""
    theta = sample_prior_theta(n_samples, cov, seed=seed)
    return knn_bits(theta[:, :n_targets], k=k)


def prior_sliced_entropy_mean(
    cov: np.ndarray,
    *,
    n_targets: int,
    n_samples: int,
    k: int,
    n_clouds: int,
    seed: int = 0,
) -> float:
    """Mean of ``n_clouds`` independent prior k-NN estimates (size ``n_samples``).

    Puts the prior on the same footing as ``E_y[H]`` (mean of ``M`` k-NN calls)
    so sweep error bars are comparable.
    """
    n_clouds = max(1, int(n_clouds))
    hs = [
        prior_sliced_entropy(
            cov,
            n_targets=n_targets,
            n_samples=n_samples,
            k=k,
            seed=seed + j,
        )
        for j in range(n_clouds)
    ]
    return float(np.mean(hs))


def posterior_sliced_entropy(
    cov: np.ndarray,
    *,
    n_targets: int,
    n_outer_y: int,
    n_samples: int,
    k: int,
    seed: int = 0,
    ys: np.ndarray | None = None,
) -> dict:
    """Production posterior path: sample theta|y, slice to S, mean over y."""
    rng = np.random.default_rng(seed)
    iy = _iy(cov)
    if ys is None:
        ys = rng.normal(0.0, math.sqrt(float(cov[iy, iy])), size=n_outer_y)
    else:
        ys = np.asarray(ys, dtype=np.float64)
        n_outer_y = int(ys.shape[0])

    theta = sample_posterior_theta_given_y(ys, n_samples, cov, seed=seed + 1)
    S = theta[:, :, :n_targets]
    per_y = [
        knn_bits(S[:, j, :], k=k, warn_duplicates=False) for j in range(n_outer_y)
    ]
    return {
        "ys": ys,
        "theta": theta,
        "S": S,
        "per_y": np.asarray(per_y, dtype=np.float64),
        "h_mean": float(np.mean(per_y)),
    }


def _bias_mean_std(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(vals, axis=0)
    if vals.shape[0] <= 1:
        std = np.zeros_like(mean)
    else:
        std = np.nanstd(vals, axis=0, ddof=1)
    return mean, std


def sample_size_sweep(
    cov: np.ndarray,
    *,
    n_targets: int,
    sample_grid: list[int],
    n_outer_y: int,
    knn_k: int,
    seed: int = 0,
    n_reps: int = 1,
) -> dict:
    """Bias vs N. Prior and posterior are both means of ``n_outer_y`` k-NN calls."""
    meta = analytic_targets(cov, n_targets=n_targets)
    n_reps = max(1, int(n_reps))
    prior_reps = np.empty((n_reps, len(sample_grid)), dtype=np.float64)
    post_reps = np.empty((n_reps, len(sample_grid)), dtype=np.float64)
    for r in range(n_reps):
        for i, n in enumerate(sample_grid):
            base = seed + 10_000 * r + 10 * i
            h_prior = prior_sliced_entropy_mean(
                cov,
                n_targets=n_targets,
                n_samples=n,
                k=knn_k,
                n_clouds=n_outer_y,
                seed=base,
            )
            post = posterior_sliced_entropy(
                cov,
                n_targets=n_targets,
                n_outer_y=n_outer_y,
                n_samples=n,
                k=knn_k,
                seed=base + 1,
            )
            prior_reps[r, i] = h_prior - meta["h_prior_S"]
            post_reps[r, i] = post["h_mean"] - meta["h_post_S_given_y"]
    prior_bias, prior_std = _bias_mean_std(prior_reps)
    post_bias, post_std = _bias_mean_std(post_reps)
    return {
        "n": np.asarray(sample_grid, dtype=int),
        "prior_bias": prior_bias,
        "post_bias": post_bias,
        "prior_bias_std": prior_std,
        "post_bias_std": post_std,
        "n_reps": n_reps,
        "n_outer_y": n_outer_y,
        "knn_k": knn_k,
        "joint_dim": meta["joint_dim"],
        "n_targets": meta["n_targets"],
        "n_nuisance": meta["n_nuisance"],
    }


def neighbor_rank_sweep(
    cov: np.ndarray,
    *,
    n_targets: int,
    k_grid: list[int],
    n_samples: int,
    n_outer_y: int,
    seed: int = 0,
    n_reps: int = 1,
) -> dict:
    """Bias vs k. Prior/post both mean of ``n_outer_y`` clouds; clouds reused across k."""
    meta = analytic_targets(cov, n_targets=n_targets)
    n_reps = max(1, int(n_reps))
    prior_reps = np.empty((n_reps, len(k_grid)), dtype=np.float64)
    post_reps = np.empty((n_reps, len(k_grid)), dtype=np.float64)
    for r in range(n_reps):
        base = seed + 10_000 * r
        # M independent prior clouds, reused across k (same footing as posterior).
        prior_clouds = [
            sample_prior_theta(n_samples, cov, seed=base + 1000 + j)[:, :n_targets]
            for j in range(n_outer_y)
        ]
        post = posterior_sliced_entropy(
            cov,
            n_targets=n_targets,
            n_outer_y=n_outer_y,
            n_samples=n_samples,
            k=max(k_grid),
            seed=base + 1,
        )
        for i, kk in enumerate(k_grid):
            if n_samples <= kk + 1:
                prior_reps[r, i] = np.nan
                post_reps[r, i] = np.nan
                continue
            prior_per = [
                knn_bits(cloud, k=kk, warn_duplicates=False) for cloud in prior_clouds
            ]
            per_y = [
                knn_bits(post["S"][:, j, :], k=kk, warn_duplicates=False)
                for j in range(n_outer_y)
            ]
            prior_reps[r, i] = float(np.mean(prior_per)) - meta["h_prior_S"]
            post_reps[r, i] = float(np.mean(per_y)) - meta["h_post_S_given_y"]
    prior_bias, prior_std = _bias_mean_std(prior_reps)
    post_bias, post_std = _bias_mean_std(post_reps)
    return {
        "k": np.asarray(k_grid, dtype=int),
        "prior_bias": prior_bias,
        "post_bias": post_bias,
        "prior_bias_std": prior_std,
        "post_bias_std": post_std,
        "n_samples": n_samples,
        "n_outer_y": n_outer_y,
        "n_reps": n_reps,
        "joint_dim": meta["joint_dim"],
        "n_targets": meta["n_targets"],
        "n_nuisance": meta["n_nuisance"],
    }


def outer_y_sweep(
    cov: np.ndarray,
    *,
    n_targets: int,
    outer_y_grid: list[int],
    n_samples: int,
    knn_k: int,
    seed: int = 0,
    n_reps: int = 1,
) -> dict:
    """Bias vs M=outer y.

    Posterior is ``E_y[H]`` over ``M`` clouds (bars shrink with ``M``).
    Prior is a single cloud of size ``N`` — independent of ``M`` by construction
    (production prior entropy does not average over outer ``y``).
    """
    meta = analytic_targets(cov, n_targets=n_targets)
    n_reps = max(1, int(n_reps))
    prior_reps = np.empty((n_reps, len(outer_y_grid)), dtype=np.float64)
    post_reps = np.empty((n_reps, len(outer_y_grid)), dtype=np.float64)
    for r in range(n_reps):
        base = seed + 10_000 * r
        # One prior estimate per seed; broadcast across M (genuinely M-independent).
        h_prior = prior_sliced_entropy(
            cov,
            n_targets=n_targets,
            n_samples=n_samples,
            k=knn_k,
            seed=base,
        )
        prior_bias = h_prior - meta["h_prior_S"]
        for i, m in enumerate(outer_y_grid):
            post = posterior_sliced_entropy(
                cov,
                n_targets=n_targets,
                n_outer_y=int(m),
                n_samples=n_samples,
                k=knn_k,
                seed=base + 1 + i,
            )
            prior_reps[r, i] = prior_bias
            post_reps[r, i] = post["h_mean"] - meta["h_post_S_given_y"]
    prior_bias, prior_std = _bias_mean_std(prior_reps)
    post_bias, post_std = _bias_mean_std(post_reps)
    return {
        "m": np.asarray(outer_y_grid, dtype=int),
        "prior_bias": prior_bias,
        "post_bias": post_bias,
        "prior_bias_std": prior_std,
        "post_bias_std": post_std,
        "n_samples": n_samples,
        "knn_k": knn_k,
        "n_reps": n_reps,
        "joint_dim": meta["joint_dim"],
        "n_targets": meta["n_targets"],
        "n_nuisance": meta["n_nuisance"],
    }


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_clouds(
    prior_theta: np.ndarray,
    post: dict,
    post_plot: dict,
    *,
    targets: dict,
    out_path: Path,
    n_show: int = 3,
) -> None:
    """Horizontally stacked: joint (x0, xn) scatter + sliced 1-D densities."""
    if prior_theta.shape[1] != 2 or targets.get("n_targets", 1) != 1:
        raise ValueError(
            "plot_clouds needs n_targets=1 and n_nuisance=1 "
            f"(got d_theta={prior_theta.shape[1]}, n_targets={targets.get('n_targets')})"
        )
    plt = _setup_mpl()

    ys = post["ys"]
    plot_theta = post_plot["theta"]
    plot_S = post_plot["S"]

    order = np.argsort(ys)
    if len(order) <= n_show:
        show_idx = order
    else:
        picks = np.linspace(0, len(order) - 1, n_show, dtype=int)
        show_idx = order[picks]

    colors = ["#0072B2", "#E69F00", "#009E73"]
    x0_pad = 0.4
    x0_all = np.concatenate([prior_theta[:, 0], plot_S[:, show_idx, 0].ravel()])
    x0_min, x0_max = float(x0_all.min()) - x0_pad, float(x0_all.max()) + x0_pad

    fig, (ax_joint, ax_dens) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.4),
        sharex=True,
        gridspec_kw={"wspace": 0.22, "width_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14)

    sub = prior_theta[:: max(1, prior_theta.shape[0] // 4000)]
    ax_joint.scatter(
        sub[:, 0],
        sub[:, 1],
        s=4,
        alpha=0.15,
        color="#999999",
        label=r"prior $p(x_0,x_n)$",
        rasterized=True,
    )
    for i, m in enumerate(show_idx):
        c = colors[i % len(colors)]
        th = plot_theta[:, m, :]
        ax_joint.scatter(
            th[:, 0],
            th[:, 1],
            s=6,
            alpha=0.25,
            color=c,
            label=rf"$p(x_0,x_n\mid y={ys[m]:+.1f})$",
            rasterized=True,
        )
    ax_joint.set_xlim(x0_min, x0_max)
    ax_joint.set_xlabel(r"$x_0$  (target)")
    ax_joint.set_ylabel(r"$x_n$  (nuisance)")
    ax_joint.set_title(
        r"joint $\theta=(x_0,x_n)$: prior + posteriors given $y$" + "\n"
        r"(production samples full $\theta$, then slices)",
        fontsize=11,
    )
    ax_joint.legend(fontsize=7, framealpha=0.95, loc="best")

    bins = np.linspace(x0_min, x0_max, 60)
    ax_dens.hist(
        prior_theta[:, 0],
        bins=bins,
        density=True,
        color="#999999",
        alpha=0.35,
        label=r"prior $p(x_0)$  (slice)",
        zorder=1,
    )
    for i, m in enumerate(show_idx):
        c = colors[i % len(colors)]
        ax_dens.hist(
            plot_S[:, m, 0],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            color=c,
            label=rf"$p(x_0\mid y={ys[m]:+.1f})$  (slice)",
            zorder=2,
        )
    ax_dens.axvline(0.0, color="#cccccc", lw=0.8)
    ax_dens.set_xlabel(r"$x_0$  (target)")
    ax_dens.set_ylabel("density")
    ax_dens.set_title(
        "sliced 1-D clouds\n"
        rf"analytic $H(S)$={targets['h_prior_S']:.3f}, "
        rf"$H(S\mid y)$={targets['h_post_S_given_y']:.3f}",
        fontsize=11,
    )
    ax_dens.legend(fontsize=7, framealpha=0.95, loc="upper right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _plot_bias_panel(
    ax,
    *,
    x,
    prior_bias,
    post_bias,
    prior_std,
    post_std,
    s_lab: str,
    prior_line: str,
    post_line: str,
    xlabel: str,
    title: str,
    logx: bool = False,
    vline: float | None = None,
    vline_label: str | None = None,
    prior_label: str | None = None,
    post_label: str | None = None,
    ylabel: str = "k-NN − analytic  (bits); bars = seed std",
) -> None:
    ax.axhline(0.0, color="#000000", lw=1.0, zorder=1)
    ax.errorbar(
        x,
        prior_bias,
        yerr=prior_std,
        fmt="o-",
        color=prior_line,
        lw=1.8,
        capsize=3,
        label=prior_label or rf"prior mean of $M$ clouds $H({s_lab})$",
    )
    ax.errorbar(
        x,
        post_bias,
        yerr=post_std,
        fmt="s-",
        color=post_line,
        lw=1.8,
        capsize=3,
        label=post_label or rf"post $E_y[H({s_lab}\mid y)]$ ($M$ clouds)",
    )
    if vline is not None:
        ax.axvline(
            vline,
            color="#999999",
            ls="--",
            lw=1.0,
            label=vline_label or f"default={vline}",
        )
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, framealpha=0.95, loc="best")
    ax.grid(True, which="both" if logx else "major", color="#eee", lw=0.8)


def plot_sweeps(
    n_sweep: dict,
    k_sweep: dict,
    m_sweep: dict,
    *,
    out_path: Path,
    default_n: int | None = None,
    default_k: int | None = None,
    default_m: int | None = None,
) -> None:
    """Horizontally stacked: bias vs k, vs N, vs M=outer y."""
    plt = _setup_mpl()

    prior_line = "#0072B2"
    post_line = "#E69F00"
    joint_dim = k_sweep.get("joint_dim", n_sweep.get("joint_dim", "?"))
    n_targets = k_sweep.get("n_targets", n_sweep.get("n_targets", "?"))
    n_nuisance = k_sweep.get("n_nuisance", n_sweep.get("n_nuisance", "?"))
    n_reps = int(k_sweep.get("n_reps", n_sweep.get("n_reps", 1)))
    s_lab = r"S" if n_targets != 1 else r"x_0"
    dim_tag = rf"joint={joint_dim}D, $|S|$={n_targets}, nuisance={n_nuisance}"

    fig, (ax_k, ax_n, ax_m) = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.4),
        gridspec_kw={"wspace": 0.28, "width_ratios": [1.0, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.16)

    _plot_bias_panel(
        ax_k,
        x=k_sweep["k"],
        prior_bias=k_sweep["prior_bias"],
        post_bias=k_sweep["post_bias"],
        prior_std=k_sweep.get("prior_bias_std"),
        post_std=k_sweep.get("post_bias_std"),
        s_lab=s_lab,
        prior_line=prior_line,
        post_line=post_line,
        xlabel=r"neighbor rank $k$",
        title=(
            rf"bias vs $k$  ({dim_tag}" + "\n"
            rf"fixed $N$={k_sweep.get('n_samples', default_n)}, "
            rf"$M$={k_sweep.get('n_outer_y', '?')}, {n_reps} seeds)"
        ),
        vline=default_k,
        vline_label=f"default $k$={default_k}" if default_k is not None else None,
        ylabel="k-NN − analytic  (bits); bars = seed std of mean-of-$M$",
    )
    _plot_bias_panel(
        ax_n,
        x=n_sweep["n"],
        prior_bias=n_sweep["prior_bias"],
        post_bias=n_sweep["post_bias"],
        prior_std=n_sweep.get("prior_bias_std"),
        post_std=n_sweep.get("post_bias_std"),
        s_lab=s_lab,
        prior_line=prior_line,
        post_line=post_line,
        xlabel=r"samples $N$ (per cloud; prior and post each average $M$ clouds)",
        title=(
            rf"bias vs $N$  ({dim_tag}" + "\n"
            rf"fixed $k$={default_k if default_k is not None else '?'}, "
            rf"$M$={n_sweep.get('n_outer_y', '?')}, {n_reps} seeds)"
        ),
        logx=True,
        vline=default_n,
        vline_label=f"default $N$={default_n}" if default_n is not None else None,
        ylabel="k-NN − analytic  (bits); bars = seed std of mean-of-$M$",
    )
    _plot_bias_panel(
        ax_m,
        x=m_sweep["m"],
        prior_bias=m_sweep["prior_bias"],
        post_bias=m_sweep["post_bias"],
        prior_std=m_sweep.get("prior_bias_std"),
        post_std=m_sweep.get("post_bias_std"),
        s_lab=s_lab,
        prior_line=prior_line,
        post_line=post_line,
        xlabel=r"outer $y$ count $M$ (post only; prior is one cloud)",
        title=(
            rf"bias vs $M$  ({dim_tag}" + "\n"
            rf"fixed $N$={m_sweep.get('n_samples', default_n)}, "
            rf"$k$={m_sweep.get('knn_k', default_k)}, {n_reps} seeds)"
        ),
        logx=True,
        vline=default_m,
        vline_label=f"default $M$={default_m}" if default_m is not None else None,
        prior_label=rf"prior $H({s_lab})$ (single cloud, indep. of $M$)",
        post_label=rf"post $E_y[H({s_lab}\mid y)]$ ($M$ clouds)",
        ylabel="k-NN − analytic  (bits); bars = seed std",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _row(label: str, estimate: float, truth: float) -> None:
    bias = estimate - truth
    print(f"  {label:<42} {estimate:>9.4f} {truth:>9.4f} {bias:>+10.4f}")


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=20_000, help="Prior theta sample count")
    ap.add_argument(
        "--n-targets",
        type=int,
        default=1,
        help="Target dims |S| kept after slicing (joint dim = n_targets+n_nuisance+1)",
    )
    ap.add_argument(
        "--n-nuisance",
        type=int,
        default=1,
        help="Nuisance dims sliced away (1→ with 1 target: 3D joint; 4→6D)",
    )
    ap.add_argument(
        "--rho-ss",
        type=float,
        default=None,
        help="Corr among targets (default: same as --rho-0n)",
    )
    ap.add_argument(
        "--rho-0n", type=float, default=0.5, help="Corr(each target, each nuisance)"
    )
    ap.add_argument(
        "--rho-nn",
        type=float,
        default=None,
        help="Corr among nuisances (default: same as --rho-0n)",
    )
    ap.add_argument("--rho-0y", type=float, default=0.7, help="Corr(each target, y)")
    ap.add_argument("--rho-ny", type=float, default=0.4, help="Corr(each nuisance, y)")
    ap.add_argument("--k", type=int, default=3, help="k-NN neighbor rank")
    ap.add_argument(
        "--n-outer-y",
        type=int,
        default=20,
        help="M outer observations y ~ p(y) for posterior term",
    )
    ap.add_argument(
        "--posterior-samples",
        type=int,
        default=500,
        help="K samples of theta|y per outer y for the k-NN estimate",
    )
    ap.add_argument(
        "--plot-samples",
        type=int,
        default=5000,
        help="K samples per y for density plots only",
    )
    ap.add_argument(
        "--sample-grid",
        type=str,
        default="100,200,500,1000,2000,5000,10000,20000",
        help="Sample sizes for bias-vs-N sweep",
    )
    ap.add_argument(
        "--k-grid",
        type=str,
        default="1,2,3,5,7,10,15,20,30",
        help="Neighbor ranks for bias-vs-k sweep (fixed N)",
    )
    ap.add_argument(
        "--outer-y-grid",
        type=str,
        default="1,2,4,8,10,20,40,80",
        help="Outer-y counts M for bias-vs-M sweep (fixed N, k)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n-reps",
        type=int,
        default=8,
        help="Independent seeds per sweep point for error bars "
        "(std of mean-of-M estimates across seeds)",
    )
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument(
        "--out-clouds",
        type=str,
        default=None,
        help="Horizontally stacked joint+density figure "
        "(only written for --n-targets 1 --n-nuisance 1). "
        "Default: plots/toy_marginal_entropy_clouds_<timestamp>.png",
    )
    ap.add_argument(
        "--out-sweep",
        type=str,
        default=None,
        help="Horizontally stacked bias vs k / N / M figure. "
        "Default: plots/toy_marginal_entropy_sweep_<timestamp>.png",
    )
    args = ap.parse_args()
    if args.out_clouds is None:
        args.out_clouds = str(_default_plot_path("toy_marginal_entropy_clouds"))
    if args.out_sweep is None:
        args.out_sweep = str(_default_plot_path("toy_marginal_entropy_sweep"))

    cov = make_joint_cov(
        n_targets=args.n_targets,
        n_nuisance=args.n_nuisance,
        rho_ss=args.rho_ss,
        rho_0n=args.rho_0n,
        rho_nn=args.rho_nn,
        rho_0y=args.rho_0y,
        rho_ny=args.rho_ny,
    )
    meta = analytic_targets(cov, n_targets=args.n_targets)
    sample_grid = _parse_int_list(args.sample_grid)
    k_grid = _parse_int_list(args.k_grid)
    outer_y_grid = _parse_int_list(args.outer_y_grid)
    rho_ss = args.rho_ss if args.rho_ss is not None else args.rho_0n
    rho_nn = args.rho_nn if args.rho_nn is not None else args.rho_0n

    print("=" * 72)
    print("SLICE-FROM-JOINT MARGINAL ENTROPY TOY  (production analog)")
    print("=" * 72)
    print(
        f"  |S|={args.n_targets}, n_nuisance={args.n_nuisance}, "
        f"theta dim={meta['d_theta']}, joint dim={meta['joint_dim']}; "
        f"rho_ss={rho_ss}, rho_0n={args.rho_0n}, rho_nn={rho_nn}, "
        f"rho_0y={args.rho_0y}, rho_ny={args.rho_ny}"
    )
    print(
        f"  N_prior={args.n}, k={args.k}, M={args.n_outer_y}, "
        f"K_est={args.posterior_samples}, K_plot={args.plot_samples}"
    )
    print(
        f"  Analytic: H(S)={meta['h_prior_S']:.4f} bits, "
        f"H(S|y)={meta['h_post_S_given_y']:.4f} bits"
    )
    print(
        f"  Joint H(theta,y) = {gaussian_entropy(cov) * BITS:.4f} bits "
        f"(sanity; not used)"
    )
    print()
    print(f"  {'check':<42} {'estimate':>9} {'truth':>9} {'bias':>10}")
    print(f"  {'-'*42} {'-'*9} {'-'*9} {'-'*10}")

    h_prior = prior_sliced_entropy(
        cov,
        n_targets=args.n_targets,
        n_samples=args.n,
        k=args.k,
        seed=args.seed,
    )
    _row("prior H(S) [sample theta, slice]", h_prior, meta["h_prior_S"])

    post = posterior_sliced_entropy(
        cov,
        n_targets=args.n_targets,
        n_outer_y=args.n_outer_y,
        n_samples=args.posterior_samples,
        k=args.k,
        seed=args.seed + 1,
    )
    _row(
        "post E_y[H(S|y)] [sample theta|y, slice]",
        post["h_mean"],
        meta["h_post_S_given_y"],
    )
    print(f"  {'  (per-y std)':<42} {float(np.std(post['per_y'])):>9.4f}")

    n_sweep = k_sweep = m_sweep = None
    if not args.skip_sweep:
        print()
        print(
            f"  Sample-size sweep (k={args.k}, M={args.n_outer_y}, "
            f"{args.n_reps} seeds): {sample_grid}"
        )
        print(
            f"  {'N':>8} {'prior bias':>12} {'±std':>10} "
            f"{'post bias':>12} {'±std':>10}"
        )
        n_sweep = sample_size_sweep(
            cov,
            n_targets=args.n_targets,
            sample_grid=sample_grid,
            n_outer_y=args.n_outer_y,
            knn_k=args.k,
            seed=args.seed + 100,
            n_reps=args.n_reps,
        )
        for n, pb, ps, qb, qs in zip(
            n_sweep["n"],
            n_sweep["prior_bias"],
            n_sweep["prior_bias_std"],
            n_sweep["post_bias"],
            n_sweep["post_bias_std"],
        ):
            print(f"  {int(n):>8} {pb:>+12.4f} {ps:>10.4f} {qb:>+12.4f} {qs:>10.4f}")

        print()
        print(
            f"  Neighbor-rank sweep (N={args.posterior_samples}, "
            f"M={args.n_outer_y}, {args.n_reps} seeds): {k_grid}"
        )
        print(
            f"  {'k':>8} {'prior bias':>12} {'±std':>10} "
            f"{'post bias':>12} {'±std':>10}"
        )
        k_sweep = neighbor_rank_sweep(
            cov,
            n_targets=args.n_targets,
            k_grid=k_grid,
            n_samples=args.posterior_samples,
            n_outer_y=args.n_outer_y,
            seed=args.seed + 200,
            n_reps=args.n_reps,
        )
        for kk, pb, ps, qb, qs in zip(
            k_sweep["k"],
            k_sweep["prior_bias"],
            k_sweep["prior_bias_std"],
            k_sweep["post_bias"],
            k_sweep["post_bias_std"],
        ):
            print(f"  {int(kk):>8} {pb:>+12.4f} {ps:>10.4f} {qb:>+12.4f} {qs:>10.4f}")

        print()
        print(
            f"  Outer-y sweep (N={args.posterior_samples}, k={args.k}, "
            f"{args.n_reps} seeds): {outer_y_grid}"
        )
        print(
            f"  {'M':>8} {'prior bias':>12} {'±std':>10} "
            f"{'post bias':>12} {'±std':>10}"
        )
        m_sweep = outer_y_sweep(
            cov,
            n_targets=args.n_targets,
            outer_y_grid=outer_y_grid,
            n_samples=args.posterior_samples,
            knn_k=args.k,
            seed=args.seed + 300,
            n_reps=args.n_reps,
        )
        for m, pb, ps, qb, qs in zip(
            m_sweep["m"],
            m_sweep["prior_bias"],
            m_sweep["prior_bias_std"],
            m_sweep["post_bias"],
            m_sweep["post_bias_std"],
        ):
            print(f"  {int(m):>8} {pb:>+12.4f} {ps:>10.4f} {qb:>+12.4f} {qs:>10.4f}")

    if args.plot:
        if n_sweep is None:
            n_sweep = sample_size_sweep(
                cov,
                n_targets=args.n_targets,
                sample_grid=sample_grid,
                n_outer_y=args.n_outer_y,
                knn_k=args.k,
                seed=args.seed + 100,
                n_reps=args.n_reps,
            )
        if k_sweep is None:
            k_sweep = neighbor_rank_sweep(
                cov,
                n_targets=args.n_targets,
                k_grid=k_grid,
                n_samples=args.posterior_samples,
                n_outer_y=args.n_outer_y,
                seed=args.seed + 200,
                n_reps=args.n_reps,
            )
        if m_sweep is None:
            m_sweep = outer_y_sweep(
                cov,
                n_targets=args.n_targets,
                outer_y_grid=outer_y_grid,
                n_samples=args.posterior_samples,
                knn_k=args.k,
                seed=args.seed + 300,
                n_reps=args.n_reps,
            )
        plot_sweeps(
            n_sweep,
            k_sweep,
            m_sweep,
            out_path=Path(args.out_sweep),
            default_n=args.posterior_samples,
            default_k=args.k,
            default_m=args.n_outer_y,
        )

        if args.n_targets == 1 and args.n_nuisance == 1:
            prior_theta = sample_prior_theta(args.n, cov, seed=args.seed)
            post_plot = posterior_sliced_entropy(
                cov,
                n_targets=1,
                n_outer_y=args.n_outer_y,
                n_samples=args.plot_samples,
                k=args.k,
                seed=args.seed + 2,
                ys=post["ys"],
            )
            plot_clouds(
                prior_theta,
                post,
                post_plot,
                targets=meta,
                out_path=Path(args.out_clouds),
            )
        else:
            print(
                "[plot] skip clouds "
                f"(needs --n-targets 1 --n-nuisance 1; "
                f"got {args.n_targets}, {args.n_nuisance})"
            )

    print()
    print(
        "Analog: prior = sample p(theta)+slice; "
        "post = sample p(theta|y)+slice+avg_y  (oracle guide)."
    )


if __name__ == "__main__":
    main()
