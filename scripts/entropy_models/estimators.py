"""Reliable differential-entropy estimators for the transformed prior H(z).

Target use case: estimate ``H(z)`` for a ``D``-dimensional prior in the
Gaussianized (``input_transform_type="joint"``) space, where the KDE plug-in is
unreliable (bandwidth/dimension bias, see ``kde_entropy_toy``). Two complementary
estimators are provided:

1. :func:`flow_plugin_entropy` -- normalizing-flow plug-in. Fits an unconditional
   zuko NSF on a train split and scores a held-out split: ``H = -E_eval[log p_flow]``.
   This is the primary estimator; the held-out split controls in-sample optimism.
2. :func:`cumulant_negentropy_entropy` -- Gaussian baseline minus a 3rd/4th-order
   cumulant (Edgeworth) negentropy. No density estimation; cheap and low-variance,
   but only captures low-order non-Gaussianity. Used as an independent cross-check.

Agreement between the two is the reliability signal. :func:`make_correlated_nongaussian`
builds a ``D``-dim correlated, non-Gaussian distribution with a tight analytic-MC
reference entropy for validation at the real problem's dimensionality.
"""

from __future__ import annotations

import math

import numpy as np
import torch

LOG_2PI = math.log(2.0 * math.pi)


def gaussian_entropy(cov: np.ndarray) -> float:
    """Differential entropy (nats) of N(mu, cov)."""
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("cov must be positive definite")
    return float(0.5 * (d * (1.0 + LOG_2PI) + logdet))


# ----------------------------------------------------------------------
# 1. Normalizing-flow plug-in entropy (primary)
# ----------------------------------------------------------------------


def flow_plugin_entropy(
    samples: np.ndarray,
    *,
    n_train: int | None = None,
    transforms: int = 3,
    bins: int = 8,
    hidden_features: tuple[int, ...] = (128, 128),
    epochs: int = 400,
    lr: float = 1e-3,
    batch_size: int = 4096,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    return_diagnostics: bool = False,
):
    """Held-out NF plug-in entropy ``H = -E_eval[log p_flow]`` in nats.

    Fits an unconditional zuko NSF on a train split (standardized inputs) and
    scores a disjoint eval split, so the estimate is not optimistically biased by
    the flow memorizing its own training points.
    """
    import zuko

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {x.shape}")
    n, d = x.shape
    if n_train is None:
        n_train = n // 2
    if not (0 < n_train < n):
        raise ValueError("n_train must split the sample set")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    x_tr = x[perm[:n_train]]
    x_ev = x[perm[n_train:]]

    # Standardize on the train split; entropy of the original variable is the
    # entropy of the standardized variable plus sum(log std).
    mu = x_tr.mean(axis=0)
    std = x_tr.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    log_jac = float(np.sum(np.log(std)))

    dev = torch.device(device)
    torch.manual_seed(seed)
    xt = torch.as_tensor((x_tr - mu) / std, dtype=torch.float32, device=dev)
    xe = torch.as_tensor((x_ev - mu) / std, dtype=torch.float32, device=dev)

    # Pin float32 explicitly: importing bedcosmo can set torch's default dtype
    # to float64, which would otherwise mismatch the float32 inputs above.
    flow = zuko.flows.NSF(
        features=d,
        context=0,
        transforms=transforms,
        bins=bins,
        hidden_features=hidden_features,
        randperm=True,
    ).to(dev, dtype=torch.float32)
    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)

    best_eval = math.inf
    best_state = None
    history = []
    n_tr = xt.shape[0]
    for _epoch in range(epochs):
        flow.train()
        idx = torch.randperm(n_tr, device=dev)
        for start in range(0, n_tr, batch_size):
            batch = xt[idx[start : start + batch_size]]
            opt.zero_grad()
            loss = -flow().log_prob(batch).mean()
            loss.backward()
            opt.step()
        flow.eval()
        with torch.no_grad():
            eval_nll = float(-flow().log_prob(xe).mean().item())
        history.append(eval_nll)
        if eval_nll < best_eval:
            best_eval = eval_nll
            best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}

    if best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval()
    with torch.no_grad():
        h_std = float(-flow().log_prob(xe).mean().item())

    h_nats = h_std + log_jac
    if return_diagnostics:
        return h_nats, {"eval_nll_history": history, "best_eval_nll": best_eval, "log_jac": log_jac}
    return h_nats


# ----------------------------------------------------------------------
# 2. Cumulant (Edgeworth) negentropy entropy (cross-check)
# ----------------------------------------------------------------------


def _fourth_moment_tensor(y: np.ndarray, chunk: int = 256) -> np.ndarray:
    """E[y_i y_j y_k y_l] accumulated in chunks to bound memory (D**4 output)."""
    n, d = y.shape
    m4 = np.zeros((d, d, d, d), dtype=np.float64)
    for start in range(0, n, chunk):
        b = y[start : start + chunk]
        m4 += np.einsum("ni,nj,nk,nl->ijkl", b, b, b, b, optimize=True)
    return m4 / n


def cumulant_negentropy_entropy(
    samples: np.ndarray,
    *,
    max_samples: int = 40_000,
    seed: int = 0,
    return_diagnostics: bool = False,
):
    """Entropy via ``H = H_gauss(cov) - J``, with negentropy ``J`` from cumulants.

    Whitens the data, then approximates the negentropy (KL from the best-fit
    Gaussian) with the multivariate Gram-Charlier / Edgeworth expansion::

        J ~= (1/12) * sum kappa3_{ijk}^2  +  (1/48) * sum kappa4_{ijkl}^2

    where kappa3/kappa4 are the standardized third/fourth cumulants of the
    whitened data. Captures only low-order non-Gaussianity by construction.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"samples must be 2D, got shape {x.shape}")
    n, d = x.shape
    if n > max_samples:
        rng = np.random.default_rng(seed)
        x = x[rng.permutation(n)[:max_samples]]

    mu = x.mean(axis=0)
    xc = x - mu
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    h_gauss = gaussian_entropy(cov)

    # Whiten so the residual non-Gaussianity is cumulant structure only.
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    w_inv_sqrt = evecs @ np.diag(evals**-0.5) @ evecs.T
    y = xc @ w_inv_sqrt  # cov(y) ~= I

    # Third cumulants == third moments (zero mean).
    k3 = np.einsum("ni,nj,nk->ijk", y, y, y, optimize=True) / y.shape[0]
    # Fourth cumulants: m4 minus the Gaussian (Isserlis) part.
    m4 = _fourth_moment_tensor(y)
    eye = np.eye(d)
    gauss4 = (
        np.einsum("ij,kl->ijkl", eye, eye)
        + np.einsum("ik,jl->ijkl", eye, eye)
        + np.einsum("il,jk->ijkl", eye, eye)
    )
    k4 = m4 - gauss4

    j3 = float(np.sum(k3**2) / 12.0)
    j4 = float(np.sum(k4**2) / 48.0)
    negentropy = j3 + j4
    h_nats = h_gauss - negentropy
    if return_diagnostics:
        return h_nats, {"h_gauss": h_gauss, "negentropy": negentropy, "j3": j3, "j4": j4}
    return h_nats


# ----------------------------------------------------------------------
# Ground truth: correlated, non-Gaussian, with a tight analytic-MC entropy
# ----------------------------------------------------------------------


def make_correlated_nongaussian(
    d: int,
    n: int,
    *,
    seed: int = 0,
    strength: float = 0.6,
    cov_jitter: float = 0.5,
    ref_samples: int = 400_000,
) -> tuple[np.ndarray, float]:
    """Return ``(samples (n, d), H_true_nats)``.

    Construction: correlated Gaussian ``z ~ N(0, Sigma)`` (analytic entropy) pushed
    through the elementwise smooth map ``y_i = z_i + strength * sin(z_i)``. The map
    is diagonal and monotone (``|strength| < 1``), so by change of variables::

        H(y) = H(z) + E_z[ sum_i log(1 + strength*cos(z_i)) ]

    The expectation is evaluated on a large independent draw for a tight reference.
    The result is jointly non-Gaussian with non-Gaussian marginals *and* dependence.
    """
    if not -1.0 < strength < 1.0:
        raise ValueError("strength must be in (-1, 1) for monotonicity")
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    cov = a @ a.T / d + np.eye(d) * cov_jitter
    chol = np.linalg.cholesky(cov)
    h_z = gaussian_entropy(cov)

    def push(z):
        return z + strength * np.sin(z)

    z = rng.standard_normal((n, d)) @ chol.T
    y = push(z)

    # Tight reference for the Jacobian term on a large independent draw.
    z_ref = rng.standard_normal((ref_samples, d)) @ chol.T
    jac_term = float(np.mean(np.sum(np.log1p(strength * np.cos(z_ref)), axis=1)))
    h_true = h_z + jac_term
    return y, h_true
