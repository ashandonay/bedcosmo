"""Sample-based differential-entropy estimators (evaluation only).

Used to compute marginal EIG over subsets of cosmological parameters, where
the joint normalizing-flow guide ``q(theta | y, d)`` and the joint empirical
prior cannot be marginalized in closed form. Both the marginal prior entropy
``H[p(theta_S)]`` and the marginal posterior entropy ``H[q(theta_S | y, d)]``
are therefore estimated from samples.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln


def knn_entropy(x, k=3):
    """Kozachenko-Leonenko k-NN differential-entropy estimate (in bits).

    Estimator (Kraskov et al. 2004 convention, Euclidean norm)::

        H = (-psi(k) + psi(n) + log(c_d) + (d / n) * sum_i log(eps_i)) / log(2)

    where ``eps_i`` is the distance from sample ``i`` to its k-th nearest
    neighbor and ``c_d = pi^(d/2) / Gamma(d/2 + 1)`` is the volume of the unit
    Euclidean ball in ``d`` dimensions.

    Args:
        x: Array of shape ``(n_samples, d)`` (or ``(n_samples,)`` for d=1).
        k: Neighbor rank used for the distance statistic (``k >= 1``).

    Returns:
        Estimated differential entropy in bits (float).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, d = x.shape
    if n <= k + 1:
        raise ValueError(f"knn_entropy needs n > k + 1 = {k + 1} samples, got {n}")

    # Jitter degenerate (constant) columns so neighbor distances are well defined
    # (mirrors the constant-column guard in BaseExperiment.get_guide_samples).
    for j in range(d):
        col = x[:, j]
        if col.max() == col.min():
            scale = abs(col[0]) * 1e-10 if col[0] != 0 else 1e-10
            x[:, j] = col + np.random.normal(0.0, scale, size=n)

    tree = cKDTree(x)
    # query returns the point itself (distance 0) as the first neighbor, so the
    # k-th nearest *other* neighbor is at column index k.
    dists, _ = tree.query(x, k=k + 1, p=2)
    eps = dists[:, -1]
    eps = np.maximum(eps, np.finfo(np.float64).tiny)  # avoid log(0)

    log_vol_unit_ball = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)

    h_nats = -digamma(k) + digamma(n) + log_vol_unit_ball + (d / n) * np.sum(np.log(eps))
    return float(h_nats / np.log(2))
