"""Differential-entropy estimators and entropy unit conversions (evaluation only).

Used to compute marginal EIG over subsets of cosmological parameters, where
the joint normalizing-flow guide ``q(theta | y, d)`` and the joint empirical
prior cannot be marginalized in closed form. Both the marginal prior entropy
``H[p(theta_S)]`` and the marginal posterior entropy ``H[q(theta_S | y, d)]``
are therefore estimated from samples.

Units
-----
**Sample-based estimators here return BITS.** The loss path in
``pyro_oed_src`` works in NATS throughout and converts once at the reporting
edge (``Evaluator._store_entropy_eval_results``). That boundary is deliberate:
keep nats inside the torch/nf_loss path, and convert with :func:`nats_to_bits`
when a number is about to be reported or compared against an estimator here.
:func:`plugin_entropy_from_log_probs` is the one function that straddles the
boundary -- it consumes the nats log-probs of the loss path, so it returns
nats, and says so in its name.

Estimator choice
----------------
:func:`knn_entropy` is the production estimator. :func:`kde_entropy` and
:func:`gaussian_entropy` exist so diagnostic scripts can cross-check it against
independent estimators of the same quantity; they are not used in the
production EIG path. Measured on the num_visits 1-D z prior marginal: k-NN
0.484 bits, KDE 0.501, Gaussian 0.579 (biased high by the skew). In higher
dimensions k-NN is biased high (+1.4 bits at 13-D) but KDE degrades faster, so
k-NN is the least-bad option rather than a good one -- prefer reducing the
dimension over trusting either.
"""

import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

_LN2 = np.log(2.0)


def nats_to_bits(values):
    """Convert entropy/EIG quantities from nats to bits (shape preserved)."""
    return np.asarray(values, dtype=np.float64) / _LN2


def bits_to_nats(values):
    """Convert entropy/EIG quantities from bits to nats (shape preserved)."""
    return np.asarray(values, dtype=np.float64) * _LN2


def knn_entropy(x, k=3, warn_duplicates=True):
    """Kozachenko-Leonenko k-NN differential-entropy estimate (in bits).

    Estimator (Kraskov et al. 2004 convention, Euclidean norm)::

        H = (-psi(k) + psi(n) + log(c_d) + (d / n) * sum_i log(eps_i)) / log(2)

    where ``eps_i`` is the distance from sample ``i`` to its k-th nearest
    neighbor and ``c_d = pi^(d/2) / Gamma(d/2 + 1)`` is the volume of the unit
    Euclidean ball in ``d`` dimensions.

    Duplicate rows are dropped before estimation: the estimator assumes a
    continuous density, under which exact ties have probability zero, so any
    tie is an artifact. This matters because ``k + 1`` identical rows drive the
    k-th neighbor distance to exactly zero and ``log(eps)`` then collapses the
    whole estimate -- 4 ties out of 200 samples is enough to take H from +2.6
    to -18 bits. The usual source is a clamp onto a prior bound (see
    ``BaseExperiment._sanitize_physical_samples``), which stacks every
    out-of-bounds row onto the identical bound value.

    Args:
        x: Array of shape ``(n_samples, d)`` (or ``(n_samples,)`` for d=1).
        k: Neighbor rank used for the distance statistic (``k >= 1``).
        warn_duplicates: Emit a ``RuntimeWarning`` when ties are dropped. The
            message is deliberately constant (no counts): Python's default
            warning filter dedups on ``(text, category, lineno)``, so embedding
            a varying count makes every call a "new" warning and spams the log.
            Callers in hot loops should pass ``False`` and report an aggregate.

    Returns:
        Estimated differential entropy in bits (float).
    """
    # copy=True: the degenerate-column jitter below writes into ``x``, which
    # would otherwise mutate the caller's array in place.
    x = np.array(x, dtype=np.float64, copy=True)
    if x.ndim == 1:
        x = x[:, None]
    n, d = x.shape
    if n <= k + 1:
        raise ValueError(f"knn_entropy needs n > k + 1 = {k + 1} samples, got {n}")

    # Jitter degenerate (constant) columns so neighbor distances are well defined
    # (mirrors the constant-column guard in BaseExperiment.get_guide_samples).
    # Runs before the de-duplication below so a fully-constant column stays a
    # jittered n-row estimate rather than collapsing to a single unique row.
    for j in range(d):
        col = x[:, j]
        if col.max() == col.min():
            scale = abs(col[0]) * 1e-10 if col[0] != 0 else 1e-10
            x[:, j] = col + np.random.normal(0.0, scale, size=n)

    unique_x = np.unique(x, axis=0)
    n_dup = n - unique_x.shape[0]
    if n_dup:
        if warn_duplicates:
            # Constant text on purpose -- see the warn_duplicates docstring note.
            warnings.warn(
                "knn_entropy: dropping duplicate rows before estimation. Ties have "
                "probability zero under a continuous density, so they usually mean "
                "samples were clamped onto a bound; the estimate would otherwise "
                "collapse toward -inf. Prefer dropping out-of-bounds samples "
                "upstream. Pass warn_duplicates=False and report an aggregate if "
                "this is a hot loop.",
                RuntimeWarning,
                stacklevel=2,
            )
        x = unique_x
        n = x.shape[0]
        if n <= k + 1:
            raise ValueError(
                f"knn_entropy needs n > k + 1 = {k + 1} unique samples, got {n} "
                f"after dropping {n_dup} duplicates"
            )

    tree = cKDTree(x)
    # query returns the point itself (distance 0) as the first neighbor, so the
    # k-th nearest *other* neighbor is at column index k.
    dists, _ = tree.query(x, k=k + 1, p=2)
    eps = dists[:, -1]
    eps = np.maximum(eps, np.finfo(np.float64).tiny)  # avoid log(0)

    log_vol_unit_ball = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)

    h_nats = -digamma(k) + digamma(n) + log_vol_unit_ball + (d / n) * np.sum(np.log(eps))
    return float(h_nats / _LN2)


def kde_entropy(x, max_eval_points=None, seed=0):
    """Gaussian-KDE plug-in differential entropy ``H = -E[log p_hat]`` (in bits).

    Structurally immune to duplicate rows (ties merely add density mass rather
    than collapsing a neighbor distance), which makes it a useful cross-check on
    :func:`knn_entropy`. Degrades above ~3 dimensions as Scott-rule bandwidth
    selection and density resolution break down.

    Args:
        x: Array of shape ``(n_samples, d)`` (or ``(n_samples,)`` for d=1).
        max_eval_points: Evaluate ``log p_hat`` on at most this many points
            (subsampled without replacement). ``gaussian_kde.logpdf`` is
            O(n_eval * n_fit), so resubstitution on large samples is quadratic.
            ``None`` (default) evaluates on every point.
        seed: Seed for the evaluation-point subsample.

    Returns:
        Estimated differential entropy in bits (float).
    """
    from scipy.stats import gaussian_kde

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, _ = x.shape
    if n < 2:
        raise ValueError(f"kde_entropy needs at least 2 samples, got {n}")

    kde = gaussian_kde(x.T)  # Scott bandwidth
    eval_x = x
    if max_eval_points is not None and n > int(max_eval_points):
        rng = np.random.default_rng(seed)
        eval_x = x[rng.choice(n, int(max_eval_points), replace=False)]
    # clip: a KDE evaluated far from its mass can underflow to exactly 0.
    log_p = np.log(np.clip(kde(eval_x.T), 1e-300, None))
    return float(-np.mean(log_p) / _LN2)


def gaussian_entropy(x):
    """Gaussian plug-in differential entropy from the sample covariance (in bits).

    Shape-blind: exact only for a Gaussian, and biased HIGH for skewed or
    bounded densities (measured +0.08 bits on the num_visits z prior marginal).
    Provided as a cheap, very low-variance baseline for diagnostics.

    Args:
        x: Array of shape ``(n_samples, d)`` (or ``(n_samples,)`` for d=1).

    Returns:
        Estimated differential entropy in bits (float).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, d = x.shape
    if n <= d:
        raise ValueError(f"gaussian_entropy needs n > d = {d} samples, got {n}")

    half_log2_2pie = 0.5 * np.log2(2.0 * np.pi * np.e)
    if d == 1:
        return float(half_log2_2pie + np.log2(max(float(np.std(x)), 1e-12)))
    cov = np.cov(x, rowvar=False)
    sign, logabsdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError(
            "gaussian_entropy: sample covariance is singular or not positive "
            "definite (degenerate support?); reduce the dimension first"
        )
    return float(d * half_log2_2pie + 0.5 * logabsdet / _LN2)


def plugin_entropy_from_log_probs(log_probs, param_names):
    """Per-design plug-in entropy ``H = -E[log p]`` in NATS, from prior log-probs.

    Returns nats, not bits: this consumes the nats log-probs of the ``nf_loss``
    path and its result is subtracted from a nats posterior entropy. Convert at
    the reporting edge with :func:`nats_to_bits`. Operates on torch tensors and
    keeps the per-design dimension (averages over the particle axis 0).

    Args:
        log_probs: Dict with either a ``"joint"`` entry or one entry per name in
            ``param_names``, each shaped ``(n_particles, n_designs)``.
        param_names: Parameter names to sum over when the prior is factorized.

    Returns:
        Per-design entropy tensor in nats.
    """
    if "joint" in log_probs:
        return -log_probs["joint"].mean(dim=0)
    return -sum(log_probs[name].mean(dim=0) for name in param_names)
