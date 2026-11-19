"""Response surfaces of EIG over a design space, and one-dimension effect traces.

A run evaluates EIG on a finite pool of designs. To see how EIG responds to one design
dimension on its own, fit a smooth polynomial to the pool and evaluate it along a path
that changes only that dimension.

Two kinds of design space:

* Mixture designs, whose rows sum to a constant (fractions of a fixed budget). A plain
  polynomial is not identifiable there (the intercept and every square are linear
  combinations of the other terms), so the fit is a Scheffe polynomial in the
  proportions ``z = x / sum(x)``. Moving one dimension forces the others to move too; the
  trace follows the Cox direction: dimension ``i`` goes from its smallest to its largest
  pooled proportion while the others keep their proportions relative to each other at
  the reference design.
* Any other design space: an ordinary polynomial with an intercept, in standardized
  coordinates. The trace varies dimension ``i`` over its pooled range with the others
  held at the reference design.

Traces stay inside each dimension's pooled range; a mixture trace can still pass through
compositions the pool did not sample, where the value is the fit's interpolation.
"""
import itertools

import numpy as np

ORDERS = ("linear", "quadratic", "cubic")


def is_mixture(designs, rtol=1e-6):
    """True when every design row sums to the same (nonzero) total."""
    sums = np.asarray(designs, dtype=float).sum(axis=1)
    scale = np.abs(sums).max()
    return bool(scale > 0 and np.ptp(sums) <= rtol * scale)


def _scheffe_basis(z, order):
    q = z.shape[1]
    cols = [z[:, i] for i in range(q)]
    pairs = list(itertools.combinations(range(q), 2))
    if order in ("quadratic", "cubic"):
        cols += [z[:, i] * z[:, j] for i, j in pairs]
    if order == "cubic":
        cols += [z[:, i] * z[:, j] * (z[:, i] - z[:, j]) for i, j in pairs]
        cols += [z[:, i] * z[:, j] * z[:, k] for i, j, k in itertools.combinations(range(q), 3)]
    return np.column_stack(cols)


def _poly_basis(u, order):
    q = u.shape[1]
    cols = [np.ones(len(u))] + [u[:, i] for i in range(q)]
    degree = ORDERS.index(order) + 1
    for d in range(2, degree + 1):
        cols += [np.prod(u[:, list(c)], axis=1)
                 for c in itertools.combinations_with_replacement(range(q), d)]
    return np.column_stack(cols)


class ResponseSurface:
    """Least-squares polynomial fit of ``values`` over ``designs``.

    Attributes: ``mixture`` (bool), ``order``, ``coef``, ``cov`` (the coefficients'
    covariance, ``s^2 (A^T A)^-1`` with ``s^2`` the residual variance on ``n - p`` degrees
    of freedom), ``dof``, ``r2``, ``rmse`` (in-sample), ``loo_rmse`` (leave-one-out, from
    the hat matrix), ``n_params``.

    The covariance treats the residuals as independent noise of one variance. For an EIG
    pool they are mostly the polynomial's misfit, not measurement noise, so the
    confidence band reads as "how precisely this pool pins down a surface of this form".
    """

    def __init__(self, designs, values, order="quadratic", mixture=None):
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}; got {order!r}")
        X = np.asarray(designs, dtype=float)
        y = np.asarray(values, dtype=float)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"designs {X.shape} and values {y.shape} do not match")
        self.order = order
        self.mixture = is_mixture(X) if mixture is None else bool(mixture)
        self.total = float(X.sum(axis=1).mean()) if self.mixture else None
        self.mu = X.mean(axis=0)
        self.sd = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        self.lo, self.hi = X.min(axis=0), X.max(axis=0)
        A = self._basis(X)
        self.n_params = A.shape[1]
        if len(y) <= self.n_params:
            raise ValueError(
                f"{len(y)} designs cannot fit a {order} surface with {self.n_params} "
                "coefficients; use a lower order")
        self.coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ self.coef
        self.rmse = float(np.sqrt(np.mean(resid ** 2)))
        self.r2 = float(1.0 - resid.var() / y.var()) if y.var() > 0 else float("nan")
        hat = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
        self.loo_rmse = float(np.sqrt(np.mean((resid / np.clip(1.0 - hat, 1e-12, None)) ** 2)))
        self.dof = len(y) - self.n_params
        self.cov = (resid @ resid / self.dof) * np.linalg.pinv(A.T @ A)

    def _basis(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.mixture:
            return _scheffe_basis(X / X.sum(axis=1, keepdims=True), self.order)
        return _poly_basis((X - self.mu) / self.sd, self.order)

    def predict(self, designs, coef=None):
        return self._basis(designs) @ (self.coef if coef is None else coef)

    def confidence_band(self, designs, level=68, reference=None):
        """``(fit, half_width)`` of the two-sided ``level``% confidence band of the fit.

        ``half_width = t * se`` with ``se^2 = a^T cov a``, ``a`` the design's basis row, and
        ``t`` Student's quantile on ``dof``. With ``reference``, the band is for the fit
        minus its value at that design (``a`` becomes the difference of the two rows), so
        it is zero there.
        """
        from scipy.stats import t as student_t

        A = self._basis(designs)
        if reference is not None:
            A = A - self._basis(np.asarray(reference, dtype=float)[None])
        fit = A @ self.coef
        se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", A, self.cov, A), 0.0, None))
        return fit, student_t.ppf(0.5 + level / 200.0, self.dof) * se

    def sample_coefs(self, n=1000, seed=0):
        """Coefficient sets drawn from ``N(coef, cov)``, shape ``(n, n_params)``."""
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.coef, self.cov, size=n, method="svd")

    def trace(self, dim, reference, n=200):
        """Designs along dimension ``dim``'s effect path through ``reference``.

        Returns ``(t, designs)``: ``t`` is the dimension's value along the path, in the
        designs' own units, spanning its pooled range.
        """
        ref = np.asarray(reference, dtype=float)
        t = np.linspace(self.lo[dim], self.hi[dim], n)
        if not self.mixture:
            designs = np.tile(ref, (n, 1))
            designs[:, dim] = t
            return t, designs
        z_ref = ref / ref.sum()
        others = np.arange(len(ref)) != dim
        rest = z_ref[others] / z_ref[others].sum()
        z = np.empty((n, len(ref)))
        z[:, dim] = t / self.total
        z[:, others] = (1.0 - z[:, [dim]]) * rest
        return t, z * self.total
