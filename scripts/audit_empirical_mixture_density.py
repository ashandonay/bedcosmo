"""Audit density evaluation of the reflected-and-normalized pattern sampler.

Densities use the simplex chart (a_1,...,a_{m-1}) on each exact support face,
plus log scale and z. Singular pattern components deliberately raise an error.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from prototype_joint_empirical_mixture import JointPatternMixture
from scipy.special import logsumexp


def reflected_log_pdf(x, centers, sigma, lower=0.0, upper=1.0):
    """Gaussian image sum for repeated reflection, with ≥8-sigma tail coverage."""
    if np.any(np.asarray(sigma) <= 0) or upper <= lower:
        raise ValueError("A continuous reflected density needs positive width and interval")
    x, centers, sigma = np.broadcast_arrays(x, centers, sigma)
    width = upper - lower
    images = int(np.ceil(1 + 4 * np.max(sigma) / width))
    value = np.full(x.shape, -np.inf)
    norm = np.log(sigma) + 0.5 * np.log(2 * np.pi)
    for image in range(-images, images + 1):
        for distance in [
            x - centers + 2 * image * width,
            x + centers - 2 * lower + 2 * image * width,
        ]:
            value = np.logaddexp(value, -0.5 * (distance / sigma) ** 2 - norm)
    return np.where((x >= lower) & (x <= upper), value, -np.inf)


def simplex_kernel_log_density(weights, centers, sigma, *, order=128):
    """Integrate the removed radius with Jacobian t**(m-1), per KDE donor.

    Before closure, x_i in [0,1]; after closure x_i=t*a_i. Thus
    q(a|donor) = integral_0^(1/max(a)) prod_i k_i(t*a_i) t^(m-1) dt.
    This is the density in the first m-1 fraction coordinates, not surface area.
    """
    weights = np.atleast_2d(weights)
    centers = np.atleast_2d(centers)
    sigma = np.asarray(sigma)
    if (
        centers.shape[1] != weights.shape[1]
        or np.any(weights <= 0)
        or not np.allclose(weights.sum(axis=1), 1)
    ):
        raise ValueError("Expected positive weights and centers on the same support face")
    if weights.shape[1] == 1:
        return np.zeros((len(weights), len(centers)))
    if np.any(sigma <= 0):
        raise ValueError(
            "Zero-width coefficient kernels need a separate singular-measure calculation"
        )
    nodes, quadrature_weights = leggauss(order)
    output = np.empty((len(weights), len(centers)))
    m = weights.shape[1]
    for j, a in enumerate(weights):
        upper_radius = 1 / a.max()
        t = (nodes + 1) * upper_radius / 2
        log_integrand = np.broadcast_to((m - 1) * np.log(t[:, None]), (order, len(centers))).copy()
        for i in range(m):
            log_integrand += reflected_log_pdf(t[:, None] * a[i], centers[None, :, i], sigma[i])
        output[j] = logsumexp(
            log_integrand + np.log(quadrature_weights[:, None] * upper_radius / 2), axis=0
        )
    return output


def continuous_pattern(model, index):
    active = model.patterns[index]
    widths = model.widths[index]
    # Conservatively support nondegenerate kernels only. Some degenerate
    # coefficient cases could be integrated analytically, but are not handled.
    return bool(np.all(widths[-2:] > 0) and (active.sum() == 1 or np.all(widths[:-2][active] > 0)))


def joint_log_density(model, rows, *, order=128):
    """Unconditional face density, including pattern probability n_pattern/N.

    Unknown patterns have zero probability. Supported singular patterns raise;
    there is no fabricated finite score for atoms or lower-dimensional measures.
    """
    rows = np.atleast_2d(rows)
    if (
        rows.shape[1] != model.rows.shape[1]
        or not np.isfinite(rows).all()
        or np.any(rows[:, :-2] < 0)
        or not np.allclose(rows[:, :-2].sum(axis=1), 1)
    ):
        raise ValueError("Expected finite simplex coefficient rows with scale and z")
    lookup = {tuple(p): i for i, p in enumerate(model.patterns)}
    output = np.full(len(rows), -np.inf)
    for j, row in enumerate(rows):
        pattern = row[:-2] > 0
        index = lookup.get(tuple(pattern))
        if index is None:
            continue
        if not continuous_pattern(model, index):
            raise ValueError(
                "Pattern has singular kernels; a continuous face density is undefined here"
            )
        donors = model.rows[model.members[index]]
        widths = model.widths[index]
        log_values = simplex_kernel_log_density(
            row[:-2][pattern], donors[:, :-2][:, pattern], widths[:-2][pattern], order=order
        )[0]
        for aux in [-2, -1]:
            log_values += reflected_log_pdf(
                row[aux], donors[:, aux], widths[aux], model.lower[aux], model.upper[aux]
            )
        output[j] = logsumexp(log_values) - np.log(len(model.rows))
    return output


def audit(fit_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(fit_path) as saved:
        model = JointPatternMixture(saved["training_rows"], bandwidth=float(saved["bandwidth"]))
    flags = np.array([continuous_pattern(model, i) for i in range(len(model.patterns))])
    draws = model.sample(1000, seed=123)
    lookup = {tuple(p): i for i, p in enumerate(model.patterns)}
    eligible = np.array([flags[lookup[tuple(row[:-2] > 0)]] for row in draws])
    queries = draws[eligible][:24]
    scores = {}
    seconds = {}
    for order in [128, 256, 512, 1024]:
        start = time.perf_counter()
        scores[order] = joint_log_density(model, queries, order=order)
        seconds[order] = time.perf_counter() - start
    # Numerical normalization in a two-component simplex is independently
    # checkable by integrating over a_1 in (0,1), with a_2=1-a_1.
    centers = np.array([[0.2, 0.8], [0.7, 0.3]])
    nodes, weights = leggauss(256)
    a = (nodes + 1) / 2
    toy = simplex_kernel_log_density(
        np.column_stack([a, 1 - a]), centers, np.array([0.12, 0.12]), order=256
    )
    integrals = np.sum(np.exp(toy) * weights[:, None] / 2, axis=0)
    report = {
        "n_queries": len(queries),
        "seconds_by_quadrature_order": seconds,
        "max_abs_log_density_difference_256_vs_1024": float(
            np.max(abs(scores[256] - scores[1024]))
        ),
        "max_abs_log_density_difference_512_vs_1024": float(
            np.max(abs(scores[512] - scores[1024]))
        ),
        "toy_simplex_kernel_integrals": integrals.tolist(),
        "continuous_pattern_mass": float(model.counts[flags].sum() / len(model.rows)),
        "unsupported_degenerate_pattern_mass": float(model.counts[~flags].sum() / len(model.rows)),
        "coordinate_measure": "Counting over exact activation patterns; within each nonsingular face, Lebesgue in first m-1 active fractions, log_c_scale and z.",
        "eig_assessment": [
            "A continuous-pattern score is feasible via one-dimensional radial quadrature, not by scoring the pre-normalized KDE directly.",
            "Singleton/degenerate patterns retain atoms; this audit refuses to score those using a continuous face density.",
            "The existing all-parameter posterior flow lives in continuous ILR coordinates; exact-zero faces have no finite ILR coordinates. Prior and posterior must use a compatible measure before subtracting entropies.",
            "Pattern probabilities, within-face densities and any singular components must be handled consistently; numerical density convergence on 24 draws does not establish EIG correctness.",
            "No BED jobs, prior entropy claim or end-to-end EIG estimate produced.",
        ],
    }
    (output_dir / "density_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(
        output_dir / "density_queries.npz",
        rows=queries,
        **{f"logp_{q}": s for q, s in scores.items()},
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    audit(**vars(parser.parse_args()))
