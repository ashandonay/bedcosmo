#!/usr/bin/env python
"""Diagnostic: compare held-out error of quick NMF bases at several ranks.

Reads the matrix written by ``desi.build_matrix``. The bases fitted here are
not used by ``desi.build_prior``, which learns its own production basis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from speclite import filters as speclite_filters  # noqa: E402

from ..paths import get_desi_training_data_dir  # noqa: E402
from .support import select_wavelength_support  # noqa: E402
from .weighted_nmf import (  # noqa: E402
    fit_weighted_nmf,
    infer_coefficients,
    weighted_reconstruction_error,
)

GRID = "#D9DDE3"


def desi_covered_lsst_color_rms(
    flux: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    basis: np.ndarray,
    wave_rest: np.ndarray,
    redshift: np.ndarray,
    *,
    minimum_band_coverage: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare reconstructed LSST colors where DESI covers each filter."""
    reconstruction = coefficients @ basis
    rms = np.full(len(flux), np.nan, dtype=float)
    band_count = np.zeros(len(flux), dtype=int)
    loaded_filters = [speclite_filters.load_filter(f"lsst2023-{band}") for band in "ugrizy"]
    full_norm = []
    for loaded in loaded_filters:
        filter_wave = np.asarray(loaded.wavelength, dtype=float)
        full_norm.append(
            float(np.trapz(np.asarray(loaded(filter_wave), float) * filter_wave, filter_wave))
        )

    for row, z in enumerate(np.asarray(redshift, dtype=float)):
        observed_wave = wave_rest * (1.0 + z)
        observed = weights[row] > 0
        delta_magnitude: list[float] = []
        for loaded, normalization in zip(loaded_filters, full_norm):
            response = np.asarray(loaded(observed_wave), dtype=float)
            kernel = response * observed_wave
            covered = float(np.trapz(kernel * observed, observed_wave) / normalization)
            if covered < minimum_band_coverage:
                continue
            reference = float(np.trapz(flux[row] * kernel * observed, observed_wave))
            predicted = float(np.trapz(reconstruction[row] * kernel * observed, observed_wave))
            if reference > 0 and predicted > 0:
                delta_magnitude.append(-2.5 * np.log10(predicted / reference))
        band_count[row] = len(delta_magnitude)
        if len(delta_magnitude) >= 2:
            delta = np.asarray(delta_magnitude)
            rms[row] = float(np.sqrt(np.mean((delta - np.mean(delta)) ** 2)))
    return rms, band_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--training-matrix",
        type=Path,
        default=None,
        help="Matrix from desi.build_matrix (default: <num_visits>/desi_training_data/"
        "desi_rest_frame_training_matrix.npz)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Default: the matrix's directory"
    )
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--smooth-sigma-aa", type=float, default=10.0)
    parser.add_argument("--observations-per-component", type=int, default=10)
    parser.add_argument("--wavelength-support-rank", type=int, default=10)
    parser.add_argument(
        "--minimum-wavelength-contributors",
        type=int,
        default=None,
        help="Explicit contributors required per wavelength; overrides the rank-scaled default",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-lsst-band-coverage", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_figure(
    metrics: pd.DataFrame,
    wave: np.ndarray,
    bases: dict[int, np.ndarray],
    coverage_wave: np.ndarray,
    coverage_fraction: np.ndarray,
    required_contributors: int,
    n_spectra: int,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(metrics["rank"], metrics["test_median_wrms"], marker="o", label="median")
    axes[0].plot(metrics["rank"], metrics["test_p90_wrms"], marker="o", label="90th percentile")
    axes[0].set(
        xlabel="DESI basis rank",
        ylabel="Held-out weighted spectral RMS",
        title="Direct-DESI reconstruction error",
        xticks=metrics["rank"],
    )
    axes[0].legend(frameon=False)

    axes[1].plot(metrics["rank"], metrics["test_median_color_rms"], marker="o", label="median")
    axes[1].plot(
        metrics["rank"], metrics["test_p90_color_rms"], marker="o", label="90th percentile"
    )
    axes[1].set(
        xlabel="DESI basis rank",
        ylabel="Held-out LSST color RMS [mag]",
        title="Colors over DESI-covered LSST bands",
        xticks=metrics["rank"],
    )
    axes[1].legend(frameon=False)

    axes[2].plot(coverage_wave, coverage_fraction, color="#4C78A8", lw=2)
    axes[2].axhline(
        required_contributors / n_spectra,
        color="#E45756",
        linestyle="--",
        label=f"minimum contributors (N={required_contributors:,})",
    )
    axes[2].fill_between(
        coverage_wave,
        0,
        coverage_fraction,
        where=coverage_fraction >= required_contributors / n_spectra,
        color="#72B7B2",
        alpha=0.25,
        label="basis wavelength support",
    )
    axes[2].set(
        xlabel="Rest wavelength [Å]",
        ylabel="Fraction of spectra observed",
        ylim=(0, 1.03),
        title="DESI rest-frame coverage",
    )
    axes[2].legend(frameon=False)

    rank = max(bases)
    basis = bases[rank]
    for index, spectrum in enumerate(basis):
        scale = float(np.mean(spectrum))
        axes[3].plot(wave, spectrum / scale, label=f"B{index + 1}")
    axes[3].set(
        xlabel="Rest wavelength [Å]",
        ylabel="Component flux / component mean",
        title=f"Rank-{rank} nonnegative DESI basis",
    )
    axes[3].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.grid(True, color=GRID, alpha=0.7, linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(f"Basis learned from {n_spectra:,} masked DESI coadd spectra")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    matrix_path = (
        Path(
            args.training_matrix
            or get_desi_training_data_dir() / "desi_rest_frame_training_matrix.npz"
        )
        .expanduser()
        .resolve()
    )
    output_dir = Path(args.output_dir or matrix_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(matrix_path) as data:
        manifest = pd.DataFrame(
            {"targetid": data["targetid"], "healpix": data["healpix"], "z": data["redshift"]}
        )
        wave = data["wave_rest_aa"]
        flux = data["flux"]
        weights = data["relative_ivar"]
    wave_step = np.diff(wave)
    if not np.allclose(wave_step, wave_step[0]):
        raise ValueError(f"{matrix_path} does not have a uniform wavelength grid")
    wave_step = float(wave_step[0])

    rng = np.random.default_rng(args.seed)
    test = rng.random(len(manifest)) < args.test_fraction
    if not np.any(test) or not np.any(~test):
        raise ValueError("Train/test split is empty")
    wavelength_coverage = np.mean(weights > 0, axis=0)
    learned, wavelength_contributors, required_contributors = select_wavelength_support(
        weights[~test],
        args.ranks,
        observations_per_component=args.observations_per_component,
        minimum_contributors=args.minimum_wavelength_contributors,
        support_rank=args.wavelength_support_rank,
    )
    if np.count_nonzero(learned) < 20:
        raise ValueError(
            f"Too few wavelength bins have at least {required_contributors} contributors"
        )
    learned_wave = wave[learned]
    learned_flux = flux[:, learned]
    learned_weights = weights[:, learned]
    pd.DataFrame(
        {
            "wave_rest_aa": wave,
            "training_contributing_spectra": wavelength_contributors,
            "basis_support": learned,
        }
    ).to_csv(output_dir / "rank_comparison_support.csv", index=False)

    rows: list[dict[str, float | int]] = []
    bases: dict[int, np.ndarray] = {}
    coefficient_tables: list[pd.DataFrame] = []
    for rank in sorted(set(args.ranks)):
        _, basis, losses = fit_weighted_nmf(
            learned_flux[~test],
            learned_weights[~test],
            rank,
            iterations=args.iterations,
            smooth_sigma_pixels=args.smooth_sigma_aa / wave_step,
            seed=args.seed,
        )
        coefficients = infer_coefficients(learned_flux, learned_weights, basis)
        error = weighted_reconstruction_error(learned_flux, learned_weights, coefficients, basis)
        color_rms, color_band_count = desi_covered_lsst_color_rms(
            learned_flux,
            learned_weights,
            coefficients,
            basis,
            learned_wave,
            manifest["z"].to_numpy(float),
            minimum_band_coverage=args.min_lsst_band_coverage,
        )
        bases[rank] = basis
        np.savetxt(
            output_dir / f"desi_basis_rank{rank}.csv",
            np.column_stack([learned_wave, basis.T]),
            delimiter=",",
            header="wave_rest_aa," + ",".join(f"B{i + 1}" for i in range(rank)),
            comments="",
        )
        total = coefficients.sum(axis=1)
        fractions = np.divide(
            coefficients,
            total[:, None],
            out=np.zeros_like(coefficients),
            where=total[:, None] > 0,
        )
        coefficient_table = manifest.copy()
        coefficient_table.insert(3, "rank", rank)
        coefficient_table["log_scale"] = np.log(np.maximum(total, 1e-300))
        coefficient_table["held_out"] = test
        coefficient_table["weighted_spectral_rms"] = error
        coefficient_table["lsst_color_rms"] = color_rms
        coefficient_table["lsst_covered_band_count"] = color_band_count
        for index in range(rank):
            coefficient_table[f"a{index + 1}"] = fractions[:, index]
        coefficient_tables.append(coefficient_table)
        rows.append(
            {
                "rank": rank,
                "n_train": int(np.sum(~test)),
                "n_test": int(np.sum(test)),
                "test_median_wrms": float(np.nanmedian(error[test])),
                "test_p90_wrms": float(np.nanpercentile(error[test], 90)),
                "train_median_wrms": float(np.nanmedian(error[~test])),
                "test_color_count": int(np.sum(test & np.isfinite(color_rms))),
                "test_median_color_rms": float(np.nanmedian(color_rms[test])),
                "test_p90_color_rms": float(np.nanpercentile(color_rms[test], 90)),
                "final_training_loss": float(losses[-1]),
            }
        )
        print(
            f"rank {rank}: held-out median WRMS={rows[-1]['test_median_wrms']:.4f}, "
            f"color RMS={rows[-1]['test_median_color_rms']:.4f} mag "
            f"(n={rows[-1]['test_color_count']})",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "rank_comparison.csv", index=False)
    pd.concat(coefficient_tables, ignore_index=True).to_csv(
        output_dir / "desi_basis_coefficients.csv", index=False
    )
    parameters = vars(args).copy()
    parameters.update(
        {
            "training_matrix": str(matrix_path),
            "output_dir": str(output_dir),
            "n_spectra": len(manifest),
            "learned_wave_min_aa": float(learned_wave.min()),
            "learned_wave_max_aa": float(learned_wave.max()),
            "n_learned_wavelength_bins": int(np.sum(learned)),
            "required_wavelength_contributors": required_contributors,
            "wavelength_support_rule": (
                "largest contiguous interval with at least the required number "
                "of observed spectra in every bin"
            ),
        }
    )
    for key, value in list(parameters.items()):
        if isinstance(value, Path):
            parameters[key] = str(value)
    (output_dir / "rank_comparison_provenance.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n"
    )
    make_figure(
        metrics,
        learned_wave,
        bases,
        wave,
        wavelength_coverage,
        required_contributors,
        len(manifest),
        output_dir / "desi_basis_rank_comparison.png",
    )
    print(f"Wrote DESI rank comparison to {output_dir}")


if __name__ == "__main__":
    main()
