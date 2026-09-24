#!/usr/bin/env python
"""Compare weighted factorization algorithms on signed DESI data."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import nnls

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .support import select_wavelength_support
from .weighted_nmf import infer_coefficients, weighted_reconstruction_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--training-matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=(4, 6, 8))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("anls", "nearly_nmf"),
        default=("anls", "nearly_nmf"),
        help="Factorization methods to fit; completed results from other methods are retained.",
    )
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--initialization-seed", type=int, default=7301)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--observations-per-component", type=int, default=10)
    parser.add_argument("--wavelength-support-rank", type=int, default=10)
    parser.add_argument(
        "--minimum-wavelength-contributors",
        type=int,
        default=None,
        help="Explicit contributors required per wavelength; overrides the rank-scaled default",
    )
    parser.add_argument("--max-spectra", type=int, default=0)
    parser.add_argument("--anls-max-updates", type=int, default=30)
    parser.add_argument("--nearly-max-updates", type=int, default=500)
    parser.add_argument("--nearly-check-every", type=int, default=10)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--polish-selected", action="store_true")
    parser.add_argument("--polish-anls-max-updates", type=int, default=120)
    parser.add_argument("--polish-nearly-max-updates", type=int, default=3000)
    parser.add_argument("--polish-relative-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def nearly_nmf_package_metadata() -> dict[str, str]:
    """Return installed package provenance without making it a hard dependency."""
    try:
        package = distribution("nearly_nmf")
    except PackageNotFoundError:
        return {"nearly_nmf_version": "not installed"}
    metadata = {"nearly_nmf_version": package.version}
    direct_url = package.read_text("direct_url.json")
    if direct_url:
        source = json.loads(direct_url)
        metadata["nearly_nmf_install_url"] = str(source.get("url", ""))
        commit = source.get("vcs_info", {}).get("commit_id")
        if commit:
            metadata["nearly_nmf_commit"] = str(commit)
    return metadata


def require_compatible_resume(path: Path, provenance: dict[str, object]) -> None:
    """Refuse to mix cached fits made on a different matrix or support grid."""
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    compatibility_keys = (
        "training_matrix",
        "max_spectra",
        "split_seed",
        "initialization_seed",
        "train_fraction",
        "validation_fraction",
        "observations_per_component",
        "minimum_wavelength_contributors",
        "wavelength_support_rank",
        "required_wavelength_contributors",
        "n_spectra",
        "n_wavelengths",
        "wave_min_aa",
        "wave_max_aa",
    )
    mismatches = [
        key for key in compatibility_keys if previous.get(key) != provenance.get(key)
    ]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            "Output directory contains incompatible cached factorization results "
            f"({joined}). Use a new --output-dir instead of mixing wavelength grids."
        )


def shared_initialization(
    n_spectra: int, n_wavelengths: int, rank: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly positive factors without inspecting the flux matrix."""
    rng = np.random.default_rng(seed)
    coefficients = rng.uniform(0.25, 1.75, size=(n_spectra, rank)) / rank
    basis = rng.uniform(0.5, 1.5, size=(rank, n_wavelengths))
    return coefficients, basis


def weighted_objective(
    flux: np.ndarray, weights: np.ndarray, coefficients: np.ndarray, basis: np.ndarray
) -> float:
    """Return the inverse-variance-weighted squared reconstruction error."""
    return float(np.sum(weights * (flux - coefficients @ basis) ** 2))


def normalize_factorization(
    coefficients: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove factor scale degeneracy while preserving the reconstruction."""
    norms = np.mean(basis, axis=1)
    valid = norms > 0
    basis[valid] /= norms[valid, None]
    coefficients[:, valid] *= norms[valid]
    return coefficients, basis


def fit_anls(
    flux: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    basis: np.ndarray,
    *,
    max_updates: int,
    relative_tolerance: float,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], int]:
    """Fit by alternating exact weighted nonnegative least squares."""
    coefficients = np.array(coefficients, dtype=float, copy=True)
    basis = np.array(basis, dtype=float, copy=True)
    losses = [weighted_objective(flux, weights, coefficients, basis)]
    elapsed = [0.0]
    started = time.perf_counter()
    stable = 0
    for _ in range(max_updates):
        coefficients = infer_coefficients(flux, weights, basis)
        for column in range(flux.shape[1]):
            observed = weights[:, column] > 0
            if np.count_nonzero(observed) < basis.shape[0]:
                basis[:, column] = 0.0
                continue
            sqrt_weight = np.sqrt(weights[observed, column])
            design = coefficients[observed] * sqrt_weight[:, None]
            response = flux[observed, column] * sqrt_weight
            basis[:, column], _ = nnls(design, response)
        coefficients, basis = normalize_factorization(coefficients, basis)
        losses.append(weighted_objective(flux, weights, coefficients, basis))
        elapsed.append(time.perf_counter() - started)
        relative_change = (losses[-2] - losses[-1]) / max(losses[-2], 1e-300)
        stable = stable + 1 if relative_change < relative_tolerance else 0
        if stable >= patience:
            break
    return coefficients, basis, losses, elapsed, len(losses) - 1


def fit_nearly_nmf(
    flux: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    basis: np.ndarray,
    *,
    max_updates: int,
    check_every: int,
    relative_tolerance: float,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], int]:
    """Fit with the paper authors' Nearly-NMF implementation."""
    try:
        from nearly_nmf.nmf import fit_NMF
    except ImportError as error:
        raise RuntimeError(
            "Install https://github.com/dylanagreen/nearly_nmf to run this evaluator"
        ) from error

    coefficients = np.array(coefficients, dtype=float, copy=True)
    basis = np.array(basis, dtype=float, copy=True)
    losses = [weighted_objective(flux, weights, coefficients, basis)]
    elapsed = [0.0]
    started = time.perf_counter()
    stable = 0
    completed = 0
    while completed < max_updates:
        updates = min(check_every, max_updates - completed)
        coefficients, basis = fit_NMF(
            flux,
            weights,
            H_start=coefficients,
            W_start=basis,
            n_iter=updates,
            algorithm="nearly",
            transpose=True,
            use_gpu=False,
        )
        coefficients, basis = normalize_factorization(coefficients, basis)
        completed += updates
        losses.append(weighted_objective(flux, weights, coefficients, basis))
        elapsed.append(time.perf_counter() - started)
        relative_change = (losses[-2] - losses[-1]) / max(losses[-2], 1e-300)
        stable = stable + 1 if relative_change < relative_tolerance else 0
        if stable >= patience:
            break
    return coefficients, basis, losses, elapsed, completed


def per_spectrum_metrics(
    flux: np.ndarray, weights: np.ndarray, coefficients: np.ndarray, basis: np.ndarray
) -> dict[str, float]:
    """Summarize held-out residuals in uncertainty and normalized-flux units."""
    residual = flux - coefficients @ basis
    observed_count = np.sum(weights > 0, axis=1)
    dof = np.maximum(observed_count - basis.shape[0], 1)
    chi2_reduced = np.sum(weights * residual**2, axis=1) / dof
    wrms = weighted_reconstruction_error(flux, weights, coefficients, basis)
    observed = weights > 0
    pulls = np.sqrt(weights[observed]) * residual[observed]
    total_weight = np.sum(weights)
    weighted_flux_bias = float(np.sum(weights * (coefficients @ basis - flux)) / total_weight)
    return {
        "median_reduced_chi2": float(np.nanmedian(chi2_reduced)),
        "p75_reduced_chi2": float(np.nanpercentile(chi2_reduced, 75)),
        "p90_reduced_chi2": float(np.nanpercentile(chi2_reduced, 90)),
        "p95_reduced_chi2": float(np.nanpercentile(chi2_reduced, 95)),
        "median_wrms": float(np.nanmedian(wrms)),
        "p90_wrms": float(np.nanpercentile(wrms, 90)),
        "pull_mean": float(np.mean(pulls)),
        "pull_std": float(np.std(pulls)),
        "weighted_flux_bias": weighted_flux_bias,
    }


def evaluate_basis(
    flux: np.ndarray,
    weights: np.ndarray,
    basis: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    coefficients = infer_coefficients(flux[indices], weights[indices], basis)
    return per_spectrum_metrics(flux[indices], weights[indices], coefficients, basis)


def make_summary_plot(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    colors = {"anls": "#4C78A8", "nearly_nmf": "#E45756"}
    labels = {"anls": "ANLS", "nearly_nmf": "Nearly-NMF"}
    ranks = np.array(sorted(metrics["rank"].unique()))
    for method, group in metrics.groupby("method"):
        summary = group.groupby("rank")
        selected = group[group["selected_for_test"]].sort_values("rank")
        axes[0].plot(
            selected["rank"],
            selected["test_pull_std"],
            marker="o",
            color=colors[method],
            label=labels[method],
        )
        median = summary["elapsed_seconds"].median().reindex(ranks).to_numpy()
        low = summary["elapsed_seconds"].min().reindex(ranks).to_numpy()
        high = summary["elapsed_seconds"].max().reindex(ranks).to_numpy()
        axes[1].plot(
            ranks, median, marker="o", color=colors[method], label=labels[method]
        )
        axes[1].fill_between(ranks, low, high, color=colors[method], alpha=0.16)
    axes[0].axhline(1.0, color="black", lw=1.0, linestyle="--", label="Ideal = 1")
    axes[0].set(
        xlabel="Basis rank",
        ylabel="Standardized residual width",
        xticks=ranks,
        title="Test residual calibration",
    )
    axes[1].set(
        xlabel="Basis rank",
        ylabel="Wall time [s]",
        xticks=ranks,
        title="Initial fit time across five starts\nmedian and min–max per start",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Signed-DESI factorization comparison")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_residual_plot(
    metrics: pd.DataFrame,
    flux: np.ndarray,
    weights: np.ndarray,
    wave: np.ndarray,
    test: np.ndarray,
    output_dir: Path,
) -> None:
    """Compare residual calibration for the highest-rank selected solutions."""
    selected = metrics[metrics["selected_for_test"]]
    rank = int(selected["rank"].max())
    selected = selected[selected["rank"] == rank]
    colors = {"anls": "#4C78A8", "nearly_nmf": "#E45756"}
    labels = {"anls": "ANLS", "nearly_nmf": "Nearly-NMF"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for _, item in selected.iterrows():
        method = str(item["method"])
        polished_path = output_dir / f"polished_{method}_rank{rank}_basis.npz"
        if polished_path.exists():
            basis = np.load(polished_path)["basis"]
        else:
            stem = f"{method}_rank{rank}_start{int(item['start'])}_basis.npz"
            basis = np.load(output_dir / stem)["basis"]
        coefficients = infer_coefficients(flux[test], weights[test], basis)
        residual = flux[test] - coefficients @ basis
        observed = weights[test] > 0
        dof = np.maximum(np.sum(observed, axis=1) - rank, 1)
        reduced_chi2 = np.sum(weights[test] * residual**2, axis=1) / dof
        axes[0, 0].hist(
            reduced_chi2,
            bins=np.linspace(0.7, 2.0, 45),
            histtype="step",
            density=True,
            linewidth=1.8,
            color=colors[method],
            label=labels[method],
        )
        weight_sum = np.sum(weights[test], axis=0)
        weighted_bias = np.divide(
            np.sum(weights[test] * -residual, axis=0),
            weight_sum,
            out=np.full(len(wave), np.nan),
            where=weight_sum > 0,
        )
        axes[0, 1].plot(wave, weighted_bias, color=colors[method], label=labels[method])
        pull = np.sqrt(weights[test]) * residual
        count = np.sum(observed, axis=0)
        pull_mean = np.divide(
            np.sum(pull, axis=0),
            count,
            out=np.full(len(wave), np.nan),
            where=count > 0,
        )
        pull_variance = np.divide(
            np.sum(np.where(observed, (pull - pull_mean) ** 2, 0.0), axis=0),
            count,
            out=np.full(len(wave), np.nan),
            where=count > 0,
        )
        axes[1, 0].plot(wave, pull_mean, color=colors[method], label=labels[method])
        axes[1, 1].plot(
            wave, np.sqrt(pull_variance), color=colors[method], label=labels[method]
        )
    axes[0, 0].set(
        xlabel="Reduced $\\chi^2$ per test spectrum",
        ylabel="Density",
        title="Held-out fit distribution",
    )
    axes[0, 1].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[0, 1].set(
        xlabel="Rest wavelength [Å]",
        ylabel="Weighted model − data flux",
        title="Wavelength-dependent flux bias",
    )
    axes[1, 0].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[1, 0].set(
        xlabel="Rest wavelength [Å]",
        ylabel="Mean standardized residual",
        title="Residual centering",
    )
    axes[1, 1].axhline(1, color="black", lw=0.8, linestyle="--")
    axes[1, 1].set(
        xlabel="Rest wavelength [Å]",
        ylabel="Standardized residual width",
        title="Residual calibration",
    )
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Rank-{rank} polished solutions on the untouched test set")
    fig.savefig(
        output_dir / f"rank{rank}_test_residual_diagnostics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.training_matrix)
    targetid = data["targetid"].astype(np.int64)
    wave = data["wave_rest_aa"].astype(float)
    flux = data["flux"].astype(float)
    weights = data["relative_ivar"].astype(float)
    if args.max_spectra and args.max_spectra < len(flux):
        subset_rng = np.random.default_rng(args.split_seed)
        keep = np.sort(subset_rng.choice(len(flux), args.max_spectra, replace=False))
        targetid, flux, weights = targetid[keep], flux[keep], weights[keep]

    split_rng = np.random.default_rng(args.split_seed)
    permutation = split_rng.permutation(len(flux))
    train_stop = int(args.train_fraction * len(flux))
    validation_stop = train_stop + int(args.validation_fraction * len(flux))
    split = np.full(len(flux), "test", dtype="U10")
    split[permutation[:train_stop]] = "train"
    split[permutation[train_stop:validation_stop]] = "validation"
    train = np.flatnonzero(split == "train")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    learned, wavelength_contributors, required_contributors = select_wavelength_support(
        weights[train],
        args.ranks,
        observations_per_component=args.observations_per_component,
        minimum_contributors=args.minimum_wavelength_contributors,
        support_rank=args.wavelength_support_rank,
    )
    if np.count_nonzero(learned) < 20:
        raise ValueError(
            f"Too few wavelength bins have at least {required_contributors} contributors"
        )
    wave = wave[learned]
    flux = flux[:, learned]
    weights = weights[:, learned]
    pd.DataFrame({"targetid": targetid, "split": split}).to_csv(
        args.output_dir / "data_split.csv", index=False
    )

    provenance = vars(args).copy()
    provenance.update(
        {
            "training_matrix": str(args.training_matrix.expanduser().resolve()),
            "output_dir": str(args.output_dir.expanduser().resolve()),
            "n_spectra": len(flux),
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
            "n_wavelengths": len(wave),
            "required_wavelength_contributors": required_contributors,
            "minimum_selected_wavelength_contributors": int(
                np.min(wavelength_contributors[learned])
            ),
            "wave_min_aa": float(wave.min()),
            "wave_max_aa": float(wave.max()),
            "smoothing": False,
            "held_out_coefficient_solver": "scipy.optimize.nnls",
            "nearly_nmf_source": "https://github.com/dylanagreen/nearly_nmf",
        }
    )
    provenance.update(nearly_nmf_package_metadata())
    for key, value in list(provenance.items()):
        if isinstance(value, Path):
            provenance[key] = str(value)
    provenance_path = args.output_dir / "evaluation_provenance.json"
    require_compatible_resume(provenance_path, provenance)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    for rank in sorted(set(args.ranks)):
        for start in range(args.starts):
            initialization_seed = args.initialization_seed + 1000 * rank + start
            coefficient_start, basis_start = shared_initialization(
                len(train), len(wave), rank, initialization_seed
            )
            for method in args.methods:
                stem = f"{method}_rank{rank}_start{start}"
                result_path = args.output_dir / f"{stem}.json"
                if result_path.exists():
                    print(f"Skipping completed {stem}", flush=True)
                    continue
                print(f"Fitting {stem}", flush=True)
                if method == "anls":
                    _, basis, losses, elapsed, updates = fit_anls(
                        flux[train],
                        weights[train],
                        coefficient_start,
                        basis_start,
                        max_updates=args.anls_max_updates,
                        relative_tolerance=args.relative_tolerance,
                        patience=args.patience,
                    )
                else:
                    _, basis, losses, elapsed, updates = fit_nearly_nmf(
                        flux[train],
                        weights[train],
                        coefficient_start,
                        basis_start,
                        max_updates=args.nearly_max_updates,
                        check_every=args.nearly_check_every,
                        relative_tolerance=args.relative_tolerance,
                        patience=args.patience,
                    )
                result: dict[str, object] = {
                    "method": method,
                    "rank": rank,
                    "start": start,
                    "initialization_seed": initialization_seed,
                    "updates": updates,
                    "elapsed_seconds": elapsed[-1],
                    "initial_training_objective": losses[0],
                    "final_training_objective": losses[-1],
                    "objective_history": losses,
                    "elapsed_history": elapsed,
                }
                result.update(
                    {
                        f"validation_{key}": value
                        for key, value in evaluate_basis(
                            flux, weights, basis, validation
                        ).items()
                    }
                )
                result_path.write_text(json.dumps(result, indent=2) + "\n")
                np.savez_compressed(
                    args.output_dir / f"{stem}_basis.npz",
                    wave_rest_aa=wave,
                    basis=basis,
                )
                print(
                    f"{stem}: validation median chi2/dof="
                    f"{result['validation_median_reduced_chi2']:.4f}, "
                    f"p90={result['validation_p90_reduced_chi2']:.4f}, "
                    f"time={elapsed[-1]:.1f}s",
                    flush=True,
                )

    records = []
    for path in sorted(args.output_dir.glob("*_rank*_start*.json")):
        records.append(json.loads(path.read_text()))
    metrics = pd.DataFrame(records)
    metrics["selected_for_test"] = False
    for _, group in metrics.groupby(["method", "rank"]):
        selected_index = group["validation_median_reduced_chi2"].idxmin()
        metrics.loc[selected_index, "selected_for_test"] = True
        selected = metrics.loc[selected_index]
        stem = (
            f"{selected['method']}_rank{int(selected['rank'])}_"
            f"start{int(selected['start'])}"
        )
        basis = np.load(args.output_dir / f"{stem}_basis.npz")["basis"]
        if args.polish_selected:
            polished_stem = f"polished_{selected['method']}_rank{int(selected['rank'])}"
            polished_path = args.output_dir / f"{polished_stem}.json"
            polished_basis_path = args.output_dir / f"{polished_stem}_basis.npz"
            if polished_path.exists() and polished_basis_path.exists():
                polished = json.loads(polished_path.read_text())
                basis = np.load(polished_basis_path)["basis"]
            else:
                print(f"Polishing {stem}", flush=True)
                coefficient_start = infer_coefficients(
                    flux[train], weights[train], basis
                )
                if selected["method"] == "anls":
                    _, basis, losses, elapsed, updates = fit_anls(
                        flux[train],
                        weights[train],
                        coefficient_start,
                        basis,
                        max_updates=args.polish_anls_max_updates,
                        relative_tolerance=args.polish_relative_tolerance,
                        patience=args.patience,
                    )
                else:
                    # Multiplicative updates cannot revive exact zeros. A tiny
                    # floor preserves the reconstruction while allowing the
                    # selected solution to continue moving.
                    coefficient_start = np.maximum(coefficient_start, 1e-12)
                    basis = np.maximum(basis, 1e-12)
                    _, basis, losses, elapsed, updates = fit_nearly_nmf(
                        flux[train],
                        weights[train],
                        coefficient_start,
                        basis,
                        max_updates=args.polish_nearly_max_updates,
                        check_every=args.nearly_check_every,
                        relative_tolerance=args.polish_relative_tolerance,
                        patience=args.patience,
                    )
                polished = {
                    "method": selected["method"],
                    "rank": int(selected["rank"]),
                    "selected_start": int(selected["start"]),
                    "additional_updates": updates,
                    "additional_elapsed_seconds": elapsed[-1],
                    "initial_training_objective": losses[0],
                    "final_training_objective": losses[-1],
                    "objective_history": losses,
                    "elapsed_history": elapsed,
                }
                polished_path.write_text(json.dumps(polished, indent=2) + "\n")
                np.savez_compressed(
                    polished_basis_path, wave_rest_aa=wave, basis=basis
                )
                print(
                    f"{polished_stem}: additional updates={updates}, "
                    f"time={elapsed[-1]:.1f}s",
                    flush=True,
                )
            metrics.loc[selected_index, "polish_additional_updates"] = polished[
                "additional_updates"
            ]
            metrics.loc[selected_index, "polish_additional_seconds"] = polished[
                "additional_elapsed_seconds"
            ]
            metrics.loc[selected_index, "polished_training_objective"] = polished[
                "final_training_objective"
            ]
            polished_validation = evaluate_basis(
                flux, weights, basis, validation
            )
            for key, value in polished_validation.items():
                metrics.loc[selected_index, f"validation_{key}"] = value
        test_metrics = evaluate_basis(flux, weights, basis, test)
        for key, value in test_metrics.items():
            metrics.loc[selected_index, f"test_{key}"] = value
        result_path = args.output_dir / f"{stem}.json"
        selected_result = json.loads(result_path.read_text())
        selected_result["selected_for_test"] = True
        selected_result.update(
            {f"test_{key}": value for key, value in test_metrics.items()}
        )
        result_path.write_text(json.dumps(selected_result, indent=2) + "\n")
    history_columns = ["objective_history", "elapsed_history"]
    metrics.drop(columns=history_columns).to_csv(
        args.output_dir / "factorization_method_metrics.csv", index=False
    )
    make_summary_plot(metrics, args.output_dir / "factorization_method_comparison.png")
    make_residual_plot(metrics, flux, weights, wave, test, args.output_dir)
    print(f"Wrote comparison to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
