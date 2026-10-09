#!/usr/bin/env python3
"""Build a NumVisits empirical prior from a direct DESI Nearly-NMF basis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..diagnostics_plots import plot_template_redshifts
from ..paths import (
    BUILD_PROVENANCE_FILENAME,
    SED_PRIOR_KDE_NATIVE_FILENAME,
    get_num_visits_scratch,
    get_prior_build_dir,
)
from ..provenance import write_provenance
from ..template_config import default_empirical_parameters
from ..templates import load_two_column_template, read_template_param
from .evaluate_factorization_methods import (
    evaluate_basis,
    fit_nearly_nmf,
    nearly_nmf_package_metadata,
    shared_initialization,
)
from .support import lsst_required_mask, lsst_support_limits, select_wavelength_support
from .weighted_nmf import infer_coefficients

KDE_MODULE = "bedcosmo.num_visits.empirical.fit_sed_prior_kde"


def training_matrix_sha256(path):
    """Fingerprint the exact matrix used to fit cached coefficients."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_prior_redshift_limits(wave, prior_z_min=None, prior_z_max=None):
    """Derive LSST-safe defaults and reject overrides outside template support."""
    _, _, supported_min, supported_max = lsst_support_limits(wave.min(), wave.max())
    lower = supported_min if prior_z_min is None else prior_z_min
    upper = supported_max if prior_z_max is None else prior_z_max
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("No nonempty, finite LSST-supported prior redshift interval")
    if lower < supported_min or upper > supported_max:
        raise ValueError(
            f"Prior redshift limits must lie within LSST-supported range "
            f"[{supported_min:g}, {supported_max:g}]"
        )
    return lower, upper


def normalize_basis_for_export(
    wave: np.ndarray,
    basis: np.ndarray,
    coefficients: np.ndarray,
    *,
    norm_min: float,
    norm_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-integral normalize components while preserving ``C @ B`` exactly."""
    wave = np.asarray(wave, dtype=float)
    basis = np.asarray(basis, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    selected = (wave >= norm_min) & (wave <= norm_max)
    if np.count_nonzero(selected) < 3:
        raise ValueError(
            f"Normalization interval [{norm_min:g}, {norm_max:g}] Angstrom "
            "is not contained in the learned wavelength support"
        )
    integrals = np.trapz(basis[:, selected], wave[selected], axis=1)
    if np.any(~np.isfinite(integrals)) or np.any(integrals <= 0):
        raise ValueError("Every learned component must have a positive finite integral")
    return basis / integrals[:, None], coefficients * integrals[None, :], integrals


def physical_prior_coefficients(
    normalized_coefficients: np.ndarray,
    normalization_scale: np.ndarray,
    redshift: np.ndarray,
) -> np.ndarray:
    """Convert normalized-matrix coefficients to NumVisits template coefficients.

    DESI coadd FLUX is tabulated in units of 1e-17 cgs. The training matrix is
    divided by one scalar per object and only its wavelength coordinate is moved
    to the rest frame. NumVisits evaluates rest-frame templates with an explicit
    ``1 / (1 + z)``. Consequently the stored coefficient scale must restore both
    the per-object normalization and ``(1 + z)``; NumVisits applies 1e-17 later.
    """
    coefficients = np.asarray(normalized_coefficients, dtype=float)
    scale = np.asarray(normalization_scale, dtype=float)
    z = np.asarray(redshift, dtype=float)
    if coefficients.shape[0] != len(scale) or len(scale) != len(z):
        raise ValueError("Coefficient rows, normalization scales, and redshifts must match")
    return coefficients * (scale * (1.0 + z))[:, None]


def make_prior_table(
    manifest: pd.DataFrame,
    coefficients: np.ndarray,
    flux: np.ndarray,
    weights: np.ndarray,
    basis: np.ndarray,
    normalization_scale: np.ndarray,
    *,
    prior_z_min: float,
    prior_z_max: float,
    max_chi2_dof: float,
) -> pd.DataFrame:
    """Create the standard empirical-prior fit table for supported redshifts."""
    z = manifest["z"].to_numpy(float)
    physical = physical_prior_coefficients(coefficients, normalization_scale, z)
    coefficient_scale = np.sum(physical, axis=1)
    fractions = np.divide(
        physical,
        coefficient_scale[:, None],
        out=np.zeros_like(physical),
        where=coefficient_scale[:, None] > 0,
    )
    residual = flux - coefficients @ basis
    observed_count = np.sum(weights > 0, axis=1)
    dof = np.maximum(observed_count - basis.shape[0], 1)
    chi2 = np.sum(weights * residual**2, axis=1)
    finite = (
        np.isfinite(z)
        & np.isfinite(normalization_scale)
        & np.isfinite(coefficient_scale)
        & (coefficient_scale > 0)
        & np.all(np.isfinite(fractions), axis=1)
    )
    supported = finite & (z >= prior_z_min) & (z <= prior_z_max)

    output = manifest[["targetid", "healpix", "z"]].copy()
    output["success"] = finite
    output["quality_pass"] = supported & (chi2 / dof <= max_chi2_dof)
    output["dof"] = dof
    output["chi2"] = chi2
    output["chi2_dof"] = chi2 / dof
    output["desi_normalization_scale"] = normalization_scale
    output["log_c_scale"] = np.log(coefficient_scale)
    for index in range(basis.shape[0]):
        output[f"c{index + 1}"] = physical[:, index]
        output[f"a{index + 1}"] = fractions[:, index]
    return output.loc[supported].reset_index(drop=True)


def write_template_bank(
    template_dir: Path,
    relative_param: Path,
    wave: np.ndarray,
    basis: np.ndarray,
) -> tuple[Path, list[str]]:
    """Write learned components and an EAZY-compatible parameter file."""
    component_dir = relative_param.parent
    relative_paths: list[str] = []
    for index, component in enumerate(basis, start=1):
        relative_path = component_dir / f"component_{index:02d}.dat"
        absolute_path = template_dir / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            absolute_path,
            np.column_stack([wave, component]),
            fmt=("%.8f", "%.12e"),
            header="rest_wavelength_angstrom normalized_f_lambda",
        )
        relative_paths.append(str(relative_path))
    param_path = template_dir / relative_param
    param_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Direct DESI empirical component bank.",
        *(f"{index} {path} 1.0" for index, path in enumerate(relative_paths, start=1)),
    ]
    param_path.write_text("\n".join(lines) + "\n")
    return param_path, relative_paths


def write_prior_args(
    path: Path,
    *,
    prior_dir: Path,
    template_param: Path,
    rank: int,
    norm_min: float,
    norm_max: float,
    log_scale: np.ndarray,
    redshift: np.ndarray,
) -> Path:
    """Write a directly usable explicit NumVisits empirical-prior config."""
    parameters = default_empirical_parameters(rank)
    parameters["log_c_scale"]["plot"] = {
        "lower": float(np.floor(np.nanmin(log_scale) * 2.0) / 2.0),
        "upper": float(np.ceil(np.nanmax(log_scale) * 2.0) / 2.0),
    }
    parameters["z"]["plot"] = {
        "lower": float(np.floor(np.nanmin(redshift) * 20.0) / 20.0),
        "upper": float(np.ceil(np.nanmax(redshift) * 20.0) / 20.0),
    }
    config = {
        "density_type": "kde",
        "prior_dir": str(prior_dir),
        "template_param": str(template_param),
        "template_norm_min": float(norm_min),
        "template_norm_max": float(norm_max),
        "flux_unit_scale": 1.0e-17,
        "prior_pool_size": 65536,
        "prior_pool_seed": 7,
        "parameters": parameters,
        "constraints": {},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--training-matrix",
        type=Path,
        default=None,
        help=(
            "Rest-frame matrix (default: "
            "<num_visits>/desi_training_data_extrapolated/desi_rest_frame_training_matrix.npz)."
        ),
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--build-name", default="desi8",
                        help="Build name below num_visits/empirical_prior (not a path)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--template-param", type=Path, default=None)
    parser.add_argument(
        "--prior-z-min",
        type=float,
        default=None,
        help="Override lower prior redshift bound; default uses the saved requested range",
    )
    parser.add_argument(
        "--prior-z-max",
        type=float,
        default=None,
        help="Override upper prior redshift bound; default uses the saved requested range",
    )
    parser.add_argument("--max-chi2-dof", type=float, default=1.5)
    parser.add_argument("--norm-min", type=float, default=3600.0)
    parser.add_argument("--norm-max", type=float, default=4200.0)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--initialization-seed", type=int, default=7301)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--observations-per-component", type=int, default=10)
    parser.add_argument("--wavelength-support-rank", type=int, default=10)
    parser.add_argument("--minimum-wavelength-contributors", type=int, default=None)
    parser.add_argument("--start-updates", type=int, default=500)
    parser.add_argument("--polish-updates", type=int, default=3000)
    parser.add_argument("--full-refit-updates", type=int, default=1000)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--kde-sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-kde", action="store_true")
    parser.add_argument(
        "--prior-only",
        action="store_true",
        help="Rebuild prior tables and KDEs from this build's saved basis and coefficients",
    )
    return parser.parse_args()


def require_compatible_checkpoints(
    output_dir: Path, args: argparse.Namespace, wave: np.ndarray, required: int
) -> None:
    """Refuse to resume factors created for a different build request."""
    path = output_dir / "factorization_request.json"
    request = {
        key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"kde_sample", "seed", "skip_kde", "max_chi2_dof", "prior_only"}
    }
    request.update(
        {
            "selected_wave_min_aa": float(wave.min()),
            "selected_wave_max_aa": float(wave.max()),
            "selected_n_wavelengths": len(wave),
            "required_wavelength_contributors": required,
        }
    )
    if path.exists():
        previous = json.loads(path.read_text())
        mismatches = [key for key, value in request.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(
                "Existing factorization checkpoints are incompatible with this request "
                f"({', '.join(mismatches)}). Use a fresh --output-dir."
            )
    else:
        path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")


def finish_prior_build(args, output_dir, weights_table):
    """Write runtime config and fit KDEs for either a fresh or reused basis."""
    weights_path = output_dir / "desi_eazy_empirical_weights.csv"
    provenance_path = output_dir / BUILD_PROVENANCE_FILENAME
    template = json.loads(provenance_path.read_text())["template"]
    prior_args_path = write_prior_args(
        output_dir / "prior_args.yaml",
        prior_dir=output_dir,
        template_param=Path(template["template_param"]),
        rank=template["rank"],
        norm_min=template["normalization"]["wave_min_aa"],
        norm_max=template["normalization"]["wave_max_aa"],
        log_scale=weights_table.loc[weights_table["quality_pass"], "log_c_scale"].to_numpy(float),
        redshift=weights_table.loc[weights_table["quality_pass"], "z"].to_numpy(float),
    )
    print(f"Wrote NumVisits prior config to {prior_args_path}")
    figure = plot_template_redshifts(output_dir, flux_max=12)
    plot_path = output_dir / "template_redshifts.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(f"Wrote template redshift diagnostic to {plot_path}")
    if args.skip_kde:
        return

    subprocess.run(
        [
            sys.executable,
            "-m",
            KDE_MODULE,
            "--weights-csv",
            str(weights_path),
            "--out",
            str(output_dir / SED_PRIOR_KDE_NATIVE_FILENAME),
            "--build-provenance",
            str(provenance_path),
            "--no-z-filter",
            "--max-chi2-dof",
            str(args.max_chi2_dof),
            "--sample",
            str(args.kde_sample),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )


def rebuild_prior(args, output_dir):
    """Reuse saved factors, verifying matrix identity and exported templates."""
    provenance_path = output_dir / BUILD_PROVENANCE_FILENAME
    metadata = json.loads(provenance_path.read_text())
    matrix_path = args.training_matrix or Path(metadata["factorization"]["training_matrix"])
    expected_digest = metadata["factorization"].get("training_matrix_sha256")
    if expected_digest is None:
        raise ValueError("Saved basis has no training-matrix fingerprint; run a full basis build")
    if training_matrix_sha256(matrix_path) != expected_digest:
        raise ValueError("Training-matrix fingerprint does not match saved coefficients")
    with np.load(output_dir / "desi_basis.npz") as saved:
        wave = saved["wave_rest_aa"]
        basis = saved["basis"]
        coefficients = saved["coefficients"]
        targetids = saved["targetid"]
        support = saved["support_mask"]
    with np.load(matrix_path) as data:
        if not np.array_equal(targetids, data["targetid"]):
            raise ValueError("Training-matrix target IDs/order do not match saved coefficients")
        if not np.array_equal(wave, data["wave_rest_aa"][support]):
            raise ValueError("Training-matrix wavelengths do not match saved basis support")
        manifest = pd.DataFrame(
            {"targetid": data["targetid"], "healpix": data["healpix"], "z": data["redshift"]}
        )
        flux = data["flux"][:, support].astype(float)
        weights = data["relative_ivar"][:, support].astype(float)
        scales = data["normalization_scale"]
    template = metadata["template"]
    template_param = Path(template["template_param"])
    template_dir = output_dir / "templates"
    paths = read_template_param(template_dir / template_param)
    if len(paths) != basis.shape[0] or coefficients.shape != (len(manifest), basis.shape[0]):
        raise ValueError("Saved template/basis/coefficient dimensions do not match")
    for component, filename in zip(basis, paths):
        exported_wave, exported_flux = load_two_column_template(template_dir / filename)
        if not np.allclose(exported_wave, wave, rtol=0, atol=1e-7) or not np.allclose(
            exported_flux, component, rtol=1e-10, atol=1e-15
        ):
            raise ValueError("Exported template bank does not match saved basis")
    rank = basis.shape[0]
    prior_z_min, prior_z_max = resolve_prior_redshift_limits(
        wave,
        metadata["selection"]["prior_z_min"] if args.prior_z_min is None else args.prior_z_min,
        metadata["selection"]["prior_z_max"] if args.prior_z_max is None else args.prior_z_max,
    )
    table = make_prior_table(
        manifest,
        coefficients,
        flux,
        weights,
        basis,
        scales,
        prior_z_min=prior_z_min,
        prior_z_max=prior_z_max,
        max_chi2_dof=args.max_chi2_dof,
    )
    if not table.quality_pass.any():
        raise ValueError("No quality-passing spectra remain for prior fitting")
    metadata["selection"].update(
        prior_z_min=prior_z_min,
        prior_z_max=prior_z_max,
        max_chi2_dof=args.max_chi2_dof,
        n_supported_redshift=len(table),
        n_prior_quality_pass=int(table.quality_pass.sum()),
    )
    metadata["prior_rebuild"] = {
        "basis_retrained": False,
        "training_matrix": str(matrix_path),
        "kde_sample": args.kde_sample,
        "seed": args.seed,
    }
    table.to_csv(output_dir / "desi_eazy_empirical_weights.csv", index=False)
    write_provenance(provenance_path, metadata)
    print(
        f"Reusing saved DESI{rank} basis; prior redshift cut "
        f"{prior_z_min:.6f}–{prior_z_max:.6f}; "
        f"{int(table.quality_pass.sum()):,} quality-passing rows",
        flush=True,
    )
    finish_prior_build(args, output_dir, table)
    print(
        "Prior flows are not rebuilt by build_prior; retrain prior_flow --space both before BED use."
    )


def main() -> None:
    args = parse_args()
    if (not args.build_name or args.build_name in {".", ".."}
            or "/" in args.build_name or "\\" in args.build_name):
        raise ValueError("--build-name must be a single directory name; use --output-dir for paths")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else get_prior_build_dir(f"empirical_prior/{args.build_name}")
    )
    if args.prior_only:
        rebuild_prior(args, output_dir)
        return
    args.training_matrix = (
        args.training_matrix.expanduser().resolve()
        if args.training_matrix is not None
        else (
            get_num_visits_scratch()
            / "desi_training_data_extrapolated"
            / "desi_rest_frame_training_matrix.npz"
        )
    )
    if args.rank < 2:
        raise ValueError("The ILR prior requires a basis rank of at least two")
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must lie between zero and one")
    if not 0 < args.validation_fraction < 1 - args.train_fraction:
        raise ValueError("--validation-fraction must leave a nonempty test split")

    template_dir = output_dir / "templates"
    relative_param = args.template_param or Path(f"desi{args.rank}.param")
    if relative_param.is_absolute():
        raise ValueError("--template-param must be relative to <prior-dir>/templates")
    output_dir.mkdir(parents=True, exist_ok=True)

    args.training_matrix_sha256 = training_matrix_sha256(args.training_matrix)
    data = np.load(args.training_matrix)
    manifest = pd.DataFrame(
        {
            "targetid": data["targetid"].astype(np.int64),
            "healpix": data["healpix"].astype(np.int64),
            "z": data["redshift"].astype(float),
        }
    )
    wave_all = data["wave_rest_aa"].astype(float)
    flux_all = data["flux"].astype(float)
    weights_all = data["relative_ivar"].astype(float)
    normalization_scale = data["normalization_scale"].astype(float)
    requested_bounds = data["prior_redshift_bounds"]
    if args.prior_z_min is None:
        args.prior_z_min = float(requested_bounds[0])
    if args.prior_z_max is None:
        args.prior_z_max = float(requested_bounds[1])

    args.prior_z_min, args.prior_z_max = resolve_prior_redshift_limits(
        wave_all, args.prior_z_min, args.prior_z_max
    )
    rng = np.random.default_rng(args.split_seed)
    permutation = rng.permutation(len(manifest))
    train_stop = int(args.train_fraction * len(manifest))
    validation_stop = train_stop + int(args.validation_fraction * len(manifest))
    train = permutation[:train_stop]
    validation = permutation[train_stop:validation_stop]
    test = permutation[validation_stop:]
    support, contributors, required = select_wavelength_support(
        weights_all[train],
        [args.rank],
        observations_per_component=args.observations_per_component,
        minimum_contributors=args.minimum_wavelength_contributors,
        support_rank=args.wavelength_support_rank,
        required_mask=lsst_required_mask(wave_all, np.array([args.prior_z_min, args.prior_z_max])),
    )
    if np.count_nonzero(support) < 20:
        raise ValueError(f"Too few wavelength bins have at least {required} contributors")
    wave = wave_all[support]
    args.prior_z_min, args.prior_z_max = resolve_prior_redshift_limits(
        wave, args.prior_z_min, args.prior_z_max
    )
    print(f"LSST-supported prior redshift cut: {args.prior_z_min:.6f}–{args.prior_z_max:.6f}")
    flux = flux_all[:, support]
    weights = weights_all[:, support]
    require_compatible_checkpoints(output_dir, args, wave, required)
    print(
        f"Training rank-{args.rank} Nearly-NMF on {len(train):,} spectra; "
        f"support={wave.min():.0f}-{wave.max():.0f} Angstrom ({len(wave)} bins)",
        flush=True,
    )

    start_records: list[dict[str, object]] = []
    best_basis: np.ndarray | None = None
    best_metric = np.inf
    for start in range(args.starts):
        seed = args.initialization_seed + 1000 * args.rank + start
        basis_path = output_dir / f"start_{start}_basis.npz"
        record_path = output_dir / f"start_{start}.json"
        if basis_path.exists() and record_path.exists():
            basis = np.load(basis_path)["basis"]
            record = json.loads(record_path.read_text())
            print(f"  start {start}: using completed checkpoint", flush=True)
        else:
            coefficient_start, basis_start = shared_initialization(
                len(train), len(wave), args.rank, seed
            )
            _, basis, losses, elapsed, updates = fit_nearly_nmf(
                flux[train],
                weights[train],
                coefficient_start,
                basis_start,
                max_updates=args.start_updates,
                check_every=args.check_every,
                relative_tolerance=args.relative_tolerance,
                patience=args.patience,
            )
            validation_metrics = evaluate_basis(flux, weights, basis, validation)
            record = {
                "start": start,
                "seed": seed,
                "updates": updates,
                "elapsed_seconds": elapsed[-1],
                "training_objective": losses[-1],
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
            np.savez_compressed(basis_path, wave=wave, basis=basis)
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        metric = float(record["validation_median_reduced_chi2"])
        start_records.append(record)
        print(f"  start {start}: validation median chi2/dof={metric:.5f}", flush=True)
        if metric < best_metric:
            best_metric = metric
            best_basis = basis

    assert best_basis is not None
    selected_path = output_dir / "selected_polished_basis.npz"
    test_path = output_dir / "untouched_test_metrics.json"
    if selected_path.exists() and test_path.exists():
        selected_basis = np.load(selected_path)["basis"]
        test_document = json.loads(test_path.read_text())
        test_metrics = test_document["metrics"]
        polish_updates = int(test_document["polish_updates"])
        print("Using completed polished-basis checkpoint", flush=True)
    else:
        train_coefficients = np.maximum(
            infer_coefficients(flux[train], weights[train], best_basis), 1e-12
        )
        _, selected_basis, _, _, polish_updates = fit_nearly_nmf(
            flux[train],
            weights[train],
            train_coefficients,
            np.maximum(best_basis, 1e-12),
            max_updates=args.polish_updates,
            check_every=args.check_every,
            relative_tolerance=args.relative_tolerance,
            patience=args.patience,
        )
        test_metrics = evaluate_basis(flux, weights, selected_basis, test)
        np.savez_compressed(selected_path, wave=wave, basis=selected_basis)
        test_path.write_text(
            json.dumps({"polish_updates": polish_updates, "metrics": test_metrics}, indent=2) + "\n"
        )
    print(
        "Untouched test: median chi2/dof="
        f"{test_metrics['median_reduced_chi2']:.5f}, "
        f"median WRMS={test_metrics['median_wrms']:.5f}",
        flush=True,
    )

    # The test set has now served its sole evaluation purpose. Refit the final
    # production factors on every object without changing the selected support.
    full_refit_path = output_dir / "full_refit_basis.npz"
    if full_refit_path.exists():
        full_refit = np.load(full_refit_path)
        final_basis = full_refit["basis"]
        all_coefficients = full_refit["coefficients"]
        full_updates = int(full_refit["updates"])
        print("Using completed full-population refit checkpoint", flush=True)
    else:
        all_coefficient_start = np.maximum(infer_coefficients(flux, weights, selected_basis), 1e-12)
        all_coefficients, final_basis, _, _, full_updates = fit_nearly_nmf(
            flux,
            weights,
            all_coefficient_start,
            np.maximum(selected_basis, 1e-12),
            max_updates=args.full_refit_updates,
            check_every=args.check_every,
            relative_tolerance=args.relative_tolerance,
            patience=args.patience,
        )
        np.savez_compressed(
            full_refit_path,
            wave=wave,
            basis=final_basis,
            coefficients=all_coefficients,
            updates=full_updates,
        )
    final_basis, all_coefficients, component_integrals = normalize_basis_for_export(
        wave,
        final_basis,
        all_coefficients,
        norm_min=args.norm_min,
        norm_max=args.norm_max,
    )
    # Re-solve once against the exact exported bank; this is also insensitive to
    # the arbitrary scale convention used internally by Nearly-NMF.
    all_coefficients = infer_coefficients(flux, weights, final_basis)

    param_path, component_paths = write_template_bank(
        template_dir, relative_param, wave, final_basis
    )
    weights_table = make_prior_table(
        manifest,
        all_coefficients,
        flux,
        weights,
        final_basis,
        normalization_scale,
        prior_z_min=args.prior_z_min,
        prior_z_max=args.prior_z_max,
        max_chi2_dof=args.max_chi2_dof,
    )
    weights_path = output_dir / "desi_eazy_empirical_weights.csv"
    weights_table.to_csv(weights_path, index=False)
    np.savez_compressed(
        output_dir / "desi_basis.npz",
        wave_rest_aa=wave,
        basis=final_basis,
        coefficients=all_coefficients,
        targetid=manifest["targetid"].to_numpy(np.int64),
        support_mask=support,
        wavelength_contributors=contributors,
    )
    pd.DataFrame(start_records).to_csv(output_dir / "initialization_metrics.csv", index=False)

    provenance_path = write_provenance(
        output_dir / BUILD_PROVENANCE_FILENAME,
        {
            "kind": "direct_desi_empirical_sed_prior_build",
            "build_name": args.build_name,
            "template": {
                "template_param": str(relative_param),
                "component_paths": component_paths,
                "rank": args.rank,
                "normalization": {
                    "method": "integral",
                    "wave_min_aa": args.norm_min,
                    "wave_max_aa": args.norm_max,
                },
                "component_integrals_before_normalization": component_integrals.tolist(),
                "wave_min_aa": float(wave.min()),
                "wave_max_aa": float(wave.max()),
            },
            "selection": {
                "population": "all quality-selected DESI spectra in training matrix",
                "n_total": len(manifest),
                "n_supported_redshift": len(weights_table),
                "n_prior_quality_pass": int(weights_table["quality_pass"].sum()),
                "prior_z_min": args.prior_z_min,
                "prior_z_max": args.prior_z_max,
                "max_chi2_dof": args.max_chi2_dof,
                "reason": "complete LSST ugrizy support without endpoint extrapolation",
            },
            "factorization": {
                "method": "Nearly-NMF",
                "training_matrix": args.training_matrix,
                "training_matrix_sha256": args.training_matrix_sha256,
                "split_seed": args.split_seed,
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
                "starts": args.starts,
                "polish_updates": polish_updates,
                "full_refit_updates": full_updates,
                "required_wavelength_contributors": required,
                "wavelength_support_policy": "require the contributor threshold throughout requested-prior LSST demand; use the selected wavelength or redshift support extensions without narrowing the prior range",
                "minimum_retained_training_contributors": int(contributors[support].min()),
                "test_metrics_before_full_refit": test_metrics,
                **nearly_nmf_package_metadata(),
            },
            "coefficient_scale": {
                "formula": "c_prior = c_normalized * component_integral * desi_normalization_scale * (1 + z)",
                "runtime_flux_unit_scale_cgs": 1e-17,
                "explanation": (
                    "The training matrix shifts wavelength but retains observed f_lambda; "
                    "NumVisits later divides the redshifted template by (1+z)."
                ),
            },
            "outputs": {
                "weights_csv": weights_path,
                "template_param": param_path,
            },
            "arguments": vars(args),
        },
    )
    print(f"Wrote {len(weights_table):,} supported prior rows to {weights_path}")
    print(f"Wrote template bank to {param_path}")
    print(f"Wrote build provenance to {provenance_path}")
    finish_prior_build(args, output_dir, weights_table)


if __name__ == "__main__":
    main()
