#!/usr/bin/env python
"""Build the direct-DESI rest-frame training matrix that basis fits learn from."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..desi_data import ensure_desi_healpix
from ..paths import (
    DEFAULT_HEALPIX,
    ZWARN_UNSTABLE_BIT,
    get_desi_candidate_manifest_path,
    get_desi_data_dir,
    get_num_visits_scratch,
)
from .support import lsst_demand_weighted_coverage
from .training_matrix import (
    build_rest_frame_matrix,
    derive_rest_frame_grid,
    discover_desi_manifest,
    load_desi_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Explicit target/redshift manifest override; default discovers directly from DESI",
    )
    parser.add_argument("--desi-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--healpix", type=int, nargs="+", default=list(DEFAULT_HEALPIX))
    parser.add_argument("--target-spectype", default="GALAXY")
    parser.add_argument("--z-min", type=float, default=0.01)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--allow-nonzero-zwarn", action="store_true")
    parser.add_argument("--zwarn-forbid-mask", type=int, default=None, metavar="BITS")
    parser.add_argument(
        "--drop-unstable-zwarn",
        action="store_true",
        help=f"Shorthand for --zwarn-forbid-mask {ZWARN_UNSTABLE_BIT}.",
    )
    parser.add_argument("--min-good-pixels", type=int, default=100)
    parser.add_argument("--max-spectra", type=int, default=1500)
    parser.add_argument(
        "--wave-min",
        type=float,
        default=None,
        help="Override lower candidate bound; default covers valid DESI pixels and LSST at all selected redshifts",
    )
    parser.add_argument(
        "--wave-max",
        type=float,
        default=None,
        help="Override upper candidate bound; default covers valid DESI pixels and LSST at all selected redshifts",
    )
    parser.add_argument("--wave-step", type=float, default=10.0)
    parser.add_argument(
        "--edge-extrapolation",
        choices=("constant", "linear", "powerlaw"),
        default="constant",
        help="Edge treatment for missing rest-frame coverage",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the --max-spectra subset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_good_pixels <= 0:
        raise ValueError("--min-good-pixels must be positive")
    if args.z_max is not None and args.z_min is not None and args.z_min >= args.z_max:
        raise ValueError("--z-min must be below --z-max")
    desi_dir = Path(args.desi_dir or get_desi_data_dir()).expanduser().resolve()
    output_dir = (
        Path(args.output_dir or (get_num_visits_scratch() / "desi_training_data_extrapolated"))
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    zwarn_forbid_mask = args.zwarn_forbid_mask
    if args.drop_unstable_zwarn:
        if zwarn_forbid_mask is not None and zwarn_forbid_mask != ZWARN_UNSTABLE_BIT:
            raise ValueError(
                "Use only one of --drop-unstable-zwarn and --zwarn-forbid-mask, "
                f"or pass --zwarn-forbid-mask {ZWARN_UNSTABLE_BIT}."
            )
        zwarn_forbid_mask = ZWARN_UNSTABLE_BIT

    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
        manifest = load_desi_manifest(manifest_path)
        sample_source = "explicit_manifest"
    else:
        for healpix in args.healpix:
            ensure_desi_healpix(int(healpix), desi_dir=desi_dir)
        manifest = discover_desi_manifest(
            args.healpix,
            desi_dir=desi_dir,
            target_spectype=args.target_spectype,
            z_min=args.z_min,
            z_max=args.z_max,
            allow_nonzero_zwarn=args.allow_nonzero_zwarn,
            zwarn_forbid_mask=zwarn_forbid_mask,
        )
        manifest_path = None
        sample_source = "direct_desi_redrock"
    if manifest.empty:
        raise ValueError("No DESI spectra passed the sample selection")
    n_candidates = len(manifest)
    candidate_manifest_path = (
        get_desi_candidate_manifest_path()
        if args.output_dir is None
        else output_dir / "desi_candidate_manifest.csv"
    )
    candidate_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(candidate_manifest_path, index=False)
    if args.max_spectra and len(manifest) > args.max_spectra:
        manifest = (
            manifest.sample(args.max_spectra, random_state=args.seed)
            .sort_values(["healpix", "targetid"])
            .reset_index(drop=True)
        )
    grid_redshift_min = float(manifest["z"].min())
    grid_redshift_max = float(manifest["z"].max())
    wave = derive_rest_frame_grid(
        manifest,
        desi_dir=desi_dir,
        wave_step=args.wave_step,
        wave_min=args.wave_min,
        wave_max=args.wave_max,
    )
    manifest, flux, weights, scales = build_rest_frame_matrix(
        manifest,
        desi_dir=desi_dir,
        rest_wave=wave,
        min_good_pixels=args.min_good_pixels,
        edge_extrapolation=args.edge_extrapolation,
    )
    sample_manifest_path = output_dir / "desi_sample_manifest.csv"
    manifest.to_csv(sample_manifest_path, index=False)
    matrix_path = output_dir / "desi_rest_frame_training_matrix.npz"
    np.savez_compressed(
        matrix_path,
        targetid=manifest["targetid"].to_numpy(np.int64),
        healpix=manifest["healpix"].to_numpy(np.int64),
        redshift=manifest["z"].to_numpy(float),
        wave_rest_aa=wave,
        flux=flux,
        relative_ivar=weights,
        normalization_scale=scales,
    )
    pd.DataFrame(
        {
            "wave_rest_aa": wave,
            "all_contributing_spectra": np.sum(weights > 0, axis=0),
            "observed_fraction": np.mean(weights > 0, axis=0),
            "lsst_demand_weighted_coverage": lsst_demand_weighted_coverage(
                wave, manifest["z"].to_numpy(float), weights
            ),
        }
    ).to_csv(output_dir / "rest_wavelength_coverage.csv", index=False)

    parameters = vars(args).copy()
    parameters.update(
        {
            "sample_source": sample_source,
            "input_manifest": str(manifest_path) if manifest_path is not None else None,
            "candidate_manifest": str(candidate_manifest_path),
            "sample_manifest": str(sample_manifest_path),
            "training_matrix": str(matrix_path),
            "desi_dir": str(desi_dir),
            "output_dir": str(output_dir),
            "n_candidate_spectra": n_candidates,
            "n_loaded_spectra": len(manifest),
            "wave_min_aa": float(wave.min()),
            "wave_max_aa": float(wave.max()),
            "n_wavelength_bins": len(wave),
            "wavelength_grid_rule": "union of valid DESI pixels and full tabulated LSST bandpasses over selected redshifts, rounded outward",
            "grid_redshift_min": grid_redshift_min,
            "grid_redshift_max": grid_redshift_max,
            "uses_eazy_selection": (False if sample_source == "direct_desi_redrock" else None),
            "selection_role": (
                "Direct Redrock/FIBERMAP galaxy selection"
                if sample_source == "direct_desi_redrock"
                else "Explicit user-supplied manifest override"
            ),
            "desi_flux_unit_scale_cgs": 1e-17,
            "edge_quality_cut": {
                "relative_ivar_threshold": 0.05,
                "consecutive_bins": 3,
                "reference_observed_aa": [50, 300],
                "minimum_reference_bins": 3,
                "insufficient_reference": "retain endpoint",
            },
        }
    )
    for key, value in list(parameters.items()):
        if isinstance(value, Path):
            parameters[key] = str(value)
    (output_dir / "desi_training_matrix_provenance.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(manifest):,} x {len(wave):,} DESI training matrix to {matrix_path}")


if __name__ == "__main__":
    main()
