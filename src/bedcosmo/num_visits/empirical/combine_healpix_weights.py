#!/usr/bin/env python
"""Concatenate per-HEALPix fit CSVs into one weights table for the KDE prior."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import pandas as pd

from .paths import (
    BUILD_PROVENANCE_FILENAME,
    DEFAULT_EMPIRICAL_PRIOR_DIR,
    DEFAULT_HEALPIX,
    find_healpix_weights_csv,
    get_prior_build_dir,
)
from .provenance import fit_provenance_path, read_provenance, write_provenance


def _template_identity(settings: dict | None) -> tuple | None:
    """Fields that determine the numerical template coefficient system."""
    if not isinstance(settings, dict):
        return None
    normalization = settings.get("normalization") or {}
    return (
        settings.get("template_param"),
        normalization.get("method"),
        normalization.get("wave_min_aa"),
        normalization.get("wave_max_aa"),
    )


def _fit_identity(record: dict) -> tuple:
    """Fit/selection settings that must agree across combined patches."""
    parameters = record.get("parameters") or {}
    keys = (
        "fit_method",
        "coeff_norm",
        "wave_obs_min",
        "wave_obs_max",
        "min_good_pixels",
        "target_spectype",
        "z_min",
        "z_max",
        "allow_nonzero_zwarn",
        "zwarn_forbid_mask",
        "max_chi2_dof",
        "n_max",
        "seed",
    )
    return tuple(parameters.get(key) for key in keys)


def _build_fit_identity(build: dict) -> tuple:
    fit = build.get("fit") or {}
    selection = build.get("selection") or {}
    quality = build.get("quality") or {}
    return (
        fit.get("fit_method"),
        fit.get("coeff_norm"),
        fit.get("wave_obs_min_aa"),
        fit.get("wave_obs_max_aa"),
        fit.get("min_good_pixels"),
        selection.get("target_spectype"),
        selection.get("z_min"),
        selection.get("z_max"),
        selection.get("allow_nonzero_zwarn"),
        selection.get("zwarn_forbid_mask"),
        quality.get("max_chi2_dof"),
        selection.get("n_max_per_healpix"),
        selection.get("seed"),
    )


def _merge_fit_provenance(prior_dir: Path, records: list[dict]) -> None:
    """Validate per-patch fit settings and attach them to build provenance."""
    if not records:
        warnings.warn(
            "No per-patch fit provenance found; combined weights cannot be fully audited.",
            stacklevel=2,
        )
        return
    reference = records[0].get("template")
    reference_identity = _template_identity(reference)
    reference_fit = _fit_identity(records[0])
    for record in records[1:]:
        if _template_identity(record.get("template")) != reference_identity:
            raise ValueError(
                "Cannot combine HEALPix fits with different template bank or normalization settings"
            )
        if _fit_identity(record) != reference_fit:
            raise ValueError("Cannot combine HEALPix fits with different fitting/selection settings")

    build_path = prior_dir / BUILD_PROVENANCE_FILENAME
    build = read_provenance(build_path) or {
        "kind": "combined_empirical_sed_fits",
        "template": reference,
    }
    if (
        build.get("template") is not None
        and _template_identity(build.get("template")) != reference_identity
    ):
        raise ValueError("Per-patch template settings do not match the existing build provenance")
    if build.get("fit") is not None and _build_fit_identity(build) != reference_fit:
        raise ValueError(
            "Existing patch fits do not match the requested build fitting/selection settings; "
            "re-run them with --force-fit"
        )
    if build.get("template") is None:
        build["template"] = reference
    build["patch_fits"] = records
    write_provenance(build_path, build)


def combine_healpix_weights(
    healpix: list[int] | tuple[int, ...],
    *,
    build_name: str = DEFAULT_EMPIRICAL_PRIOR_DIR,
    prior_dir: Path | None = None,
    out: Path | None = None,
    quality_pass_only: bool = False,
) -> Path:
    """
    Concatenate per-patch ``desi_eazy_empirical_weights.csv`` files.

    Looks under ``<prior_dir>/healpix/hp<HEALPIX>/`` (legacy ``desi_eazy_hp*`` also accepted).
    """
    prior_dir = Path(prior_dir) if prior_dir is not None else get_prior_build_dir(build_name)
    out = Path(out) if out is not None else prior_dir / "desi_eazy_empirical_weights.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    fit_records = []
    missing_provenance = []
    for hp in healpix:
        csv_path = find_healpix_weights_csv(hp, build_name=build_name, prior_dir=prior_dir)
        if csv_path is None:
            raise FileNotFoundError(
                f"No weights CSV for HEALPIX {hp} under {prior_dir} "
                "(expected healpix/hp{hp}/desi_eazy_empirical_weights.csv)"
            )
        df = pd.read_csv(csv_path)
        df["healpix"] = hp
        if quality_pass_only and "quality_pass" in df.columns:
            df = df[df["quality_pass"].astype(bool)]
        frames.append(df)
        record = read_provenance(fit_provenance_path(csv_path.parent))
        if record is None:
            missing_provenance.append(int(hp))
        else:
            fit_records.append(record)
        print(f"  HEALPIX {hp}: {len(df)} rows from {csv_path}")

    if missing_provenance:
        if (prior_dir / BUILD_PROVENANCE_FILENAME).is_file():
            raise RuntimeError(
                "Cannot create an auditable build from patch CSVs missing fit provenance "
                f"(HEALPix {missing_provenance}). Re-run those fits with --force-fit."
            )
        warnings.warn(
            "Missing fit provenance for HEALPix " + ", ".join(str(hp) for hp in missing_provenance),
            stacklevel=2,
        )
    else:
        _merge_fit_provenance(prior_dir, fit_records)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out, index=False)
    n_qp = (
        int(combined["quality_pass"].sum()) if "quality_pass" in combined.columns else len(combined)
    )
    print(f"\nWrote {len(combined)} rows ({n_qp} quality_pass) -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healpix", type=int, nargs="+", default=DEFAULT_HEALPIX)
    parser.add_argument(
        "--build-name",
        default=DEFAULT_EMPIRICAL_PRIOR_DIR,
        help=f"Prior build directory name under num_visits (default: {DEFAULT_EMPIRICAL_PRIOR_DIR}).",
    )
    parser.add_argument(
        "--prior-dir",
        type=Path,
        default=None,
        help="Override prior build root (default: $SCRATCH/bedcosmo/num_visits/<build-name>).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Combined weights CSV (default: <prior-dir>/desi_eazy_empirical_weights.csv).",
    )
    parser.add_argument(
        "--quality-pass-only",
        action="store_true",
        help="Keep only rows with quality_pass True (recommended for KDE training export).",
    )
    args = parser.parse_args()

    prior_dir = (
        Path(os.path.expanduser(args.prior_dir))
        if args.prior_dir is not None
        else get_prior_build_dir(args.build_name)
    )
    out = (
        Path(os.path.expanduser(args.out))
        if args.out is not None
        else prior_dir / "desi_eazy_empirical_weights.csv"
    )

    combine_healpix_weights(
        args.healpix,
        build_name=args.build_name,
        prior_dir=prior_dir,
        out=out,
        quality_pass_only=args.quality_pass_only,
    )


if __name__ == "__main__":
    main()
