#!/usr/bin/env python
"""Concatenate per-HEALPix fit CSVs into one weights table for the KDE prior."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .paths import (
    DEFAULT_EMPIRICAL_PRIOR_DIR,
    DEFAULT_HEALPIX,
    find_healpix_weights_csv,
    get_prior_build_dir,
    get_prior_weights_csv,
)


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
        print(f"  HEALPIX {hp}: {len(df)} rows from {csv_path}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out, index=False)
    n_qp = (
        int(combined["quality_pass"].sum())
        if "quality_pass" in combined.columns
        else len(combined)
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
