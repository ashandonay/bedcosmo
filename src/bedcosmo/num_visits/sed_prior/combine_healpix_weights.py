#!/usr/bin/env python
"""Concatenate per-HEALPix fit CSVs into one weights table for the KDE prior."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_HEALPIX = [
    23040,
    27257,
    27245,
    27259,
    27247,
    27256,
    27258,
    27344,
    26282,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healpix", type=int, nargs="+", default=DEFAULT_HEALPIX)
    parser.add_argument(
        "--scratch-base",
        default="~/scratch",
        help="Parent of desi_eazy_hp<HEALPIX>/ directories.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("~/scratch/bedcosmo/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv"),
    )
    parser.add_argument(
        "--quality-pass-only",
        action="store_true",
        help="Keep only rows with quality_pass True (recommended for KDE training export).",
    )
    args = parser.parse_args()

    scratch = Path(os.path.expanduser(args.scratch_base))
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for hp in args.healpix:
        path = scratch / f"desi_eazy_hp{hp}" / "desi_eazy_empirical_weights.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["healpix"] = hp
        if args.quality_pass_only and "quality_pass" in df.columns:
            df = df[df["quality_pass"].astype(bool)]
        frames.append(df)
        print(f"  HEALPIX {hp}: {len(df)} rows from {path}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out, index=False)
    n_qp = int(combined["quality_pass"].sum()) if "quality_pass" in combined.columns else len(combined)
    print(f"\nWrote {len(combined)} rows ({n_qp} quality_pass) -> {out}")


if __name__ == "__main__":
    main()
