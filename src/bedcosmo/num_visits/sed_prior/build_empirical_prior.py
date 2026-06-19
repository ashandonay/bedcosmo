#!/usr/bin/env python3
"""
End-to-end empirical SED prior build: DESI download → NNLS fits → combine → KDE.

Default output tree (``--build-name empirical_prior``)::

    $SCRATCH/bedcosmo/num_visits/empirical_prior/
      healpix/hp23040/desi_eazy_empirical_weights.csv
      healpix/hp27257/...
      desi_eazy_empirical_weights.csv   # combined
      sed_prior_kde.joblib
      sed_prior_kde.json

Shared inputs (downloaded once, reused across builds)::

    $SCRATCH/bedcosmo/desi/tiny_dr1/
    $SCRATCH/bedcosmo/eazy/

Example::

  python -m bedcosmo.num_visits.sed_prior.build_empirical_prior
  python -m bedcosmo.num_visits.sed_prior.build_empirical_prior --build-name empirical_prior_test --n-max 600
  python -m bedcosmo.num_visits.sed_prior.build_empirical_prior --healpix 23040 --skip-kde
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .combine_healpix_weights import combine_healpix_weights
from .desi_data import ensure_desi_healpix
from .paths import (
    DEFAULT_EMPIRICAL_PRIOR_DIR,
    DEFAULT_HEALPIX,
    ZWARN_UNSTABLE_BIT,
    add_desi_dir_argument,
    get_healpix_fit_dir,
    get_prior_build_dir,
    get_prior_kde_path,
    get_prior_weights_csv,
    resolve_desi_dir,
)

DEFAULT_MAX_CHI2_DOF = 1.2
DEFAULT_Z_MIN = 0.01

FIT_MODULE = "bedcosmo.num_visits.sed_prior.fit_eazy_weights_to_desi"
KDE_MODULE = "bedcosmo.num_visits.sed_prior.build_empirical_sed_prior_kde"


def resolve_kde_python(explicit: str | None = None) -> str:
    """
    KDE training needs torch. With fitsio installed, the full pipeline can run in bedcosmo.

    Resolution order: ``explicit`` → ``$BEDCOSMO_PYTHON`` → current interpreter
    if torch importable → ``$CONDA_PREFIX/../bedcosmo/bin/python``.
    """
    if explicit:
        return str(Path(explicit).expanduser())
    env_py = os.environ.get("BEDCOSMO_PYTHON")
    if env_py:
        return str(Path(env_py).expanduser())
    try:
        import torch  # noqa: F401

        return sys.executable
    except ImportError:
        pass
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        sibling = Path(conda_prefix).parent / "bedcosmo" / "bin" / "python"
        if sibling.is_file():
            return str(sibling)
    bedcosmo_py = shutil.which("bedcosmo-python")
    if bedcosmo_py:
        return bedcosmo_py
    raise RuntimeError(
        "KDE build requires torch. Activate the bedcosmo conda env, set "
        "BEDCOSMO_PYTHON=/path/to/python, or pass --kde-python."
    )


def _run(cmd: list[str], *, step: str) -> None:
    print(f"\n{'=' * 72}\n{step}\n{'=' * 72}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def build_empirical_prior(
    *,
    build_name: str = DEFAULT_EMPIRICAL_PRIOR_DIR,
    healpix: list[int] | tuple[int, ...] = DEFAULT_HEALPIX,
    desi_dir: Path | None = None,
    force_desi: bool = False,
    force_fit: bool = False,
    n_max: int | None = None,
    seed: int = 7,
    z_min: float = DEFAULT_Z_MIN,
    fit_method: str = "nnls",
    skip_desi: bool = False,
    skip_fit: bool = False,
    skip_combine: bool = False,
    skip_kde: bool = False,
    kde_sample: int = 20_000,
    max_chi2_dof: float = DEFAULT_MAX_CHI2_DOF,
    kde_python: str | None = None,
    allow_nonzero_zwarn: bool = False,
    zwarn_forbid_mask: int | None = None,
) -> dict[str, Path]:
    """
    Run the full empirical prior pipeline for ``build_name``.

    Returns paths to the prior build directory, combined CSV, and KDE artifact.
    """
    desi_dir = resolve_desi_dir(desi_dir)
    prior_dir = get_prior_build_dir(build_name)
    prior_dir.mkdir(parents=True, exist_ok=True)
    weights_csv = get_prior_weights_csv(build_name)
    kde_path = get_prior_kde_path(build_name)

    fit_python = sys.executable
    kde_py = resolve_kde_python(kde_python)

    if not skip_desi:
        print(f"\nStep 1/4: ensure DESI coadd + redrock under {desi_dir}")
        for hp in healpix:
            coadd, redrock = ensure_desi_healpix(
                hp,
                desi_dir=desi_dir,
                skip_existing=not force_desi,
            )
            print(f"  HEALPIX {hp}: OK ({coadd.name}, {redrock.name})")
    else:
        print("\nStep 1/4: skipped (--skip-desi)")

    if not skip_fit:
        print(f"\nStep 2/4: fit EAZY template weights → {prior_dir}/healpix/hp*/")
        for hp in healpix:
            outdir = get_healpix_fit_dir(hp, build_name=build_name)
            csv_path = outdir / "desi_eazy_empirical_weights.csv"
            if csv_path.exists() and not force_fit:
                print(f"  HEALPIX {hp}: skip existing {csv_path}")
                continue

            cmd = [
                fit_python,
                "-m",
                FIT_MODULE,
                "--healpix",
                str(hp),
                "--desi-dir",
                str(desi_dir),
                "--outdir",
                str(outdir),
                "--fit-method",
                fit_method,
                "--z-min",
                str(z_min),
                "--seed",
                str(seed),
                "--plot-n-examples",
                "0",
                "--no-triangle-plots",
                "--no-raw-coeff-triangle",
                "--max-chi2-dof",
                str(max_chi2_dof),
                "--no-auto-download-desi",
            ]
            if n_max is not None:
                cmd.extend(["--n-max", str(n_max)])
            if zwarn_forbid_mask is not None:
                cmd.extend(["--zwarn-forbid-mask", str(zwarn_forbid_mask)])
            elif allow_nonzero_zwarn:
                cmd.append("--allow-nonzero-zwarn")
            _run(cmd, step=f"Fitting HEALPIX {hp}")
    else:
        print("\nStep 2/4: skipped (--skip-fit)")

    if not skip_combine:
        print(f"\nStep 3/4: combine patch CSVs → {weights_csv}")
        combine_healpix_weights(
            healpix,
            build_name=build_name,
            prior_dir=prior_dir,
            out=weights_csv,
        )
    else:
        print("\nStep 3/4: skipped (--skip-combine)")

    if not skip_kde:
        print(f"\nStep 4/4: build KDE → {kde_path} (python: {kde_py})")
        cmd = [
            kde_py,
            "-m",
            KDE_MODULE,
            "--weights-csv",
            str(weights_csv),
            "--out",
            str(kde_path),
            "--sample",
            str(kde_sample),
            "--seed",
            str(seed),
            "--max-chi2-dof",
            str(max_chi2_dof),
            "--z-min",
            str(z_min),
        ]
        _run(cmd, step="Building empirical KDE")
    else:
        print("\nStep 4/4: skipped (--skip-kde)")

    print(f"\nDone. Prior build directory: {prior_dir}")
    if weights_csv.exists():
        print(f"  Combined weights: {weights_csv}")
    if kde_path.exists():
        print(f"  KDE artifact:     {kde_path}")

    return {
        "prior_dir": prior_dir,
        "weights_csv": weights_csv,
        "kde_path": kde_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build empirical SED prior: DESI → fits → combine → KDE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--build-name",
        default=DEFAULT_EMPIRICAL_PRIOR_DIR,
        help="Output directory name under $SCRATCH/bedcosmo/num_visits/.",
    )
    parser.add_argument("--healpix", type=int, nargs="+", default=list(DEFAULT_HEALPIX))
    add_desi_dir_argument(parser)
    parser.add_argument(
        "--force-desi",
        action="store_true",
        help="Re-download DESI patches even if coadd files exist.",
    )
    parser.add_argument(
        "--force-fit",
        action="store_true",
        help="Re-fit HEALPix patches even if weights CSV exists.",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Max spectra per HEALPix patch (default: fit all passing candidates).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN)
    parser.add_argument("--fit-method", choices=("nnls", "wls"), default="nnls")
    parser.add_argument(
        "--max-chi2-dof",
        type=float,
        default=DEFAULT_MAX_CHI2_DOF,
        help="Quality cut used in fits and KDE training table.",
    )
    parser.add_argument(
        "--kde-sample",
        type=int,
        default=20_000,
        help="Diagnostic KDE draws after save (0 to skip triangle plots).",
    )
    parser.add_argument(
        "--kde-python",
        default=None,
        help="Python executable for KDE step (needs torch; default: bedcosmo env or $BEDCOSMO_PYTHON).",
    )
    parser.add_argument(
        "--allow-nonzero-zwarn",
        action="store_true",
        help="Include all redrock ZWARN values (default: require ZWARN == 0).",
    )
    parser.add_argument(
        "--zwarn-forbid-mask",
        type=int,
        default=None,
        metavar="BITS",
        help=(
            "Drop rows with forbidden ZWARN bits set, e.g. 2048=UNSTABLE. "
            "Overrides --allow-nonzero-zwarn."
        ),
    )
    parser.add_argument(
        "--drop-unstable-zwarn",
        action="store_true",
        help=f"Shorthand for --zwarn-forbid-mask {ZWARN_UNSTABLE_BIT} (drop UNSTABLE only).",
    )
    parser.add_argument("--skip-desi", action="store_true")
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-combine", action="store_true")
    parser.add_argument("--skip-kde", action="store_true")
    args = parser.parse_args()
    zwarn_forbid_mask = args.zwarn_forbid_mask
    if args.drop_unstable_zwarn:
        if zwarn_forbid_mask is not None and zwarn_forbid_mask != ZWARN_UNSTABLE_BIT:
            parser.error(
                "Use only one of --drop-unstable-zwarn and --zwarn-forbid-mask, "
                "or pass --zwarn-forbid-mask 2048 explicitly."
            )
        zwarn_forbid_mask = ZWARN_UNSTABLE_BIT

    build_empirical_prior(
        build_name=args.build_name,
        healpix=args.healpix,
        desi_dir=args.desi_dir,
        force_desi=args.force_desi,
        force_fit=args.force_fit,
        n_max=args.n_max,
        seed=args.seed,
        z_min=args.z_min,
        fit_method=args.fit_method,
        skip_desi=args.skip_desi,
        skip_fit=args.skip_fit,
        skip_combine=args.skip_combine,
        skip_kde=args.skip_kde,
        kde_sample=args.kde_sample,
        max_chi2_dof=args.max_chi2_dof,
        kde_python=args.kde_python,
        allow_nonzero_zwarn=args.allow_nonzero_zwarn,
        zwarn_forbid_mask=zwarn_forbid_mask,
    )


if __name__ == "__main__":
    main()
