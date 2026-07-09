"""Default filesystem locations for the empirical SED prior pipeline."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DR = "dr1"
DEFAULT_SPECPROD = "iron"
DEFAULT_SURVEY = "main"
DEFAULT_PROGRAM = "dark"
NUM_VISITS_EXPERIMENT = "num_visits"
DEFAULT_EMPIRICAL_PRIOR_DIR = "empirical_prior"
SED_PRIOR_KDE_NATIVE_FILENAME = "sed_prior_kde_native.joblib"
SED_PRIOR_KDE_GAUSSIANIZED_FILENAME = "sed_prior_kde_gaussianized.joblib"
ZWARN_UNSTABLE_BIT = 2048
HEALPIX_FITS_SUBDIR = "healpix"
HEALPIX_DIR_PREFIX = "hp"

DEFAULT_HEALPIX = (
    23040,
    27257,
    27245,
    27259,
    27247,
    27256,
    27258,
    27344,
    26282,
)


def get_scratch_root() -> Path:
    """Return ``$SCRATCH`` when set, otherwise ``~/scratch``."""
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch)
    return Path.home() / "scratch"


def get_bedcosmo_scratch() -> Path:
    """Bedcosmo artifact root: ``$SCRATCH/bedcosmo`` (or ``~/scratch/bedcosmo``)."""
    return get_scratch_root() / "bedcosmo"


def get_num_visits_scratch() -> Path:
    """NumVisits experiment scratch dir: ``$SCRATCH/bedcosmo/num_visits``."""
    return get_bedcosmo_scratch() / NUM_VISITS_EXPERIMENT


def get_desi_data_dir(*, dr: str = DEFAULT_DR) -> Path:
    """Local DESI DR subset tree used by the fit scripts."""
    return get_bedcosmo_scratch() / "desi" / f"tiny_{dr.lower()}"


DESI_DIR_ARG_HELP = (
    "DESI tree root (default: $SCRATCH/bedcosmo/desi/tiny_dr1, "
    "or ~/scratch/bedcosmo/desi/tiny_dr1 when SCRATCH is unset)."
)


def resolve_desi_dir(desi_dir: Path | str | None, *, dr: str = DEFAULT_DR) -> Path:
    """Return ``desi_dir`` or the standard tiny-DR path under bedcosmo scratch."""
    if desi_dir is None:
        return get_desi_data_dir(dr=dr)
    return Path(desi_dir).expanduser().resolve()


def add_desi_dir_argument(parser) -> None:
    """Register ``--desi-dir`` with the bedcosmo scratch default."""
    parser.add_argument(
        "--desi-dir",
        type=Path,
        default=None,
        help=DESI_DIR_ARG_HELP,
    )


def get_template_dir() -> Path:
    """Cached EAZY template bank (auto-downloaded from GitHub)."""
    return get_bedcosmo_scratch() / "eazy"


def get_prior_build_dir(name: str = DEFAULT_EMPIRICAL_PRIOR_DIR) -> Path:
    """Prior build root: combined CSV, KDE, and per-HEALPix fit subdirs."""
    return get_num_visits_scratch() / name


def healpix_dir_name(healpix: int) -> str:
    """Directory name for one patch, e.g. ``hp23040``."""
    return f"{HEALPIX_DIR_PREFIX}{int(healpix)}"


def get_healpix_fits_root(build_name: str = DEFAULT_EMPIRICAL_PRIOR_DIR) -> Path:
    """Parent directory for per-patch fit outputs: ``<prior_build>/healpix/``."""
    return get_prior_build_dir(build_name) / HEALPIX_FITS_SUBDIR


def get_healpix_fit_dir(
    healpix: int,
    build_name: str = DEFAULT_EMPIRICAL_PRIOR_DIR,
) -> Path:
    """Per-HEALPix NNLS fit outputs under ``<prior_build>/healpix/hp<HEALPIX>/``."""
    return get_healpix_fits_root(build_name) / healpix_dir_name(healpix)


def get_healpix_weights_csv(
    healpix: int,
    build_name: str = DEFAULT_EMPIRICAL_PRIOR_DIR,
) -> Path:
    return get_healpix_fit_dir(healpix, build_name) / "desi_eazy_empirical_weights.csv"


def find_healpix_weights_csv(
    healpix: int,
    *,
    build_name: str = DEFAULT_EMPIRICAL_PRIOR_DIR,
    prior_dir: Path | None = None,
) -> Path | None:
    """Return weights CSV path if found (current layout, then legacy names)."""
    root = Path(prior_dir) if prior_dir is not None else get_prior_build_dir(build_name)
    hp = int(healpix)
    candidates = [
        root / HEALPIX_FITS_SUBDIR / healpix_dir_name(hp) / "desi_eazy_empirical_weights.csv",
        root / f"desi_eazy_hp{hp}" / "desi_eazy_empirical_weights.csv",
        get_num_visits_scratch() / f"desi_eazy_hp{hp}" / "desi_eazy_empirical_weights.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def get_prior_weights_csv(name: str = DEFAULT_EMPIRICAL_PRIOR_DIR) -> Path:
    return get_prior_build_dir(name) / "desi_eazy_empirical_weights.csv"


def get_prior_kde_path(name: str = DEFAULT_EMPIRICAL_PRIOR_DIR) -> Path:
    return get_prior_build_dir(name) / SED_PRIOR_KDE_NATIVE_FILENAME


def get_prior_kde_gaussianized_path(name: str = DEFAULT_EMPIRICAL_PRIOR_DIR) -> Path:
    return get_prior_build_dir(name) / SED_PRIOR_KDE_GAUSSIANIZED_FILENAME
