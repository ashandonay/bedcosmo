"""Ensure local DESI spectroscopy files exist for SED prior fits."""

from __future__ import annotations

from pathlib import Path

from .desi_get_dr_subset import download_healpix_patches
from .paths import (
    DEFAULT_DR,
    DEFAULT_PROGRAM,
    DEFAULT_SPECPROD,
    DEFAULT_SURVEY,
    get_desi_data_dir,
)


def healpix_subdir(healpix: int) -> str:
    """DESI HEALPix directories use prefix/healpix, e.g. 230/23040."""
    hp = str(int(healpix))
    prefix = hp if len(hp) < 3 else hp[:3]
    return f"{prefix}/{hp}"


def get_local_desi_paths(
    desi_dir: Path,
    specprod: str,
    survey: str,
    program: str,
    healpix: int,
) -> tuple[Path, Path]:
    hpdir = (
        desi_dir
        / "spectro"
        / "redux"
        / specprod
        / "healpix"
        / survey
        / program
        / healpix_subdir(healpix)
    )
    coadd = hpdir / f"coadd-{survey}-{program}-{healpix}.fits"
    redrock = hpdir / f"redrock-{survey}-{program}-{healpix}.fits"
    return coadd, redrock


def ensure_desi_healpix(
    healpix: int,
    *,
    desi_dir: Path | None = None,
    data_release: str = DEFAULT_DR,
    specprod: str = DEFAULT_SPECPROD,
    survey: str = DEFAULT_SURVEY,
    program: str = DEFAULT_PROGRAM,
    skip_existing: bool = True,
) -> tuple[Path, Path]:
    """
    Download DESI coadd + redrock for ``healpix`` when missing locally.

    Returns
    -------
    coadd, redrock
        Paths under ``desi_dir`` (default: ``$SCRATCH/bedcosmo/desi/tiny_dr1``).
    """
    desi_dir = Path(desi_dir) if desi_dir is not None else get_desi_data_dir(dr=data_release)
    coadd, redrock = get_local_desi_paths(
        desi_dir=desi_dir,
        specprod=specprod,
        survey=survey,
        program=program,
        healpix=healpix,
    )
    if coadd.exists() and redrock.exists():
        return coadd, redrock

    ok = download_healpix_patches(
        [int(healpix)],
        local_base_path=desi_dir,
        data_release=data_release,
        specprod=specprod,
        skip_existing=skip_existing,
        skip_catalog=True,
        no_tiles=True,
    )
    if not ok or not coadd.exists() or not redrock.exists():
        raise FileNotFoundError(
            f"DESI coadd/redrock for HEALPIX {healpix} not available under {desi_dir}. "
            "Download failed or files are missing."
        )
    return coadd, redrock
