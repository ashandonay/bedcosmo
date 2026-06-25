"""Load EAZY template SEDs onto a common rest-frame wavelength grid."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from .paths import get_eazy_templates_dir

EAZY_RAW_BASE = "https://raw.githubusercontent.com/gbrammer/eazy-photoz/master/"
DEFAULT_TEMPLATES_DIR = get_eazy_templates_dir()
DEFAULT_PARAM_12D = "templates/fsps_full/fsps_QSF_12_v3.param"
# Rest-frame tabulation range for NumVisits / template bank (Angstrom).
# Native EAZY files span ~91–1e8 Å; LSST needs dense sampling in the optical/NIR only.
DEFAULT_BANK_WAVE_MIN_AA = 500.0
DEFAULT_BANK_WAVE_MAX_AA = 50000.0


def download(url: str, path: Path, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    urlretrieve(url, path)


def read_template_param(param_path: Path) -> list[str]:
    template_paths: list[str] = []
    for line in param_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            template_paths.append(parts[1])
        elif parts[0].endswith((".dat", ".sed")):
            template_paths.append(parts[0])
    if not template_paths:
        raise ValueError(f"No templates found in {param_path}")
    return template_paths


def load_two_column_template(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}, got shape {arr.shape}")
    wave = arr[:, 0].astype(float)
    flux = arr[:, 1].astype(float)
    good = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave = wave[good]
    flux = flux[good]
    order = np.argsort(wave)
    return wave[order], flux[order]


def normalize_shape(
    wave: np.ndarray,
    flux: np.ndarray,
    norm_min: float,
    norm_max: float,
) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux = np.clip(flux, 0.0, None)
    mask = (wave >= norm_min) & (wave <= norm_max)
    if mask.sum() < 3:
        mask = np.ones_like(wave, dtype=bool)
    norm = np.trapz(flux[mask], wave[mask])
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Template has non-positive normalization")
    return flux / norm


def load_eazy_templates(
    param: str = DEFAULT_PARAM_12D,
    *,
    overwrite: bool = False,
    norm_min: float = 4000.0,
    norm_max: float = 8000.0,
    templates_dir: Path | str = DEFAULT_TEMPLATES_DIR,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Download/load per-template wavelength and normalized flux arrays."""
    templates_dir = Path(os.path.expanduser(templates_dir))
    local_param = templates_dir / param
    download(EAZY_RAW_BASE + param, local_param, overwrite=overwrite)
    rel_paths = read_template_param(local_param)
    waves: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    for rel in rel_paths:
        local_path = templates_dir / rel
        download(EAZY_RAW_BASE + rel, local_path, overwrite=overwrite)
        wave, flux = load_two_column_template(local_path)
        flux = normalize_shape(wave, flux, norm_min=norm_min, norm_max=norm_max)
        waves.append(np.asarray(wave, dtype=float))
        fluxes.append(np.asarray(flux, dtype=float))
    return waves, fluxes, rel_paths


def build_common_rest_grid(
    template_waves: list[np.ndarray],
    *,
    n_points: int = 4000,
    wave_min: float | None = None,
    wave_max: float | None = None,
    log_spacing: bool = True,
) -> np.ndarray:
    """
    Common rest-frame wavelength grid for stacking template SEDs.

    Defaults clip the native EAZY span to ``[500, 50000]`` Å and use log spacing
    so the optical (1–12 µm rest) is densely sampled. A linear grid over the
    full native 91–1e8 Å range leaves no points in the optical.
    """
    native_min = min(float(w.min()) for w in template_waves)
    native_max = max(float(w.max()) for w in template_waves)
    wmin = DEFAULT_BANK_WAVE_MIN_AA if wave_min is None else float(wave_min)
    wmax = DEFAULT_BANK_WAVE_MAX_AA if wave_max is None else float(wave_max)
    wmin = max(wmin, native_min)
    wmax = min(wmax, native_max)
    if wmin >= wmax:
        raise ValueError(f"Invalid template bank grid: {wmin} >= {wmax}")
    if log_spacing:
        return np.logspace(np.log10(wmin), np.log10(wmax), n_points, dtype=np.float64)
    return np.linspace(wmin, wmax, n_points, dtype=np.float64)


def stack_templates_on_grid(
    template_waves: list[np.ndarray],
    template_fluxes: list[np.ndarray],
    wave_common: np.ndarray,
) -> np.ndarray:
    """Interpolate each template onto wave_common; shape (n_templates, n_points)."""
    n_templates = len(template_waves)
    stack = np.zeros((n_templates, wave_common.size), dtype=np.float64)
    for i, (tw, tf) in enumerate(zip(template_waves, template_fluxes)):
        stack[i] = np.interp(wave_common, tw, tf, left=0.0, right=0.0)
    return stack


def load_eazy_template_bank(
    param: str = DEFAULT_PARAM_12D,
    *,
    templates_dir: Path | str = DEFAULT_TEMPLATES_DIR,
    n_grid: int = 4000,
    wave_min: float | None = None,
    wave_max: float | None = None,
    log_spacing: bool = True,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns
    -------
    wave_rest_aa : (n_grid,) common rest-frame Angstrom grid
    template_flux : (n_templates, n_grid) normalized template flux
    rel_paths : template filenames
    """
    waves, fluxes, rel_paths = load_eazy_templates(
        param, templates_dir=templates_dir, overwrite=overwrite
    )
    wave_common = build_common_rest_grid(
        waves,
        n_points=n_grid,
        wave_min=wave_min,
        wave_max=wave_max,
        log_spacing=log_spacing,
    )
    stack = stack_templates_on_grid(waves, fluxes, wave_common)
    return wave_common, stack, rel_paths
