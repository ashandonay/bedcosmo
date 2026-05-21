#!/usr/bin/env python3
"""
Download EAZY galaxy SED templates and draw random continuous template-mixture spectra.

This is a standalone exploratory script. It does NOT need bedcosmo/Pyro.
It creates:
  outputs/eazy_templates_rest.png       -- normalized input templates
  outputs/random_sed_mixtures_rest.png  -- random rest-frame mixtures
  outputs/random_sed_mixtures_obs.png   -- same mixtures redshifted for display
  outputs/sample_seds.npz               -- wavelength grid, templates, weights, samples

Example:
  python generate_template_sed_examples.py --n-samples 12 --outdir outputs/sed_examples
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import matplotlib.pyplot as plt



EAZY_RAW_BASE = "https://raw.githubusercontent.com/gbrammer/eazy-photoz/master/"
EAZY_TEMPLATES_DIR = Path(os.path.expanduser("~/data/num_visits/eazy"))

# 12 template basis
DEFAULT_PARAM_12D = "templates/fsps_full/fsps_QSF_12_v3.param"

# 6 template basis
DEFAULT_PARAM_6D = "templates/eazy_v1.0.spectra.param"

def download(url: str, path: Path, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    print(f"Downloading {url}")
    urlretrieve(url, path)


def read_template_param(param_path: Path) -> list[str]:
    """Return template file paths listed in an EAZY .param file."""
    template_paths: list[str] = []
    for line in param_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # EAZY param rows usually look like:
        # index templates/path.dat scale [other optional columns...]
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


def normalize_shape(wave: np.ndarray, flux: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    """Normalize an SED shape by its integrated positive flux in a wavelength window."""
    flux = np.clip(flux, 0.0, None)
    mask = (wave >= norm_min) & (wave <= norm_max)
    if mask.sum() < 3:
        mask = np.ones_like(wave, dtype=bool)
    norm = np.trapz(flux[mask], wave[mask])
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Template has non-positive normalization")
    return flux / norm


def make_common_grid(all_waves: list[np.ndarray], wave_min: float, wave_max: float, n_grid: int) -> np.ndarray:
    lower = max(w.min() for w in all_waves) if wave_min is None else wave_min
    upper = min(w.max() for w in all_waves) if wave_max is None else wave_max
    if lower >= upper:
        raise ValueError(f"Invalid common wavelength grid: {lower} >= {upper}")
    return np.linspace(lower, upper, n_grid)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def sample_template_mixtures(
    templates: np.ndarray,
    n_samples: int,
    logit_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw continuous SED mixtures.

    templates: shape (K, L)
    logits u ~ Normal(0, logit_scale)
    weights a = softmax(u)
    sample_sed = sum_k a_k T_k(lambda)
    """
    n_templates = templates.shape[0]
    logits = rng.normal(loc=0.0, scale=logit_scale, size=(n_samples, n_templates))
    weights = softmax(logits, axis=-1)
    samples = weights @ templates
    return weights, samples


def redshift_for_display(wave_rest: np.ndarray, sed_rest: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Display-only redshifting.

    For f_lambda shape, the observed spectrum scales as 1/(1+z) and shifts to
    lambda_obs = lambda_rest * (1+z). This omits luminosity-distance scaling
    because here we only want to visualize shapes.
    """
    return wave_rest * (1.0 + z), sed_rest / (1.0 + z)


def plot_templates(wave: np.ndarray, templates: np.ndarray, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, tpl in enumerate(templates):
        ax.plot(wave, tpl / np.nanmax(tpl), lw=1.4, alpha=0.85, label=f"T{i+1}")
    ax.set_xlabel("Rest-frame wavelength [Angstrom]")
    ax.set_ylabel("Normalized shape")
    ax.set_title("Downloaded EAZY template basis spectra")
    ax.set_xlim(wave.min(), wave.max())
    if templates.shape[0] <= 12:
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_samples_rest(wave: np.ndarray, samples: np.ndarray, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for sed in samples:
        ax.plot(wave, sed / np.nanmax(sed), lw=1.2, alpha=0.75)
    ax.set_xlabel("Rest-frame wavelength [Angstrom]")
    ax.set_ylabel("Normalized shape")
    ax.set_title("Random continuous mixtures of EAZY templates")
    ax.set_xlim(wave.min(), wave.max())
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_samples_observed(wave: np.ndarray, samples: np.ndarray, zs: np.ndarray, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for sed, z in zip(samples, zs):
        wobs, fobs = redshift_for_display(wave, sed, float(z))
        ax.plot(wobs, fobs / np.nanmax(fobs), lw=1.2, alpha=0.75, label=f"z={z:.2f}")
    ax.set_xlabel("Observed wavelength [Angstrom]")
    ax.set_ylabel("Normalized observed shape")
    ax.set_title("Same random SED mixtures shifted to random redshifts")
    ax.set_xlim(wave.min(), wave.max() * (1.0 + zs.max()))
    if len(zs) <= 12:
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs/sed_template_examples")
    parser.add_argument("--param", default=DEFAULT_PARAM_12D, help="EAZY template .param file inside eazy-photoz repo")
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--n-grid", type=int, default=2000)
    parser.add_argument("--wave-min", type=float, default=900.0)
    parser.add_argument("--wave-max", type=float, default=12000.0)
    parser.add_argument("--norm-min", type=float, default=4000.0)
    parser.add_argument("--norm-max", type=float, default=8000.0)
    parser.add_argument("--logit-scale", type=float, default=1.5, help="Larger values make mixtures closer to single templates")
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    templates_dir = EAZY_TEMPLATES_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download the EAZY parameter file that lists the template files.
    param_url = EAZY_RAW_BASE + args.param
    local_param = templates_dir / args.param
    download(param_url, local_param, overwrite=args.overwrite)

    # 2. Download each listed template file.
    rel_template_paths = read_template_param(local_param)
    local_template_paths = []
    for rel in rel_template_paths:
        local_path = templates_dir / rel
        download(EAZY_RAW_BASE + rel, local_path, overwrite=args.overwrite)
        local_template_paths.append(local_path)

    # 3. Load, interpolate onto a common rest-frame wavelength grid, and normalize shapes.
    raw_waves, raw_fluxes = [], []
    for path in local_template_paths:
        wave, flux = load_two_column_template(path)
        raw_waves.append(wave)
        raw_fluxes.append(flux)

    wave_grid = make_common_grid(raw_waves, args.wave_min, args.wave_max, args.n_grid)
    templates = []
    for wave, flux in zip(raw_waves, raw_fluxes):
        interp_flux = np.interp(wave_grid, wave, flux, left=0.0, right=0.0)
        templates.append(normalize_shape(wave_grid, interp_flux, args.norm_min, args.norm_max))
    templates = np.stack(templates, axis=0)  # (K, L)

    # 4. Draw continuous random mixtures using softmax-normal weights.
    rng = np.random.default_rng(args.seed)
    weights, sample_seds = sample_template_mixtures(
        templates=templates,
        n_samples=args.n_samples,
        logit_scale=args.logit_scale,
        rng=rng,
    )
    z_samples = rng.uniform(args.z_min, args.z_max, size=args.n_samples)

    # 5. Save figures and arrays.
    plot_templates(wave_grid, templates, outdir / "eazy_templates_rest.png")
    plot_samples_rest(wave_grid, sample_seds, outdir / "random_sed_mixtures_rest.png")
    plot_samples_observed(wave_grid, sample_seds, z_samples, outdir / "random_sed_mixtures_obs.png")

    np.savez(
        outdir / "sample_seds.npz",
        wavelength_rest_aa=wave_grid,
        templates=templates,
        weights=weights,
        sample_seds_rest=sample_seds,
        z_samples=z_samples,
        template_files=np.array(rel_template_paths),
    )

    print("Done.")
    print(f"Templates: {templates.shape[0]}")
    print(f"Wavelength grid: {wave_grid[0]:.1f} - {wave_grid[-1]:.1f} Angstrom, N={len(wave_grid)}")
    print(f"Outputs written to: {outdir.resolve()}")
    print("Example first sample weights:")
    for rel, w in zip(rel_template_paths, weights[0]):
        print(f"  {Path(rel).name}: {w:.3f}")


if __name__ == "__main__":
    main()
