#!/usr/bin/env python3
"""
SED prior diagnostic plots (not part of the production build pipeline).

Subcommands:

  clr-triangle
      CLR-feature triangle with low-weight template highlighting, plus the same
      highlighting in Cholesky-whitened gaussianized coordinates when the KDE
      artifact includes a gaussianizer.

  redshift-histograms
      Low-z redshift distributions: DESI redrock GALAXY vs STAR, plus empirical
      weights CSV with an optional z-cutoff line.

  sed-examples
      Sample galaxy SEDs through NumVisits (rest/observed spectra, LSST mags,
      template weights).

  mag-leakage
      Compare LSST magnitudes from smooth KDE weights vs threshold-zeroed /
      support-masked post-processing on identical draws.

Examples:

  python -m bedcosmo.num_visits.empirical.diagnostic_plots all \\
    --prior-dir ~/scratch/bedcosmo/num_visits/empirical_prior

  python -m bedcosmo.num_visits.empirical.diagnostic_plots redshift-histograms \\
    --prior-dir ~/scratch/bedcosmo/num_visits/empirical_prior

  python -m bedcosmo.num_visits.empirical.diagnostic_plots clr-triangle \\
    --prior-dir ~/scratch/bedcosmo/num_visits/empirical_prior \\
    --also-training
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from bedcosmo.num_visits import NumVisits
from bedcosmo.util import get_experiment_config_path

from .fit_eazy_weights_to_desi import prior_a_column_names, read_redrock
from .fit_sed_prior_kde import (
    DEFAULT_KDE_DIAGNOSTIC_SAMPLES,
    apply_training_support_mask,
    gaussianize_with,
    get_empirical_gaussianizer,
    load_prior_training_table,
    load_sed_prior_kde,
    sample_sed_prior,
)
from .paths import (
    ZWARN_UNSTABLE_BIT,
    add_desi_dir_argument,
    get_prior_build_dir,
    resolve_desi_dir,
)
from .sed_prior import sample_prior_batch
from .simplex import clr_to_weights, prior_clr_feature_names

DEFAULT_INACTIVE_WEIGHT_THRESHOLD = 1e-4
DEFAULT_PRIOR_DIR = get_prior_build_dir()
PRIOR_KDE_FILENAME = "sed_prior_kde_native.joblib"
PRIOR_WEIGHTS_FILENAME = "desi_eazy_empirical_weights.csv"
DIAGNOSTICS_DIRNAME = "diagnostics"


@dataclass(frozen=True)
class PriorBuild:
    """Resolved paths for one empirical-prior build directory."""

    prior_dir: Path
    kde_path: Path
    weights_csv: Path


def add_prior_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prior-dir",
        type=Path,
        default=None,
        help=(
            "Prior build directory containing "
            f"{PRIOR_KDE_FILENAME} and {PRIOR_WEIGHTS_FILENAME} "
            f"(default: {DEFAULT_PRIOR_DIR})."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            f"Diagnostics output root (default: <prior-dir>/{DIAGNOSTICS_DIRNAME}). "
            "Each subcommand writes to a named subdirectory."
        ),
    )


def resolve_prior_build(
    *,
    prior_dir: Path | None = None,
    kde_path: Path | None = None,
    weights_csv: Path | None = None,
) -> PriorBuild:
    root = Path(prior_dir or DEFAULT_PRIOR_DIR).expanduser().resolve()
    kde = Path(kde_path).expanduser().resolve() if kde_path else root / PRIOR_KDE_FILENAME
    weights = (
        Path(weights_csv).expanduser().resolve()
        if weights_csv
        else root / PRIOR_WEIGHTS_FILENAME
    )
    if not kde.is_file():
        raise FileNotFoundError(kde)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return PriorBuild(prior_dir=root, kde_path=kde, weights_csv=weights)


def diagnostics_root(args: argparse.Namespace) -> Path:
    build = resolve_prior_build(
        prior_dir=args.prior_dir,
        kde_path=getattr(args, "kde_path", None),
        weights_csv=getattr(args, "weights_csv", None),
    )
    if args.outdir is not None:
        return Path(args.outdir).expanduser().resolve()
    return (build.prior_dir / DIAGNOSTICS_DIRNAME).resolve()


def diagnostics_subdir(args: argparse.Namespace, name: str) -> Path:
    out = diagnostics_root(args) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def prior_build_from_args(args: argparse.Namespace) -> PriorBuild:
    return resolve_prior_build(
        prior_dir=args.prior_dir,
        kde_path=getattr(args, "kde_path", None),
        weights_csv=getattr(args, "weights_csv", None),
    )


def z_cutoff_from_prior(build: PriorBuild) -> float | None:
    artifact = load_sed_prior_kde(build.kde_path)
    z_min = artifact.get("metadata", {}).get("z_min")
    return float(z_min) if z_min is not None else None


def load_experiment_configs(
    *,
    kde_path: Path,
    pool_size: int,
    pool_seed: int,
) -> tuple[dict, dict]:
    prior_path = get_experiment_config_path("num_visits", "prior_args_empirical.yaml")
    design_path = get_experiment_config_path("num_visits", "design_args.yaml")
    with open(prior_path) as f:
        prior_args = yaml.safe_load(f)
    with open(design_path) as f:
        design_args = yaml.safe_load(f)
    design_args["input_type"] = "nominal"
    prior_args["prior_dir"] = str(kde_path.parent.resolve())
    prior_args["prior_pool_size"] = pool_size
    prior_args["prior_pool_seed"] = pool_seed
    return prior_args, design_args


def plot_templates(wave: np.ndarray, templates: np.ndarray, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, tpl in enumerate(templates):
        peak = np.nanmax(tpl)
        y = tpl / peak if peak > 0 else tpl
        ax.plot(wave, y, lw=1.4, alpha=0.85, label=f"T{i + 1}")
    ax.set_xlabel("Rest-frame wavelength [Angstrom]")
    ax.set_ylabel("Normalized shape")
    ax.set_title("EAZY template basis (NumVisits)")
    ax.set_xlim(wave.min(), wave.max())
    if templates.shape[0] <= 12:
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_rest_seds(
    wave: np.ndarray,
    seds: np.ndarray,
    z: np.ndarray,
    log_s: np.ndarray,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, sed in enumerate(seds):
        ax.plot(
            wave,
            sed,
            lw=1.2,
            alpha=0.8,
            label=f"z={z[i]:.2f}, log s={log_s[i]:.2f}",
        )
    ax.set_xlabel("Rest-frame wavelength [Angstrom]")
    ax.set_ylabel(r"Rest-frame $f_\lambda$ [arb.]")
    ax.set_title("Empirical-prior SED draws (NumVisits rest grid)")
    ax.set_yscale("log")
    ax.set_xlim(wave.min(), wave.max())
    if len(seds) <= 12:
        ax.legend(ncol=2, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_observed_seds(
    wave_obs: np.ndarray,
    seds_obs: np.ndarray,
    z: np.ndarray,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, sed in enumerate(seds_obs):
        ax.plot(
            wave_obs,
            sed,
            lw=1.2,
            alpha=0.8,
            label=f"z={z[i]:.2f}",
        )
    ax.set_xlabel("Observed wavelength [Angstrom]")
    ax.set_ylabel(r"Observed $f_\lambda$ [arb.]")
    ax.set_title("Observed-frame SEDs (NumVisits LSST wavelength grid)")
    ax.set_yscale("log")
    ax.set_xlim(wave_obs.min(), wave_obs.max())
    if len(seds_obs) <= 12:
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_magnitudes(
    filters: list[str],
    mags: np.ndarray,
    z: np.ndarray,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(filters))
    for i, row in enumerate(mags):
        ax.plot(x, row, marker="o", lw=1.2, alpha=0.85, label=f"z={z[i]:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(filters)
    ax.set_xlabel("LSST filter")
    ax.set_ylabel("AB magnitude")
    ax.set_title("LSST magnitudes from NumVisits._calculate_magnitudes")
    ax.invert_yaxis()
    if len(mags) <= 12:
        ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_weight_heatmap(a: np.ndarray, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.35 * a.shape[0] + 1.5)))
    im = ax.imshow(a, aspect="auto", cmap="viridis")
    ax.set_xlabel("Template index")
    ax.set_ylabel("Sample")
    ax.set_title("Template mixture weights a_k")
    ax.set_xticks(range(a.shape[1]))
    ax.set_xticklabels([f"a{k + 1}" for k in range(a.shape[1])], rotation=45, ha="right")
    ax.set_yticks(range(a.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def get_raw_training_weights(artifact: dict) -> np.ndarray:
    """NNLS template weights from the fit CSV (hard zeros on inactive templates)."""
    meta = artifact.get("metadata", {})
    weights_csv = meta.get("weights_csv")
    if not weights_csv:
        raise KeyError("KDE artifact metadata missing weights_csv for support-mask comparison.")
    df = load_prior_training_table(
        Path(weights_csv).expanduser(),
        max_chi2_dof=meta.get("max_chi2_dof"),
        z_min=meta.get("z_min"),
    )
    return df[prior_a_column_names(df)].to_numpy(dtype=float)


def renormalize_weights(a: torch.Tensor) -> torch.Tensor:
    out = a.clone()
    denom = out.sum(dim=-1, keepdim=True).clamp_min(1e-300)
    return out / denom


def apply_inactive_threshold(a: torch.Tensor, threshold: float) -> torch.Tensor:
    out = a.clone()
    out[out <= float(threshold)] = 0.0
    return renormalize_weights(out)


def apply_support_mask_torch(
    a: torch.Tensor,
    training_a: np.ndarray,
    *,
    seed: int,
) -> torch.Tensor:
    a_np = a.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    masked = apply_training_support_mask(a_np, training_a, rng)
    s = masked.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(s)) | (s <= 0)
    masked = masked / np.where(bad, 1.0, s)
    if np.any(bad):
        masked[bad[:, 0]] = 1.0 / masked.shape[1]
    return torch.tensor(masked, device=a.device, dtype=a.dtype)


@torch.no_grad()
def magnitudes_from_weights(
    exp: NumVisits,
    z: torch.Tensor,
    a: torch.Tensor,
    log_s: torch.Tensor,
) -> torch.Tensor:
    flux = exp._observed_spectral_flux(z, a=a, log_s=log_s)
    return exp._calculate_magnitudes(flux)


def summarize_mag_deltas(
    filters: list[str],
    mags_ref: np.ndarray,
    mags_other: np.ndarray,
    *,
    label: str,
) -> pd.DataFrame:
    delta = mags_other - mags_ref
    rows = []
    for j, band in enumerate(filters):
        d = delta[:, j]
        ad = np.abs(d)
        rows.append(
            {
                "variant": label,
                "filter": band,
                "median_delta_mag": float(np.median(d)),
                "mean_delta_mag": float(np.mean(d)),
                "p95_abs_delta_mag": float(np.percentile(ad, 95)),
                "max_abs_delta_mag": float(np.max(ad)),
                "frac_abs_gt_0.001": float(np.mean(ad > 0.001)),
                "frac_abs_gt_0.01": float(np.mean(ad > 0.01)),
            }
        )
    max_abs = np.max(np.abs(delta), axis=1)
    rows.append(
        {
            "variant": label,
            "filter": "max_over_bands",
            "median_delta_mag": float(np.median(max_abs)),
            "mean_delta_mag": float(np.mean(max_abs)),
            "p95_abs_delta_mag": float(np.percentile(max_abs, 95)),
            "max_abs_delta_mag": float(np.max(max_abs)),
            "frac_abs_gt_0.001": float(np.mean(max_abs > 0.001)),
            "frac_abs_gt_0.01": float(np.mean(max_abs > 0.01)),
        }
    )
    return pd.DataFrame(rows)


def plot_mag_delta_histograms(
    filters: list[str],
    delta_threshold: np.ndarray,
    delta_masked: np.ndarray,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=False)
    axes = axes.ravel()
    for j, band in enumerate(filters):
        ax = axes[j]
        d_thr = delta_threshold[:, j]
        d_msk = delta_masked[:, j]
        bins = np.linspace(
            min(d_thr.min(), d_msk.min(), -0.05),
            max(d_thr.max(), d_msk.max(), 0.05),
            40,
        )
        ax.hist(d_thr, bins=bins, alpha=0.55, label="threshold − smooth", color="C0")
        ax.hist(d_msk, bins=bins, alpha=0.55, label="masked − smooth", color="C1")
        ax.axvline(0.0, color="k", lw=0.8, alpha=0.6)
        ax.set_title(band)
        ax.set_xlabel(r"$\Delta m$ [AB]")
        ax.set_ylabel("count")
    axes[0].legend(fontsize=8)
    fig.suptitle(r"$\Delta$ magnitude vs smooth KDE (same draws)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mag_max_delta_summary(
    max_abs_threshold: np.ndarray,
    max_abs_masked: np.ndarray,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        0.0,
        max(max_abs_threshold.max(), max_abs_masked.max(), 0.02) * 1.05,
        40,
    )
    ax.hist(max_abs_threshold, bins=bins, alpha=0.6, label="threshold − smooth", color="C0")
    ax.hist(max_abs_masked, bins=bins, alpha=0.6, label="masked − smooth", color="C1")
    ax.set_xlabel(r"max$_{\mathrm{bands}}$ $|\Delta m|$ [AB]")
    ax.set_ylabel("count")
    ax.set_title("Per-sample maximum magnitude shift across LSST bands")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def inactive_mask_from_weights(
    a: np.ndarray,
    n_templates: int,
    *,
    threshold: float,
) -> np.ndarray:
    """(n_samples, n_features) with True for template dims with a_k <= threshold."""
    a = np.asarray(a, dtype=float)
    inactive = a[:, :n_templates] <= float(threshold)
    mask = np.zeros((a.shape[0], n_templates + 2), dtype=bool)
    mask[:, :n_templates] = inactive
    return mask


def save_clr_triangle_zero_highlight(
    outdir: Path,
    joint: np.ndarray,
    param_labels: list[str],
    zero_mask: np.ndarray,
    *,
    filename: str,
    title: str,
    panel_size: float = 1.35,
) -> None:
    """Lower-triangle plot with red/gray split by low template weights."""
    n_params = joint.shape[1]
    if n_params < 2 or joint.shape[0] < 3:
        return
    if zero_mask.shape != joint.shape:
        raise ValueError(
            f"zero_mask shape {zero_mask.shape} must match joint {joint.shape}"
        )

    fig, axes = plt.subplots(
        n_params,
        n_params,
        figsize=(panel_size * n_params, panel_size * n_params),
        squeeze=False,
    )

    gray_color = "0.45"
    red_color = "#c0392b"

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                vals = joint[:, i]
                active = ~zero_mask[:, i]
                inactive = zero_mask[:, i]
                vals_active = vals[np.isfinite(vals) & active]
                vals_inactive = vals[np.isfinite(vals) & inactive]
                if vals_active.size == 0 and vals_inactive.size == 0:
                    ax.text(
                        0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
                    )
                else:
                    all_vals = np.concatenate(
                        [v for v in (vals_active, vals_inactive) if v.size]
                    )
                    bins = np.linspace(all_vals.min(), all_vals.max(), 31)
                    if bins[0] == bins[-1]:
                        bins = np.linspace(bins[0] - 0.5, bins[-1] + 0.5, 31)
                    if vals_active.size:
                        ax.hist(
                            vals_active,
                            bins=bins,
                            color=gray_color,
                            alpha=0.75,
                            histtype="stepfilled",
                            label="active",
                        )
                    if vals_inactive.size:
                        ax.hist(
                            vals_inactive,
                            bins=bins,
                            color=red_color,
                            alpha=0.65,
                            histtype="stepfilled",
                            label="inactive",
                        )
            else:
                xj = joint[:, j]
                yi = joint[:, i]
                both_active = (~zero_mask[:, j]) & (~zero_mask[:, i])
                either_inactive = zero_mask[:, j] | zero_mask[:, i]
                ok_gray = np.isfinite(xj) & np.isfinite(yi) & both_active
                ok_red = np.isfinite(xj) & np.isfinite(yi) & either_inactive
                if np.any(ok_gray):
                    ax.scatter(
                        xj[ok_gray],
                        yi[ok_gray],
                        s=8,
                        alpha=0.45,
                        color=gray_color,
                        edgecolors="none",
                        label="both active",
                    )
                if np.any(ok_red):
                    ax.scatter(
                        xj[ok_red],
                        yi[ok_red],
                        s=10,
                        alpha=0.55,
                        color=red_color,
                        edgecolors="none",
                        label="inactive in axis",
                    )

            ax.tick_params(axis="both", labelsize=7)
            if i == n_params - 1:
                ax.set_xlabel(param_labels[j], fontsize=8)
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(param_labels[i], fontsize=8)
            else:
                ax.tick_params(labelleft=False)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if not ax.get_visible():
                continue
            if i != n_params - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _clr_param_labels(
    feature_names: list[str],
    n_templates: int,
) -> list[str]:
    if feature_names == prior_clr_feature_names(n_templates):
        return [rf"$f_{{{k + 1}}}$" for k in range(n_templates)] + [
            r"$\log s$",
            r"$z$",
        ]
    return [rf"${name}$" for name in feature_names]


def _gaussianized_param_labels(feature_names: list[str]) -> list[str]:
    return [f"g_{name}" for name in feature_names]


def collect_redrock_redshifts(
    desi_dir: Path,
    *,
    zwarn_zero: bool = True,
    zwarn_forbid_mask: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (galaxy_z, star_z) arrays from all redrock files under desi_dir."""
    rr_files = sorted(glob.glob(str(desi_dir / "**/redrock-*.fits"), recursive=True))
    if not rr_files:
        raise FileNotFoundError(f"No redrock-*.fits files under {desi_dir}")

    all_gal_z: list[float] = []
    all_star_z: list[float] = []
    for rr_path in rr_files:
        rr = read_redrock(Path(rr_path))
        z = np.asarray(rr["Z"], dtype=float)
        st = np.char.strip(np.asarray(rr["SPECTYPE"]).astype(str))
        zw = np.asarray(rr["ZWARN"], dtype=int)
        gal = (st == "GALAXY") & np.isfinite(z)
        star = (st == "STAR") & np.isfinite(z)
        if zwarn_zero:
            gal &= zw == 0
            star &= zw == 0
        elif zwarn_forbid_mask is not None:
            zwarn_ok = (zw & zwarn_forbid_mask) == 0
            gal &= zwarn_ok
            star &= zwarn_ok
        all_gal_z.extend(z[gal].tolist())
        all_star_z.extend(z[star].tolist())

    return np.asarray(all_gal_z, dtype=float), np.asarray(all_star_z, dtype=float)


def redrock_zwarn_filter_label(
    *,
    zwarn_zero: bool,
    zwarn_forbid_mask: int | None = None,
) -> str:
    """Human-readable label for the redrock ZWARN selection."""
    if zwarn_zero:
        return "ZWARN=0"
    if zwarn_forbid_mask == ZWARN_UNSTABLE_BIT:
        return "no UNSTABLE (ZWARN & 2048 = 0)"
    if zwarn_forbid_mask is not None:
        return f"ZWARN forbid mask {zwarn_forbid_mask}"
    return "no ZWARN filter"


def add_redrock_zwarn_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI flags for redrock ZWARN selection in redshift-histograms."""
    parser.add_argument(
        "--allow-zwarn",
        action="store_true",
        help=(
            "Loosen redrock panel to match non-strict fits: drop UNSTABLE only "
            f"(same as --zwarn-forbid-mask {ZWARN_UNSTABLE_BIT}). Default: ZWARN=0."
        ),
    )
    parser.add_argument(
        "--zwarn-forbid-mask",
        type=int,
        default=None,
        metavar="BITS",
        help=(
            "Redrock panel keeps rows with (ZWARN & BITS) == 0. "
            "Overrides default ZWARN=0 when set."
        ),
    )
    parser.add_argument(
        "--drop-unstable-zwarn",
        action="store_true",
        help=f"Shorthand for --zwarn-forbid-mask {ZWARN_UNSTABLE_BIT}.",
    )


def resolve_redrock_zwarn_selection(
    args: argparse.Namespace,
) -> tuple[bool, int | None, str]:
    """Map CLI flags to (zwarn_zero, zwarn_forbid_mask, label)."""
    mask = getattr(args, "zwarn_forbid_mask", None)
    if getattr(args, "drop_unstable_zwarn", False):
        if mask is not None and mask != ZWARN_UNSTABLE_BIT:
            raise SystemExit(
                "Use only one of --drop-unstable-zwarn and --zwarn-forbid-mask, "
                f"or pass --zwarn-forbid-mask {ZWARN_UNSTABLE_BIT} explicitly."
            )
        mask = ZWARN_UNSTABLE_BIT
    elif getattr(args, "allow_zwarn", False) and mask is None:
        mask = ZWARN_UNSTABLE_BIT

    if mask is not None:
        label = redrock_zwarn_filter_label(zwarn_zero=False, zwarn_forbid_mask=mask)
        return False, mask, label
    label = redrock_zwarn_filter_label(zwarn_zero=True)
    return True, None, label


def weights_histogram_title(df: pd.DataFrame, weights_z: np.ndarray) -> str:
    """Title for the empirical-weights panel, including ZWARN breakdown when available."""
    n = len(weights_z)
    if "zwarn" not in df.columns:
        return f"Empirical weights CSV (n={n})"
    zw = df["zwarn"].to_numpy(dtype=int)
    n_zero = int((zw == 0).sum())
    if n_zero == n:
        return f"Empirical weights (n={n}, ZWARN=0)"
    return f"Empirical weights (n={n}, ZWARN=0: {n_zero}, non-zero: {n - n_zero})"


def save_redshift_histograms(
    outdir: Path,
    *,
    galaxy_z: np.ndarray,
    star_z: np.ndarray,
    weights_z: np.ndarray,
    z_cutoff: float | None = 0.05,
    low_z_max: float = 0.15,
    filename: str = "redshift_histograms.png",
    redrock_zwarn_label: str = "ZWARN=0",
    weights_title: str = "Empirical weights CSV (all rows)",
) -> None:
    """Two-panel low-z diagnostic: redrock spectypes vs empirical weights CSV."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bins = np.linspace(-0.01, low_z_max, 60)
    gal_low = galaxy_z[(galaxy_z > -0.01) & (galaxy_z < low_z_max)]
    star_low = star_z[(star_z > -0.01) & (star_z < low_z_max)]
    axes[0].hist(gal_low, bins=bins, alpha=0.7, label="GALAXY", color="C0")
    axes[0].hist(star_low, bins=bins, alpha=0.5, label="STAR", color="C1")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Redrock ({redrock_zwarn_label}), z<{low_z_max:g}")
    axes[0].legend()

    axes[1].hist(weights_z, bins=50, color="C2", alpha=0.8)
    if z_cutoff is not None:
        axes[1].axvline(float(z_cutoff), color="r", ls="--", label=f"z={float(z_cutoff):g}")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("count")
    axes[1].set_title(weights_title)
    axes[1].legend()

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_redshift_summary(
    galaxy_z: np.ndarray,
    star_z: np.ndarray,
    *,
    redrock_zwarn_label: str,
) -> None:
    print(f"Full catalog GALAXY ({redrock_zwarn_label}): n={len(galaxy_z)}")
    for cut in (0.001, 0.01, 0.05):
        n = int((galaxy_z < cut).sum())
        print(f"  z<{cut:g}: {n} ({100 * n / max(len(galaxy_z), 1):.2f}%)")
    print(f"  z<0: {(galaxy_z < 0).sum()}")
    print(f"\nFull catalog STAR ({redrock_zwarn_label}): n={len(star_z)}")
    n_near0 = int((np.abs(star_z) < 0.001).sum())
    print(
        f"  |z|<0.001: {n_near0} ({100 * n_near0 / max(len(star_z), 1):.2f}%)"
    )


def run_clr_triangle(args: argparse.Namespace) -> None:
    build = prior_build_from_args(args)
    kde_path = build.kde_path
    artifact = load_sed_prior_kde(kde_path)
    outdir = diagnostics_subdir(args, "clr_triangle")
    print(f"Prior dir: {build.prior_dir}")
    print(f"Output dir: {outdir}")
    n_templates = int(artifact["n_templates"])
    support_mode = artifact.get("support_mode", "smooth")
    threshold = float(args.inactive_weight_threshold)

    mask_arg = True if support_mode == "masked" else False
    draws = sample_sed_prior(
        artifact,
        args.sample,
        seed=args.seed,
        apply_support_mask=mask_arg,
    )
    feature_names = list(artifact["feature_names"])
    labels = _clr_param_labels(feature_names, n_templates)

    a_kde = clr_to_weights(draws[:, :n_templates])
    zero_mask = inactive_mask_from_weights(
        a_kde, n_templates, threshold=threshold
    )
    n_inactive = zero_mask[:, :n_templates].sum(axis=0)
    print(
        f"Inactive counts per template (a_k <= {threshold:g}): "
        + ", ".join(f"f{k+1}={n_inactive[k]}" for k in range(n_templates))
    )

    thresh_label = f"{threshold:.0e}".replace("e-0", "e-")
    out_name = "kde_samples_clr_triangle_zero_highlight.png"
    save_clr_triangle_zero_highlight(
        outdir,
        draws,
        labels,
        zero_mask,
        filename=out_name,
        title=(
            rf"CLR KDE samples with low-weight templates highlighted "
            rf"($N={args.sample}$, red = $a_k \leq {thresh_label}$)"
        ),
        panel_size=1.35,
    )
    print(f"Saved {outdir / out_name}")

    if args.also_training:
        training_x = np.asarray(artifact["training_x"], dtype=float)
        meta = artifact.get("metadata", {})
        weights_csv = Path(meta["weights_csv"]).expanduser()
        df = load_prior_training_table(
            weights_csv,
            max_chi2_dof=meta.get("max_chi2_dof"),
            z_min=meta.get("z_min"),
        )
        a_raw = df[prior_a_column_names(df)].to_numpy(dtype=float)
        train_mask = inactive_mask_from_weights(
            a_raw, n_templates, threshold=threshold
        )
        train_name = "training_clr_triangle_zero_highlight.png"
        save_clr_triangle_zero_highlight(
            outdir,
            training_x,
            labels,
            train_mask,
            filename=train_name,
            title=(
                rf"CLR training data with low-weight templates highlighted "
                rf"($N={training_x.shape[0]}, red = $a_k \leq {thresh_label}$)"
            ),
            panel_size=1.35,
        )
        print(f"Saved {outdir / train_name}")

    if artifact.get("gaussianizer_state") is not None:
        gaussianizer = get_empirical_gaussianizer(artifact)
        gaussian_labels = _gaussianized_param_labels(feature_names)
        draws_whitened = gaussianize_with(gaussianizer, draws, whitening="cholesky")
        whitened_name = "kde_samples_whitened_gaussianized_triangle_zero_highlight.png"
        save_clr_triangle_zero_highlight(
            outdir,
            draws_whitened,
            gaussian_labels,
            zero_mask,
            filename=whitened_name,
            title=(
                rf"Cholesky-whitened gaussianized KDE samples with low-weight templates "
                rf"highlighted ($N={args.sample}$, red = $a_k \leq {thresh_label}$)"
            ),
            panel_size=1.35,
        )
        print(f"Saved {outdir / whitened_name}")

        if args.also_training:
            training_whitened = gaussianize_with(
                gaussianizer, training_x, whitening="cholesky"
            )
            train_whitened_name = (
                "training_whitened_gaussianized_triangle_zero_highlight.png"
            )
            save_clr_triangle_zero_highlight(
                outdir,
                training_whitened,
                gaussian_labels,
                train_mask,
                filename=train_whitened_name,
                title=(
                    rf"Cholesky-whitened gaussianized training data with low-weight "
                    rf"templates highlighted ($N={training_x.shape[0]}, "
                    rf"red = $a_k \leq {thresh_label}$)"
                ),
                panel_size=1.35,
            )
            print(f"Saved {outdir / train_whitened_name}")
    else:
        print(
            "Skipping whitened gaussianized zero-highlight plots "
            "(artifact has no gaussianizer_state)."
        )


def run_redshift_histograms(args: argparse.Namespace) -> None:
    build = prior_build_from_args(args)
    desi_dir = resolve_desi_dir(args.desi_dir)
    weights_csv = build.weights_csv
    outdir = diagnostics_subdir(args, "redshift_histograms")
    print(f"Prior dir: {build.prior_dir}")
    print(f"Output dir: {outdir}")

    df = pd.read_csv(weights_csv)
    if "z" not in df.columns:
        raise KeyError(f"Missing z column in {weights_csv}")
    weights_z = df["z"].to_numpy(dtype=float)

    zwarn_zero, zwarn_forbid_mask, redrock_zwarn_label = resolve_redrock_zwarn_selection(
        args
    )
    galaxy_z, star_z = collect_redrock_redshifts(
        desi_dir,
        zwarn_zero=zwarn_zero,
        zwarn_forbid_mask=zwarn_forbid_mask,
    )
    _print_redshift_summary(galaxy_z, star_z, redrock_zwarn_label=redrock_zwarn_label)

    if args.no_z_cutoff_line:
        z_cutoff = None
    elif args.z_cutoff is not None:
        z_cutoff = args.z_cutoff
    else:
        z_cutoff = z_cutoff_from_prior(build)
    save_redshift_histograms(
        outdir,
        galaxy_z=galaxy_z,
        star_z=star_z,
        weights_z=weights_z,
        z_cutoff=z_cutoff,
        low_z_max=args.low_z_max,
        filename=args.filename,
        redrock_zwarn_label=redrock_zwarn_label,
        weights_title=weights_histogram_title(df, weights_z),
    )
    print(f"Saved {outdir / args.filename}")


def run_sed_examples(args: argparse.Namespace) -> None:
    build = prior_build_from_args(args)
    kde_path = build.kde_path
    outdir = diagnostics_subdir(args, "sed_examples")
    print(f"Prior dir: {build.prior_dir}")
    print(f"Output dir: {outdir}")

    prior_args, design_args = load_experiment_configs(
        kde_path=kde_path,
        pool_size=args.prior_pool_size,
        pool_seed=args.prior_pool_seed,
    )

    exp = NumVisits(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model="empirical",
        device=args.device,
        cdf_samples=2000,
        transform_input=False,
        verbose=True,
    )
    if exp.prior_pool is None:
        raise RuntimeError("NumVisits did not initialize an empirical prior pool")

    print(f"NumVisits empirical prior: {exp.prior_kde_path}")
    print(f"  parameterization: {exp._prior_parameterization}")
    print(f"  feature_names: {exp.prior_feature_names}")
    print(f"  prior pool: {exp.prior_pool.pool.shape[0]} rows on {exp.device}")

    generator = torch.Generator(device=exp.device)
    generator.manual_seed(args.seed)
    rows = sample_prior_batch(exp.prior_pool, args.n_samples, generator=generator)

    with torch.no_grad():
        a, log_s, z = exp._empirical_rows_to_physical(rows, ())
        flux_obs = exp._observed_spectral_flux(z, a=a, log_s=log_s)
        mags = exp._calculate_magnitudes(flux_obs)
        coeffs = torch.exp(log_s).unsqueeze(-1) * a
        flux_rest = coeffs @ exp._template_flux

    wave_rest = exp._template_wave_rest.detach().cpu().numpy()
    wave_obs = exp._wlen_aa_tensor.detach().cpu().numpy()
    templates = exp._template_flux.detach().cpu().numpy()
    a_np = a.detach().cpu().numpy()
    log_s_np = log_s.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()
    seds_rest = flux_rest.detach().cpu().numpy()
    seds_obs = flux_obs.detach().cpu().numpy()
    mags_np = mags.detach().cpu().numpy()
    rows_np = rows.detach().cpu().numpy()

    plot_templates(wave_rest, templates, outdir / "eazy_templates_rest.png")
    plot_rest_seds(wave_rest, seds_rest, z_np, log_s_np, outdir / "empirical_seds_rest.png")
    plot_observed_seds(wave_obs, seds_obs, z_np, outdir / "empirical_seds_observed.png")
    plot_magnitudes(exp.filters_list, mags_np, z_np, outdir / "empirical_lsst_magnitudes.png")
    plot_weight_heatmap(a_np, outdir / "empirical_weights.png")

    np.savez(
        outdir / "empirical_seds.npz",
        wavelength_rest_aa=wave_rest,
        wavelength_obs_aa=wave_obs,
        templates=templates,
        feature_rows=rows_np,
        feature_names=np.array(exp.prior_feature_names),
        a=a_np,
        log_c_scale=log_s_np,
        z=z_np,
        sample_seds_rest=seds_rest,
        sample_seds_obs=seds_obs,
        lsst_magnitudes=mags_np,
        lsst_filters=np.array(exp.filters_list),
        kde_path=str(kde_path),
        parameterization=exp._prior_parameterization,
    )

    print("Done.")
    print(f"Wrote {args.n_samples} empirical SED examples to {outdir.resolve()}")
    print("First draw:")
    print(f"  z={z_np[0]:.4f}, log_c_scale={log_s_np[0]:.4f}")
    print(f"  LSST mags: {dict(zip(exp.filters_list, mags_np[0]))}")
    print("  top template weights:")
    top = np.argsort(a_np[0])[::-1][:5]
    for idx in top:
        print(f"    a{idx + 1}: {a_np[0, idx]:.4f}")


def run_mag_leakage(args: argparse.Namespace) -> None:
    build = prior_build_from_args(args)
    kde_path = build.kde_path
    outdir = diagnostics_subdir(args, "mag_leakage")
    print(f"Prior dir: {build.prior_dir}")
    print(f"Output dir: {outdir}")

    artifact = load_sed_prior_kde(kde_path)
    rows_np = sample_sed_prior(artifact, args.n_samples, seed=args.seed)
    training_a_raw = get_raw_training_weights(artifact)

    prior_args, design_args = load_experiment_configs(
        kde_path=kde_path,
        pool_size=args.prior_pool_size,
        pool_seed=args.seed,
    )
    exp = NumVisits(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model="empirical",
        device=args.device,
        cdf_samples=2000,
        transform_input=False,
        verbose=False,
    )

    rows = torch.tensor(rows_np, device=exp.device, dtype=torch.float64)
    with torch.no_grad():
        a, log_s, z = exp._empirical_rows_to_physical(rows, ())
        mags_smooth = magnitudes_from_weights(exp, z, a, log_s)

        a_thr = apply_inactive_threshold(a, args.inactive_weight_threshold)
        mags_thr = magnitudes_from_weights(exp, z, a_thr, log_s)

        a_msk = apply_support_mask_torch(a, training_a_raw, seed=args.seed + 99991)
        mags_msk = magnitudes_from_weights(exp, z, a_msk, log_s)

    filters = list(exp.filters_list)
    m_smooth = mags_smooth.cpu().numpy()
    m_thr = mags_thr.cpu().numpy()
    m_msk = mags_msk.cpu().numpy()

    df_thr = summarize_mag_deltas(filters, m_smooth, m_thr, label="threshold")
    df_msk = summarize_mag_deltas(filters, m_smooth, m_msk, label="masked")
    summary = pd.concat([df_thr, df_msk], ignore_index=True)
    summary_path = outdir / "mag_delta_summary.csv"
    summary.to_csv(summary_path, index=False)

    delta_thr = m_thr - m_smooth
    delta_msk = m_msk - m_smooth
    plot_mag_delta_histograms(
        filters,
        delta_thr,
        delta_msk,
        outdir / "mag_delta_by_filter.png",
    )
    plot_mag_max_delta_summary(
        np.max(np.abs(delta_thr), axis=1),
        np.max(np.abs(delta_msk), axis=1),
        outdir / "mag_max_delta_hist.png",
    )

    np.savez(
        outdir / "mag_comparison_samples.npz",
        lsst_filters=np.array(filters),
        mags_smooth=m_smooth,
        mags_threshold=m_thr,
        mags_masked=m_msk,
        delta_threshold=delta_thr,
        delta_masked=delta_msk,
        z=z.cpu().numpy(),
        inactive_threshold=float(args.inactive_weight_threshold),
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        kde_path=str(kde_path),
    )

    print(f"Compared {args.n_samples} KDE draws (smooth reference)")
    print(f"Inactive threshold: a_k <= {args.inactive_weight_threshold:g}")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {outdir / 'mag_delta_by_filter.png'}")
    print(f"Wrote {outdir / 'mag_max_delta_hist.png'}")
    print(f"Wrote {outdir / 'mag_comparison_samples.npz'}")
    print("\nSummary (max |Δm| over bands per variant):")
    for label in ("threshold", "masked"):
        row = summary[(summary["variant"] == label) & (summary["filter"] == "max_over_bands")].iloc[0]
        print(
            f"  {label:9s}: median={row['median_delta_mag']:.5f}, "
            f"p95={row['p95_abs_delta_mag']:.5f}, "
            f"max={row['max_abs_delta_mag']:.5f} AB"
        )
    print("\nPer-filter median Δm (threshold − smooth):")
    thr_only = summary[summary["variant"] == "threshold"]
    for _, row in thr_only[thr_only["filter"] != "max_over_bands"].iterrows():
        print(f"  {row['filter']:>1s}: {row['median_delta_mag']:+.5f} mag")


def run_all(args: argparse.Namespace) -> None:
    build = prior_build_from_args(args)
    root = diagnostics_root(args)
    print(f"Running all diagnostics for {build.prior_dir}")
    print(f"Diagnostics root: {root}")

    steps: list[tuple[str, object, dict]] = [
        ("clr_triangle", run_clr_triangle, {}),
        ("redshift_histograms", run_redshift_histograms, {"filename": "redshift_histograms.png"}),
        ("sed_examples", run_sed_examples, {"n_samples": 12}),
        ("mag_leakage", run_mag_leakage, {"n_samples": 5000}),
    ]
    for subname, func, overrides in steps:
        print(f"\n========== {subname} ==========")
        sub_args = argparse.Namespace(**{**vars(args), **overrides})
        sub_args.outdir = root
        func(sub_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    add_prior_build_args(common)

    p_all = sub.add_parser(
        "all",
        parents=[common],
        help="Run all diagnostic plot subcommands.",
    )
    add_desi_dir_argument(p_all)
    p_all.add_argument("--seed", type=int, default=7)
    p_all.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_KDE_DIAGNOSTIC_SAMPLES,
        help="KDE draws for clr_triangle.",
    )
    p_all.add_argument(
        "--inactive-weight-threshold",
        type=float,
        default=DEFAULT_INACTIVE_WEIGHT_THRESHOLD,
    )
    p_all.add_argument(
        "--also-training",
        action="store_true",
        default=True,
        help="Include training CLR triangle in clr_triangle step.",
    )
    p_all.add_argument("--no-z-cutoff-line", action="store_true")
    p_all.add_argument("--z-cutoff", type=float, default=None)
    p_all.add_argument("--low-z-max", type=float, default=0.15, dest="low_z_max")
    add_redrock_zwarn_arguments(p_all)
    p_all.add_argument("--prior-pool-size", type=int, default=4096)
    p_all.add_argument("--prior-pool-seed", type=int, default=7)
    p_all.add_argument("--device", default="cpu")
    p_all.set_defaults(func=run_all)

    p_clr = sub.add_parser(
        "clr-triangle",
        parents=[common],
        help="CLR triangle plot with inactive template highlighting.",
    )
    p_clr.add_argument(
        "--kde-path",
        type=Path,
        default=None,
        help="Override KDE path (default: <prior-dir>/sed_prior_kde_native.joblib).",
    )
    p_clr.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_KDE_DIAGNOSTIC_SAMPLES,
        help="Number of KDE draws to plot.",
    )
    p_clr.add_argument("--seed", type=int, default=7)
    p_clr.add_argument(
        "--inactive-weight-threshold",
        type=float,
        default=DEFAULT_INACTIVE_WEIGHT_THRESHOLD,
        help=(
            "Template k is inactive (red) when a_k <= this weight threshold. "
            f"Default {DEFAULT_INACTIVE_WEIGHT_THRESHOLD:g}."
        ),
    )
    p_clr.add_argument(
        "--also-training",
        action="store_true",
        help="Also write training_clr_triangle_zero_highlight.png from training_x.",
    )
    p_clr.set_defaults(func=run_clr_triangle)

    p_z = sub.add_parser(
        "redshift-histograms",
        parents=[common],
        help="Low-z redshift histograms (redrock spectypes vs weights CSV).",
    )
    add_desi_dir_argument(p_z)
    p_z.add_argument(
        "--weights-csv",
        type=Path,
        default=None,
        help="Override weights CSV (default: <prior-dir>/desi_eazy_empirical_weights.csv).",
    )
    p_z.add_argument(
        "--z-cutoff",
        type=float,
        default=None,
        help="Dashed line on weights panel (default: z_min from KDE metadata).",
    )
    p_z.add_argument(
        "--no-z-cutoff-line",
        action="store_true",
        help="Omit the z-cutoff vertical line on the weights panel.",
    )
    p_z.add_argument(
        "--low-z-max",
        type=float,
        default=0.15,
        dest="low_z_max",
        help="Upper z limit for the redrock spectype panel (default 0.15).",
    )
    add_redrock_zwarn_arguments(p_z)
    p_z.add_argument(
        "--filename",
        default="redshift_histograms.png",
        help="Output PNG filename.",
    )
    p_z.set_defaults(func=run_redshift_histograms)

    p_sed = sub.add_parser(
        "sed-examples",
        parents=[common],
        help="Sample empirical-prior SEDs through NumVisits.",
    )
    p_sed.add_argument(
        "--kde-path",
        type=Path,
        default=None,
        help="Override KDE path (default: <prior-dir>/sed_prior_kde_native.joblib).",
    )
    p_sed.add_argument("--n-samples", type=int, default=12)
    p_sed.add_argument("--seed", type=int, default=7)
    p_sed.add_argument(
        "--prior-pool-size",
        type=int,
        default=4096,
        help="Prior pool size passed to NumVisits.",
    )
    p_sed.add_argument(
        "--prior-pool-seed",
        type=int,
        default=7,
        help="Seed used when building the NumVisits prior pool.",
    )
    p_sed.add_argument("--device", default="cpu")
    p_sed.set_defaults(func=run_sed_examples)

    p_mag = sub.add_parser(
        "mag-leakage",
        parents=[common],
        help="Compare LSST mags: smooth KDE vs threshold/masked weights.",
    )
    p_mag.add_argument(
        "--kde-path",
        type=Path,
        default=None,
        help="Override KDE path (default: <prior-dir>/sed_prior_kde_native.joblib).",
    )
    p_mag.add_argument("--n-samples", type=int, default=5000)
    p_mag.add_argument("--seed", type=int, default=7)
    p_mag.add_argument(
        "--inactive-weight-threshold",
        type=float,
        default=DEFAULT_INACTIVE_WEIGHT_THRESHOLD,
        help=(
            "Zero templates with a_k <= this before renormalize "
            f"(default {DEFAULT_INACTIVE_WEIGHT_THRESHOLD:g})."
        ),
    )
    p_mag.add_argument("--device", default="cpu")
    p_mag.add_argument(
        "--prior-pool-size",
        type=int,
        default=4096,
        help="Passed to NumVisits init (keeps config consistent).",
    )
    p_mag.set_defaults(func=run_mag_leakage)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
