"""Diagnose how efficiently an EAZY template bank spans the fitted DESI population.

The figure answers two related questions:

1. Which original EAZY templates are actually used by the quality-passing DESI
   fits, and which can be reconstructed from the rest of the bank?
2. How much fidelity is lost when the template simplex is replaced by a
   truncated linear PCA basis?

All comparisons use the EAZY reconstruction of each fitted DESI spectrum.  The
LSST-color diagnostics evaluate every reconstruction at that galaxy's fitted
DESI redshift and remove the mean magnitude, so ``log_c_scale`` does not enter.

Usage::

    conda run -n bedcosmo python \
      experiments/num_visits/scripts/plot_eazy_basis_representativeness.py \
      --build eazy6

Use ``--build eazy12`` for the production twelve-template bank. The default
prior layout is ``$SCRATCH/bedcosmo/num_visits/empirical_prior/<build>``.
Override it with ``--prior-dir`` when inspecting another build.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import nnls
from speclite import filters as speclite_filters

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _paths import plot_path  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from bedcosmo.num_visits.empirical.templates import load_eazy_template_bank  # noqa: E402

INK = "#25272B"
MUTED = "#6B7280"
GRID = "#D9DDE3"
PC_COLOR = "#3366CC"
ILR_COLOR = "#DC3912"

BUILD_CONFIG = {
    "eazy6": "templates/eazy_v1.0.spectra.param",
    "eazy12": "templates/fsps_full/fsps_QSF_12_v3.param",
}


def default_scratch() -> Path:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise RuntimeError("$SCRATCH is not set; pass --prior-dir and --template-dir")
    return Path(scratch).expanduser()


def read_fit_table(path: Path, n_templates: int) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    required = {
        "success",
        "quality_pass",
        "z",
        *(f"a{i}" for i in range(1, n_templates + 1)),
    }
    missing = required.difference(table.dtype.names or ())
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    keep = table["success"].astype(bool) & table["quality_pass"].astype(bool)
    weights = np.column_stack([table[f"a{i}"][keep] for i in range(1, n_templates + 1)]).astype(
        float
    )
    redshift = np.asarray(table["z"][keep], dtype=float)
    finite = np.all(np.isfinite(weights), axis=1) & np.isfinite(redshift)
    weights = weights[finite]
    redshift = redshift[finite]
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, redshift


def read_templates(template_dir: Path, template_param: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the exact normalized, gridded template bank used by ``NumVisits``."""
    wave, templates, _ = load_eazy_template_bank(
        template_param=template_param,
        template_dir=template_dir,
    )
    return np.asarray(wave, dtype=float), np.asarray(templates, dtype=float)


def ilr_features(weights: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    smooth = weights + floor
    smooth /= smooth.sum(axis=1, keepdims=True)
    clr = np.log(smooth) - np.log(smooth).mean(axis=1, keepdims=True)
    # Any orthonormal basis of the sum-zero subspace has identical PCA eigenvalues.
    n_templates = weights.shape[1]
    basis = np.zeros((n_templates, n_templates - 1))
    for j in range(n_templates - 1):
        basis[: j + 1, j] = 1.0 / np.sqrt((j + 1) * (j + 2))
        basis[j + 1, j] = -(j + 1) / np.sqrt((j + 1) * (j + 2))
    return clr @ basis


def explained_variance(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0)
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
    positive = eigenvalues > eigenvalues.max() * 1e-12
    eigenvalues = eigenvalues[positive]
    return eigenvalues / eigenvalues.sum()


def spectral_pca(
    weights: np.ndarray, templates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a compact PCA transform under the rest-spectrum L2 metric."""
    gram = templates @ templates.T
    values, vectors = np.linalg.eigh(gram)
    sqrt_gram = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    inv_sqrt_gram = np.linalg.pinv(sqrt_gram, rcond=1e-12)

    mean = weights.mean(axis=0)
    encoded = (weights - mean) @ sqrt_gram
    covariance = encoded.T @ encoded / (len(encoded) - 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = components[:, order]
    return mean, encoded, components, inv_sqrt_gram


def reconstruct_pca_weights(
    mean: np.ndarray,
    encoded: np.ndarray,
    components: np.ndarray,
    inv_sqrt_gram: np.ndarray,
    n_components: int,
) -> np.ndarray:
    projection = components[:, :n_components]
    encoded_hat = (encoded @ projection) @ projection.T
    return mean + encoded_hat @ inv_sqrt_gram


def lsst_template_responses(
    wave_rest: np.ndarray,
    templates: np.ndarray,
    redshift: np.ndarray,
    chunk_size: int = 500,
) -> np.ndarray:
    """Band-integrated flux for every (galaxy, template, LSST band)."""
    filter_data = []
    for band in "ugrizy":
        loaded = speclite_filters.load_filter("lsst2023-" + band)
        wave = np.asarray(loaded.wavelength, dtype=float)[::5]
        transmission = np.asarray(loaded(wave), dtype=float)
        filter_data.append((wave, transmission))

    n_wave = max(len(wave) for wave, _ in filter_data)
    wave_obs = np.linspace(
        min(wave.min() for wave, _ in filter_data),
        max(wave.max() for wave, _ in filter_data),
        n_wave,
    )
    transmission = np.stack(
        [np.interp(wave_obs, wave, response, left=0.0, right=0.0) for wave, response in filter_data]
    )
    photon_kernel = transmission * wave_obs[None, :]

    n_templates = len(templates)
    responses = np.empty((len(redshift), n_templates, 6), dtype=float)
    for start in range(0, len(redshift), chunk_size):
        stop = min(start + chunk_size, len(redshift))
        z = redshift[start:stop]
        rest_wave = wave_obs[None, :] / (1.0 + z[:, None])
        observed_templates = np.stack(
            [
                np.interp(rest_wave.ravel(), wave_rest, flux, left=0.0, right=0.0).reshape(
                    len(z), n_wave
                )
                / (1.0 + z[:, None])
                for flux in templates
            ],
            axis=1,
        )
        responses[start:stop] = np.trapz(
            observed_templates[:, :, None, :] * photon_kernel[None, None, :, :],
            wave_obs,
            axis=-1,
        )
    return responses


def centered_colors(weights: np.ndarray, responses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flux = np.einsum("nk,nkb->nb", weights, responses)
    valid = np.all(np.isfinite(flux) & (flux > 0.0), axis=1)
    magnitudes = np.full_like(flux, np.nan)
    magnitudes[valid] = -2.5 * np.log10(flux[valid])
    magnitudes[valid] -= magnitudes[valid].mean(axis=1, keepdims=True)
    return magnitudes, valid


def color_error(
    reference: np.ndarray,
    candidate_weights: np.ndarray,
    responses: np.ndarray,
    reference_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate, candidate_valid = centered_colors(candidate_weights, responses)
    valid = reference_valid & candidate_valid
    rms = np.sqrt(np.mean((candidate[valid] - reference[valid]) ** 2, axis=1))
    return rms, valid


def leave_one_template_out(
    weights: np.ndarray, templates: np.ndarray, omitted: int
) -> tuple[np.ndarray, float]:
    """Replace one template with its best nonnegative combination of the others."""
    n_templates = templates.shape[0]
    retained = [i for i in range(n_templates) if i != omitted]
    replacement, _ = nnls(templates[retained].T, templates[omitted])
    approximation = replacement @ templates[retained]
    residual = np.linalg.norm(templates[omitted] - approximation) / np.linalg.norm(
        templates[omitted]
    )
    reconstructed = weights.copy()
    reconstructed[:, retained] += weights[:, omitted, None] * replacement[None, :]
    reconstructed[:, omitted] = 0.0
    return reconstructed, float(residual)


def make_figure(
    weights: np.ndarray,
    redshift: np.ndarray,
    wave: np.ndarray,
    templates: np.ndarray,
    responses: np.ndarray,
    output: Path,
) -> dict[str, np.ndarray]:
    n_templates = templates.shape[0]
    n_shape = n_templates - 1
    labels = [f"T{i}" for i in range(1, n_templates + 1)]
    color_map = plt.get_cmap("tab20")
    colors = [color_map(i / max(n_templates - 1, 1)) for i in range(n_templates)]
    mean_weight = weights.mean(axis=0)
    active_fraction = (weights > 1e-10).mean(axis=0)
    dominant_fraction = np.array(
        [(np.argmax(weights, axis=1) == i).mean() for i in range(n_templates)]
    )

    reference_colors, reference_valid = centered_colors(weights, responses)
    omit_median = np.empty(n_templates)
    omit_p95 = np.empty(n_templates)
    omit_spectral = np.empty(n_templates)
    for i in range(n_templates):
        reconstructed, omit_spectral[i] = leave_one_template_out(weights, templates, i)
        errors, _ = color_error(reference_colors, reconstructed, responses, reference_valid)
        omit_median[i], omit_p95[i] = np.percentile(errors, [50, 95])

    mean, encoded, components, inv_sqrt = spectral_pca(weights, templates)
    spectral_variance = explained_variance(encoded)
    ilr_variance = explained_variance(ilr_features(weights))
    pca_median = np.empty(n_shape)
    pca_p95 = np.empty(n_shape)
    pca_valid = np.empty(n_shape)
    for n_components in range(1, n_shape + 1):
        reconstructed = reconstruct_pca_weights(mean, encoded, components, inv_sqrt, n_components)
        errors, valid = color_error(reference_colors, reconstructed, responses, reference_valid)
        pca_median[n_components - 1], pca_p95[n_components - 1] = np.percentile(errors, [50, 95])
        pca_valid[n_components - 1] = valid.mean()

    fig = plt.figure(figsize=(15.0, 12.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 0.92])
    ax_sed = fig.add_subplot(grid[0, :])
    ax_use = fig.add_subplot(grid[1, 0])
    ax_omit = fig.add_subplot(grid[1, 1])
    ax_var = fig.add_subplot(grid[2, 0])
    ax_pca = fig.add_subplot(grid[2, 1])

    # Show the same 4000--8000 Angstrom-normalized templates used by the analysis.
    display_mask = (wave >= 1200.0) & (wave <= 11000.0)
    for i, (template, color) in enumerate(zip(templates, colors)):
        ax_sed.plot(
            wave[display_mask], template[display_mask], color=color, lw=1.8, label=labels[i]
        )
    ax_sed.set_yscale("log")
    ax_sed.set_xlim(1200, 11000)
    positive_flux = templates[:, display_mask][templates[:, display_mask] > 0.0]
    ax_sed.set_ylim(np.percentile(positive_flux, 0.2) / 1.3, positive_flux.max() * 1.25)
    ax_sed.set_xlabel(r"Rest wavelength $\lambda$ [$\AA$]")
    ax_sed.set_ylabel(r"Template $f_\lambda$ (unit integral over 4000--8000 $\AA$)")
    ax_sed.set_title(f"The {n_templates} candidate directions in rest-frame SED space", loc="left")
    ax_sed.legend(ncol=min(n_templates, 6), frameon=False, loc="upper right")

    # Weight use combines the full distribution with easy-to-read prevalence summaries.
    violin = ax_use.violinplot(
        [weights[:, i] for i in range(n_templates)],
        positions=np.arange(1, n_templates + 1),
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        quantiles=[[0.05, 0.95]] * n_templates,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.34)
    for key in ("cmedians", "cquantiles"):
        violin[key].set_color(INK)
        violin[key].set_linewidth(1.1)
    template_x = np.arange(1, n_templates + 1)
    ax_use.scatter(
        template_x,
        mean_weight,
        c=colors,
        edgecolor=INK,
        s=45,
        zorder=4,
    )
    ax_use.scatter(
        template_x,
        active_fraction,
        marker="^",
        facecolors="none",
        edgecolors=INK,
        s=48,
        linewidths=1.2,
        zorder=4,
    )
    ax_use.scatter(
        template_x,
        dominant_fraction,
        marker="x",
        c=INK,
        s=42,
        linewidths=1.4,
        zorder=4,
    )
    ax_use.set_ylim(0, 1.05)
    ax_use.set_xticks(np.arange(1, n_templates + 1), labels)
    ax_use.set_xlabel("EAZY template")
    ax_use.set_ylabel("Normalized fitted coefficient $a_k$")
    ax_use.set_title(f"Fitted to {len(weights):,} DESI Spectra", loc="left", pad=38)
    ax_use.text(
        0.0,
        1.015,
        "colored dot: mean; black bar: 5–95%\n"
        r"$\triangle$ active fraction: fraction with $a_k>10^{-10}$; "
        r"$\times$ dominant fraction: fraction where $a_k$ is largest",
        transform=ax_use.transAxes,
        va="bottom",
        color=MUTED,
        fontsize=8.5 if n_templates > 8 else 9,
    )

    # Leave-one-template-out impact; the spectral replacement itself is nonnegative.
    x = np.arange(1, n_templates + 1)
    ax_omit.vlines(x, omit_median, omit_p95, color=colors, lw=5, alpha=0.42)
    ax_omit.scatter(x, omit_p95, c=colors, marker="_", s=240, linewidths=2.2)
    ax_omit.scatter(x, omit_median, c=colors, edgecolor=INK, s=48, zorder=3)
    for i in range(n_templates):
        ax_omit.text(
            i + 1,
            omit_p95[i] + max(omit_p95.max() * 0.035, 0.002),
            f"{omit_spectral[i]:.0%}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=MUTED,
        )
    ax_omit.set_xticks(x, labels)
    ax_omit.set_xlabel("Omitted template")
    ax_omit.set_ylabel("LSST color reconstruction RMS [mag]")
    ax_omit.set_ylim(0, omit_p95.max() * 1.26)
    ax_omit.set_title(f"Can the other {n_templates - 1} templates replace it?", loc="left")
    ax_omit.text(
        0.02,
        0.96,
        "dot: median; cap: 95th percentile; label: template spectral residual",
        transform=ax_omit.transAxes,
        va="top",
        color=MUTED,
        fontsize=9,
    )

    n_ilr = np.arange(1, len(ilr_variance) + 1)
    n_sed = np.arange(1, len(spectral_variance) + 1)
    ax_var.plot(
        n_sed, np.cumsum(spectral_variance), "o-", color=PC_COLOR, lw=2, label="rest-SED flux PCA"
    )
    ax_var.plot(
        n_ilr, np.cumsum(ilr_variance), "s-", color=ILR_COLOR, lw=2, label="ILR coefficient PCA"
    )
    ax_var.axhline(0.95, color=MUTED, lw=1, ls="--")
    ax_var.set_xlim(0.8, n_shape + 0.2)
    ax_var.set_ylim(0, 1.04)
    ax_var.set_xticks(np.arange(1, n_shape + 1))
    ax_var.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_var.set_xlabel("Retained shape components")
    ax_var.set_ylabel("Cumulative variance explained")
    ax_var.set_title("Variance alone gives conflicting answers", loc="left")
    ax_var.legend(frameon=False, loc="lower right")

    # The full shape rank reproduces the original model to machine precision,
    # which would stretch the log axis and hide every genuinely reduced model.
    n_reduced = n_shape - 1
    n_pc = np.arange(1, n_reduced + 1)
    ax_pca.fill_between(
        n_pc, pca_median[:n_reduced], pca_p95[:n_reduced], color=PC_COLOR, alpha=0.18
    )
    ax_pca.plot(
        n_pc,
        pca_p95[:n_reduced],
        "o-",
        color=PC_COLOR,
        lw=2,
        label="95th percentile",
    )
    ax_pca.plot(n_pc, pca_median[:n_reduced], "o--", color=INK, lw=1.5, label="median")
    ax_pca.axhline(0.01, color=MUTED, lw=1, ls=":", label="0.01 mag")
    ax_pca.set_yscale("log")
    ax_pca.set_xlim(0.8, n_reduced + 0.2)
    ax_pca.set_xticks(n_pc)
    ax_pca.set_xlabel("Retained rest-SED principal components")
    ax_pca.set_ylabel("LSST color reconstruction RMS [mag]")
    ax_pca.set_title("Photometric-redshift fidelity needs more than variance", loc="left")
    ax_pca.legend(frameon=False, loc="upper right")

    for ax in (ax_sed, ax_use, ax_omit, ax_var, ax_pca):
        ax.tick_params(colors=INK)
        ax.grid(True, color=GRID, lw=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    fig.suptitle(
        f"Representative power of the EAZY{n_templates} basis for fitted DESI galaxy spectra",
        fontsize=17,
        color=INK,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "mean_weight": mean_weight,
        "active_fraction": active_fraction,
        "dominant_fraction": dominant_fraction,
        "omit_median": omit_median,
        "omit_p95": omit_p95,
        "omit_spectral": omit_spectral,
        "spectral_variance": spectral_variance,
        "ilr_variance": ilr_variance,
        "pca_median": pca_median,
        "pca_p95": pca_p95,
        "pca_valid": pca_valid,
        "redshift": redshift,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build",
        choices=sorted(BUILD_CONFIG),
        default="eazy6",
        help="Named template bank and default prior subdirectory",
    )
    parser.add_argument(
        "--prior-dir",
        type=Path,
        default=None,
        help="Prior build containing desi_eazy_empirical_weights.csv",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=None,
        help="EAZY download root containing the selected template parameter file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scratch = default_scratch() if args.prior_dir is None or args.template_dir is None else None
    prior_dir = (
        args.prior_dir
        if args.prior_dir is not None
        else scratch / "bedcosmo/num_visits/empirical_prior" / args.build
    )
    template_dir = args.template_dir if args.template_dir is not None else scratch / "bedcosmo/eazy"
    template_param = BUILD_CONFIG[args.build]
    wave, templates = read_templates(template_dir, template_param)
    weights, redshift = read_fit_table(
        prior_dir / "desi_eazy_empirical_weights.csv", n_templates=len(templates)
    )
    responses = lsst_template_responses(wave, templates, redshift)
    output = args.output or plot_path(f"{args.build}_basis_representativeness.png")
    metrics = make_figure(weights, redshift, wave, templates, responses, output)

    print(f"Wrote {output}")
    print(f"Quality-passing DESI fits: {len(weights):,}")
    print("Mean weights:", np.round(metrics["mean_weight"], 4))
    print("Active fractions:", np.round(metrics["active_fraction"], 4))
    print("Omission color RMS p95 [mag]:", np.round(metrics["omit_p95"], 4))
    print("PCA color RMS median [mag]:", np.round(metrics["pca_median"], 4))
    print("PCA color RMS p95 [mag]:", np.round(metrics["pca_p95"], 4))


if __name__ == "__main__":
    main()
