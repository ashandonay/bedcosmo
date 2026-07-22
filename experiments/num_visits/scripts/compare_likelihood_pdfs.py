"""
Compare joint likelihood PDFs P(y_u, y_g | d) from pyro_model (sampling) vs unnorm_lfunc (grid).

Cross-checks the two independent routes to the same P(y|d): the pyro_model
sampler (2D histogram) and the unnorm_lfunc grid marginal via GridCalculation
(the same path as grid_calc.py). Agreement of the 1/2/3-sigma contours, and a
residual map that is noise-level, is the likelihood-correctness check.

Single nominal LSST design (u=70, g=100), uniform prior on z over --z-min/--z-max.
Filled/solid contours = pyro_model (histogram), solid = unnorm_lfunc (grid).

Usage:
  python compare_likelihood_pdfs.py                       # z in [0.1, 2.0]
  python compare_likelihood_pdfs.py --z-min 0.5 \\
      --out compare_likelihood_pdfs_narrow_prior.png      # the narrow-prior variant

The narrow prior (z >= 0.5) drops the low-z end where the magnitude track is
steepest and the per-z Gaussians are sharpest, so it is the easier grid problem;
running both is how we separated genuine likelihood error from grid resolution.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from _paths import PLOTS_DIR, plot_path

from bedcosmo.grid_calc import GridCalculation
from bedcosmo.util import get_experiment_config_path


def contour_levels_from_pdf(pdf_2d, fracs=(0.9973, 0.9545, 0.6827)):
    """Return density thresholds enclosing given mass fractions."""
    flat = pdf_2d.ravel()
    idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[idx])
    cumsum /= cumsum[-1]
    thresholds = []
    for frac in fracs:
        k = min(np.searchsorted(cumsum, frac), len(flat) - 1)
        thresholds.append(flat[idx[k]])
    return sorted(thresholds)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z-min", type=float, default=0.1, help="Uniform prior lower bound on z.")
    p.add_argument("--z-max", type=float, default=2.0, help="Uniform prior upper bound on z.")
    p.add_argument("--n-samples", type=int, default=200_000, help="pyro_model draws.")
    p.add_argument("--out", default="compare_likelihood_pdfs.png",
                   help="Output PNG; bare names land in experiments/num_visits/plots/.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cpu"
    z_min, z_max = args.z_min, args.z_max

    # --- Load configs ---
    design_args_path = get_experiment_config_path("num_visits", "design_args_2d.yaml")

    with open(design_args_path) as f:
        design_args = yaml.safe_load(f)

    # Uniform prior on z, built inline so the range is a CLI knob rather than a
    # second copy of prior_args_uniform.yaml.
    prior_args = {
        "parameters": {
            "z": {
                "distribution": {"type": "uniform", "lower": z_min, "upper": z_max},
                "plot": {"lower": z_min, "upper": z_max},
                "latex": "z",
            }
        },
        "constraints": {},
    }

    # Keep input_type="variable" from the YAML. GridCalculation needs
    # experiment.designs_grid, and NumVisits.init_designs deliberately leaves it
    # None for explicit/nominal designs (a Grid is a Cartesian product, so
    # scattered points would materialize prod(unique values per axis) cells).
    # We build the full design grid here and slice out the nominal design from
    # the marginal afterwards.
    from bedcosmo.num_visits import NumVisits

    exp = NumVisits(
        prior_args=prior_args,
        design_args=design_args,
        cosmo_model="bb",
        temperature=10000,
        device=device,
        verbose=True,
    )

    nominal = exp.nominal_design
    filters = exp.filters_list
    print(f"Nominal design: {nominal}")

    # Collapse the design grid onto the single nominal design. Without this the
    # YAML's variable grid has 9 designs (the sum==170 constraint cuts the 9x9
    # product to its anti-diagonal) and the marginal would be evaluated at
    # 500x500 features for every one of them -- 9x more work than this
    # one-design comparison needs, enough to get OOM-killed on a login node.
    # Pinning lower==upper==nominal keeps designs_grid defined (which
    # GridCalculation requires) at a single point.
    nominal_vals = [float(v) for v in nominal.cpu()]
    exp.init_designs(
        input_type="variable",
        labels=design_args["labels"],
        step=design_args["step"],
        lower=nominal_vals,
        upper=nominal_vals,
    )
    print(f"Design grid collapsed to {exp.designs.shape[0]} design(s)")

    # =====================================================================
    # 1.  pyro_model samples at nominal design
    # =====================================================================
    n_samples = args.n_samples
    print(f"\nSampling {n_samples} points from pyro_model...")
    d = nominal.unsqueeze(0)
    with torch.no_grad():
        samples = exp.sample_data(d, num_samples=n_samples)
    pyro_samp = samples.squeeze(1).cpu().numpy()
    print("  Done")

    # =====================================================================
    # 2.  Grid-based marginal via GridCalculation
    # =====================================================================
    # Use GridCalculation for grid setup, but compute the marginal directly
    # to avoid calculateEIG's normalization assertion (unnorm_lfunc's global
    # max subtraction can cause underflow for some z slices).
    # Feature ranges from eval log; dense regions around the magnitude track
    # to resolve the sharp Gaussians at low z.
    feature_ranges = {"u": (-15.0, 100.0), "g": (15.0, 55.0)}

    # Get magnitude track to set dense regions
    z_check = torch.linspace(z_min, z_max, 200, dtype=torch.float64)
    with torch.no_grad():
        mag_track = exp._calculate_magnitudes(
            exp._observed_spectral_flux(z_check)
        ).cpu().numpy()
    feature_dense_ranges = {
        "u": (float(mag_track[:, 0].min()) - 2.0, float(mag_track[:, 0].max()) + 2.0),
        "g": (float(mag_track[:, 1].min()) - 2.0, float(mag_track[:, 1].max()) + 2.0),
    }
    print(f"  Dense ranges: u={feature_dense_ranges['u']}, g={feature_dense_ranges['g']}")

    print("\nRunning GridCalculation...")
    gc = GridCalculation(
        experiment=exp,
        param_pts=500,
        feature_pts=500,
        device=device,
        feature_ranges=feature_ranges,
        feature_dense_ranges=feature_dense_ranges,
        feature_dense_fraction=0.8,
        # Required: grid_calc.py does os.makedirs(plt_save_path) unconditionally,
        # so the documented plt_save_path=None default raises TypeError. Its own
        # diagnostic (grid_prior_pdf.png) lands beside our figure.
        plt_save_path=str(PLOTS_DIR),
    )
    gc.compute_prior_pdf(use_experiment_prior=True, normalize=True)

    # Marginalize P(y|d) = sum_z p(y|z,d) prior(z) directly instead of calling
    # gc.run(). run() goes through ExperimentDesigner.calculateEIG, which iterates
    # designs.subgrid(); bed.Grid cannot build a single-design constrained subgrid
    # (it squeezes the mask to 0-d and jnp.nonzero rejects it). We only ever wanted
    # the marginal, so take the weighted sum here -- and computing the full EIG
    # would mean evaluating all 9 grid designs to use one.
    print("\nMarginalizing the grid likelihood over z...")
    like = np.asarray(
        exp.unnorm_lfunc(gc.parameter_grid, gc.feature_grid, gc.design_grid),
        dtype=np.float64,
    )  # shape: feature_shape + design_shape + param_shape
    prior_pmf = np.asarray(gc.prior_pmf, dtype=np.float64)
    n_param_axes = prior_pmf.ndim
    param_axes = tuple(range(like.ndim - n_param_axes, like.ndim))
    marginal = (like * prior_pmf).sum(axis=param_axes)
    del like
    print(f"  likelihood -> marginal: reduced axes {param_axes}, marginal {marginal.shape}")

    # marginal shape: (n_y_u, n_y_g, *design_shape). Index the design axes at the
    # nominal design so we compare like with like against the pyro_model samples,
    # which were drawn at that same design.
    feature_names = list(gc.feature_grid.names)
    n_feat_axes = len(feature_names)
    design_names = list(gc.design_grid.names)
    design_idx = []
    for k, name in enumerate(design_names):
        vals = np.asarray(gc.design_grid.axes_in[name], dtype=np.float64).ravel()
        j = int(np.argmin(np.abs(vals - float(nominal[k].cpu()))))
        if not np.isclose(vals[j], float(nominal[k].cpu())):
            raise ValueError(
                f"Nominal {name}={float(nominal[k].cpu())} is not on the design grid "
                f"(nearest {vals[j]}); the comparison would use a different design."
            )
        design_idx.append(j)
    grid_2d = marginal[(slice(None),) * n_feat_axes + tuple(design_idx)]
    print(f"  Marginal shape: {marginal.shape}, grid_2d: {grid_2d.shape}")
    print(f"  Nominal design index: {dict(zip(design_names, design_idx))}")

    # =====================================================================
    # 3.  2D contour comparison
    # =====================================================================
    feature_names = list(gc.feature_grid.names)
    y_u = np.asarray(gc.feature_grid.axes_in[feature_names[0]]).ravel()
    y_g = np.asarray(gc.feature_grid.axes_in[feature_names[1]]).ravel()
    Y_U, Y_G = np.meshgrid(y_u, y_g, indexing="ij")

    # Normalize grid PDF
    grid_2d = np.maximum(grid_2d, 0.0)
    grid_norm = grid_2d.sum()
    if grid_norm > 0:
        grid_2d /= grid_norm

    # pyro_model: 2D histogram on UNIFORM bins (avoid banding from dense grid)
    n_hist_bins = 200
    u_bins = np.linspace(y_u.min(), y_u.max(), n_hist_bins + 1)
    g_bins = np.linspace(y_g.min(), y_g.max(), n_hist_bins + 1)
    hist_2d, _, _ = np.histogram2d(
        pyro_samp[:, 0], pyro_samp[:, 1],
        bins=[u_bins, g_bins],
    )
    y_u_c = 0.5 * (u_bins[:-1] + u_bins[1:])
    y_g_c = 0.5 * (g_bins[:-1] + g_bins[1:])
    Y_U_c, Y_G_c = np.meshgrid(y_u_c, y_g_c, indexing="ij")
    hist_2d = np.maximum(hist_2d, 0.0)
    hist_2d /= hist_2d.sum()

    # Contour levels — ensure strictly increasing for matplotlib
    def ensure_increasing(levels):
        out = sorted(set(levels))
        if len(out) < len(levels):
            out = sorted(set(l + i * 1e-15 for i, l in enumerate(levels)))
        return out

    # Interpolate grid PDF onto uniform histogram grid for fair comparison
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((y_u, y_g), grid_2d, method="linear",
                                     bounds_error=False, fill_value=0.0)
    pts = np.stack([Y_U_c.ravel(), Y_G_c.ravel()], axis=-1)
    grid_on_hist = interp(pts).reshape(Y_U_c.shape)
    grid_on_hist = np.maximum(grid_on_hist, 0.0)
    grid_on_hist /= grid_on_hist.sum()

    hist_levels = ensure_increasing(contour_levels_from_pdf(hist_2d))
    grid_interp_levels = ensure_increasing(contour_levels_from_pdf(grid_on_hist))

    print(f"\n  Histogram contour levels (3σ,2σ,1σ): {hist_levels}")
    print(f"  Grid-interp contour levels (3σ,2σ,1σ): {grid_interp_levels}")
    print(f"  Hist peak: {hist_2d.max():.6e}, Grid-interp peak: {grid_on_hist.max():.6e}")
    print(f"  Hist sum: {hist_2d.sum():.6f}, Grid-interp sum: {grid_on_hist.sum():.6f}")

    du = float(nominal[0].cpu())
    dg = float(nominal[1].cpu())

    # Colors: blue for pyro_model, orange for grid
    blue_colors = ["#c6dbef", "#6baed6", "#2171b5"]    # light/mid/dark blue (3σ/2σ/1σ)
    orange_colors = ["#fdd49e", "#f16913", "#d94801"]   # light/mid/dark orange (3σ/2σ/1σ)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: contour overlay ---
    ax = axes[0]
    ax.contourf(Y_U_c, Y_G_c, hist_2d,
                levels=[hist_levels[0], hist_levels[1], hist_levels[2], hist_2d.max()],
                colors=blue_colors, alpha=0.7)
    ax.contour(Y_U_c, Y_G_c, hist_2d, levels=hist_levels,
               colors=blue_colors, linewidths=1.5)
    ax.contour(Y_U_c, Y_G_c, grid_on_hist, levels=grid_interp_levels,
               colors=orange_colors, linewidths=1.5)
    ax.set_xlim(20, 45)
    ax.set_ylim(18, 30)
    ax.set_xlabel("$y_u$ (mag)", fontsize=12)
    ax.set_ylabel("$y_g$ (mag)", fontsize=12)
    ax.legend(handles=[
        Patch(facecolor="#6baed6", alpha=0.7, label="pyro_model (histogram)"),
        Line2D([0], [0], color="#d94801", linewidth=1.5, label="unnorm_lfunc (grid)"),
    ], fontsize=10, loc="upper right")
    ax.set_title("Contour comparison", fontsize=11)

    # --- Right panel: residual (histogram - grid) ---
    ax2 = axes[1]
    residual = hist_2d - grid_on_hist
    abs_max = max(np.abs(residual.min()), np.abs(residual.max()))
    im = ax2.pcolormesh(y_u_c, y_g_c, residual.T,
                        cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                        shading="auto")
    ax2.set_xlim(20, 45)
    ax2.set_ylim(18, 30)
    ax2.set_xlabel("$y_u$ (mag)", fontsize=12)
    ax2.set_ylabel("$y_g$ (mag)", fontsize=12)
    ax2.set_title("Residual: pyro_model $-$ grid", fontsize=11)
    cb = fig.colorbar(im, ax=ax2, shrink=0.85)
    cb.set_label("$\\Delta P$", fontsize=11)

    # Print residual stats
    print("\n  Residual stats:")
    print(f"    max:  {residual.max():.6e}")
    print(f"    min:  {residual.min():.6e}")
    print(f"    mean: {residual.mean():.6e}")
    print(f"    std:  {residual.std():.6e}")
    print(f"    |residual| / peak(hist): {np.abs(residual).max() / hist_2d.max():.4f}")

    fig.suptitle(
        f"Joint $P(y_u, y_g \\mid d)$ — nominal design ($n_u$={du:.0f}, $n_g$={dg:.0f})\n"
        f"Uniform prior on $z \\in [{z_min:g}, {z_max:g}]$  |  contours at 1$\\sigma$, 2$\\sigma$, 3$\\sigma$",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    out_path = (args.out if os.path.isabs(args.out) or os.path.dirname(args.out)
                else str(plot_path(args.out)))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
