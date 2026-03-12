"""Run the ShapeFit Fisher pipeline for a single set of parameters and print the full Fisher matrix."""

import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from desilike import Fisher
from desilike.likelihoods.galaxy_clustering import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import (
    CutskyFootprint,
    ObservablesCovarianceMatrix,
    TracerPowerSpectrumMultipolesObservable,
)
from desilike.theories.galaxy_clustering import (
    KaiserTracerPowerSpectrumMultipoles,
    ShapeFitPowerSpectrumTemplate,
)
from desilike.theories.primordial_cosmology import get_cosmo

warnings.filterwarnings("ignore", message=".*EisensteinHu.*")

# --- Single fiducial cosmology (DESI-like) ---
theta_cosmo = {
    "omega_cdm": 0.1200,
    "omega_b": 0.02237,
    "h": 0.6736,
    "ln10A_s": 3.044,
    "n_s": 0.9649,
}
nbar_list = [1000.0, 5000.0]  # deg^-2 per unit redshift
zrange = (0.4, 0.6)
area = 14000.0      # deg^2
b0 = 0.84
resolution = 3

# --- Build the pipeline (shared across nbar values) ---
z = np.mean(zrange)
dz = np.diff(zrange)[0]

cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
fo = cosmo.get_fourier()

r = 0.5
sigmaper = r * 12.4 * 0.758 * fo.sigma8_z(z, of="delta_cb") / 0.9
f = fo.sigma8_z(z, of="theta_cb") / fo.sigma8_z(z, of="delta_cb")
b1 = b0 * fo.sigma8_cb / fo.sigma8_z(z, of="delta_cb")
params = {
    "b1": b1,
    "sigmapar": (1.0 + f) * sigmaper,
    "sigmaper": sigmaper,
}

template = ShapeFitPowerSpectrumTemplate(
    z=z, fiducial=("DESI", dict(theta_cosmo)), apmode="qisoqap"
)
theory = KaiserTracerPowerSpectrumMultipoles(template=template)

observable = TracerPowerSpectrumMultipolesObservable(
    data=params,
    klim={0: [0.01, 0.5, 0.01], 2: [0.01, 0.5, 0.01]},
    theory=theory,
)

# Transform setup (shared)
f_sigmar_fid = float(template.f_sigmar_fid)
m_fid = float(template.m_fid)
J = np.diag([1.0, 1.0, 1.0, f_sigmar_fid])
phys_names = ["qiso", "qap", "m", "f_sigmar"]
shapefit_names = ["qiso", "qap", "dm", "df"]

# --- Run Fisher for each nbar ---
cov_phys_dict = {}

for nbar in nbar_list:
    print(f"\n{'='*60}")
    print(f"=== nbar = {nbar:.0f} deg^-2 / dz ===")
    print(f"{'='*60}")

    print(f"  z         = {z}")
    print(f"  area      = {area} deg^2")
    print(f"  b1        = {b1:.4f}")
    print(f"  f         = {f:.4f}")
    print(f"  sigmaper  = {sigmaper:.4f} Mpc/h")
    print(f"  sigmapar  = {params['sigmapar']:.4f} Mpc/h")

    footprint = CutskyFootprint(area=area, zrange=zrange, nbar=nbar * dz, cosmo=cosmo)
    print(f"  V_survey  = {footprint.volume:.2e} (Mpc/h)^3")
    print(f"  P_shot    = {footprint.shotnoise:.2f} (Mpc/h)^3")

    covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=resolution)

    likelihood = ObservablesGaussianLikelihood(
        observables=observable,
        covariance=covariance(**params),
    )

    fisher = Fisher(likelihood)
    fisher_result = fisher(**params)

    param_names = [str(p) for p in fisher_result.names()]
    n = len(param_names)

    print(f"\n  Fisher matrix ({n}x{n}), parameters: {param_names}")

    F = np.array(fisher_result._hessian)
    F_matrix = -F

    cov = np.linalg.inv(F_matrix)

    sf_idx = [param_names.index(p) for p in shapefit_names]
    cov_sf = cov[np.ix_(sf_idx, sf_idx)]
    cov_phys = J @ cov_sf @ J.T
    std_phys = np.sqrt(np.diag(cov_phys))

    print(f"\n  Physical ShapeFit covariance (marginalized over b1, sn0):")
    print(f"  f_sigmar_fid = {f_sigmar_fid:.6f},  m_fid = {m_fid:.6f}")
    phys_header = "  " + "".ljust(12) + "".join(p.rjust(14) for p in phys_names)
    print(phys_header)
    for i, name in enumerate(phys_names):
        row = "  " + name.ljust(12) + "".join(f"{cov_phys[i, j]:14.6f}" for j in range(4))
        print(row)

    print(f"\n  Marginalized 1-sigma errors:")
    for i, name in enumerate(phys_names):
        print(f"    sigma({name:>10s}) = {std_phys[i]:.6f}")

    cov_phys_dict[nbar] = cov_phys

# --- Plot: two covariance grids stacked vertically, shared colorbar ---
n_sf = len(phys_names)

# Determine shared symmetric-log color limits
all_vals = np.concatenate([c.ravel() for c in cov_phys_dict.values()])
vmax = np.max(np.abs(all_vals))
linthresh = np.min(np.abs(all_vals[all_vals != 0])) * 0.5
norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)

fig, axes = plt.subplots(2, 1, figsize=(7, 9), constrained_layout=True)
fig.suptitle(
    f"ShapeFit Fisher forecast  |  z_eff={z:.2f}",
    fontsize=12,
)

for ax, nbar in zip(axes, nbar_list):
    cov_phys = cov_phys_dict[nbar]
    im = ax.imshow(cov_phys, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(n_sf))
    ax.set_xticklabels(phys_names, rotation=45, ha="right")
    ax.set_yticks(range(n_sf))
    ax.set_yticklabels(phys_names)
    ax.set_title(f"nbar = {nbar:.0f} deg$^{{-2}}$/dz")
    for i in range(n_sf):
        for j in range(n_sf):
            ax.text(j, i, f"{cov_phys[i, j]:.1e}", ha="center", va="center", fontsize=8)

fig.colorbar(im, ax=axes, shrink=0.6, label="Covariance", pad=0.02)

savename = f"fisher_z{z:.1f}_nbar_comparison.png"
plt.savefig(savename, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {savename}")
plt.show()
