"""Run BAO Fisher forecasts for multiple N_gal values and plot DH/rd, DM/rd constraint ellipses."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from desilike import Fisher
from desilike.likelihoods.galaxy_clustering import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import (
    CutskyFootprint,
    ObservablesCovarianceMatrix,
    TracerPowerSpectrumMultipolesObservable,
)
from desilike.theories.galaxy_clustering import (
    BAOPowerSpectrumTemplate,
    SimpleBAOWigglesTracerPowerSpectrumMultipoles,
)
from desilike.theories.primordial_cosmology import get_cosmo

warnings.filterwarnings("ignore", message=".*EisensteinHu.*")

# --- Survey and cosmology configuration ---
theta_cosmo = {
    "omega_cdm": 0.1200,
    "omega_b": 0.02237,
    "h": 0.6736,
    "ln10A_s": 3.044,
    "n_s": 0.9649,
}
N_gal_values = [1e6, 2.8e6, 7e6]  # different galaxy counts to compare
zrange = (0.4, 0.6)
area = 14000.0      # survey area in deg^2
b0 = 0.84           # bias normalization at z=0
resolution = 3      # covariance matrix resolution (lower = faster)

# --- Derived quantities at the effective redshift ---
z = np.mean(zrange)
dz = np.diff(zrange)[0]

# Build fiducial cosmology and compute growth quantities
cosmo = get_cosmo(("DESI", dict(theta_cosmo)))
fo = cosmo.get_fourier()

# BAO damping scales and linear bias at the effective redshift
r = 0.5
sigmaper = r * 12.4 * 0.758 * fo.sigma8_z(z, of="delta_cb") / 0.9
f = fo.sigma8_z(z, of="theta_cb") / fo.sigma8_z(z, of="delta_cb")
b1 = b0 * fo.sigma8_cb / fo.sigma8_z(z, of="delta_cb")
params = {
    "b1": b1,
    "sigmapar": (1.0 + f) * sigmaper,
    "sigmaper": sigmaper,
}

print("=== Fiducial point ===")
print(f"  z         = {z}")
print(f"  area      = {area} deg^2")
print(f"  b1        = {b1:.4f}")
print(f"  f         = {f:.4f}")
print(f"  sigmaper  = {sigmaper:.4f} Mpc/h")
print(f"  sigmapar  = {params['sigmapar']:.4f} Mpc/h")

# --- Run Fisher for each N_gal ---
colors = ["steelblue", "indianred", "seagreen", "darkorange", "mediumpurple"]
results = []

for N_gal in N_gal_values:
    print(f"\n--- N_gal = {N_gal:.1e} ---")

    # Survey footprint (volume + shot noise from N_gal and area)
    nbar = N_gal / area  # convert to angular density in deg^-2
    footprint = CutskyFootprint(area=area, zrange=zrange, nbar=nbar, cosmo=cosmo)
    print(f"  P_shot = {footprint.shotnoise:.2f} (Mpc/h)^3")

    # BAOPowerSpectrumTemplate with qpar/qper (alpha_parallel, alpha_perpendicular)
    template = BAOPowerSpectrumTemplate(
        z=z, fiducial=("DESI", dict(theta_cosmo)), apmode="qparqper"
    )
    # SimpleBAOWigglesTracerPowerSpectrumMultipoles: moves only BAO wiggles with
    # scaling parameters, designed for Fisher forecasts (see fisher_desi.ipynb)
    theory = SimpleBAOWigglesTracerPowerSpectrumMultipoles(template=template)

    # Compress theory P(k,mu) into monopole + quadrupole observables
    observable = TracerPowerSpectrumMultipolesObservable(
        data=params,
        klim={0: [0.01, 0.5, 0.01], 2: [0.01, 0.5, 0.01]},
        theory=theory,
    )

    # Gaussian covariance from theory power spectrum + survey footprint
    covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=resolution)

    likelihood = ObservablesGaussianLikelihood(
        observables=observable,
        covariance=covariance(**params),
    )
    # Fix the FoG damping parameter (sigmas): it does not affect BAO wiggles and
    # would create a flat direction in the Fisher matrix, making it singular.
    # This is standard practice for BAO Fisher forecasts (see fisher_desi.ipynb).
    likelihood.all_params['sigmas'].update(fixed=True)

    # Run Fisher forecast
    fisher = Fisher(likelihood)
    fisher_result = fisher(**params)

    # Extract covariance and transform to DH/rd, DM/rd
    param_names = [str(p) for p in fisher_result.names()]
    F_matrix = -np.array(fisher_result._hessian)
    cov = np.linalg.inv(F_matrix)

    # Fiducial BAO distances
    DH_over_rd_fid = float(template.DH_over_rd_fid)
    DM_over_rd_fid = float(template.DM_over_rd_fid)

    # Transform qpar/qper covariance to DH/rd, DM/rd via Jacobian:
    #   DH/rd = qpar * (DH/rd)_fid,  DM/rd = qper * (DM/rd)_fid
    bao_idx = [param_names.index(p) for p in ["qpar", "qper"]]
    cov_q = cov[np.ix_(bao_idx, bao_idx)]
    J = np.diag([DH_over_rd_fid, DM_over_rd_fid])
    cov_dist = J @ cov_q @ J.T

    std_dist = np.sqrt(np.diag(cov_dist))
    print(f"  sigma(DH/rd) = {std_dist[0]:.4f}  ({std_dist[0]/DH_over_rd_fid*100:.2f}%)")
    print(f"  sigma(DM/rd) = {std_dist[1]:.4f}  ({std_dist[1]/DM_over_rd_fid*100:.2f}%)")

    results.append({
        "N_gal": N_gal,
        "cov_dist": cov_dist,
        "std_dist": std_dist,
        "DH_over_rd_fid": DH_over_rd_fid,
        "DM_over_rd_fid": DM_over_rd_fid,
    })

# --- Plot: overlaid 1-sigma ellipses for each N_gal ---
fig, ax = plt.subplots(figsize=(7, 6))

fig.suptitle(
    f"BAO Fisher forecast  |  z_eff={z:.2f}, area={area:.0f} deg$^2$",
    fontsize=10,
)

DH_fid = results[0]["DH_over_rd_fid"]
DM_fid = results[0]["DM_over_rd_fid"]
max_std = 0.0

for i, res in enumerate(results):
    color = colors[i % len(colors)]
    cov_dist = res["cov_dist"]
    eigvals, eigvecs = np.linalg.eigh(cov_dist)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # 1-sigma ellipse with legend label
    w = 2 * np.sqrt(eigvals[0])
    h = 2 * np.sqrt(eigvals[1])
    label = (
        f"$N_{{gal}}={res['N_gal']:.1e}$: "
        f"$\\sigma(D_H/r_d)={res['std_dist'][0]:.3f}$, "
        f"$\\sigma(D_M/r_d)={res['std_dist'][1]:.3f}$"
    )
    ellipse = Ellipse(
        (DH_fid, DM_fid), w, h, angle=angle,
        facecolor=color, alpha=0.3, edgecolor=color, lw=2, label=label,
    )
    ax.add_patch(ellipse)

    max_std = max(max_std, *res["std_dist"])

margin = 5 * max_std
ax.set_xlim(DH_fid - margin, DH_fid + margin)
ax.set_ylim(DM_fid - margin, DM_fid + margin)
ax.set_xlabel(r"$D_H / r_d$")
ax.set_ylabel(r"$D_M / r_d$")
ax.set_title(r"1$\sigma$ constraints: $D_H/r_d$ vs $D_M/r_d$")
ax.axhline(DM_fid, color="gray", ls="--", lw=0.5)
ax.axvline(DH_fid, color="gray", ls="--", lw=0.5)
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
savename = f"fisher_bao_z{z:.1f}_Ngal_comparison.png"
plt.savefig(savename, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {savename}")
plt.show()
