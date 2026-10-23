"""Generate a scaling-mode reference covariance from a DESI DR1 BAO VAC catalog version.

Scaling mode rescales a fixed reference covariance (``ref_cov``, default ``desi_cov.npy``:
the published DESI 2024 BAO result, built on the v1.2 clustering catalogs). The DR1
``full-shape-bao-clustering`` VAC ships per-tracer post-reconstruction BAO likelihoods
for two catalog versions:

  - ``data/``       v1.5 (DESI's default; the DR1 full-shape + BAO analysis)
  - ``data_v1.2/``  v1.2 (the DR1 BAO analysis)

Each file holds the covariance of the BAO dilation parameters: ``qiso`` for BGS/QSO, and
``(qpar, qper)`` for the anisotropic tracers, i.e. fractional errors on (D_V) or (D_H, D_M).
This script places those fractional covariances on the ``desi_data.csv`` rows,
C_ij = cov_q[a, b] * value_i * value_j, with value the row's ``value_at_z``. Lya QSO has
no VAC BAO likelihood and keeps its block from ``desi_cov.npy``; cross-tracer entries are
zero, as in ``desi_cov.npy``. The output is a drop-in ``ref_cov``.

Run (in the `bedcosmo` env):
    python scripts/num_tracers/generate_ref_cov_vac.py --catalog v1.5
    # then, in a scaling-mode train_args block:
    #   ref_cov: <printed path>
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd

VAC_DIR = "/global/cfs/cdirs/desi/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0"
CATALOG_SUBDIR = {"v1.5": "data", "v1.2": "data_v1.2"}
# desi_data.csv tracer -> VAC file stem
VAC_TRACER = {
    "BGS": "BGS_BRIGHT-21.5_GCcomb_z0.1-0.4",
    "LRG1": "LRG_GCcomb_z0.4-0.6",
    "LRG2": "LRG_GCcomb_z0.6-0.8",
    "LRG3+ELG1": "LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1",
    "ELG2": "ELG_LOPnotqso_GCcomb_z1.1-1.6",
    "QSO": "QSO_GCcomb_z0.8-2.1",
}
# VAC dilation parameter -> desi_data.csv quantity
Q_TO_QUANTITY = {"qiso": "DV_over_rs", "qpar": "DH_over_rs", "qper": "DM_over_rs"}


def _parse_args():
    p = argparse.ArgumentParser(description="Generate a scaling-mode ref_cov from a DR1 BAO VAC version")
    p.add_argument("--catalog", choices=sorted(CATALOG_SUBDIR), default="v1.5")
    p.add_argument("--kind", choices=["syst", "stat-only"], default="syst",
                   help="VAC likelihood with or without DESI's systematic budget")
    p.add_argument("--vac-dir", default=VAC_DIR)
    p.add_argument("--data-dir", default=os.path.expanduser("~/data/desi/bao_dr1"),
                   help="Holds desi_data.csv and desi_cov.npy (the row layout and the Lya block)")
    p.add_argument("--output", default=None,
                   help="Output .npy path. Default: "
                        "$SCRATCH/bedcosmo/num_tracers/ref_cov/ref_cov_vac_<catalog>_<kind>_dr1.npy")
    return p.parse_args()


def read_vac_dilation_cov(path):
    """(parameter names, covariance) of the BAO dilation parameters in one VAC likelihood."""
    with h5py.File(path, "r") as h:
        names = [n.decode() for n in h["covariance/observable/baorecon/labels_values"][...]]
        cov = np.atleast_2d(np.asarray(h["covariance/value"][...], dtype=np.float64))
    if cov.shape != (len(names), len(names)):
        raise ValueError(f"{path}: covariance {cov.shape} does not match parameters {names}")
    return names, cov


def build_ref_cov(desi_data, desi_cov, vac_likelihood_dir, kind):
    """Reference covariance on the desi_data rows from the VAC dilation covariances."""
    cov = np.zeros_like(desi_cov)
    for tracer in desi_data["tracer"].unique():
        rows = desi_data.index[desi_data["tracer"] == tracer].tolist()
        if tracer not in VAC_TRACER:  # Lya QSO: no VAC BAO likelihood
            cov[np.ix_(rows, rows)] = desi_cov[np.ix_(rows, rows)]
            continue
        path = os.path.join(vac_likelihood_dir, f"likelihood_bao-recon_{kind}_{VAC_TRACER[tracer]}.h5")
        names, cov_q = read_vac_dilation_cov(path)
        row_of = {desi_data.loc[r, "quantity"]: r for r in rows}
        quantities = [Q_TO_QUANTITY[n] for n in names]
        if sorted(quantities) != sorted(row_of):
            raise ValueError(f"{tracer}: VAC parameters {names} do not match rows {sorted(row_of)}")
        idx = [row_of[q] for q in quantities]
        values = desi_data.loc[idx, "value_at_z"].to_numpy(dtype=np.float64)
        cov[np.ix_(idx, idx)] = cov_q * np.outer(values, values)
    return cov


def main():
    args = _parse_args()
    desi_data = pd.read_csv(os.path.join(args.data_dir, "desi_data.csv"))
    desi_cov = np.load(os.path.join(args.data_dir, "desi_cov.npy"))
    likelihood_dir = os.path.join(args.vac_dir, CATALOG_SUBDIR[args.catalog], "likelihood")
    cov = build_ref_cov(desi_data, desi_cov, likelihood_dir, args.kind)

    if args.output:
        out_path = os.path.expanduser(args.output)
    else:
        out_dir = os.path.join(os.environ["SCRATCH"], "bedcosmo", "num_tracers", "ref_cov")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ref_cov_vac_{args.catalog}_{args.kind}_dr1.npy")
    np.save(out_path, cov)

    vac_sig, desi_sig = np.sqrt(np.diag(cov)), np.sqrt(np.diag(desi_cov))
    print(f"\nVAC {args.catalog} ({args.kind}) reference covariance from {likelihood_dir}")
    print(f"{'row':>3} {'tracer':<10}{'quantity':<12}{'VAC_sigma':>11}{'DESI_sigma':>12}{'ratio':>8}")
    for r in range(len(desi_data)):
        print(f"{r:>3} {desi_data['tracer'].iloc[r]:<10}{desi_data['quantity'].iloc[r]:<12}"
              f"{vac_sig[r]:>11.5f}{desi_sig[r]:>12.5f}{vac_sig[r] / desi_sig[r]:>8.3f}")
    print(f"\nSaved: {out_path}")
    print("Use in a scaling-mode train_args.yaml block as:")
    print(f"  ref_cov: {out_path}")


if __name__ == "__main__":
    main()
