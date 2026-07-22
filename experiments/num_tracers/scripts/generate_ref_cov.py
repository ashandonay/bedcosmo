"""Generate a reference covariance from the NumTracers emulators.

Scaling mode (``likelihood_mode='scaling'``) scales a fixed *reference* covariance
by the design. By default that reference is the DESI published ``desi_cov.npy``, but
you can override it via the ``ref_cov`` train_arg. This script builds an alternative
reference straight from the trained covariance emulators: the emulator-predicted
covariance at the NOMINAL DESI design N_tracers and the DESI fiducial cosmology --
exactly the matrix NumTracers caches internally as ``self._sqrtn_ref_cov_cache``
(see ``_sqrtn_fiducial_cov_ref``). Feeding it back through ``ref_cov`` lets scaling
mode scale the emulator's fiducial covariance instead of DESI's published one.

The output ``.npy`` is a (n_data, n_data) matrix in the same desi_data row order as
``desi_cov.npy``, so it is a drop-in ``ref_cov``. Null-checkpoint tracer bins (e.g.
Lya_QSO in base) fall back to their DESI nominal block, matching the emulator path.

Run (in the `bedcosmo` env):
    python experiments/num_tracers/scripts/generate_ref_cov.py --dataset dr1 --cosmo-model base
    # then, in a scaling-mode train_args block:
    #   ref_cov: <printed path>            (absolute)   OR
    #   ref_cov: ref_cov_emulator_dr1_base.npy  (if copied next to desi_cov.npy)
"""
import argparse
import os

import numpy as np

from bedcosmo.util import init_experiment


def _parse_args():
    p = argparse.ArgumentParser(description="Generate an emulator-based reference covariance")
    p.add_argument("--dataset", default="dr1", help="DESI dataset (dr1, dr2, ...)")
    p.add_argument("--cosmo-model", default="base", help="Cosmology model (selects emulator set)")
    p.add_argument("--analysis", default="bao")
    p.add_argument("--prior-args-path", default="prior_args_hrdrag.yaml")
    p.add_argument("--design-args-path", default=None,
                   help="Defaults to design_args_<dataset>.yaml")
    p.add_argument("--apply-desi-syst", action="store_true",
                   help="Inflate sigma_stat -> sigma_tot with DESI's systematic budget")
    p.add_argument("--output", default=None,
                   help="Output .npy path. Default: "
                        "$SCRATCH/bedcosmo/num_tracers/ref_cov/ref_cov_emulator_<dataset>_<model>.npy")
    return p.parse_args()


def main():
    args = _parse_args()
    design_args_path = args.design_args_path or f"design_args_{args.dataset}.yaml"

    exp = init_experiment(
        cosmo_exp="num_tracers",
        prior_args_path=args.prior_args_path,
        design_args_path=design_args_path,
        dataset=args.dataset,
        analysis=args.analysis,
        cosmo_model=args.cosmo_model,
        likelihood_mode="emulator",  # loads the covariance emulators
        include_D_M=True,
        include_D_V=True,
        device="cpu",
        mode="eval",
    )

    # The emulator covariance at nominal-design N and the DESI fiducial cosmology --
    # the matrix cached as self._sqrtn_ref_cov_cache. This is the statistical cov; the
    # DESI-systematic inflation (if requested) is applied on top, matching how the
    # emulator likelihood path composes them.
    cov = exp._sqrtn_fiducial_cov_ref()
    if args.apply_desi_syst:
        exp.apply_desi_syst = True
        cov = exp._maybe_apply_desi_syst(cov)
    cov = cov.detach().cpu().numpy().astype(np.float64)

    # Resolve output path (default under SCRATCH so we never touch the read-only data dir).
    if args.output:
        out_path = os.path.expanduser(args.output)
    else:
        scratch = os.environ.get("SCRATCH", os.path.expanduser("~"))
        out_dir = os.path.join(scratch, "bedcosmo", "num_tracers", "ref_cov")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"ref_cov_emulator_{args.dataset}_{args.cosmo_model}.npy"
        )
    np.save(out_path, cov)

    # Report: per-row emulator sigma vs the DESI published sigma, for a sanity check.
    desi_cov = np.asarray(exp.ref_cov, dtype=np.float64)
    emu_sig = np.sqrt(np.diag(cov))
    desi_sig = np.sqrt(np.diag(desi_cov))
    dd = exp.desi_data
    print(f"\nGenerated emulator reference covariance: shape {cov.shape}, "
          f"apply_desi_syst={args.apply_desi_syst}")
    print(f"{'row':>3} {'tracer':<10}{'quantity':<12}{'emu_sigma':>12}{'DESI_sigma':>12}{'ratio':>8}")
    for r in range(len(dd)):
        print(f"{r:>3} {dd['tracer'].iloc[r]:<10}{dd['quantity'].iloc[r]:<12}"
              f"{emu_sig[r]:>12.5f}{desi_sig[r]:>12.5f}{emu_sig[r]/desi_sig[r]:>8.3f}")
    print(f"\nSaved: {out_path}")
    print("Use in a scaling-mode train_args.yaml block as:")
    print(f"  ref_cov: {out_path}")


if __name__ == "__main__":
    main()
