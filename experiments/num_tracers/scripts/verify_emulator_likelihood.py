"""Basic verification of the emulator-mode likelihood in NumTracers.

Most direct sanity check: drive the emulator likelihood at the NOMINAL DESI
design (class ratios) and the DESI FIDUCIAL cosmology, then compare the
resulting data values + errors against the real DESI measurements:

    means  (D_H/D_M/D_V at fiducial)  vs  desi_data['value_at_z']
    sqrt(diag(cov)) from emulators     vs  desi_data['std']
    full emulator covariance           vs  desi_cov.npy (nominal)

At nominal design the per-bin N_tracers == the DESI passed (redshift-confirmed)
counts the emulators were trained around (desilike_emulator _get_ntracers reads
the 'passed' column), so a well-behaved emulator likelihood should reproduce the
DESI errors closely. Bins with a null checkpoint (Lya_QSO in base) fall back to
the fixed DESI nominal covariance by construction.

Run:  python experiments/num_tracers/scripts/verify_emulator_likelihood.py
"""
import numpy as np
import torch

from bedcosmo.util import init_experiment

# DESI 2024 fiducial cosmology (Planck18 base): Omega_m and h*r_d.
#   omega_b=0.02237, omega_cdm=0.1200, mnu=0.06 -> Omega_m ~ 0.3152
#   h=0.6736, r_d~147.09 Mpc -> h*r_d ~ 99.08
# The bedcosmo 'hrdrag' raw param is h*r_d / 100 (hrdrag_multiplier=10000 ->
# hrdrag_phys = raw * 10000/100 = raw * 100).
OM_FID = 0.3152
HRD_PHYS_FID = 99.08
HRD_RAW_FID = HRD_PHYS_FID / 100.0

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    torch.set_printoptions(linewidth=200, sci_mode=False)

    exp = init_experiment(
        cosmo_exp="num_tracers",
        prior_args_path="prior_args_hrdrag.yaml",
        design_args_path="design_args_dr1.yaml",
        dataset="dr1",
        analysis="bao",
        cosmo_model="base",
        likelihood_mode="emulator",
        include_D_M=True,
        include_D_V=True,
        device=DEVICE,
        mode="eval",
    )
    print(f"\nDevice: {DEVICE}")
    print(f"Loaded emulators: {list(exp._emulators.keys())}")
    print(f"Fallback (null-checkpoint) bins: {exp._emulator_fallback_bins}")

    dd = exp.desi_data
    n = len(dd)

    # --- Nominal design = nominal class ratios [BGS, LRG, ELG, QSO] (sum to 1) ---
    class_ratio = exp.nominal_design.to(torch.float64).view(1, 4)
    print(f"\nNominal design (class ratios {list(exp.design_labels)}): "
          f"{class_ratio.squeeze().cpu().numpy()}  (sum={class_ratio.sum().item():.4f})")

    # --- Fiducial cosmology parameters dict (shape (1,1) so squeeze(-1) works) ---
    params = {
        "Om": torch.tensor([[OM_FID]], device=DEVICE, dtype=torch.float64),
        "hrdrag": torch.tensor([[HRD_RAW_FID]], device=DEVICE, dtype=torch.float64),
    }

    # --- Means: same D_H/D_M/D_V computation the pyro_model emulator branch uses ---
    means = torch.zeros(n, device=DEVICE, dtype=torch.float64)
    z_dh = torch.tensor(dd[dd["quantity"] == "DH_over_rs"]["z"].to_list(), device=DEVICE)
    z_dm = torch.tensor(dd[dd["quantity"] == "DM_over_rs"]["z"].to_list(), device=DEVICE)
    z_dv = torch.tensor(dd[dd["quantity"] == "DV_over_rs"]["z"].to_list(), device=DEVICE)
    means[exp.DH_idx] = exp.D_H_func(z_dh, Om=OM_FID, hrdrag=HRD_RAW_FID).flatten()
    means[exp.DM_idx] = exp.D_M_func(z_dm, Om=OM_FID, hrdrag=HRD_RAW_FID).flatten()
    means[exp.DV_idx] = exp.D_V_func(z_dv, Om=OM_FID, hrdrag=HRD_RAW_FID).flatten()

    # --- Covariance: from the emulators (passed_ratio drives the N_tracers feed) ---
    passed_ratio = exp.calc_passed(class_ratio)
    cov = exp._build_emulator_covariance(passed_ratio, params)[0]  # (n, n)
    emu_sigma = torch.sqrt(torch.diag(cov))

    # Show the emulator inputs we feed, relative to each checkpoint's training center.
    n_tracers = exp._passed_ratio_to_n_tracers(passed_ratio)
    print("\nEmulator inputs at nominal design (value | normalized vs training x_mu/x_sigma):")
    print(f"  feed Om={OM_FID}, hrdrag_phys={HRD_PHYS_FID}")
    for tb, emu in exp._emulators.items():
        nt = n_tracers[tb].item()
        xmu, xsig = emu["x_mu"].flatten(), emu["x_sigma"].flatten()
        z_nt = (nt - xmu[0].item()) / xsig[0].item()
        z_om = (OM_FID - xmu[1].item()) / xsig[1].item()
        z_hr = (HRD_PHYS_FID - xmu[2].item()) / xsig[2].item()
        print(f"  {tb:<10} N_tracers={nt:>10.0f} (z={z_nt:+.2f})  "
              f"Om z={z_om:+.2f}  hrdrag z={z_hr:+.2f}")

    # --- References (real DESI measurements) ---
    central = exp.central_val.to(torch.float64)
    desi_sigma = exp.sigmas.to(torch.float64)
    ref_cov = torch.as_tensor(exp.ref_cov, device=DEVICE, dtype=torch.float64)

    # ---------------------------- Report ----------------------------
    print("\n" + "=" * 100)
    print("MEANS  (emulator-likelihood mean at fiducial cosmology)  vs  DESI value_at_z")
    print("=" * 100)
    print(f"{'tracer':<12}{'quantity':<12}{'z':>7}{'mean_fid':>13}{'DESI_val':>13}{'ratio':>9}{'%diff':>9}")
    for i in range(n):
        r = means[i].item() / central[i].item()
        print(f"{dd['tracer'].iloc[i]:<12}{dd['quantity'].iloc[i]:<12}{dd['z'].iloc[i]:>7.3f}"
              f"{means[i].item():>13.4f}{central[i].item():>13.4f}{r:>9.4f}{(r-1)*100:>8.2f}%")

    print("\n" + "=" * 100)
    print("ERRORS  sqrt(diag) emulator covariance  vs  DESI std")
    print("=" * 100)
    print(f"{'tracer':<12}{'quantity':<12}{'emu_sigma':>13}{'DESI_std':>13}{'ratio':>9}{'%diff':>9}  note")
    for i in range(n):
        r = emu_sigma[i].item() / desi_sigma[i].item()
        tb = None
        for k, v in exp._EMULATOR_TRACER_TO_DESI.items():
            if v == dd["tracer"].iloc[i]:
                tb = k
        note = "FALLBACK" if tb in exp._emulator_fallback_bins else "emulator"
        print(f"{dd['tracer'].iloc[i]:<12}{dd['quantity'].iloc[i]:<12}"
              f"{emu_sigma[i].item():>13.5f}{desi_sigma[i].item():>13.5f}{r:>9.4f}{(r-1)*100:>8.2f}%  {note}")

    # Correlations within anisotropic 2x2 blocks (DH-DM) from the emulator.
    print("\n" + "=" * 100)
    print("CORRELATIONS  rho(DH,DM) emulator  vs  DESI nominal covariance")
    print("=" * 100)
    for tracer in dd["tracer"].unique():
        rows = dd.index[dd["tracer"] == tracer].tolist()
        if len(rows) != 2:
            continue
        a, b = rows
        rho_emu = cov[a, b] / torch.sqrt(cov[a, a] * cov[b, b])
        rho_desi = ref_cov[a, b] / torch.sqrt(ref_cov[a, a] * ref_cov[b, b])
        print(f"{tracer:<12} rho_emu={rho_emu.item():>8.4f}   rho_DESI={rho_desi.item():>8.4f}")

    # Aggregate diagnostics.
    sig_ratio = (emu_sigma / desi_sigma).cpu().numpy()
    mean_ratio = (means / central).cpu().numpy()
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"mean ratio   : min={mean_ratio.min():.4f}  max={mean_ratio.max():.4f}  "
          f"median={np.median(mean_ratio):.4f}")
    print(f"sigma ratio  : min={sig_ratio.min():.4f}  max={sig_ratio.max():.4f}  "
          f"median={np.median(sig_ratio):.4f}")

    # --- Draw data samples from the likelihood to confirm it samples cleanly ---
    # (sample on CPU: batched cuSOLVER triangular-solve is flaky on login-node GPUs)
    cov_stab = exp._stabilize_covariance(cov.unsqueeze(0))[0].cpu()
    mvn = torch.distributions.MultivariateNormal(means.cpu(), cov_stab)
    draws = mvn.sample((20000,))
    means = means.cpu()
    emu_sigma = emu_sigma.cpu()
    emp_mean = draws.mean(0)
    emp_std = draws.std(0)
    print("\nSampled 20000 draws from MultivariateNormal(means, emulator_cov):")
    print(f"  empirical mean matches means : max|Δ|={torch.max(torch.abs(emp_mean-means)).item():.4e}")
    print(f"  empirical std  matches sqrt(diag): "
          f"max ratio dev={torch.max(torch.abs(emp_std/emu_sigma - 1)).item():.4f}")
    print(f"  all finite: {torch.isfinite(draws).all().item()}")


if __name__ == "__main__":
    main()
