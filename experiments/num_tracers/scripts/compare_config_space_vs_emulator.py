"""Wiring check: NumTracers emulator-likelihood errors vs desilike config-space truth.

The covariance emulators used by ``NumTracers`` (likelihood_mode='emulator') were
trained to reproduce the first-principles config-space (xi) Grieb/Fisher
sigma(D_H/rd, D_M/rd, D_V/rd) produced by ``~/desilike-emulator``'s
``bao/config_space.py``. This script checks that the experiment is wired up
correctly by comparing, at the NOMINAL DESI design + FIDUCIAL cosmology and
matched N_tracers, the per-bin errors from two independent paths:

    NumTracers (this repo, `bedcosmo` env)  -- NN emulator covariance
    desilike   (`emulator` env, on the fly) -- config-space ground truth

If the emulator is wired up and trained well, the two agree to a few percent.
The DESI published `std` is overlaid as a reference (we already expect the
forecast errors to sit below it).

Both pieces live in different conda envs, so the config-space side is computed
by shelling out to ``_gen_config_space_sigmas.py`` under the `emulator` env.

Run (in the `bedcosmo` env):
    python experiments/num_tracers/scripts/compare_config_space_vs_emulator.py
"""
import json
import os
import subprocess
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bedcosmo.util import init_experiment

# DESI 2024 fiducial cosmology (see verify_emulator_likelihood.py).
OM_FID = 0.3152
HRD_PHYS_FID = 99.08          # physical h*r_d
HRD_RAW_FID = HRD_PHYS_FID / 100.0

HERE = os.path.dirname(os.path.abspath(__file__))
EMU_DIR = os.path.expanduser("~/desilike-emulator")
QUANTS = ("DH_over_rs", "DM_over_rs", "DV_over_rs")


def compute_numtracers_side():
    """Per-bin NumTracers emulator errors at nominal design + fiducial cosmology."""
    exp = init_experiment(
        cosmo_exp="num_tracers", prior_args_path="prior_args_hrdrag.yaml",
        design_args_path="design_args_dr1.yaml", dataset="dr1", analysis="bao",
        cosmo_model="base", likelihood_mode="emulator", include_D_M=True,
        include_D_V=True, device="cpu", mode="eval",
    )
    dd = exp.desi_data
    cr = exp.nominal_design.to(torch.float64).view(1, 4)

    params = {
        "Om": torch.tensor([[OM_FID]], dtype=torch.float64),
        "hrdrag": torch.tensor([[HRD_RAW_FID]], dtype=torch.float64),
    }
    # Statistical-only emulator covariance, then DESI-systematic-inflated (the new
    # NumTracers `apply_desi_syst` option). rho is unchanged by the diagonal inflation.
    # passed_ratio drives the emulator N_tracers feed (computed once, reused).
    passed_ratio = exp.calc_passed(cr)
    cov = exp._build_emulator_covariance(passed_ratio, params)[0]
    exp.apply_desi_syst = True
    cov_syst = exp._build_emulator_covariance(passed_ratio, params)[0]
    exp.apply_desi_syst = False
    emu_sigma = torch.sqrt(torch.diag(cov))
    emu_sigma_syst = torch.sqrt(torch.diag(cov_syst))
    desi_std = exp.sigmas.to(torch.float64)
    n_tracers = exp._passed_ratio_to_n_tracers(passed_ratio)
    ref_cov = torch.as_tensor(exp.ref_cov, dtype=torch.float64)  # DESI published cov (default ref)

    # Map each non-fallback emulator bin -> {quantity: (emu, emu_syst, desi)} and rho.
    bins = {}
    for tb in exp._emulators:  # excludes fallback (null-checkpoint) bins
        desi_name = exp._EMULATOR_TRACER_TO_DESI[tb]
        rows = dd.index[dd["tracer"] == desi_name].tolist()
        per_q = {}
        for r in rows:
            q = dd["quantity"].iloc[r]
            per_q[q] = {"emu": emu_sigma[r].item(), "emu_syst": emu_sigma_syst[r].item(),
                        "desi": desi_std[r].item(), "row": r}
        rho_emu = rho_desi = None
        if len(rows) == 2:
            a, b = rows
            rho_emu = (cov[a, b] / torch.sqrt(cov[a, a] * cov[b, b])).item()
            rho_desi = (ref_cov[a, b]
                        / torch.sqrt(ref_cov[a, a] * ref_cov[b, b])).item()
        bins[tb] = {
            "N_tracers": float(n_tracers[tb]),
            "quantities": per_q,
            "rho_emu": rho_emu,
            "rho_desi": rho_desi,
        }
    return bins


def compute_configspace_side(bins, workdir):
    """Shell out to the `emulator` env for config-space ground-truth sigmas."""
    req = {"Om": OM_FID, "hrdrag": HRD_PHYS_FID,
           "bins": {tb: bins[tb]["N_tracers"] for tb in bins}}
    req_path = os.path.join(workdir, "config_req.json")
    out_path = os.path.join(workdir, "config_out.json")
    with open(req_path, "w") as f:
        json.dump(req, f, indent=2)

    gen = os.path.join(HERE, "_gen_config_space_sigmas.py")
    bash = (
        "module load conda >/dev/null 2>&1; "
        "module load cray-mpich-abi >/dev/null 2>&1; "
        "export MPICH_GPU_SUPPORT_ENABLED=0; "
        f"conda run -n emulator python {gen} {req_path} {out_path} {EMU_DIR}"
    )
    print("Running config-space ground truth in the `emulator` env "
          "(loads desilike + DESI bundles; ~1-2 min)...")
    res = subprocess.run(["bash", "-lc", bash], cwd=os.path.join(EMU_DIR, "bao"),
                         capture_output=True, text=True)
    sys.stdout.write(res.stdout)
    if res.returncode != 0 or not os.path.exists(out_path):
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"config-space generator failed (rc={res.returncode}).")
    with open(out_path) as f:
        return json.load(f)


def report_and_plot(nt, cs, workdir):
    labels, gt_v, emu_v, syst_v, desi_v = [], [], [], [], []
    print("\n" + "=" * 104)
    print(f"{'bin':<11}{'quantity':<12}{'config-GT':>12}{'NumTracers':>12}"
          f"{'emu+syst':>12}{'DESI_std':>11}{'syst/emu':>10}{'GT/DESI':>9}")
    print("=" * 104)
    for tb in nt:
        for q in QUANTS:
            if q not in nt[tb]["quantities"]:
                continue
            emu = nt[tb]["quantities"][q]["emu"]
            emu_syst = nt[tb]["quantities"][q]["emu_syst"]
            desi = nt[tb]["quantities"][q]["desi"]
            gt = cs[tb][q]
            labels.append(f"{tb}\n{q.replace('_over_rs','')}")
            gt_v.append(gt); emu_v.append(emu); syst_v.append(emu_syst); desi_v.append(desi)
            print(f"{tb:<11}{q:<12}{gt:>12.5f}{emu:>12.5f}{emu_syst:>12.5f}"
                  f"{desi:>11.5f}{emu_syst/emu:>10.3f}{gt/desi:>9.3f}")

    # Independent convention check: bedcosmo feeds the emulator PASSED N_tracers,
    # and config-space's own fiducial (N_fid, from its _get_ntracers -> 'passed' column)
    # must agree at the nominal design. This catches a future observed-vs-passed regression
    # that the matched-N comparison above would otherwise cancel out.
    print("\nN_tracers convention check  (bedcosmo passed  vs  config-space _get_ntracers):")
    max_rel = 0.0
    for tb in nt:
        n_bed = nt[tb]["N_tracers"]
        n_fid = cs[tb].get("N_fid")
        if n_fid is None:
            continue
        rel = abs(n_bed - n_fid) / n_fid
        max_rel = max(max_rel, rel)
        flag = "" if rel < 1e-3 else "  <-- MISMATCH (observed vs passed?)"
        print(f"             {tb:<11}bedcosmo={n_bed:>12.0f}  config-space={n_fid:>12.0f}"
              f"  rel={rel:.2e}{flag}")
    print(f"             max rel diff = {max_rel:.2e}  "
          f"({'PASS' if max_rel < 1e-3 else 'FAIL'})")

    rho_labels, rho_emu, rho_cs, rho_desi = [], [], [], []
    print("\nrho(DH,DM):  bin        NumTracers   config-GT   DESI")
    for tb in nt:
        if nt[tb]["rho_emu"] is not None and "rho_DH_DM" in cs[tb]:
            rd = nt[tb]["rho_desi"]
            print(f"             {tb:<11}{nt[tb]['rho_emu']:>9.3f}{cs[tb]['rho_DH_DM']:>12.3f}"
                  f"{rd:>9.3f}")
            rho_labels.append(tb)
            rho_emu.append(nt[tb]["rho_emu"])
            rho_cs.append(cs[tb]["rho_DH_DM"])
            rho_desi.append(rd)

    # Bar colors: ground truth blue, emulator orange, emulator+syst green, DESI gray.
    C_GT, C_EMU, C_SYST, C_DESI = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"
    L_GT = "desilike pipeline (ground truth)"
    L_EMU = "NumTracers (emulator)"
    L_SYST = "NumTracers (emulator + syst)"
    L_DESI = "DESI published"

    # --- Plot: sigma magnitudes (top) and rho(DH,DM) (bottom) ---
    x = np.arange(len(labels))
    w = 0.2
    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(max(11, 1.25 * len(labels)), 9),
        gridspec_kw={"height_ratios": [2, 1]})

    ax1.bar(x - 1.5 * w, gt_v, w, label=L_GT, color=C_GT)
    ax1.bar(x - 0.5 * w, emu_v, w, label=L_EMU, color=C_EMU)
    ax1.bar(x + 0.5 * w, syst_v, w, label=L_SYST, color=C_SYST)
    ax1.bar(x + 1.5 * w, desi_v, w, label=L_DESI, color=C_DESI, alpha=0.7)
    ax1.set_ylabel(r"$\sigma$ (D/$r_d$)")
    ax1.set_title("NumTracers emulator errors vs desilike config-space ground truth\n"
                  "(nominal DESI design, fiducial cosmology, matched $N_{tracers}$)")
    ax1.legend()
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # rho(DH,DM) for the anisotropic bins (iso bins have no DH-DM correlation).
    # The diagonal syst inflation leaves rho unchanged, so there is no separate
    # emulator+syst bar here.
    xr = np.arange(len(rho_labels))
    wr = 0.27
    ax3.bar(xr - wr, rho_cs, wr, label=L_GT, color=C_GT)
    ax3.bar(xr, rho_emu, wr, label=L_EMU, color=C_EMU)
    ax3.bar(xr + wr, rho_desi, wr, label=L_DESI, color=C_DESI, alpha=0.7)
    ax3.axhline(0.0, color="k", lw=0.8)
    ax3.set_ylabel(r"$\rho(D_H, D_M)$")
    ax3.set_xticks(xr); ax3.set_xticklabels(rho_labels, fontsize=8)
    ax3.set_ylim(min(rho_emu + rho_cs + rho_desi) - 0.08, 0.05)
    ax3.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(workdir, f"config_space_vs_emulator_{ts}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {out_png}")


def main():
    scratch = os.environ.get("SCRATCH", "/tmp")
    workdir = os.path.join(scratch, "bedcosmo", "num_tracers", "verification")
    os.makedirs(workdir, exist_ok=True)

    nt = compute_numtracers_side()
    cs = compute_configspace_side(nt, workdir)
    report_and_plot(nt, cs, workdir)


if __name__ == "__main__":
    main()
