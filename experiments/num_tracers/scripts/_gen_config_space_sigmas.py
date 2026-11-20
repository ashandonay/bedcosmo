"""Config-space ground-truth sigma generator (runs in the `emulator` conda env).

Standalone helper for ``compare_config_space_vs_emulator.py``. It imports the
tracer-forecast-emulator config-space pipeline
(``tracer_forecast_emulator.bao.config_space``) and, for each
requested tracer bin, evaluates the first-principles Grieb/Fisher
sigma(D_H/rd, D_M/rd, D_V/rd) at the supplied N_tracers and cosmology -- the
exact quantity the NN covariance emulator was trained to reproduce.

This MUST run in the `emulator` env (it needs `desilike`); the orchestrator
shells out to it with the cray-mpich-abi module loaded and
MPICH_GPU_SUPPORT_ENABLED=0. It talks to the orchestrator over two JSON files:

    request.json : {"Om": float, "hrdrag": float, "bins": {bin: N_tracers, ...}}
    out.json     : {bin: {"apmode", "DH_over_rs", "DM_over_rs", "DV_over_rs",
                          "rho_DH_DM"(aniso only), "N_tracers"}, ...}

`hrdrag` is PHYSICAL h*r_d (e.g. 99.08); N_tracers are absolute counts.

Usage:  python _gen_config_space_sigmas.py <request.json> <out.json> [emulator_checkout]

`emulator_checkout`, if given, is a tracer-forecast-emulator checkout whose
``src/`` is imported ahead of the installed package; omitted, the installed
``tracer_forecast_emulator`` is used.
"""
import json
import os
import sys


def main():
    req_path, out_path = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        sys.path.insert(0, os.path.join(os.path.abspath(sys.argv[3]), "src"))

    # heavy: desilike + DESI bundles
    from tracer_forecast_emulator.bao import config_space

    with open(req_path) as f:
        req = json.load(f)
    Om = float(req["Om"])
    hrdrag = float(req["hrdrag"])  # physical h*r_d
    bins = req["bins"]

    out = {}
    for tracer, n_tracers in bins.items():
        gen = config_space.XiSigmaGenerator(tracer)
        s = gen.sigma_triplet(N_tracers=float(n_tracers), Om=Om, hrdrag=hrdrag)
        rec = {
            "apmode": s["apmode"],
            "DH_over_rs": float(s["DH_over_rs"]),
            "DM_over_rs": float(s["DM_over_rs"]),
            "DV_over_rs": float(s["DV_over_rs"]),
            "N_tracers": float(n_tracers),
            "N_fid": float(gen._N_fid),
        }
        if "rho_DH_DM" in s:
            rec["rho_DH_DM"] = float(s["rho_DH_DM"])
        out[tracer] = rec
        print(f"  {tracer:<10} apmode={rec['apmode']:<9} "
              f"DH={rec['DH_over_rs']:.5f} DM={rec['DM_over_rs']:.5f} "
              f"DV={rec['DV_over_rs']:.5f} N={n_tracers:.0f}", flush=True)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote config-space sigmas to {out_path}")


if __name__ == "__main__":
    main()
