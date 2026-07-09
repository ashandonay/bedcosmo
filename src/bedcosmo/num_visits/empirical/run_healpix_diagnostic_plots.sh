#!/usr/bin/env bash
# Per-HEALPix fit diagnostic plots (--plot-only; see build_prior.py for full pipeline).
set -euo pipefail

BUILD_NAME="${BUILD_NAME:-empirical_prior}"
NUM_VISITS_SCRATCH="${NUM_VISITS_SCRATCH:-${SCRATCH:+${SCRATCH}/bedcosmo/num_visits}}"
NUM_VISITS_SCRATCH="${NUM_VISITS_SCRATCH:-${HOME}/scratch/bedcosmo/num_visits}"
PRIOR_DIR="${NUM_VISITS_SCRATCH}/${BUILD_NAME}"
HEALPIX_DIR="${PRIOR_DIR}/healpix"
SEED="${SEED:-7}"
PLOT_N_EXAMPLES="${PLOT_N_EXAMPLES:-8}"
PLOT_TOP_OUTLIERS="${PLOT_TOP_OUTLIERS:-5}"
# Template-bank .param; set for non-default banks, e.g.
# TEMPLATE_PARAM=templates/eazy_v1.0.spectra.param
TEMPLATE_PARAM="${TEMPLATE_PARAM:-}"
PY="conda run -n bedcosmo python -m bedcosmo.num_visits.empirical"

# Optional override; omit to use Python default ($SCRATCH/bedcosmo/desi/tiny_dr1).
DESI_DIR_ARGS=()
if [[ -n "${DESI_DIR:-}" ]]; then
  DESI_DIR_ARGS=(--desi-dir "${DESI_DIR}")
fi
TEMPLATE_PARAM_ARGS=()
if [[ -n "${TEMPLATE_PARAM}" ]]; then
  TEMPLATE_PARAM_ARGS=(--template-param "${TEMPLATE_PARAM}")
fi

HEALPIX=(23040 27257 27245 27259 27247 27256 27258 27344 26282)

for hp in "${HEALPIX[@]}"; do
  out="${HEALPIX_DIR}/hp${hp}"
  csv="${out}/desi_eazy_empirical_weights.csv"
  if [[ ! -f "$csv" ]]; then
    echo "HEALPIX ${hp}: missing ${csv}; skip"
    continue
  fi
  echo "=== Diagnostic plots HEALPIX ${hp} ==="
  ${PY}.fit_eazy_weights_to_desi \
    --plot-only \
    --build-name "${BUILD_NAME}" \
    --healpix "${hp}" \
    --outdir "${out}" \
    "${DESI_DIR_ARGS[@]}" \
    "${TEMPLATE_PARAM_ARGS[@]}" \
    --plot-n-examples "${PLOT_N_EXAMPLES}" \
    --plot-top-outliers "${PLOT_TOP_OUTLIERS}" \
    --seed "${SEED}"
done

echo "=== Comparison plots ==="
${PY}.compare_healpix_prior_params \
  --build-name "${BUILD_NAME}" \
  --prior-dir "${PRIOR_DIR}" \
  --outdir "${PRIOR_DIR}/healpix_prior_comparison" \
  --n-subsample 500 \
  --seed "${SEED}"
