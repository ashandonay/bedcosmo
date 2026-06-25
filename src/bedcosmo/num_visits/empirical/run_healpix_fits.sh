#!/usr/bin/env bash
# Fit EAZY NNLS weights per HEALPix (see build_prior.py for full pipeline).
set -euo pipefail

BUILD_NAME="${BUILD_NAME:-empirical_prior}"
NUM_VISITS_SCRATCH="${NUM_VISITS_SCRATCH:-${SCRATCH:+${SCRATCH}/bedcosmo/num_visits}}"
NUM_VISITS_SCRATCH="${NUM_VISITS_SCRATCH:-${HOME}/scratch/bedcosmo/num_visits}"
PRIOR_DIR="${NUM_VISITS_SCRATCH}/${BUILD_NAME}"
HEALPIX_DIR="${PRIOR_DIR}/healpix"
N_MAX="${N_MAX:-}"
SEED="${SEED:-7}"
Z_MIN="${Z_MIN:-0.01}"
FORCE="${FORCE:-0}"
PY="conda run -n bedcosmo python -m bedcosmo.num_visits.empirical"

HEALPIX=(23040 27257 27245 27259 27247 27256 27258 27344 26282)

for hp in "${HEALPIX[@]}"; do
  out="${HEALPIX_DIR}/hp${hp}"
  csv="${out}/desi_eazy_empirical_weights.csv"
  if [[ "$FORCE" != "1" && -f "$csv" ]]; then
    echo "HEALPIX ${hp}: already exists ${csv} (set FORCE=1 to refit)"
    continue
  fi
  if [[ -n "${N_MAX}" ]]; then
    echo "=== Fitting HEALPIX ${hp} (n-max=${N_MAX}) ==="
  else
    echo "=== Fitting HEALPIX ${hp} (all passing candidates) ==="
  fi
  n_max_args=()
  if [[ -n "${N_MAX}" ]]; then
    n_max_args=(--n-max "${N_MAX}")
  fi
  ${PY}.fit_eazy_weights_to_desi \
    --build-name "${BUILD_NAME}" \
    --healpix "${hp}" \
    --fit-method nnls \
    "${n_max_args[@]}" \
    --z-min "${Z_MIN}" \
    --seed "${SEED}" \
    --plot-n-examples 0 \
    --no-triangle-plots \
    --no-raw-coeff-triangle \
    --outdir "${out}"
done

echo "=== Comparison plots ==="
${PY}.compare_healpix_prior_params \
  --build-name "${BUILD_NAME}" \
  --prior-dir "${PRIOR_DIR}" \
  --outdir "${PRIOR_DIR}/healpix_prior_comparison" \
  --n-subsample 500 \
  --seed "${SEED}"
