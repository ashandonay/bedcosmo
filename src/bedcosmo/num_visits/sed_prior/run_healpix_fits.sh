#!/usr/bin/env bash
# Fit EAZY NNLS weights per HEALPix (matched --n-max for fair cross-patch comparison).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESI_DIR="${DESI_DIR:-$HOME/data/desi/tiny_dr1}"
SCRATCH="${SCRATCH:-$HOME/scratch/bedcosmo}"
N_MAX="${N_MAX:-600}"
SEED="${SEED:-7}"

HEALPIX=(23040 27257 27245 27259 27247 27256 27258 27344 26282)

for hp in "${HEALPIX[@]}"; do
  out="${SCRATCH}/desi_eazy_hp${hp}"
  csv="${out}/desi_eazy_empirical_weights.csv"
  if [[ "$hp" == "23040" && -f "${SCRATCH}/desi_eazy_empirical_prior_nnls/desi_eazy_empirical_weights.csv" ]]; then
    echo "HEALPIX ${hp}: using existing ${SCRATCH}/desi_eazy_empirical_prior_nnls"
    continue
  fi
  if [[ -f "$csv" ]]; then
    echo "HEALPIX ${hp}: already exists ${csv}"
    continue
  fi
  echo "=== Fitting HEALPIX ${hp} (n-max=${N_MAX}) ==="
  conda run -n sedprior python "${SCRIPT_DIR}/fit_eazy_weights_to_desi.py" \
    --desi-dir "${DESI_DIR}" \
    --healpix "${hp}" \
    --fit-method nnls \
    --n-max "${N_MAX}" \
    --seed "${SEED}" \
    --plot-n-examples 0 \
    --no-triangle-plots \
    --no-raw-coeff-triangle \
    --outdir "${out}"
done

echo "=== Comparison plots ==="
conda run -n sedprior python "${SCRIPT_DIR}/compare_healpix_prior_params.py" \
  --scratch-base "${SCRATCH}" \
  --outdir "${SCRATCH}/healpix_prior_comparison" \
  --n-subsample 500 \
  --seed "${SEED}"
