#!/usr/bin/env bash
# Fit EAZY NNLS weights per HEALPix. By default fits all passing candidates;
# set N_MAX to subsample (e.g. N_MAX=600) for faster runs or equal patch counts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESI_DIR="${DESI_DIR:-$HOME/data/desi/tiny_dr1}"
SCRATCH="${SCRATCH:-$HOME/scratch/bedcosmo}"
N_MAX="${N_MAX:-}"
SEED="${SEED:-7}"
Z_MIN="${Z_MIN:-0.01}"
FORCE="${FORCE:-0}"

HEALPIX=(23040 27257 27245 27259 27247 27256 27258 27344 26282)

for hp in "${HEALPIX[@]}"; do
  out="${SCRATCH}/desi_eazy_hp${hp}"
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
  conda run -n sedprior python "${SCRIPT_DIR}/fit_eazy_weights_to_desi.py" \
    --desi-dir "${DESI_DIR}" \
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
conda run -n sedprior python "${SCRIPT_DIR}/compare_healpix_prior_params.py" \
  --scratch-base "${SCRATCH}" \
  --outdir "${SCRATCH}/healpix_prior_comparison" \
  --n-subsample 500 \
  --seed "${SEED}"
