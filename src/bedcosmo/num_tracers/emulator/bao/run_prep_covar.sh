#!/usr/bin/env bash
# Wrapper script to generate BAO Fisher covariance training data for
# multiple redshift bins.  All bins are saved under a single versioned
# directory as {name}_train.npz / {name}_test.npz.
#
# Usage:
#   bash run_prep_covar.sh                                    # run all bins with defaults
#   bash run_prep_covar.sh --cosmo-model base --n-samples 1000  # specify cosmo model + extra args
#
# The redshift bins below are from DESI DR2 BAO analysis.
# z_eff does NOT have to equal the midpoint of zrange.
# N_tracers ranges are roughly ±50% around DR2 Table 3 "passed" counts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Redshift bins: (name, z_min, z_max, z_eff, ntracers_low, ntracers_high) ──
# N_tracers ranges centered on DR2 passed counts:
#   BGS=1188526, LRG1=1052151, LRG2=1613562,
#   LRG3+ELG1: 1802770+2737573=4540343, ELG2=3797271,
#   QSO=1461588, Lya_QSO=1289874
BINS=(
    "BGS       0.1  0.4  0.295  6e5  1.8e6"
    "LRG1      0.4  0.6  0.510  5e5  1.6e6"
    "LRG2      0.6  0.8  0.706  8e5  2.4e6"
    "LRG3_ELG1 0.8  1.1  0.934  2.3e6  6.8e6"
    "ELG2      1.1  1.6  1.321  1.9e6  5.7e6"
    "QSO       0.8  2.1  1.484  7e5  2.2e6"
    "Lya_QSO   1.8  4.2  2.330  6.5e5  1.9e6"
)

# Parse --cosmo-model from args (default: base_omegak_w_wa), pass rest through.
COSMO_MODEL="base_omegak_w_wa"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cosmo-model)
            COSMO_MODEL="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Determine the next version number once so all bins land in the same directory.
VERSION=$(python -c "
import sys, os
sys.path.insert(0, os.path.abspath('$SCRIPT_DIR/..'))
from util import get_default_save_path, _next_version
save_path = get_default_save_path(analysis='bao', quantity='covar', cosmo_model='$COSMO_MODEL')
print(_next_version(save_path))
")
echo "Cosmo model: ${COSMO_MODEL}"
echo "Using version: v${VERSION}"

for bin in "${BINS[@]}"; do
    read -r NAME Z_MIN Z_MAX Z_EFF NTRACERS_LOW NTRACERS_HIGH <<< "$bin"
    echo ""
    echo "================================================================"
    echo "  $NAME:  zrange=($Z_MIN, $Z_MAX)  z_eff=$Z_EFF  N_tracers=[$NTRACERS_LOW, $NTRACERS_HIGH]"
    echo "================================================================"
    python "$SCRIPT_DIR/prep_covar.py" \
        --name "$NAME" \
        --zrange "$Z_MIN" "$Z_MAX" \
        --z-eff "$Z_EFF" \
        --ntracers-range "$NTRACERS_LOW" "$NTRACERS_HIGH" \
        --version "$VERSION" \
        --cosmo-model "$COSMO_MODEL" \
        "${EXTRA_ARGS[@]}"
done

echo ""
echo "All redshift bins complete (saved to training_data/v${VERSION})."
