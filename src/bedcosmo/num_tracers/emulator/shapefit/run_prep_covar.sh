#!/usr/bin/env bash
# Wrapper script to generate ShapeFit Fisher covariance training data for
# multiple redshift bins.  All bins are saved under a single versioned
# directory as {name}_train.npz / {name}_test.npz.
#
# Usage:
#   bash run_prep_covar.sh                   # run all bins with defaults
#   bash run_prep_covar.sh --n-samples 1000  # pass extra args to prep_covar.py
#
# Uses the same DESI DR2 redshift bins and N_tracers ranges as the BAO wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Redshift bins: (name, z_min, z_max, z_eff, ntracers_low, ntracers_high) ──
# N_tracers ranges centered on DR2 passed counts:
#   BGS=1188526, LRG1=1052151, LRG2=1613562,
#   LRG3+ELG1: 1802770+2737573=4540343, ELG2=3797271,
#   QSO=1461588, Lya_QSO=1289874
BINS=(
    "LRG2      0.6  0.8  0.706  8e5  2.4e6"
)

# Extra arguments forwarded to prep_covar.py (e.g. --n-samples 2000)
EXTRA_ARGS=("$@")

# Determine the next version number once so all bins land in the same directory.
VERSION=$(python -c "
import sys, os
sys.path.insert(0, os.path.abspath('$SCRIPT_DIR/..'))
from util import get_default_save_path, _next_version
save_path = get_default_save_path(analysis='shapefit', quantity='covar')
print(_next_version(save_path))
")
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
        "${EXTRA_ARGS[@]}"
done

echo ""
echo "All redshift bins complete (saved to training_data/v${VERSION})."
