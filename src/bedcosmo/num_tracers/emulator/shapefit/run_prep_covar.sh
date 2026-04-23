#!/usr/bin/env bash
# Wrapper script to generate ShapeFit Fisher covariance training data for
# multiple redshift bins.  All bins are saved under a single versioned
# directory as {name}_train.npz / {name}_test.npz.
#
# Usage:
#   bash run_prep_covar.sh                   # run all bins with defaults
#   bash run_prep_covar.sh --n-samples 1000  # pass extra args to prep_covar.py
#
# Bins are read from ../tracers.yaml. By default only LRG2 is run; set
# SHAPEFIT_TRACERS to a space-separated list (e.g. "BGS LRG2") or "all".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t BINS < <(python -c "
import os, sys
sys.path.insert(0, os.path.abspath('$SCRIPT_DIR/..'))
from util import TRACER_CONFIGS
raw = os.environ.get('SHAPEFIT_TRACERS', 'LRG2').strip() or 'LRG2'
names = None if raw.lower() == 'all' else [n.upper() for n in raw.split()]
for name, cfg in TRACER_CONFIGS.items():
    if names is not None and name not in names:
        continue
    z_min, z_max = cfg['zrange']
    print(f'{name} {z_min} {z_max} {cfg[\"z_eff\"]} {cfg[\"low\"]} {cfg[\"high\"]}')
")

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
