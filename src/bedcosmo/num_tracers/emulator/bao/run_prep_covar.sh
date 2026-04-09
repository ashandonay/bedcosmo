#!/usr/bin/env bash
# Wrapper script to generate BAO Fisher covariance training data for
# multiple redshift bins.  All bins are saved under a single versioned
# directory as {name}_train.npz / {name}_test.npz.
#
# Usage:
#   bash run_prep_covar.sh                                    # run all bins with defaults
#   bash run_prep_covar.sh --cosmo-model base --n-samples 1000  # specify cosmo model + extra args
#
# Redshift bins and N_tracers ranges come from ../tracers.yaml (via util.TRACER_CONFIGS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t BINS < <(python -c "
import sys, os
sys.path.insert(0, os.path.abspath('$SCRIPT_DIR/..'))
from util import TRACER_CONFIGS
for name, cfg in TRACER_CONFIGS.items():
    z_min, z_max = cfg['zrange']
    print(f'{name} {z_min} {z_max} {cfg[\"z_eff\"]} {cfg[\"low\"]} {cfg[\"high\"]}')
")

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
        --tracer-bin "$NAME" \
        --zrange "$Z_MIN" "$Z_MAX" \
        --z-eff "$Z_EFF" \
        --ntracers-range "$NTRACERS_LOW" "$NTRACERS_HIGH" \
        --version "$VERSION" \
        --cosmo-model "$COSMO_MODEL" \
        "${EXTRA_ARGS[@]}"
done

echo ""
echo "All redshift bins complete (saved to training_data/v${VERSION})."
