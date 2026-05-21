#!/bin/bash
#SBATCH -A desi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Note: --job-name, --constraint (cpu/gpu), --time, --qos, and --nodes are set by submit.sh via sbatch CLI flags

# Parse named arguments
COSMO_EXP=""
NODE_TYPE="cpu"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --cosmo-exp)
            COSMO_EXP="$2"
            shift 2
            ;;
        --node-type)
            NODE_TYPE="$2"
            shift 2
            ;;
        *)
            # Collect any additional arguments to pass to Python script
            EXTRA_ARGS+=("$1")
            if [[ $2 != --* ]] && [[ -n $2 ]]; then
                EXTRA_ARGS+=("$2")
                shift 2
            else
                shift 1
            fi
            ;;
    esac
done

# Validate required arguments
if [ -z "$COSMO_EXP" ]; then
    echo "Error: --cosmo-exp is required"
    echo "Usage: sbatch grid.sh --cosmo-exp <value> [additional args...]"
    exit 1
fi

# Set log directory based on cosmo_exp
LOG_DIR="${SCRATCH}/bedcosmo/${COSMO_EXP}/logs"
mkdir -p "$LOG_DIR"

# Capture all stdout/stderr in a single log file.
JOB_LOG="${LOG_DIR}/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log"
touch "$JOB_LOG"
exec > >(tee -a "$JOB_LOG") 2>&1

# Print job information and CLI overrides
echo "=========================================="
echo "SLURM Job Started (grid_calc)"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Job Name:     $SLURM_JOB_NAME"
echo "Cosmo Exp:    $COSMO_EXP"
echo "Node Type:    $NODE_TYPE"
echo "Start Time:   $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
if [ -n "$BED_CLI_OVERRIDES" ]; then
    echo "CLI Overrides (non-default values):"
    echo "------------------------------------"
    set -- $BED_CLI_OVERRIDES
    while [ $# -gt 0 ]; do
        if [[ "$1" == --* ]]; then
            if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
                echo "  $1 $2"
                shift 2
            else
                echo "  $1"
                shift 1
            fi
        else
            shift 1
        fi
    done
else
    echo "CLI Overrides: (none)"
fi
echo "=========================================="
echo ""

# Node resource diagnostics (captured in job log)
echo "SLURM mem per node: ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "SLURM cpus per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Node MemTotal:"
grep -i "^MemTotal" /proc/meminfo
echo "free -h:"
free -h
echo ""

# Load conda and activate environment
module load conda
conda activate bedcosmo

if [ "$NODE_TYPE" = "gpu" ]; then
    # Prefer torch's bundled cuBLAS to avoid CUDA version mismatches after NERSC updates
    CUBLAS_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib"
    if [ -d "$CUBLAS_LIB" ]; then
        export LD_LIBRARY_PATH="$CUBLAS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # Let JAX auto-detect the GPU plugin and avoid upfront GPU memory reservation
    unset JAX_PLATFORMS
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    echo "GPU mode: JAX will use CUDA backend"
    nvidia-smi || true
else
    # Force JAX to use the CPU backend so it doesn't try to initialize the
    # bundled CUDA plugin on CPU nodes.
    export JAX_PLATFORMS=cpu
fi

# Background memory logger — prints node RSS every 30s so we can see how
# memory evolves across the design loop and distinguish steady-state usage
# from progressive accumulation.
(
    while true; do
        mem_line=$(free -g | awk '/^Mem:/ {print $3" GiB used, "$7" GiB available"}')
        echo "[mem $(date '+%H:%M:%S')] ${mem_line}"
        sleep 30
    done
) &
MEM_LOGGER_PID=$!
trap "kill $MEM_LOGGER_PID 2>/dev/null" EXIT

export PYTHONUNBUFFERED=1
srun python -u -m bedcosmo.grid_calc \
    "$COSMO_EXP" \
    "${EXTRA_ARGS[@]}"

# Copy the SLURM log into the grid_calc output directory so everything is together
OUT_DIR=$(grep "All outputs saved to:" "$JOB_LOG" | tail -n 1 | sed 's/.*All outputs saved to: //')
if [ -n "$OUT_DIR" ] && [ -d "$OUT_DIR" ]; then
    cp "$JOB_LOG" "$OUT_DIR/slurm_${SLURM_JOB_ID}.log"
    echo "SLURM log copied to: $OUT_DIR/slurm_${SLURM_JOB_ID}.log"
fi
