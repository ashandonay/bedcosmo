#!/bin/bash
#SBATCH -C gpu
#SBATCH -A desi
#SBATCH --job-name=eval
#SBATCH --ntasks-per-node=1     # 1 primary Slurm task per node
#SBATCH --cpus-per-task=32      # CPUs for eval (single GPU)
#SBATCH --gpus-per-node=1       # Single GPU for evaluation
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Note: --time, --qos, and --nodes are set by submit.sh via sbatch CLI flags

# Parse named arguments
COSMO_EXP=""
RUN_ID=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --cosmo_exp)
            COSMO_EXP="$2"
            shift 2
            ;;
        --run_id)
            RUN_ID="$2"
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
    echo "Error: --cosmo_exp is required"
    echo "Usage: sbatch eval.sh --cosmo_exp <value> --run_id <value> [additional args...]"
    exit 1
fi

if [ -z "$RUN_ID" ]; then
    echo "Error: --run_id is required"
    echo "Usage: sbatch eval.sh --cosmo_exp <value> --run_id <value> [additional args...]"
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
echo "SLURM Job Started"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Job Name:     $SLURM_JOB_NAME"
echo "Cosmo Exp:    $COSMO_EXP"
echo "Run ID:       $RUN_ID"
echo "Start Time:   $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
if [ -n "$BED_CLI_OVERRIDES" ]; then
    echo "CLI Overrides (non-default values):"
    echo "------------------------------------"
    # Parse and print key-value pairs
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
    echo "CLI Overrides: (none - using YAML defaults)"
fi
echo "=========================================="
echo ""

# Load conda first, then activate, then other GPU libraries
module load conda
conda activate bed-cosmo

# Load NERSC CUDA and NCCL modules AFTER conda activation
module load nccl/2.21.5 # NERSC NCCL for Slingshot
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export NCCL_NET_GDR_LEVEL=PHB # PCI Host Bridge to use GPUdirect
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn # high-speed network interface
export NCCL_DEBUG=WARN

# PyTorch CUDA memory allocation configuration to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    -m bedcosmo.evaluate \
    --cosmo_exp "$COSMO_EXP" \
    --run_id "$RUN_ID" \
    "${EXTRA_ARGS[@]}"