#!/bin/bash
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -A desi
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 primary Slurm task per node
#SBATCH --cpus-per-task=32     # CPUs for all DDP workers on the node (e.g., 4 workers * 32 cpus/worker)
#SBATCH --gpus-per-node=1       # Number of GPUs to request per node
#SBATCH --time=2:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

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