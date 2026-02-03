#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -A desi
#SBATCH --job-name=resume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 primary Slurm task per node
#SBATCH --cpus-per-task=128     # CPUs for all DDP workers on the node (e.g., 4 workers * 32 cpus/worker)
#SBATCH --gpus-per-node=4       # Number of GPUs to request per node
#SBATCH --time=03:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashandon@uci.edu
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

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

# Define number of DDP processes per node
NPROC_PER_NODE=$SLURM_GPUS_PER_NODE

# Set environment variables for distributed training
# WORLD_SIZE is total number of DDP processes across all nodes
export WORLD_SIZE=$(($SLURM_NNODES * $NPROC_PER_NODE))

# Get the IP address of the first node (master node) using SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 10000-60000 -n 1)

# Additional environment variables for DDP
export OMP_NUM_THREADS=$((128 / $NPROC_PER_NODE)) # Example: Distribute cpus_per_task among OMP threads if needed, or set to a fixed val like 4 or 8

# Tell CUDA to order GPUs by PCI_BUS_ID for consistency
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# The srun command will launch one instance of torch.distributed.run per node.
# torch.distributed.run will then spawn NPROC_PER_NODE worker processes on each node.
# SLURM_PROCID can be used for node_rank as srun launches one task per node here.

# Parse named arguments
COSMO_EXP=""
RESUME_ID=""
RESUME_STEP=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --cosmo_exp)
            COSMO_EXP="$2"
            shift 2
            ;;
        --resume_id)
            RESUME_ID="$2"
            shift 2
            ;;
        --resume_step)
            RESUME_STEP="$2"
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
    echo "Usage: sbatch resume.sh --cosmo_exp <value> --resume_id <value> --resume_step <value> [additional args...]"
    exit 1
fi

if [ -z "$RESUME_ID" ]; then
    echo "Error: --resume_id is required"
    echo "Usage: sbatch resume.sh --cosmo_exp <value> --resume_id <value> --resume_step <value> [additional args...]"
    exit 1
fi

if [ -z "$RESUME_STEP" ]; then
    echo "Error: --resume_step is required"
    echo "Usage: sbatch resume.sh --cosmo_exp <value> --resume_id <value> --resume_step <value> [additional args...]"
    exit 1
fi

# Set log directory based on cosmo_exp
LOG_DIR="${SCRATCH}/bedcosmo/${COSMO_EXP}/logs"
mkdir -p "$LOG_DIR"

# Capture all stdout/stderr into the same log file that torchrun uses.
JOB_LOG="${LOG_DIR}/${SLURM_JOB_ID}_${SLURM_JOB_NAME}.log"
touch "$JOB_LOG"
exec > >(tee -a "$JOB_LOG") 2>&1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRUNCATE_SCRIPT="${SCRIPT_DIR}/../truncate_metrics.py"

echo "=========================================="
echo "Resuming training run: $RESUME_ID"
echo "Resume step: $RESUME_STEP"
echo "=========================================="

# Step 1: Truncate metrics before resuming
echo ""
echo "Step 1: Truncating metrics to resume step..."
echo "--------------------------------------------"

if [[ -f "$TRUNCATE_SCRIPT" ]]; then
    echo "Proceeding with metrics truncation..."
    if python3 "$TRUNCATE_SCRIPT" --run_id "$RESUME_ID" --resume_step "$RESUME_STEP" --cosmo_exp "$COSMO_EXP"; then
        echo "Metrics truncation completed successfully!"
    else
        echo "Error: Metrics truncation failed. Aborting resume to avoid corrupt metrics."
        exit 1
    fi
else
    echo "Error: Truncate script not found at $TRUNCATE_SCRIPT"
    echo "Aborting resume. Please ensure the truncate_metrics.py script is available."
    exit 1
fi

echo ""
echo "Step 2: Resuming training..."
echo "----------------------------"

# Step 2: Resume the training run
srun torchrun \
     --nproc_per_node=$NPROC_PER_NODE \
     --nnodes=$SLURM_NNODES \
     --node_rank=$SLURM_PROCID \
     --rdzv_backend=c10d \
     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     -m bedcosmo.train \
     --cosmo_exp $COSMO_EXP \
     --resume_id $RESUME_ID \
     --resume_step $RESUME_STEP \
     "${EXTRA_ARGS[@]}"
