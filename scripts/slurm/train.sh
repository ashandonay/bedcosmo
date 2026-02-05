#!/bin/bash
#SBATCH -C gpu
#SBATCH -A desi
#SBATCH --job-name=train
#SBATCH --ntasks-per-node=1     # 1 primary Slurm task per node
#SBATCH --cpus-per-task=128     # CPUs for all DDP workers on the node
#SBATCH --gpus-per-node=4       # Default GPUs per node (overridable via submit.sh --gpus)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashandon@uci.edu
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Note: --time, --qos, and --nodes are set by submit.sh via sbatch CLI flags

# Parse named arguments
COSMO_EXP=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --cosmo_exp)
            COSMO_EXP="$2"
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
    echo "Usage: sbatch train.sh --cosmo_exp <value> [additional args...]"
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
echo "Nodes:        $SLURM_NNODES"
echo "GPUs/Node:    $SLURM_GPUS_PER_NODE"
echo "Cosmo Exp:    $COSMO_EXP"
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
conda activate bedcosmo

# Load NERSC CUDA and NCCL modules AFTER conda activation
module load nccl/2.21.5 # NERSC NCCL for Slingshot
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export NCCL_NET_GDR_LEVEL=PHB # tells NCCL to use GPUdirect when on the same PCI bus
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
export OMP_NUM_THREADS=$((128 / $NPROC_PER_NODE))

# Tell CUDA to order GPUs by PCI_BUS_ID for consistency
export CUDA_DEVICE_ORDER=PCI_BUS_ID

srun torchrun \
     --nproc_per_node=$NPROC_PER_NODE \
     --nnodes=$SLURM_NNODES \
     --node_rank=$SLURM_PROCID \
     --rdzv_backend=c10d \
     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     -m bedcosmo.train \
     --cosmo_exp "$COSMO_EXP" \
     "${EXTRA_ARGS[@]}"