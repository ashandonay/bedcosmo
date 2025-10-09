#!/bin/bash
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -A desi
#SBATCH --job-name=resume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 primary Slurm task per node
#SBATCH --cpus-per-task=128     # CPUs for all DDP workers on the node (e.g., 4 workers * 32 cpus/worker)
#SBATCH --gpus-per-node=4       # Number of GPUs to request per node
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x_%a.log
#SBATCH --error=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x_%a.log

# Load conda first, then activate, then other GPU libraries
module load conda
conda activate bed-cosmo

# Load NERSC CUDA and NCCL modules AFTER conda activation
module load nccl/2.21.5 # NERSC NCCL for Slingshot
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

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

RUN_ID=c15b5b2695cc43bfb4b61842d8ee6274
RESUME_STEP=28000

# Get the directory where this script is located
TRUNCATE_SCRIPT="/global/homes/a/ashandon/bed/BED_cosmo/scripts/truncate_metrics.py"

echo "=========================================="
echo "Resuming training run: $RUN_ID"
echo "Resume step: $RESUME_STEP"
echo "=========================================="

# Step 1: Truncate metrics before resuming
echo ""
echo "Step 1: Truncating metrics to resume step..."
echo "--------------------------------------------"

if [[ -f "$TRUNCATE_SCRIPT" ]]; then
    echo "Proceeding with metrics truncation..."
    python3 "$TRUNCATE_SCRIPT" --run_id "$RUN_ID" --resume_step "$RESUME_STEP"
    
    if [[ $? -eq 0 ]]; then
        echo "Metrics truncation completed successfully!"
    else
        echo "Warning: Metrics truncation failed, but continuing with training resume..."
    fi
else
    echo "Warning: Truncate script not found at $TRUNCATE_SCRIPT"
    echo "Continuing without metrics truncation..."
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
     /global/homes/a/ashandon/bed/BED_cosmo/train.py \
     --resume_id $RUN_ID \
     --resume_step $RESUME_STEP



