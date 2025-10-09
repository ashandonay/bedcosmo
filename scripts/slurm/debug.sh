#!/bin/bash
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -A desi
#SBATCH --job-name=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 primary Slurm task per node
#SBATCH --cpus-per-task=128     # CPUs for all DDP workers on the node (e.g., 4 workers * 32 cpus/worker)
#SBATCH --gpus-per-node=4       # Number of GPUs to request per node
#SBATCH --time=00:10:00
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
export OMP_NUM_THREADS=$((128 / $NPROC_PER_NODE))

# Tell CUDA to order GPUs by PCI_BUS_ID for consistency
export CUDA_DEVICE_ORDER=PCI_BUS_ID

srun torchrun \
     --nproc_per_node=$NPROC_PER_NODE \
     --nnodes=$SLURM_NNODES \
     --node_rank=$SLURM_PROCID \
     --rdzv_backend=c10d \
     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     /global/homes/a/ashandon/bed/BED_cosmo/train.py \
     --cosmo_model base_omegak_w_wa \
     --mlflow_exp debug \
     --cosmo_exp num_tracers \
     --pyro_seed 1 \
     --nf_seed 1 \
     --flow_type MAF \
     --activation elu \
     --n_transforms 10 \
     --cond_hidden_size 256 \
     --cond_n_layers 8 \
     --mnn_hidden_size 256 \
     --mnn_n_layers 4 \
     --mnn_signal 64 \
     --spline_bins 20 \
     --n_particles_per_device 1000 \
     --total_steps 10000 \
     --scheduler_type cosine \
     --initial_lr 0.0002 \
     --final_lr 0.0 \
     --warmup_fraction 0.0 \
     --design_step "[0.025, 0.05, 0.05, 0.025]" \
     --design_lower "[0.025, 0.1, 0.1, 0.1]" \
     --fixed_design \
     --verbose