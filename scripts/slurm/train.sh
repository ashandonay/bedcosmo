#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -A desi
#SBATCH --job-name=train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1     # 1 primary Slurm task per node
#SBATCH --cpus-per-task=128     # CPUs for all DDP workers on the node (e.g., 4 workers * 32 cpus/worker)
#SBATCH --gpus-per-node=4       # Request 4 GPUs for the 1 task on the node
#SBATCH --time=10:10:00
#SBATCH --output=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x_%a.log
#SBATCH --error=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x_%a.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashandon@uci.edu

# Load conda first, then activate, then other GPU libraries
module load conda
conda activate bed-cosmo

# Load NERSC CUDA and NCCL modules AFTER conda activation
module load nccl/2.21.5 # NERSC NCCL for Slingshot

# Define number of DDP processes per node
NPROC_PER_NODE=4

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
     /global/homes/a/ashandon/bed/BED_cosmo/num_tracers/n_tracers_train_distributed.py \
     --mlflow_exp base \
     --cosmo_exp num_tracers \
     --flow_type MAF \
     --transform affine \
     --hyper_hidden_size 128 \
     --hyper_n_layers 4 \
     --n_particles_per_device 20000 \
     --design_lower 0.05 \
     --design_step 0.05 \
     --total_steps 250000 \
     --scheduler_type linear \
     --initial_lr 0.001 \
     --final_lr 0.000005 \
     --verbose