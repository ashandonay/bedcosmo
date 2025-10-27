#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -A desi
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 primary Slurm task per node
#SBATCH --cpus-per-task=32     # CPUs for all DDP workers on the node (e.g., 4 workers * 32 cpus/worker)
#SBATCH --gpus-per-node=1       # Number of GPUs to request per node
#SBATCH --time=01:00:00
#SBATCH --output=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x.log
#SBATCH --error=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x.log

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

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    /global/homes/a/ashandon/bed/BED_cosmo/evaluate.py \
    --run_id bc59983ae5634ffcaf761f40084b6f8a \
    --eval_step last \
    --global_rank "[0, 1, 2, 3]" \
    --levels 0.68 \
    --guide_samples 100000 \
    --n_particles 101 \
    --param_space physical \
    --design_step "[0.01, 0.02, 0.02, 0.02]" \
    --design_lower "[0.02, 0.1, 0.1, 0.1]" \
    --profile