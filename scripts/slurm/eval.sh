#!/bin/bash
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -A desi
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 primary Slurm task per node
#SBATCH --cpus-per-task=128     # CPUs for all DDP workers on the node (e.g., 4 workers * 32 cpus/worker)
#SBATCH --gpus-per-node=4       # Request 4 GPUs for the 1 task on the node
#SBATCH --time=00:05:00
#SBATCH --output=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x.log
#SBATCH --error=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x.log

# Load conda first, then activate, then other GPU libraries
module load conda
conda activate bed-cosmo

# Load NERSC CUDA and NCCL modules AFTER conda activation
module load nccl/2.21.5 # NERSC NCCL for Slingshot

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    /global/homes/a/ashandon/bed/BED_cosmo/num_tracers/eval.py \
    --run_id "e03e77daa0c8402daf8eb55e9761eca0" \
    --eval_step 200000 \
    --global_rank 4 \
    --design_step 0.01 \
    --eval_particles 2000