#!/bin/bash
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -A desi
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --array=0-3
#SBATCH -J n_tracers_seed
#SBATCH --output=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x_%a.log
#SBATCH --error=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x_%a.log
#SBATCH -t 11:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashandon@uci.edu

module load conda
conda activate bed
export OMP_NUM_THREADS=32

SEEDS=(1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

srun python -u num_tracers/n_tracers_train.py \
     --device "cuda" \
     --pyro_seed $SEED \
     --scheduler_type "cosine" \
     --steps 200000 \
     --exp_name "base_NAF_seed8_fixed" \