#!/bin/bash
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -A desi
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --array=0-5
#SBATCH -J n_tracers_schedule
#SBATCH --output=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x_%a.log
#SBATCH --error=/pscratch/sd/a/ashandon/bed/BED_cosmo/num_tracers/logs/%A_%x_%a.log
#SBATCH -t 10:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashandon@uci.edu

module load conda
conda activate bed
export OMP_NUM_THREADS=32

# Parameter lists
GAMMAS=(0.9794977288060767 0.9847666521101581)
SEEDS=(1 2 3)
EXP_NAME="base_NAF_schedule2_fixed"

# Map task ID to indices
gamma_idx=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))  # 0,1,2
seed_idx=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))   # 0,1,2

GAMMA=${GAMMAS[$gamma_idx]}
SEED=${SEEDS[$seed_idx]}

srun python -u /global/u1/a/ashandon/bed/BED_cosmo/num_tracers/n_tracers_train.py \
     --device "cuda" \
     --pyro_seed $SEED \
     --gamma $GAMMA \
     --gamma_freq 1000 \
     --steps 300000 \
     --exp_name $EXP_NAME