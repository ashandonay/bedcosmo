#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J brute_force
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

mkdir -p logs

# Load conda, then activate environment
module load conda
conda activate bedcosmo

cd /global/homes/a/ashandon/bedcosmo

echo "SLURM mem per node: ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "SLURM mem per cpu: ${SLURM_MEM_PER_CPU:-unknown} MB"
echo "Node MemTotal:"
grep -i "^MemTotal" /proc/meminfo
echo "free -h:"
free -h

srun python -m bedcosmo.grid_calc num_visits \
    --design-args-path design_args_2d.yaml \
    --prior-args-path prior_args_uniform.yaml \
    --device cpu \
    --param-pts 1000 \
    --feature-pts 500
