#!/bin/bash
#SBATCH -A desi
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -J brute_force
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

mkdir -p logs

# Load conda, then activate environment
module load conda
conda activate bedcosmo

# Prefer torch's bundled cuBLAS over system libraries to avoid CUBLAS_STATUS_INVALID_VALUE
# errors from CUDA version mismatches after NERSC system updates. Without this the
# cudatoolkit module's cuBLAS shadows torch's and every batched GEMM fails (torch.bmm
# with batch >= 2, any shape or dtype). Same block as train.sh/eval.sh/grid.sh.
CUBLAS_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib"
if [ -d "$CUBLAS_LIB" ]; then
    export LD_LIBRARY_PATH="$CUBLAS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd /global/homes/a/ashandon/bedcosmo

echo "SLURM mem per node: ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "SLURM mem per cpu: ${SLURM_MEM_PER_CPU:-unknown} MB"
echo "Node MemTotal:"
grep -i "^MemTotal" /proc/meminfo
echo "free -h:"
free -h

# Reduce JAX/XLA upfront GPU reservation to lower OOM risk.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# If no GPU is visible to the job, force JAX onto the CPU backend so it doesn't
# try (and noisily fail) to initialize the bundled CUDA12 plugin.
if ! nvidia-smi -L >/dev/null 2>&1; then
    export JAX_PLATFORMS=cpu
    echo "No GPU detected — forcing JAX_PLATFORMS=cpu"
else
    unset JAX_PLATFORMS
fi

srun python -m bedcosmo.grid_calc num_visits \
    --design-args-path design_args_2d.yaml \
    --prior-args-path /pscratch/sd/a/ashandon/bedcosmo/num_visits/mlruns/228253452122836442/2c669af599124f08a4f5666f9e15d36b/artifacts/prior_args.yaml \
    --device cuda:0 \
    --param-pts 200 \
    --feature-pts 120 \
    --temperature 10000 \
    --param-dense-range z:0.1,0.7 \
    --param-dense-fraction 0.9 \
    --feature-range u:-15,100 \
    --feature-range g:0,80 \
    --central-z 1.0