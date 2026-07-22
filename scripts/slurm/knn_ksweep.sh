#!/bin/bash
# Diagnostic: does the k-NN marginal EIG_z converge to the focused-guide result
# as the inner-sample count K grows? Drives the production estimator
# (Evaluator._marginal_posterior_entropy path) via
# experiments/num_visits/scripts/knn_marginal_convergence.py over a K sweep.
#
# GPU job: the joint 13-D guide flow is sampled on the GPU (K_max draws per
# (eval, design-chunk, outer-y)); only the cheap k-NN/KDE/Gaussian entropy
# repeats per K, reusing the draws so noise differences come from sample count
# alone. Read-only w.r.t. src/bedcosmo.
#
# Usage (args after the script name pass through to the Python script):
#   sbatch scripts/slurm/knn_ksweep.sh --k-sweep 250,1000,4000,16000 --n-evals 3
#
#SBATCH -A desi
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J knn_ksweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

mkdir -p logs

module load conda
conda activate bedcosmo

# Prefer torch's bundled cuBLAS over system libraries to avoid CUBLAS_STATUS_INVALID_VALUE
# errors from CUDA version mismatches after NERSC system updates. Without this the
# cudatoolkit module's cuBLAS shadows torch's and EVERY batched GEMM fails (torch.bmm
# with batch>=2, any shape/dtype), which killed job 56051637 in the empirical model's
# template-mixing einsum. Same block as train.sh/eval.sh/grid.sh.
CUBLAS_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib"
if [ -d "$CUBLAS_LIB" ]; then
    export LD_LIBRARY_PATH="$CUBLAS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd /global/homes/a/ashandon/bedcosmo

export JAX_PLATFORMS=cpu   # keep transitive JAX off the CUDA plugin

THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$THREADS"

echo "Node: $(hostname)   threads: $THREADS   args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Default: 64x lever arm around production's K=1000, matched outer_y/n_evals.
if [ "$#" -eq 0 ]; then
    set -- --k-sweep 250,1000,4000,16000 --outer-y 10 --n-evals 3 \
        --out-dir /pscratch/sd/a/ashandon/bedcosmo/num_visits/knn_ksweep
fi

python -u experiments/num_visits/scripts/knn_marginal_convergence.py --device cuda:0 "$@"
