#!/bin/bash
# Train the empirical SED prior normalizing flow(s) on a compute node.
#
# Wraps `python -m bedcosmo.num_visits.empirical.prior_flow`. NSF training
# is CPU-heavy, so we grab a full node and let torch use all its cores rather
# than thrash a thread-capped login node. Writes sed_prior_flow_native.pt and
# sed_prior_flow_gaussianized.pt beside the KDE ($SCRATCH/.../empirical_prior/).
#
# Usage (all args after the script name pass through to the Python module):
#   sbatch scripts/slurm/train_prior_flow.sh --space both --n 100000 --epochs 400
#   sbatch -t 00:30:00 -q debug scripts/slurm/train_prior_flow.sh --space native --n 50000
#
# CPU-only job: train on a CPU node (torch/JAX are pinned to CPU below), not a
# GPU node. Use account desi (CPU); -C cpu; override -q debug at submit if wanted.
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J train_prior_flow
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
# ~8 min for one flow at n=100k/400ep (128x128); --space both runs them
# concurrently, so 20 min is ~2x headroom. Also fits the faster debug QOS.
#SBATCH -t 00:20:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

mkdir -p logs

module load conda
conda activate bedcosmo

cd /global/homes/a/ashandon/bedcosmo

# CPU-only job: keep torch off the GPU and JAX (pulled in transitively) on CPU.
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

# Use the whole node's cores for the flow. Leave a little headroom. For --space
# both these are split in half across the two concurrent per-space processes.
THREADS="${SLURM_CPUS_PER_TASK:-128}"
if [ "$THREADS" -gt 8 ]; then
    THREADS=$((THREADS - 4))
fi

echo "Node: $(hostname)   threads: $THREADS   args: $*"
free -h | head -2

# Default to both spaces at production size if the caller passes no args.
if [ "$#" -eq 0 ]; then
    set -- --space both --n 100000 --epochs 400 --hidden 128 128 --batch-size 4096
fi

# We're now on the compute node, which is just a (big) local run. Hand off to the
# unified launcher's --local path so the training logic lives in one place. Our
# --threads goes last so the full-node count wins over any caller-supplied value.
exec src/bedcosmo/num_visits/empirical/train_prior_flow.sh --local "$@" --threads "$THREADS"
