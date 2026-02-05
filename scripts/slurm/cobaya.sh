#!/bin/bash
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -J cobaya
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -t 00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Polychord typically benefits from multiple cores but not necessarily multiple nodes
#SBATCH -c 64  # Request 64 cores per task (Milan has 128 cores per node)
#SBATCH --ntasks=1  # Single task

# OpenMP settings for polychord
export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Create logs directory if it doesn't exist
LOG_DIR="${SCRATCH}/bedcosmo/num_tracers/logs"
mkdir -p "$LOG_DIR"

# Capture all stdout/stderr in a single log file
JOB_LOG="${LOG_DIR}/${SLURM_JOB_ID}_cobaya.log"
touch "$JOB_LOG"
exec > >(tee -a "$JOB_LOG") 2>&1

# Change to working directory
cd /global/homes/a/ashandon/bed/BED_cosmo

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Number of cores: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

# Load the DESI cosmological environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# Create unique output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL="base_omegak_w_wa"
SAMPLER="polychord" # "mcmc" or "polychord"
OUTPUT_DIR="${SCRATCH}/bedcosmo/num_tracers/chains/${SAMPLER}/${MODEL}/camb/run${TIMESTAMP}"
mkdir -p $OUTPUT_DIR
FILE_NAME="${OUTPUT_DIR}/chain"

# Update the YAML file to use the new output directory
sed -i "s|output: .*|output: $FILE_NAME|" experiments/num_tracers/cobaya/${SAMPLER}/${MODEL}.yaml

echo "Starting polychord cobaya run..."
echo "Config file: experiments/num_tracers/cobaya/${SAMPLER}/${MODEL}.yaml"
echo "Output directory: $OUTPUT_DIR"
echo "Monitor progress with: tail -f $OUTPUT_DIR/*.txt"

# Run cobaya with force flag to overwrite existing runs
time srun cobaya-run experiments/num_tracers/cobaya/${SAMPLER}/${MODEL}.yaml -f

echo "Job completed at: $(date)"