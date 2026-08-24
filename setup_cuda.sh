#!/bin/bash

# CUDA paths
export CUDA_HOME=/home/ashandonay/miniconda3/envs/tf-py39-2.13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH

# Additional CUDA settings
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=0

# Print CUDA information
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH" 