#!/bin/bash

export OMP_NUM_THREADS=4 

# Loop over 3 entropy seeds with n_particles=5000 (2 GPUs)
GPUS=1
python -m torch.distributed.run --nnodes=1 --nproc_per_node=$GPUS num_tracers/n_tracers_eval.py
