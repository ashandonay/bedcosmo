#!/bin/bash

export OMP_NUM_THREADS=4 
GPUS=2

python -m torch.distributed.run --nnodes=1 --nproc_per_node=$GPUS num_tracers/n_tracers_train_distributed.py \
    --cosmo_model base \
    --device cuda \
    --exp_name ddp_test \
    --n_particles 5000 \
    --steps 50000 \
    --verbose