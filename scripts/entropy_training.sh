#!/bin/bash

export OMP_NUM_THREADS=4 
GPUS=1

python -m torch.distributed.run --nnodes=1 --nproc_per_node=$GPUS num_tracers/n_tracers_train_distributed.py \
    --cosmo_model base \
    --device cuda \
    --exp_name ddp_test \
    --n_particles 10000 \
    --steps 50000 \
    --verbose