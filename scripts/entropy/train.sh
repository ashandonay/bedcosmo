#!/bin/bash

export OMP_NUM_THREADS=32
GPUS=2

torchrun \
    --nproc_per_node=$GPUS \
    num_tracers/n_tracers_train_distributed.py \
    --cosmo_model base \
    --exp_name ddp_test \
    --n_particles_per_device 500 \
    --steps 500 \
    --pyro_seed 1 \
    --verbose