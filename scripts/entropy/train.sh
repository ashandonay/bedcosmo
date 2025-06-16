#!/bin/bash

export OMP_NUM_THREADS=32

GPUS=2
n_particles_per_device=1000
torchrun \
    --nproc_per_node=$GPUS \
    num_tracers/n_tracers_train_distributed.py \
    --cosmo_model base \
    --exp_name base_NAF \
    --n_particles_per_device $n_particles_per_device \
    --steps 10000 \
    --pyro_seed 1 \
    --final_lr 0.0001 \
    --verbose