#!/bin/bash

export OMP_NUM_THREADS=32

GPUS=2
n_particles_per_device=2000
torchrun \
    --nproc_per_node=$GPUS \
    num_tracers/n_tracers_train.py \
    --cosmo_model base \
    --mlflow_exp base_small_design \
    --n_particles_per_device $n_particles_per_device \
    --total_steps 3000 \
    --pyro_seed 1 \
    --initial_lr 0.0005 \
    --final_lr 0.0001 \
    --design_step "[0.025, 0.05, 0.05, 0.025]" \
    --design_lower "[0.025, 0.1, 0.1, 0.1]" \
    --restart_id 0957d8ba072e4511acfc7d07afc5dc8e \
    --restart_step 5000 \
    --verbose