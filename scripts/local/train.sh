#!/bin/bash

export OMP_NUM_THREADS=32

GPUS=2
n_particles_per_device=1000
hidden_size=128
    
torchrun \
    --nproc_per_node=$GPUS \
    -m bedcosmo.train \
    --cosmo_model base_omegak_w_wa \
    --mlflow_exp base_omegak_w_wa \
    --cosmo_exp num_tracers \
    --flow_type MAF \
    --cond_hidden_size $hidden_size \
    --n_particles_per_device $n_particles_per_device \
    --total_steps 3000 \
    --flow_type MAF \
    --pyro_seed 1 \
    --initial_lr 0.001 \
    --final_lr 0.0001 \
    --design_step "[0.025, 0.05, 0.05, 0.025]" \
    --design_lower "[0.025, 0.1, 0.1, 0.1]" \
    --verbose
    