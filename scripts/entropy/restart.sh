#!/bin/bash

export OMP_NUM_THREADS=32

GPUS=2
n_particles_per_device=100

torchrun \
    --nproc_per_node=$GPUS \
    train.py \
    --cosmo_model base_omegak_w_wa \
    --mlflow_exp debug \
    --priors_path "num_tracers/priors_small.yaml" \
    --n_particles_per_device $n_particles_per_device \
    --flow_type MAF \
    --activation relu \
    --total_steps 5000 \
    --pyro_seed 1 \
    --initial_lr 0.0005499549954995499 \
    --final_lr 0.0001 \
    --design_step "[0.025, 0.05, 0.05, 0.025]" \
    --design_lower "[0.025, 0.1, 0.1, 0.1]" \
    --restart_id c135063999824d0da5940742dfa49739 \
    --restart_step 5000 \
    --verbose