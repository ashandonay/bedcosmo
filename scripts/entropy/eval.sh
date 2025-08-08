#!/bin/bash

export OMP_NUM_THREADS=4 
GPUS=1

torchrun \
    --nproc_per_node=$GPUS \
    num_tracers/evaluate.py \
    --run_id a7ffddc7f3b3452eb34104a4416137f1 \
    --eval_step last \
    --global_rank "[0, 1]" \
    --guide_samples 10000 \
    --n_particles 501
