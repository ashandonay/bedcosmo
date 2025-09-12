#!/bin/bash

export OMP_NUM_THREADS=4 
GPUS=1

torchrun \
    --nproc_per_node=$GPUS \
    evaluate.py \
    --run_id 0957d8ba072e4511acfc7d07afc5dc8e \
    --eval_step last \
    --global_rank "[0, 1]" \
    --guide_samples 10000 \
    --n_particles 501
