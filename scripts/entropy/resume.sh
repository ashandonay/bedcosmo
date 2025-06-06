#!/bin/bash

export OMP_NUM_THREADS=32
GPUS=2

python -m torch.distributed.run --nproc_per_node=$GPUS num_tracers/n_tracers_train_distributed.py \
    --resume_id "78b2a800934e472ebb03046af25c84f7" \
    --resume_step 304 \