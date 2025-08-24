#!/bin/bash

export OMP_NUM_THREADS=32
GPUS=2

python -m torch.distributed.run --nproc_per_node=$GPUS num_tracers/n_tracers_train.py \
    --resume_id "4232c68c4787449687c7ea616943796f" \
    --resume_step 1000 \