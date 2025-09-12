#!/bin/bash

export OMP_NUM_THREADS=32
GPUS=2

python -m torch.distributed.run --nproc_per_node=$GPUS train.py \
    --resume_id fa335caaaa924bb996f696ed7bd871a6 \
    --resume_step 8000 \