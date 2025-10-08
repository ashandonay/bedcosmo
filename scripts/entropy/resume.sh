#!/bin/bash

export OMP_NUM_THREADS=32
GPUS=2

RUN_ID=3b68d103f21646c5aa222485f6a0f813
RESUME_STEP=8000

# Get the directory where this script is located
TRUNCATE_SCRIPT="/home/ashandonay/bed/BED_cosmo/scripts/truncate_metrics.py"

echo "=========================================="
echo "Resuming training run: $RUN_ID"
echo "Resume step: $RESUME_STEP"
echo "=========================================="

# Step 1: Truncate metrics before resuming
echo ""
echo "Step 1: Truncating metrics to resume step..."
echo "--------------------------------------------"

if [[ -f "$TRUNCATE_SCRIPT" ]]; then
    echo "Proceeding with metrics truncation..."
    python3 "$TRUNCATE_SCRIPT" --run_id "$RUN_ID" --resume_step "$RESUME_STEP"
    
    if [[ $? -eq 0 ]]; then
        echo "Metrics truncation completed successfully!"
    else
        echo "Warning: Metrics truncation failed, but continuing with training resume..."
    fi
else
    echo "Warning: Truncate script not found at $TRUNCATE_SCRIPT"
    echo "Continuing without metrics truncation..."
fi

echo ""
echo "Step 2: Resuming training..."
echo "----------------------------"


python -m torch.distributed.run --nproc_per_node=$GPUS train.py \
    --resume_id $RUN_ID \
    --resume_step $RESUME_STEP