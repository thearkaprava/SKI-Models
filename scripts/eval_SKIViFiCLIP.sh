#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run the distributed training
python -m torch.distributed.launch \
    --master_port 29501 \
    --nproc_per_node=8 \
    P2_SKIVLM_main.py \
    -cfg /PATH/TO/CONFIG \
    --output /PATH/TO/OUTPUT \
    --only_test \
    --resume /PATH/TO/Phase2_MODEL \
