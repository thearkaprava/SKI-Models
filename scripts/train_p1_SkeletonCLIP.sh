d#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run the distributed training
python -m torch.distributed.launch \
    --master_port 29501 \
    --nproc_per_node=8 \
    P1_SkelCLIP_main.py \
    -cfg /PATH/TO/CONFIG \
    --output /PATH/TO/Phase1_OUTPUT/ \
    --resume_pose /PATH/TO/PRETRAINED_HYPERFORMER_MODEL/
