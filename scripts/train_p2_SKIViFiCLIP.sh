#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the distributed training
python -m torch.distributed.launch \
    --master_port 29501 \
    --nproc_per_node=4 \
    P2_SKIVLM_main.py \
    -cfg /PATH/TO/CONFIG \
    --output /PATH/TO/Phase2_OUTPUT/ \
    --resume /PATH/TO/PRETRAINED_VLM_MODEL/ \
    --resume_pose /PATH/TO/Phase1_MODEL/ \
    --alpha 10.0