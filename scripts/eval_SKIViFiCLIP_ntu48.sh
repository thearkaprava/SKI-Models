#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Run the distributed training
python -m torch.distributed.launch \
    --master_port 29507 \
    --nproc_per_node=4 \
    P2_SKIVLM_main.py \
    -cfg ./configs/16_16_zs_eval_48_12.yaml \
    --output ./workdirs/TEST_eval_SKIViFiCLIP_zs_48_3Jan24/ \
    --only_test \
    --resume ./workdirs/TEST_P2_SKIViFiCLIP_zs_48_25Dec24/ckpt_epoch_0.pth \
