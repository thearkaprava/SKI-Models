d#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Run the distributed training
python -m torch.distributed.launch \
    --master_port 29501 \
    --nproc_per_node=4 \
    P1_SkelCLIP_main.py \
    -cfg ./configs/16_16_skeletonclip_base_train_48_12.yaml \
    --output ./workdirs/TEST_P1_SKeletonCLIP_alltrn_HFinit_zs_48_17Dec24/ \
    --resume_pose /data/users/asinha13/projects/CLIP4ADL/model_chkpnts/Hyperformer_joint_zs_110_AS_27Jan24/runs-140-228340.pt
