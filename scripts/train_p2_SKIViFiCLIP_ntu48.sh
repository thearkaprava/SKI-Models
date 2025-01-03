#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the distributed training
python -m torch.distributed.launch \
    --master_port 29501 \
    --nproc_per_node=4 \
    P2_SKIVLM_main.py \
    -cfg ./configs/16_16_skivlm_base_train_48_12.yaml \
    --output ./workdirs/TEST_P2_SKIViFiCLIP_zs_48_25Dec24/ \
    --resume ./workdirs/TEST_P1_SKeletonCLIP_alltrn_HFinit_zs_48_17Dec24/ckpt_epoch_0.pth \
    --resume_pose /data/users/asinha13/projects/CLIP4ADL/model_chkpnts/ntu_CS_P1_HF_pretrain_frzn_txt_zs_48_AS_16Jan24/ckpt_epoch_99.pth \
    --alpha 10.0