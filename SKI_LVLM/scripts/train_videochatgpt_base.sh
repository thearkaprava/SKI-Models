#!/bin/bash

# Exporting PYTHONPATH
export PYTHONPATH="./:$PYTHONPATH"

# Running the training command
torchrun --nproc_per_node=4 --master_port 29001 llavidal/train/train_mem.py \
          --model_name_or_path /data/users/rchakra6/Video-ChatGPT/LLaVA-7B-Lightening-v1-1 \
          --version v1 \
          --data_path /data/vidlab_datasets/ntu_cogvlm_qa/ntu120_cogvlm_annotation_traindata_v2.json \
          --video_folder /data/users/asinha13/projects/Video-ChatGPT/ntu120_video_features_24Jul24/ \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./LLAVIDAL_7B-1.1_Checkpoints_NTU_base_17Dec24 \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 3000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True
