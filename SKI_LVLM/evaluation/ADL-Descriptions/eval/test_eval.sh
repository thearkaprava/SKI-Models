#!/bin/bash

# Define common arguments for all scripts
PRED_PATH="/data/users/asinha13/projects/Video-ChatGPT/neurips_workshop_inria/data/charades_vlma2/charades_cropped_videos_vllama2_q1.json"
OUTPUT_DIR="/data/users/asinha13/projects/home_dir/CLIP4ADL/SKI_Models/SKI_LVLM/work_dirs/temp/"
OUTPUT_NAME="test"

rm -r $OUTPUT_DIR
# Run the "correctness" evaluation script
python evaluate_benchmark_1_correctness.py \
      --pred_path ${PRED_PATH} \
      --output_dir $OUTPUT_DIR \
      --output_json ${OUTPUT_NAME}-corr.json \
      --num_tasks 8

rm -r $OUTPUT_DIR