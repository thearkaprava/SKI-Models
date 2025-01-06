#!/bin/bash

# Define common arguments for all scripts
PRED_PATH="path to json generated with ADL-Descriptions/inference/multiproc_run_inference_benchmark_general.py"
PRED_PATH_CONSISTENCY="path to json generated with ADL-Descriptions/inference/multiproc_run_inference_benchmark_consistency.py"
OUTPUT_DIR="output directory to store intermediate files, will be deleted after each run"
OUTPUT_NAME="filename for evaluation outputs"

rm -r $OUTPUT_DIR
# Run the "correctness" evaluation script
python evaluate_benchmark_1_correctness.py \
      --pred_path ${PRED_PATH} \
      --output_dir $OUTPUT_DIR \
      --output_json ${OUTPUT_NAME}-corr.json \
      --num_tasks 8

rm -r $OUTPUT_DIR

# # Run the "detailed orientation" evaluation script
python evaluate_benchmark_2_detailed_orientation.py \
  --pred_path $PRED_PATH \
  --output_dir $OUTPUT_DIR \
  --output_json ${OUTPUT_NAME}-do.json \
  --num_tasks 8

rm -r $OUTPUT_DIR

python evaluate_benchmark_3_context.py \
  --pred_path $PRED_PATH \
  --output_dir $OUTPUT_DIR \
  --output_json ${OUTPUT_NAME}-con.json \
  --num_tasks 8

rm -r $OUTPUT_DIR
python evaluate_benchmark_4_temporal.py \
  --pred_path $PRED_PATH \
  --output_dir $OUTPUT_DIR \
  --output_json ${OUTPUT_NAME}-temp.json \
  --num_tasks 8

rm -r $OUTPUT_DIR

python evaluate_benchmark_5_consistency.py \
  --pred_path $PRED_PATH_CONSISTENCY \
  --output_dir $OUTPUT_DIR \
  --output_json ${OUTPUT_NAME}_des_cons.json \
  --num_tasks 8

rm -r $OUTPUT_DIR

echo "All evaluations completed!"
