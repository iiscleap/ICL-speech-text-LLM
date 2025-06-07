#!/bin/bash

echo "Running unified inference: python ${SCRIPT_PATH} with arguments"
echo "RUN_NAME: ${RUN_NAME}"

# Common arguments (SAME FORMAT AS INFERENCE.SH)
COMMON_ARGS="--model_type \"${model_type}\" \
    --checkpoint_path \"${checkpoint_path}\" \
    --run_name \"${RUN_NAME}\" \
    --dataset_type \"${dataset_type}\" \
    --split \"${split}\" \
    --batch_size ${batch_size} \
    --max_samples ${max_samples} \
    --max_length ${max_length} \
    --temperature ${temperature} \
    --top_p ${top_p}"

# Run the inference script
eval "python ${SCRIPT_PATH} ${COMMON_ARGS}"

# Print completion message
echo "Unified inference job ${RUN_NAME} completed at $(date)"