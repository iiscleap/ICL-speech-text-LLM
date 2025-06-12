#!/bin/bash

echo "Running unified inference: python ${SCRIPT_PATH} with arguments"
echo "RUN_NAME: ${RUN_NAME}"

# Set conda environment
source /home/share/anaconda3/etc/profile.d/conda.sh
conda activate salmon

COMMON_ARGS="--peft_model_path \"${checkpoint_path}\" \
    --run_name \"${RUN_NAME}\" \
    --today \"${TODAY}\" \
    --model_type \"${model_type}\" \
    --dataset_type \"${dataset_type}\" \
    --split \"${split}\" \
    --batch_size ${batch_size} \
    --max_samples ${max_samples} \
    --max_length ${max_length} \
    --temperature ${temperature} \
    --top_p ${top_p} \
    --num_examples ${num_examples} \
    --use_mlp ${use_mlp} \
    --compare_modes ${compare_modes} \
    --symbol_mode \"${symbol_mode}\" \
    --device cuda:0"

# Run the inference script
eval "python ${SCRIPT_PATH} ${COMMON_ARGS}"

echo "Unified inference job ${RUN_NAME} completed at $(date)"