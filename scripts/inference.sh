#!/bin/bash

echo "Running inference: python ${SCRIPT_PATH} with arguments"
echo "RUN_NAME: ${RUN_NAME}"

# Common arguments
COMMON_ARGS="--peft_model_path \"${peft_model_path}\" \
    --run_name \"${RUN_NAME}\" \
    --today \"${TODAY}\" \
    --output_suffix \"${output_suffix}\" \
    --input_mode ${input_mode} \
    --dataset_type ${dataset_type} \
    --fewshot_mode ${fewshot_mode} \
    --model_type ${model_type} \
    --batch_size ${batch_size:-1} \
    --num_examples ${num_examples:-5} \
    --num_workers ${num_workers:-4} \
    --seed ${seed:-42} \
    --debug_samples ${debug_samples:-0} \
    --fp16"

# Run the inference script
eval "python ${SCRIPT_PATH} ${COMMON_ARGS}"

# Print completion message
echo "Inference job ${RUN_NAME} completed at $(date)" 