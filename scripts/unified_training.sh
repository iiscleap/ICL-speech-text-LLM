#!/bin/bash
# filepath: /data2/neeraja/neeraja/code/ICL/scripts/unified_training.sh

echo "Running: python ${SCRIPT_PATH} with arguments"
echo "RUN_NAME: ${RUN_NAME}"

# Build the command with all arguments
python ${SCRIPT_PATH} \
    --initial_model_path "${initial_model_path}" \
    --dataset_type "${dataset_type}" \
    --model_type "${model_type}" \
    --lora_lr ${lora_lr} \
    --mlp_lr ${mlp_lr} \
    --lora_epochs ${lora_epochs} \
    --mlp_epochs ${mlp_epochs} \
    --total_cycles ${total_cycles} \
    --hidden_dim ${hidden_dim} \
    --batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_grad_norm ${max_grad_norm} \
    --warmup_steps ${warmup_steps} \
    --max_samples ${max_samples} \
    --run_name "${RUN_NAME}" \
    --output_dir "/data2/neeraja/neeraja/results/model_ICL/unified_training/${RUN_NAME}" \
    ${OPTIMIZATION_FLAGS}

# Print completion message
echo "Unified symbol training job ${RUN_NAME} completed at $(date)"