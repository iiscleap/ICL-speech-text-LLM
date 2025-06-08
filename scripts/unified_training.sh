#!/bin/bash
# filepath: /data2/neeraja/neeraja/code/ICL/scripts/unified_training.sh

echo "Starting unified symbol training..."
echo "Python script: ${SCRIPT_PATH}"
echo "Run name: ${RUN_NAME}"

# Navigate to script directory
cd "$(dirname "${SCRIPT_PATH}")"

# Execute Python script with all parameters
python3 "${SCRIPT_PATH}" \
    --model_type "${model_type}" \
    --device "${device}" \
    --dataset_type "${dataset_type}" \
    --lora_lr "${lora_lr}" \
    --mlp_lr "${mlp_lr}" \
    --lora_epochs "${lora_epochs}" \
    --mlp_epochs "${mlp_epochs}" \
    --total_cycles "${total_cycles}" \
    --lora_final_epochs "${lora_final_epochs}" \
    --hidden_dim "${hidden_dim}" \
    --batch_size "${batch_size}" \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --max_grad_norm "${max_grad_norm}" \
    --max_samples "${max_samples}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}"

echo "Unified training completed with exit code: $?"