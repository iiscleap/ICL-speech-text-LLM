# inference_sentiment.sh
#!/bin/bash
echo "Running: python ${SCRIPT_PATH} with arguments"

python ${SCRIPT_PATH} \
    --base_model_path "lmsys/vicuna-13b-v1.1" \
    --peft_model_path "${MODEL_PATH}" \
    --run_name "${RUN_NAME}" \
    --today "${TODAY}" \
    --output_suffix "${RUN_NAME}" \
    --num_examples "${num_examples}" \
    --input_mode "${input_mode}" \
    --dataset_type "${dataset_type}" \
    --model_type "${model_type}" \
    --fewshot_mode "${fewshot_mode}"