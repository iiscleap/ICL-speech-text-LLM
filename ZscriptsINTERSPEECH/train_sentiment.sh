# train_sentiment.sh
#!/bin/bash

# export CUDA_LAUNCH_BLOCKING=1

echo "Running: python ${SCRIPT_PATH} with arguments"
echo "RUN_NAME: ${RUN_NAME}"

# Remove the resume_from_checkpoint argument if it's empty
if [ -z "$resume_from_checkpoint" ]; then
    python ${SCRIPT_PATH} \
        --model_path "lmsys/vicuna-13b-v1.1" \
        --output_dir "./results/trained_models/${RUN_NAME}" \
        --log_file "./results/logs/train/${RUN_NAME}.log" \
        --input_mode ${input_mode} \
        --dataset_type ${dataset_type} \
        --fewshot_mode ${fewshot_mode} \
        --model_type ${model_type}
else
    python ${SCRIPT_PATH} \
        --model_path "lmsys/vicuna-13b-v1.1" \
        --output_dir "./results/trained_models/${RUN_NAME}" \
        --log_file "./results/logs/train/${RUN_NAME}.log" \
        --input_mode ${input_mode} \
        --dataset_type ${dataset_type} \
        --fewshot_mode ${fewshot_mode} \
        --model_type ${model_type} \
        --resume_from_checkpoint ${resume_from_checkpoint}
fi






