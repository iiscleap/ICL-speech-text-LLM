#!/bin/bash

echo "Running: python ${SCRIPT_PATH} with arguments"
echo "RUN_NAME: ${RUN_NAME}"

# Common arguments for both cases
COMMON_ARGS="--output_dir \"/data2/neeraja/neeraja/results/model_ICL/trained_models/${RUN_NAME}\" \
        --input_mode ${input_mode} \
        --dataset_type ${dataset_type} \
        --fewshot_mode ${fewshot_mode} \
        --model_type ${model_type} \
        --batch_size ${batch_size:-8} \
        --gradient_accumulation_steps ${gradient_accumulation_steps:-1} \
        --learning_rate ${learning_rate:-1e-5} \
        --num_epochs ${num_epochs:-10} \
        --warmup_steps ${warmup_steps:-100} \
        --save_every ${save_every:-1} \
        --eval_every ${eval_every:-1} \
        --num_examples ${num_examples:-5} \
        --run_name \"${RUN_NAME}\" \
        --dataloader_num_workers ${dataloader_num_workers:-4} \
        --scheduler ${scheduler:-linear} \
        --seed ${seed:-42} \
        --weight_decay ${weight_decay:-0.01} \
        --max_grad_norm ${max_grad_norm:-1.0} \
        --debug_samples ${debug_samples:-0} \
        --randomize_swap ${randomize_swap:-true} \
        ${OPTIMIZATION_FLAGS}"

# Remove the resume_from_checkpoint argument if it's empty
if [ -z "$resume_from_checkpoint" ]; then
    echo "Starting new training run"
    eval "python ${SCRIPT_PATH} ${COMMON_ARGS}"
else
    echo "Resuming from checkpoint: ${resume_from_checkpoint}"
    eval "python ${SCRIPT_PATH} ${COMMON_ARGS} --resume_from_checkpoint ${resume_from_checkpoint}"
fi

# Print completion message
echo "Training job ${RUN_NAME} completed at $(date)" 