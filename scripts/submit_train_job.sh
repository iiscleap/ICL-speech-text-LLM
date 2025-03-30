#!/bin/bash

# Configuration - Edit these values as needed
model_type="salmonn"  # Options: "salmonn" or "qwen2"
# dataset_type="voxceleb,hvb"  # Options: "voxceleb", "hvb", "voxpopuli", etc.
dataset_type="voxceleb_swap,hvb_swap"  # Options: "voxceleb", "hvb", "voxpopuli", etc.
input_mode="speech_only"  # Options: "speech_only", "text_only", "speech_and_text"
fewshot_mode="text"  # Options: "text" or "speech"
num_examples=5
batch_size=1
gradient_accumulation_steps=8
learning_rate=1e-5
num_epochs=20
warmup_steps=100
save_every=1
eval_every=5000

# Add near the top with other configuration options
randomize_swap=true  # Default to true for training

# Performance optimization options
use_fp16=true  # Enable mixed precision training with FP16
use_bf16=false  # Enable mixed precision training with BF16 (better for newer GPUs)
use_gradient_checkpointing=true  # Enable gradient checkpointing to save memory
dataloader_num_workers=4  # Number of workers for data loading
scheduler="cosine"  # Options: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
seed=42  # Random seed for reproducibility
weight_decay=0.01  # Weight decay for AdamW
max_grad_norm=1.0  # Max gradient norm for clipping

# Debug options
debug_samples=0  # Set to 0 to use full dataset, or >0 for limited samples

# If in debug mode (debug_samples > 0), override training parameters for faster iteration
if [ "$debug_samples" -gt 0 ]; then
    echo "Debug mode enabled with ${debug_samples} samples"
    gradient_accumulation_steps=1
    num_epochs=1
    warmup_steps=2
    save_every=1
    eval_every=5
fi

# Set conda environment based on model type
if [ "$model_type" == "salmonn" ]; then
    export CONDA_ENV="salmon"
elif [ "$model_type" == "qwen2" ]; then
    export CONDA_ENV="qwen2"
else
    echo "Invalid model type. Please specify 'salmonn' or 'qwen2'"
    exit 1
fi

echo "Set conda environment to: $CONDA_ENV"
source /home/share/anaconda3/etc/profile.d/conda.sh  
conda deactivate
conda activate $CONDA_ENV  


if [[ $dataset_type == *","* ]]; then
    # For file names (replace commas with hyphens)
    CLEAN_DATASET_TYPE=$(echo $dataset_type | tr ',' '-' | tr -d ' ')
else
    # No cleaning needed
    CLEAN_DATASET_TYPE=$dataset_type
fi

# Calculate effective batch size
effective_batch_size=$((batch_size * gradient_accumulation_steps))

# Get current date and time in DD_MM_HHMM format
CURRENT_DATETIME=$(date +"%d%m_%H%M")

# Generate a descriptive run name with date and time at the start
RUN_NAME="${CURRENT_DATETIME}_ft_${num_examples}ex_${num_epochs}e${effective_batch_size}b_${model_type}_${input_mode}_${fewshot_mode}_${CLEAN_DATASET_TYPE}_BothShu"

# Set script path
SCRIPT_PATH="/data2/neeraja/neeraja/code/ICL/train/train.py"
TODAY=$(date +"%Y-%m-%d")

OUTPUT_DIR="/data2/neeraja/neeraja/results/model_ICL"
LOG_DIR="${OUTPUT_DIR}/logs/train/${TODAY}"
MODEL_DIR="${OUTPUT_DIR}/trained_models/${RUN_NAME}"

# Create directories and check permissions
for dir in "$LOG_DIR" "$MODEL_DIR"; do
    if ! mkdir -p "$dir"; then
        echo "Error: Cannot create directory $dir"
        exit 1
    fi
done
# Remove old log file if it exists
rm -f "${LOG_DIR}/${RUN_NAME}.log"

# Build optimization flags
OPTIMIZATION_FLAGS=""

if [ "$use_fp16" = true ]; then
    OPTIMIZATION_FLAGS="${OPTIMIZATION_FLAGS} --fp16"
fi

if [ "$use_bf16" = true ]; then
    OPTIMIZATION_FLAGS="${OPTIMIZATION_FLAGS} --bf16"
fi

if [ "$use_gradient_checkpointing" = true ]; then
    OPTIMIZATION_FLAGS="${OPTIMIZATION_FLAGS} --gradient_checkpointing"
fi

# Submit job
qsub -q long.q -V -cwd \
    -l hostname=compute-0-9 \
    -l h_rt=72:00:00 \
    -o "${LOG_DIR}/${RUN_NAME}.log" \
    -j y \
    -v CUDA_VISIBLE_DEVICES=2,\
TODAY=${TODAY},\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
debug_samples=${debug_samples},\
num_examples=${num_examples},\
SCRIPT_PATH=${SCRIPT_PATH},\
input_mode=${input_mode},\
dataset_type=${CLEAN_DATASET_TYPE},\
fewshot_mode=${fewshot_mode},\
model_type=${model_type},\
batch_size=${batch_size},\
gradient_accumulation_steps=${gradient_accumulation_steps},\
learning_rate=${learning_rate},\
num_epochs=${num_epochs},\
warmup_steps=${warmup_steps},\
save_every=${save_every},\
eval_every=${eval_every},\
dataloader_num_workers=${dataloader_num_workers},\
scheduler=${scheduler},\
seed=${seed},\
weight_decay=${weight_decay},\
max_grad_norm=${max_grad_norm},\
randomize_swap=${randomize_swap},\
OPTIMIZATION_FLAGS="${OPTIMIZATION_FLAGS}" \
    -S /bin/bash /data2/neeraja/neeraja/code/ICL/scripts/train.sh

echo "Submitted training job for ${RUN_NAME}" 