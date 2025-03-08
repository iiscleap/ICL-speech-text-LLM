#!/bin/bash

# Configuration - Edit these values as needed
model_type="salmonn"  # Options: "salmonn" or "qwen2"
dataset_types="voxceleb,hvb"  # Comma-separated list of datasets
input_mode="speech_only"  # Options: "speech_only", "text_only", "speech_and_text"
fewshot_mode="text"  # Options: "text" or "speech"
num_examples=5
batch_size=8
gradient_accumulation_steps=1
learning_rate=1e-4
num_epochs=10
warmup_steps=100
save_every=1000
eval_every=500

# Set conda environment based on model type
if [ "$model_type" == "salmonn" ]; then
    export CONDA_ENV="salmon"
    model_path="/path/to/salmonn/model"  # Update with actual path
elif [ "$model_type" == "qwen2" ]; then
    export CONDA_ENV="qwen2"
    model_path="/path/to/qwen2/model"  # Update with actual path
else
    echo "Invalid model type. Please specify 'salmonn' or 'qwen2'"
    exit 1
fi

echo "Set conda environment to: $CONDA_ENV"

# Generate a descriptive run name
DATASET_NAME=$(echo $dataset_types | tr ',' '_')
RUN_NAME="ft_${num_epochs}e${batch_size}b_${model_type}_multi_${DATASET_NAME}"

# Set script path
SCRIPT_PATH="/data2/neeraja/neeraja/code/ICL/train/train.py"
TODAY=$(date +"%Y-%m-%d")

# Create output directories
mkdir -p "./results/logs/train/${TODAY}"
mkdir -p "./results/trained_models/${RUN_NAME}"

# Remove old log file if it exists
rm -f "./results/logs/train/${TODAY}/${RUN_NAME}.log"

# Submit job
qsub -q long.q -V -cwd \
    -l hostname=compute-0-9 \
    -l h_rt=72:00:00 \
    -o "./results/logs/train/${TODAY}/${RUN_NAME}.log" \
    -j y \
    -v CUDA_VISIBLE_DEVICES=0,\
TODAY=${TODAY},\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
num_examples=${num_examples},\
model_path=${model_path},\
SCRIPT_PATH=${SCRIPT_PATH},\
input_mode=${input_mode},\
dataset_type=${dataset_types},\
fewshot_mode=${fewshot_mode},\
model_type=${model_type},\
batch_size=${batch_size},\
gradient_accumulation_steps=${gradient_accumulation_steps},\
learning_rate=${learning_rate},\
num_epochs=${num_epochs},\
warmup_steps=${warmup_steps},\
save_every=${save_every},\
eval_every=${eval_every},\
is_multi_task=true \
    -S /bin/bash /data2/neeraja/neeraja/code/ICL/scripts/train.sh

echo "Submitted multi-task training job for ${RUN_NAME}" 