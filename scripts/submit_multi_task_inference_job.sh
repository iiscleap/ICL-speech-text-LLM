#!/bin/bash

# Configuration - Edit these values as needed
model_type="salmonn"  # Options: "salmonn" or "qwen2"
dataset_types="voxceleb,hvb"  # Comma-separated list of datasets
input_mode="speech_only"  # Options: "speech_only", "text_only", "speech_and_text"
fewshot_mode="text"  # Options: "text" or "speech"
num_examples=5
batch_size=1

# Performance optimization options
use_fp16=true  # Enable mixed precision inference with FP16
use_bf16=false  # Enable mixed precision inference with BF16 (better for newer GPUs)
optimize_batch_size=true  # Automatically find optimal batch size
max_batch_size=32  # Maximum batch size to try when optimizing
num_workers=4  # Number of workers for data loading
use_compile=false  # Use torch.compile for faster inference (requires PyTorch 2.0+)
save_per_dataset=true  # Save separate result files for each dataset type
seed=42  # Random seed for reproducibility

# Path to the trained model - REQUIRED
peft_model_path="./results/trained_models/ft_10e8b_salmonn_multi_voxceleb_hvb/final_model.pt"

# Check if model path is provided
if [ -z "$peft_model_path" ] || [ ! -f "$peft_model_path" ]; then
    echo "Error: Valid PEFT model path is required"
    echo "Please set peft_model_path in the script to point to a valid model file"
    exit 1
fi

# Set conda environment based on model type
if [ "$model_type" == "salmonn" ]; then
    export CONDA_ENV="salmon"
    base_model_path="/path/to/salmonn/model"  # Update with actual path
elif [ "$model_type" == "qwen2" ]; then
    export CONDA_ENV="qwen2"
    base_model_path="/path/to/qwen2/model"  # Update with actual path
else
    echo "Invalid model type. Please specify 'salmonn' or 'qwen2'"
    exit 1
fi

echo "Set conda environment to: $CONDA_ENV"

# Extract run name from model path
RUN_NAME=$(echo "$peft_model_path" | sed -n 's/.*trained_models\/\([^/]*\).*/\1/p')
RUN_NAME=$(echo "$RUN_NAME" | sed 's/speech/sp/g; s/text/txt/g; s/salmonn/sal/g; s/qwen2/qw/g; s/voxceleb/vox/g')

# Set script path
SCRIPT_PATH="/data2/neeraja/neeraja/code/ICL/inference/inference.py"
TODAY=$(date +"%Y-%m-%d")

# Create output directories
mkdir -p "/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}"
mkdir -p "/data2/neeraja/neeraja/results/model_ICL/metrics/${TODAY}"

# Remove old log file if it exists
rm -f "/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}/${RUN_NAME}_${dataset_type}_${input_mode}_${fewshot_mode}_${num_examples}shots.qsub.log"

# Build optimization flags
OPTIMIZATION_FLAGS=""

if [ "$use_fp16" = true ]; then
    OPTIMIZATION_FLAGS="${OPTIMIZATION_FLAGS} --fp16"
fi

if [ "$use_bf16" = true ]; then
    OPTIMIZATION_FLAGS="${OPTIMIZATION_FLAGS} --bf16"
fi

if [ "$optimize_batch_size" = true ]; then
    OPTIMIZATION_FLAGS="${OPTIMIZATION_FLAGS} --optimize_batch_size --max_batch_size ${max_batch_size}"
fi

if [ "$use_compile" = true ]; then
    OPTIMIZATION_FLAGS="${OPTIMIZATION_FLAGS} --compile"
fi

if [ "$save_per_dataset" = true ]; then
    OPTIMIZATION_FLAGS="${OPTIMIZATION_FLAGS} --save_per_dataset"
fi

# Submit job
qsub -q long.q -V -cwd \
    -l hostname=compute-0-9 \
    -l h_rt=72:00:00 \
    -o "/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}/${RUN_NAME}_multi_${input_mode}_${fewshot_mode}_${num_examples}shots.qsub.log" \
    -j y \
    -v CUDA_VISIBLE_DEVICES=0,\
TODAY=${TODAY},\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
num_examples=${num_examples},\
base_model_path=${base_model_path},\
peft_model_path=${peft_model_path},\
SCRIPT_PATH=${SCRIPT_PATH},\
input_mode=${input_mode},\
dataset_type=${dataset_types},\
fewshot_mode=${fewshot_mode},\
model_type=${model_type},\
batch_size=${batch_size},\
num_workers=${num_workers},\
seed=${seed},\
output_suffix="_multi",\
OPTIMIZATION_FLAGS="${OPTIMIZATION_FLAGS}" \
    -S /bin/bash /data2/neeraja/neeraja/code/ICL/scripts/inference.sh

echo "Submitted multi-task inference job for ${RUN_NAME}" 