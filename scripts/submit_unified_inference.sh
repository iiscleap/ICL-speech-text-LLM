#!/bin/bash

# Configuration - Edit these values as needed
model_type="salmonn"  # Options: "salmonn" or "qwen2"
dataset_type="voxceleb"  # Dataset(s) to use for inference
split="test"  # Dataset split: train, validation, test


checkpoint_path="/data2/neeraja/neeraja/results/model_ICL/unified_training/3105_0053_unified_4c_2le_1me_salmonn_voxceleb_greek_hvb_greek/3105_0053_unified_4c_2le_1me_salmonn_voxceleb_greek_hvb_greek/cycle_4_lora_epoch_2/model.pt"  # Path to trained model checkpoint

# Inference parameters
batch_size=1

max_samples=10  # 0 = use all samples
max_length=512
temperature=0.7
top_p=0.9

# Node configuration
queue_name="longgpu.q"      # Queue to submit job to
hostname="compute-0-8"      # Hostname to run on
cuda_device=0              # CUDA device to use
hold_job_id=""              # Job ID to wait for (empty = don't wait)

# Clean dataset type for file names
CLEAN_DATASET_TYPE=$(echo $dataset_type | tr ',' '-' | tr -d ' ')

# Check if checkpoint exists
if [ ! -f "$checkpoint_path" ]; then
    echo "ERROR: Checkpoint file not found: $checkpoint_path"
    exit 1
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

# Get current date and time
CURRENT_DATETIME=$(date +"%d%m_%H%M")

# Generate run name (same logic as inference script)
RUN_NAME="unified_${CURRENT_DATETIME}_${CLEAN_DATASET_TYPE}"

# Set script path - THIS IS THE ONLY DIFFERENCE
SCRIPT_PATH="/data2/neeraja/neeraja/code/ICL/models/unified_inference.py"
TODAY=$(date +"%Y-%m-%d")

# Create output directories (SAME AS INFERENCE)
mkdir -p "/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}"
mkdir -p "/data2/neeraja/neeraja/results/model_ICL/metrics/${TODAY}"

# Remove old log file
rm -f "/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}/${RUN_NAME}.log"

# Common qsub arguments (SAME AS INFERENCE)
QSUB_ARGS="-q ${queue_name} -V -cwd -l hostname=${hostname} -l h_rt=72:00:00"
LOG_PATH="/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}/${RUN_NAME}.log"

# Common variables for qsub (ADAPTED FOR UNIFIED INFERENCE)
COMMON_VARS="CUDA_VISIBLE_DEVICES=${cuda_device},\
TODAY=${TODAY},\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
SCRIPT_PATH=${SCRIPT_PATH},\
model_type=${model_type},\
checkpoint_path=${checkpoint_path},\
batch_size=${batch_size},\
dataset_type=${CLEAN_DATASET_TYPE},\
split=${split},\
max_samples=${max_samples},\
max_length=${max_length},\
temperature=${temperature},\
top_p=${top_p}"

# Submit job (SAME AS INFERENCE)
if [ -n "$hold_job_id" ]; then
    echo "Submitting job to queue ${queue_name} with dependency on job ID: ${hold_job_id}"
    JOB_ID=$(qsub ${QSUB_ARGS} -hold_jid ${hold_job_id} -o "${LOG_PATH}" -j y \
    -v ${COMMON_VARS} \
    -S /bin/bash /data2/neeraja/neeraja/code/ICL/scripts/unified_inference.sh)
else
    echo "Submitting job to queue ${queue_name} without dependencies"
    JOB_ID=$(qsub ${QSUB_ARGS} -o "${LOG_PATH}" -j y \
    -v ${COMMON_VARS} \
    -S /bin/bash /data2/neeraja/neeraja/code/ICL/scripts/unified_inference.sh)
fi

# Extract job ID and report
JOB_ID_NUM=$(echo $JOB_ID | cut -d' ' -f3)

echo "Submitted unified inference job for ${RUN_NAME} to queue ${queue_name}"
echo "Job ID: ${JOB_ID_NUM}"