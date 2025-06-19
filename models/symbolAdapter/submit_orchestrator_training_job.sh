#!/bin/bash
# filepath: /data2/neeraja/neeraja/code/ICL/models/symbolAdapter/submit_orchestrator_training_job.sh

# Configuration - Edit these values as needed
model_type="salmonn"  # Options: "salmonn" or "qwen2"
dataset_type="voxceleb-hvb"  # Dataset type(s) to use
device="cuda:0"  # GPU device

# Training parameters
lora_lr=1e-5
mlp_lr=1e-5

lora_epochs=1
lora_final_epochs=1 

mlp_epochs=1
total_cycles=2

# MLP Architecture parameters
use_output_mlp=False  # Enable/disable output MLP


bypass_mlp=False 


hidden_dim=4
batch_size=1

gradient_accumulation_steps=8
max_grad_norm=1.0
max_samples=0  # Set reasonable default

# NEW: Orchestrator-specific parameters
schedule_type="mlp_first"  # Options: "lora_first", "mlp_first", "joint_training"


dynamic_symbols_per_epoch=False  # Generate new symbols each epoch

# Set conda environment
export CONDA_ENV="salmon"
echo "Set conda environment to: $CONDA_ENV"
source /home/share/anaconda3/etc/profile.d/conda.sh  
conda deactivate
conda activate $CONDA_ENV

# Clean dataset type for filenames
if [[ $dataset_type == *"-"* ]]; then
    CLEAN_DATASET_TYPE=$(echo $dataset_type | tr '-' '_')
else
    CLEAN_DATASET_TYPE=$dataset_type
fi

# Get current date and time
CURRENT_DATETIME=$(date +"%d%m_%H%M")

# Generate descriptive run name
if [ "$bypass_mlp" = "True" ] || [ "$bypass_mlp" = "true" ]; then
    if [ "$dynamic_symbols_per_epoch" = "True" ] || [ "$dynamic_symbols_per_epoch" = "true" ]; then
        MLP_SUFFIX="bypass_mlp_sym"
    else
        MLP_SUFFIX="bypass_mlp_org"
    fi
else
    if [ "$use_output_mlp" = "True" ] || [ "$use_output_mlp" = "true" ]; then
        MLP_SUFFIX="io_mlp"  # Input + Output MLP
    else
        MLP_SUFFIX="i_mlp"   # Input MLP only
    fi
fi

RUN_NAME="${CURRENT_DATETIME}_orchestrator_${schedule_type}_${total_cycles}c_${lora_epochs}le_${mlp_epochs}me_${MLP_SUFFIX}_${model_type}_${CLEAN_DATASET_TYPE}"

# Set script path
SCRIPT_PATH="/data2/neeraja/neeraja/code/ICL/models/symbolAdapter/orchestrator_training.py"
TODAY=$(date +"%Y-%m-%d")

# Directory setup
OUTPUT_DIR="/data2/neeraja/neeraja/results/model_ICL/orchestrator_training"
LOG_DIR="/data2/neeraja/neeraja/results/model_ICL/orchestrator_training/logs/${TODAY}"

# Create directories
for dir in "$LOG_DIR" "$OUTPUT_DIR"; do
    if ! mkdir -p "$dir"; then
        echo "Error: Cannot create directory $dir"
        exit 1
    fi
done

# Remove old log file if it exists
rm -f "${LOG_DIR}/${RUN_NAME}.log"

# Print configuration
echo "=========================================="
echo "Orchestrator Symbol Training Job Configuration"
echo "=========================================="
echo "Run Name: ${RUN_NAME}"
echo "Dataset: ${dataset_type}"
echo "Device: ${device}"
echo "Schedule Type: ${schedule_type}"
echo "Cycles: ${total_cycles}"
echo "LoRA Epochs/Cycle: ${lora_epochs}"
echo "MLP Epochs/Cycle: ${mlp_epochs}"
echo "LoRA LR: ${lora_lr}"
echo "MLP LR: ${mlp_lr}"
echo "Use Output MLP: ${use_output_mlp}"
echo "Bypass MLP: ${bypass_mlp}"
echo "Dynamic Symbols: ${dynamic_symbols_per_epoch}"
echo "Hidden Dim: ${hidden_dim}"
echo "Max Samples: ${max_samples}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Log File: ${LOG_DIR}/${RUN_NAME}.log"
echo "=========================================="

# Submit job
qsub -q gpu.q -V -cwd \
    -l hostname=compute-0-9 \
    -l h_rt=72:00:00 \
    -o "${LOG_DIR}/${RUN_NAME}.log" \
    -j y \
    -v CUDA_VISIBLE_DEVICES=2,\
TODAY=${TODAY},\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
SCRIPT_PATH=${SCRIPT_PATH},\
dataset_type=${dataset_type},\
model_type=${model_type},\
device=${device},\
lora_lr=${lora_lr},\
mlp_lr=${mlp_lr},\
lora_epochs=${lora_epochs},\
mlp_epochs=${mlp_epochs},\
total_cycles=${total_cycles},\
lora_final_epochs=${lora_final_epochs},\
use_output_mlp=${use_output_mlp},\
bypass_mlp=${bypass_mlp},\
hidden_dim=${hidden_dim},\
batch_size=${batch_size},\
gradient_accumulation_steps=${gradient_accumulation_steps},\
max_grad_norm=${max_grad_norm},\
max_samples=${max_samples},\
schedule_type=${schedule_type},\
dynamic_symbols_per_epoch=${dynamic_symbols_per_epoch},\
OUTPUT_DIR=${OUTPUT_DIR} \
    -S /bin/bash /data2/neeraja/neeraja/code/ICL/models/symbolAdapter/orchestrator_training.sh

echo "Submitted orchestrator symbol training job: ${RUN_NAME}"
echo "Monitor with: tail -f ${LOG_DIR}/${RUN_NAME}.log"