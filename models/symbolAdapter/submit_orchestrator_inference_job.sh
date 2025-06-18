#!/bin/bash
# filepath: /data2/neeraja/neeraja/code/ICL/models/symbolAdapter/submit_orchestrator_inference_job.sh

# ========================================
# Configuration - Edit these values as needed
# ========================================
checkpoint_path="/data2/neeraja/neeraja/results/model_ICL/orchestrator_training/training/1606_1315_orchestrator_joint_training_1c_5le_1me_bypass_mlp_sym_salmonn_voxceleb_hvb/checkpoints/lora_step0_cycle0_epoch1_periodic.pt"
dataset_type="hvb-voxceleb-voxpopuli"  # Dataset type to evaluate on
max_val_samples=0            # 0 = use all samples

# Optional parameters
device="cuda:0"
output_dir="/data2/neeraja/neeraja/results/model_ICL"

# Node configuration
queue_name="gpu.q"
hostname="compute-0-9"
cuda_device=1

# ========================================
# Validation and Setup
# ========================================

# Set conda environment
export CONDA_ENV="salmon"
echo "Set conda environment to: $CONDA_ENV"
source /home/share/anaconda3/etc/profile.d/conda.sh  
conda deactivate
conda activate $CONDA_ENV   

# Check if checkpoint exists
if [ ! -f "$checkpoint_path" ]; then
    echo "ERROR: Checkpoint file not found: $checkpoint_path"
    exit 1
fi

# Clean dataset type for file names
CLEAN_DATASET_TYPE=$(echo $dataset_type | tr ',' '-' | tr -d ' ')

# Get current date and time
CURRENT_DATETIME=$(date +"%d%m_%H%M")

# Generate run name (same logic as orchestrator_inference.py)
CHECKPOINT_NAME=$(basename "$checkpoint_path" | sed 's/\.pt$//' | sed 's/\.pth$//')
RUN_NAME="${CURRENT_DATETIME}_inference_${CHECKPOINT_NAME}_${CLEAN_DATASET_TYPE}"

# Set script path
SCRIPT_PATH="/data2/neeraja/neeraja/code/ICL/models/symbolAdapter/orchestrator_inference.py"
TODAY=$(date +"%Y-%m-%d")

# Create output directories
LOG_DIR="/data2/neeraja/neeraja/results/model_ICL/orchestrator_logs/${TODAY}"
mkdir -p "$LOG_DIR"

# Remove old log file if exists
rm -f "${LOG_DIR}/${RUN_NAME}.log"

# ========================================
# Display Configuration
# ========================================
echo "=========================================="
echo "Orchestrator Inference Job Configuration"
echo "=========================================="
echo "Run Name: ${RUN_NAME}"
echo "Checkpoint: ${checkpoint_path}"
echo "Dataset: ${dataset_type}"
echo "Max Samples: ${max_val_samples}"
echo "Device: ${device}"
echo "Output Dir: ${output_dir}"
echo "Log File: ${LOG_DIR}/${RUN_NAME}.log"
echo "Queue: ${queue_name}"
echo "Hostname: ${hostname}"
echo "CUDA Device: ${cuda_device}"
echo "=========================================="

# ========================================
# Submit Job
# ========================================
JOB_ID=$(qsub -q ${queue_name} -V -cwd \
    -l hostname=${hostname} \
    -l h_rt=24:00:00 \
    -o "${LOG_DIR}/${RUN_NAME}.log" \
    -j y \
    -v CUDA_VISIBLE_DEVICES=${cuda_device},\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
SCRIPT_PATH=${SCRIPT_PATH},\
checkpoint_path=${checkpoint_path},\
dataset_type=${dataset_type},\
max_val_samples=${max_val_samples},\
device=${device},\
output_dir=${output_dir} \
    -S /bin/bash << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "Starting Orchestrator Inference Job"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Python path: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo ""

echo "Environment variables:"
env | grep -E "(RUN_NAME|checkpoint_path|dataset_type|max_val_samples|device|output_dir)" | sort
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /home/share/anaconda3/etc/profile.d/conda.sh
conda activate salmon

echo "Conda environment: $(conda info --envs | grep '*')"
echo "Python version: $(python --version)"
echo ""

# Validate inputs
echo "Validating inputs..."
if [ ! -f "${checkpoint_path}" ]; then
    echo "ERROR: Checkpoint file not found: ${checkpoint_path}"
    exit 1
fi
echo "✅ Checkpoint file exists"

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script file not found: ${SCRIPT_PATH}"
    exit 1
fi
echo "✅ Script file exists"

echo ""
echo "=========================================="
echo "Running Orchestrator Inference"
echo "=========================================="

# Run inference with detailed logging
python ${SCRIPT_PATH} \
    --checkpoint_path "${checkpoint_path}" \
    --dataset_type "${dataset_type}" \
    --device "${device}" \
    --max_val_samples ${max_val_samples} \
    --output_dir "${output_dir}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job Completion"
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Orchestrator inference completed successfully!"
else
    echo "❌ Orchestrator inference failed with exit code: ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
EOF
)

# ========================================
# Report Job Submission
# ========================================
JOB_ID_NUM=$(echo $JOB_ID | cut -d' ' -f3)

echo ""
echo "=========================================="
echo "Job Submitted Successfully"
echo "=========================================="
echo "Job Name: ${RUN_NAME}"
echo "Job ID: ${JOB_ID_NUM}"
echo "Queue: ${queue_name}"
echo "Host: ${hostname}"
echo ""
echo "Monitor commands:"
echo "  tail -f ${LOG_DIR}/${RUN_NAME}.log"
echo "  qstat | grep ${JOB_ID_NUM}"
echo "  qstat -j ${JOB_ID_NUM}"
echo ""
echo "Results will be saved to:"
echo "  ${output_dir}/orchestrator_metrics/"
echo "  ${output_dir}/orchestrator_logs/"
echo "=========================================="