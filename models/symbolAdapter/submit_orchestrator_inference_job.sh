#!/bin/bash
# filepath: /data2/neeraja/neeraja/code/ICL/models/symbolAdapter/submit_orchestrator_inference_job.sh

# ========================================
# Configuration - Edit these values as needed
# ========================================

#vox + vop
# checkpoint_path="/data1/chandnia/neeraja/results/model_ICL/orchestrator_training/checkpoints/0507_1713_orchestrator__1c_8le_1me_bypass_mlp_sym_salmonn_voxceleb_voxpopuli/lora_step0_cycle0_epoch5_periodic.pt"
# checkpoint_path="/data1/chandnia/neeraja/results/model_ICL/orchestrator_training/checkpoints/0507_1713_orchestrator__1c_8le_1me_bypass_mlp_sym_salmonn_voxceleb_voxpopuli/lora_step0_cycle0_epoch2_periodic.pt"
#vox + hvb
# checkpoint_path="/data1/chandnia/neeraja/results/model_ICL/orchestrator_training/checkpoints/0507_1713_orchestrator__1c_8le_1me_bypass_mlp_sym_salmonn_voxceleb_hvb/lora_step0_cycle0_epoch2_periodic.pt"
#meld + hvb
# checkpoint_path="/data1/chandnia/neeraja/results/model_ICL/orchestrator_training/checkpoints/0507_1712_orchestrator__1c_8le_1me_bypass_mlp_sym_salmonn_meld_emotion_hvb/lora_step0_cycle0_epoch3_periodic.pt"
checkpoint_path="/data1/chandnia/neeraja/results/model_ICL/orchestrator_training/checkpoints/0507_1712_orchestrator__1c_8le_1me_bypass_mlp_sym_salmonn_meld_emotion_hvb/lora_step0_cycle0_epoch2_periodic.pt"

dataset_type="hvb-voxceleb-voxpopuli-meld_emotion"  # Dataset type to evaluate on
max_val_samples=0          # 0 = use all samples

num_examples=0

# Optional parameters
device="cuda:0"
output_dir="/data1/chandnia/neeraja/results/model_ICL"

# Node configuration
queue_name="longgpu.q"
hostname="compute-0-9"
cuda_device=2

hold_job_id=""
# ========================================
# Validation and Setup
# ========================================

# Set conda environment
export CONDA_ENV="salmon"
echo "Set conda environment to: $CONDA_ENV"
source /home/share/anaconda3/etc/profile.d/conda.sh  
conda deactivate
conda activate $CONDA_ENV   


if [ -n "$hold_job_id" ]; then
    echo "Job will wait for completion of job: $hold_job_id"
    HOLD_FLAG="-hold_jid $hold_job_id"
else
    HOLD_FLAG=""
fi




# Check if checkpoint exists
if [ ! -f "$checkpoint_path" ]; then
    echo "ERROR: Checkpoint file not found: $checkpoint_path"
    exit 1
fi

# Clean dataset type for file names
CLEAN_DATASET_TYPE=$(echo $dataset_type | tr ',' '-' | tr -d ' ')

# Get current date and time
CURRENT_DATETIME=$(date +"%d%m_%H%M")


# Extract training timestamp, dataset, and epoch from checkpoint path
CHECKPOINT_DIR=$(basename "$(dirname "$checkpoint_path")")
TRAINING_TIMESTAMP=$(echo "$CHECKPOINT_DIR" | cut -d'_' -f1-2)
EPOCH_NUM=$(basename "$checkpoint_path" | sed -n 's/.*epoch\([0-9]\+\).*/\1/p')

# ✅ IMPROVED: Extract everything after "salmonn_" to get full training dataset
TRAINING_DATASET=$(echo "$CHECKPOINT_DIR" | sed 's/.*salmonn_//')

clean_dataset_name() {
    local dataset=$1
    dataset=$(echo "$dataset" | sed 's/voxceleb/vox/g')
    dataset=$(echo "$dataset" | sed 's/voxpopuli/vop/g') 
    dataset=$(echo "$dataset" | sed 's/meld_emotion/meld/g')
    dataset=$(echo "$dataset" | sed 's/_/-/g')  # Replace underscores with hyphens
    echo "$dataset"
}

TRAINING_DATASET_CLEAN=$(clean_dataset_name "$TRAINING_DATASET")
CLEAN_DATASET_TYPE_COMPACT=$(clean_dataset_name "$CLEAN_DATASET_TYPE")
# Create compact checkpoint identifier with training dataset
CHECKPOINT_NAME="${TRAINING_TIMESTAMP}_ep${EPOCH_NUM}_${TRAINING_DATASET_CLEAN}"

# Update RUN_NAME
RUN_NAME="${CURRENT_DATETIME}_infer_${CHECKPOINT_NAME}_on_${CLEAN_DATASET_TYPE_COMPACT}_${num_examples}ex"





# Set script path
SCRIPT_PATH="/data2/neeraja/neeraja/code/ICL/models/symbolAdapter/orchestrator_inference.py"
TODAY=$(date +"%Y-%m-%d")

# Create output directories
# LOG_DIR="/data2/neeraja/neeraja/results/model_ICL/orchestrator_logs/${TODAY}"
LOG_DIR="/data1/chandnia/neeraja/results/model_ICL/orchestrator_logs/${TODAY}"

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
    ${HOLD_FLAG} \
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
num_examples=${num_examples},\
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
    --num_examples ${num_examples} \
    --output_dir "${output_dir}"\
    --run_name "${RUN_NAME}"

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