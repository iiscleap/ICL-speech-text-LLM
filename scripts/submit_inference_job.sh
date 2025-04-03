#!/bin/bash

# Configuration - Edit these values as needed
model_type="qwen2"  # Options: "salmonn" or "qwen2"
dataset_type="voxceleb"  # Options: "voxceleb", "hvb", "voxpopuli", etc., sqa, vp_nel
input_mode="speech_only"  # Options: "speech_only", "text_only", "speech_and_text"
fewshot_mode="text"  # Options: "text" or "speech"
num_examples=0
batch_size=1

debug_samples=10  # Add debug_samples parameter (0 = use all samples)

# Add near the top with other configuration options
randomize_swap=false  # Set to true to randomize swap configurations

# Node configuration
queue_name="long.q"      # Queue to submit job to (gpu.q, med.q, etc.)
hostname="compute-0-8"  # Hostname to run on
cuda_device=1         # CUDA device to use
hold_job_id=""          # Job ID to wait for (empty = don't wait)


# Path to the trained model - REQUIRED
# peft_model_path=""

peft_model_path="/data2/neeraja/neeraja/code/SALMONN/results/trained_models/ft_20e8b_qwen2_speech_text_voxceleb/final_model.pt"


# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_swap_symbol/final_model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_swap/final_model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/1803_1820_ft_5ex_10e8b_salmonn_speech_only_text_voxceleb/final_model.pt"


# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_hvb_swap/checkpoints/epoch_10_loss_0.0046/model.pt"

# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb-hvb/final_model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_swap-hvb_swap/final_model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_greek-hvb_greek/final_model.pt"

# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb-hvb/checkpoints/epoch_10_loss_0.0060/model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_swap-hvb_swap/checkpoints/epoch_10_loss_0.0117/model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_greek-hvb_greek/checkpoints/epoch_10_loss_0.0055/model.pt"


# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/1503_0227_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb-hvb/checkpoints/epoch_10_loss_0.0060/model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/1503_0227_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_swap-hvb_swap/checkpoints/epoch_10_loss_0.0061/model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/1503_0227_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_greek-hvb_greek/checkpoints/epoch_10_loss_0.0055/model.pt"


# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/1503_0227_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb-hvb/checkpoints/epoch_15_loss_0.0002/model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/1503_0227_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_swap-hvb_swap/checkpoints/epoch_15_loss_0.0005/model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/1503_0227_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_greek-hvb_greek/checkpoints/epoch_15_loss_0.0005/model.pt"

# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/2303_0105_ft_5ex_10e8b_salmonn_speech_only_text_voxceleb-hvb/final_model.pt" #(this is with mix + vox training)

# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/2303_0046_ft_5ex_10e8b_salmonn_text_only_text_voxceleb-hvb/final_model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/2503_0016_ft_5ex_20e8b_salmonn_text_only_text_voxceleb_greek-hvb_greek/checkpoints/epoch_10_loss_0.0997/model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/2503_1535_ft_5ex_20e8b_salmonn_text_only_text_voxceleb_swap-hvb_swap/checkpoints/epoch_10_loss_0.1935/model.pt"


# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/2503_1539_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_swap-hvb_swap_ss/checkpoints/epoch_10_loss_0.0063/model.pt"

# peft_model_path="/data2/neeraja/neeraja/code/SALMONN/results/trained_models/INTERSPEECH/finetune_llama2_salmon_speech_15e8b_Q_voxceleb_swap/final_model.pt"

# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/2503_1539_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_swap-hvb_swap_ss/checkpoints/epoch_10_loss_0.0063/model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/3003_0050_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_swap_Inter/final_model.pt"

# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/3003_0157_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_swap-hvb_swap_NEIter/checkpoints/epoch_10_loss_0.0081/model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/3003_0201_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_swap-hvb_swap_BothShu/checkpoints/epoch_10_loss_0.0468/model.pt"
# peft_model_path="/data2/neeraja/neeraja/results/model_ICL/trained_models/3003_0201_ft_5ex_20e8b_salmonn_speech_only_text_voxceleb_swap-hvb_swap_BothShu/final_model.pt"

# Clean dataset type for file names and Python
if [[ $dataset_type == *","* ]]; then
    # For file names (replace commas with hyphens)
    CLEAN_DATASET_TYPE=$(echo $dataset_type | tr ',' '-' | tr -d ' ')
else
    # No cleaning needed
    CLEAN_DATASET_TYPE=$dataset_type
fi

# Check if model path is provided
echo "Checking model path: $peft_model_path"
if [ -z "$peft_model_path" ]; then
    echo "No PEFT model path provided, using empty string"
    peft_model_path=""
else
    if [ ! -f "$peft_model_path" ]; then
        echo "Error: PEFT model file does not exist at path: $peft_model_path"
        echo "Current directory: $(pwd)"
        echo "Please set peft_model_path in the script to point to a valid model file"
        exit 1
    fi
fi

# Set conda environment based on model type
if [ "$model_type" == "salmonn" ]; then
    export CONDA_ENV="salmon" # Update with actual path
elif [ "$model_type" == "qwen2" ]; then
    export CONDA_ENV="qwen2" # Update with actual path
else
    echo "Invalid model type. Please specify 'salmonn' or 'qwen2'"
    exit 1
fi

echo "Set conda environment to: $CONDA_ENV"
source /home/share/anaconda3/etc/profile.d/conda.sh  
conda deactivate
conda activate $CONDA_ENV   

# Get current date and time in DD_MM_HHMM format
CURRENT_DATETIME=$(date +"%d%m_%H%M")

# Extract run name from model path or set default
if [ -z "$peft_model_path" ]; then
    RUN_NAME="default"
else
    RUN_NAME=$(echo "$peft_model_path" | sed -n 's/.*trained_models\/\([^/]*\).*/\1/p')
    RUN_NAME=$(echo "$RUN_NAME" | sed 's/speech/sp/g; s/text/txt/g; s/salmonn/sal/g; s/qwen2/qw/g; s/voxceleb/vox/g')
    
    # Extract epoch number if path contains checkpoints
    if [[ $peft_model_path == *"checkpoints/epoch_"* ]]; then
        EPOCH_NUM=$(echo "$peft_model_path" | sed -n 's/.*epoch_\([0-9]*\)_.*/\1/p')
        RUN_NAME="${RUN_NAME}_e${EPOCH_NUM}"
    fi
fi

# Add datetime to the start of RUN_NAME
RUN_NAME="${CURRENT_DATETIME}_${RUN_NAME}"

# Set script path
SCRIPT_PATH="/data2/neeraja/neeraja/code/ICL/inference/inference.py"
TODAY=$(date +"%Y-%m-%d")

# Create output directories
mkdir -p "/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}"
mkdir -p "/data2/neeraja/neeraja/results/model_ICL/metrics/${TODAY}"

# Remove old log file
rm -f "/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}/${RUN_NAME}_${CLEAN_DATASET_TYPE}_${input_mode}_${fewshot_mode}_${num_examples}shots.log"

# Add debug prints before qsub
echo "Original dataset_type: ${dataset_type}"
echo "CLEAN_DATASET_TYPE: ${CLEAN_DATASET_TYPE}"
echo "Queue: ${queue_name}"
echo "Hostname: ${hostname}"
echo "CUDA device: ${cuda_device}"
echo "Hold job ID: ${hold_job_id}"

# Common qsub arguments
QSUB_ARGS="-q ${queue_name} -V -cwd -l hostname=${hostname} -l h_rt=72:00:00"
LOG_PATH="/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}/${RUN_NAME}_${CLEAN_DATASET_TYPE}_${input_mode}_${fewshot_mode}_${num_examples}shots.log"

# Create a job log file for the day (JSON format)
JOB_LOG_FILE="/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}/submitted_jobs.json"

# Create the JSON file if it doesn't exist
if [ ! -f "${JOB_LOG_FILE}" ]; then
    echo "[]" > "${JOB_LOG_FILE}"
fi

# Common variables for qsub
COMMON_VARS="CUDA_VISIBLE_DEVICES=${cuda_device},\
TODAY=${TODAY},\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
num_examples=${num_examples},\
peft_model_path=${peft_model_path},\
SCRIPT_PATH=${SCRIPT_PATH},\
input_mode=${input_mode},\
dataset_type=${CLEAN_DATASET_TYPE},\
fewshot_mode=${fewshot_mode},\
model_type=${model_type},\
batch_size=${batch_size},\
num_workers=${num_workers},\
seed=${seed},\
debug_samples=${debug_samples},\
randomize_swap=${randomize_swap},\
output_suffix="

# Conditionally add hold_jid if provided
if [ -n "$hold_job_id" ]; then
    echo "Submitting job to queue ${queue_name} with dependency on job ID: ${hold_job_id}"
    JOB_ID=$(qsub ${QSUB_ARGS} -hold_jid ${hold_job_id} -o "${LOG_PATH}" -j y \
    -v ${COMMON_VARS} \
    -S /bin/bash /data2/neeraja/neeraja/code/ICL/scripts/inference.sh)
else
    echo "Submitting job to queue ${queue_name} without dependencies"
    JOB_ID=$(qsub ${QSUB_ARGS} -o "${LOG_PATH}" -j y \
    -v ${COMMON_VARS} \
    -S /bin/bash /data2/neeraja/neeraja/code/ICL/scripts/inference.sh)
fi

# Extract just the job ID number
JOB_ID_NUM=$(echo $JOB_ID | cut -d' ' -f3)

# Create a JSON entry for this job
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
JOB_ENTRY="{
    \"timestamp\": \"${TIMESTAMP}\",
    \"job_id\": \"${JOB_ID_NUM}\",
    \"dataset\": \"${CLEAN_DATASET_TYPE}\",
    \"input_mode\": \"${input_mode}\",
    \"fewshot_mode\": \"${fewshot_mode}\",
    \"model_type\": \"${model_type}\",
    \"num_examples\": ${num_examples},
    \"queue\": \"${queue_name}\",
    \"hostname\": \"${hostname}\",
    \"cuda_device\": ${cuda_device},
    \"hold_job_id\": \"${hold_job_id}\",
    \"debug_samples\": ${debug_samples},
    \"run_name\": \"${RUN_NAME}\",
    \"model_path\": \"${peft_model_path}\"
}"

# Append the job entry to the JSON file (requires jq)
# First, read the current JSON array
CURRENT_JOBS=$(cat "${JOB_LOG_FILE}")
# Then append the new job and write back
echo "${CURRENT_JOBS}" | jq ". += [${JOB_ENTRY}]" > "${JOB_LOG_FILE}"

echo "Submitted inference job for ${RUN_NAME} to queue ${queue_name}"
echo "Job ID: ${JOB_ID_NUM}"
echo "Job details logged to: ${JOB_LOG_FILE}"

