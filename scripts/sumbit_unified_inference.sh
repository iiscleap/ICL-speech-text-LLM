#!/bin/bash

# Configuration - Edit these values as needed
model_type="salmonn"  # Options: "salmonn" or "qwen2"
dataset_type="voxceleb_greek-hvb_greek"  # Dataset(s) to use for inference
split="test"  # Dataset split: train, validation, test
checkpoint_path="/data2/neeraja/neeraja/results/model_ICL/unified_training/3105_0053_unified_4c_2le_1me_salmonn_voxceleb_greek_hvb_greek/cycle_3_lora_epoch_2/model.pt"  # Path to trained model checkpoint

# Inference parameters
batch_size=1
max_samples=0  # 0 = use all samples
max_length=512
temperature=0.7
top_p=0.9
do_sample=""  # Add "--do_sample" to enable sampling

# Hardware settings
device="cuda:0"
partition="gpu"
time_limit="04:00:00"  # 4 hours
memory="32G"
cpus_per_task=8
gpus_per_node=1

# Output settings
output_base_dir="/data2/neeraja/neeraja/results/model_ICL/unified_inference"
run_name=""  # Leave empty for auto-generated name

# Generate job name and directories
timestamp=$(date +"%m%d_%H%M")
if [ -z "$run_name" ]; then
    job_name="unified_inf_${timestamp}_${dataset_type//[-_]/_}"
    run_name="unified_inference_${timestamp}_${dataset_type//[-_]/_}"
else
    job_name="unified_inf_${run_name}"
fi

# Create output directories
output_dir="${output_base_dir}/${run_name}"
log_dir="${output_dir}/logs"
mkdir -p "$log_dir"

# Log file paths
log_file="${log_dir}/${job_name}.log"
error_file="${log_dir}/${job_name}.err"

echo "=== Unified Symbol Discovery Inference Job Submission ==="
echo "Job name: $job_name"
echo "Model type: $model_type"
echo "Dataset: $dataset_type"
echo "Split: $split"
echo "Checkpoint: $checkpoint_path"
echo "Output directory: $output_dir"
echo "Log file: $log_file"
echo ""

# Check if checkpoint exists
if [ ! -f "$checkpoint_path" ]; then
    echo "ERROR: Checkpoint file not found: $checkpoint_path"
    echo "Please verify the checkpoint path and try again."
    exit 1
fi

# Build the command
cmd="python /data2/neeraja/neeraja/code/ICL/models/unified_inference.py \
    --model_type $model_type \
    --device $device \
    --checkpoint_path $checkpoint_path \
    --batch_size $batch_size \
    --dataset_type $dataset_type \
    --split $split \
    --max_samples $max_samples \
    --max_length $max_length \
    --temperature $temperature \
    --top_p $top_p \
    $do_sample \
    --output_dir $output_base_dir \
    --run_name $run_name"

echo "Command to be executed:"
echo "$cmd"
echo ""

# Create SLURM job script
cat > "${log_dir}/job_script.sh" << EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --partition=$partition
#SBATCH --time=$time_limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --mem=$memory
#SBATCH --gres=gpu:$gpus_per_node
#SBATCH --output=$log_file
#SBATCH --error=$error_file

# Print job information
echo "=== Job Information ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job name: \$SLURM_JOB_NAME"
echo "Node: \$SLURM_NODELIST"
echo "Started at: \$(date)"
echo "Working directory: \$(pwd)"
echo ""

# Load environment
echo "=== Environment Setup ==="
source ~/.bashrc
conda activate salmon
echo "Conda environment: \$CONDA_DEFAULT_ENV"
echo "Python version: \$(python --version)"
echo "PyTorch version: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: \$(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Set environment variables
export PYTHONPATH="/data2/neeraja/neeraja/code/ICL:\$PYTHONPATH"
export HF_HOME="/data2/neeraja/neeraja/.cache/huggingface"
export TRANSFORMERS_CACHE="/data2/neeraja/neeraja/.cache/huggingface/transformers"

# Print resource usage before starting
echo "=== Resource Usage (Before) ==="
nvidia-smi
free -h
echo ""

# Change to working directory
cd /data2/neeraja/neeraja/code/ICL

# Run the inference
echo "=== Starting Unified Symbol Discovery Inference ==="
echo "Command: $cmd"
echo ""

$cmd

exit_code=\$?

# Print resource usage after completion
echo ""
echo "=== Resource Usage (After) ==="
nvidia-smi
free -h

echo ""
echo "=== Job Completion ==="
echo "Finished at: \$(date)"
echo "Exit code: \$exit_code"

if [ \$exit_code -eq 0 ]; then
    echo "✓ Inference completed successfully!"
    echo "Results saved to: $output_dir"
    
    # Print summary of results
    if [ -f "$output_dir/metrics.json" ]; then
        echo ""
        echo "=== Results Summary ==="
        python -c "
import json
try:
    with open('$output_dir/metrics.json', 'r') as f:
        metrics = json.load(f)
    print(f'Accuracy: {metrics.get(\"accuracy\", 0):.4f}')
    print(f'Correct: {metrics.get(\"correct\", 0)}/{metrics.get(\"total\", 0)}')
except Exception as e:
    print(f'Could not read metrics: {e}')
"
    fi
else
    echo "✗ Inference failed with exit code \$exit_code"
fi

exit \$exit_code
EOF

# Submit the job
echo "Submitting job to SLURM..."
job_id=$(sbatch "${log_dir}/job_script.sh" | awk '{print $4}')

if [ $? -eq 0 ]; then
    echo "✓ Job submitted successfully!"
    echo "Job ID: $job_id"
    echo "Monitor with: tail -f $log_file"
    echo "Check status: squeue -j $job_id"
    echo "Cancel job: scancel $job_id"
    echo ""
    echo "Results will be saved to: $output_dir"
else
    echo "✗ Job submission failed!"
    exit 1
fi

echo "Unified symbol discovery inference job submitted at $(date)"