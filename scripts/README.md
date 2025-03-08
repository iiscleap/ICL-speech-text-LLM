# ICL Framework Job Scripts

This directory contains scripts for running training and inference jobs for the In-Context Learning (ICL) framework. The scripts are designed to be used with the SGE job scheduler (qsub).

## Available Scripts

### Training Scripts

1. **train.sh**: The base script for running training. This is called by the job submission scripts.
2. **submit_train_job.sh**: Script for submitting a single-task training job.
3. **submit_multi_task_job.sh**: Script for submitting a multi-task training job.

### Inference Scripts

1. **inference.sh**: The base script for running inference. This is called by the job submission scripts.
2. **submit_inference_job.sh**: Script for submitting a single-task inference job.
3. **submit_multi_task_inference_job.sh**: Script for submitting a multi-task inference job.

## Usage

### Single-Task Training

To submit a single-task training job, edit the configuration parameters in `submit_train_job.sh` and then run:

```bash
bash submit_train_job.sh
```

### Multi-Task Training

To submit a multi-task training job, edit the configuration parameters in `submit_multi_task_job.sh` and then run:

```bash
bash submit_multi_task_job.sh
```

### Single-Task Inference

To submit a single-task inference job, edit the configuration parameters in `submit_inference_job.sh` and then run:

```bash
bash submit_inference_job.sh
```

### Multi-Task Inference

To submit a multi-task inference job, edit the configuration parameters in `submit_multi_task_inference_job.sh` and then run:

```bash
bash submit_multi_task_inference_job.sh
```

## Important Notes

1. **Configuration**: Each script contains configuration parameters at the top that you should edit before running the script. These include:
   - `model_type`: The type of model to use (e.g., "salmonn" or "qwen2")
   - `dataset_type` or `dataset_types`: The dataset(s) to use
   - `input_mode`: The input mode (e.g., "speech_only", "text_only", "speech_and_text")
   - `fewshot_mode`: The few-shot mode (e.g., "text" or "speech")
   - `num_examples`: The number of few-shot examples to use
   - `batch_size`: The batch size for training or inference
   - Other parameters specific to training or inference

2. **Model Paths**: You need to update the model paths in the scripts to point to the correct locations on your system.

3. **Job Scheduler**: The scripts use the SGE job scheduler with `qsub`. You may need to adjust the job submission parameters based on your cluster configuration.

4. **Output Directories**: The scripts create output directories for logs and results. Make sure these directories exist or have appropriate permissions.

5. **Conda Environment**: The scripts set the conda environment based on the model type. Make sure these environments exist on your system.

## Example

Here's an example of how to use the scripts:

1. Edit `submit_train_job.sh` to set the appropriate configuration parameters.
2. Run `bash submit_train_job.sh` to submit the training job.
3. Once training is complete, edit `submit_inference_job.sh` to set the appropriate configuration parameters, including the path to the trained model.
4. Run `bash submit_inference_job.sh` to submit the inference job. 