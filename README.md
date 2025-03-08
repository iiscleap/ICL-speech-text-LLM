# ICL Framework

A high-performance framework for In-Context Learning (ICL) with audio and text models.

## Overview

This framework provides tools for training and evaluating ICL models on various audio and text datasets. It supports both single-task and multi-task learning, with a focus on performance and scalability.

## Features

- **Multi-modal Support**: Process both audio and text inputs
- **Multi-task Learning**: Train on multiple datasets simultaneously
- **High Performance**: Optimized for speed and memory efficiency
- **Flexible Architecture**: Support for different model types (SALMONN, Qwen2)
- **Distributed Training**: Scale across multiple GPUs
- **Comprehensive Evaluation**: Detailed metrics and result analysis

## Directory Structure

```
code/ICL/
├── config/                 # Configuration files
├── data/                   # Dataset handling
├── models/                 # Model implementations
├── train/                  # Training scripts
├── inference/              # Inference scripts
├── utils/                  # Utility functions
├── scripts/                # Job submission scripts
├── OPTIMIZATIONS.md        # Performance optimization details
└── README.md               # This file
```

## Quick Start

### Training

```bash
# Single-task training
python train/train.py \
  --model_path /path/to/model \
  --output_dir ./output \
  --log_file ./logs/train.log \
  --dataset_type VOXCELEB \
  --input_mode speech_only \
  --fewshot_mode text \
  --num_examples 5 \
  --batch_size 8 \
  --fp16

# Multi-task training
python train/train.py \
  --model_path /path/to/model \
  --output_dir ./output \
  --log_file ./logs/train.log \
  --dataset_type VOXCELEB,HVB \
  --input_mode speech_only \
  --fewshot_mode text \
  --num_examples 5 \
  --batch_size 8 \
  --fp16
```

### Inference

```bash
# Single-task inference
python inference/inference.py \
  --base_model_path /path/to/base_model \
  --peft_model_path /path/to/fine_tuned_model \
  --run_name test_run \
  --today 2023-03-03 \
  --dataset_type VOXCELEB \
  --input_mode speech_only \
  --fewshot_mode text \
  --num_examples 5 \
  --optimize_batch_size \
  --fp16

# Multi-task inference
python inference/inference.py \
  --base_model_path /path/to/base_model \
  --peft_model_path /path/to/fine_tuned_model \
  --run_name test_run \
  --today 2023-03-03 \
  --dataset_type VOXCELEB,HVB \
  --input_mode speech_only \
  --fewshot_mode text \
  --num_examples 5 \
  --optimize_batch_size \
  --fp16 \
  --save_per_dataset
```

### Using Job Submission Scripts

```bash
# Submit a training job
./scripts/submit_train_job.sh

# Submit an inference job
./scripts/submit_inference_job.sh

# Submit a multi-task training job
./scripts/submit_multi_task_job.sh

# Submit a multi-task inference job
./scripts/submit_multi_task_inference_job.sh
```

## Performance Optimizations

This framework includes numerous optimizations for faster training and inference:

- **Mixed Precision Training**: FP16/BF16 support for faster computation
- **Gradient Checkpointing**: Reduced memory usage for large models
- **Dataset Caching**: Faster data loading with in-memory caching
- **Automatic Batch Size Optimization**: Find the optimal batch size for your hardware
- **Model Caching**: Avoid reloading the same model multiple times
- **Optimized Data Loading**: Parallel processing and efficient memory usage
- **Performance Monitoring**: Track training speed, memory usage, and throughput

For detailed information about performance optimizations, see [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

## Configuration

### Training Configuration

Training parameters can be configured in `config/training_config.py`. Key parameters include:

- Learning rate and scheduler settings
- Optimizer parameters
- Model-specific configurations
- Dataset-specific settings

### Inference Configuration

Inference parameters can be configured in `config/inference_config.py`. Key parameters include:

- Generation parameters
- Model-specific configurations
- Dataset-specific settings

## Supported Models

- **SALMONN**: Speech Audio Language Music Open Neural Network
- **Qwen2**: Qwen2 language model

## Supported Datasets

- **VOXCELEB**: Speaker verification dataset
- **HVB**: Human Voice Bank dataset
- **VOXPOPULI**: Multi-lingual speech dataset

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.25+
- CUDA 11.6+ (for GPU acceleration)

## License

[MIT License](LICENSE)

## Citation

If you use this framework in your research, please cite:

```
@misc{icl-framework,
  author = {Your Name},
  title = {ICL Framework: A High-Performance Framework for In-Context Learning},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/icl-framework}
}
``` 