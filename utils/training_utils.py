import os
import logging
import torch
import time
from typing import Dict, Any, Optional, List, Union
import traceback

logger = logging.getLogger(__name__)

def setup_logging(log_file: str):
    """
    Setup logging to file and console with timestamps.
    
    Args:
        log_file: Path to the log file
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """
    Save checkpoint with only trainable parameters.
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        scheduler: The scheduler state to save
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save the checkpoint
    """
    # Get state dict
    full_state_dict = model.state_dict()
    
    # Create a new state dict with only trainable parameters
    trainable_state_dict = {}
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_state_dict[name] = full_state_dict[name]
            trainable_params += param.numel()
    
    # Save checkpoint with only trainable parameters
    checkpoint = {
        "model_state_dict": trainable_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "loss": loss,
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, path)
    
    # Log savings
    logger = logging.getLogger(__name__)
    logger.info(f"Saved checkpoint to {path}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters saved: {trainable_params:,}")
    logger.info(f"Reduction in checkpoint size: {(1 - trainable_params/total_params)*100:.2f}%")

def load_checkpoint(path, map_location=None):
    """
    Load checkpoint and handle partial state dict (trainable parameters only).
    
    Args:
        path: Path to the checkpoint
        map_location: Device to load the checkpoint to
    
    Returns:
        dict: Loaded checkpoint
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint from {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    checkpoint = torch.load(path, map_location=map_location)
    
    # Log loaded checkpoint info
    if "model_state_dict" in checkpoint:
        num_params = sum(p.numel() for p in checkpoint["model_state_dict"].values())
        logger.info(f"Loaded checkpoint with {num_params:,} parameters")
    
    return checkpoint

def get_gpu_memory_usage():
    """
    Get GPU memory usage for all available GPUs.
    
    Returns:
        Dictionary mapping GPU indices to their memory usage in MB
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    memory_usage = {}
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)    # Convert to MB
        memory_usage[f"gpu_{i}"] = {
            "allocated_mb": memory_allocated,
            "reserved_mb": memory_reserved
        }
    
    return memory_usage

def log_gpu_memory_usage(logger):
    """
    Log GPU memory usage for all available GPUs.
    
    Args:
        logger: Logger to use for logging
    """
    memory_usage = get_gpu_memory_usage()
    if "error" in memory_usage:
        logger.info(f"GPU memory usage: {memory_usage['error']}")
        return
    
    for gpu, usage in memory_usage.items():
        logger.info(f"{gpu.upper()} memory usage: {usage['allocated_mb']:.2f}MB allocated, {usage['reserved_mb']:.2f}MB reserved")

def get_learning_rates(optimizer):
    """
    Get current learning rates from optimizer.
    
    Args:
        optimizer: The optimizer
        
    Returns:
        List of current learning rates
    """
    return [group['lr'] for group in optimizer.param_groups]
