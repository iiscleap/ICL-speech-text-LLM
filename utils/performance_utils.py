import os
import time
import logging
import traceback
import psutil
import torch
import gc
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Utility class to track performance metrics during training and inference"""
    
    def __init__(self, log_interval: int = 100, logger=None):
        """
        Initialize the performance tracker.
        
        Args:
            log_interval: How often to log metrics (in steps)
            logger: Logger to use for logging
        """
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger(__name__)
        self.reset()
    
    def reset(self):
        """Reset all tracking metrics"""
        self.step_times = []
        self.batch_sizes = []
        self.loss_values = []
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.total_examples = 0
        self.total_tokens = 0
        self.step_count = 0
    
    def update(self, step_time: float, batch_size: int, loss: Optional[float] = None, token_count: Optional[int] = None):
        """
        Update tracking metrics with a new step.
        
        Args:
            step_time: Time taken for the step in seconds
            batch_size: Batch size for the step
            loss: Loss value for the step (optional)
            token_count: Number of tokens processed in the step (optional)
        """
        self.step_times.append(step_time)
        self.batch_sizes.append(batch_size)
        if loss is not None:
            self.loss_values.append(loss)
        
        self.total_examples += batch_size
        if token_count is not None:
            self.total_tokens += token_count
        
        self.step_count += 1
        
        # Log if interval is reached
        if self.step_count % self.log_interval == 0:
            self.log_metrics()
    
    def log_metrics(self):
        """Log current performance metrics"""
        current_time = time.time()
        time_elapsed = current_time - self.last_log_time
        
        # Calculate metrics
        avg_step_time = np.mean(self.step_times[-self.log_interval:])
        examples_per_second = sum(self.batch_sizes[-self.log_interval:]) / time_elapsed
        
        metrics = {
            "avg_step_time": f"{avg_step_time:.4f}s",
            "examples_per_second": f"{examples_per_second:.2f}",
            "total_examples": self.total_examples
        }
        
        # Add loss if available
        if self.loss_values:
            metrics["avg_loss"] = f"{np.mean(self.loss_values[-self.log_interval:]):.4f}"
        
        # Add tokens per second if available
        if self.total_tokens > 0:
            tokens_per_second = sum(self.batch_sizes[-self.log_interval:]) * (self.total_tokens / self.total_examples) / time_elapsed
            metrics["tokens_per_second"] = f"{tokens_per_second:.2f}"
        
        # Log metrics
        self.logger.info(f"Performance metrics: {metrics}")
        
        # Update last log time
        self.last_log_time = current_time
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        total_time = time.time() - self.start_time
        
        summary = {
            "total_time": f"{total_time:.2f}s",
            "total_examples": self.total_examples,
            "avg_step_time": f"{np.mean(self.step_times):.4f}s",
            "examples_per_second": f"{self.total_examples / total_time:.2f}",
            "step_count": self.step_count
        }
        
        # Add loss if available
        if self.loss_values:
            summary["avg_loss"] = f"{np.mean(self.loss_values):.4f}"
        
        # Add tokens per second if available
        if self.total_tokens > 0:
            summary["tokens_per_second"] = f"{self.total_tokens / total_time:.2f}"
            summary["total_tokens"] = self.total_tokens
        
        return summary
    
    def log_summary(self):
        """Log a summary of all performance metrics"""
        summary = self.get_summary()
        self.logger.info(f"Performance summary: {summary}")


@contextmanager
def timer(name: str = None, logger=None):
    """
    Context manager for timing code blocks.
    
    Args:
        name: Name of the timed section
        logger: Logger to use for logging
    
    Yields:
        None
    """
    logger = logger or logging.getLogger(__name__)
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        logger.info(f"{name} completed in {elapsed:.4f}s")
    else:
        logger.info(f"Operation completed in {elapsed:.4f}s")


def time_function(func: Callable) -> Callable:
    """
    Decorator to time a function's execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"Function '{func.__name__}' failed after {execution_time:.4f} seconds: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    return wrapper


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage statistics
    """
    # System memory
    system_memory = psutil.virtual_memory()
    
    memory_stats = {
        "system": {
            "total_gb": system_memory.total / (1024 ** 3),
            "available_gb": system_memory.available / (1024 ** 3),
            "used_gb": system_memory.used / (1024 ** 3),
            "percent": system_memory.percent
        }
    }
    
    # GPU memory if available
    if torch.cuda.is_available():
        gpu_stats = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            gpu_stats[f"gpu_{i}"] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "utilization": torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else None
            }
        memory_stats["gpu"] = gpu_stats
    
    # Process memory
    process = psutil.Process(os.getpid())
    memory_stats["process"] = {
        "rss_gb": process.memory_info().rss / (1024 ** 3),
        "vms_gb": process.memory_info().vms / (1024 ** 3)
    }
    
    return memory_stats


def log_memory_usage(logger=None):
    """
    Log current memory usage statistics.
    
    Args:
        logger: Logger to use for logging
    """
    logger = logger or logging.getLogger(__name__)
    memory_stats = get_memory_usage()
    
    # Log system memory
    system = memory_stats["system"]
    logger.info(f"System memory: {system['used_gb']:.2f}GB used / {system['total_gb']:.2f}GB total ({system['percent']}%)")
    
    # Log process memory
    process = memory_stats["process"]
    logger.info(f"Process memory: {process['rss_gb']:.2f}GB RSS, {process['vms_gb']:.2f}GB VMS")
    
    # Log GPU memory if available
    if "gpu" in memory_stats:
        for gpu_id, gpu in memory_stats["gpu"].items():
            logger.info(f"{gpu_id.upper()}: {gpu['allocated_gb']:.2f}GB allocated, {gpu['reserved_gb']:.2f}GB reserved")
            if gpu['utilization'] is not None:
                logger.info(f"{gpu_id.upper()} utilization: {gpu['utilization']}%")


def optimize_memory():
    """
    Optimize memory usage by clearing caches and running garbage collection.
    
    Returns:
        Dictionary with memory usage before and after optimization
    """
    # Get memory usage before optimization
    before = get_memory_usage()
    
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    
    # Run garbage collection
    gc.collect()
    
    # Get memory usage after optimization
    after = get_memory_usage()
    
    # Calculate difference
    diff = {}
    if "gpu" in before and "gpu" in after:
        for gpu_id in before["gpu"]:
            if gpu_id in after["gpu"]:
                allocated_diff = before["gpu"][gpu_id]["allocated_gb"] - after["gpu"][gpu_id]["allocated_gb"]
                reserved_diff = before["gpu"][gpu_id]["reserved_gb"] - after["gpu"][gpu_id]["reserved_gb"]
                diff[gpu_id] = {
                    "allocated_gb_freed": allocated_diff,
                    "reserved_gb_freed": reserved_diff
                }
    
    return {
        "before": before,
        "after": after,
        "diff": diff
    }


def log_optimization_results(results: Dict[str, Any], logger=None):
    """
    Log memory optimization results.
    
    Args:
        results: Results from optimize_memory()
        logger: Logger to use for logging
    """
    logger = logger or logging.getLogger(__name__)
    
    logger.info("Memory optimization results:")
    
    # Log GPU memory differences
    if "diff" in results:
        for gpu_id, gpu_diff in results["diff"].items():
            logger.info(f"{gpu_id.upper()}: Freed {gpu_diff['allocated_gb_freed']:.2f}GB allocated, {gpu_diff['reserved_gb_freed']:.2f}GB reserved")
    
    # Log system memory after optimization
    system = results["after"]["system"]
    logger.info(f"System memory after optimization: {system['used_gb']:.2f}GB used / {system['total_gb']:.2f}GB total ({system['percent']}%)")


def optimize_torch_settings():
    """
    Apply optimal PyTorch settings for performance.
    
    Returns:
        Dictionary with applied optimizations
    """
    optimizations = {}
    
    # Enable cuDNN benchmark mode for better performance with fixed input sizes
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        optimizations["cudnn_benchmark"] = True
    
    # Set optimal thread settings
    num_threads = min(os.cpu_count() or 4, 8)  # Limit to 8 threads max
    torch.set_num_threads(num_threads)
    optimizations["num_threads"] = num_threads
    
    # Set optimal interop threads
    if hasattr(torch, 'set_num_interop_threads'):
        interop_threads = min(os.cpu_count() or 2, 4)  # Limit to 4 interop threads max
        torch.set_num_interop_threads(interop_threads)
        optimizations["interop_threads"] = interop_threads
    
    return optimizations


def log_system_info():
    """
    Log system information including CPU, memory, and GPU details.
    """
    try:
        # CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024 ** 3)
        memory_available_gb = memory.available / (1024 ** 3)
        memory_used_gb = memory.used / (1024 ** 3)
        memory_percent = memory.percent
        
        # Log CPU and memory info
        logger.info(f"System Info - CPU: {cpu_count} physical cores, {cpu_count_logical} logical cores, {cpu_percent}% used")
        logger.info(f"System Info - Memory: {memory_total_gb:.2f}GB total, {memory_used_gb:.2f}GB used ({memory_percent}%), {memory_available_gb:.2f}GB available")
        
        # GPU info if available
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"System Info - GPU: {device_count} devices available")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_capability = torch.cuda.get_device_capability(i)
                logger.info(f"System Info - GPU {i}: {device_name}, Compute Capability {device_capability[0]}.{device_capability[1]}")
                
                # Log memory for each GPU
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
                logger.info(f"System Info - GPU {i} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        else:
            logger.info("System Info - GPU: No CUDA devices available")
    except Exception as e:
        logger.error(f"Error logging system information: {str(e)}")
        logger.debug(traceback.format_exc())


def clear_gpu_memory():
    """
    Clear GPU memory by emptying cache and running garbage collection.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not torch.cuda.is_available():
            logger.info("No CUDA devices available to clear memory")
            return True
        
        # Log memory before clearing
        logger.info("GPU memory before clearing:")
        log_gpu_memory_usage()
        
        # Empty cache and run garbage collection
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log memory after clearing
        logger.info("GPU memory after clearing:")
        log_gpu_memory_usage()
        
        return True
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def log_gpu_memory_usage(message: str = "Current GPU memory usage"):
    """
    Log the current GPU memory usage.
    
    Args:
        message: Message prefix for the log
    """
    try:
        if not torch.cuda.is_available():
            logger.info(f"{message}: CUDA not available")
            return
        
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
            logger.info(f"{message} - GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception as e:
        logger.error(f"Error logging GPU memory usage: {str(e)}")
        logger.debug(traceback.format_exc())


def log_cpu_memory_usage(message: str = "Current CPU memory usage"):
    """
    Log the current CPU memory usage.
    
    Args:
        message: Message prefix for the log
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Convert to GB for readability
        rss_gb = memory_info.rss / (1024 ** 3)
        vms_gb = memory_info.vms / (1024 ** 3)
        
        logger.info(f"{message}: RSS {rss_gb:.2f}GB, VMS {vms_gb:.2f}GB")
    except Exception as e:
        logger.error(f"Error logging CPU memory usage: {str(e)}")
        logger.debug(traceback.format_exc())


class BatchSizeOptimizer:
    """Utility class to find the optimal batch size for training or inference"""
    
    def __init__(
        self,
        model,
        sample_batch_fn: Callable[[int], Dict[str, Any]],
        forward_fn: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        max_memory_usage: float = 0.8,
        logger=None
    ):
        """
        Initialize the batch size optimizer.
        
        Args:
            model: Model to optimize batch size for
            sample_batch_fn: Function that returns a sample batch of given size
            forward_fn: Function that runs the forward pass (defaults to model's forward method)
            min_batch_size: Minimum batch size to try
            max_batch_size: Maximum batch size to try
            max_memory_usage: Maximum memory usage as a fraction of available memory
            logger: Logger to use for logging
        """
        self.model = model
        self.sample_batch_fn = sample_batch_fn
        self.forward_fn = forward_fn or (lambda m, b: m(b))
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_memory_usage = max_memory_usage
        self.logger = logger or logging.getLogger(__name__)
        
        # Ensure model is on CUDA
        self.device = next(model.parameters()).device
        if not self.device.type == 'cuda':
            self.logger.warning("Model is not on CUDA, batch size optimization may not be accurate")
    
    def find_optimal_batch_size(self) -> int:
        """
        Find the optimal batch size for the model.
        
        Returns:
            Optimal batch size
        """
        self.logger.info(f"Finding optimal batch size between {self.min_batch_size} and {self.max_batch_size}")
        
        # Start with binary search
        low = self.min_batch_size
        high = self.max_batch_size
        optimal = low
        
        while low <= high:
            mid = (low + high) // 2
            
            # Try this batch size
            success, memory_info = self._try_batch_size(mid)
            
            if success:
                # This batch size works, try larger
                optimal = mid
                low = mid + 1
                self.logger.info(f"Batch size {mid} works, trying larger batch size")
                self.logger.debug(f"Memory usage: {memory_info}")
            else:
                # This batch size is too large, try smaller
                high = mid - 1
                self.logger.info(f"Batch size {mid} failed, trying smaller batch size")
        
        # Fine-tune with linear search
        for bs in range(optimal + 1, min(optimal + 4, self.max_batch_size + 1)):
            success, memory_info = self._try_batch_size(bs)
            if success:
                optimal = bs
                self.logger.info(f"Batch size {bs} works, updating optimal")
                self.logger.debug(f"Memory usage: {memory_info}")
            else:
                break
        
        self.logger.info(f"Optimal batch size: {optimal}")
        return optimal
    
    def _try_batch_size(self, batch_size: int) -> Tuple[bool, Dict[str, float]]:
        """
        Try a specific batch size and check if it works.
        
        Args:
            batch_size: Batch size to try
            
        Returns:
            Tuple of (success, memory_info)
        """
        # Clear cache before trying
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Get a sample batch
            batch = self.sample_batch_fn(batch_size)
            
            # Move batch to device if needed
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            
            # Run forward pass
            with torch.no_grad():
                self.forward_fn(self.model, batch)
            
            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            
            # Check if memory usage is acceptable
            if memory_allocated / total_memory > self.max_memory_usage:
                return False, {"allocated_gb": memory_allocated, "reserved_gb": memory_reserved, "total_gb": total_memory}
            
            return True, {"allocated_gb": memory_allocated, "reserved_gb": memory_reserved, "total_gb": total_memory}
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                self.logger.debug(f"CUDA OOM for batch size {batch_size}")
                return False, {}
            else:
                self.logger.error(f"Error trying batch size {batch_size}: {e}")
                return False, {}
        except Exception as e:
            self.logger.error(f"Unexpected error trying batch size {batch_size}: {e}")
            return False, {} 