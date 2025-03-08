import logging
import os
import torch
import time
import traceback
import gc
from typing import Dict, Any, Optional, List, Union, Tuple
from functools import lru_cache

from .custom_salmon import CustomSALMONN
from .multi_task_model import MultiTaskModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading models
_MODEL_CACHE = {}

class ModelFactory:
    """
    Factory class for creating model instances.
    Supports both single-task and multi-task models for SALMONN and Qwen2.
    """
    
    @staticmethod
    def create_model(
        model_type: str,
        multi_task: bool = False,
        task_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        default_task: Optional[str] = None,
        use_cache: bool = True,
        device: Optional[torch.device] = None,
        **model_kwargs
    ):
        """
        Create a model instance based on configuration.
        
        Args:
            model_type: Type of model ("salmonn" or "qwen2")
            multi_task: Whether to create a multi-task model
            task_configs: Dictionary of task configurations for multi-task models
            default_task: Default task for multi-task models
            use_cache: Whether to use the model cache
            device: Device to load the model on
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            Instantiated model
            
        Raises:
            ValueError: If model_type is unknown or task_configs is missing for multi-task models
            RuntimeError: If model creation fails
        """
        try:
            model_type = model_type.lower()
            
            if model_type not in ["salmonn", "qwen2"]:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Generate a cache key based on model parameters
            cache_key = ModelFactory._generate_cache_key(
                model_type=model_type,
                multi_task=multi_task,
                task_configs=task_configs,
                default_task=default_task,
                **model_kwargs
            )
            
            # Check if model is in cache
            if use_cache and cache_key in _MODEL_CACHE:
                logger.info(f"Using cached {model_type} model")
                model = _MODEL_CACHE[cache_key]
                
                # Move to specified device if needed
                if device is not None and next(model.parameters()).device != device:
                    logger.info(f"Moving cached model to device: {device}")
                    model = model.to(device)
                
                return model
                
            # Create new model
            start_time = time.time()
            
            if multi_task:
                if not task_configs:
                    raise ValueError("task_configs required for multi-task models")
                    
                logger.info(f"Creating multi-task {model_type} model")
                model = MultiTaskModel(
                    model_type=model_type,
                    task_configs=task_configs,
                    default_task=default_task,
                    **model_kwargs
                )
            else:
                logger.info(f"Creating single-task {model_type} model")
                if model_type == "salmonn":
                    model = CustomSALMONN(**model_kwargs)
                else:
                    from .custom_qwen import CustomQwen
                    model = CustomQwen(**model_kwargs)
            
            # Move to specified device if needed
            if device is not None:
                logger.info(f"Moving model to device: {device}")
                model = model.to(device)
            
            # Cache the model if requested
            if use_cache:
                logger.debug(f"Caching model with key: {cache_key}")
                _MODEL_CACHE[cache_key] = model
            
            creation_time = time.time() - start_time
            logger.info(f"Model created in {creation_time:.2f}s")
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to create model: {str(e)}") from e
    
    @staticmethod
    def from_config(config: Dict[str, Any], device: Optional[torch.device] = None, use_cache: bool = True):
        """
        Create a model from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            device: Device to load the model on
            use_cache: Whether to use the model cache
            
        Returns:
            Instantiated model
            
        Raises:
            ValueError: If required configuration keys are missing
            RuntimeError: If model creation fails
        """
        try:
            # Extract model type
            model_type = config.get("model_type")
            if not model_type:
                raise ValueError("model_type not specified in config")
            
            # Check if this is a multi-task model
            multi_task = config.get("multi_task", False)
            
            # Extract task configs for multi-task models
            task_configs = config.get("task_configs")
            default_task = config.get("default_task")
            
            if multi_task and not task_configs:
                raise ValueError("task_configs required for multi-task models")
            
            # Extract model-specific parameters
            model_params = config.get("model_params", {})
            
            # Create the model
            return ModelFactory.create_model(
                model_type=model_type,
                multi_task=multi_task,
                task_configs=task_configs,
                default_task=default_task,
                use_cache=use_cache,
                device=device,
                **model_params
            )
            
        except Exception as e:
            logger.error(f"Error creating model from config: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to create model from config: {str(e)}") from e
    
    @staticmethod
    def _generate_cache_key(model_type: str, multi_task: bool, task_configs: Optional[Dict], default_task: Optional[str], **kwargs) -> str:
        """
        Generate a cache key for a model based on its parameters.
        
        Args:
            model_type: Type of model
            multi_task: Whether this is a multi-task model
            task_configs: Task configurations for multi-task models
            default_task: Default task for multi-task models
            **kwargs: Additional model parameters
            
        Returns:
            Cache key string
        """
        # Start with model type and multi-task flag
        key_parts = [f"model_type={model_type}", f"multi_task={multi_task}"]
        
        # Add task configs for multi-task models
        if multi_task and task_configs:
            task_names = sorted(task_configs.keys())
            key_parts.append(f"tasks={','.join(task_names)}")
            
            if default_task:
                key_parts.append(f"default_task={default_task}")
        
        # Add important model parameters
        important_params = [
            "lora", "lora_rank", "lora_alpha", "lora_dropout",
            "freeze_whisper", "freeze_beats", "freeze_speech_QFormer",
            "num_speech_query_token", "window_level_Qformer",
            "second_per_window", "second_stride"
        ]
        
        for param in important_params:
            if param in kwargs:
                key_parts.append(f"{param}={kwargs[param]}")
        
        # Add model paths if present
        path_params = ["model_path", "llama_path", "whisper_path", "beats_path", "ckpt_path"]
        for param in path_params:
            if param in kwargs:
                # Use just the filename to avoid path differences
                path = kwargs[param]
                if path:
                    filename = os.path.basename(path)
                    key_parts.append(f"{param}={filename}")
        
        return "|".join(key_parts)
    
    @staticmethod
    def clear_cache():
        """
        Clear the model cache to free memory.
        
        Returns:
            Number of models cleared from cache
        """
        global _MODEL_CACHE
        num_models = len(_MODEL_CACHE)
        
        # Clear the cache
        _MODEL_CACHE.clear()
        
        # Run garbage collection to free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Model cache cleared: {num_models} models")
        return num_models
    
    @staticmethod
    def optimize_for_inference(model, device=None):
        """
        Optimize a model for inference.
        
        Args:
            model: The model to optimize
            device: Device to move the model to
            
        Returns:
            Optimized model
        """
        try:
            # Move to specified device if needed
            if device is not None:
                logger.info(f"Moving model to device: {device}")
                model = model.to(device)
            
            # Set to evaluation mode
            model.eval()
            
            # Apply torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile') and callable(getattr(torch, 'compile')):
                try:
                    logger.info("Applying torch.compile for faster inference")
                    model = torch.compile(model)
                except Exception as e:
                    logger.warning(f"Failed to apply torch.compile: {str(e)}")
            
            # Optimize CUDA graphs if applicable
            if torch.cuda.is_available() and hasattr(torch, 'cuda') and hasattr(torch.cuda, 'make_graphed_callables'):
                try:
                    logger.info("Optimizing with CUDA graphs")
                    # This would need sample inputs to trace the graph
                    # Implementation depends on the model architecture
                    pass
                except Exception as e:
                    logger.warning(f"Failed to optimize with CUDA graphs: {str(e)}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error optimizing model for inference: {str(e)}")
            logger.debug(traceback.format_exc())
            # Return the original model if optimization fails
            return model
    
    @staticmethod
    @lru_cache(maxsize=8)
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with model information
            
        Raises:
            ValueError: If model_type is unknown
        """
        model_type = model_type.lower()
        
        if model_type == "salmonn":
            return {
                "name": "SALMONN",
                "full_name": "Speech Audio Language Music Open Neural Network",
                "description": "Multimodal model for speech, audio, and text",
                "default_paths": {
                    "llama_path": "/path/to/llama",
                    "whisper_path": "openai/whisper-large-v2",
                    "beats_path": "microsoft/beats"
                },
                "supports_lora": True,
                "supports_speech": True,
                "supports_audio": True
            }
        elif model_type == "qwen2":
            return {
                "name": "Qwen2",
                "full_name": "Qwen2 Audio",
                "description": "Multimodal model for audio and text",
                "default_paths": {
                    "model_path": "Qwen/Qwen2-Audio-7B-Instruct"
                },
                "supports_lora": True,
                "supports_speech": True,
                "supports_audio": True
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get a list of available model types.
        
        Returns:
            List of available model types
        """
        return ["salmonn", "qwen2"]
    
    @staticmethod
    def get_model_from_checkpoint(checkpoint_path: str, base_model_path: str, model_type: str, device=None):
        """
        Load a model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            base_model_path: Path to the base model
            model_type: Type of model
            device: Device to load the model on
            
        Returns:
            Loaded model
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If loading fails
        """
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            
            # Create base model
            if model_type == "salmonn":
                model = CustomSALMONN(llama_path=base_model_path, device=device)
            else:
                from .custom_qwen import CustomQwen
                model = CustomQwen(model_path=base_model_path, device=device)
            
            # Load checkpoint weights using utility function
            from utils.training_utils import load_checkpoint
            checkpoint = load_checkpoint(checkpoint_path, map_location=device)
            
            # Extract state dict based on checkpoint structure
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
                logger.info("Found weights under 'model' key")
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                logger.info("Found weights under 'model_state_dict' key")
            else:
                state_dict = checkpoint
                logger.info("Using checkpoint directly as state dict")
            
            # Load weights with strict=False to allow partial loading
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
            
            logger.info(f"Model loaded successfully from checkpoint")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from checkpoint: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to load model from checkpoint: {str(e)}") from e 