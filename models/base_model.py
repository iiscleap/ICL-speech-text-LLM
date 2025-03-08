import abc
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import nullcontext

class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for all models in the ICL framework.
    Defines the common interface that all model implementations must follow.
    """
    
    def __init__(self, device=None, use_fp16=False):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16
        self.batch_counter = 0
        logging.info(f"Initializing model on device: {self.device}, FP16: {self.use_fp16}")
    
    @abc.abstractmethod
    def forward(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for training.
        
        Args:
            samples: Dictionary containing input data
            
        Returns:
            Dictionary containing at minimum 'loss' for backpropagation
        """
        pass
    
    @abc.abstractmethod
    def generate_output(self, samples: Dict[str, Any]) -> List[str]:
        """
        Generate predictions for inference.
        
        Args:
            samples: Dictionary containing input data
            
        Returns:
            List of string predictions
        """
        pass
    
    @abc.abstractmethod
    def get_speech_embeddings(self, samples: Dict[str, Any]) -> Tuple:
        """
        Extract speech embeddings from input samples.
        
        Args:
            samples: Dictionary containing input data
            
        Returns:
            Tuple containing speech embeddings and related data
        """
        pass
    
    def maybe_autocast(self):
        """Context manager for mixed precision training"""
        return torch.cuda.amp.autocast() if self.use_fp16 else nullcontext()
    
    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseModel":
        """
        Create a model instance from a configuration dictionary.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            Instantiated model
        """
        pass
    
    def save_checkpoint(self, path: str, optimizer=None, scheduler=None, epoch=None, loss=None):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            epoch: Optional current epoch
            loss: Optional current loss
        """
        checkpoint = {
            "model": self.state_dict(),
            "config": {
                "use_fp16": self.use_fp16,
            }
        }
        
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if loss is not None:
            checkpoint["loss"] = loss
            
        torch.save(checkpoint, path)
        logging.info(f"Model checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, config: Dict[str, Any] = None, map_location=None) -> "BaseModel":
        """
        Load a model from a checkpoint.
        
        Args:
            path: Path to the checkpoint file
            config: Optional configuration dictionary
            map_location: Optional device to map tensors to
            
        Returns:
            Tuple of (loaded model, checkpoint)
        """
        logging.info(f"Loading checkpoint from {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=map_location)
        
        # Create model from config if provided
        if config is not None:
            model = cls.from_config(config)
        else:
            # Try to get config from checkpoint
            if "config" in checkpoint:
                model = cls.from_config(checkpoint["config"])
            else:
                raise ValueError("No config provided and no config found in checkpoint")
        
        # Load state dict
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
            
        logging.info(f"Model loaded from checkpoint: {path}")
        return model, checkpoint

    # Removed load_checkpoint_weights method as it's redundant with the utility function 