import logging
from typing import Dict, List, Optional, Any

from .multi_task_dataset import BaseMultiTaskDataset
from .master_config import DatasetType, DatasetSplit, get_dataset_config, get_swap_config

logger = logging.getLogger(__name__)

class InferenceDataset(BaseMultiTaskDataset):
    """Dataset class for inference/evaluation"""
    
    def __init__(
        self, 
        dataset_type: DatasetType, 
        dataset, 
        processor,
        input_mode: str = 'speech_only',
        fewshot_mode: str = 'text', 
        num_examples: int = 5, 
        random_examples: bool = False,  # Default to False for consistent evaluation
        model_type: str = "salmonn"
    ):
        """
        Initialize the inference dataset.
        
        Args:
            dataset_type: Type of dataset (e.g., VOXCELEB, HVB)
            dataset: The actual dataset object
            processor: Model-specific processor for text/audio
            input_mode: 'speech_and_text', 'speech_only', or 'text_only'
            fewshot_mode: 'text' or 'speech' for few-shot examples
            num_examples: Number of few-shot examples to use
            random_examples: Whether to randomly select few-shot examples (default False for inference)
            model_type: Type of model ('salmonn', 'qwen2', etc.)
        """
        # Always use TEST split for inference dataset
        super().__init__(
            dataset_type=dataset_type,
            dataset=dataset,
            processor=processor,
            input_mode=input_mode,
            fewshot_mode=fewshot_mode,
            num_examples=num_examples,
            random_examples=random_examples,
            split=DatasetSplit.TEST,
            model_type=model_type
        )
    
    def _is_training(self):
        """Override to indicate this is not a training dataset"""
        return False
    
