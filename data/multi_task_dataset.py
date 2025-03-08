import abc
import logging
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Tuple, Union
import time

from .master_config import DatasetType, DatasetSplit, get_dataset_config, get_swap_config
from .model_processors import ModelProcessor
from datasets import load_from_disk

logger = logging.getLogger(__name__)

class BaseMultiTaskDataset(Dataset):
    """Base class for multi-task datasets"""
    
    def __init__(
        self, 
        dataset_type: DatasetType, 
        dataset, 
        processor,
        input_mode: str = 'speech_only',
        fewshot_mode: str = 'text', 
        num_examples: int = 5, 
        random_examples: bool = False, 
        split: DatasetSplit = DatasetSplit.TEST, 
        model_type: str = "salmonn", 
        run_name: str = ""
    ):
        """
        Initialize the base multi-task dataset.
        
        Args:
            dataset_type: Type of dataset (e.g., VOXCELEB, HVB)
            dataset: The actual dataset object
            processor: Model-specific processor for text/audio
            input_mode: 'speech_and_text', 'speech_only', or 'text_only'
            fewshot_mode: 'text' or 'speech' for few-shot examples
            num_examples: Number of few-shot examples to use
            random_examples: Whether to randomly select few-shot examples
            split: Dataset split (TRAIN, VAL, TEST)
            model_type: Type of model ('salmonn', 'qwen2', etc.)
            run_name: Name of the run (for logging)
        """
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.processor = processor
        self.input_mode = input_mode
        self.fewshot_mode = fewshot_mode
        self.num_examples = num_examples
        self.random_examples = random_examples
        self.split = split
        self.model_type = model_type.lower()
        self.run_name = run_name
        
        # Get dataset configuration
        self.config = get_dataset_config(dataset_type)
        
        # Set up prompt template and label mapping for swap configurations
        self.prompt_template = self.config.prompt_template
        self.label_mapping = self.config.label_mapping
        
        if dataset_type in [DatasetType.VOXCELEB_SWAP, DatasetType.HVB_SWAP, DatasetType.VOXPOPULI_SWAP]:
            # For swap configurations, get both prompt template and label mapping
            swap_prompt, swap_mapping = get_swap_config(dataset_type)
            self.prompt_template = swap_prompt
            self.label_mapping = swap_mapping

        # Pre-load audio lookup if using speech examples
        self.audio_lookup = None
        self.audio_index_map = None
        if fewshot_mode == 'speech':
            audio_lookup_path = self.config.get_audio_lookup_path(self.split)
            if audio_lookup_path:
                load_time = time.time()
                self.audio_lookup = load_from_disk(audio_lookup_path)
                self.audio_index_map = {str(idx): i for i, idx in enumerate(self.audio_lookup['index'])}
                logger.info(f"Initialized audio lookup dataset in {time.time() - load_time:.3f}s")

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.dataset)

    def _get_audio_by_index(self, index_str):
        start_time = time.time()
        
        if not index_str:
            return None
        
        if self.audio_lookup is None:
            logger.warning(f"Audio lookup not initialized for {self.dataset_type}")
            return None
        
        try:
            # lookup_time = time.time()
            lookup_idx = self.audio_index_map.get(index_str)
            if lookup_idx is None:
                logger.warning(f"No matching audio found for index {index_str}")
                return None
            
            audio = self.audio_lookup[lookup_idx]['audio']
            # logger.info(f"Audio lookup took: {time.time() - lookup_time:.3f}s")
            return audio
        
        except Exception as e:
            logger.error(f"Error loading audio for index {index_str}: {e}")
            return None

    def _select_examples(self, few_shot_examples):
        """Helper method to select examples consistently"""
        if self.random_examples:
            num_examples = min(self.num_examples, len(few_shot_examples))
            return random.sample(few_shot_examples, num_examples) if num_examples > 0 else []
        return few_shot_examples[:self.num_examples]
    


    def _format_label(self, example_or_label, is_example=True):
        """
        Format label based on dataset type.
        
        Args:
            example_or_label: Either a full example dictionary or just a label value
            is_example: If True, treat input as example dict; if False, treat as direct label
            
        Returns:
            Formatted label as string
        """
        # Extract the label from example if needed
        if is_example:
            label = example_or_label['label']
        else:
            label = example_or_label
        
        # Handle different dataset types first
        if self.dataset_type in [DatasetType.VOXPOPULI, DatasetType.VOXPOPULI_SWAP, DatasetType.VOXPOPULI_GREEK]:
            if isinstance(label, dict):
                # Filter out keys with None values
                label_dict = [k for k, v in label.items() if v]
                return ', '.join(label_dict) if label_dict else 'None'
        
        # Handle list-type labels (for HVB datasets)
        if isinstance(label, list):
            # Convert list to comma-separated string
            label = ', '.join(label)
        
        # Normalize case to lowercase
        label = label.lower()
        
        # Apply label mapping if needed
        if self.label_mapping and isinstance(label, str):
            if "," in label:
                # Handle multi-label case (e.g., HVB)
                parts = [part.strip().lower() for part in label.split(",")]
                mapped_parts = [self.label_mapping.get(part, part) for part in parts]
                label = ", ".join(mapped_parts)
            else:
                # Single label case
                label = self.label_mapping.get(label.lower(), label.lower())
                
        return label

    def __getitem__(self, idx):

        
        # total_start = time.time()
        
        # Get item from dataset
        # item_start = time.time()
        item = self.dataset[idx]
        
        # Get few-shot examples
        # examples_start = time.time()
        few_shot_examples = item.get('few_shot_examples', [])
        selected_examples = self._select_examples(few_shot_examples)
        # logger.info(f"Selecting examples took: {time.time() - examples_start:.3f}s")
        
        # Process completion
        # completion_start = time.time()
        completion = self._format_label(item[self.config.completion_key], is_example=False)
        # logger.info(f"Processing completion took: {time.time() - completion_start:.3f}s")
        
        # Format examples
        # format_start = time.time()
        formatted_examples = []
        for example in selected_examples:
            formatted_example = {
                "text": example['text'],
                "label": self._format_label(example, is_example=True)
            }
            formatted_examples.append(formatted_example)
        # logger.info(f"Formatting examples took: {time.time() - format_start:.3f}s")
        
        # Create prompt
        # prompt_start = time.time()
        prompt = self.processor.format_prompt(
            template=self.prompt_template,
            text=item[self.config.text_key],
            examples=formatted_examples,
            input_mode=self.input_mode,
            fewshot_mode=self.fewshot_mode
        )
        # logger.info(f"Creating prompt took: {time.time() - prompt_start:.3f}s")
        
        # Get audio data
        # audio_start = time.time()
        audio_data = self._get_main_audio(item)
        # logger.info(f"Getting main audio took: {time.time() - audio_start:.3f}s")
        
        # Get examples audio
        # examples_audio_start = time.time()
        examples_audio = self._get_examples_audio(selected_examples)
        # logger.info(f"Getting examples audio took: {time.time() - examples_audio_start:.3f}s")
        
        # Process inputs
        # process_start = time.time()
        inputs = self.processor.process_inputs(
            data={
                "prompt": prompt,
                "fewshot_mode": self.fewshot_mode,
                "input_mode": self.input_mode,
                "completion": completion,
                "audio": audio_data,
                "examples_audio": examples_audio
            },
            is_training=self._is_training()
        )
        # logger.info(f"Processing inputs took: {time.time() - process_start:.3f}s")
        
        # Build result
        # build_start = time.time()
        result = self._build_result(
            prompt=prompt,
            item=item,
            completion=completion,
            text=item[self.config.text_key],
            inputs=inputs
        )
        # logger.info(f"Building result took: {time.time() - build_start:.3f}s")
        
        # logger.info(f"Total __getitem__ time: {time.time() - total_start:.3f}s")

        return result
    def _get_main_audio(self, item):
        """Get main audio data if needed (can be overridden)"""
        audio_data = None
        if "speech" in self.input_mode and "audio" in item:
            audio_data = item["audio"]["array"]
        return audio_data
        
    def _get_examples_audio(self, selected_examples):
        """Get audio data for examples if needed (can be overridden)"""
        examples_audio = None
        if self.fewshot_mode == 'speech':
            examples_audio = []
            # logger.info(f"Getting audio for {len(selected_examples)} examples")

            for i, example in enumerate(selected_examples):
                if 'index' in example:
                    example_audio = self._get_audio_by_index(example['index'])
                    if example_audio is not None:
                        if isinstance(example_audio['array'], list):
                            import numpy as np
                            examples_audio.append(np.array(example_audio['array']))
                        else:
                            examples_audio.append(example_audio['array'])

            if not examples_audio:
                examples_audio = None

        return examples_audio
        
    def _is_training(self):
        """Whether this dataset is for training (can be overridden)"""
        return False
        
    def _build_result(self, prompt, item, completion, text,inputs):
        """Build the result dictionary (can be overridden)"""
        return {
            "prompt": prompt,
            "text": text,
            "completion": completion,
            "dataset_type": self.dataset_type,
            **inputs
        }

    def set_task(self, task_type: DatasetType):
        """
        Switch the current task/dataset type.
        
        Args:
            task_type: The dataset type to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if task_type not in self.few_shot_examples:
            logger.warning(f"Task {task_type} not found in available tasks")
            return False
        
        # Update the current dataset type
        self.dataset_type = task_type
        
        # Update the config and label mapping for the new task
        self.config = get_dataset_config(task_type)
        self.label_mapping = self.config.label_mapping
        
        logger.info(f"Switched to task: {task_type}")
        return True

    def get_available_tasks(self):
        """
        Get a list of available tasks/dataset types.
        
        Returns:
            List of available dataset types
        """
        return list(self.few_shot_examples.keys())

class MultiTaskDataset(Dataset):
    """Dataset that combines multiple datasets into one"""
    
    def __init__(
        self,
        datasets: Dict[DatasetType, BaseMultiTaskDataset],
        processor
    ):
        """
        Initialize the multi-task dataset.
        
        Args:
            datasets: Dictionary mapping dataset types to dataset objects
            processor: Model-specific processor for text/audio
        """
        self.datasets = datasets
        self.dataset_types = list(datasets.keys())
        self.processor = processor
        
        # Calculate dataset sizes and create index mapping
        self.dataset_sizes = {dt: len(dataset) for dt, dataset in datasets.items()}
        self.total_size = sum(self.dataset_sizes.values())
        
        # Create a mapping from global index to (dataset_type, local_index)
        self.index_mapping = []
        for dt in self.dataset_types:
            for local_idx in range(self.dataset_sizes[dt]):
                self.index_mapping.append((dt, local_idx))
        
        logger.info(f"Created MultiTaskDataset with {len(self.dataset_types)} tasks")
        for dt in self.dataset_types:
            logger.info(f"  - {dt}: {self.dataset_sizes[dt]} examples")
        logger.info(f"Total examples: {self.total_size}")
    
    def __len__(self):
        """Return the total size of all datasets"""
        return self.total_size
    
    def __getitem__(self, idx):
        # Log before mapping
        
        # Map global index to dataset type and local index
        dataset_type, local_idx = self.index_mapping[idx]
        
        # Get the item from the appropriate dataset
        item = self.datasets[dataset_type][local_idx]
        
        # Add the dataset type to the result if not already present
        if "dataset_type" not in item:
            item["dataset_type"] = dataset_type
        
        return item

class MultiTaskTrainingDataset(MultiTaskDataset):
    """Multi-task dataset for training"""
    
    def __init__(self, datasets: Dict[DatasetType, BaseMultiTaskDataset], processor):
        super().__init__(datasets, processor)
    
    def _is_training(self):
        return True

class MultiTaskInferenceDataset(MultiTaskDataset):
    """Multi-task dataset for inference"""
    
    def __init__(self, datasets: Dict[DatasetType, BaseMultiTaskDataset], processor):
        super().__init__(datasets, processor)
    
    def _is_training(self):
        return False