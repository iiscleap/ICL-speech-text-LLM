import logging
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple

from .master_config import DatasetType, DatasetSplit, get_dataset_config
from .multi_task_dataset import BaseMultiTaskDataset, MultiTaskDataset, MultiTaskTrainingDataset, MultiTaskInferenceDataset
from .training_dataset import TrainingDataset
from .inference_dataset import InferenceDataset
from .model_processors import get_processor

logger = logging.getLogger(__name__)

class DatasetFactory:
    """Factory class for creating datasets"""
    
    @staticmethod
    def create_dataset(
        dataset_type: Union[DatasetType, List[DatasetType]],
        dataset,
        processor,
        is_training: bool = False,
        input_mode: str = 'speech_only',
        fewshot_mode: str = 'text',
        num_examples: int = 5,
        random_examples: Optional[bool] = None,
        model_type: str = "salmonn",
        run_name: str = "",
        randomize_swap: bool = False,
        balance_datasets: bool = True,
        interleave: bool = True
    ):
        """
        Create a dataset based on the provided parameters.
        
        Args:
            dataset_type: Type of dataset or list of dataset types for multi-task
            dataset: The actual dataset object or dict mapping types to datasets
            processor: Model-specific processor
            is_training: Whether this is for training
            input_mode: 'speech_and_text', 'speech_only', or 'text_only'
            fewshot_mode: 'text' or 'speech' for few-shot examples
            num_examples: Number of few-shot examples to use
            random_examples: Whether to randomly select few-shot examples
                            (defaults to True for training, False for inference)
            model_type: Type of model ('salmonn', 'qwen2', etc.)
            run_name: Optional run name for logging
            
        Returns:
            The created dataset
            
        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If dataset creation fails
        """
        try:
            # Validate input parameters
            if input_mode not in ['speech_only', 'text_only', 'speech_and_text']:
                raise ValueError(f"Invalid input_mode: {input_mode}. Must be one of: speech_only, text_only, speech_and_text")
            
            if fewshot_mode not in ['text', 'speech']:
                raise ValueError(f"Invalid fewshot_mode: {fewshot_mode}. Must be one of: text, speech")
            
            if num_examples < 0:
                raise ValueError(f"Invalid num_examples: {num_examples}. Must be non-negative")
            
            # Set default for random_examples based on is_training
            if random_examples is None:
                random_examples = is_training
            
            logger.info(f"Creating {'training' if is_training else 'inference'} dataset with parameters:")
            logger.info(f"  dataset_type: {dataset_type}")
            logger.info(f"  input_mode: {input_mode}")
            logger.info(f"  fewshot_mode: {fewshot_mode}")
            logger.info(f"  num_examples: {num_examples}")
            logger.info(f"  random_examples: {random_examples}")
            logger.info(f"  model_type: {model_type}")
            
            # Create dataset class based on parameters
            if isinstance(dataset_type, list):
                    # Multi-task dataset
                return DatasetFactory._create_multi_task_dataset(
                    dataset_types=dataset_type,
                    dataset=dataset,
                    processor=processor,
                    is_training=is_training,
                    input_mode=input_mode,
                    fewshot_mode=fewshot_mode,
                    num_examples=num_examples,
                    random_examples=random_examples,
                    model_type=model_type,
                    run_name=run_name,
                    randomize_swap=randomize_swap,
                    balance_datasets=balance_datasets,
                    interleave=interleave
                )

            else:
                # Single-task dataset
                return DatasetFactory._create_single_task_dataset(
                    dataset_type=dataset_type,
                    dataset=dataset,
                    processor=processor,
                    is_training=is_training,
                    input_mode=input_mode,
                    fewshot_mode=fewshot_mode,
                    num_examples=num_examples,
                    random_examples=random_examples,
                    model_type=model_type
                )
                
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to create dataset: {str(e)}") from e
    
    @staticmethod
    def _create_single_task_dataset(
        dataset_type: DatasetType,
        dataset,
        processor,
        is_training: bool,
        input_mode: str,
        fewshot_mode: str,
        num_examples: int,
        random_examples: bool,
        model_type: str
    ):
        """
        Create a single-task dataset.
        
        Args:
            dataset_type: Type of dataset
            dataset: The actual dataset object
            processor: Model-specific processor
            is_training: Whether this is for training
            input_mode: Input mode
            fewshot_mode: Few-shot mode
            num_examples: Number of few-shot examples
            random_examples: Whether to randomly select examples
            model_type: Type of model
            
        Returns:
            The created dataset
        """
        if is_training:
            logger.info(f"Creating TrainingDataset for {dataset_type}")
            return TrainingDataset(
                dataset_type=dataset_type,
                dataset=dataset,
                processor=processor,
                input_mode=input_mode,
                fewshot_mode=fewshot_mode,
                num_examples=num_examples,
                random_examples=random_examples,
                model_type=model_type
            )
        else:
            logger.info(f"Creating InferenceDataset for {dataset_type}")
            return InferenceDataset(
                dataset_type=dataset_type,
                dataset=dataset,
                processor=processor,
                input_mode=input_mode,
                fewshot_mode=fewshot_mode,
                num_examples=num_examples,
                random_examples=random_examples,
                model_type=model_type
            )
    
    @staticmethod
    def _create_multi_task_dataset(
        dataset_types: List[DatasetType],
        dataset,
        processor,
        is_training: bool,
        input_mode: str,
        fewshot_mode: str,
        num_examples: int,
        random_examples: bool,
        model_type: str,
        run_name: str,
        randomize_swap: bool,
        balance_datasets: bool,
        interleave: bool
    ):
        """
        Create a multi-task dataset.
        
        Args:
            dataset_types: List of dataset types
            dataset: The actual dataset object or dict mapping types to datasets
            processor: Model-specific processor
            is_training: Whether this is for training
            input_mode: Input mode
            fewshot_mode: Few-shot mode
            num_examples: Number of few-shot examples
            random_examples: Whether to randomly select examples
            model_type: Type of model
            run_name: Run name for logging
            
        Returns:
            The created multi-task dataset
        """
        datasets = {}
        
        for dt in dataset_types:
            # Get the dataset for this type
            if isinstance(dataset, dict):
                dt_dataset = dataset.get(dt)
            else:
                dt_dataset = dataset
            
            if dt_dataset is None:
                logger.warning(f"No dataset provided for {dt}, skipping")
                continue
            
            # Create the appropriate dataset
            try:
                if is_training:
                    datasets[dt] = TrainingDataset(
                        dataset_type=dt,
                        dataset=dt_dataset,
                        processor=processor,
                        input_mode=input_mode,
                        fewshot_mode=fewshot_mode,
                        num_examples=num_examples,
                        random_examples=random_examples,
                        model_type=model_type,
                        run_name=run_name,
                        randomize_swap=randomize_swap
                    )
                else:
                    datasets[dt] = InferenceDataset(
                        dataset_type=dt,
                        dataset=dt_dataset,
                        processor=processor,
                        input_mode=input_mode,
                        fewshot_mode=fewshot_mode,
                        num_examples=num_examples,
                        random_examples=random_examples,
                        model_type=model_type,
                        run_name=run_name,
                        randomize_swap=randomize_swap
                    )
                logger.info(f"Added {dt} to multi-task dataset")
            except Exception as e:
                logger.error(f"Error creating dataset for {dt}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        if not datasets:
            raise ValueError("No valid datasets created for multi-task dataset")
        
        # Create multi-task dataset
        if is_training:
            logger.info(f"Creating MultiTaskTrainingDataset with {len(datasets)} tasks")
            return MultiTaskTrainingDataset(
                datasets=datasets,
                processor=processor,
                balance_datasets=balance_datasets,
                interleave=interleave
            )
        else:
            logger.info(f"Creating MultiTaskInferenceDataset with {len(datasets)} tasks")
            return MultiTaskInferenceDataset(
                datasets=datasets,
                processor=processor,
                balance_datasets=balance_datasets,
                interleave=interleave
            )
    
    @staticmethod
    def from_config(config: Dict[str, Any], datasets, processor=None):
        """
        Create a dataset from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            datasets: Dataset object or dictionary of datasets
            processor: Model-specific processor (optional)
            
        Returns:
            The created dataset
            
        Raises:
            ValueError: If required configuration keys are missing
            RuntimeError: If dataset creation fails
        """
        try:
            # Extract required parameters
            dataset_type = config.get("dataset_type")
            if not dataset_type:
                raise ValueError("dataset_type not specified in config")
            
            # Convert string dataset type to enum or list of enums
            if isinstance(dataset_type, str):
                if "," in dataset_type:
                    # Multi-task dataset
                    dataset_types = [DatasetType(dt.strip()) for dt in dataset_type.split(",")]
                else:
                    # Single-task dataset
                    dataset_types = DatasetType(dataset_type)
            else:
                dataset_types = dataset_type
            
            # Get processor if not provided
            if processor is None:
                model_type = config.get("model_type", "salmonn")
                processor_config = config.get("processor_config", {})
                processor = get_processor(model_type, **processor_config)
            
            # Extract other parameters with defaults
            is_training = config.get("is_training", False)
            input_mode = config.get("input_mode", "speech_only")
            fewshot_mode = config.get("fewshot_mode", "text")
            num_examples = config.get("num_examples", 5)
            random_examples = config.get("random_examples")
            model_type = config.get("model_type", "salmonn")
            run_name = config.get("run_name", "")
            
            # Create the dataset
            return DatasetFactory.create_dataset(
                dataset_type=dataset_types,
                dataset=datasets,
                processor=processor,
                is_training=is_training,
                input_mode=input_mode,
                fewshot_mode=fewshot_mode,
                num_examples=num_examples,
                random_examples=random_examples,
                model_type=model_type,
                run_name=run_name
            )
            
        except Exception as e:
            logger.error(f"Error creating dataset from config: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to create dataset from config: {str(e)}") from e
    
    @staticmethod
    def get_dataset_info(dataset_type: DatasetType) -> Dict[str, Any]:
        """
        Get information about a dataset type.
        
        Args:
            dataset_type: Type of dataset
            
        Returns:
            Dictionary with dataset information
        """
        try:
            # Get dataset config
            config = get_dataset_config(dataset_type)
            
            # Return dataset information
            return {
                "name": dataset_type.value,
                "prompt_template": config.prompt_template,
                "valid_labels": config.valid_labels,
                "completion_key": config.completion_key,
                "text_key": config.text_key,
                "has_audio": config.audio_lookup_paths is not None
            }
        except Exception as e:
            logger.error(f"Error getting dataset info for {dataset_type}: {str(e)}")
            logger.debug(traceback.format_exc())
            return {"name": dataset_type.value, "error": str(e)}