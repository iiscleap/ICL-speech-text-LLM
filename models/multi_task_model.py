import logging
import torch
from typing import Dict, List, Optional, Tuple, Union, Any

from .custom_salmon import CustomSALMONN


class MultiTaskModel:
    """
    Multi-task model that supports both SALMONN and Qwen2.
    Each task can have its own configuration and prompt template.
    """
    def __init__(self, 
                 model_type: str,  # "salmonn" or "qwen2"
                 task_configs: Dict[str, Dict[str, Any]] = None,
                 default_task: str = None,
                 *args, **kwargs):
        """
        Initialize the MultiTaskModel.
        
        Args:
            model_type: Type of model to use ("salmonn" or "qwen2")
            task_configs: Dictionary mapping task names to their configurations
            default_task: Name of the default task to use
            *args, **kwargs: Arguments to pass to the specific model constructor
        """
        self.model_type = model_type.lower()
        
        # Initialize the base model based on type
        if self.model_type == "salmonn":
            self.model = CustomSALMONN(*args, **kwargs)
        elif self.model_type == "qwen2":
            from .custom_qwen import CustomQwen
            self.model = CustomQwen(*args, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Task-specific configurations
        self.task_configs = task_configs or {}
        self.current_task = default_task
        
        # Initialize task-specific prompt templates
        self.task_prompt_templates = {}
        for task, config in self.task_configs.items():
            if 'prompt_template' in config:
                self.task_prompt_templates[task] = config['prompt_template']
        
        logging.info(f"Initialized MultiTaskModel ({model_type}) with {len(self.task_configs)} tasks")
        if default_task:
            logging.info(f"Default task set to: {default_task}")
    
    def set_task(self, task_name: str) -> bool:
        """Set the current active task."""
        if task_name in self.task_configs:
            self.current_task = task_name
            logging.info(f"Active task set to: {task_name}")
            return True
        logging.warning(f"Task '{task_name}' not found in configured tasks")
        return False
    
    def get_task_prompt_template(self, task_name: str = None) -> str:
        """Get the prompt template for a specific task."""
        task = task_name or self.current_task
        if task in self.task_prompt_templates:
            return self.task_prompt_templates[task]
        return self.model.prompt_template
    
    def forward(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass with task-specific handling."""
        # Extract task information from samples if available
        batch_tasks = samples.get("task", [self.current_task] * len(samples["prompt"]))
        
        # Log task distribution in batch (only occasionally)
        if self.model.batch_counter % 10 == 0:
            task_counts = {}
            for task in batch_tasks:
                task_counts[task] = task_counts.get(task, 0) + 1
            logging.info(f"Batch task distribution: {task_counts}")
        
        # First batch logging
        if self.model.batch_counter == 0:
            logging.info("=== First Batch Token Debug ===")
            logging.info(f"Current task: {self.current_task}")
            
            # Common logging for both models
            logging.info(f"Input IDs shape: {samples['input_ids'].shape}")
            logging.info(f"Last 20 input_ids: {samples['input_ids'][0][-20:].tolist()}")
            logging.info(f"Last 20 attention_mask: {samples['attention_mask'][0][-20:].tolist()}")
            
            # Qwen2-specific logging
            if self.model_type == "qwen2":
                logging.info(f"Input features shape: {samples['input_features'].shape}")
                logging.info(f"Last 20 feature_attention_mask: {samples['feature_attention_mask'][0][-20:].tolist()}")
                logging.info(f"Input features min: {samples['input_features'].min().item()}")
                logging.info(f"Input features max: {samples['input_features'].max().item()}")
                logging.info(f"Input features mean: {samples['input_features'].mean().item()}")
                decoded_input = self.model.processor.tokenizer.decode(samples['input_ids'][0])
                logging.info(f"Example input decoded: {decoded_input}")
            
            # SALMONN-specific logging
            elif self.model_type == "salmonn":
                if "spectrogram" in samples:
                    logging.info(f"Spectrogram shape: {samples['spectrogram'].shape}")
                if "raw_wav" in samples:
                    logging.info(f"Raw wav shape: {samples['raw_wav'].shape}")
                if "padding_mask" in samples:
                    logging.info(f"Padding mask shape: {samples['padding_mask'].shape}")
            
            logging.info("==============================")
        
        # Apply task-specific prompt templates if needed
        if "task" in samples and any(task is not None for task in samples["task"]):
            original_prompts = samples["prompt"]
            modified_prompts = []
            
            for i, (prompt, task) in enumerate(zip(original_prompts, batch_tasks)):
                if task is not None and task in self.task_prompt_templates:
                    task_template = self.task_prompt_templates[task]
                    modified_prompts.append(task_template + prompt.split(self.model.prompt_template, 1)[-1])
                else:
                    modified_prompts.append(prompt)
            
            samples["prompt"] = modified_prompts
        
        # Call model's forward method
        outputs = self.model.forward(samples)
        outputs["task"] = self.current_task
        return outputs
    
    def generate_output(self, samples: Dict[str, Any]) -> List[str]:
        """Generate predictions with task-specific handling."""
        # Extract task information if available
        task = samples.get("task", [self.current_task])[0]
        if task:
            self.set_task(task)
            
        # Apply task-specific parameters
        if task and task in self.task_configs:
            task_config = self.task_configs[task]
            generation_params = {
                "max_new_tokens": task_config.get("max_new_tokens", 10),
                "num_beams": task_config.get("num_beams", 1),
                "do_sample": task_config.get("do_sample", False),
                "temperature": task_config.get("temperature", 0.8),
            }
            samples.update(generation_params)
        
        # Call model's generate method
        return self.model.generate_output(samples)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create a MultiTaskModel instance from a configuration dictionary."""
        model_type = config.pop("model_type")
        task_configs = config.pop("task_configs", {})
        default_task = config.pop("default_task", None)
        
        return cls(
            model_type=model_type,
            task_configs=task_configs,
            default_task=default_task,
            **config
        ) 