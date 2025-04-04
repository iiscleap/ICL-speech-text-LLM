from typing import Dict, Any, Optional
from data.master_config import DatasetType

def get_inference_config(model_type: str, dataset_type: Optional[DatasetType] = None) -> Dict[str, Any]:
    """
    Get inference configuration for a specific model type and dataset type.
    
    Args:
        model_type: Type of model ("salmonn" or "qwen2")
        dataset_type: Type of dataset (optional)
        
    Returns:
        Dictionary containing inference configuration
    """
    # Base configuration
    base_config = {
        "num_workers": 2,
        "batch_size": 1,
        "generation_args": {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
    }
    
    # Model-specific configuration
    if model_type == "salmonn":
        model_config = {
            "model_args": {
                "llama_path": "lmsys/vicuna-13b-v1.1",
                "whisper_path": "openai/whisper-large-v2",
                "beats_path": "/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
                "lora": True,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "max_txt_len": 128,
            }
        }
    elif model_type == "qwen2":
        model_config = {
            "model_args": {
                "model_path": "Qwen/Qwen2-Audio-7B-Instruct",
                "lora": True,
                "max_txt_len": 512,
                "ckpt_path": "/data2/neeraja/neeraja/code/SALMONN/results/trained_models/ft_20e8b_qwen2_speech_text_voxceleb/final_model.pt"
            }
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Merge configurations
    config = {**base_config, **model_config}
    
    # Add dataset-specific configuration if provided
    if dataset_type is not None:
        if dataset_type in [DatasetType.VOXCELEB, DatasetType.VOXCELEB_SWAP, DatasetType.VOXCELEB_GREEK]:
            dataset_config = {
                "prompt_template": "Determine the sentiment of the following speech: {text}",
                "valid_labels": ["positive", "negative", "neutral"],
            }
        elif dataset_type in [DatasetType.HVB, DatasetType.HVB_SWAP, DatasetType.HVB_GREEK]:
            dataset_config = {
                "prompt_template": "Identify the emotions in the following speech: {text}",
                "valid_labels": ["happy", "sad", "angry", "surprised", "fearful", "disgusted"],
            }
        elif dataset_type in [DatasetType.VOXPOPULI, DatasetType.VOXPOPULI_SWAP, DatasetType.VOXPOPULI_GREEK]:
            dataset_config = {
                "prompt_template": "Classify the sentiment of the following speech: {text}",
                "valid_labels": ["positive", "negative", "neutral"],
            }
        else:
            dataset_config = {}
        
        config.update(dataset_config)
    
    return config 