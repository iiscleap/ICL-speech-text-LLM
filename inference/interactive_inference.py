import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import logging
import argparse
import torch
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import ModelFactory
from utils.training_utils import setup_logging, load_checkpoint
from config.inference_config import get_inference_config
from data.model_processors import get_processor
from data.master_config import DatasetType

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive inference with ICL models")
    
    # Model loading arguments
    parser.add_argument("--model_type", type=str, default="salmonn", help="Type of model to use")
    parser.add_argument("--peft_model_path", type=str, default="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb-hvb/checkpoints/epoch_10_loss_0.0060/model.pt", help="Path to the fine-tuned model")
    # parser.add_argument("--peft_model_path", type=str, default="/data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_greek-hvb_greek/checkpoints/epoch_10_loss_0.0055/model.pt", help="Path to the fine-tuned model")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference")
    
    # Performance options
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for inference")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster inference")
    
    # Query arguments
    parser.add_argument("--query", type=str, default="", help="Text query to process")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    
    return parser.parse_args()

def setup_model(args):
    # Get model config
    config = get_inference_config(args.model_type)
    
    # Force a specific GPU device if using CUDA
    if args.device.startswith('cuda'):
        # Extract device index if specified, default to 0
        device_idx = 0
        if ':' in args.device:
            device_idx = int(args.device.split(':')[1])
        args.device = f"cuda:{device_idx}"
        torch.cuda.set_device(device_idx)
        
        # Check available GPU memory before proceeding
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        reserved_memory = torch.cuda.memory_reserved(device_idx)
        allocated_memory = torch.cuda.memory_allocated(device_idx)
        free_memory = total_memory - reserved_memory
        
        logger.info(f"CUDA device {device_idx}: {torch.cuda.get_device_name(device_idx)}")
        logger.info(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
        logger.info(f"Reserved memory: {reserved_memory / 1e9:.2f} GB")
        logger.info(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        logger.info(f"Free memory: {free_memory / 1e9:.2f} GB")
        
        # If free memory is low, warn user and adjust settings
        if free_memory < 10 * 1e9:  # less than 10GB free
            logger.warning(f"Low GPU memory detected ({free_memory / 1e9:.2f} GB free)!")
            logger.warning("Loading model with low_resource=True and offload options")
            
            # Update config to use more aggressive memory settings
            config["model_args"]["low_resource"] = True
            if "device_map" not in config["model_args"]:
                config["model_args"]["device_map"] = "auto"
        
        logger.info(f"Forcing CUDA device to: {args.device}")
    
    # Create model with memory-efficient settings
    try:
        logger.info(f"Creating model of type {args.model_type}")
        
        # Force garbage collection before model creation
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Try to create model with current settings
        model = ModelFactory.create_model(
            model_type=args.model_type,
            multi_task=False,
            device=args.device,
            low_resource=True,  # Always use low_resource=True to be safe
            **config.get("model_args", {})
        )
        
    except RuntimeError as e:
            # Re-raise other errors
            raise
    
    # Load checkpoint if provided
    if args.peft_model_path and args.peft_model_path.strip():
        logger.info(f"Loading checkpoint from {args.peft_model_path}")
        checkpoint = load_checkpoint(args.peft_model_path, map_location=args.device)
        
        try:
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                finetuned_state_dict = checkpoint["model_state_dict"]
                finetuned_keys = set(finetuned_state_dict.keys())
                logging.info(f"Updating {len(finetuned_keys)} keys from finetuned model")
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            elif "model" in checkpoint:
                model.salmonn.load_state_dict(checkpoint["model"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("Out of memory when loading checkpoint. Try running with --device cpu")
                raise
            else:
                # Log but continue if there are missing keys - it might still work
                logger.warning(f"Error loading checkpoint: {str(e)}")
    
    # Apply torch.compile if requested and available
    if args.compile and hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        try:
            logger.info("Applying torch.compile for faster inference")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile: {e}")
    
    # Move model to device
    model.to(args.device)
    logger.info(f"Model device check:")
    device_info = next(model.parameters()).device
    logger.info(f"Model is on device: {device_info}")
    # Verify model device placement
    logger.info(f"Verifying model device placement:")
    param_devices = {}
    for name, param in model.named_parameters():
        device_name = param.device
        if device_name not in param_devices:
            param_devices[device_name] = 0
        param_devices[device_name] += param.numel()
    
    # Log device distribution
    logger.info("Parameter distribution by device:")
    for device, count in param_devices.items():
        logger.info(f"  {device}: {count:,} parameters ({count * 4 / 1e9:.2f} GB)")
    
    logger.info("Model device verification complete")
    
    # Create processor for the model
    if args.model_type == "salmonn":    
        processor = get_processor(args.model_type, model.input_processor, model.llama_tokenizer)
    else:
        processor = get_processor(args.model_type, model.input_processor)
    
    return model, processor

def run_interactive_inference(model, processor, query, args):
    logger.info(f"Processing query: {query}")
    
    # First process the raw inputs using process_inputs
    # This creates the tokenized representation
    processed_inputs = processor.process_inputs(
        data={
            "prompt": query,  # The text query
            "fewshot_mode": "text",
            "input_mode": "text_only",
            "completion": "",  # Empty for inference
            "audio": None,  # No audio for text-only mode
            "examples_audio": None,
            "dataset_type": DatasetType.VOXCELEB  # Default dataset type
        },
        is_training=False  # We're doing inference, not training
    )
    
    # Create a batch item with the processed inputs
    batch_item = {
        "input_ids": processed_inputs["input_ids"].squeeze(0),
        "attention_mask": processed_inputs["attention_mask"].squeeze(0),
        "prompt": query,
        "text": query,
        "completion": "",
        "dataset_type": DatasetType.VOXCELEB,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True,
        "num_examples": 0  # Add missing required key for collate_batch
    }
    
    # Now create a batch with just this single item
    batch = processor.collate_batch([batch_item])
    
    # Move to device - explicitly verify device placement and force if needed
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.device != torch.device(args.device):
                logger.info(f"Moving batch tensor {k} from {v.device} to {args.device}")
                batch[k] = v.to(args.device)
    
    # Verify device placement after moving
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.info(f"Batch tensor {k} is on device: {v.device}")
    
    # Generate output
    model.eval()
    with torch.no_grad():
        # Force CUDA device synchronization before generation
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)
        
        output = model.generate_output(batch)
    
    return output[0] if isinstance(output, list) else output

def main():
    # Parse arguments
    args = parse_args()
    
    
    try:
        # Log CUDA device information
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
        
        # Setup model and processor
        model, processor = setup_model(args)
        
        # Get query from args or command line
        query = args.query
        if not query:
            query = input("Enter your query: ")
        
        # Run inference
        output = run_interactive_inference(model, processor, query, args)
        
        # Print output
        print("\nModel output:")
        print(output)
        
        # Interactive mode
        while True:
            query = input("\nEnter a new query (or 'exit' to quit): ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            output = run_interactive_inference(model, processor, query, args)
            print("\nModel output:")
            print(output)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


# What is the definition of positive?
# What is the definition of negative?
# What is the definition of neutral?

# Define positive?

# Positive means having a good attitude, being optimistic, and looking for the good in situations. It is a mindset that focuses on the positive aspects of life and looks for solutions rather than dwelling on problems.
# Negative is a term used to describe something that is not positive or constructive. It can refer to a person, situation, or attitude that is characterized by negative emotions, thoughts, or actions.
# Neutral is a term used to describe a color or design element that is not associated with any particular emotion or feeling.

# Positive means having a good attitude, being optimistic, and looking for the good in situations. It is the opposite of negative.
# Negative is a term used to describe something that is not positive or constructive. It can refer to a person's attitude, emotions, or actions, as well as situations or circumstances that are not favorable or beneficial.
# Neutral refers to having no charge or bias. It is the opposite of positive or negative.

#Positive means having a good attitude, being optimistic, and looking for the best in situations. It is the opposite of negative.


