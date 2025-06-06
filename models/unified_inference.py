#!/usr/bin/env python3
"""
Unified Symbol Discovery Inference Script
Generates new random symbols, discovers meaningful symbols via MLP, and performs inference
"""

import os
import sys
import logging
import argparse
import torch
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_salmonn import MLPSalmonn, generate_one_word_two_token_symbols, create_label_mapping
from utils.data_utils import load_dataset
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType, get_dataset_config
from utils.training_utils import load_checkpoint
from torch.utils.data import DataLoader
from data.model_processors import get_processor
from transformers import LlamaTokenizer, WhisperFeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Symbol Discovery Inference")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="salmonn", help="Model type (salmonn or qwen2)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for inference")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    
    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, default="voxceleb_greek-hvb_greek", 
                      help="Dataset type(s) to use, hyphen-separated for multi-dataset")
    parser.add_argument("--max_samples", type=int, default=0, 
                      help="Maximum number of samples to process (0 = use all samples)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"],
                      help="Dataset split to use for inference")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/data2/neeraja/neeraja/results/model_ICL/unified_inference", 
                      help="Output directory for results")
    parser.add_argument("--run_name", type=str, default="", help="Name for the inference run")
    
    return parser.parse_args()

def setup_inference_logging(args):
    """Setup logging for unified inference"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if not args.run_name:
        args.run_name = f"unified_inference_{timestamp}_{args.dataset_type.replace('-', '_')}"
    
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, "unified_inference.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logging.info(f"Arguments: {args}")
    return args

def load_inference_datasets(args, datasets):
    """Load inference datasets"""
    from utils.data_utils import load_dataset
    
    inference_datasets = {}
    
    for dataset_name in datasets:
        try:
            dataset_type = DatasetType(dataset_name)
            
            # Load inference dataset
            dataset = load_dataset(dataset_type, split=args.split)
            
            # Apply sample limiting if specified
            if args.max_samples > 0:
                logging.info(f"Limiting to {args.max_samples} samples for dataset {dataset_name}")
                inference_datasets[dataset_type] = dataset.select(range(args.max_samples))
            else:
                inference_datasets[dataset_type] = dataset
            
            logging.info(f"✓ Loaded {dataset_name} ({args.split}): {len(inference_datasets[dataset_type])} samples")
            
        except Exception as e:
            logging.error(f"✗ Failed to load dataset {dataset_name}: {e}")
            continue
    
    logging.info(f"Successfully loaded {len(inference_datasets)} inference datasets")
    return inference_datasets

def create_inference_dataloader(datasets, processor, args):
    """Create inference dataloader"""
    from torch.utils.data import DataLoader
    
    if not datasets:
        raise ValueError("No datasets provided")
    
    dataset_types = list(datasets.keys())
    
    # Create dataset using DatasetFactory
    inference_dataset = DatasetFactory.create_dataset(
        dataset_type=dataset_types,
        dataset=datasets,
        processor=processor,
        is_training=False,  # Inference mode
        input_mode="speech_only",
        fewshot_mode="text",
        num_examples=5,
        random_examples=False,
        model_type=args.model_type,
        run_name=args.run_name,
        randomize_swap=False,
        balance_datasets=False,
        interleave=False
    )
    
    # Create dataloader
    num_workers = min(os.cpu_count() or 4, 4)
    dataloader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffling for inference
        collate_fn=processor.collate_batch,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=False  # Don't drop last batch in inference
    )
    
    return dataloader

def generate_responses(model, batch, args):
    """Generate responses using the model (copied from custom_salmon.py pattern)"""
    try:
        # Move batch to device
        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Set model to eval mode for generation
        model.eval()
        
        with torch.no_grad():
            # Get model inputs
            if hasattr(model, 'salmonn'):
                # Use SALMONN's generate method
                generated_outputs = model.salmonn.generate(
                    batch,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    pad_token_id=model.llama_tokenizer.pad_token_id,
                    eos_token_id=model.llama_tokenizer.eos_token_id,
                )
            else:
                # Fallback generation
                outputs = model(batch)
                generated_outputs = outputs.get("generated_text", [""])
        
        return generated_outputs
        
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return [""] * len(batch.get("prompt", [""]))

def replace_symbols_in_batch_inference(batch, symbol_mappings):
    """Replace discovered symbols in inference batch"""
    if not symbol_mappings:
        return batch
    
    updated_batch = batch.copy()
    
    # Replace in prompts
    if "prompt" in batch:
        updated_prompts = []
        for prompt in batch["prompt"]:
            updated_prompt = prompt
            for original, discovered in symbol_mappings.items():
                updated_prompt = updated_prompt.replace(original, discovered)
            updated_prompts.append(updated_prompt)
        updated_batch["prompt"] = updated_prompts
    
    return updated_batch

def evaluate_predictions(predictions, references, dataset_types):
    """Evaluate predictions using the same logic as inference.py"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    
    # Simple accuracy calculation for classification tasks
    correct = 0
    total = 0
    
    detailed_results = []
    
    for pred, ref in zip(predictions, references):
        total += 1
        # Simple exact match for now (can be enhanced based on task)
        if pred.strip().lower() == ref.strip().lower():
            correct += 1
            result = "correct"
        else:
            result = "incorrect"
        
        detailed_results.append({
            "prediction": pred,
            "reference": ref,
            "result": result
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "detailed_results": detailed_results
    }
    
    return metrics

def main():
    args = parse_args()
    setup_inference_logging(args)
    
    logging.info("=== Unified Symbol Discovery Inference ===")
    
    if args.model_type == "salmonn":
        # Setup tokenizers and processors
        logging.info("Loading tokenizer and feature extractor...")
        llama_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1", use_fast=False)
        llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        llama_tokenizer.padding_side = "right"
        
        whisper_path = "openai/whisper-large-v2"
        input_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        
        processor = get_processor(args.model_type, input_processor, llama_tokenizer)
        
        # Load inference datasets
        inference_datasets = load_inference_datasets(args, args.dataset_type.split('-'))
        inference_dataloader = create_inference_dataloader(inference_datasets, processor, args)
        
        # Get dataset labels for random symbol generation
        dataset_names = args.dataset_type.split('-')
        all_valid_labels = set()
        for dataset_name in dataset_names:
            try:
                dataset_type = DatasetType(dataset_name)
                config = get_dataset_config(dataset_type)
                valid_labels = config.valid_labels
                all_valid_labels.update(valid_labels)
                logging.info(f"Dataset {dataset_name} valid labels: {valid_labels}")
            except Exception as e:
                logging.error(f"Failed to get config for dataset {dataset_name}: {e}")
                continue
        
        dataset_labels = sorted(list(all_valid_labels))
        logging.info(f"Combined valid labels for symbol generation: {dataset_labels}")
        
        # Generate NEW random symbols for inference
        logging.info("Generating NEW random symbols for inference...")
        random_symbols = generate_one_word_two_token_symbols(len(dataset_labels), llama_tokenizer)
        initial_symbol_mappings = create_label_mapping(dataset_labels, random_symbols)
        
        logging.info("=== Initial Random Symbol Mappings for Inference ===")
        for original, random_symbol in initial_symbol_mappings.items():
            logging.info(f"'{original}' -> '{random_symbol}'")
        
        # Create model and load checkpoint
        logging.info("Creating model and loading checkpoint...")
        model = MLPSalmonn(
            device=args.device,
            label_tokens=list(initial_symbol_mappings.values()),
            hidden_dim=8,  # Should match training
            dropout=0.1,
            freeze_base=True,
            lora=True,
            lora_rank=8,
            lora_alpha=32,
            lora_dropout=0.05,
            low_resource=True
        )
        
        # Load trained weights
        logging.info(f"Loading checkpoint from: {args.checkpoint_path}")
        checkpoint = load_checkpoint(args.checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logging.info("Checkpoint loaded successfully")
        
        # Update model with initial random symbols
        model.update_label_tokens(initial_symbol_mappings)
        
        # Discover symbols using trained MLP
        logging.info("Discovering symbols using trained MLP...")
        model.eval()  # Set to eval mode for hard quantization
        
        discoveries_dir = os.path.join(args.output_dir, "discoveries")
        os.makedirs(discoveries_dir, exist_ok=True)
        
        try:
            # Use the same discovery logic as training
            token_mappings = model.discover_symbols(save_path=os.path.join(discoveries_dir, "inference_tokens.json"))
            
            if token_mappings:
                discovered_symbol_mappings = model.convert_token_mappings_to_text(token_mappings)
                
                # Save discovered symbols
                with open(os.path.join(discoveries_dir, "inference_symbols.json"), 'w') as f:
                    json.dump(discovered_symbol_mappings, f, indent=2)
                
                logging.info("=== Discovered Symbol Mappings ===")
                for original, discovered in discovered_symbol_mappings.items():
                    logging.info(f"'{original}' -> '{discovered}'")
                
                # Use discovered symbols for inference
                final_symbol_mappings = discovered_symbol_mappings
            else:
                logging.warning("No symbols discovered, using random symbols")
                final_symbol_mappings = initial_symbol_mappings
                
        except Exception as e:
            logging.error(f"Symbol discovery failed: {e}")
            logging.info("Using random symbols for inference")
            final_symbol_mappings = initial_symbol_mappings
        
        # Perform inference with discovered symbols
        logging.info("Starting inference with discovered symbols...")
        
        predictions = []
        references = []
        all_prompts = []
        all_responses = []
        
        model.eval()
        total_samples = len(inference_dataloader.dataset)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(inference_dataloader, desc="Inference")):
                try:
                    # Apply discovered symbol mappings to batch
                    updated_batch = replace_symbols_in_batch_inference(batch, final_symbol_mappings)
                    
                    # Log first batch for debugging
                    if batch_idx == 0:
                        logging.info("=== Sample Inference Batch ===")
                        logging.info(f"Original prompt: {batch['prompt'][0]}")
                        logging.info(f"Updated prompt: {updated_batch['prompt'][0]}")
                    
                    # Generate responses
                    generated_responses = generate_responses(model, updated_batch, args)
                    
                    # Store results
                    predictions.extend(generated_responses)
                    references.extend(batch.get("completion", [""] * len(generated_responses)))
                    all_prompts.extend(updated_batch["prompt"])
                    all_responses.extend(generated_responses)
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logging.info(f"Processed {batch_idx + 1}/{len(inference_dataloader)} batches")
                    
                except Exception as e:
                    logging.error(f"Error in inference batch {batch_idx}: {e}")
                    continue
        
        # Evaluate results
        logging.info("Evaluating results...")
        metrics = evaluate_predictions(predictions, references, list(inference_datasets.keys()))
        
        # Save results (same format as inference.py)
        results = {
            "metrics": metrics,
            "symbol_mappings": {
                "initial_random": initial_symbol_mappings,
                "discovered": final_symbol_mappings
            },
            "predictions": predictions,
            "references": references,
            "prompts": all_prompts,
            "responses": all_responses
        }
        
        # Save metrics
        metrics_file = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save full results
        results_file = os.path.join(args.output_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to: {args.output_dir}")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Correct: {metrics['correct']}/{metrics['total']}")
        
    logging.info("=== Inference Complete ===")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Inference failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)