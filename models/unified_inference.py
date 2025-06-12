#!/usr/bin/env python3
"""
Unified Symbol Discovery Inference Script
Can run inference with or without MLP transformations
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

# FIXED: Use existing evaluation functions from inference.py
from utils.evaluation_utils import evaluate_predictions, clean_prediction
from models.unified_symbol_training import replace_symbols_in_batch

def parse_args():
    """Parse command line arguments (SAME FORMAT AS INFERENCE.PY)"""
    parser = argparse.ArgumentParser(description="Unified Symbol Discovery Inference")
    
    # SAME ARGUMENTS AS INFERENCE.PY
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
    parser.add_argument("--today", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="Date for output directory")
    parser.add_argument("--model_type", type=str, default="salmonn", help="Type of model to use")
    parser.add_argument("--dataset_type", type=str, required=True, help="Dataset type(s) to use")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    
    # SIMPLIFIED: Only one MLP flag needed
    parser.add_argument("--use_mlp", type=bool, default=True, help="Use MLP transformations (True) or bypass them (False)")
    parser.add_argument("--compare_modes", type=bool, default=False, help="Run both MLP and non-MLP inference for comparison")
    parser.add_argument("--symbol_mode", type=str, choices=["random", "original"], default="training", 
                       help="Symbol mode: 'training' (use same symbols as training), 'original' (use original labels)")
    
    # ADDED: num_examples parameter (from inference_job.sh)
    parser.add_argument("--num_examples", type=int, default=5, help="Number of few-shot examples")
    
    # ADDED: Missing generation parameters
    parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    return parser.parse_args()

def setup_inference_logging(args):
    """Setup logging (SAME AS INFERENCE.PY)"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True
    )
    return args

def load_inference_datasets(args, datasets):
    """Load inference datasets (SAME LOGIC AS INFERENCE.PY)"""
    inference_datasets = {}
    
    for dataset_name in datasets:
        try:
            dataset_type = DatasetType(dataset_name)
            dataset = load_dataset(dataset_type, split=args.split)
            
            if args.max_samples > 0:
                logging.info(f"Limiting to {args.max_samples} samples for dataset {dataset_name}")
                inference_datasets[dataset_type] = dataset.select(range(min(args.max_samples, len(dataset))))
            else:
                inference_datasets[dataset_type] = dataset
            
            logging.info(f"✓ Loaded {dataset_name} ({args.split}): {len(inference_datasets[dataset_type])} samples")
            
        except Exception as e:
            logging.error(f"✗ Failed to load dataset {dataset_name}: {e}")
            continue
    
    return inference_datasets

def get_symbol_mappings(args, dataset_labels, llama_tokenizer):
    """Get symbol mappings based on the specified mode"""
    
    if args.symbol_mode == "original":
        logging.info("Using original labels (no symbol mapping)")
        return {}
    
    elif args.symbol_mode == "random":
        logging.info("Using same symbols as random...")
        random_symbols = generate_one_word_two_token_symbols(len(dataset_labels), llama_tokenizer)
        symbol_mappings = create_label_mapping(dataset_labels, random_symbols)
        
        logging.info("=== Symbol Mappings ===")
        for original, symbol in symbol_mappings.items():
            logging.info(f"'{original}' -> '{symbol}'")
        
        return symbol_mappings
    
    return {}

def create_reverse_mappings(symbol_mappings):
    """Create reverse mappings for symbol conversion"""
    reverse_mappings = {}
    for original_label, random_symbol in symbol_mappings.items():
        reverse_mappings[random_symbol.lower()] = original_label
        reverse_mappings[random_symbol] = original_label
    return reverse_mappings

def convert_symbols_back(text, reverse_mappings):
    """Convert random symbols back to original labels"""
    if not reverse_mappings:
        return text
    
    converted = text
    for random_symbol, original_label in reverse_mappings.items():
        if random_symbol in converted:
            converted = converted.replace(random_symbol, original_label)
        elif random_symbol.lower() in converted.lower():
            import re
            pattern = re.compile(re.escape(random_symbol), re.IGNORECASE)
            if pattern.search(converted):
                converted = pattern.sub(original_label, converted)
    
    return converted

def run_inference_mode(model, dataloader, args, symbol_mappings, reverse_mappings, mode_name, use_mlp=True):
    """Run inference in a specific mode"""
    
    logging.info(f"=== Running {mode_name} Inference ===")
    logging.info(f"MLP enabled: {use_mlp}")
    
    # Set MLP bypass mode
    model.set_bypass_mlp(not use_mlp)
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{mode_name} Inference")):
            try:
                # Apply symbol replacements if we have mappings
                if symbol_mappings:
                    updated_batch = replace_symbols_in_batch(batch, symbol_mappings)
                else:
                    updated_batch = batch
                
                # Log first batch for debugging
                if batch_idx == 0:
                    logging.info(f"=== {mode_name} Sample Batch ===")
                    logging.info(f"Original prompt: {batch['prompt'][0][:200]}...")
                    if symbol_mappings:
                        logging.info(f"Updated prompt: {updated_batch['prompt'][0][:200]}...")
                    logging.info(f"True completion: {batch.get('completion', ['N/A'])[0]}")
                
                # Generate outputs
                outputs = model.generate_output(updated_batch)
                
                # Process outputs
                for i, (output, true_label) in enumerate(zip(outputs, batch.get("completion", [""] * len(outputs)))):
                    
                    # Convert symbols back only for training mode
                    if args.symbol_mode == "training" and reverse_mappings:
                        converted_output = convert_symbols_back(output, reverse_mappings)
                    else:
                        converted_output = output
                    
                    # Get dataset type for this sample
                    if isinstance(batch.get("dataset_type"), list):
                        dataset_type = batch["dataset_type"][i]
                    else:
                        dataset_type = batch.get("dataset_type", list(dataloader.dataset.datasets.keys())[0])
                    
                    # Clean the prediction (USE EXISTING FUNCTION)
                    cleaned_output = clean_prediction(converted_output, dataset_type)
                    
                    # Log first few samples
                    # if batch_idx < 2:
                    logging.info(f"Batch {batch_idx+1}, Sample {i+1} ({mode_name}):")
                    logging.info(f"Raw output: {output}")
                    if args.symbol_mode == "training" and symbol_mappings:
                        logging.info(f"Converted: {converted_output}")
                    logging.info(f"Cleaned: {cleaned_output}")
                    logging.info(f"True: {true_label}")
                    logging.info("-" * 30)
                    
                    # Store result (SAME FORMAT AS INFERENCE.PY)
                    prediction = {
                        "text": batch.get("text", [""])[i] if isinstance(batch.get("text"), list) else batch.get("text", ""),
                        "true_label": true_label,
                        "predicted_label (cleaned)": cleaned_output,
                        "predicted_label": converted_output.strip() if args.symbol_mode == "training" and symbol_mappings else output.strip(),
                        "dataset_type": dataset_type.value if hasattr(dataset_type, 'value') else str(dataset_type)
                    }
                    
                    results.append(prediction)
                
            except Exception as e:
                logging.error(f"Error in {mode_name} batch {batch_idx}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
    
    return results

def save_final_results(results, args, results_dir, suffix=""):
    """Save final results (SAME LOGIC AS INFERENCE.PY)"""
    try:
        # Clean dataset type for filename (SAME AS INFERENCE.PY)
        if "-" in args.dataset_type:
            clean_dataset_type = args.dataset_type.replace(" ", "").replace("-", "-")
        else:
            clean_dataset_type = args.dataset_type

        # Generate output file names (SAME AS INFERENCE.PY)
        base_filename = f"{args.run_name}_{clean_dataset_type}_unified{suffix}"
        
        # Save raw results (SAME PATH AS INFERENCE.PY)
        results_file = os.path.join(results_dir, f"{base_filename}_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Saved results to {results_file}")
        
        # Evaluate results (USE EXISTING FUNCTION)
        logging.info("Evaluating results")
        all_metrics = {}
        
        if "-" in args.dataset_type:
            dataset_types = [DatasetType(dt.strip()) for dt in args.dataset_type.split("-")]
        else:
            dataset_types = [DatasetType(args.dataset_type)]
        
        for dt in dataset_types:
            dt_results = [r for r in results if r["dataset_type"] == dt.value]
            if dt_results:
                metrics = evaluate_predictions(dt_results, dt)
                all_metrics[dt.value] = metrics
                logging.info(f"Results for {dt}:")
                for metric, value in metrics.items():
                    if isinstance(value, (float, int)):
                        logging.info(f"  {metric}: {value:.4f}")
                    else:
                        logging.info(f"  {metric}: {value}")
        
        # Save metrics (SAME PATH AS INFERENCE.PY)
        metrics_file = os.path.join(results_dir, f"{base_filename}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        logging.info(f"Saved metrics to {metrics_file}")
        
        return all_metrics
        
    except Exception as e:
        logging.error(f"Error saving final results: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {}

def main():
    args = parse_args()
    setup_inference_logging(args)
    
    # SIMPLIFIED: No need for conflicting argument handling
    logging.info("=== Unified Symbol Discovery Inference ===")
    logging.info(f"Symbol mode: {args.symbol_mode}")
    logging.info(f"MLP usage: {args.use_mlp}")
    logging.info(f"Compare modes: {args.compare_modes}")
    
    # Create output directory (SAME PATH AS INFERENCE.PY)
    results_dir = f"/data2/neeraja/neeraja/results/model_ICL/metrics/{args.today}"
    os.makedirs(results_dir, exist_ok=True)
    
    if args.model_type == "salmonn":
        # Setup tokenizers and processors
        logging.info("Loading tokenizer and feature extractor...")
        llama_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1", use_fast=False)
        llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        llama_tokenizer.padding_side = "right"
        
        whisper_path = "openai/whisper-large-v2"
        input_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        processor = get_processor(args.model_type, input_processor, llama_tokenizer)
        
        # Load datasets
        inference_datasets = load_inference_datasets(args, args.dataset_type.split('-'))
        
        # Create dataloader
        dataloader = DataLoader(
            DatasetFactory.create_dataset(
                dataset_type=list(inference_datasets.keys()),
                dataset=inference_datasets,
                processor=processor,
                is_training=False,
                input_mode="speech_only",
                fewshot_mode="text",
                num_examples=args.num_examples,  # FIXED: Use parameter instead of hardcoded 5
                random_examples=False,
                model_type=args.model_type,
                run_name=args.run_name,
                randomize_swap=False,
                balance_datasets=False,
                interleave=False
            ),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=processor.collate_batch,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
        
        # Get dataset labels
        dataset_names = args.dataset_type.split('-')
        all_valid_labels = set()
        for dataset_name in dataset_names:
            try:
                dataset_type = DatasetType(dataset_name)
                config = get_dataset_config(dataset_type)
                all_valid_labels.update(config.valid_labels)
            except Exception as e:
                logging.error(f"Failed to get config for dataset {dataset_name}: {e}")
                continue
        
        dataset_labels = sorted(list(all_valid_labels))
        logging.info(f"Dataset labels: {dataset_labels}")
        
        # Get symbol mappings
        symbol_mappings = get_symbol_mappings(args, dataset_labels, llama_tokenizer)
        reverse_mappings = create_reverse_mappings(symbol_mappings)
        
        # Create model
        logging.info("Creating model and loading checkpoint...")
        model = MLPSalmonn(
            device=args.device,
            label_tokens=list(symbol_mappings.values()) if symbol_mappings else [],
            hidden_dim=8,
            dropout=0.1,
            freeze_base=True,
            lora=True,
            lora_rank=8,
            lora_alpha=32,
            lora_dropout=0.05,
            low_resource=True,
            use_output_mlp=False
        )
        
        # Load checkpoint (SAME LOGIC AS INFERENCE.PY)
        if args.peft_model_path and args.peft_model_path.strip():
            logging.info(f"Loading checkpoint from {args.peft_model_path}")
            checkpoint = torch.load(args.peft_model_path, map_location=args.device)
            
            if "trainable_state_dict" in checkpoint:
                state_dict = checkpoint["trainable_state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logging.warning(f"Missing keys in checkpoint: {missing_keys[:5]}...")
            if unexpected_keys:
                logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
                
            logging.info("Checkpoint loaded successfully")
        
        # Update model with symbol mappings
        if symbol_mappings:
            model.update_label_tokens(symbol_mappings)
        
        model.eval()
        
        # Run inference
        if args.compare_modes:
            # Run both MLP and non-MLP inference
            logging.info("Running comparison between MLP and non-MLP modes...")
            
            # MLP mode
            mlp_results = run_inference_mode(
                model, dataloader, args, symbol_mappings, reverse_mappings, 
                "MLP", use_mlp=True
            )
            mlp_metrics = save_final_results(mlp_results, args, results_dir, "_mlp")
            
            # Non-MLP mode  
            no_mlp_results = run_inference_mode(
                model, dataloader, args, symbol_mappings, reverse_mappings, 
                "No-MLP", use_mlp=False
            )
            no_mlp_metrics = save_final_results(no_mlp_results, args, results_dir, "_no_mlp")
            
            # Log comparison
            logging.info("=== Mode Comparison ===")
            for dataset_name in mlp_metrics.keys():
                if dataset_name in no_mlp_metrics:
                    mlp_acc = mlp_metrics[dataset_name].get('accuracy', 0.0)
                    no_mlp_acc = no_mlp_metrics[dataset_name].get('accuracy', 0.0)
                    logging.info(f"{dataset_name}:")
                    logging.info(f"  MLP Mode Accuracy: {mlp_acc:.4f}")
                    logging.info(f"  No-MLP Mode Accuracy: {no_mlp_acc:.4f}")
                    logging.info(f"  Difference: {mlp_acc - no_mlp_acc:+.4f}")
            
        else:
            # Run single mode
            mode_name = "MLP" if args.use_mlp else "No-MLP"
            results = run_inference_mode(
                model, dataloader, args, symbol_mappings, reverse_mappings, 
                mode_name, use_mlp=args.use_mlp
            )
            
            suffix = "_mlp" if args.use_mlp else "_no_mlp"
            save_final_results(results, args, results_dir, suffix)
        
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