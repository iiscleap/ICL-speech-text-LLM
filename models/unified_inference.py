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

# ✅ CORRECT IMPORTS from evaluation_utils
from utils.evaluation_utils import evaluate_predictions, save_evaluation_results, clean_prediction

# ✅ IMPORT the exact same function from training
from models.unified_symbol_training import replace_symbols_in_batch

def parse_args():
    """Parse command line arguments - SAME AS INFERENCE.PY"""
    parser = argparse.ArgumentParser(description="Unified Symbol Discovery Inference")
    
    # SAME ARGS AS INFERENCE.PY
    parser.add_argument("--model_type", type=str, default="salmonn", help="Type of model to use")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
    parser.add_argument("--dataset_type", type=str, required=True, help="Dataset type(s) to use")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    
    return parser.parse_args()

def setup_inference_logging(args):
    """Setup logging - SAME AS INFERENCE.PY"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    # SAVE IN METRICS FOLDER (SAME AS INFERENCE.PY)
    results_dir = f"/data2/neeraja/neeraja/results/model_ICL/metrics/{today}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Log file in the results directory
    # log_file = os.path.join(results_dir, f"{args.run_name}_unified_inference.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    
    args.output_dir = results_dir  # SET OUTPUT DIR TO METRICS FOLDER
    return args

def load_inference_datasets(args, datasets):
    """Load inference datasets"""
    inference_datasets = {}
    
    for dataset_name in datasets:
        try:
            dataset_type = DatasetType(dataset_name)
            
            # Load inference dataset
            dataset = load_dataset(dataset_type, split=args.split)
            
            # Apply sample limiting if specified
            if args.max_samples > 0:
                logging.info(f"Limiting to {args.max_samples} samples for dataset {dataset_name}")
                inference_datasets[dataset_type] = dataset.select(range(min(args.max_samples, len(dataset))))
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

def convert_symbols_back(text, reverse_mappings):
    """Convert random symbols back to original labels - handles multi-label outputs"""
    logging.info(f"Converting text: '{text}' using mappings: {reverse_mappings}")
    
    converted = text
    conversions_made = []
    
    for random_symbol, original_label in reverse_mappings.items():
        # Check if the symbol is in the text (case-insensitive for robustness)
        if random_symbol in converted:
            logging.info(f"Found '{random_symbol}' in text, replacing with '{original_label}'")
            converted = converted.replace(random_symbol, original_label)
            conversions_made.append(f"'{random_symbol}' -> '{original_label}'")
        elif random_symbol.lower() in converted.lower():
            # Case-insensitive fallback
            import re
            pattern = re.compile(re.escape(random_symbol), re.IGNORECASE)
            if pattern.search(converted):
                logging.info(f"Found '{random_symbol}' (case-insensitive) in text, replacing with '{original_label}'")
                converted = pattern.sub(original_label, converted)
                conversions_made.append(f"'{random_symbol}' -> '{original_label}' (case-insensitive)")
    
    if conversions_made:
        logging.info(f"Conversions made: {', '.join(conversions_made)}")
    else:
        logging.info(f"No symbols found to convert in text")
    
    logging.info(f"Final converted text: '{converted}'")
    return converted

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
        
        # Discover symbols using trained MLP (ONLY FIRST ITERATION)
        logging.info("Discovering symbols using trained MLP...")
        model.eval()

        # ✅ Enable discovery collection for first iteration only
        model.store_discoveries = True
        model.discovered_mappings = {}

        discoveries_dir = os.path.join(args.output_dir, "discoveries")
        os.makedirs(discoveries_dir, exist_ok=True)

        results = []
        reverse_symbol_mappings = {}
        for original_label, random_symbol in initial_symbol_mappings.items():
            reverse_symbol_mappings[random_symbol.lower()] = original_label
            reverse_symbol_mappings[random_symbol] = original_label  # Also exact case

        logging.info("=== Reverse Mapping (random symbol -> original label) ===")
        for symbol, original in reverse_symbol_mappings.items():
            logging.info(f"'{symbol}' -> '{original}'")

        # Perform inference 
        logging.info("Starting inference with symbol discovery...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(inference_dataloader, desc="Inference")):
                try:
                    # ✅ Use RANDOM symbols for prompts (same as training)
                    updated_batch = replace_symbols_in_batch(batch, initial_symbol_mappings)
                    
                    # Log first batch for debugging
                    if batch_idx == 0:
                        logging.info("=== Sample Inference Batch ===")
                        logging.info(f"Original prompt: {batch['prompt'][0]}")
                        logging.info(f"Updated prompt (with random symbols): {updated_batch['prompt'][0]}")
                    
                    # ✅ DISCOVERY HAPPENS AUTOMATICALLY during first iteration
                    outputs = model.generate_output(updated_batch)

                    # ✅ DEBUG: Check discovery status after first batch
                    if batch_idx == 0:
                        logging.info(f"DEBUG: After first batch, discovered_mappings: {getattr(model, 'discovered_mappings', {})}")
                        logging.info(f"DEBUG: store_discoveries flag: {getattr(model, 'store_discoveries', False)}")
                        logging.info(f"DEBUG: Has discovery_similarities: {hasattr(model, 'discovery_similarities')}")
                        
                        # ✅ ALWAYS process discovery data (even if empty)
                        logging.info("=== Symbol Discovery Status After First Iteration ===")
                        
                        if hasattr(model, 'discovered_mappings') and model.discovered_mappings:
                            logging.info("✓ Discoveries found!")
                            discovered_symbol_mappings = model.convert_token_mappings_to_text(model.discovered_mappings)
                            
                            # Save discoveries with similarities
                            discovery_data = {
                                'token_mappings': model.discovered_mappings,
                                'text_mappings': discovered_symbol_mappings,
                                'similarities': getattr(model, 'discovery_similarities', {}),
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            with open(os.path.join(discoveries_dir, f"{args.run_name}_symbols.json"), 'w') as f:
                                json.dump(discovery_data, f, indent=2)
                            
                            for original, discovered in discovered_symbol_mappings.items():
                                similarity = model.discovery_similarities.get(original, 0.0)
                                logging.info(f"Discovery: '{original}' -> '{discovered}' [similarity: {similarity:.4f}]")
                        else:
                            logging.warning("✗ No discoveries found - using random symbols for conversion")
                            
                            # Save empty discovery data for debugging
                            discovery_data = {
                                'token_mappings': {},
                                'text_mappings': {},
                                'similarities': {},
                                'timestamp': datetime.now().isoformat(),
                                'note': 'No discoveries made during first iteration'
                            }
                            
                            with open(os.path.join(discoveries_dir, f"{args.run_name}_no_discoveries.json"), 'w') as f:
                                json.dump(discovery_data, f, indent=2)
                        
                        # ✅ Disable discovery for remaining iterations
                        model.store_discoveries = False
                        logging.info("Discovery phase complete - continuing with normal inference")
                    
                    # ✅ Convert outputs using the convert_symbols_back function
                    for i, (output, true_label) in enumerate(zip(outputs, batch.get("completion", [""] * len(outputs)))):
                        
                        # ✅ USE THE FUNCTION instead of inline conversion
                        converted_output = convert_symbols_back(output, reverse_symbol_mappings)
                        
                        # Get dataset type for this sample
                        if isinstance(batch.get("dataset_type"), list):
                            dataset_type = batch["dataset_type"][i]
                        else:
                            dataset_type = batch.get("dataset_type", list(inference_datasets.keys())[0])
                        
                        # Clean the converted output
                        cleaned_output = clean_prediction(converted_output, dataset_type)
                        
                        # Log first few samples like inference.py
                        if batch_idx < 2:
                            logging.info(f"Batch {batch_idx+1}, Sample {i+1}:")
                            logging.info(f"Predicted (original): {output}")
                            logging.info(f"Predicted (converted): {converted_output}")
                            logging.info(f"Predicted (cleaned): {cleaned_output}")
                            logging.info(f"True: {true_label}")
                            logging.info("-" * 50)

                        # ✅ SAME FORMAT AS INFERENCE.PY - but store converted output
                        prediction = {
                            "text": batch.get("text", [""])[i] if isinstance(batch.get("text"), list) else batch.get("text", ""),
                            "true_label": true_label,
                            "predicted_label_cleaned": cleaned_output,  # This is the final cleaned version
                            "predicted_label": converted_output.strip(),  # This is the converted version
                            "predicted_label_raw": output.strip(),  # Keep the raw output for debugging
                            "dataset_type": dataset_type.value if hasattr(dataset_type, 'value') else str(dataset_type),
                            "original_prompt": batch["prompt"][i],
                            "updated_prompt": updated_batch["prompt"][i]
                        }
                        
                        results.append(prediction)
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logging.info(f"Processed {batch_idx + 1}/{len(inference_dataloader)} batches")
                    
                except Exception as e:
                    logging.error(f"Error in inference batch {batch_idx}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    continue
        
        # ✅ FOLLOW INFERENCE.PY PATTERN: Evaluate by dataset type
        logging.info("Evaluating results...")
        
        # Group results by dataset type and evaluate each separately
        dataset_metrics = {}
        for dataset_type in inference_datasets.keys():
            # Filter results for this dataset
            dataset_results = [r for r in results if r["dataset_type"] == dataset_type.value]
            
            if dataset_results:
                logging.info(f"Evaluating {len(dataset_results)} samples for dataset {dataset_type.value}")
                
                # ✅ USE evaluate_predictions from evaluation_utils (same as inference.py)
                metrics = evaluate_predictions(dataset_results, dataset_type)
                dataset_metrics[dataset_type.value] = metrics
                
                logging.info(f"Dataset {dataset_type.value} metrics: {metrics}")
            else:
                logging.warning(f"No results found for dataset {dataset_type.value}")
                dataset_metrics[dataset_type.value] = {"error": "No results"}
        
        # Calculate overall metrics
        overall_accuracy = 0.0
        total_correct = 0
        total_samples = len(results)
        
        if total_samples > 0:
            for result in results:
                if result["predicted_label_cleaned"].lower() == result["true_label"].lower():
                    total_correct += 1
            overall_accuracy = total_correct / total_samples
        
        # ✅ SAME STRUCTURE AS INFERENCE.PY
        final_results = {
            "dataset_metrics": dataset_metrics,
            "overall_metrics": {
                "accuracy": overall_accuracy,
                "correct": total_correct,
                "total": total_samples
            },
            "symbol_mappings": {
                "initial_random": initial_symbol_mappings,
                "discovered": reverse_symbol_mappings
            },
            "detailed_results": results,
            "summary": {
                "total_samples": total_samples,
                "dataset_types": [str(dt) for dt in inference_datasets.keys()],
                "num_datasets": len(inference_datasets)
            }
        }
        
        # ✅ SAVE RESULTS with run_name in filename
        save_evaluation_results(final_results, args.output_dir, f"{args.run_name}_unified_inference_results.json")
        save_evaluation_results(dataset_metrics, args.output_dir, f"{args.run_name}_metrics.json")

        logging.info(f"Results saved to: {args.output_dir}")
        logging.info(f"Main results: {args.run_name}_unified_inference_results.json")
        logging.info(f"Metrics: {args.run_name}_metrics.json")
        logging.info(f"Discoveries: {args.run_name}_symbols.json (if any)")
        
        # Log per-dataset results
        for dataset_name, metrics in dataset_metrics.items():
            if "accuracy" in metrics:
                logging.info(f"Dataset {dataset_name}: Accuracy = {metrics['accuracy']:.4f}")
        
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