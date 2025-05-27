import os
import sys
import logging
import argparse
import torch
import json
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
import time
import traceback
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_salmonn import MLPSalmonn, generate_one_word_two_token_symbols, create_label_mapping
from utils.data_utils import load_dataset
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType, get_dataset_config
from utils.training_utils import setup_logging, load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader, SubsetRandomSampler
from data.model_processors import get_processor

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Symbol Discovery and LoRA Training")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="salmonn", help="Model type (salmonn or qwen2)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--initial_model_path", type=str, required=True, help="Path to initial pretrained model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    
    # Training parameters
    parser.add_argument("--lora_lr", type=float, default=1e-5, help="Learning rate for LoRA training")
    parser.add_argument("--mlp_lr", type=float, default=1e-4, help="Learning rate for MLP training")
    parser.add_argument("--lora_epochs", type=int, default=2, help="LoRA epochs per cycle")
    parser.add_argument("--mlp_epochs", type=int, default=1, help="MLP epochs per cycle")
    parser.add_argument("--total_cycles", type=int, default=3, help="Total alternating cycles")
    parser.add_argument("--hidden_dim", type=int, default=8, help="Hidden dimension for MLP")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for scheduler")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision training")
    
    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, default="voxceleb_greek-hvb_greek", 
                      help="Dataset type(s) to use, hyphen-separated for multi-dataset")
    parser.add_argument("--max_samples", type=int, default=64, 
                      help="Maximum number of samples to use (0 = use all samples)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/data2/neeraja/neeraja/results/model_ICL/unified_training", 
                      help="Output directory for results")
    parser.add_argument("--run_name", type=str, default="", help="Name for the run")
    
    return parser.parse_args()

def setup_unified_logging(args):
    """Setup standard logging for unified training"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if not args.run_name:
        args.run_name = f"unified_{timestamp}_{args.dataset_type.replace('-', '_')}"
    
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup standard logging to match mlp_salmonn.py
    log_file = os.path.join(args.output_dir, "unified_training.log")
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

def train_lora_phase(model, dataloader, args, cycle, current_symbols=None):
    """Train LoRA weights with discovered symbols applied"""
    logging.info(f"=== Starting LoRA Training Phase - Cycle {cycle} ===")
    
    # Log current symbols being used
    if current_symbols:
        logging.info(f"Using discovered symbols in LoRA training:")
        for orig, disc in current_symbols.items():
            logging.info(f"  '{orig}' -> '{disc}'")
    else:
        logging.info("No symbol mappings available - using original symbols")
    
    # Freeze MLP, unfreeze LoRA
    model.freeze_mlp_weights()
    model.unfreeze_lora_weights()
    
    # IMPORTANT: Disable MLP bypass for LoRA training to use discovered symbols
    model.set_mlp_bypass(bypass=True)
    logging.info("MLP bypass disabled - LoRA will train with discovered symbol mappings")
    
    # LoRA optimizer (using AdamW like train.py)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lora_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Scheduler
    total_steps = len(dataloader) * args.lora_epochs // args.gradient_accumulation_steps
    from transformers import get_scheduler
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision setup
    from torch.cuda.amp import autocast, GradScaler
    if args.fp16:
        scaler = GradScaler()
        amp_dtype = torch.float16
        logging.info("Using float16 mixed precision training for LoRA")
    elif args.bf16:
        scaler = None
        amp_dtype = torch.bfloat16
        logging.info("Using bfloat16 mixed precision training for LoRA")
    else:
        scaler = None
        amp_dtype = None
        logging.info("Using full precision training for LoRA")
    
    # Training loop
    global_step = 0
    for epoch in range(args.lora_epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        logging.info(f"Starting LoRA Cycle {cycle} Epoch {epoch+1}/{args.lora_epochs}")
        logging.info(f"Total batches to process: {len(dataloader)}")
        
        progress_bar = tqdm(dataloader, desc=f"LoRA Cycle {cycle} Epoch {epoch+1}", disable=False)
        
        for step, batch in enumerate(progress_bar):
            step_start = time.time()
            
            try:
                # Apply symbol mappings to batch data
                if current_symbols:
                    batch = replace_symbols_in_batch(batch, current_symbols, model.llama_tokenizer)
                
                # Move batch to device
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Log first few batches with symbol replacement
                if step < 3:
                    logging.info(f"=== LoRA Batch {step+1} with Symbol Replacement ===")
                    if current_symbols:
                        logging.info("Symbol-replaced prompt:")
                        logging.info(batch["prompt"][0])
                        logging.info("Symbol-replaced completion:")
                        logging.info(batch["completion"][0])
                    else:
                        logging.info("Original prompt (no symbols):")
                        logging.info(batch["prompt"][0])
                
                # Forward pass (MLP will be applied since bypass=False)
                if amp_dtype is not None:
                    with autocast(dtype=amp_dtype):
                        outputs = model(batch)
                        loss = outputs["loss"]
                        loss = loss / args.gradient_accumulation_steps
                    
                    # Backward pass
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Update weights if gradient accumulation is complete
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                            
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        global_step += 1
                else:
                    # Standard precision training
                    outputs = model(batch)
                    loss = outputs["loss"]
                    loss = loss / args.gradient_accumulation_steps
                    
                    loss.backward()
                    
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        global_step += 1
                
                total_loss += loss.item() * args.gradient_accumulation_steps
                valid_batches += 1
                
                # Calculate and log detailed metrics every 10 steps (like inference.py pattern)
                if (step + 1) % 10 == 0:
                    # Calculate speed
                    examples_per_second = args.batch_size / (time.time() - step_start)
                    current_lr = optimizer.param_groups[0]['lr']
                    current_loss = loss.item() * args.gradient_accumulation_steps
                    
                    # Update progress bar with metrics
                    progress_bar.set_postfix({
                        "loss": f"{current_loss:.4f}",
                        "lr": f"{current_lr:.8f}",
                        "speed": f"{examples_per_second:.2f}ex/s"
                    })
                    
                    # Log detailed progress (like inference.py)
                    logging.info(
                        f"LoRA Cycle {cycle}, Epoch {epoch+1}, Batch {step+1}/{len(dataloader)}, "
                        f"Loss: {current_loss:.4f}, "
                        f"LR: {current_lr:.8f}, "
                        f"Speed: {examples_per_second:.2f} examples/s, "
                        f"Step time: {time.time() - step_start:.2f}s, "
                        f"Global step: {global_step}"
                    )
                
            except Exception as e:
                logging.error(f"Error in LoRA training batch {step}: {e}")
                continue
        
        # Close progress bar to prevent log interference (like inference.py)
        progress_bar.close()
        
        if valid_batches > 0:
            epoch_loss = total_loss / valid_batches
            logging.info(f"✓ LoRA Cycle {cycle} Epoch {epoch+1} COMPLETED: Average loss = {epoch_loss:.4f}")
            logging.info(f"  Total batches processed: {len(dataloader)}")
            logging.info(f"  Valid batches: {valid_batches}")
            logging.info(f"  Global steps: {global_step}")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(args.output_dir, f"cycle_{cycle}_lora_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=epoch_loss,
                path=os.path.join(checkpoint_dir, "model.pt")
            )
            logging.info(f"Saved LoRA checkpoint: {checkpoint_dir}")
        else:
            logging.error(f"No valid batches in LoRA epoch {epoch+1}")
    
    logging.info(f"=== Completed LoRA Training Phase - Cycle {cycle} ===")
    return model

def train_mlp_phase(model, dataloader, args, cycle, current_symbols=None):
    """Train MLP embeddings with discovered symbols applied"""
    logging.info(f"=== Starting MLP Training Phase - Cycle {cycle} ===")
    
    # Log current symbols being used
    if current_symbols:
        logging.info(f"Using discovered symbols in MLP training:")
        for orig, disc in current_symbols.items():
            logging.info(f"  '{orig}' -> '{disc}'")
    else:
        logging.info("No symbol mappings available - using original symbols")
    
    # Freeze LoRA, unfreeze MLP
    model.freeze_lora_weights()
    model.unfreeze_mlp_weights()
    
    # Ensure MLP bypass is disabled for MLP training
    model.set_mlp_bypass(bypass=False)
    
    # Use even more conservative optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.mlp_lr * 0.1,  # Much smaller learning rate
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    # Training loop
    for epoch in range(args.mlp_epochs):
        model.train()
        total_loss = 0
        valid_update_steps = 0
        accumulated_loss = 0
        
        # Log epoch start
        logging.info(f"Starting MLP Cycle {cycle} Epoch {epoch+1}/{args.mlp_epochs}")
        logging.info(f"Total batches to process: {len(dataloader)}")
        
        # Create progress bar with detailed description
        progress_bar = tqdm(dataloader, desc=f"MLP Cycle {cycle} Epoch {epoch+1}", disable=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Apply symbol mappings to batch data (SAME AS LORA PHASE)
                if current_symbols:
                    batch = replace_symbols_in_batch(batch, current_symbols, model.llama_tokenizer)
                
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Update progress bar description with current batch
                progress_bar.set_description(f"MLP Cycle {cycle} Epoch {epoch+1} - Processing batch {batch_idx+1}")
                
                # Log progress every 10 batches
                if batch_idx % 10 == 0:
                    logging.info(f"Processing MLP batch {batch_idx+1}/{len(dataloader)} (Cycle {cycle}, Epoch {epoch+1})")
                
                # Add detailed logging for first 5 iterations with symbol replacement
                if batch_idx < 5:
                    logging.info(f"=== MLP Cycle {cycle}, Epoch {epoch+1}, Batch {batch_idx+1} ===")
                    
                    if current_symbols:
                        logging.info("Symbol-replaced prompt:")
                        logging.info(batch["prompt"][0])
                        logging.info("Symbol-replaced completion:")
                        logging.info(batch["completion"][0])
                    else:
                        logging.info("Original prompt (no symbols):")
                        logging.info(batch["prompt"][0])
                        logging.info("Original completion:")
                        logging.info(batch["completion"][0])
                    
                    logging.info("=" * 60)
                
                # Clear memory every 50 steps
                if batch_idx > 0 and batch_idx % 50 == 0:
                    logging.info(f"Clearing memory at MLP batch {batch_idx}")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                        logging.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
                # Check MLP health before each batch
                is_healthy, health_info = model.check_mlp_health()
                if not is_healthy:
                    logging.warning(f"MLP unhealthy at batch {batch_idx}: {health_info}")
                    model.reset_mlp_weights()
                    optimizer.zero_grad()
                    accumulated_loss = 0
                    continue
                
                # Forward pass
                outputs = model(batch)
                loss = outputs["loss"]
                
                # Check for NaN/inf immediately
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"Invalid loss at batch {batch_idx}: {loss}")
                    model.reset_mlp_weights()
                    optimizer.zero_grad()
                    accumulated_loss = 0
                    continue
                
                # Scale loss more conservatively
                loss = loss / 16  # Larger accumulation for stability
                
                loss.backward()
                accumulated_loss += loss.item()
                
                # Update every 16 steps instead of 8
                if (batch_idx + 1) % 16 == 0:
                    # More aggressive gradient clipping
                    max_grad_norm = torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), 
                        max_norm=0.5  # Much smaller clipping
                    )
                    
                    if max_grad_norm > 5.0:  # Lower threshold
                        logging.warning(f"Large gradient norm {max_grad_norm:.2f}, skipping update")
                        optimizer.zero_grad()
                        accumulated_loss = 0
                        continue
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Track progress
                    valid_update_steps += 1
                    avg_loss = accumulated_loss / 16
                    total_loss += accumulated_loss
                    accumulated_loss = 0
                    
                    # Update progress bar with metrics
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.6f}", 
                        "steps": valid_update_steps,
                        "max_grad": f"{max_grad_norm:.3f}"
                    })
                    
                    # Log detailed progress every 5 update steps
                    if valid_update_steps % 5 == 0:
                        logging.info(
                            f"MLP Cycle {cycle}, Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, "
                            f"Update Step {valid_update_steps}, Loss: {avg_loss:.6f}, "
                            f"Max Grad Norm: {max_grad_norm:.3f}"
                        )
            
            except Exception as e:
                logging.error(f"Error in MLP training batch {batch_idx}: {e}")
                model.reset_mlp_weights()
                optimizer.zero_grad()
                accumulated_loss = 0
                continue
        
        # Close progress bar to prevent log interference
        progress_bar.close()
        
        if valid_update_steps > 0:
            epoch_loss = total_loss / valid_update_steps
            logging.info(f"✓ MLP Cycle {cycle} Epoch {epoch+1} COMPLETED: Average loss = {epoch_loss:.6f}")
            logging.info(f"  Total batches processed: {len(dataloader)}")
            logging.info(f"  Valid update steps: {valid_update_steps}")
            
            # Step the scheduler
            scheduler.step(epoch_loss)
            
            # Save MLP checkpoint
            checkpoint_dir = os.path.join(args.output_dir, f"cycle_{cycle}_mlp_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=epoch_loss,
                path=os.path.join(checkpoint_dir, "model.pt")
            )
            logging.info(f"Saved MLP checkpoint: {checkpoint_dir}")
        else:
            logging.error(f"No valid update steps in MLP epoch {epoch+1}")
    
    # Discover new symbols
    logging.info(f"=== Symbol Discovery - Cycle {cycle} ===")
    discovered_symbols = discover_symbols(model, args, cycle)
    
    logging.info(f"=== Completed MLP Training Phase - Cycle {cycle} ===")
    return model, discovered_symbols

def discover_symbols(model, args, cycle):
    """Enhanced symbol discovery with random baseline"""
    try:
        logging.info("Starting enhanced symbol discovery...")
        
        # Temporarily disable MLP bypass for discovery
        original_bypass = model.bypass_mlp_during_lora
        model.set_mlp_bypass(bypass=False)
        
        # Get actual labels
        actual_labels = model.label_tokens if hasattr(model, 'label_tokens') else []
        logging.info(f"Using actual labels: {actual_labels}")
        
        if not actual_labels:
            model.set_mlp_bypass(original_bypass)
            return {}
        
        # Run discovery with multiple samples for stability
        all_mappings = []
        for sample_idx in range(5):  # Try 5 different samples
            try:
                raw_mappings = model.find_symbol_mappings()
                if raw_mappings:
                    all_mappings.append(raw_mappings)
            except Exception as e:
                logging.warning(f"Discovery sample {sample_idx} failed: {e}")
                continue
        
        # Find consensus mappings
        consensus_mappings = {}
        if all_mappings:
            # Only keep mappings that appear in multiple samples
            for mapping in all_mappings:
                for source, target in mapping.items():
                    if source not in consensus_mappings:
                        consensus_mappings[source] = {}
                    if target not in consensus_mappings[source]:
                        consensus_mappings[source][target] = 0
                    consensus_mappings[source][target] += 1
            
            # Keep only mappings with consensus
            final_mappings = {}
            for source, targets in consensus_mappings.items():
                max_count = max(targets.values())
                if max_count >= 2:  # Appeared in at least 2 samples
                    target = max(targets, key=targets.get)
                    final_mappings[source] = target
            
            logging.info(f"Found {len(final_mappings)} consensus mappings")
        else:
            final_mappings = {}
            logging.warning("No symbol mappings discovered")
        
        # Restore original bypass setting
        model.set_mlp_bypass(original_bypass)
        
        # Convert to label mappings
        label_mappings = {}
        if final_mappings:
            tokenizer = model.llama_tokenizer
            for source_token_id, target_token_id in final_mappings.items():
                try:
                    source_text = tokenizer.decode([int(source_token_id)], skip_special_tokens=False).strip()
                    target_text = tokenizer.decode([int(target_token_id)], skip_special_tokens=False).strip()
                    
                    if source_text and target_text and source_text != target_text:
                        label_mappings[source_text] = target_text
                        logging.info(f"Stable mapping: '{source_text}' -> '{target_text}'")
                except Exception as e:
                    logging.warning(f"Error processing mapping {source_token_id}->{target_token_id}: {e}")
        
        return label_mappings
        
    except Exception as e:
        logging.error(f"Error during symbol discovery: {e}")
        return {}

def replace_symbols_in_batch(batch, symbol_mappings, tokenizer):
    """Replace original symbols with discovered symbols in batch data"""
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
    
    # Replace in completions
    if "completion" in batch:
        updated_completions = []
        for completion in batch["completion"]:
            updated_completion = completion
            for original, discovered in symbol_mappings.items():
                updated_completion = updated_completion.replace(original, discovered)
            updated_completions.append(updated_completion)
        updated_batch["completion"] = updated_completions
    
    return updated_batch

def load_datasets(args, datasets):
    """Load training and meta datasets following train.py pattern"""
    from utils.data_utils import load_dataset
    
    train_datasets = {}
    meta_datasets = {}
    
    for dataset_name in datasets:
        try:
            dataset_type = DatasetType(dataset_name)
            
            # Load datasets using the same pattern as train.py
            full_train_dataset = load_dataset(dataset_type, split="train")
            full_val_dataset = load_dataset(dataset_type, split="validation")
            
            # Apply debug sample limiting if specified
            if args.max_samples > 0:
                logging.info(f"Limiting to {args.max_samples} samples for dataset {dataset_name}")
                train_datasets[dataset_type] = full_train_dataset.select(range(args.max_samples))
                meta_datasets[dataset_type] = full_val_dataset.select(range(args.max_samples // 2))
            else:
                train_datasets[dataset_type] = full_train_dataset
                meta_datasets[dataset_type] = full_val_dataset.select(range(100))  # Match train.py
            
            logging.info(f"✓ Loaded {dataset_name}: train={len(train_datasets[dataset_type])}, val={len(meta_datasets[dataset_type])}")
            
        except Exception as e:
            logging.error(f"✗ Failed to load dataset {dataset_name}: {e}")
            logging.error(traceback.format_exc())
            continue
    
    logging.info(f"Successfully loaded {len(train_datasets)} training datasets and {len(meta_datasets)} meta datasets")
    return train_datasets, meta_datasets

def create_combined_dataloader(datasets, processor, args, shuffle=False):
    """Create a combined dataloader following train.py pattern"""
    from torch.utils.data import DataLoader
    
    if not datasets:
        raise ValueError("No datasets provided")
    
    # Convert to list format that train.py expects
    dataset_types = list(datasets.keys())
    
    # Create dataset using DatasetFactory like train.py
    combined_dataset = DatasetFactory.create_dataset(
        dataset_type=dataset_types,
        dataset=datasets,
        processor=processor,
        is_training=shuffle,  # Training mode for shuffle=True
        input_mode="speech_only",
        fewshot_mode="text",
        num_examples=5,
        random_examples=False,
        model_type=args.model_type,
        run_name=args.run_name,
        randomize_swap=False,  # Don't randomize during symbol training
        balance_datasets=False,
        interleave=False
    )
    
    # Determine number of workers for data loading
    num_workers = min(os.cpu_count() or 4, 4)  # Match train.py
    
    # Create dataloader with same settings as train.py
    dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=processor.collate_batch,  # IMPORTANT: Use processor's collate function
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True
    )
    
    return dataloader

def main():
    """Main function for unified symbol discovery training"""
    try:
        args = parse_args()
        args = setup_unified_logging(args)
        
        logging.info("Starting Unified Symbol Discovery Training")
        logging.info(f"Training arguments: {vars(args)}")
        
        # Load dataset configs
        datasets = args.dataset_type.split('-')
        logging.info(f"Training on datasets: {datasets}")
        
        # Get labels from ALL datasets
        all_labels = set()
        dataset_configs = {}
        
        for dataset_name in datasets:
            try:
                dataset_type = DatasetType(dataset_name)
                config = get_dataset_config(dataset_type)
                dataset_configs[dataset_type] = config
                
                if hasattr(config, 'valid_labels') and config.valid_labels:
                    for label in config.valid_labels:
                        all_labels.add(label)
                    logging.info(f"Dataset {dataset_name} has labels: {config.valid_labels}")
                else:
                    logging.warning(f"Dataset {dataset_name} has no valid_labels")
            except Exception as e:
                logging.error(f"Failed to get config for dataset {dataset_name}: {e}")
                return 1
        
        if not all_labels:
            logging.error("No valid labels found in any dataset")
            return 1
        
        # Convert to sorted list for consistency
        original_labels = sorted(list(all_labels))
        logging.info(f"Combined labels from all datasets ({len(original_labels)}): {original_labels}")
        
        # Generate random symbols BEFORE initializing the model
        # Use the FIXED function that actually produces 2-token symbols
        from transformers import LlamaTokenizer
        temp_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1")
        
        logging.info("=== Generating ACTUAL 2-Token Symbols ===")
        random_symbols = generate_one_word_two_token_symbols(len(original_labels), temp_tokenizer)
        
        # Verify ALL symbols are exactly 2 tokens
        logging.info("=== Verification of 2-Token Symbols ===")
        all_valid = True
        for symbol in random_symbols:
            token_ids = temp_tokenizer.encode(symbol, add_special_tokens=False)
            if len(token_ids) != 2:
                logging.error(f"FAILED: '{symbol}' -> {len(token_ids)} tokens: {token_ids}")
                all_valid = False
            else:
                logging.info(f"✓ '{symbol}' -> 2 tokens: {token_ids}")
        
        if not all_valid:
            logging.error("Some symbols are not 2 tokens! Cannot proceed.")
            return 1
        
        # Create simple mapping
        initial_symbol_mapping = create_label_mapping(original_labels, random_symbols)
        
        # Clean up temporary tokenizer
        del temp_tokenizer
        
        # NOW Initialize the model with the VERIFIED 2-token symbols
        logging.info("=== Initializing Model with VERIFIED 2-Token Symbols ===")
        model = MLPSalmonn(
            llama_path="lmsys/vicuna-13b-v1.1",
            whisper_path="openai/whisper-large-v2",
            beats_path="/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            label_tokens=random_symbols,  # PASS THE VERIFIED SYMBOLS HERE
            hidden_dim=args.hidden_dim,
            dropout=0.1,
            freeze_base=True,
            lora=True,
            lora_rank=8,
            lora_alpha=32,
            lora_dropout=0.05,
            device=args.device,
            low_resource=True
        )
        
        # Verify that label_token_ids got populated correctly
        logging.info(f"Model initialized with {len(model.label_token_ids)} label token IDs: {model.label_token_ids}")
        
        # Double-check tokenization with the model's tokenizer
        logging.info("=== Final Tokenization Verification with Model Tokenizer ===")
        for symbol in random_symbols:
            token_ids = model.llama_tokenizer.encode(symbol, add_special_tokens=False)
            logging.info(f"'{symbol}' -> {len(token_ids)} tokens: {token_ids}")
            if len(token_ids) != 2:
                logging.warning(f"Expected 2 tokens, got {len(token_ids)}!")
        
        # Create processor
        processor = get_processor(args.model_type, model.input_processor, model.llama_tokenizer)
        
        # Load datasets
        logging.info("=== Loading Datasets ===")
        train_datasets, meta_datasets = load_datasets(args, datasets)
        
        # Create data loaders
        train_loader = create_combined_dataloader(train_datasets, processor, args, shuffle=True)
        meta_loader = create_combined_dataloader(meta_datasets, processor, args, shuffle=True)
        
        logging.info(f"Training loader: {len(train_loader)} batches")
        logging.info(f"Meta loader: {len(meta_loader)} batches")
        
        # Store current symbol mappings
        current_symbols = initial_symbol_mapping.copy()
        
        # Main alternating training loop
        for cycle in range(1, args.total_cycles + 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Starting Training Cycle {cycle}/{args.total_cycles}")
            logging.info(f"Current symbols: {current_symbols}")
            logging.info(f"{'='*60}")
            
            # Phase 1: Train MLP first
            logging.info(f"Phase 1: MLP training - Cycle {cycle}")
            model, discovered_symbols = train_mlp_phase(model, meta_loader, args, cycle, current_symbols)
            
            # Update symbols after MLP training
            if discovered_symbols:
                current_symbols.update(discovered_symbols)
                logging.info(f"Updated symbol mappings after MLP training:")
                for orig, disc in discovered_symbols.items():
                    logging.info(f"  '{orig}' -> '{disc}'")
            
            # Phase 2: Train LoRA with discovered symbols
            logging.info(f"Phase 2: LoRA training with discovered symbols - Cycle {cycle}")
            model = train_lora_phase(model, train_loader, args, cycle, current_symbols)
            
            logging.info(f"Completed Training Cycle {cycle}/{args.total_cycles}")
        
        # Final symbol discovery and save results
        logging.info("=== Final Symbol Discovery ===")
        final_symbols = discover_symbols(model, args, args.total_cycles)
        
        if final_symbols:
            current_symbols.update(final_symbols)
            
            # Save final symbol mappings
            symbol_file = os.path.join(args.output_dir, "final_symbol_mappings.json")
            with open(symbol_file, 'w') as f:
                json.dump(current_symbols, f, indent=2)
            logging.info(f"Saved final symbol mappings to {symbol_file}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())