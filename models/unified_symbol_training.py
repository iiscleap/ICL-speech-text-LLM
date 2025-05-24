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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_salmonn import MLPSalmonn
from utils.data_utils import load_dataset
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType
from utils.training_utils import setup_logging, load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader, SubsetRandomSampler

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
    """Train LoRA weights with detailed progress logging like inference.py"""
    logging.info(f"=== Starting LoRA Training Phase - Cycle {cycle} ===")
    
    # Freeze MLP, unfreeze LoRA
    model.freeze_mlp_weights()
    model.unfreeze_lora_weights()
    
    # Log trainable parameters
    trainable, frozen = model.get_trainable_parameters()
    logging.info(f"Trainable parameters: {len(trainable)}")
    logging.info(f"Frozen parameters: {len(frozen)}")
    
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
        
        # Log epoch start like inference.py
        logging.info(f"Starting LoRA Cycle {cycle} Epoch {epoch+1}/{args.lora_epochs}")
        logging.info(f"Total batches to process: {len(dataloader)}")
        
        # Create progress bar with detailed description
        progress_bar = tqdm(dataloader, desc=f"LoRA Cycle {cycle} Epoch {epoch+1}", disable=False)
        
        for step, batch in enumerate(progress_bar):
            step_start = time.time()
            
            try:
                # Move batch to device
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Update progress bar description with current batch (like inference.py)
                progress_bar.set_description(f"LoRA Cycle {cycle} Epoch {epoch+1} - Processing batch {step+1}")
                
                # Log progress every 10 batches (like inference.py pattern)
                if step % 10 == 0:
                    logging.info(f"Processing LoRA batch {step+1}/{len(dataloader)} (Cycle {cycle}, Epoch {epoch+1})")
                
                # Add detailed logging for first 5 iterations (like inference.py)
                if step < 5:
                    logging.info(f"=== LoRA Cycle {cycle}, Epoch {epoch+1}, Batch {step+1} ===")
                    
                    # Log input prompt
                    if "prompt" in batch:
                        logging.info("Input Prompt:")
                        logging.info(batch["prompt"][0])
                    
                    # Log target completion
                    if "completion" in batch:
                        logging.info("Expected Output:")
                        logging.info(batch["completion"][0])
                    
                    logging.info("=" * 60)
                
                # Clear memory every 50 steps (like inference.py)
                if step > 0 and step % 50 == 0:
                    logging.info(f"Clearing memory at LoRA step {step}")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                        logging.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
                # Forward pass with mixed precision
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

def train_mlp_phase(model, dataloader, args, cycle):
    """Train MLP embeddings with detailed progress logging like inference.py"""
    logging.info(f"=== Starting MLP Training Phase - Cycle {cycle} ===")
    
    # Freeze LoRA, unfreeze MLP
    model.freeze_lora_weights()
    model.unfreeze_mlp_weights()
    
    # MLP optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.mlp_lr,
        momentum=0.0,
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
        
        # Log epoch start like inference.py
        logging.info(f"Starting MLP Cycle {cycle} Epoch {epoch+1}/{args.mlp_epochs}")
        logging.info(f"Total batches to process: {len(dataloader)}")
        
        # Create progress bar with detailed description
        progress_bar = tqdm(dataloader, desc=f"MLP Cycle {cycle} Epoch {epoch+1}", disable=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Update progress bar description with current batch (like inference.py)
                progress_bar.set_description(f"MLP Cycle {cycle} Epoch {epoch+1} - Processing batch {batch_idx+1}")
                
                # Log progress every 10 batches (like inference.py pattern)
                if batch_idx % 10 == 0:
                    logging.info(f"Processing MLP batch {batch_idx+1}/{len(dataloader)} (Cycle {cycle}, Epoch {epoch+1})")
                
                # Add detailed logging for first 5 iterations (like inference.py)
                if batch_idx < 5:
                    logging.info(f"=== MLP Cycle {cycle}, Epoch {epoch+1}, Batch {batch_idx+1} ===")
                    
                    if "prompt" in batch:
                        logging.info("Input Prompt:")
                        logging.info(batch["prompt"][0])
                    
                    if "completion" in batch:
                        logging.info("Expected Output:")
                        logging.info(batch["completion"][0])
                    
                    logging.info("=" * 60)
                
                # Clear memory every 50 steps (like inference.py)
                if batch_idx > 0 and batch_idx % 50 == 0:
                    logging.info(f"Clearing memory at MLP batch {batch_idx}")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                        logging.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                
                # Check MLP health
                is_healthy, _ = model.check_mlp_health()
                if not is_healthy:
                    logging.warning(f"MLP unhealthy at batch {batch_idx}, resetting weights")
                    model.reset_mlp_weights()
                    optimizer.zero_grad()
                    accumulated_loss = 0
                    continue
                
                # Forward pass
                outputs = model(batch)
                loss = outputs["loss"]
                loss = loss / 8
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"Invalid loss at batch {batch_idx}: {loss}")
                    continue
                
                loss.backward()
                accumulated_loss += loss.item()
                
                # Update every accumulation_steps
                if (batch_idx + 1) % 8 == 0:
                    # Check gradients
                    max_grad_norm = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            max_grad_norm = max(max_grad_norm, param.grad.norm().item())
                    
                    if max_grad_norm > 10.0:
                        logging.warning(f"Large gradient norm {max_grad_norm:.2f} at batch {batch_idx}, skipping update")
                        optimizer.zero_grad()
                        accumulated_loss = 0
                        continue
                    
                    # Clip and update
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), 
                        max_norm=1.0
                    )
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Track progress
                    valid_update_steps += 1
                    avg_loss = accumulated_loss / 8
                    total_loss += accumulated_loss
                    accumulated_loss = 0
                    
                    # Update progress bar with metrics (like inference.py)
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.6f}", 
                        "steps": valid_update_steps,
                        "max_grad": f"{max_grad_norm:.3f}"
                    })
                    
                    # Log detailed progress every 5 update steps (like inference.py)
                    if valid_update_steps % 5 == 0:
                        logging.info(
                            f"MLP Cycle {cycle}, Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, "
                            f"Update Step {valid_update_steps}, Loss: {avg_loss:.6f}, "
                            f"Max Grad Norm: {max_grad_norm:.3f}"
                        )
                
            except Exception as e:
                logging.error(f"Error in MLP training batch {batch_idx}: {e}")
                continue
        
        # Close progress bar to prevent log interference (like inference.py)
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
    """Discover symbol mappings using actual dataset labels and model's existing methods"""
    try:
        logging.info("Starting symbol discovery with actual dataset labels...")
        
        # Get actual label tokens from the model
        actual_labels = model.label_tokens if hasattr(model, 'label_tokens') else []
        logging.info(f"Using actual labels from dataset: {actual_labels}")
        
        if not actual_labels:
            logging.warning("No label tokens found in model!")
            return {}
        
        # Store labels in model for easy access during discovery
        if not hasattr(model, 'label_tokens'):
            model.label_tokens = actual_labels
        
        # Find symbol mappings using the model's existing method
        logging.info("Finding symbol mappings using model.find_symbol_mappings()...")
        raw_mappings = model.find_symbol_mappings()
        
        # Convert raw token mappings to human-readable format
        token_mappings = {}
        if raw_mappings:
            tokenizer = model.llama_tokenizer
            
            logging.info(f"Found {len(raw_mappings)} raw token mappings")
            for source_token_id, target_token_id in raw_mappings.items():
                source_text = tokenizer.decode([int(source_token_id)], skip_special_tokens=False).strip()
                target_text = tokenizer.decode([int(target_token_id)], skip_special_tokens=False).strip()
                
                if source_text and target_text and source_text != target_text:
                    token_mappings[source_text] = target_text
                    logging.info(f"  Token mapping: '{source_text}' -> '{target_text}'")
        
        # Calculate label mappings using actual dataset labels and model's methods
        label_mappings = {}
        
        logging.info("Calculating label mappings for actual dataset labels...")
        for label in actual_labels:
            logging.info(f"Processing label: '{label}'")
            
            # Use the model's existing label_to_token_ids mapping
            if hasattr(model, 'label_to_token_ids') and label in model.label_to_token_ids:
                label_token_ids = model.label_to_token_ids[label]
            else:
                # Fallback: tokenize the label
                label_token_ids = tokenizer.encode(label, add_special_tokens=False)
            
            logging.info(f"  Label '{label}' has token IDs: {label_token_ids}")
            
            # Build combined result by applying token mappings
            combined_result = ""
            original_tokens = []
            mapped_tokens = []
            
            for token_id in label_token_ids:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                original_tokens.append(token_text)
                
                # Check if this token has a mapping in raw_mappings
                if token_id in raw_mappings:
                    mapped_token_id = raw_mappings[token_id]
                    mapped_text = tokenizer.decode([mapped_token_id], skip_special_tokens=False).strip()
                    mapped_tokens.append(mapped_text)
                    combined_result += mapped_text
                    logging.info(f"    Token '{token_text}' (ID: {token_id}) -> '{mapped_text}' (ID: {mapped_token_id})")
                else:
                    mapped_tokens.append(token_text)
                    combined_result += token_text
                    logging.info(f"    Token '{token_text}' (ID: {token_id}) -> unchanged")
            
            # Only save if the combined result is different from original
            if combined_result and combined_result != label:
                label_mappings[label] = combined_result
                logging.info(f"  ✓ Label mapping: '{label}' -> '{combined_result}'")
                logging.info(f"    Original tokens: {original_tokens}")
                logging.info(f"    Mapped tokens: {mapped_tokens}")
            else:
                logging.info(f"  ✗ No change for label: '{label}'")
        
        # Use model's existing symbol replacement dict method
        symbol_replacement_dict = {}
        if hasattr(model, 'get_symbol_replacement_dict'):
            symbol_replacement_dict = model.get_symbol_replacement_dict()
            logging.info(f"Got symbol replacement dict from model: {symbol_replacement_dict}")
        
        # Save comprehensive results
        results_file = os.path.join(args.output_dir, "symbol_discovery_results.json")
        all_results = {}
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except:
                all_results = {}
        
        # Save detailed results for this cycle
        all_results[f"cycle_{cycle}"] = {
            "actual_labels": actual_labels,
            "token_mappings": token_mappings,
            "label_mappings": label_mappings,
            "symbol_replacement_dict": symbol_replacement_dict,
            "raw_token_mappings": {str(k): int(v) for k, v in raw_mappings.items()} if raw_mappings else {},
            "num_actual_labels": len(actual_labels),
            "num_token_mappings": len(token_mappings),
            "num_label_mappings": len(label_mappings),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Log summary
        logging.info("=== Symbol Discovery Summary ===")
        logging.info(f"Actual dataset labels: {len(actual_labels)} -> {actual_labels}")
        logging.info(f"Raw token mappings found: {len(raw_mappings)}")
        logging.info(f"Human-readable token mappings: {len(token_mappings)}")
        logging.info(f"Label mappings found: {len(label_mappings)}")
        
        if label_mappings:
            logging.info("Label transformations discovered:")
            for old, new in label_mappings.items():
                logging.info(f"  '{old}' -> '{new}'")
        else:
            logging.info("No label transformations discovered")
        
        if symbol_replacement_dict:
            logging.info("Symbol replacement dictionary:")
            for old, new in symbol_replacement_dict.items():
                logging.info(f"  '{old}' -> '{new}'")
        
        return label_mappings
        
    except Exception as e:
        logging.error(f"Error during symbol discovery: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {}

def replace_symbols_in_batch(batch, symbol_mappings, tokenizer):
    """Replace old symbols with new symbols in batch data"""
    if not symbol_mappings:
        return batch
    
    # This is a simplified version - you'll need to implement based on your batch structure
    # The key is to find text fields and replace old symbols with new ones
    
    try:
        # Example: if batch has 'text' field
        if 'text' in batch:
            for i, text in enumerate(batch['text']):
                for old_symbol, new_symbol in symbol_mappings.items():
                    text = text.replace(old_symbol, new_symbol)
                batch['text'][i] = text
        
        # Re-tokenize if needed
        # You might need to update input_ids, attention_mask, etc.
        
    except Exception as e:
        logging.error(f"Error replacing symbols in batch: {e}")
    
    return batch

def main():
    # Parse arguments and setup
    args = parse_args()
    args = setup_unified_logging(args)
    
    logging.info("=== Starting Unified Symbol Discovery Training ===")
    
    try:
        # Parse dataset types
        dataset_types = [DatasetType(dt.strip()) for dt in args.dataset_type.split("-")]
        
        # Get label tokens
        from train_mlp_embeddings import get_label_tokens_from_dataset_types
        label_tokens = get_label_tokens_from_dataset_types(dataset_types)
        
        # Create model
        logging.info(f"Creating MLPSalmonn model with label tokens: {label_tokens}")
        model = MLPSalmonn(
            llama_path="lmsys/vicuna-13b-v1.1",
            whisper_path="openai/whisper-large-v2",
            beats_path="/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            label_tokens=label_tokens,
            hidden_dim=args.hidden_dim,
            freeze_base=True,
            lora=True,
            lora_rank=8,
            lora_alpha=32,
            lora_dropout=0.05,
            device=args.device,
            low_resource=True
        )
        
        # checkpoint = load_checkpoint(args.initial_model_path)
        # model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Create dataset and dataloader
        from data.model_processors import get_processor
        processor = get_processor(args.model_type, model.input_processor, model.llama_tokenizer)
        
        # Load datasets - separate train/val for different purposes
        train_datasets = {}  # For normal training
        meta_datasets = {}   # For meta-learning (MLP training)
        
        for dt in dataset_types:
            full_train_dataset = load_dataset(dt, split="train")
            full_val_dataset = load_dataset(dt, split="validation")  # Use val for meta-learning
            
            if args.max_samples > 0:
                train_datasets[dt] = full_train_dataset.select(range(args.max_samples))
                meta_datasets[dt] = full_val_dataset.select(range(args.max_samples // 2))
            else:
                train_datasets[dt] = full_train_dataset
                meta_datasets[dt] = full_val_dataset
            
            logging.info(f"Dataset {dt}: {len(train_datasets[dt])} train, {len(meta_datasets[dt])} meta")

        # Create separate dataloaders
        train_dataset = DatasetFactory.create_dataset(
            dataset_type=dataset_types,
            dataset=train_datasets,  # Regular training data
            processor=processor,
            is_training=True,
            input_mode="speech_only",
            fewshot_mode="text",
            num_examples=5,
            random_examples=True,
            model_type=args.model_type,
            run_name=args.run_name
        )
        
        meta_dataset = DatasetFactory.create_dataset(
            dataset_type=dataset_types,
            dataset=meta_datasets,   # Meta-learning data (validation)
            processor=processor,
            is_training=True,
            input_mode="speech_only",
            fewshot_mode="text",
            num_examples=5,
            random_examples=True,
            model_type=args.model_type,
            run_name=args.run_name
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=processor.collate_batch)
        meta_loader = DataLoader(meta_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=processor.collate_batch)
        
        logging.info(f"Created dataloaders: {len(train_loader)} train batches, {len(meta_loader)} meta batches")
        
        # Main alternating training loop - SAME pattern for all cycles
        current_symbols = {}
        
        for cycle in range(1, args.total_cycles + 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Starting Training Cycle {cycle}/{args.total_cycles}")
            logging.info(f"{'='*60}")
            
            # Phase 1: Train MLP first (stabilizes symbol mappings)
            logging.info(f"Phase 1: MLP training - Cycle {cycle}")
            model, discovered_symbols = train_mlp_phase(model, meta_loader, args, cycle)
            
            # Update symbols after MLP training
            if discovered_symbols:
                current_symbols.update(discovered_symbols)
                logging.info(f"Updated symbol mappings: {current_symbols}")
            
            # Phase 2: Train LoRA (adapts to stabilized mappings)
            logging.info(f"Phase 2: LoRA training - Cycle {cycle}")
            model = train_lora_phase(model, train_loader, args, cycle, current_symbols)
            
            logging.info(f"Completed Training Cycle {cycle}/{args.total_cycles}")
        
        logging.info("=== Unified Symbol Discovery Training Completed ===")
        
    except Exception as e:
        logging.error(f"Error during unified training: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


# CUDA_VISIBLE_DEVICES=0 python /data2/neeraja/neeraja/code/ICL/models/unified_symbol_training.py \
#     --initial_model_path /data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_greek-hvb_greek/checkpoints/epoch_1_loss_0.3191/model.pt \
#     --dataset_type voxceleb_greek-hvb_greek \
#     --lora_lr 1e-5 \
#     --mlp_lr 1e-4 \
#     --lora_epochs 2 \
#     --mlp_epochs 1 \
#     --total_cycles 3 \
#     --max_samples 64 \
#     --batch_size 1 \
#     --hidden_dim 8 \
#     --gradient_accumulation_steps 8 \
#     --max_grad_norm 1.0 \
#     --warmup_steps 100 \
#     --fp16 \
#     --run_name "unified_symbol_discovery_v2"