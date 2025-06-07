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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
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
    
    # NEW: MLP training frequency options
    parser.add_argument("--mlp_training_mode", type=str, default="first_and_second", 
                       choices=["every_cycle", "first_only", "first_and_last"],
                       help="When to train MLP: every_cycle, first_only, or first_and_second")
    
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

def should_train_mlp(cycle, total_cycles, mlp_training_mode):
    """Determine if MLP should be trained in this cycle"""
    if mlp_training_mode == "every_cycle":
        return True
    elif mlp_training_mode == "first_only":
        return cycle == 0
    elif mlp_training_mode == "first_and_second":
        return cycle == 0 or cycle == 1
    else:
        return True  # Default to every cycle

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
                    batch = replace_symbols_in_batch(batch, current_symbols)
                
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
    """Train MLP for symbol discovery"""
    logging.info(f"=== Starting MLP Training Phase - Cycle {cycle} ===")
    
    # Freeze LoRA, unfreeze MLP
    model.freeze_lora_weights()
    model.unfreeze_mlp_weights()
    
    # ✅ FIX: Much more lenient gradient clipping
    max_grad_norm = 100.0  # Increased from 50.0
    
    # MLP optimizer with higher learning rate
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.mlp_lr,  # Usually 1e-4
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     weight_decay=0.01
    # )

    optimizer = torch.optim.SGD(
        model.position_wise_mlp.parameters(),
        lr=args.mlp_lr * 0.1,  # Reduce LR for SGD (0.0001 -> 0.00001)
        momentum=0.9,          # Add momentum for better convergence
        weight_decay=1e-5,     # Small weight decay for regularization
        nesterov=True          # Nesterov momentum for better performance
    )
    
    # Scheduler
    total_steps = len(dataloader) * args.mlp_epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=2, verbose=True
    )
    
    # ✅ FIX: More lenient health check
    def is_mlp_healthy(model, loss):
        """Check if MLP is healthy with more lenient thresholds"""
        if torch.isnan(loss) or torch.isinf(loss):
            return False, "NaN/Inf loss"
        
        max_weight = 0
        max_grad = 0
        
        for param in model.position_wise_mlp.parameters():
            if param.data is not None:
                max_weight = max(max_weight, param.data.abs().max().item())
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())
        
        # ✅ FIX: Very lenient thresholds
        if max_weight > 1000.0:  # Much higher threshold
            return False, f"Extreme weights: {max_weight}"
        if max_grad > 10000.0:  # Much higher threshold
            return False, f"Extreme gradients: {max_grad}"
        
        return True, None
    
    # Training loop
    unhealthy_count = 0
    valid_update_steps = 0
    
    for epoch in range(args.mlp_epochs):
        model.train()
        total_loss = 0
        
        logging.info(f"Starting MLP Cycle {cycle} Epoch {epoch+1}/{args.mlp_epochs}")
        
        progress_bar = tqdm(dataloader, desc=f"MLP Cycle {cycle} Epoch {epoch+1} - Processing batch {len(dataloader)}")
        
        for step, batch in enumerate(progress_bar):
            try:

                if current_symbols:
                    batch = replace_symbols_in_batch(batch, current_symbols, model.llama_tokenizer)
                # Move batch to device
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # ✅ FIX: Check gradient norm but be more permissive
                total_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                # ✅ FIX: Only clip if really necessary, don't skip updates
                if total_norm > max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    logging.info(f"Gradient norm {total_norm:.2f} clipped to {max_grad_norm}")
                
                # ✅ FIX: Don't skip updates unless gradients are truly insane
                if total_norm > 10000.0:  # Much higher threshold
                    logging.error(f"Extreme gradient norm {total_norm:.2f}, skipping update")
                    optimizer.zero_grad()
                    continue
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad()
                valid_update_steps += 1
                
                # Health check with more tolerance
                is_healthy, reason = is_mlp_healthy(model, loss)
                if not is_healthy:
                    logging.warning(f"MLP unhealthy at batch {step}: {reason}")
                    unhealthy_count += 1
                    if unhealthy_count >= 10:  # Much more tolerance
                        logging.warning("Resetting MLP weights due to persistent instability")
                        model.reset_mlp_weights()
                        unhealthy_count = 0
                else:
                    unhealthy_count = 0
                
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "grad_norm": f"{total_norm:.2f}",
                    "valid_steps": valid_update_steps
                })
                
            except Exception as e:
                logging.error(f"Error in MLP training batch {step}: {e}")
                continue
        
        progress_bar.close()
        
        if valid_update_steps > 0:
            epoch_loss = total_loss / len(dataloader)
            logging.info(f"✓ MLP Cycle {cycle} Epoch {epoch+1} COMPLETED: Average loss = {epoch_loss:.6f}")
            logging.info(f"  Valid update steps: {valid_update_steps}")
            scheduler.step(epoch_loss)
        else:
            logging.error(f"❌ NO VALID UPDATE STEPS in MLP epoch {epoch+1} - MLP training failed!")

    
    logging.info(f"=== Completed MLP Training Phase - Cycle {cycle} ===")
    return model


def replace_symbols_in_batch(batch, symbol_mappings):
    """Replace symbols with minimal logging"""
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
            original_completion = completion
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
    args = parse_args()
    setup_unified_logging(args)
    
    logging.info("=== Unified Symbol Discovery and LoRA Training ===")
    

    # ✅ FIX: Clean up temp_model properly
    if args.model_type == "salmonn":
        from models.mlp_salmonn import MLPSalmonn, generate_one_word_two_token_symbols, create_label_mapping
        from transformers import LlamaTokenizer
        
        logging.info("Loading LLaMA tokenizer...")
        llama_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1",use_fast=False)
        llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        llama_tokenizer.padding_side = "right"

        logging.info("Tokenizer loaded successfully")

        train_datasets, meta_datasets = load_datasets(args, args.dataset_type.split('-'))
        processor = get_processor(args.model_type,tokenizer=llama_tokenizer)
        
        train_dataloader = create_combined_dataloader(train_datasets, processor, args, shuffle=True)
        meta_dataloader = create_combined_dataloader(meta_datasets, processor, args, shuffle=True)

        # Generate initial symbols with proper tokenizer
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
        

        random_symbols = generate_one_word_two_token_symbols(len(dataset_labels), llama_tokenizer)
        symbol_mappings = create_label_mapping(dataset_labels, random_symbols)
        
        
        model = MLPSalmonn(
            device=args.device,
            label_tokens=list(symbol_mappings.values()),
            hidden_dim=args.hidden_dim,
            dropout=0.1,
            freeze_base=True,
            lora=True,
            lora_rank=8,
            lora_alpha=32,
            lora_dropout=0.05,
            low_resource=True
        )
        
        # Store original mapping for discovery
        model.update_label_tokens(symbol_mappings)
    
    # Training cycles with IMPROVED ORDER
    for cycle in range(args.total_cycles):
        logging.info(f"=== Cycle {cycle + 1}/{args.total_cycles} ===")
        
        # 🔄 PHASE 1: LoRA Training (Adapt language model to current symbols)
        logging.info("Phase 1: Adapting language model to current symbols...")
        train_lora_phase(model, train_dataloader, args, cycle, symbol_mappings)
        
        # 🔄 PHASE 2: MLP Training (Learn symbol transformations with adapted model)
        if should_train_mlp(cycle, args.total_cycles, args.mlp_training_mode):
            logging.info("Phase 2: Learning symbol transformations with adapted model...")
            train_mlp_phase(model, meta_dataloader, args, cycle, symbol_mappings)
        
        # 🔄 PHASE 3: Symbol Discovery (Find better symbols)
        if cycle < args.total_cycles - 1:
            logging.info("Phase 3: Discovering improved symbols...")
            discoveries_dir = os.path.join(args.output_dir, "discoveries")
            os.makedirs(discoveries_dir, exist_ok=True)
            
            try:
                # Discover new token mappings
                token_mappings = model.discover_symbols(save_path=os.path.join(discoveries_dir, f"cycle_{cycle}_tokens.json"))
                
                # Convert to symbol mappings with error handling
                if token_mappings:
                    new_symbol_mappings = model.convert_token_mappings_to_text(token_mappings)
                    
                    # Save symbol mappings
                    with open(os.path.join(discoveries_dir, f"cycle_{cycle}_symbols.json"), 'w') as f:
                        json.dump(new_symbol_mappings, f, indent=2)
                    
                    # Update model with new mappings
                    model.update_label_tokens(new_symbol_mappings)
                    symbol_mappings = new_symbol_mappings  # Update for next cycle
                    
                    logging.info(f"Cycle {cycle} complete. Updated model with new symbol mappings.")
                    logging.info(f"Discovery files saved to: {discoveries_dir}")
                else:
                    logging.warning(f"No token mappings discovered in cycle {cycle}")
                    
            except Exception as e:
                logging.error(f"Symbol discovery failed in cycle {cycle}: {e}")
                # Continue with existing symbols

    logging.info("=== Training Complete ===")

# ✅ FIX: Correct the __name__ check
if __name__ == "__main__":  # Fixed the typo
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)