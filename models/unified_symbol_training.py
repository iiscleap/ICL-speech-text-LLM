#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import torch
import json
from datetime import datetime
from tqdm import tqdm
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_salmonn import MLPSalmonn, generate_one_word_two_token_symbols, create_label_mapping
from utils.data_utils import load_dataset
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType, get_dataset_config
from utils.training_utils import setup_logging, save_checkpoint
from torch.utils.data import DataLoader
from data.model_processors import get_processor

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Symbol Discovery and LoRA Training")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="salmonn")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    
    # Training parameters
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--mlp_lr", type=float, default=1e-4)
    parser.add_argument("--lora_epochs", type=int, default=2)
    parser.add_argument("--mlp_epochs", type=int, default=2)
    parser.add_argument("--total_cycles", type=int, default=3)
    parser.add_argument("--lora_final_epochs", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, default="voxceleb_greek")
    parser.add_argument("--max_samples", type=int, default=64)
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/data2/neeraja/neeraja/results/model_ICL/unified_training")
    parser.add_argument("--run_name", type=str, default="")
    
    return parser.parse_args()

def setup_unified_logging(args):
    """Setup logging with timestamp"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if not args.run_name:
        args.run_name = f"unified_{timestamp}_{args.dataset_type.replace('-', '_')}"
    
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
        force=True
    )
    
    logging.info(f"Arguments: {args}")
    return args

def train_mlp_phase_adaptive(model, dataloader, args, cycle, num_epochs, current_mappings):
    """Train MLP with adaptive learning rate and scale factor"""
    logging.info(f"=== MLP Training: {num_epochs} epochs, building on current mappings ===")
    
    model.set_mlp_training_mode()
    model.store_discoveries = True
    
    # ✅ Set current cycle for adaptive scale factor
    model.current_cycle = cycle
    logging.info(f"Set model.current_cycle = {cycle}")
    
    model.update_label_tokens(current_mappings)
    logging.info(f"Starting MLP training with symbols: {list(current_mappings.values())}")
    
    # Clear previous discoveries
    if hasattr(model, 'discovered_mappings'):
        model.discovered_mappings.clear()
    
    # ✅ Progressive learning rate strategy
    # base_mlp_lr = args.mlp_lr
    
    # Increase learning rate for early cycles to encourage exploration
    # if cycle == 0:
    #     # First cycle: Higher LR for breaking identity mappings
    #     mlp_lr = base_mlp_lr * 3.0  # 3x higher for initial exploration
    #     logging.info(f"Cycle {cycle}: Using exploration LR = {mlp_lr:.2e} (3x base)")
    # elif cycle == 1:
    #     # Second cycle: Medium LR for continued exploration
    #     mlp_lr = base_mlp_lr * 2.0  # 2x higher
    #     logging.info(f"Cycle {cycle}: Using medium LR = {mlp_lr:.2e} (2x base)")
    # else:
    #     # Later cycles: Gradual decay for refinement
    #     mlp_lr = base_mlp_lr * (0.8 ** (cycle - 2))
    #     logging.info(f"Cycle {cycle}: Using refinement LR = {mlp_lr:.2e} (decay)")

    mlp_lr = args.mlp_lr
    
    # ✅ Use SGD with higher momentum for exploration
    optimizer = torch.optim.SGD(
        model.position_wise_mlp.parameters(),
        lr=mlp_lr,
        momentum=0.95,  # Higher momentum for smoother exploration
        weight_decay=1e-5,
        nesterov=True
    )
    
    # ✅ Add learning rate scheduler within each epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(dataloader) * num_epochs,
        eta_min=mlp_lr * 0.1  # Minimum LR is 10% of starting LR
    )
    
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    logging.info(f"MLP Cycle {cycle}: SGD LR = {mlp_lr:.2e}, Momentum = 0.95, Effective batch size = {effective_batch_size}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        accumulated_loss = 0
        
        logging.info(f"MLP Cycle {cycle} Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"MLP C{cycle} E{epoch+1}")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            try:
                # ✅ ADD LOGGING FOR STEP 1 ONLY
                if step == 0:  # Log only for step 1
                    logging.info("=== STEP 1 BATCH CONTENT ===")
                    
                    # Log ORIGINAL batch content
                    if "prompt" in batch:
                        logging.info("ORIGINAL PROMPTS:")
                        for i, prompt in enumerate(batch["prompt"]):  # Show first 2
                            logging.info(f"  [{i}] {prompt}")
                    
                    if "completion" in batch:
                        logging.info("ORIGINAL COMPLETIONS:")
                        for i, completion in enumerate(batch["completion"]):  # Show first 2
                            logging.info(f"  [{i}] {completion}")
                    
                    logging.info(f"SYMBOL MAPPINGS: {current_mappings}")
                
                # Apply symbol replacement
                batch = replace_symbols_in_batch(batch, current_mappings)
                
                # ✅ LOG UPDATED BATCH CONTENT FOR STEP 1
                if step == 0:
                    logging.info("UPDATED AFTER SYMBOL REPLACEMENT:")
                    
                    if "prompt" in batch:
                        logging.info("UPDATED PROMPTS:")
                        for i, prompt in enumerate(batch["prompt"]):  # Show first 2
                            logging.info(f"  [{i}] {prompt}")
                    
                    if "completion" in batch:
                        logging.info("UPDATED COMPLETIONS:")
                        for i, completion in enumerate(batch["completion"]):  # Show first 2
                            logging.info(f"  [{i}] {completion}")
                    
                    logging.info("=== END STEP 1 BATCH CONTENT ===")
                
                # ✅ Clear cache every 3 steps to prevent OOM
                if step % 3 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = model(batch)
                loss = outputs["loss"]
                
                # ✅ Check for NaN loss and invalid mappings
                if torch.isnan(loss).any() or loss.item() == 0.0:
                    logging.error(f"Invalid loss at step {step}: {loss.item()} - skipping batch")
                    optimizer.zero_grad()
                    continue
                
                scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
                
                accumulated_loss += loss.item()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # ✅ More lenient gradient checking for exploration
                    has_nan_grad = False
                    total_norm = 0
                    for name, param in model.named_parameters():
                        if param.grad is not None and 'position_wise_mlp' in name:
                            grad_norm = param.grad.data.norm(2)
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                logging.error(f"NaN/Inf gradient in {name}")
                                has_nan_grad = True
                            else:
                                total_norm += grad_norm.item() ** 2
                    
                    total_norm = total_norm ** 0.5
                    
                    # ✅ Allow larger gradients for exploration (especially early cycles)
                    max_grad_norm = 5.0 if cycle == 0 else 3.0 if cycle == 1 else 1.0
                    
                    if has_nan_grad:
                        logging.warning(f"NaN gradient detected - zeroing gradients")
                        optimizer.zero_grad()
                        accumulated_loss = 0
                        continue
                    max_grad_norm = max_grad_norm*10
                    if total_norm > max_grad_norm :  # Only warn for very large gradients
                        logging.warning(f"Large gradient norm: {total_norm:.2f} (max allowed: {max_grad_norm})")
                    
                    # ✅ Clip gradients based on cycle
                    torch.nn.utils.clip_grad_norm_(
                        [p for n, p in model.named_parameters() if 'position_wise_mlp' in n], 
                        max_norm=max_grad_norm
                    )
                    
                    optimizer.step()
                    scheduler.step()  # Update learning rate
                    optimizer.zero_grad()
                    
                    # ✅ Log current learning rate occasionally
                    current_lr = scheduler.get_last_lr()[0]
                    avg_accumulated_loss = accumulated_loss / args.gradient_accumulation_steps
                    progress_bar.set_postfix({
                        "acc_loss": f"{avg_accumulated_loss:.6f}",
                        "lr": f"{current_lr:.2e}"
                    })
                    
                    total_loss += accumulated_loss
                    valid_batches += 1
                    accumulated_loss = 0
                
                # Show discoveries every 10 gradient updates (not every step)
                if step > 0 and step % (1 * args.gradient_accumulation_steps) == 0:
                    current_discoveries = model.get_final_discovered_symbols()
                    if current_discoveries:
                        unique_discoveries = len(set(current_discoveries.values()))
                        logging.info(f"Step {step}: {unique_discoveries} unique discoveries so far")
                    
                    if current_discoveries:
                        logging.info(f"Step {step} discoveries: {list(current_discoveries.values())[:3]}...")
                    
                    # ✅ Simple detailed logging every 1 steps
                    if hasattr(model, 'discovery_similarities') and model.discovery_similarities:
                        total = len(model.discovery_similarities)
                        identity_count = sum(1 for d in model.discovery_similarities.values() if d['random_token'] == d['discovered_token'])
                        avg_sim = sum(d['similarity'] for d in model.discovery_similarities.values()) / total
                        
                        logging.info(f"Step {step} Status: {total} discoveries, {identity_count} identity ({identity_count/total*100:.1f}%), avg_sim: {avg_sim:.3f}")
                        
                        # Show first 3 discoveries with full details
                        for i, (key, disc) in enumerate(list(model.discovery_similarities.items())[:3]):
                            logging.info(f"  {i+1}. {disc['random_token']}('{disc['random_text']}') -> {disc['discovered_token']}('{disc['discovered_text']}') [sim: {disc['similarity']:.3f}]")
                
            except Exception as e:
                logging.error(f"Error in MLP batch {step}: {e}")
                optimizer.zero_grad()
                # ✅ Clear cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # Handle remaining gradients at epoch end
        if accumulated_loss > 0:
            max_grad_norm = 5.0 if cycle == 0 else 3.0 if cycle == 1 else 1.0
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in model.named_parameters() if 'position_wise_mlp' in n], 
                max_norm=max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss
            valid_batches += 1
        
        progress_bar.close()
        
        # ✅ Clear cache at end of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if valid_batches > 0:
            epoch_loss = total_loss / valid_batches
            final_lr = scheduler.get_last_lr()[0]
            logging.info(f"✓ MLP Epoch {epoch+1} completed: Average loss = {epoch_loss:.6f}, Final LR = {final_lr:.2e}")
            
            # ✅ Log discovery progress after each epoch
            current_discoveries = model.get_final_discovered_symbols()
            if current_discoveries:
                unique_discoveries = len(set(current_discoveries.values()))
                identity_mappings = sum(1 for orig, disc in current_discoveries.items() if orig == disc)
                logging.info(f"Epoch {epoch+1} Discovery Status: {unique_discoveries} unique, {identity_mappings} identity mappings")
    
    # Extract final discoveries
    final_discoveries = model.get_final_discovered_symbols()
    model.store_discoveries = False
    
    return final_discoveries

def train_lora_phase_adaptive(model, dataloader, args, cycle, current_mappings, num_epochs):
    """Train LoRA with gradient accumulation"""
    logging.info(f"=== LoRA Training: {num_epochs} epochs ===")
    
    model.set_lora_training_mode()
    
    if current_mappings:
        model.update_label_tokens(current_mappings)
        logging.info(f"LoRA training with symbols: {list(current_mappings.values())}")
    
    lora_lr = args.lora_lr * (1.1 ** min(cycle, 2))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lora_lr,
        weight_decay=0.01
    )
    
    # ✅ Effective batch size calculation
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    logging.info(f"LoRA Cycle {cycle}: LR = {lora_lr:.2e}, Effective batch size = {effective_batch_size}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        accumulated_loss = 0
        
        logging.info(f"LoRA Cycle {cycle} Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc=f"LoRA C{cycle} E{epoch+1}")
        
        optimizer.zero_grad()  # ✅ Initialize gradients
        
        for step, batch in enumerate(progress_bar):
            try:

                if step == 0:  # Log only for step 1
                    logging.info("=== STEP 1 BATCH CONTENT ===")
                    
                    # Log ORIGINAL batch content
                    if "prompt" in batch:
                        logging.info("ORIGINAL PROMPTS:")
                        for i, prompt in enumerate(batch["prompt"]):  # Show first 2
                            logging.info(f"  [{i}] {prompt}")
                    
                    if "completion" in batch:
                        logging.info("ORIGINAL COMPLETIONS:")
                        for i, completion in enumerate(batch["completion"]):  # Show first 2
                            logging.info(f"  [{i}] {completion}")
                    
                    logging.info(f"SYMBOL MAPPINGS: {current_mappings}")
                
                # Apply symbol replacement
                if current_mappings:
                    batch = replace_symbols_in_batch(batch, current_mappings)
                
                # ✅ LOG UPDATED BATCH CONTENT FOR STEP 1
                if step == 0:
                    logging.info("UPDATED AFTER SYMBOL REPLACEMENT:")
                    
                    if "prompt" in batch:
                        logging.info("UPDATED PROMPTS:")
                        for i, prompt in enumerate(batch["prompt"]):  # Show first 2
                            logging.info(f"  [{i}] {prompt}")
                    
                    if "completion" in batch:
                        logging.info("UPDATED COMPLETIONS:")
                        for i, completion in enumerate(batch["completion"]):  # Show first 2
                            logging.info(f"  [{i}] {completion}")
                    
                    logging.info("=== END STEP 1 BATCH CONTENT ===")
                
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = model(batch)
                loss = outputs["loss"]
                
                # ✅ Scale loss by accumulation steps
                scaled_loss = loss / args.gradient_accumulation_steps
                scaled_loss.backward()
                
                accumulated_loss += loss.item()
                
                # ✅ Update weights every gradient_accumulation_steps
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Log accumulated loss
                    avg_accumulated_loss = accumulated_loss / args.gradient_accumulation_steps
                    progress_bar.set_postfix({"acc_loss": f"{avg_accumulated_loss:.4f}"})
                    
                    total_loss += accumulated_loss
                    valid_batches += 1
                    accumulated_loss = 0
                
            except Exception as e:
                logging.error(f"Error in LoRA batch {step}: {e}")
                continue
        
        # ✅ Handle remaining gradients at epoch end
        if accumulated_loss > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss
            valid_batches += 1
        
        progress_bar.close()
        
        if valid_batches > 0:
            epoch_loss = total_loss / valid_batches
            logging.info(f"✓ LoRA Epoch {epoch+1} completed: Average accumulated loss = {epoch_loss:.4f}")
    
    return current_mappings

def replace_symbols_in_batch(batch, symbol_mappings):
    """Replace symbols in batch prompts and completions"""
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
    """Load training and validation datasets"""
    train_datasets = {}
    val_datasets = {}
    
    for dataset_name in datasets:
        try:
            dataset_type = DatasetType(dataset_name)
            
            # Load both train and validation
            full_train_dataset = load_dataset(dataset_type, split="train")
            full_val_dataset = load_dataset(dataset_type, split="validation")
            
            if args.max_samples > 0:
                train_datasets[dataset_type] = full_train_dataset.select(range(args.max_samples))
                # Use smaller validation set for MLP discovery
                val_samples = min(args.max_samples // 2, len(full_val_dataset))
                val_datasets[dataset_type] = full_val_dataset.select(range(val_samples))
            else:
                train_datasets[dataset_type] = full_train_dataset
                val_datasets[dataset_type] = full_val_dataset
            
            logging.info(f"✓ Loaded {dataset_name}: {len(train_datasets[dataset_type])} train, {len(val_datasets[dataset_type])} val samples")
            
        except Exception as e:
            logging.error(f"✗ Failed to load dataset {dataset_name}: {e}")
            continue
    
    return train_datasets, val_datasets

def create_combined_dataloader(datasets, processor, args, shuffle=False):
    """Create combined dataloader"""
    dataset_types = list(datasets.keys())
    
    combined_dataset = DatasetFactory.create_dataset(
        dataset_type=dataset_types,
        dataset=datasets,
        processor=processor,
        is_training=shuffle,
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
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=processor.collate_batch,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

def generate_training_schedule(args):
    """Generate dynamic training schedule based on parameters"""
    logging.info(f"Generating schedule: {args.total_cycles} cycles, {args.lora_epochs} LoRA epochs, {args.mlp_epochs} MLP epochs")
    
    schedule = [
        {"phase": "lora_initial", "epochs": 1, "description": "Initial task learning"}
    ]
    
    for cycle in range(args.total_cycles):
        current_mlp_epochs = max(1, args.mlp_epochs - (cycle // 2))
        
        schedule.extend([
            {"phase": "mlp", "epochs": current_mlp_epochs, "description": f"Cycle {cycle+1} symbol discovery"},
            {"phase": "lora", "epochs": args.lora_epochs, "description": f"Cycle {cycle+1} adaptation"}
        ])
    
    schedule.append({
        "phase": "lora_final", 
        "epochs": args.lora_final_epochs, 
        "description": "Final task optimization"
    })
    
    logging.info(f"Generated training schedule with {len(schedule)} steps:")
    for i, step in enumerate(schedule):
        logging.info(f"  Step {i+1}: {step['phase']} ({step['epochs']} epochs) - {step['description']}")
    
    return schedule

def save_trainable_checkpoint(model, path, epoch=0, loss=0.0):
    """Save only trainable parameters to reduce checkpoint size and save time"""
    full_state_dict = model.state_dict()
    trainable_state_dict = {}
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        # Save if parameter requires grad OR if it's MLP (always save MLP)
        if param.requires_grad or 'position_wise_mlp' in name:
            trainable_state_dict[name] = full_state_dict[name]
            trainable_params += param.numel()
    
    checkpoint = {
        "trainable_state_dict": trainable_state_dict,
        "epoch": epoch,
        "loss": loss,
        "total_params": total_params,
        "trainable_params": trainable_params
    }
    
    torch.save(checkpoint, path)
    logging.info(f"✓ Saved {trainable_params:,} trainable params (of {total_params:,} total) to {path}")

def main():
    args = parse_args()
    setup_unified_logging(args)
    
    logging.info("=== Unified Symbol Discovery and LoRA Training ===")
    logging.info(f"Training configuration:")
    logging.info(f"  Total cycles: {args.total_cycles}")
    logging.info(f"  LoRA epochs per cycle: {args.lora_epochs}")
    logging.info(f"  MLP epochs per cycle: {args.mlp_epochs}")
    logging.info(f"  Final LoRA epochs: {args.lora_final_epochs}")
    
    # Setup tokenizer and datasets
    from transformers import LlamaTokenizer
    llama_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1", use_fast=False)
    llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llama_tokenizer.padding_side = "right"
    
    # ✅ Load both train and validation datasets
    train_datasets, val_datasets = load_datasets(args, args.dataset_type.split('-'))
    processor = get_processor(args.model_type, tokenizer=llama_tokenizer)
    
    # ✅ Create separate dataloaders for train and validation
    train_dataloader = create_combined_dataloader(train_datasets, processor, args, shuffle=True)
    val_dataloader = create_combined_dataloader(val_datasets, processor, args, shuffle=False)
    
    logging.info(f"✓ Created train dataloader: {len(train_dataloader)} batches")
    logging.info(f"✓ Created validation dataloader: {len(val_dataloader)} batches")
    
    # Generate initial symbols
    dataset_names = args.dataset_type.split('-')
    all_valid_labels = set()
    for dataset_name in dataset_names:
        dataset_type = DatasetType(dataset_name)
        config = get_dataset_config(dataset_type)
        all_valid_labels.update(config.valid_labels)
    
    dataset_labels = sorted(list(all_valid_labels))
    random_symbols = generate_one_word_two_token_symbols(len(dataset_labels), llama_tokenizer)
    current_symbol_mappings = create_label_mapping(dataset_labels, random_symbols)
    
    # ✅ ADD THIS ONE LINE
    
    
    # Initialize model
    model = MLPSalmonn(
        device=args.device,
        label_tokens=list(current_symbol_mappings.values()),
        hidden_dim=args.hidden_dim,
        dropout=0.1,
        freeze_base=True,
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        low_resource=True
    )

    model.original_to_random_mapping = current_symbol_mappings.copy()
    
    # Apply initial random symbol mappings to model
    logging.info(f"Initial random symbol mappings: {current_symbol_mappings}")
    model.update_label_tokens(current_symbol_mappings)
    
    # Generate dynamic training schedule
    training_schedule = generate_training_schedule(args)
    
    cycle = 0
    for step, schedule in enumerate(training_schedule):
        logging.info(f"=== Step {step+1}/{len(training_schedule)}: {schedule['description']} ({schedule['epochs']} epochs) ===")
        
        if schedule["phase"] in ["lora_initial", "lora", "lora_final"]:
            # ✅ LoRA training uses TRAIN set
            logging.info("Using TRAIN set for LoRA training")
            current_symbol_mappings = train_lora_phase_adaptive(
                model, train_dataloader, args, cycle, current_symbol_mappings, schedule["epochs"]
            )
        
        elif schedule["phase"] == "mlp":
            # ✅ MLP training uses VALIDATION set for better generalization
            logging.info("Using VALIDATION set for MLP symbol discovery")
            discovered_symbols = train_mlp_phase_adaptive(
                model, val_dataloader, args, cycle, schedule["epochs"], current_symbol_mappings
            )
            
            if discovered_symbols:
                current_symbol_mappings = discovered_symbols
                
                # Save discoveries with run_name prefix
                discoveries_dir = os.path.join(args.output_dir, "discoveries")
                os.makedirs(discoveries_dir, exist_ok=True)
                with open(os.path.join(discoveries_dir, f"{args.run_name}_step_{step}_symbols.json"), 'w') as f:
                    json.dump(discovered_symbols, f, indent=2)
                
                logging.info(f"Step {step}: Updated symbols: {discovered_symbols}")
            
            cycle += 1
        
        # Save checkpoint
        logging.info(f"Saving checkpoint for step {step}...")
        checkpoint_dir = os.path.join(args.output_dir, f"{args.run_name}_step_{step}_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        save_trainable_checkpoint(
            model=model,
            path=os.path.join(checkpoint_dir, "model.pt"),
            epoch=step,
            loss=0.0
        )
    
    # Save final model
    logging.info("Saving final model...")
    final_checkpoint_dir = os.path.join(args.output_dir, f"{args.run_name}_final_model")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    save_trainable_checkpoint(
        model=model,
        path=os.path.join(final_checkpoint_dir, "model.pt"),
        epoch=-1,
        loss=0.0
    )
    
    final_mappings_file = os.path.join(args.output_dir, f"{args.run_name}_final_symbol_mappings.json")
    with open(final_mappings_file, 'w') as f:
        json.dump({
            "final_mappings": current_symbol_mappings,
            "training_schedule": training_schedule,
            "args": vars(args)
        }, f, indent=2)
    
    logging.info(f"Training complete! Files saved with prefix: {args.run_name}")
    logging.info(f"Final model: {final_checkpoint_dir}/model.pt")
    logging.info(f"Final mappings: {final_mappings_file}")

if __name__ == "__main__":
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