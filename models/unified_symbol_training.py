import os
import sys
import logging
import argparse
import torch
import json
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_salmonn import MLPSalmonn
from utils.data_utils import load_dataset
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType
from utils.training_utils import setup_logging, load_checkpoint, save_checkpoint
from torch.utils.data import DataLoader, SubsetRandomSampler

logger = logging.getLogger(__name__)

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
    """Setup logging for unified training"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if not args.run_name:
        args.run_name = f"unified_{timestamp}_{args.dataset_type.replace('-', '_')}"
    
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, "unified_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    logger.info(f"Arguments: {args}")
    return args

def train_lora_phase(model, dataloader, args, cycle, current_symbols=None):
    """Train LoRA weights using train.py pattern"""
    logger.info(f"=== Starting LoRA Training Phase - Cycle {cycle} ===")
    
    # Freeze MLP, unfreeze LoRA
    model.freeze_mlp_weights()
    model.unfreeze_lora_weights()
    
    # Log trainable parameters
    trainable, frozen = model.get_trainable_parameters()
    logger.info(f"Trainable parameters: {len(trainable)}")
    logger.info(f"Frozen parameters: {len(frozen)}")
    
    # LoRA optimizer (using AdamW like train.py)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lora_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Scheduler like train.py
    total_steps = len(dataloader) * args.lora_epochs // args.gradient_accumulation_steps
    from transformers import get_scheduler
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision setup like train.py
    from torch.cuda.amp import autocast, GradScaler
    if args.fp16:
        scaler = GradScaler()
        amp_dtype = torch.float16
    elif args.bf16:
        scaler = None
        amp_dtype = torch.bfloat16
    else:
        scaler = None
        amp_dtype = None
    
    # Training loop
    global_step = 0
    for epoch in range(args.lora_epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"LoRA Cycle {cycle} Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            try:
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # DON'T replace symbols in training data!
                # The MLP layer should learn to map symbols internally
                # Only use symbol replacement during inference/evaluation
                
                # Forward pass with mixed precision
                if amp_dtype is not None:
                    with autocast(dtype=amp_dtype):
                        outputs = model(batch)
                        loss = outputs["loss"]
                        loss = loss / args.gradient_accumulation_steps
                else:
                    outputs = model(batch)
                    loss = outputs["loss"]
                    loss = loss / args.gradient_accumulation_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                
                total_loss += loss.item() * args.gradient_accumulation_steps
                valid_batches += 1
                
                # Update progress
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.8f}"
                })
                
            except Exception as e:
                logger.error(f"Error in LoRA training batch {step}: {e}")
                continue
        
        if valid_batches > 0:
            epoch_loss = total_loss / valid_batches
            logger.info(f"LoRA Cycle {cycle} Epoch {epoch+1}: Average loss = {epoch_loss:.4f}")
            
            # Save checkpoint after each epoch like train.py
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
            logger.info(f"Saved LoRA checkpoint: {checkpoint_dir}")
        else:
            logger.error(f"No valid batches in LoRA epoch {epoch+1}")
    
    logger.info(f"=== Completed LoRA Training Phase - Cycle {cycle} ===")
    return model

def train_mlp_phase(model, dataloader, args, cycle):
    """Train MLP embeddings with scheduler"""
    logger.info(f"=== Starting MLP Training Phase - Cycle {cycle} ===")
    
    # Freeze LoRA, unfreeze MLP
    model.freeze_lora_weights()
    model.unfreeze_mlp_weights()
    
    # MLP optimizer (SGD to avoid NaN issues)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.mlp_lr,
        momentum=0.0,
        weight_decay=0.0
    )
    
    # Add scheduler for MLP training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    # Training loop with scheduler    
    for epoch in range(args.mlp_epochs):
        model.train()
        total_loss = 0
        valid_update_steps = 0
        accumulated_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"MLP Cycle {cycle} Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Check MLP health
                is_healthy, _ = model.check_mlp_health()
                if not is_healthy:
                    model.reset_mlp_weights()
                    optimizer.zero_grad()
                    accumulated_loss = 0
                    continue
                
                # Forward pass
                outputs = model(batch)
                loss = outputs["loss"]
                loss = loss / 8
                
                if torch.isnan(loss) or torch.isinf(loss):
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
                    
                    progress_bar.set_postfix({"loss": f"{avg_loss:.6f}", "steps": valid_update_steps})
                
            except Exception as e:
                logger.error(f"Error in MLP training batch {batch_idx}: {e}")
                continue
        
        if valid_update_steps > 0:
            epoch_loss = total_loss / valid_update_steps
            logger.info(f"MLP Cycle {cycle} Epoch {epoch+1}: Average loss = {epoch_loss:.6f}")
            
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
            logger.info(f"Saved MLP checkpoint: {checkpoint_dir}")
        else:
            logger.error(f"No valid update steps in MLP epoch {epoch+1}")
    
    # Discover new symbols
    logger.info(f"=== Symbol Discovery - Cycle {cycle} ===")
    discovered_symbols = discover_symbols(model, args, cycle)
    
    logger.info(f"=== Completed MLP Training Phase - Cycle {cycle} ===")
    return model, discovered_symbols

def discover_symbols(model, args, cycle):
    """Discover symbol mappings and save results"""
    try:
        # Find symbol mappings
        raw_mappings = model.find_symbol_mappings()
        
        # Convert to human-readable format
        symbol_mappings = {}
        if raw_mappings:
            tokenizer = model.llama_tokenizer
            
            for source_token_id, target_token_id in raw_mappings.items():
                source_text = tokenizer.decode([int(source_token_id)], skip_special_tokens=False).strip()
                target_text = tokenizer.decode([int(target_token_id)], skip_special_tokens=False).strip()
                
                if source_text and target_text and source_text != target_text:
                    symbol_mappings[source_text] = target_text
        
        # Reconstruct label mappings
        label_mappings = {}
        potential_labels = ['alpha', 'beta', 'gamma', 'delta', 'fred', 'plugh', 'xyzzy', 'thud', 'wibble', 'wobble', 'wubble', 'flob', 'zoop']
        
        for label in potential_labels:
            combined_result = ""
            for token in tokenizer.encode(label, add_special_tokens=False):
                token_text = tokenizer.decode([token], skip_special_tokens=False).strip()
                if token_text in symbol_mappings:
                    combined_result += symbol_mappings[token_text]
                else:
                    combined_result += token_text
            
            if combined_result and combined_result != label:
                label_mappings[label] = combined_result
        
        # Save results
        results_file = os.path.join(args.output_dir, "symbol_discovery_results.json")
        all_results = {}
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except:
                all_results = {}
        
        all_results[f"cycle_{cycle}"] = {
            "token_mappings": symbol_mappings,
            "label_mappings": label_mappings,
            "num_token_mappings": len(symbol_mappings),
            "num_label_mappings": len(label_mappings),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Discovered {len(label_mappings)} label mappings in cycle {cycle}:")
        for old, new in label_mappings.items():
            logger.info(f"  '{old}' -> '{new}'")
        
        return label_mappings
        
    except Exception as e:
        logger.error(f"Error during symbol discovery: {e}")
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
        logger.error(f"Error replacing symbols in batch: {e}")
    
    return batch

def main():
    # Parse arguments and setup
    args = parse_args()
    args = setup_unified_logging(args)
    
    logger.info("=== Starting Unified Symbol Discovery Training ===")
    
    try:
        # Parse dataset types
        dataset_types = [DatasetType(dt.strip()) for dt in args.dataset_type.split("-")]
        
        # Get label tokens
        from train_mlp_embeddings import get_label_tokens_from_dataset_types
        label_tokens = get_label_tokens_from_dataset_types(dataset_types)
        
        # Create model
        logger.info(f"Creating MLPSalmonn model with label tokens: {label_tokens}")
        model = MLPSalmonn(
            llama_path="lmsys/vicuna-13b-v1.1",
            whisper_path="openai/whisper-large-v2",
            beats_path="/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            label_tokens=label_tokens,
            hidden_dim=args.hidden_dim,
            freeze_base=True,
            lora=True,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            device=args.device,
            low_resource=True
        )
        
        # Load initial weights
        logger.info(f"Loading initial checkpoint from {args.initial_model_pathcheckpoint = load_checkpoint(args.initial_model_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Create dataset and dataloader
        from data.model_processors import get_processor
        processor = get_processor(args.model_type, model.input_processor, model.llama_tokenizer)
        
        # Load datasets - separate train/val for different purposes
        train_datasets = {}  # For normal training
        meta_datasets = {}   # For meta-learning (MLP training)
        
        for dt in dataset_types:
            full_train_dataset = load_dataset(dt, split="train")
            full_val_dataset = load_dataset(dt, split="val")  # Use val for meta-learning
            
            if args.max_samples > 0:
                train_datasets[dt] = full_train_dataset.select(range(args.max_samples))
                meta_datasets[dt] = full_val_dataset.select(range(args.max_samples // 2))
            else:
                train_datasets[dt] = full_train_dataset
                meta_datasets[dt] = full_val_dataset
            
            logger.info(f"Dataset {dt}: {len(train_datasets[dt])} train, {len(meta_datasets[dt])} meta")
}")
        checkpoint = load_checkpoint(args.initial_model_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Create dataset and dataloader
        from data.model_processors import get_processor
        processor = get_processor(args.model_type, model.input_processor, model.llama_tokenizer)
        
        # Load datasets - separate train/val for different purposes
        train_datasets = {}  # For normal training
        meta_datasets = {}   # For meta-learning (MLP training)
        
        for dt in dataset_types:
            full_train_dataset = load_dataset(dt, split="train")
            full_val_dataset = load_dataset(dt, split="val")  # Use val for meta-learning
            
            if args.max_samples > 0:
                train_datasets[dt] = full_train_dataset.select(range(args.max_samples))
                meta_datasets[dt] = full_val_dataset.select(range(args.max_samples // 2))
            else:
                train_datasets[dt] = full_train_dataset
                meta_datasets[dt] = full_val_dataset
            
            logger.info(f"Dataset {dt}: {len(train_datasets[dt])} train, {len(meta_datasets[dt])} meta")
        
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
        
        logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(meta_loader)} meta batches")
        
        # Main alternating training loop
        current_symbols = {}
        
        for cycle in range(1, args.total_cycles + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Training Cycle {cycle}/{args.total_cycles}")
            logger.info(f"{'='*60}")
            
            # Phase 1: Train LoRA on regular training data
            model = train_lora_phase(model, train_loader, args, cycle)
            
            # Phase 2: Train MLP on meta-learning data (validation set)
            model, discovered_symbols = train_mlp_phase(model, meta_loader, args, cycle)
            
            # Update current symbols for next cycle
            if discovered_symbols:
                current_symbols.update(discovered_symbols)
                logger.info(f"Updated symbol mappings for cycle {cycle+1}: {current_symbols}")
            
            logger.info(f"Completed Training Cycle {cycle}/{args.total_cycles}")
        
        logger.info("=== Unified Symbol Discovery Training Completed ===")
        
    except Exception as e:
        logger.error(f"Error during unified training: {e}", exc_info=True)
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