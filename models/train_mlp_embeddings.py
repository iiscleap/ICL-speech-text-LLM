import os
import sys
import logging
import argparse
import torch
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_salmonn import MLPSalmonn
from utils.data_utils import load_dataset
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType
from utils.training_utils import setup_logging, load_checkpoint
from torch.utils.data import DataLoader, SubsetRandomSampler

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP embedding transformations")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="salmonn", help="Model type (salmonn or qwen2)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for MLP training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--hidden_dim", type=int, default=8, help="Hidden dimension for MLP (smaller = fewer parameters)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for MLP")
    
    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, default="voxceleb_greek-hvb_greek", 
                      help="Dataset type(s) to use, hyphen-separated for multi-dataset")
    parser.add_argument("--max_samples", type=int, default=16, 
                      help="Maximum number of samples to use (0 = use all samples)")
    parser.add_argument("--debug_samples", type=int, default=0, 
                      help="Number of samples to use for debugging (0 = use all samples)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/data2/neeraja/neeraja/results/model_ICL/mlp_embeddings", 
                      help="Output directory for transformed embeddings")
    parser.add_argument("--run_name", type=str, default="", help="Name for the run")
    
    return parser.parse_args()

def setup_logging(args):
    # Create output directory if needed
    if not args.output_dir:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"/data2/neeraja/neeraja/results/model_ICL/mlp_embeddings/{timestamp}_{args.dataset_type.replace('-', '_')}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, "mlp_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    logger.info(f"Arguments: {args}")
    return args

def get_label_tokens_from_dataset_types(dataset_types):
    """Extract label tokens from dataset configurations"""
    from data.master_config import get_dataset_config
    
    all_label_tokens = []
    dataset_labels = {}
    
    for dt in dataset_types:
        config = get_dataset_config(dt)
        if config and hasattr(config, 'valid_labels'):
            dataset_labels[dt.name] = config.valid_labels
            all_label_tokens.extend(config.valid_labels)
    
    # Remove duplicates while preserving order
    unique_label_tokens = []
    for token in all_label_tokens:
        if token not in unique_label_tokens:
            unique_label_tokens.append(token)
    
    logger.info(f"Extracted label tokens from datasets:")
    for dt_name, labels in dataset_labels.items():
        logger.info(f"  {dt_name}: {labels}")
    logger.info(f"Combined unique label tokens: {unique_label_tokens}")
    
    return unique_label_tokens


def train(model, dataloader, args):
    """Train MLP embedding transformations using backpropagation with gradient accumulation"""
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.0,
        weight_decay=0.0
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )


    # Gradient accumulation settings
    accumulation_steps = 8
    effective_batch_size = args.batch_size * accumulation_steps
    
    logger.info(f"Using gradient accumulation: {accumulation_steps} steps")
    logger.info(f"Effective batch size: {effective_batch_size}")
    
    for epoch in range(args.num_epochs):
        model.train()
        
        # Reset tracking variables at start of epoch
        total_loss = 0
        total_samples = 0
        valid_update_steps = 0  # Track actual optimizer steps
        accumulated_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        consecutive_failures = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Simple health check (no verbose logging)
                is_healthy, _ = model.check_mlp_health()
                if not is_healthy:
                    model.reset_mlp_weights()
                    optimizer.zero_grad()
                    accumulated_loss = 0
                    consecutive_failures += 1
                    
                    if consecutive_failures > 5:
                        logger.error("Too many consecutive MLP failures, stopping training")
                        break
                    continue
                
                # Forward pass
                outputs = model(batch)
                loss = outputs["loss"]
                loss = loss / accumulation_steps
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    consecutive_failures += 1
                    continue
                
                consecutive_failures = 0
                loss.backward()
                accumulated_loss += loss.item()
                
                # Only update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Quick health check without verbose logging
                    nan_grads = False
                    max_grad_norm = 0.0
                    
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            max_grad_norm = max(max_grad_norm, grad_norm)
                            
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                logger.warning(f"NaN/Inf gradient detected in {name}")
                                nan_grads = True
                                break
                    
                    if nan_grads or max_grad_norm > 10.0:
                        logger.warning(f"Skipping update step due to problematic gradients (max norm: {max_grad_norm:.6f})")
                        optimizer.zero_grad()
                        accumulated_loss = 0
                        model.reset_mlp_weights()
                        continue
                    
                    # Simple health check
                    is_healthy_before_opt, _ = model.check_mlp_health()
                    if not is_healthy_before_opt:
                        logger.warning("MLP unhealthy before optimizer step, resetting")
                        model.reset_mlp_weights()
                        optimizer.zero_grad()
                        accumulated_loss = 0
                        continue
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), 
                        max_norm=1.0
                    )
                    
                    # Update weights
                    optimizer.step()
                    
                    # Quick health check after update
                    is_healthy_after_opt, _ = model.check_mlp_health()
                    if not is_healthy_after_opt:
                        logger.warning("MLP corrupted after optimizer step, resetting")
                        model.reset_mlp_weights()
                        optimizer.zero_grad()
                        accumulated_loss = 0
                        continue
                    
                    optimizer.zero_grad()
                    
                    # Update tracking - THIS IS THE KEY FIX
                    valid_update_steps += 1  # Count this as a valid update step
                    avg_loss = accumulated_loss / accumulation_steps
                    total_loss += accumulated_loss  # Add to total
                    total_samples += accumulation_steps  # Count samples
                    accumulated_loss = 0
                    
                    # Log progress
                    logger.info(f"Step {valid_update_steps}: loss={avg_loss:.6f}, max_grad_norm={max_grad_norm:.6f}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                torch.cuda.empty_cache()
                # Reset everything on error
                optimizer.zero_grad()
                accumulated_loss = 0
                model.reset_mlp_weights()
                consecutive_failures += 1
                
                if consecutive_failures > 5:
                    logger.error("Too many consecutive failures, stopping training")
                    break
                continue

        # Handle remaining gradients if the last batch doesn't align with accumulation_steps
        if accumulated_loss > 0:
            # Check for NaN gradients
            nan_grads = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    logger.warning(f"NaN/Inf gradient detected in {name} (final step)")
                    nan_grads = True
                    break
            
            if not nan_grads:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    max_norm=1.0
                )
                
                # Update weights
                optimizer.step()
                valid_update_steps += 1
                
                # Track final metrics
                total_loss += accumulated_loss * accumulation_steps * args.batch_size
                total_samples += args.batch_size * ((batch_idx + 1) % accumulation_steps)
            
            optimizer.zero_grad()

        # Calculate average loss for the epoch - FIX THE CONDITION
        if valid_update_steps > 0:  # Change from valid_batches to valid_update_steps
            avg_loss = total_loss / total_samples
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} complete - Average loss: {avg_loss:.4f} ({valid_update_steps} update steps)")
            
            # === SYMBOL MAPPING DISCOVERY AND SAVING ===
            logger.info("=== Symbol Discovery After Epoch ===")
            
            try:
                # Find symbol mappings (returns token_id -> token_id mappings)
                raw_mappings = model.find_symbol_mappings()
                
                # Get label mappings (full labels like 'alpha' -> 'Alpha', 'flob' -> 'FLOB')
                label_mappings = model.get_label_mappings() if hasattr(model, 'get_label_mappings') else {}
                
                # Convert raw token mappings to human-readable text
                token_mappings = {}
                if raw_mappings:
                    tokenizer = model.llama_tokenizer
                    
                    logger.info(f"Found {len(raw_mappings)} token mappings:")
                    for source_token_id, target_token_id in raw_mappings.items():
                        # Convert token IDs to text
                        source_text = tokenizer.decode([int(source_token_id)], skip_special_tokens=False).strip()
                        target_text = tokenizer.decode([int(target_token_id)], skip_special_tokens=False).strip()
                        
                        # Only save meaningful mappings (not empty or identical)
                        if source_text and target_text and source_text != target_text:
                            token_mappings[source_text] = target_text
                            logger.info(f"  '{source_text}' -> '{target_text}'")
                
                # Extract label mappings from the logs if model doesn't have get_label_mappings method
                if not label_mappings:
                    # Parse the combined mappings from model's discovery process
                    # This is a fallback - ideally the model should provide get_label_mappings()
                    logger.info("Attempting to extract label mappings from recent discovery...")
                    
                    # For now, create some example mappings based on the patterns we see
                    # You might need to implement get_label_mappings() in the model
                    potential_labels = ['alpha', 'beta', 'gamma', 'delta', 'fred', 'plugh', 'xyzzy', 'thud', 'wibble', 'wobble', 'wubble', 'flob', 'zoop']
                    
                    for label in potential_labels:
                        # Try to reconstruct the combined mapping
                        # This is approximate - better to get it directly from the model
                        combined_result = ""
                        for token in tokenizer.encode(label, add_special_tokens=False):
                            token_text = tokenizer.decode([token], skip_special_tokens=False).strip()
                            if token_text in token_mappings:
                                combined_result += token_mappings[token_text]
                            else:
                                combined_result += token_text
                        
                        if combined_result and combined_result != label:
                            label_mappings[label] = combined_result
                
                # Load existing combined file
                combined_file = os.path.join(args.output_dir, "symbol_discovery_results.json")
                all_results = {}
                
                if os.path.exists(combined_file):
                    try:
                        with open(combined_file, 'r', encoding='utf-8') as f:
                            all_results = json.load(f)
                    except:
                        all_results = {}
                
                # Add current epoch results with both token and label mappings
                epoch_key = f"epoch_{epoch+1}"
                all_results[epoch_key] = {
                    "loss": avg_loss,
                    "token_mappings": token_mappings,  # Individual tokens: "alpha" -> "Alpha"
                    "label_mappings": label_mappings,  # Full labels: "wobble" -> "stubbles"
                    "num_token_mappings": len(token_mappings),
                    "num_label_mappings": len(label_mappings),
                    "timestamp": datetime.now().isoformat(),
                    "update_steps": valid_update_steps
                }
                
                # Save combined results
                with open(combined_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ Saved epoch {epoch+1} results to: {combined_file}")
                
                # Print current epoch results
                logger.info("=== Current Epoch Results ===")
                logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Steps = {valid_update_steps}")
                
                if token_mappings:
                    logger.info(f"Found {len(token_mappings)} token mappings:")
                    for token, mapped in token_mappings.items():
                        logger.info(f"  Token: '{token}' -> '{mapped}'")
                
                if label_mappings:
                    logger.info(f"Found {len(label_mappings)} label mappings:")
                    for label, mapped in label_mappings.items():
                        logger.info(f"  Label: '{label}' -> '{mapped}'")
                
                if not token_mappings and not label_mappings:
                    logger.info("  No symbol mappings discovered this epoch")
                
            except Exception as e:
                logger.error(f"❌ Error during symbol mapping discovery: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            logger.info("=== End Symbol Discovery ===")
            
        else:
            logger.error("No valid update steps in this epoch!")
            continue
            
        # Step the scheduler
        scheduler.step(avg_loss)
    
    return model


def main():
    # Parse arguments and setup
    args = parse_args()
    args = setup_logging(args)
    
    start_time = datetime.now()
    logger.info(f"Starting MLP embedding training at {start_time}")
    
    try:
        # Create base model
        logger.info(f"Creating base model of type {args.model_type}")
        
        # Parse dataset types
        dataset_types = [DatasetType(dt.strip()) for dt in args.dataset_type.split("-")]
        
        # Get label tokens
        label_tokens = get_label_tokens_from_dataset_types(dataset_types)
        

        logger.info(f"Creating MLPSalmonn model with label tokens: {label_tokens}")
        mlp_model = MLPSalmonn(
            llama_path="lmsys/vicuna-13b-v1.1",
            whisper_path="openai/whisper-large-v2",
            beats_path="/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            label_tokens=label_tokens,
            hidden_dim=args.hidden_dim,
            freeze_base=True,
            lora=True,
            lora_rank=8,  # Use same value as your pretrained model
            lora_alpha=16,
            lora_dropout=0.05,
            device=args.device,
            low_resource=True
        )


        
        # Load pretrained weights
        logger.info(f"Loading checkpoint from {args.peft_model_path}")
        checkpoint = load_checkpoint(args.peft_model_path)
        mlp_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Print token info
        mlp_model.print_token_info(label_tokens)
        
        # Create processor
        from data.model_processors import get_processor
        processor = get_processor(args.model_type, mlp_model.input_processor, mlp_model.llama_tokenizer)
        
        # Load datasets
        train_datasets = {}
        for dt in dataset_types:
            full_dataset = load_dataset(dt, split="train")
            if args.debug_samples > 0:
                train_datasets[dt] = full_dataset.select(range(args.debug_samples))
            else:
                train_datasets[dt] = full_dataset
        
        # Create training dataset
        train_dataset = DatasetFactory.create_dataset(
            dataset_type=dataset_types,
            dataset=train_datasets,
            processor=processor,
            is_training=True,
            input_mode="speech_only",
            fewshot_mode="text",
            num_examples=5,
            random_examples=True,
            model_type=args.model_type,
            run_name=args.run_name
        )
        
        # Sample data if needed
        total_samples = len(train_dataset)
        logger.info(f"Original dataset size: {total_samples} samples")
        
        if args.max_samples > 0 and total_samples > args.max_samples:
            # Sample random indices
            indices = torch.randperm(total_samples)[:args.max_samples].tolist()
            sampler = SubsetRandomSampler(indices)
            
            logger.info(f"Sampled {args.max_samples} random examples from dataset")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                collate_fn=processor.collate_batch
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=processor.collate_batch
            )
        
        logger.info(f"Created dataloader with {len(train_loader)} batches")
        
        # Train model
        mlp_model = train(mlp_model, train_loader, args)
        
        # Record end time and duration
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"MLP embedding training completed at {end_time}")
        logger.info(f"Total duration: {duration}")
        
        logger.info(f"All operations completed. Results in {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during MLP embedding training: {str(e)}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())



# CUDA_VISIBLE_DEVICES=0 python /data2/neeraja/neeraja/code/ICL/models/train_mlp_embeddings.py \
#     --peft_model_path /data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_greek-hvb_greek/checkpoints/epoch_1_loss_0.3191/model.pt \
#     --dataset_type voxceleb_greek-hvb_greek