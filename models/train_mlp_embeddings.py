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

from models.mlp_embedding_model import MLPEmbeddingModel
from models.custom_salmon import CustomSALMONN
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
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension for MLP (smaller = fewer parameters)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for MLP")
    
    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, default="voxceleb_greek-hvb_greek", 
                      help="Dataset type(s) to use, hyphen-separated for multi-dataset")
    parser.add_argument("--max_samples", type=int, default=10, 
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
    """Train MLP embedding transformations using backpropagation"""
    # Set up optimizer for MLP parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    # Training loop
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        model.train()
        total_loss = 0
        total_samples = 0
        

        optimizer.zero_grad()
        # Process batches
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                
                # Track metrics
                batch_size = batch["input_ids"].size(0) if "input_ids" in batch else 1
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Update progress bar
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss/total_samples:.4f}")
                
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                torch.cuda.empty_cache()
                continue

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), 
            max_norm=1.0
                )
        optimizer.step()   

        # Calculate average loss for the epoch
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} complete - Average loss: {avg_loss:.4f}")
        

        # Find and save nearest tokens
        results = model.find_nearest_token_embeddings(
            exclude_label_tokens=False,
            top_k=20,
            min_similarity=0.5
        )
        
        # Save results to JSON
        results_path = os.path.join(args.output_dir, f"nearest_tokens_epoch_{epoch+1}.json")
        save_results_to_json(results, results_path)
    
    return model

def save_results_to_json(results, path):
    """Save token results to JSON file"""
    clean_results = []
    for result in results:
        clean_result = {
            "original_label": result["original_label"],
            "token_results": [],
            "combined_neighbors": []
        }
        
        # Process token results
        for token_result in result["token_results"]:
            clean_token_result = {
                "token_id": int(token_result["token_id"]),
                "token_text": token_result["token_text"],
                "neighbors": []
            }
            
            for neighbor in token_result["neighbors"]:
                clean_token_result["neighbors"].append({
                    "token_id": int(neighbor["token_id"]),
                    "token_text": neighbor["token_text"],
                    "similarity": float(neighbor["similarity"])
                })
                
            clean_result["token_results"].append(clean_token_result)
        
        # Process combined neighbors
        for neighbor in result["combined_neighbors"]:
            clean_result["combined_neighbors"].append({
                "token_id": int(neighbor["token_id"]),
                "token_text": neighbor["token_text"],
                "similarity": float(neighbor["similarity"])
            })
            
        clean_results.append(clean_result)

    with open(path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    logger.info(f"Saved nearest token results to {path}")

def main():
    # Parse arguments and setup
    args = parse_args()
    args = setup_logging(args)
    
    start_time = datetime.now()
    logger.info(f"Starting MLP embedding training at {start_time}")
    
    try:
        # Create base model
        logger.info(f"Creating base model of type {args.model_type}")
        base_model = CustomSALMONN(device=args.device, low_resource=True)
        
        # Parse dataset types
        dataset_types = [DatasetType(dt.strip()) for dt in args.dataset_type.split("-")]
        
        # Get label tokens
        label_tokens = get_label_tokens_from_dataset_types(dataset_types)
        
        # Create MLP embedding model
        mlp_model = MLPEmbeddingModel.from_custom_salmon(
            base_model,
            label_tokens=label_tokens,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
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
            num_examples=0,
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
#     --peft_model_path /data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_greek-hvb_greek/checkpoints/epoch_10_loss_0.0055/model.pt \
#     --dataset_type voxceleb_greek