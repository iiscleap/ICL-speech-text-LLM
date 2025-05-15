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

from models.embedding_discovery_model import EmbeddingDiscoveryModel
from models.custom_salmon import CustomSALMONN
from models.model_factory import ModelFactory
from data.model_processors import get_processor
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType
from utils.data_utils import load_dataset
from utils.training_utils import setup_logging, load_checkpoint
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train embedding discovery model")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="salmonn", help="Model type (salmonn or qwen2)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to pretrained model with nonsense labels")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for embedding training")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train")
    
    # Dataset parameters
    parser.add_argument("--dataset_type", type=str, default="voxceleb", 
                      help="Dataset type(s) to use, hyphen-separated for multi-dataset (e.g., voxceleb-hvb)")
    parser.add_argument("--input_mode", type=str, default="speech_only", 
                      choices=["speech_only", "text_only", "speech_and_text"],
                      help="Input mode for the model")
    parser.add_argument("--debug_samples", type=int, default=0, 
                      help="Number of samples to use for debugging (0 = use all samples)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="/data2/neeraja/neeraja/results/model_ICL/", help="Output directory for discovered embeddings")
    parser.add_argument("--run_name", type=str, default="", help="Name for the run")
    
    # Target token parameters
    
    return parser.parse_args()

def setup_logging(args):
    # Create output directory if needed
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"/data2/neeraja/neeraja/results/model_ICL/embedding_discovery/{timestamp}_{args.dataset_type.replace('-', '_')}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging to file and console
    log_file = os.path.join(args.output_dir, "embedding_discovery.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Arguments: {args}")
    return args

def create_model(args):
    logger.info(f"Creating base model of type {args.model_type}")
    
    # Create the base model
    model = CustomSALMONN(
        device=args.device, 
        low_resource=True
    )
    
    # Load checkpoint
    # logger.info(f"Loading checkpoint from {args.peft_model_path}")
    # checkpoint = load_checkpoint(args.peft_model_path)

    #  # Load state dict from the custom salmon model
    # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    return model

def train(model, dataloader, args):
    """
    Train the model with gradient accumulation over an entire epoch.
    Only update weights once per epoch.
    """
    # Set up optimizer - only embedding parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    # Training loop
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        model.train()
        total_loss = 0
        
        # Zero gradients at the start of the epoch
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch)
                loss = outputs["loss"]
                
                # Accumulate gradients (backward pass)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    max_grad_norm
                )
                
                # Log progress (don't update weights yet)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
                    
                # Clear CUDA cache periodically
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue
        
        # Log epoch results
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Avg Loss: {avg_loss:.4f}")
        
        # Apply accumulated gradients and update weights once per epoch
        logger.info(f"Applying accumulated gradients from entire epoch")
        optimizer.step()
        optimizer.zero_grad()
        
        # Save intermediate embeddings
        if args.output_dir:
            embed_path = os.path.join(args.output_dir, f"token_embeddings_epoch_{epoch+1}.pt")
            # model.save_token_embeddings(embed_path)
            
            # Find nearest tokens after each epoch
            results = model.find_nearest_token_embeddings(
                exclude_label_tokens=False,
                top_k=args.top_k,
                min_similarity=0.5
            )
            
            # Log results
            logger.info(f"\nNearest tokens after epoch {epoch+1}:")
            for result in results:
                original = result.get("original_label", "unknown")
                logger.info(f"\n{original} → Nearest tokens:")
                for idx, neighbor in enumerate(result["neighbors"][:5]):
                    logger.info(f"  {idx+1}. '{neighbor['token_text']}' (similarity: {neighbor['similarity']:.4f})")
            
            # Save results
            epoch_suffix = f"_epoch_{epoch+1}"
            results_path = os.path.join(args.output_dir, f"nearest_tokens{epoch_suffix}.json")
            
            # Convert tensor items to regular Python types for JSON serialization
            clean_results = []
            for result in results:
                clean_result = {
                    "original_label": result.get("original_label", "unknown"),
                    "target_token": result["target_token"],
                    "target_id": int(result["target_id"]),
                    "neighbors": []
                }
                for neighbor in result["neighbors"]:
                    clean_result["neighbors"].append({
                        "token_id": int(neighbor["token_id"]),
                        "token_text": neighbor["token_text"],
                        "similarity": float(neighbor["similarity"])
                    })
                clean_results.append(clean_result)
                    
            with open(results_path, 'w') as f:
                json.dump(clean_results, f, indent=2)
    
    logger.info("Training complete")
    return model

def get_label_tokens_from_dataset_types(dataset_types):
    """Extract all label tokens from the dataset configurations"""
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

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging and directories
    args = setup_logging(args)
    
    # Record start time
    start_time = datetime.now()
    logger.info(f"Starting embedding discovery at {start_time}")
    
    try:
        # Create model
        model = create_model(args)

        from data.model_processors import get_processor
        from utils.data_utils import load_dataset
        from data.dataset_factory import DatasetFactory
        from data.master_config import get_dataset_config

        # Parse dataset types
        dataset_types = [DatasetType(dt.strip()) for dt in args.dataset_type.split("-")]


        # Create model with specific label tokens to update
        label_tokens = get_label_tokens_from_dataset_types(dataset_types)
        discovery_model = EmbeddingDiscoveryModel.from_custom_salmon(model, label_tokens=label_tokens)
        # discovery_model = EmbeddingDiscoveryModel(label_tokens=label_tokens)

        logger.info(f"Loading checkpoint from {args.peft_model_path}")
        checkpoint = load_checkpoint(args.peft_model_path)

        # Load state dict from the custom salmon model
        discovery_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # Print token info before training
        discovery_model.print_token_info(label_tokens)


        
        # Create processor
        processor = get_processor(args.model_type, discovery_model.input_processor, discovery_model.llama_tokenizer)
        
        # Get dataset and dataloader
        
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
            input_mode="speech_only",  # Use speech-only mode for training
            fewshot_mode="text",
            num_examples=0,
            random_examples=True,
            model_type=args.model_type,
            run_name=args.run_name
        )
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=processor.collate_batch
        )
        
        logger.info(f"Created dataloader with {len(train_loader)} batches")
        
       

        # Train model (only label token embeddings will be updated)
        train(discovery_model, train_loader, args)

        # After training, find nearest tokens to the updated label embeddings
        results = discovery_model.find_nearest_token_embeddings(
            exclude_label_tokens=False,  # Don't include the original labels
            top_k=20,  # Find top 20 nearest tokens
            min_similarity=0.5  # Only include tokens with similarity > 0.5
        )

        # Print results
        for result in results:
            original = result["original_label"]
            print(f"\n{original} → Nearest tokens:")
            for neighbor in result["neighbors"][:5]:  # Show top 5
                print(f"  {neighbor['token_text']} (similarity: {neighbor['similarity']:.4f})")
        
        # Record end time and duration
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Embedding discovery completed at {end_time}")
        logger.info(f"Total duration: {duration}")
        
        logger.info(f"All operations completed. Results in {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during embedding discovery: {str(e)}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())


# CUDA_VISIBLE_DEVICES=2 python /data2/neeraja/neeraja/code/ICL/data/train_embedding_discovery.py \
#     --peft_model_path /data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_greek-hvb_greek/checkpoints/epoch_10_loss_0.0055/model.pt \
#     --dataset_type voxceleb_greek-hvb_greek 

