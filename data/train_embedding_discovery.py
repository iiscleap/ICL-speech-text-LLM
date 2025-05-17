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

class EpochEmbeddingUpdater:
    """Updates all token embeddings for each label"""
    def __init__(self, model, lr=2e-4):
        self.model = model
        self.embed_module = model.get_embedding_module()
        self.label_token_ids = model.label_token_ids
        self.label_to_token_ids = model.label_to_token_ids  # Use the new mapping
        self.lr = lr
        
        # Track by original label instead of individual tokens
        self.label_batch_losses = {label: [] for label in self.label_to_token_ids.keys()}
        self.label_batch_counts = {label: 0 for label in self.label_to_token_ids.keys()}
        self.label_occurrence_mask = {label: 0.0 for label in self.label_to_token_ids.keys()}
        
        # Store original embeddings
        with torch.no_grad():
            self.original_embeds = {}
            for label, token_ids in self.label_to_token_ids.items():
                self.original_embeds[label] = self.embed_module.weight[token_ids].clone()
    
    def record_batch(self, input_ids, loss):
        """Record which labels appear in this batch"""
        for label, token_ids in self.label_to_token_ids.items():
            for token_id in token_ids:
                if (input_ids == token_id).any():
                    self.label_batch_losses[label].append(loss.item())
                    self.label_batch_counts[label] += 1
                    self.label_occurrence_mask[label] = 1.0
                    break  # Once we find any token from this label, we can stop

    def get_embedding_changes(self):
        """Calculate how much embeddings have changed from original"""
        with torch.no_grad():
            avg_changes = []
            max_changes = []
            
            # Calculate changes for each label separately
            for label, token_ids in self.label_to_token_ids.items():
                if label in self.original_embeds:
                    current = self.embed_module.weight[token_ids]
                    original = self.original_embeds[label]
                    
                    # Calculate absolute differences
                    diff = torch.abs(current - original)
                    
                    # Track metrics for this label
                    avg_change = torch.mean(diff).item()
                    max_change = torch.max(diff).item()
                    
                    avg_changes.append(avg_change)
                    max_changes.append(max_change)
            
            # Average across all labels
            if avg_changes:
                return sum(avg_changes) / len(avg_changes), max(max_changes)
            else:
                return 0.0, 0.0
    
    def update_embeddings_at_epoch_end(self):
        """Update all token embeddings for each label that appeared"""
        with torch.no_grad():
            # Track which labels actually appeared
            labels_with_data = sum(1 for label in self.label_to_token_ids.keys() 
                                if self.label_batch_counts[label] > 0)
            
            if labels_with_data == 0:
                logging.warning("No batches contained target labels! Embeddings not updated.")
                return False
                
            logging.info(f"{labels_with_data}/{len(self.label_to_token_ids)} labels appeared in this epoch")
            
            # Update each label's token embeddings
            for label, token_ids in self.label_to_token_ids.items():
                if self.label_occurrence_mask[label] > 0:
                    # Get current embeddings for all tokens of this label
                    current_embeds = self.embed_module.weight[token_ids].clone()
                    
                    # Create perturbation (same for all tokens of this label)
                    perturbation = torch.randn_like(current_embeds) * self.lr
                    
                    # Apply update to all tokens for this label
                    self.embed_module.weight[token_ids] = current_embeds + perturbation
                    
                    # Log stats
                    count = self.label_batch_counts[label]
                    if count > 0:
                        avg_loss = sum(self.label_batch_losses[label]) / count
                        token_texts = [self.model.salmonn.llama_tokenizer.decode([tid]) for tid in token_ids]
                        logging.info(f"Label '{label}' ({token_texts}) appeared in {count} batches, avg loss: {avg_loss:.4f}")
            
            # Reset counters for next epoch
            self.label_batch_losses = {label: [] for label in self.label_to_token_ids.keys()}
            self.label_batch_counts = {label: 0 for label in self.label_to_token_ids.keys()}
            self.label_occurrence_mask = {label: 0.0 for label in self.label_to_token_ids.keys()}
            
            return True

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
    # Initialize our custom embedding updater
    updater = EpochEmbeddingUpdater(model, lr=args.lr)
    
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        model.eval()  # Use eval mode since we're not using gradients
        total_loss = 0
        total_samples = 0
        
        # Create progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        # Process all batches to collect information
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass only (no backward)
                with torch.no_grad():
                    outputs = model(batch)
                    loss = outputs["loss"]
                
                # Record batch info for later update
                if "input_ids" in batch:
                    updater.record_batch(batch["input_ids"], loss)
                
                # Track metrics
                batch_size = batch["input_ids"].size(0) if "input_ids" in batch else 1
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Update progress bar
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss/total_samples:.4f}")
                
                # Clear CUDA cache periodically
                if batch_idx % 30 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                torch.cuda.empty_cache()
                continue
        
        # Apply a single update at the end of the epoch
        logger.info(f"Applying token embedding updates at the end of epoch {epoch+1}")
        update_made = updater.update_embeddings_at_epoch_end()
        
        if update_made:
            # Log embedding change metrics
            avg_change, max_change = updater.get_embedding_changes()
            logger.info(f"Average embedding change: {avg_change:.6f}, Maximum change: {max_change:.6f}")
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} complete - Average loss: {avg_loss:.4f}")
        


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

        # Replace the select approach with a custom sampler
        total_samples = len(train_dataset)
        logger.info(f"Original dataset size: {total_samples} samples")

        # Create a custom sampler for the DataLoader
        if total_samples > 100:
            # Generate 100 random indices
            random_indices = torch.randperm(total_samples)[:10].tolist()
            
            # Create a SubsetRandomSampler with these indices
            from torch.utils.data import SubsetRandomSampler
            sampler = SubsetRandomSampler(random_indices)
            
            logger.info(f"Sampled 100 random examples from dataset")
            
            # Create dataloader with sampler instead of shuffle
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,  # Use sampler instead of shuffle
                collate_fn=processor.collate_batch
            )
        else:
            logger.info(f"Dataset has {total_samples} samples, using all of them")
            
            # Create standard dataloader with shuffle
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
            exclude_label_tokens=False, 
            top_k=20,  # Find top 20 nearest tokens
            min_similarity=0.5  # Only include tokens with similarity > 0.5
        )

        # Save final results to file
        results_path = os.path.join(args.output_dir, "nearest_tokens_final.json")

        # Convert tensor elements to Python types for JSON serialization
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

        # Save to JSON file
        with open(results_path, 'w') as f:
            json.dump(clean_results, f, indent=2)

        logger.info(f"Saved final nearest token results to {results_path}")


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

