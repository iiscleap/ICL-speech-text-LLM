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

class EpochEmbeddingUpdater:
    """Updates token embeddings directly without gradients, only at the end of each epoch"""
    def __init__(self, model, lr=2e-4):
        self.model = model
        self.embed_module = model.get_embedding_module()
        self.label_token_ids = model.label_token_ids
        self.lr = lr
        
        # Track batches with label tokens and their losses
        self.token_batch_losses = {token_id: [] for token_id in self.label_token_ids}
        self.token_batch_counts = {token_id: 0 for token_id in self.label_token_ids}
        self.token_occurrence_mask = torch.zeros(len(self.label_token_ids), dtype=torch.float32)
        
        # Store original embeddings
        with torch.no_grad():
            self.original_embeds = self.embed_module.weight[self.label_token_ids].clone()
    
    def record_batch(self, input_ids, loss):
        """Record a batch with its loss for end-of-epoch update"""
        # Check which label tokens appear in this batch
        for i, token_id in enumerate(self.label_token_ids):
            if (input_ids == token_id).any():
                self.token_batch_losses[token_id].append(loss.item())
                self.token_batch_counts[token_id] += 1
                self.token_occurrence_mask[i] = 1.0
    
    def update_embeddings_at_epoch_end(self):
        """Apply a single embedding update at the end of the epoch"""
        with torch.no_grad():
            # Get current embeddings
            current_embeds = self.embed_module.weight[self.label_token_ids].clone()
            
            # Track which tokens actually appeared during the epoch
            tokens_with_data = sum(1 for tid in self.label_token_ids if self.token_batch_counts[tid] > 0)
            
            if tokens_with_data == 0:
                logging.warning("No batches contained target tokens! Embeddings not updated.")
                return False
                
            logging.info(f"{tokens_with_data}/{len(self.label_token_ids)} token types appeared in this epoch")
            
            # Create update direction - small random perturbation weighted by token occurrence
            # IMPORTANT: Create with the same dtype as the embedding weights
            perturbation = torch.randn_like(current_embeds) * self.lr
            
            # Create occurrence mask with correct device and dtype
            occurrence_mask = self.token_occurrence_mask.to(
                device=perturbation.device, 
                dtype=perturbation.dtype  # Match perturbation dtype
            ).unsqueeze(1)
            
            # Apply update only to tokens that appeared in the epoch
            # Ensure all tensors have the same dtype
            update = current_embeds + perturbation * occurrence_mask
            
            # For quantized models, might need to dequantize/requantize
            if hasattr(self.embed_module, 'dequantize') and hasattr(self.embed_module, 'quantize'):
                # Dequantize, update, requantize
                dequantized = self.embed_module.dequantize(self.embed_module.weight)
                dequantized[self.label_token_ids] = update
                self.embed_module.weight = self.embed_module.quantize(dequantized)
            else:
                # Standard update
                self.embed_module.weight[self.label_token_ids] = update
            
            # Log token stats
            for i, token_id in enumerate(self.label_token_ids):
                count = self.token_batch_counts[token_id]
                if count > 0:
                    avg_loss = sum(self.token_batch_losses[token_id]) / count
                    logging.info(f"Token '{self.model.label_tokens[i]}' (ID: {token_id}) appeared in {count} batches, avg loss: {avg_loss:.4f}")
            
            # Reset counters for next epoch
            self.token_batch_losses = {token_id: [] for token_id in self.label_token_ids}
            self.token_batch_counts = {token_id: 0 for token_id in self.label_token_ids}
            self.token_occurrence_mask = torch.zeros(len(self.label_token_ids), dtype=torch.float32)
            
            return True
    
    def get_embedding_changes(self):
        """Calculate how much embeddings have changed from original"""
        with torch.no_grad():
            current = self.embed_module.weight[self.label_token_ids]
            diff = current - self.original_embeds
            avg_change = torch.mean(torch.abs(diff)).item()
            max_change = torch.max(torch.abs(diff)).item()
        return avg_change, max_change

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
        
        # Save embeddings after each epoch
        if args.output_dir:
            save_path = os.path.join(args.output_dir, f"embeddings_epoch_{epoch+1}.pt")
            model.save_token_embeddings(save_path)
            
        # Find and log nearest tokens
        results = model.find_nearest_token_embeddings(
            exclude_label_tokens=False,
            top_k=20,
            min_similarity=0.5
        )
        
        # Save results and continue with rest of your code...

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


# CUDA_VISIBLE_DEVICES=1 python /data2/neeraja/neeraja/code/ICL/data/train_embedding_discovery.py \
#     --peft_model_path /data2/neeraja/neeraja/results/model_ICL/trained_models/ft_5ex_20e8b_salmonn_speech_only_voxceleb_greek-hvb_greek/checkpoints/epoch_10_loss_0.0055/model.pt \
#     --dataset_type voxceleb_greek-hvb_greek

