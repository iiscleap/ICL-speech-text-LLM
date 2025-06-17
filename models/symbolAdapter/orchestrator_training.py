#!/usr/bin/env python3
"""
Orchestrator Training Script
Uses the new Symbol Training Orchestrator with TrainingConfig
"""

import os
import sys
import logging
import torch
from datetime import datetime
from typing import List

# Add parent directory to path for imports
ICL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ICL_ROOT)

# Import the orchestrator and configurations
from models.symbolAdapter.training.symbol_training import SymbolTrainingOrchestrator
from models.symbolAdapter.configs.training_configs import TrainingConfig, parse_training_args, TrainingMode
from models.mlp_salmonn import MLPSalmonn
from models.symbolAdapter.symbol_manager import SymbolManager

# Import data utilities
from utils.data_utils import load_dataset
from data.dataset_factory import DatasetFactory
from data.master_config import DatasetType, get_dataset_config
from data.model_processors import get_processor
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer



def setup_tokenizer():
    """Setup tokenizer"""
    llama_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1", use_fast=False)
    llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llama_tokenizer.padding_side = "right"
    return llama_tokenizer


def load_datasets_for_config(config: TrainingConfig, inference_mode: bool = False):
    """Load datasets based on configuration
    
    Args:
        config: Training configuration
        inference_mode: If True, load test split only
    """
    dataset_type_str = config.data_config.dataset_type
    dataset_names = dataset_type_str.split('-') if '-' in dataset_type_str else [dataset_type_str]
    
    train_datasets = {}
    val_datasets = {}  # Will contain test datasets when inference_mode=True
    
    for dataset_name in dataset_names:
        try:
            dataset_type = DatasetType(dataset_name)
            
            if inference_mode:
                # ✅ INFERENCE MODE: Load only test split
                logging.info(f"🔍 Loading test split for {dataset_name}")
                full_test_dataset = load_dataset(dataset_type, split="test")
                
                if config.data_config.max_samples > 0:
                    test_samples = min(config.data_config.max_samples, len(full_test_dataset))
                    val_datasets[dataset_type] = full_test_dataset.select(range(test_samples))
                else:
                    val_datasets[dataset_type] = full_test_dataset
                
                # Empty train_datasets for inference
                train_datasets[dataset_type] = None
                
                logging.info(f"✓ Loaded {dataset_name} TEST: {len(val_datasets[dataset_type])} samples")
                
            else:
                # ✅ TRAINING MODE: Load train and validation splits
                logging.info(f"📚 Loading train/val splits for {dataset_name}")
                full_train_dataset = load_dataset(dataset_type, split="train")
                full_val_dataset = load_dataset(dataset_type, split="validation")
                
                if config.data_config.max_samples > 0:
                    train_datasets[dataset_type] = full_train_dataset.select(range(config.data_config.max_samples))
                    val_samples = min(config.data_config.val_max_samples, len(full_val_dataset))
                    val_datasets[dataset_type] = full_val_dataset.select(range(val_samples))
                else:
                    train_datasets[dataset_type] = full_train_dataset
                    val_datasets[dataset_type] = full_val_dataset
                
                logging.info(f"✓ Loaded {dataset_name}: {len(train_datasets[dataset_type])} train, {len(val_datasets[dataset_type])} val samples")
            
        except Exception as e:
            logging.error(f"✗ Failed to load dataset {dataset_name}: {e}")
            continue
    
    return train_datasets, val_datasets


def create_combined_dataloader(datasets, processor, config: TrainingConfig, shuffle=False):
    """Create combined dataloader from datasets"""
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
        model_type=config.model_type.value,
        run_name=config.run_name,
        randomize_swap=False,
        balance_datasets=False,
        interleave=False
    )
    
    batch_size = config.data_config.batch_size if not shuffle else config.data_config.val_batch_size
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=processor.collate_batch,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def extract_dataset_labels(config: TrainingConfig) -> List[str]:
    """Extract dataset labels once - centralized function"""
    dataset_type_str = config.data_config.dataset_type
    dataset_names = dataset_type_str.split('-') if '-' in dataset_type_str else [dataset_type_str]
    
    all_valid_labels = set()
    for dataset_name in dataset_names:
        try:
            dataset_type = DatasetType(dataset_name)
            dataset_config = get_dataset_config(dataset_type)
            all_valid_labels.update(dataset_config.valid_labels)
        except Exception as e:
            logging.warning(f"Could not get labels for {dataset_name}: {e}")
            
    
    dataset_labels = sorted(list(all_valid_labels))
    logging.info(f"Extracted dataset labels: {dataset_labels}")
    return dataset_labels


def initialize_model(config: TrainingConfig, tokenizer, symbol_manager) -> MLPSalmonn:
    """Initialize the MLPSalmonn model using SymbolManager"""
    
    # Get initial symbol mappings from SymbolManager
    initial_symbol_mappings = symbol_manager.get_symbols_for_epoch(0)  # Get epoch 0 symbols
    
    logging.info(f"Initial symbol mappings from SymbolManager: {initial_symbol_mappings}")
    
    bypass_mlp = config.mode in [TrainingMode.BYPASS_MLP_SYM, TrainingMode.BYPASS_MLP_ORG]
    
    logging.info(f"Training mode: {config.mode.value}, bypass_mlp: {bypass_mlp}")
    
    # Initialize model
    model = MLPSalmonn(
        device=config.device,
        label_tokens=list(initial_symbol_mappings.values()),
        hidden_dim=config.mlp_config.hidden_dim,
        dropout=config.mlp_config.dropout,
        lora=True,
        lora_rank=config.lora_config.rank,
        lora_alpha=config.lora_config.alpha,
        lora_dropout=config.lora_config.dropout,
        low_resource=False,
        use_output_mlp=config.mlp_config.use_output_mlp,
        bypass_mlp=bypass_mlp  # NEW: Pass bypass_mlp parameter
    )
    

    model.update_label_tokens(initial_symbol_mappings)
    
    return model


def setup_logging() -> str:
    """Setup logging with run_name based paths"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    return None 


def main():
    try:
        # Parse arguments and create configuration
        log_file = setup_logging()
        logging.info("🚀 Starting orchestrator training...")
        
        # Parse arguments
        logging.info("📋 Parsing arguments...")
        args = parse_training_args()

        config = TrainingConfig.from_args(args)
        logging.info(f"Training configuration: {config.to_dict()}")

        # Setup tokenizer
        tokenizer = setup_tokenizer()
        logging.info("✓ Tokenizer initialized")
        
        # Extract dataset labels ONCE
        dataset_labels = extract_dataset_labels(config)
        logging.info(f"✓ Dataset labels extracted: {dataset_labels}")
        
        # Create SymbolManager ONCE
        
        symbol_manager = SymbolManager(
            original_labels=dataset_labels,
            tokenizer=tokenizer,
            dynamic_per_epoch=(config.symbol_config.mode.value == "dynamic_per_epoch"),
            symbol_type=config.symbol_config.symbol_type
        )
        logging.info("✓ SymbolManager initialized")
        
        # Load datasets
        train_datasets, val_datasets = load_datasets_for_config(config)
        logging.info(f"✓ Loaded datasets: {list(train_datasets.keys())}")
        
        # Create processor and dataloaders
        processor = get_processor(config.model_type.value, tokenizer=tokenizer)
        
        train_dataloader = create_combined_dataloader(train_datasets, processor, config, shuffle=True)
        val_dataloader = create_combined_dataloader(val_datasets, processor, config, shuffle=False)
        
        logging.info(f"✓ Created dataloaders: {len(train_dataloader)} train, {len(val_dataloader)} val batches")
        
        # Initialize model using SymbolManager
        model = initialize_model(config, tokenizer, symbol_manager)
        logging.info("✓ Model initialized")
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Run training using the orchestrator - PASS THE EXISTING SYMBOL_MANAGER
        logging.info("🚀 Starting Symbol Training Orchestrator...")
        
        try:
            # Create orchestrator with existing symbol_manager
            orchestrator = SymbolTrainingOrchestrator(
                config=config,
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                tokenizer=tokenizer,
                symbol_manager=symbol_manager  # PASS EXISTING SYMBOL_MANAGER
            )
            
            results = orchestrator.run_complete_training()
            
            logging.info("✅ Training completed successfully!")
            logging.info(f"Best overall score: {results.get('best_overall_score', 0):.4f}")
            logging.info(f"Training time: {results.get('training_time', 0):.1f} seconds")
            
            return results
            
        except KeyboardInterrupt:
            logging.info("❌ Training interrupted by user")
            return None
            
        except Exception as e:
            logging.error(f"❌ Training failed: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logging.error(f"❌ Training failed: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()