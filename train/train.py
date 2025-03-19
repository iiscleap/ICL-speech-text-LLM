import os
import sys
import time
import logging
import argparse
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import random
import psutil
import gc
from functools import partial

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.master_config import DatasetType
from data.dataset_factory import DatasetFactory
from data.model_processors import get_processor
from models.model_factory import ModelFactory
from utils.data_utils import load_dataset
from utils.training_utils import setup_logging, save_checkpoint, load_checkpoint, log_gpu_memory_usage, get_learning_rates
from utils.evaluation_utils import evaluate_predictions
from config.training_config import get_training_config

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for ICL models")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--input_mode", type=str, default="speech_only", 
                        choices=["speech_only", "text_only", "speech_and_text"],
                        help="Input mode for the model")
    parser.add_argument("--dataset_type", type=str, required=True, 
                        help="Dataset type(s) to use, comma-separated for multi-task")
    parser.add_argument("--fewshot_mode", type=str, default="text", 
                        choices=["text", "speech"],
                        help="Mode for few-shot examples")
    parser.add_argument("--model_type", type=str, default="salmonn", 
                        choices=["salmonn", "qwen2"],
                        help="Type of model to use")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=-1, 
                        help="Local rank for distributed training")
    
    # Add training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--run_name", type=str, default="", help="Name for the run")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision training (better for newer GPUs)")
    parser.add_argument("--early_stopping_patience", type=int, default=3, 
                        help="Number of evaluations with no improvement after which training will be stopped")
    parser.add_argument("--val_split", type=str, default="test", 
                        choices=["val", "test"],
                        help="Dataset split to use for validation")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for AdamW")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for AdamW")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, 
                        help="Number of workers for data loading")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2, 
                        help="Number of batches to prefetch")
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=True,
                        help="Use pinned memory for data loading")
    parser.add_argument("--log_steps", type=int, default=10, 
                        help="Log training stats every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--find_unused_parameters", action="store_true", 
                        help="Find unused parameters in DDP (use only if getting DDP errors)")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Batch size for evaluation (defaults to training batch size)")
    parser.add_argument("--scheduler", type=str, default="linear", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler type")
    
    # Add debug argument
    parser.add_argument("--debug_samples", type=int, default=10, 
                        help="Number of samples to use for debugging (0 = use all samples)")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Make CuDNN deterministic (slightly reduces performance)
        torch.backends.cudnn.deterministic = True

def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    """Get the appropriate learning rate scheduler"""
    from transformers import get_scheduler as get_hf_scheduler
    return get_hf_scheduler(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    try:
        # Setup distributed training
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend="nccl")
            is_main_process = args.local_rank == 0
        else:
            is_main_process = True
        
        # Setup logging
        # if is_main_process:
        #     setup_logging(args.log_file)
        
        logger = logging.getLogger(__name__)
        if is_main_process:
            logger.info(f"Starting training with args: {args}")
            # Log system info
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                log_gpu_memory_usage(logger)
            
            # Log CPU info
            logger.info(f"CPU count: {os.cpu_count()}")
            logger.info(f"Memory available: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
        
        # Parse dataset types for multi-task
        try:
            dataset_types = [DatasetType(dt.strip()) for dt in args.dataset_type.split("-")]
            if not dataset_types:
                raise ValueError("No valid dataset types provided")
        except ValueError as e:
            logger.error(f"Error parsing dataset types: {e}")
            return 1
        
        # Load training config
        try:
            config = get_training_config(args.model_type)
            if is_main_process:
                logger.info(f"Loaded training config for {args.model_type}")
        except Exception as e:
            logger.error(f"Error loading training config: {e}")
            return 1
        
        # Create model
        try:
            model = ModelFactory.create_model(
                model_type=args.model_type,
                # multi_task=len(dataset_types) > 1,
                multi_task=False,
                task_configs={dt: get_training_config(args.model_type, dt) for dt in dataset_types},
                default_task=dataset_types[0] if dataset_types else None,
                **config.get("model_args", {})
            )
            if is_main_process:
                logger.info(f"Created model: {args.model_type}")
                # Log model size
                param_count = sum(p.numel() for p in model.parameters())
                trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Model parameters: {param_count:,} (trainable: {trainable_param_count:,})")
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return 1
        
        # Move model to device
        device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda")
        model = model.to(device)
        
        # Enable gradient checkpointing if requested (saves memory)
        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if is_main_process:
                logger.info("Gradient checkpointing enabled")
        
        # Resume from checkpoint if specified
        start_epoch = 0
        best_val_metric = float('inf')  # Lower is better for loss
        no_improvement_count = 0
        if args.resume_from_checkpoint:
            try:
                if is_main_process:
                    logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
                checkpoint = load_checkpoint(args.resume_from_checkpoint, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"],state_dict=False)
                start_epoch = checkpoint.get("epoch", 0) + 1
                # if "best_val_metric" in checkpoint:
                    # best_val_metric = checkpoint["best_val_metric"]
                    # if is_main_process:
                    #     logger.info(f"Loaded best validation metric: {best_val_metric:.4f}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                if is_main_process:
                    logger.info("Starting training from scratch")
                start_epoch = 0
        
        # Wrap model with DDP
        if args.local_rank != -1:
            model = DDP(
                model, 
                device_ids=[args.local_rank], 
                output_device=args.local_rank,
                find_unused_parameters=args.find_unused_parameters
            )
        
        # Create processor
        if not isinstance(model, DDP):
            if args.model_type == "salmonn":    
                processor = get_processor(args.model_type, model.input_processor,model.llama_tokenizer)
            else:
                processor = get_processor(args.model_type, model.processor)
        else:
            processor = get_processor(model.module.processor)
        
        # Load datasets
        train_datasets = {}
        val_datasets = {}
        start_time = time.time()
        for dt in dataset_types:
            try:
                full_train_dataset = load_dataset(dt, split="train")
                full_val_dataset = load_dataset(dt, split=args.val_split)
                
                # Apply debug sample limiting if specified
                if args.debug_samples > 0:
                    if is_main_process:
                        logger.info(f"DEBUG MODE: Limiting to {args.debug_samples} samples for dataset {dt}")
                    train_datasets[dt] = full_train_dataset.select(range(args.debug_samples))
                    val_datasets[dt] = full_val_dataset.select(range(args.debug_samples))
                else:
                    train_datasets[dt] = full_train_dataset
                    val_datasets[dt] = full_val_dataset.select(range(100))  # Keep original validation subset
                
                if is_main_process:
                    logger.info(f"Loaded dataset {dt}: {len(train_datasets[dt])} training examples, "
                              f"{len(val_datasets[dt])} validation examples")
            except Exception as e:
                logger.error(f"Error loading dataset {dt}: {e}")
                return 1
        
        if is_main_process:
            logger.info(f"Dataset loading time: {time.time() - start_time:.2f} seconds")
            if args.debug_samples > 0:
                logger.info("Running in DEBUG mode with limited samples")
        
        # Create training dataset using DatasetFactory
        try:
            train_dataset = DatasetFactory.create_dataset(
                dataset_type=dataset_types,
                dataset=train_datasets,
                processor=processor,
                is_training=True,
                input_mode=args.input_mode,
                fewshot_mode=args.fewshot_mode,
                num_examples=args.num_examples,
                random_examples=True,
                model_type=args.model_type,
                run_name=args.run_name
            )
            if is_main_process:
                logger.info(f"Created training dataset with {len(train_dataset)} examples")
        except Exception as e:
            logger.error(f"Error creating training dataset: {e}")
            return 1
        
        # Create validation dataset
        try:
            val_dataset = DatasetFactory.create_dataset(
                dataset_type=dataset_types,
                dataset=val_datasets,
                processor=processor,
                is_training=False,  # Use inference mode for validation
                input_mode=args.input_mode,
                fewshot_mode=args.fewshot_mode,
                num_examples=args.num_examples,
                random_examples=False,
                model_type=args.model_type,
                run_name=args.run_name
            )
            if is_main_process:
                logger.info(f"Created validation dataset with {len(val_dataset)} examples")
        except Exception as e:
            logger.error(f"Error creating validation dataset: {e}")
            return 1
        
        # Create data sampler
        if args.local_rank != -1:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
        
        # Determine number of workers for data loading
        num_workers = min(
            os.cpu_count() or 4, 
            args.dataloader_num_workers
        )
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            collate_fn=processor.collate_batch,
            num_workers=num_workers,
            pin_memory=args.dataloader_pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=False
        )
        
        # Use separate batch size for evaluation if specified
        eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=processor.collate_batch,
            num_workers=num_workers,
            pin_memory=args.dataloader_pin_memory,
            prefetch_factor=args.dataloader_prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        )
        
        # Create optimizer with better defaults
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay
        )
        
        # Create learning rate scheduler
        total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
        
        # Create scheduler based on user choice
        scheduler = get_scheduler(
            name=args.scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup mixed precision training
        if args.bf16 and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            if is_main_process:
                logger.info("Using bfloat16 mixed precision training")
        elif args.fp16:
            amp_dtype = torch.float16
            if is_main_process:
                logger.info("Using float16 mixed precision training")
        else:
            amp_dtype = None
            if is_main_process:
                logger.info("Using full precision training")
        
        scaler = GradScaler() if amp_dtype == torch.float16 else None
        
        if is_main_process:
            logger.info(f"Starting training for {args.num_epochs} epochs")
            logger.info(f"Total optimization steps: {total_steps}")
            logger.info(f"Using {num_workers} workers for data loading")
        
        # Training loop
        global_step = 0
        train_start_time = time.time()
        
        for epoch in range(start_epoch, args.num_epochs):
            epoch_start_time = time.time()
            if is_main_process:
                logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
                log_gpu_memory_usage(logger)
            
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            model.train()
            total_loss = 0
            
            # Use tqdm for progress bar if main process
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process)
            
            for step, batch in enumerate(train_iter):
                step_start = time.time()
                
                # Add detailed logging for first epoch and first 10 iterations
                if epoch == 0 and step < 10 and is_main_process:
                    logger.info(f"\n{'='*20} Epoch {epoch+1}, Batch {step+1} {'='*20}")
                    
                    # Log input prompt
                    logger.info("\n=== Input Prompt ===")
                    if "prompt" in batch:
                        for i in range(min(1, len(batch["prompt"]))):  # Log first example in batch
                            logger.info(f"Prompt {i+1}:")
                            logger.info(batch["prompt"][i])
                    
                    # Log target completion
                    logger.info("\n=== Expected Output ===")
                    if "completion" in batch:
                        for i in range(min(1, len(batch["completion"]))):
                            logger.info(f"Target {i+1}:")
                            logger.info(batch["completion"][i])
                    
                    
                    logger.info("\n" + "="*60 + "\n")
                
                # Clear memory every 50 steps
                if step > 0 and step % 50 == 0:
                    if is_main_process:
                        logger.info(f"Clearing memory at step {step}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        log_gpu_memory_usage(logger)
                
                try:
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    # Log which precision path we're taking
                    if step == 0:  # Only log on first step to avoid spam
                        if amp_dtype is not None:
                            logging.info(f"=== Using Mixed Precision Training with dtype: {amp_dtype} ===")
                            if scaler is not None:
                                logging.info("Using GradScaler for FP16 training")
                            else:
                                logging.info("Using BF16 - no GradScaler needed")
                        else:
                            logging.info("=== Using Standard Precision Training ===")

                    # Forward pass with mixed precision if enabled
                    if amp_dtype is not None:
                        with autocast(dtype=amp_dtype):
                            outputs = model(batch)
                            loss = outputs["loss"]
                            # Scale loss for gradient accumulation
                            loss = loss / args.gradient_accumulation_steps
                        
                        # Backward pass with gradient scaling for float16
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            if step % 100 == 0:  # Log every 100 steps
                                logging.info("Using scaled backward pass (FP16)")
                        else:
                            loss.backward()
                            if step % 100 == 0:
                                logging.info("Using regular backward pass (BF16)")
                        
                        # Update weights if gradient accumulation is complete
                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            if step % 100 == 0:
                                logging.info(f"Updating weights after {args.gradient_accumulation_steps} accumulation steps")
                            
                            if scaler is not None:
                                # Unscale gradients for clipping with float16
                                scaler.unscale_(optimizer)
                                if step % 100 == 0:
                                    logging.info("Unscaled gradients for clipping (FP16)")
                            
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            
                            # Update weights with or without gradient scaling
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                                if step % 100 == 0:
                                    logging.info("Updated weights with scaler (FP16)")
                            else:
                                optimizer.step()
                                if step % 100 == 0:
                                    logging.info("Updated weights without scaler (BF16)")
                                
                            optimizer.zero_grad(set_to_none=True)
                            scheduler.step()
                            global_step += 1
                    else:
                        # Standard precision training
                        if step % 100 == 0:
                            logging.info("=== Standard precision forward pass ===")
                        outputs = model(batch)
                        loss = outputs["loss"]
                        loss = loss / args.gradient_accumulation_steps
                        
                        loss.backward()
                        
                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            if step % 100 == 0:
                                logging.info("=== Standard precision weight update ===")
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            scheduler.step()
                            global_step += 1
                    
                    # Log progress
                    total_loss += loss.item() * args.gradient_accumulation_steps
                    
                    # Update progress bar
                    if is_main_process:
                        train_iter.set_postfix({
                            'loss': f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                            'lr': f"{optimizer.param_groups[0]['lr']:.8f}",
                            'step_time': f"{time.time() - step_start:.2f}s"
                        })
                    
                    if is_main_process and (step + 1) % args.log_steps == 0:
                        # Calculate speed
                        examples_per_second = args.batch_size / (time.time() - step_start)
                        
                        logger.info(
                            f"Epoch {epoch+1}, Step {step+1}/{len(train_loader)}, "
                            f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f}, "
                            f"LR: {optimizer.param_groups[0]['lr']:.8f}, "
                            f"Speed: {examples_per_second:.2f} examples/s, "
                            f"Step time: {time.time() - step_start:.2f}s"
                        )


                    # Validation
                    if step % args.eval_every == 0:
                        val_start_time = time.time()
                        val_loss, val_metrics = validate(
                            model=model,
                            val_loader=val_loader,
                            device=device,
                            dataset_types=dataset_types,
                            amp_dtype=amp_dtype,
                            is_main_process=is_main_process,
                            logger=logger
                        )
                        
                        val_time = time.time() - val_start_time
                        
                        if is_main_process:
                            logger.info(
                                f"Validation completed in {val_time:.2f}s, "
                                f"Validation Loss: {val_loss:.4f}"
                            )
                            
                            # Log validation metrics
                            for dt in val_metrics:
                                logger.info(f"Metrics for {dt}:")
                                for metric, value in val_metrics[dt].items():
                                    if isinstance(value, (float, int)):
                                        logger.info(f"  {metric}: {value:.4f}")
                                    else:
                                        logger.info(f"  {metric}: {value}")

                except Exception as e:
                    logger.error(f"Error during training step: {e}")
                    if is_main_process:
                        logger.info(f"Skipping problematic batch at step {step+1}")
                    continue
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - epoch_start_time
            
            if is_main_process:
                logger.info(
                    f"Epoch {epoch+1} completed in {epoch_time:.2f}s, "
                    f"Average Loss: {avg_loss:.4f}"
                )
                
                # Force garbage collection to free memory
                gc.collect()
                torch.cuda.empty_cache()
                log_gpu_memory_usage(logger)
            

                
                # Force garbage collection after validation
                if is_main_process:
                    gc.collect()
                    torch.cuda.empty_cache()
                    log_gpu_memory_usage(logger)
            
            # Save checkpoint
            if is_main_process and (epoch + 1) % args.save_every == 0:
                checkpoint_dir = os.path.join(
                    args.output_dir, "checkpoints", f"epoch_{epoch+1}_loss_{avg_loss:.4f}"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint(
                    model=model.module if args.local_rank != -1 else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=avg_loss,
                    path=os.path.join(checkpoint_dir, "model.pt")
                )
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save final model
        if is_main_process:
            final_path = os.path.join(args.output_dir, "final_model.pt")
            save_checkpoint(
                model=model.module if args.local_rank != -1 else model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=args.num_epochs - 1,
                loss=avg_loss,
                path=final_path
            )
            logger.info(f"Saved final model to {final_path}")
            
            # Log total training time
            total_time = time.time() - train_start_time
            logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
            
            return 0
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        return 1

def validate(model, val_loader, device, dataset_types, amp_dtype=None, is_main_process=True, logger=None):
    """Run validation on the validation set"""
    model.eval()
    total_loss = 0
    all_results = {dt.value: [] for dt in dataset_types}
    
    with torch.no_grad():
        total_batches = len(val_loader)
        
        # Log start of validation
        if is_main_process and logger:
            logger.info(f"Starting validation with {total_batches} batches")
        
        for batch_idx, batch in enumerate(val_loader):
            # Log progress at 10% intervals
            if is_main_process and logger and batch_idx % max(1, total_batches // 10) == 0:
                logger.info(f"Validation progress: {batch_idx}/{total_batches} batches ({batch_idx/total_batches*100:.0f}%)")
            
            try:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass with mixed precision if enabled
                if amp_dtype is not None:
                    with autocast(dtype=amp_dtype):
                        outputs = model(batch)
                        loss = outputs["loss"]
                else:
                    outputs = model(batch)
                    loss = outputs["loss"]
                
                total_loss += loss.item()
                
                # Generate predictions for evaluation
                predictions = model.generate_output(batch)
                
                # Process results
                for i, pred in enumerate(predictions):
                    dt = batch["dataset_type"][i] if isinstance(batch["dataset_type"], list) else batch["dataset_type"]
                    dt_key = dt.value if hasattr(dt, 'value') else str(dt)  # Convert to string
                    true_label = batch["completion"][i] if isinstance(batch["completion"], list) else batch["completion"]

                    result = {
                        "text": batch["text"][i] if isinstance(batch["text"], list) else batch["text"],
                        "true_label": true_label,
                        "predicted_label":  str(pred).strip(),
                        "dataset_type": dt_key  # Use string value
                    }
                    all_results[dt_key].append(result)

                    logger.info(f"Predicted: {pred.strip()}")
                    logger.info(f"True label: {true_label}")
                    logger.info(f"dataset type: {dt_key}")
                    logger.info("=" * 50)
            except Exception as e:
                if logger:
                    logger.error(f"Error during validation batch: {str(e)}")
                continue
    
    # Calculate average loss
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics for each dataset type
    metrics = {}
    for dt in dataset_types:
        dt_key = dt.value if hasattr(dt, 'value') else str(dt)
        dt_results = all_results.get(dt_key, [])
        if dt_results:
            try:
                dt_metrics = evaluate_predictions(dt_results, dt)
                metrics[dt_key] = dt_metrics
            except Exception as e:
                if logger:
                    logger.error(f"Error evaluating predictions for {dt_key}: {str(e)}")
                metrics[dt_key] = {"error": str(e)}
    
    # Log completion
    if is_main_process and logger:
        logger.info(f"Validation complete: {total_batches}/{total_batches} batches (100%)")
    
    return avg_loss, metrics

if __name__ == "__main__":
    sys.exit(main())


            
