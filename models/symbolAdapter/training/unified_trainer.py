import logging
import os
import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configs.training_configs import TrainingConfig
from ..symbol_manager import SymbolManager
from .validation import ValidationManager
from .schedulers import TrainingStep


class UnifiedTrainer:
    """
    Unified trainer that can handle all training phases: LoRA, MLP, Joint
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        symbol_manager: SymbolManager,
        tokenizer=None
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.symbol_manager = symbol_manager
        self.tokenizer = tokenizer
        
        # Training state
        self.current_epoch = 0
        self.optimizer = None
        self.scheduler = None
        
        # Initialize validator
        self.validator = ValidationManager(
            config=config,
            symbol_manager=symbol_manager,
            tokenizer=tokenizer,
            max_val_samples=getattr(config.data_config, 'val_max_samples', 200)
        )
        
        logging.info("UnifiedTrainer initialized")
    
    def train_step(self, step: TrainingStep) -> dict:
        """Universal training method for any phase"""
        logging.info(f"🚀 Starting {step.phase.upper()} Training: {step.description}")
        
        # Setup parameters and optimizer based on phase
        self._setup_training_phase(step)
        
        validation_scores = {}
        step_metrics = {
            "train_loss": [],
            "validation_scores": {},
            "best_epoch": 0,
            "best_score": 0.0
        }
        
        # Train for specified epochs
        for epoch in range(step.epochs):
            self.current_epoch = epoch
            logging.info(f"--- {step.phase.upper()} Epoch {epoch+1}/{step.epochs} ---")
            
            # Training epoch
            epoch_loss = self._train_epoch(step, epoch)
            step_metrics["train_loss"].append(epoch_loss)
            
            logging.info(f"{step.phase.upper()} Epoch {epoch+1} Loss: {epoch_loss:.6f}")
            
            # Validation after each epoch
            if getattr(self.config, 'validate_every_epoch', True):
                epoch_validation = self._validate_epoch(step, epoch)
                validation_scores.update(epoch_validation)
                step_metrics["validation_scores"][epoch] = epoch_validation
                
                # ✅ TRACK EACH EPOCH INDIVIDUALLY
                if hasattr(self, 'orchestrator') and hasattr(self.orchestrator, '_track_epoch_summary'):
                    self.orchestrator._track_epoch_summary(step, epoch, epoch_validation)
                
                # Log epoch summary
                self._log_epoch_summary(step, epoch + 1, epoch_loss, epoch_validation)
                
                # Track best performance
                main_score = self._get_main_score(epoch_validation, step.phase)
                if main_score > step_metrics["best_score"]:
                    step_metrics["best_score"] = main_score
                    step_metrics["best_epoch"] = epoch
                    
                    # Save best checkpoint
                    if self.config.checkpoint_frequency > 0:
                        self._save_checkpoint(step, epoch, "best")
            
            # Save periodic checkpoint
            if self.config.checkpoint_frequency > 0 and (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(step, epoch, "periodic")
        
        # Log step summary
        self._log_step_summary(step, step_metrics)
        
        return validation_scores
    
    def _setup_training_phase(self, step: TrainingStep):
        """Setup model parameters and optimizer based on training phase"""
        if step.phase == "lora":
            # Setup LoRA training
            self._setup_lora_training(step)
            
        elif step.phase == "mlp":
            # Setup MLP training
            self._setup_mlp_training(step)
            
        elif step.phase == "joint":
            # Setup joint training (both LoRA and MLP)
            self._setup_joint_training(step)
        
        else:
            raise ValueError(f"Unknown training phase: {step.phase}")
    
    def _setup_lora_training(self, step: TrainingStep):
        """Setup LoRA parameters for training"""
        logging.info(f"Setting up LoRA training for {step.description}")
        
        # Freeze MLP parameters and unfreeze LoRA parameters
        self._freeze_mlp_parameters()
        self._unfreeze_lora_parameters()
        
        # Log parameter status
        self._log_parameter_status("LoRA")
        
        # Setup optimizer for LoRA parameters only
        self._setup_lora_optimizer(step)
        
        # Set model to training mode
        self.model.train()
        
        logging.info("✓ LoRA training setup complete")
    
    def _setup_mlp_training(self, step: TrainingStep):
        """Setup MLP parameters for training"""
        logging.info(f"Setting up MLP training for {step.description}")
        
        # Check if MLP training is possible
        if getattr(self.model, 'bypass_mlp', False):
            raise ValueError("Cannot perform MLP training when bypass_mlp=True. No MLP layers exist.")
        
        # Freeze LoRA parameters and unfreeze MLP parameters
        self._freeze_lora_parameters()
        self._unfreeze_mlp_parameters()
        
        # Log parameter status
        self._log_parameter_status("MLP")
        
        # Setup optimizer for MLP parameters only
        self._setup_mlp_optimizer(step)
        
        # Set model to training mode
        self.model.train()
        
        logging.info("✓ MLP training setup complete")
    
    def _setup_joint_training(self, step: TrainingStep):
        """Setup joint LoRA + MLP training"""
        logging.info(f"Setting up Joint MLP+LoRA training for {step.description}")
        
        # Check if this is effectively LoRA-only training
        if getattr(self.model, 'bypass_mlp', False):
            logging.warning("BYPASS_MLP=True: Joint training will only train LoRA + SALMONN (no MLP layers)")
        
        # Unfreeze both MLP and LoRA parameters
        self._unfreeze_mlp_parameters()
        self._unfreeze_lora_parameters()
        
        # Log parameter status
        self._log_parameter_status("Joint")
        
        # Setup optimizer for both MLP and LoRA parameters
        self._setup_joint_optimizer(step)
        
        # Set model to training mode
        self.model.train()
        
        logging.info("✓ Joint MLP+LoRA training setup complete")
    
    def _freeze_mlp_parameters(self):
        """Freeze all MLP parameters (if any exist)"""
        mlp_param_count = 0
        
        # Check if MLPs are bypassed at the model level
        if getattr(self.model, 'bypass_mlp', False):
            logging.info("BYPASS_MLP=True: No MLP parameters to freeze")
            return
        
        # Only count actual MLP layers from our architecture
        for name, param in self.model.named_parameters():
            # Only count our specific MLP layers
            if ('input_mlp' in name or 'output_mlp' in name) and 'lora' not in name.lower():
                param.requires_grad = False
                mlp_param_count += param.numel()
        
        if mlp_param_count > 0:
            logging.info(f"Frozen {mlp_param_count:,} MLP parameters")
        else:
            logging.info("No MLP parameters found to freeze")
    
    def _freeze_lora_parameters(self):
        """Freeze all LoRA parameters AND unfreeze SALMONN speech components"""
        
        # Freeze LoRA parameters
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = False
        
        if hasattr(self.model, 'salmonn'):
            # Freeze QFormer components during MLP training
            if hasattr(self.model.salmonn, 'speech_Qformer'):
                for name, param in self.model.salmonn.speech_Qformer.named_parameters():
                    param.requires_grad = False
            
            # Freeze speech_query_tokens
            if hasattr(self.model.salmonn, 'speech_query_tokens'):
                self.model.salmonn.speech_query_tokens.requires_grad = False
            
            # Freeze speech_llama_proj
            if hasattr(self.model.salmonn, 'speech_llama_proj'):
                for name, param in self.model.salmonn.speech_llama_proj.named_parameters():
                    param.requires_grad = False
    
    def _unfreeze_lora_parameters(self):
        """Unfreeze LoRA parameters and SALMONN speech components"""
        lora_param_count = 0
        
        # Unfreeze LoRA parameters
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                lora_param_count += param.numel()
        
        # For SALMONN model, unfreeze speech components during LoRA training
        speech_param_count = 0
        if hasattr(self.model, 'salmonn'):
            # Unfreeze QFormer components
            if hasattr(self.model.salmonn, 'speech_Qformer'):
                for name, param in self.model.salmonn.speech_Qformer.named_parameters():
                    param.requires_grad = True
                    speech_param_count += param.numel()
            
            # Unfreeze speech_query_tokens
            if hasattr(self.model.salmonn, 'speech_query_tokens'):
                self.model.salmonn.speech_query_tokens.requires_grad = True
                speech_param_count += self.model.salmonn.speech_query_tokens.numel() if hasattr(self.model.salmonn.speech_query_tokens, 'numel') else 1
            
            # Unfreeze speech_llama_proj
            if hasattr(self.model.salmonn, 'speech_llama_proj'):
                for name, param in self.model.salmonn.speech_llama_proj.named_parameters():
                    param.requires_grad = True
                    speech_param_count += param.numel()
        
        logging.info(f"Unfrozen {lora_param_count:,} LoRA parameters")
        if speech_param_count > 0:
            logging.info(f"Unfrozen {speech_param_count:,} SALMONN speech parameters")
    
    def _unfreeze_mlp_parameters(self):
        """Unfreeze MLP parameters based on configuration"""
        
        # Check if MLPs are bypassed at the model level
        if getattr(self.model, 'bypass_mlp', False):
            logging.warning("BYPASS_MLP=True: No MLP parameters to unfreeze")
            return
        
        mlp_param_count = 0
        
        # Unfreeze input MLP if enabled
        if self.config.mlp_config.use_input_mlp:
            for name, param in self.model.named_parameters():
                if 'input_mlp' in name.lower() or 'symbol_mlp' in name.lower():
                    param.requires_grad = True
                    mlp_param_count += param.numel()
        
        # Unfreeze output MLP if enabled
        if self.config.mlp_config.use_output_mlp:
            for name, param in self.model.named_parameters():
                if 'output_mlp' in name.lower():
                    param.requires_grad = True
                    mlp_param_count += param.numel()
        
        if mlp_param_count > 0:
            logging.info(f"Unfrozen {mlp_param_count:,} MLP parameters")
        else:
            logging.info("No MLP parameters found to unfreeze")
    
    def _log_parameter_status(self, training_mode: str):
        """Log current parameter training status with breakdown"""
        # Count different parameter types
        mlp_params = sum(p.numel() for name, p in self.model.named_parameters() 
                        if p.requires_grad and ('mlp' in name.lower() or 'symbol' in name.lower()) and 'lora' not in name.lower())
        
        lora_params = sum(p.numel() for name, p in self.model.named_parameters() 
                         if p.requires_grad and 'lora' in name.lower())
        
        speech_params = 0
        if hasattr(self.model, 'salmonn'):
            speech_params = sum(p.numel() for name, p in self.model.named_parameters() 
                              if p.requires_grad and ('speech_Qformer' in name or 'speech_query_tokens' in name or 'speech_llama_proj' in name))
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = total_params - trainable_params
        
        logging.info("=" * 60)
        logging.info(f"{training_mode.upper()} TRAINING PARAMETER STATUS:")
        logging.info(f"  MLP Parameters:     {mlp_params:,} ({mlp_params/total_params*100:.1f}%)")
        logging.info(f"  LoRA Parameters:    {lora_params:,} ({lora_params/total_params*100:.1f}%)")
        if speech_params > 0:
            logging.info(f"  SALMONN Parameters: {speech_params:,} ({speech_params/total_params*100:.1f}%)")
        logging.info(f"  Total Trainable:    {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        logging.info(f"  Frozen Parameters:  {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        logging.info(f"  Total Parameters:   {total_params:,}")
        logging.info("=" * 60)
    
    def _setup_lora_optimizer(self, step: TrainingStep):
        """Setup optimizer for LoRA parameters"""
        # Get trainable LoRA parameters
        lora_params = []
        salmonn_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append(param)
                elif ('speech_Qformer' in name or 'speech_query_tokens' in name or 'speech_llama_proj' in name):
                    salmonn_params.append(param)
        
        if not lora_params and not salmonn_params:
            raise ValueError("No trainable LoRA or SALMONN parameters found!")
        
        # Create parameter groups with different learning rates if needed
        param_groups = []
        
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': step.learning_rate or self.config.lora_config.learning_rate,
                'weight_decay': self.config.lora_config.weight_decay,
                'name': 'lora'
            })
        
        if salmonn_params:
            param_groups.append({
                'params': salmonn_params,
                'lr': (step.learning_rate or self.config.lora_config.learning_rate),
                'weight_decay': self.config.lora_config.weight_decay,
                'name': 'salmonn'
            })
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(param_groups)
        
        logging.info(f"Created AdamW optimizer with {len(param_groups)} parameter groups:")
        for group in param_groups:
            logging.info(f"  {group['name']}: {len(group['params'])} layers, LR={group['lr']}")
    
    def _setup_mlp_optimizer(self, step: TrainingStep):
        """Setup optimizer for MLP parameters"""
        # Get trainable MLP parameters
        mlp_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and ('input_mlp' in name.lower() or 'output_mlp' in name.lower()):
                mlp_params.append(param)
        
        if not mlp_params:
            raise ValueError("No trainable MLP parameters found!")
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            mlp_params,
            lr=step.learning_rate or self.config.mlp_config.learning_rate,
            weight_decay=self.config.mlp_config.weight_decay
        )
        
        logging.info(f"Created AdamW optimizer for {len(mlp_params)} MLP parameter groups")
        logging.info(f"Learning rate: {step.learning_rate or self.config.mlp_config.learning_rate}")
        logging.info(f"Weight decay: {self.config.mlp_config.weight_decay}")
    
    def _setup_joint_optimizer(self, step: TrainingStep):
        """Setup optimizer for both MLP and LoRA parameters with different learning rates"""
        # Get different parameter groups
        mlp_params = []
        lora_params = []
        salmonn_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append(param)
                elif ('input_mlp' in name.lower() or 'output_mlp' in name.lower()  or 'symbol' in name.lower()) and 'lora' not in name.lower():
                    mlp_params.append(param)
                elif ('speech_Qformer' in name or 'speech_query_tokens' in name or 'speech_llama_proj' in name):
                    salmonn_params.append(param)
        
        if not mlp_params and not lora_params and not salmonn_params:
            raise ValueError("No trainable parameters found for joint training!")
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        # MLP parameters - use MLP learning rate
        if mlp_params:
            param_groups.append({
                'params': mlp_params,
                'lr': step.learning_rate or self.config.mlp_config.learning_rate,
                'weight_decay': self.config.mlp_config.weight_decay,
                'name': 'mlp'
            })
        
        # LoRA parameters - use LoRA learning rate
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': (step.learning_rate or self.config.lora_config.learning_rate),  
                'weight_decay': self.config.lora_config.weight_decay,
                'name': 'lora'
            })
        
        # SALMONN parameters - use lowest learning rate
        if salmonn_params:
            param_groups.append({
                'params': salmonn_params,
                'lr': (step.learning_rate or self.config.lora_config.learning_rate),  
                'weight_decay': self.config.lora_config.weight_decay,
                'name': 'salmonn'
            })
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(param_groups)
        
        logging.info(f"Created Joint AdamW optimizer with {len(param_groups)} parameter groups:")
        for group in param_groups:
            logging.info(f"  {group['name']}: {len(group['params'])} params, LR={group['lr']:.2e}, WD={group['weight_decay']}")
    
    def _train_epoch(self, step: TrainingStep, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Get symbols for this epoch
        epoch_symbols = self.symbol_manager.get_symbols_for_epoch(epoch)
        
        # Create progress bar
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"{step.phase.upper()} Epoch {epoch+1}",
            leave=False
        )
        
        # Gradient accumulation steps
        if step.phase == "lora":
            accumulation_steps = step.gradient_accumulation_steps or self.config.lora_config.gradient_accumulation_steps
        elif step.phase == "mlp":
            accumulation_steps = step.gradient_accumulation_steps or self.config.mlp_config.gradient_accumulation_steps
        else:  # joint
            accumulation_steps = step.gradient_accumulation_steps or max(
                self.config.mlp_config.gradient_accumulation_steps,
                self.config.lora_config.gradient_accumulation_steps
            )
        
        try:
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # DETAILED BATCH LOGGING FOR FIRST BATCH
                    if batch_idx == 0:
                        logging.info(f"=== {step.phase.upper()} EPOCH BATCH 1 CONTENT ===")
                        if "prompt" in batch:
                            logging.info("ORIGINAL PROMPTS:")
                            for i, prompt in enumerate(batch["prompt"][:2]):
                                logging.info(f"  [{i}] {prompt}")
                        if "completion" in batch:
                            logging.info("ORIGINAL COMPLETIONS:")
                            for i, completion in enumerate(batch["completion"][:2]):
                                logging.info(f"  [{i}] {completion}")
                        
                        if getattr(step, 'use_symbols', True):
                            current_symbols = self.symbol_manager.get_symbols_for_epoch(epoch)
                            logging.info(f"EPOCH {epoch} SYMBOL MAPPINGS: {current_symbols}")
                        else:
                            logging.info("NO SYMBOL REPLACEMENT")
                    
                    # Apply symbol replacement
                    if getattr(step, 'use_symbols', True):
                        updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, epoch=epoch)
                    else:
                        updated_batch = batch
                    
                    # DETAILED BATCH LOGGING AFTER REPLACEMENT
                    if batch_idx == 0:
                        if getattr(step, 'use_symbols', True):
                            logging.info("UPDATED AFTER SYMBOL REPLACEMENT:")
                            if "prompt" in updated_batch:
                                logging.info("UPDATED PROMPTS:")
                                for i, prompt in enumerate(updated_batch["prompt"][:2]):
                                    logging.info(f"  [{i}] {prompt}")
                            if "completion" in updated_batch:
                                logging.info("UPDATED COMPLETIONS:")
                                for i, completion in enumerate(updated_batch["completion"][:2]):
                                    logging.info(f"  [{i}] {completion}")
                        else:
                            logging.info("NO CHANGES (symbols not used)")
                        logging.info(f"=== END {step.phase.upper()} EPOCH BATCH 1 CONTENT ===")
                    
                    # Move to device
                    updated_batch = self._move_batch_to_device(updated_batch)
                    
                    # Forward pass
                    outputs = self.model(updated_batch)
                    loss = outputs.get("loss")
                    
                    if loss is None:
                        logging.warning(f"No loss returned for batch {batch_idx}")
                        continue
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % accumulation_steps == 0:
                        # Gradient clipping
                        max_grad_norm = getattr(self.config, 'max_grad_norm', 0)
                        if step.phase == "lora":
                            max_grad_norm = max_grad_norm or self.config.lora_config.max_grad_norm
                        elif step.phase == "mlp":
                            max_grad_norm = max_grad_norm or self.config.mlp_config.max_grad_norm
                        else:  # joint
                            max_grad_norm = max_grad_norm or min(
                                self.config.mlp_config.max_grad_norm if self.config.mlp_config.max_grad_norm > 0 else float('inf'),
                                self.config.lora_config.max_grad_norm if self.config.lora_config.max_grad_norm > 0 else float('inf')
                            )
                        
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        
                        # Optimizer step
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    # Update metrics
                    total_loss += loss.item() * accumulation_steps
                    num_batches += 1
                    
                    # Update progress bar
                    avg_loss = total_loss / num_batches
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * accumulation_steps:.6f}",
                        'avg_loss': f"{avg_loss:.6f}"
                    })
                    
                    # Log frequently during first epoch
                    if epoch == 0 and batch_idx % getattr(self.config, 'log_frequency', 10) == 0:
                        logging.info(f"Batch {batch_idx}: loss = {loss.item() * accumulation_steps:.6f}")
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue
        
        except KeyboardInterrupt:
            logging.info(f"{step.phase.upper()} training interrupted by user")
        
        finally:
            progress_bar.close()
        
        # Final gradient step if needed
        if num_batches % accumulation_steps != 0:
            max_grad_norm = getattr(self.config, 'max_grad_norm', 0)
            if step.phase == "lora":
                max_grad_norm = max_grad_norm or self.config.lora_config.max_grad_norm
            elif step.phase == "mlp":
                max_grad_norm = max_grad_norm or self.config.mlp_config.max_grad_norm
            else:  # joint
                max_grad_norm = max_grad_norm or min(
                    self.config.mlp_config.max_grad_norm if self.config.mlp_config.max_grad_norm > 0 else float('inf'),
                    self.config.lora_config.max_grad_norm if self.config.lora_config.max_grad_norm > 0 else float('inf')
                )
            
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return total_loss / max(num_batches, 1)
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the correct device"""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.config.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device_batch[key] = [v.to(self.config.device) for v in value]
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _validate_epoch(self, step: TrainingStep, epoch: int, is_final: bool = False) -> dict:
        """Universal validation - DYNAMIC mapping based on actual validation results"""
        logging.info(f"Validating {step.phase.upper()} (Epoch {epoch+1})")
        
        # Use the existing validation method to get comprehensive results
        comprehensive_results = self.validator.run_comprehensive_validation(
            model=self.model,
            val_dataloader=self.val_dataloader,
            epoch=epoch,
            phase=step.phase,
            cycle=step.cycle,
            step=step
        )
        
        # ✅ DYNAMIC MAPPING: Map based on what validation modes were actually run
        validation_scores = {}
        
        # Standard mapping with FRESH SYMBOLS support
        result_to_display_mapping = {
            'mlp_symbols': 'MLP+Symbols',
            'no_mlp_symbols': 'NoMLP+Symbols',
            'mlp_original': 'MLP+Original', 
            'no_mlp_original': 'NoMLP+Original',
            'mlp_fresh': 'MLP+Fresh',        # ✅ ADD FRESH SYMBOLS
            'no_mlp_fresh': 'NoMLP+Fresh'    # ✅ ADD FRESH SYMBOLS
        }
        
        for result_key, display_key in result_to_display_mapping.items():
            if result_key in comprehensive_results:
                validation_scores[display_key] = comprehensive_results[result_key]
        
        return validation_scores

    def _get_main_score(self, validation_scores: dict, phase: str) -> float:
        """Get the main score for tracking best performance"""
        if phase == "lora":
            return validation_scores.get('NoMLP+Symbols', 0.0)
        elif phase == "mlp":
            return validation_scores.get('MLP+Symbols', 0.0)
        elif phase == "joint":
            return validation_scores.get('MLP+Symbols', 0.0)
        else:
            return max(validation_scores.values()) if validation_scores else 0.0
    
    def _log_epoch_summary(self, step: TrainingStep, epoch: int, epoch_loss: float, validation_scores: dict):
        """Log summary after each epoch - DYNAMIC based on actual validation results"""
        logging.info("=" * 120)
        logging.info(f"EPOCH SUMMARY - {step.phase.upper()} Cycle {step.cycle} Epoch {epoch}")
        logging.info("=" * 120)
        logging.info(f"Phase: {step.phase.upper():<6} | Cycle: {step.cycle:<2} | Epoch: {epoch:<2} | Loss: {epoch_loss:.4f}")
        
        # ✅ DYNAMIC: Only log scores that actually exist, including FRESH
        score_parts = []
        
        # Map validation results to display names with FRESH support
        display_mapping = {
            'MLP+Symbols': 'MLP+Sym',
            'NoMLP+Symbols': 'NoMLP+Sym', 
            'MLP+Original': 'MLP+Orig',
            'NoMLP+Original': 'NoMLP+Orig',
            'MLP+Fresh': 'MLP+Fresh',        # ✅ ADD FRESH SYMBOLS
            'NoMLP+Fresh': 'NoMLP+Fresh'     # ✅ ADD FRESH SYMBOLS
        }
        
        for key, display_name in display_mapping.items():
            if key in validation_scores:
                score_parts.append(f"{display_name}: {validation_scores[key]:.4f}")
        
        if score_parts:
            logging.info(" | ".join(score_parts))
        else:
            logging.info("No validation scores available")
        
        logging.info("=" * 120)
    
    def _log_step_summary(self, step: TrainingStep, step_metrics: dict):
        """Log summary after each training step"""
        logging.info("=" * 80)
        logging.info(f"STEP SUMMARY - {step.phase.upper()} Step {step.step_id}")
        logging.info("=" * 80)
        logging.info(f"Best Epoch: {step_metrics['best_epoch'] + 1}")
        logging.info(f"Best Score: {step_metrics['best_score']:.4f}") 
        logging.info("=" * 80)
    
    def _save_checkpoint(self, step: TrainingStep, epoch: int, checkpoint_type: str):
        """Save model checkpoint with only trainable parameters"""
        checkpoint_dir = self.config.get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_name = f"{step.phase}_step{step.step_id}_cycle{step.cycle}_epoch{epoch+1}_{checkpoint_type}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # Save only trainable parameters
        trainable_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_state[name] = param.data.clone()
        
        checkpoint_data = {
            'step_info': {
                'phase': step.phase,
                'step_id': step.step_id,
                'cycle': step.cycle,
                'epoch': epoch + 1,
                'description': step.description
            },
            'model_state': trainable_state,
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"Saved {checkpoint_type} checkpoint: {checkpoint_name}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and restore trainable parameters"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        
        # Load trainable parameters
        model_state = checkpoint['model_state']
        for name, param in self.model.named_parameters():
            if name in model_state:
                param.data.copy_(model_state[name])
                logging.info(f"Loaded parameter: {name}")
        
        # Load optimizer state if available
        if checkpoint.get('optimizer_state') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        step_info = checkpoint.get('step_info', {})
        logging.info(f"Loaded checkpoint from {step_info.get('phase', 'unknown')} "
                   f"step {step_info.get('step_id', 'unknown')} "
                   f"epoch {step_info.get('epoch', 'unknown')}")
        
        return checkpoint