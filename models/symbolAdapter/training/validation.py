"""
Validation Logic for Symbol Adapter Training
Handles validation with different symbol modes and configurations
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from tqdm import tqdm
import numpy as np
import re

from ..symbol_manager import SymbolManager
from ..configs.training_configs import TrainingConfig, SymbolMode


class ValidationManager:
    """Manages validation for different training modes and symbol configurations"""
    
    def __init__(
        self,
        config: TrainingConfig,
        symbol_manager: SymbolManager,
        tokenizer,
        max_val_samples: int = 200
    ):
        self.config = config
        self.symbol_manager = symbol_manager
        self.tokenizer = tokenizer
        self.max_val_samples = max_val_samples
        
    def validate_model(
        self,
        model,
        val_dataloader,
        epoch: int,
        phase: str,
        cycle: int = 0,
        bypass_mlp: bool = False,
        use_original_labels: bool = False,
        use_dynamic_symbols: bool = False
    ) -> Dict[str, float]:
        """
        Comprehensive validation function that handles all training modes
        
        Args:
            model: The model to validate
            val_dataloader: Validation data loader
            epoch: Current epoch
            phase: Training phase ("lora", "mlp", "joint", etc.)
            cycle: Current training cycle
            bypass_mlp: Whether to bypass MLP during validation
            use_original_labels: Whether to use original labels instead of symbols
            use_dynamic_symbols: Whether to generate fresh symbols for this validation
            
        Returns:
            Dictionary with validation metrics
        """
        model.eval()
        
        # Configure model for validation
        if bypass_mlp:
            model.set_lora_training_mode()
        else:
            model.set_mlp_training_mode()
        
        # Get symbols for validation
        if use_original_labels:
            symbol_mappings = {}  # No symbol replacement
            mode_name = "Original"
        elif use_dynamic_symbols:
            # Generate fresh symbols for this validation
            symbol_mappings = self.symbol_manager._generate_symbol_mappings()
            mode_name = "Fresh-Symbols"
        else:
            # Use symbols from training epoch
            symbol_mappings = self.symbol_manager.get_symbols_for_epoch(epoch)
            mode_name = "Fixed-Symbols"
        
        # Configure validation mode name
        mlp_mode = "NoMLP" if bypass_mlp else "MLP"
        full_mode_name = f"{mlp_mode}+{mode_name}"
        
        logging.info(f"=== Validation: {full_mode_name} (Epoch {epoch}, {phase.upper()}) ===")
        if symbol_mappings:
            logging.info(f"Using symbols: {symbol_mappings}")
        
        # Run validation
        try:
            with torch.no_grad():
                metrics = self._run_validation_loop(
                    model=model,
                    val_dataloader=val_dataloader,
                    symbol_mappings=symbol_mappings,
                    mode_name=full_mode_name,
                    epoch=epoch,
                    phase=phase
                )
            
            logging.info(f"✓ {full_mode_name} Validation Score: {metrics['accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            logging.error(f"❌ Validation failed for {full_mode_name}: {str(e)}")
            return {"accuracy": 0.0, "loss": float('inf'), "total_samples": 0}
        
        finally:
            model.train()  # Reset to training mode
    
    def _run_validation_loop(
        self,
        model,
        val_dataloader,
        symbol_mappings: Dict[str, str],
        mode_name: str,
        epoch: int,
        phase: str
    ) -> Dict[str, float]:
        """Run the actual validation loop"""
        
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0.0
        processed_samples = 0
        
        # Create progress bar
        progress_bar = tqdm(
            val_dataloader, 
            desc=f"Val {mode_name}",
            total=min(len(val_dataloader), self.max_val_samples)
        )
        
        try:
            for batch_idx, batch in enumerate(progress_bar):
                # Limit validation samples
                if processed_samples >= self.max_val_samples:
                    break
                
                # Apply symbol replacement if needed
                if symbol_mappings:
                    batch = self.symbol_manager.replace_symbols_in_batch(batch, epoch)
                
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                try:
                    outputs = model(**batch)
                    loss = outputs.get("loss", 0.0)
                    
                    if loss is not None and not torch.isnan(loss):
                        total_loss += loss.item()
                    
                    # Extract predictions and targets
                    batch_correct, batch_total = self._compute_accuracy(
                        outputs=outputs,
                        batch=batch,
                        symbol_mappings=symbol_mappings
                    )
                    
                    correct_predictions += batch_correct
                    total_predictions += batch_total
                    processed_samples += batch_total
                    
                    # Update progress bar
                    current_acc = correct_predictions / max(total_predictions, 1)
                    progress_bar.set_postfix({
                        'acc': f'{current_acc:.3f}',
                        'samples': processed_samples
                    })
                    
                except Exception as e:
                    logging.warning(f"Batch {batch_idx} failed: {str(e)}")
                    continue
                    
        except KeyboardInterrupt:
            logging.info(f"Validation interrupted at {processed_samples} samples")
        
        finally:
            progress_bar.close()
        
        # Compute final metrics
        accuracy = correct_predictions / max(total_predictions, 1)
        avg_loss = total_loss / max(len(val_dataloader), 1)
        
        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "total_samples": processed_samples
        }
    
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
    
    def _compute_accuracy(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        symbol_mappings: Dict[str, str]
    ) -> Tuple[int, int]:
        """
        Compute accuracy from model outputs
        
        Returns:
            Tuple of (correct_predictions, total_predictions)
        """
        try:
            # Get logits from outputs
            if "logits" in outputs:
                logits = outputs["logits"]
            elif "prediction_scores" in outputs:
                logits = outputs["prediction_scores"]
            else:
                # Try to get from loss computation
                return self._compute_accuracy_from_generation(outputs, batch, symbol_mappings)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Get targets
            if "labels" in batch:
                targets = batch["labels"]
            elif "target_ids" in batch:
                targets = batch["target_ids"]
            else:
                logging.warning("No targets found in batch")
                return 0, 1
            
            # Mask out padding tokens (-100)
            mask = targets != -100
            
            if mask.sum() == 0:
                return 0, 1
            
            # Compute accuracy
            correct = (predictions == targets) & mask
            return correct.sum().item(), mask.sum().item()
            
        except Exception as e:
            logging.warning(f"Accuracy computation failed: {str(e)}")
            return 0, 1
    
    def _compute_accuracy_from_generation(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        symbol_mappings: Dict[str, str]
    ) -> Tuple[int, int]:
        """
        Compute accuracy for generation-based models
        """
        try:
            # This is for models that generate text outputs
            if "generated_text" in outputs:
                generated_texts = outputs["generated_text"]
            else:
                # Try to generate text from model
                return 0, len(batch.get("prompt", []))
            
            # Get expected completions
            expected_completions = batch.get("completion", [])
            
            if len(generated_texts) != len(expected_completions):
                return 0, len(expected_completions)
            
            correct = 0
            
            for generated, expected in zip(generated_texts, expected_completions):
                # Convert symbols back to original labels if needed
                if symbol_mappings:
                    generated = self.symbol_manager.convert_symbols_back(generated)
                    expected = self.symbol_manager.convert_symbols_back(expected)
                
                # Simple exact match (can be made more sophisticated)
                if self._is_correct_prediction(generated, expected):
                    correct += 1
            
            return correct, len(expected_completions)
            
        except Exception as e:
            logging.warning(f"Generation accuracy computation failed: {str(e)}")
            return 0, len(batch.get("prompt", []))
    
    def _is_correct_prediction(self, generated: str, expected: str) -> bool:
        """
        Check if generated text is correct
        Can be customized for different accuracy criteria
        """
        # Clean and normalize text
        generated_clean = generated.strip().lower()
        expected_clean = expected.strip().lower()
        
        # Exact match
        if generated_clean == expected_clean:
            return True
        
        # Check if expected answer is contained in generated text
        if expected_clean in generated_clean:
            return True
        
        # For classification tasks, extract the label
        generated_label = self._extract_label(generated_clean)
        expected_label = self._extract_label(expected_clean)
        
        return generated_label == expected_label
    
    def _extract_label(self, text: str) -> str:
        """Extract classification label from text"""
        # Remove common prefixes/suffixes
        text = re.sub(r'^(the answer is|answer:|prediction:|label:)\s*', '', text)
        text = re.sub(r'\s*(\.|\!|\?)$', '', text)
        
        # Extract first word that could be a label
        words = text.split()
        if words:
            return words[0]
        
        return text
    
    def run_comprehensive_validation(
        self,
        model,
        val_dataloader,
        epoch: int,
        phase: str,
        cycle: int = 0
    ) -> Dict[str, float]:
        """
        Run comprehensive validation with all modes (MLP+Symbols, NoMLP+Symbols, etc.)
        
        Returns:
            Dictionary with all validation metrics
        """
        validation_results = {}
        
        # Determine which validation modes to run based on training phase
        if phase in ["bypass_mlp_sym"]:
            # For bypass MLP mode, only test with fresh symbols and original labels
            modes = [
                ("fresh_symbols", True, False, True),    # NoMLP + Fresh Symbols
                ("original_labels", True, True, False),  # NoMLP + Original Labels
            ]
        elif phase in ["bypass_mlp_org"]:
            # For bypass MLP original mode, only test without symbols
            modes = [
                ("original_labels", True, True, False),  # NoMLP + Original Labels
            ]
        elif phase == "joint":
            # For joint training, test all combinations
            modes = [
                ("mlp_symbols", False, False, False),     # MLP + Fixed Symbols
                ("no_mlp_symbols", True, False, False),   # NoMLP + Fixed Symbols
                ("mlp_original", False, True, False),     # MLP + Original Labels
                ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
            ]
        else:
            # Standard validation modes
            modes = [
                ("mlp_symbols", False, False, False),     # MLP + Fixed Symbols
                ("no_mlp_symbols", True, False, False),   # NoMLP + Fixed Symbols
                ("mlp_original", False, True, False),     # MLP + Original Labels
                ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
            ]
        
        # Run each validation mode
        for mode_key, bypass_mlp, use_original, use_dynamic in modes:
            try:
                metrics = self.validate_model(
                    model=model,
                    val_dataloader=val_dataloader,
                    epoch=epoch,
                    phase=phase,
                    cycle=cycle,
                    bypass_mlp=bypass_mlp,
                    use_original_labels=use_original,
                    use_dynamic_symbols=use_dynamic
                )
                
                validation_results[mode_key] = metrics["accuracy"]
                validation_results[f"{mode_key}_loss"] = metrics["loss"]
                
            except Exception as e:
                logging.error(f"Validation mode {mode_key} failed: {str(e)}")
                validation_results[mode_key] = 0.0
                validation_results[f"{mode_key}_loss"] = float('inf')
        
        return validation_results
    
    def log_validation_summary(
        self,
        validation_results: Dict[str, float],
        epoch: int,
        phase: str,
        cycle: int = 0
    ):
        """Log validation summary in a readable format"""
        logging.info("=" * 60)
        logging.info(f"{phase.upper()} CYCLE {cycle} EPOCH {epoch} VALIDATION SUMMARY:")
        logging.info("=" * 60)
        
        # Group results by type
        accuracy_results = {k: v for k, v in validation_results.items() if not k.endswith('_loss')}
        
        for mode, accuracy in accuracy_results.items():
            logging.info(f"{mode:<20}: {accuracy:.4f}")
        
        # Find best performing mode
        if accuracy_results:
            best_mode = max(accuracy_results, key=accuracy_results.get)
            best_score = accuracy_results[best_mode]
            logging.info("-" * 60)
            logging.info(f"Best performance: {best_mode} = {best_score:.4f}")
        
        logging.info("=" * 60)


# Backwards compatibility functions
def validate_model(
    model,
    val_dataloader,
    args,
    current_mappings: Dict[str, str],
    phase: str,
    epoch: int,
    bypass_mlp: bool = False,
    use_original_labels: bool = False,
    max_val_samples: int = 200
) -> float:
    """
    Legacy validation function for backwards compatibility
    """
    # Create minimal config and symbol manager
    from ..configs.training_configs import TrainingConfig, SymbolMode
    
    config = TrainingConfig(
        device=getattr(args, 'device', 'cuda:0'),
        data_config=TrainingConfig().data_config
    )
    
    # Create symbol manager with current mappings
    from ..symbol_manager import SymbolManager
    
    symbol_manager = SymbolManager(
        original_labels=list(current_mappings.keys()) if current_mappings else [],
        tokenizer=getattr(args, 'tokenizer', None),
        dynamic_per_epoch=False
    )
    
    if current_mappings:
        symbol_manager.fixed_mappings = current_mappings
    
    # Create validation manager
    validator = ValidationManager(
        config=config,
        symbol_manager=symbol_manager,
        tokenizer=getattr(args, 'tokenizer', None),
        max_val_samples=max_val_samples
    )
    
    # Run validation
    metrics = validator.validate_model(
        model=model,
        val_dataloader=val_dataloader,
        epoch=epoch,
        phase=phase,
        bypass_mlp=bypass_mlp,
        use_original_labels=use_original_labels
    )
    
    return metrics["accuracy"]


def run_validation_suite(
    model,
    val_dataloader,
    symbol_manager: SymbolManager,
    config: TrainingConfig,
    epoch: int,
    phase: str,
    cycle: int = 0
) -> Dict[str, float]:
    """
    Run complete validation suite for an epoch
    
    Returns:
        Dictionary with all validation metrics
    """
    validator = ValidationManager(
        config=config,
        symbol_manager=symbol_manager,
        tokenizer=None,  # Will be set by caller
        max_val_samples=config.data_config.val_max_samples
    )
    
    validation_results = validator.run_comprehensive_validation(
        model=model,
        val_dataloader=val_dataloader,
        epoch=epoch,
        phase=phase,
        cycle=cycle
    )
    
    validator.log_validation_summary(
        validation_results=validation_results,
        epoch=epoch,
        phase=phase,
        cycle=cycle
    )
    
    return validation_results