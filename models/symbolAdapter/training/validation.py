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
from ..training.schedulers import TrainingStep

# Import the evaluation functions from utils.evaluation_utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from utils.evaluation_utils import evaluate_predictions, clean_prediction
from data.master_config import DatasetType


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
        Uses utils.evaluation_utils for evaluation
        """
        model.eval()
        
        # Set bypass flag for this validation run
        original_bypass_state = getattr(model, 'bypass_mlp_for_inference', False)
        model.bypass_mlp_for_inference = bypass_mlp
        
        # Get symbols for validation using SymbolManager methods
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
        
        # Run validation using utils.evaluation_utils
        try:
            with torch.no_grad():
                metrics = self._run_validation_with_utils(
                    model=model,
                    val_dataloader=val_dataloader,
                    symbol_mappings=symbol_mappings,
                    mode_name=full_mode_name,
                    epoch=epoch,
                    phase=phase,
                    use_original_labels=use_original_labels,
                    use_dynamic_symbols=use_dynamic_symbols
                )
            
            logging.info(f"✓ {full_mode_name} Validation Score: {metrics['accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            logging.error(f"❌ Validation failed for {full_mode_name}: {str(e)}")
            return {"accuracy": 0.0, "loss": float('inf'), "total_samples": 0}
        
        finally:
            # Restore original bypass state
            model.bypass_mlp_for_inference = original_bypass_state
            model.train()  # Reset to training mode

    def _run_validation_with_utils(
        self,
        model,
        val_dataloader,
        symbol_mappings: Dict[str, str],
        mode_name: str,
        epoch: int,
        phase: str,
        use_original_labels: bool = False,
        use_dynamic_symbols: bool = False
    ) -> Dict[str, float]:
        """Run validation using utils.evaluation_utils functions"""
        
        all_results = {}
        processed_samples = 0
        
        # Initialize results dict based on dataset types
        dataset_names = [self.config.data_config.dataset_type.value] if hasattr(self.config.data_config.dataset_type, 'value') else [str(self.config.data_config.dataset_type)]
        for dataset_name in dataset_names:
            all_results[dataset_name] = []
        
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
                
                try:
                    # Apply symbol replacement using SymbolManager methods
                    if use_original_labels:
                        # Use original batch without symbol replacement
                        updated_batch = batch
                    elif use_dynamic_symbols:
                        # For dynamic symbols, use the fresh mappings directly
                        updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, mappings=symbol_mappings)
                    else:
                        # Use SymbolManager's method with the epoch
                        updated_batch = self.symbol_manager.replace_symbols_in_batch(batch, epoch=epoch)
                    
                    # Generate outputs using the model's generate_output method
                    predictions = model.generate_output(updated_batch)
                    
                    # Process results using SymbolManager and evaluation_utils
                    for i, pred in enumerate(predictions):
                        dt = batch["dataset_type"][i] if isinstance(batch["dataset_type"], list) else batch["dataset_type"]
                        dt_key = dt.value if hasattr(dt, 'value') else str(dt)
                        true_label = batch["completion"][i] if isinstance(batch["completion"], list) else batch["completion"]
                        
                        # Convert symbols back to original labels using SymbolManager
                        converted_pred = pred
                        if not use_original_labels and symbol_mappings:
                            if use_dynamic_symbols:
                                # For fresh symbols, use the fresh mappings directly
                                converted_pred = self.symbol_manager.convert_symbols_back(pred, mappings=symbol_mappings)
                            else:
                                # For training symbols, use SymbolManager's method with epoch
                                converted_pred = self.symbol_manager.convert_symbols_back(pred, epoch=epoch)
                        
                        # Clean the prediction using utils.evaluation_utils
                        try:
                            dataset_type = DatasetType(dt_key)
                            cleaned_pred = clean_prediction(converted_pred, dataset_type)
                        except:
                            cleaned_pred = converted_pred.strip()
                        
                        result = {
                            "text": batch["text"][i] if isinstance(batch["text"], list) else batch["text"],
                            "true_label": true_label,
                            "predicted_label": str(cleaned_pred).strip(),
                            "dataset_type": dt_key
                        }
                        
                        # Add to results
                        if dt_key in all_results:
                            all_results[dt_key].append(result)
                        
                        processed_samples += 1
                        
                        # Log first few samples for debugging
                        if batch_idx < 10 and i < 2:
                            logging.info(f"Original pred: {pred}")
                            logging.info(f"Converted pred: {converted_pred}")
                            logging.info(f"Cleaned pred: {cleaned_pred}")
                            logging.info(f"True label: {true_label}")
                            logging.info(f"Dataset type: {dt_key}")
                            logging.info("=" * 50)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'samples': processed_samples
                    })
                        
                except Exception as e:
                    logging.error(f"Error during validation batch {batch_idx}: {str(e)}")
                    continue
                    
        except KeyboardInterrupt:
            logging.info(f"Validation interrupted at {processed_samples} samples")
        
        finally:
            progress_bar.close()
        
        # Calculate metrics using utils.evaluation_utils.evaluate_predictions
        main_metric_value = 0.0
        
        for dataset_name in dataset_names:
            dt_results = all_results.get(dataset_name, [])
            
            if dt_results:
                try:
                    dataset_type = DatasetType(dataset_name)
                    dt_metrics = evaluate_predictions(dt_results, dataset_type)
                    
                    # Log metrics
                    logging.info(f"Metrics for {dataset_name} ({mode_name}):")
                    for metric, value in dt_metrics.items():
                        if isinstance(value, (float, int)):
                            logging.info(f"  {metric}: {value:.4f}")
                        else:
                            logging.info(f"  {metric}: {value}")
                    
                    # Extract main metric for return value (same logic as unified_symbol_training.py)
                    if dataset_name.lower() == 'voxceleb':
                        main_metric_value = dt_metrics.get('macro_f1_with_invalid', 0.0)
                    elif dataset_name.lower() == 'hvb':
                        main_metric_value = dt_metrics.get('macro_f1', 0.0)
                    elif dataset_name.lower() == 'meld_emotion':
                        main_metric_value = dt_metrics.get('macro_f1_filtered', 0.0)
                    else:
                        main_metric_value = dt_metrics.get('macro_f1', 0.0)
                        
                except Exception as e:
                    logging.error(f"Error evaluating predictions for {dataset_name}: {str(e)}")
                    main_metric_value = 0.0
        
        return {
            "accuracy": main_metric_value,
            "loss": 0.0,  # Not calculated in this implementation
            "total_samples": len(all_results.get(dataset_names[0], []))
        }

    # Keep all the existing methods unchanged
    def run_comprehensive_validation(
        self,
        model,
        val_dataloader,
        epoch: int,
        phase: str,  # This comes from TrainingStep.phase: "lora", "mlp", "joint"
        cycle: int = 0,
        step: Optional[TrainingStep] = None  # Add step parameter
    ) -> Dict[str, float]:
        """
        Run comprehensive validation with all modes (MLP+Symbols, NoMLP+Symbols, etc.)
        """
        validation_results = {}
        
        # Check model state
        bypass_mlp = getattr(model, 'bypass_mlp', False)
        use_symbols = step.use_symbols if step else True  # Default to True if no step provided
        
        # Determine validation modes based on phase + model state + step config
        if phase == "lora":
            if bypass_mlp and use_symbols:
                # LoRA training with bypass_mlp=True and symbols
                modes = [
                    ("no_mlp_symbols", True, False, False),   # NoMLP + Fixed Symbols (from training)
                    ("no_mlp_fresh", True, False, True),      # NoMLP + Fresh Symbols
                    ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
                ]
            elif bypass_mlp and not use_symbols:
                # LoRA training with bypass_mlp=True and no symbols
                modes = [
                    ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
                ]
            elif not bypass_mlp and use_symbols:
                # LoRA training with MLP enabled and symbols
                modes = [
                    ("mlp_symbols", False, False, False),     # MLP + Fixed Symbols
                    ("no_mlp_symbols", True, False, False),   # NoMLP + Fixed Symbols
                    ("mlp_original", False, True, False),     # MLP + Original Labels
                    ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
                    ("mlp_fresh", False, False, True),     # MLP + Fresh Labels
                    ("no_mlp_fresh", True, False, True),   # NoMLP + Fresh Labels

                ]
            else:
                # LoRA training with MLP enabled and no symbols
                modes = [
                    ("mlp_original", False, True, False),     # MLP + Original Labels
                    ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
                ]
        
        elif phase == "mlp":
            if bypass_mlp:
                # This should not happen - MLP training with bypass_mlp=True
                logging.error("Cannot run MLP validation when bypass_mlp=True")
                return {"error": 0.0}
            else:
                # Normal MLP training - always uses symbols
                modes = [
                    ("mlp_symbols", False, False, False),     # MLP + Fixed Symbols (main focus)
                    ("no_mlp_symbols", True, False, False),   # NoMLP + Fixed Symbols (comparison)
                    ("mlp_original", False, True, False),     # MLP + Original Labels (baseline)
                    ("no_mlp_original", True, True, False),   # NoMLP + Original Labels (baseline)
                    ("mlp_fresh", False, False, True),     # MLP + Fresh Labels
                    ("no_mlp_fresh", True, False, True),   # NoMLP + Fresh Labels
                ]
        
        elif phase == "joint":
            if bypass_mlp and use_symbols:
                # Joint training with bypass_mlp=True (effectively LoRA-only)
                modes = [
                    ("no_mlp_symbols", True, False, False),   # NoMLP + Fixed Symbols
                    ("no_mlp_fresh", True, False, True),      # NoMLP + Fresh Symbols
                    ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
                ]
            elif bypass_mlp and not use_symbols:
                # Joint training with bypass_mlp=True and no symbols
                modes = [
                    ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
                ]
            elif not bypass_mlp and use_symbols:
                # Normal joint training with symbols
                modes = [
                    ("mlp_symbols", False, False, False),     # MLP + Fixed Symbols
                    ("no_mlp_symbols", True, False, False),   # NoMLP + Fixed Symbols
                    ("mlp_original", False, True, False),     # MLP + Original Labels
                    ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
                    ("mlp_fresh", False, False, True),     # MLP + Fresh Labels
                    ("no_mlp_fresh", True, False, True),   # NoMLP + Fresh Labels
                ]
            else:
                # Joint training without symbols
                modes = [
                    ("mlp_original", False, True, False),     # MLP + Original Labels
                    ("no_mlp_original", True, True, False),   # NoMLP + Original Labels
                ]
        
        else:
            # Default fallback
            modes = [
                ("mlp_symbols", False, False, False),
                ("no_mlp_symbols", True, False, False),
                ("mlp_original", False, True, False),
                ("no_mlp_original", True, True, False),
            ]
        
        logging.info(f"Validation modes for {phase.upper()} (bypass_mlp={bypass_mlp}, use_symbols={use_symbols}):")
        for mode_name, bypass, orig, fresh in modes:
            logging.info(f"  - {mode_name}")
        
        # Run each validation mode
        for mode_key, bypass_mlp_val, use_original, use_dynamic in modes:
            try:
                metrics = self.validate_model(
                    model=model,
                    val_dataloader=val_dataloader,
                    epoch=epoch,
                    phase=phase,
                    cycle=cycle,
                    bypass_mlp=bypass_mlp_val,
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


