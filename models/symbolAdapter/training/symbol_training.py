"""
Symbol Training Orchestrator
Main training pipeline that coordinates MLP, LoRA, and Joint trainers
with symbol management and comprehensive validation
"""

import logging
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import os
import json
from datetime import datetime

from ..symbol_manager import SymbolManager
from ..configs.training_configs import TrainingConfig, SymbolMode
from .validation import ValidationManager
from .schedulers import TrainingScheduler, TrainingStep
from .unified_trainer import UnifiedTrainer


class SymbolTrainingOrchestrator:
    """
    Main orchestrator for Symbol Adapter training pipeline
    Manages the complete training process with multiple cycles and phases
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tokenizer=None,
        symbol_manager=None  # NEW: Accept existing symbol_manager
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        
        # Use provided symbol_manager or create new one
        self.symbol_manager = symbol_manager
        logging.info("Using provided SymbolManager")

        
        # Rest of initialization...
        self.scheduler = TrainingScheduler(config)
        self.trainer = UnifiedTrainer(
            config=config,
            symbol_manager=symbol_manager,
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            tokenizer=self.tokenizer
        )
        
        self.validator = ValidationManager(
            config=config,
            symbol_manager=self.symbol_manager,
            tokenizer=tokenizer,
            max_val_samples=config.data_config.val_max_samples
        )
        
        # Training state
        self.current_cycle = 0
        self.current_step = 0
        self.training_history = []
        self.best_scores = {}
        
        # Add summary tracking
        self.training_summary = []
        self.best_scores = {
            'MLP+Symbols': {'score': 0.0, 'step': None},
            'NoMLP+Symbols': {'score': 0.0, 'step': None}, 
            'MLP+Original': {'score': 0.0, 'step': None},
            'NoMLP+Original': {'score': 0.0, 'step': None}
        }
        
        # Setup logging and output directories
        self._setup_training_environment()
    
    
    def _setup_training_environment(self):
        """Setup logging and output directories"""
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "logs"), exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.config.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logging.info(f"Training environment setup complete")
        logging.info(f"Output directory: {self.config.output_dir}")
    
    def run_complete_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        logging.info("🚀 Starting Complete Symbol Training Pipeline")
        
        # ✅ GIVE TRAINER ACCESS TO ORCHESTRATOR FOR EPOCH TRACKING
        self.trainer.orchestrator = self
        
        # Get training schedule
        training_steps = self.scheduler.generate_schedule()
        
        current_cycle = -1
        
        for step_idx, step in enumerate(training_steps):
            # Log cycle summary when cycle changes
            if step.cycle != current_cycle and current_cycle >= 0:
                self._log_cycle_summary(current_cycle)
            current_cycle = step.cycle
            
            # Execute training step with unified trainer
            validation_scores = self.trainer.train_step(step)
            
            # Track summary data - this will be replaced by epoch tracking
            # self._track_step_summary(step, validation_scores)  # COMMENTED OUT
        
        # Log final summaries
        self._log_cycle_summary(current_cycle)
        self._log_final_summary()
        
        return self.best_scores
    
    def _execute_training_step(self, step: TrainingStep) -> Dict[str, Any]:
        """Execute a single training step"""
        
        # Update symbol manager for this step
        if step.cycle != self.current_cycle:
            self.current_cycle = step.cycle
            logging.info(f"Starting new cycle {step.cycle}")
        
        # Route to appropriate trainer based on phase
        if step.phase == "mlp":
            logging.info("🔧 Executing MLP Training Step")
            return self.mlp_trainer.train_mlp_step(step)
        
        elif step.phase == "lora":
            logging.info("🔗 Executing LoRA Training Step")
            return self.lora_trainer.train_lora_step(step)
        
        elif step.phase == "bypass_mlp_sym":
            logging.info("🔄 Executing Dynamic LoRA Training Step (Bypass MLP)")
            return self.lora_trainer.train_lora_step(step)
        
        elif step.phase == "bypass_mlp_org":
            logging.info("🚫 Executing LoRA Training Step (No Symbols)")
            step.use_symbols = False
            return self.lora_trainer.train_lora_step(step)
        
        elif step.phase == "joint":
            logging.info("🤝 Executing Joint MLP+LoRA Training Step")
            return self.joint_trainer.train_joint_step(step)
        
        else:
            raise ValueError(f"Unknown training phase: {step.phase}")
    
    def _log_training_configuration(self):
        """Log the complete training configuration"""
        logging.info("TRAINING CONFIGURATION:")
        logging.info(f"  Dataset: {self.config.data_config.dataset_type}")
        logging.info(f"  Device: {self.config.device}")
        logging.info(f"  Cycles: {self.config.total_cycles}")
        logging.info(f"  Symbol Mode: {self.config.symbol_config.mode}")
        logging.info(f"  Output Dir: {self.config.output_dir}")
        
        # Model configuration
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"  Model Parameters: {total_params:,}")
        
        # Data configuration
        logging.info(f"  Train Batches: {len(self.train_dataloader)}")
        logging.info(f"  Val Batches: {len(self.val_dataloader)}")
        
        logging.info("=" * 80)
    
    def _summarize_cycle(self, cycle: int, training_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize results for a completed cycle"""
        cycle_steps = [step for step in training_steps if step.get("step_info", {}).get("cycle") == cycle]
        
        if not cycle_steps:
            return {"cycle": cycle, "steps": 0, "best_score": 0.0}
        
        # Find best score in this cycle
        best_score = max(step.get("best_score", 0) for step in cycle_steps)
        best_step = max(cycle_steps, key=lambda x: x.get("best_score", 0))
        
        cycle_summary = {
            "cycle": cycle,
            "steps": len(cycle_steps),
            "best_score": best_score,
            "best_step": best_step.get("step_info", {}),
            "phases": list(set(step.get("step_info", {}).get("phase") for step in cycle_steps))
        }
        
        logging.info(f"\n📊 CYCLE {cycle} SUMMARY:")
        logging.info(f"  Steps completed: {len(cycle_steps)}")
        logging.info(f"  Phases executed: {', '.join(cycle_summary['phases'])}")
        logging.info(f"  Best score: {best_score:.4f}")
        logging.info(f"  Best phase: {best_step.get('step_info', {}).get('phase', 'Unknown')}")
        
        return cycle_summary
    
    def _log_final_summary(self):
        """Log complete training summary - DYNAMIC columns with EPOCHS"""
        logging.info("=" * 140)
        logging.info("COMPLETE TRAINING SUMMARY - ALL EPOCHS")
        logging.info("=" * 140)
        
        # ✅ DYNAMIC: Determine which columns to show
        all_score_keys = set()
        for entry in self.training_summary:
            for key in entry.keys():
                if key.endswith(('_sym', '_orig', '_fresh')) and entry[key] is not None:
                    all_score_keys.add(key)
        
        # Create header with EPOCH support
        header_parts = ["Phase", "Cycle", "Step", "Epoch"]  # ✅ CHANGE TO EPOCH
        column_mapping = {
            'mlp_sym': 'MLP+Sym',
            'nomlp_sym': 'NoMLP+Sym', 
            'mlp_orig': 'MLP+Orig',
            'nomlp_orig': 'NoMLP+Orig',
            'nomlp_fresh': 'NoMLP+Fresh'
        }
        
        for key in sorted(all_score_keys):
            header_parts.append(column_mapping.get(key, key))
        
        header = "  ".join(f"{h:<12}" for h in header_parts)
        logging.info(header)
        logging.info("-" * len(header))
        
        # Log data rows - ONE ROW PER EPOCH
        for entry in self.training_summary:
            row_parts = [
                entry['phase'][:6], 
                str(entry['cycle']), 
                str(entry['step']),
                str(entry['epoch'])  # ✅ SHOW EPOCH NUMBER
            ]
            
            for key in sorted(all_score_keys):
                value = entry.get(key)
                if value is not None:
                    row_parts.append(f"{value:.4f}")
                else:
                    row_parts.append("N/A")
            
            row = "  ".join(f"{r:<12}" for r in row_parts)
            logging.info(row)
        
        logging.info("=" * 140)
        
        # ✅ DYNAMIC: Best scores based on what actually exists
        logging.info("BEST PERFORMING CONFIGURATIONS:")
        for config_name, best_info in self.best_scores.items():
            if best_info and best_info.get('score', 0) > 0:
                logging.info(f"Best {config_name}: {best_info['step']} = {best_info['score']:.4f}")
        logging.info("=" * 140)
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save complete training results to file"""
        results_path = os.path.join(self.config.output_dir, "training_results.json")
        
        # Convert any non-serializable objects
        serializable_results = self._make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logging.info(f"Training results saved to: {results_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def run_validation_only(self, epoch: int = 0, phase: str = "joint") -> Dict[str, float]:
        """
        Run only validation without training
        Useful for evaluating pre-trained models
        """
        logging.info(f"🔍 Running validation-only mode")
        
        validation_results = self.validator.run_comprehensive_validation(
            model=self.model,
            val_dataloader=self.val_dataloader,
            epoch=epoch,
            phase=phase,
            cycle=0
        )
        
        self.validator.log_validation_summary(
            validation_results=validation_results,
            epoch=epoch,
            phase=phase,
            cycle=0
        )
        
        return validation_results
    
    def run_single_step(self, step: TrainingStep) -> Dict[str, Any]:
        """
        Run a single training step
        Useful for debugging or custom training schedules
        """
        logging.info(f"🎯 Running single training step: {step.description}")
        
        step_results = self._execute_training_step(step)
        
        logging.info(f"✅ Single step completed")
        logging.info(f"Best score: {step_results.get('best_score', 0):.4f}")
        
        return step_results
    
    def continue_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Continue training from a saved checkpoint
        """
        logging.info(f"📂 Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore training state
        step_info = checkpoint.get("step_info", {})
        self.current_cycle = step_info.get("cycle", 0)
        
        # Load symbol mappings
        if "symbol_mappings" in checkpoint:
            # Update symbol manager with saved mappings
            epoch = checkpoint.get("epoch", 0)
            self.symbol_manager.fixed_mappings = checkpoint["symbol_mappings"]
        
        logging.info(f"✅ Checkpoint loaded successfully")
        logging.info(f"Resuming from cycle {self.current_cycle}")
        
        # Continue with remaining training
        return self.run_complete_training()
    
    def _log_epoch_summary(self, step: TrainingStep, epoch: int, epoch_loss: float, validation_scores: dict):
        """Log and track epoch summary"""
        
        # Extract validation scores (assuming these keys exist)
        mlp_sym = validation_scores.get('MLP+Symbols', 0.0)
        nomlp_sym = validation_scores.get('NoMLP+Symbols', 0.0) 
        mlp_orig = validation_scores.get('MLP+Original', 0.0)
        nomlp_orig = validation_scores.get('NoMLP+Original', 0.0)
        
        # Add to summary
        summary_entry = {
            'phase': step.phase.upper(),
            'cycle': step.cycle,
            'epoch': epoch,
            'loss': epoch_loss,
            'mlp_sym': mlp_sym,
            'nomlp_sym': nomlp_sym,
            'mlp_orig': mlp_orig,
            'nomlp_orig': nomlp_orig,
            'step_description': step.description
        }
        self.training_summary.append(summary_entry)
        
        # Update best scores
        self._update_best_scores(summary_entry)
        
        # Log epoch summary
        logging.info("=" * 100)
        logging.info(f"EPOCH SUMMARY - {step.phase.upper()} Cycle {step.cycle} Epoch {epoch}")
        logging.info("=" * 100)
        logging.info(f"Phase: {step.phase.upper():<6} | Cycle: {step.cycle:<2} | Epoch: {epoch:<2} | Loss: {epoch_loss:.4f}")
        logging.info(f"MLP+Sym: {mlp_sym:.4f} | NoMLP+Sym: {nomlp_sym:.4f} | MLP+Orig: {mlp_orig:.4f} | NoMLP+Orig: {nomlp_orig:.4f}")
        logging.info("=" * 100)
    
    def _update_best_scores(self, entry):
        """Update best scores tracking with FRESH support"""
        configs = {
            'MLP+Symbols': entry.get('mlp_sym'),
            'NoMLP+Symbols': entry.get('nomlp_sym'), 
            'MLP+Original': entry.get('mlp_orig'),
            'NoMLP+Original': entry.get('nomlp_orig'),
            'MLP+Fresh': entry.get('mlp_fresh'),        # ✅ ADD FRESH SYMBOLS
            'NoMLP+Fresh': entry.get('nomlp_fresh')     # ✅ ADD FRESH SYMBOLS
        }
        
        for config_name, score in configs.items():
            if score is not None and score > self.best_scores.get(config_name, {}).get('score', 0):
                if config_name not in self.best_scores:
                    self.best_scores[config_name] = {'score': 0.0, 'step': None}
                self.best_scores[config_name]['score'] = score
                self.best_scores[config_name]['step'] = f"{entry['phase']} Cycle {entry['cycle']} Step {entry['step']}"
    
    def _log_cycle_summary(self, cycle: int):
        """Log summary at end of each cycle - DYNAMIC columns with FRESH"""
        cycle_entries = [e for e in self.training_summary if e['cycle'] == cycle]
        
        if not cycle_entries:
            return
            
        logging.info("=" * 140)
        logging.info(f"CYCLE {cycle} SUMMARY")
        logging.info("=" * 140)
        
        # ✅ DYNAMIC: Determine which columns to show based on actual data
        all_score_keys = set()
        for entry in cycle_entries:
            for key in entry.keys():
                if key.endswith(('_sym', '_orig', '_fresh')) and entry[key] is not None:
                    all_score_keys.add(key)
        
        # Create header with FRESH support
        header_parts = ["Phase", "Step"]
        column_mapping = {
            'mlp_sym': 'MLP+Sym',
            'nomlp_sym': 'NoMLP+Sym',
            'mlp_orig': 'MLP+Orig',
            'nomlp_orig': 'NoMLP+Orig',
            'mlp_fresh': 'MLP+Fresh',      # ✅ ADD FRESH SYMBOLS
            'nomlp_fresh': 'NoMLP+Fresh'   # ✅ ADD FRESH SYMBOLS
        }
        
        for key in sorted(all_score_keys):
            header_parts.append(column_mapping.get(key, key))
        
        header = "  ".join(f"{h:<12}" for h in header_parts)
        logging.info(header)
        logging.info("-" * len(header))
        
        # Log data rows
        for entry in cycle_entries:
            row_parts = [entry['phase'][:6], str(entry['step'])]
            
            for key in sorted(all_score_keys):
                value = entry.get(key)
                if value is not None:
                    row_parts.append(f"{value:.4f}")
                else:
                    row_parts.append("N/A")
            
            row = "  ".join(f"{r:<12}" for r in row_parts)
            logging.info(row)
        
        logging.info("=" * 140)

    def _track_epoch_summary(self, step: TrainingStep, epoch: int, validation_scores: dict):
        """Track summary data for each EPOCH"""
        summary_entry = {
            'phase': step.phase.upper(),
            'cycle': step.cycle,
            'step': step.step_id,
            'epoch': epoch,  # ✅ ADD EPOCH TRACKING
            'epochs_total': step.epochs,
        }
        
        # Add scores dynamically
        possible_scores = {
            'MLP+Symbols': 'mlp_sym',
            'NoMLP+Symbols': 'nomlp_sym',
            'MLP+Original': 'mlp_orig', 
            'NoMLP+Original': 'nomlp_orig',
            'NoMLP+Fresh': 'nomlp_fresh'
        }
        
        for display_key, entry_key in possible_scores.items():
            summary_entry[entry_key] = validation_scores.get(display_key, None)
        
        self.training_summary.append(summary_entry)
        
        # Update best scores
        self._update_best_scores(summary_entry)
