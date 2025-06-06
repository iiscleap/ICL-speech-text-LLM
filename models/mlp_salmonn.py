import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import nullcontext

# Import PEFT for LoRA
from peft import LoraConfig, TaskType, get_peft_model
from transformers import WhisperFeatureExtractor

# Path setup for SALMONN import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from SALMONN.models.salmonn_org import SALMONN

class MLPSalmonn(nn.Module):
    """SIMPLIFIED SALMONN with MLP - compatible with unified_symbol_training.py"""
    
    def __init__(
        self,
        llama_path="lmsys/vicuna-13b-v1.1",
        whisper_path="openai/whisper-large-v2",
        beats_path="/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        label_tokens=None,
        hidden_dim=None,
        dropout=0.1,
        freeze_base=True,
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        device=None,
        low_resource=False
    ):
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # SIMPLIFIED SALMONN config
        salmonn_config = {
            "llama_path": llama_path,
            "whisper_path": whisper_path,
            "beats_path": beats_path,
            "lora": True,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "low_resource": False,
            "use_speech_Qformer": True,
            "freeze_whisper": True,
            "freeze_beats": True,
            "freeze_speech_QFormer": False,
            "num_speech_query_token": 1,
            "window_level_Qformer": True,
            "second_per_window": 0.333333,
            "second_stride": 0.333333,
            "speech_llama_proj_model": "",
            "freeze_speech_llama_proj": False,
            "ckpt": "/data2/neeraja/neeraja/salmonn_v1.pth"
        }

        # Initialize SALMONN
        logging.info("Loading base SALMONN model...")
        self.salmonn = SALMONN.from_config(salmonn_config)
        logging.info("Base SALMONN model loaded successfully")
        self.salmonn.to(self.device)
        
        # SIMPLIFIED attributes
        self.label_tokens = label_tokens or []
        self.label_token_ids = []
        self.batch_counter = 0
        self.speech_placeholder = "<SpeechHere>"
        self.bypass_mlp_during_lora = False
        
        # ✅ FIX: Store original label mapping for discovery
        self.original_to_random_mapping = {}  # Will store {"alpha": "duhl"}
        
        # Get model components
        self.llama_model = self.salmonn.llama_model
        self.llama_tokenizer = self.salmonn.llama_tokenizer
        
        # Get embedding module
        if hasattr(self.llama_model, 'model'):
            if hasattr(self.llama_model.model, 'model'):
                self.embed_module = self.llama_model.model.model.embed_tokens
            else:
                self.embed_module = self.llama_model.model.embed_tokens
        else:
            self.embed_module = self.llama_model.embed_tokens
        
        self.embed_dim = self.embed_module.weight.shape[1]
        self.hidden_dim = hidden_dim or self.embed_dim
        
        # SIMPLIFIED MLP
        self.position_wise_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  # ✅ Add layer norm for stability
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        # SIMPLIFIED initialization
        with torch.no_grad():
            for layer in self.position_wise_mlp:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    torch.nn.init.constant_(layer.weight, 1.0)
                    torch.nn.init.constant_(layer.bias, 0.0)
        
        self.position_wise_mlp.to(self.device)
        self.input_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        
        # Convert label tokens to token IDs
        if self.label_tokens:
            self.label_token_ids = self._get_label_token_ids()
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.salmonn.parameters():
                param.requires_grad = False
            for param in self.position_wise_mlp.parameters():
                param.requires_grad = True
        
        # SIMPLIFIED attributes for compatibility
        self.original_embed_matrix = self.embed_module.weight.clone().detach()
        self.original_embed_matrix.requires_grad = False
        self.symbol_mappings = {}
    
    
    def apply_mlp_to_embeddings(self, embeddings, token_ids, scale_factor=0.2, temperature=0.1):
        # ✅ EMERGENCY FIX: Force dtype consistency
        target_dtype = embeddings.dtype
        target_device = embeddings.device
        
        # Ensure MLP matches input dtype
        if next(self.position_wise_mlp.parameters()).dtype != target_dtype:
            logging.info(f"Converting MLP from {next(self.position_wise_mlp.parameters()).dtype} to {target_dtype}")
            self.position_wise_mlp = self.position_wise_mlp.to(dtype=target_dtype)
        
        # Ensure all computations use the same dtype
        positions_to_transform = []
        for label_token_id in self.label_token_ids:
            positions = (token_ids == label_token_id).nonzero(as_tuple=True)[0]
            positions_to_transform.extend(positions.tolist())
        
        if not positions_to_transform:
            return embeddings
        
        positions_tensor = torch.tensor(positions_to_transform, device=target_device)
        to_transform = embeddings[positions_tensor].to(dtype=target_dtype)
        
        # Apply MLP with error handling
        try:
            with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for this operation
                mlp_output = self.position_wise_mlp(to_transform)
                transformed = to_transform + scale_factor * mlp_output
        except Exception as e:
            logging.error(f"MLP transformation failed: {e}")
            return embeddings  # Return original embeddings on failure
        
        # Continue with quantization...
        try:
            # Normalize transformed embeddings
            transformed_norm = F.normalize(transformed, p=2, dim=-1)
            
            # Get vocabulary embeddings (normalized)
            vocab_embeds = self.embed_module.weight
            vocab_norm = F.normalize(vocab_embeds, p=2, dim=-1)
            
            # Compute cosine similarities
            similarities = torch.mm(transformed_norm, vocab_norm.t())
            
            if self.training:
                # ✅ TRAINING: Use soft quantization (differentiable)
                soft_weights = F.softmax(similarities / temperature, dim=-1)
                final_embeddings = torch.mm(soft_weights, vocab_embeds)
                
                # if self.batch_counter < 3:
                logging.info(f"Training mode: Using soft quantization (temp={temperature})")
            else:
                # ✅ INFERENCE: Use hard quantization (argmax)
                hard_indices = torch.argmax(similarities, dim=-1)  # [num_positions]
                final_embeddings = vocab_embeds[hard_indices]  # Direct lookup
                
                if self.batch_counter < 3:
                    logging.info(f"Inference mode: Using hard quantization (argmax)")
                    # Log the discovered tokens
                    for i, pos_idx in enumerate(positions_to_transform[:2]):
                        orig_token_id = token_ids[positions_tensor[i]].item()
                        discovered_token_id = hard_indices[i].item()
                        orig_text = self.llama_tokenizer.decode([orig_token_id], skip_special_tokens=True)
                        discovered_text = self.llama_tokenizer.decode([discovered_token_id], skip_special_tokens=True)
                        similarity = similarities[i, discovered_token_id].item()
                        logging.info(f"Hard quantization: {orig_token_id}('{orig_text}') -> {discovered_token_id}('{discovered_text}') [sim: {similarity:.4f}]")
            
            # Update embeddings
            output_embeds = embeddings.clone()
            if len(embeddings.shape) == 2:
                output_embeds[positions_tensor] = final_embeddings
            else:
                batch_indices = torch.zeros_like(positions_tensor)
                output_embeds[batch_indices, positions_tensor] = final_embeddings
            
            return output_embeds
            
        except Exception as e:
            logging.error(f"Error in MLP transformation: {e}")
            return embeddings
    
    def forward(self, samples):
        """Forward pass - reduced logging"""
        # Process speech embeddings
        speech_embeds, speech_atts, example_embeds, example_atts = self.get_speech_embeddings(samples)
        
        # Get number of examples
        num_examples = samples.get("num_examples", torch.zeros(len(samples["prompt"]), dtype=torch.long))
        
        # Wrap embeddings with prompts
        wrapped_embeds, wrapped_atts = self.custom_prompt_wrap(
            speech_embeds, speech_atts, samples["prompt"],
            num_examples, example_embeds, example_atts
        )
        
        # Process target text
        target_tokens = self.llama_tokenizer(
            samples["completion"],
            padding='longest',
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
        ).to(wrapped_embeds.device)
        
        # Get target embeddings
        target_embeds = self.embed_module(target_tokens.input_ids)
        
        # Create labels
        prompt_length = wrapped_embeds.size(1)
        labels = torch.full(
            [wrapped_atts.shape[0], wrapped_atts.shape[1] + target_tokens.input_ids.size(1)],
            fill_value=-100,
            dtype=torch.long,
            device=wrapped_embeds.device
        )
        labels[:, prompt_length:] = target_tokens.input_ids
        
        # Create attention mask
        attention_mask = torch.cat([wrapped_atts, target_tokens.attention_mask], dim=1)
        labels[attention_mask == 0] = -100
        
        # Forward through LLaMA
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=torch.cat([wrapped_embeds, target_embeds], dim=1),
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        
        self.batch_counter += 1
        return {"loss": outputs.loss, "logits": outputs.logits, "labels": labels}
    
    def get_speech_embeddings(self, samples):
        """
        Processes speech inputs to generate embeddings.
        Implementation from CustomSALMONN.
        """
        start_time = time.time()

        # Check if this is SQA dataset with question and document audio
        is_sqa = (
            "question_spectrogram" in samples and 
            "document_spectrogram" in samples
        )

        if is_sqa:
            # Process question and document audio
            has_main_speech = (
                samples["question_spectrogram"] is not None and 
                samples["document_spectrogram"] is not None
            )
            has_examples = (
                "example_question_spectrograms" in samples and 
                "example_document_spectrograms" in samples and 
                samples["example_question_spectrograms"] is not None and 
                samples["example_document_spectrograms"] is not None
            )
        else:
            # Original logic for other datasets
            has_main_speech = "spectrogram" in samples and samples["spectrogram"] is not None
            has_examples = "example_spectrograms" in samples and samples["example_spectrograms"] is not None

        # Add detailed logging for first batch
        if self.batch_counter == 0 and not is_sqa:
            if has_main_speech:
                logging.info("=== Input Data Debug (First 5 values) ===")
                logging.info(f"Spectrogram dtype: {samples['spectrogram'].dtype}")
                logging.info("Spectrogram first 5 values:")
                spec_flat = samples["spectrogram"][0].flatten()  # First batch item
                logging.info(f"{spec_flat[:10].tolist()}")
                logging.info(f"{spec_flat[-10:].tolist()}")
                
                if samples.get("raw_wav") is not None:
                    logging.info(f"\nRaw WAV dtype: {samples['raw_wav'].dtype}")
                    logging.info("Raw WAV first 5 values:")
                    wav_flat = samples["raw_wav"][0].flatten()  # First batch item
                    logging.info(f"{wav_flat[:10].tolist()}")
                    logging.info(f"{wav_flat[-10:].tolist()}")

                if "padding_mask" in samples:
                    logging.info("=== Padding Mask Debug ===")
                    padding_mask = samples["padding_mask"]
                    logging.info(f"Padding mask dtype: {padding_mask.dtype}")
                    logging.info(f"Padding mask shape: {padding_mask.shape}")
                    logging.info(f"Padding mask first 5 values: {padding_mask[0, :5].tolist()}")
                    logging.info(f"Non-padded length: {padding_mask.sum().item()}")
                    logging.info(f"Padding percentage: {(1 - padding_mask.float().mean()) * 100:.2f}%")

        if self.batch_counter == 0:
            if has_main_speech:
                logging.info("Valid main speech data detected")
            else:
                logging.info("No valid main speech data detected")
            
            if has_examples:
                logging.info("Valid speech examples detected")
            else:
                logging.info("No valid speech examples detected")
        
        def to_device_and_dtype(tensor):
            if tensor is None:
                return None
            tensor = tensor.to(device=self.device)
            if hasattr(self, 'use_fp16') and self.use_fp16 and tensor.dtype == torch.float32:
                tensor = tensor.to(dtype=torch.float16)
            return tensor

        # Process main inputs
        speech_embeds = speech_atts = None
        if has_main_speech:

            # Standard speech processing
            spectrograms = to_device_and_dtype(samples["spectrogram"])
            padding_mask = to_device_and_dtype(samples.get("padding_mask"))
            raw_wav = to_device_and_dtype(samples.get("raw_wav"))
            
            speech_embeds, speech_atts = self.encode_speech(spectrograms, raw_wav, padding_mask)
    
        # Process examples
        example_embeds = example_atts = None
        if has_examples:
            # Standard example processing
            example_specs = to_device_and_dtype(samples["example_spectrograms"])
            example_padding = to_device_and_dtype(samples.get("example_padding_masks"))
            example_raw = to_device_and_dtype(samples.get("example_raw_wavs"))
            
            # Initialize lists to store example embeddings
            batch_example_embeds = []
            batch_example_atts = []
            
            # Process each batch
            for b in range(len(example_specs)):
                example_count = len(example_specs[b])
                batch_embeds = []
                batch_atts = []
                
                # Process each example in this batch
                for i in range(example_count):
                    spec = example_specs[b][i]
                    
                    # Get padding mask and raw wav if available
                    pad = example_padding[b][i] if example_padding is not None else None
                    raw = example_raw[b][i] if example_raw is not None else None
                    
                    # Encode example speech
                    embed, att = self.encode_speech(spec.unsqueeze(0), raw, pad)
                    
                    # Add to batch
                    batch_embeds.append(embed.squeeze(0))
                    batch_atts.append(att.squeeze(0))
                
                batch_example_embeds.append(batch_embeds)
                batch_example_atts.append(batch_atts)
                
                example_embeds = batch_example_embeds
                example_atts = batch_example_atts
        
        logging.info(f"Speech embedding generation took {time.time() - start_time:.2f} seconds")
        
        return speech_embeds, speech_atts, example_embeds, example_atts
        
    def custom_prompt_wrap(self, embeds, atts, prompts, num_examples=None, example_embeds=None, example_atts=None):
        """Wraps speech embeddings with text prompts and examples - reduced logging"""
        # Infer device from model parameters
        device = next(self.parameters()).device
        
        # Add flag for SQA dataset - only check embeds
        is_sqa = isinstance(embeds, tuple)
        
        batch_size = len(prompts)
        max_examples = num_examples.max().item() if num_examples is not None else 0
        batch_embeds, batch_atts = [], []
        
        for b in range(batch_size):
            parts = []
            suffix = prompts[b]
            if max_examples > 0 and example_embeds is not None:

                # Original example handling
                for i in range(max_examples):
                    example_marker = f"<Example{i}>"
                    if example_marker in suffix:
                        before_example, after_example = suffix.split(example_marker, 1)
                        parts.append(before_example)
                        suffix = after_example
                    else:
                        parts.append("")

            if self.speech_placeholder in suffix:
                before_speech, after_speech = suffix.split(self.speech_placeholder)
                parts.append(before_speech)
                suffix = after_speech
            else:
                # For text-only mode, we don't split on speech placeholder
                parts.append(suffix)
                suffix = ""
            parts.append(suffix)
            
            # Process text parts - Ensure everything is on the correct device and dtype
            tokenized_parts = [
                self.llama_tokenizer(part, padding="longest", return_tensors="pt", add_special_tokens=False) 
                for part in parts
            ]
            # Move to device and ensure input_ids are Long
            tokenized_parts = [
                {k: v.to(device=device, dtype=torch.long if k == 'input_ids' else v.dtype) 
                 for k, v in tokens.items()}
                for tokens in tokenized_parts
            ]
            
            # Get embeddings from the original embedding layer
            part_embeds = []
            for tokens in tokenized_parts:
                part_embed = self.embed_module(tokens['input_ids'].squeeze(0))

                # Only log for first few batches
                if self.batch_counter < 3:
                    part_text = self.llama_tokenizer.decode(tokens['input_ids'].squeeze(0), skip_special_tokens=True)
                    logging.info(f"Processing prompt part: '{part_text[:50]}...' ({len(tokens['input_ids'].squeeze(0))} tokens)")
                
                # Apply MLP if training and not bypassed
                if self.training and self.label_token_ids:
                    original_shape = part_embed.shape
                    part_embed = self.apply_mlp_to_embeddings(
                        part_embed, 
                        tokens['input_ids'].squeeze(0)
                    )
                    if self.batch_counter < 3:
                        logging.info(f"Prompt part processed: shape {original_shape} -> {part_embed.shape}")
                
                part_embeds.append(part_embed)
                
            part_atts = [tokens['attention_mask'].squeeze(0) for tokens in tokenized_parts]
            
            # Combine embeddings
            combined_embeds, combined_atts = [], []
            
            

            for i in range(len(part_embeds) - 2):
                combined_embeds.append(part_embeds[i])
                combined_atts.append(part_atts[i])
                
                if i < max_examples and example_embeds is not None:
                    # Add example embeddings if available for this batch and example index
                    if b < len(example_embeds) and i < len(example_embeds[b]):
                        combined_embeds.append(example_embeds[b][i])
                        combined_atts.append(example_atts[b][i])
            


            # Add final parts
            if embeds is not None:  # Speech mode
                combined_embeds.extend([part_embeds[-2], embeds[b], part_embeds[-1]])
                combined_atts.extend([part_atts[-2], atts[b], part_atts[-1]])
            else:  # Text-only mode
                # For text-only, concatenate the final parts without speech embeddings
                if len(part_embeds) >= 2:
                    combined_embeds.extend([part_embeds[-2], part_embeds[-1]])
                    combined_atts.extend([part_atts[-2], part_atts[-1]])
                else:
                    # If there's only one part (e.g., no speech placeholder)
                    combined_embeds.append(part_embeds[-1])
                    combined_atts.append(part_atts[-1])
            
            # Concatenate all parts
            batch_embeds.append(torch.cat(combined_embeds, dim=0))
            batch_atts.append(torch.cat(combined_atts, dim=0))
        
        return torch.stack(batch_embeds, dim=0), torch.stack(batch_atts, dim=0)
    
    # SIMPLIFIED supporting functions
    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        """Encode speech inputs using SALMONN"""
        return self.salmonn.encode_speech(    
            spectrogram=spectrogram,
            raw_wav=raw_wav,
            audio_padding_mask=audio_padding_mask
        )
    
    def _get_label_token_ids(self):
        """SIMPLIFIED: Convert label tokens to token IDs"""
        token_ids = []
        self.label_to_token_ids = {}
        
        for label in self.label_tokens:
            if isinstance(label, int):
                token_ids.append(label)
                self.label_to_token_ids[label] = [label]
            else:
                tokenized = self.llama_tokenizer.encode(label, add_special_tokens=False)
                if len(tokenized) > 0:
                    token_ids.extend(tokenized)
                    self.label_to_token_ids[label] = tokenized
                    logging.info(f"Label '{label}' -> {tokenized}")
        
        return token_ids
    
    def maybe_autocast(self):
        """Context manager for autocast"""
        if self.device != "cpu" and hasattr(torch.cuda, 'amp'):
            return torch.cuda.amp.autocast(enabled=True)
        else:
            return nullcontext()
    
    
    def freeze_mlp_weights(self):
        """Freeze MLP weights"""
        for param in self.position_wise_mlp.parameters():
            param.requires_grad = False

    def unfreeze_mlp_weights(self):
        """Unfreeze MLP weights"""
        for param in self.position_wise_mlp.parameters():
            param.requires_grad = True

    def freeze_lora_weights(self):
        """Freeze LoRA weights AND QFormer for MLP training"""
        frozen_count = 0
        
        # Freeze LoRA weights
        for name, param in self.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = False
                frozen_count += 1
        
        # Freeze QFormer components (speech_Qformer and speech_query_tokens)
        if hasattr(self.salmonn, 'speech_Qformer'):
            for name, param in self.salmonn.speech_Qformer.named_parameters():
                param.requires_grad = False
                frozen_count += 1
        
        if hasattr(self.salmonn, 'speech_query_tokens'):
            self.salmonn.speech_query_tokens.requires_grad = False
            frozen_count += 1
        
        # Freeze speech_llama_proj if it should be trainable during LoRA
        if hasattr(self.salmonn, 'speech_llama_proj'):
            for name, param in self.salmonn.speech_llama_proj.named_parameters():
                param.requires_grad = False
                frozen_count += 1
        
        logging.info(f"Frozen LoRA and QFormer weights ({frozen_count} parameters)")

    def unfreeze_lora_weights(self):
        """Unfreeze LoRA weights AND QFormer for LoRA training"""
        unfrozen_count = 0
        
        # Unfreeze LoRA weights
        for name, param in self.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                unfrozen_count += 1
        
        # Unfreeze QFormer components for training (since freeze_speech_QFormer=False in your config)
        if hasattr(self.salmonn, 'speech_Qformer'):
            for name, param in self.salmonn.speech_Qformer.named_parameters():
                param.requires_grad = True
                unfrozen_count += 1
        
        if hasattr(self.salmonn, 'speech_query_tokens'):
            self.salmonn.speech_query_tokens.requires_grad = True
            unfrozen_count += 1
        
        # Unfreeze speech_llama_proj if it should be trainable during LoRA
        # Check your SALMONN config - if freeze_speech_llama_proj=False, then unfreeze it
        if hasattr(self.salmonn, 'speech_llama_proj'):
            for name, param in self.salmonn.speech_llama_proj.named_parameters():
                param.requires_grad = True  # Set to True if freeze_speech_llama_proj=False
                unfrozen_count += 1
        
        logging.info(f"Unfrozen LoRA and QFormer weights ({unfrozen_count} parameters)")

    def get_trainable_parameters(self):
        """Get trainable parameters"""
        trainable = []
        frozen = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
            else:
                frozen.append(name)
        
        return trainable, frozen


    def check_mlp_health(self):
        """Check if MLP weights are healthy"""
        max_weight = 0.0
        max_grad = 0.0
        
        for param in self.position_wise_mlp.parameters():
            if param.data is not None:
                max_weight = max(max_weight, param.data.abs().max().item())
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())
        
        is_healthy = (max_weight < 100.0 and max_grad < 10.0 and 
                      not torch.isnan(torch.tensor(max_weight)) and 
                      not torch.isnan(torch.tensor(max_grad)))
        
        return is_healthy, {"max_weight": max_weight, "max_grad": max_grad}

    def reset_mlp_weights(self):
        """Reset MLP weights with better initialization"""
        for layer in self.position_wise_mlp:
            if isinstance(layer, nn.Linear):
                # Xavier initialization for better stability
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)
        
        logging.info("MLP weights reset with Xavier initialization")

    

    def update_label_tokens(self, symbol_mappings):
        """Update with COMPLETE symbol mappings"""
        if not symbol_mappings:
            return
        
        # ✅ FIX: Store the current mapping for discovery
        self.original_to_random_mapping = symbol_mappings.copy()
        logging.info(f"Stored mapping for discovery: {self.original_to_random_mapping}")
        
        # Get tokens for random symbols
        all_tokens = []
        for random_symbol in symbol_mappings.values():
            tokens = self.llama_tokenizer.encode(random_symbol, add_special_tokens=False)
            all_tokens.extend(tokens)
            logging.info(f"Symbol '{random_symbol}' -> tokens {tokens}")
        
        self.label_token_ids = list(set(all_tokens))
        logging.info(f"Updated to track {len(self.label_token_ids)} tokens from {len(symbol_mappings)} symbols")

    def convert_token_mappings_to_text(self, token_mappings):
        """Convert to ORIGINAL -> DISCOVERED mapping (FIXED - no spaces)"""
        if not hasattr(self, 'original_to_random_mapping'):
            logging.warning("No original mapping available")
            return {}
        
        text_mappings = {}
        
        for original_label, current_symbol in self.original_to_random_mapping.items():
            logging.info(f"Processing mapping for '{original_label}' -> '{current_symbol}'")
            
            # Get tokens for the CURRENT SYMBOL
            current_tokens = self.llama_tokenizer.encode(current_symbol, add_special_tokens=False)
            logging.info(f"Current symbol '{current_symbol}' has tokens: {current_tokens}")
            
            # Find what these current tokens map to
            discovered_tokens = []
            any_changed = False
            
            for token_id in current_tokens:
                if token_id in token_mappings and token_mappings[token_id] != token_id:
                    # Token was discovered/changed
                    discovered_tokens.append(token_mappings[token_id])
                    any_changed = True
                    logging.info(f"Token {token_id} mapped to {token_mappings[token_id]}")
                else:
                    # Keep original token
                    discovered_tokens.append(token_id)
                    logging.info(f"Token {token_id} kept as original")
            
            try:
                # Convert discovered tokens back to text
                discovered_text = self.llama_tokenizer.decode(discovered_tokens, skip_special_tokens=True)
                
                # ✅ FIX: Remove all spaces to create compound word
                discovered_text_no_spaces = discovered_text.replace(' ', '')
                
                if any_changed and discovered_text_no_spaces and discovered_text_no_spaces != current_symbol:
                    # ✅ SUCCESS: We found a new mapping (without spaces)
                    text_mappings[original_label] = discovered_text_no_spaces
                    logging.info(f"✅ Discovered mapping: '{original_label}' -> '{discovered_text_no_spaces}' (was '{current_symbol}')")
                else:
                    # ✅ NO CHANGE: Keep current symbol
                    text_mappings[original_label] = current_symbol
                    logging.info(f"No change for '{original_label}', keeping current symbol '{current_symbol}'")
                    
            except Exception as e:
                logging.warning(f"Error converting tokens for {original_label}: {e}")
                text_mappings[original_label] = current_symbol
        
        return text_mappings

    def get_english_token_range(self):
        """Get range of English tokens in vocabulary (expanded)"""
        # Common English tokens are usually in range 100-32000 for LLaMA
        vocab_size = self.embed_module.weight.shape[0]
        start_idx = 100  # Skip special tokens
        end_idx = min(30000, vocab_size - 1000)  # Expanded range
        
        logging.info(f"Using English token range: {start_idx} to {end_idx}")
        return start_idx, end_idx

    def is_english_token(self, token_id):
        """Check if token contains only English characters"""
        try:
            token_text = self.llama_tokenizer.decode([token_id], skip_special_tokens=True)
            # Check if contains only ASCII letters, numbers, and basic punctuation
            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_')
            return all(c in allowed_chars for c in token_text) and len(token_text.strip()) > 0
        except:
            return False

    def discover_symbols(self, save_path=None):
        """Simplified symbol discovery using apply_mlp_to_embeddings in inference mode"""
        if not self.training or not self.label_token_ids:
            logging.info("No label tokens to discover")
            return {}
        
        logging.info("=== Symbol Discovery ===")
        logging.info(f"Discovering symbols for {len(self.label_token_ids)} label tokens")
        
        # Temporarily set to eval mode for hard quantization
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # Get original embeddings for label tokens
            label_token_tensor = torch.tensor(self.label_token_ids, device=self.device)
            original_embeds = self.embed_module(label_token_tensor)
            
            # ✅ USE apply_mlp_to_embeddings in inference mode (hard quantization)
            transformed_embeds = self.apply_mlp_to_embeddings(
                original_embeds, 
                label_token_tensor, 
                scale_factor=0.2, 
                temperature=0.1
            )
            
            # Get English token range for filtering
            start_idx, end_idx = self.get_english_token_range()
            
            mappings = {}
            discovery_log = []
            
            for i, original_token_id in enumerate(self.label_token_ids):
                original_text = self.llama_tokenizer.decode([original_token_id], skip_special_tokens=True)
                
                # Get the transformed embedding
                transformed_embed = transformed_embeds[i:i+1]  # [1, hidden_dim]
                
                # Find closest vocabulary tokens
                vocab_embeds = self.embed_module.weight
                vocab_norm = F.normalize(vocab_embeds, p=2, dim=-1)
                transformed_norm = F.normalize(transformed_embed, p=2, dim=-1)
                
                similarities = torch.mm(transformed_norm, vocab_norm.t()).squeeze(0)
                top_similarities, top_indices = torch.topk(similarities, k=20, largest=True)
                
                # Find best English match (excluding original)
                best_match_id = None
                best_similarity = -1
                
                for j in range(len(top_indices)):
                    candidate_id = top_indices[j].item()
                    candidate_sim = top_similarities[j].item()
                    
                    # Skip original token
                    if candidate_id == original_token_id:
                        continue
                    
                    # Check if it's in English range and meets threshold
                    if (start_idx <= candidate_id <= end_idx and 
                        candidate_sim > 0.3 and 
                        self.is_english_token(candidate_id)):
                        
                        best_match_id = candidate_id
                        best_similarity = candidate_sim
                        break
                
                # Record discovery
                if best_match_id is not None:
                    best_match_text = self.llama_tokenizer.decode([best_match_id], skip_special_tokens=True)
                    mappings[original_token_id] = best_match_id
                    
                    discovery_info = {
                        'original_token_id': original_token_id,
                        'original_text': original_text,
                        'discovered_token_id': best_match_id,
                        'discovered_text': best_match_text,
                        'similarity': best_similarity
                    }
                    discovery_log.append(discovery_info)
                    
                    logging.info(f"Discovery: {original_token_id}('{original_text}') -> {best_match_id}('{best_match_text}') [sim: {best_similarity:.4f}]")
                else:
                    # Keep original
                    mappings[original_token_id] = original_token_id
                    discovery_info = {
                        'original_token_id': original_token_id,
                        'original_text': original_text,
                        'discovered_token_id': original_token_id,
                        'discovered_text': original_text,
                        'similarity': 1.0
                    }
                    discovery_log.append(discovery_info)
                    
                    logging.info(f"No change: {original_token_id}('{original_text}') kept")
            
            # Restore training mode
            if was_training:
                self.train()
            
            # Save discovery log if path provided
            if save_path:
                import json
                from datetime import datetime
                
                save_data = {
                    'timestamp': datetime.now().isoformat(),
                    'total_tokens': len(self.label_token_ids),
                    'changed_tokens': len([d for d in discovery_log if d['original_token_id'] != d['discovered_token_id']]),
                    'discoveries': discovery_log
                }
                
                with open(save_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                logging.info(f"Discovery log saved to: {save_path}")
                logging.info(f"Total discoveries: {save_data['changed_tokens']}/{save_data['total_tokens']}")
            
            return mappings


# SIMPLIFIED symbol generation functions
def generate_one_word_two_token_symbols(num_symbols, tokenizer):
    """SIMPLIFIED: Generate 2-token symbols"""
    import random
    import string
    
    chars = string.ascii_lowercase
    two_token_words = []
    used_words = set()
    
    logging.info("Searching for 2-token words...")
    
    attempts = 0
    max_attempts = 10000
    
    while len(two_token_words) < num_symbols and attempts < max_attempts:
        attempts += 1
        
        word_length = random.choice([4, 5])
        word = ''.join(random.choice(chars) for _ in range(word_length))
        
        if word in used_words:
            continue
        
        used_words.add(word)
        
        try:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            
            if len(token_ids) == 2:
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                if decoded.lower() == word.lower():
                    two_token_words.append(word)
                    logging.info(f"Found 2-token word #{len(two_token_words)}: '{word}' -> {token_ids}")
                    
                    token_texts = [tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]
                    logging.info(f"  Tokens: {token_texts}")
        except:
            continue
    
    logging.info(f"Generated {len(two_token_words)} two-token words from {attempts} attempts")
    
    
    return two_token_words[:num_symbols]

def create_label_mapping(original_labels, random_symbols):
    """Create simple label to symbol mapping"""
    return {orig: rand for orig, rand in zip(original_labels, random_symbols)}
