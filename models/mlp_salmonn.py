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
        beats_path=None,
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
    
    def set_mlp_bypass(self, bypass=True):
        """Enable/disable MLP bypass"""
        self.bypass_mlp_during_lora = bypass
        logging.info(f"MLP bypass set to: {bypass}")
    
    def apply_mlp_to_embeddings(self, embeddings, token_ids):
        """SIMPLIFIED: Apply MLP only if not bypassed"""
        if self.bypass_mlp_during_lora or not self.training:
            return embeddings
        
        return self.transform_text_embeddings(embeddings, token_ids)
    
    def transform_text_embeddings(self, embeddings, token_ids):
        """Transform embeddings with MLP - reduced logging"""
        if not self.training or not self.label_token_ids:
            return embeddings
        
        # Convert token_ids to tensor if needed
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, device=embeddings.device)
        elif token_ids.dim() > 1:
            token_ids = token_ids.squeeze()
        
        # LOG: Only for first few batches
        if self.batch_counter < 3:
            logging.info(f"=== MLP Transform Debug ===")
            logging.info(f"Input embeddings shape: {embeddings.shape}")
            logging.info(f"Token sequence length: {len(token_ids)}")
            logging.info(f"Looking for {len(self.label_token_ids)} label tokens: {self.label_token_ids}")
        
        # Find positions where label tokens appear
        positions_to_transform = []
        tokens_found = {}
        
        for label_token_id in self.label_token_ids:
            positions = (token_ids == label_token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                positions_to_transform.extend(positions.tolist())
                if self.batch_counter < 3:  # Only log for first few batches
                    token_text = self.llama_tokenizer.decode([label_token_id], skip_special_tokens=True)
                    tokens_found[label_token_id] = {
                        'text': token_text,
                        'positions': positions.tolist(),
                        'count': len(positions)
                    }
                    logging.info(f"✓ Found label token {label_token_id} ('{token_text}') at positions: {positions.tolist()}")
        
        # Return early if no tokens found
        if not positions_to_transform:
            return embeddings
        
        # Only log summary for first few batches
        if self.batch_counter < 3:
            total_transformations = len(positions_to_transform)
            unique_positions = len(set(positions_to_transform))
            logging.info(f"📊 TRANSFORMATION SUMMARY:")
            logging.info(f"   • Total label tokens found: {total_transformations}")
            logging.info(f"   • Unique positions to transform: {unique_positions}")
            
            for token_id, info in tokens_found.items():
                logging.info(f"   • Token '{info['text']}' (ID: {token_id}): {info['count']} occurrences")
        
        try:
            positions_tensor = torch.tensor(positions_to_transform, device=embeddings.device)
            
            # Check bounds
            max_pos = positions_tensor.max().item()
            seq_len = embeddings.shape[-2]
            if max_pos >= seq_len:
                logging.error(f"❌ BOUNDS ERROR: Max position {max_pos} >= sequence length {seq_len}")
                return embeddings
            
            # Extract embeddings to transform
            if len(embeddings.shape) == 2:
                to_transform = embeddings[positions_tensor]
            else:
                batch_indices = torch.zeros_like(positions_tensor)
                to_transform = embeddings[batch_indices, positions_tensor]
            
            # Convert to MLP's dtype if needed
            original_dtype = to_transform.dtype
            mlp_dtype = next(self.position_wise_mlp.parameters()).dtype
            
            if to_transform.dtype != mlp_dtype:
                to_transform = to_transform.to(dtype=mlp_dtype)
            
            # Apply MLP with residual connection
            mlp_output = self.position_wise_mlp(to_transform)
            transformed = to_transform + mlp_output  # old + residue
            
            # Convert back to original dtype
            if transformed.dtype != original_dtype:
                transformed = transformed.to(dtype=original_dtype)
            
            # Create output
            output_embeds = embeddings.clone()
            if len(embeddings.shape) == 2:
                output_embeds[positions_tensor] = transformed
            else:
                output_embeds[batch_indices, positions_tensor] = transformed
            
            # Only log success for first few batches
            if self.batch_counter < 3:
                logging.info(f"✅ Successfully applied MLP transformation to {len(positions_tensor)} token positions")
            
            return output_embeds
            
        except Exception as e:
            logging.error(f"❌ Error in MLP transformation: {e}")
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
        
        # Apply MLP if training and not bypassed
        if self.training and self.label_token_ids and not self.bypass_mlp_during_lora:
            target_sequence = target_tokens.input_ids[0] if target_tokens.input_ids.dim() > 1 else target_tokens.input_ids
            
            # Only log target info for first few batches
            if self.batch_counter < 3:
                target_text = self.llama_tokenizer.decode(target_sequence, skip_special_tokens=True)
                logging.info(f"=== TARGET TRANSFORMATION ===")
                logging.info(f"Target text: '{target_text}'")
                logging.info(f"Target sequence: {target_sequence.tolist()}")
                logging.info(f"Target embeddings shape: {target_embeds.shape}")
            
            original_target_norm = torch.norm(target_embeds, dim=-1).mean().item()
            target_embeds = self.apply_mlp_to_embeddings(target_embeds, target_sequence)
            new_target_norm = torch.norm(target_embeds, dim=-1).mean().item()
            
            if self.batch_counter < 3:
                logging.info(f"Target transformation complete:")
                logging.info(f"   Original norm: {original_target_norm:.4f}")
                logging.info(f"   New norm: {new_target_norm:.4f}")
                logging.info(f"   Change ratio: {new_target_norm/original_target_norm:.4f}")
        
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
    
    def find_symbol_mappings(self):
        """SIMPLIFIED symbol discovery"""
        if not self.label_token_ids:
            return {}
        
        # Simplified random mapping for now
        mappings = {}
        for token_id in self.label_token_ids:
            mappings[token_id] = token_id  # Identity mapping for simplicity
        
        return mappings
    
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

    def find_closest_symbols(self):
        """Symbol discovery with minimal logging"""
        if not self.training or not self.label_token_ids:
            logging.info("No label tokens to discover")
            return {}
        
        logging.info("=== Simple Symbol Discovery ===")
        logging.info(f"Finding closest symbols for {len(self.label_token_ids)} label tokens")
        
        with torch.no_grad():
            # Get original embedding matrix
            label_token_tensor = torch.tensor(self.label_token_ids, device=self.device)
            original_embeds = self.embed_module(label_token_tensor)
            
            mlp_dtype = next(self.position_wise_mlp.parameters()).dtype
            if original_embeds.dtype != mlp_dtype:
                original_embeds = original_embeds.to(dtype=mlp_dtype)
            
            mlp_has_trained = any(param.abs().max().item() > 0.001 for param in self.position_wise_mlp.parameters())
            transformed_embeds = self.position_wise_mlp(original_embeds) if mlp_has_trained else original_embeds
            
            original_dtype = self.embed_module.weight.dtype
            if transformed_embeds.dtype != original_dtype:
                transformed_embeds = transformed_embeds.to(dtype=original_dtype)
            
            full_embedding_matrix = self.embed_module.weight
            mappings = {}
            
            for i, original_token_id in enumerate(self.label_token_ids):
                transformed_embed = transformed_embeds[i:i+1]
                transformed_norm = F.normalize(transformed_embed, p=2, dim=1)
                vocab_norm = F.normalize(full_embedding_matrix, p=2, dim=1)
                similarities = torch.mm(transformed_norm, vocab_norm.t()).squeeze(0)
                
                top_similarities, top_indices = torch.topk(similarities, k=10, largest=True)
                best_match_id = next((candidate_id for candidate_id in top_indices if candidate_id != original_token_id and candidate_id >= 100), original_token_id)
                
                mappings[original_token_id] = best_match_id
                original_text = self.llama_tokenizer.decode([original_token_id], skip_special_tokens=True)
                best_match_text = self.llama_tokenizer.decode([best_match_id], skip_special_tokens=True)
                logging.info(f"Mapping found: {original_token_id} ('{original_text}') -> {best_match_id} ('{best_match_text}')")
        
        logging.info(f"=== Discovery Complete: Found {len(mappings)} mappings ===")
        return mappings

    def update_label_tokens(self, symbol_mappings):
        """Update with COMPLETE symbol mappings"""
        if not symbol_mappings:
            return
        
        self.original_to_random_mapping = symbol_mappings.copy()
        
        all_tokens = []
        for random_symbol in symbol_mappings.values():
            tokens = self.llama_tokenizer.encode(random_symbol, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        self.label_token_ids = list(set(all_tokens))
        logging.info(f"Updated to track {len(self.label_token_ids)} tokens from {len(symbol_mappings)} symbols")

    def convert_token_mappings_to_text(self, token_mappings):
        """Convert to ORIGINAL -> DISCOVERED mapping"""
        if not hasattr(self, 'original_to_random_mapping'):
            logging.warning("No original mapping available")
            return {}
        
        text_mappings = {}
        
        for original_label, random_symbol in self.original_to_random_mapping.items():
            random_tokens = self.llama_tokenizer.encode(random_symbol, add_special_tokens=False)
            discovered_tokens = [token_mappings.get(token_id, token_id) for token_id in random_tokens]
            
            try:
                discovered_text = self.llama_tokenizer.decode(discovered_tokens, skip_special_tokens=True).strip()
                if discovered_text and discovered_text != random_symbol:
                    text_mappings[original_label] = discovered_text
                    logging.info(f"Discovered mapping: '{original_label}' -> '{discovered_text}'")
            except Exception as e:
                logging.warning(f"Error converting tokens for {original_label}: {e}")
        
        return text_mappings


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
