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
    """
    Enhanced SALMONN model with MLP embedding transformations,
    applied BEFORE LoRA adapters to ensure proper gradient flow.
    Uses CustomSALMONN implementations of forward, get_speech_embeddings, etc.
    """
    
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
        lora_alpha=16,
        lora_dropout=0.05,
        device=None,
        low_resource=True
    ):
        super().__init__()
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure SALMONN without LoRA first
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
        "freeze_speech_QFormer": False,  # <-- Set to False to make QFormer trainable
        "num_speech_query_token": 1,
        "window_level_Qformer": True,
        "second_per_window": 0.333333,
        "second_stride": 0.333333,
        "speech_llama_proj_model": "",
        "freeze_speech_llama_proj": False,  # <-- Set to False if you want to train the projection layer
        "ckpt": "/data2/neeraja/neeraja/salmonn_v1.pth"
    }

        # Initialize the SALMONN model using from_config
        logging.info("Loading base SALMONN model without LoRA...")
        self.salmonn = SALMONN.from_config(salmonn_config)
        self.salmonn.to(self.device)
        
        # Store model attributes
        self.label_tokens = label_tokens or []
        self.label_token_ids = []
        self.batch_counter = 0
        self.speech_placeholder = "<SpeechHere>"
        self.use_lora = True
        
        # Get references to SALMONN components for easier access
        self.llama_model = self.salmonn.llama_model
        self.llama_tokenizer = self.salmonn.llama_tokenizer
        
        # Get embedding module and dimensions
        if hasattr(self.llama_model, 'model'):
            if hasattr(self.llama_model.model, 'model'):
                self.embed_module = self.llama_model.model.model.embed_tokens
            else:
                self.embed_module = self.llama_model.model.embed_tokens
        else:
            self.embed_module = self.llama_model.embed_tokens
        
        self.embed_dim = self.embed_module.weight.shape[1]
        self.hidden_dim = hidden_dim or self.embed_dim
        
        # Create the MLP for embedding transformation
        self.position_wise_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        # Use MUCH more conservative initialization
        with torch.no_grad():
            for layer in self.position_wise_mlp:
                if isinstance(layer, nn.Linear):
                    # Use Xavier uniform with very small gain
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)  # Much smaller gain
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
        
        base_dtype = self.embed_module.weight.dtype
        self.position_wise_mlp = self.position_wise_mlp.to(dtype=base_dtype)
        logging.info(f"Initialized MLP with conservative initialization")

        self.position_wise_mlp.to(self.device)

        self.input_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        
        # Convert label tokens to token IDs
        if self.label_tokens:
            self.label_token_ids = self._get_label_token_ids()
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.salmonn.parameters():
                param.requires_grad = False
            
            # Make MLP trainable
            for param in self.position_wise_mlp.parameters():
                param.requires_grad = True
                
            trainable_params = sum(p.numel() for p in self.position_wise_mlp.parameters())
            logging.info(f"Model has {trainable_params:,} trainable MLP parameters")
        
        # Keep original embeddings frozen for reference
        self.original_embed_matrix = self.embed_module.weight.clone().detach()
        self.original_embed_matrix.requires_grad = False
        
        # Store discovered symbol mappings
        self.symbol_mappings = {}  # Maps original_token_id -> discovered_token_id

        self.bypass_mlp_during_lora = False  # Add this flag
    
    def set_mlp_bypass(self, bypass=True):
        """Enable/disable MLP bypass during LoRA training"""
        self.bypass_mlp_during_lora = bypass
        logging.info(f"MLP bypass set to: {bypass}")
    
    def apply_mlp_to_embeddings(self, embeddings, token_ids):
        """Apply MLP transformation with bypass option"""
        if self.bypass_mlp_during_lora:
            logging.debug("MLP bypass active - returning original embeddings")
            return embeddings
        
        # Apply MLP transformation to specific tokens in embeddings
        return self.transform_text_embeddings(embeddings, token_ids)
    
    def get_transformed_embeddings(self):
        """Get embeddings with transformations applied for label tokens - consistent with transform_text_embeddings"""
        # Start with a copy of the original embeddings
        transformed_matrix = self.embed_module.weight.clone()
        
        # Get MLP properties
        device = next(self.position_wise_mlp.parameters()).device
        dtype = next(self.position_wise_mlp.parameters()).dtype
        
        # Apply transformation to label tokens using the SAME method as transform_text_embeddings
        label_embeds = self.embed_module.weight[self.label_token_ids].to(device=device, dtype=dtype)
        
        # Apply MLP to get residual component (same as transform_text_embeddings)
        residual_values = self.position_wise_mlp(label_embeds)
        
        # Add residuals to original embeddings (same as transform_text_embeddings)
        transformed_embeds = label_embeds + residual_values
        
        # Move back to original device and dtype if needed
        if (transformed_embeds.device != transformed_matrix.device or 
            transformed_embeds.dtype != transformed_matrix.dtype):
            transformed_embeds = transformed_embeds.to(
                device=transformed_matrix.device, 
                dtype=transformed_matrix.dtype
            )
        
        # Update in the transformed matrix
        transformed_matrix[self.label_token_ids] = transformed_embeds
        
        return transformed_matrix
    
    def save_transformed_embeddings(self, path):
        """Save the transformed token embeddings to a file"""
        with torch.no_grad():
            transformed_matrix = self.get_transformed_embeddings()
        
        # Save transformed embeddings with metadata
        torch.save({
            "embeddings": transformed_matrix,
            "vocab_size": transformed_matrix.shape[0],
            "embedding_dim": transformed_matrix.shape[1],
            "label_tokens": self.label_tokens,
            "label_token_ids": self.label_token_ids,
            "mlp_state_dict": self.position_wise_mlp.state_dict()
        }, path)
        logging.info(f"Saved transformed embeddings to {path}")
    
    def forward(self, samples):
        """
        Forward pass that handles all the sample processing similar to CustomSALMONN.
        Uses transformed embeddings during training.
        """
        start_time = time.time()
        
        # Process speech embeddings
        speech_embeds, speech_atts, example_embeds, example_atts = self.get_speech_embeddings(samples)
        
        
        # First batch logging
        if self.batch_counter == 0:
            if speech_embeds is not None:
                # Original logging for single speech embedding
                logging.info("Speech data detected and processed")
                logging.info(f"Speech embeddings device: {speech_embeds.device}")
                logging.info(f"Speech embeddings range: {speech_embeds.min():.3f} to {speech_embeds.max():.3f}")
            else:
                logging.info("No speech data detected, using text-only mode")
        
        if self.batch_counter == 0:
            logging.info(f"Prompt example:\n{samples['prompt'][0]}")
            logging.info(f"Output:\n{samples['completion'][0]}")
        
        # Get number of examples per sample
        num_examples = samples.get("num_examples", torch.zeros(len(samples["prompt"]), dtype=torch.long))
        
        # Wrap embeddings with prompts
        wrapped_embeds, wrapped_atts = self.custom_prompt_wrap(
            speech_embeds, speech_atts, samples["prompt"],
            num_examples, example_embeds, example_atts
        )
        
        if self.batch_counter == 0:
            logging.info(f"Wrapped embeddings shape: {wrapped_embeds.shape}")
            logging.info(f"Wrapped attention mask shape: {wrapped_atts.shape}")
        
        gen_start_time = time.time()
        
        # Process target text
        target_tokens = self.llama_tokenizer(
            samples["completion"],
            padding='longest',
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
        ).to(wrapped_embeds.device)
        
        # Get normal embeddings first 
        target_embeds = self.embed_module(target_tokens.input_ids)

        # Apply MLP to label token embeddings (with bypass check)
        if self.label_token_ids and not self.bypass_mlp_during_lora:
            logging.info(f"Applying MLP to {len(self.label_token_ids)} label tokens")
            target_embeds = self.apply_mlp_to_embeddings(target_embeds, self.label_token_ids)
        elif self.bypass_mlp_during_lora:
            logging.debug("MLP bypass active - using original embeddings for LoRA training")
        
        prompt_length = wrapped_embeds.size(1)
        
        # Create labels tensor
        labels = torch.full(
            [wrapped_atts.shape[0], wrapped_atts.shape[1] + target_tokens.input_ids.size(1)],
            fill_value=-100,
            dtype=torch.long,
            device=wrapped_embeds.device
        )
        labels[:, prompt_length:] = target_tokens.input_ids
        
        # Mask padding tokens
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
        
        # logging.info(f"Forward pass took {time.time() - start_time:.2f} seconds")

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
        """
        Wraps speech embeddings with text prompts and examples.
        Implementation from CustomSALMONN.
        """
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
                if self.use_lora:
                    part_embed = self.llama_model.model.model.embed_tokens(tokens['input_ids'].squeeze(0))
                else:
                    part_embed = self.llama_model.model.embed_tokens(tokens['input_ids'].squeeze(0))

                # Transform embeddings for label tokens
                if self.training and self.label_token_ids:
                    part_embed = self.transform_text_embeddings(
                        part_embed, 
                        tokens['input_ids'].squeeze(0)
                    )
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
        
    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        """
        Encode speech inputs using the SALMONN speech encoder.
        """
        return self.salmonn.encode_speech(    
            spectrogram=spectrogram,
            raw_wav=raw_wav,
            audio_padding_mask=audio_padding_mask
        )

    
    def _get_label_token_ids(self):
        """Convert label tokens to token IDs and maintain token-to-label mapping"""
        token_ids = []
        # Create a mapping from token ID to original label
        self.token_id_to_label = {}
        # Create a mapping from label to all its token IDs
        self.label_to_token_ids = {}
        
        for label in self.label_tokens:
            if isinstance(label, int):
                token_ids.append(label)
                self.token_id_to_label[label] = label
                self.label_to_token_ids[label] = [label]
            else:
                # Token is a string, encode it
                tokenized = self.llama_tokenizer.encode(label, add_special_tokens=False)
                if len(tokenized) > 0:
                    # Add ALL token IDs from tokenization
                    for token_id in tokenized:
                        token_ids.append(token_id)
                        self.token_id_to_label[token_id] = label
                    
                    self.label_to_token_ids[label] = tokenized
                    logging.info(f"Label '{label}' tokenized to {len(tokenized)} tokens: {tokenized}")
                    token_texts = [self.llama_tokenizer.decode([tid]) for tid in tokenized]
                    logging.info(f"  Decoded tokens: {token_texts}")
                else:
                    logging.warning(f"Could not tokenize '{label}'")
        
        if token_ids:
            logging.info(f"Label token IDs: {token_ids}")
            for token_id in token_ids:
                token_text = self.llama_tokenizer.decode([token_id])
                original_label = self.token_id_to_label.get(token_id, "unknown")
                logging.info(f"Token ID {token_id} ('{token_text}') belongs to label '{original_label}'")
        else:
            logging.warning("No valid label tokens provided")
        
        return token_ids




    def print_token_info(self, tokens):
        """Print tokenization information for debugging"""
        tokenizer = self.llama_tokenizer
        
        for token in tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            token_texts = [tokenizer.decode([tid]) for tid in token_ids]
            
            logging.info(f"Token '{token}':")
            logging.info(f"  Token IDs: {token_ids}")
            logging.info(f"  Decoded as: {token_texts}")

    def transform_text_embeddings(self, embeddings, token_ids):
        """Apply MLP transformation to specific tokens in embeddings - with robust safety checks"""
        if not self.training or not self.label_token_ids:
            return embeddings
        
        # Create a mask for tokens that need transformation
        is_label_token = torch.zeros_like(token_ids, dtype=torch.bool)
        for token_id in self.label_token_ids:
            is_label_token = is_label_token | (token_ids == token_id)
        
        # If we have no tokens to transform, return original
        if not is_label_token.any():
            return embeddings
        
        # Get positions where we need to apply MLP
        nonzero_result = is_label_token.nonzero(as_tuple=True)
        
        if isinstance(nonzero_result, tuple) and len(nonzero_result) == 2:
            batch_indices, token_positions = nonzero_result
        elif isinstance(nonzero_result, tuple) and len(nonzero_result) == 1:
            token_positions = nonzero_result[0]
            batch_indices = torch.zeros_like(token_positions)
        else:
            logging.error(f"Unexpected nonzero result: {type(nonzero_result)}")
            return embeddings
        
        # Extract embeddings at those positions
        if len(embeddings.shape) == 2:  # 1D case
            to_transform = embeddings[token_positions]
        else:  # 2D case
            to_transform = embeddings[batch_indices, token_positions]
        
        # Check input embeddings for NaN/Inf
        if torch.isnan(to_transform).any() or torch.isinf(to_transform).any():
            logging.error("NaN/Inf detected in input embeddings to MLP!")
            return embeddings
        
        # Use safe MLP forward pass
        residual_values = self.safe_mlp_forward(to_transform)
        
        # If safe_mlp_forward returned original embeddings, it means MLP failed
        if torch.equal(residual_values, to_transform):
            logging.warning("MLP forward failed, using original embeddings")
            return embeddings
        
        # Scale down the residual to prevent instability
        residual_values = residual_values * 0.001  # Very small scale
        
        # Create residual mask
        if len(embeddings.shape) == 2:  # 1D case
            residual_mask = torch.zeros_like(embeddings)
            residual_mask[token_positions] = residual_values
        else:  # 2D case
            residual_mask = torch.zeros_like(embeddings)
            residual_mask[batch_indices, token_positions] = residual_values
        
        # Add residuals to create new tensor
        transformed_embeddings = embeddings + residual_mask
        
        # Final safety check
        if torch.isnan(transformed_embeddings).any() or torch.isinf(transformed_embeddings).any():
            logging.error("NaN/Inf detected in final transformed embeddings!")
            return embeddings
        
        # logging.info(f"Applied MLP to {is_label_token.sum().item()} label tokens")
        return transformed_embeddings

    def maybe_autocast(self):
        """Context manager for autocast mixed precision where appropriate"""
        # Check if we're on CUDA and should use autocast
        if self.device != "cpu" and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            return torch.cuda.amp.autocast(enabled=True)
        else:
            # Return a dummy context manager that does nothing
            return nullcontext()

    def find_symbol_mappings(self):
        """Find which original vocabulary tokens are most similar to transformed symbols"""
        if not self.label_token_ids:
            return {}
        
        with torch.no_grad():
            # Get device and dtype from MLP
            device = next(self.position_wise_mlp.parameters()).device
            dtype = next(self.position_wise_mlp.parameters()).dtype
            
            # Use the SAME transformation method as transform_text_embeddings
            mappings = {}
            
            print("\n=== Symbol Discovery After Epoch ===")
            
            # Normalize original matrix for similarity search - ensure float32 for CPU ops
            original_norm = F.normalize(self.original_embed_matrix.float(), p=2, dim=1)
            
            # Process each label
            for label, token_ids in self.label_to_token_ids.items():
                print(f"\nLabel: '{label}'")
                label_mappings = []
                
                for token_idx, token_id in enumerate(token_ids):
                    # Get original embedding for this token
                    original_embed = self.embed_module.weight[token_id].unsqueeze(0).to(device=device, dtype=dtype)
                    
                    # Apply MLP transformation (same as transform_text_embeddings)
                    residual_values = self.safe_mlp_forward(original_embed)
                    
                    # Check if MLP worked
                    if torch.equal(residual_values, original_embed):
                        print(f"  '{self.llama_tokenizer.decode([token_id])}' (ID: {token_id}) -> MLP failed, using original")
                        mappings[token_id] = token_id
                        label_mappings.append(token_id)
                        continue
                    
                    # Scale residual (same as transform_text_embeddings)
                    residual_values = residual_values * 0.1
                    
                    # Add residual to original embedding
                    transformed_embed = original_embed + residual_values
                    
                    # Check for NaN/Inf
                    if torch.isnan(transformed_embed).any() or torch.isinf(transformed_embed).any():
                        print(f"  '{self.llama_tokenizer.decode([token_id])}' (ID: {token_id}) -> NaN/Inf detected, using original")
                        mappings[token_id] = token_id
                        label_mappings.append(token_id)
                        continue
                    
                    # Normalize transformed embedding - convert to float32 for CPU operations
                    transformed_norm = F.normalize(transformed_embed.float(), p=2, dim=1)
                    
                    # Move to CPU for similarity computation
                    transformed_norm_cpu = transformed_norm.cpu()
                    original_norm_cpu = original_norm.cpu()
                    
                    try:
                        # Find most similar token in original vocabulary
                        similarity = torch.matmul(transformed_norm_cpu, original_norm_cpu.transpose(0, 1))[0]
                        
                        # Exclude the original token itself
                        similarity[token_id] = -1.0
                        
                        # Get best match
                        best_match_id = torch.argmax(similarity).item()
                        best_similarity = similarity[best_match_id].item()
                        
                        # Check if similarity is valid
                        if torch.isnan(torch.tensor(best_similarity)) or torch.isinf(torch.tensor(best_similarity)):
                            print(f"  '{self.llama_tokenizer.decode([token_id])}' (ID: {token_id}) -> Invalid similarity, using original")
                            mappings[token_id] = token_id
                            label_mappings.append(token_id)
                            continue
                        
                        # Store mapping
                        mappings[token_id] = best_match_id
                        label_mappings.append(best_match_id)
                        
                        # Print the discovery
                        original_text = self.llama_tokenizer.decode([token_id])
                        discovered_text = self.llama_tokenizer.decode([best_match_id])
                        print(f"  '{original_text}' (ID: {token_id}) -> '{discovered_text}' (ID: {best_match_id}) [similarity: {best_similarity:.4f}]")
                    
                    except Exception as e:
                        print(f"  '{self.llama_tokenizer.decode([token_id])}' (ID: {token_id}) -> Error in similarity computation: {e}")
                        mappings[token_id] = token_id
                        label_mappings.append(token_id)
                        continue
                
                # Show combined mapping for multi-token labels
                if len(label_mappings) > 1:
                    combined_original = "".join([self.llama_tokenizer.decode([tid]) for tid in token_ids])
                    combined_discovered = "".join([self.llama_tokenizer.decode([tid]) for tid in label_mappings])
                    print(f"  Combined: '{combined_original}' -> '{combined_discovered}'")
            
            # Update stored mappings
            self.symbol_mappings = mappings
            
            return mappings

    def get_symbol_replacement_dict(self):
        """Get a dictionary for replacing symbols during inference"""
        if not self.symbol_mappings:
            return {}
        
        replacement_dict = {}
        
        for label, token_ids in self.label_to_token_ids.items():
            # Get the discovered tokens for this label
            discovered_ids = [self.symbol_mappings[tid] for tid in token_ids if tid in self.symbol_mappings]
            
            if discovered_ids:
                # Create text representations
                original_text = "".join([self.llama_tokenizer.decode([tid]) for tid in token_ids])
                discovered_text = "".join([self.llama_tokenizer.decode([tid]) for tid in discovered_ids])
                
                replacement_dict[original_text] = discovered_text
        
        return replacement_dict

    def prepare_inference_text(self, text):
        """Replace symbols in text for inference using discovered mappings"""
        if not self.symbol_mappings:
            return text
        
        replacement_dict = self.get_symbol_replacement_dict()
        
        # Replace symbols in the text
        for original, discovered in replacement_dict.items():
            text = text.replace(original, discovered)
        
        return text

    def print_symbol_mappings(self):
        """Print current symbol mappings in a readable format"""
        if not self.symbol_mappings:
            print("No symbol mappings discovered yet.")
            return
        
        print("\n=== Current Symbol Mappings ===")
        replacement_dict = self.get_symbol_replacement_dict()
        
        for original, discovered in replacement_dict.items():
            print(f"'{original}' -> '{discovered}'")
        
        print("\nFor inference, replace these symbols in your prompts:")
        print("Example usage:")
        for original, discovered in replacement_dict.items():
            print(f"  text = text.replace('{original}', '{discovered}')")

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
        """Reset MLP weights to conservative initialization"""
        logging.warning("Resetting MLP weights due to instability")
        with torch.no_grad():
            for layer in self.position_wise_mlp:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def safe_mlp_forward(self, embeddings):
        """Safely apply MLP with health checks"""
        # Check MLP health before forward pass
        is_healthy, message = self.check_mlp_health()
        if not is_healthy:
            logging.error(f"MLP unhealthy: {message}")
            self.reset_mlp_weights()
            return embeddings  # Return original embeddings if MLP is corrupted
        
        # Ensure input is the right dtype
        target_dtype = next(self.position_wise_mlp.parameters()).dtype
        if embeddings.dtype != target_dtype:
            embeddings = embeddings.to(dtype=target_dtype)
        
        # Apply MLP
        try:
            output = self.position_wise_mlp(embeddings)
            
            # Check output for NaN/Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                logging.error("NaN/Inf detected in MLP output")
                self.reset_mlp_weights()
                return embeddings
            
            return output
        except Exception as e:
            logging.error(f"Error in MLP forward pass: {e}")
            self.reset_mlp_weights()
            return embeddings

    def debug_mlp_forward(self, embeddings):
        """Debug version of MLP forward to see where corruption happens"""
        logging.info(f"=== MLP Forward Debug ===")
        logging.info(f"Input shape: {embeddings.shape}")
        logging.info(f"Input stats: min={embeddings.min():.6f}, max={embeddings.max():.6f}, mean={embeddings.mean():.6f}")
        
        x = embeddings
        for i, layer in enumerate(self.position_wise_mlp):
            if isinstance(layer, nn.Linear):
                logging.info(f"Layer {i} ({type(layer).__name__}):")
                logging.info(f"  Weight stats: min={layer.weight.min():.6f}, max={layer.weight.max():.6f}, mean={layer.weight.mean():.6f}")
                logging.info(f"  Bias stats: min={layer.bias.min():.6f}, max={layer.bias.max():.6f}, mean={layer.bias.mean():.6f}")
            
            x_before = x.clone()
            x = layer(x)
            
            logging.info(f"  After layer {i}: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}")
            
            if torch.isnan(x).any():
                logging.error(f"  NaN detected after layer {i}!")
                logging.info(f"  Input to this layer: min={x_before.min():.6f}, max={x_before.max():.6f}")
                break
            if torch.isinf(x).any():
                logging.error(f"  Inf detected after layer {i}!")
                break
                
        return x

    def freeze_mlp_weights(self):
        """Freeze MLP weights for LoRA training"""
        for param in self.position_wise_mlp.parameters():
            param.requires_grad = False
        logging.info("Frozen MLP weights")

    def unfreeze_mlp_weights(self):
        """Unfreeze MLP weights for MLP training"""
        for param in self.position_wise_mlp.parameters():
            param.requires_grad = True
        logging.info("Unfrozen MLP weights")

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
        """Get currently trainable parameters with detailed breakdown"""
        lora_params = []
        mlp_params = []
        qformer_params = []
        projection_params = []
        other_params = []
        
        frozen_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append(name)
                elif 'position_wise_mlp' in name:
                    mlp_params.append(name)
                elif 'speech_qformer' in name.lower() or 'speech_query_tokens' in name.lower():
                    qformer_params.append(name)
                elif 'speech_llama_proj' in name.lower():
                    projection_params.append(name)
                else:
                    other_params.append(name)
            else:
                frozen_params.append(name)
        
        # Log breakdown
        logging.info(f"Trainable parameter breakdown:")
        logging.info(f"  LoRA parameters: {len(lora_params)}")
        logging.info(f"  MLP parameters: {len(mlp_params)}")
        logging.info(f"  QFormer parameters: {len(qformer_params)}")
        logging.info(f"  Projection parameters: {len(projection_params)}")
        logging.info(f"  Other parameters: {len(other_params)}")
        logging.info(f"  Frozen parameters: {len(frozen_params)}")
        
        trainable = lora_params + mlp_params + qformer_params + projection_params + other_params
        return trainable, frozen_params

def generate_random_symbols(num_symbols, symbol_length=2):
    """Generate random symbols of fixed length"""
    import random
    import string
    
    symbols = []
    used_symbols = set()
    
    # Create character pool (avoid confusing characters)
    chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
    excluded = {'0', 'O', '1', 'l', 'I'}  # Avoid confusing characters
    chars = [c for c in chars if c not in excluded]
    
    while len(symbols) < num_symbols:
        # Generate random symbol of fixed length
        symbol = ''.join(random.choices(chars, k=symbol_length))
        
        # Ensure uniqueness and no semantic meaning
        if symbol not in used_symbols and not symbol.lower() in ['no', 'yes', 'ok', 'hi']:
            symbols.append(symbol)
            used_symbols.add(symbol)
    
    return symbols

# Usage in unified_symbol_training.py:
def get_random_label_tokens(original_labels, symbol_length=2):
    """Replace original labels with random symbols"""
    random_symbols = generate_random_symbols(len(original_labels), symbol_length)
    
    # Create mapping
    label_mapping = {}
    for orig, rand in zip(original_labels, random_symbols):
        label_mapping[orig] = rand
    
    logging.info(f"Generated random label mapping: {label_mapping}")
    return random_symbols, label_mapping