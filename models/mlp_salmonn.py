import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple, Union, Any

# Import PEFT for LoRA
from peft import LoraConfig, TaskType, get_peft_model

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
        low_resource=False
    ):
        super().__init__()
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure SALMONN without LoRA first
        salmonn_config = {
            "llama_path": llama_path,
            "whisper_path": whisper_path,
            "beats_path": beats_path,
            "lora": False,  # Important: Start without LoRA
            "low_resource": low_resource
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
        self.use_lora = lora
        
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
        self.position_wise_mlp.to(self.device)
        
        # Convert label tokens to token IDs
        if self.label_tokens:
            for token in self.label_tokens:
                token_id = self.llama_tokenizer.convert_tokens_to_ids(token)
                if token_id != self.llama_tokenizer.unk_token_id:
                    self.label_token_ids.append(token_id)
                else:
                    logging.warning(f"Label token '{token}' not found in vocabulary")
            logging.info(f"Initialized with {len(self.label_token_ids)} label tokens")
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.salmonn.parameters():
                param.requires_grad = False
            
            # Make MLP trainable
            for param in self.position_wise_mlp.parameters():
                param.requires_grad = True
                
            trainable_params = sum(p.numel() for p in self.position_wise_mlp.parameters())
            logging.info(f"Model has {trainable_params:,} trainable MLP parameters")
        
        # NOW apply LoRA if it was originally enabled
        if self.use_lora:
            logging.info("Applying LoRA adapters AFTER MLP setup")
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()
    
    def get_transformed_embeddings(self):
        """Get embeddings with transformations applied for label tokens"""
        # Start with a copy of the original embeddings
        transformed_matrix = self.embed_module.weight.clone()
        
        # Get MLP properties
        device = next(self.position_wise_mlp.parameters()).device
        dtype = next(self.position_wise_mlp.parameters()).dtype
        
        # Apply transformation to label tokens
        label_embeds = self.embed_module.weight[self.label_token_ids].to(device=device, dtype=dtype)
        transformed_embeds = self.position_wise_mlp(label_embeds)
        transformed_embeds = label_embeds + transformed_embeds  # residual connection
        
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
        
        if hasattr(self, 'batch_counter'):
            self.batch_counter += 1
        else:
            self.batch_counter = 0
        
        # First batch logging
        if self.batch_counter == 0:
            if speech_embeds is not None:
                if isinstance(speech_embeds, tuple):
                    # For SQA dataset with question and document embeddings
                    q_embeds, d_embeds = speech_embeds
                    logging.info("SQA Speech data detected and processed")
                    logging.info(f"Question embeddings device: {q_embeds.device}")
                    logging.info(f"Question embeddings range: {q_embeds.min():.3f} to {q_embeds.max():.3f}")
                    logging.info(f"Document embeddings device: {d_embeds.device}")
                    logging.info(f"Document embeddings range: {d_embeds.min():.3f} to {d_embeds.max():.3f}")
                else:
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
        
        # Important: Get transformed embeddings if in training mode
        if self.training and self.label_token_ids:
            # Get transformed embeddings for all tokens
            logging.info("Using transformed embeddings in forward pass")
            transformed_matrix = self.get_transformed_embeddings()
            
            # Use these for the target embeddings
            # This is the key step that ensures MLP gradients flow
            if self.use_lora:
                original_weight = self.llama_model.model.model.embed_tokens.weight
                self.llama_model.model.model.embed_tokens.weight = transformed_matrix
                target_embeds = self.llama_model.model.model.embed_tokens(target_tokens.input_ids)
                self.llama_model.model.model.embed_tokens.weight = original_weight
            else:
                original_weight = self.llama_model.model.embed_tokens.weight
                self.llama_model.model.embed_tokens.weight = transformed_matrix
                target_embeds = self.llama_model.model.embed_tokens(target_tokens.input_ids)
                self.llama_model.model.embed_tokens.weight = original_weight
        else:
            # Just use the original embeddings during inference
            target_embeds = self.embed_module(target_tokens.input_ids)
        
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
        
        logging.info(f"Forward pass took {time.time() - start_time:.2f} seconds")
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
            if is_sqa:
                # Process question and document spectrograms
                q_spectrograms = to_device_and_dtype(samples["question_spectrogram"])
                d_spectrograms = to_device_and_dtype(samples["document_spectrogram"])
                q_padding_mask = to_device_and_dtype(samples.get("question_padding_mask"))
                d_padding_mask = to_device_and_dtype(samples.get("document_padding_mask"))
                q_raw_wav = to_device_and_dtype(samples.get("question_raw_wav"))
                d_raw_wav = to_device_and_dtype(samples.get("document_raw_wav"))
                
                # Encode question and document
                q_embeds, q_atts = self.encode_speech(q_spectrograms, q_raw_wav, q_padding_mask)
                d_embeds, d_atts = self.encode_speech(d_spectrograms, d_raw_wav, d_padding_mask)
                
                # Combine as tuple
                speech_embeds = (q_embeds, d_embeds)
                speech_atts = (q_atts, d_atts)
            else:
                # Standard speech processing
                spectrograms = to_device_and_dtype(samples["spectrogram"])
                padding_mask = to_device_and_dtype(samples.get("padding_mask"))
                raw_wav = to_device_and_dtype(samples.get("raw_wav"))
                
                speech_embeds, speech_atts = self.encode_speech(spectrograms, raw_wav, padding_mask)
        
        # Process examples
        example_embeds = example_atts = None
        if has_examples:
            if is_sqa:
                # Process SQA examples
                example_q_specs = to_device_and_dtype(samples["example_question_spectrograms"])
                example_d_specs = to_device_and_dtype(samples["example_document_spectrograms"])
                example_q_padding = to_device_and_dtype(samples.get("example_question_padding_masks"))
                example_d_padding = to_device_and_dtype(samples.get("example_document_padding_masks"))
                example_q_raw = to_device_and_dtype(samples.get("example_question_raw_wavs"))
                example_d_raw = to_device_and_dtype(samples.get("example_document_raw_wavs"))
                
                # Initialize lists to store example embeddings
                batch_example_embeds = []
                batch_example_atts = []
                
                # Process each batch
                for b in range(len(example_q_specs)):
                    example_count = len(example_q_specs[b])
                    batch_embeds = []
                    batch_atts = []
                    
                    # Process each example in this batch
                    for i in range(example_count):
                        q_spec = example_q_specs[b][i]
                        d_spec = example_d_specs[b][i]
                        
                        # Get padding masks and raw wavs if available
                        q_pad = example_q_padding[b][i] if example_q_padding is not None else None
                        d_pad = example_d_padding[b][i] if example_d_padding is not None else None
                        q_raw = example_q_raw[b][i] if example_q_raw is not None else None
                        d_raw = example_d_raw[b][i] if example_d_raw is not None else None
                        
                        # Encode example speech
                        q_embed, q_att = self.encode_speech(q_spec.unsqueeze(0), q_raw, q_pad)
                        d_embed, d_att = self.encode_speech(d_spec.unsqueeze(0), d_raw, d_pad)
                        
                        # Add to batch
                        batch_embeds.append((q_embed.squeeze(0), d_embed.squeeze(0)))
                        batch_atts.append((q_att.squeeze(0), d_att.squeeze(0)))
                    
                    batch_example_embeds.append(batch_embeds)
                    batch_example_atts.append(batch_atts)
                
                example_embeds = batch_example_embeds
                example_atts = batch_example_atts
            else:
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
                if is_sqa:
                    # First handle the examples with Question{i} and Document{i}
                    for i in range(max_examples):
                        q_marker = f"<Question{i}>"
                        d_marker = f"<Document{i}>"
                        
                        if q_marker in suffix and d_marker in suffix:
                            before_d, rest = suffix.split(d_marker, 1)
                            middle, after_q = rest.split(q_marker, 1)
                            parts.extend([before_d, middle])
                            suffix = after_q
                            
                            if self.batch_counter == 0:
                                logging.info(f"After processing example {i}, parts list length: {len(parts)}")

                else:
                    # Original example handling
                    for i in range(max_examples):
                        example_marker = f"<Example{i}>"
                        if example_marker in suffix:
                            before_example, after_example = suffix.split(example_marker, 1)
                            parts.append(before_example)
                            suffix = after_example
                        else:
                            parts.append("")

            # Split for main speech input if speech placeholder exists
            if "<Question>" in suffix:
                    before_d, rest = suffix.split("<Document>", 1)
                    middle, after_q = rest.split("<Question>", 1)
                    parts.extend([before_d, middle])
                    suffix = after_q
            elif self.speech_placeholder in suffix:
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
                part_embeds.append(part_embed)
                
            part_atts = [tokens['attention_mask'].squeeze(0) for tokens in tokenized_parts]
            
            # Combine embeddings
            combined_embeds, combined_atts = [], []
            
            # Add text parts and examples
            if is_sqa:
                # Handle SQA dataset
                for i in range(max_examples):
                    combined_embeds.append(part_embeds[2*i])
                    combined_atts.append(part_atts[2*i])
                    
                    if example_embeds is not None:
                        if b < len(example_embeds) and i < len(example_embeds[b]):
                            q_embed, d_embed = example_embeds[b][i]
                            q_att, d_att = example_atts[b][i]
                            combined_embeds.extend([d_embed, part_embeds[2*i+1], q_embed])
                            combined_atts.extend([d_att, part_atts[2*i+1], q_att])
                    
                # Add final question and document embeddings
                if embeds is not None:  # Speech mode
                    q_embeds, d_embeds = embeds
                    q_atts, d_atts = atts
                    
                    combined_embeds.extend([part_embeds[-3], d_embeds[b], part_embeds[-2], q_embeds[b], part_embeds[-1]])
                    combined_atts.extend([part_atts[-3], d_atts[b], part_atts[-2], q_atts[b], part_atts[-1]])
                else:
                    combined_embeds.extend([part_embeds[-3], part_embeds[-2], part_embeds[-1]])
                    combined_atts.extend([part_atts[-3],part_atts[-2], part_atts[-1]])
            else:
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
        
    @classmethod
    def from_config(cls, config):
        """Create MLPSalmonn from configuration dictionary"""
        # Extract MLP-specific settings
        label_tokens = config.pop("label_tokens", None)
        hidden_dim = config.pop("hidden_dim", None)
        dropout = config.pop("dropout", 0.1)
        freeze_base = config.pop("freeze_base", True)
        
        # Save LoRA settings but disable it initially
        use_lora = config.pop("lora", True)
        lora_rank = config.pop("lora_rank", 8)
        lora_alpha = config.pop("lora_alpha", 16)
        lora_dropout = config.pop("lora_dropout", 0.05)
        
        # Create base SALMONN model without LoRA
        config["lora"] = False
        salmonn_model = SALMONN.from_config(config)
        
        # Create MLPSalmonn instance
        model = cls.__new__(cls)
        
        # Initialize with the base model's attributes
        for key, val in vars(salmonn_model).items():
            setattr(model, key, val)
        
        # Initialize additional attributes
        model.batch_counter = 0
        model.speech_placeholder = "<SpeechHere>"
        model.label_tokens = label_tokens or []
        model.label_token_ids = []
        model.use_lora = use_lora
        model.lora_rank = lora_rank
        model.lora_alpha = lora_alpha
        model.lora_dropout = lora_dropout
        
        # Get embedding module and dimensions
        if hasattr(model.llama_model, 'model'):
            if hasattr(model.llama_model.model, 'model'):
                model.embed_module = model.llama_model.model.model.embed_tokens
            else:
                model.embed_module = model.llama_model.model.embed_tokens
        else:
            model.embed_module = model.llama_model.embed_tokens
        
        model.embed_dim = model.embed_module.weight.shape[1]
        model.hidden_dim = hidden_dim or model.embed_dim
        
        # Create the MLP for embedding transformation
        model.position_wise_mlp = nn.Sequential(
            nn.Linear(model.embed_dim, model.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model.hidden_dim, model.embed_dim)
        )
        
        # Convert label tokens to token IDs
        if model.label_tokens:
            for token in model.label_tokens:
                token_id = model.llama_tokenizer.convert_tokens_to_ids(token)
                if token_id != model.llama_tokenizer.unk_token_id:
                    model.label_token_ids.append(token_id)
                else:
                    logging.warning(f"Label token '{token}' not found in vocabulary")
            logging.info(f"Initialized with {len(model.label_token_ids)} label tokens")
        
        # Move MLP to appropriate device
        device = next(model.parameters()).device
        model.position_wise_mlp.to(device)
        
        # Freeze base model if requested
        if freeze_base:
            for name, param in model.named_parameters():
                if not name.startswith('position_wise_mlp'):
                    param.requires_grad = False
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"Model has {trainable_params:,} trainable MLP parameters")
        
        # Apply LoRA if needed
        if model.use_lora:
            logging.info("Applying LoRA adapters AFTER MLP setup")
            model.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=model.lora_rank, 
                lora_alpha=model.lora_alpha, 
                lora_dropout=model.lora_dropout,
            )
            model.llama_model = get_peft_model(model.llama_model, model.peft_config)
            model.llama_model.print_trainable_parameters()
        
        return model