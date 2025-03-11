import sys
import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor

# Import the original SALMONN model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from SALMONN.models.salmonn_org import SALMONN

# Import our base model
from .base_model import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CustomSALMONN(BaseModel):
    """
    Custom implementation of SALMONN model that inherits from BaseModel.
    Provides a standardized interface for training and inference.
    """
    def __init__(self, 
                 llama_path: str = "lmsys/vicuna-13b-v1.1",
                 whisper_path: str = "openai/whisper-large-v2",
                 beats_path: str = "/data2/neeraja/neeraja/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
                 use_speech_Qformer: bool = True,
                 freeze_whisper: bool = True,
                 freeze_beats: bool = True,
                 freeze_speech_QFormer: bool = False,
                 num_speech_query_token: int = 1,
                 window_level_Qformer: bool = True,
                 second_per_window: float = 0.333333,
                 second_stride: float = 0.333333,
                 speech_llama_proj_model: str = "",
                 freeze_speech_llama_proj: bool = False,
                 lora: bool = True,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 multi_prompt: bool = False,
                 prompt_path: str = "",
                 prompt_template: str = "",
                 max_txt_len: int = 128,
                 end_sym: str = "</s>",
                 low_resource: bool = False,
                 ckpt_path: str = "/data2/neeraja/neeraja/salmonn_v1.pth",  # Default checkpoint path
                 device=None, 
                 use_fp16: bool = False):
        """
        Initialize the CustomSALMONN model.
        """
        # Initialize the BaseModel
        super().__init__(device=device, use_fp16=use_fp16)
        
        # Initialize the processor for compatibility with the ICL framework
        logging.info(f"Initializing WhisperFeatureExtractor from {whisper_path}")
        self.input_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        
        # Create config for SALMONN
        salmonn_config = {
            "llama_path": llama_path,
            "whisper_path": whisper_path,
            "beats_path": beats_path,
            "use_speech_Qformer": use_speech_Qformer,
            "freeze_whisper": freeze_whisper,
            "freeze_beats": freeze_beats,
            "freeze_speech_QFormer": freeze_speech_QFormer,
            "num_speech_query_token": num_speech_query_token,
            "window_level_Qformer": window_level_Qformer,
            "second_per_window": second_per_window,
            "second_stride": second_stride,
            "speech_llama_proj_model": speech_llama_proj_model,
            "freeze_speech_llama_proj": freeze_speech_llama_proj,
            "lora": lora,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "multi_prompt": multi_prompt,
            "prompt_path": prompt_path,
            "prompt_template": prompt_template,
            "max_txt_len": max_txt_len,
            "end_sym": end_sym,
            "low_resource": low_resource,
            "ckpt": ckpt_path
        }
    
        # Initialize the SALMONN model using from_config
        self.salmonn = SALMONN.from_config(salmonn_config)

        # Move model to the specified device
        self.salmonn.to(self.device)

        # Store model attributes
        self.prompt_template = prompt_template
        self.lora = lora
        
        # Initialize model settings
        self.speech_tag_start = "<Speech>"
        self.speech_tag_end = "</Speech>"
        self.speech_placeholder = "<SpeechHere>"
        
        # Get references to SALMONN components for easier access
        self.llama_model = self.salmonn.llama_model
        self.llama_tokenizer = self.salmonn.llama_tokenizer

        self.batch_counter = 0
        
        logging.info(f"Initialized CustomSALMONN with model path: {llama_path}")

    def custom_prompt_wrap(self, embeds, atts, prompts, num_examples=None, example_embeds=None, example_atts=None):
        """
        Wraps speech embeddings with text prompts and examples.
        """
        # Infer device from model parameters
        device = next(self.parameters()).device
        
        batch_size = len(prompts)
        # logging.info(f"Batch size: {batch_size}")
        max_examples = num_examples.max().item() if num_examples is not None else 0
        # logging.info(f"Max examples: {max_examples}")
        batch_embeds, batch_atts = [], []
        
        for b in range(batch_size):
            parts = []
            suffix = prompts[b]
            
            # Split prompt for examples
            if max_examples > 0 and example_embeds is not None:
                for i in range(max_examples):
                    example_marker = f"<Example{i}>"
                    if example_marker in suffix:
                        before_example, after_example = suffix.split(example_marker, 1)
                        parts.append(before_example)
                        suffix = after_example
                    else:
                        parts.append("")

            # Split for main speech input if speech placeholder exists
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
            
            part_embeds = [
                self.llama_model.model.embed_tokens(tokens['input_ids'].squeeze(0))
                if not self.lora else 
                self.llama_model.model.model.embed_tokens(tokens['input_ids'].squeeze(0))
                for tokens in tokenized_parts
            ]
            part_atts = [tokens['attention_mask'].squeeze(0) for tokens in tokenized_parts]
            
            # Combine embeddings
            combined_embeds, combined_atts = [], []
            
            # Add text parts and examples
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
                if self.batch_counter == 0:
                    logging.info(f"Speech embeds shape: {embeds[b].shape}")
                    logging.info(f"Part embeds shapes: {part_embeds[-2].shape}, {part_embeds[-1].shape}")
                combined_embeds.extend([part_embeds[-2], embeds[b], part_embeds[-1]])
                combined_atts.extend([part_atts[-2], atts[b], part_atts[-1]])
                if self.batch_counter == 0:
                    logging.info(f"Combined embeds length: {len(combined_embeds)}")
                    logging.info(f"Individual shapes in combined_embeds: {[e.shape for e in combined_embeds]}")
            else:  # Text-only mode
                # For text-only, concatenate the final parts without speech embeddings
                if len(part_embeds) >= 2:
                    combined_embeds.extend([part_embeds[-2], part_embeds[-1]])
                    combined_atts.extend([part_atts[-2], part_atts[-1]])
                else:
                    # If there's only one part (e.g., no speech placeholder)
                    combined_embeds.append(part_embeds[-1])
                    combined_atts.append(part_atts[-1])
                
                if self.batch_counter == 0:
                    logging.info(f"Text-only mode: combined {len(combined_embeds)} parts")

            if self.batch_counter == 0:
                logging.info("\n=== Attention Mask Debug ===")
                for i, att in enumerate(combined_atts):
                    logging.info(f"Attention mask {i} shape: {att.shape}")

            # Concatenate all parts
            batch_embeds.append(torch.cat(combined_embeds, dim=0))
            batch_atts.append(torch.cat(combined_atts, dim=0))
        
        # Only log embedding stats for speech mode
        if self.batch_counter == 0 and embeds is not None:
            for i, embed in enumerate(combined_embeds):
                logging.info(f"Part {i} stats:")
                logging.info(f"Mean: {embed.mean().item():.3f}")
                logging.info(f"Std: {embed.std().item():.3f}")
                logging.info(f"Min: {embed.min().item():.3f}")
                logging.info(f"Max: {embed.max().item():.3f}")
                # Log what this part represents
                if i % 2 == 1:
                    logging.info("(Speech embedding)")
                else:
                    logging.info("(Text part)")
        
        return torch.stack(batch_embeds, dim=0), torch.stack(batch_atts, dim=0)

    def get_speech_embeddings(self, samples):
        """
        Generates speech embeddings for both training and inference.
        """
        start_time = time.time()
        
        # Add detailed logging for first batch
        if self.batch_counter == 0:
            if samples["spectrogram"] is not None:
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
        
        # Check if we have valid speech data
        has_main_speech = "spectrogram" in samples and samples["spectrogram"] is not None and samples["spectrogram"].numel() > 1
        has_examples = "example_spectrograms" in samples and samples["example_spectrograms"] is not None and samples["example_spectrograms"].numel() > 1
        
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
            if self.use_fp16 and tensor.dtype == torch.float32:
                tensor = tensor.to(dtype=torch.float16)
            return tensor

        # Process main inputs - only if we have valid main speech
        speech_embeds = speech_atts = None
        if has_main_speech:
            samples["spectrogram"] = to_device_and_dtype(samples["spectrogram"])
            samples["raw_wav"] = to_device_and_dtype(samples.get("raw_wav"))
            if "padding_mask" in samples:
                samples["padding_mask"] = samples["padding_mask"].to(self.device)
            
            # Encode main speech
            speech_embeds, speech_atts = self.encode_speech(
                spectrogram=samples["spectrogram"],
                raw_wav=samples.get("raw_wav"),
                audio_padding_mask=samples.get("padding_mask")
            )
            
            if self.batch_counter == 0:
                logging.info("\n=== Speech Embeddings Debug ===")
                logging.info(f"Main speech embeddings shape: {speech_embeds.shape}")
                logging.info(f"Main speech attention mask shape: {speech_atts.shape}")
                logging.info("\nFirst 5 values of speech embeddings:")
                embed_flat = speech_embeds[0].flatten()  # First batch item
                logging.info(f"{embed_flat[:5].tolist()}")
                logging.info("\nStats for speech embeddings:")
                logging.info(f"Mean: {speech_embeds.mean().item():.3f}")
                logging.info(f"Std: {speech_embeds.std().item():.3f}")
                logging.info(f"Min: {speech_embeds.min().item():.3f}")
                logging.info(f"Max: {speech_embeds.max().item():.3f}")
                logging.info(f"Has NaN after encode: {torch.isnan(speech_embeds).any().item()}")

                logging.info(f"Spectrogram dtype: {samples['spectrogram'].dtype}")
                logging.info(f"Raw WAV dtype: {samples['raw_wav'].dtype}")
                logging.info(f"Padding mask dtype: {padding_mask.dtype}")
                 # Add attention mask stats
                logging.info("\n=== Attention Mask Stats ===")
                logging.info(f"Attention mask first 5 values: {speech_atts[0, :5].tolist()}")
                logging.info(f"Non-masked length: {speech_atts.sum().item()}")
                logging.info(f"Masked percentage: {(1 - speech_atts.float().mean()) * 100:.2f}%")
        
        # Process examples if present - regardless of whether main speech exists
        example_embeds = example_atts = None
        if has_examples:
            # Process example inputs
            samples["example_spectrograms"] = to_device_and_dtype(samples["example_spectrograms"])
            samples["example_wavs"] = to_device_and_dtype(samples.get("example_wavs"))
            if "example_padding_masks" in samples:
                samples["example_padding_masks"] = samples["example_padding_masks"].to(self.device)
                
            B, E = samples["example_spectrograms"].shape[:2]
            all_example_embeds, all_example_atts = [], []
            
            if self.batch_counter == 0:
                logging.info(f"=== Example Processing Debug ===")
                logging.info(f"Number of examples: {E}")
                logging.info(f"Example spectrograms shape: {samples['example_spectrograms'].shape}")
            
            for b in range(B):
                batch_embeds, batch_atts = [], []
                for e in range(E):
                    if self.batch_counter == 0:
                        logging.info(f"\nProcessing example {e}:")
                        logging.info(f"Example spectrogram shape: {samples['example_spectrograms'][b, e].shape}")
                        logging.info(f"Has NaN before encode: {torch.isnan(samples['example_spectrograms'][b, e]).any().item()}")
                    
                    single_spec = samples["example_spectrograms"][b, e].unsqueeze(0)
                    single_wav = samples["example_wavs"][b, e].unsqueeze(0) if samples["example_wavs"] is not None else None
                    single_padding = samples["example_padding_masks"][b, e].unsqueeze(0) if "example_padding_masks" in samples else None
                    
                    example_embed, example_att = self.encode_speech(
                        spectrogram=single_spec,
                        raw_wav=single_wav,
                        audio_padding_mask=single_padding
                    )
                    
                    if self.batch_counter == 0:
                        logging.info(f"Example {e} embeddings shape: {example_embed.shape}")
                        logging.info(f"Has NaN after encode: {torch.isnan(example_embed).any().item()}")
                        if torch.isnan(example_embed).any():
                            logging.info(f"NaN percentage: {torch.isnan(example_embed).float().mean()*100:.2f}%")
                    
                    example_att = example_att.squeeze(0)
                    example_embed = example_embed.squeeze(0)
                    batch_embeds.append(example_embed)
                    batch_atts.append(example_att)

                    if self.batch_counter == 0 and b == 0 and e == 0:
                        logging.info(f"Example embeddings shape: {example_embed.shape}")
                        logging.info(f"Example attention mask shape: {example_att.shape}")
                
                all_example_embeds.append(batch_embeds)
                all_example_atts.append(batch_atts)
                     
            example_embeds = all_example_embeds
            example_atts = all_example_atts
        
        logging.info(f"Speech embedding generation took {time.time() - start_time:.2f} seconds")
        
        return speech_embeds, speech_atts, example_embeds, example_atts

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        """
        Encode speech inputs using the SALMONN speech encoder.
        """
        return self.salmonn.encode_speech(
            spectrogram=spectrogram,
            raw_wav=raw_wav,
            audio_padding_mask=audio_padding_mask
        )

    def forward(self, samples):
        """
        Forward pass for training.
        """
        start_time = time.time()
        
        # Process speech embeddings
        speech_embeds, speech_atts, example_embeds, example_atts = self.get_speech_embeddings(samples)
        
        if self.batch_counter == 0:
            if speech_embeds is not None:
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
        
        target_embeds = self.llama_model.model.embed_tokens(target_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(target_tokens.input_ids)
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
        
        logging.info(f"Forward pass took {time.time() - gen_start_time:.2f} seconds")
        self.batch_counter += 1
        return {"loss": outputs.loss, "logits": outputs.logits, "labels": labels}

    def generate_output(self, samples):
        """
        Generate predictions for speech or text input.
        """
        start_time = time.time()
        
        # Process speech embeddings
        speech_embeds, speech_atts, example_embeds, example_atts = self.get_speech_embeddings(samples)
        
        if self.batch_counter == 0:
            if speech_embeds is not None:
                logging.info("Speech data detected and processed")
                logging.info(f"Speech embeddings device: {speech_embeds.device}")
                logging.info(f"Speech embeddings range: {speech_embeds.min():.3f} to {speech_embeds.max():.3f}")
            else:
                logging.info("No speech data detected, using text-only mode")
        
        if self.batch_counter == 0:
            logging.info(f"Prompt example:\n{samples['prompt'][0]}")
        
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
        
        # Generate with LLaMA
        with torch.inference_mode():
            outputs = self.llama_model.generate(
                inputs_embeds=wrapped_embeds,
                attention_mask=wrapped_atts,
                max_new_tokens=samples.get("max_new_tokens", 10),
                num_beams=samples.get("num_beams", 1),
                do_sample=samples.get("do_sample", False),
                min_length=samples.get("min_length", 1),
                top_p=samples.get("top_p", 0.9),
                repetition_penalty=samples.get("repetition_penalty", 1.0),
                length_penalty=samples.get("length_penalty", 1.0),
                temperature=samples.get("temperature", 0.8),
                pad_token_id=self.llama_tokenizer.pad_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                return_dict_in_generate=False,  # Faster, since we don't need extra outputs
                output_scores=False,  
            )
        
        gen_time = time.time() - gen_start_time
        
        if self.batch_counter == 0:
            logging.info(f"Raw output tokens: {outputs[0].tolist()}")
            logging.info(f"Tokenizer vocab size: {self.llama_tokenizer.vocab_size}")
            first_pred = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"First raw prediction: {first_pred}")
        
        # Decode predictions
        predictions = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        total_time = time.time() - start_time
        if self.batch_counter == 0:
            logging.info(f"Time breakdown - Generation: {gen_time:.2f}s, Total: {total_time:.2f}s")
        
        self.batch_counter += 1
        logging.info(f"Generation took {time.time() - start_time:.2f} seconds")
        return predictions

    @classmethod
    def from_config(cls, config):
        """
        Create a CustomSALMONN instance from a configuration dictionary.
        This is a simple factory method that passes the config to the constructor.
        """
        # We don't need to duplicate the parameter handling logic here
        # Just pass the config directly to the constructor
        return cls(**config)