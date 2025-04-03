import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging
from dataset_config import DatasetType, DATASET_CONFIGS, get_swapped_config, apply_label_mapping, DatasetSplit
import random
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_from_disk
import time
import torch.nn.functional as F
from utils.generate_fewshots import convert_ner_to_dict

class BaseSalmonDataset(Dataset):
    def __init__(self, dataset_type: DatasetType, dataset, wav_processor, input_mode='speech_and_text', 
                 fewshot_mode='text', num_examples=5, random_examples=False, split=DatasetSplit.TEST, 
                 model_type="salmonn", run_name=""):
        self.dataset_type = dataset_type
        self.config = DATASET_CONFIGS[dataset_type]
        self.dataset = dataset
        self.wav_processor = wav_processor
        self.input_mode = input_mode
        self.fewshot_mode = fewshot_mode
        self.num_examples = num_examples
        self.random_examples = random_examples
        self.split = split
        self.model_type = model_type
        self.run_name = run_name
        
        if input_mode not in ['speech_only', 'speech_and_text', 'text_only']:
            raise ValueError("input_mode must be one of 'speech_only', 'speech_and_text', or 'text_only'")
        if fewshot_mode not in ['text', 'speech']:
            raise ValueError("fewshot_mode must be either 'text' or 'speech'")

        # Create index lookup dictionary
        self.index_lookup = {}
        for i, item in enumerate(dataset):
            if dataset_type in [DatasetType.HVB, DatasetType.HVB_GREEK, DatasetType.HVB_SWAP]:
                key = f"{item['issue_id']}_{item['utt_index']}"
            else:
                key = str(item['index'])
            self.index_lookup[key] = i

        # Load audio lookup dataset if needed for speech fewshots
        self.audio_lookup = None
        self.audio_index_map = {}
        if fewshot_mode == 'speech':
            dataset_config = DATASET_CONFIGS[dataset_type]
            audio_lookup_path = dataset_config.get_audio_lookup_path(split)
            if audio_lookup_path:
                self.audio_lookup = load_from_disk(audio_lookup_path)
                # Create index mapping during initialization
                self.audio_index_map = {str(idx): i for i, idx in enumerate(self.audio_lookup['index'])}
                logging.info(f"Loaded audio lookup dataset for {split.value} from {audio_lookup_path}")

    def __len__(self):
        return len(self.dataset)

    def _process_speech(self, audio_data):
        """Process speech data into required format"""
        # Convert list to numpy array if needed
        if isinstance(audio_data['array'], list):
            import numpy as np
            array_data = np.array(audio_data['array'])
        else:
            array_data = audio_data['array']
        
        raw_wav = torch.from_numpy(array_data)
        spectrogram = self.wav_processor(raw_wav, sampling_rate=16000, return_tensors="pt")["input_features"]
        return {
            "raw_wav": raw_wav,
            "spectrogram": spectrogram.squeeze(0),
            "wav_length": len(raw_wav)
        }

    def _get_audio_by_index(self, index_str):
        """Fetch audio data using the index string"""
        if self.audio_lookup is not None:
            lookup_idx = self.audio_index_map.get(index_str)
            if lookup_idx is None:
                logging.warning(f"No matching audio found for index {index_str} in lookup dataset")
                return None
            return self.audio_lookup[lookup_idx]['audio']
        else:
            # Use original method for training
            dataset_idx = self.index_lookup.get(index_str)
            if dataset_idx is None:
                logging.warning(f"No matching audio found for index {index_str}, skipping this example")
                return None
            return self.dataset[dataset_idx]['audio']

    def _select_examples(self, few_shot_examples):
        """Helper method to select examples consistently"""
        if self.random_examples:
            num_examples = min(self.num_examples, len(few_shot_examples))
            return random.sample(few_shot_examples, num_examples) if num_examples > 0 else []
        return few_shot_examples[:self.num_examples]

    def _create_prompt(self, item, selected_examples):
        # Common function to create input section (for SALMONN)
        def get_input_section(model_type):
            if self.input_mode == 'speech_only':
                return "<|audio_bos|><|AUDIO|><|audio_eos|>" if model_type == "qwen2" else "<Speech><SpeechHere></Speech>"
            elif self.input_mode == 'text_only':
                return f"Text: {item[self.config.text_key]}"
            else:  # speech_and_text
                speech_tag = "<|audio_bos|><|AUDIO|><|audio_eos|>" if model_type == "qwen2" else "<Speech><SpeechHere></Speech>"
                return f"{speech_tag}\nTranscript: {item[self.config.text_key]}"

        def format_label(example):
            if self.dataset_type == DatasetType.VOXPOPULI:
                # Filter out keys with None values
                # label_dict = {k: v for k, v in example['label'].items() if v}  # This will remove None values
                label_dict = [k for k, v in example['label'].items() if v]
                # return str(label_dict)  # Dictionary format for NER
                return ', '.join(label_dict) if label_dict else 'None'

            elif isinstance(example['label'], list):
                return ', '.join(example['label'])  # HVB format
            return example['label']  # VoxCeleb format

        if self.model_type == "qwen2":
            # Create simplified format for Qwen2
            conversation = [
                {'role': 'system', 'content': self.prompt_template if hasattr(self, 'prompt_template') else self.config.prompt_template}
            ]

            user_content =[]

            if selected_examples:
                for example in selected_examples:
                    user_content = [{"type": "text", "text": "Here are few examples to learn from:\n"}]
                    if self.fewshot_mode == 'speech':
                        # For speech examples, include both audio and text
                        user_content.extend([
                            {"type": "audio", "audio_url": "dummy_url"},
                            {"type": "text", "text": f"Label: {format_label(example)}\n"}
                        ])
                    else:  # text mode
                        user_content.extend([
                            {"type": "text", "text": f"Text: {example['text']}\n"},
                            {"type": "text", "text": f"Label: {format_label(example)}\n"}
                        ])

            user_content.append({"type": "text", "text": "\nNow analyze this input:\n"})
            # Add current input
            
            if self.input_mode in ['speech_only', 'speech_and_text']:
                user_content.append({"type": "audio", "audio_url": "dummy_url"})
            if self.input_mode in ['text_only', 'speech_and_text']:
                user_content.append({"type": "text", "text": item[self.config.text_key]})
            
            conversation.append({
                "role": "user", 
                "content": user_content
            })

            return conversation
            
        is_target_model = "list" in self.run_name
        # Original SALMONN prompt creation
        if selected_examples:
            if self.fewshot_mode == 'speech':
                examples_text = "\n\n".join([
                    f"<Speech><Example{i}></Speech>\n"
                    f"Output: {format_label(example)}"
                    for i, example in enumerate(selected_examples)
                ])
            else:  # text mode
                examples_text = "\n\n".join([
                    f"Text: {example['text']}\n"
                    f"Output: {format_label(example)}"
                    for example in selected_examples
                ])

            prompt = f"""{self.prompt_template if hasattr(self, 'prompt_template') else self.config.prompt_template}

Here are few examples to learn from:
{examples_text}

Now analyze this input:
{get_input_section("salmonn")}
Output:"""
        else:
            prompt = f"""{self.prompt_template if hasattr(self, 'prompt_template') else self.config.prompt_template}

Now analyze this input:
{get_input_section("salmonn")}
Output:"""

        return prompt

    def __getitem__(self, idx):
        pass

class FinetuneDataset(BaseSalmonDataset):
    def __init__(self, dataset_type: DatasetType, dataset, wav_processor, input_mode='speech_and_text', 
                 balance_strategy=None, fewshot_mode='text', model_type="salmonn"):
        super().__init__(dataset_type, dataset, wav_processor, input_mode, 
                        fewshot_mode=fewshot_mode, random_examples=True, split=DatasetSplit.TRAIN, model_type=model_type)
        
        self.balance_strategy = balance_strategy
        self._setup_class_weights()
        
        # Get dataset config and handle swapped configurations
        if dataset_type in [DatasetType.VOXCELEB_SWAP, DatasetType.HVB_SWAP]:
            self.prompt_template, self.label_mapping = get_swapped_config(dataset_type)
        else:
            self.prompt_template = self.config.prompt_template
            self.label_mapping = self.config.label_mapping if hasattr(self.config, 'label_mapping') else None

    def _setup_class_weights(self):
        # Calculate class weights and sampling weights
        label_counts = {}
        for item in self.dataset:
            label = item[self.config.completion_key]
            # Handle list labels (for HVB datasets)
            if isinstance(label, list):
                label = ', '.join(sorted(label))
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total_samples = sum(label_counts.values())
        self.class_weights = {
            label: total_samples / (len(label_counts) * count)
            for label, count in label_counts.items()
        }
        
        if self.balance_strategy == 'weighted':
            self.sample_weights = [
                self.class_weights[
                    ', '.join(sorted(self.dataset[i][self.config.completion_key])) 
                    if isinstance(self.dataset[i][self.config.completion_key], list)
                    else self.dataset[i][self.config.completion_key]
                ]
                for i in range(len(self.dataset))
            ]

    def get_sampler(self):
        if self.balance_strategy == 'weighted':
            return torch.utils.data.WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.dataset),
                replacement=True
            )
        return None

    def _create_prompt(self, item, selected_examples):
        """Override _create_prompt to handle label mapping for both VOXCELEB_SWAP and HVB_GREEK"""
        if self.dataset_type in [DatasetType.VOXCELEB_SWAP, DatasetType.HVB_SWAP, DatasetType.HVB_GREEK]:
            # Map few-shot examples if they exist
            if selected_examples:
                selected_examples = apply_label_mapping(selected_examples, self.label_mapping)
        
        # Use parent class's _create_prompt with potentially modified examples
        return super()._create_prompt(item, selected_examples)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Process examples
        few_shot_examples = item.get('few_shot_examples', [])
        selected_examples = self._select_examples(few_shot_examples)
        
        # Handle completion/label mapping
        completion = item[self.config.completion_key]
        if isinstance(completion, list):
            if self.label_mapping:
                completion = [self.label_mapping.get(label, label) for label in completion]
            completion = ', '.join(completion)
        elif self.label_mapping:
            completion = self.label_mapping.get(completion, completion)

        if self.model_type == "qwen2":
            conversation = self._create_prompt(item, selected_examples)
            
            # Get prompt text using chat template
            prompt = self.wav_processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Calculate prompt length by tokenizing prompt separately
            prompt_tokens = self.wav_processor.tokenizer(prompt, return_tensors="pt").input_ids
            prompt_length = prompt_tokens.size(1)
            
            # Tokenize completion and eos separately
            completion_with_eos = f"{completion}{self.wav_processor.tokenizer.eos_token}"
            
            # Create full text for training
            full_text = f"{prompt}{completion_with_eos}"
            
            # Process audios
            audios = []
            if self.fewshot_mode == 'speech':
                for example in selected_examples:
                    audio_data = self._get_audio_by_index(example['index'])
                    if audio_data is not None:
                        example_audio = audio_data['array'] 
                        audios.append(example_audio)

            main_audio = item["audio"]["array"]
            audios.append(main_audio)
            
            # Process everything together
            inputs = self.wav_processor(
                text=full_text,
                audios=audios,
                return_tensors="pt",
                sampling_rate=16000
            )
            
            inputs.input_features = inputs.input_features.to(torch.float16)
            
            return {
                "input_ids": inputs.input_ids.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0),
                "input_features": inputs.input_features.squeeze(0),
                "feature_attention_mask": inputs.feature_attention_mask.squeeze(0),
                "prompt_length": prompt_length,
                "completion": completion,
                "text": item[self.config.text_key],
                "prompt": prompt
            }

        # For SALMONN model:
        
        # Process main audio only if not text_only mode
        if self.input_mode != 'text_only':
            main_speech = self._process_speech(item["audio"])
        else:
            # For text-only mode, return None instead of dummy tensors
            main_speech = {
                "spectrogram": None,
                "raw_wav": None,
                "wav_length": 0
            }
        
        # Process examples if in speech mode - regardless of input_mode
        examples_data = []
        if self.fewshot_mode == 'speech':
            for example in selected_examples:
                audio_data = self._get_audio_by_index(example['index'])
                if audio_data is not None:
                    examples_data.append(self._process_speech(audio_data))

        return {
            "spectrogram": main_speech["spectrogram"],
            "raw_wav": main_speech["raw_wav"],
            "wav_length": main_speech["wav_length"],
            "text": item[self.config.text_key],
            "prompt": self._create_prompt(item, selected_examples),
            "completion": completion,
            "num_examples": len(examples_data),
            "examples_speech": examples_data
        }

class InferenceDataset(BaseSalmonDataset):
    def __init__(self, dataset_type: DatasetType, dataset, wav_processor, num_examples=5, 
                 input_mode='speech_and_text', fewshot_mode='text', model_type="salmonn", run_name=""):
        super().__init__(dataset_type, dataset, wav_processor, input_mode, 
                        fewshot_mode=fewshot_mode, num_examples=num_examples, 
                        random_examples=False, split=DatasetSplit.TEST, 
                        model_type=model_type, run_name=run_name)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Process examples for both cases
        few_shot_examples = item.get('few_shot_examples', [])
        selected_examples = self._select_examples(few_shot_examples)
        
        # Create prompt
        
        
        if self.model_type == "qwen2":
            # Process with Qwen2 processor
            conversation = self._create_prompt(item, selected_examples)  # Now returns conversation dict
            
            # Apply chat template to get formatted text
            prompt = self.wav_processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )

            audios = []
            if self.fewshot_mode == 'speech':
                # First add few-shot example audios
                for example in selected_examples:
                    audio_data = self._get_audio_by_index(example['index'])
                    if audio_data is not None:
                        example_audio = audio_data['array'] 
                        audios.append(example_audio)

            main_audio = item["audio"]["array"]
            audios.append(main_audio)
            
            inputs = self.wav_processor(
                text=prompt,
                audios=audios,
                return_tensors="pt",
                sampling_rate=16000
            )
            
            # Convert to float16 immediately
            inputs.input_features = inputs.input_features.to(torch.float16)
            
            return {
                "input_ids": inputs.input_ids.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0),
                "input_features": inputs.input_features.squeeze(0),  # Already float16
                "feature_attention_mask": inputs.feature_attention_mask.squeeze(0),
                "prompt_length": len(inputs.input_ids[0]),
                "completion": item[self.config.completion_key],
                "text": item[self.config.text_key],
                "prompt": prompt
            }
        
        prompt = self._create_prompt(item, selected_examples)
        # Original SALMONN processing
        
        # Process main speech only if not in text-only mode
        if self.input_mode != 'text_only':
            main_speech = self._process_speech(item["audio"])
        else:
            # For text-only mode, return None instead of dummy tensors
            main_speech = {
                "spectrogram": None,
                "raw_wav": None,
                "wav_length": 0
            }
            
        # Process examples if in speech mode - regardless of input_mode
        examples_data = []
        if self.fewshot_mode == 'speech':
            for example in selected_examples:
                audio_data = self._get_audio_by_index(example['index'])
                if audio_data is not None:
                    examples_data.append(self._process_speech(audio_data))

        completion = item[self.config.completion_key]
        if self.dataset_type == DatasetType.VOXPOPULI:
            completion_dict = convert_ner_to_dict(item[self.config.text_key], completion)
            completion_dict = [k for k, v in completion_dict.items() if v]
            completion = ', '.join(completion_dict) if completion_dict else 'None'


        return {
            "spectrogram": main_speech["spectrogram"],
            "raw_wav": main_speech["raw_wav"],
            "wav_length": main_speech["wav_length"],
            "text": item[self.config.text_key],
            "prompt": prompt,
            "completion": completion,
            "num_examples": len(examples_data),
            "examples_speech": examples_data
        }

def collate_fn(batch):
    # Handle Qwen2 batch
    if "prompt_length" in batch[0]:  # Qwen2 model
        # Qwen2 processing remains unchanged
        if len(batch[0]["input_features"].shape) == 3:  # Has examples
            # Concatenate all input features and their masks
            return {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
                "input_features": torch.cat([item["input_features"] for item in batch]),
                "feature_attention_mask": torch.cat([item["feature_attention_mask"] for item in batch]),
                "prompt_length": torch.tensor([item["prompt_length"] for item in batch]),
                "completion": [item["completion"] for item in batch],
                "text": [item["text"] for item in batch],
                "prompt": [item["prompt"] for item in batch]
            }
        else:  # No examples, just stack normally
            return {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
                "input_features": torch.stack([item["input_features"] for item in batch]),
                "feature_attention_mask": torch.stack([item["feature_attention_mask"] for item in batch]),
                "prompt_length": torch.tensor([item["prompt_length"] for item in batch]),
                "completion": [item["completion"] for item in batch],
                "text": [item["text"] for item in batch],
                "prompt": [item["prompt"] for item in batch]
            }

    # Original SALMONN collate_fn code with text-only mode support
    
    # Get input_mode from args (not from batch)
    # This will be passed directly from the training script
    
    # Process speech data based on whether we have valid speech data
    has_valid_speech = all(item['spectrogram'] is not None for item in batch)
    
    if not has_valid_speech:
        # For text-only mode, create minimal placeholder tensors for main input
        wav_lengths = torch.zeros(len(batch), dtype=torch.long)
        raw_wavs_padded = torch.zeros((len(batch), 1), dtype=torch.float)  # Minimal tensor
        padding_mask = torch.ones((len(batch), 1), dtype=torch.bool)  # All masked
        spectrograms = torch.zeros((len(batch), 1, 1, 1), dtype=torch.float)  # Minimal tensor
    else:
        # Original speech processing for main input
        wav_lengths = torch.tensor([item['wav_length'] for item in batch])
        raw_wavs = [item['raw_wav'] for item in batch]
        raw_wavs_padded = pad_sequence(raw_wavs, batch_first=True, padding_value=0)
        padding_mask = torch.arange(raw_wavs_padded.size(1)).unsqueeze(0) >= wav_lengths.unsqueeze(1)
        spectrograms = torch.stack([item['spectrogram'] for item in batch])
    
    result = {
        "spectrogram": spectrograms,
        "raw_wav": raw_wavs_padded,
        "padding_mask": padding_mask,
        "prompt": [item['prompt'] for item in batch],
        "num_examples": torch.tensor([item['num_examples'] for item in batch])
        # No input_mode in result
    }

    # Add text field if available
    if 'text' in batch[0]:
        result['text'] = [item['text'] for item in batch]

    # Handle speech examples - independent of input_mode
    max_examples = max(item['num_examples'] for item in batch)
    
    if max_examples > 0:
        # Check if we have actual speech examples to process
        has_speech_examples = any(
            'examples_speech' in item and 
            len(item['examples_speech']) > 0 and
            item['examples_speech'][0]['spectrogram'] is not None
            for item in batch
        )
        
        if has_speech_examples:
            # Process examples with actual speech data
            example_specs = []
            example_wavs = []
            example_masks = []
            
            # Find max length for padding
            max_length = max(
                example['wav_length']
                for item in batch if 'examples_speech' in item
                for example in item['examples_speech'][:item['num_examples']]
            )
            
            # Process each batch item
            for item in batch:
                # Get actual examples and pad if needed
                examples = item.get('examples_speech', [])[:item['num_examples']]
                
                # Process actual examples
                batch_specs = []
                batch_wavs = []
                batch_masks = []
                
                for example in examples:
                    # Add spectrogram (no need to add channel dimension)
                    spec = example['spectrogram']
                    batch_specs.append(spec)
                    
                    # Pad and add waveform
                    wav = example['raw_wav']
                    if wav.size(0) < max_length:
                        wav = torch.nn.functional.pad(wav, (0, max_length - wav.size(0)), value=0)
                    batch_wavs.append(wav)
                    
                    # Create mask
                    mask = torch.arange(max_length, device=wav.device) >= example['wav_length']
                    batch_masks.append(mask)
                
                # Pad with zeros if needed
                while len(batch_specs) < max_examples:
                    # Use same shape as other spectrograms for padding
                    pad_spec = torch.zeros_like(batch_specs[0]) if batch_specs else torch.zeros((1, 80, 3000))
                    batch_specs.append(pad_spec)
                    batch_wavs.append(torch.zeros(max_length, device=wav.device))
                    batch_masks.append(torch.ones(max_length, device=wav.device, dtype=torch.bool))
                
                # Stack for this batch item
                example_specs.append(torch.stack(batch_specs))
                example_wavs.append(torch.stack(batch_wavs))
                example_masks.append(torch.stack(batch_masks))
            
            # Stack across batch dimension
            result['example_spectrograms'] = torch.stack(example_specs)
            result['example_wavs'] = torch.stack(example_wavs)
            result['example_padding_masks'] = torch.stack(example_masks)
        else:
            # No actual speech examples, create minimal placeholder tensors
            result['example_spectrograms'] = torch.zeros((len(batch), max_examples, 1, 1, 1), dtype=torch.float)
            result['example_wavs'] = torch.zeros((len(batch), max_examples, 1), dtype=torch.float)
            result['example_padding_masks'] = torch.ones((len(batch), max_examples, 1), dtype=torch.bool)

    # Add completion if available
    if 'completion' in batch[0]:
        result['completion'] = [item['completion'] for item in batch]

    return result