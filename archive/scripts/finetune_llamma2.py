# finetune_sentiment_debug.py
import torch
from datasets import load_from_disk
from transformers import (
    LlamaTokenizer, 
    LlamaForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
from tqdm import tqdm
# import wandb  # Optional: for better logging
import gc
import sys
import os



parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--output_dir", type=str, default="./sentiment_model")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                    help="Path to previous checkpoint to resume training from")
parser.add_argument('--log_file', type=str, default='finetune_llamma2.log', 
                   help='Name of the log file')
parser.add_argument("--input_mode", type=str, default='speech_only', 
                   choices=['speech_only', 'speech_and_text'],
                   help="Whether to use speech only or both speech and text")
args = parser.parse_args()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{args.log_file}'),
        logging.StreamHandler()
    ]
)

class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        valid_labels = ["Positive", "Negative", "Neutral"]
        filtered_indices = [i for i in range(len(dataset)) if dataset[i]['sentiment'] in valid_labels]
        self.dataset = dataset.select(filtered_indices)
        
        # Log dataset statistics
        logging.info(f"Original dataset size: {len(dataset)}")
        logging.info(f"Filtered dataset size: {len(self.dataset)}")
        
        # Log label distribution
        label_counts = {}
        for item in self.dataset:
            label = item['sentiment']
            label_counts[label] = label_counts.get(label, 0) + 1
        logging.info(f"Label distribution: {label_counts}")
        # Log max sequence length in dataset
        logging.info("Checking sequence lengths...")
        max_len = 0
        for idx in range(min(100, len(dataset))):  # Check first 100 samples
            item = dataset[idx]
            prompt = self._create_prompt(item)
            length = len(tokenizer.encode(prompt['prompt']))
            max_len = max(max_len, length)
        logging.info(f"Maximum sequence length in sample: {max_len}")

    def __len__(self):
        return len(self.dataset)

    def _create_prompt(self, item):
        text = item['normalized_text']
        sentiment = item['sentiment']
        
        base_prompt = """You are a sentiment analysis expert. Based on the statement below, respond with EXACTLY ONE WORD from these options: Positive, Negative, or Neutral.

Guidelines:
- Choose Positive if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose Negative if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose Neutral ONLY IF the statement is purely factual with zero emotional content
"""
        max_available = min(5, len(item['few_shot_examples']))
        num_examples = np.random.randint(0, max_available) 
        # Get random examples from few_shot_examples
        if 'few_shot_examples' in item and len(item['few_shot_examples']) > 0 and num_examples > 0:
            # Randomly select up to 3 examples
            selected_examples = item['few_shot_examples'][:num_examples]
            examples_text = "\n\n".join([
            f"Text: {example['text']}\n"
            f"Output: {example['label']}"
            for example in selected_examples
            ])
            
            prompt = f"""{base_prompt}
Here are few examples to learn from:
{examples_text}

Now analyze this input:
Text: {text}
Output: """
        else:
            prompt = f"""{base_prompt}

Now analyze this input:            
Text: {text}
Output: """
            
        return {"prompt": prompt, "completion": f"{sentiment}</s>"}

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt_data = self._create_prompt(item)
        return {
            "prompt": prompt_data["prompt"],
            "completion": prompt_data["completion"]
        }

def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    completions = [item["completion"] for item in batch]
    
    return {
        "prompts": prompts,
        "completions": completions
    }


def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logging.info(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.2f}%"
        )

class MemoryTracker:
    def __init__(self, device, threshold_mb=20000):
        self.device = device
        self.threshold_mb = threshold_mb
        self.peak_memory = 0
        
    def check_memory(self, batch_idx):
        if not torch.cuda.is_available():
            return
            
        current_memory = torch.cuda.memory_allocated(self.device) / 1024**2
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # if current_memory > self.threshold_mb:
        print(f"\nHigh memory usage detected: {current_memory:.2f} MB")
        print(f"Clearing cache at batch {batch_idx}")
        torch.cuda.empty_cache()
        gc.collect()
        
        return current_memory




def train(model, train_dataloader, tokenizer, optimizer, scheduler, num_epochs, device, start_epoch=0):
    model.train()
    best_loss = float('inf')
    loss_history = []
    
    # Add gradient clipping
    max_grad_norm = 0.5  # Reduce this if still unstable
    
    # Use in training loop
    memory_tracker = MemoryTracker(device, threshold_mb=20000)

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        num_batches = len(train_dataloader)
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Force flush print statements
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx}", flush=True)
                current_memory = memory_tracker.check_memory(batch_idx)
                print(f"\nCurrent Memory: {current_memory:.2f} MB", flush=True)
                print(f"Peak Memory: {memory_tracker.peak_memory:.2f} MB", flush=True)

            # Tokenize prompts without padding
            prompt_encodings = tokenizer(
                batch["prompts"],
                padding="longest",
                return_tensors="pt",
                add_special_tokens=False,
                return_attention_mask=True,
            )
            
            # Get completion token ids with attention mask
            completion_encodings = tokenizer(
                batch["completions"],
                padding='longest',
                return_tensors="pt",
                add_special_tokens=False,
                return_attention_mask=True,
            )

            # Get prompt lengths directly from input_ids size
            prompt_lengths = prompt_encodings.input_ids.size(1)
            
            # Combine input_ids and attention_mask
            input_ids = torch.cat([prompt_encodings.input_ids, completion_encodings.input_ids], dim=1)
            attention_mask = torch.cat([prompt_encodings.attention_mask, completion_encodings.attention_mask], dim=1)
            
            # Create labels tensor with -100s for prompt tokens and padding tokens
            labels = torch.full_like(input_ids, -100)
            labels[:, prompt_lengths:] = completion_encodings.input_ids
            labels[attention_mask == 0] = -100  # Mask padding tokens
            
            # Move everything to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # loss = outputs.loss
            loss = outputs.loss 

            if torch.isnan(loss):
                print(f"\nNaN loss detected at batch {batch_idx}")
                print(f"Last 5 losses: {loss_history[-5:] if loss_history else []}")
                print(f"Logits stats: min={outputs.logits.min()}, max={outputs.logits.max()}, mean={outputs.logits.mean()}")
                continue

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Check for NaN gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print("Skipping batch due to NaN gradients")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            loss_history.append(loss_val)
            epoch_loss += loss_val
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })
            
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        logging.info(f'Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}')
        
        # Save checkpoint if best loss
        if avg_epoch_loss < best_loss:
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = f'{args.output_dir}/checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # best_loss = avg_epoch_loss
            
            # Save checkpoint with epoch info
            checkpoint_path = f'{args.output_dir}/checkpoints/epoch_{epoch + 1}_loss_{avg_epoch_loss:.4f}'
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save optimizer and scheduler states
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }, f'{checkpoint_path}/training_state.pt')
            model.save_pretrained(f'{checkpoint_dir}/epoch_{epoch + 1}_loss_{avg_epoch_loss:.4f}')
            logging.info(f'Saved checkpoint for epoch {epoch + 1} with loss {avg_epoch_loss:.4f}')

def main():
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("Loading model and tokenizer...")

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
        # max_memory={0: "40GB"}  # Limit GPU memory usage
    )

    for name, param in model.named_parameters():
        param.requires_grad = False

    print_trainable_parameters(model)

    logging.info("Configuring LoRA...")
    start_epoch = 0
    if args.resume_from_checkpoint:
        logging.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        model = PeftModel.from_pretrained(
            model,
            args.resume_from_checkpoint,
            is_trainable=True,
            torch_dtype=torch.float16,
        )
        training_state_path = f'{args.resume_from_checkpoint}/training_state.pt'
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            start_epoch = training_state['epoch'] + 1
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
            scheduler.load_state_dict(training_state['scheduler_state_dict'])
            logging.info(f"Resuming from epoch {start_epoch}")
    else:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)
    model.print_trainable_parameters()

    # Enable gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    logging.info("Loading dataset...")
    dataset = load_from_disk("/data2/neeraja/neeraja/data/voxceleb_train_fewshots")
    # dataset = dataset.select(range(50))
    train_dataset = SentimentDataset(dataset, tokenizer)

    # Reduced batch size and sequence length
    batch_size = 1  # Reduced from 4
    num_epochs = 15
    learning_rate = 1e-5
    warmup_steps = 100

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs) 
    )

    # Clear CUDA cache before training
    torch.cuda.empty_cache()
    
    logging.info("Starting training...")
    train(model, train_dataloader, tokenizer, optimizer, scheduler, 
          num_epochs, model.device, start_epoch=start_epoch)
    logging.info("Training completed!")
    model.save_pretrained(args.output_dir)
    logging.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

# qsub -q med.q -V -cwd \
#     -l hostname=compute-0-9 \
#     -o ./results/logs/train/train_debug_long5.log \
#     -j y \
#     -v CUDA_VISIBLE_DEVICES=2 \
#     -S /bin/bash train_sentiment.sh

# qlogin -q all.q -l hostname=compute-0-9

## Duplicates are remove for topK in hvb
## For testing I wil remove filter for similar score