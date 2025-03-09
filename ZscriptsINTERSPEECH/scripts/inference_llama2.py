import torch
from datasets import load_from_disk
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import json


parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="lmsys/vicuna-13b-v1.1")
parser.add_argument("--peft_model_path", type=str, default="./sentiment_model")
parser.add_argument("--test_data_path", type=str, default="/data2/neeraja/neeraja/data/voxceleb_test_fewshots")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_examples", type=int, default=0, 
                    help="Number of few-shot examples to include (0-10)")
parser.add_argument("--output_suffix", type=str, default="",
                    help="Suffix to add to output file name (e.g., '_2shot')")
parser.add_argument('--log_file', type=str, default='inference_llamma2.log', 
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
        logging.FileHandler(f'{args.log_file}',mode='w'),
        logging.StreamHandler()
    ]
)



class InferenceDataset(Dataset):
    def __init__(self, dataset, num_examples=0):
        self.dataset = dataset
        self.num_examples = min(num_examples, 10)  # Cap at 10 examples

    def __len__(self):
        return len(self.dataset)

    def _create_prompt(self, item):
        text = item['normalized_text']
        
        base_prompt = """You are a sentiment analysis expert. Based on the statement below, respond with EXACTLY ONE WORD from these options: Positive, Negative, or Neutral.

Guidelines:
- Choose Positive if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose Negative if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose Neutral ONLY IF the statement is purely factual with zero emotional content"""

        if 'few_shot_examples' in item and len(item['few_shot_examples']) > 0 and self.num_examples > 0:
            # Take the specified number of examples
            selected_examples = item['few_shot_examples'][:self.num_examples]
            examples_text = "\n\n".join([
                f"Text: {example['text']}\n"
                f"Output: {example['label']}"
                for example in selected_examples
            ])
            
            prompt = f"""{base_prompt}

{examples_text}

Text: {text}
Output: """
        else:
            prompt = f"""{base_prompt}

Text: {text}
Output: """
            
        return prompt

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = self._create_prompt(item)
        return {
            "prompt": prompt,
            "true_sentiment": item['sentiment'],
            "text": item['normalized_text']
        }

def collate_fn(batch):
    return {
        "prompts": [item["prompt"] for item in batch],
        "true_sentiments": [item["true_sentiment"] for item in batch],
        "texts": [item["text"] for item in batch]
    }

def generate_response(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # print(prompt)
    # print(inputs)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # We only need one word
            num_return_sequences=1,
            temperature=0.1,    # Lower temperature for more focused sampling
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # print(outputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the sentiment prediction (last word after "Answer:")
    
    # print(response, flush=True)
    sentiment = response.split("Output:")[-1].strip()
    return sentiment





def evaluate_results(predictions, output_file):
    """
    Evaluate the model predictions and save metrics
    """
    # Convert predictions to DataFrame
    df = pd.DataFrame(predictions)
    df.columns = ['text', 'gt', 'pd']  # gt: ground truth, pd: predicted

    # Filter only valid classes
    valid_classes = ['neutral', 'negative', 'positive']
    
    # Log initial length
    total_samples = len(df)
    
    # Filter ground truth
    df = df[df['gt'].str.lower().isin(valid_classes)]
    after_gt_filter = len(df)
    
    # Filter predictions
    df = df[df['pd'].str.lower().isin(valid_classes)]
    after_pred_filter = len(df)
    
    # Calculate confusion matrix and metrics
    matrix = confusion_matrix(
        df['gt'].str.lower().values, 
        df['pd'].str.lower().values,
        labels=valid_classes
    )
    
    # Calculate per-class accuracy
    class_accuracy = matrix.diagonal()/matrix.sum(axis=1)
    class_samples = matrix.sum(axis=1)
    
    # Calculate macro F1 score
    macro_f1 = f1_score(
        df['gt'].str.lower().values,
        df['pd'].str.lower().values,
        average="macro",
        labels=valid_classes
    )
    
    # Save results
    with open(output_file, "w") as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Samples after ground truth filter: {after_gt_filter}\n")
        f.write(f"Samples after prediction filter: {after_pred_filter}\n")
        f.write(f"\nMacro F1 Score: {macro_f1:.4f}\n")
        f.write("\nPer-class metrics:\n")
        for i, class_name in enumerate(valid_classes):
            f.write(f"{class_name}:\n")
            f.write(f"  Accuracy: {class_accuracy[i]:.4f}\n")
            f.write(f"  Samples: {class_samples[i]}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(f"Classes: {valid_classes}\n")
        f.write(str(matrix))
    
    return {
        'macro_f1': macro_f1,
        'class_accuracy': class_accuracy,
        'confusion_matrix': matrix,
        'valid_samples': after_pred_filter,
        'filtered_samples': total_samples - after_pred_filter
    }

def main():
    logging.info("Loading model and tokenizer...")
    
    # Load base model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    # model = base_model
    model.eval()
    
    logging.info("Loading test dataset...")
    test_dataset = load_from_disk(args.test_data_path)
    test_dataset = InferenceDataset(test_dataset, num_examples=args.num_examples)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Prepare results storage
    results = {
        'predictions': []
    }
    
    logging.info("Starting inference...")
    for batch in tqdm(test_dataloader):
        for prompt, true_sentiment, text in zip(batch["prompts"], batch["true_sentiments"], batch["texts"]):
            predicted_sentiment = generate_response(model, tokenizer, prompt, model.device)
            
            # Print with flush=True
            print(f"\n True: {true_sentiment.lower()}, Predicted: {predicted_sentiment.lower()}", flush=True)
            
            # Store prediction
            results['predictions'].append({
                'text': text,
                'true_sentiment': true_sentiment.lower(),
                'predicted_sentiment': predicted_sentiment.lower()
            })
    # Base filename for results
    base_filename = f'{args.output_suffix}_{args.num_examples}shot'
    
    # Save raw predictions
    with open(f'./results/final_results/{base_filename}_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Evaluate and save metrics
    metrics = evaluate_results(
        results['predictions'],
        f'./results/final_results/{base_filename}_metrics.txt'
    )
    
    # Log summary
    logging.info(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    logging.info(f"Valid samples: {metrics['valid_samples']}")
    logging.info(f"Filtered samples: {metrics['filtered_samples']}")
    logging.info(f"Results saved to {base_filename}_*")

if __name__ == "__main__":
    main() 

# qsub -q all.q -V -cwd \
#     -l hostname=compute-0-9 \
#     -o ./results/test_llama4.log \
#     -j y \
#     -v CUDA_VISIBLE_DEVICES=0 \
#     -S /bin/bash train_sentiment.sh