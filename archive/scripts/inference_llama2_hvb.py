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
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="lmsys/vicuna-13b-v1.1")
parser.add_argument("--peft_model_path", type=str, default="./sentiment_model")
parser.add_argument("--test_data_path", type=str, default="")
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
        self.num_examples = min(num_examples, 3)  # Cap at 3 examples

    def _create_prompt(self, item):
        text = item['text']
        
        base_prompt = """You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- acknowledge: Shows understanding or receipt of information
- answer_agree: Expresses agreement
- answer_dis: Expresses disagreement
- answer_general: General response to a question
- apology: Expression of regret or sorry
- backchannel: Brief verbal/textual feedback (like "uh-huh", "mm-hmm")
- disfluency: Speech repairs, repetitions, or corrections
- other: Actions that don't fit other categories
- question_check: Questions to verify understanding
- question_general: General information-seeking questions
- question_repeat: Requests for repetition
- self: Self-directed speech
- statement_close: Concluding statements
- statement_general: General statements or information
- statement_instruct: Instructions or directions
- statement_open: Opening statements or greetings
- statement_problem: Statements describing issues or problems
- thanks: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions"""

        if 'few_shot_examples' in item and len(item['few_shot_examples']) > 0 and self.num_examples > 0:
            selected_examples = item['few_shot_examples'][:self.num_examples]
            examples_text = "\n\n".join([
                f"Text: {example['text']}\n"
                f"Output: {', '.join(example['label'])}"
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
            
        return prompt

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = self._create_prompt(item)
        return {
            "prompt": prompt,
            "true_actions": item['dialog_acts'],
            "text": item['text']
        }

    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    return {
        "prompts": [item["prompt"] for item in batch],
        "true_actions": [item["true_actions"] for item in batch],
        "texts": [item["text"] for item in batch]
    }

def generate_response(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Increased for multiple labels
            num_return_sequences=1,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the dialogue actions prediction
    predicted_actions = response.split("Output:")[-1].strip()
    # Convert to list and clean up
    predicted_actions = [act.strip() for act in predicted_actions.split(",")]
    return predicted_actions





def evaluate_results(predictions, output_file):
    """
    Evaluate the model predictions for multi-label classification
    """
    # Convert predictions to DataFrame
    df = pd.DataFrame(predictions)
    df.columns = ['text', 'true_actions', 'pred_actions']

    # Get all unique dialogue actions
    all_actions = set()
    for actions in df['true_actions']:
        all_actions.update(actions)
    for actions in df['pred_actions']:
        all_actions.update(actions)
    all_actions = sorted(list(all_actions))

    # Calculate metrics
    # Initialize arrays for true and predicted labels
    y_true = np.zeros((len(df), len(all_actions)))
    y_pred = np.zeros((len(df), len(all_actions)))

    # Fill the arrays
    for i, row in df.iterrows():
        for action in row['true_actions']:
            if action in all_actions:
                y_true[i, all_actions.index(action)] = 1
        for action in row['pred_actions']:
            if action in all_actions:
                y_pred[i, all_actions.index(action)] = 1

    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    per_class_f1 = f1_score(y_true, y_pred, average=None)

    # Save results
    with open(output_file, "w") as f:
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"\nMacro F1 Score: {macro_f1:.4f}\n")
        f.write(f"Micro F1 Score: {micro_f1:.4f}\n")
        f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")
        
        f.write("\nPer-class F1 Scores:\n")
        for action, score in zip(all_actions, per_class_f1):
            f.write(f"{action}: {score:.4f}\n")
        
        # Calculate and write support for each class
        support = y_true.sum(axis=0)
        f.write("\nClass Support (number of occurrences):\n")
        for action, count in zip(all_actions, support):
            f.write(f"{action}: {int(count)}\n")

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': dict(zip(all_actions, per_class_f1)),
        'support': dict(zip(all_actions, support))
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
        for prompt, true_action, text in zip(batch["prompts"], batch["true_actions"], batch["texts"]):
            predicted_action = generate_response(model, tokenizer, prompt, model.device)
            
            # Print with flush=True
            print(f"\n True: {','.join([act.lower() for act in true_action])}, Predicted: {','.join([act.lower() for act in predicted_action])}",flush=True)
            
            # Store prediction
            results['predictions'].append({
                'text': text,
                'true_actions': [act.lower() for act in true_action],
                'pred_actions': [act.lower() for act in predicted_action]
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
    logging.info(f"Micro F1 Score: {metrics['micro_f1']:.4f}")
    logging.info(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")
    logging.info(f"Per-class F1 Scores:")
    for action, f1 in metrics['per_class_f1'].items():
        logging.info(f"  {action}: {f1:.4f} (support: {metrics['support'][action]})")
    logging.info(f"Results saved to {base_filename}_*")

if __name__ == "__main__":
    main() 

# qsub -q all.q -V -cwd \
#     -l hostname=compute-0-9 \
#     -o ./results/test_llama4.log \
#     -j y \
#     -v CUDA_VISIBLE_DEVICES=0 \
#     -S /bin/bash train_sentiment.sh