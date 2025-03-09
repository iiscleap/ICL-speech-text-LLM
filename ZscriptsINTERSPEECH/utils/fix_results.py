import json
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

def fix_predictions_and_metrics(input_file, output_file_prefix):
    # Load the JSON file
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    # Fix predictions by extracting sentiment from Answer: instead of Output:
    fixed_predictions = []
    for pred in results['predictions']:
        text = pred['text']
        true_sentiment = pred['true_sentiment']
        response = pred['predicted_sentiment']
        
        # Try to extract sentiment after "Answer:", if not found, try "Output:"
        try:
            if "Answer:" in response:
                fixed_sentiment = response.split("Answer:")[-1].strip().lower()
            elif "output:" in response:
                fixed_sentiment = response.split("output:")[-1].strip().lower()
            else:
                fixed_sentiment = response.strip().lower()
        except:
            fixed_sentiment = "invalid"
            
        fixed_predictions.append({
            'text': text,
            'true_sentiment': true_sentiment,
            'predicted_sentiment': fixed_sentiment
        })
    
    # Save fixed predictions
    results['predictions'] = fixed_predictions
    with open(f'{output_file_prefix}_fixed_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Convert predictions to format needed for metrics
    predictions_list = [[p['text'], p['true_sentiment'], p['predicted_sentiment']] 
                       for p in fixed_predictions]
    
    # Calculate metrics
    df = pd.DataFrame(predictions_list)
    df.columns = ['text', 'gt', 'pd']
    
    # Filter valid classes
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
    
    # Save metrics
    with open(f'{output_file_prefix}_fixed_metrics.txt', 'w') as f:
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
    
    print(f"Fixed predictions saved to: {output_file_prefix}_fixed_predictions.json")
    print(f"Fixed metrics saved to: {output_file_prefix}_fixed_metrics.txt")
    print(f"Macro F1 Score: {macro_f1:.4f}")

# Usage example:
if __name__ == "__main__":
    input_file = "./results/final_results/inference_results_5shot_predictions.json"  # Update this path
    output_prefix = "./results/final_results/inference_finetune_llama2_5shot"  # Update this path
    fix_predictions_and_metrics(input_file, output_prefix) 