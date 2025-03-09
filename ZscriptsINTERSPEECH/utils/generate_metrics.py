import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation_utils import evaluate_results
import json
import logging
import argparse

def clean_prediction(prediction: str, dataset_type: str = None) -> str:
    """Clean prediction by removing escape characters and dataset-specific artifacts
    
    Args:
        prediction: The prediction string to clean
        dataset_type: The type of dataset ('hvb' or 'voxceleb')
    """
    # First remove basic escape characters
    cleaned = prediction.replace('\\', '')
    
    # Handle VoxCeleb specific cleaning

    if '\n' in cleaned:
        cleaned = cleaned.split('\n')[0]
    
    return cleaned.strip()

def process_single_file(predictions_file: str, output_file: str, dataset_type: str):
    """Process a single predictions file and generate metrics"""
    logging.info(f"Processing {predictions_file}")
    
    with open(predictions_file, 'r') as f:
        data = json.load(f)

    # Each item in data should be a dictionary with text, true_sentiment, and predicted_sentiment
    predictions = []
    for p in data:
        if isinstance(p, dict) and 'text' in p and 'true_sentiment' in p and 'predicted_sentiment' in p:
            # Handle case where true_sentiment is a list
            true_sent = p['true_sentiment']
            if isinstance(true_sent, list):
                true_sent = true_sent[0]  # Take first sentiment if it's a list
            
            # Clean predicted sentiment with dataset type
            pred_sent = clean_prediction(p['predicted_sentiment'], dataset_type)
            
            predictions.append([p['text'], true_sent.lower(), pred_sent.lower()])

    # Generate metrics
    metrics = evaluate_results(
        predictions=predictions,
        output_file=output_file,
        dataset_type=dataset_type
    )
    
    logging.info(f"Metrics saved to {output_file}")
    return metrics

def process_directory(directory: str, dataset_type: str):
    """Process all prediction files in a directory"""
    # Find all prediction JSON files
    prediction_files = glob.glob(os.path.join(directory, "**/*predictions.json"), recursive=True)
    
    for pred_file in prediction_files:
        # Generate output metrics filename by replacing 'predictions.json' with 'metrics_new.txt'
        output_file = pred_file.replace('predictions.json', 'metrics_new.txt')
        process_single_file(pred_file, output_file, dataset_type)

def determine_dataset_type(filename: str) -> str:
    """Determine dataset type from filename"""
    filename_lower = filename.lower()
    if filename_lower.startswith('hvb_'):
        return 'hvb'
    elif filename_lower.startswith('voxceleb_'):
        return 'voxceleb'
    elif filename_lower.startswith('voxpopuli_'):
        return 'voxpopuli'
    else:
        raise ValueError(f"Could not determine dataset type from filename: {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str, required=False,
                        help="Path to the predictions JSON file. If not provided, will process all prediction files in the directory")
    parser.add_argument("--output_file", type=str, required=False,
                        help="Path to save the metrics. If not provided, will be auto-generated from predictions file")
    parser.add_argument("--dataset_type", type=str, required=False,
                        choices=['voxceleb', 'hvb'],
                        help="Type of dataset to evaluate. If not provided, will be determined from filename")
    parser.add_argument("--directory", type=str, required=False,
                        help="Directory containing prediction files to process")
    
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.predictions_file:
        # Auto-generate output file name if not provided
        output_file = args.output_file or args.predictions_file.replace('predictions.json', 'metrics_new.txt')
        # Determine dataset type if not provided
        dataset_type = args.dataset_type or determine_dataset_type(os.path.basename(args.predictions_file))
        process_single_file(args.predictions_file, output_file, dataset_type)
    elif args.directory:
        for pred_file in glob.glob(os.path.join(args.directory, "**/*predictions.json"), recursive=True):
            output_file = pred_file.replace('predictions.json', 'metrics_new.txt')
            dataset_type = args.dataset_type or determine_dataset_type(os.path.basename(pred_file))
            process_single_file(pred_file, output_file, dataset_type)
    else:
        parser.error("Either --predictions_file or --directory must be provided")

if __name__ == "__main__":
    main() 





# python scripts_new/utils/generate_metrics.py \
#     --predictions_file /data2/neeraja/neeraja/code/SALMONN/results/final_results/2025-02-17/voxpopuli_finetune_llama2_salmon_sp_15e8b_Q_vox_swap_speech_only_text_5shots_predictions.json 


# python scripts_new/utils/generate_metrics.py \
#     --directory /data2/neeraja/neeraja/code/SALMONN/results/final_results/2025-02-01
