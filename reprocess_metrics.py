import json
import os
from utils.evaluation_utils import evaluate_predictions
from data.master_config import DatasetType
import traceback

def reprocess_results(results_file: str):
    """Reprocess results file to generate new metrics for all dataset types"""
    try:
        # Get directory path
        dir_path = os.path.dirname(results_file)
        base_name = os.path.basename(results_file)
        
        # Make sure we're using the results file, not metrics file
        if 'metrics' in base_name:
            results_file = results_file.replace('_metrics.json', '_results.json')
            base_name = os.path.basename(results_file)
        
        print(f"Processing file: {results_file}")
        
        # Check if results file exists
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        # Load raw data
        with open(results_file, 'r') as f:
            raw_results = json.load(f)
        
        # Convert to list if needed
        if isinstance(raw_results, dict):
            results = [raw_results[key] for key in raw_results.keys() if isinstance(raw_results[key], dict)]
        else:
            results = raw_results
            
        # Verify we have valid results
        if not results:
            raise ValueError(f"No valid predictions found in file")
            
        print(f"Processing {len(results)} total predictions")
        
        # Create combined metrics dictionary
        combined_metrics = {}
        
        # Process each dataset type
        dataset_types = {
            'voxceleb': DatasetType.VOXCELEB,
            'hvb': DatasetType.HVB,
            'voxpopuli': DatasetType.VOXPOPULI
        }
        
        for dataset_name, dataset_type in dataset_types.items():
            # Filter results for current dataset type
            dataset_results = [r for r in results if r['dataset_type'].lower() == dataset_name]
            
            if dataset_results:
                print(f"\nProcessing {len(dataset_results)} predictions for {dataset_name}")
                # Print sample for verification
                print(f"Sample {dataset_name} prediction:")
                print(f"Text: {dataset_results[0]['text']}")
                print(f"True label: {dataset_results[0]['true_label']}")
                print(f"Predicted: {dataset_results[0]['predicted_label']}")
                
                # Generate metrics for this dataset
                metrics = evaluate_predictions(dataset_results, dataset_type)
                combined_metrics[dataset_name] = metrics
            else:
                print(f"No predictions found for {dataset_name}")
        
        # Create output filename in same directory
        output_file = os.path.join(dir_path, base_name.replace('_results.json', '_metrics_combined.json'))
        
        # Save combined metrics
        with open(output_file, 'w') as f:
            json.dump(combined_metrics, f, indent=2)
        
        print(f"\nProcessed {results_file}")
        print(f"Saved combined metrics to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()

def process_folder(folder_path: str):
    """Process all results.json files in the given folder and its subfolders"""
    print(f"Processing folder: {folder_path}")
    
    # Walk through all subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if file is a results.json file
            if file.endswith('_results.json'):
                results_file = os.path.join(root, file)
                print(f"\nProcessing file: {results_file}")
                try:
                    reprocess_results(results_file)
                except Exception as e:
                    print(f"Error processing {results_file}: {str(e)}")
                    traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process results files in a folder')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing results files')
    args = parser.parse_args()
    
    if os.path.exists(args.folder_path):
        process_folder(args.folder_path)
    else:
        print(f"Folder not found: {args.folder_path}") 