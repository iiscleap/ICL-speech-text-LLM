import json
import os
from utils.evaluation_utils import evaluate_predictions
from data.master_config import DatasetType

def reprocess_results(results_file: str, dataset_name: str):
    """Reprocess results file to generate new metrics"""
    try:
        # Get directory path
        dir_path = os.path.dirname(results_file)
        base_name = os.path.basename(results_file)
        
        # Assign dataset_type based on dataset_name
        if dataset_name.lower() == 'voxceleb':
            dataset_type = DatasetType.VOXCELEB
        elif dataset_name.lower() == 'hvb':
            dataset_type = DatasetType.HVB
        elif dataset_name.lower() == 'voxpopuli':
            dataset_type = DatasetType.VOXPOPULI
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        
        # Make sure we're using the results file, not metrics file
        if 'metrics' in base_name:
            results_file = results_file.replace('_metrics.json', '_results.json')
            base_name = os.path.basename(results_file)
        
        print(f"Processing file: {results_file}")
        print(f"Dataset type: {dataset_type}")
        
        # Check if results file exists
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        # Load and print the raw data structure
        with open(results_file, 'r') as f:
            raw_results = json.load(f)
        
        print("Raw data type:", type(raw_results))
        if isinstance(raw_results, dict):
            # Convert dictionary to list if needed
            results = [raw_results[key] for key in raw_results.keys() if isinstance(raw_results[key], dict)]
        else:
            results = raw_results
            
        # Verify we have valid results
        if not results:
            raise ValueError(f"No valid predictions found in file")
            
        print(f"Processing {len(results)} predictions")
        
        # Print first few results for verification
        print("\nFirst few results before processing:")
        for i, result in enumerate(results[:3]):
            print(f"\nResult {i+1}:")
            print(f"Text: {result['text']}")
            print(f"True label: {result['true_label']}")
            print(f"Predicted: {result['predicted_label']}")
            print(f"Dataset type: {result['dataset_type']}")
        
        # Generate new metrics
        metrics = evaluate_predictions(results, dataset_type)
        
        # Create output filename in same directory
        output_file = os.path.join(dir_path, base_name.replace('_results.json', '_metrics_new.json'))
        
        # Save new metrics
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nProcessed {results_file}")
        print(f"Saved new metrics to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # Make sure to use the results file path, not metrics
    results_file = "/data2/neeraja/neeraja/results/model_ICL/metrics/2025-03-24/2403_1802_ft_5ex_20e8b_sal_sp_only_vox-hvb_voxpopuli_speech_only_text_1shots_results.json"
    
    if os.path.exists(results_file):
        reprocess_results(results_file, "voxpopuli")
    else:
        print(f"Results file not found: {results_file}") 