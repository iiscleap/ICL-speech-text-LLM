from datasets import load_from_disk
import os
from huggingface_hub import HfFolder
import logging

def list_local_datasets(cache_dir=None):
    """
    List all datasets in the local Hugging Face cache directory
    """
    if cache_dir is None:
        cache_dir = HfFolder().cache_dir
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    
    logging.info(f"Checking datasets in: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        logging.info("No local cache directory found")
        return []
    
    local_datasets = []
    
    # Walk through the cache directory
    for root, dirs, files in os.walk(cache_dir):
        if "dataset_info.json" in files:
            # This directory contains a dataset
            dataset_path = root
            try:
                # Try to load the dataset to get its info
                dataset = load_from_disk(dataset_path)
                info = {
                    "path": dataset_path,
                    "num_rows": len(dataset),
                    "features": list(dataset.features.keys()),
                    "split": os.path.basename(os.path.dirname(dataset_path))
                }
                local_datasets.append(info)
                logging.info(f"\nFound dataset:")
                logging.info(f"Path: {info['path']}")
                logging.info(f"Number of rows: {info['num_rows']}")
                logging.info(f"Features: {info['features']}")
                logging.info(f"Split: {info['split']}")
                logging.info("-" * 50)
            except Exception as e:
                logging.warning(f"Could not load dataset at {dataset_path}: {str(e)}")
    
    return local_datasets

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # You can specify a custom cache directory if needed
    # datasets = list_local_datasets("/path/to/custom/cache")
    datasets = list_local_datasets("/data2/neeraja/neeraja/data")
    
    if not datasets:
        logging.info("No datasets found in local cache") 