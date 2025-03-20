import os
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk, concatenate_datasets
import logging
import json
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_sqa5_dataset():
    dataset_name = "asapp/slue-phase-2"
    subset = "sqa5"
    cache_dir = '/data2/neeraja/neeraja/data'
    
    try:
        for split in ['train', 'validation', 'test']:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {split} split...")
            
            # Load with optimized settings
            if split == 'train':
                split_dataset = load_dataset(
                    dataset_name, 
                    subset, 
                    split='train', 
                    cache_dir=cache_dir,
                    num_proc=8,  # Use multiple processes for loading
                    batch_size=1000  # Larger batch size
                ).select(range(10000))
                logger.info(f"Selected first 10000 examples from train split")
            else:
                split_dataset = load_dataset(
                    dataset_name, 
                    subset, 
                    split=split, 
                    cache_dir=cache_dir,
                    num_proc=8,
                    batch_size=1000
                )
            
            logger.info(f"Initial number of examples: {len(split_dataset)}")
            
            # Process with optimized settings
            logger.info("\n=== Adding unique_id column ===")
            split_dataset = split_dataset.map(
                lambda x: {'unique_id': x['question_id'] + "_" + x['document_id']},
                batch_size=16,
                num_proc=1,
                desc="Creating unique IDs"
            )
            
            logger.info("\n=== Adding answer_text column ===")
            split_dataset = split_dataset.map(
                lambda x: {'answer_text': json.loads(x['answer_spans'])['answer'][0] if isinstance(x['answer_spans'], str) else x['answer_spans']['answer'][0]},
                batch_size=16,
                num_proc=1,
                desc="Extracting answers"
            )
            
            logger.info("\n=== Adding time_spans column ===")
            split_dataset = split_dataset.map(
                lambda x: {'time_spans': [json.loads(x['answer_spans'])['start_second'][0], json.loads(x['answer_spans'])['end_second'][0]] if isinstance(x['answer_spans'], str) else [x['answer_spans']['start_second'][0], x['answer_spans']['end_second'][0]]},
                batch_size=16,
                num_proc=1,
                desc="Extracting time spans"
            )
            
            # Log unique IDs stats
            num_examples = len(split_dataset)
            num_unique_ids = len(set(split_dataset['unique_id']))
            logger.info(f"Total examples: {num_examples}")
            logger.info(f"Unique IDs: {num_unique_ids}")
            if num_examples != num_unique_ids:
                logger.warning(f"Found {num_examples - num_unique_ids} duplicate IDs!")
            
            # Log answer stats
            answer_types = split_dataset['answer_text'][:5]
            logger.info("First 5 answers:")
            for i, ans in enumerate(answer_types):
                logger.info(f"Answer {i+1}: {ans} (type: {type(ans)})")
            
            # Log time spans stats
            time_spans = split_dataset['time_spans'][:5]
            logger.info("First 5 time spans:")
            for i, span in enumerate(time_spans):
                logger.info(f"Time span {i+1}: {span} (type: {type(span)})")
            
            # Final verification
            logger.info("\n=== Final Dataset Stats ===")
            logger.info(f"Number of examples: {len(split_dataset)}")
            logger.info(f"Columns: {split_dataset.column_names}")
            
            # Show a complete example
            logger.info("\nExample data point:")
            example = split_dataset[0]
            for k, v in example.items():
                logger.info(f"{k}: {v} (type: {type(v)})")
            
            # Save processed dataset
            output_path = f"/data2/neeraja/neeraja/data/{dataset_name}_{subset}_{split}"
            split_dataset.save_to_disk(output_path)
            logger.info(f"\nSaved {split} split to {output_path}")
            logger.info("="*50)

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    process_sqa5_dataset()