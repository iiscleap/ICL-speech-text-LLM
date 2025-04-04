from .base_config import DatasetType, DatasetSplit, DatasetConfig

SQA_CONFIG = DatasetConfig(
    name=DatasetType.SQA,
    paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_sqa5_train",
        DatasetSplit.VAL: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_sqa5_validation",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_sqa5_test",
    },
    prompt_template="""You are a spoken question answering expert. Your task is to identify both the start and end timestamps of the answer along with the exact words in a given document.

Guidelines:
- Provide the answer in the format: words|start_time end_time
- The words should be the exact text from the document that answers the question
- The first timestamp marks where the answer begins
- The second timestamp marks where the answer ends
- Be precise with both timestamps for accurate answer extraction
- If multiple occurrences exist, choose the first relevant instance
- Timestamps should be in seconds, separated by a space

Example format: twenty five dollars|10.42 10.86

Remember: Output should be the answer text, followed by a pipe symbol (|), followed by two timestamps separated by a space.""",
    
    valid_labels=None,  # Since we're predicting timestamps, not classification
    completion_key="time_spans",
    text_key="normalized_document_text",  # Primary text key
    
    # Use new generic fields
    additional_text_keys={
        'question': 'normalized_question_text',
    },
    additional_audio_keys={
        'question_audio': 'question_audio',
        'document_audio': 'document_audio'
    },
    additional_metadata_keys={
        'unique_id': 'unique_id',
        'question_id': 'question_id',
        'document_id': 'document_id',
        'speaker_ids': {
            'question': 'question_speaker_id',
            'document': 'document_speaker_id'
        }
    },
    
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_sqa5_train",
        DatasetSplit.VAL: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_sqa5_validation",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_sqa5_test",
    },
    output_format="timestamps_pair"  # Indicates we expect "start_time end_time" format
) 