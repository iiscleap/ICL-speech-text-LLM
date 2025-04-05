from .base_config import DatasetType, DatasetSplit, DatasetConfig

SQA_CONFIG = DatasetConfig(
    name=DatasetType.SQA,
    paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_sqa5_train",
        DatasetSplit.VAL: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_sqa5_validation",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_sqa5_test",
    },
        prompt_template="""You are a spoken question answering expert. Your task is to identify the answer in a given document.
    
    Guidelines:
    - Provide a clear and concise answer to the question
    - Keep answers short (1-2 words whenever possible)
    - Base your answer solely on the information provided in the document
    - Keep the answer focused and relevant to the question
    - Use natural, conversational language
    - Avoid including unnecessary context or explanations
    
    Remember: Output should be just the answer text.""",
    
    valid_labels=None,  # Since we're predicting timestamps, not classification
    completion_key="answer_text",
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