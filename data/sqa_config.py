from .base_config import DatasetType, DatasetSplit, DatasetConfig

SQA_CONFIG = DatasetConfig(
    name=DatasetType.SQA,
    paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_sqa_train",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_sqa_test",
    },
    prompt_template="""You are a spoken question answering expert. Given a question and a document, identify the timestamp in the document where the answer begins.

Question: {normalized_question_text}
Document: {normalized_document_text}

Provide the timestamp in seconds where the answer to this question begins in the document. Respond with ONLY the numerical timestamp value.""",
    
    valid_labels=None,  # Since we're predicting timestamps, not classification
    completion_key="answer_spans",
    text_key="normalized_document_text",
    question_key="normalized_question_text",
    
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_sqa_train_audio_lookup",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_sqa_test_audio_lookup",
    },
    
  
    # Audio fields
    audio_keys={
        "question_audio": "question_audio",
        "document_audio": "document_audio"
    }
) 