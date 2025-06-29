from .base_config import DatasetType, DatasetSplit, DatasetConfig

VP_NEL_CONFIG = DatasetConfig(
    name=DatasetType.VOXPOPULI_NEL,
    paths={
        DatasetSplit.VAL: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_vp_nel_validation",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_vp_nel_test",
    },
    prompt_template="""You are a named entity recognition expert. Your task is to identify each word and its timestamps in the given text.

Guidelines:
- For each word in the text, provide the word and its precise start and end timestamps
- Format: word1:start1-end1 word2:start2-end2 word3:start3-end3
- Timestamps should be in seconds with decimal precision
- Include all words, not just named entities
- Words should be in the exact order they appear in the text
- Ensure timestamps are sequential and don't overlap

Example format:
the:1.79-1.85 president:1.85-2.14 of:2.14-2.25 France:2.25-2.64 said:2.64-2.89

Remember: Each word should have its own timestamp pair, connected with a hyphen, and words should be separated by spaces.""",
    
    valid_labels=None,
    completion_key="ne_spans",
    text_key="text",
    
    additional_metadata_keys={
        'unique_id': 'unique_id',
        'speaker_id': 'speaker_id'
    },
    
    additional_audio_keys={
        'audio': 'audio'
    },
    
    # Use validation split for all audio lookups
    audio_lookup_paths={
        DatasetSplit.VAL: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_vp_nel_validation",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_vp_nel_validation",  # Use validation for test too
    },
    output_format="entity_timestamps"  # Indicates we expect "TYPE: start end" format
) 