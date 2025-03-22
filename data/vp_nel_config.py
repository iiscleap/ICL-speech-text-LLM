from .base_config import DatasetType, DatasetSplit, DatasetConfig

VP_NEL_CONFIG = DatasetConfig(
    name=DatasetType.VOXPOPULI_NEL,
    paths={
        DatasetSplit.VAL: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_vp_nel_validation",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_vp_nel_test",
    },
    prompt_template="""You are a named entity recognition expert. Your task is to identify the timestamps and types of named entities in the given text.

Guidelines:
- For each entity, provide its type followed by start and end timestamps
- Format: ENTITY_TYPE: start_time end_time
- Separate multiple entities with semicolons
- Timestamps should be in seconds
- If no entities are found, respond with 'none'
- Entity types can be: DATE, GPE, PERSON, ORG, etc.

Example format:
DATE: 1.79 2.64; GPE: 2.96 3.19; DATE: 7.09 8.47

Remember: Output should be "TYPE: start end" pairs separated by semicolons.""",
    
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