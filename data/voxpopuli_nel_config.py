from .base_config import DatasetType, DatasetSplit, DatasetConfig

VOXPOPULI_NEL_CONFIG = DatasetConfig(
    name=DatasetType.VOXPOPULI_NEL,
    paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_nel_train",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_nel_test",
    },
    prompt_template="""You are a named entity linking expert. Given a document, identify the timestamp where the specified entity is mentioned.

Guidelines for identifying entities:
- An entity can be a person's name (e.g., Barack Obama, Angela Merkel)
- An entity can be an organization (e.g., European Union, United Nations)
- An entity can be a location (e.g., Brussels, Germany)
- An entity can be a political party (e.g., Democrats, Republicans)
- Focus on exact matches or clear references to the entity

Document: {normalized_text}
Entity: {entity_name}

Provide the timestamp in seconds where this entity is mentioned in the document. Respond with ONLY the numerical timestamp value.""",
    
    valid_labels=None,  # Since we're predicting timestamps, not classification
    completion_key="entity_spans",  # The timestamps we're predicting
    text_key="normalized_text",
    entity_key="entity_name",  # Key for the entity we're looking for
    
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_nel_train_audio_lookup",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_nel_test_audio_lookup",
    },
    

) 