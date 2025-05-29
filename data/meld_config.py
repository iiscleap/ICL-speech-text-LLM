from .base_config import DatasetType, DatasetSplit, DatasetConfig
import random

MELD_CONFIG = DatasetConfig(
    name=DatasetType.MELD,
    paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/meld_train",
        DatasetSplit.VAL: "/data2/neeraja/neeraja/data/meld_validation",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/meld_test",
        
    },
    prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: positive, negative, or neutral.

Guidelines:
- Choose positive if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose negative if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose neutral ONLY IF the statement is purely factual with zero emotional content""",
    valid_labels=["positive", "negative", "neutral"],
    completion_key="sentiment_label",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/meld_train",
        DatasetSplit.VAL: "/data2/neeraja/neeraja/data/meld_train",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/meld_train",
        
    }
)

# Similar to VoxCeleb, you can also add Greek version and permutation configs
MELD_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.MELD_GREEK,
    paths=MELD_CONFIG.paths,
    prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: alpha, beta, or gamma.

Guidelines:
- Choose alpha if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose beta if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose gamma ONLY IF the statement is purely factual with zero emotional content""",
    valid_labels=["alpha", "beta", "gamma"],
    label_mapping={"positive": "alpha", "negative": "beta", "neutral": "gamma"},
    audio_lookup_paths=MELD_CONFIG.audio_lookup_paths,
    text_key=MELD_CONFIG.text_key,
    completion_key=MELD_CONFIG.completion_key
)