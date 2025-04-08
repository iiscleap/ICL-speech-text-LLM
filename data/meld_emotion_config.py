from .base_config import DatasetType, DatasetSplit, DatasetConfig
import random

MELD_EMOTION_CONFIG = DatasetConfig(
    name=DatasetType.MELD_EMOTION,
    paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/meld_train",
        DatasetSplit.VAL: "/data2/neeraja/neeraja/data/meld_validation",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/meld_test",
    },
    prompt_template="""You are an emotion recognition expert. Based on the input, respond with EXACTLY ONE WORD from these options: neutral, joy, sadness, anger, fear, disgust, or surprise.

Guidelines:
- Choose joy if there is happiness, excitement, delight, pleasure, or positive enthusiasm
- Choose sadness if there is unhappiness, sorrow, grief, disappointment, or regret
- Choose anger if there is irritation, rage, fury, annoyance, or hostility
- Choose fear if there is terror, anxiety, worry, concern, or nervousness
- Choose disgust if there is repulsion, distaste, revulsion, or strong dislike
- Choose surprise if there is astonishment, shock, amazement, or unexpected reaction
- Choose neutral ONLY IF the statement expresses no distinct emotional state""",
    valid_labels=["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"],
    completion_key="emotion_label",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/meld_train",
        DatasetSplit.VAL: "/data2/neeraja/neeraja/data/meld_validation",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/meld_test",
    }
)

# Greek version with arbitrary greek letter mappings
MELD_EMOTION_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.MELD_EMOTION_GREEK,
    paths=MELD_EMOTION_CONFIG.paths,
    prompt_template="""You are an emotion recognition expert. Based on the input, respond with EXACTLY ONE WORD from these options: alpha, beta, gamma, delta, epsilon, zeta, eta.

Guidelines:
- Choose alpha if there is no distinct emotional state (neutral)
- Choose beta if there is happiness, excitement, delight, pleasure, or positive enthusiasm
- Choose gamma if there is unhappiness, sorrow, grief, disappointment, or regret
- Choose delta if there is irritation, rage, fury, annoyance, or hostility
- Choose epsilon if there is terror, anxiety, worry, concern, or nervousness
- Choose zeta if there is repulsion, distaste, revulsion, or strong dislike
- Choose eta if there is astonishment, shock, amazement, or unexpected reaction""",
    valid_labels=["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"],
    label_mapping={
        "neutral": "alpha", 
        "joy": "beta", 
        "sadness": "gamma", 
        "anger": "delta", 
        "fear": "epsilon", 
        "disgust": "zeta", 
        "surprise": "eta"
    },
    audio_lookup_paths=MELD_EMOTION_CONFIG.audio_lookup_paths,
    text_key=MELD_EMOTION_CONFIG.text_key,
    completion_key=MELD_EMOTION_CONFIG.completion_key
)