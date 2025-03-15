from .base_config import DatasetType, DatasetSplit, DatasetConfig
import random

# VOXCELEB_CONFIG = DatasetConfig(
#     name=DatasetType.VOXCELEB,
#     paths={
#         DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_train_20fewshots",
#         DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_test_20fewshots",
#     },
#     prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: Positive, Negative, or Neutral.

# Guidelines:
# - Choose Positive if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
# - Choose Negative if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
# - Choose Neutral ONLY IF the statement is purely factual with zero emotional content""",
#     valid_labels=["Positive", "Negative", "Neutral"],
#     completion_key="sentiment",
#     text_key="normalized_text",
#     audio_lookup_paths={
#         DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_train_audio_lookup",
#         DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_test_audio_lookup",
#     }
# )

VOXCELEB_CONFIG = DatasetConfig(
    name=DatasetType.VOXCELEB,
    paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_train_20fewshots",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_test_20fewshots",
    },
    prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: positive, negative, or neutral.

Guidelines:
- Choose positive if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose negative if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose neutral ONLY IF the statement is purely factual with zero emotional content""",
    valid_labels=["positive", "negative", "neutral"],
    completion_key="sentiment",
    text_key="normalized_text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_train_audio_lookup",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_test_audio_lookup",
    }
)

VOXCELEB_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.VOXCELEB_GREEK,
    paths=VOXCELEB_CONFIG.paths,
    prompt_template="""You are a sentiment analysis expert. Based on the input,, respond with EXACTLY ONE WORD from these options: alpha, beta, or gamma.

Guidelines:
- Choose alpha if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose beta if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose gamma ONLY IF the statement is purely factual with zero emotional content""",
    valid_labels=["alpha", "beta", "gamma"],
    label_mapping={"positive": "alpha", "negative": "beta", "neutral": "gamma"},
    audio_lookup_paths=VOXCELEB_CONFIG.audio_lookup_paths,
    text_key=VOXCELEB_CONFIG.text_key,
    completion_key=VOXCELEB_CONFIG.completion_key
)

VOXCELEB_PERMUTATIONS = [
    ["negative", "positive", "neutral"],
    ["negative", "neutral", "positive"],
    ["positive", "neutral", "negative"],
    ["positive", "negative", "neutral"],
    ["neutral", "negative", "positive"],
    ["neutral", "positive", "negative"]
]

# VOXCELEB_PERMUTATIONS = [
#     ["beta", "alpha", "gamma"],
#     ["beta", "gamma", "alpha"],
#     ["alpha", "gamma", "beta"],
#     ["alpha", "beta", "gamma"],
#     ["gamma", "beta", "alpha"],
#     ["gamma", "alpha", "beta"]
# ]

VOXCELEB_SWAP_CONFIGS = []
for perm in VOXCELEB_PERMUTATIONS:
    mapping = {orig: swapped for orig, swapped in zip(VOXCELEB_CONFIG.valid_labels, perm)}
    VOXCELEB_SWAP_CONFIGS.append(DatasetConfig(
        prompt_template=f"""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: {perm[0]}, {perm[1]}, or {perm[2]}.

Guidelines:
- Choose {perm[0]} if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose {perm[1]} if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose {perm[2]} ONLY IF the statement is purely factual with zero emotional content""",
        label_mapping=mapping,
        paths=VOXCELEB_CONFIG.paths,
        audio_lookup_paths=VOXCELEB_CONFIG.audio_lookup_paths,
        text_key=VOXCELEB_CONFIG.text_key,
        completion_key=VOXCELEB_CONFIG.completion_key,
        valid_labels=perm,
        name=DatasetType.VOXCELEB_SWAP
    ))




def get_voxceleb_swap_config():
    return random.choice(VOXCELEB_SWAP_CONFIGS) 