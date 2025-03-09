from .base_config import DatasetType, DatasetSplit, DatasetConfig
import random

VOXPOPULI_CONFIG = DatasetConfig(
    name=DatasetType.VOXPOPULI,
    paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_train_20fewshots",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_test_20fewshots",
    },
    prompt_template="""You are an Entity Type Classification system. For the given input, identify which of the following entity types are present:

- LAW: Laws, regulations, directives, and legal frameworks
- NORP: Nationalities, religious, or political groups
- ORG: Companies, agencies, institutions
- PERSON: People, including fictional characters
- PLACE: Countries, cities, locations
- QUANT: Numbers, quantities, percentages
- WHEN: Dates, times, durations, periods

Guidelines:
1. Return ONLY the entity type if present (e.g., 'PLACE', 'PERSON')
2. Return 'None' if no entity types are found
3. Be precise in identifying entity types""",
    valid_labels=["LAW", "NORP", "ORG", "PERSON", "PLACE", "QUANT", "WHEN"],
    completion_key="normalized_combined_ner",
    text_key="normalized_text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_train_audio_lookup",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_test_audio_lookup",
    }
)

VOXPOPULI_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.VOXPOPULI_GREEK,
    paths=VOXPOPULI_CONFIG.paths,
    prompt_template="""You are an Entity Type Classification system. For the given input, identify which of the following entity types are present:

- Zeta1: Laws, regulations, directives, and legal frameworks
- Zeta2: Nationalities, religious, or political groups
- Zeta3: Companies, agencies, institutions
- Zeta4: People, including fictional characters
- Zeta5: Countries, cities, locations
- Zeta6: Numbers, quantities, percentages
- Zeta7: Dates, times, durations, periods

Guidelines:
1. Return ONLY the entity type if present (e.g., 'Zeta5', 'Zeta4')
2. Return 'None' if no entity types are found
3. Be precise in identifying entity types""",
    valid_labels=["zeta1", "zeta2", "zeta3", "zeta4", "zeta5", "zeta6", "zeta7"],
    completion_key="normalized_combined_ner",
    text_key="normalized_text",
    audio_lookup_paths=VOXPOPULI_CONFIG.audio_lookup_paths,
    label_mapping={
        "LAW": "zeta1",
        "NORP": "zeta2",
        "ORG": "zeta3",
        "PERSON": "zeta4",
        "PLACE": "zeta5",
        "QUANT": "zeta6",
        "WHEN": "zeta7"
    }
)

VOXPOPULI_PERMUTATIONS = [
    ["NORP", "ORG", "PERSON", "PLACE", "QUANT", "WHEN", "LAW"],
    ["LAW", "NORP", "ORG", "PERSON", "PLACE", "QUANT", "WHEN"],
    ["WHEN", "LAW", "NORP", "ORG", "PERSON", "PLACE", "QUANT"],
    ["QUANT", "WHEN", "LAW", "NORP", "ORG", "PERSON", "PLACE"],
    ["PLACE", "QUANT", "WHEN", "LAW", "NORP", "ORG", "PERSON"],
    ["PERSON", "PLACE", "QUANT", "WHEN", "LAW", "NORP", "ORG"]
]

VOXPOPULI_SWAP_CONFIGS = []
for perm in VOXPOPULI_PERMUTATIONS:
    mapping = {orig: swapped for orig, swapped in zip(VOXPOPULI_CONFIG.valid_labels, perm)}
    VOXPOPULI_SWAP_CONFIGS.append(DatasetConfig(
        prompt_template=f"""You are an Entity Type Classification system. For the given input, identify which of the following entity types are present:

{chr(10).join(f'- {label}: {desc}' for label, desc in zip(perm, [
    "Laws, regulations, directives, and legal frameworks",
    "Nationalities, religious, or political groups",
    "Companies, agencies, institutions",
    "People, including fictional characters",
    "Countries, cities, locations",
    "Numbers, quantities, percentages",
    "Dates, times, durations, periods"
]))}

Guidelines:
1. Return ONLY the entity type if present (e.g., '{perm[4]}', '{perm[3]}')
2. Return 'None' if no entity types are found
3. Be precise in identifying entity types""",
        label_mapping=mapping,
        paths=VOXPOPULI_CONFIG.paths,
        audio_lookup_paths=VOXPOPULI_CONFIG.audio_lookup_paths,
        text_key=VOXPOPULI_CONFIG.text_key,
        completion_key=VOXPOPULI_CONFIG.completion_key,
        valid_labels=perm,
        name=DatasetType.VOXPOPULI_SWAP
    ))

def get_voxpopuli_swap_config():
    return random.choice(VOXPOPULI_SWAP_CONFIGS) 