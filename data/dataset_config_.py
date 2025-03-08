from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import random
import logging

logger = logging.getLogger(__name__)

class DatasetType(Enum):
    # Original datasets
    VOXCELEB = "voxceleb"
    HVB = "hvb"
    VOXPOPULI = "voxpopuli"
    
    # Symbol/Greek versions
    VOXCELEB_GREEK = "voxceleb_greek"
    HVB_GREEK = "hvb_greek"
    VOXPOPULI_GREEK = "voxpopuli_greek"
    
    # Swap/Permuted versions
    VOXCELEB_SWAP = "voxceleb_swap"
    HVB_SWAP = "hvb_swap"
    VOXPOPULI_SWAP = "voxpopuli_swap"

class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"

@dataclass
class DatasetConfig:
    name: DatasetType
    paths: Dict[DatasetSplit, str]
    prompt_template: str
    valid_labels: List[str]
    completion_key: str
    text_key: str
    audio_lookup_paths: Dict[DatasetSplit, str] = None
    label_mapping: Dict[str, str] = None

    def get_path(self, split: DatasetSplit) -> str:
        return self.paths[split]
    
    def get_audio_lookup_path(self, split: DatasetSplit) -> Optional[str]:
        if self.audio_lookup_paths:
            return self.audio_lookup_paths.get(split)
        return None

@dataclass
class SwapConfig:
    """Configuration for label swapping"""
    prompt_template: str
    label_mapping: Dict[str, str]  # Original label -> Swapped label

# ===== DATASET CONFIGURATIONS =====

# ===== VOXCELEB =====
VOXCELEB_CONFIG = DatasetConfig(
    name=DatasetType.VOXCELEB,
    paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_train_20fewshots",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_test_20fewshots",
    },
    prompt_template="""You are a sentiment analysis expert. Based on the statement below, respond with EXACTLY ONE WORD from these options: positive, negative, or Neutral.

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

# Greek/Symbol version for VoxCeleb
VOXCELEB_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.VOXCELEB_GREEK,
    paths=VOXCELEB_CONFIG.paths,
    prompt_template="""You are a sentiment analysis expert. Based on the statement below, respond with EXACTLY ONE WORD from these options: alpha, beta, or gamma.

Guidelines:
- Choose alpha if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose beta if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose gamma ONLY IF the statement is purely factual with zero emotional content""",
    valid_labels=["alpha", "beta", "gamma"],
    completion_key="sentiment",
    text_key="normalized_text",
    audio_lookup_paths=VOXCELEB_CONFIG.audio_lookup_paths,
    label_mapping={"positive": "alpha", "negative": "beta", "neutral": "gamma"}
)

# Define fixed permutations for each dataset
VOXCELEB_PERMUTATIONS = [
    ["negative", "positive", "neutral"],
    ["negative", "neutral", "positive"],
    ["positive", "neutral", "negative"],
    ["positive", "negative", "neutral"],
    ["neutral", "negative", "positive"],
    ["neutral", "positive", "negative"]
]

# Modify the swap config creation
VOXCELEB_SWAP_CONFIGS = []
for perm in VOXCELEB_PERMUTATIONS:
    mapping = {orig: swapped for orig, swapped in zip(VOXCELEB_CONFIG.valid_labels, perm)}
    VOXCELEB_SWAP_CONFIGS.append(SwapConfig(
        prompt_template=f"""You are a sentiment analysis expert. Based on the statement below, respond with EXACTLY ONE WORD from these options: {perm[0]}, {perm[1]}, or {perm[2]}.

Guidelines:
- Choose {perm[0]} if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose {perm[1]} if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose {perm[2]} ONLY IF the statement is purely factual with zero emotional content""",
        label_mapping=mapping
    ))

# ===== HVB =====
HVB_CONFIG = DatasetConfig(
    name=DatasetType.HVB,
    paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_train_20fewshots",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_test_20fewshots",
    },
    prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- acknowledge: Shows understanding or receipt of information
- answer_agree: Expresses agreement
- answer_dis: Expresses disagreement
- answer_general: General response to a question
- apology: Expression of regret or sorry
- backchannel: Brief verbal/textual feedback (like "uh-huh", "mm-hmm")
- disfluency: Speech repairs, repetitions, or corrections
- other: Actions that don't fit other categories
- question_check: Questions to verify understanding
- question_general: General information-seeking questions
- question_repeat: Requests for repetition
- self: Self-directed speech
- statement_close: Concluding statements
- statement_general: General statements or information
- statement_instruct: Instructions or directions
- statement_open: Opening statements or greetings
- statement_problem: Statements describing issues or problems
- thanks: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
    valid_labels=[
        "acknowledge", "answer_agree", "answer_dis", "answer_general",
        "apology", "backchannel", "disfluency", "other",
        "question_check", "question_general", "question_repeat",
        "self", "statement_close", "statement_general",
        "statement_instruct", "statement_open", "statement_problem",
        "thanks"
    ],
    completion_key="dialog_acts",
    text_key="text",
    audio_lookup_paths={
        DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_train_audio_lookup",
        DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_test_audio_lookup",
    }
)

# Greek/Symbol version for HVB
HVB_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.HVB_GREEK,
    paths=HVB_CONFIG.paths,
    prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- Alpha: Shows understanding or receipt of information
- Beta: Expresses agreement
- Gamma: Expresses disagreement
- Delta: General response to a question
- Epsilon: Expression of regret or sorry
- Zeta: Brief verbal/textual feedback (like "uh-huh", "mm-hmm")
- Eta: Speech repairs, repetitions, or corrections
- Theta: Actions that don't fit other categories
- Iota: Questions to verify understanding
- Kappa: General information-seeking questions
- Lambda: Requests for repetition
- Mu: Self-directed speech
- Nu: Concluding statements
- Xi: General statements or information
- Omicron: Instructions or directions
- Pi: Opening statements or greetings
- Rho: Statements describing issues or problems
- Sigma: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
    valid_labels=[
        "alpha", "beta", "gamma", "delta", "epsilon", 
        "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "omicron",
        "pi", "rho", "sigma"
    ],
    completion_key="dialog_acts",
    text_key="text",
    audio_lookup_paths=HVB_CONFIG.audio_lookup_paths,
    label_mapping={
        "acknowledge": "alpha",
        "answer_agree": "beta",
        "answer_dis": "gamma",
        "answer_general": "delta",
        "apology": "epsilon",
        "backchannel": "zeta",
        "disfluency": "eta",
        "other": "theta",
        "question_check": "iota",
        "question_general": "kappa",
        "question_repeat": "lambda",
        "self": "mu",
        "statement_close": "nu",
        "statement_general": "xi",
        "statement_instruct": "omicron",
        "statement_open": "pi",
        "statement_problem": "rho",
        "thanks": "sigma"
    }
)

# Define fixed permutations for HVB
HVB_PERMUTATIONS = [
    # Original order
    ["acknowledge", "answer_agree", "answer_dis", "answer_general", "apology", 
     "backchannel", "disfluency", "other", "question_check", "question_general",
     "question_repeat", "self", "statement_close", "statement_general",
     "statement_instruct", "statement_open", "statement_problem", "thanks"],
    
    # Permutation 2: Rotate questions to front
    ["question_check", "question_general", "question_repeat", "acknowledge", 
     "answer_agree", "answer_dis", "answer_general", "apology", "backchannel", 
     "disfluency", "other", "self", "statement_close", "statement_general",
     "statement_instruct", "statement_open", "statement_problem", "thanks"],
    
    # Permutation 3: Rotate statements to front
    ["statement_close", "statement_general", "statement_instruct", "statement_open", 
     "statement_problem", "acknowledge", "answer_agree", "answer_dis", "answer_general", 
     "apology", "backchannel", "disfluency", "other", "question_check", 
     "question_general", "question_repeat", "self", "thanks"],
    
    # Permutation 4: Rotate answers to front
    ["answer_agree", "answer_dis", "answer_general", "acknowledge", "apology", 
     "backchannel", "disfluency", "other", "question_check", "question_general",
     "question_repeat", "self", "statement_close", "statement_general",
     "statement_instruct", "statement_open", "statement_problem", "thanks"],
    
    # Permutation 5: Group similar actions
    ["acknowledge", "backchannel", "disfluency", "self", "answer_agree", 
     "answer_dis", "answer_general", "question_check", "question_general",
     "question_repeat", "statement_close", "statement_general", "statement_instruct", 
     "statement_open", "statement_problem", "apology", "thanks", "other"],
    
    # Permutation 6: Reverse original
    ["thanks", "statement_problem", "statement_open", "statement_instruct", 
     "statement_general", "statement_close", "self", "question_repeat",
     "question_general", "question_check", "other", "disfluency", "backchannel", 
     "apology", "answer_general", "answer_dis", "answer_agree", "acknowledge"],
    
    # Permutation 7: Group by conversation flow
    ["statement_open", "question_general", "answer_general", "question_check", 
     "answer_agree", "answer_dis", "acknowledge", "backchannel", "disfluency",
     "question_repeat", "statement_general", "statement_problem", "statement_instruct", 
     "apology", "self", "other", "statement_close", "thanks"],
    
    # Permutation 8: Group by response type
    ["question_general", "question_check", "question_repeat", "answer_general", 
     "answer_agree", "answer_dis", "statement_general", "statement_open",
     "statement_close", "statement_problem", "statement_instruct", "acknowledge", 
     "backchannel", "disfluency", "self", "apology", "thanks", "other"],
    
    # Permutation 9: Alternate question/answer/statement
    ["question_general", "answer_general", "statement_general", "question_check", 
     "answer_agree", "statement_open", "question_repeat", "answer_dis",
     "statement_close", "acknowledge", "backchannel", "statement_problem", 
     "disfluency", "self", "statement_instruct", "apology", "thanks", "other"],
    
    # Permutation 10: Group by formality
    ["statement_instruct", "statement_general", "question_general", "answer_general", 
     "statement_problem", "question_check", "answer_agree", "answer_dis",
     "statement_open", "statement_close", "acknowledge", "question_repeat", 
     "backchannel", "disfluency", "self", "apology", "thanks", "other"]
]

# Descriptions for HVB labels - will be used in the same order for all permutations
HVB_DESCRIPTIONS = [
    "Shows understanding or receipt of information",
    "Expresses agreement",
    "Expresses disagreement",
    "General response to a question",
    "Expression of regret or sorry",
    "Brief verbal/textual feedback (like 'uh-huh', 'mm-hmm')",
    "Speech repairs, repetitions, or corrections",
    "Actions that don't fit other categories",
    "Questions to verify understanding",
    "General information-seeking questions",
    "Requests for repetition",
    "Self-directed speech",
    "Concluding statements",
    "General statements or information",
    "Instructions or directions",
    "Opening statements or greetings",
    "Statements describing issues or problems",
    "Expressions of gratitude"
]

# Modify the swap config creation
HVB_SWAP_CONFIGS = []
for perm in HVB_PERMUTATIONS:
    mapping = {orig: swapped for orig, swapped in zip(HVB_CONFIG.valid_labels, perm)}
    descriptions = {label: desc for label, desc in zip(HVB_CONFIG.valid_labels, HVB_DESCRIPTIONS)}
    HVB_SWAP_CONFIGS.append(SwapConfig(
        prompt_template=f"""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
{chr(10).join(f'- {label}: {descriptions[orig]}' for label, orig in zip(perm, HVB_CONFIG.valid_labels))}

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping=mapping
    ))

# ===== VOXPOPULI =====
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

# Greek/Symbol version for VoxPopuli
VOXPOPULI_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.VOXPOPULI_GREEK,
    paths=VOXPOPULI_CONFIG.paths,
    prompt_template="""You are an Entity Type Classification system. For the given input, identify which of the following entity types are present:

- Alpha: Laws, regulations, directives, and legal frameworks
- Beta: Nationalities, religious, or political groups
- Gamma: Companies, agencies, institutions
- Delta: People, including fictional characters
- Epsilon: Countries, cities, locations
- Zeta: Numbers, quantities, percentages
- Eta: Dates, times, durations, periods

Guidelines:
1. Return ONLY the entity type if present (e.g., 'Epsilon', 'Delta')
2. Return 'None' if no entity types are found
3. Be precise in identifying entity types""",
    valid_labels=["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"],
    completion_key="normalized_combined_ner",
    text_key="normalized_text",
    audio_lookup_paths=VOXPOPULI_CONFIG.audio_lookup_paths,
    label_mapping={
        "LAW": "alpha",
        "NORP": "beta", 
        "ORG": "gamma",
        "PERSON": "delta",
        "PLACE": "epsilon",
        "QUANT": "zeta",
        "WHEN": "eta"
    }
)

# Define fixed permutations for VOXPOPULI
VOXPOPULI_PERMUTATIONS = [
    ["NORP", "ORG", "PERSON", "PLACE", "QUANT", "WHEN", "LAW"],
    ["LAW", "NORP", "ORG", "PERSON", "PLACE", "QUANT", "WHEN"],
    ["WHEN", "LAW", "NORP", "ORG", "PERSON", "PLACE", "QUANT"],
    ["QUANT", "WHEN", "LAW", "NORP", "ORG", "PERSON", "PLACE"],
    ["PLACE", "QUANT", "WHEN", "LAW", "NORP", "ORG", "PERSON"],
    ["PERSON", "PLACE", "QUANT", "WHEN", "LAW", "NORP", "ORG"]
]

# Modify the swap config creation
VOXPOPULI_SWAP_CONFIGS = []
for perm in VOXPOPULI_PERMUTATIONS:
    mapping = {orig: swapped for orig, swapped in zip(VOXPOPULI_CONFIG.valid_labels, perm)}
    VOXPOPULI_SWAP_CONFIGS.append(SwapConfig(
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
        label_mapping=mapping
    ))

# ===== HELPER FUNCTIONS =====

DATASET_CONFIGS = {
    DatasetType.VOXCELEB: VOXCELEB_CONFIG,
    DatasetType.VOXCELEB_GREEK: VOXCELEB_GREEK_CONFIG,
    DatasetType.HVB: HVB_CONFIG,
    DatasetType.HVB_GREEK: HVB_GREEK_CONFIG,
    DatasetType.VOXPOPULI: VOXPOPULI_CONFIG,
    DatasetType.VOXPOPULI_GREEK: VOXPOPULI_GREEK_CONFIG
}

def get_dataset_config(dataset_type: DatasetType):
    """Get the configuration for a specific dataset type."""
    return DATASET_CONFIGS.get(dataset_type)

def get_swap_config(dataset_type: DatasetType) -> Tuple[str, Dict[str, str]]:
    """Get a random swap configuration for a dataset type"""
    if dataset_type == DatasetType.VOXCELEB_SWAP:
        config = random.choice(VOXCELEB_SWAP_CONFIGS)
        return config.prompt_template, config.label_mapping
    elif dataset_type == DatasetType.HVB_SWAP:
        config = random.choice(HVB_SWAP_CONFIGS)
        return config.prompt_template, config.label_mapping
    elif dataset_type == DatasetType.VOXPOPULI_SWAP:
        config = random.choice(VOXPOPULI_SWAP_CONFIGS)
        return config.prompt_template, config.label_mapping
    else:
        raise ValueError(f"No swap configuration available for dataset type: {dataset_type}")

def apply_label_mapping(examples: List[Dict], label_mapping: Dict[str, str]) -> List[Dict]:
    """Apply label mapping to a list of examples"""
    for example in examples:
        if "sentiment" in example:
            if example["sentiment"] in label_mapping:
                example["sentiment"] = label_mapping[example["sentiment"]]
        elif "dialog_acts" in example:
            acts = example["dialog_acts"].split(",")
            mapped_acts = []
            for act in acts:
                act = act.strip()
                if act in label_mapping:
                    mapped_acts.append(label_mapping[act])
                else:
                    mapped_acts.append(act)
            example["dialog_acts"] = ",".join(mapped_acts)
        elif "language" in example:
            if example["language"] in label_mapping:
                example["language"] = label_mapping[example["language"]]
    return examples

__all__ = ['DatasetType', 'get_dataset_config']
