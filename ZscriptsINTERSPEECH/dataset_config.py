from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

class DatasetType(Enum):
    VOXCELEB = "voxceleb"
    VOXCELEB_SWAP = "voxceleb_swap"
    HVB = "hvb"
    HVB_GREEK = "hvb_greek"
    HVB_SWAP = "hvb_swap"
    VOXPOPULI = "voxpopuli"

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
    audio_lookup_paths: Dict[DatasetSplit, str] = None  # Updated to have paths per split
    label_mapping: Dict[str, str] = None

    def get_path(self, split: DatasetSplit) -> str:
        return self.paths[split]
    
    def get_audio_lookup_path(self, split: DatasetSplit) -> Optional[str]:
        if self.audio_lookup_paths:
            return self.audio_lookup_paths.get(split)
        return None

@dataclass
class SwapConfig:
    """Configuration for sentiment label swapping"""
    prompt_template: str
    label_mapping: Dict[str, str]  # Original label -> Swapped label

@dataclass
class HVBSwapConfig:
    """Configuration for HVB swap"""
    prompt_template: str
    label_mapping: Dict[str, str]  # Original label -> Swapped label

# Define different swap configurations
SWAP_CONFIGS = [
    SwapConfig(
        prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: alpha, beta, or gamma.

Guidelines:
- Choose beta if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose alpha if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose gamma ONLY IF the statement is purely factual with zero emotional content""",
        label_mapping={
            "Positive": "beta",
            "Negative": "alpha",
            "Neutral": "gamma"
        }
    ),
    SwapConfig(
        prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: alpha, beta, or gamma.

Guidelines:
- Choose gamma if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose alpha if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose beta ONLY IF the statement is purely factual with zero emotional content""",
        label_mapping={
            "Positive": "gamma",
            "Negative": "alpha",
            "Neutral": "beta"
        }
    ),
    SwapConfig(
        prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: alpha, beta, or gamma.

Guidelines:
- Choose gamma if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose beta if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose alpha ONLY IF the statement is purely factual with zero emotional content""",
        label_mapping={
            "Positive": "gamma",
            "Negative": "beta",
            "Neutral": "alpha"
        }
    ),
    SwapConfig(
        prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: alpha, beta, or gamma.

Guidelines:
- Choose alpha if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose gamma if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose beta ONLY IF the statement is purely factual with zero emotional content""",
        label_mapping={
            "Positive": "gamma",
            "Negative": "alpha",
            "Neutral": "beta"
        }
    ),
    SwapConfig(
        prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: alpha, beta, or gamma.

Guidelines:
- Choose beta if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose alpha if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose gamma ONLY IF the statement is purely factual with zero emotional content""",
        label_mapping={
            "Positive": "alpha",
            "Negative": "beta",
            "Neutral": "gamma"
        }
    ),
    SwapConfig(
        prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: alpha, beta, or gamma.

Guidelines:
- Choose alpha if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose beta if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose gamma ONLY IF the statement is purely factual with zero emotional content""",
        label_mapping={
            "Positive": "alpha",
            "Negative": "beta",
            "Neutral": "gamma"
        }
    )
]

# Move this to the top, before DATASET_CONFIGS
HVB_GREEK_MAPPING = {
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


HVB_GREEK_PROMPT = """You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- alpha: Shows understanding or receipt of information
- beta: Expresses agreement
- gamma: Expresses disagreement
- delta: General response to a question
- epsilon: Expression of regret or sorry
- zeta: Brief verbal/textual feedback
- eta: Speech repairs, repetitions, or corrections
- theta: Actions that don't fit other categories
- iota: Questions to verify understanding
- kappa: General information-seeking questions
- lambda: Requests for repetition
- mu: Self-directed speech
- nu: Concluding statements
- xi: General statements or information
- omicron: Instructions or directions
- pi: Opening statements or greetings
- rho: Statements describing issues or problems
- sigma: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""" 

HVB_SWAP_CONFIGS = [
    # Configuration 1: Complete role reversal
    HVBSwapConfig(
        prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from these categories:

Available dialogue actions:
- acknowledge: Expresses disagreement or rejection
- backchannel: Shows regret or apologizes
- answer_agree: Brief verbal/textual feedback (like "uh-huh", "mm-hmm")
- answer_dis: Shows understanding or receipt of information
- answer_general: Requests for repetition
- apology: Expresses agreement or acceptance
- disfluency: General information-seeking questions
- other: Instructions or directions
- question_check: General response to a question
- question_general: Speech repairs, repetitions, or corrections
- question_repeat: Actions that don't fit other categories
- self: Opening statements or greetings
- statement_close: Statements describing issues or problems
- statement_general: Self-directed speech
- statement_instruct: Expressions of gratitude
- statement_open: Questions to verify understanding
- statement_problem: Concluding statements
- thanks: General statements or information

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping={
            "acknowledge": "answer_dis",
            "backchannel": "apology",
            "answer_agree": "backchannel",
            "answer_dis": "acknowledge",
            "answer_general": "question_repeat",
            "apology": "answer_agree",
            "disfluency": "question_general",
            "other": "statement_instruct",
            "question_check": "answer_general",
            "question_general": "disfluency",
            "question_repeat": "other",
            "self": "statement_open",
            "statement_close": "statement_problem",
            "statement_general": "self",
            "statement_instruct": "thanks",
            "statement_open": "question_check",
            "statement_problem": "statement_close",
            "thanks": "statement_general"
        }
    ),

    # Configuration 2: Question-Statement Swap
    HVBSwapConfig(
        prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from these categories:

Available dialogue actions:
- acknowledge: General statements or information
- backchannel: Questions to verify understanding
- answer_agree: Opening statements or greetings
- answer_dis: Concluding statements
- answer_general: Instructions or directions
- apology: Statements describing issues or problems
- disfluency: Brief verbal/textual feedback
- other: Shows understanding or receipt of information
- question_check: Expresses agreement
- question_general: Expresses disagreement
- question_repeat: General response to a question
- self: Expression of regret or sorry
- statement_close: Speech repairs, repetitions, or corrections
- statement_general: Actions that don't fit other categories
- statement_instruct: Requests for repetition
- statement_open: Self-directed speech
- statement_problem: Expressions of gratitude
- thanks: General information-seeking questions

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping={
            "acknowledge": "statement_general",
            "backchannel": "question_check",
            "answer_agree": "statement_open",
            "answer_dis": "statement_close",
            "answer_general": "statement_instruct",
            "apology": "statement_problem",
            "disfluency": "backchannel",
            "other": "acknowledge",
            "question_check": "answer_agree",
            "question_general": "answer_dis",
            "question_repeat": "answer_general",
            "self": "apology",
            "statement_close": "disfluency",
            "statement_general": "other",
            "statement_instruct": "question_repeat",
            "statement_open": "self",
            "statement_problem": "thanks",
            "thanks": "question_general"
        }
    ),

    # Configuration 3: Response Type Swap
    HVBSwapConfig(
        prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from these categories:

Available dialogue actions:
- acknowledge: Instructions or directions
- backchannel: General response to a question
- answer_agree: Questions to verify understanding
- answer_dis: Brief verbal/textual feedback
- answer_general: Shows understanding or receipt of information
- apology: General information-seeking questions
- disfluency: Expresses agreement
- other: Concluding statements
- question_check: Expression of regret or sorry
- question_general: Statements describing issues or problems
- question_repeat: Opening statements or greetings
- self: General statements or information
- statement_close: Expresses disagreement
- statement_general: Requests for repetition
- statement_instruct: Actions that don't fit other categories
- statement_open: Speech repairs, repetitions, or corrections
- statement_problem: Self-directed speech
- thanks: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping={
            "acknowledge": "statement_instruct",
            "backchannel": "answer_general",
            "answer_agree": "question_check",
            "answer_dis": "backchannel",
            "answer_general": "acknowledge",
            "apology": "question_general",
            "disfluency": "answer_agree",
            "other": "statement_close",
            "question_check": "apology",
            "question_general": "statement_problem",
            "question_repeat": "statement_open",
            "self": "statement_general",
            "statement_close": "answer_dis",
            "statement_general": "question_repeat",
            "statement_instruct": "other",
            "statement_open": "disfluency",
            "statement_problem": "self",
            "thanks": "thanks"
        }
    ),

    # Configuration 4: Feedback-Statement Swap
    HVBSwapConfig(
        prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from these categories:

Available dialogue actions:
- acknowledge: Opening statements or greetings
- backchannel: Concluding statements
- answer_agree: Instructions or directions
- answer_dis: Statements describing issues or problems
- answer_general: General statements or information
- apology: Self-directed speech
- disfluency: Requests for repetition
- other: Questions to verify understanding
- question_check: Brief verbal/textual feedback
- question_general: Shows understanding or receipt of information
- question_repeat: Expresses agreement
- self: General response to a question
- statement_close: Expression of regret or sorry
- statement_general: Actions that don't fit other categories
- statement_instruct: Expresses disagreement
- statement_open: General information-seeking questions
- statement_problem: Speech repairs, repetitions, or corrections
- thanks: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping={
            "acknowledge": "statement_open",
            "backchannel": "statement_close",
            "answer_agree": "statement_instruct",
            "answer_dis": "statement_problem",
            "answer_general": "statement_general",
            "apology": "self",
            "disfluency": "question_repeat",
            "other": "question_check",
            "question_check": "backchannel",
            "question_general": "acknowledge",
            "question_repeat": "answer_agree",
            "self": "answer_general",
            "statement_close": "apology",
            "statement_general": "other",
            "statement_instruct": "answer_dis",
            "statement_open": "question_general",
            "statement_problem": "disfluency",
            "thanks": "thanks"
        }
    ),

    # Configuration 5: Action-Response Swap
    HVBSwapConfig(
        prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from these categories:

Available dialogue actions:
- acknowledge: Requests for repetition
- backchannel: General information-seeking questions
- answer_agree: Speech repairs, repetitions, or corrections
- answer_dis: Shows understanding or receipt of information
- answer_general: Questions to verify understanding
- apology: Opening statements or greetings
- disfluency: Statements describing issues or problems
- other: Expressions of gratitude
- question_check: General statements or information
- question_general: Instructions or directions
- question_repeat: Brief verbal/textual feedback
- self: Concluding statements
- statement_close: Expresses agreement
- statement_general: Expression of regret or sorry
- statement_instruct: Actions that don't fit other categories
- statement_open: General response to a question
- statement_problem: Expresses disagreement
- thanks: Self-directed speech

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping={
            "acknowledge": "question_repeat",
            "backchannel": "question_general",
            "answer_agree": "disfluency",
            "answer_dis": "acknowledge",
            "answer_general": "question_check",
            "apology": "statement_open",
            "disfluency": "statement_problem",
            "other": "thanks",
            "question_check": "statement_general",
            "question_general": "statement_instruct",
            "question_repeat": "backchannel",
            "self": "statement_close",
            "statement_close": "answer_agree",
            "statement_general": "apology",
            "statement_instruct": "other",
            "statement_open": "answer_general",
            "statement_problem": "answer_dis",
            "thanks": "self"
        }
    ),

    # Configuration 6: Communication Flow Swap
    HVBSwapConfig(
        prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from these categories:

Available dialogue actions:
- acknowledge: Statements describing issues or problems
- backchannel: Self-directed speech
- answer_agree: General statements or information
- answer_dis: Questions to verify understanding
- answer_general: Speech repairs, repetitions, or corrections
- apology: Brief verbal/textual feedback
- disfluency: General response to a question
- other: Opening statements or greetings
- question_check: Expression of regret or sorry
- question_general: Expresses disagreement
- question_repeat: Shows understanding or receipt of information
- self: Instructions or directions
- statement_close: General information-seeking questions
- statement_general: Expresses agreement
- statement_instruct: Concluding statements
- statement_open: Actions that don't fit other categories
- statement_problem: Requests for repetition
- thanks: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping={
            "acknowledge": "statement_problem",
            "backchannel": "self",
            "answer_agree": "statement_general",
            "answer_dis": "question_check",
            "answer_general": "disfluency",
            "apology": "backchannel",
            "disfluency": "answer_general",
            "other": "statement_open",
            "question_check": "apology",
            "question_general": "answer_dis",
            "question_repeat": "acknowledge",
            "self": "statement_instruct",
            "statement_close": "question_general",
            "statement_general": "answer_agree",
            "statement_instruct": "statement_close",
            "statement_open": "other",
            "statement_problem": "question_repeat",
            "thanks": "thanks"
        }
    ),

    # Configuration 7: Interaction Pattern Swap
    HVBSwapConfig(
        prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from these categories:

Available dialogue actions:
- acknowledge: General information-seeking questions
- backchannel: Statements describing issues or problems
- answer_agree: Questions to verify understanding
- answer_dis: General statements or information
- answer_general: Expression of regret or sorry
- apology: Actions that don't fit other categories
- disfluency: Opening statements or greetings
- other: Shows understanding or receipt of information
- question_check: General response to a question
- question_general: Concluding statements
- question_repeat: Expresses agreement
- self: Brief verbal/textual feedback
- statement_close: Speech repairs, repetitions, or corrections
- statement_general: Expresses disagreement
- statement_instruct: Self-directed speech
- statement_open: Requests for repetition
- statement_problem: Instructions or directions
- thanks: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping={
            "acknowledge": "question_general",
            "backchannel": "statement_problem",
            "answer_agree": "question_check",
            "answer_dis": "statement_general",
            "answer_general": "apology",
            "apology": "other",
            "disfluency": "statement_open",
            "other": "acknowledge",
            "question_check": "answer_general",
            "question_general": "statement_close",
            "question_repeat": "answer_agree",
            "self": "backchannel",
            "statement_close": "disfluency",
            "statement_general": "answer_dis",
            "statement_instruct": "self",
            "statement_open": "question_repeat",
            "statement_problem": "statement_instruct",
            "thanks": "thanks"
        }
    ),

    # Configuration 8: Response-Question Swap
    HVBSwapConfig(
        prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from these categories:

Available dialogue actions:
- acknowledge: Speech repairs, repetitions, or corrections
- backchannel: Instructions or directions
- answer_agree: Statements describing issues or problems
- answer_dis: Opening statements or greetings
- answer_general: Actions that don't fit other categories
- apology: Shows understanding or receipt of information
- disfluency: General information-seeking questions
- other: Expression of regret or sorry
- question_check: Self-directed speech
- question_general: Brief verbal/textual feedback
- question_repeat: General statements or information
- self: Questions to verify understanding
- statement_close: Expresses agreement
- statement_general: Expresses disagreement
- statement_instruct: General response to a question
- statement_open: Concluding statements
- statement_problem: Requests for repetition
- thanks: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping={
            "acknowledge": "disfluency",
            "backchannel": "statement_instruct",
            "answer_agree": "statement_problem",
            "answer_dis": "statement_open",
            "answer_general": "other",
            "apology": "acknowledge",
            "disfluency": "question_general",
            "other": "apology",
            "question_check": "self",
            "question_general": "backchannel",
            "question_repeat": "statement_general",
            "self": "question_check",
            "statement_close": "answer_agree",
            "statement_general": "answer_dis",
            "statement_instruct": "answer_general",
            "statement_open": "statement_close",
            "statement_problem": "question_repeat",
            "thanks": "thanks"
        }
    ),

    # Configuration 9: Statement-Feedback Swap
    HVBSwapConfig(
        prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from these categories:

Available dialogue actions:
- acknowledge: Concluding statements
- backchannel: Expresses agreement
- answer_agree: Shows understanding or receipt of information
- answer_dis: General information-seeking questions
- answer_general: Statements describing issues or problems
- apology: Speech repairs, repetitions, or corrections
- disfluency: Expression of regret or sorry
- other: General statements or information
- question_check: Instructions or directions
- question_general: Requests for repetition
- question_repeat: Actions that don't fit other categories
- self: Opening statements or greetings
- statement_close: Brief verbal/textual feedback
- statement_general: Questions to verify understanding
- statement_instruct: Expresses disagreement
- statement_open: General response to a question
- statement_problem: Self-directed speech
- thanks: Expressions of gratitude

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping={
            "acknowledge": "statement_close",
            "backchannel": "answer_agree",
            "answer_agree": "acknowledge",
            "answer_dis": "question_general",
            "answer_general": "statement_problem",
            "apology": "disfluency",
            "disfluency": "apology",
            "other": "statement_general",
            "question_check": "statement_instruct",
            "question_general": "question_repeat",
            "question_repeat": "other",
            "self": "statement_open",
            "statement_close": "backchannel",
            "statement_general": "question_check",
            "statement_instruct": "answer_dis",
            "statement_open": "answer_general",
            "statement_problem": "self",
            "thanks": "thanks"
        }
    ),

    # Configuration 10: orignal
    HVBSwapConfig(
        prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from these categories:

Available dialogue actions:
- acknowledge: Shows understanding or receipt of information
- answer_agree: Expresses agreement
- answer_dis: Expresses disagreement
- answer_general: General response to a question
- apology: Expression of regret or sorry
- backchannel: Brief verbal/textual feedback
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
        label_mapping={
            "acknowledge": "acknowledge",
            "answer_agree": "answer_agree",
            "answer_dis": "answer_dis",
            "answer_general": "answer_general",
            "apology": "apology",
            "backchannel": "backchannel",
            "disfluency": "disfluency",
            "other": "other",
            "question_check": "question_check",
            "question_general": "question_general",
            "question_repeat": "question_repeat",
            "self": "self",
            "statement_close": "statement_close",
            "statement_general": "statement_general",
            "statement_instruct": "statement_instruct",
            "statement_open": "statement_open",
            "statement_problem": "statement_problem",
            "thanks": "thanks"
        }
    )
]

# First define the HVB paths and configuration separately
HVB_PATHS = {
    DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_train_20fewshots",
    DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_test_20fewshots"
}

HVB_AUDIO_PATHS = {
    DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_train_audio_lookup",
    DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue-phase-2_hvb_test_audio_lookup"
}

HVB_VALID_LABELS = [
    "acknowledge", "answer_agree", "answer_dis", "answer_general",
    "apology", "backchannel", "disfluency", "other", "question_check",
    "question_general", "question_repeat", "self", "statement_close",
    "statement_general", "statement_instruct", "statement_open",
    "statement_problem", "thanks"
]

# Add VoxPopuli paths
VOXPOPULI_PATHS = {
    DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_train_20fewshots",
    DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_test_20fewshots"
}

VOXPOPULI_AUDIO_PATHS = {
    DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_train_audio_lookup",
    DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxpopuli_test_audio_lookup"
}

DATASET_CONFIGS = {
    DatasetType.VOXCELEB: DatasetConfig(
        name=DatasetType.VOXCELEB,
        paths={
            DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_train_20fewshots",
            DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_test_20fewshots",
        },
        prompt_template="""You are a sentiment analysis expert. Based on the input, respond with EXACTLY ONE WORD from these options: Positive, Negative, or Neutral.

Guidelines:
- Choose Positive if there is ANY hint of: approval, optimism, happiness, success, laughter, enjoyment, pride, or satisfaction
- Choose Negative if there is ANY hint of: criticism, pessimism, sadness, failure, frustration, anger, disappointment, or concern
- Choose Neutral ONLY IF the statement is purely factual with zero emotional content""",
        valid_labels=["Positive", "Negative", "Neutral"],
        completion_key="sentiment",
        text_key="normalized_text",
        audio_lookup_paths={
            DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_train_audio_lookup",
            DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_test_audio_lookup"
        }
    ),
    
    DatasetType.VOXCELEB_SWAP: DatasetConfig(
        name=DatasetType.VOXCELEB_SWAP,
        paths={
            DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_train_20fewshots",
            DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_test_20fewshots",
        },
        prompt_template="<DYNAMIC>",  # Will be replaced dynamically
        valid_labels=["Positive", "Negative", "Neutral"],
        completion_key="sentiment",
        text_key="normalized_text",
        audio_lookup_paths={
            DatasetSplit.TRAIN: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_train_audio_lookup",
            DatasetSplit.TEST: "/data2/neeraja/neeraja/data/asapp/slue_voxceleb_test_audio_lookup"
        }
    ),
    
    DatasetType.HVB: DatasetConfig(
        name=DatasetType.HVB,
        paths=HVB_PATHS,
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
        valid_labels=HVB_VALID_LABELS,
        completion_key="dialog_acts",
        text_key="text",
        audio_lookup_paths=HVB_AUDIO_PATHS
    ),
    
    DatasetType.HVB_GREEK: DatasetConfig(
        name=DatasetType.HVB_GREEK,
        paths=HVB_PATHS,
        prompt_template=HVB_GREEK_PROMPT,
        valid_labels=list(HVB_GREEK_MAPPING.values()),
        completion_key="dialog_acts",
        text_key="text",
        audio_lookup_paths=HVB_AUDIO_PATHS,
        label_mapping=HVB_GREEK_MAPPING
    ),
    
    DatasetType.HVB_SWAP: DatasetConfig(
        name=DatasetType.HVB_SWAP,
        paths=HVB_PATHS,  # Use the shared paths
        prompt_template="<DYNAMIC>",
        valid_labels=HVB_VALID_LABELS,  # Use the shared labels
        completion_key="dialog_acts",
        text_key="text",
        audio_lookup_paths=HVB_AUDIO_PATHS  # Use the shared audio paths
    ),
    
    DatasetType.VOXPOPULI: DatasetConfig(
        name=DatasetType.VOXPOPULI,
        paths=VOXPOPULI_PATHS,
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
        audio_lookup_paths=VOXPOPULI_AUDIO_PATHS
    ),
}

def get_swapped_config(dataset_type: DatasetType = None) -> Tuple[str, Dict[str, str]]:
    """Get a random swap configuration based on dataset type"""
    if dataset_type == DatasetType.HVB_SWAP:
        swap_config = random.choice(HVB_SWAP_CONFIGS)
        # swap_config = HVB_SWAP_CONFIGS[0]  # Using first config for consistency
        return swap_config.prompt_template, swap_config.label_mapping
    else:  # Default to VoxCeleb swap
        swap_config = random.choice(SWAP_CONFIGS)
        # swap_config = SWAP_CONFIGS[0]  # Using first config for consistency
        return swap_config.prompt_template, swap_config.label_mapping

def apply_label_mapping(examples: List[Dict], label_mapping: Dict[str, str]) -> List[Dict]:
    """Apply label mapping to few-shot examples"""
    mapped_examples = []
    for example in examples:
        mapped_example = example.copy()
        if isinstance(example['label'], list):
            mapped_example['label'] = [label_mapping.get(l, l) for l in example['label']]
        else:
            mapped_example['label'] = label_mapping.get(example['label'], example['label'])
        mapped_examples.append(mapped_example)
    return mapped_examples

# Prompt template for HVB Greek
