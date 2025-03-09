from .base_config import DatasetType, DatasetSplit, DatasetConfig
import random

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

HVB_GREEK_CONFIG = DatasetConfig(
    name=DatasetType.HVB_GREEK,
    paths=HVB_CONFIG.paths,
    prompt_template="""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
- Foo: Shows understanding or receipt of information
- Bar: Expresses agreement
- Baz: Expresses disagreement
- Qux: General response to a question
- Quux: Expression of regret or sorry
- Corge: Brief verbal/textual feedback
- Grault: Speech repairs, repetitions, or corrections
- Garply: Actions that don't fit other categories
- Waldo: Questions to verify understanding
- Fred: General information-seeking questions
- Plugh: Requests for repetition
- Xyzzy: Self-directed speech
- Thud: Concluding statements
- Wibble: General statements or information
- Wobble: Instructions or directions
- Wubble: Opening statements or greetings
- Flob: Statements describing issues or problems
- Zoop: Expressions of gratitude""",
    valid_labels=[
        "foo", "bar", "baz", "qux", "quux", 
        "corge", "grault", "garply", "waldo", "fred",
        "plugh", "xyzzy", "thud", "wibble", "wobble",
        "wubble", "flob", "zoop"
    ],
    completion_key=HVB_CONFIG.completion_key,
    text_key=HVB_CONFIG.text_key,
    audio_lookup_paths=HVB_CONFIG.audio_lookup_paths,
    label_mapping={
        "acknowledge": "foo",
        "answer_agree": "bar",
        "answer_dis": "baz",
        "answer_general": "qux",
        "apology": "quux",
        "backchannel": "corge",
        "disfluency": "grault",
        "other": "garply",
        "question_check": "waldo",
        "question_general": "fred",
        "question_repeat": "plugh",
        "self": "xyzzy",
        "statement_close": "thud",
        "statement_general": "wibble",
        "statement_instruct": "wobble",
        "statement_open": "wubble",
        "statement_problem": "flob",
        "thanks": "zoop"
    }
)

# Include HVB_PERMUTATIONS and HVB_SWAP_CONFIGS from original file
HVB_PERMUTATIONS = [
    # Original order
    HVB_CONFIG.valid_labels,
    # Add other permutations as in original file
]

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

HVB_SWAP_CONFIGS = []
for perm in HVB_PERMUTATIONS:
    mapping = {orig: swapped for orig, swapped in zip(HVB_CONFIG.valid_labels, perm)}
    descriptions = {label: desc for label, desc in zip(HVB_CONFIG.valid_labels, HVB_DESCRIPTIONS)}
    HVB_SWAP_CONFIGS.append(DatasetConfig(
        prompt_template=f"""You are a dialogue analysis expert for banking conversations. Based on the statement below, identify all applicable dialogue actions from the following options:

Available dialogue actions:
{chr(10).join(f'- {label}: {descriptions[orig]}' for label, orig in zip(perm, HVB_CONFIG.valid_labels))}

Guidelines:
- Multiple actions can apply to a single statement
- List all applicable actions separated by commas
- Consider the banking context when analyzing
- Be precise in identifying the dialogue actions""",
        label_mapping=mapping,
        valid_labels=perm,
        name=DatasetType.HVB_SWAP,
        paths=HVB_CONFIG.paths,
        audio_lookup_paths=HVB_CONFIG.audio_lookup_paths,
        text_key=HVB_CONFIG.text_key,
        completion_key=HVB_CONFIG.completion_key
    ))

def get_hvb_swap_config():
    return random.choice(HVB_SWAP_CONFIGS) 