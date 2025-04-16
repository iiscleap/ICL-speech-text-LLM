from typing import Dict, List, Tuple
import logging
from .base_config import DatasetType, DatasetConfig, DatasetSplit
from .voxceleb_config import (
    VOXCELEB_CONFIG, 
    VOXCELEB_GREEK_CONFIG, 
    get_voxceleb_swap_config
)
from .hvb_config import (
    HVB_CONFIG, 
    HVB_GREEK_CONFIG, 
    get_hvb_swap_config
)
from .voxpopuli_config import (
    VOXPOPULI_CONFIG, 
    VOXPOPULI_GREEK_CONFIG, 
    get_voxpopuli_swap_config
)
from .sqa_config import SQA_CONFIG
from .vp_nel_config import VP_NEL_CONFIG

from .meld_config import (
    MELD_CONFIG,
    MELD_GREEK_CONFIG
)
from .meld_emotion_config import (
    MELD_EMOTION_CONFIG, 
    MELD_EMOTION_GREEK_CONFIG,
    get_meld_emotion_swap_config
)


logger = logging.getLogger(__name__)

DATASET_CONFIGS: Dict[DatasetType, DatasetConfig] = {
    DatasetType.VOXCELEB: VOXCELEB_CONFIG,
    DatasetType.VOXCELEB_GREEK: VOXCELEB_GREEK_CONFIG,
    DatasetType.HVB: HVB_CONFIG,
    DatasetType.HVB_GREEK: HVB_GREEK_CONFIG,
    DatasetType.VOXPOPULI: VOXPOPULI_CONFIG,
    DatasetType.VOXPOPULI_GREEK: VOXPOPULI_GREEK_CONFIG,
    DatasetType.SQA: SQA_CONFIG,
    DatasetType.VP_NEL: VP_NEL_CONFIG,
    DatasetType.MELD: MELD_CONFIG,
    DatasetType.MELD_GREEK: MELD_GREEK_CONFIG,
    DatasetType.MELD_EMOTION: MELD_EMOTION_CONFIG,
    DatasetType.MELD_EMOTION_GREEK: MELD_EMOTION_GREEK_CONFIG,
    DatasetType.MELD_EMOTION_SWAP: MELD_EMOTION_CONFIG,
    DatasetType.VOXPOPULI_SWAP: VOXPOPULI_CONFIG,

}

def get_dataset_config(dataset_type: DatasetType) -> DatasetConfig:
    """Get the configuration for a specific dataset type."""
    return DATASET_CONFIGS.get(dataset_type)

def get_swap_config(dataset_type: DatasetType, randomize: bool = False):
    """Get a random swap configuration for a dataset type"""
    if dataset_type == DatasetType.VOXCELEB_SWAP:
        return get_voxceleb_swap_config(randomize)
    elif dataset_type == DatasetType.HVB_SWAP:
        return get_hvb_swap_config(randomize)
    elif dataset_type == DatasetType.VOXPOPULI_SWAP:
        return get_voxpopuli_swap_config(randomize)
    elif dataset_type == DatasetType.MELD_EMOTION_SWAP:
        return get_meld_emotion_swap_config(randomize)
    else:
        raise ValueError(f"No swap config available for dataset type: {dataset_type}")

def apply_label_mapping(examples: List[Dict], label_mapping: Dict[str, str]) -> List[Dict]:
    """Apply label mapping to a list of examples"""
    for example in examples:
        if "sentiment" in example:
            if example["sentiment"] in label_mapping:
                example["sentiment"] = label_mapping[example["sentiment"]]
        elif "sentiment_label" in example:
            if example["sentiment_label"] in label_mapping:
                example["sentiment_label"] = label_mapping[example["sentiment_label"]]
        elif "emotion_label" in example:
            if example["emotion_label"] in label_mapping:
                example["emotion_label"] = label_mapping[example["emotion_label"]] 
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
        elif "normalized_combined_ner" in example:
            if example["normalized_combined_ner"] in label_mapping:
                example["normalized_combined_ner"] = label_mapping[example["normalized_combined_ner"]]
    return examples

__all__ = [
    'DatasetType',
    'DatasetSplit',
    'get_dataset_config',
    'get_swap_config',
    'apply_label_mapping'
] 