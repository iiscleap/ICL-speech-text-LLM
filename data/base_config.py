from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    name: DatasetType  # Keep as DatasetType
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