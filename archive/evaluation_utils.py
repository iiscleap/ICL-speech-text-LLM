import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from typing import List, Dict
import logging
from dataset_config import DatasetType, DATASET_CONFIGS
# from utils.generate_fewshots import convert_ner_to_dict
import ast

def evaluate_voxceleb(df: pd.DataFrame, valid_classes: List[str], output_file: str) -> Dict:
    """Evaluate VoxCeleb predictions (single-label classification)"""
    total_samples = len(df)
    
    # Filter for valid ground truth labels
    df = df[df['gt'].str.lower().isin(valid_classes)]
    after_gt_filter = len(df)
    
    # Calculate metrics with invalid class
    df_with_invalid = df.copy()
    df_with_invalid['pd'] = df_with_invalid['pd'].str.lower().apply(
        lambda x: x if x in valid_classes else 'invalid'
    )
    
    macro_f1_with_invalid = f1_score(
        df_with_invalid['gt'].str.lower().values,
        df_with_invalid['pd'].values,
        average="macro",
        labels=valid_classes
    )
    
    # Calculate standard metrics (filtered)
    df_filtered = df[df['pd'].str.lower().isin(valid_classes)]
    after_pred_filter = len(df_filtered)
    
    matrix_filtered = confusion_matrix(
        df_filtered['gt'].str.lower().values,
        df_filtered['pd'].str.lower().values,
        labels=valid_classes
    )
    
    class_accuracy_filtered = matrix_filtered.diagonal()/matrix_filtered.sum(axis=1)
    class_samples_filtered = matrix_filtered.sum(axis=1)
    
    macro_f1_filtered = f1_score(
        df_filtered['gt'].str.lower().values,
        df_filtered['pd'].str.lower().values,
        average="macro",
        labels=valid_classes
    )
    
    # Save detailed metrics
    with open(output_file, "w") as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Valid ground truth samples: {after_gt_filter}\n")
        f.write(f"Valid prediction samples: {after_pred_filter}\n")
        f.write(f"Invalid predictions: {len(df[~df['pd'].str.lower().isin(valid_classes)])}\n")
        
        f.write("\nMetrics (with invalid predictions counted):\n")
        f.write(f"Macro F1 Score: {macro_f1_with_invalid:.4f}\n")
        
        f.write("\nMetrics (filtered - standard evaluation):\n")
        f.write(f"Macro F1 Score: {macro_f1_filtered:.4f}\n")
        
        f.write("\nPer-class metrics (filtered):\n")
        for i, class_name in enumerate(valid_classes):
            f.write(f"{class_name}:\n")
            f.write(f"  Accuracy: {class_accuracy_filtered[i]:.4f}\n")
            f.write(f"  Samples: {class_samples_filtered[i]}\n")
        
        f.write("\nConfusion Matrix (filtered):\n")
        f.write(f"Classes: {valid_classes}\n")
        f.write(str(matrix_filtered))
    
    return {
        # Standard (filtered) metrics
        'macro_f1_filtered': macro_f1_filtered,
        'class_accuracy_filtered': class_accuracy_filtered,
        'confusion_matrix_filtered': matrix_filtered,
        'valid_samples': after_pred_filter,
        
        # Metrics including invalid predictions
        'macro_f1_with_invalid': macro_f1_with_invalid,
        'invalid_predictions': len(df[~df['pd'].str.lower().isin(valid_classes)]),
        
        # General statistics
        'total_samples': total_samples,
        'valid_gt_samples': after_gt_filter
    }

def evaluate_hvb(df: pd.DataFrame, valid_classes: List[str], output_file: str) -> Dict:
    """Evaluate HVB predictions (multi-label classification)"""
    total_samples = len(df)
    
    # Convert string labels to lists if needed
    df['gt'] = df['gt'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
    df['pd'] = df['pd'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
    
    # Count samples with no valid predictions
    invalid_samples = sum(1 for pred_labels in df['pd'] 
                         if not any(label.lower() in valid_classes for label in pred_labels))
    
    # Convert to binary matrix format with invalid handling
    def to_binary_vector(labels: List[str], valid_classes: List[str]) -> np.ndarray:
        # Check if any valid label exists
        has_valid_label = any(label.lower() in valid_classes for label in labels)
        if not has_valid_label:
            # Return zero vector for invalid predictions
            return np.zeros(len(valid_classes))
        return np.array([1 if label.lower() in labels else 0 for label in valid_classes])
    
    y_true = np.array([to_binary_vector(labels, valid_classes) for labels in df['gt']])
    y_pred = np.array([to_binary_vector(labels, valid_classes) for labels in df['pd']])
    
    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate per-class metrics
    class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate support for each class
    support = y_true.sum(axis=0)
    
    # Save detailed metrics
    with open(output_file, "w") as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Samples with no valid predictions: {invalid_samples}\n")
        f.write(f"\nOverall Metrics:\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Micro F1: {micro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")
        
        f.write("\nPer-class metrics:\n")
        for i, class_name in enumerate(valid_classes):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {class_precision[i]:.4f}\n")
            f.write(f"  Recall: {class_recall[i]:.4f}\n")
            f.write(f"  F1 Score: {class_f1[i]:.4f}\n")
            f.write(f"  Support: {support[i]}\n")
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'class_precision': class_precision.tolist(),
        'class_recall': class_recall.tolist(),
        'class_f1': class_f1.tolist(),
        'support': support.tolist(),
        'total_samples': total_samples,
        'invalid_samples': invalid_samples,
        'valid_classes': valid_classes
    }

def evaluate_voxpopuli(df: pd.DataFrame, valid_classes: List[str], output_file: str) -> Dict:
    """Evaluate VoxPopuli entity type predictions"""
    total_samples = len(df)
    
    # Convert predictions and ground truth to lowercase
    df['gt'] = df['gt'].str.lower()
    df['pd'] = df['pd'].str.lower()
    
    # Calculate metrics
    macro_f1 = f1_score(
        df['gt'].values,
        df['pd'].values,
        average="macro",
        labels=[label.lower() for label in valid_classes] + ['none']
    )
    
    micro_f1 = f1_score(
        df['gt'].values,
        df['pd'].values,
        average="micro",
        labels=[label.lower() for label in valid_classes] + ['none']
    )
    
    weighted_f1 = f1_score(
        df['gt'].values,
        df['pd'].values,
        average="weighted",
        labels=[label.lower() for label in valid_classes] + ['none']
    )
    
    # Calculate per-class metrics
    class_precision = precision_score(
        df['gt'].values,
        df['pd'].values,
        average=None,
        labels=[label.lower() for label in valid_classes] + ['none'],
        zero_division=0
    )
    
    class_recall = recall_score(
        df['gt'].values,
        df['pd'].values,
        average=None,
        labels=[label.lower() for label in valid_classes] + ['none'],
        zero_division=0
    )
    
    class_f1 = f1_score(
        df['gt'].values,
        df['pd'].values,
        average=None,
        labels=[label.lower() for label in valid_classes] + ['none'],
        zero_division=0
    )
    
    # Calculate support for each class
    class_support = pd.Series(df['gt'].values).value_counts()
    
    # Save detailed metrics
    with open(output_file, "w") as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"\nOverall Metrics:\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Micro F1: {micro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")
        
        f.write("\nPer-class metrics:\n")
        for i, class_name in enumerate([*valid_classes, 'None']):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {class_precision[i]:.4f}\n")
            f.write(f"  Recall: {class_recall[i]:.4f}\n")
            f.write(f"  F1 Score: {class_f1[i]:.4f}\n")
            f.write(f"  Support: {class_support.get(class_name.lower(), 0)}\n")
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'class_precision': class_precision.tolist(),
        'class_recall': class_recall.tolist(),
        'class_f1': class_f1.tolist(),
        'support': class_support.tolist(),
        'total_samples': total_samples,
        'valid_classes': [*valid_classes, 'None']
    }

def evaluate_results(predictions: List, output_file: str, dataset_type: str) -> Dict:
    """Main evaluation function that handles all dataset types"""
    df = pd.DataFrame(predictions)
    df.columns = ['text', 'gt', 'pd']
    
    dataset_type_enum = DatasetType(dataset_type)
    dataset_config = DATASET_CONFIGS[dataset_type_enum]
    
    # Convert valid labels to lowercase for consistent comparison
    valid_labels = [label.lower() for label in dataset_config.valid_labels]
    
    if dataset_type_enum == DatasetType.HVB:
        return evaluate_hvb(df, valid_labels, output_file)
    elif dataset_type_enum == DatasetType.VOXPOPULI:
        return evaluate_voxpopuli(df, valid_labels, output_file)
    else:
        return evaluate_voxceleb(df, valid_labels, output_file) 