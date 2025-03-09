import pandas as pd
import numpy as np
import logging
import json
import os
import re
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data.master_config import get_dataset_config, get_swap_config, DatasetType

logger = logging.getLogger(__name__)

def evaluate_predictions(predictions: List[Dict[str, Any]], dataset_type: DatasetType) -> Dict[str, Any]:
    """
    Evaluate model predictions based on dataset type.
    
    Args:
        predictions: List of prediction dictionaries with true_label and predicted_sentiment
        dataset_type: Type of dataset being evaluated
        
    Returns:
        Dictionary of evaluation metrics
    """
    if not predictions:
        logger.warning("Empty predictions list provided for evaluation")
        return {"error": "Empty predictions list", "accuracy": 0.0}
    
    try:
        # Get the appropriate config based on dataset type
        if dataset_type in [DatasetType.VOXCELEB_SWAP, DatasetType.HVB_SWAP, DatasetType.VOXPOPULI_SWAP]:
            config = get_swap_config(dataset_type)
        else:
            config = get_dataset_config(dataset_type)
        
        if not config:
            logger.warning(f"No config found for dataset type: {dataset_type}")
            return {"error": "Invalid dataset type"}
        
        # Extract and clean labels
        true_labels = [p.get("true_label", "") for p in predictions]
        # Clean the predicted labels
        pred_labels = [clean_prediction(p.get("predicted_label", "")) for p in predictions]
        
        # Log some example predictions for debugging
        logger.info("\nExample predictions after cleaning:")
        for i in range(min(5, len(predictions))):
            logger.info(f"Original: {predictions[i].get('predicted_label', '')}")
            logger.info(f"Cleaned: {pred_labels[i]}")
            logger.info(f"True: {true_labels[i]}")
            logger.info("-" * 50)
        
        # Create DataFrame for evaluation
        df = pd.DataFrame({
            "text": [p.get("text", "") for p in predictions],
            "gt": true_labels,
            "pd": pred_labels
        })
        
        # Get valid labels for this dataset type
        valid_labels = [label.lower() for label in config.valid_labels]
        
        # Calculate metrics based on dataset type
        if dataset_type in [DatasetType.VOXCELEB, DatasetType.VOXCELEB_SWAP, DatasetType.VOXCELEB_GREEK]:
            metrics = evaluate_voxceleb(df, valid_labels)
        elif dataset_type in [DatasetType.HVB, DatasetType.HVB_SWAP, DatasetType.HVB_GREEK]:
            metrics = evaluate_hvb(df, valid_labels)
        elif dataset_type in [DatasetType.VOXPOPULI, DatasetType.VOXPOPULI_SWAP, DatasetType.VOXPOPULI_GREEK]:
            metrics = evaluate_voxpopuli(df, valid_labels)
        else:
            logger.warning(f"Unsupported dataset type for evaluation: {dataset_type}")
            return {"accuracy": 0.0}
        
        # Add error analysis
        error_analysis = analyze_errors(true_labels, pred_labels, dataset_type)
        metrics["error_analysis"] = error_analysis
        
        # Log error analysis summary
        logger.info("\nError Analysis Summary:")
        logger.info(f"Total errors: {error_analysis.get('num_errors', 0)}")
        logger.info(f"Error rate: {error_analysis.get('error_rate', 0):.4f}")
        
        if 'common_confusions' in error_analysis:
            logger.info("\nMost common confusions:")
            for confusion, count in error_analysis['common_confusions'].items():
                logger.info(f"  {confusion}: {count}")
        
        if 'common_missing_labels' in error_analysis:
            logger.info("\nMost common missing labels:")
            for label, count in error_analysis['common_missing_labels'].items():
                logger.info(f"  {label}: {count}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error in evaluate_predictions: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"error": str(e), "accuracy": 0.0}

def evaluate_voxceleb(df: pd.DataFrame, valid_classes: List[str]) -> Dict:
    """Evaluate VoxCeleb predictions (single-label classification)"""
    total_samples = len(df)
    
    # Convert to lowercase for consistent comparison
    df['gt'] = df['gt'].str.lower()
    df['pd'] = df['pd'].str.lower()
    
    # Filter for valid ground truth labels
    df = df[df['gt'].isin(valid_classes)]
    after_gt_filter = len(df)
    
    # Calculate metrics with invalid class
    df_with_invalid = df.copy()
    df_with_invalid['pd'] = df_with_invalid['pd'].apply(
        lambda x: x if x in valid_classes else 'invalid'
    )
    
    macro_f1_with_invalid = f1_score(
        df_with_invalid['gt'].values,
        df_with_invalid['pd'].values,
        average="macro",
        labels=valid_classes,
        zero_division=0
    )
    
    # Calculate standard metrics (filtered)
    df_filtered = df[df['pd'].isin(valid_classes)]
    after_pred_filter = len(df_filtered)
    
    if after_pred_filter == 0:
        logger.warning("No valid predictions found for evaluation")
        return {
            'macro_f1_filtered': 0.0,
            'macro_f1_with_invalid': 0.0,
            'invalid_predictions': len(df[~df['pd'].isin(valid_classes)]),
            'total_samples': total_samples,
            'valid_gt_samples': after_gt_filter,
            'valid_samples': 0
        }
    
    matrix_filtered = confusion_matrix(
        df_filtered['gt'].values,
        df_filtered['pd'].values,
        labels=valid_classes
    )
    
    class_accuracy_filtered = matrix_filtered.diagonal()/matrix_filtered.sum(axis=1)
    class_samples_filtered = matrix_filtered.sum(axis=1)
    
    macro_f1_filtered = f1_score(
        df_filtered['gt'].values,
        df_filtered['pd'].values,
        average="macro",
        labels=valid_classes,
        zero_division=0
    )
    
    # Calculate per-class metrics
    class_precision = precision_score(
        df_filtered['gt'].values,
        df_filtered['pd'].values,
        average=None,
        labels=valid_classes,
        zero_division=0
    )
    
    class_recall = recall_score(
        df_filtered['gt'].values,
        df_filtered['pd'].values,
        average=None,
        labels=valid_classes,
        zero_division=0
    )
    
    class_f1 = f1_score(
        df_filtered['gt'].values,
        df_filtered['pd'].values,
        average=None,
        labels=valid_classes,
        zero_division=0
    )
    
    # Calculate overall accuracy
    accuracy = accuracy_score(df_filtered['gt'].values, df_filtered['pd'].values)
    
    return {
        # Standard (filtered) metrics
        'accuracy': accuracy,
        'macro_f1_filtered': macro_f1_filtered,
        'class_accuracy_filtered': class_accuracy_filtered.tolist(),
        'class_precision': class_precision.tolist(),
        'class_recall': class_recall.tolist(),
        'class_f1': class_f1.tolist(),
        'confusion_matrix_filtered': matrix_filtered.tolist(),
        'valid_samples': after_pred_filter,
        
        # Metrics including invalid predictions
        'macro_f1_with_invalid': macro_f1_with_invalid,
        'invalid_predictions': len(df[~df['pd'].isin(valid_classes)]),
        
        # General statistics
        'total_samples': total_samples,
        'valid_gt_samples': after_gt_filter,
        'valid_classes': valid_classes
    }

def evaluate_hvb(df: pd.DataFrame, valid_classes: List[str]) -> Dict:
    """Evaluate HVB predictions (multi-label classification)"""
    total_samples = len(df)
    
    # Convert string labels to lists if needed
    df['gt'] = df['gt'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
    df['pd'] = df['pd'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
    
    # Convert to lowercase for consistent comparison
    df['gt'] = df['gt'].apply(lambda labels: [label.lower() for label in labels])
    df['pd'] = df['pd'].apply(lambda labels: [label.lower() for label in labels])
    
    # Filter for samples with at least one valid ground truth label
    df = df[df['gt'].apply(lambda labels: any(label in valid_classes for label in labels))]
    after_gt_filter = len(df)
    
    # Count samples with no valid predictions
    invalid_samples = sum(1 for pred_labels in df['pd'] 
                         if not any(label in valid_classes for label in pred_labels))
    
    # Convert to binary matrix format with invalid handling
    def to_binary_vector(labels: List[str], valid_classes: List[str]) -> np.ndarray:
        # Check if any valid label exists
        has_valid_label = any(label in valid_classes for label in labels)
        if not has_valid_label:
            # Return zero vector for invalid predictions
            return np.zeros(len(valid_classes))
        return np.array([1 if label in labels else 0 for label in valid_classes])
    
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
    
    # Calculate exact match accuracy
    exact_match = sum(np.array_equal(t, p) for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
    
    return {
        'exact_match': exact_match,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'class_precision': class_precision.tolist(),
        'class_recall': class_recall.tolist(),
        'class_f1': class_f1.tolist(),
        'support': support.tolist(),
        'total_samples': total_samples,
        'valid_gt_samples': after_gt_filter,
        'invalid_samples': invalid_samples,
        'valid_classes': valid_classes
    }

def evaluate_voxpopuli(df: pd.DataFrame, valid_classes: List[str]) -> Dict:
    """Evaluate VoxPopuli entity type predictions"""
    total_samples = len(df)
    
    # Convert predictions and ground truth to lowercase
    df['gt'] = df['gt'].str.lower()
    df['pd'] = df['pd'].str.lower()
    
    # Filter for valid ground truth labels (including 'none')
    all_valid_classes = valid_classes + ['none']
    df = df[df['gt'].isin(all_valid_classes)]
    after_gt_filter = len(df)
    
    # Calculate metrics
    macro_f1 = f1_score(
        df['gt'].values,
        df['pd'].values,
        average="macro",
        labels=all_valid_classes,
        zero_division=0
    )
    
    micro_f1 = f1_score(
        df['gt'].values,
        df['pd'].values,
        average="micro",
        labels=all_valid_classes,
        zero_division=0
    )
    
    weighted_f1 = f1_score(
        df['gt'].values,
        df['pd'].values,
        average="weighted",
        labels=all_valid_classes,
        zero_division=0
    )
    
    # Calculate per-class metrics
    class_precision = precision_score(
        df['gt'].values,
        df['pd'].values,
        average=None,
        labels=all_valid_classes,
        zero_division=0
    )
    
    class_recall = recall_score(
        df['gt'].values,
        df['pd'].values,
        average=None,
        labels=all_valid_classes,
        zero_division=0
    )
    
    class_f1 = f1_score(
        df['gt'].values,
        df['pd'].values,
        average=None,
        labels=all_valid_classes,
        zero_division=0
    )
    
    # Calculate support for each class
    class_support = pd.Series(df['gt'].values).value_counts()
    support_list = [class_support.get(label, 0) for label in all_valid_classes]
    
    # Calculate confusion matrix
    confusion_mat = confusion_matrix(
        df['gt'].values,
        df['pd'].values,
        labels=all_valid_classes
    )
    
    # Calculate accuracy
    accuracy = accuracy_score(df['gt'].values, df['pd'].values)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'class_precision': class_precision.tolist(),
        'class_recall': class_recall.tolist(),
        'class_f1': class_f1.tolist(),
        'support': support_list,
        'confusion_matrix': confusion_mat.tolist(),
        'total_samples': total_samples,
        'valid_gt_samples': after_gt_filter,
        'valid_classes': all_valid_classes
    }

def clean_prediction(prediction: str) -> str:
    """Clean prediction by removing escape characters and dataset-specific artifacts"""
    if prediction is None:
        return ""
    
    # Convert to lowercase
    prediction = prediction.lower().strip()
    
    # Remove common prefixes
    prefixes = ["the answer is", "the label is", "output:", "label:", "the correct answer is", "answer:"]
    for prefix in prefixes:
        if prediction.startswith(prefix):
            prediction = prediction[len(prefix):].strip()
    
    # Remove punctuation at the beginning and end
    prediction = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', prediction).strip()
    
    # Remove newlines and take first line
    if '\n' in prediction:
        prediction = prediction.split('\n')[0]
    
    return prediction.strip()

def analyze_errors(true_labels: List[Any], pred_labels: List[Any], dataset_type: DatasetType) -> Dict[str, Any]:
    """
    Analyze prediction errors to identify patterns.
    
    Args:
        true_labels: List of true labels
        pred_labels: List of predicted labels
        dataset_type: Type of dataset
        
    Returns:
        Dictionary with error analysis results
    """
    try:
        # For multi-label datasets, convert to sets for comparison
        if dataset_type in [DatasetType.HVB, DatasetType.HVB_SWAP, DatasetType.HVB_GREEK]:
            errors = []
            for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
                true_set = set(true) if isinstance(true, list) else set([true])
                pred_set = set(pred) if isinstance(pred, list) else set([pred])
                
                if true_set != pred_set:
                    errors.append({
                        "index": i,
                        "true": true,
                        "pred": pred,
                        "missing": list(true_set - pred_set),
                        "extra": list(pred_set - true_set)
                    })
            
            # Analyze common missing and extra labels
            missing_counts = Counter()
            extra_counts = Counter()
            
            for error in errors:
                for label in error["missing"]:
                    missing_counts[label] += 1
                for label in error["extra"]:
                    extra_counts[label] += 1
            
            return {
                "num_errors": len(errors),
                "error_rate": len(errors) / len(true_labels),
                "common_missing_labels": dict(missing_counts.most_common(5)),
                "common_extra_labels": dict(extra_counts.most_common(5)),
                "example_errors": errors[:5] if errors else []  # Include first 5 errors as examples
            }
        else:
            # For single-label classification
            errors = []
            for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
                if true != pred:
                    errors.append({
                        "index": i,
                        "true": true,
                        "pred": pred
                    })
            
            # Count confusion pairs
            confusion_pairs = Counter()
            for error in errors:
                confusion_pairs[(error["true"], error["pred"])] += 1
            
            # Get most common confusion pairs
            common_confusions = {}
            for (true, pred), count in confusion_pairs.most_common(5):
                common_confusions[f"{true} → {pred}"] = count
            
            return {
                "num_errors": len(errors),
                "error_rate": len(errors) / len(true_labels),
                "common_confusions": common_confusions,
                "example_errors": errors[:5] if errors else []  # Include first 5 errors as examples
            }
    
    except Exception as e:
        logger.error(f"Error in analyze_errors: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            "error": str(e),
            "num_errors": 0,
            "error_rate": 0.0
        }

def save_evaluation_results(metrics: Dict, output_dir: str, filename: str) -> None:
    """
    Save evaluation metrics to a file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save results
        filename: Name of the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Convert numpy values to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    metrics_converted = convert_numpy(metrics)
    
    with open(output_path, 'w') as f:
        json.dump(metrics_converted, f, indent=2)
    
    logger.info(f"Saved evaluation results to {output_path}")
