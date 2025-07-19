"""
Classification Validation and Evaluation System.

Implements comprehensive fault classification evaluation metrics,
confusion matrix analysis, and per-fault-type performance assessment.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import json
from pathlib import Path

from .fault_classifier import FaultType, FaultPrediction, ClassificationMetrics, FaultTypeMapper

logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """Complete validation results for fault classification."""
    overall_metrics: ClassificationMetrics
    per_fault_metrics: Dict[FaultType, Dict[str, float]]
    confusion_matrix: np.ndarray
    roc_curves: Dict[FaultType, Tuple[np.ndarray, np.ndarray]]
    pr_curves: Dict[FaultType, Tuple[np.ndarray, np.ndarray]]
    uncertainty_analysis: Dict[str, float]
    reliability_analysis: Dict[str, Any]


class ClassificationValidator:
    """
    Comprehensive fault classification evaluation system.
    
    Provides detailed performance analysis including confusion matrices,
    per-fault-type metrics, uncertainty analysis, and reliability assessment.
    """
    
    def __init__(self, fault_mapper: Optional[FaultTypeMapper] = None):
        self.fault_mapper = fault_mapper or FaultTypeMapper()
        self.validation_history = []
        
    def evaluate_predictions(self, 
                           predictions: List[FaultPrediction],
                           true_labels: List[FaultType],
                           return_detailed: bool = True) -> ValidationResults:
        """
        Comprehensive evaluation of fault classification predictions.
        
        Args:
            predictions: List of FaultPrediction objects
            true_labels: List of true fault types
            return_detailed: Whether to compute detailed analysis
            
        Returns:
            ValidationResults object with comprehensive metrics
        """
        if len(predictions) != len(true_labels):
            raise ValueError("Number of predictions must match number of true labels")
        
        # Extract predicted labels and probabilities
        pred_labels = [pred.fault_type for pred in predictions]
        pred_indices = [self.fault_mapper.type_to_index(label) for label in pred_labels]
        true_indices = [self.fault_mapper.type_to_index(label) for label in true_labels]
        
        # Extract probabilities for multi-class analysis
        n_classes = self.fault_mapper.n_classes
        prob_matrix = np.zeros((len(predictions), n_classes))
        
        for i, pred in enumerate(predictions):
            for fault_type, prob in pred.probabilities.items():
                class_idx = self.fault_mapper.type_to_index(fault_type)
                prob_matrix[i, class_idx] = prob
        
        # Compute basic metrics
        overall_metrics = self._compute_overall_metrics(true_indices, pred_indices, prob_matrix)
        
        # Compute per-fault metrics
        per_fault_metrics = self._compute_per_fault_metrics(true_indices, pred_indices, prob_matrix)
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(true_indices, pred_indices)
        
        if not return_detailed:
            return ValidationResults(
                overall_metrics=overall_metrics,
                per_fault_metrics=per_fault_metrics,
                confusion_matrix=conf_matrix,
                roc_curves={},
                pr_curves={},
                uncertainty_analysis={},
                reliability_analysis={}
            )
        
        # Detailed analysis
        roc_curves = self._compute_roc_curves(true_indices, prob_matrix)
        pr_curves = self._compute_pr_curves(true_indices, prob_matrix)
        uncertainty_analysis = self._analyze_uncertainty(predictions, true_labels)
        reliability_analysis = self._analyze_reliability(predictions, true_labels)
        
        results = ValidationResults(
            overall_metrics=overall_metrics,
            per_fault_metrics=per_fault_metrics,
            confusion_matrix=conf_matrix,
            roc_curves=roc_curves,
            pr_curves=pr_curves,
            uncertainty_analysis=uncertainty_analysis,
            reliability_analysis=reliability_analysis
        )
        
        # Store in history
        self.validation_history.append(results)
        
        return results
    
    def _compute_overall_metrics(self, true_indices: List[int], pred_indices: List[int],
                               prob_matrix: np.ndarray) -> ClassificationMetrics:
        """Compute overall classification metrics."""
        accuracy = accuracy_score(true_indices, pred_indices)
        
        # Precision, recall, F1 for each class
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_indices, pred_indices, average=None, zero_division=0
        )
        
        # Convert to per-fault-type dictionaries
        precision_dict = {}
        recall_dict = {}
        f1_dict = {}
        per_class_accuracy = {}
        
        for i, fault_type in enumerate(self.fault_mapper.get_all_types()):
            precision_dict[fault_type] = precision[i] if i < len(precision) else 0.0
            recall_dict[fault_type] = recall[i] if i < len(recall) else 0.0
            f1_dict[fault_type] = f1[i] if i < len(f1) else 0.0
            
            # Per-class accuracy
            class_mask = np.array(true_indices) == i
            if np.sum(class_mask) > 0:
                class_correct = np.sum((np.array(pred_indices)[class_mask] == i))
                per_class_accuracy[fault_type] = class_correct / np.sum(class_mask)
            else:
                per_class_accuracy[fault_type] = 0.0
        
        conf_matrix = confusion_matrix(true_indices, pred_indices)
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            confusion_matrix=conf_matrix,
            per_class_accuracy=per_class_accuracy
        )
    
    def _compute_per_fault_metrics(self, true_indices: List[int], pred_indices: List[int],
                                 prob_matrix: np.ndarray) -> Dict[FaultType, Dict[str, float]]:
        """Compute detailed metrics for each fault type."""
        per_fault_metrics = {}
        
        for i, fault_type in enumerate(self.fault_mapper.get_all_types()):
            # Binary classification metrics for this fault type
            true_binary = (np.array(true_indices) == i).astype(int)
            pred_binary = (np.array(pred_indices) == i).astype(int)
            
            if len(np.unique(true_binary)) > 1:  # Check if we have both classes
                try:
                    auc_score = roc_auc_score(true_binary, prob_matrix[:, i])
                except ValueError:
                    auc_score = 0.0
            else:
                auc_score = 0.0
            
            # Compute metrics
            tp = np.sum((true_binary == 1) & (pred_binary == 1))
            tn = np.sum((true_binary == 0) & (pred_binary == 0))
            fp = np.sum((true_binary == 0) & (pred_binary == 1))
            fn = np.sum((true_binary == 1) & (pred_binary == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_fault_metrics[fault_type] = {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'auc_score': auc_score,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            }
        
        return per_fault_metrics
    
    def _compute_roc_curves(self, true_indices: List[int], 
                          prob_matrix: np.ndarray) -> Dict[FaultType, Tuple[np.ndarray, np.ndarray]]:
        """Compute ROC curves for each fault type."""
        roc_curves = {}
        
        for i, fault_type in enumerate(self.fault_mapper.get_all_types()):
            true_binary = (np.array(true_indices) == i).astype(int)
            
            if len(np.unique(true_binary)) > 1:
                fpr, tpr, _ = roc_curve(true_binary, prob_matrix[:, i])
                roc_curves[fault_type] = (fpr, tpr)
            else:
                # Handle case where we don't have both classes
                roc_curves[fault_type] = (np.array([0, 1]), np.array([0, 1]))
        
        return roc_curves
    
    def _compute_pr_curves(self, true_indices: List[int],
                         prob_matrix: np.ndarray) -> Dict[FaultType, Tuple[np.ndarray, np.ndarray]]:
        """Compute Precision-Recall curves for each fault type."""
        pr_curves = {}
        
        for i, fault_type in enumerate(self.fault_mapper.get_all_types()):
            true_binary = (np.array(true_indices) == i).astype(int)
            
            if len(np.unique(true_binary)) > 1:
                precision, recall, _ = precision_recall_curve(true_binary, prob_matrix[:, i])
                pr_curves[fault_type] = (precision, recall)
            else:
                # Handle case where we don't have both classes
                pr_curves[fault_type] = (np.array([1, 0]), np.array([0, 1]))
        
        return pr_curves
    
    def _analyze_uncertainty(self, predictions: List[FaultPrediction],
                           true_labels: List[FaultType]) -> Dict[str, float]:
        """Analyze uncertainty calibration and reliability."""
        uncertainties = [pred.uncertainty for pred in predictions]
        confidences = [pred.confidence for pred in predictions]
        
        # Check if predictions are correct
        correct_predictions = [
            pred.fault_type == true_label 
            for pred, true_label in zip(predictions, true_labels)
        ]
        
        # Uncertainty statistics
        uncertainty_stats = {
            'mean_uncertainty': np.mean(uncertainties),
            'std_uncertainty': np.std(uncertainties),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences)
        }
        
        # Uncertainty vs accuracy correlation
        if len(uncertainties) > 1:
            # Higher uncertainty should correlate with lower accuracy
            correct_array = np.array(correct_predictions).astype(float)
            uncertainty_array = np.array(uncertainties)
            
            # Correlation between uncertainty and correctness (should be negative)
            uncertainty_correlation = np.corrcoef(uncertainty_array, correct_array)[0, 1]
            uncertainty_stats['uncertainty_accuracy_correlation'] = uncertainty_correlation
            
            # Confidence calibration: high confidence should mean high accuracy
            confidence_array = np.array(confidences)
            confidence_correlation = np.corrcoef(confidence_array, correct_array)[0, 1]
            uncertainty_stats['confidence_accuracy_correlation'] = confidence_correlation
        
        return uncertainty_stats
    
    def _analyze_reliability(self, predictions: List[FaultPrediction],
                           true_labels: List[FaultType]) -> Dict[str, Any]:
        """Analyze prediction reliability for safety-critical decisions."""
        # Count reliable vs unreliable predictions
        reliable_count = 0
        unreliable_count = 0
        reliable_correct = 0
        unreliable_correct = 0
        
        for pred, true_label in zip(predictions, true_labels):
            is_correct = pred.fault_type == true_label
            
            # Simple reliability check based on confidence
            is_reliable = pred.confidence > 0.8 and pred.uncertainty < 0.3
            
            if is_reliable:
                reliable_count += 1
                if is_correct:
                    reliable_correct += 1
            else:
                unreliable_count += 1
                if is_correct:
                    unreliable_correct += 1
        
        total_predictions = len(predictions)
        
        reliability_analysis = {
            'total_predictions': total_predictions,
            'reliable_predictions': reliable_count,
            'unreliable_predictions': unreliable_count,
            'reliable_accuracy': reliable_correct / reliable_count if reliable_count > 0 else 0.0,
            'unreliable_accuracy': unreliable_correct / unreliable_count if unreliable_count > 0 else 0.0,
            'reliability_ratio': reliable_count / total_predictions,
            'safety_threshold_met': reliable_count / total_predictions > 0.7  # 70% should be reliable
        }
        
        return reliability_analysis
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix with fault type labels."""
        fault_labels = [fault_type.value for fault_type in self.fault_mapper.get_all_types()]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=fault_labels, yticklabels=fault_labels, ax=ax)
        
        ax.set_title('Fault Classification Confusion Matrix')
        ax.set_xlabel('Predicted Fault Type')
        ax.set_ylabel('True Fault Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, roc_curves: Dict[FaultType, Tuple[np.ndarray, np.ndarray]],
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curves for all fault types."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for fault_type, (fpr, tpr) in roc_curves.items():
            # Compute AUC
            auc = np.trapz(tpr, fpr)
            ax.plot(fpr, tpr, label=f'{fault_type.value} (AUC = {auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves for Fault Classification')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def generate_classification_report(self, results: ValidationResults) -> str:
        """Generate a comprehensive text report of classification performance."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("FAULT CLASSIFICATION EVALUATION REPORT")
        report_lines.append("=" * 60)
        
        # Overall metrics
        report_lines.append(f"\nOVERALL PERFORMANCE:")
        report_lines.append(f"Accuracy: {results.overall_metrics.accuracy:.4f}")
        
        # Per-fault type metrics
        report_lines.append(f"\nPER-FAULT-TYPE METRICS:")
        for fault_type in self.fault_mapper.get_all_types():
            metrics = results.per_fault_metrics.get(fault_type, {})
            report_lines.append(f"\n{fault_type.value.upper()}:")
            report_lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
            report_lines.append(f"  Recall: {metrics.get('recall', 0):.4f}")
            report_lines.append(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
            report_lines.append(f"  AUC: {metrics.get('auc_score', 0):.4f}")
        
        # Uncertainty analysis
        if results.uncertainty_analysis:
            report_lines.append(f"\nUNCERTAINTY ANALYSIS:")
            ua = results.uncertainty_analysis
            report_lines.append(f"Mean Uncertainty: {ua.get('mean_uncertainty', 0):.4f}")
            report_lines.append(f"Mean Confidence: {ua.get('mean_confidence', 0):.4f}")
            if 'confidence_accuracy_correlation' in ua:
                report_lines.append(f"Confidence-Accuracy Correlation: {ua['confidence_accuracy_correlation']:.4f}")
        
        # Reliability analysis
        if results.reliability_analysis:
            report_lines.append(f"\nRELIABILITY ANALYSIS:")
            ra = results.reliability_analysis
            report_lines.append(f"Reliable Predictions: {ra.get('reliable_predictions', 0)}/{ra.get('total_predictions', 0)}")
            report_lines.append(f"Reliability Ratio: {ra.get('reliability_ratio', 0):.4f}")
            report_lines.append(f"Reliable Accuracy: {ra.get('reliable_accuracy', 0):.4f}")
            report_lines.append(f"Safety Threshold Met: {ra.get('safety_threshold_met', False)}")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)
    
    def save_results(self, results: ValidationResults, output_dir: str):
        """Save validation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_dict = {
            'overall_accuracy': results.overall_metrics.accuracy,
            'per_fault_metrics': {
                fault_type.value: metrics 
                for fault_type, metrics in results.per_fault_metrics.items()
            },
            'uncertainty_analysis': results.uncertainty_analysis,
            'reliability_analysis': results.reliability_analysis
        }
        
        with open(output_path / 'classification_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Save confusion matrix
        np.save(output_path / 'confusion_matrix.npy', results.confusion_matrix)
        
        # Generate and save plots
        self.plot_confusion_matrix(results.confusion_matrix, 
                                 str(output_path / 'confusion_matrix.png'))
        
        if results.roc_curves:
            self.plot_roc_curves(results.roc_curves, 
                               str(output_path / 'roc_curves.png'))
        
        # Save text report
        report = self.generate_classification_report(results)
        with open(output_path / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"Validation results saved to {output_path}")


def evaluate_fault_classification(predictions: List[FaultPrediction],
                                true_labels: List[FaultType],
                                output_dir: Optional[str] = None) -> ValidationResults:
    """
    Convenience function for comprehensive fault classification evaluation.
    
    Args:
        predictions: List of FaultPrediction objects
        true_labels: List of true fault types
        output_dir: Optional directory to save results
        
    Returns:
        ValidationResults object
    """
    validator = ClassificationValidator()
    results = validator.evaluate_predictions(predictions, true_labels)
    
    if output_dir:
        validator.save_results(results, output_dir)
    
    return results