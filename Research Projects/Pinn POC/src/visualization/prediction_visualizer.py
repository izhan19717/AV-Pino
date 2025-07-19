"""
Prediction visualization with confidence intervals and uncertainty displays.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd

@dataclass
class PredictionResult:
    """Container for prediction results with uncertainty."""
    predictions: np.ndarray
    uncertainties: np.ndarray
    confidence_intervals: np.ndarray
    fault_types: List[str]
    timestamps: np.ndarray
    true_labels: Optional[np.ndarray] = None

@dataclass
class UncertaintyMetrics:
    """Container for uncertainty quantification metrics."""
    epistemic_uncertainty: np.ndarray
    aleatoric_uncertainty: np.ndarray
    total_uncertainty: np.ndarray
    confidence_scores: np.ndarray

class PredictionVisualizer:
    """
    Visualization tools for fault predictions with uncertainty quantification.
    
    Provides comprehensive plotting capabilities for prediction results,
    confidence intervals, uncertainty displays, and reliability assessment.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8'):
        """
        Initialize prediction visualizer.
        
        Args:
            figsize: Default figure size for plots
            style: Matplotlib style to use
        """
        self.figsize = figsize
        plt.style.use(style)
        self.fault_colors = {
            'Normal': '#2E8B57',
            'Inner Race': '#DC143C', 
            'Outer Race': '#FF8C00',
            'Ball': '#4169E1',
            'Unknown': '#808080'
        }
        
    def plot_prediction_timeline(self, 
                               results: PredictionResult,
                               title: str = "Fault Prediction Timeline",
                               show_uncertainty: bool = True,
                               show_confidence: bool = True) -> plt.Figure:
        """
        Plot prediction timeline with uncertainty bands.
        
        Args:
            results: PredictionResult object containing predictions and uncertainties
            title: Plot title
            show_uncertainty: Whether to show uncertainty bands
            show_confidence: Whether to show confidence intervals
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])
        fig.suptitle(title, fontsize=16)
        
        # Main prediction plot
        ax1 = axes[0]
        
        # Plot predictions as colored timeline
        for i, (pred, timestamp) in enumerate(zip(results.predictions, results.timestamps)):
            fault_type = results.fault_types[int(pred)] if pred < len(results.fault_types) else 'Unknown'
            color = self.fault_colors.get(fault_type, '#808080')
            
            ax1.scatter(timestamp, pred, c=color, s=50, alpha=0.7, label=fault_type if i == 0 or fault_type not in [results.fault_types[int(p)] for p in results.predictions[:i]] else "")
        
        # Add uncertainty bands if requested
        if show_uncertainty and results.uncertainties is not None:
            lower_bound = results.predictions - results.uncertainties
            upper_bound = results.predictions + results.uncertainties
            ax1.fill_between(results.timestamps, lower_bound, upper_bound, 
                           alpha=0.3, color='gray', label='Uncertainty')
        
        # Add confidence intervals if requested
        if show_confidence and results.confidence_intervals is not None:
            ci_lower = results.confidence_intervals[:, 0]
            ci_upper = results.confidence_intervals[:, 1]
            ax1.fill_between(results.timestamps, ci_lower, ci_upper,
                           alpha=0.2, color='blue', label='95% Confidence')
        
        # Add true labels if available
        if results.true_labels is not None:
            ax1.plot(results.timestamps, results.true_labels, 'k--', 
                    linewidth=2, alpha=0.8, label='True Labels')
        
        ax1.set_ylabel('Fault Type Index')
        ax1.set_title('Prediction Timeline')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Uncertainty evolution plot
        ax2 = axes[1]
        if results.uncertainties is not None:
            ax2.plot(results.timestamps, results.uncertainties, 'r-', 
                    linewidth=1.5, label='Uncertainty')
            ax2.fill_between(results.timestamps, 0, results.uncertainties,
                           alpha=0.3, color='red')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Uncertainty')
        ax2.set_title('Uncertainty Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_uncertainty_decomposition(self,
                                     metrics: UncertaintyMetrics,
                                     timestamps: np.ndarray,
                                     title: str = "Uncertainty Decomposition") -> plt.Figure:
        """
        Plot decomposition of uncertainty into epistemic and aleatoric components.
        
        Args:
            metrics: UncertaintyMetrics object
            timestamps: Time points for the measurements
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Epistemic uncertainty (model uncertainty)
        axes[0, 0].plot(timestamps, metrics.epistemic_uncertainty, 'b-', linewidth=1.5)
        axes[0, 0].fill_between(timestamps, 0, metrics.epistemic_uncertainty, 
                               alpha=0.3, color='blue')
        axes[0, 0].set_title('Epistemic Uncertainty (Model)')
        axes[0, 0].set_ylabel('Uncertainty')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Aleatoric uncertainty (data uncertainty)
        axes[0, 1].plot(timestamps, metrics.aleatoric_uncertainty, 'g-', linewidth=1.5)
        axes[0, 1].fill_between(timestamps, 0, metrics.aleatoric_uncertainty,
                               alpha=0.3, color='green')
        axes[0, 1].set_title('Aleatoric Uncertainty (Data)')
        axes[0, 1].set_ylabel('Uncertainty')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total uncertainty
        axes[1, 0].plot(timestamps, metrics.total_uncertainty, 'r-', linewidth=1.5, label='Total')
        axes[1, 0].plot(timestamps, metrics.epistemic_uncertainty, 'b--', alpha=0.7, label='Epistemic')
        axes[1, 0].plot(timestamps, metrics.aleatoric_uncertainty, 'g--', alpha=0.7, label='Aleatoric')
        axes[1, 0].set_title('Total Uncertainty')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Uncertainty')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence scores
        axes[1, 1].plot(timestamps, metrics.confidence_scores, 'purple', linewidth=1.5)
        axes[1, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='High Confidence')
        axes[1, 1].axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence')
        axes[1, 1].set_title('Confidence Scores')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_prediction_confidence_matrix(self,
                                        predictions: np.ndarray,
                                        confidences: np.ndarray,
                                        fault_types: List[str],
                                        title: str = "Prediction Confidence Matrix") -> plt.Figure:
        """
        Plot confidence matrix showing prediction reliability for each fault type.
        
        Args:
            predictions: Array of predicted fault type indices
            confidences: Array of confidence scores
            fault_types: List of fault type names
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Create confidence matrix
        n_types = len(fault_types)
        confidence_matrix = np.zeros((n_types, 5))  # 5 confidence bins
        confidence_bins = np.linspace(0, 1, 6)
        
        for pred, conf in zip(predictions, confidences):
            if pred < n_types:
                bin_idx = np.digitize(conf, confidence_bins) - 1
                bin_idx = max(0, min(4, bin_idx))  # Clamp to valid range
                confidence_matrix[int(pred), bin_idx] += 1
        
        # Normalize by row
        row_sums = confidence_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        confidence_matrix_norm = confidence_matrix / row_sums
        
        # Plot heatmap
        im = axes[0].imshow(confidence_matrix_norm, cmap='YlOrRd', aspect='auto')
        axes[0].set_xticks(range(5))
        axes[0].set_xticklabels(['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
        axes[0].set_yticks(range(n_types))
        axes[0].set_yticklabels(fault_types)
        axes[0].set_xlabel('Confidence Range')
        axes[0].set_ylabel('Fault Type')
        axes[0].set_title('Confidence Distribution by Fault Type')
        
        # Add text annotations
        for i in range(n_types):
            for j in range(5):
                text = axes[0].text(j, i, f'{confidence_matrix_norm[i, j]:.2f}',
                                  ha="center", va="center", color="black" if confidence_matrix_norm[i, j] < 0.5 else "white")
        
        plt.colorbar(im, ax=axes[0])
        
        # Plot confidence distribution
        axes[1].hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].axvline(np.mean(confidences), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        axes[1].axvline(np.median(confidences), color='green', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(confidences):.3f}')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Overall Confidence Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_reliability_assessment(self,
                                  predictions: np.ndarray,
                                  uncertainties: np.ndarray,
                                  true_labels: Optional[np.ndarray] = None,
                                  title: str = "Reliability Assessment") -> plt.Figure:
        """
        Plot reliability assessment showing relationship between uncertainty and accuracy.
        
        Args:
            predictions: Array of predictions
            uncertainties: Array of uncertainty values
            true_labels: Array of true labels (optional)
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Uncertainty vs prediction scatter
        axes[0, 0].scatter(predictions, uncertainties, alpha=0.6, s=30)
        axes[0, 0].set_xlabel('Predictions')
        axes[0, 0].set_ylabel('Uncertainty')
        axes[0, 0].set_title('Uncertainty vs Predictions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Uncertainty distribution
        axes[0, 1].hist(uncertainties, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].axvline(np.mean(uncertainties), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(uncertainties):.3f}')
        axes[0, 1].set_xlabel('Uncertainty')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Uncertainty Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        if true_labels is not None:
            # Accuracy vs uncertainty
            correct_predictions = (predictions == true_labels).astype(int)
            
            # Bin uncertainties and compute accuracy for each bin
            n_bins = 10
            uncertainty_bins = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
            bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2
            bin_accuracies = []
            
            for i in range(n_bins):
                mask = (uncertainties >= uncertainty_bins[i]) & (uncertainties < uncertainty_bins[i + 1])
                if mask.sum() > 0:
                    bin_accuracies.append(correct_predictions[mask].mean())
                else:
                    bin_accuracies.append(0)
            
            axes[1, 0].plot(bin_centers, bin_accuracies, 'bo-', linewidth=2, markersize=6)
            axes[1, 0].set_xlabel('Uncertainty')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Accuracy vs Uncertainty')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Calibration plot
            # Sort by uncertainty and compute cumulative accuracy
            sorted_indices = np.argsort(uncertainties)
            sorted_uncertainties = uncertainties[sorted_indices]
            sorted_correct = correct_predictions[sorted_indices]
            
            # Compute cumulative accuracy for different uncertainty thresholds
            thresholds = np.percentile(sorted_uncertainties, np.linspace(0, 100, 21))
            cumulative_accuracies = []
            
            for threshold in thresholds:
                mask = uncertainties <= threshold
                if mask.sum() > 0:
                    cumulative_accuracies.append(correct_predictions[mask].mean())
                else:
                    cumulative_accuracies.append(0)
            
            axes[1, 1].plot(thresholds, cumulative_accuracies, 'g-', linewidth=2, marker='o')
            axes[1, 1].set_xlabel('Uncertainty Threshold')
            axes[1, 1].set_ylabel('Cumulative Accuracy')
            axes[1, 1].set_title('Calibration Curve')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # If no true labels, show uncertainty statistics
            axes[1, 0].text(0.5, 0.5, 'True labels not available\nfor accuracy analysis',
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 0].set_title('Accuracy Analysis')
            
            # Show uncertainty percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            uncertainty_percentiles = np.percentile(uncertainties, percentiles)
            
            axes[1, 1].bar(range(len(percentiles)), uncertainty_percentiles, alpha=0.7, color='purple')
            axes[1, 1].set_xticks(range(len(percentiles)))
            axes[1, 1].set_xticklabels([f'{p}%' for p in percentiles])
            axes[1, 1].set_xlabel('Percentile')
            axes[1, 1].set_ylabel('Uncertainty')
            axes[1, 1].set_title('Uncertainty Percentiles')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_fault_type_analysis(self,
                               predictions: np.ndarray,
                               uncertainties: np.ndarray,
                               fault_types: List[str],
                               true_labels: Optional[np.ndarray] = None,
                               title: str = "Fault Type Analysis") -> plt.Figure:
        """
        Plot detailed analysis for each fault type.
        
        Args:
            predictions: Array of predicted fault type indices
            uncertainties: Array of uncertainty values
            fault_types: List of fault type names
            true_labels: Array of true labels (optional)
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        n_types = len(fault_types)
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Prediction distribution
        pred_counts = np.bincount(predictions.astype(int), minlength=n_types)
        axes[0, 0].bar(range(n_types), pred_counts, alpha=0.7, 
                      color=[self.fault_colors.get(ft, '#808080') for ft in fault_types])
        axes[0, 0].set_xticks(range(n_types))
        axes[0, 0].set_xticklabels(fault_types, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Prediction Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Uncertainty by fault type
        uncertainty_by_type = []
        for i in range(n_types):
            mask = predictions == i
            if mask.sum() > 0:
                uncertainty_by_type.append(uncertainties[mask])
            else:
                uncertainty_by_type.append([])
        
        # Box plot of uncertainties
        box_data = [unc for unc in uncertainty_by_type if len(unc) > 0]
        box_labels = [fault_types[i] for i, unc in enumerate(uncertainty_by_type) if len(unc) > 0]
        
        if box_data:
            bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, label in zip(bp['boxes'], box_labels):
                patch.set_facecolor(self.fault_colors.get(label, '#808080'))
                patch.set_alpha(0.7)
        
        axes[0, 1].set_ylabel('Uncertainty')
        axes[0, 1].set_title('Uncertainty by Fault Type')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        if true_labels is not None:
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_labels, predictions, labels=range(n_types))
            
            im = axes[1, 0].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[1, 0].set_title('Confusion Matrix')
            tick_marks = np.arange(n_types)
            axes[1, 0].set_xticks(tick_marks)
            axes[1, 0].set_yticks(tick_marks)
            axes[1, 0].set_xticklabels(fault_types, rotation=45, ha='right')
            axes[1, 0].set_yticklabels(fault_types)
            axes[1, 0].set_ylabel('True Label')
            axes[1, 0].set_xlabel('Predicted Label')
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(n_types):
                for j in range(n_types):
                    axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > thresh else "black")
            
            plt.colorbar(im, ax=axes[1, 0])
            
            # Per-class accuracy
            class_accuracies = cm.diagonal() / cm.sum(axis=1)
            axes[1, 1].bar(range(n_types), class_accuracies, alpha=0.7,
                          color=[self.fault_colors.get(ft, '#808080') for ft in fault_types])
            axes[1, 1].set_xticks(range(n_types))
            axes[1, 1].set_xticklabels(fault_types, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].set_title('Per-Class Accuracy')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'True labels not available\nfor confusion matrix',
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 0].set_title('Confusion Matrix')
            
            axes[1, 1].text(0.5, 0.5, 'True labels not available\nfor accuracy analysis',
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 1].set_title('Per-Class Accuracy')
        
        plt.tight_layout()
        return fig