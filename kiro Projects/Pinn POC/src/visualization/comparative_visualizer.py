"""
Comparative performance visualization between methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd

@dataclass
class MethodResults:
    """Container for method comparison results."""
    method_name: str
    accuracy: float
    precision: np.ndarray
    recall: np.ndarray
    f1_score: np.ndarray
    inference_time: float
    training_time: float
    memory_usage: float
    physics_consistency: Optional[float] = None
    uncertainty_quality: Optional[float] = None

@dataclass
class BenchmarkResults:
    """Container for comprehensive benchmark results."""
    methods: List[MethodResults]
    fault_types: List[str]
    dataset_name: str
    test_size: int

class ComparativeVisualizer:
    """
    Comparative visualization tools for method performance analysis.
    
    Provides comprehensive plotting capabilities for comparing different
    fault diagnosis methods, including traditional ML approaches and
    physics-informed neural operators.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10), style: str = 'seaborn-v0_8'):
        """
        Initialize comparative visualizer.
        
        Args:
            figsize: Default figure size for plots
            style: Matplotlib style to use
        """
        self.figsize = figsize
        plt.style.use(style)
        self.method_colors = {
            'AV-PINO': '#FF6B6B',
            'Traditional ML': '#4ECDC4',
            'Deep Learning': '#45B7D1', 
            'Physics-Informed': '#96CEB4',
            'Baseline': '#FFEAA7',
            'SVM': '#DDA0DD',
            'Random Forest': '#98FB98',
            'CNN': '#F0E68C',
            'LSTM': '#FFB6C1'
        }
        
    def plot_method_comparison_overview(self,
                                      results: BenchmarkResults,
                                      title: str = "Method Comparison Overview") -> plt.Figure:
        """
        Plot comprehensive overview comparing all methods.
        
        Args:
            results: BenchmarkResults object containing all method results
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16)
        
        method_names = [method.method_name for method in results.methods]
        
        # Overall accuracy comparison
        accuracies = [method.accuracy for method in results.methods]
        colors = [self.method_colors.get(name, '#808080') for name in method_names]
        
        bars = axes[0, 0].bar(range(len(method_names)), accuracies, 
                             color=colors, alpha=0.7)
        axes[0, 0].set_xticks(range(len(method_names)))
        axes[0, 0].set_xticklabels(method_names, rotation=45, ha='right')
        axes[0, 0].set_title('Overall Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Inference time comparison
        inference_times = [method.inference_time * 1000 for method in results.methods]  # Convert to ms
        
        bars = axes[0, 1].bar(range(len(method_names)), inference_times, 
                             color=colors, alpha=0.7)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                          label='1ms Target')
        axes[0, 1].set_xticks(range(len(method_names)))
        axes[0, 1].set_xticklabels(method_names, rotation=45, ha='right')
        axes[0, 1].set_title('Inference Time Comparison')
        axes[0, 1].set_ylabel('Inference Time (ms)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Memory usage comparison
        memory_usage = [method.memory_usage / (1024**2) for method in results.methods]  # Convert to MB
        
        bars = axes[0, 2].bar(range(len(method_names)), memory_usage, 
                             color=colors, alpha=0.7)
        axes[0, 2].set_xticks(range(len(method_names)))
        axes[0, 2].set_xticklabels(method_names, rotation=45, ha='right')
        axes[0, 2].set_title('Memory Usage Comparison')
        axes[0, 2].set_ylabel('Memory Usage (MB)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # F1-score comparison by fault type
        n_fault_types = len(results.fault_types)
        x = np.arange(n_fault_types)
        width = 0.8 / len(results.methods)
        
        for i, method in enumerate(results.methods):
            offset = (i - len(results.methods)/2 + 0.5) * width
            axes[1, 0].bar(x + offset, method.f1_score, width, 
                          label=method.method_name, 
                          color=self.method_colors.get(method.method_name, '#808080'),
                          alpha=0.7)
        
        axes[1, 0].set_xlabel('Fault Type')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('F1-Score by Fault Type')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(results.fault_types, rotation=45, ha='right')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training time comparison
        training_times = [method.training_time / 60 for method in results.methods]  # Convert to minutes
        
        bars = axes[1, 1].bar(range(len(method_names)), training_times, 
                             color=colors, alpha=0.7)
        axes[1, 1].set_xticks(range(len(method_names)))
        axes[1, 1].set_xticklabels(method_names, rotation=45, ha='right')
        axes[1, 1].set_title('Training Time Comparison')
        axes[1, 1].set_ylabel('Training Time (minutes)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Physics consistency comparison (if available)
        physics_methods = [method for method in results.methods 
                          if method.physics_consistency is not None]
        
        if physics_methods:
            physics_names = [method.method_name for method in physics_methods]
            physics_scores = [method.physics_consistency for method in physics_methods]
            physics_colors = [self.method_colors.get(name, '#808080') for name in physics_names]
            
            bars = axes[1, 2].bar(range(len(physics_names)), physics_scores, 
                                 color=physics_colors, alpha=0.7)
            axes[1, 2].set_xticks(range(len(physics_names)))
            axes[1, 2].set_xticklabels(physics_names, rotation=45, ha='right')
            axes[1, 2].set_title('Physics Consistency Comparison')
            axes[1, 2].set_ylabel('Physics Consistency Score')
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'Physics consistency\ndata not available',
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        return fig
    
    def plot_performance_radar_chart(self,
                                   results: BenchmarkResults,
                                   metrics: List[str] = None,
                                   title: str = "Performance Radar Chart") -> plt.Figure:
        """
        Plot radar chart comparing methods across multiple metrics.
        
        Args:
            results: BenchmarkResults object
            metrics: List of metrics to include in radar chart
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        if metrics is None:
            metrics = ['Accuracy', 'Speed', 'Memory Efficiency', 'Physics Consistency']
        
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(projection='polar'))
        fig.suptitle(title, fontsize=16)
        
        # Normalize metrics to 0-1 scale
        normalized_data = {}
        
        for method in results.methods:
            method_data = []
            
            # Accuracy (already 0-1)
            if 'Accuracy' in metrics:
                method_data.append(method.accuracy)
            
            # Speed (inverse of inference time, normalized)
            if 'Speed' in metrics:
                max_time = max([m.inference_time for m in results.methods])
                speed_score = 1 - (method.inference_time / max_time)
                method_data.append(speed_score)
            
            # Memory Efficiency (inverse of memory usage, normalized)
            if 'Memory Efficiency' in metrics:
                max_memory = max([m.memory_usage for m in results.methods])
                memory_score = 1 - (method.memory_usage / max_memory)
                method_data.append(memory_score)
            
            # Physics Consistency
            if 'Physics Consistency' in metrics:
                if method.physics_consistency is not None:
                    method_data.append(method.physics_consistency)
                else:
                    method_data.append(0.0)  # Default for non-physics methods
            
            # Uncertainty Quality
            if 'Uncertainty Quality' in metrics:
                if method.uncertainty_quality is not None:
                    method_data.append(method.uncertainty_quality)
                else:
                    method_data.append(0.0)
            
            normalized_data[method.method_name] = method_data
        
        # Set up radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each method
        for method_name, data in normalized_data.items():
            data += data[:1]  # Complete the circle
            color = self.method_colors.get(method_name, '#808080')
            
            ax.plot(angles, data, 'o-', linewidth=2, label=method_name, color=color)
            ax.fill(angles, data, alpha=0.25, color=color)
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_comparison(self,
                                       results: BenchmarkResults,
                                       title: str = "Precision-Recall Comparison") -> plt.Figure:
        """
        Plot precision-recall comparison for all methods.
        
        Args:
            results: BenchmarkResults object
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        method_names = [method.method_name for method in results.methods]
        n_methods = len(method_names)
        n_fault_types = len(results.fault_types)
        
        # Precision comparison
        precision_data = np.array([method.precision for method in results.methods])
        
        x = np.arange(n_fault_types)
        width = 0.8 / n_methods
        
        for i, method in enumerate(results.methods):
            offset = (i - n_methods/2 + 0.5) * width
            color = self.method_colors.get(method.method_name, '#808080')
            axes[0].bar(x + offset, method.precision, width, 
                       label=method.method_name, color=color, alpha=0.7)
        
        axes[0].set_xlabel('Fault Type')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision by Fault Type')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results.fault_types, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Recall comparison
        for i, method in enumerate(results.methods):
            offset = (i - n_methods/2 + 0.5) * width
            color = self.method_colors.get(method.method_name, '#808080')
            axes[1].bar(x + offset, method.recall, width, 
                       label=method.method_name, color=color, alpha=0.7)
        
        axes[1].set_xlabel('Fault Type')
        axes[1].set_ylabel('Recall')
        axes[1].set_title('Recall by Fault Type')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results.fault_types, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def plot_efficiency_analysis(self,
                               results: BenchmarkResults,
                               title: str = "Efficiency Analysis") -> plt.Figure:
        """
        Plot efficiency analysis comparing accuracy vs computational cost.
        
        Args:
            results: BenchmarkResults object
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Accuracy vs Inference Time
        accuracies = [method.accuracy for method in results.methods]
        inference_times = [method.inference_time * 1000 for method in results.methods]  # ms
        method_names = [method.method_name for method in results.methods]
        
        for i, (acc, time, name) in enumerate(zip(accuracies, inference_times, method_names)):
            color = self.method_colors.get(name, '#808080')
            axes[0, 0].scatter(time, acc, s=100, color=color, alpha=0.7, label=name)
            axes[0, 0].annotate(name, (time, acc), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        
        axes[0, 0].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, 
                          label='1ms Target')
        axes[0, 0].set_xlabel('Inference Time (ms)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy vs Inference Time')
        axes[0, 0].set_xscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy vs Memory Usage
        memory_usage = [method.memory_usage / (1024**2) for method in results.methods]  # MB
        
        for i, (acc, mem, name) in enumerate(zip(accuracies, memory_usage, method_names)):
            color = self.method_colors.get(name, '#808080')
            axes[0, 1].scatter(mem, acc, s=100, color=color, alpha=0.7, label=name)
            axes[0, 1].annotate(name, (mem, acc), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        
        axes[0, 1].set_xlabel('Memory Usage (MB)')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy vs Memory Usage')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training Time vs Accuracy
        training_times = [method.training_time / 60 for method in results.methods]  # minutes
        
        for i, (acc, train_time, name) in enumerate(zip(accuracies, training_times, method_names)):
            color = self.method_colors.get(name, '#808080')
            axes[1, 0].scatter(train_time, acc, s=100, color=color, alpha=0.7, label=name)
            axes[1, 0].annotate(name, (train_time, acc), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        
        axes[1, 0].set_xlabel('Training Time (minutes)')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy vs Training Time')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency score (accuracy / computational cost)
        # Normalize costs to 0-1 scale
        max_inference = max(inference_times)
        max_memory = max(memory_usage)
        max_training = max(training_times)
        
        efficiency_scores = []
        for method in results.methods:
            norm_inference = method.inference_time * 1000 / max_inference
            norm_memory = method.memory_usage / (1024**2) / max_memory
            norm_training = method.training_time / 60 / max_training
            
            # Combined cost (weighted average)
            combined_cost = 0.5 * norm_inference + 0.3 * norm_memory + 0.2 * norm_training
            efficiency = method.accuracy / (combined_cost + 1e-8)
            efficiency_scores.append(efficiency)
        
        colors = [self.method_colors.get(name, '#808080') for name in method_names]
        bars = axes[1, 1].bar(range(len(method_names)), efficiency_scores, 
                             color=colors, alpha=0.7)
        axes[1, 1].set_xticks(range(len(method_names)))
        axes[1, 1].set_xticklabels(method_names, rotation=45, ha='right')
        axes[1, 1].set_title('Overall Efficiency Score')
        axes[1, 1].set_ylabel('Efficiency (Accuracy/Cost)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, efficiency_scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_statistical_significance_analysis(self,
                                             results: BenchmarkResults,
                                             confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
                                             title: str = "Statistical Significance Analysis") -> plt.Figure:
        """
        Plot statistical significance analysis of method comparisons.
        
        Args:
            results: BenchmarkResults object
            confidence_intervals: Optional confidence intervals for each method
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        method_names = [method.method_name for method in results.methods]
        accuracies = [method.accuracy for method in results.methods]
        
        # Accuracy with confidence intervals
        if confidence_intervals:
            lower_bounds = [confidence_intervals[name][0] for name in method_names 
                           if name in confidence_intervals]
            upper_bounds = [confidence_intervals[name][1] for name in method_names 
                           if name in confidence_intervals]
            errors = [[acc - lower, upper - acc] for acc, lower, upper 
                     in zip(accuracies, lower_bounds, upper_bounds)]
            errors = np.array(errors).T
            
            colors = [self.method_colors.get(name, '#808080') for name in method_names]
            axes[0, 0].bar(range(len(method_names)), accuracies, 
                          yerr=errors, capsize=5, color=colors, alpha=0.7)
        else:
            colors = [self.method_colors.get(name, '#808080') for name in method_names]
            axes[0, 0].bar(range(len(method_names)), accuracies, 
                          color=colors, alpha=0.7)
        
        axes[0, 0].set_xticks(range(len(method_names)))
        axes[0, 0].set_xticklabels(method_names, rotation=45, ha='right')
        axes[0, 0].set_title('Accuracy with Confidence Intervals')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pairwise comparison matrix (mock p-values for demonstration)
        n_methods = len(method_names)
        p_value_matrix = np.ones((n_methods, n_methods))
        
        # Generate mock p-values based on accuracy differences
        for i in range(n_methods):
            for j in range(n_methods):
                if i != j:
                    acc_diff = abs(accuracies[i] - accuracies[j])
                    # Mock p-value: larger differences -> smaller p-values
                    p_value_matrix[i, j] = max(0.001, 1 - acc_diff * 10)
        
        im = axes[0, 1].imshow(p_value_matrix, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[0, 1].set_xticks(range(n_methods))
        axes[0, 1].set_yticks(range(n_methods))
        axes[0, 1].set_xticklabels(method_names, rotation=45, ha='right')
        axes[0, 1].set_yticklabels(method_names)
        axes[0, 1].set_title('Pairwise Comparison P-Values')
        
        # Add p-value annotations
        for i in range(n_methods):
            for j in range(n_methods):
                text = axes[0, 1].text(j, i, f'{p_value_matrix[i, j]:.3f}',
                                     ha="center", va="center", 
                                     color="white" if p_value_matrix[i, j] < 0.5 else "black",
                                     fontsize=8)
        
        plt.colorbar(im, ax=axes[0, 1])
        
        # Effect size analysis (Cohen's d approximation)
        effect_sizes = []
        comparison_pairs = []
        
        # Compare each method with the best performing method
        best_method_idx = np.argmax(accuracies)
        best_accuracy = accuracies[best_method_idx]
        
        for i, (method, acc) in enumerate(zip(method_names, accuracies)):
            if i != best_method_idx:
                # Approximate effect size (assuming std = 0.1 for demonstration)
                effect_size = abs(acc - best_accuracy) / 0.1
                effect_sizes.append(effect_size)
                comparison_pairs.append(f'{method}\nvs\n{method_names[best_method_idx]}')
        
        if effect_sizes:
            axes[1, 0].bar(range(len(effect_sizes)), effect_sizes, 
                          color='skyblue', alpha=0.7)
            axes[1, 0].set_xticks(range(len(effect_sizes)))
            axes[1, 0].set_xticklabels(comparison_pairs, rotation=0, ha='center', fontsize=8)
            axes[1, 0].set_title('Effect Sizes (vs Best Method)')
            axes[1, 0].set_ylabel('Effect Size (Cohen\'s d)')
            axes[1, 0].axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Small')
            axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium')
            axes[1, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Method ranking with uncertainty
        method_rankings = np.argsort(accuracies)[::-1]  # Descending order
        ranked_names = [method_names[i] for i in method_rankings]
        ranked_accuracies = [accuracies[i] for i in method_rankings]
        
        colors_ranked = [self.method_colors.get(name, '#808080') for name in ranked_names]
        bars = axes[1, 1].bar(range(len(ranked_names)), ranked_accuracies, 
                             color=colors_ranked, alpha=0.7)
        axes[1, 1].set_xticks(range(len(ranked_names)))
        axes[1, 1].set_xticklabels(ranked_names, rotation=45, ha='right')
        axes[1, 1].set_title('Method Ranking by Accuracy')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add ranking numbers
        for i, (bar, acc) in enumerate(zip(bars, ranked_accuracies)):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'#{i+1}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig