"""
Diagnostic visualization tools for model debugging and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import pandas as pd

@dataclass
class ModelDiagnostics:
    """Container for model diagnostic information."""
    layer_activations: Dict[str, np.ndarray]
    gradient_norms: Dict[str, np.ndarray]
    weight_distributions: Dict[str, np.ndarray]
    loss_components: Dict[str, np.ndarray]
    training_metrics: Dict[str, List[float]]

@dataclass
class PerformanceMetrics:
    """Container for performance analysis metrics."""
    inference_times: np.ndarray
    memory_usage: np.ndarray
    accuracy_over_time: np.ndarray
    loss_over_time: np.ndarray
    learning_rates: np.ndarray
    epochs: np.ndarray

class DiagnosticVisualizer:
    """
    Diagnostic visualization tools for model debugging and analysis.
    
    Provides comprehensive plotting capabilities for analyzing model behavior,
    training dynamics, performance bottlenecks, and debugging issues.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10), style: str = 'seaborn-v0_8'):
        """
        Initialize diagnostic visualizer.
        
        Args:
            figsize: Default figure size for plots
            style: Matplotlib style to use
        """
        self.figsize = figsize
        plt.style.use(style)
        self.layer_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
    def plot_training_diagnostics(self,
                                metrics: PerformanceMetrics,
                                title: str = "Training Diagnostics") -> plt.Figure:
        """
        Plot comprehensive training diagnostics.
        
        Args:
            metrics: PerformanceMetrics object containing training data
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16)
        
        # Loss evolution
        axes[0, 0].plot(metrics.epochs, metrics.loss_over_time, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss Evolution')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Accuracy evolution
        axes[0, 1].plot(metrics.epochs, metrics.accuracy_over_time, 'g-', linewidth=2)
        axes[0, 1].set_title('Training Accuracy Evolution')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # Learning rate schedule
        axes[0, 2].plot(metrics.epochs, metrics.learning_rates, 'r-', linewidth=2)
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_yscale('log')
        
        # Inference time distribution
        axes[1, 0].hist(metrics.inference_times * 1000, bins=30, alpha=0.7, 
                       color='orange', edgecolor='black')
        axes[1, 0].axvline(np.mean(metrics.inference_times) * 1000, color='red', 
                          linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(metrics.inference_times)*1000:.2f}ms')
        axes[1, 0].axvline(1.0, color='green', linestyle='--', linewidth=2, 
                          label='1ms Target')
        axes[1, 0].set_title('Inference Time Distribution')
        axes[1, 0].set_xlabel('Inference Time (ms)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Memory usage over time
        axes[1, 1].plot(range(len(metrics.memory_usage)), metrics.memory_usage / (1024**2), 
                       'purple', linewidth=2)
        axes[1, 1].set_title('Memory Usage Evolution')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Memory Usage (MB)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Training stability analysis
        loss_gradient = np.gradient(metrics.loss_over_time)
        axes[1, 2].plot(metrics.epochs[:len(loss_gradient)], loss_gradient, 'brown', linewidth=2)
        axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('Loss Gradient (Training Stability)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss Gradient')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_architecture_analysis(self,
                                       diagnostics: ModelDiagnostics,
                                       title: str = "Model Architecture Analysis") -> plt.Figure:
        """
        Plot analysis of model architecture and layer behavior.
        
        Args:
            diagnostics: ModelDiagnostics object
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Layer activation distributions
        layer_names = list(diagnostics.layer_activations.keys())
        activation_data = [diagnostics.layer_activations[name].flatten() 
                          for name in layer_names]
        
        bp = axes[0, 0].boxplot(activation_data, labels=layer_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], self.layer_colors[:len(layer_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0, 0].set_title('Layer Activation Distributions')
        axes[0, 0].set_ylabel('Activation Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient norms by layer
        if diagnostics.gradient_norms:
            gradient_names = list(diagnostics.gradient_norms.keys())
            gradient_values = [np.mean(diagnostics.gradient_norms[name]) 
                             for name in gradient_names]
            
            bars = axes[0, 1].bar(range(len(gradient_names)), gradient_values, 
                                 color=self.layer_colors[:len(gradient_names)], alpha=0.7)
            axes[0, 1].set_xticks(range(len(gradient_names)))
            axes[0, 1].set_xticklabels(gradient_names, rotation=45, ha='right')
            axes[0, 1].set_title('Gradient Norms by Layer')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, gradient_values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.2e}', ha='center', va='bottom', fontsize=8)
        
        # Weight distribution analysis
        if diagnostics.weight_distributions:
            weight_names = list(diagnostics.weight_distributions.keys())
            
            for i, name in enumerate(weight_names[:4]):  # Limit to 4 layers
                weights = diagnostics.weight_distributions[name].flatten()
                axes[1, 0].hist(weights, bins=30, alpha=0.6, 
                               color=self.layer_colors[i], label=name, density=True)
            
            axes[1, 0].set_title('Weight Distributions')
            axes[1, 0].set_xlabel('Weight Value')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss component breakdown
        if diagnostics.loss_components:
            loss_names = list(diagnostics.loss_components.keys())
            loss_values = [np.mean(diagnostics.loss_components[name]) 
                          for name in loss_names]
            
            # Pie chart for loss components
            axes[1, 1].pie(loss_values, labels=loss_names, autopct='%1.1f%%', 
                          colors=self.layer_colors[:len(loss_names)])
            axes[1, 1].set_title('Loss Component Breakdown')
        
        plt.tight_layout()
        return fig
    
    def plot_gradient_flow_analysis(self,
                                  diagnostics: ModelDiagnostics,
                                  title: str = "Gradient Flow Analysis") -> plt.Figure:
        """
        Plot gradient flow analysis for debugging training issues.
        
        Args:
            diagnostics: ModelDiagnostics object
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        if not diagnostics.gradient_norms:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No gradient data available',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            return fig
        
        layer_names = list(diagnostics.gradient_norms.keys())
        
        # Gradient magnitude by layer
        gradient_mags = [np.mean(diagnostics.gradient_norms[name]) for name in layer_names]
        
        axes[0, 0].plot(range(len(layer_names)), gradient_mags, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xticks(range(len(layer_names)))
        axes[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[0, 0].set_title('Gradient Magnitude by Layer')
        axes[0, 0].set_ylabel('Gradient Magnitude')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient variance by layer
        gradient_vars = [np.var(diagnostics.gradient_norms[name]) for name in layer_names]
        
        axes[0, 1].bar(range(len(layer_names)), gradient_vars, 
                      color='orange', alpha=0.7)
        axes[0, 1].set_xticks(range(len(layer_names)))
        axes[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[0, 1].set_title('Gradient Variance by Layer')
        axes[0, 1].set_ylabel('Gradient Variance')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient ratio analysis (consecutive layers)
        if len(layer_names) > 1:
            gradient_ratios = []
            ratio_labels = []
            
            for i in range(len(layer_names) - 1):
                ratio = gradient_mags[i+1] / (gradient_mags[i] + 1e-8)
                gradient_ratios.append(ratio)
                ratio_labels.append(f'{layer_names[i]}\nto\n{layer_names[i+1]}')
            
            axes[1, 0].bar(range(len(gradient_ratios)), gradient_ratios, 
                          color='green', alpha=0.7)
            axes[1, 0].set_xticks(range(len(gradient_ratios)))
            axes[1, 0].set_xticklabels(ratio_labels, rotation=0, ha='center', fontsize=8)
            axes[1, 0].set_title('Gradient Ratios (Layer to Layer)')
            axes[1, 0].set_ylabel('Gradient Ratio')
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient evolution over time (if available)
        if len(list(diagnostics.gradient_norms.values())[0]) > 1:
            for i, name in enumerate(layer_names[:5]):  # Limit to 5 layers
                gradients = diagnostics.gradient_norms[name]
                axes[1, 1].plot(range(len(gradients)), gradients, 
                               color=self.layer_colors[i], linewidth=1.5, 
                               label=name, alpha=0.8)
            
            axes[1, 1].set_title('Gradient Evolution Over Time')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_activation_analysis(self,
                               diagnostics: ModelDiagnostics,
                               title: str = "Activation Analysis") -> plt.Figure:
        """
        Plot detailed activation analysis for debugging dead neurons and saturation.
        
        Args:
            diagnostics: ModelDiagnostics object
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16)
        
        if not diagnostics.layer_activations:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No activation data available',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            return fig
        
        layer_names = list(diagnostics.layer_activations.keys())
        
        # Activation statistics by layer
        activation_means = []
        activation_stds = []
        dead_neuron_counts = []
        
        for name in layer_names:
            activations = diagnostics.layer_activations[name]
            activation_means.append(np.mean(activations))
            activation_stds.append(np.std(activations))
            
            # Count dead neurons (activations close to zero)
            dead_neurons = np.sum(np.abs(activations) < 1e-6)
            dead_neuron_counts.append(dead_neurons / activations.size * 100)
        
        # Mean activations
        axes[0, 0].bar(range(len(layer_names)), activation_means, 
                      color='blue', alpha=0.7)
        axes[0, 0].set_xticks(range(len(layer_names)))
        axes[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[0, 0].set_title('Mean Activations by Layer')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Activation standard deviations
        axes[0, 1].bar(range(len(layer_names)), activation_stds, 
                      color='green', alpha=0.7)
        axes[0, 1].set_xticks(range(len(layer_names)))
        axes[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[0, 1].set_title('Activation Standard Deviations')
        axes[0, 1].set_ylabel('Std Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dead neuron percentage
        axes[0, 2].bar(range(len(layer_names)), dead_neuron_counts, 
                      color='red', alpha=0.7)
        axes[0, 2].set_xticks(range(len(layer_names)))
        axes[0, 2].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[0, 2].set_title('Dead Neurons (%)')
        axes[0, 2].set_ylabel('Percentage')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Activation histograms for selected layers
        selected_layers = layer_names[:3]  # Show first 3 layers
        
        for i, name in enumerate(selected_layers):
            activations = diagnostics.layer_activations[name].flatten()
            
            axes[1, i].hist(activations, bins=50, alpha=0.7, 
                           color=self.layer_colors[i], density=True)
            axes[1, i].axvline(np.mean(activations), color='red', linestyle='--',
                              linewidth=2, label=f'Mean: {np.mean(activations):.3f}')
            axes[1, i].axvline(0, color='black', linestyle='-', alpha=0.5)
            axes[1, i].set_title(f'{name} Activation Distribution')
            axes[1, i].set_xlabel('Activation Value')
            axes[1, i].set_ylabel('Density')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        # Fill remaining subplots if fewer than 3 layers
        for i in range(len(selected_layers), 3):
            axes[1, i].text(0.5, 0.5, 'No additional\nlayer data',
                           ha='center', va='center', transform=axes[1, i].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        return fig
    
    def plot_performance_bottleneck_analysis(self,
                                           metrics: PerformanceMetrics,
                                           layer_times: Optional[Dict[str, np.ndarray]] = None,
                                           title: str = "Performance Bottleneck Analysis") -> plt.Figure:
        """
        Plot performance bottleneck analysis for optimization.
        
        Args:
            metrics: PerformanceMetrics object
            layer_times: Optional dictionary of per-layer execution times
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Inference time vs accuracy trade-off
        if len(metrics.inference_times) == len(metrics.accuracy_over_time):
            scatter = axes[0, 0].scatter(metrics.inference_times * 1000, 
                                       metrics.accuracy_over_time,
                                       c=range(len(metrics.inference_times)), 
                                       cmap='viridis', alpha=0.6)
            axes[0, 0].axvline(1.0, color='red', linestyle='--', alpha=0.7, 
                              label='1ms Target')
            axes[0, 0].set_xlabel('Inference Time (ms)')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Inference Time vs Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 0], label='Training Progress')
        
        # Memory usage efficiency
        memory_mb = metrics.memory_usage / (1024**2)
        axes[0, 1].plot(range(len(memory_mb)), memory_mb, 'purple', linewidth=2)
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Per-layer timing analysis (if available)
        if layer_times:
            layer_names = list(layer_times.keys())
            avg_times = [np.mean(times) * 1000 for times in layer_times.values()]  # Convert to ms
            
            bars = axes[1, 0].bar(range(len(layer_names)), avg_times, 
                                 color=self.layer_colors[:len(layer_names)], alpha=0.7)
            axes[1, 0].set_xticks(range(len(layer_names)))
            axes[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
            axes[1, 0].set_title('Average Layer Execution Time')
            axes[1, 0].set_ylabel('Time (ms)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add percentage labels
            total_time = sum(avg_times)
            for bar, time in zip(bars, avg_times):
                height = bar.get_height()
                percentage = (time / total_time) * 100
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'Per-layer timing\ndata not available',
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Performance trend analysis
        if len(metrics.inference_times) > 10:
            # Moving average of inference times
            window_size = min(50, len(metrics.inference_times) // 5)
            moving_avg = np.convolve(metrics.inference_times, 
                                   np.ones(window_size)/window_size, mode='valid')
            
            axes[1, 1].plot(range(len(metrics.inference_times)), 
                           metrics.inference_times * 1000, 'lightblue', alpha=0.5, 
                           linewidth=0.5, label='Raw')
            axes[1, 1].plot(range(len(moving_avg)), moving_avg * 1000, 
                           'blue', linewidth=2, label=f'Moving Avg ({window_size})')
            axes[1, 1].axhline(1.0, color='red', linestyle='--', alpha=0.7, 
                              label='1ms Target')
            axes[1, 1].set_title('Inference Time Trend')
            axes[1, 1].set_xlabel('Sample')
            axes[1, 1].set_ylabel('Inference Time (ms)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor trend analysis',
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        return fig