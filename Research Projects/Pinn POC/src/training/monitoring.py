"""
Training monitoring and metrics collection system for AV-PINO.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
import time
from collections import defaultdict, deque
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

logger = logging.getLogger(__name__)


@dataclass
class PhysicsConsistencyMetrics:
    """Physics consistency monitoring metrics."""
    maxwell_residual: float
    heat_equation_residual: float
    structural_dynamics_residual: float
    coupling_residual: float
    total_physics_loss: float
    constraint_violations: int
    energy_conservation_error: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingProgressMetrics:
    """Comprehensive training progress metrics."""
    epoch: int
    step: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    
    # Loss components
    data_loss: float
    physics_loss: float
    consistency_loss: float
    variational_loss: float
    
    # Physics consistency
    physics_metrics: PhysicsConsistencyMetrics
    
    # Performance metrics
    epoch_time: float
    step_time: float
    gpu_memory_used: float
    cpu_usage: float
    
    # Gradient metrics
    grad_norm: float
    param_norm: float
    grad_to_param_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['physics_metrics'] = self.physics_metrics.to_dict()
        return result


class MetricsCollector:
    """Collects and aggregates training metrics."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Size of rolling window for metrics averaging
        """
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.metrics_history = []
        self.rolling_metrics = defaultdict(lambda: deque(maxlen=self.window_size))
        self.epoch_metrics = defaultdict(list)
        self.step_count = 0
        
    def add_metrics(self, metrics: TrainingProgressMetrics):
        """Add new metrics."""
        self.metrics_history.append(metrics)
        self.step_count += 1
        
        # Update rolling metrics
        metrics_dict = metrics.to_dict()
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.rolling_metrics[key].append(value)
        
        # Update epoch metrics
        if not self.epoch_metrics[metrics.epoch]:
            self.epoch_metrics[metrics.epoch] = []
        self.epoch_metrics[metrics.epoch].append(metrics)
    
    def get_rolling_average(self, metric_name: str, window: Optional[int] = None) -> float:
        """Get rolling average of a metric."""
        window = window or self.window_size
        values = list(self.rolling_metrics[metric_name])[-window:]
        return np.mean(values) if values else 0.0
    
    def get_rolling_std(self, metric_name: str, window: Optional[int] = None) -> float:
        """Get rolling standard deviation of a metric."""
        window = window or self.window_size
        values = list(self.rolling_metrics[metric_name])[-window:]
        return np.std(values) if len(values) > 1 else 0.0
    
    def get_trend(self, metric_name: str, window: Optional[int] = None) -> str:
        """Get trend direction for a metric."""
        window = window or min(20, self.window_size)
        values = list(self.rolling_metrics[metric_name])[-window:]
        
        if len(values) < 5:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 1e-6:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def get_epoch_summary(self, epoch: int) -> Dict[str, Any]:
        """Get summary statistics for an epoch."""
        if epoch not in self.epoch_metrics:
            return {}
        
        epoch_data = self.epoch_metrics[epoch]
        
        summary = {
            'epoch': epoch,
            'num_steps': len(epoch_data),
            'avg_train_loss': np.mean([m.train_loss for m in epoch_data]),
            'avg_val_loss': epoch_data[-1].val_loss if epoch_data else 0.0,
            'avg_train_accuracy': np.mean([m.train_accuracy for m in epoch_data]),
            'avg_val_accuracy': epoch_data[-1].val_accuracy if epoch_data else 0.0,
            'avg_physics_loss': np.mean([m.physics_loss for m in epoch_data]),
            'total_epoch_time': sum([m.epoch_time for m in epoch_data]),
            'avg_gpu_memory': np.mean([m.gpu_memory_used for m in epoch_data]),
            'final_learning_rate': epoch_data[-1].learning_rate if epoch_data else 0.0
        }
        
        return summary
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect training anomalies."""
        anomalies = []
        
        if len(self.metrics_history) < 15:
            return anomalies
        
        # Use longer baseline excluding recent values for better detection
        baseline_metrics = self.metrics_history[-20:-3] if len(self.metrics_history) >= 20 else self.metrics_history[:-3]
        recent_metrics = self.metrics_history[-3:]
        
        # Check for loss spikes
        baseline_losses = [m.train_loss for m in baseline_metrics]
        if len(baseline_losses) < 5:
            return anomalies
            
        loss_mean = np.mean(baseline_losses)
        loss_std = np.std(baseline_losses)
        
        # Ensure minimum std to avoid division issues
        loss_std = max(loss_std, 0.01)
        
        for i, metrics in enumerate(recent_metrics):
            loss = metrics.train_loss
            if loss > loss_mean + 3 * loss_std:
                anomalies.append({
                    'type': 'loss_spike',
                    'step': metrics.step,
                    'value': loss,
                    'threshold': loss_mean + 3 * loss_std,
                    'severity': 'high' if loss > loss_mean + 5 * loss_std else 'medium'
                })
        
        # Check for gradient explosion
        grad_norms = [m.grad_norm for m in recent_metrics if m.grad_norm > 0]
        if grad_norms:
            grad_mean = np.mean(grad_norms)
            grad_std = np.std(grad_norms)
            
            for i, grad_norm in enumerate(grad_norms[-3:]):
                if grad_norm > grad_mean + 4 * grad_std:
                    anomalies.append({
                        'type': 'gradient_explosion',
                        'step': recent_metrics[-(len(grad_norms)-i)].step,
                        'value': grad_norm,
                        'threshold': grad_mean + 4 * grad_std,
                        'severity': 'high'
                    })
        
        # Check for physics constraint violations
        for metrics in recent_metrics[-3:]:
            if metrics.physics_metrics.constraint_violations > 5:
                anomalies.append({
                    'type': 'physics_violation',
                    'step': metrics.step,
                    'violations': metrics.physics_metrics.constraint_violations,
                    'severity': 'medium'
                })
        
        return anomalies


class PhysicsConsistencyMonitor:
    """Monitor physics consistency during training."""
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize physics consistency monitor.
        
        Args:
            tolerance: Tolerance for constraint violations
        """
        self.tolerance = tolerance
        self.violation_history = defaultdict(list)
        self.constraint_weights = {}
        
    def check_consistency(self, physics_residuals: Dict[str, torch.Tensor]) -> PhysicsConsistencyMetrics:
        """
        Check physics consistency from residuals.
        
        Args:
            physics_residuals: Dictionary of physics constraint residuals
            
        Returns:
            Physics consistency metrics
        """
        # Extract individual residuals
        maxwell_residual = physics_residuals.get('maxwell_residual', torch.tensor(0.0)).item()
        heat_residual = physics_residuals.get('heat_equation_residual', torch.tensor(0.0)).item()
        structural_residual = physics_residuals.get('structural_dynamics_residual', torch.tensor(0.0)).item()
        coupling_residual = physics_residuals.get('coupling_residual', torch.tensor(0.0)).item()
        total_physics_loss = physics_residuals.get('total_physics_loss', torch.tensor(0.0)).item()
        
        # Count constraint violations
        violations = 0
        if maxwell_residual > self.tolerance:
            violations += 1
            self.violation_history['maxwell'].append(maxwell_residual)
        if heat_residual > self.tolerance:
            violations += 1
            self.violation_history['heat'].append(heat_residual)
        if structural_residual > self.tolerance:
            violations += 1
            self.violation_history['structural'].append(structural_residual)
        if coupling_residual > self.tolerance:
            violations += 1
            self.violation_history['coupling'].append(coupling_residual)
        
        # Compute energy conservation error (simplified)
        energy_error = abs(maxwell_residual + heat_residual - structural_residual)
        
        return PhysicsConsistencyMetrics(
            maxwell_residual=maxwell_residual,
            heat_equation_residual=heat_residual,
            structural_dynamics_residual=structural_residual,
            coupling_residual=coupling_residual,
            total_physics_loss=total_physics_loss,
            constraint_violations=violations,
            energy_conservation_error=energy_error
        )
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of constraint violations."""
        summary = {}
        
        for constraint_type, violations in self.violation_history.items():
            if violations:
                summary[constraint_type] = {
                    'count': len(violations),
                    'avg_magnitude': np.mean(violations),
                    'max_magnitude': np.max(violations),
                    'recent_trend': 'increasing' if len(violations) > 5 and 
                                   np.mean(violations[-5:]) > np.mean(violations[:-5]) else 'stable'
                }
        
        return summary


class TrainingVisualizer:
    """Create visualizations for training progress and physics consistency."""
    
    def __init__(self, save_dir: str = "training_plots"):
        """
        Initialize training visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def plot_training_curves(self, metrics_history: List[TrainingProgressMetrics], 
                           save_path: Optional[str] = None) -> str:
        """
        Plot training and validation curves.
        
        Args:
            metrics_history: List of training metrics
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if not metrics_history:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        epochs = [m.epoch for m in metrics_history]
        train_losses = [m.train_loss for m in metrics_history]
        val_losses = [m.val_loss for m in metrics_history]
        train_accs = [m.train_accuracy for m in metrics_history]
        val_accs = [m.val_accuracy for m in metrics_history]
        learning_rates = [m.learning_rate for m in metrics_history]
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', alpha=0.8)
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, train_accs, label='Train Accuracy', alpha=0.8)
        axes[0, 1].plot(epochs, val_accs, label='Val Accuracy', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, learning_rates, color='orange', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # GPU memory usage
        gpu_memory = [m.gpu_memory_used for m in metrics_history]
        axes[1, 1].plot(epochs, gpu_memory, color='red', alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('GPU Memory (MB)')
        axes[1, 1].set_title('GPU Memory Usage')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "training_curves.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_loss_components(self, metrics_history: List[TrainingProgressMetrics],
                           save_path: Optional[str] = None) -> str:
        """
        Plot individual loss components.
        
        Args:
            metrics_history: List of training metrics
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if not metrics_history:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Loss Components Analysis', fontsize=16)
        
        epochs = [m.epoch for m in metrics_history]
        data_losses = [m.data_loss for m in metrics_history]
        physics_losses = [m.physics_loss for m in metrics_history]
        consistency_losses = [m.consistency_loss for m in metrics_history]
        variational_losses = [m.variational_loss for m in metrics_history]
        
        # Individual loss components
        axes[0, 0].plot(epochs, data_losses, label='Data Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Data Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, physics_losses, label='Physics Loss', color='orange', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Physics Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, consistency_losses, label='Consistency Loss', color='green', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Consistency Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, variational_losses, label='Variational Loss', color='red', alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Variational Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "loss_components.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_physics_consistency(self, metrics_history: List[TrainingProgressMetrics],
                               save_path: Optional[str] = None) -> str:
        """
        Plot physics consistency metrics.
        
        Args:
            metrics_history: List of training metrics
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if not metrics_history:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Physics Consistency Monitoring', fontsize=16)
        
        epochs = [m.epoch for m in metrics_history]
        maxwell_residuals = [m.physics_metrics.maxwell_residual for m in metrics_history]
        heat_residuals = [m.physics_metrics.heat_equation_residual for m in metrics_history]
        structural_residuals = [m.physics_metrics.structural_dynamics_residual for m in metrics_history]
        violations = [m.physics_metrics.constraint_violations for m in metrics_history]
        
        # PDE residuals
        axes[0, 0].plot(epochs, maxwell_residuals, label='Maxwell', alpha=0.8)
        axes[0, 0].plot(epochs, heat_residuals, label='Heat Equation', alpha=0.8)
        axes[0, 0].plot(epochs, structural_residuals, label='Structural', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].set_title('PDE Residuals')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Constraint violations
        axes[0, 1].plot(epochs, violations, color='red', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Violations Count')
        axes[0, 1].set_title('Constraint Violations')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy conservation
        energy_errors = [m.physics_metrics.energy_conservation_error for m in metrics_history]
        axes[1, 0].plot(epochs, energy_errors, color='purple', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Energy Error')
        axes[1, 0].set_title('Energy Conservation Error')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total physics loss
        physics_losses = [m.physics_metrics.total_physics_loss for m in metrics_history]
        axes[1, 1].plot(epochs, physics_losses, color='orange', alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Physics Loss')
        axes[1, 1].set_title('Total Physics Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "physics_consistency.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_gradient_analysis(self, metrics_history: List[TrainingProgressMetrics],
                             save_path: Optional[str] = None) -> str:
        """
        Plot gradient analysis.
        
        Args:
            metrics_history: List of training metrics
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        if not metrics_history:
            return ""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Gradient Analysis', fontsize=16)
        
        epochs = [m.epoch for m in metrics_history]
        grad_norms = [m.grad_norm for m in metrics_history]
        param_norms = [m.param_norm for m in metrics_history]
        grad_to_param_ratios = [m.grad_to_param_ratio for m in metrics_history]
        
        # Gradient norms
        axes[0].plot(epochs, grad_norms, alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Gradient Norm')
        axes[0].set_title('Gradient Norms')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Parameter norms
        axes[1].plot(epochs, param_norms, color='orange', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Parameter Norm')
        axes[1].set_title('Parameter Norms')
        axes[1].grid(True, alpha=0.3)
        
        # Gradient to parameter ratio
        axes[2].plot(epochs, grad_to_param_ratios, color='green', alpha=0.8)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Grad/Param Ratio')
        axes[2].set_title('Gradient to Parameter Ratio')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "gradient_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)


class TrainingMonitor:
    """Comprehensive training monitoring system."""
    
    def __init__(self, save_dir: str = "training_monitoring", 
                 physics_tolerance: float = 1e-6,
                 metrics_window: int = 100):
        """
        Initialize training monitor.
        
        Args:
            save_dir: Directory to save monitoring outputs
            physics_tolerance: Tolerance for physics constraint violations
            metrics_window: Window size for rolling metrics
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_collector = MetricsCollector(window_size=metrics_window)
        self.physics_monitor = PhysicsConsistencyMonitor(tolerance=physics_tolerance)
        self.visualizer = TrainingVisualizer(save_dir=str(self.save_dir / "plots"))
        
        self.monitoring_active = True
        self.alert_thresholds = {
            'loss_spike_factor': 3.0,
            'gradient_explosion_threshold': 100.0,
            'physics_violation_limit': 5,
            'memory_usage_limit': 0.9  # 90% of available memory
        }
        
    def update(self, epoch: int, step: int, model: torch.nn.Module,
               loss_components: Dict[str, Any], 
               train_accuracy: float, val_accuracy: float,
               learning_rate: float, epoch_time: float) -> TrainingProgressMetrics:
        """
        Update monitoring with new training step.
        
        Args:
            epoch: Current epoch
            step: Current step
            model: Training model
            loss_components: Dictionary of loss components
            train_accuracy: Training accuracy
            val_accuracy: Validation accuracy
            learning_rate: Current learning rate
            epoch_time: Time taken for epoch
            
        Returns:
            Training progress metrics
        """
        if not self.monitoring_active:
            return None
        
        # Compute gradient and parameter norms
        grad_norm = self._compute_gradient_norm(model)
        param_norm = self._compute_parameter_norm(model)
        grad_to_param_ratio = grad_norm / (param_norm + 1e-8)
        
        # Extract physics residuals
        physics_residuals = {k: v for k, v in loss_components.items() 
                           if 'residual' in k or k == 'total_physics_loss'}
        
        # Check physics consistency
        physics_metrics = self.physics_monitor.check_consistency(physics_residuals)
        
        # Create comprehensive metrics
        metrics = TrainingProgressMetrics(
            epoch=epoch,
            step=step,
            train_loss=loss_components.get('total_loss', 0.0),
            val_loss=loss_components.get('val_loss', 0.0),
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate,
            data_loss=loss_components.get('data_loss', 0.0),
            physics_loss=loss_components.get('physics_loss', 0.0),
            consistency_loss=loss_components.get('consistency_loss', 0.0),
            variational_loss=loss_components.get('variational_loss', 0.0),
            physics_metrics=physics_metrics,
            epoch_time=epoch_time,
            step_time=0.0,  # Would need to be measured separately
            gpu_memory_used=self._get_gpu_memory_usage(),
            cpu_usage=0.0,  # Would need psutil for accurate measurement
            grad_norm=grad_norm,
            param_norm=param_norm,
            grad_to_param_ratio=grad_to_param_ratio
        )
        
        # Add to collector
        self.metrics_collector.add_metrics(metrics)
        
        # Check for anomalies
        anomalies = self.metrics_collector.detect_anomalies()
        if anomalies:
            self._handle_anomalies(anomalies)
        
        return metrics
    
    def generate_report(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive training report.
        
        Args:
            save_plots: Whether to save visualization plots
            
        Returns:
            Training report dictionary
        """
        if not self.metrics_collector.metrics_history:
            return {"error": "No metrics available"}
        
        # Generate visualizations
        plot_paths = {}
        if save_plots:
            plot_paths['training_curves'] = self.visualizer.plot_training_curves(
                self.metrics_collector.metrics_history
            )
            plot_paths['loss_components'] = self.visualizer.plot_loss_components(
                self.metrics_collector.metrics_history
            )
            plot_paths['physics_consistency'] = self.visualizer.plot_physics_consistency(
                self.metrics_collector.metrics_history
            )
            plot_paths['gradient_analysis'] = self.visualizer.plot_gradient_analysis(
                self.metrics_collector.metrics_history
            )
        
        # Compute summary statistics
        latest_metrics = self.metrics_collector.metrics_history[-1]
        
        # Get trends
        trends = {
            'train_loss': self.metrics_collector.get_trend('train_loss'),
            'val_loss': self.metrics_collector.get_trend('val_loss'),
            'physics_loss': self.metrics_collector.get_trend('physics_loss'),
            'grad_norm': self.metrics_collector.get_trend('grad_norm')
        }
        
        # Physics consistency summary
        physics_summary = self.physics_monitor.get_violation_summary()
        
        # Recent anomalies
        anomalies = self.metrics_collector.detect_anomalies()
        
        report = {
            'timestamp': time.time(),
            'total_epochs': latest_metrics.epoch + 1,
            'total_steps': latest_metrics.step,
            'latest_metrics': latest_metrics.to_dict(),
            'trends': trends,
            'physics_consistency': physics_summary,
            'recent_anomalies': anomalies,
            'plot_paths': plot_paths,
            'rolling_averages': {
                'train_loss': self.metrics_collector.get_rolling_average('train_loss'),
                'val_loss': self.metrics_collector.get_rolling_average('val_loss'),
                'physics_loss': self.metrics_collector.get_rolling_average('physics_loss'),
                'grad_norm': self.metrics_collector.get_rolling_average('grad_norm')
            }
        }
        
        # Save report
        report_path = self.save_dir / "training_report.json"
        with open(report_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_report = self._convert_for_json(report)
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")
        
        return report
    
    def _compute_gradient_norm(self, model: torch.nn.Module) -> float:
        """Compute gradient norm."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def _compute_parameter_norm(self, model: torch.nn.Module) -> float:
        """Compute parameter norm."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
    
    def _handle_anomalies(self, anomalies: List[Dict[str, Any]]):
        """Handle detected anomalies."""
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'medium')
            anomaly_type = anomaly.get('type', 'unknown')
            
            if severity == 'high':
                logger.warning(f"High severity anomaly detected: {anomaly_type}")
                logger.warning(f"Details: {anomaly}")
            else:
                logger.info(f"Anomaly detected: {anomaly_type} - {anomaly}")
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        logger.info("Training monitoring stopped")
    
    def resume_monitoring(self):
        """Resume monitoring."""
        self.monitoring_active = True
        logger.info("Training monitoring resumed")