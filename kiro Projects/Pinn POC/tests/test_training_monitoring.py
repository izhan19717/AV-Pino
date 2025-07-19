"""
Unit tests for training monitoring system.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import Mock, patch

from src.training.monitoring import (
    PhysicsConsistencyMetrics, TrainingProgressMetrics,
    MetricsCollector, PhysicsConsistencyMonitor,
    TrainingVisualizer, TrainingMonitor
)


class SimpleTestModel(torch.nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        
    def forward(self, x):
        return self.linear(x)


class TestPhysicsConsistencyMetrics:
    """Test PhysicsConsistencyMetrics dataclass."""
    
    def test_physics_metrics_creation(self):
        """Test creating PhysicsConsistencyMetrics."""
        metrics = PhysicsConsistencyMetrics(
            maxwell_residual=0.01,
            heat_equation_residual=0.02,
            structural_dynamics_residual=0.015,
            coupling_residual=0.005,
            total_physics_loss=0.05,
            constraint_violations=2,
            energy_conservation_error=0.001
        )
        
        assert metrics.maxwell_residual == 0.01
        assert metrics.constraint_violations == 2
        assert metrics.energy_conservation_error == 0.001
    
    def test_physics_metrics_to_dict(self):
        """Test converting physics metrics to dictionary."""
        metrics = PhysicsConsistencyMetrics(
            maxwell_residual=0.01,
            heat_equation_residual=0.02,
            structural_dynamics_residual=0.015,
            coupling_residual=0.005,
            total_physics_loss=0.05,
            constraint_violations=2,
            energy_conservation_error=0.001
        )
        
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['maxwell_residual'] == 0.01
        assert metrics_dict['constraint_violations'] == 2


class TestTrainingProgressMetrics:
    """Test TrainingProgressMetrics dataclass."""
    
    def test_training_progress_metrics_creation(self):
        """Test creating TrainingProgressMetrics."""
        physics_metrics = PhysicsConsistencyMetrics(
            maxwell_residual=0.01, heat_equation_residual=0.02,
            structural_dynamics_residual=0.015, coupling_residual=0.005,
            total_physics_loss=0.05, constraint_violations=2,
            energy_conservation_error=0.001
        )
        
        metrics = TrainingProgressMetrics(
            epoch=5, step=100, train_loss=0.5, val_loss=0.6,
            train_accuracy=0.8, val_accuracy=0.75, learning_rate=1e-3,
            data_loss=0.4, physics_loss=0.1, consistency_loss=0.05,
            variational_loss=0.02, physics_metrics=physics_metrics,
            epoch_time=120.0, step_time=1.2, gpu_memory_used=512.0,
            cpu_usage=50.0, grad_norm=0.1, param_norm=10.0,
            grad_to_param_ratio=0.01
        )
        
        assert metrics.epoch == 5
        assert metrics.train_loss == 0.5
        assert metrics.physics_metrics.constraint_violations == 2
    
    def test_training_progress_metrics_to_dict(self):
        """Test converting training progress metrics to dictionary."""
        physics_metrics = PhysicsConsistencyMetrics(
            maxwell_residual=0.01, heat_equation_residual=0.02,
            structural_dynamics_residual=0.015, coupling_residual=0.005,
            total_physics_loss=0.05, constraint_violations=2,
            energy_conservation_error=0.001
        )
        
        metrics = TrainingProgressMetrics(
            epoch=5, step=100, train_loss=0.5, val_loss=0.6,
            train_accuracy=0.8, val_accuracy=0.75, learning_rate=1e-3,
            data_loss=0.4, physics_loss=0.1, consistency_loss=0.05,
            variational_loss=0.02, physics_metrics=physics_metrics,
            epoch_time=120.0, step_time=1.2, gpu_memory_used=512.0,
            cpu_usage=50.0, grad_norm=0.1, param_norm=10.0,
            grad_to_param_ratio=0.01
        )
        
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['epoch'] == 5
        assert 'physics_metrics' in metrics_dict
        assert isinstance(metrics_dict['physics_metrics'], dict)


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.collector = MetricsCollector(window_size=10)
        
    def create_sample_metrics(self, epoch: int, train_loss: float) -> TrainingProgressMetrics:
        """Create sample metrics for testing."""
        physics_metrics = PhysicsConsistencyMetrics(
            maxwell_residual=0.01, heat_equation_residual=0.02,
            structural_dynamics_residual=0.015, coupling_residual=0.005,
            total_physics_loss=0.05, constraint_violations=0,
            energy_conservation_error=0.001
        )
        
        return TrainingProgressMetrics(
            epoch=epoch, step=epoch*10, train_loss=train_loss, val_loss=train_loss*1.1,
            train_accuracy=0.8, val_accuracy=0.75, learning_rate=1e-3,
            data_loss=train_loss*0.8, physics_loss=train_loss*0.2, 
            consistency_loss=0.05, variational_loss=0.02,
            physics_metrics=physics_metrics, epoch_time=120.0, step_time=1.2,
            gpu_memory_used=512.0, cpu_usage=50.0, grad_norm=0.1,
            param_norm=10.0, grad_to_param_ratio=0.01
        )
    
    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        assert self.collector.window_size == 10
        assert len(self.collector.metrics_history) == 0
        assert self.collector.step_count == 0
    
    def test_add_metrics(self):
        """Test adding metrics to collector."""
        metrics = self.create_sample_metrics(epoch=0, train_loss=0.5)
        self.collector.add_metrics(metrics)
        
        assert len(self.collector.metrics_history) == 1
        assert self.collector.step_count == 1
        assert self.collector.metrics_history[0] == metrics
    
    def test_rolling_average(self):
        """Test rolling average calculation."""
        # Add multiple metrics
        for i in range(5):
            metrics = self.create_sample_metrics(epoch=i, train_loss=0.5 + i*0.1)
            self.collector.add_metrics(metrics)
        
        avg_loss = self.collector.get_rolling_average('train_loss')
        expected_avg = np.mean([0.5, 0.6, 0.7, 0.8, 0.9])
        assert abs(avg_loss - expected_avg) < 1e-6
    
    def test_rolling_std(self):
        """Test rolling standard deviation calculation."""
        # Add multiple metrics
        losses = [0.5, 0.6, 0.7, 0.8, 0.9]
        for i, loss in enumerate(losses):
            metrics = self.create_sample_metrics(epoch=i, train_loss=loss)
            self.collector.add_metrics(metrics)
        
        std_loss = self.collector.get_rolling_std('train_loss')
        expected_std = np.std(losses)
        assert abs(std_loss - expected_std) < 1e-6
    
    def test_trend_detection(self):
        """Test trend detection."""
        # Add increasing losses
        for i in range(10):
            metrics = self.create_sample_metrics(epoch=i, train_loss=0.5 + i*0.1)
            self.collector.add_metrics(metrics)
        
        trend = self.collector.get_trend('train_loss')
        assert trend == "increasing"
        
        # Add decreasing losses
        self.collector.reset()
        for i in range(10):
            metrics = self.create_sample_metrics(epoch=i, train_loss=1.0 - i*0.05)
            self.collector.add_metrics(metrics)
        
        trend = self.collector.get_trend('train_loss')
        assert trend == "decreasing"
    
    def test_epoch_summary(self):
        """Test epoch summary generation."""
        # Add metrics for epoch 0
        for step in range(5):
            metrics = self.create_sample_metrics(epoch=0, train_loss=0.5)
            metrics.step = step
            self.collector.add_metrics(metrics)
        
        summary = self.collector.get_epoch_summary(epoch=0)
        assert summary['epoch'] == 0
        assert summary['num_steps'] == 5
        assert summary['avg_train_loss'] == 0.5
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Add normal metrics with some variation
        base_losses = [0.5, 0.51, 0.49, 0.52, 0.48, 0.50, 0.51, 0.49, 0.50, 0.52]
        for i, loss in enumerate(base_losses):
            metrics = self.create_sample_metrics(epoch=i, train_loss=loss)
            self.collector.add_metrics(metrics)
        
        # Add more normal metrics to establish baseline
        for i in range(10, 20):
            metrics = self.create_sample_metrics(epoch=i, train_loss=0.5)
            self.collector.add_metrics(metrics)
        
        # Add anomalous loss spike - much larger than baseline
        spike_metrics = self.create_sample_metrics(epoch=20, train_loss=3.0)  # 6x larger than baseline
        self.collector.add_metrics(spike_metrics)
        
        anomalies = self.collector.detect_anomalies()
        assert len(anomalies) > 0
        assert any(a['type'] == 'loss_spike' for a in anomalies)


class TestPhysicsConsistencyMonitor:
    """Test PhysicsConsistencyMonitor functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = PhysicsConsistencyMonitor(tolerance=1e-3)
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.tolerance == 1e-3
        assert len(self.monitor.violation_history) == 0
    
    def test_consistency_check(self):
        """Test physics consistency checking."""
        physics_residuals = {
            'maxwell_residual': torch.tensor(0.01),
            'heat_equation_residual': torch.tensor(0.02),
            'structural_dynamics_residual': torch.tensor(0.015),
            'coupling_residual': torch.tensor(0.005),
            'total_physics_loss': torch.tensor(0.05)
        }
        
        metrics = self.monitor.check_consistency(physics_residuals)
        
        assert isinstance(metrics, PhysicsConsistencyMetrics)
        assert abs(metrics.maxwell_residual - 0.01) < 1e-6
        assert abs(metrics.heat_equation_residual - 0.02) < 1e-6
        assert metrics.constraint_violations == 4  # All above tolerance
    
    def test_violation_summary(self):
        """Test violation summary generation."""
        # Add some violations
        physics_residuals = {
            'maxwell_residual': torch.tensor(0.01),
            'heat_equation_residual': torch.tensor(0.02),
        }
        
        # Check consistency multiple times
        for _ in range(5):
            self.monitor.check_consistency(physics_residuals)
        
        summary = self.monitor.get_violation_summary()
        assert 'maxwell' in summary
        assert 'heat' in summary
        assert summary['maxwell']['count'] == 5


class TestTrainingVisualizer:
    """Test TrainingVisualizer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = TrainingVisualizer(save_dir=self.temp_dir)
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_sample_metrics_list(self, n_epochs: int = 5) -> list:
        """Create sample metrics list for testing."""
        metrics_list = []
        
        for epoch in range(n_epochs):
            physics_metrics = PhysicsConsistencyMetrics(
                maxwell_residual=0.01 * (1 + epoch*0.1),
                heat_equation_residual=0.02 * (1 + epoch*0.1),
                structural_dynamics_residual=0.015 * (1 + epoch*0.1),
                coupling_residual=0.005 * (1 + epoch*0.1),
                total_physics_loss=0.05 * (1 + epoch*0.1),
                constraint_violations=epoch % 3,
                energy_conservation_error=0.001 * (1 + epoch*0.1)
            )
            
            metrics = TrainingProgressMetrics(
                epoch=epoch, step=epoch*10, 
                train_loss=1.0 - epoch*0.1, val_loss=1.1 - epoch*0.1,
                train_accuracy=0.5 + epoch*0.1, val_accuracy=0.45 + epoch*0.1,
                learning_rate=1e-3 * (0.9**epoch),
                data_loss=0.8 - epoch*0.08, physics_loss=0.2 - epoch*0.02,
                consistency_loss=0.05, variational_loss=0.02,
                physics_metrics=physics_metrics, epoch_time=120.0, step_time=1.2,
                gpu_memory_used=512.0 + epoch*10, cpu_usage=50.0,
                grad_norm=0.1 + epoch*0.01, param_norm=10.0,
                grad_to_param_ratio=0.01 + epoch*0.001
            )
            metrics_list.append(metrics)
        
        return metrics_list
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        assert Path(self.temp_dir).exists()
        assert self.visualizer.save_dir == Path(self.temp_dir)
    
    def test_plot_training_curves(self):
        """Test plotting training curves."""
        metrics_list = self.create_sample_metrics_list()
        
        plot_path = self.visualizer.plot_training_curves(metrics_list)
        
        assert Path(plot_path).exists()
        assert plot_path.endswith('.png')
    
    def test_plot_loss_components(self):
        """Test plotting loss components."""
        metrics_list = self.create_sample_metrics_list()
        
        plot_path = self.visualizer.plot_loss_components(metrics_list)
        
        assert Path(plot_path).exists()
        assert plot_path.endswith('.png')
    
    def test_plot_physics_consistency(self):
        """Test plotting physics consistency."""
        metrics_list = self.create_sample_metrics_list()
        
        plot_path = self.visualizer.plot_physics_consistency(metrics_list)
        
        assert Path(plot_path).exists()
        assert plot_path.endswith('.png')
    
    def test_plot_gradient_analysis(self):
        """Test plotting gradient analysis."""
        metrics_list = self.create_sample_metrics_list()
        
        plot_path = self.visualizer.plot_gradient_analysis(metrics_list)
        
        assert Path(plot_path).exists()
        assert plot_path.endswith('.png')
    
    def test_empty_metrics_handling(self):
        """Test handling of empty metrics list."""
        plot_path = self.visualizer.plot_training_curves([])
        assert plot_path == ""


class TestTrainingMonitor:
    """Test TrainingMonitor integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = TrainingMonitor(save_dir=self.temp_dir)
        self.model = SimpleTestModel()
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert Path(self.temp_dir).exists()
        assert self.monitor.monitoring_active
        assert isinstance(self.monitor.metrics_collector, MetricsCollector)
        assert isinstance(self.monitor.physics_monitor, PhysicsConsistencyMonitor)
        assert isinstance(self.monitor.visualizer, TrainingVisualizer)
    
    def test_update_monitoring(self):
        """Test updating monitoring with new data."""
        # Create some gradients for the model
        dummy_input = torch.randn(4, 10)
        output = self.model(dummy_input)
        loss = output.sum()
        loss.backward()
        
        loss_components = {
            'total_loss': 0.5,
            'data_loss': 0.4,
            'physics_loss': 0.1,
            'consistency_loss': 0.05,
            'variational_loss': 0.02,
            'maxwell_residual': torch.tensor(0.01),
            'heat_equation_residual': torch.tensor(0.02),
            'total_physics_loss': torch.tensor(0.1)
        }
        
        metrics = self.monitor.update(
            epoch=0, step=1, model=self.model,
            loss_components=loss_components,
            train_accuracy=0.8, val_accuracy=0.75,
            learning_rate=1e-3, epoch_time=120.0
        )
        
        assert isinstance(metrics, TrainingProgressMetrics)
        assert metrics.epoch == 0
        assert metrics.step == 1
        assert metrics.train_loss == 0.5
        assert metrics.grad_norm > 0  # Should have computed gradient norm
    
    def test_generate_report(self):
        """Test generating comprehensive training report."""
        # Add some monitoring data
        for epoch in range(3):
            # Create gradients
            dummy_input = torch.randn(4, 10)
            output = self.model(dummy_input)
            loss = output.sum()
            loss.backward()
            
            loss_components = {
                'total_loss': 1.0 - epoch*0.1,
                'data_loss': 0.8 - epoch*0.08,
                'physics_loss': 0.2 - epoch*0.02,
                'consistency_loss': 0.05,
                'variational_loss': 0.02,
                'maxwell_residual': torch.tensor(0.01),
                'heat_equation_residual': torch.tensor(0.02),
                'total_physics_loss': torch.tensor(0.2 - epoch*0.02)
            }
            
            self.monitor.update(
                epoch=epoch, step=epoch*10, model=self.model,
                loss_components=loss_components,
                train_accuracy=0.5 + epoch*0.1, val_accuracy=0.45 + epoch*0.1,
                learning_rate=1e-3, epoch_time=120.0
            )
        
        # Generate report
        report = self.monitor.generate_report(save_plots=True)
        
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'total_epochs' in report
        assert 'latest_metrics' in report
        assert 'trends' in report
        assert 'plot_paths' in report
        
        # Check that report file was saved
        report_path = Path(self.temp_dir) / "training_report.json"
        assert report_path.exists()
        
        # Check that plots were saved
        plots_dir = Path(self.temp_dir) / "plots"
        assert plots_dir.exists()
        assert len(list(plots_dir.glob("*.png"))) > 0
    
    def test_stop_resume_monitoring(self):
        """Test stopping and resuming monitoring."""
        assert self.monitor.monitoring_active
        
        self.monitor.stop_monitoring()
        assert not self.monitor.monitoring_active
        
        self.monitor.resume_monitoring()
        assert self.monitor.monitoring_active
    
    @patch('src.training.monitoring.torch.cuda.is_available', return_value=False)
    def test_cpu_mode(self, mock_cuda):
        """Test monitoring in CPU mode."""
        gpu_memory = self.monitor._get_gpu_memory_usage()
        assert gpu_memory == 0.0


if __name__ == "__main__":
    pytest.main([__file__])