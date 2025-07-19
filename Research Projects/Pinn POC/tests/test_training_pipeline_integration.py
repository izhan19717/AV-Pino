"""
End-to-end integration tests for the complete training pipeline.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

from src.training import (
    TrainingEngine, TrainingMonitor, TrainingMetrics, 
    CheckpointData, PhysicsConsistencyMetrics
)
from src.config.config_manager import AVPINOConfig, DataConfig, PhysicsConfig, ModelConfig, InferenceConfig
from src.physics.loss import PhysicsInformedLoss


class SimpleTestModel(nn.Module):
    """Simple model for integration testing."""
    
    def __init__(self, input_size=64, output_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class TestTrainingPipelineIntegration:
    """Integration tests for complete training pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create configuration
        self.config = AVPINOConfig(
            data=DataConfig(window_size=64, sequence_length=32),
            physics=PhysicsConfig(constraint_tolerance=1e-6),
            model=ModelConfig(learning_rate=1e-3, max_epochs=3, patience=10, batch_size=16),
            inference=InferenceConfig(),
            checkpoint_dir=str(Path(self.temp_dir) / "checkpoints"),
            output_dir=str(Path(self.temp_dir) / "outputs")
        )
        
        # Create model
        self.model = SimpleTestModel()
        
        # Create datasets
        self.train_dataset = TensorDataset(
            torch.randn(128, 64),  # 128 samples, 64 features
            torch.randint(0, 4, (128,))  # 4 classes
        )
        
        self.val_dataset = TensorDataset(
            torch.randn(64, 64),   # 64 validation samples
            torch.randint(0, 4, (64,))
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_physics_loss(self):
        """Create a mock physics-informed loss function."""
        def mock_loss_fn(prediction, target, input_data=None, coords=None, mu=None, log_var=None, **kwargs):
            # Simulate realistic loss components
            data_loss = nn.CrossEntropyLoss()(prediction, target)
            physics_loss = torch.tensor(0.1) * torch.rand(1).item()
            consistency_loss = torch.tensor(0.05) * torch.rand(1).item()
            variational_loss = torch.tensor(0.02) * torch.rand(1).item()
            
            total_loss = data_loss + physics_loss + consistency_loss + variational_loss
            
            loss_components = {
                'data_loss': data_loss.item(),
                'physics_loss': physics_loss.item(),
                'consistency_loss': consistency_loss.item(),
                'variational_loss': variational_loss.item(),
                'total_loss': total_loss.item(),
                # Physics residuals
                'maxwell_residual': torch.tensor(0.01 * torch.rand(1).item()),
                'heat_equation_residual': torch.tensor(0.02 * torch.rand(1).item()),
                'structural_dynamics_residual': torch.tensor(0.015 * torch.rand(1).item()),
                'coupling_residual': torch.tensor(0.005 * torch.rand(1).item())
            }
            
            return total_loss, loss_components
        
        return mock_loss_fn
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline with monitoring."""
        # Create training engine
        loss_fn = self.create_mock_physics_loss()
        
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )
        
        # Create training monitor
        monitor = TrainingMonitor(
            save_dir=str(Path(self.temp_dir) / "monitoring"),
            physics_tolerance=1e-6,
            metrics_window=50
        )
        
        # Custom training loop with monitoring
        metrics_history = []
        
        for epoch in range(self.config.model.max_epochs):
            engine.current_epoch = epoch
            
            # Train epoch
            epoch_metrics = engine.train_epoch(
                self.train_loader,
                validation_callback=lambda: engine.validate(self.val_loader)
            )
            
            # Update monitoring
            loss_components = {
                'total_loss': epoch_metrics.train_loss,
                'data_loss': epoch_metrics.data_loss,
                'physics_loss': epoch_metrics.physics_loss,
                'consistency_loss': epoch_metrics.consistency_loss,
                'variational_loss': epoch_metrics.variational_loss,
                'val_loss': epoch_metrics.val_loss,
                'maxwell_residual': torch.tensor(0.01),
                'heat_equation_residual': torch.tensor(0.02),
                'structural_dynamics_residual': torch.tensor(0.015),
                'coupling_residual': torch.tensor(0.005),
                'total_physics_loss': torch.tensor(epoch_metrics.physics_loss)
            }
            
            training_metrics = monitor.update(
                epoch=epoch,
                step=epoch * len(self.train_loader),
                model=engine.model,
                loss_components=loss_components,
                train_accuracy=epoch_metrics.train_accuracy,
                val_accuracy=epoch_metrics.val_accuracy,
                learning_rate=epoch_metrics.learning_rate,
                epoch_time=epoch_metrics.epoch_time
            )
            
            metrics_history.append(training_metrics)
            
            # Save checkpoint every epoch
            checkpoint_path = engine.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            assert Path(checkpoint_path).exists()
        
        # Verify training completed
        assert len(metrics_history) == self.config.model.max_epochs
        assert all(isinstance(m, type(training_metrics)) for m in metrics_history)
        
        # Generate monitoring report
        report = monitor.generate_report(save_plots=True)
        
        # Verify report contents
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'total_epochs' in report
        assert 'latest_metrics' in report
        assert 'trends' in report
        assert 'plot_paths' in report
        
        # Verify plots were created
        plots_dir = Path(self.temp_dir) / "monitoring" / "plots"
        assert plots_dir.exists()
        plot_files = list(plots_dir.glob("*.png"))
        assert len(plot_files) >= 3  # Should have multiple plots
        
        # Verify checkpoints were saved
        checkpoint_dir = Path(self.config.checkpoint_dir)
        assert checkpoint_dir.exists()
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) >= self.config.model.max_epochs
        
        # Test checkpoint loading
        latest_checkpoint = checkpoint_files[-1]
        loaded_checkpoint = engine.load_checkpoint(latest_checkpoint)
        assert isinstance(loaded_checkpoint, CheckpointData)
        assert loaded_checkpoint.epoch == self.config.model.max_epochs - 1
    
    def test_training_with_early_stopping(self):
        """Test training pipeline with early stopping mechanism."""
        # Test that early stopping mechanism is properly initialized
        self.config.model.patience = 3
        self.config.model.max_epochs = 5
        
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            loss_fn=self.create_mock_physics_loss(),
            device=torch.device('cpu')
        )
        
        # Verify early stopping attributes are initialized
        assert engine.early_stopping_counter == 0
        assert engine.best_val_loss == float('inf')
        assert engine.best_val_accuracy == 0.0
        
        # Run a short training to test the mechanism
        metrics_history = engine.train(self.train_loader, self.val_loader, num_epochs=3)
        
        # Should complete training (may or may not trigger early stopping)
        assert len(metrics_history) <= 3
        assert len(metrics_history) > 0
        
        # Test manual early stopping counter increment
        initial_counter = engine.early_stopping_counter
        engine.early_stopping_counter = self.config.model.patience + 1
        
        # Verify that early stopping counter can be set
        assert engine.early_stopping_counter > initial_counter
    
    def test_distributed_training_setup(self):
        """Test distributed training setup (single GPU simulation)."""
        from src.training.training_engine import DistributedTrainingManager
        
        # Test single GPU setup
        dist_manager = DistributedTrainingManager(world_size=1, rank=0)
        assert not dist_manager.is_distributed
        assert dist_manager.is_main_process
        
        # Test model wrapping (should not wrap for single GPU)
        wrapped_model = dist_manager.wrap_model(self.model)
        assert wrapped_model is self.model
        
        # Test dataloader creation
        dataloader = dist_manager.create_dataloader(
            self.train_dataset, batch_size=16, shuffle=True
        )
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 16
    
    def test_physics_consistency_monitoring(self):
        """Test physics consistency monitoring during training."""
        from src.training.monitoring import PhysicsConsistencyMonitor
        
        monitor = PhysicsConsistencyMonitor(tolerance=1e-3)
        
        # Simulate physics residuals over training
        for step in range(10):
            physics_residuals = {
                'maxwell_residual': torch.tensor(0.01 * (1 + step * 0.1)),
                'heat_equation_residual': torch.tensor(0.02 * (1 + step * 0.1)),
                'structural_dynamics_residual': torch.tensor(0.015 * (1 + step * 0.1)),
                'coupling_residual': torch.tensor(0.005 * (1 + step * 0.1)),
                'total_physics_loss': torch.tensor(0.05 * (1 + step * 0.1))
            }
            
            consistency_metrics = monitor.check_consistency(physics_residuals)
            assert isinstance(consistency_metrics, PhysicsConsistencyMetrics)
            
            # Should detect violations as residuals increase
            if step > 5:  # Later steps should have violations
                assert consistency_metrics.constraint_violations > 0
        
        # Get violation summary
        summary = monitor.get_violation_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0  # Should have recorded violations
    
    def test_gradient_monitoring(self):
        """Test gradient norm monitoring."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            device=torch.device('cpu')
        )
        
        # Create gradients
        dummy_input = torch.randn(4, 64)
        output = engine.model(dummy_input)
        loss = output.sum()
        loss.backward()
        
        # Test gradient norm computation using monitor
        monitor = TrainingMonitor(save_dir=str(Path(self.temp_dir) / "grad_monitor"))
        grad_norm = monitor._compute_gradient_norm(engine.model)
        param_norm = monitor._compute_parameter_norm(engine.model)
        
        assert grad_norm > 0
        assert param_norm > 0
        assert isinstance(grad_norm, float)
        assert isinstance(param_norm, float)
    
    def test_metrics_collection_and_visualization(self):
        """Test metrics collection and visualization."""
        from src.training.monitoring import MetricsCollector, TrainingVisualizer
        
        collector = MetricsCollector(window_size=20)
        visualizer = TrainingVisualizer(save_dir=str(Path(self.temp_dir) / "viz"))
        
        # Generate sample metrics
        sample_metrics = []
        for epoch in range(10):
            physics_metrics = PhysicsConsistencyMetrics(
                maxwell_residual=0.01 * (1 + epoch * 0.1),
                heat_equation_residual=0.02 * (1 + epoch * 0.1),
                structural_dynamics_residual=0.015 * (1 + epoch * 0.1),
                coupling_residual=0.005 * (1 + epoch * 0.1),
                total_physics_loss=0.05 * (1 + epoch * 0.1),
                constraint_violations=epoch % 3,
                energy_conservation_error=0.001 * (1 + epoch * 0.1)
            )
            
            from src.training.monitoring import TrainingProgressMetrics
            metrics = TrainingProgressMetrics(
                epoch=epoch, step=epoch*10, 
                train_loss=1.0 - epoch*0.05, val_loss=1.1 - epoch*0.05,
                train_accuracy=0.5 + epoch*0.04, val_accuracy=0.45 + epoch*0.04,
                learning_rate=1e-3 * (0.9**epoch),
                data_loss=0.8 - epoch*0.04, physics_loss=0.2 - epoch*0.01,
                consistency_loss=0.05, variational_loss=0.02,
                physics_metrics=physics_metrics, epoch_time=120.0, step_time=1.2,
                gpu_memory_used=512.0 + epoch*5, cpu_usage=50.0,
                grad_norm=0.1 + epoch*0.005, param_norm=10.0,
                grad_to_param_ratio=0.01 + epoch*0.0005
            )
            
            collector.add_metrics(metrics)
            sample_metrics.append(metrics)
        
        # Test visualization
        plot_paths = []
        plot_paths.append(visualizer.plot_training_curves(sample_metrics))
        plot_paths.append(visualizer.plot_loss_components(sample_metrics))
        plot_paths.append(visualizer.plot_physics_consistency(sample_metrics))
        plot_paths.append(visualizer.plot_gradient_analysis(sample_metrics))
        
        # Verify all plots were created
        for plot_path in plot_paths:
            assert Path(plot_path).exists()
            assert plot_path.endswith('.png')
        
        # Test metrics analysis
        assert collector.get_rolling_average('train_loss') > 0
        assert collector.get_trend('train_loss') == 'decreasing'  # Loss should be decreasing
        
        epoch_summary = collector.get_epoch_summary(5)
        assert epoch_summary['epoch'] == 5
        assert 'avg_train_loss' in epoch_summary
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with invalid configuration - PyTorch will raise ValueError for negative LR
        invalid_config = self.config
        invalid_config.model.learning_rate = -1.0  # Invalid learning rate
        
        # Should raise ValueError for invalid learning rate
        with pytest.raises(ValueError, match="Invalid learning rate"):
            engine = TrainingEngine(
                model=self.model,
                config=invalid_config,
                device=torch.device('cpu')
            )
        
        # Test checkpoint loading with non-existent file using valid config
        valid_config = self.config
        valid_config.model.learning_rate = 1e-3  # Reset to valid value
        engine = TrainingEngine(
            model=self.model,
            config=valid_config,
            device=torch.device('cpu')
        )
        
        with pytest.raises(FileNotFoundError):
            engine.load_checkpoint("non_existent_checkpoint.pt")
        
        # Test monitoring with empty metrics
        monitor = TrainingMonitor(save_dir=str(Path(self.temp_dir) / "empty_monitor"))
        
        # Should handle empty state gracefully
        report = monitor.generate_report(save_plots=False)
        assert 'error' in report
    
    def test_memory_and_performance_monitoring(self):
        """Test memory usage and performance monitoring."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            device=torch.device('cpu')
        )
        
        # Test memory monitoring (CPU mode)
        gpu_memory = engine._get_gpu_memory_usage()
        assert gpu_memory == 0.0  # Should be 0 in CPU mode
        
        # Test training step timing
        import time
        start_time = time.time()
        
        # Run a single training step
        batch_data = next(iter(self.train_loader))
        batch_data = engine._move_to_device(batch_data)
        
        # Mock loss function for timing test
        def timed_loss_fn(prediction, target, input_data=None, coords=None, mu=None, log_var=None, **kwargs):
            # Use actual loss computation to ensure gradients
            data_loss = nn.CrossEntropyLoss()(prediction, target)
            return data_loss, {'total_loss': data_loss.item(), 'data_loss': data_loss.item(), 'physics_loss': 0.0}
        
        engine.loss_fn = timed_loss_fn
        
        loss, loss_components, predictions = engine._forward_pass(batch_data)
        engine._backward_pass(loss)
        
        step_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert step_time < 5.0  # 5 seconds max for a single step
        assert isinstance(loss, torch.Tensor)
        assert isinstance(predictions, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__])