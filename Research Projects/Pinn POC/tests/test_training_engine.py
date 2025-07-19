"""
Unit tests for TrainingEngine and related components.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from src.training.training_engine import (
    TrainingEngine, TrainingMetrics, CheckpointData,
    LearningRateScheduler, DistributedTrainingManager
)
from src.config.config_manager import AVPINOConfig
from src.physics.loss import PhysicsInformedLoss


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=64, output_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics object."""
        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            train_accuracy=0.8,
            val_accuracy=0.75,
            physics_loss=0.1,
            data_loss=0.4,
            consistency_loss=0.05,
            variational_loss=0.02,
            learning_rate=1e-3,
            epoch_time=120.5,
            gpu_memory_used=512.0
        )
        
        assert metrics.epoch == 1
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.6
        assert metrics.train_accuracy == 0.8
        assert metrics.val_accuracy == 0.75
        
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = TrainingMetrics(
            epoch=1, train_loss=0.5, val_loss=0.6, train_accuracy=0.8,
            val_accuracy=0.75, physics_loss=0.1, data_loss=0.4,
            consistency_loss=0.05, variational_loss=0.02,
            learning_rate=1e-3, epoch_time=120.5, gpu_memory_used=512.0
        )
        
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['epoch'] == 1
        assert metrics_dict['train_loss'] == 0.5
        assert 'learning_rate' in metrics_dict


class TestCheckpointData:
    """Test CheckpointData functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = Path(self.temp_dir) / "test_checkpoint.pt"
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoint data."""
        # Create real config instead of mock to avoid pickling issues
        from src.config.config_manager import DataConfig, PhysicsConfig, ModelConfig, InferenceConfig
        
        config = AVPINOConfig(
            data=DataConfig(),
            physics=PhysicsConfig(), 
            model=ModelConfig(),
            inference=InferenceConfig()
        )
        
        checkpoint_data = CheckpointData(
            epoch=10,
            model_state_dict={'layer.weight': torch.randn(10, 5)},
            optimizer_state_dict={'state': {}, 'param_groups': []},
            scheduler_state_dict={'last_epoch': 10},
            scaler_state_dict={},
            loss_state_dict={'weights': {'data_loss': 1.0}},
            metrics_history=[],
            config=config,
            best_val_loss=0.5,
            best_val_accuracy=0.8,
            physics_constraints={}
        )
        
        # Save checkpoint
        checkpoint_data.save(self.checkpoint_path)
        assert self.checkpoint_path.exists()
        
        # Load checkpoint
        loaded_data = CheckpointData.load(self.checkpoint_path)
        assert loaded_data.epoch == 10
        assert loaded_data.best_val_loss == 0.5
        assert loaded_data.best_val_accuracy == 0.8
        
    def test_checkpoint_load_nonexistent(self):
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError):
            CheckpointData.load("nonexistent_checkpoint.pt")


class TestLearningRateScheduler:
    """Test LearningRateScheduler functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleTestModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = LearningRateScheduler(
            self.optimizer,
            scheduler_type="cosine_annealing",
            initial_lr=1e-3,
            warmup_epochs=5
        )
        
        assert scheduler.initial_lr == 1e-3
        assert scheduler.warmup_epochs == 5
        assert not scheduler.divergence_detected
        
    def test_warmup_phase(self):
        """Test learning rate warmup."""
        scheduler = LearningRateScheduler(
            self.optimizer,
            scheduler_type="cosine_annealing",
            initial_lr=1e-3,
            warmup_epochs=5
        )
        
        # Test warmup epochs
        for epoch in range(5):
            continue_training = scheduler.step(epoch)
            assert continue_training
            expected_lr = 1e-3 * (epoch + 1) / 5
            assert abs(scheduler.get_lr() - expected_lr) < 1e-6
    
    def test_divergence_detection(self):
        """Test training divergence detection."""
        scheduler = LearningRateScheduler(
            self.optimizer,
            scheduler_type="reduce_on_plateau",
            divergence_threshold=1.5,
            warmup_epochs=2
        )
        
        # Simulate normal training
        for epoch in range(2, 12):
            val_loss = 1.0 if epoch < 10 else 2.0  # Sudden increase
            continue_training = scheduler.step(epoch, val_loss)
            
            if epoch >= 10:
                # Should detect divergence after a few epochs of high loss
                if not continue_training:
                    assert scheduler.divergence_detected
                    break
    
    def test_different_scheduler_types(self):
        """Test different scheduler types."""
        scheduler_types = ["cosine_annealing", "reduce_on_plateau", "exponential"]
        
        for scheduler_type in scheduler_types:
            scheduler = LearningRateScheduler(
                self.optimizer,
                scheduler_type=scheduler_type,
                warmup_epochs=2
            )
            
            # Should not raise exception
            continue_training = scheduler.step(5, 0.5)
            assert continue_training
            
    def test_invalid_scheduler_type(self):
        """Test invalid scheduler type."""
        with pytest.raises(ValueError):
            LearningRateScheduler(
                self.optimizer,
                scheduler_type="invalid_scheduler"
            )


class TestDistributedTrainingManager:
    """Test DistributedTrainingManager functionality."""
    
    def test_single_gpu_initialization(self):
        """Test initialization with single GPU."""
        with patch('torch.cuda.device_count', return_value=1):
            manager = DistributedTrainingManager()
            assert not manager.is_distributed
            assert manager.is_main_process
    
    def test_multi_gpu_initialization(self):
        """Test initialization with multiple GPUs."""
        with patch('torch.cuda.device_count', return_value=4):
            manager = DistributedTrainingManager(world_size=4, rank=0)
            assert manager.world_size == 4
            assert manager.rank == 0
            assert manager.is_main_process
    
    def test_model_wrapping(self):
        """Test model wrapping for distributed training."""
        model = SimpleTestModel()
        manager = DistributedTrainingManager(world_size=1)  # Single GPU
        
        wrapped_model = manager.wrap_model(model)
        assert wrapped_model is model  # Should not wrap for single GPU
    
    def test_dataloader_creation(self):
        """Test distributed dataloader creation."""
        dataset = TensorDataset(torch.randn(100, 64), torch.randint(0, 4, (100,)))
        manager = DistributedTrainingManager(world_size=1)
        
        dataloader = manager.create_dataloader(dataset, batch_size=16)
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 16


class TestTrainingEngine:
    """Test TrainingEngine functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AVPINOConfig(
            data=Mock(),
            physics=Mock(),
            model=Mock(learning_rate=1e-3, max_epochs=10, patience=5),
            inference=Mock(),
            checkpoint_dir=self.temp_dir
        )
        self.model = SimpleTestModel()
        self.loss_fn = Mock(spec=PhysicsInformedLoss)
        
        # Setup mock loss function return values
        self.loss_fn.return_value = (
            torch.tensor(0.5),  # loss
            {  # loss_components
                'data_loss': 0.4,
                'physics_loss': 0.1,
                'consistency_loss': 0.0,
                'variational_loss': 0.0,
                'total_loss': 0.5
            }
        )
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_training_engine_initialization(self):
        """Test TrainingEngine initialization."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            loss_fn=self.loss_fn,
            device=torch.device('cpu')
        )
        
        assert engine.model is not None
        assert engine.config == self.config
        assert engine.device == torch.device('cpu')
        assert engine.current_epoch == 0
        assert engine.best_val_loss == float('inf')
        
    def test_training_engine_with_default_optimizer(self):
        """Test TrainingEngine with default optimizer creation."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            device=torch.device('cpu')
        )
        
        assert engine.optimizer is not None
        assert isinstance(engine.optimizer, torch.optim.AdamW)
    
    def test_move_to_device(self):
        """Test moving batch data to device."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            device=torch.device('cpu')
        )
        
        batch_data = {
            'input': torch.randn(16, 64),
            'target': torch.randint(0, 4, (16,)),
            'metadata': 'some_string'
        }
        
        moved_data = engine._move_to_device(batch_data)
        assert moved_data['input'].device == torch.device('cpu')
        assert moved_data['target'].device == torch.device('cpu')
        assert moved_data['metadata'] == 'some_string'
    
    def test_get_batch_size(self):
        """Test getting batch size from batch data."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            device=torch.device('cpu')
        )
        
        batch_data = {
            'input': torch.randn(16, 64),
            'target': torch.randint(0, 4, (16,))
        }
        
        batch_size = engine._get_batch_size(batch_data)
        assert batch_size == 16
    
    def test_calculate_accuracy_classification(self):
        """Test accuracy calculation for classification."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            device=torch.device('cpu')
        )
        
        # Classification case
        predictions = torch.tensor([[0.1, 0.9, 0.0, 0.0],
                                   [0.0, 0.0, 0.8, 0.2],
                                   [0.7, 0.1, 0.1, 0.1]])
        targets = torch.tensor([1, 2, 0])
        
        correct = engine._calculate_accuracy(predictions, targets)
        assert correct == 3  # All predictions correct
    
    def test_calculate_accuracy_regression(self):
        """Test accuracy calculation for regression."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            device=torch.device('cpu')
        )
        
        # Regression case (single output)
        predictions = torch.tensor([1.0, 2.05, 3.15])  # Close to targets
        targets = torch.tensor([1.0, 2.0, 3.0])
        
        correct = engine._calculate_accuracy(predictions, targets)
        assert correct == 3  # All within threshold
    
    def test_forward_pass(self):
        """Test forward pass computation."""
        # Create a proper mock that doesn't conflict with kwargs
        def mock_loss_fn(prediction, target, input_data=None, coords=None, mu=None, log_var=None, **kwargs):
            return (
                torch.tensor(0.5),
                {
                    'data_loss': 0.4,
                    'physics_loss': 0.1,
                    'consistency_loss': 0.0,
                    'variational_loss': 0.0,
                    'total_loss': 0.5
                }
            )
        
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            loss_fn=mock_loss_fn,
            device=torch.device('cpu')
        )
        
        batch_data = {
            'input': torch.randn(4, 64),
            'target': torch.randint(0, 4, (4,))
        }
        
        loss, loss_components, predictions = engine._forward_pass(batch_data)
        
        assert isinstance(loss, torch.Tensor)
        assert isinstance(loss_components, dict)
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape[0] == 4  # Batch size
    
    def test_backward_pass(self):
        """Test backward pass computation."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            device=torch.device('cpu')
        )
        
        # Create a simple loss
        loss = torch.tensor(0.5, requires_grad=True)
        
        # Should not raise exception
        engine._backward_pass(loss)
    
    def test_validation(self):
        """Test validation loop."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            loss_fn=self.loss_fn,
            device=torch.device('cpu')
        )
        
        # Create validation dataset
        val_dataset = TensorDataset(
            torch.randn(32, 64),
            torch.randint(0, 4, (32,))
        )
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        val_loss, val_accuracy = engine.validate(val_loader)
        
        assert isinstance(val_loss, float)
        assert isinstance(val_accuracy, float)
        assert val_loss >= 0
        assert 0 <= val_accuracy <= 1
    
    def test_train_epoch(self):
        """Test single epoch training."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            loss_fn=self.loss_fn,
            device=torch.device('cpu')
        )
        
        # Create training dataset
        train_dataset = TensorDataset(
            torch.randn(32, 64),
            torch.randint(0, 4, (32,))
        )
        train_loader = DataLoader(train_dataset, batch_size=8)
        
        # Mock validation callback
        validation_callback = Mock(return_value=(0.6, 0.75))
        
        metrics = engine.train_epoch(train_loader, validation_callback)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.epoch == 0
        assert metrics.train_loss >= 0
        assert metrics.val_loss == 0.6
        assert metrics.val_accuracy == 0.75
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            device=torch.device('cpu')
        )
        
        # Save checkpoint
        checkpoint_path = engine.save_checkpoint("test_checkpoint.pt")
        assert Path(checkpoint_path).exists()
        
        # Modify engine state
        engine.current_epoch = 5
        engine.best_val_loss = 0.3
        
        # Load checkpoint
        checkpoint_data = engine.load_checkpoint(checkpoint_path)
        
        assert engine.current_epoch == 0  # Should be restored
        assert engine.best_val_loss == float('inf')  # Should be restored
        assert isinstance(checkpoint_data, CheckpointData)
    
    def test_training_summary(self):
        """Test getting training summary."""
        engine = TrainingEngine(
            model=self.model,
            config=self.config,
            device=torch.device('cpu')
        )
        
        # Add some mock metrics
        engine.metrics_history = [
            TrainingMetrics(
                epoch=0, train_loss=0.8, val_loss=0.9, train_accuracy=0.6,
                val_accuracy=0.5, physics_loss=0.1, data_loss=0.7,
                consistency_loss=0.0, variational_loss=0.0,
                learning_rate=1e-3, epoch_time=60.0, gpu_memory_used=256.0
            )
        ]
        
        summary = engine.get_training_summary()
        
        assert isinstance(summary, dict)
        assert 'training_status' in summary
        assert 'latest_metrics' in summary
        assert 'training_progress' in summary
        assert 'hardware_utilization' in summary


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AVPINOConfig(
            data=Mock(),
            physics=Mock(),
            model=Mock(learning_rate=1e-3, max_epochs=3, patience=10),
            inference=Mock(),
            checkpoint_dir=self.temp_dir
        )
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_training_loop(self):
        """Test complete training loop integration."""
        model = SimpleTestModel()
        
        # Create simple physics-informed loss
        loss_fn = Mock(spec=PhysicsInformedLoss)
        loss_fn.return_value = (
            torch.tensor(0.5),
            {
                'data_loss': 0.4,
                'physics_loss': 0.1,
                'consistency_loss': 0.0,
                'variational_loss': 0.0,
                'total_loss': 0.5
            }
        )
        
        engine = TrainingEngine(
            model=model,
            config=self.config,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.randn(64, 64),
            torch.randint(0, 4, (64,))
        )
        val_dataset = TensorDataset(
            torch.randn(32, 64),
            torch.randint(0, 4, (32,))
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # Run training
        metrics_history = engine.train(train_loader, val_loader, num_epochs=2)
        
        assert len(metrics_history) == 2
        assert all(isinstance(m, TrainingMetrics) for m in metrics_history)
        assert metrics_history[0].epoch == 0
        assert metrics_history[1].epoch == 1
    
    def test_training_with_early_stopping(self):
        """Test training with early stopping."""
        model = SimpleTestModel()
        
        # Create loss function that returns increasing validation loss
        loss_fn = Mock(spec=PhysicsInformedLoss)
        
        def mock_loss_side_effect(*args, **kwargs):
            # Simulate increasing loss for early stopping
            epoch = getattr(mock_loss_side_effect, 'call_count', 0)
            mock_loss_side_effect.call_count = epoch + 1
            loss_value = 1.0 + epoch * 0.1  # Increasing loss
            
            return (
                torch.tensor(loss_value),
                {
                    'data_loss': loss_value * 0.8,
                    'physics_loss': loss_value * 0.2,
                    'consistency_loss': 0.0,
                    'variational_loss': 0.0,
                    'total_loss': loss_value
                }
            )
        
        loss_fn.side_effect = mock_loss_side_effect
        
        # Set very low patience for quick early stopping
        self.config.model.patience = 2
        
        engine = TrainingEngine(
            model=model,
            config=self.config,
            loss_fn=loss_fn,
            device=torch.device('cpu')
        )
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.randn(32, 64),
            torch.randint(0, 4, (32,))
        )
        val_dataset = TensorDataset(
            torch.randn(16, 64),
            torch.randint(0, 4, (16,))
        )
        
        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        # Run training - should stop early
        metrics_history = engine.train(train_loader, val_loader, num_epochs=10)
        
        # Should stop before 10 epochs due to early stopping
        assert len(metrics_history) < 10
        assert engine.early_stopping_counter > 0


if __name__ == "__main__":
    pytest.main([__file__])