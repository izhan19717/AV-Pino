"""
Integration tests for the complete PhysicsInformedLoss system.

Tests the integration of all loss components and their interactions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.physics.loss import PhysicsInformedLoss
from src.physics.constraints import PDEConstraint


class MockPDEConstraint(PDEConstraint):
    """Mock PDE constraint for testing."""
    
    def __init__(self, name: str = "mock_constraint", weight: float = 1.0):
        super().__init__(weight, name)  # Fix parameter order
    
    def compute_residual(self, prediction: torch.Tensor, input_data: torch.Tensor, 
                        coords: torch.Tensor) -> torch.Tensor:
        """Mock residual computation."""
        return torch.mean(prediction ** 2)


class TestPhysicsInformedLossIntegration:
    """Test cases for complete PhysicsInformedLoss integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.num_classes = 5
        self.feature_dim = 32
        self.latent_dim = 16
        
        # Create mock constraints
        self.mock_constraints = [
            MockPDEConstraint("maxwell", 1.0),
            MockPDEConstraint("heat", 0.8),
            MockPDEConstraint("structural", 0.6)
        ]
        
        # Configuration for PhysicsInformedLoss
        self.loss_config = {
            'data_loss_config': {
                'loss_type': 'cross_entropy',
                'weight': 1.0
            },
            'physics_loss_config': {
                'constraints': self.mock_constraints,
                'weight': 1.0,
                'adaptive_weighting': True
            },
            'consistency_loss_config': {
                'weight': 0.5,
                'coupling_strength': 1.0
            },
            'variational_loss_config': {
                'weight': 0.1,
                'kl_weight': 1.0,
                'prior_std': 1.0
            },
            'adaptive_weighting_config': {
                'initial_weights': {
                    'data_loss': 1.0,
                    'physics_loss': 1.0,
                    'consistency_loss': 0.5,
                    'variational_loss': 0.1
                },
                'adaptation_method': 'uncertainty_weighting',
                'update_frequency': 5
            }
        }
        
        # Create test tensors
        self.prediction = torch.randn(self.batch_size, self.num_classes)
        self.target = torch.randint(0, self.num_classes, (self.batch_size,))
        self.input_data = torch.randn(self.batch_size, self.feature_dim)
        self.coords = torch.randn(self.batch_size, 3)  # 3D coordinates
        
        # Multi-physics features
        self.em_features = torch.randn(self.batch_size, self.feature_dim)
        self.thermal_features = torch.randn(self.batch_size, self.feature_dim)
        self.mechanical_features = torch.randn(self.batch_size, self.feature_dim)
        
        # Variational parameters
        self.mu = torch.randn(self.batch_size, self.latent_dim)
        self.log_var = torch.randn(self.batch_size, self.latent_dim)
    
    def test_physics_informed_loss_initialization(self):
        """Test PhysicsInformedLoss initialization."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        assert hasattr(loss_system, 'data_loss')
        assert hasattr(loss_system, 'physics_loss')
        assert hasattr(loss_system, 'consistency_loss')
        assert hasattr(loss_system, 'variational_loss')
        assert hasattr(loss_system, 'adaptive_weighting')
        
        assert loss_system.use_physics_loss == True
        assert loss_system.use_consistency_loss == True
        assert loss_system.use_variational_loss == True
        assert loss_system.use_adaptive_weighting == True
    
    def test_complete_loss_computation(self):
        """Test complete loss computation with all components."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        total_loss, loss_components = loss_system(
            prediction=self.prediction,
            target=self.target,
            input_data=self.input_data,
            coords=self.coords,
            electromagnetic_features=self.em_features,
            thermal_features=self.thermal_features,
            mechanical_features=self.mechanical_features,
            mu=self.mu,
            log_var=self.log_var
        )
        
        # Check that total loss is computed
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0
        assert torch.isfinite(total_loss)
        
        # Check that all loss components are present
        expected_components = [
            'data_loss', 'physics_loss', 'consistency_loss', 
            'variational_loss', 'total_loss', 'weights'
        ]
        for component in expected_components:
            assert component in loss_components
        
        # Check that weights are provided
        assert isinstance(loss_components['weights'], dict)
        assert len(loss_components['weights']) > 0
    
    def test_partial_loss_computation(self):
        """Test loss computation with only some components available."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        # Only provide data loss inputs
        total_loss, loss_components = loss_system(
            prediction=self.prediction,
            target=self.target
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert torch.isfinite(total_loss)
        
        # Data loss should be computed, others should be zero
        assert loss_components['data_loss'].item() > 0
        assert loss_components['physics_loss'].item() == 0
        assert loss_components['consistency_loss'].item() == 0
        assert loss_components['variational_loss'].item() == 0
    
    def test_loss_scheduling(self):
        """Test loss weight scheduling over epochs."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        # Test early epoch weights
        loss_system.set_epoch(0, 1000)
        early_weights = loss_system.get_current_weights()
        
        # Test mid-training weights
        loss_system.set_epoch(500, 1000)
        mid_weights = loss_system.get_current_weights()
        
        # Test late training weights
        loss_system.set_epoch(900, 1000)
        late_weights = loss_system.get_current_weights()
        
        # Physics loss should ramp up
        assert early_weights['physics_loss'] < mid_weights['physics_loss']
        assert mid_weights['physics_loss'] <= late_weights['physics_loss']
        
        # Data loss should remain constant
        assert early_weights['data_loss'] == mid_weights['data_loss'] == late_weights['data_loss']
    
    def test_adaptive_weighting_integration(self):
        """Test adaptive weighting integration with loss computation."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        initial_weights = loss_system.get_current_weights()
        
        # Run multiple forward passes to trigger adaptive weighting
        for i in range(15):  # More than update_frequency
            # Vary loss magnitudes to trigger adaptation
            varied_prediction = self.prediction + 0.1 * i * torch.randn_like(self.prediction)
            
            total_loss, loss_components = loss_system(
                prediction=varied_prediction,
                target=self.target,
                input_data=self.input_data,
                coords=self.coords,
                electromagnetic_features=self.em_features,
                thermal_features=self.thermal_features,
                mechanical_features=self.mechanical_features,
                mu=self.mu,
                log_var=self.log_var
            )
        
        final_weights = loss_system.get_current_weights()
        
        # Weights should be valid
        for weight in final_weights.values():
            assert isinstance(weight, float)
            assert weight >= 0  # Allow zero weights (components can be disabled)
            assert np.isfinite(weight)
    
    def test_component_enable_disable(self):
        """Test enabling and disabling loss components."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        # Disable consistency loss
        loss_system.enable_component('consistency_loss', False)
        
        total_loss, loss_components = loss_system(
            prediction=self.prediction,
            target=self.target,
            input_data=self.input_data,
            coords=self.coords,
            electromagnetic_features=self.em_features,
            thermal_features=self.thermal_features,
            mechanical_features=self.mechanical_features,
            mu=self.mu,
            log_var=self.log_var
        )
        
        # Consistency loss should be zero when disabled
        assert loss_components['consistency_loss'].item() == 0
        
        # Re-enable and check it's computed again
        loss_system.enable_component('consistency_loss', True)
        
        total_loss2, loss_components2 = loss_system(
            prediction=self.prediction,
            target=self.target,
            input_data=self.input_data,
            coords=self.coords,
            electromagnetic_features=self.em_features,
            thermal_features=self.thermal_features,
            mechanical_features=self.mechanical_features,
            mu=self.mu,
            log_var=self.log_var
        )
        
        # Consistency loss should be non-zero when re-enabled
        assert loss_components2['consistency_loss'].item() > 0
    
    def test_loss_history_tracking(self):
        """Test loss history tracking functionality."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        # Run several forward passes
        for _ in range(5):
            total_loss, loss_components = loss_system(
                prediction=self.prediction,
                target=self.target,
                input_data=self.input_data,
                coords=self.coords,
                electromagnetic_features=self.em_features,
                thermal_features=self.thermal_features,
                mechanical_features=self.mechanical_features,
                mu=self.mu,
                log_var=self.log_var
            )
        
        # Check loss history
        history = loss_system.get_loss_history()
        
        assert isinstance(history, dict)
        assert 'total_loss' in history
        assert 'data_loss' in history
        assert len(history['total_loss']) == 5
        assert len(history['data_loss']) == 5
    
    def test_loss_summary_statistics(self):
        """Test loss summary statistics computation."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        # Run several forward passes with varying inputs
        for i in range(10):
            varied_prediction = self.prediction + 0.1 * i * torch.randn_like(self.prediction)
            
            total_loss, loss_components = loss_system(
                prediction=varied_prediction,
                target=self.target,
                input_data=self.input_data,
                coords=self.coords,
                electromagnetic_features=self.em_features,
                thermal_features=self.thermal_features,
                mechanical_features=self.mechanical_features,
                mu=self.mu,
                log_var=self.log_var
            )
        
        # Get summary statistics
        summary = loss_system.get_loss_summary()
        
        assert isinstance(summary, dict)
        
        # Check that summary contains expected statistics
        expected_keys = ['total_loss_mean', 'total_loss_std', 'total_loss_latest',
                        'data_loss_mean', 'data_loss_std', 'data_loss_latest']
        
        for key in expected_keys:
            assert key in summary
            assert isinstance(summary[key], (float, np.floating))
            assert np.isfinite(summary[key])
    
    def test_reset_adaptive_weights(self):
        """Test resetting adaptive weights to initial state."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        # Run forward passes to change adaptive weights
        for _ in range(15):
            total_loss, loss_components = loss_system(
                prediction=self.prediction,
                target=self.target,
                input_data=self.input_data,
                coords=self.coords,
                electromagnetic_features=self.em_features,
                thermal_features=self.thermal_features,
                mechanical_features=self.mechanical_features,
                mu=self.mu,
                log_var=self.log_var
            )
        
        # Reset adaptive weights
        loss_system.reset_adaptive_weights()
        
        # Check that weights are reset to initial values
        current_weights = loss_system.adaptive_weighting.get_weights()
        initial_weights = self.loss_config['adaptive_weighting_config']['initial_weights']
        
        for key in initial_weights:
            assert abs(current_weights[key] - initial_weights[key]) < 0.1
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the loss system."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        # Create parameters that require gradients
        prediction = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        mu = torch.randn(self.batch_size, self.latent_dim, requires_grad=True)
        log_var = torch.randn(self.batch_size, self.latent_dim, requires_grad=True)
        
        total_loss, loss_components = loss_system(
            prediction=prediction,
            target=self.target,
            input_data=self.input_data,
            coords=self.coords,
            electromagnetic_features=self.em_features,
            thermal_features=self.thermal_features,
            mechanical_features=self.mechanical_features,
            mu=mu,
            log_var=log_var
        )
        
        # Compute gradients
        total_loss.backward()
        
        # Check that gradients are computed
        assert prediction.grad is not None
        assert mu.grad is not None
        assert log_var.grad is not None
        
        # Check that gradients are finite
        assert torch.isfinite(prediction.grad).all()
        assert torch.isfinite(mu.grad).all()
        assert torch.isfinite(log_var.grad).all()
    
    def test_device_compatibility(self):
        """Test that loss system works on different devices."""
        loss_system = PhysicsInformedLoss(**self.loss_config)
        
        # Test on CPU (default)
        total_loss_cpu, _ = loss_system(
            prediction=self.prediction,
            target=self.target,
            input_data=self.input_data,
            coords=self.coords,
            electromagnetic_features=self.em_features,
            thermal_features=self.thermal_features,
            mechanical_features=self.mechanical_features,
            mu=self.mu,
            log_var=self.log_var
        )
        
        assert total_loss_cpu.device.type == 'cpu'
        
        # Test GPU compatibility if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            loss_system = loss_system.to(device)
            
            prediction_gpu = self.prediction.to(device)
            target_gpu = self.target.to(device)
            input_data_gpu = self.input_data.to(device)
            coords_gpu = self.coords.to(device)
            em_features_gpu = self.em_features.to(device)
            thermal_features_gpu = self.thermal_features.to(device)
            mechanical_features_gpu = self.mechanical_features.to(device)
            mu_gpu = self.mu.to(device)
            log_var_gpu = self.log_var.to(device)
            
            total_loss_gpu, _ = loss_system(
                prediction=prediction_gpu,
                target=target_gpu,
                input_data=input_data_gpu,
                coords=coords_gpu,
                electromagnetic_features=em_features_gpu,
                thermal_features=thermal_features_gpu,
                mechanical_features=mechanical_features_gpu,
                mu=mu_gpu,
                log_var=log_var_gpu
            )
            
            assert total_loss_gpu.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__])