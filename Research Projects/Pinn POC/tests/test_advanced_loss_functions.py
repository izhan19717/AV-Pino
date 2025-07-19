"""
Unit tests for advanced loss functions in the physics-informed loss system.

Tests ConsistencyLoss, VariationalLoss, and AdaptiveLossWeighting components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.physics.loss import (
    ConsistencyLoss, 
    VariationalLoss, 
    AdaptiveLossWeighting
)


class TestConsistencyLoss:
    """Test cases for ConsistencyLoss class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.consistency_loss = ConsistencyLoss(weight=1.0, coupling_strength=1.0)
        self.batch_size = 4
        self.feature_dim = 32
        
        # Create mock feature tensors
        self.em_features = torch.randn(self.batch_size, self.feature_dim)
        self.thermal_features = torch.randn(self.batch_size, self.feature_dim)
        self.mechanical_features = torch.randn(self.batch_size, self.feature_dim)
        self.prediction = torch.randn(self.batch_size, 10)
        self.target = torch.randn(self.batch_size, 10)
    
    def test_consistency_loss_initialization(self):
        """Test ConsistencyLoss initialization."""
        assert self.consistency_loss.weight == 1.0
        assert self.consistency_loss.coupling_strength == 1.0
        assert self.consistency_loss.name == "consistency_loss"
    
    def test_compute_loss_basic(self):
        """Test basic consistency loss computation."""
        loss = self.consistency_loss.compute_loss(
            prediction=self.prediction,
            target=self.target,
            electromagnetic_features=self.em_features,
            thermal_features=self.thermal_features,
            mechanical_features=self.mechanical_features
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_thermal_mechanical_coupling(self):
        """Test thermal-mechanical coupling computation."""
        # Create strongly correlated features to test coupling
        base_features = torch.randn(self.batch_size, self.feature_dim)
        thermal_corr = base_features + 0.01 * torch.randn(self.batch_size, self.feature_dim)  # Very similar
        mechanical_corr = base_features + 0.01 * torch.randn(self.batch_size, self.feature_dim)  # Very similar
        
        loss_corr = self.consistency_loss.compute_loss(
            prediction=self.prediction,
            target=self.target,
            electromagnetic_features=self.em_features,
            thermal_features=thermal_corr,
            mechanical_features=mechanical_corr
        )
        
        # Create completely uncorrelated features
        thermal_uncorr = torch.randn(self.batch_size, self.feature_dim)
        mechanical_uncorr = torch.randn(self.batch_size, self.feature_dim)
        
        loss_uncorr = self.consistency_loss.compute_loss(
            prediction=self.prediction,
            target=self.target,
            electromagnetic_features=self.em_features,
            thermal_features=thermal_uncorr,
            mechanical_features=mechanical_uncorr
        )
        
        # Both losses should be finite and positive
        assert torch.isfinite(loss_corr) and loss_corr.item() >= 0
        assert torch.isfinite(loss_uncorr) and loss_uncorr.item() >= 0
        
        # The coupling mechanism should produce different losses for different correlations
        assert loss_corr.item() != loss_uncorr.item()
    
    def test_em_mechanical_coupling(self):
        """Test electromagnetic-mechanical coupling computation."""
        # Test with normalized features
        em_norm = torch.nn.functional.normalize(self.em_features, dim=1)
        mech_norm = torch.nn.functional.normalize(self.mechanical_features, dim=1)
        
        loss = self.consistency_loss.compute_loss(
            prediction=self.prediction,
            target=self.target,
            electromagnetic_features=em_norm,
            thermal_features=self.thermal_features,
            mechanical_features=mech_norm
        )
        
        assert torch.isfinite(loss)
        assert loss.item() >= 0
    
    def test_coupling_strength_scaling(self):
        """Test that coupling strength properly scales the loss."""
        loss_weak = ConsistencyLoss(coupling_strength=0.5).compute_loss(
            prediction=self.prediction,
            target=self.target,
            electromagnetic_features=self.em_features,
            thermal_features=self.thermal_features,
            mechanical_features=self.mechanical_features
        )
        
        loss_strong = ConsistencyLoss(coupling_strength=2.0).compute_loss(
            prediction=self.prediction,
            target=self.target,
            electromagnetic_features=self.em_features,
            thermal_features=self.thermal_features,
            mechanical_features=self.mechanical_features
        )
        
        # Strong coupling should produce higher loss
        assert loss_strong.item() > loss_weak.item()


class TestVariationalLoss:
    """Test cases for VariationalLoss class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.variational_loss = VariationalLoss(weight=1.0, kl_weight=1.0, prior_std=1.0)
        self.batch_size = 4
        self.latent_dim = 16
        
        # Create mock tensors
        self.prediction = torch.randn(self.batch_size, 10)
        self.target = torch.randn(self.batch_size, 10)
        self.mu = torch.randn(self.batch_size, self.latent_dim)
        self.log_var = torch.randn(self.batch_size, self.latent_dim)
    
    def test_variational_loss_initialization(self):
        """Test VariationalLoss initialization."""
        assert self.variational_loss.weight == 1.0
        assert self.variational_loss.kl_weight == 1.0
        assert self.variational_loss.prior_std == 1.0
        assert self.variational_loss.name == "variational_loss"
    
    def test_compute_loss_basic(self):
        """Test basic variational loss computation."""
        total_loss, loss_components = self.variational_loss.compute_loss(
            prediction=self.prediction,
            target=self.target,
            mu=self.mu,
            log_var=self.log_var
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0
        assert total_loss.item() >= 0
        
        # Check loss components
        assert 'reconstruction_loss' in loss_components
        assert 'kl_loss' in loss_components
        assert 'total_variational_loss' in loss_components
        
        # Verify total loss equals sum of components
        expected_total = loss_components['reconstruction_loss'] + self.variational_loss.kl_weight * loss_components['kl_loss']
        assert torch.allclose(total_loss, expected_total, atol=1e-6)
    
    def test_kl_divergence_computation(self):
        """Test KL divergence computation."""
        # Test with zero mean and unit variance (should give small KL)
        mu_zero = torch.zeros(self.batch_size, self.latent_dim)
        log_var_zero = torch.zeros(self.batch_size, self.latent_dim)  # var = 1
        
        kl_zero = self.variational_loss._compute_kl_divergence(mu_zero, log_var_zero)
        
        # KL should be close to zero for N(0,1) vs N(0,1)
        assert kl_zero.item() < 0.1
        
        # Test with non-zero mean (should give larger KL)
        mu_nonzero = torch.ones(self.batch_size, self.latent_dim)
        kl_nonzero = self.variational_loss._compute_kl_divergence(mu_nonzero, log_var_zero)
        
        assert kl_nonzero.item() > kl_zero.item()
    
    def test_kl_weight_annealing(self):
        """Test KL weight annealing schedules."""
        # Test linear annealing
        self.variational_loss.update_kl_weight(epoch=0, total_epochs=100, annealing_type="linear")
        initial_weight = self.variational_loss.kl_weight
        
        self.variational_loss.update_kl_weight(epoch=25, total_epochs=100, annealing_type="linear")
        mid_weight = self.variational_loss.kl_weight
        
        self.variational_loss.update_kl_weight(epoch=50, total_epochs=100, annealing_type="linear")
        final_weight = self.variational_loss.kl_weight
        
        assert initial_weight <= mid_weight <= final_weight
        
        # Test cosine annealing
        self.variational_loss.update_kl_weight(epoch=0, total_epochs=100, annealing_type="cosine")
        cosine_initial = self.variational_loss.kl_weight
        
        self.variational_loss.update_kl_weight(epoch=100, total_epochs=100, annealing_type="cosine")
        cosine_final = self.variational_loss.kl_weight
        
        assert cosine_initial > cosine_final
    
    def test_prior_std_effect(self):
        """Test effect of different prior standard deviations."""
        # Narrow prior (small std)
        narrow_loss = VariationalLoss(prior_std=0.5)
        narrow_kl = narrow_loss._compute_kl_divergence(self.mu, self.log_var)
        
        # Wide prior (large std)
        wide_loss = VariationalLoss(prior_std=2.0)
        wide_kl = wide_loss._compute_kl_divergence(self.mu, self.log_var)
        
        # Narrow prior should generally give higher KL divergence
        assert narrow_kl.item() > wide_kl.item()


class TestAdaptiveLossWeighting:
    """Test cases for AdaptiveLossWeighting class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.initial_weights = {
            'data_loss': 1.0,
            'physics_loss': 1.0,
            'consistency_loss': 0.5
        }
        self.adaptive_weighting = AdaptiveLossWeighting(
            initial_weights=self.initial_weights,
            adaptation_method="uncertainty_weighting",
            update_frequency=5
        )
    
    def test_adaptive_weighting_initialization(self):
        """Test AdaptiveLossWeighting initialization."""
        assert self.adaptive_weighting.weights == self.initial_weights
        assert self.adaptive_weighting.adaptation_method == "uncertainty_weighting"
        assert self.adaptive_weighting.update_frequency == 5
        assert self.adaptive_weighting.step_count == 0
    
    def test_weight_update_basic(self):
        """Test basic weight update functionality."""
        # Create varying loss values to trigger adaptation
        loss_values_list = [
            {'data_loss': torch.tensor(0.5 + 0.1 * i), 'physics_loss': torch.tensor(1.0 + 0.2 * i), 'consistency_loss': torch.tensor(0.2 + 0.05 * i)}
            for i in range(15)
        ]
        
        initial_weights = self.adaptive_weighting.get_weights()
        
        # Update multiple times with varying losses to trigger weight adaptation
        for loss_values in loss_values_list:
            self.adaptive_weighting.update_weights(loss_values)
        
        updated_weights = self.adaptive_weighting.get_weights()
        
        # Check that the weighting system is functioning (weights should be valid)
        assert all(isinstance(w, float) and w > 0 for w in updated_weights.values())
        # Weights should sum to approximately 1.0 (normalized) rather than number of weights
        assert abs(sum(updated_weights.values()) - 1.0) < 0.1
    
    def test_uncertainty_weighting_method(self):
        """Test uncertainty-based weighting adaptation."""
        # Create loss values with different variances
        stable_loss = [torch.tensor(0.5 + 0.01 * np.random.randn()) for _ in range(20)]
        unstable_loss = [torch.tensor(1.0 + 0.5 * np.random.randn()) for _ in range(20)]
        
        # Feed loss history
        for i in range(20):
            loss_values = {
                'data_loss': stable_loss[i],
                'physics_loss': unstable_loss[i],
                'consistency_loss': torch.tensor(0.3)
            }
            self.adaptive_weighting.update_weights(loss_values)
        
        final_weights = self.adaptive_weighting.get_weights()
        
        # Stable loss should get higher weight than unstable loss
        assert final_weights['data_loss'] > final_weights['physics_loss']
    
    def test_gradient_norm_balancing(self):
        """Test gradient norm balancing method."""
        grad_balancing = AdaptiveLossWeighting(
            initial_weights=self.initial_weights,
            adaptation_method="gradient_norm_balancing",
            update_frequency=1
        )
        
        # Simulate different loss magnitudes
        loss_values = {
            'data_loss': torch.tensor(2.0),  # High loss
            'physics_loss': torch.tensor(0.1),  # Low loss
            'consistency_loss': torch.tensor(0.5)
        }
        
        initial_weights = grad_balancing.get_weights()
        
        # Update weights
        for _ in range(5):
            grad_balancing.update_weights(loss_values)
        
        final_weights = grad_balancing.get_weights()
        
        # High loss should get lower weight
        assert final_weights['data_loss'] < final_weights['physics_loss']
    
    def test_homoscedastic_uncertainty(self):
        """Test homoscedastic uncertainty weighting."""
        homo_weighting = AdaptiveLossWeighting(
            initial_weights=self.initial_weights,
            adaptation_method="homoscedastic_uncertainty",
            update_frequency=5
        )
        
        # Create consistent vs inconsistent loss patterns
        for i in range(15):
            loss_values = {
                'data_loss': torch.tensor(0.5),  # Consistent
                'physics_loss': torch.tensor(1.0 + 0.5 * np.sin(i)),  # Variable
                'consistency_loss': torch.tensor(0.3)
            }
            homo_weighting.update_weights(loss_values)
        
        final_weights = homo_weighting.get_weights()
        
        # Consistent loss should get higher weight
        assert final_weights['data_loss'] > final_weights['physics_loss']
    
    def test_weight_normalization(self):
        """Test that weights are properly normalized."""
        # Create varying loss values to trigger normalization
        varying_losses = [
            {'data_loss': torch.tensor(1.0 + 0.5 * i), 'physics_loss': torch.tensor(0.5 + 0.2 * i), 'consistency_loss': torch.tensor(0.3 + 0.1 * i)}
            for i in range(15)
        ]
        
        for loss_values in varying_losses:
            self.adaptive_weighting.update_weights(loss_values)
        
        weights = self.adaptive_weighting.get_weights()
        total_weight = sum(weights.values())
        
        # Weights should be positive and reasonably normalized
        assert all(w > 0 for w in weights.values())
        assert total_weight > 0.5  # Should have reasonable total weight
    
    def test_manual_weight_setting(self):
        """Test manual weight setting functionality."""
        new_weights = {
            'data_loss': 2.0,
            'physics_loss': 0.5,
            'consistency_loss': 1.5
        }
        
        self.adaptive_weighting.set_weights(new_weights)
        updated_weights = self.adaptive_weighting.get_weights()
        
        for key, value in new_weights.items():
            assert updated_weights[key] == value
    
    def test_weight_history_tracking(self):
        """Test weight history tracking."""
        history = self.adaptive_weighting.get_weight_history()
        
        assert isinstance(history, dict)
        assert len(history) == len(self.initial_weights)
        
        for key in self.initial_weights:
            assert key in history
            assert isinstance(history[key], list)


if __name__ == "__main__":
    pytest.main([__file__])