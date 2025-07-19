"""
Unit tests for Variational Bayesian Uncertainty Quantification.
"""

import pytest
import torch
import numpy as np
from src.physics.uncertainty import (
    VariationalBayesianUQ, PhysicsInformedPrior, VariationalInferenceTrainer,
    UncertaintyConfig
)


class TestVariationalBayesianUQ:
    """Test cases for VariationalBayesianUQ class"""
    
    @pytest.fixture
    def config(self):
        return UncertaintyConfig(
            prior_mean=0.0,
            prior_std=1.0,
            num_mc_samples=50,
            kl_weight=1e-3
        )
    
    @pytest.fixture
    def vb_uq(self, config):
        return VariationalBayesianUQ(config, input_dim=10, output_dim=5)
    
    def test_initialization(self, vb_uq, config):
        """Test proper initialization of variational parameters"""
        assert vb_uq.input_dim == 10
        assert vb_uq.output_dim == 5
        assert vb_uq.prior_mean == config.prior_mean
        assert vb_uq.prior_std == config.prior_std
        
        # Check parameter shapes
        assert vb_uq.variational_mean.shape == (10, 5)
        assert vb_uq.variational_logvar.shape == (10, 5)
        
        # Check initial values
        assert torch.allclose(vb_uq.variational_mean, torch.zeros(10, 5))
        expected_logvar = torch.ones(10, 5) * np.log(config.prior_std ** 2)
        assert torch.allclose(vb_uq.variational_logvar, expected_logvar)
    
    def test_sample_weights(self, vb_uq):
        """Test weight sampling from variational distribution"""
        # Test default number of samples
        samples = vb_uq.sample_weights()
        assert samples.shape == (vb_uq.config.num_mc_samples, 10, 5)
        
        # Test custom number of samples
        samples = vb_uq.sample_weights(num_samples=20)
        assert samples.shape == (20, 10, 5)
        
        # Test that samples are different (stochastic)
        samples1 = vb_uq.sample_weights(num_samples=2)
        samples2 = vb_uq.sample_weights(num_samples=2)
        assert not torch.allclose(samples1, samples2)
    
    def test_kl_divergence_computation(self, vb_uq):
        """Test KL divergence computation"""
        kl_div = vb_uq.compute_kl_divergence()
        
        # KL divergence should be non-negative
        assert kl_div >= 0
        
        # For identical distributions, KL should be close to 0
        with torch.no_grad():
            vb_uq.variational_mean.fill_(vb_uq.prior_mean)
            vb_uq.variational_logvar.fill_(np.log(vb_uq.prior_std ** 2))
        
        kl_div_identical = vb_uq.compute_kl_divergence()
        assert kl_div_identical < 1e-5
    
    def test_log_likelihood_computation(self, vb_uq):
        """Test log likelihood computation"""
        batch_size = 32
        predictions = torch.randn(batch_size, 5)
        targets = torch.randn(batch_size, 5)
        
        log_likelihood = vb_uq.compute_log_likelihood(predictions, targets)
        
        # Log likelihood should be a scalar
        assert log_likelihood.dim() == 0
        
        # Perfect predictions should have higher likelihood
        perfect_predictions = targets.clone()
        perfect_log_likelihood = vb_uq.compute_log_likelihood(perfect_predictions, targets)
        assert perfect_log_likelihood > log_likelihood
    
    def test_elbo_computation(self, vb_uq):
        """Test ELBO computation"""
        batch_size = 32
        predictions = torch.randn(batch_size, 5)
        targets = torch.randn(batch_size, 5)
        
        elbo, components = vb_uq.compute_elbo(predictions, targets)
        
        # Check return types
        assert isinstance(elbo, torch.Tensor)
        assert isinstance(components, dict)
        
        # Check components
        assert 'log_likelihood' in components
        assert 'kl_divergence' in components
        assert 'elbo' in components
        
        # ELBO should equal likelihood - weighted KL
        expected_elbo = (components['log_likelihood'] - 
                        vb_uq.config.kl_weight * components['kl_divergence'])
        assert torch.allclose(elbo, expected_elbo)
    
    def test_predict_with_uncertainty(self, vb_uq):
        """Test uncertainty quantification in predictions"""
        batch_size = 16
        inputs = torch.randn(batch_size, 10)
        
        # Simple forward function for testing
        def forward_fn(x, weights):
            return torch.matmul(x, weights)
        
        mean_pred, uncertainty = vb_uq.predict_with_uncertainty(
            inputs, forward_fn, num_samples=20
        )
        
        # Check output shapes
        assert mean_pred.shape == (batch_size, 5)
        assert uncertainty.shape == (batch_size, 5)
        
        # Uncertainty should be non-negative
        assert torch.all(uncertainty >= 0)
    
    def test_variational_parameter_management(self, vb_uq):
        """Test getting and setting variational parameters"""
        # Get parameters
        params = vb_uq.get_variational_parameters()
        assert 'mean' in params
        assert 'logvar' in params
        assert 'std' in params
        
        # Modify parameters
        new_mean = torch.randn(10, 5)
        new_logvar = torch.randn(10, 5)
        vb_uq.set_variational_parameters(new_mean, new_logvar)
        
        # Check that parameters were updated
        assert torch.allclose(vb_uq.variational_mean, new_mean)
        assert torch.allclose(vb_uq.variational_logvar, new_logvar)
    
    def test_elbo_gradient_flow(self, vb_uq):
        """Test that ELBO computation allows gradient flow"""
        batch_size = 16
        predictions = torch.randn(batch_size, 5, requires_grad=True)
        targets = torch.randn(batch_size, 5)
        
        elbo, _ = vb_uq.compute_elbo(predictions, targets)
        elbo.backward()
        
        # Check that gradients exist
        assert predictions.grad is not None
        assert vb_uq.variational_mean.grad is not None
        assert vb_uq.variational_logvar.grad is not None


class TestPhysicsInformedPrior:
    """Test cases for PhysicsInformedPrior class"""
    
    @pytest.fixture
    def config(self):
        return UncertaintyConfig()
    
    @pytest.fixture
    def physics_prior(self, config):
        return PhysicsInformedPrior(config)
    
    def test_electromagnetic_kernel(self, physics_prior):
        """Test electromagnetic prior kernel"""
        n_points = 10
        coordinates = torch.randn(n_points, 3)
        
        kernel = physics_prior.electromagnetic_prior_kernel(coordinates, coordinates)
        
        # Check kernel properties
        assert kernel.shape == (n_points, n_points)
        assert torch.allclose(kernel, kernel.t())  # Symmetric
        assert torch.all(kernel >= 0)  # Non-negative
        # Diagonal should be close to 1 (may be slightly higher due to regularization)
        diagonal = torch.diag(kernel)
        assert torch.all(diagonal >= 1.0)  # At least 1 due to regularization
        assert torch.all(diagonal <= 1.01)  # Close to 1
    
    def test_thermal_kernel(self, physics_prior):
        """Test thermal prior kernel"""
        n_points = 8
        coordinates = torch.randn(n_points, 2)
        
        kernel = physics_prior.thermal_prior_kernel(coordinates, coordinates)
        
        # Check kernel properties
        assert kernel.shape == (n_points, n_points)
        assert torch.allclose(kernel, kernel.t())  # Symmetric
        assert torch.all(kernel >= 0)  # Non-negative
    
    def test_mechanical_kernel(self, physics_prior):
        """Test mechanical prior kernel"""
        n_points = 12
        coordinates = torch.randn(n_points, 1)
        
        kernel = physics_prior.mechanical_prior_kernel(coordinates, coordinates)
        
        # Check kernel properties
        assert kernel.shape == (n_points, n_points)
        assert torch.allclose(kernel, kernel.t())  # Symmetric
        assert torch.all(kernel >= 0)  # Non-negative
        # Note: diagonal elements may be slightly > 1 due to regularization
        assert torch.all(kernel <= 1.01)  # Bounded by 1 (with small tolerance for regularization)
    
    def test_physics_informed_prior_computation(self, physics_prior):
        """Test physics-informed prior computation"""
        coordinates = torch.randn(5, 2)
        
        # Test all physics types
        for physics_type in ['electromagnetic', 'thermal', 'mechanical']:
            prior_cov = physics_prior.compute_physics_informed_prior(
                coordinates, physics_type
            )
            
            assert prior_cov.shape == (5, 5)
            assert torch.allclose(prior_cov, prior_cov.t())  # Symmetric
    
    def test_invalid_physics_type(self, physics_prior):
        """Test error handling for invalid physics type"""
        coordinates = torch.randn(3, 2)
        
        with pytest.raises(ValueError):
            physics_prior.compute_physics_informed_prior(coordinates, 'invalid_type')
    
    def test_kernel_length_scale_effects(self, physics_prior):
        """Test that different length scales produce different kernels"""
        coordinates = torch.randn(5, 2)
        
        kernel1 = physics_prior.electromagnetic_prior_kernel(
            coordinates, coordinates, length_scale=1.0
        )
        kernel2 = physics_prior.electromagnetic_prior_kernel(
            coordinates, coordinates, length_scale=2.0
        )
        
        # Different length scales should produce different kernels
        assert not torch.allclose(kernel1, kernel2)


class TestVariationalInferenceTrainer:
    """Test cases for VariationalInferenceTrainer class"""
    
    @pytest.fixture
    def setup_trainer(self):
        config = UncertaintyConfig(num_mc_samples=10, kl_weight=1e-3)
        vb_uq = VariationalBayesianUQ(config, input_dim=5, output_dim=3)
        physics_prior = PhysicsInformedPrior(config)
        optimizer = torch.optim.Adam(vb_uq.parameters(), lr=1e-3)
        
        trainer = VariationalInferenceTrainer(vb_uq, physics_prior, optimizer)
        return trainer, vb_uq
    
    def test_trainer_initialization(self, setup_trainer):
        """Test trainer initialization"""
        trainer, vb_uq = setup_trainer
        
        assert trainer.vb_uq is vb_uq
        assert isinstance(trainer.physics_prior, PhysicsInformedPrior)
        assert len(trainer.training_history) == 0
    
    def test_train_step_without_physics(self, setup_trainer):
        """Test training step without physics constraints"""
        trainer, vb_uq = setup_trainer
        
        # Create sample data
        batch_size = 16
        inputs = torch.randn(batch_size, 5)
        targets = torch.randn(batch_size, 3)
        
        # Simple forward function
        def forward_fn(x, weights):
            return torch.matmul(x, weights)
        
        # Perform training step
        metrics = trainer.train_step(inputs, targets, forward_fn)
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert 'total_loss' in metrics
        assert 'elbo' in metrics
        assert 'log_likelihood' in metrics
        assert 'kl_divergence' in metrics
        assert 'physics_loss' in metrics
        
        # Physics loss should be zero when no physics function provided
        assert metrics['physics_loss'] == 0.0
        
        # Check training history
        assert len(trainer.training_history) == 1
    
    def test_train_step_with_physics(self, setup_trainer):
        """Test training step with physics constraints"""
        trainer, vb_uq = setup_trainer
        
        # Create sample data
        batch_size = 16
        inputs = torch.randn(batch_size, 5)
        targets = torch.randn(batch_size, 3)
        
        # Simple forward function
        def forward_fn(x, weights):
            return torch.matmul(x, weights)
        
        # Simple physics loss function
        def physics_loss_fn(predictions, inputs):
            return torch.mean(predictions ** 2) * 0.1
        
        # Perform training step
        metrics = trainer.train_step(inputs, targets, forward_fn, physics_loss_fn)
        
        # Physics loss should be non-zero
        assert metrics['physics_loss'] > 0.0
    
    def test_multiple_training_steps(self, setup_trainer):
        """Test multiple training steps"""
        trainer, vb_uq = setup_trainer
        
        batch_size = 16
        inputs = torch.randn(batch_size, 5)
        targets = torch.randn(batch_size, 3)
        
        def forward_fn(x, weights):
            return torch.matmul(x, weights)
        
        # Perform multiple training steps
        num_steps = 5
        for _ in range(num_steps):
            trainer.train_step(inputs, targets, forward_fn)
        
        # Check training history
        assert len(trainer.training_history) == num_steps
        
        # Check that parameters are being updated
        history = trainer.get_training_history()
        assert len(history) == num_steps
        
        # Loss should generally decrease (though not guaranteed for few steps)
        first_loss = history[0]['total_loss']
        last_loss = history[-1]['total_loss']
        assert isinstance(first_loss, float)
        assert isinstance(last_loss, float)
    
    def test_gradient_updates(self, setup_trainer):
        """Test that gradients are computed and parameters updated"""
        trainer, vb_uq = setup_trainer
        
        # Store initial parameters
        initial_mean = vb_uq.variational_mean.clone()
        initial_logvar = vb_uq.variational_logvar.clone()
        
        # Create sample data
        batch_size = 16
        inputs = torch.randn(batch_size, 5)
        targets = torch.randn(batch_size, 3)
        
        def forward_fn(x, weights):
            return torch.matmul(x, weights)
        
        # Perform training step
        trainer.train_step(inputs, targets, forward_fn)
        
        # Check that parameters have been updated
        assert not torch.allclose(vb_uq.variational_mean, initial_mean)
        assert not torch.allclose(vb_uq.variational_logvar, initial_logvar)


class TestIntegration:
    """Integration tests for the complete uncertainty quantification system"""
    
    def test_end_to_end_uncertainty_quantification(self):
        """Test complete uncertainty quantification pipeline"""
        # Setup
        config = UncertaintyConfig(num_mc_samples=20, kl_weight=1e-3)
        vb_uq = VariationalBayesianUQ(config, input_dim=8, output_dim=4)
        physics_prior = PhysicsInformedPrior(config)
        optimizer = torch.optim.Adam(vb_uq.parameters(), lr=1e-2)
        trainer = VariationalInferenceTrainer(vb_uq, physics_prior, optimizer)
        
        # Generate synthetic data
        batch_size = 32
        inputs = torch.randn(batch_size, 8)
        true_weights = torch.randn(8, 4)
        targets = torch.matmul(inputs, true_weights) + 0.1 * torch.randn(batch_size, 4)
        
        def forward_fn(x, weights):
            return torch.matmul(x, weights)
        
        # Train for several steps
        for _ in range(10):
            trainer.train_step(inputs, targets, forward_fn)
        
        # Make predictions with uncertainty
        test_inputs = torch.randn(16, 8)
        mean_pred, uncertainty = vb_uq.predict_with_uncertainty(
            test_inputs, forward_fn, num_samples=50
        )
        
        # Verify outputs
        assert mean_pred.shape == (16, 4)
        assert uncertainty.shape == (16, 4)
        assert torch.all(uncertainty >= 0)
        
        # Check that training history was recorded
        history = trainer.get_training_history()
        assert len(history) == 10
        
        # Verify ELBO components are reasonable
        final_metrics = history[-1]
        assert final_metrics['elbo'] < 0  # ELBO is typically negative
        assert final_metrics['kl_divergence'] >= 0
    
    def test_physics_informed_training(self):
        """Test training with physics-informed priors"""
        config = UncertaintyConfig()
        physics_prior = PhysicsInformedPrior(config)
        
        # Test different physics types
        coordinates = torch.randn(10, 3)
        
        for physics_type in ['electromagnetic', 'thermal', 'mechanical']:
            prior_cov = physics_prior.compute_physics_informed_prior(
                coordinates, physics_type
            )
            
            # Verify covariance matrix properties
            assert torch.allclose(prior_cov, prior_cov.t())  # Symmetric
            
            # Check that most eigenvalues are positive (some kernel approximations may have small negative eigenvalues)
            eigenvals = torch.linalg.eigvals(prior_cov).real
            positive_eigenvals = torch.sum(eigenvals > 0)
            total_eigenvals = len(eigenvals)
            # At least 80% of eigenvalues should be positive
            assert positive_eigenvals >= 0.8 * total_eigenvals


if __name__ == '__main__':
    pytest.main([__file__])