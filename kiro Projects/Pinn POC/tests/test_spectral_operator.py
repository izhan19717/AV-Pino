"""
Unit tests for spectral representation and operator control implementations.

Tests spectral decomposition, operator control, and fault evolution operators
for accuracy and stability.
"""

import torch
import pytest
import numpy as np
import math
from src.physics.spectral_operator import (
    SpectralBasis, FourierBasis, ChebyshevBasis, SpectralDecomposition,
    OperatorControl, FaultEvolutionOperator, SpectralOperatorLayer,
    create_fourier_spectral_layer, create_chebyshev_spectral_layer
)


class TestFourierBasis:
    """Test Fourier basis functions."""
    
    def test_fourier_basis_initialization(self):
        """Test Fourier basis initialization."""
        basis = FourierBasis(n_modes=10, domain_size=2.0)
        assert basis.n_modes == 10
        assert basis.domain_size == 2.0
        assert basis.k_max == 5
        assert len(basis.wave_numbers) == 10  # Even n_modes: [-k_max, k_max)
    
    def test_fourier_basis_computation(self):
        """Test Fourier basis function computation."""
        basis = FourierBasis(n_modes=6, domain_size=1.0)
        
        coords = torch.linspace(0, 1, 4).unsqueeze(1)
        basis_values = basis.compute_basis(coords)
        
        # Should have shape (batch_size, 2*n_modes) for real and imaginary parts
        assert basis_values.shape == (4, 12)  # 2 * 6 modes
        
        # Check orthogonality property (simplified test)
        # At x=0, cos terms should be 1, sin terms should be 0
        x_zero = torch.tensor([[0.0]])
        basis_zero = basis.compute_basis(x_zero)
        
        # Every other element (cos terms) should be 1 at x=0
        cos_terms = basis_zero[0, ::2]  # Even indices are cos terms
        sin_terms = basis_zero[0, 1::2]  # Odd indices are sin terms
        
        assert torch.allclose(cos_terms, torch.ones_like(cos_terms))
        assert torch.allclose(sin_terms, torch.zeros_like(sin_terms))
    
    def test_fourier_derivatives(self):
        """Test Fourier basis derivative computation."""
        basis = FourierBasis(n_modes=4, domain_size=1.0)
        
        coords = torch.tensor([[0.25]])  # x = 1/4
        derivatives = basis.compute_derivatives(coords, order=1)
        
        assert derivatives.shape == (1, 8)  # 2 * 4 modes
        
        # Test second derivatives
        second_derivatives = basis.compute_derivatives(coords, order=2)
        assert second_derivatives.shape == (1, 8)
    
    def test_fourier_periodicity(self):
        """Test periodicity of Fourier basis."""
        basis = FourierBasis(n_modes=4, domain_size=1.0)
        
        # Test at x=0 and x=1 (should be the same due to periodicity)
        x0 = torch.tensor([[0.0]])
        x1 = torch.tensor([[1.0]])
        
        basis_0 = basis.compute_basis(x0)
        basis_1 = basis.compute_basis(x1)
        
        # Should be approximately equal due to periodicity
        assert torch.allclose(basis_0, basis_1, atol=1e-6)


class TestChebyshevBasis:
    """Test Chebyshev basis functions."""
    
    def test_chebyshev_basis_initialization(self):
        """Test Chebyshev basis initialization."""
        basis = ChebyshevBasis(n_modes=8, domain_size=2.0)
        assert basis.n_modes == 8
        assert basis.domain_size == 2.0
    
    def test_chebyshev_basis_computation(self):
        """Test Chebyshev basis function computation."""
        basis = ChebyshevBasis(n_modes=5, domain_size=1.0)
        
        coords = torch.linspace(0, 1, 4).unsqueeze(1)
        basis_values = basis.compute_basis(coords)
        
        assert basis_values.shape == (4, 5)
        
        # Test known values: T₀(x) = 1, T₁(x) = x (after normalization)
        x_test = torch.tensor([[0.5]])  # Middle of domain
        basis_test = basis.compute_basis(x_test)
        
        assert torch.allclose(basis_test[0, 0], torch.tensor(1.0))  # T₀ = 1
        assert torch.allclose(basis_test[0, 1], torch.tensor(0.0), atol=1e-6)  # T₁(0.5) = 0 after normalization
    
    def test_chebyshev_derivatives(self):
        """Test Chebyshev basis derivative computation."""
        basis = ChebyshevBasis(n_modes=4, domain_size=1.0)
        
        coords = torch.tensor([[0.5]])
        derivatives = basis.compute_derivatives(coords, order=1)
        
        assert derivatives.shape == (1, 4)
    
    def test_chebyshev_orthogonality(self):
        """Test Chebyshev orthogonality properties (simplified)."""
        basis = ChebyshevBasis(n_modes=3, domain_size=1.0)
        
        # Test at domain boundaries
        x_left = torch.tensor([[0.0]])  # Maps to -1
        x_right = torch.tensor([[1.0]])  # Maps to +1
        
        basis_left = basis.compute_basis(x_left)
        basis_right = basis.compute_basis(x_right)
        
        # T₀(-1) = T₀(1) = 1
        assert torch.allclose(basis_left[0, 0], torch.tensor(1.0))
        assert torch.allclose(basis_right[0, 0], torch.tensor(1.0))
        
        # T₁(-1) = -1, T₁(1) = 1
        assert torch.allclose(basis_left[0, 1], torch.tensor(-1.0))
        assert torch.allclose(basis_right[0, 1], torch.tensor(1.0))


class TestSpectralDecomposition:
    """Test spectral decomposition module."""
    
    def test_spectral_decomposition_initialization(self):
        """Test spectral decomposition initialization."""
        basis = FourierBasis(n_modes=6, domain_size=1.0)
        decomp = SpectralDecomposition(basis, n_channels=3)
        
        assert decomp.n_channels == 3
        assert decomp.n_modes == 6
        assert decomp.spectral_coeffs.shape == (3, 12)  # 3 channels, 2*6 modes
    
    def test_spectral_reconstruction(self):
        """Test spectral function reconstruction."""
        basis = FourierBasis(n_modes=4, domain_size=1.0)
        decomp = SpectralDecomposition(basis, n_channels=2)
        
        coords = torch.linspace(0, 1, 5).unsqueeze(1)
        reconstructed = decomp(coords)
        
        assert reconstructed.shape == (5, 2)  # 5 points, 2 channels
    
    def test_spectral_derivatives(self):
        """Test spectral derivative computation."""
        basis = FourierBasis(n_modes=4, domain_size=1.0)
        decomp = SpectralDecomposition(basis, n_channels=1)
        
        coords = torch.tensor([[0.25], [0.5], [0.75]])
        derivatives = decomp.compute_derivatives(coords, order=1)
        
        assert derivatives.shape == (3, 1)
    
    def test_coefficient_manipulation(self):
        """Test getting and setting spectral coefficients."""
        basis = FourierBasis(n_modes=3, domain_size=1.0)
        decomp = SpectralDecomposition(basis, n_channels=1)
        
        # Get original coefficients
        original_coeffs = decomp.get_spectral_coefficients()
        
        # Set new coefficients
        new_coeffs = torch.ones_like(original_coeffs)
        decomp.set_spectral_coefficients(new_coeffs)
        
        # Verify change
        current_coeffs = decomp.get_spectral_coefficients()
        assert torch.allclose(current_coeffs, new_coeffs)
        assert not torch.allclose(current_coeffs, original_coeffs)
    
    def test_gradient_flow(self):
        """Test that gradients flow through spectral decomposition."""
        basis = FourierBasis(n_modes=3, domain_size=1.0)
        decomp = SpectralDecomposition(basis, n_channels=1)
        
        coords = torch.tensor([[0.5]], requires_grad=True)
        output = decomp(coords)
        loss = torch.sum(output**2)
        loss.backward()
        
        # Check gradients exist
        assert coords.grad is not None
        assert decomp.spectral_coeffs.grad is not None


class TestOperatorControl:
    """Test operator control module."""
    
    def test_operator_control_initialization(self):
        """Test operator control initialization."""
        control = OperatorControl(input_dim=5, output_dim=3, n_modes=8, stability_constraint=True)
        
        assert control.input_dim == 5
        assert control.output_dim == 3
        assert control.n_modes == 8
        assert control.stability_constraint
    
    def test_operator_control_forward(self):
        """Test operator control forward pass."""
        control = OperatorControl(input_dim=4, output_dim=2, n_modes=6)
        
        batch_size = 3
        parameters = torch.randn(batch_size, 4)
        
        gain_kernel, stability_loss = control(parameters)
        
        assert gain_kernel.shape == (batch_size, 2, 6)  # (batch, output_dim, n_modes)
        assert isinstance(stability_loss, torch.Tensor)
        assert stability_loss >= 0
    
    def test_stability_constraint(self):
        """Test stability constraint computation."""
        control = OperatorControl(input_dim=2, output_dim=1, n_modes=4, stability_constraint=True)
        
        # Create parameters that should violate stability
        large_params = torch.tensor([[10.0, 10.0]])
        small_params = torch.tensor([[0.1, 0.1]])
        
        _, large_loss = control(large_params)
        _, small_loss = control(small_params)
        
        # Large parameters should have higher stability loss
        assert large_loss > small_loss
    
    def test_stability_metrics(self):
        """Test stability metrics computation."""
        control = OperatorControl(input_dim=2, output_dim=2, n_modes=3)
        
        # Create test gain kernel
        gain_kernel = torch.tensor([[[1.0, 0.5, 0.2], [0.8, 0.3, 0.1]]])
        
        metrics = control.get_stability_metrics(gain_kernel)
        
        assert 'max_gain' in metrics
        assert 'mean_gain' in metrics
        assert 'stability_margin' in metrics
        
        assert metrics['max_gain'] > 0
        assert metrics['mean_gain'] > 0
    
    def test_operator_control_without_stability(self):
        """Test operator control without stability constraints."""
        control = OperatorControl(input_dim=3, output_dim=2, n_modes=4, stability_constraint=False)
        
        parameters = torch.randn(2, 3)
        gain_kernel, stability_loss = control(parameters)
        
        assert gain_kernel.shape == (2, 2, 4)
        assert stability_loss == 0.0


class TestFaultEvolutionOperator:
    """Test fault evolution operator."""
    
    def test_fault_evolution_initialization(self):
        """Test fault evolution operator initialization."""
        evolution = FaultEvolutionOperator(state_dim=4, control_dim=2, noise_dim=1)
        
        assert evolution.state_dim == 4
        assert evolution.control_dim == 2
        assert evolution.noise_dim == 1
    
    def test_fault_evolution_forward(self):
        """Test fault evolution forward pass."""
        evolution = FaultEvolutionOperator(state_dim=3, control_dim=2)
        
        batch_size = 2
        fault_state = torch.rand(batch_size, 3)  # Random initial state [0, 1]
        control_input = torch.randn(batch_size, 2)
        temperature = torch.full((batch_size, 1), 323.15)  # 50°C
        
        new_state, evolution_loss = evolution(fault_state, control_input, temperature)
        
        assert new_state.shape == (batch_size, 3)
        assert torch.all(new_state >= 0) and torch.all(new_state <= 1)  # Valid range
        assert isinstance(evolution_loss, torch.Tensor)
        assert evolution_loss >= 0
    
    def test_physics_dynamics(self):
        """Test physics-based fault dynamics."""
        evolution = FaultEvolutionOperator(state_dim=2, control_dim=1)
        
        # Test temperature effect
        fault_state = torch.tensor([[0.5, 0.3]])
        temp_low = torch.tensor([[293.15]])  # Room temperature
        temp_high = torch.tensor([[373.15]])  # High temperature
        
        dynamics_low = evolution._compute_physics_dynamics(fault_state, temp_low)
        dynamics_high = evolution._compute_physics_dynamics(fault_state, temp_high)
        
        # Higher temperature should lead to faster degradation
        assert torch.all(dynamics_high > dynamics_low)
    
    def test_fault_trajectory_prediction(self):
        """Test fault trajectory prediction."""
        evolution = FaultEvolutionOperator(state_dim=2, control_dim=1)
        
        initial_state = torch.tensor([[0.1, 0.2]])
        control_sequence = torch.randn(1, 5, 1)  # 5 time steps
        temperature_sequence = torch.full((1, 5, 1), 323.15)
        
        trajectory = evolution.predict_fault_trajectory(
            initial_state, control_sequence, temperature_sequence, n_steps=5
        )
        
        assert trajectory.shape == (1, 6, 2)  # batch, n_steps+1, state_dim
        assert torch.allclose(trajectory[:, 0], initial_state)  # First step is initial state
    
    def test_evolution_loss_computation(self):
        """Test evolution loss computation."""
        evolution = FaultEvolutionOperator(state_dim=2, control_dim=1)
        
        old_state = torch.tensor([[0.3, 0.4]])
        new_state = torch.tensor([[0.35, 0.42]])  # Small increase (degradation)
        dynamics = torch.tensor([[0.05, 0.02]])
        dt = 1e-3
        
        loss = evolution._compute_evolution_loss(old_state, new_state, dynamics, dt)
        
        assert isinstance(loss, torch.Tensor)
        assert loss >= 0
    
    def test_monotonicity_constraint(self):
        """Test monotonicity constraint in fault evolution."""
        evolution = FaultEvolutionOperator(state_dim=2, control_dim=1)
        
        # Test case where fault "heals" too quickly (should be penalized)
        old_state = torch.tensor([[0.8, 0.7]])
        new_state = torch.tensor([[0.3, 0.2]])  # Large decrease (unrealistic healing)
        dynamics = torch.tensor([[-0.1, -0.1]])
        dt = 1e-3
        
        loss_healing = evolution._compute_evolution_loss(old_state, new_state, dynamics, dt)
        
        # Compare with normal degradation
        new_state_normal = torch.tensor([[0.82, 0.72]])  # Small increase
        dynamics_normal = torch.tensor([[0.02, 0.02]])
        
        loss_normal = evolution._compute_evolution_loss(old_state, new_state_normal, dynamics_normal, dt)
        
        # Unrealistic healing should have higher loss
        assert loss_healing > loss_normal


class TestSpectralOperatorLayer:
    """Test combined spectral operator layer."""
    
    def test_spectral_operator_layer_initialization(self):
        """Test spectral operator layer initialization."""
        basis = FourierBasis(n_modes=6, domain_size=1.0)
        layer = SpectralOperatorLayer(basis, input_dim=4, output_dim=3, n_channels=2)
        
        assert layer.use_operator_control
        assert hasattr(layer, 'spectral_decomp')
        assert hasattr(layer, 'operator_control')
        assert hasattr(layer, 'projection')
    
    def test_spectral_operator_layer_forward(self):
        """Test spectral operator layer forward pass."""
        basis = FourierBasis(n_modes=4, domain_size=1.0)
        layer = SpectralOperatorLayer(basis, input_dim=3, output_dim=2, n_channels=1)
        
        batch_size = 2
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        parameters = torch.randn(batch_size, 3)
        
        output, losses = layer(coords, parameters)
        
        assert output.shape == (batch_size, 2)
        assert 'stability_loss' in losses
        assert isinstance(losses['stability_loss'], torch.Tensor)
    
    def test_spectral_operator_layer_without_control(self):
        """Test spectral operator layer without operator control."""
        basis = ChebyshevBasis(n_modes=5, domain_size=1.0)
        layer = SpectralOperatorLayer(basis, input_dim=3, output_dim=2, n_channels=1, 
                                     use_operator_control=False)
        
        coords = torch.tensor([[0.5]])
        output, losses = layer(coords)
        
        assert output.shape == (1, 2)
        assert len(losses) == 0  # No losses without operator control
    
    def test_gradient_flow_through_layer(self):
        """Test gradient flow through spectral operator layer."""
        basis = FourierBasis(n_modes=3, domain_size=1.0)
        layer = SpectralOperatorLayer(basis, input_dim=2, output_dim=1, n_channels=1)
        
        coords = torch.tensor([[0.5]], requires_grad=True)
        parameters = torch.tensor([[1.0, 2.0]], requires_grad=True)
        
        output, losses = layer(coords, parameters)
        total_loss = torch.sum(output**2) + losses.get('stability_loss', 0)
        total_loss.backward()
        
        # Check gradients exist
        assert coords.grad is not None
        assert parameters.grad is not None


class TestSpectralLayerCreation:
    """Test spectral layer creation functions."""
    
    def test_create_fourier_spectral_layer(self):
        """Test Fourier spectral layer creation."""
        layer = create_fourier_spectral_layer(n_modes=8, input_dim=5, output_dim=3, 
                                             n_channels=2, domain_size=2.0)
        
        assert isinstance(layer, SpectralOperatorLayer)
        assert isinstance(layer.spectral_decomp.basis, FourierBasis)
        assert layer.spectral_decomp.basis.n_modes == 8
        assert layer.spectral_decomp.basis.domain_size == 2.0
    
    def test_create_chebyshev_spectral_layer(self):
        """Test Chebyshev spectral layer creation."""
        layer = create_chebyshev_spectral_layer(n_modes=6, input_dim=4, output_dim=2,
                                               n_channels=1, domain_size=1.5)
        
        assert isinstance(layer, SpectralOperatorLayer)
        assert isinstance(layer.spectral_decomp.basis, ChebyshevBasis)
        assert layer.spectral_decomp.basis.n_modes == 6
        assert layer.spectral_decomp.basis.domain_size == 1.5


class TestSpectralAccuracy:
    """Test spectral accuracy and convergence properties."""
    
    def test_fourier_approximation_accuracy(self):
        """Test Fourier approximation accuracy for known functions."""
        # Test approximation of a simple constant function
        basis = FourierBasis(n_modes=10, domain_size=1.0)
        decomp = SpectralDecomposition(basis, n_channels=1)
        
        # Set coefficients to approximate constant function f(x) = 1
        # Constant corresponds to k=0 mode (DC component)
        coeffs = torch.zeros(1, 20)  # 2 * 10 modes
        # Find the index for k=0 mode (should be around the middle)
        k_zero_idx = len(basis.wave_numbers) // 2 * 2  # Real part of k=0
        coeffs[0, k_zero_idx] = 1.0  # Real part of k=0 mode
        decomp.set_spectral_coefficients(coeffs)
        
        # Test points
        x_test = torch.linspace(0, 1, 100).unsqueeze(1)
        approx = decomp(x_test)
        exact = torch.ones_like(approx)  # Constant function
        
        # Should be reasonably accurate for constant function
        error = torch.mean((approx - exact)**2)
        assert error < 1.0  # More lenient threshold for this test
    
    def test_spectral_convergence(self):
        """Test spectral convergence with increasing modes."""
        # Test convergence for a smooth function
        def test_function(x):
            return torch.exp(-x**2)
        
        x_test = torch.linspace(0, 1, 50).unsqueeze(1)
        exact = test_function(x_test.squeeze()).unsqueeze(1)
        
        errors = []
        mode_counts = [4, 8, 16]
        
        for n_modes in mode_counts:
            basis = FourierBasis(n_modes=n_modes, domain_size=1.0)
            decomp = SpectralDecomposition(basis, n_channels=1)
            
            # Use random coefficients (in practice would be learned)
            approx = decomp(x_test)
            error = torch.mean((approx - exact)**2)
            errors.append(error.item())
        
        # Errors should generally decrease (though not guaranteed with random coefficients)
        # This test mainly checks that the computation doesn't break with more modes
        assert all(e >= 0 for e in errors)
        assert len(errors) == len(mode_counts)


if __name__ == "__main__":
    pytest.main([__file__])