"""
Unit tests for physics constraint implementations.

Tests PDE residual computation accuracy and constraint layer functionality.
"""

import torch
import pytest
import numpy as np
from src.physics.constraints import (
    PDEConstraint, MaxwellConstraint, HeatEquationConstraint,
    StructuralDynamicsConstraint, PhysicsConstraintLayer
)


class TestPDEConstraint:
    """Test base PDEConstraint class."""
    
    def test_constraint_initialization(self):
        """Test constraint initialization with weight and name."""
        class DummyConstraint(PDEConstraint):
            def compute_residual(self, prediction, input_data, coords):
                return torch.tensor(0.0)
        
        constraint = DummyConstraint(weight=2.0, name="dummy")
        assert constraint.get_constraint_weight() == 2.0
        assert constraint.name == "dummy"
    
    def test_weight_update(self):
        """Test constraint weight updating."""
        class DummyConstraint(PDEConstraint):
            def compute_residual(self, prediction, input_data, coords):
                return torch.tensor(0.0)
        
        constraint = DummyConstraint(weight=1.0)
        constraint.set_constraint_weight(3.0)
        assert constraint.get_constraint_weight() == 3.0


class TestMaxwellConstraint:
    """Test Maxwell equation constraints."""
    
    def test_maxwell_initialization(self):
        """Test Maxwell constraint initialization."""
        constraint = MaxwellConstraint(weight=1.5)
        assert constraint.get_constraint_weight() == 1.5
        assert constraint.name == "maxwell"
        assert constraint.mu_0 == 4e-7 * np.pi
        assert constraint.epsilon_0 == 8.854e-12
    
    def test_maxwell_residual_computation(self):
        """Test Maxwell equation residual computation."""
        constraint = MaxwellConstraint()
        
        # Create test data with E and B fields
        batch_size = 4
        prediction = torch.randn(batch_size, 10)  # E_x,E_y,E_z,B_x,B_y,B_z,...
        input_data = torch.randn(batch_size, 6)   # Current density J
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        residual = constraint.compute_residual(prediction, input_data, coords)
        
        assert isinstance(residual, torch.Tensor)
        assert residual.shape == torch.Size([])  # Scalar residual
        assert residual >= 0  # Residual should be non-negative (squared)
    
    def test_maxwell_curl_computation(self):
        """Test curl computation for Maxwell constraints."""
        constraint = MaxwellConstraint()
        
        # Test with simple field
        field = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        dx = torch.tensor([0.1])
        
        curl = constraint._compute_curl(field, dx)
        assert curl.shape == field.shape
    
    def test_maxwell_divergence_computation(self):
        """Test divergence computation for Maxwell constraints."""
        constraint = MaxwellConstraint()
        
        field = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        dx = torch.tensor([0.1])
        
        div = constraint._compute_divergence(field, dx)
        assert div.shape[0] == field.shape[0]


class TestHeatEquationConstraint:
    """Test heat equation constraints."""
    
    def test_heat_initialization(self):
        """Test heat equation constraint initialization."""
        constraint = HeatEquationConstraint(weight=2.0, thermal_diffusivity=1e-4)
        assert constraint.get_constraint_weight() == 2.0
        assert constraint.name == "heat_equation"
        assert constraint.alpha == 1e-4
    
    def test_heat_residual_computation(self):
        """Test heat equation residual computation."""
        constraint = HeatEquationConstraint()
        
        batch_size = 4
        prediction = torch.randn(batch_size, 8)  # Temperature at index 6
        input_data = torch.randn(batch_size, 3)  # Heat source
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        residual = constraint.compute_residual(prediction, input_data, coords)
        
        assert isinstance(residual, torch.Tensor)
        assert residual.shape == torch.Size([])
        assert residual >= 0
    
    def test_heat_laplacian_computation(self):
        """Test Laplacian computation for heat equation."""
        constraint = HeatEquationConstraint()
        
        # Test with quadratic function (should have constant Laplacian)
        x = torch.linspace(0, 1, 5)
        field = (x**2).unsqueeze(1)  # f(x) = x²
        dx = torch.tensor([0.25])
        
        laplacian = constraint._compute_laplacian(field, dx)
        assert laplacian.shape == field.shape


class TestStructuralDynamicsConstraint:
    """Test structural dynamics constraints."""
    
    def test_structural_initialization(self):
        """Test structural dynamics constraint initialization."""
        constraint = StructuralDynamicsConstraint(
            weight=1.5, density=8000.0, youngs_modulus=2.1e11
        )
        assert constraint.get_constraint_weight() == 1.5
        assert constraint.name == "structural_dynamics"
        assert constraint.rho == 8000.0
        assert constraint.E == 2.1e11
    
    def test_structural_residual_computation(self):
        """Test structural dynamics residual computation."""
        constraint = StructuralDynamicsConstraint()
        
        batch_size = 4
        prediction = torch.randn(batch_size, 12)  # Displacement at indices 7-9
        input_data = torch.randn(batch_size, 8)   # Body forces at indices 3-5
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        residual = constraint.compute_residual(prediction, input_data, coords)
        
        assert isinstance(residual, torch.Tensor)
        assert residual.shape == torch.Size([])
        assert residual >= 0
    
    def test_stress_divergence_computation(self):
        """Test stress divergence computation."""
        constraint = StructuralDynamicsConstraint()
        
        displacement = torch.randn(4, 3)
        coords = torch.linspace(0, 1, 4).unsqueeze(1)
        
        stress_div = constraint._compute_stress_divergence(displacement, coords)
        assert stress_div.shape == displacement.shape


class TestPhysicsConstraintLayer:
    """Test physics constraint layer."""
    
    def test_layer_initialization(self):
        """Test constraint layer initialization."""
        constraints = [
            MaxwellConstraint(weight=1.0),
            HeatEquationConstraint(weight=2.0),
            StructuralDynamicsConstraint(weight=1.5)
        ]
        
        layer = PhysicsConstraintLayer(constraints)
        assert len(layer.constraints) == 3
        assert torch.allclose(layer.constraint_weights, torch.tensor([1.0, 2.0, 1.5]))
    
    def test_layer_forward_pass(self):
        """Test constraint layer forward pass."""
        constraints = [
            MaxwellConstraint(weight=1.0),
            HeatEquationConstraint(weight=2.0)
        ]
        
        layer = PhysicsConstraintLayer(constraints)
        
        batch_size = 4
        prediction = torch.randn(batch_size, 10)
        input_data = torch.randn(batch_size, 6)
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        output_pred, residuals = layer.forward(prediction, input_data, coords)
        
        # Check output prediction is unchanged
        assert torch.allclose(output_pred, prediction)
        
        # Check residuals dictionary
        assert "maxwell" in residuals
        assert "heat_equation" in residuals
        assert "total_physics_loss" in residuals
        
        # Check residual values are scalars
        for key, value in residuals.items():
            assert isinstance(value, torch.Tensor)
            if key != "total_physics_loss":
                assert value.shape == torch.Size([])
    
    def test_physics_loss_computation(self):
        """Test physics loss computation."""
        constraints = [MaxwellConstraint(weight=1.0)]
        layer = PhysicsConstraintLayer(constraints)
        
        batch_size = 4
        prediction = torch.randn(batch_size, 10)
        input_data = torch.randn(batch_size, 6)
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        physics_loss = layer.get_physics_loss(prediction, input_data, coords)
        
        assert isinstance(physics_loss, torch.Tensor)
        assert physics_loss.shape == torch.Size([])
        assert physics_loss >= 0
    
    def test_constraint_weight_update(self):
        """Test constraint weight updating."""
        constraints = [
            MaxwellConstraint(weight=1.0),
            HeatEquationConstraint(weight=2.0)
        ]
        
        layer = PhysicsConstraintLayer(constraints)
        
        # Update weights
        new_weights = {"maxwell": 3.0, "heat_equation": 4.0}
        layer.update_constraint_weights(new_weights)
        
        assert layer.constraint_weights[0] == 3.0
        assert layer.constraint_weights[1] == 4.0
    
    def test_gradient_flow(self):
        """Test that gradients flow through constraint layer."""
        constraints = [MaxwellConstraint(weight=1.0)]
        layer = PhysicsConstraintLayer(constraints)
        
        batch_size = 4
        prediction = torch.randn(batch_size, 10, requires_grad=True)
        input_data = torch.randn(batch_size, 6)
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        physics_loss = layer.get_physics_loss(prediction, input_data, coords)
        physics_loss.backward()
        
        # Check that gradients exist
        assert prediction.grad is not None
        assert not torch.allclose(prediction.grad, torch.zeros_like(prediction.grad))


class TestConstraintIntegration:
    """Test integration between different constraints."""
    
    def test_multi_constraint_consistency(self):
        """Test that multiple constraints work together consistently."""
        constraints = [
            MaxwellConstraint(weight=1.0),
            HeatEquationConstraint(weight=1.0),
            StructuralDynamicsConstraint(weight=1.0)
        ]
        
        layer = PhysicsConstraintLayer(constraints)
        
        batch_size = 4
        prediction = torch.randn(batch_size, 12)  # Enough dimensions for all constraints
        input_data = torch.randn(batch_size, 8)   # Enough dimensions for all constraints
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        # Multiple forward passes should be consistent
        _, residuals1 = layer.forward(prediction, input_data, coords)
        _, residuals2 = layer.forward(prediction, input_data, coords)
        
        for key in residuals1:
            assert torch.allclose(residuals1[key], residuals2[key])
    
    def test_constraint_scaling(self):
        """Test that constraint weights properly scale residuals."""
        constraint1 = MaxwellConstraint(weight=1.0)
        constraint2 = MaxwellConstraint(weight=2.0)
        
        layer1 = PhysicsConstraintLayer([constraint1])
        layer2 = PhysicsConstraintLayer([constraint2])
        
        batch_size = 4
        prediction = torch.randn(batch_size, 10)
        input_data = torch.randn(batch_size, 6)
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        loss1 = layer1.get_physics_loss(prediction, input_data, coords)
        loss2 = layer2.get_physics_loss(prediction, input_data, coords)
        
        # Loss2 should be approximately 2x loss1 due to weight scaling
        assert torch.allclose(loss2, 2.0 * loss1, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])