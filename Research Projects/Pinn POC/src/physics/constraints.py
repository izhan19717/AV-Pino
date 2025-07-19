"""
Physics constraint implementations for neural operator training.

Implements PDE constraints for electromagnetic, thermal, and mechanical physics
to enforce physical consistency during neural operator training.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np


class PDEConstraint(ABC):
    """Base class for physics constraints in neural operators."""
    
    def __init__(self, weight: float = 1.0, name: str = "constraint"):
        self.weight = weight
        self.name = name
    
    @abstractmethod
    def compute_residual(self, prediction: torch.Tensor, input_data: torch.Tensor, 
                        coords: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual for the constraint.
        
        Args:
            prediction: Neural operator prediction
            input_data: Input data to the operator
            coords: Spatial/temporal coordinates
            
        Returns:
            PDE residual tensor
        """
        pass
    
    def get_constraint_weight(self) -> float:
        """Get the weight for this constraint in the loss function."""
        return self.weight
    
    def set_constraint_weight(self, weight: float):
        """Set the weight for this constraint."""
        self.weight = weight


class MaxwellConstraint(PDEConstraint):
    """Electromagnetic field constraints based on Maxwell's equations."""
    
    def __init__(self, weight: float = 1.0, mu_0: float = 4e-7 * np.pi, 
                 epsilon_0: float = 8.854e-12):
        super().__init__(weight, "maxwell")
        self.mu_0 = mu_0  # Permeability of free space
        self.epsilon_0 = epsilon_0  # Permittivity of free space
    
    def compute_residual(self, prediction: torch.Tensor, input_data: torch.Tensor,
                        coords: torch.Tensor) -> torch.Tensor:
        """Compute Maxwell equation residuals.
        
        Enforces:
        - ∇ × E = -∂B/∂t (Faraday's law)
        - ∇ × B = μ₀J + μ₀ε₀∂E/∂t (Ampère's law)
        - ∇ · E = ρ/ε₀ (Gauss's law)
        - ∇ · B = 0 (No magnetic monopoles)
        """
        batch_size = prediction.shape[0]
        
        # Extract E and B fields from prediction
        # Assuming prediction format: [E_x, E_y, E_z, B_x, B_y, B_z, ...]
        E_field = prediction[:, :3]  # Electric field components
        B_field = prediction[:, 3:6]  # Magnetic field components
        
        # Compute spatial derivatives using finite differences
        dx = coords[:, 1:] - coords[:, :-1]  # Spatial step
        dt = 1e-6  # Time step (assumed)
        
        # Faraday's law: ∇ × E + ∂B/∂t = 0
        curl_E = self._compute_curl(E_field, dx)
        dB_dt = self._compute_time_derivative(B_field, dt)
        faraday_residual = curl_E + dB_dt
        
        # Ampère's law: ∇ × B - μ₀ε₀∂E/∂t - μ₀J = 0
        curl_B = self._compute_curl(B_field, dx)
        dE_dt = self._compute_time_derivative(E_field, dt)
        J_current = input_data[:, :3] if input_data.shape[1] >= 3 else torch.zeros_like(E_field)
        ampere_residual = curl_B - self.mu_0 * self.epsilon_0 * dE_dt - self.mu_0 * J_current
        
        # Gauss's law: ∇ · E - ρ/ε₀ = 0
        div_E = self._compute_divergence(E_field, dx)
        rho_charge = torch.zeros_like(div_E)  # Assume no free charges
        gauss_E_residual = div_E - rho_charge / self.epsilon_0
        
        # No magnetic monopoles: ∇ · B = 0
        div_B = self._compute_divergence(B_field, dx)
        gauss_B_residual = div_B
        
        # Combine all residuals
        total_residual = (torch.mean(faraday_residual**2) + 
                         torch.mean(ampere_residual**2) +
                         torch.mean(gauss_E_residual**2) + 
                         torch.mean(gauss_B_residual**2))
        
        return total_residual
    
    def _compute_curl(self, field: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """Compute curl of a 3D vector field using finite differences."""
        # Simplified 1D curl approximation for demonstration
        # In practice, would need proper 3D finite difference stencils
        curl = torch.zeros_like(field)
        if field.shape[1] >= 2:
            curl[:, 0] = torch.gradient(field[:, 2], dim=0)[0] - torch.gradient(field[:, 1], dim=0)[0]
            curl[:, 1] = torch.gradient(field[:, 0], dim=0)[0] - torch.gradient(field[:, 2], dim=0)[0]
            curl[:, 2] = torch.gradient(field[:, 1], dim=0)[0] - torch.gradient(field[:, 0], dim=0)[0]
        return curl
    
    def _compute_divergence(self, field: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """Compute divergence of a 3D vector field."""
        # Simplified divergence computation
        div = torch.sum(torch.gradient(field, dim=0)[0], dim=1)
        return div
    
    def _compute_time_derivative(self, field: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute time derivative using finite differences."""
        # Simplified time derivative - in practice would use proper temporal discretization
        return torch.zeros_like(field)


class HeatEquationConstraint(PDEConstraint):
    """Thermal dynamics constraints based on heat equation."""
    
    def __init__(self, weight: float = 1.0, thermal_diffusivity: float = 1e-5):
        super().__init__(weight, "heat_equation")
        self.alpha = thermal_diffusivity
    
    def compute_residual(self, prediction: torch.Tensor, input_data: torch.Tensor,
                        coords: torch.Tensor) -> torch.Tensor:
        """Compute heat equation residual.
        
        Enforces: ∂T/∂t = α∇²T + Q/ρc
        where T is temperature, α is thermal diffusivity, Q is heat source
        """
        # Extract temperature from prediction
        temperature = prediction[:, 6:7] if prediction.shape[1] > 6 else prediction[:, 0:1]
        
        # Compute spatial derivatives
        dx = coords[:, 1:] - coords[:, :-1] if coords.shape[1] > 1 else torch.ones(1)
        
        # Time derivative ∂T/∂t
        dT_dt = self._compute_time_derivative(temperature)
        
        # Laplacian ∇²T
        laplacian_T = self._compute_laplacian(temperature, dx)
        
        # Heat source term
        heat_source = input_data[:, -1:] if input_data.shape[1] > 0 else torch.zeros_like(temperature)
        
        # Heat equation residual: ∂T/∂t - α∇²T - Q = 0
        residual = dT_dt - self.alpha * laplacian_T - heat_source
        
        return torch.mean(residual**2)
    
    def _compute_time_derivative(self, field: torch.Tensor) -> torch.Tensor:
        """Compute time derivative using finite differences."""
        # Simplified - would use proper temporal discretization in practice
        return torch.zeros_like(field)
    
    def _compute_laplacian(self, field: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using finite differences."""
        # Simplified 1D Laplacian
        if field.shape[0] > 2:
            laplacian = (field[2:] - 2*field[1:-1] + field[:-2]) / (dx[0]**2)
            # Pad to maintain shape
            laplacian = torch.cat([laplacian[:1], laplacian, laplacian[-1:]], dim=0)
        else:
            laplacian = torch.zeros_like(field)
        return laplacian


class StructuralDynamicsConstraint(PDEConstraint):
    """Mechanical vibration constraints based on structural dynamics."""
    
    def __init__(self, weight: float = 1.0, density: float = 7850.0, 
                 youngs_modulus: float = 2e11):
        super().__init__(weight, "structural_dynamics")
        self.rho = density  # Material density
        self.E = youngs_modulus  # Young's modulus
    
    def compute_residual(self, prediction: torch.Tensor, input_data: torch.Tensor,
                        coords: torch.Tensor) -> torch.Tensor:
        """Compute structural dynamics residual.
        
        Enforces: ρ∂²u/∂t² = ∇·σ + f
        where u is displacement, σ is stress tensor, f is body force
        """
        # Extract displacement from prediction
        displacement = prediction[:, 7:10] if prediction.shape[1] > 9 else prediction[:, :3]
        
        # Compute acceleration ∂²u/∂t²
        acceleration = self._compute_acceleration(displacement)
        
        # Compute stress divergence ∇·σ
        stress_div = self._compute_stress_divergence(displacement, coords)
        
        # Body forces
        body_force = input_data[:, 3:6] if input_data.shape[1] >= 6 else torch.zeros_like(displacement)
        
        # Equation of motion residual: ρ∂²u/∂t² - ∇·σ - f = 0
        residual = self.rho * acceleration - stress_div - body_force
        
        return torch.mean(residual**2)
    
    def _compute_acceleration(self, displacement: torch.Tensor) -> torch.Tensor:
        """Compute acceleration from displacement."""
        # Simplified - would use proper temporal discretization
        return torch.zeros_like(displacement)
    
    def _compute_stress_divergence(self, displacement: torch.Tensor, 
                                  coords: torch.Tensor) -> torch.Tensor:
        """Compute divergence of stress tensor."""
        # Simplified linear elasticity: σ = E∇u (for small deformations)
        dx = coords[:, 1:] - coords[:, :-1] if coords.shape[1] > 1 else torch.ones(1)
        
        # Compute strain (gradient of displacement)
        strain = torch.gradient(displacement, dim=0)[0]
        
        # Stress (simplified)
        stress = self.E * strain
        
        # Stress divergence
        stress_div = torch.gradient(stress, dim=0)[0]
        
        return stress_div


class PhysicsConstraintLayer(nn.Module):
    """Neural network layer that enforces physics constraints during forward pass."""
    
    def __init__(self, constraints: list[PDEConstraint]):
        super().__init__()
        self.constraints = constraints
        self.constraint_weights = nn.Parameter(
            torch.tensor([c.get_constraint_weight() for c in constraints])
        )
    
    def forward(self, prediction: torch.Tensor, input_data: torch.Tensor,
                coords: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply physics constraints and return prediction with residuals.
        
        Args:
            prediction: Neural operator prediction
            input_data: Input data
            coords: Spatial/temporal coordinates
            
        Returns:
            Tuple of (prediction, residuals_dict)
        """
        residuals = {}
        total_physics_loss = 0.0
        
        for i, constraint in enumerate(self.constraints):
            residual = constraint.compute_residual(prediction, input_data, coords)
            residuals[constraint.name] = residual
            total_physics_loss += self.constraint_weights[i] * residual
        
        # Store total physics loss for backward pass
        residuals['total_physics_loss'] = total_physics_loss
        
        return prediction, residuals
    
    def get_physics_loss(self, prediction: torch.Tensor, input_data: torch.Tensor,
                        coords: torch.Tensor) -> torch.Tensor:
        """Compute total physics loss for training."""
        _, residuals = self.forward(prediction, input_data, coords)
        return residuals['total_physics_loss']
    
    def update_constraint_weights(self, weights: Dict[str, float]):
        """Update constraint weights during training."""
        with torch.no_grad():
            for i, constraint in enumerate(self.constraints):
                if constraint.name in weights:
                    self.constraint_weights[i] = weights[constraint.name]