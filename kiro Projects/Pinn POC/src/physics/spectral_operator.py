"""
Spectral representation and operator control for AGT-NO architecture.

Implements spectral decomposition, operator control modules, and fault evolution
operators for the AV-PINO motor fault diagnosis system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Callable
import numpy as np
from abc import ABC, abstractmethod
import math


class SpectralBasis(ABC):
    """Base class for spectral basis functions."""
    
    def __init__(self, n_modes: int, domain_size: float = 1.0):
        self.n_modes = n_modes
        self.domain_size = domain_size
    
    @abstractmethod
    def compute_basis(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute basis functions at given coordinates.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, spatial_dim)
            
        Returns:
            Basis functions of shape (batch_size, n_modes)
        """
        pass
    
    @abstractmethod
    def compute_derivatives(self, coords: torch.Tensor, order: int = 1) -> torch.Tensor:
        """Compute derivatives of basis functions.
        
        Args:
            coords: Coordinate tensor
            order: Derivative order
            
        Returns:
            Derivative tensor
        """
        pass


class FourierBasis(SpectralBasis):
    """Fourier basis functions for spectral decomposition."""
    
    def __init__(self, n_modes: int, domain_size: float = 1.0):
        super().__init__(n_modes, domain_size)
        # Precompute wave numbers - use exactly n_modes
        self.k_max = n_modes // 2
        if n_modes % 2 == 0:
            # Even number of modes: [-k_max, ..., -1, 0, 1, ..., k_max-1]
            self.wave_numbers = torch.arange(-self.k_max, self.k_max, dtype=torch.float32)
        else:
            # Odd number of modes: [-k_max, ..., -1, 0, 1, ..., k_max]
            self.wave_numbers = torch.arange(-self.k_max, self.k_max + 1, dtype=torch.float32)
    
    def compute_basis(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute Fourier basis functions: φₖ(x) = exp(2πikx/L)."""
        batch_size = coords.shape[0]
        device = coords.device
        
        # Move wave numbers to same device
        k = self.wave_numbers.to(device)
        
        # Handle multi-dimensional coordinates by using only the first dimension
        if coords.dim() == 3:  # (batch_size, n_points, coord_dim)
            # Use first coordinate dimension for Fourier basis
            x_coord = coords[:, :, 0]  # (batch_size, n_points)
            # Flatten to (batch_size * n_points,) for processing
            x_norm = x_coord.flatten() / self.domain_size
        else:  # (batch_size, coord_dim) or (batch_size,)
            x_norm = coords.squeeze(-1) / self.domain_size
            if x_norm.dim() == 0:
                x_norm = x_norm.unsqueeze(0)
        
        # Compute Fourier modes: exp(2πikx)
        # Shape: (batch_size*n_points,) outer (n_modes,) -> (batch_size*n_points, n_modes)
        phases = 2 * math.pi * torch.outer(x_norm, k)
        basis_real = torch.cos(phases)
        basis_imag = torch.sin(phases)
        
        # Combine real and imaginary parts
        basis = torch.stack([basis_real, basis_imag], dim=-1)  # (batch_size*n_points, n_modes, 2)
        basis = basis.view(basis.shape[0], -1)  # (batch_size*n_points, 2*n_modes)
        
        # Reshape back to match input batch structure
        if coords.dim() == 3:
            n_points = coords.shape[1]
            basis = basis.view(batch_size, n_points, -1)
        
        return basis
    
    def compute_derivatives(self, coords: torch.Tensor, order: int = 1) -> torch.Tensor:
        """Compute derivatives of Fourier basis functions."""
        batch_size = coords.shape[0]
        device = coords.device
        
        k = self.wave_numbers.to(device)
        
        # Handle multi-dimensional coordinates by using only the first dimension
        if coords.dim() == 3:  # (batch_size, n_points, coord_dim)
            # Use first coordinate dimension for Fourier basis
            x_coord = coords[:, :, 0]  # (batch_size, n_points)
            # Flatten to (batch_size * n_points,) for processing
            x_norm = x_coord.flatten() / self.domain_size
        else:  # (batch_size, coord_dim) or (batch_size,)
            x_norm = coords.squeeze(-1) / self.domain_size
            if x_norm.dim() == 0:
                x_norm = x_norm.unsqueeze(0)
        
        # Derivative factor: (2πik/L)^order
        deriv_factor = (2 * math.pi * k / self.domain_size) ** order
        
        phases = 2 * math.pi * torch.outer(x_norm, k)
        
        if order % 2 == 0:
            # Even derivatives
            basis_real = torch.cos(phases) * deriv_factor.unsqueeze(0)
            basis_imag = torch.sin(phases) * deriv_factor.unsqueeze(0)
        else:
            # Odd derivatives
            basis_real = -torch.sin(phases) * deriv_factor.unsqueeze(0)
            basis_imag = torch.cos(phases) * deriv_factor.unsqueeze(0)
        
        basis = torch.stack([basis_real, basis_imag], dim=-1)
        basis = basis.view(basis.shape[0], -1)
        
        # Reshape back to match input batch structure
        if coords.dim() == 3:
            n_points = coords.shape[1]
            basis = basis.view(batch_size, n_points, -1)
        
        return basis


class ChebyshevBasis(SpectralBasis):
    """Chebyshev polynomial basis functions."""
    
    def __init__(self, n_modes: int, domain_size: float = 1.0):
        super().__init__(n_modes, domain_size)
    
    def compute_basis(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute Chebyshev basis functions: Tₙ(x)."""
        batch_size = coords.shape[0]
        device = coords.device
        
        # Normalize coordinates to [-1, 1] for Chebyshev polynomials
        x_norm = 2 * coords / self.domain_size - 1
        x_norm = torch.clamp(x_norm, -1, 1)
        
        # Initialize basis tensor
        basis = torch.zeros(batch_size, self.n_modes, device=device)
        
        if self.n_modes > 0:
            basis[:, 0] = 1.0  # T₀(x) = 1
        if self.n_modes > 1:
            basis[:, 1] = x_norm.squeeze()  # T₁(x) = x
        
        # Recurrence relation: Tₙ₊₁(x) = 2xTₙ(x) - Tₙ₋₁(x)
        for n in range(2, self.n_modes):
            basis[:, n] = 2 * x_norm.squeeze() * basis[:, n-1] - basis[:, n-2]
        
        return basis
    
    def compute_derivatives(self, coords: torch.Tensor, order: int = 1) -> torch.Tensor:
        """Compute derivatives of Chebyshev polynomials."""
        # Simplified implementation - would need proper Chebyshev derivative formulas
        batch_size = coords.shape[0]
        device = coords.device
        
        # For now, return finite difference approximation
        h = 1e-6
        coords_plus = coords + h
        coords_minus = coords - h
        
        basis_plus = self.compute_basis(coords_plus)
        basis_minus = self.compute_basis(coords_minus)
        
        derivatives = (basis_plus - basis_minus) / (2 * h)
        
        return derivatives


class SpectralDecomposition(nn.Module):
    """Spectral decomposition module: v(x) = Σₖ vₖ φₖ(x)."""
    
    def __init__(self, basis: SpectralBasis, n_channels: int = 1):
        super().__init__()
        self.basis = basis
        self.n_channels = n_channels
        self.n_modes = basis.n_modes
        
        # Learnable spectral coefficients
        if isinstance(basis, FourierBasis):
            # For Fourier basis, we have real and imaginary parts
            # The actual size depends on the number of wave numbers
            coeff_size = 2 * len(basis.wave_numbers)
        else:
            coeff_size = self.n_modes
            
        self.spectral_coeffs = nn.Parameter(
            torch.randn(n_channels, coeff_size) * 0.1
        )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Reconstruct function from spectral coefficients.
        
        Args:
            coords: Spatial coordinates of shape (batch_size, spatial_dim)
            
        Returns:
            Reconstructed function of shape (batch_size, n_channels)
        """
        # Compute basis functions
        basis_values = self.basis.compute_basis(coords)  # (batch_size, n_modes)
        
        # Reconstruct: v(x) = Σₖ vₖ φₖ(x)
        # (batch_size, n_modes) @ (n_modes, n_channels) -> (batch_size, n_channels)
        reconstructed = torch.matmul(basis_values, self.spectral_coeffs.T)
        
        return reconstructed
    
    def compute_derivatives(self, coords: torch.Tensor, order: int = 1) -> torch.Tensor:
        """Compute derivatives of the reconstructed function."""
        # Compute basis derivatives
        basis_derivs = self.basis.compute_derivatives(coords, order)
        
        # Reconstruct derivatives
        deriv_reconstructed = torch.matmul(basis_derivs, self.spectral_coeffs.T)
        
        return deriv_reconstructed
    
    def get_spectral_coefficients(self) -> torch.Tensor:
        """Get current spectral coefficients."""
        return self.spectral_coeffs.clone()
    
    def set_spectral_coefficients(self, coeffs: torch.Tensor):
        """Set spectral coefficients."""
        with torch.no_grad():
            self.spectral_coeffs.copy_(coeffs)


class OperatorControl(nn.Module):
    """Operator control module: χ_G: Parameters → Gain Kernel with stability constraints."""
    
    def __init__(self, input_dim: int, output_dim: int, n_modes: int,
                 stability_constraint: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.stability_constraint = stability_constraint
        
        # Parameter-to-gain mapping network
        self.param_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_modes * output_dim)
        )
        
        # Stability constraint parameters
        if stability_constraint:
            self.stability_weight = nn.Parameter(torch.tensor(1.0))
            self.max_eigenvalue = nn.Parameter(torch.tensor(1.0))  # Ensure stability
    
    def forward(self, parameters: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map parameters to gain kernel with stability constraints.
        
        Args:
            parameters: Input parameters of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (gain_kernel, stability_loss)
        """
        batch_size = parameters.shape[0]
        
        # Encode parameters to gain values
        gain_values = self.param_encoder(parameters)  # (batch_size, n_modes * output_dim)
        gain_kernel = gain_values.view(batch_size, self.output_dim, self.n_modes)
        
        stability_loss = torch.tensor(0.0, device=parameters.device)
        
        if self.stability_constraint:
            # Compute stability constraint
            stability_loss = self._compute_stability_constraint(gain_kernel)
        
        return gain_kernel, stability_loss
    
    def _compute_stability_constraint(self, gain_kernel: torch.Tensor) -> torch.Tensor:
        """Compute stability constraint for the gain kernel."""
        # Simplified stability constraint: limit maximum gain
        max_gain = torch.max(torch.abs(gain_kernel), dim=-1)[0]  # (batch_size, output_dim)
        
        # Penalty for gains exceeding stability threshold
        stability_violation = F.relu(max_gain - self.max_eigenvalue)
        stability_loss = self.stability_weight * torch.mean(stability_violation**2)
        
        return stability_loss
    
    def get_stability_metrics(self, gain_kernel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get stability metrics for analysis."""
        max_gain = torch.max(torch.abs(gain_kernel), dim=-1)[0]
        mean_gain = torch.mean(torch.abs(gain_kernel), dim=-1)
        
        return {
            'max_gain': torch.max(max_gain),
            'mean_gain': torch.mean(mean_gain),
            'stability_margin': self.max_eigenvalue - torch.max(max_gain)
        }


class FaultEvolutionOperator(nn.Module):
    """Fault evolution operator: ∂α/∂t = F[α, σ, T] + W(t)."""
    
    def __init__(self, state_dim: int, control_dim: int, noise_dim: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.noise_dim = noise_dim
        
        # Nonlinear dynamics function F[α, σ, T]
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim + control_dim + 1, 64),  # +1 for temperature
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, state_dim)
        )
        
        # Noise model parameters
        self.noise_scale = nn.Parameter(torch.tensor(0.1))
        
        # Physics-based constraints
        self.degradation_rate = nn.Parameter(torch.tensor(0.01))
        self.recovery_rate = nn.Parameter(torch.tensor(0.001))
    
    def forward(self, fault_state: torch.Tensor, control_input: torch.Tensor,
                temperature: torch.Tensor, dt: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fault evolution over time step dt.
        
        Args:
            fault_state: Current fault state α of shape (batch_size, state_dim)
            control_input: Control input σ of shape (batch_size, control_dim)
            temperature: Temperature T of shape (batch_size, 1)
            dt: Time step
            
        Returns:
            Tuple of (new_fault_state, evolution_loss)
        """
        batch_size = fault_state.shape[0]
        device = fault_state.device
        
        # Concatenate inputs for dynamics function
        dynamics_input = torch.cat([fault_state, control_input, temperature], dim=1)
        
        # Compute dynamics: F[α, σ, T]
        dynamics = self.dynamics_net(dynamics_input)
        
        # Add physics-based constraints
        physics_dynamics = self._compute_physics_dynamics(fault_state, temperature)
        total_dynamics = dynamics + physics_dynamics
        
        # Add noise term: W(t)
        noise = self.noise_scale * torch.randn(batch_size, self.state_dim, device=device)
        
        # Euler integration: α(t+dt) = α(t) + dt * (F[α, σ, T] + W(t))
        new_fault_state = fault_state + dt * (total_dynamics + noise)
        
        # Ensure fault state remains in valid range [0, 1]
        new_fault_state = torch.clamp(new_fault_state, 0.0, 1.0)
        
        # Compute evolution loss (physics consistency)
        evolution_loss = self._compute_evolution_loss(fault_state, new_fault_state, 
                                                     total_dynamics, dt)
        
        return new_fault_state, evolution_loss
    
    def _compute_physics_dynamics(self, fault_state: torch.Tensor, 
                                 temperature: torch.Tensor) -> torch.Tensor:
        """Compute physics-based fault dynamics."""
        # Temperature-dependent degradation
        temp_factor = torch.exp((temperature - 293.15) / 50.0)  # Arrhenius-like
        degradation = self.degradation_rate * temp_factor * (1 - fault_state)
        
        # Natural recovery (very slow)
        recovery = -self.recovery_rate * fault_state
        
        physics_dynamics = degradation + recovery
        
        return physics_dynamics
    
    def _compute_evolution_loss(self, old_state: torch.Tensor, new_state: torch.Tensor,
                               dynamics: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute physics consistency loss for fault evolution."""
        # Consistency with dynamics
        predicted_change = dt * dynamics
        actual_change = new_state - old_state
        
        consistency_loss = torch.mean((actual_change - predicted_change)**2)
        
        # Monotonicity constraint (faults generally don't heal quickly)
        healing_penalty = F.relu(old_state - new_state - 0.01 * dt)  # Allow small healing
        monotonicity_loss = torch.mean(healing_penalty**2)
        
        total_loss = consistency_loss + 0.1 * monotonicity_loss
        
        return total_loss
    
    def predict_fault_trajectory(self, initial_state: torch.Tensor,
                                control_sequence: torch.Tensor,
                                temperature_sequence: torch.Tensor,
                                n_steps: int, dt: float = 1e-3) -> torch.Tensor:
        """Predict fault evolution trajectory over multiple time steps."""
        trajectory = [initial_state]
        current_state = initial_state
        
        for i in range(n_steps):
            control_input = control_sequence[:, i] if control_sequence.dim() > 1 else control_sequence
            temperature = temperature_sequence[:, i] if temperature_sequence.dim() > 1 else temperature_sequence
            
            current_state, _ = self.forward(current_state, control_input, temperature, dt)
            trajectory.append(current_state)
        
        return torch.stack(trajectory, dim=1)  # (batch_size, n_steps+1, state_dim)


class SpectralOperatorLayer(nn.Module):
    """Combined spectral representation and operator control layer."""
    
    def __init__(self, basis: SpectralBasis, input_dim: int, output_dim: int,
                 n_channels: int = 1, use_operator_control: bool = True):
        super().__init__()
        self.spectral_decomp = SpectralDecomposition(basis, n_channels)
        self.use_operator_control = use_operator_control
        
        if use_operator_control:
            self.operator_control = OperatorControl(input_dim, output_dim, basis.n_modes)
        
        # Projection layer to combine spectral and control information
        self.projection = nn.Linear(n_channels + (output_dim if use_operator_control else 0), 
                                   output_dim)
    
    def forward(self, coords: torch.Tensor, parameters: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass combining spectral decomposition and operator control.
        
        Args:
            coords: Spatial coordinates
            parameters: Control parameters (optional)
            
        Returns:
            Tuple of (output, losses_dict)
        """
        # Spectral reconstruction
        spectral_output = self.spectral_decomp(coords)
        
        losses = {}
        
        if self.use_operator_control and parameters is not None:
            # Operator control
            gain_kernel, stability_loss = self.operator_control(parameters)
            losses['stability_loss'] = stability_loss
            
            # Apply gain to spectral output - simplified approach
            # In practice, this would involve more sophisticated operator application
            batch_size, n_points = coords.shape[:2]
            
            # Create controlled output with proper shape
            if len(spectral_output.shape) == 3:  # (batch, n_points, features)
                controlled_features = torch.zeros(batch_size, n_points, 
                                                self.projection.in_features - spectral_output.shape[-1],
                                                device=coords.device)
                combined_input = torch.cat([spectral_output, controlled_features], dim=-1)
            else:  # (batch, features)
                controlled_features = torch.zeros(batch_size, 
                                                self.projection.in_features - spectral_output.shape[-1],
                                                device=coords.device)
                combined_input = torch.cat([spectral_output, controlled_features], dim=-1)
        else:
            combined_input = spectral_output
        
        # Final projection
        output = self.projection(combined_input)
        
        return output, losses


def create_fourier_spectral_layer(n_modes: int, input_dim: int, output_dim: int,
                                 n_channels: int = 1, domain_size: float = 1.0
                                ) -> SpectralOperatorLayer:
    """Create a spectral operator layer with Fourier basis."""
    basis = FourierBasis(n_modes, domain_size)
    return SpectralOperatorLayer(basis, input_dim, output_dim, n_channels)


def create_chebyshev_spectral_layer(n_modes: int, input_dim: int, output_dim: int,
                                   n_channels: int = 1, domain_size: float = 1.0
                                  ) -> SpectralOperatorLayer:
    """Create a spectral operator layer with Chebyshev basis."""
    basis = ChebyshevBasis(n_modes, domain_size)
    return SpectralOperatorLayer(basis, input_dim, output_dim, n_channels)