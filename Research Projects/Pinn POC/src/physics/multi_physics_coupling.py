"""
Multi-physics coupling implementation for electromagnetic-thermal-mechanical interactions.

Implements coupling mechanisms between different physics domains and energy conservation
constraints for the AV-PINO motor fault diagnosis system.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import numpy as np
from abc import ABC, abstractmethod


class CouplingTerm(ABC):
    """Base class for multi-physics coupling terms."""
    
    def __init__(self, weight: float = 1.0, name: str = "coupling"):
        self.weight = weight
        self.name = name
    
    @abstractmethod
    def compute_coupling(self, fields: Dict[str, torch.Tensor], 
                        coords: torch.Tensor) -> torch.Tensor:
        """Compute coupling term between physics domains.
        
        Args:
            fields: Dictionary of field variables from different physics domains
            coords: Spatial/temporal coordinates
            
        Returns:
            Coupling term tensor
        """
        pass
    
    def get_coupling_weight(self) -> float:
        """Get the weight for this coupling term."""
        return self.weight


class ElectromagneticThermalCoupling(CouplingTerm):
    """Coupling between electromagnetic and thermal domains."""
    
    def __init__(self, weight: float = 1.0, electrical_conductivity: float = 5.8e7):
        super().__init__(weight, "em_thermal")
        self.sigma = electrical_conductivity  # Electrical conductivity
    
    def compute_coupling(self, fields: Dict[str, torch.Tensor], 
                        coords: torch.Tensor) -> torch.Tensor:
        """Compute electromagnetic-thermal coupling.
        
        Implements Joule heating: Q = σ|E|² + ω|B|²/μ
        where Q is heat source, σ is conductivity, E is electric field, B is magnetic field
        """
        E_field = fields.get('electric_field', torch.zeros(coords.shape[0], 3))
        B_field = fields.get('magnetic_field', torch.zeros(coords.shape[0], 3))
        
        # Joule heating from electric field
        joule_heating_E = self.sigma * torch.sum(E_field**2, dim=1, keepdim=True)
        
        # Magnetic heating (eddy currents)
        mu_0 = 4e-7 * np.pi  # Permeability of free space
        omega = 2 * np.pi * 60  # Assume 60 Hz frequency
        magnetic_heating = omega * torch.sum(B_field**2, dim=1, keepdim=True) / mu_0
        
        # Total heat generation
        total_heating = joule_heating_E + magnetic_heating
        
        return total_heating
    
    def compute_temperature_dependent_conductivity(self, temperature: torch.Tensor,
                                                  T_ref: float = 293.15,
                                                  alpha_temp: float = 0.004) -> torch.Tensor:
        """Compute temperature-dependent electrical conductivity.
        
        σ(T) = σ₀ / (1 + α(T - T_ref))
        """
        conductivity = self.sigma / (1 + alpha_temp * (temperature - T_ref))
        return conductivity


class ThermalMechanicalCoupling(CouplingTerm):
    """Coupling between thermal and mechanical domains."""
    
    def __init__(self, weight: float = 1.0, thermal_expansion_coeff: float = 12e-6,
                 reference_temperature: float = 293.15):
        super().__init__(weight, "thermal_mechanical")
        self.alpha_thermal = thermal_expansion_coeff
        self.T_ref = reference_temperature
    
    def compute_coupling(self, fields: Dict[str, torch.Tensor], 
                        coords: torch.Tensor) -> torch.Tensor:
        """Compute thermal-mechanical coupling.
        
        Implements thermal stress: σ_thermal = E * α * ΔT
        where E is Young's modulus, α is thermal expansion coefficient, ΔT is temperature change
        """
        temperature = fields.get('temperature', torch.full((coords.shape[0], 1), self.T_ref))
        youngs_modulus = fields.get('youngs_modulus', torch.full_like(temperature, 2e11))
        
        # Temperature difference from reference
        delta_T = temperature - self.T_ref
        
        # Thermal stress (simplified isotropic case)
        thermal_stress = youngs_modulus * self.alpha_thermal * delta_T
        
        # Thermal strain
        thermal_strain = self.alpha_thermal * delta_T
        
        return thermal_stress
    
    def compute_thermal_strain(self, temperature: torch.Tensor) -> torch.Tensor:
        """Compute thermal strain from temperature field."""
        delta_T = temperature - self.T_ref
        thermal_strain = self.alpha_thermal * delta_T
        return thermal_strain


class MechanicalElectromagneticCoupling(CouplingTerm):
    """Coupling between mechanical and electromagnetic domains."""
    
    def __init__(self, weight: float = 1.0, magnetostrictive_coeff: float = 1e-5):
        super().__init__(weight, "mechanical_em")
        self.lambda_s = magnetostrictive_coeff  # Magnetostrictive coefficient
    
    def compute_coupling(self, fields: Dict[str, torch.Tensor], 
                        coords: torch.Tensor) -> torch.Tensor:
        """Compute mechanical-electromagnetic coupling.
        
        Implements magnetostriction and inverse magnetostriction effects
        """
        B_field = fields.get('magnetic_field', torch.zeros(coords.shape[0], 3))
        displacement = fields.get('displacement', torch.zeros(coords.shape[0], 3))
        
        # Magnetostrictive strain: ε_ms = λ_s * (B·B)/|B|²
        B_magnitude_sq = torch.sum(B_field**2, dim=1, keepdim=True)
        B_magnitude_sq = torch.clamp(B_magnitude_sq, min=1e-12)  # Avoid division by zero
        
        # Simplified magnetostrictive strain
        magnetostrictive_strain = self.lambda_s * B_magnitude_sq
        
        # Inverse effect: stress-induced magnetic field changes
        stress_induced_B_change = self.lambda_s * torch.sum(displacement**2, dim=1, keepdim=True)
        
        return magnetostrictive_strain + stress_induced_B_change


class EnergyConservationConstraint:
    """Energy conservation constraint for multi-physics coupling."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def compute_energy_residual(self, fields: Dict[str, torch.Tensor],
                               coords: torch.Tensor, dt: float = 1e-6) -> torch.Tensor:
        """Compute energy conservation residual.
        
        Enforces: dE_total/dt = P_input - P_dissipated
        where E_total is total energy, P_input is input power, P_dissipated is dissipated power
        """
        # Extract field variables
        E_field = fields.get('electric_field', torch.zeros(coords.shape[0], 3))
        B_field = fields.get('magnetic_field', torch.zeros(coords.shape[0], 3))
        temperature = fields.get('temperature', torch.full((coords.shape[0], 1), 293.15))
        displacement = fields.get('displacement', torch.zeros(coords.shape[0], 3))
        velocity = fields.get('velocity', torch.zeros(coords.shape[0], 3))
        
        # Electromagnetic energy density
        epsilon_0 = 8.854e-12
        mu_0 = 4e-7 * np.pi
        em_energy = 0.5 * (epsilon_0 * torch.sum(E_field**2, dim=1) + 
                          torch.sum(B_field**2, dim=1) / mu_0)
        
        # Thermal energy density (simplified)
        rho = 7850.0  # Material density
        c_p = 460.0   # Specific heat capacity
        thermal_energy = rho * c_p * (temperature.squeeze() - 293.15)
        
        # Mechanical energy density
        mechanical_energy = 0.5 * rho * torch.sum(velocity**2, dim=1)
        
        # Total energy density
        total_energy = em_energy + thermal_energy + mechanical_energy
        
        # Energy conservation residual (simplified - would need proper time derivatives)
        # For now, just ensure energy is bounded
        energy_residual = torch.mean((total_energy - torch.mean(total_energy))**2)
        
        return energy_residual


class MultiPhysicsCoupling(nn.Module):
    """Multi-physics coupling module for electromagnetic-thermal-mechanical interactions."""
    
    def __init__(self, coupling_terms: List[CouplingTerm], 
                 energy_conservation: bool = True):
        super().__init__()
        self.coupling_terms = coupling_terms
        self.use_energy_conservation = energy_conservation
        
        if energy_conservation:
            self.energy_constraint = EnergyConservationConstraint()
        
        # Learnable coupling weights
        self.coupling_weights = nn.Parameter(
            torch.tensor([term.get_coupling_weight() for term in coupling_terms])
        )
    
    def forward(self, fields: Dict[str, torch.Tensor], 
                coords: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Apply multi-physics coupling and return modified fields and coupling loss.
        
        Args:
            fields: Dictionary of field variables from different physics domains
            coords: Spatial/temporal coordinates
            
        Returns:
            Tuple of (modified_fields, coupling_loss)
        """
        modified_fields = fields.copy()
        total_coupling_loss = 0.0
        coupling_contributions = {}
        
        # Apply each coupling term
        for i, coupling_term in enumerate(self.coupling_terms):
            coupling_contribution = coupling_term.compute_coupling(fields, coords)
            coupling_contributions[coupling_term.name] = coupling_contribution
            
            # Add coupling contribution to appropriate field
            if coupling_term.name == "em_thermal":
                # Add heat source to temperature field
                if 'heat_source' in modified_fields:
                    modified_fields['heat_source'] += self.coupling_weights[i] * coupling_contribution
                else:
                    modified_fields['heat_source'] = self.coupling_weights[i] * coupling_contribution
                    
            elif coupling_term.name == "thermal_mechanical":
                # Add thermal stress to mechanical field
                if 'thermal_stress' in modified_fields:
                    modified_fields['thermal_stress'] += self.coupling_weights[i] * coupling_contribution
                else:
                    modified_fields['thermal_stress'] = self.coupling_weights[i] * coupling_contribution
                    
            elif coupling_term.name == "mechanical_em":
                # Add magnetostrictive effects
                if 'magnetostrictive_strain' in modified_fields:
                    modified_fields['magnetostrictive_strain'] += self.coupling_weights[i] * coupling_contribution
                else:
                    modified_fields['magnetostrictive_strain'] = self.coupling_weights[i] * coupling_contribution
            
            # Accumulate coupling loss
            coupling_loss = torch.mean(coupling_contribution**2)
            total_coupling_loss += self.coupling_weights[i] * coupling_loss
        
        # Apply energy conservation constraint
        if self.use_energy_conservation:
            energy_residual = self.energy_constraint.compute_energy_residual(modified_fields, coords)
            total_coupling_loss += self.energy_constraint.weight * energy_residual
            coupling_contributions['energy_conservation'] = energy_residual
        
        # Store coupling contributions in modified fields for analysis
        modified_fields['coupling_contributions'] = coupling_contributions
        
        return modified_fields, total_coupling_loss
    
    def get_coupling_loss(self, fields: Dict[str, torch.Tensor], 
                         coords: torch.Tensor) -> torch.Tensor:
        """Compute total coupling loss for training."""
        _, coupling_loss = self.forward(fields, coords)
        return coupling_loss
    
    def update_coupling_weights(self, weights: Dict[str, float]):
        """Update coupling weights during training."""
        with torch.no_grad():
            for i, coupling_term in enumerate(self.coupling_terms):
                if coupling_term.name in weights:
                    self.coupling_weights[i] = weights[coupling_term.name]
    
    def get_coupling_contributions(self, fields: Dict[str, torch.Tensor],
                                  coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get individual coupling contributions for analysis."""
        modified_fields, _ = self.forward(fields, coords)
        return modified_fields.get('coupling_contributions', {})


class AdaptiveCouplingWeights(nn.Module):
    """Adaptive coupling weight adjustment based on field magnitudes."""
    
    def __init__(self, initial_weights: Dict[str, float], 
                 adaptation_rate: float = 0.01):
        super().__init__()
        self.adaptation_rate = adaptation_rate
        self.weight_names = list(initial_weights.keys())
        
        # Initialize weights as learnable parameters
        initial_values = torch.tensor([initial_weights[name] for name in self.weight_names])
        self.log_weights = nn.Parameter(torch.log(initial_values))
    
    def forward(self, field_magnitudes: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute adaptive coupling weights based on field magnitudes."""
        weights = torch.exp(self.log_weights)
        
        # Normalize weights based on field magnitudes
        normalized_weights = {}
        for i, name in enumerate(self.weight_names):
            if name in field_magnitudes:
                # Scale weight inversely with field magnitude to maintain balance
                field_mag = torch.clamp(field_magnitudes[name], min=1e-12)
                normalized_weights[name] = weights[i] / torch.sqrt(field_mag)
            else:
                normalized_weights[name] = weights[i]
        
        return {name: float(weight) for name, weight in normalized_weights.items()}
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current coupling weights."""
        weights = torch.exp(self.log_weights)
        return {name: float(weights[i]) for i, name in enumerate(self.weight_names)}


def create_motor_coupling_system(em_thermal_weight: float = 1.0,
                                thermal_mechanical_weight: float = 1.0,
                                mechanical_em_weight: float = 0.5,
                                energy_conservation: bool = True) -> MultiPhysicsCoupling:
    """Create a complete multi-physics coupling system for motor applications.
    
    Args:
        em_thermal_weight: Weight for electromagnetic-thermal coupling
        thermal_mechanical_weight: Weight for thermal-mechanical coupling
        mechanical_em_weight: Weight for mechanical-electromagnetic coupling
        energy_conservation: Whether to include energy conservation constraint
        
    Returns:
        Configured MultiPhysicsCoupling module
    """
    coupling_terms = [
        ElectromagneticThermalCoupling(weight=em_thermal_weight),
        ThermalMechanicalCoupling(weight=thermal_mechanical_weight),
        MechanicalElectromagneticCoupling(weight=mechanical_em_weight)
    ]
    
    return MultiPhysicsCoupling(coupling_terms, energy_conservation)