"""
Unit tests for multi-physics coupling implementations.

Tests coupling mechanisms between electromagnetic, thermal, and mechanical domains
and energy conservation constraints.
"""

import torch
import pytest
import numpy as np
from src.physics.multi_physics_coupling import (
    CouplingTerm, ElectromagneticThermalCoupling, ThermalMechanicalCoupling,
    MechanicalElectromagneticCoupling, EnergyConservationConstraint,
    MultiPhysicsCoupling, AdaptiveCouplingWeights, create_motor_coupling_system
)


class TestCouplingTerm:
    """Test base CouplingTerm class."""
    
    def test_coupling_term_initialization(self):
        """Test coupling term initialization."""
        class DummyCoupling(CouplingTerm):
            def compute_coupling(self, fields, coords):
                return torch.tensor(0.0)
        
        coupling = DummyCoupling(weight=2.5, name="dummy")
        assert coupling.get_coupling_weight() == 2.5
        assert coupling.name == "dummy"


class TestElectromagneticThermalCoupling:
    """Test electromagnetic-thermal coupling."""
    
    def test_em_thermal_initialization(self):
        """Test electromagnetic-thermal coupling initialization."""
        coupling = ElectromagneticThermalCoupling(weight=1.5, electrical_conductivity=6e7)
        assert coupling.get_coupling_weight() == 1.5
        assert coupling.name == "em_thermal"
        assert coupling.sigma == 6e7
    
    def test_joule_heating_computation(self):
        """Test Joule heating computation."""
        coupling = ElectromagneticThermalCoupling()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        # Create test fields
        fields = {
            'electric_field': torch.randn(batch_size, 3),
            'magnetic_field': torch.randn(batch_size, 3)
        }
        
        heating = coupling.compute_coupling(fields, coords)
        
        assert heating.shape == (batch_size, 1)
        assert torch.all(heating >= 0)  # Heating should be non-negative
    
    def test_temperature_dependent_conductivity(self):
        """Test temperature-dependent conductivity computation."""
        coupling = ElectromagneticThermalCoupling()
        
        # Test at reference temperature
        T_ref = torch.tensor([[293.15]])
        conductivity_ref = coupling.compute_temperature_dependent_conductivity(T_ref)
        assert torch.allclose(conductivity_ref, torch.tensor([[coupling.sigma]]), rtol=1e-5)
        
        # Test at higher temperature (conductivity should decrease)
        T_high = torch.tensor([[373.15]])  # 100°C higher
        conductivity_high = coupling.compute_temperature_dependent_conductivity(T_high)
        assert conductivity_high < coupling.sigma
    
    def test_coupling_with_zero_fields(self):
        """Test coupling computation with zero fields."""
        coupling = ElectromagneticThermalCoupling()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        fields = {
            'electric_field': torch.zeros(batch_size, 3),
            'magnetic_field': torch.zeros(batch_size, 3)
        }
        
        heating = coupling.compute_coupling(fields, coords)
        assert torch.allclose(heating, torch.zeros(batch_size, 1))


class TestThermalMechanicalCoupling:
    """Test thermal-mechanical coupling."""
    
    def test_thermal_mechanical_initialization(self):
        """Test thermal-mechanical coupling initialization."""
        coupling = ThermalMechanicalCoupling(
            weight=2.0, thermal_expansion_coeff=15e-6, reference_temperature=300.0
        )
        assert coupling.get_coupling_weight() == 2.0
        assert coupling.name == "thermal_mechanical"
        assert coupling.alpha_thermal == 15e-6
        assert coupling.T_ref == 300.0
    
    def test_thermal_stress_computation(self):
        """Test thermal stress computation."""
        coupling = ThermalMechanicalCoupling()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        # Create temperature field above reference
        fields = {
            'temperature': torch.full((batch_size, 1), 373.15),  # 80°C above reference
            'youngs_modulus': torch.full((batch_size, 1), 2e11)
        }
        
        thermal_stress = coupling.compute_coupling(fields, coords)
        
        assert thermal_stress.shape == (batch_size, 1)
        assert torch.all(thermal_stress > 0)  # Should be positive for temperature increase
    
    def test_thermal_strain_computation(self):
        """Test thermal strain computation."""
        coupling = ThermalMechanicalCoupling()
        
        temperature = torch.tensor([[373.15], [293.15], [273.15]])  # Hot, ref, cold
        thermal_strain = coupling.compute_thermal_strain(temperature)
        
        assert thermal_strain.shape == temperature.shape
        assert thermal_strain[0] > 0  # Expansion for hot temperature
        assert torch.allclose(thermal_strain[1], torch.zeros(1, 1))  # Zero at reference
        assert thermal_strain[2] < 0  # Contraction for cold temperature
    
    def test_coupling_at_reference_temperature(self):
        """Test coupling at reference temperature."""
        coupling = ThermalMechanicalCoupling()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        fields = {
            'temperature': torch.full((batch_size, 1), coupling.T_ref),
            'youngs_modulus': torch.full((batch_size, 1), 2e11)
        }
        
        thermal_stress = coupling.compute_coupling(fields, coords)
        assert torch.allclose(thermal_stress, torch.zeros(batch_size, 1))


class TestMechanicalElectromagneticCoupling:
    """Test mechanical-electromagnetic coupling."""
    
    def test_mechanical_em_initialization(self):
        """Test mechanical-electromagnetic coupling initialization."""
        coupling = MechanicalElectromagneticCoupling(weight=1.2, magnetostrictive_coeff=2e-5)
        assert coupling.get_coupling_weight() == 1.2
        assert coupling.name == "mechanical_em"
        assert coupling.lambda_s == 2e-5
    
    def test_magnetostrictive_coupling(self):
        """Test magnetostrictive coupling computation."""
        coupling = MechanicalElectromagneticCoupling()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        fields = {
            'magnetic_field': torch.randn(batch_size, 3),
            'displacement': torch.randn(batch_size, 3)
        }
        
        magnetostrictive_effect = coupling.compute_coupling(fields, coords)
        
        assert magnetostrictive_effect.shape == (batch_size, 1)
        assert torch.all(magnetostrictive_effect >= 0)  # Should be non-negative
    
    def test_coupling_with_zero_fields(self):
        """Test coupling with zero magnetic field and displacement."""
        coupling = MechanicalElectromagneticCoupling()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        fields = {
            'magnetic_field': torch.zeros(batch_size, 3),
            'displacement': torch.zeros(batch_size, 3)
        }
        
        magnetostrictive_effect = coupling.compute_coupling(fields, coords)
        assert torch.allclose(magnetostrictive_effect, torch.zeros(batch_size, 1))


class TestEnergyConservationConstraint:
    """Test energy conservation constraint."""
    
    def test_energy_conservation_initialization(self):
        """Test energy conservation constraint initialization."""
        constraint = EnergyConservationConstraint(weight=2.0)
        assert constraint.weight == 2.0
    
    def test_energy_residual_computation(self):
        """Test energy conservation residual computation."""
        constraint = EnergyConservationConstraint()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        fields = {
            'electric_field': torch.randn(batch_size, 3),
            'magnetic_field': torch.randn(batch_size, 3),
            'temperature': torch.full((batch_size, 1), 323.15),
            'displacement': torch.randn(batch_size, 3),
            'velocity': torch.randn(batch_size, 3)
        }
        
        energy_residual = constraint.compute_energy_residual(fields, coords)
        
        assert isinstance(energy_residual, torch.Tensor)
        assert energy_residual.shape == torch.Size([])
        assert energy_residual >= 0
    
    def test_energy_residual_with_minimal_fields(self):
        """Test energy residual with minimal field set."""
        constraint = EnergyConservationConstraint()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        # Only provide electric field
        fields = {'electric_field': torch.randn(batch_size, 3)}
        
        energy_residual = constraint.compute_energy_residual(fields, coords)
        assert isinstance(energy_residual, torch.Tensor)
        assert energy_residual >= 0


class TestMultiPhysicsCoupling:
    """Test multi-physics coupling module."""
    
    def test_multi_physics_initialization(self):
        """Test multi-physics coupling initialization."""
        coupling_terms = [
            ElectromagneticThermalCoupling(weight=1.0),
            ThermalMechanicalCoupling(weight=2.0)
        ]
        
        coupling_module = MultiPhysicsCoupling(coupling_terms, energy_conservation=True)
        
        assert len(coupling_module.coupling_terms) == 2
        assert coupling_module.use_energy_conservation
        assert hasattr(coupling_module, 'energy_constraint')
        assert torch.allclose(coupling_module.coupling_weights, torch.tensor([1.0, 2.0]))
    
    def test_multi_physics_forward_pass(self):
        """Test multi-physics coupling forward pass."""
        coupling_terms = [
            ElectromagneticThermalCoupling(weight=1.0),
            ThermalMechanicalCoupling(weight=1.0)
        ]
        
        coupling_module = MultiPhysicsCoupling(coupling_terms, energy_conservation=True)
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        fields = {
            'electric_field': torch.randn(batch_size, 3),
            'magnetic_field': torch.randn(batch_size, 3),
            'temperature': torch.full((batch_size, 1), 323.15),
            'displacement': torch.randn(batch_size, 3),
            'velocity': torch.randn(batch_size, 3)
        }
        
        modified_fields, coupling_loss = coupling_module.forward(fields, coords)
        
        # Check that new fields are added
        assert 'heat_source' in modified_fields
        assert 'thermal_stress' in modified_fields
        assert 'coupling_contributions' in modified_fields
        
        # Check coupling loss
        assert isinstance(coupling_loss, torch.Tensor)
        assert coupling_loss >= 0
    
    def test_coupling_loss_computation(self):
        """Test coupling loss computation."""
        coupling_terms = [ElectromagneticThermalCoupling(weight=1.0)]
        coupling_module = MultiPhysicsCoupling(coupling_terms, energy_conservation=False)
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        fields = {
            'electric_field': torch.randn(batch_size, 3),
            'magnetic_field': torch.randn(batch_size, 3)
        }
        
        coupling_loss = coupling_module.get_coupling_loss(fields, coords)
        
        assert isinstance(coupling_loss, torch.Tensor)
        assert coupling_loss.shape == torch.Size([])
        assert coupling_loss >= 0
    
    def test_coupling_weight_update(self):
        """Test coupling weight updating."""
        coupling_terms = [
            ElectromagneticThermalCoupling(weight=1.0),
            ThermalMechanicalCoupling(weight=2.0)
        ]
        
        coupling_module = MultiPhysicsCoupling(coupling_terms, energy_conservation=False)
        
        new_weights = {"em_thermal": 3.0, "thermal_mechanical": 4.0}
        coupling_module.update_coupling_weights(new_weights)
        
        assert coupling_module.coupling_weights[0] == 3.0
        assert coupling_module.coupling_weights[1] == 4.0
    
    def test_coupling_contributions_analysis(self):
        """Test coupling contributions analysis."""
        coupling_terms = [
            ElectromagneticThermalCoupling(weight=1.0),
            ThermalMechanicalCoupling(weight=1.0)
        ]
        
        coupling_module = MultiPhysicsCoupling(coupling_terms, energy_conservation=True)
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        fields = {
            'electric_field': torch.randn(batch_size, 3),
            'magnetic_field': torch.randn(batch_size, 3),
            'temperature': torch.full((batch_size, 1), 323.15)
        }
        
        contributions = coupling_module.get_coupling_contributions(fields, coords)
        
        assert "em_thermal" in contributions
        assert "thermal_mechanical" in contributions
        assert "energy_conservation" in contributions
    
    def test_gradient_flow(self):
        """Test that gradients flow through coupling module."""
        coupling_terms = [ElectromagneticThermalCoupling(weight=1.0)]
        coupling_module = MultiPhysicsCoupling(coupling_terms, energy_conservation=False)
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        fields = {
            'electric_field': torch.randn(batch_size, 3, requires_grad=True),
            'magnetic_field': torch.randn(batch_size, 3, requires_grad=True)
        }
        
        coupling_loss = coupling_module.get_coupling_loss(fields, coords)
        coupling_loss.backward()
        
        # Check that gradients exist
        assert fields['electric_field'].grad is not None
        assert fields['magnetic_field'].grad is not None


class TestAdaptiveCouplingWeights:
    """Test adaptive coupling weights."""
    
    def test_adaptive_weights_initialization(self):
        """Test adaptive coupling weights initialization."""
        initial_weights = {"em_thermal": 1.0, "thermal_mechanical": 2.0}
        adaptive_weights = AdaptiveCouplingWeights(initial_weights)
        
        assert len(adaptive_weights.weight_names) == 2
        assert "em_thermal" in adaptive_weights.weight_names
        assert "thermal_mechanical" in adaptive_weights.weight_names
    
    def test_adaptive_weight_computation(self):
        """Test adaptive weight computation."""
        initial_weights = {"em_thermal": 1.0, "thermal_mechanical": 2.0}
        adaptive_weights = AdaptiveCouplingWeights(initial_weights)
        
        field_magnitudes = {
            "em_thermal": torch.tensor(4.0),
            "thermal_mechanical": torch.tensor(1.0)
        }
        
        computed_weights = adaptive_weights.forward(field_magnitudes)
        
        assert isinstance(computed_weights, dict)
        assert "em_thermal" in computed_weights
        assert "thermal_mechanical" in computed_weights
        assert all(isinstance(w, float) for w in computed_weights.values())
    
    def test_get_current_weights(self):
        """Test getting current weights."""
        initial_weights = {"em_thermal": 1.0, "thermal_mechanical": 2.0}
        adaptive_weights = AdaptiveCouplingWeights(initial_weights)
        
        current_weights = adaptive_weights.get_current_weights()
        
        assert isinstance(current_weights, dict)
        assert len(current_weights) == 2
        # Weights should be close to initial values (within exp/log precision)
        assert abs(current_weights["em_thermal"] - 1.0) < 0.1
        assert abs(current_weights["thermal_mechanical"] - 2.0) < 0.1


class TestMotorCouplingSystem:
    """Test complete motor coupling system creation."""
    
    def test_create_motor_coupling_system(self):
        """Test creation of complete motor coupling system."""
        coupling_system = create_motor_coupling_system(
            em_thermal_weight=1.5,
            thermal_mechanical_weight=2.0,
            mechanical_em_weight=0.8,
            energy_conservation=True
        )
        
        assert isinstance(coupling_system, MultiPhysicsCoupling)
        assert len(coupling_system.coupling_terms) == 3
        assert coupling_system.use_energy_conservation
        
        # Check coupling term types
        term_names = [term.name for term in coupling_system.coupling_terms]
        assert "em_thermal" in term_names
        assert "thermal_mechanical" in term_names
        assert "mechanical_em" in term_names
    
    def test_motor_coupling_system_integration(self):
        """Test integration of complete motor coupling system."""
        coupling_system = create_motor_coupling_system()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        # Complete field set for motor application
        fields = {
            'electric_field': torch.randn(batch_size, 3),
            'magnetic_field': torch.randn(batch_size, 3),
            'temperature': torch.full((batch_size, 1), 323.15),
            'displacement': torch.randn(batch_size, 3),
            'velocity': torch.randn(batch_size, 3)
        }
        
        modified_fields, coupling_loss = coupling_system.forward(fields, coords)
        
        # Check that all coupling effects are present
        assert 'heat_source' in modified_fields
        assert 'thermal_stress' in modified_fields
        assert 'magnetostrictive_strain' in modified_fields
        assert 'coupling_contributions' in modified_fields
        
        # Check that coupling loss is reasonable
        assert isinstance(coupling_loss, torch.Tensor)
        assert coupling_loss >= 0
        
        # Check coupling contributions
        contributions = modified_fields['coupling_contributions']
        assert len(contributions) == 4  # 3 coupling terms + energy conservation


class TestCouplingConsistency:
    """Test consistency between different coupling mechanisms."""
    
    def test_coupling_symmetry(self):
        """Test that coupling effects are symmetric where expected."""
        # Test electromagnetic-thermal coupling symmetry
        em_thermal = ElectromagneticThermalCoupling()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        # Same field magnitudes, different orientations
        fields1 = {
            'electric_field': torch.tensor([[1.0, 0.0, 0.0]] * batch_size),
            'magnetic_field': torch.tensor([[0.0, 1.0, 0.0]] * batch_size)
        }
        
        fields2 = {
            'electric_field': torch.tensor([[0.0, 1.0, 0.0]] * batch_size),
            'magnetic_field': torch.tensor([[1.0, 0.0, 0.0]] * batch_size)
        }
        
        heating1 = em_thermal.compute_coupling(fields1, coords)
        heating2 = em_thermal.compute_coupling(fields2, coords)
        
        # Should be equal due to symmetry in |E|² and |B|² terms
        assert torch.allclose(heating1, heating2, rtol=1e-5)
    
    def test_coupling_scaling(self):
        """Test that coupling scales properly with field magnitudes."""
        em_thermal = ElectromagneticThermalCoupling()
        
        batch_size = 4
        coords = torch.linspace(0, 1, batch_size).unsqueeze(1)
        
        # Base fields
        base_fields = {
            'electric_field': torch.randn(batch_size, 3),
            'magnetic_field': torch.randn(batch_size, 3)
        }
        
        # Scaled fields (2x magnitude)
        scaled_fields = {
            'electric_field': 2.0 * base_fields['electric_field'],
            'magnetic_field': 2.0 * base_fields['magnetic_field']
        }
        
        heating_base = em_thermal.compute_coupling(base_fields, coords)
        heating_scaled = em_thermal.compute_coupling(scaled_fields, coords)
        
        # Should scale as square of field magnitude (Joule heating ~ |E|²)
        assert torch.allclose(heating_scaled, 4.0 * heating_base, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])