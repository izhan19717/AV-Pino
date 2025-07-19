"""
Unit tests for physics-based feature extraction.
"""

import unittest
import numpy as np
from src.physics.feature_extractor import PhysicsFeatureExtractor, PhysicsFeatures
from src.physics.visualization import PhysicsVisualizer


class TestPhysicsFeatureExtractor(unittest.TestCase):
    """Test cases for physics feature extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PhysicsFeatureExtractor(sampling_rate=12000)
        
        # Create test signals
        self.sampling_rate = 12000
        duration = 1.0
        time = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        
        # Vibration signal with multiple frequency components
        self.vibration_signal = (
            0.5 * np.sin(2 * np.pi * 30 * time) +  # Shaft frequency
            0.2 * np.sin(2 * np.pi * 105 * time) +  # BPFO
            0.1 * np.sin(2 * np.pi * 162 * time) +  # BPFI
            0.05 * np.random.randn(len(time))  # Noise
        )
        
        # Current signal
        self.current_signal = (
            10 + 5 * np.sin(2 * np.pi * 50 * time) +  # 50 Hz fundamental
            0.5 * np.sin(2 * np.pi * 100 * time) +  # 2nd harmonic
            0.1 * np.random.randn(len(time))  # Noise
        )
        
        # Voltage signal
        self.voltage_signal = (
            230 * np.sin(2 * np.pi * 50 * time + np.pi/6) +  # 50 Hz with phase shift
            0.2 * np.random.randn(len(time))  # Noise
        )
        
        # Temperature signal
        self.temperature_signal = (
            25 + 10 * (1 - np.exp(-time/5)) +  # Exponential rise
            0.5 * np.random.randn(len(time))  # Noise
        )
        
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertEqual(self.extractor.sampling_rate, 12000)
        self.assertIsInstance(self.extractor.motor_params, dict)
        
        # Test with custom parameters
        custom_params = {'pole_pairs': 4, 'rotor_inertia': 0.02}
        custom_extractor = PhysicsFeatureExtractor(
            sampling_rate=48000, 
            motor_params=custom_params
        )
        self.assertEqual(custom_extractor.sampling_rate, 48000)
        self.assertEqual(custom_extractor.motor_params['pole_pairs'], 4)
        
    def test_physics_features_dataclass(self):
        """Test PhysicsFeatures dataclass."""
        em_features = {'flux_density_variation': 0.1}
        thermal_features = {'heat_generation_rate': 50.0}
        mechanical_features = {'first_mode_frequency': 100.0}
        coupling_features = {'em_mech_coupling_strength': 0.5}
        
        features = PhysicsFeatures(
            electromagnetic=em_features,
            thermal=thermal_features,
            mechanical=mechanical_features,
            coupling_terms=coupling_features
        )
        
        self.assertEqual(features.electromagnetic['flux_density_variation'], 0.1)
        self.assertEqual(features.thermal['heat_generation_rate'], 50.0)
        self.assertEqual(features.mechanical['first_mode_frequency'], 100.0)
        self.assertEqual(features.coupling_terms['em_mech_coupling_strength'], 0.5)
        
    def test_extract_electromagnetic_features(self):
        """Test electromagnetic feature extraction."""
        features = self.extractor.extract_electromagnetic_features(
            self.vibration_signal, self.current_signal, self.voltage_signal
        )
        
        # Check that all expected features are present
        expected_features = [
            'flux_density_variation', 'flux_density_peak', 'em_force_rms', 'em_force_peak',
            'air_gap_variation_std', 'air_gap_asymmetry', 'fundamental_frequency',
            'fundamental_amplitude', 'total_harmonic_distortion', 'average_power',
            'power_factor', 'impedance_magnitude'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
            
        # Test specific values
        self.assertGreater(features['flux_density_variation'], 0)
        self.assertGreater(features['flux_density_peak'], 0)
        self.assertAlmostEqual(features['fundamental_frequency'], 50.0, places=0)
        
    def test_extract_thermal_features(self):
        """Test thermal feature extraction."""
        features = self.extractor.extract_thermal_features(
            self.vibration_signal, self.temperature_signal
        )
        
        # Check that all expected features are present
        expected_features = [
            'heat_generation_rate', 'heat_generation_peak', 'heat_generation_std',
            'thermal_time_constant', 'temperature_gradient_rms', 'temperature_mean',
            'temperature_std', 'temperature_trend', 'thermal_diffusivity',
            'thermal_stability'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
            
        # Test specific values
        self.assertGreater(features['heat_generation_rate'], 0)
        self.assertGreater(features['thermal_time_constant'], 0)
        self.assertAlmostEqual(features['temperature_mean'], 25.0, delta=2.0)  # Allow for noise
        
    def test_extract_mechanical_features(self):
        """Test mechanical feature extraction."""
        features = self.extractor.extract_mechanical_features(self.vibration_signal)
        
        # Check that all expected features are present
        expected_features = [
            'first_mode_frequency', 'first_mode_amplitude', 'modal_damping',
            'bpfo_amplitude', 'bpfi_amplitude', 'envelope_crest_factor',
            'shaft_unbalance', 'shaft_misalignment', 'structural_stiffness',
            'damping_ratio', 'num_resonances', 'primary_resonance_freq',
            'primary_resonance_amplitude'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
            
        # Test specific values
        self.assertGreaterEqual(features['num_resonances'], 0)
        self.assertGreater(features['structural_stiffness'], 0)
        self.assertGreater(features['envelope_crest_factor'], 1.0)  # Should be > 1 for any signal
        
    def test_compute_coupling_terms(self):
        """Test multi-physics coupling computation."""
        features = self.extractor.compute_coupling_terms(
            self.vibration_signal, self.current_signal, self.temperature_signal
        )
        
        # Check that all expected features are present
        expected_features = [
            'em_mech_coupling_strength', 'em_mech_coupling_phase',
            'thermal_mech_coupling', 'em_thermal_coupling',
            'energy_transfer_efficiency'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
            
        # Test specific constraints
        self.assertGreaterEqual(features['energy_transfer_efficiency'], 0)
        self.assertLessEqual(features['energy_transfer_efficiency'], 1)
        self.assertGreaterEqual(features['em_mech_coupling_strength'], 0)
        
    def test_extract_all_features(self):
        """Test extraction of all physics features."""
        features = self.extractor.extract_all_features(
            self.vibration_signal, self.current_signal, 
            self.voltage_signal, self.temperature_signal
        )
        
        self.assertIsInstance(features, PhysicsFeatures)
        self.assertIsInstance(features.electromagnetic, dict)
        self.assertIsInstance(features.thermal, dict)
        self.assertIsInstance(features.mechanical, dict)
        self.assertIsInstance(features.coupling_terms, dict)
        
        # Check that features are not empty
        self.assertGreater(len(features.electromagnetic), 0)
        self.assertGreater(len(features.thermal), 0)
        self.assertGreater(len(features.mechanical), 0)
        self.assertGreater(len(features.coupling_terms), 0)
        
    def test_extract_features_minimal_inputs(self):
        """Test feature extraction with minimal inputs (vibration only)."""
        features = self.extractor.extract_all_features(self.vibration_signal)
        
        self.assertIsInstance(features, PhysicsFeatures)
        
        # Should still extract features even without current/voltage/temperature
        self.assertGreater(len(features.electromagnetic), 0)
        self.assertGreater(len(features.thermal), 0)
        self.assertGreater(len(features.mechanical), 0)
        self.assertGreater(len(features.coupling_terms), 0)
        
    def test_electromagnetic_force_computation(self):
        """Test electromagnetic force computation."""
        em_force = self.extractor._compute_electromagnetic_force(self.vibration_signal)
        
        self.assertEqual(len(em_force), len(self.vibration_signal))
        self.assertIsInstance(em_force, np.ndarray)
        
        # Force should be proportional to acceleration
        acceleration = np.gradient(np.gradient(self.vibration_signal))
        expected_force = self.extractor.motor_params['rotor_inertia'] * acceleration
        np.testing.assert_array_almost_equal(em_force, expected_force)
        
    def test_air_gap_variations(self):
        """Test air gap variation estimation."""
        air_gap_vars = self.extractor._estimate_air_gap_variations(self.vibration_signal)
        
        self.assertEqual(len(air_gap_vars), len(self.vibration_signal))
        self.assertIsInstance(air_gap_vars, np.ndarray)
        
        # Air gap variations should have reduced DC component
        self.assertLess(np.mean(air_gap_vars), np.mean(self.vibration_signal))
        
    def test_heat_generation_estimation(self):
        """Test heat generation estimation."""
        heat_gen = self.extractor._estimate_heat_generation(self.vibration_signal)
        
        self.assertEqual(len(heat_gen), len(self.vibration_signal))
        self.assertIsInstance(heat_gen, np.ndarray)
        
        # Heat generation should be non-negative
        self.assertTrue(np.all(heat_gen >= 0))
        
    def test_modal_analysis(self):
        """Test modal analysis."""
        modal_features = self.extractor._modal_analysis(self.vibration_signal)
        
        expected_features = ['first_mode_frequency', 'first_mode_amplitude', 'modal_damping']
        for feature in expected_features:
            self.assertIn(feature, modal_features)
            self.assertIsInstance(modal_features[feature], (int, float))
            
        # First mode frequency should be positive
        self.assertGreaterEqual(modal_features['first_mode_frequency'], 0)
        
    def test_bearing_fault_indicators(self):
        """Test bearing fault indicators."""
        bearing_features = self.extractor._bearing_fault_indicators(self.vibration_signal)
        
        expected_features = ['bpfo_amplitude', 'bpfi_amplitude', 'envelope_crest_factor']
        for feature in expected_features:
            self.assertIn(feature, bearing_features)
            self.assertIsInstance(bearing_features[feature], (int, float))
            
        # Crest factor should be greater than 1
        self.assertGreater(bearing_features['envelope_crest_factor'], 1.0)
        
    def test_current_signature_analysis(self):
        """Test motor current signature analysis."""
        current_features = self.extractor._current_signature_features(self.current_signal)
        
        expected_features = ['fundamental_frequency', 'fundamental_amplitude', 'total_harmonic_distortion']
        for feature in expected_features:
            self.assertIn(feature, current_features)
            self.assertIsInstance(current_features[feature], (int, float))
            
        # Fundamental frequency should be around 50 Hz
        self.assertAlmostEqual(current_features['fundamental_frequency'], 50.0, places=0)
        
    def test_power_impedance_features(self):
        """Test power and impedance feature extraction."""
        power_features = self.extractor._power_impedance_features(
            self.voltage_signal, self.current_signal
        )
        
        expected_features = ['average_power', 'power_factor', 'impedance_magnitude']
        for feature in expected_features:
            self.assertIn(feature, power_features)
            self.assertIsInstance(power_features[feature], (int, float))
            
        # Power factor should be between -1 and 1
        self.assertGreaterEqual(power_features['power_factor'], -1)
        self.assertLessEqual(power_features['power_factor'], 1)
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very short signal
        short_signal = np.array([1, 2, 3])
        features = self.extractor.extract_all_features(short_signal)
        self.assertIsInstance(features, PhysicsFeatures)
        
        # Test with zero signal
        zero_signal = np.zeros(1000)
        features = self.extractor.extract_all_features(zero_signal)
        self.assertIsInstance(features, PhysicsFeatures)
        
        # Test with constant signal
        constant_signal = np.ones(1000) * 5.0
        features = self.extractor.extract_all_features(constant_signal)
        self.assertIsInstance(features, PhysicsFeatures)


class TestPhysicsVisualizer(unittest.TestCase):
    """Test cases for physics visualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = PhysicsVisualizer()
        
        # Create test features
        self.features = PhysicsFeatures(
            electromagnetic={'flux_density_variation': 0.1, 'em_force_rms': 50.0},
            thermal={'heat_generation_rate': 25.0, 'thermal_stability': 0.8},
            mechanical={'first_mode_frequency': 100.0, 'damping_ratio': 0.05},
            coupling_terms={'em_mech_coupling_strength': 0.3}
        )
        
        # Create test signal
        time = np.linspace(0, 1, 1000)
        self.test_signal = np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.randn(len(time))
        
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertEqual(self.visualizer.figsize, (12, 8))
        
        custom_visualizer = PhysicsVisualizer(figsize=(10, 6))
        self.assertEqual(custom_visualizer.figsize, (10, 6))
        
    def test_plot_physics_features_overview(self):
        """Test physics features overview plot."""
        fig = self.visualizer.plot_physics_features_overview(self.features)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # Should have 4 subplots
        
        # Check that all subplots have titles
        for ax in fig.axes:
            self.assertIsNotNone(ax.get_title())
            
    def test_plot_electromagnetic_analysis(self):
        """Test electromagnetic analysis plot."""
        fig = self.visualizer.plot_electromagnetic_analysis(self.test_signal)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # Should have 4 subplots
        
    def test_plot_thermal_analysis(self):
        """Test thermal analysis plot."""
        fig = self.visualizer.plot_thermal_analysis(self.test_signal)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # Should have 4 subplots
        
    def test_plot_mechanical_analysis(self):
        """Test mechanical analysis plot."""
        fig = self.visualizer.plot_mechanical_analysis(self.test_signal)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # Should have 4 subplots
        
    def test_plot_coupling_analysis(self):
        """Test coupling analysis plot."""
        fig = self.visualizer.plot_coupling_analysis(self.test_signal)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # Should have 4 subplots
        
    def test_plot_feature_comparison(self):
        """Test feature comparison plot."""
        # Create multiple feature sets
        features_list = [self.features, self.features]  # Same features for simplicity
        labels = ['Sample 1', 'Sample 2']
        
        fig = self.visualizer.plot_feature_comparison(features_list, labels)
        
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.axes), 0)


if __name__ == '__main__':
    unittest.main()