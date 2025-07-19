"""
Unit tests for signal processing functionality.
"""

import unittest
import numpy as np
from src.data.signal_processor import SignalProcessor


class TestSignalProcessor(unittest.TestCase):
    """Test cases for signal processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = SignalProcessor(sampling_rate=12000)
        
        # Create test signals
        self.test_signal_sine = self._generate_sine_wave(frequency=100, duration=1.0)
        self.test_signal_noise = np.random.randn(len(self.test_signal_sine))
        self.test_signal_mixed = self.test_signal_sine + 0.1 * self.test_signal_noise
        
    def _generate_sine_wave(self, frequency: float, duration: float, 
                           sampling_rate: int = 12000) -> np.ndarray:
        """Generate a sine wave for testing."""
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        return np.sin(2 * np.pi * frequency * t)
    
    def test_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.sampling_rate, 12000)
        
    def test_time_domain_features_sine_wave(self):
        """Test time-domain feature extraction on sine wave."""
        features = self.processor.extract_time_domain_features(self.test_signal_sine)
        
        # Check that all expected features are present
        expected_features = [
            'mean', 'std', 'variance', 'rms', 'peak', 'peak_to_peak',
            'crest_factor', 'skewness', 'kurtosis', 'energy', 'power',
            'zero_crossing_rate'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
            
        # Sine wave should have near-zero mean
        self.assertAlmostEqual(features['mean'], 0.0, places=2)
        
        # RMS of sine wave should be approximately 1/sqrt(2)
        expected_rms = 1.0 / np.sqrt(2)
        self.assertAlmostEqual(features['rms'], expected_rms, places=1)
        
    def test_time_domain_features_noise(self):
        """Test time-domain feature extraction on noise."""
        features = self.processor.extract_time_domain_features(self.test_signal_noise)
        
        # Noise should have near-zero mean
        self.assertAlmostEqual(features['mean'], 0.0, places=1)
        
        # Standard deviation should be close to 1 for standard normal noise
        self.assertAlmostEqual(features['std'], 1.0, places=1)
        
    def test_frequency_domain_features_sine_wave(self):
        """Test frequency-domain feature extraction on sine wave."""
        features = self.processor.extract_frequency_domain_features(self.test_signal_sine)
        
        # Check that all expected features are present
        expected_features = [
            'spectral_centroid', 'spectral_spread', 'spectral_rolloff',
            'peak_frequency', 'peak_magnitude', 'low_freq_energy',
            'mid_freq_energy', 'high_freq_energy'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
            
        # Peak frequency should be close to 100 Hz
        self.assertAlmostEqual(features['peak_frequency'], 100.0, places=0)
        
        # Most energy should be in low frequency band for 100 Hz sine wave
        self.assertGreater(features['low_freq_energy'], 0.8)
        
    def test_time_frequency_features(self):
        """Test time-frequency feature extraction."""
        features = self.processor.extract_time_frequency_features(self.test_signal_mixed)
        
        # Check that STFT results are present
        expected_features = ['stft_frequencies', 'stft_times', 'stft_magnitude', 'spectrogram']
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], np.ndarray)
            
        # Check dimensions make sense
        freq_bins, time_bins = features['stft_magnitude'].shape
        self.assertEqual(len(features['stft_frequencies']), freq_bins)
        self.assertEqual(len(features['stft_times']), time_bins)
        
    def test_envelope_spectrum(self):
        """Test envelope spectrum computation."""
        frequencies, envelope_spectrum = self.processor.compute_envelope_spectrum(self.test_signal_mixed)
        
        self.assertIsInstance(frequencies, np.ndarray)
        self.assertIsInstance(envelope_spectrum, np.ndarray)
        self.assertEqual(len(frequencies), len(envelope_spectrum))
        
        # All frequencies should be positive
        self.assertTrue(np.all(frequencies > 0))
        
        # Envelope spectrum should be non-negative
        self.assertTrue(np.all(envelope_spectrum >= 0))
        
    def test_skewness_calculation(self):
        """Test skewness calculation."""
        # Symmetric signal should have near-zero skewness
        symmetric_signal = np.array([-2, -1, 0, 1, 2])
        skewness = self.processor._calculate_skewness(symmetric_signal)
        self.assertAlmostEqual(skewness, 0.0, places=10)
        
        # Right-skewed signal should have positive skewness
        right_skewed = np.array([1, 1, 1, 2, 10])
        skewness = self.processor._calculate_skewness(right_skewed)
        self.assertGreater(skewness, 0)
        
    def test_kurtosis_calculation(self):
        """Test kurtosis calculation."""
        # Normal distribution should have kurtosis close to 0 (excess kurtosis)
        normal_signal = np.random.normal(0, 1, 10000)
        kurtosis = self.processor._calculate_kurtosis(normal_signal)
        self.assertAlmostEqual(kurtosis, 0.0, places=0)  # Rough approximation
        
    def test_zero_crossing_rate(self):
        """Test zero crossing rate calculation."""
        # Sine wave should have predictable zero crossing rate
        zcr = self.processor._zero_crossing_rate(self.test_signal_sine)
        
        # For 100 Hz sine wave sampled at 12000 Hz for 1 second,
        # we expect about 200 zero crossings
        expected_zcr = 200.0 / len(self.test_signal_sine)
        self.assertAlmostEqual(zcr, expected_zcr, places=2)
        
    def test_frequency_band_energy(self):
        """Test frequency band energy calculation."""
        # Create test frequency and magnitude arrays
        frequencies = np.linspace(0, 6000, 1000)
        magnitude = np.ones_like(frequencies)  # Uniform magnitude
        
        band_energies = self.processor._frequency_band_energy(frequencies, magnitude)
        
        # Check that all bands are present
        expected_bands = ['low_freq_energy', 'mid_freq_energy', 'high_freq_energy']
        for band in expected_bands:
            self.assertIn(band, band_energies)
            
        # Energy should sum to approximately 1
        total_energy = sum(band_energies.values())
        self.assertAlmostEqual(total_energy, 1.0, places=2)
        
    def test_custom_sampling_rate(self):
        """Test using custom sampling rate."""
        custom_processor = SignalProcessor(sampling_rate=48000)
        self.assertEqual(custom_processor.sampling_rate, 48000)
        
        # Generate signal with custom sampling rate
        custom_sine = self._generate_sine_wave(frequency=100, duration=1.0, sampling_rate=48000)
        
        # Test frequency domain features with custom rate
        features = custom_processor.extract_frequency_domain_features(
            custom_sine, sampling_rate=48000
        )
        
        # Should still detect 100 Hz peak
        self.assertAlmostEqual(features['peak_frequency'], 100.0, places=0)


if __name__ == '__main__':
    unittest.main()