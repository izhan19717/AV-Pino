"""
Unit tests for data preprocessing and augmentation.
"""

import unittest
import numpy as np
from src.data.preprocessor import (
    DataPreprocessor, DataAugmentor, 
    PreprocessingConfig, AugmentationConfig
)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for data preprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PreprocessingConfig(
            window_size=512,
            overlap_ratio=0.5,
            normalization_method='standard',
            sequence_length=256,
            target_sampling_rate=12000,
            remove_dc=True,
            apply_filtering=True,
            filter_cutoff=(10.0, 1000.0)
        )
        self.preprocessor = DataPreprocessor(self.config)
        
        # Create test signals
        self.sampling_rate = 12000
        duration = 2.0
        time = np.linspace(0, duration, int(self.sampling_rate * duration), endpoint=False)
        
        # Signal with DC offset and multiple frequency components
        self.test_signal = (
            5.0 +  # DC offset
            2.0 * np.sin(2 * np.pi * 50 * time) +  # 50 Hz
            1.0 * np.sin(2 * np.pi * 150 * time) +  # 150 Hz
            0.5 * np.sin(2 * np.pi * 500 * time) +  # 500 Hz
            0.1 * np.random.randn(len(time))  # Noise
        )
        
        self.test_signals = [self.test_signal, self.test_signal * 0.8]
        self.test_labels = [0, 1]
        self.sampling_rates = [self.sampling_rate, self.sampling_rate]
        
    def test_preprocessing_config(self):
        """Test preprocessing configuration."""
        config = PreprocessingConfig()
        self.assertEqual(config.window_size, 1024)
        self.assertEqual(config.overlap_ratio, 0.5)
        self.assertEqual(config.normalization_method, 'standard')
        
        # Test custom config
        custom_config = PreprocessingConfig(
            window_size=256,
            normalization_method='minmax'
        )
        self.assertEqual(custom_config.window_size, 256)
        self.assertEqual(custom_config.normalization_method, 'minmax')
        
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertEqual(self.preprocessor.config.window_size, 512)
        self.assertFalse(self.preprocessor.fitted)
        self.assertEqual(len(self.preprocessor.scalers), 0)
        
    def test_fit(self):
        """Test fitting preprocessor."""
        self.preprocessor.fit(self.test_signals, self.test_labels)
        
        self.assertTrue(self.preprocessor.fitted)
        self.assertIn('signal', self.preprocessor.scalers)
        self.assertIn('mean', self.preprocessor.global_stats)
        self.assertIn('std', self.preprocessor.global_stats)
        
        # Check that global stats are reasonable
        self.assertGreater(self.preprocessor.global_stats['std'], 0)
        
    def test_fit_different_normalization_methods(self):
        """Test fitting with different normalization methods."""
        # Test MinMax normalization
        config_minmax = PreprocessingConfig(normalization_method='minmax')
        preprocessor_minmax = DataPreprocessor(config_minmax)
        preprocessor_minmax.fit(self.test_signals)
        self.assertTrue(preprocessor_minmax.fitted)
        
        # Test invalid normalization method
        config_invalid = PreprocessingConfig(normalization_method='invalid')
        preprocessor_invalid = DataPreprocessor(config_invalid)
        with self.assertRaises(ValueError):
            preprocessor_invalid.fit(self.test_signals)
            
    def test_preprocess_signal(self):
        """Test single signal preprocessing."""
        self.preprocessor.fit(self.test_signals)
        
        processed = self.preprocessor.preprocess_signal(self.test_signal, self.sampling_rate)
        
        # Check that signal is processed
        self.assertEqual(len(processed), len(self.test_signal))
        self.assertIsInstance(processed, np.ndarray)
        
        # DC should be removed (mean should be smaller than original after processing)
        original_mean = abs(np.mean(self.test_signal))
        processed_mean = abs(np.mean(processed))
        # After DC removal and normalization, mean should be different from original
        self.assertNotAlmostEqual(processed_mean, original_mean, places=1)
        
    def test_preprocess_signal_not_fitted(self):
        """Test preprocessing without fitting."""
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess_signal(self.test_signal, self.sampling_rate)
            
    def test_create_windows(self):
        """Test window creation."""
        self.preprocessor.fit(self.test_signals)
        
        windows, window_labels = self.preprocessor.create_windows(self.test_signal, 0)
        
        # Check window dimensions
        self.assertEqual(windows.shape[1], self.config.window_size)
        self.assertGreater(windows.shape[0], 1)  # Should create multiple windows
        
        # Check labels
        if window_labels is not None:
            self.assertEqual(len(window_labels), windows.shape[0])
            
    def test_create_windows_short_signal(self):
        """Test window creation with short signal."""
        self.preprocessor.fit(self.test_signals)
        
        short_signal = np.random.randn(100)  # Shorter than window size
        windows, window_labels = self.preprocessor.create_windows(short_signal, 1)
        
        # Should create one padded window
        self.assertEqual(windows.shape[0], 1)
        self.assertEqual(windows.shape[1], self.config.window_size)
        
    def test_prepare_sequences(self):
        """Test sequence preparation."""
        self.preprocessor.fit(self.test_signals)
        
        # Create some test windows
        test_windows = np.random.randn(10, 512)
        test_labels = np.arange(10)
        
        sequences, seq_labels = self.preprocessor.prepare_sequences(test_windows, test_labels)
        
        # Check sequence dimensions
        self.assertEqual(sequences.shape[0], test_windows.shape[0])
        self.assertEqual(sequences.shape[1], self.config.sequence_length)
        
        # Check labels
        if seq_labels is not None:
            np.testing.assert_array_equal(seq_labels, test_labels)
            
    def test_prepare_sequences_different_lengths(self):
        """Test sequence preparation with different input lengths."""
        self.preprocessor.fit(self.test_signals)
        
        # Test with longer windows (should truncate/subsample)
        long_windows = np.random.randn(5, 1024)
        sequences_long, _ = self.preprocessor.prepare_sequences(long_windows)
        self.assertEqual(sequences_long.shape[1], self.config.sequence_length)
        
        # Test with shorter windows (should interpolate)
        short_windows = np.random.randn(5, 128)
        sequences_short, _ = self.preprocessor.prepare_sequences(short_windows)
        self.assertEqual(sequences_short.shape[1], self.config.sequence_length)
        
    def test_process_dataset(self):
        """Test complete dataset processing."""
        self.preprocessor.fit(self.test_signals, self.test_labels)
        
        sequences, labels = self.preprocessor.process_dataset(
            self.test_signals, self.test_labels, self.sampling_rates
        )
        
        # Check output dimensions
        self.assertGreater(sequences.shape[0], 0)
        self.assertEqual(sequences.shape[1], self.config.sequence_length)
        self.assertEqual(len(labels), sequences.shape[0])
        
        # Check that we have sequences from both input signals
        self.assertGreater(sequences.shape[0], 2)
        
    def test_robust_normalization(self):
        """Test robust normalization method."""
        config_robust = PreprocessingConfig(normalization_method='robust')
        preprocessor_robust = DataPreprocessor(config_robust)
        preprocessor_robust.fit(self.test_signals)
        
        processed = preprocessor_robust.preprocess_signal(self.test_signal, self.sampling_rate)
        
        # Should be processed without errors
        self.assertEqual(len(processed), len(self.test_signal))
        
    def test_resampling(self):
        """Test signal resampling."""
        config_resample = PreprocessingConfig(target_sampling_rate=6000)
        preprocessor_resample = DataPreprocessor(config_resample)
        preprocessor_resample.fit(self.test_signals)
        
        processed = preprocessor_resample.preprocess_signal(self.test_signal, self.sampling_rate)
        
        # Should be resampled to half length
        expected_length = len(self.test_signal) // 2
        self.assertAlmostEqual(len(processed), expected_length, delta=10)
        
    def test_filtering_edge_cases(self):
        """Test filtering with edge cases."""
        # Test with invalid filter cutoffs
        config_invalid_filter = PreprocessingConfig(filter_cutoff=(1000.0, 10.0))  # High < Low
        preprocessor_invalid = DataPreprocessor(config_invalid_filter)
        preprocessor_invalid.fit(self.test_signals)
        
        # Should not crash, just skip filtering
        processed = preprocessor_invalid.preprocess_signal(self.test_signal, self.sampling_rate)
        self.assertEqual(len(processed), len(self.test_signal))


class TestDataAugmentor(unittest.TestCase):
    """Test cases for data augmentor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig(
            noise_std_ratio=0.05,
            time_shift_ratio=0.1,
            amplitude_scale_range=(0.8, 1.2),
            frequency_shift_range=(0.95, 1.05),
            synthetic_fault_ratio=0.2,
            enable_mixup=True,
            mixup_alpha=0.2
        )
        self.augmentor = DataAugmentor(self.config)
        
        # Create test signal
        time = np.linspace(0, 1, 1000)
        self.test_signal = np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.randn(len(time))
        self.test_signals = np.array([self.test_signal, self.test_signal * 0.5])
        self.test_labels = np.array([0, 1])
        
    def test_augmentation_config(self):
        """Test augmentation configuration."""
        config = AugmentationConfig()
        self.assertEqual(config.noise_std_ratio, 0.05)
        self.assertEqual(config.time_shift_ratio, 0.1)
        self.assertTrue(config.enable_mixup)
        
    def test_initialization(self):
        """Test augmentor initialization."""
        self.assertEqual(self.augmentor.config.noise_std_ratio, 0.05)
        self.assertIsNotNone(self.augmentor.rng)
        
    def test_augment_signal(self):
        """Test single signal augmentation."""
        augmented = self.augmentor.augment_signal(self.test_signal)
        
        # Should have same length
        self.assertEqual(len(augmented), len(self.test_signal))
        
        # Should be different from original (due to noise and other augmentations)
        self.assertFalse(np.array_equal(augmented, self.test_signal))
        
        # Should have similar magnitude
        original_rms = np.sqrt(np.mean(self.test_signal**2))
        augmented_rms = np.sqrt(np.mean(augmented**2))
        self.assertLess(abs(augmented_rms - original_rms) / original_rms, 0.5)  # Within 50%
        
    def test_augment_signal_with_label(self):
        """Test signal augmentation with label."""
        augmented = self.augmentor.augment_signal(self.test_signal, label=1)
        
        # Should work without errors
        self.assertEqual(len(augmented), len(self.test_signal))
        
    def test_generate_synthetic_faults(self):
        """Test synthetic fault generation."""
        normal_signals = [self.test_signal]
        fault_types = ['bearing_fault', 'unbalance', 'misalignment']
        
        synthetic_signals, synthetic_labels = self.augmentor.generate_synthetic_faults(
            normal_signals, fault_types
        )
        
        # Should generate one signal per fault type
        self.assertEqual(len(synthetic_signals), len(fault_types))
        self.assertEqual(len(synthetic_labels), len(fault_types))
        
        # Check that synthetic signals are different from original
        for synthetic_signal in synthetic_signals:
            self.assertEqual(len(synthetic_signal), len(self.test_signal))
            self.assertFalse(np.array_equal(synthetic_signal, self.test_signal))
            
        # Check labels
        self.assertEqual(set(synthetic_labels), set(fault_types))
        
    def test_bearing_fault_generation(self):
        """Test bearing fault generation."""
        bearing_fault = self.augmentor._generate_bearing_fault(self.test_signal)
        
        self.assertEqual(len(bearing_fault), len(self.test_signal))
        self.assertFalse(np.array_equal(bearing_fault, self.test_signal))
        
        # Should have higher peak values due to impulses
        self.assertGreater(np.max(np.abs(bearing_fault)), np.max(np.abs(self.test_signal)))
        
    def test_unbalance_fault_generation(self):
        """Test unbalance fault generation."""
        unbalance_fault = self.augmentor._generate_unbalance_fault(self.test_signal)
        
        self.assertEqual(len(unbalance_fault), len(self.test_signal))
        self.assertFalse(np.array_equal(unbalance_fault, self.test_signal))
        
    def test_misalignment_fault_generation(self):
        """Test misalignment fault generation."""
        misalignment_fault = self.augmentor._generate_misalignment_fault(self.test_signal)
        
        self.assertEqual(len(misalignment_fault), len(self.test_signal))
        self.assertFalse(np.array_equal(misalignment_fault, self.test_signal))
        
    def test_generic_fault_generation(self):
        """Test generic fault generation."""
        generic_fault = self.augmentor._generate_generic_fault(self.test_signal)
        
        self.assertEqual(len(generic_fault), len(self.test_signal))
        self.assertFalse(np.array_equal(generic_fault, self.test_signal))
        
    def test_mixup_augmentation(self):
        """Test mixup augmentation."""
        mixed_signals, mixed_labels = self.augmentor.mixup_augmentation(
            self.test_signals, self.test_labels
        )
        
        # Should have same dimensions
        self.assertEqual(mixed_signals.shape, self.test_signals.shape)
        self.assertEqual(len(mixed_labels), len(self.test_labels))
        
        # Mixed signals should be different from originals
        self.assertFalse(np.array_equal(mixed_signals, self.test_signals))
        
        # Labels should be from original set
        for label in mixed_labels:
            self.assertIn(label, self.test_labels)
            
    def test_mixup_with_custom_alpha(self):
        """Test mixup with custom alpha parameter."""
        mixed_signals, mixed_labels = self.augmentor.mixup_augmentation(
            self.test_signals, self.test_labels, alpha=0.5
        )
        
        self.assertEqual(mixed_signals.shape, self.test_signals.shape)
        self.assertEqual(len(mixed_labels), len(self.test_labels))
        
    def test_augment_dataset(self):
        """Test complete dataset augmentation."""
        augmented_signals, augmented_labels = self.augmentor.augment_dataset(
            self.test_signals, self.test_labels, augmentation_factor=3
        )
        
        # Should have more samples than original
        self.assertGreater(len(augmented_signals), len(self.test_signals))
        self.assertEqual(len(augmented_signals), len(augmented_labels))
        
        # Should include original signals
        self.assertGreaterEqual(len(augmented_signals), len(self.test_signals) * 3)
        
    def test_frequency_shift(self):
        """Test frequency domain shifting."""
        shifted = self.augmentor._frequency_shift(self.test_signal)
        
        self.assertEqual(len(shifted), len(self.test_signal))
        # Should be different due to frequency shifting
        self.assertFalse(np.array_equal(shifted, self.test_signal))
        
    def test_augmentation_reproducibility(self):
        """Test that augmentation is reproducible with same random state."""
        # Create two augmentors with same random state
        augmentor1 = DataAugmentor(self.config)
        augmentor2 = DataAugmentor(self.config)
        
        # Both should produce same results
        aug1 = augmentor1.augment_signal(self.test_signal)
        aug2 = augmentor2.augment_signal(self.test_signal)
        
        np.testing.assert_array_almost_equal(aug1, aug2)
        
    def test_no_augmentation_config(self):
        """Test augmentation with disabled features."""
        no_aug_config = AugmentationConfig(
            noise_std_ratio=0.0,
            time_shift_ratio=0.0,
            amplitude_scale_range=(1.0, 1.0),
            frequency_shift_range=(1.0, 1.0),
            enable_mixup=False
        )
        no_augmentor = DataAugmentor(no_aug_config)
        
        # Should still work but produce minimal changes
        augmented = no_augmentor.augment_signal(self.test_signal)
        self.assertEqual(len(augmented), len(self.test_signal))


class TestIntegration(unittest.TestCase):
    """Integration tests for preprocessing and augmentation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.prep_config = PreprocessingConfig(window_size=256, sequence_length=128)
        self.aug_config = AugmentationConfig(noise_std_ratio=0.02)
        
        self.preprocessor = DataPreprocessor(self.prep_config)
        self.augmentor = DataAugmentor(self.aug_config)
        
        # Create test dataset
        time = np.linspace(0, 2, 2000)
        signal1 = np.sin(2 * np.pi * 50 * time) + 0.1 * np.random.randn(len(time))
        signal2 = np.sin(2 * np.pi * 100 * time) + 0.1 * np.random.randn(len(time))
        
        self.signals = [signal1, signal2]
        self.labels = [0, 1]
        self.sampling_rates = [1000, 1000]
        
    def test_preprocessing_then_augmentation(self):
        """Test preprocessing followed by augmentation."""
        # Fit and process dataset
        self.preprocessor.fit(self.signals, self.labels)
        sequences, seq_labels = self.preprocessor.process_dataset(
            self.signals, self.labels, self.sampling_rates
        )
        
        # Augment processed data
        augmented_sequences, augmented_labels = self.augmentor.augment_dataset(
            sequences, seq_labels, augmentation_factor=2
        )
        
        # Check final results
        self.assertGreater(len(augmented_sequences), len(sequences))
        self.assertEqual(len(augmented_sequences), len(augmented_labels))
        self.assertEqual(augmented_sequences.shape[1], self.prep_config.sequence_length)
        
    def test_synthetic_fault_integration(self):
        """Test integration with synthetic fault generation."""
        # Process normal signals
        self.preprocessor.fit(self.signals, self.labels)
        sequences, seq_labels = self.preprocessor.process_dataset(
            self.signals, self.labels, self.sampling_rates
        )
        
        # Generate synthetic faults from processed sequences
        normal_sequences = sequences[seq_labels == 0]  # Assuming 0 is normal
        
        if len(normal_sequences) > 0:
            synthetic_signals, synthetic_labels = self.augmentor.generate_synthetic_faults(
                [normal_sequences[0]], ['bearing_fault']
            )
            
            self.assertEqual(len(synthetic_signals), 1)
            self.assertEqual(len(synthetic_labels), 1)
            self.assertEqual(len(synthetic_signals[0]), len(normal_sequences[0]))


if __name__ == '__main__':
    unittest.main()