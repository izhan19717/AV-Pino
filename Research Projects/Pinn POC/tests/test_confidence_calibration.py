"""
Unit tests for confidence calibration and safety thresholds.
"""

import pytest
import torch
import numpy as np
from src.physics.uncertainty import (
    ConfidenceCalibration, SafetyThresholds, UncertaintyConfig,
    CalibrationMetrics
)


class TestConfidenceCalibration:
    """Test cases for ConfidenceCalibration class"""
    
    @pytest.fixture
    def config(self):
        return UncertaintyConfig(
            calibration_bins=10,
            safety_threshold=0.8
        )
    
    @pytest.fixture
    def calibration_system(self, config):
        return ConfidenceCalibration(config)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample prediction data for testing"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        n_samples = 1000
        n_classes = 4
        
        # Generate predictions with some calibration error
        predictions = torch.randn(n_samples, n_classes)
        targets = torch.randint(0, n_classes, (n_samples,))
        
        # Generate confidences that are slightly overconfident
        confidences = torch.sigmoid(torch.randn(n_samples)) * 0.8 + 0.2
        
        return predictions, targets, confidences
    
    def test_calibration_initialization(self, calibration_system):
        """Test proper initialization of calibration system"""
        assert not calibration_system.is_calibrated
        assert calibration_system.calibration_map is None
        assert calibration_system.config.calibration_bins == 10
    
    def test_calibration_process(self, calibration_system, sample_data):
        """Test the calibration process"""
        predictions, targets, confidences = sample_data
        
        # Perform calibration
        calibration_system.calibrate(predictions, targets, confidences)
        
        assert calibration_system.is_calibrated
        assert calibration_system.calibration_map is not None
        assert len(calibration_system.calibration_map) > 0
    
    def test_apply_calibration(self, calibration_system, sample_data):
        """Test applying calibration to new confidence scores"""
        predictions, targets, confidences = sample_data
        
        # Calibrate first
        calibration_system.calibrate(predictions, targets, confidences)
        
        # Apply calibration to new confidences
        new_confidences = torch.tensor([0.3, 0.5, 0.7, 0.9])
        calibrated = calibration_system.apply_calibration(new_confidences)
        
        assert calibrated.shape == new_confidences.shape
        assert torch.all(calibrated >= 0.0)
        assert torch.all(calibrated <= 1.0)
    
    def test_calibration_metrics_computation(self, calibration_system, sample_data):
        """Test computation of calibration metrics"""
        predictions, targets, confidences = sample_data
        
        metrics = calibration_system.compute_calibration_metrics(
            predictions, targets, confidences
        )
        
        assert isinstance(metrics, CalibrationMetrics)
        assert 0.0 <= metrics.expected_calibration_error <= 1.0
        assert 0.0 <= metrics.maximum_calibration_error <= 1.0
        assert 0.0 <= metrics.average_confidence <= 1.0
        assert 0.0 <= metrics.average_accuracy <= 1.0
        assert 0.0 <= metrics.brier_score <= 2.0
        
        # Check reliability diagram data
        assert 'bin_centers' in metrics.reliability_diagram
        assert 'accuracies' in metrics.reliability_diagram
        assert 'confidences' in metrics.reliability_diagram
        assert 'counts' in metrics.reliability_diagram
    
    def test_ece_computation(self, calibration_system, sample_data):
        """Test Expected Calibration Error computation"""
        predictions, targets, confidences = sample_data
        
        # Create perfectly calibrated data
        perfect_predictions = torch.eye(4)[targets]
        perfect_confidences = torch.max(perfect_predictions, dim=1)[0]
        
        metrics_perfect = calibration_system.compute_calibration_metrics(
            perfect_predictions, targets, perfect_confidences
        )
        
        metrics_imperfect = calibration_system.compute_calibration_metrics(
            predictions, targets, confidences
        )
        
        # Perfect calibration should have lower ECE
        assert metrics_perfect.expected_calibration_error <= metrics_imperfect.expected_calibration_error
    
    def test_brier_score_computation(self, calibration_system):
        """Test Brier score computation"""
        # Perfect predictions should have Brier score of 0
        n_samples = 100
        n_classes = 3
        targets = torch.randint(0, n_classes, (n_samples,))
        
        # Create perfect probabilistic predictions (one-hot encoded)
        perfect_predictions = torch.zeros(n_samples, n_classes)
        perfect_predictions[range(n_samples), targets] = 1.0
        perfect_confidences = torch.ones(n_samples)
        
        metrics = calibration_system.compute_calibration_metrics(
            perfect_predictions, targets, perfect_confidences
        )
        
        assert metrics.brier_score < 0.01  # Should be very close to 0
    
    def test_calibration_with_empty_bins(self, calibration_system):
        """Test calibration behavior with sparse data"""
        # Create data that will result in empty bins
        predictions = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        confidences = torch.tensor([0.1] * 5 + [0.9] * 5)  # Only two confidence levels
        
        calibration_system.calibrate(predictions, targets, confidences)
        
        assert calibration_system.is_calibrated
        # Should handle empty bins gracefully
        assert len(calibration_system.calibration_map) >= 1


class TestSafetyThresholds:
    """Test cases for SafetyThresholds class"""
    
    @pytest.fixture
    def config(self):
        return UncertaintyConfig(safety_threshold=0.8)
    
    @pytest.fixture
    def safety_thresholds(self, config):
        return SafetyThresholds(config)
    
    def test_initialization(self, safety_thresholds):
        """Test proper initialization of safety thresholds"""
        assert len(safety_thresholds.thresholds) > 0
        assert 'critical_fault' in safety_thresholds.thresholds
        assert 'warning_fault' in safety_thresholds.thresholds
        assert 'normal_operation' in safety_thresholds.thresholds
    
    def test_set_threshold(self, safety_thresholds):
        """Test setting custom thresholds"""
        safety_thresholds.set_threshold('custom_decision', 0.75)
        assert safety_thresholds.get_threshold('custom_decision') == 0.75
        
        # Test invalid threshold
        with pytest.raises(ValueError):
            safety_thresholds.set_threshold('invalid', 1.5)
    
    def test_get_threshold(self, safety_thresholds):
        """Test getting thresholds"""
        # Existing threshold
        threshold = safety_thresholds.get_threshold('critical_fault')
        assert 0.0 <= threshold <= 1.0
        
        # Non-existing threshold should return default
        default_threshold = safety_thresholds.get_threshold('non_existing')
        assert default_threshold == safety_thresholds.config.safety_threshold
    
    def test_is_decision_safe(self, safety_thresholds):
        """Test safety decision logic"""
        # High confidence should be safe for critical decisions
        assert safety_thresholds.is_decision_safe('critical_fault', 0.98)
        
        # Low confidence should not be safe for critical decisions
        assert not safety_thresholds.is_decision_safe('critical_fault', 0.5)
        
        # Moderate confidence might be safe for less critical decisions
        assert safety_thresholds.is_decision_safe('normal_operation', 0.75)
    
    def test_get_safety_margin(self, safety_thresholds):
        """Test safety margin calculation"""
        margin = safety_thresholds.get_safety_margin('critical_fault', 0.98)
        threshold = safety_thresholds.get_threshold('critical_fault')
        expected_margin = 0.98 - threshold
        
        assert abs(margin - expected_margin) < 1e-6
    
    def test_validate_thresholds(self, safety_thresholds):
        """Test threshold validation"""
        validation_results = safety_thresholds.validate_thresholds()
        
        # All default thresholds should be valid
        assert all(validation_results.values())
        
        # Set invalid threshold and test
        safety_thresholds.thresholds['invalid'] = 1.5
        validation_results = safety_thresholds.validate_thresholds()
        assert not validation_results['invalid']
    
    def test_get_recommended_action(self, safety_thresholds):
        """Test recommended action logic"""
        # Critical fault with high confidence
        action, message = safety_thresholds.get_recommended_action('critical', 0.98)
        assert action == 'emergency_shutdown'
        assert 'shutdown' in message.lower()
        
        # Warning fault with moderate confidence
        action, message = safety_thresholds.get_recommended_action('warning', 0.85)
        assert action == 'maintenance_alert'
        assert 'maintenance' in message.lower()
        
        # Normal operation
        action, message = safety_thresholds.get_recommended_action('normal', 0.75)
        assert action == 'continue_operation'
        
        # Low confidence case
        action, message = safety_thresholds.get_recommended_action('critical', 0.5)
        assert action == 'uncertain'
        assert 'insufficient' in message.lower()
    
    def test_threshold_hierarchy(self, safety_thresholds):
        """Test that critical thresholds are higher than warning thresholds"""
        critical_threshold = safety_thresholds.get_threshold('critical_fault')
        warning_threshold = safety_thresholds.get_threshold('warning_fault')
        normal_threshold = safety_thresholds.get_threshold('normal_operation')
        
        assert critical_threshold >= warning_threshold
        assert warning_threshold >= normal_threshold


class TestUncertaintyConfig:
    """Test cases for UncertaintyConfig dataclass"""
    
    def test_default_initialization(self):
        """Test default configuration values"""
        config = UncertaintyConfig()
        
        assert config.prior_mean == 0.0
        assert config.prior_std == 1.0
        assert config.num_mc_samples == 100
        assert config.kl_weight == 1e-3
        assert config.calibration_bins == 15
        assert config.safety_threshold == 0.8
        assert len(config.confidence_levels) == 3
    
    def test_custom_initialization(self):
        """Test custom configuration values"""
        custom_levels = [0.5, 0.8, 0.95]
        config = UncertaintyConfig(
            prior_mean=1.0,
            prior_std=0.5,
            confidence_levels=custom_levels
        )
        
        assert config.prior_mean == 1.0
        assert config.prior_std == 0.5
        assert config.confidence_levels == custom_levels


if __name__ == '__main__':
    pytest.main([__file__])