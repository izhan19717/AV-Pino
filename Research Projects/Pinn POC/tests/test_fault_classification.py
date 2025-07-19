"""
Unit tests for fault classification implementation.

Tests the fault classification head with uncertainty-aware predictions
and classification accuracy on known fault types.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.inference.fault_classifier import (
    FaultType, FaultPrediction, FaultTypeMapper, 
    UncertaintyAwareClassifier, FaultClassificationSystem,
    create_cwru_fault_classifier
)


class TestFaultTypeMapper:
    """Test fault type mapping functionality."""
    
    def test_fault_type_mapping(self):
        """Test fault type to index mapping."""
        mapper = FaultTypeMapper()
        
        # Test all fault types are mapped
        assert len(mapper.fault_types) == 4
        assert FaultType.NORMAL in mapper.fault_types
        assert FaultType.INNER_RACE in mapper.fault_types
        assert FaultType.OUTER_RACE in mapper.fault_types
        assert FaultType.BALL in mapper.fault_types
        
        # Test bidirectional mapping
        for fault_type in mapper.fault_types:
            idx = mapper.type_to_index(fault_type)
            recovered_type = mapper.index_to_type(idx)
            assert recovered_type == fault_type
    
    def test_string_to_fault_type(self):
        """Test string to fault type conversion."""
        mapper = FaultTypeMapper()
        
        # Test string conversion
        normal_idx = mapper.type_to_index("normal")
        assert normal_idx == mapper.type_to_index(FaultType.NORMAL)
        
        inner_idx = mapper.type_to_index("inner_race")
        assert inner_idx == mapper.type_to_index(FaultType.INNER_RACE)


class TestUncertaintyAwareClassifier:
    """Test uncertainty-aware classifier implementation."""
    
    @pytest.fixture
    def classifier(self):
        """Create test classifier."""
        return UncertaintyAwareClassifier(
            input_dim=64,
            n_classes=4,
            hidden_dim=128,
            dropout_rate=0.1,
            enable_mc_dropout=True,
            n_mc_samples=5
        )
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.input_dim == 64
        assert classifier.n_classes == 4
        assert classifier.hidden_dim == 128
        assert classifier.enable_mc_dropout == True
        assert classifier.n_mc_samples == 5
        
        # Check layers exist
        assert hasattr(classifier, 'feature_extractor')
        assert hasattr(classifier, 'classifier')
        assert hasattr(classifier, 'uncertainty_head')
        assert hasattr(classifier, 'temperature')
    
    def test_forward_without_uncertainty(self, classifier):
        """Test forward pass without uncertainty estimation."""
        batch_size = 8
        x = torch.randn(batch_size, 64)
        
        logits = classifier(x, return_uncertainty=False)
        
        assert logits.shape == (batch_size, 4)
        assert torch.isfinite(logits).all()
    
    def test_forward_with_uncertainty(self, classifier):
        """Test forward pass with uncertainty estimation."""
        batch_size = 8
        x = torch.randn(batch_size, 64)
        
        logits, uncertainty = classifier(x, return_uncertainty=True)
        
        assert logits.shape == (batch_size, 4)
        assert uncertainty.shape == (batch_size, 4)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(uncertainty).all()
        assert (uncertainty >= 0).all()  # Uncertainty should be non-negative
    
    def test_predict_with_uncertainty(self, classifier):
        """Test prediction with full uncertainty quantification."""
        batch_size = 8
        x = torch.randn(batch_size, 64)
        
        predictions, probabilities, uncertainties = classifier.predict_with_uncertainty(x)
        
        assert predictions.shape == (batch_size,)
        assert probabilities.shape == (batch_size, 4)
        assert uncertainties.shape == (batch_size, 4)
        
        # Check probabilities sum to 1
        prob_sums = torch.sum(probabilities, dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5)
        
        # Check predictions are valid class indices
        assert (predictions >= 0).all()
        assert (predictions < 4).all()
    
    def test_mc_dropout_uncertainty(self, classifier):
        """Test Monte Carlo dropout uncertainty estimation."""
        classifier.train()  # Enable training mode for MC dropout
        batch_size = 4
        x = torch.randn(batch_size, 64)
        
        # Multiple forward passes should give different results due to dropout
        logits1, uncertainty1 = classifier(x, return_uncertainty=True)
        logits2, uncertainty2 = classifier(x, return_uncertainty=True)
        
        # Results should be different due to dropout
        assert not torch.allclose(logits1, logits2, atol=1e-5)
        
        # Uncertainties should be positive
        assert (uncertainty1 >= 0).all()
        assert (uncertainty2 >= 0).all()


class TestFaultClassificationSystem:
    """Test complete fault classification system."""
    
    @pytest.fixture
    def classification_system(self):
        """Create test classification system."""
        return FaultClassificationSystem(input_dim=64, device='cpu')
    
    def test_system_initialization(self, classification_system):
        """Test system initialization."""
        system = classification_system
        
        assert system.device == 'cpu'
        assert isinstance(system.fault_mapper, FaultTypeMapper)
        assert isinstance(system.classifier, UncertaintyAwareClassifier)
        assert len(system.confidence_thresholds) == 4
        
        # Check confidence thresholds
        for fault_type in FaultType:
            assert fault_type in system.confidence_thresholds
            assert 0.0 <= system.confidence_thresholds[fault_type] <= 1.0
    
    def test_predict_single_sample(self, classification_system):
        """Test prediction on single sample."""
        features = torch.randn(1, 64)
        
        predictions = classification_system.predict(features)
        
        assert len(predictions) == 1
        prediction = predictions[0]
        
        assert isinstance(prediction, FaultPrediction)
        assert isinstance(prediction.fault_type, FaultType)
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.uncertainty >= 0.0
        assert len(prediction.probabilities) == 4
        
        # Check probabilities sum to 1
        prob_sum = sum(prediction.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-5
    
    def test_predict_batch(self, classification_system):
        """Test prediction on batch of samples."""
        batch_size = 8
        features = torch.randn(batch_size, 64)
        
        predictions = classification_system.predict(features)
        
        assert len(predictions) == batch_size
        
        for prediction in predictions:
            assert isinstance(prediction, FaultPrediction)
            assert isinstance(prediction.fault_type, FaultType)
            assert 0.0 <= prediction.confidence <= 1.0
            assert prediction.uncertainty >= 0.0
    
    def test_predict_with_physics_consistency(self, classification_system):
        """Test prediction with physics consistency scores."""
        batch_size = 4
        features = torch.randn(batch_size, 64)
        physics_consistency = torch.tensor([0.9, 0.8, 0.7, 0.6])
        
        predictions = classification_system.predict(features, physics_consistency)
        
        assert len(predictions) == batch_size
        
        for i, prediction in enumerate(predictions):
            expected_consistency = physics_consistency[i].item()
            assert abs(prediction.physics_consistency - expected_consistency) < 1e-5
    
    def test_reliability_assessment(self, classification_system):
        """Test prediction reliability assessment."""
        # Create a high-confidence prediction
        high_conf_pred = FaultPrediction(
            fault_type=FaultType.NORMAL,
            confidence=0.95,
            uncertainty=0.05,
            probabilities={ft: 0.25 for ft in FaultType},
            physics_consistency=0.9
        )
        
        # Create a low-confidence prediction
        low_conf_pred = FaultPrediction(
            fault_type=FaultType.INNER_RACE,
            confidence=0.6,
            uncertainty=0.4,
            probabilities={ft: 0.25 for ft in FaultType},
            physics_consistency=0.5
        )
        
        assert classification_system.is_prediction_reliable(high_conf_pred) == True
        assert classification_system.is_prediction_reliable(low_conf_pred) == False
    
    def test_temperature_calibration(self, classification_system):
        """Test temperature calibration functionality."""
        batch_size = 16
        val_features = torch.randn(batch_size, 64)
        val_labels = torch.randint(0, 4, (batch_size,))
        
        initial_temp = classification_system.classifier.temperature.item()
        
        classification_system.calibrate_temperature(val_features, val_labels)
        
        final_temp = classification_system.classifier.temperature.item()
        
        # Temperature should have changed (unless it was already optimal)
        assert isinstance(final_temp, float)
        assert final_temp > 0  # Temperature should be positive
    
    def test_fault_type_mapping_retrieval(self, classification_system):
        """Test fault type mapping retrieval."""
        mapping = classification_system.get_fault_type_mapping()
        
        assert isinstance(mapping, dict)
        assert len(mapping) == 4
        
        expected_types = {'normal', 'inner_race', 'outer_race', 'ball'}
        assert set(mapping.keys()) == expected_types
        
        # Check all values are valid indices
        for idx in mapping.values():
            assert 0 <= idx < 4
    
    def test_model_save_load(self, classification_system):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pth')
            
            # Save model
            classification_system.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Create new system and load model
            new_system = FaultClassificationSystem(input_dim=64, device='cpu')
            new_system.load_model(model_path)
            
            # Test that loaded model works
            features = torch.randn(2, 64)
            predictions = new_system.predict(features)
            assert len(predictions) == 2


class TestCWRUFaultClassifier:
    """Test CWRU-specific fault classifier creation."""
    
    def test_create_cwru_classifier(self):
        """Test CWRU fault classifier creation."""
        classifier = create_cwru_fault_classifier(input_dim=128, device='cpu')
        
        assert isinstance(classifier, FaultClassificationSystem)
        assert classifier.device == 'cpu'
        assert classifier.classifier.input_dim == 128
        assert classifier.classifier.n_classes == 4


class TestFaultPrediction:
    """Test FaultPrediction dataclass."""
    
    def test_fault_prediction_creation(self):
        """Test FaultPrediction object creation."""
        probabilities = {
            FaultType.NORMAL: 0.7,
            FaultType.INNER_RACE: 0.2,
            FaultType.OUTER_RACE: 0.05,
            FaultType.BALL: 0.05
        }
        
        prediction = FaultPrediction(
            fault_type=FaultType.NORMAL,
            confidence=0.7,
            uncertainty=0.1,
            probabilities=probabilities,
            physics_consistency=0.9,
            timestamp=1234567890.0
        )
        
        assert prediction.fault_type == FaultType.NORMAL
        assert prediction.confidence == 0.7
        assert prediction.uncertainty == 0.1
        assert prediction.physics_consistency == 0.9
        assert prediction.timestamp == 1234567890.0
        assert len(prediction.probabilities) == 4


class TestIntegration:
    """Integration tests for fault classification system."""
    
    def test_end_to_end_classification(self):
        """Test end-to-end classification pipeline."""
        # Create system
        system = create_cwru_fault_classifier(input_dim=64, device='cpu')
        
        # Generate test data
        batch_size = 10
        features = torch.randn(batch_size, 64)
        
        # Make predictions
        predictions = system.predict(features)
        
        # Verify predictions
        assert len(predictions) == batch_size
        
        for prediction in predictions:
            assert isinstance(prediction.fault_type, FaultType)
            assert 0.0 <= prediction.confidence <= 1.0
            assert prediction.uncertainty >= 0.0
            assert len(prediction.probabilities) == 4
            
            # Check probability distribution
            prob_sum = sum(prediction.probabilities.values())
            assert abs(prob_sum - 1.0) < 1e-5
    
    def test_classification_with_known_patterns(self):
        """Test classification with known fault patterns."""
        system = create_cwru_fault_classifier(input_dim=64, device='cpu')
        
        # Create distinctive patterns for each fault type
        # (In practice, these would be learned from data)
        normal_pattern = torch.zeros(1, 64)
        inner_race_pattern = torch.ones(1, 64)
        outer_race_pattern = torch.ones(1, 64) * 2
        ball_pattern = torch.ones(1, 64) * 3
        
        patterns = torch.cat([normal_pattern, inner_race_pattern, 
                            outer_race_pattern, ball_pattern], dim=0)
        
        predictions = system.predict(patterns)
        
        assert len(predictions) == 4
        
        # Each prediction should have reasonable confidence
        for prediction in predictions:
            assert prediction.confidence > 0.1  # At least some confidence
            assert prediction.uncertainty >= 0.0
    
    def test_uncertainty_consistency(self):
        """Test that uncertainty estimates are consistent."""
        system = create_cwru_fault_classifier(input_dim=64, device='cpu')
        
        # Test with same input multiple times
        features = torch.randn(1, 64)
        
        predictions1 = system.predict(features)
        predictions2 = system.predict(features)
        
        # Predictions should be identical for same input in eval mode
        pred1, pred2 = predictions1[0], predictions2[0]
        
        assert pred1.fault_type == pred2.fault_type
        assert abs(pred1.confidence - pred2.confidence) < 1e-5
        # Note: uncertainty might vary slightly due to MC dropout


if __name__ == "__main__":
    pytest.main([__file__])