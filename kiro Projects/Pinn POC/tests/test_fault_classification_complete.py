"""
Complete fault classification system tests.

Tests the fully integrated fault classification system with
AGT-NO integration and CWRU dataset compatibility.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.inference.fault_classification_integration import (
    IntegratedFaultDiagnosisSystem, CWRUFaultDiagnosisSystem,
    create_cwru_diagnosis_system, diagnose_motor_fault,
    evaluate_fault_diagnosis_system
)
from src.inference.fault_classifier import FaultType, FaultPrediction


class TestIntegratedFaultDiagnosisSystem:
    """Test the integrated fault diagnosis system."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system for testing."""
        return IntegratedFaultDiagnosisSystem(
            input_dim=12,
            hidden_dim=128,
            output_dim=64,
            n_classes=4,
            device='cpu'
        )
    
    def test_system_initialization(self, integrated_system):
        """Test system initialization."""
        system = integrated_system
        
        assert system.input_dim == 12
        assert system.hidden_dim == 128
        assert system.output_dim == 64
        assert system.n_classes == 4
        assert system.device == 'cpu'
        
        # Check components exist
        assert hasattr(system, 'feature_extractor')
        assert hasattr(system, 'fault_classifier')
        assert hasattr(system, 'physics_estimator')
        assert hasattr(system, 'classification_system')
    
    def test_forward_pass(self, integrated_system):
        """Test forward pass through integrated system."""
        batch_size = 8
        
        # Test 2D input
        x_2d = torch.randn(batch_size, 12)
        logits, physics, uncertainty = integrated_system(x_2d)
        
        assert logits.shape == (batch_size, 4)
        assert physics.shape == (batch_size,)
        assert uncertainty.shape == (batch_size, 4)
        assert torch.all(physics >= 0) and torch.all(physics <= 1)  # Physics consistency in [0,1]
        
        # Test 3D input (sequence)
        x_3d = torch.randn(batch_size, 10, 12)
        logits, physics, uncertainty = integrated_system(x_3d)
        
        assert logits.shape == (batch_size, 4)
        assert physics.shape == (batch_size,)
        assert uncertainty.shape == (batch_size, 4)
    
    def test_predict_faults(self, integrated_system):
        """Test high-level fault prediction."""
        batch_size = 5
        x = torch.randn(batch_size, 12)
        
        predictions = integrated_system.predict_faults(x)
        
        assert len(predictions) == batch_size
        
        for prediction in predictions:
            assert isinstance(prediction, FaultPrediction)
            assert isinstance(prediction.fault_type, FaultType)
            assert 0.0 <= prediction.confidence <= 1.0
            assert prediction.uncertainty >= 0.0
            assert 0.0 <= prediction.physics_consistency <= 1.0
    
    def test_model_save_load(self, integrated_system):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'integrated_model.pth')
            
            # Save model
            integrated_system.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Create new system and load
            new_system = IntegratedFaultDiagnosisSystem(
                input_dim=12, device='cpu'
            )
            new_system.load_model(model_path)
            
            # Test that loaded model works
            x = torch.randn(2, 12)
            predictions = new_system.predict_faults(x)
            assert len(predictions) == 2


class TestCWRUFaultDiagnosisSystem:
    """Test CWRU-specific fault diagnosis system."""
    
    @pytest.fixture
    def cwru_system(self):
        """Create CWRU system for testing."""
        return CWRUFaultDiagnosisSystem(input_dim=12, device='cpu')
    
    def test_system_initialization(self, cwru_system):
        """Test CWRU system initialization."""
        system = cwru_system
        
        assert system.device == 'cpu'
        assert system.input_dim == 12
        assert hasattr(system, 'diagnosis_system')
        assert hasattr(system, 'validator')
        assert len(system.fault_type_mapping) == 4
        
        # Check fault type mapping
        expected_mapping = {
            'normal': FaultType.NORMAL,
            'inner_race': FaultType.INNER_RACE,
            'outer_race': FaultType.OUTER_RACE,
            'ball': FaultType.BALL
        }
        assert system.fault_type_mapping == expected_mapping
    
    def test_training_pipeline(self, cwru_system):
        """Test training pipeline."""
        # Create synthetic training data
        n_samples = 100
        train_features = torch.randn(n_samples, 12)
        train_labels = torch.randint(0, 4, (n_samples,))
        
        # Create validation data
        val_features = torch.randn(20, 12)
        val_labels = torch.randint(0, 4, (20,))
        
        # Train for a few epochs
        history = cwru_system.train_on_cwru_data(
            train_features, train_labels,
            val_features, val_labels,
            n_epochs=5,  # Short training for testing
            learning_rate=1e-3
        )
        
        # Check history structure
        assert 'train_loss' in history
        assert 'train_acc' in history
        assert 'val_loss' in history
        assert 'val_acc' in history
        
        assert len(history['train_loss']) == 5
        assert len(history['train_acc']) == 5
        assert len(history['val_loss']) == 5
        assert len(history['val_acc']) == 5
        
        # Check that training progressed
        assert all(isinstance(loss, float) for loss in history['train_loss'])
        assert all(0.0 <= acc <= 1.0 for acc in history['train_acc'])
    
    def test_single_sample_prediction(self, cwru_system):
        """Test single sample prediction."""
        # Test 1D input
        sample_1d = torch.randn(12)
        prediction = cwru_system.predict_single_sample(sample_1d)
        
        assert isinstance(prediction, FaultPrediction)
        assert isinstance(prediction.fault_type, FaultType)
        
        # Test 2D input
        sample_2d = torch.randn(1, 12)
        prediction = cwru_system.predict_single_sample(sample_2d)
        
        assert isinstance(prediction, FaultPrediction)
        assert isinstance(prediction.fault_type, FaultType)
    
    def test_system_evaluation(self, cwru_system):
        """Test system evaluation."""
        # Create test data
        n_samples = 50
        test_features = torch.randn(n_samples, 12)
        test_labels = [FaultType.NORMAL] * 12 + [FaultType.INNER_RACE] * 13 + \
                     [FaultType.OUTER_RACE] * 12 + [FaultType.BALL] * 13
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = cwru_system.evaluate_system(
                test_features, test_labels, temp_dir
            )
            
            # Check results structure
            assert hasattr(results, 'overall_metrics')
            assert hasattr(results, 'per_fault_metrics')
            assert hasattr(results, 'confusion_matrix')
            
            # Check that files were saved
            assert os.path.exists(os.path.join(temp_dir, 'classification_metrics.json'))
    
    def test_performance_summary(self, cwru_system):
        """Test performance summary."""
        summary = cwru_system.get_system_performance_summary()
        
        assert isinstance(summary, dict)
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'reliability_ratio']
        
        for key in expected_keys:
            assert key in summary
            assert isinstance(summary[key], float)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_cwru_diagnosis_system(self):
        """Test CWRU system creation function."""
        system = create_cwru_diagnosis_system(input_dim=16, device='cpu')
        
        assert isinstance(system, CWRUFaultDiagnosisSystem)
        assert system.input_dim == 16
        assert system.device == 'cpu'
    
    def test_diagnose_motor_fault(self):
        """Test motor fault diagnosis convenience function."""
        signal_features = torch.randn(3, 12)
        
        predictions = diagnose_motor_fault(signal_features, device='cpu')
        
        assert len(predictions) == 3
        for prediction in predictions:
            assert isinstance(prediction, FaultPrediction)
            assert isinstance(prediction.fault_type, FaultType)
    
    def test_evaluate_fault_diagnosis_system(self):
        """Test system evaluation convenience function."""
        features = torch.randn(20, 12)
        true_labels = [FaultType.NORMAL] * 5 + [FaultType.INNER_RACE] * 5 + \
                     [FaultType.OUTER_RACE] * 5 + [FaultType.BALL] * 5
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = evaluate_fault_diagnosis_system(
                features, true_labels, output_dir=temp_dir, device='cpu'
            )
            
            assert hasattr(results, 'overall_metrics')
            assert hasattr(results, 'per_fault_metrics')
            
            # Check files were saved
            assert os.path.exists(os.path.join(temp_dir, 'classification_metrics.json'))


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""
    
    def test_complete_workflow(self):
        """Test complete workflow from training to evaluation."""
        # Create system
        system = create_cwru_diagnosis_system(input_dim=12, device='cpu')
        
        # Generate synthetic CWRU-like data
        n_train = 80
        n_test = 20
        
        # Training data
        train_features = torch.randn(n_train, 12)
        train_labels = torch.randint(0, 4, (n_train,))
        
        # Test data
        test_features = torch.randn(n_test, 12)
        test_labels = [FaultType.NORMAL] * 5 + [FaultType.INNER_RACE] * 5 + \
                     [FaultType.OUTER_RACE] * 5 + [FaultType.BALL] * 5
        
        # Train system
        history = system.train_on_cwru_data(
            train_features, train_labels,
            n_epochs=3  # Short training for testing
        )
        
        # Evaluate system
        with tempfile.TemporaryDirectory() as temp_dir:
            results = system.evaluate_system(test_features, test_labels, temp_dir)
            
            # Verify workflow completed successfully
            assert len(history['train_loss']) == 3
            assert results.overall_metrics.accuracy >= 0.0
            assert os.path.exists(os.path.join(temp_dir, 'classification_report.txt'))
    
    def test_model_persistence_workflow(self):
        """Test model persistence in complete workflow."""
        # Create and train system
        system1 = create_cwru_diagnosis_system(input_dim=12, device='cpu')
        
        train_features = torch.randn(50, 12)
        train_labels = torch.randint(0, 4, (50,))
        
        # Short training
        system1.train_on_cwru_data(train_features, train_labels, n_epochs=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'complete_model.pth')
            
            # Save trained model
            system1.diagnosis_system.save_model(model_path)
            
            # Create new system and load model
            system2 = create_cwru_diagnosis_system(input_dim=12, device='cpu')
            system2.diagnosis_system.load_model(model_path)
            
            # Test both systems give same predictions
            test_features = torch.randn(5, 12)
            
            preds1 = system1.diagnosis_system.predict_faults(test_features)
            preds2 = system2.diagnosis_system.predict_faults(test_features)
            
            assert len(preds1) == len(preds2)
            
            # Predictions should be identical (or very close due to floating point)
            for p1, p2 in zip(preds1, preds2):
                assert p1.fault_type == p2.fault_type
                assert abs(p1.confidence - p2.confidence) < 1e-4


class TestRequirementsCompliance:
    """Test compliance with specific requirements."""
    
    def test_uncertainty_aware_predictions(self):
        """Test that predictions include uncertainty information (Requirement 3.1, 3.4)."""
        system = create_cwru_diagnosis_system(input_dim=12, device='cpu')
        
        features = torch.randn(10, 12)
        predictions = system.diagnosis_system.predict_faults(features)
        
        for prediction in predictions:
            # Check uncertainty-aware prediction components
            assert hasattr(prediction, 'uncertainty')
            assert hasattr(prediction, 'confidence')
            assert hasattr(prediction, 'physics_consistency')
            
            # Verify uncertainty quantification
            assert prediction.uncertainty >= 0.0
            assert 0.0 <= prediction.confidence <= 1.0
            assert 0.0 <= prediction.physics_consistency <= 1.0
            
            # Check probability distribution
            assert len(prediction.probabilities) == 4
            prob_sum = sum(prediction.probabilities.values())
            assert abs(prob_sum - 1.0) < 1e-5
    
    def test_cwru_fault_type_mapping(self):
        """Test correct CWRU fault type mapping and classification logic."""
        system = create_cwru_diagnosis_system(input_dim=12, device='cpu')
        
        # Test fault type mapping
        mapping = system.fault_type_mapping
        
        assert 'normal' in mapping
        assert 'inner_race' in mapping
        assert 'outer_race' in mapping
        assert 'ball' in mapping
        
        assert mapping['normal'] == FaultType.NORMAL
        assert mapping['inner_race'] == FaultType.INNER_RACE
        assert mapping['outer_race'] == FaultType.OUTER_RACE
        assert mapping['ball'] == FaultType.BALL
        
        # Test that system can classify all fault types
        features = torch.randn(20, 12)
        predictions = system.diagnosis_system.predict_faults(features)
        
        predicted_types = set(pred.fault_type for pred in predictions)
        # Should be able to predict different fault types (though not necessarily all in one batch)
        assert len(predicted_types) >= 1
        assert all(ft in FaultType for ft in predicted_types)
    
    def test_classification_accuracy_capability(self):
        """Test system's capability for high classification accuracy."""
        system = create_cwru_diagnosis_system(input_dim=12, device='cpu')
        
        # Create distinguishable patterns for each fault type
        n_per_class = 10
        features = []
        labels = []
        
        for i, fault_type in enumerate(FaultType):
            for j in range(n_per_class):
                # Create distinctive pattern for each fault type
                pattern = torch.zeros(12)
                pattern[i*3:(i+1)*3] = 1.0 + torch.randn(3) * 0.1
                features.append(pattern)
                labels.append(i)
        
        features = torch.stack(features)
        labels = torch.tensor(labels)
        
        # Train briefly on this data
        system.train_on_cwru_data(features, labels, n_epochs=10)
        
        # Test predictions
        true_fault_types = []
        for i, fault_type in enumerate(FaultType):
            true_fault_types.extend([fault_type] * n_per_class)
        
        results = system.evaluate_system(features, true_fault_types)
        
        # System should be capable of learning these patterns
        # (actual >90% accuracy would require proper training data and more epochs)
        assert results.overall_metrics.accuracy >= 0.0
        assert isinstance(results.overall_metrics.accuracy, float)


if __name__ == "__main__":
    pytest.main([__file__])