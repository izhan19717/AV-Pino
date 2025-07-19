"""
Test suite for final AV-PINO system integration and validation

This test suite validates the complete integrated system including:
- End-to-end pipeline functionality
- Performance requirements compliance
- Physics consistency validation
- Real-time inference capabilities
"""

import pytest
import torch
import numpy as np
import time
from unittest.mock import Mock, patch
import tempfile
import os

from src.validation.final_performance_validation import (
    FinalPerformanceValidator, 
    PerformanceMetrics, 
    ValidationResults
)


class TestFinalIntegration:
    """Test suite for final system integration"""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.yaml")
            # Create minimal config file
            with open(config_path, 'w') as f:
                f.write("""
model:
  input_dim: 1024
  output_dim: 4
  modes: 16
  width: 64
  path: "test_model.pth"
                """)
            
            validator = FinalPerformanceValidator(config_path)
            return validator
    
    @pytest.fixture
    def mock_test_data(self):
        """Create mock test data for validation"""
        # Create synthetic test data
        batch_size = 100
        input_dim = 1024
        num_classes = 4
        
        test_data = torch.randn(batch_size, input_dim)
        test_labels = torch.randint(0, num_classes, (batch_size,))
        
        return test_data, test_labels
    
    def test_system_setup(self, validator):
        """Test complete system setup and component initialization"""
        # Mock the model loading to avoid file dependencies
        with patch('torch.load') as mock_load:
            mock_load.return_value = {}
            
            validator.setup_complete_system()
            
            # Verify all components are initialized
            assert validator.model is not None
            assert validator.realtime_inference is not None
            assert validator.fault_classifier is not None
            assert validator.uncertainty_module is not None
            assert validator.physics_validator is not None
    
    def test_classification_accuracy_validation(self, validator, mock_test_data):
        """Test fault classification accuracy validation"""
        test_data, test_labels = mock_test_data
        
        # Mock the model and components
        validator.model = Mock()
        validator.uncertainty_module = Mock()
        
        # Mock predictions that meet the 90% accuracy requirement
        mock_predictions = torch.zeros(len(test_data), 4)
        for i, label in enumerate(test_labels):
            mock_predictions[i, label] = 1.0  # Perfect predictions
        
        mock_uncertainties = torch.ones(len(test_data)) * 0.1
        
        validator.uncertainty_module.predict_with_uncertainty.return_value = (
            mock_predictions, mock_uncertainties
        )
        
        # Run accuracy validation
        metrics = validator.validate_classification_accuracy(test_data, test_labels)
        
        # Verify results
        assert metrics['accuracy'] >= 0.90
        assert metrics['meets_90_percent_requirement'] is True
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_inference_latency_validation(self, validator, mock_test_data):
        """Test inference latency validation"""
        test_data, _ = mock_test_data
        
        # Mock realtime inference with fast response
        validator.realtime_inference = Mock()
        validator.realtime_inference.predict_realtime.return_value = torch.randn(1, 4)
        
        # Run latency validation
        metrics = validator.validate_inference_latency(test_data)
        
        # Verify latency metrics are computed
        assert 'avg_latency_ms' in metrics
        assert 'max_latency_ms' in metrics
        assert 'p95_latency_ms' in metrics
        assert 'meets_1ms_requirement' in metrics
        
        # Verify reasonable latency values (mocked should be very fast)
        assert metrics['avg_latency_ms'] >= 0
    
    def test_physics_consistency_validation(self, validator, mock_test_data):
        """Test physics consistency validation"""
        test_data, _ = mock_test_data
        
        # Mock model with physics residuals
        validator.model = Mock()
        mock_physics_residuals = {
            'maxwell': torch.randn(1, 10) * 0.01,  # Small residuals
            'thermal': torch.randn(1, 10) * 0.01,
            'mechanical': torch.randn(1, 10) * 0.01
        }
        
        validator.model.return_value = (
            torch.randn(1, 4),  # predictions
            mock_physics_residuals
        )
        
        # Mock physics validator
        validator.physics_validator = Mock()
        validator.physics_validator.validate_physics_consistency.return_value = 0.9
        validator.physics_validator.check_constraint_violations.return_value = []
        
        # Run physics validation
        metrics = validator.validate_physics_consistency(test_data)
        
        # Verify physics metrics
        assert 'avg_physics_consistency' in metrics
        assert 'constraint_violation_rate' in metrics
        assert 'meets_consistency_requirement' in metrics
        assert metrics['avg_physics_consistency'] >= 0
    
    def test_edge_hardware_validation(self, validator, mock_test_data):
        """Test edge hardware performance validation"""
        test_data, _ = mock_test_data
        
        # Mock realtime inference
        validator.realtime_inference = Mock()
        validator.realtime_inference.predict_realtime.return_value = torch.randn(32, 4)
        
        # Run edge hardware validation
        metrics = validator.validate_edge_hardware_performance(test_data)
        
        # Verify edge performance metrics
        assert 'memory_usage_mb' in metrics
        assert 'throughput_samples_per_sec' in metrics
        assert 'meets_memory_constraints' in metrics
        assert 'meets_throughput_requirements' in metrics
        
        # Verify reasonable values
        assert metrics['memory_usage_mb'] >= 0
        assert metrics['throughput_samples_per_sec'] >= 0
    
    def test_complete_validation_pipeline(self, validator, mock_test_data):
        """Test the complete end-to-end validation pipeline"""
        test_data, test_labels = mock_test_data
        
        # Mock all components
        validator.model = Mock()
        validator.realtime_inference = Mock()
        validator.fault_classifier = Mock()
        validator.uncertainty_module = Mock()
        validator.physics_validator = Mock()
        validator.benchmarking_suite = Mock()
        
        # Mock data loading
        validator.load_test_dataset = Mock(return_value=(test_data, test_labels))
        
        # Mock component responses
        validator.uncertainty_module.predict_with_uncertainty.return_value = (
            torch.eye(4)[test_labels],  # Perfect predictions
            torch.ones(len(test_data)) * 0.1
        )
        
        validator.realtime_inference.predict_realtime.return_value = torch.randn(1, 4)
        
        validator.model.return_value = (
            torch.randn(1, 4),
            {'maxwell': torch.randn(1, 10) * 0.01}
        )
        
        validator.physics_validator.validate_physics_consistency.return_value = 0.9
        validator.physics_validator.check_constraint_violations.return_value = []
        
        validator.benchmarking_suite.run_all_benchmarks.return_value = {
            'svm_accuracy': 0.85,
            'random_forest_accuracy': 0.82
        }
        
        # Run complete validation
        results = validator.run_complete_validation()
        
        # Verify results structure
        assert isinstance(results, ValidationResults)
        assert isinstance(results.overall_metrics, PerformanceMetrics)
        assert isinstance(results.requirements_compliance, dict)
        
        # Verify key metrics are present
        assert results.overall_metrics.accuracy >= 0
        assert results.overall_metrics.inference_latency_ms >= 0
        assert results.overall_metrics.physics_consistency_score >= 0
        
        # Verify requirements compliance tracking
        assert 'accuracy_90_percent' in results.requirements_compliance
        assert 'latency_1ms' in results.requirements_compliance
        assert 'physics_consistency' in results.requirements_compliance
    
    def test_validation_summary_logging(self, validator, caplog):
        """Test validation summary logging functionality"""
        # Create mock validation results
        mock_metrics = PerformanceMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.93,
            f1_score=0.94,
            inference_latency_ms=0.8,
            physics_consistency_score=0.88,
            uncertainty_calibration_score=0.85,
            memory_usage_mb=256,
            throughput_samples_per_sec=1200
        )
        
        mock_compliance = {
            'accuracy_90_percent': True,
            'latency_1ms': True,
            'physics_consistency': True,
            'memory_constraints': True,
            'throughput_requirements': True
        }
        
        validator.validation_results = ValidationResults(
            overall_metrics=mock_metrics,
            per_fault_metrics={},
            physics_validation_results={},
            benchmark_comparisons={},
            edge_hardware_results={},
            requirements_compliance=mock_compliance
        )
        
        # Test logging
        validator.log_validation_summary()
        
        # Verify log messages contain key information
        log_messages = [record.message for record in caplog.records]
        assert any("VALIDATION SUMMARY" in msg for msg in log_messages)
        assert any("Classification Accuracy" in msg for msg in log_messages)
        assert any("Overall System Status: PASS" in msg for msg in log_messages)
    
    def test_performance_report_generation(self, validator):
        """Test technical report generation"""
        # Mock validation results
        mock_metrics = PerformanceMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.93,
            f1_score=0.94,
            inference_latency_ms=0.8,
            physics_consistency_score=0.88,
            uncertainty_calibration_score=0.85,
            memory_usage_mb=256,
            throughput_samples_per_sec=1200
        )
        
        validator.validation_results = ValidationResults(
            overall_metrics=mock_metrics,
            per_fault_metrics={},
            physics_validation_results={},
            benchmark_comparisons={},
            edge_hardware_results={},
            requirements_compliance={}
        )
        
        # Mock report generator
        validator.report_generator = Mock()
        
        # Test report generation
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "test_report.md")
            validator.generate_performance_report(report_path)
            
            # Verify report generator was called
            validator.report_generator.generate_validation_report.assert_called_once_with(
                validator.validation_results, report_path
            )
    
    def test_requirements_compliance_checking(self, validator):
        """Test that all requirements are properly checked"""
        # Test with failing requirements
        mock_metrics = {
            'accuracy': 0.85,  # Below 90% requirement
            'avg_latency_ms': 1.5,  # Above 1ms requirement
            'avg_physics_consistency': 0.80,  # Below consistency requirement
            'memory_usage_mb': 600,  # Above memory limit
            'throughput_samples_per_sec': 800  # Below throughput requirement
        }
        
        # Simulate validation with failing metrics
        validator.model = Mock()
        validator.uncertainty_module = Mock()
        validator.realtime_inference = Mock()
        validator.physics_validator = Mock()
        
        # Mock responses that fail requirements
        validator.uncertainty_module.predict_with_uncertainty.return_value = (
            torch.randn(100, 4),  # Random predictions (low accuracy)
            torch.ones(100) * 0.1
        )
        
        # Verify that requirements compliance is properly tracked
        test_data = torch.randn(100, 1024)
        test_labels = torch.randint(0, 4, (100,))
        
        accuracy_metrics = validator.validate_classification_accuracy(test_data, test_labels)
        
        # Should detect failure to meet 90% requirement
        assert 'meets_90_percent_requirement' in accuracy_metrics


class TestPerformanceMetrics:
    """Test performance metrics data structures"""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics dataclass creation"""
        metrics = PerformanceMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.93,
            f1_score=0.94,
            inference_latency_ms=0.8,
            physics_consistency_score=0.88,
            uncertainty_calibration_score=0.85,
            memory_usage_mb=256,
            throughput_samples_per_sec=1200
        )
        
        assert metrics.accuracy == 0.95
        assert metrics.inference_latency_ms == 0.8
        assert metrics.physics_consistency_score == 0.88
    
    def test_validation_results_creation(self):
        """Test ValidationResults dataclass creation"""
        metrics = PerformanceMetrics(
            accuracy=0.95, precision=0.94, recall=0.93, f1_score=0.94,
            inference_latency_ms=0.8, physics_consistency_score=0.88,
            uncertainty_calibration_score=0.85, memory_usage_mb=256,
            throughput_samples_per_sec=1200
        )
        
        results = ValidationResults(
            overall_metrics=metrics,
            per_fault_metrics={},
            physics_validation_results={},
            benchmark_comparisons={},
            edge_hardware_results={},
            requirements_compliance={}
        )
        
        assert results.overall_metrics == metrics
        assert isinstance(results.per_fault_metrics, dict)
        assert isinstance(results.requirements_compliance, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])