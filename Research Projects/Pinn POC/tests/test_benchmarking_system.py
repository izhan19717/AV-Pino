"""
Integration test for the complete benchmarking and validation system.

This test demonstrates the full benchmarking system including baseline comparisons,
physics consistency validation, and comprehensive performance evaluation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.validation.benchmarking_suite import BenchmarkingSuite
from src.validation.baseline_comparisons import BaselineComparator
from src.validation.physics_validation import PhysicsConsistencyValidator
from src.validation.generalization_testing import GeneralizationTester
from src.inference.fault_classifier import FaultClassificationSystem, FaultPrediction, FaultType
from src.physics.constraints import PhysicsConstraintLayer


class TestBenchmarkingSystem:
    """Test the complete benchmarking and validation system."""
    
    @pytest.fixture
    def mock_model(self):
        """Create comprehensive mock model."""
        model = Mock(spec=FaultClassificationSystem)
        model.fault_mapper = Mock()
        model.fault_mapper.index_to_type.return_value = FaultType.NORMAL
        model.fault_mapper.n_classes = 4
        model.classifier = Mock()
        model.classifier.parameters.return_value = [torch.randn(256, 128), torch.randn(128, 64), torch.randn(64, 4)]
        model.classifier.eval = Mock()
        
        # Mock predict to return appropriate number of predictions
        def mock_predict(data):
            batch_size = data.shape[0]
            return [
                FaultPrediction(
                    fault_type=FaultType.NORMAL if i % 4 == 0 else 
                              FaultType.INNER_RACE if i % 4 == 1 else
                              FaultType.OUTER_RACE if i % 4 == 2 else FaultType.BALL,
                    confidence=0.85 + 0.1 * np.random.random(),
                    uncertainty=0.1 + 0.05 * np.random.random(),
                    probabilities={
                        FaultType.NORMAL: 0.7 if i % 4 == 0 else 0.1,
                        FaultType.INNER_RACE: 0.7 if i % 4 == 1 else 0.1,
                        FaultType.OUTER_RACE: 0.7 if i % 4 == 2 else 0.1,
                        FaultType.BALL: 0.7 if i % 4 == 3 else 0.1
                    },
                    physics_consistency=0.9 + 0.05 * np.random.random()
                )
                for i in range(batch_size)
            ]
        model.predict.side_effect = mock_predict
        return model
    
    @pytest.fixture
    def mock_physics_constraints(self):
        """Create mock physics constraints."""
        constraints = Mock(spec=PhysicsConstraintLayer)
        
        def mock_forward(prediction, input_data, coords):
            batch_size = prediction.shape[0]
            residuals = {
                'maxwell': torch.tensor(0.01 + 0.005 * np.random.random()),
                'heat_equation': torch.tensor(0.005 + 0.002 * np.random.random()),
                'structural_dynamics': torch.tensor(0.02 + 0.01 * np.random.random()),
                'total_physics_loss': torch.tensor(0.035 + 0.01 * np.random.random())
            }
            return prediction, residuals
        
        constraints.side_effect = mock_forward
        return constraints
    
    @pytest.fixture
    def benchmarking_suite(self, mock_model, mock_physics_constraints):
        """Create benchmarking suite with mocked components."""
        return BenchmarkingSuite(mock_model, mock_physics_constraints)
    
    def test_complete_benchmarking_system_integration(self, benchmarking_suite):
        """Test complete benchmarking system integration."""
        # Create test data for multiple configurations
        test_data = {
            'baseline_cwru': torch.randn(50, 20),
            'high_power_motor': torch.randn(40, 20),
            'low_power_motor': torch.randn(30, 20),
            'variable_speed': torch.randn(35, 20)
        }
        
        test_labels = {
            'baseline_cwru': torch.randint(0, 4, (50,)),
            'high_power_motor': torch.randint(0, 4, (40,)),
            'low_power_motor': torch.randint(0, 4, (30,)),
            'variable_speed': torch.randint(0, 4, (35,))
        }
        
        # Create baseline training data
        baseline_data = (
            np.random.randn(200, 20),  # X_train
            np.random.randint(0, 4, 200)  # y_train
        )
        
        # Create coordinates for physics validation
        coords = torch.randn(50, 3)  # spatial + temporal coordinates
        
        # Run comprehensive benchmark
        with tempfile.TemporaryDirectory() as temp_dir:
            report = benchmarking_suite.run_comprehensive_benchmark(
                test_data=test_data,
                test_labels=test_labels,
                baseline_data=baseline_data,
                coords=coords,
                save_reports=True,
                output_dir=temp_dir
            )
            
            # Verify report structure
            assert report is not None
            assert report.test_name == "Comprehensive Benchmark"
            assert report.performance_metrics is not None
            assert report.hardware_info is not None
            assert len(report.recommendations) > 0
            
            # Verify performance metrics
            metrics = report.performance_metrics
            assert 0 <= metrics.accuracy <= 1
            assert metrics.inference_latency >= 0
            assert metrics.throughput >= 0
            assert metrics.memory_usage >= 0
            assert metrics.model_size > 0
            assert 0 <= metrics.physics_consistency_score <= 1
            assert 0 <= metrics.constraint_violation_rate <= 1
            assert 0 <= metrics.mean_confidence <= 1
            
            # Verify files were created
            assert os.path.exists(os.path.join(temp_dir, "comprehensive_benchmark_report.json"))
    
    def test_baseline_comparison_integration(self, mock_model):
        """Test baseline comparison integration."""
        comparator = BaselineComparator(mock_model)
        
        # Create training and test data
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 4, 100)
        X_test = np.random.randn(50, 20)
        y_test = np.random.randint(0, 4, 50)
        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        y_test_torch = torch.tensor(y_test, dtype=torch.long)
        
        # Run comprehensive comparison
        comparison_report = comparator.run_comprehensive_comparison(
            X_train, y_train, X_test, y_test, X_test_torch, y_test_torch
        )
        
        # Verify comparison report
        assert comparison_report is not None
        assert comparison_report.av_pino_metrics is not None
        assert len(comparison_report.baseline_metrics) > 0
        assert len(comparison_report.performance_improvements) > 0
        assert len(comparison_report.recommendations) > 0
        
        # Verify AV-PINO metrics
        av_pino_metrics = comparison_report.av_pino_metrics
        assert av_pino_metrics.method_name == "AV-PINO"
        assert 0 <= av_pino_metrics.accuracy <= 1
        assert av_pino_metrics.model_size > 0
        
        # Verify baseline metrics exist
        baseline_methods = ['random_forest', 'svm', 'mlp', 'logistic_regression']
        for method in baseline_methods:
            if method in comparison_report.baseline_metrics:
                baseline_metric = comparison_report.baseline_metrics[method]
                assert baseline_metric.method_name == method
                assert 0 <= baseline_metric.accuracy <= 1
    
    def test_physics_validation_integration(self, mock_physics_constraints):
        """Test physics validation integration."""
        validator = PhysicsConsistencyValidator(mock_physics_constraints)
        
        # Create test data
        predictions = torch.randn(20, 10)
        input_data = torch.randn(20, 6)
        coords = torch.randn(20, 3)
        
        # Run physics validation
        physics_report = validator.validate_physics_consistency(
            predictions, input_data, coords, "Physics Integration Test"
        )
        
        # Verify physics report
        assert physics_report is not None
        assert physics_report.test_name == "Physics Integration Test"
        assert len(physics_report.constraint_metrics) > 0
        assert 0 <= physics_report.overall_consistency_score <= 1
        assert len(physics_report.recommendations) > 0
        
        # Verify constraint metrics
        for constraint_name, metrics in physics_report.constraint_metrics.items():
            assert metrics.constraint_name == constraint_name
            assert metrics.mean_residual >= 0
            assert metrics.max_residual >= 0
            assert 0 <= metrics.consistency_score <= 1
            assert 0 <= metrics.violation_percentage <= 100
    
    def test_generalization_testing_integration(self, mock_model, mock_physics_constraints):
        """Test generalization testing integration."""
        # Test simplified generalization assessment instead of full comprehensive test
        tester = GeneralizationTester(mock_model, mock_physics_constraints)
        
        # Create test data for assessment
        test_data = {
            'motor_config_1': torch.randn(25, 20),
            'motor_config_2': torch.randn(20, 20)
        }
        
        test_labels = {
            'motor_config_1': torch.randint(0, 4, (25,)),
            'motor_config_2': torch.randint(0, 4, (20,))
        }
        
        # Run quick generalization assessment
        assessment_results = tester.assess_model_generalization(test_data, test_labels)
        
        # Verify assessment results
        assert assessment_results is not None
        assert len(assessment_results) == 2
        assert 'motor_config_1' in assessment_results
        assert 'motor_config_2' in assessment_results
        
        # Verify accuracy values
        for config_name, accuracy in assessment_results.items():
            assert 0 <= accuracy <= 1, f"Invalid accuracy for {config_name}: {accuracy}"
    
    def test_regression_detection(self, benchmarking_suite):
        """Test automated regression detection."""
        test_data = {'config1': torch.randn(20, 20)}
        test_labels = {'config1': torch.randint(0, 4, (20,))}
        
        # Test with very lenient thresholds that should definitely pass
        passing_thresholds = {
            'accuracy': 0.1,  # Very low threshold should pass
            'inference_latency': 100.0,  # Very high threshold should pass
            'physics_consistency': 0.1,  # Very low threshold should pass
            'memory_usage': 50000.0  # Very high threshold should pass
        }
        
        passed = benchmarking_suite.run_regression_test(
            test_data, test_labels, passing_thresholds
        )
        
        # Should pass with very lenient thresholds
        assert passed is True
        
        # Test with strict thresholds that should fail
        strict_thresholds = {
            'accuracy': 0.99,  # Very high threshold should fail
            'inference_latency': 0.001,  # Very low threshold should fail
            'physics_consistency': 0.99,  # Very high threshold should fail
            'memory_usage': 1.0  # Very low threshold should fail
        }
        
        passed = benchmarking_suite.run_regression_test(
            test_data, test_labels, strict_thresholds
        )
        
        # Should fail with strict thresholds
        assert passed is False
    
    @patch('src.validation.benchmarking_suite.PerformanceProfiler.get_hardware_info')
    def test_hardware_profiling_integration(self, mock_hardware_info, benchmarking_suite):
        """Test hardware profiling integration."""
        # Mock hardware info
        mock_hardware_info.return_value = {
            'cpu_count': 8,
            'total_memory': 16.0,
            'cuda_available': True,
            'gpu_count': 1,
            'gpu_name': 'Test GPU'
        }
        
        test_data = {'config1': torch.randn(10, 20)}
        test_labels = {'config1': torch.randint(0, 4, (10,))}
        
        # Run benchmark to test hardware profiling
        report = benchmarking_suite.run_comprehensive_benchmark(
            test_data, test_labels, save_reports=False
        )
        
        # Verify hardware info is included
        assert report.hardware_info is not None
        assert 'cpu_count' in report.hardware_info
        assert 'total_memory' in report.hardware_info
        assert 'cuda_available' in report.hardware_info
        
        # Verify hardware info values
        assert report.hardware_info['cpu_count'] == 8
        assert report.hardware_info['total_memory'] == 16.0
        assert report.hardware_info['cuda_available'] is True
    
    def test_recommendation_generation(self, benchmarking_suite):
        """Test recommendation generation based on performance."""
        test_data = {'config1': torch.randn(10, 20)}
        test_labels = {'config1': torch.randint(0, 4, (10,))}
        
        # Run benchmark
        report = benchmarking_suite.run_comprehensive_benchmark(
            test_data, test_labels, save_reports=False
        )
        
        # Verify recommendations are generated
        assert len(report.recommendations) > 0
        
        # Recommendations should be strings
        for recommendation in report.recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0
    
    def test_performance_metrics_consistency(self, benchmarking_suite):
        """Test consistency of performance metrics across runs."""
        test_data = {'config1': torch.randn(20, 20)}
        test_labels = {'config1': torch.randint(0, 4, (20,))}
        
        # Run benchmark multiple times
        reports = []
        for _ in range(3):
            report = benchmarking_suite.run_comprehensive_benchmark(
                test_data, test_labels, save_reports=False
            )
            reports.append(report)
        
        # Verify metrics are consistent (within reasonable bounds)
        accuracies = [r.performance_metrics.accuracy for r in reports]
        model_sizes = [r.performance_metrics.model_size for r in reports]
        
        # Model size should be exactly the same
        assert len(set(model_sizes)) == 1, "Model size should be consistent"
        
        # Accuracy should be reasonably consistent (within 0.2 range)
        accuracy_range = max(accuracies) - min(accuracies)
        assert accuracy_range <= 0.2, f"Accuracy range too large: {accuracy_range}"


if __name__ == '__main__':
    pytest.main([__file__])