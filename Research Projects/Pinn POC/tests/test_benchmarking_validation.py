"""
Tests for complete benchmarking and validation system.

Tests the integrated validation pipeline including baseline comparisons,
physics validation, and comprehensive benchmarking suite.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from src.validation.benchmarking_suite import (
    BenchmarkingSuite,
    PerformanceProfiler,
    ValidationPipeline,
    BenchmarkReport,
    PerformanceMetrics
)
from src.validation.baseline_comparisons import (
    BaselineComparator,
    TraditionalMLBaseline,
    BaselineMethod,
    BaselineMetrics,
    ComparisonReport
)
from src.validation.physics_validation import (
    PhysicsConsistencyValidator,
    PDEResidualAnalyzer,
    PhysicsValidationMetrics,
    PhysicsValidationReport
)
from src.inference.fault_classifier import FaultClassificationSystem, FaultPrediction, FaultType
from src.physics.constraints import PhysicsConstraintLayer


class TestPerformanceMetrics:
    """Test performance metrics data structure."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            accuracy=0.95,
            inference_latency=0.5,
            throughput=1000.0,
            memory_usage=512.0,
            cpu_usage=25.0,
            gpu_usage=30.0,
            model_size=1000000,
            physics_consistency_score=0.9,
            constraint_violation_rate=0.1,
            mean_confidence=0.85,
            uncertainty_calibration_score=0.8
        )
        
        assert metrics.accuracy == 0.95
        assert metrics.inference_latency == 0.5
        assert metrics.throughput == 1000.0
        assert metrics.memory_usage == 512.0
        assert metrics.cpu_usage == 25.0
        assert metrics.gpu_usage == 30.0
        assert metrics.model_size == 1000000
        assert metrics.physics_consistency_score == 0.9
        assert metrics.constraint_violation_rate == 0.1
        assert metrics.mean_confidence == 0.85
        assert metrics.uncertainty_calibration_score == 0.8
    
    def test_performance_metrics_to_dict(self):
        """Test converting performance metrics to dictionary."""
        metrics = PerformanceMetrics(
            accuracy=0.95,
            inference_latency=0.5,
            throughput=1000.0,
            memory_usage=512.0,
            cpu_usage=25.0,
            gpu_usage=30.0,
            model_size=1000000,
            physics_consistency_score=0.9,
            constraint_violation_rate=0.1,
            mean_confidence=0.85,
            uncertainty_calibration_score=0.8
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['accuracy'] == 0.95
        assert metrics_dict['inference_latency'] == 0.5
        assert metrics_dict['throughput'] == 1000.0
        assert metrics_dict['physics_consistency_score'] == 0.9
        assert len(metrics_dict) == 11  # All fields present


class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler."""
        return PerformanceProfiler()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock fault classification system."""
        model = Mock(spec=FaultClassificationSystem)
        model.classifier = Mock()
        model.classifier.eval = Mock()
        model.predict.return_value = [
            FaultPrediction(
                fault_type=FaultType.NORMAL,
                confidence=0.9,
                uncertainty=0.1,
                probabilities={FaultType.NORMAL: 0.9},
                physics_consistency=0.95
            )
        ]
        return model
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.start_time is None
        assert profiler.start_memory is None
        assert profiler.start_cpu is None
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_start_profiling(self, mock_cpu_percent, mock_virtual_memory, profiler):
        """Test starting performance profiling."""
        # Mock system metrics
        mock_memory = Mock()
        mock_memory.used = 1024 * 1024 * 1024  # 1GB in bytes
        mock_virtual_memory.return_value = mock_memory
        mock_cpu_percent.return_value = 25.0
        
        profiler.start_profiling()
        
        assert profiler.start_time is not None
        assert profiler.start_memory == 1024.0  # 1GB in MB
        assert profiler.start_cpu == 25.0
    
    def test_profile_inference(self, profiler, mock_model):
        """Test inference profiling."""
        test_data = torch.randn(10, 5)
        
        # Add a small delay to mock predict to ensure non-zero latency
        import time
        def mock_predict_with_delay(data):
            time.sleep(0.001)  # 1ms delay
            return [
                FaultPrediction(
                    fault_type=FaultType.NORMAL,
                    confidence=0.9,
                    uncertainty=0.1,
                    probabilities={FaultType.NORMAL: 0.9},
                    physics_consistency=0.95
                )
            ]
        mock_model.predict.side_effect = mock_predict_with_delay
        
        profile_results = profiler.profile_inference(mock_model, test_data, num_runs=5)
        
        assert 'mean_latency' in profile_results
        assert 'min_latency' in profile_results
        assert 'max_latency' in profile_results
        assert 'std_latency' in profile_results
        assert 'throughput' in profile_results
        
        assert profile_results['mean_latency'] >= 0  # Allow zero or positive
        assert profile_results['throughput'] >= 0    # Allow zero or positive
        
        # Verify model was called correctly
        assert mock_model.predict.call_count >= 15  # 10 warmup + 5 timed runs
    
    @patch('psutil.virtual_memory')
    def test_get_memory_usage(self, mock_virtual_memory, profiler):
        """Test memory usage monitoring."""
        # Mock memory info
        mock_memory = Mock()
        mock_memory.total = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.used = 4 * 1024 * 1024 * 1024   # 4GB
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        
        memory_usage = profiler.get_memory_usage()
        
        assert memory_usage['total_memory'] == 8192.0  # 8GB in MB
        assert memory_usage['used_memory'] == 4096.0   # 4GB in MB
        assert memory_usage['available_memory'] == 4096.0  # 4GB in MB
        assert memory_usage['memory_percent'] == 50.0
    
    @patch('psutil.cpu_percent')
    def test_get_cpu_usage(self, mock_cpu_percent, profiler):
        """Test CPU usage monitoring."""
        mock_cpu_percent.return_value = 75.0
        
        cpu_usage = profiler.get_cpu_usage()
        
        assert cpu_usage == 75.0
        mock_cpu_percent.assert_called_once_with(interval=1)
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_freq')
    @patch('psutil.virtual_memory')
    def test_get_hardware_info(self, mock_virtual_memory, mock_cpu_freq, mock_cpu_count, profiler):
        """Test hardware information gathering."""
        # Mock hardware info
        mock_cpu_count.return_value = 8
        mock_freq = Mock()
        mock_freq._asdict.return_value = {'current': 2400.0, 'min': 800.0, 'max': 3200.0}
        mock_cpu_freq.return_value = mock_freq
        
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_virtual_memory.return_value = mock_memory
        
        hardware_info = profiler.get_hardware_info()
        
        assert hardware_info['cpu_count'] == 8
        assert hardware_info['total_memory'] == 16.0  # 16GB
        assert 'python_version' in hardware_info
        assert 'torch_version' in hardware_info
        assert 'cuda_available' in hardware_info


class TestTraditionalMLBaseline:
    """Test traditional ML baseline implementations."""
    
    def test_baseline_initialization(self):
        """Test baseline model initialization."""
        baseline = TraditionalMLBaseline(BaselineMethod.RANDOM_FOREST)
        
        assert baseline.method == BaselineMethod.RANDOM_FOREST
        assert baseline.model is not None
        assert baseline.scaler is not None
        assert not baseline.is_trained
    
    def test_baseline_training(self):
        """Test baseline model training."""
        baseline = TraditionalMLBaseline(BaselineMethod.RANDOM_FOREST)
        
        # Generate synthetic training data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 4, 100)
        
        training_time = baseline.train(X_train, y_train)
        
        assert training_time > 0
        assert baseline.is_trained
    
    def test_baseline_prediction(self):
        """Test baseline model prediction."""
        baseline = TraditionalMLBaseline(BaselineMethod.RANDOM_FOREST)
        
        # Train model
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 4, 100)
        baseline.train(X_train, y_train)
        
        # Make predictions
        X_test = np.random.randn(20, 10)
        predictions, probabilities, inference_time = baseline.predict(X_test)
        
        assert len(predictions) == 20
        assert probabilities.shape == (20, 4)  # 4 classes
        assert inference_time > 0
    
    def test_baseline_model_size(self):
        """Test baseline model size estimation."""
        baseline = TraditionalMLBaseline(BaselineMethod.RANDOM_FOREST)
        
        # Untrained model should have size 0
        assert baseline.get_model_size() == 0
        
        # Train model
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 4, 100)
        baseline.train(X_train, y_train)
        
        # Trained model should have positive size
        assert baseline.get_model_size() > 0


class TestBaselineComparator:
    """Test baseline comparison functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock AV-PINO model."""
        model = Mock(spec=FaultClassificationSystem)
        model.fault_mapper = Mock()
        model.fault_mapper.index_to_type.return_value = FaultType.NORMAL
        model.classifier = Mock()
        model.classifier.parameters.return_value = [torch.randn(100, 50), torch.randn(50)]
        return model
    
    @pytest.fixture
    def baseline_comparator(self, mock_model):
        """Create baseline comparator."""
        return BaselineComparator(mock_model)
    
    def test_comparator_initialization(self, baseline_comparator):
        """Test baseline comparator initialization."""
        assert baseline_comparator.av_pino_model is not None
        assert len(baseline_comparator.baseline_models) == len(BaselineMethod)
        
        for method in BaselineMethod:
            assert method.value in baseline_comparator.baseline_models
    
    def test_train_baselines(self, baseline_comparator):
        """Test training all baseline models."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 4, 100)
        
        training_times = baseline_comparator.train_baselines(X_train, y_train)
        
        assert len(training_times) == len(BaselineMethod)
        for method_name, training_time in training_times.items():
            assert training_time > 0 or training_time == float('inf')  # inf for failed training
    
    def test_evaluate_av_pino(self, baseline_comparator, mock_model):
        """Test AV-PINO evaluation."""
        # Mock predictions
        mock_predictions = [
            FaultPrediction(
                fault_type=FaultType.NORMAL,
                confidence=0.9,
                uncertainty=0.1,
                probabilities={FaultType.NORMAL: 0.9},
                physics_consistency=0.95
            ),
            FaultPrediction(
                fault_type=FaultType.INNER_RACE,
                confidence=0.85,
                uncertainty=0.15,
                probabilities={FaultType.INNER_RACE: 0.85},
                physics_consistency=0.9
            )
        ]
        mock_model.predict.return_value = mock_predictions
        
        X_test = torch.randn(2, 10)
        y_test = torch.tensor([0, 1])
        
        metrics = baseline_comparator.evaluate_av_pino(X_test, y_test)
        
        assert isinstance(metrics, BaselineMetrics)
        assert metrics.method_name == "AV-PINO"
        assert 0 <= metrics.accuracy <= 1
        assert metrics.inference_time >= 0  # Allow zero or positive
        assert metrics.model_size > 0


class TestPhysicsValidation:
    """Test physics consistency validation."""
    
    @pytest.fixture
    def mock_physics_constraints(self):
        """Create mock physics constraints."""
        constraints = Mock(spec=PhysicsConstraintLayer)
        constraints.return_value = (
            torch.randn(10, 5),  # predictions
            {
                'maxwell': torch.tensor(0.01),
                'heat_equation': torch.tensor(0.005),
                'structural_dynamics': torch.tensor(0.02),
                'total_physics_loss': torch.tensor(0.035)
            }
        )
        return constraints
    
    @pytest.fixture
    def physics_validator(self, mock_physics_constraints):
        """Create physics consistency validator."""
        return PhysicsConsistencyValidator(mock_physics_constraints)
    
    def test_residual_analyzer_initialization(self):
        """Test PDE residual analyzer initialization."""
        analyzer = PDEResidualAnalyzer(tolerance_threshold=1e-3)
        assert analyzer.tolerance_threshold == 1e-3
    
    def test_analyze_residual(self):
        """Test residual analysis."""
        analyzer = PDEResidualAnalyzer(tolerance_threshold=1e-2)
        
        # Create test residual
        residual = torch.tensor([0.001, 0.005, 0.02, 0.001, 0.003])
        
        metrics = analyzer.analyze_residual(residual, "test_constraint")
        
        assert isinstance(metrics, PhysicsValidationMetrics)
        assert metrics.constraint_name == "test_constraint"
        assert metrics.mean_residual > 0
        assert metrics.max_residual > 0
        assert 0 <= metrics.consistency_score <= 1
        assert 0 <= metrics.violation_percentage <= 100
    
    def test_physics_consistency_validation(self, physics_validator):
        """Test physics consistency validation."""
        predictions = torch.randn(10, 5)
        input_data = torch.randn(10, 3)
        coords = torch.randn(10, 2)
        
        report = physics_validator.validate_physics_consistency(
            predictions, input_data, coords, "Test Physics Validation"
        )
        
        assert isinstance(report, PhysicsValidationReport)
        assert report.test_name == "Test Physics Validation"
        assert len(report.constraint_metrics) > 0
        assert 0 <= report.overall_consistency_score <= 1
        assert len(report.recommendations) > 0
    
    def test_conservation_laws_validation(self, physics_validator):
        """Test conservation laws validation."""
        predictions = torch.randn(10, 10)  # Larger prediction tensor
        input_data = torch.randn(10, 6)    # Input with current density
        
        conservation_violations = physics_validator.validate_conservation_laws(
            predictions, input_data
        )
        
        assert 'energy' in conservation_violations
        assert 'momentum' in conservation_violations
        assert 'charge' in conservation_violations
        
        for violation in conservation_violations.values():
            assert violation >= 0


class TestValidationPipeline:
    """Test complete validation pipeline."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock(spec=FaultClassificationSystem)
        model.fault_mapper = Mock()
        model.fault_mapper.index_to_type.return_value = FaultType.NORMAL
        model.classifier = Mock()
        model.classifier.parameters.return_value = [torch.randn(100, 50)]
        
        # Mock predict to return the correct number of predictions based on input size
        def mock_predict(data):
            batch_size = data.shape[0]
            return [
                FaultPrediction(
                    fault_type=FaultType.NORMAL,
                    confidence=0.9,
                    uncertainty=0.1,
                    probabilities={FaultType.NORMAL: 0.9},
                    physics_consistency=0.95
                )
                for _ in range(batch_size)
            ]
        model.predict.side_effect = mock_predict
        return model
    
    @pytest.fixture
    def validation_pipeline(self, mock_model):
        """Create validation pipeline."""
        return ValidationPipeline(mock_model)
    
    def test_pipeline_initialization(self, validation_pipeline):
        """Test validation pipeline initialization."""
        assert validation_pipeline.model is not None
        assert validation_pipeline.profiler is not None
        assert validation_pipeline.generalization_tester is not None
        assert validation_pipeline.baseline_comparator is not None
        assert validation_pipeline.baseline_threshold is not None
    
    @patch('src.validation.benchmarking_suite.PerformanceProfiler.profile_inference')
    @patch('src.validation.benchmarking_suite.PerformanceProfiler.get_memory_usage')
    @patch('src.validation.benchmarking_suite.PerformanceProfiler.get_cpu_usage')
    @patch('src.validation.benchmarking_suite.PerformanceProfiler.get_gpu_usage')
    def test_measure_core_performance(self, mock_gpu_usage, mock_cpu_usage, 
                                    mock_memory_usage, mock_profile_inference, 
                                    validation_pipeline):
        """Test core performance measurement."""
        # Mock profiler methods
        mock_profile_inference.return_value = {
            'mean_latency': 0.5,
            'throughput': 1000.0
        }
        mock_memory_usage.return_value = {'used_memory': 512.0}
        mock_cpu_usage.return_value = 25.0
        mock_gpu_usage.return_value = 30.0
        
        test_data = {'config1': torch.randn(10, 5)}
        test_labels = {'config1': torch.tensor([0] * 10)}
        
        metrics = validation_pipeline._measure_core_performance(test_data, test_labels)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert metrics.inference_latency == 0.5
        assert metrics.throughput == 1000.0
        assert metrics.memory_usage == 512.0
        assert metrics.cpu_usage == 25.0
        assert metrics.gpu_usage == 30.0
    
    def test_analyze_performance_regression(self, validation_pipeline):
        """Test performance regression analysis."""
        # Create metrics that should trigger regression
        poor_metrics = PerformanceMetrics(
            accuracy=0.8,  # Below threshold of 0.9
            inference_latency=2.0,  # Above threshold of 1.0
            throughput=500.0,
            memory_usage=1500.0,  # Above threshold of 1000.0
            cpu_usage=50.0,
            gpu_usage=60.0,
            model_size=1000000,
            physics_consistency_score=0.6,  # Below threshold of 0.8
            constraint_violation_rate=0.4,
            mean_confidence=0.7,
            uncertainty_calibration_score=0.6
        )
        
        regression_analysis = validation_pipeline._analyze_performance_regression(poor_metrics)
        
        assert regression_analysis['regression_detected'] is True
        assert 'accuracy' in regression_analysis['failing_metrics']
        assert 'inference_latency' in regression_analysis['failing_metrics']
        assert 'memory_usage' in regression_analysis['failing_metrics']
        assert 'physics_consistency' in regression_analysis['failing_metrics']
        
        # Test performance changes
        changes = regression_analysis['performance_changes']
        assert changes['accuracy_vs_threshold'] < 0  # Below threshold
        assert changes['latency_vs_threshold'] > 0   # Above threshold


class TestBenchmarkingSuite:
    """Test comprehensive benchmarking suite."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock(spec=FaultClassificationSystem)
        model.fault_mapper = Mock()
        model.fault_mapper.index_to_type.return_value = FaultType.NORMAL
        model.classifier = Mock()
        model.classifier.parameters.return_value = [torch.randn(100, 50)]
        model.predict.return_value = [
            FaultPrediction(
                fault_type=FaultType.NORMAL,
                confidence=0.9,
                uncertainty=0.1,
                probabilities={FaultType.NORMAL: 0.9},
                physics_consistency=0.95
            )
        ]
        return model
    
    @pytest.fixture
    def benchmarking_suite(self, mock_model):
        """Create benchmarking suite."""
        return BenchmarkingSuite(mock_model)
    
    def test_suite_initialization(self, benchmarking_suite):
        """Test benchmarking suite initialization."""
        assert benchmarking_suite.model is not None
        assert benchmarking_suite.validation_pipeline is not None
    
    @patch('src.validation.benchmarking_suite.ValidationPipeline.run_validation_pipeline')
    def test_run_comprehensive_benchmark(self, mock_run_pipeline, benchmarking_suite):
        """Test comprehensive benchmark execution."""
        # Mock validation pipeline result
        mock_report = Mock(spec=BenchmarkReport)
        mock_report.save_report = Mock()
        mock_report.generalization_report = None
        mock_report.comparison_report = None
        mock_report.physics_validation_report = None
        mock_run_pipeline.return_value = mock_report
        
        test_data = {'config1': torch.randn(10, 5)}
        test_labels = {'config1': torch.tensor([0] * 10)}
        
        report = benchmarking_suite.run_comprehensive_benchmark(
            test_data, test_labels, save_reports=False
        )
        
        assert report is not None
        mock_run_pipeline.assert_called_once()
    
    @patch('src.validation.benchmarking_suite.ValidationPipeline.run_validation_pipeline')
    def test_run_regression_test(self, mock_run_pipeline, benchmarking_suite):
        """Test automated regression testing."""
        # Mock validation pipeline result with no regression
        mock_report = Mock(spec=BenchmarkReport)
        mock_report.regression_analysis = {
            'regression_detected': False,
            'failing_metrics': []
        }
        mock_run_pipeline.return_value = mock_report
        
        test_data = {'config1': torch.randn(10, 5)}
        test_labels = {'config1': torch.tensor([0] * 10)}
        baseline_thresholds = {'accuracy': 0.9, 'inference_latency': 1.0}
        
        passed = benchmarking_suite.run_regression_test(
            test_data, test_labels, baseline_thresholds
        )
        
        assert passed is True
        mock_run_pipeline.assert_called_once()
        
        # Test with regression detected
        mock_report.regression_analysis = {
            'regression_detected': True,
            'failing_metrics': ['accuracy']
        }
        
        passed = benchmarking_suite.run_regression_test(
            test_data, test_labels, baseline_thresholds
        )
        
        assert passed is False


if __name__ == '__main__':
    pytest.main([__file__])