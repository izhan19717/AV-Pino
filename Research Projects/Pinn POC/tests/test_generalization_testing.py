"""
Tests for generalization testing module.

Tests cross-motor configuration validation and performance evaluation
on unseen motor types and operating conditions.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List

from src.validation.generalization_testing import (
    GeneralizationTester,
    CrossMotorValidator,
    OperatingConditionValidator,
    MotorConfiguration,
    MotorType,
    OperatingCondition,
    GeneralizationMetrics,
    GeneralizationReport
)
from src.inference.fault_classifier import FaultClassificationSystem, FaultPrediction, FaultType


class TestMotorConfiguration:
    """Test motor configuration data structure."""
    
    def test_motor_configuration_creation(self):
        """Test creating motor configuration."""
        config = MotorConfiguration(
            motor_type=MotorType.INDUCTION_MOTOR,
            power_rating=2.0,
            voltage=460,
            frequency=60,
            pole_pairs=2,
            bearing_type='6205-2RS',
            operating_condition=OperatingCondition.MEDIUM_LOAD
        )
        
        assert config.motor_type == MotorType.INDUCTION_MOTOR
        assert config.power_rating == 2.0
        assert config.voltage == 460
        assert config.frequency == 60
        assert config.pole_pairs == 2
        assert config.bearing_type == '6205-2RS'
        assert config.operating_condition == OperatingCondition.MEDIUM_LOAD
    
    def test_motor_configuration_to_dict(self):
        """Test converting motor configuration to dictionary."""
        config = MotorConfiguration(
            motor_type=MotorType.SYNCHRONOUS_MOTOR,
            power_rating=5.0,
            voltage=400,
            frequency=50,
            pole_pairs=3,
            bearing_type='6206',
            operating_condition=OperatingCondition.HIGH_LOAD
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['motor_type'] == 'synchronous'
        assert config_dict['power_rating'] == 5.0
        assert config_dict['voltage'] == 400
        assert config_dict['frequency'] == 50
        assert config_dict['pole_pairs'] == 3
        assert config_dict['bearing_type'] == '6206'
        assert config_dict['operating_condition'] == 'high_load'


class TestGeneralizationMetrics:
    """Test generalization metrics computation."""
    
    def test_generalization_metrics_creation(self):
        """Test creating generalization metrics."""
        metrics = GeneralizationMetrics(
            accuracy=0.85,
            precision={'normal': 0.9, 'inner_race': 0.8},
            recall={'normal': 0.85, 'inner_race': 0.9},
            f1_score={'normal': 0.87, 'inner_race': 0.85},
            confusion_matrix=np.array([[10, 2], [1, 8]]),
            confidence_scores=[0.9, 0.8, 0.85],
            uncertainty_scores=[0.1, 0.2, 0.15],
            physics_consistency_scores=[0.95, 0.9, 0.92]
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.precision['normal'] == 0.9
        assert metrics.recall['inner_race'] == 0.9
        assert metrics.f1_score['normal'] == 0.87
        assert metrics.confusion_matrix.shape == (2, 2)
        assert len(metrics.confidence_scores) == 3
        assert len(metrics.uncertainty_scores) == 3
        assert len(metrics.physics_consistency_scores) == 3
    
    def test_generalization_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = GeneralizationMetrics(
            accuracy=0.85,
            precision={'normal': 0.9},
            recall={'normal': 0.85},
            f1_score={'normal': 0.87},
            confusion_matrix=np.array([[10, 2], [1, 8]]),
            confidence_scores=[0.9, 0.8],
            uncertainty_scores=[0.1, 0.2],
            physics_consistency_scores=[0.95, 0.9]
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['accuracy'] == 0.85
        assert metrics_dict['precision']['normal'] == 0.9
        assert isinstance(metrics_dict['confusion_matrix'], list)
        assert len(metrics_dict['confidence_scores']) == 2


class TestCrossMotorValidator:
    """Test cross-motor configuration validation."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock fault classification system."""
        model = Mock(spec=FaultClassificationSystem)
        model.fault_mapper = Mock()
        model.fault_mapper.index_to_type.return_value = FaultType.NORMAL
        return model
    
    @pytest.fixture
    def cross_motor_validator(self, mock_model):
        """Create cross-motor validator."""
        return CrossMotorValidator(mock_model)
    
    def test_motor_configurations_creation(self, cross_motor_validator):
        """Test creation of diverse motor configurations."""
        configs = cross_motor_validator.motor_configs
        
        assert 'baseline_cwru' in configs
        assert 'high_power_induction' in configs
        assert 'low_power_induction' in configs
        assert 'synchronous_motor' in configs
        assert 'brushless_dc' in configs
        
        # Check baseline configuration
        baseline = configs['baseline_cwru']
        assert baseline.motor_type == MotorType.INDUCTION_MOTOR
        assert baseline.power_rating == 2.0
        assert baseline.voltage == 460
        
        # Check diversity in configurations
        power_ratings = [config.power_rating for config in configs.values()]
        assert len(set(power_ratings)) > 1  # Different power ratings
        
        motor_types = [config.motor_type for config in configs.values()]
        assert len(set(motor_types)) > 1  # Different motor types
    
    def test_validate_cross_motor_performance(self, cross_motor_validator, mock_model):
        """Test cross-motor performance validation."""
        # Mock predictions
        mock_predictions = [
            FaultPrediction(
                fault_type=FaultType.NORMAL,
                confidence=0.9,
                uncertainty=0.1,
                probabilities={FaultType.NORMAL: 0.9, FaultType.INNER_RACE: 0.1},
                physics_consistency=0.95
            ),
            FaultPrediction(
                fault_type=FaultType.INNER_RACE,
                confidence=0.85,
                uncertainty=0.15,
                probabilities={FaultType.NORMAL: 0.15, FaultType.INNER_RACE: 0.85},
                physics_consistency=0.9
            )
        ]
        mock_model.predict.return_value = mock_predictions
        
        # Test data
        test_data = {
            'baseline_cwru': torch.randn(2, 10),
            'high_power_induction': torch.randn(2, 10)
        }
        test_labels = {
            'baseline_cwru': torch.tensor([0, 1]),
            'high_power_induction': torch.tensor([0, 1])
        }
        
        results = cross_motor_validator.validate_cross_motor_performance(
            test_data, test_labels
        )
        
        assert 'baseline_cwru' in results
        assert 'high_power_induction' in results
        
        # Check metrics structure
        for config_name, metrics in results.items():
            assert isinstance(metrics, GeneralizationMetrics)
            assert 0 <= metrics.accuracy <= 1
            assert len(metrics.confidence_scores) == 2
            assert len(metrics.uncertainty_scores) == 2
            assert len(metrics.physics_consistency_scores) == 2
    
    def test_compute_generalization_metrics(self, cross_motor_validator):
        """Test generalization metrics computation."""
        true_labels = ['normal', 'inner_race', 'normal', 'inner_race']
        pred_labels = ['normal', 'inner_race', 'inner_race', 'inner_race']
        confidences = [0.9, 0.85, 0.7, 0.8]
        uncertainties = [0.1, 0.15, 0.3, 0.2]
        physics_scores = [0.95, 0.9, 0.8, 0.85]
        
        metrics = cross_motor_validator._compute_generalization_metrics(
            true_labels, pred_labels, confidences, uncertainties, physics_scores
        )
        
        assert isinstance(metrics, GeneralizationMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 'normal' in metrics.precision
        assert 'inner_race' in metrics.precision
        assert metrics.confusion_matrix.shape == (2, 2)
        assert len(metrics.confidence_scores) == 4
        assert len(metrics.uncertainty_scores) == 4
        assert len(metrics.physics_consistency_scores) == 4


class TestOperatingConditionValidator:
    """Test operating condition validation."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock fault classification system."""
        model = Mock(spec=FaultClassificationSystem)
        model.fault_mapper = Mock()
        model.fault_mapper.index_to_type.return_value = FaultType.NORMAL
        return model
    
    @pytest.fixture
    def operating_condition_validator(self, mock_model):
        """Create operating condition validator."""
        return OperatingConditionValidator(mock_model)
    
    def test_validate_operating_conditions(self, operating_condition_validator, mock_model):
        """Test operating condition validation."""
        # Mock predictions
        mock_predictions = [
            FaultPrediction(
                fault_type=FaultType.NORMAL,
                confidence=0.9,
                uncertainty=0.1,
                probabilities={FaultType.NORMAL: 0.9},
                physics_consistency=0.95
            )
        ]
        mock_model.predict.return_value = mock_predictions
        
        # Test data
        test_data = {
            'low_load': torch.randn(1, 10),
            'high_load': torch.randn(1, 10)
        }
        test_labels = {
            'low_load': torch.tensor([0]),
            'high_load': torch.tensor([0])
        }
        
        results = operating_condition_validator.validate_operating_conditions(
            test_data, test_labels
        )
        
        assert 'low_load' in results
        assert 'high_load' in results
        
        for condition, metrics in results.items():
            assert isinstance(metrics, GeneralizationMetrics)
            assert 0 <= metrics.accuracy <= 1
    
    def test_compute_operating_condition_metrics(self, operating_condition_validator):
        """Test operating condition specific metrics computation."""
        true_labels = ['normal']
        pred_labels = ['normal']
        confidences = [0.9]
        uncertainties = [0.1]
        physics_scores = [0.95]
        
        # High load condition (should trigger accuracy drop calculation)
        condition_params = {'load_range': (2, 3), 'rpm_range': (1720, 1750)}
        
        metrics = operating_condition_validator._compute_operating_condition_metrics(
            true_labels, pred_labels, confidences, uncertainties, 
            physics_scores, condition_params
        )
        
        assert isinstance(metrics, GeneralizationMetrics)
        assert metrics.accuracy_drop >= 0  # Should be calculated for high load


class TestGeneralizationTester:
    """Test main generalization testing functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock fault classification system."""
        model = Mock(spec=FaultClassificationSystem)
        model.fault_mapper = Mock()
        model.fault_mapper.index_to_type.return_value = FaultType.NORMAL
        return model
    
    @pytest.fixture
    def generalization_tester(self, mock_model):
        """Create generalization tester."""
        return GeneralizationTester(mock_model)
    
    def test_generalization_tester_initialization(self, generalization_tester):
        """Test generalization tester initialization."""
        assert generalization_tester.model is not None
        assert generalization_tester.cross_motor_validator is not None
        assert generalization_tester.operating_condition_validator is not None
    
    def test_assess_model_generalization(self, generalization_tester, mock_model):
        """Test quick generalization assessment."""
        # Mock predictions
        mock_predictions = [
            FaultPrediction(
                fault_type=FaultType.NORMAL,
                confidence=0.9,
                uncertainty=0.1,
                probabilities={FaultType.NORMAL: 0.9},
                physics_consistency=0.95
            )
        ]
        mock_model.predict.return_value = mock_predictions
        
        test_data = {
            'config1': torch.randn(1, 10),
            'config2': torch.randn(1, 10)
        }
        test_labels = {
            'config1': torch.tensor([0]),
            'config2': torch.tensor([0])
        }
        
        results = generalization_tester.assess_model_generalization(
            test_data, test_labels
        )
        
        assert 'config1' in results
        assert 'config2' in results
        assert all(0 <= acc <= 1 for acc in results.values())
    
    def test_compute_summary_statistics(self, generalization_tester):
        """Test summary statistics computation."""
        baseline_metrics = GeneralizationMetrics(
            accuracy=0.95,
            precision={'normal': 0.95},
            recall={'normal': 0.95},
            f1_score={'normal': 0.95},
            confusion_matrix=np.array([[10, 0], [0, 10]]),
            confidence_scores=[0.9, 0.95],
            uncertainty_scores=[0.1, 0.05],
            physics_consistency_scores=[0.95, 0.98]
        )
        
        test_results = {
            'config1': GeneralizationMetrics(
                accuracy=0.85,
                precision={'normal': 0.85},
                recall={'normal': 0.85},
                f1_score={'normal': 0.85},
                confusion_matrix=np.array([[8, 2], [1, 9]]),
                confidence_scores=[0.8, 0.85],
                uncertainty_scores=[0.2, 0.15],
                physics_consistency_scores=[0.9, 0.92]
            ),
            'config2': GeneralizationMetrics(
                accuracy=0.80,
                precision={'normal': 0.80},
                recall={'normal': 0.80},
                f1_score={'normal': 0.80},
                confusion_matrix=np.array([[7, 3], [2, 8]]),
                confidence_scores=[0.75, 0.8],
                uncertainty_scores=[0.25, 0.2],
                physics_consistency_scores=[0.85, 0.88]
            )
        }
        
        summary_stats = generalization_tester._compute_summary_statistics(
            baseline_metrics, test_results
        )
        
        assert 'baseline_accuracy' in summary_stats
        assert 'mean_generalization_accuracy' in summary_stats
        assert 'std_generalization_accuracy' in summary_stats
        assert 'min_generalization_accuracy' in summary_stats
        assert 'max_generalization_accuracy' in summary_stats
        assert 'mean_accuracy_drop' in summary_stats
        assert 'generalization_robustness' in summary_stats
        
        assert summary_stats['baseline_accuracy'] == 0.95
        assert summary_stats['mean_generalization_accuracy'] == 0.825  # (0.85 + 0.80) / 2
        assert summary_stats['min_generalization_accuracy'] == 0.80
        assert summary_stats['max_generalization_accuracy'] == 0.85
    
    def test_generate_recommendations(self, generalization_tester):
        """Test recommendation generation."""
        baseline_metrics = GeneralizationMetrics(
            accuracy=0.95,
            precision={'normal': 0.95},
            recall={'normal': 0.95},
            f1_score={'normal': 0.95},
            confusion_matrix=np.array([[10, 0], [0, 10]]),
            confidence_scores=[0.9, 0.95],
            uncertainty_scores=[0.1, 0.05],
            physics_consistency_scores=[0.95, 0.98]
        )
        
        # Test case with significant accuracy drop
        test_results_poor = {
            'config1': GeneralizationMetrics(
                accuracy=0.70,  # Significant drop
                precision={'normal': 0.70},
                recall={'normal': 0.70},
                f1_score={'normal': 0.70},
                confusion_matrix=np.array([[7, 3], [3, 7]]),
                confidence_scores=[0.6, 0.65],  # Low confidence
                uncertainty_scores=[0.4, 0.35],
                physics_consistency_scores=[0.5, 0.55]  # Poor physics consistency
            )
        }
        
        recommendations = generalization_tester._generate_recommendations(
            baseline_metrics, test_results_poor
        )
        
        assert len(recommendations) > 0
        assert any('accuracy drop' in rec.lower() for rec in recommendations)
        assert any('confidence' in rec.lower() for rec in recommendations)
        assert any('physics consistency' in rec.lower() for rec in recommendations)
        
        # Test case with good performance
        test_results_good = {
            'config1': GeneralizationMetrics(
                accuracy=0.92,
                precision={'normal': 0.92},
                recall={'normal': 0.92},
                f1_score={'normal': 0.92},
                confusion_matrix=np.array([[9, 1], [1, 9]]),
                confidence_scores=[0.9, 0.92],
                uncertainty_scores=[0.1, 0.08],
                physics_consistency_scores=[0.9, 0.92]
            )
        }
        
        recommendations_good = generalization_tester._generate_recommendations(
            baseline_metrics, test_results_good
        )
        
        assert any('satisfactory' in rec.lower() for rec in recommendations_good)


class TestGeneralizationReport:
    """Test generalization report functionality."""
    
    def test_generalization_report_creation(self):
        """Test creating generalization report."""
        baseline_metrics = GeneralizationMetrics(
            accuracy=0.95,
            precision={'normal': 0.95},
            recall={'normal': 0.95},
            f1_score={'normal': 0.95},
            confusion_matrix=np.array([[10, 0], [0, 10]]),
            confidence_scores=[0.9, 0.95],
            uncertainty_scores=[0.1, 0.05],
            physics_consistency_scores=[0.95, 0.98]
        )
        
        generalization_metrics = {
            'config1': baseline_metrics
        }
        
        motor_configs = {
            'config1': MotorConfiguration(
                motor_type=MotorType.INDUCTION_MOTOR,
                power_rating=2.0,
                voltage=460,
                frequency=60,
                pole_pairs=2,
                bearing_type='6205-2RS',
                operating_condition=OperatingCondition.MEDIUM_LOAD
            )
        }
        
        report = GeneralizationReport(
            test_name="Test Report",
            baseline_metrics=baseline_metrics,
            generalization_metrics=generalization_metrics,
            motor_configurations=motor_configs,
            summary_statistics={'mean_accuracy': 0.95},
            recommendations=['Good performance'],
            timestamp='2024-01-01T00:00:00'
        )
        
        assert report.test_name == "Test Report"
        assert report.baseline_metrics == baseline_metrics
        assert 'config1' in report.generalization_metrics
        assert 'config1' in report.motor_configurations
        assert report.summary_statistics['mean_accuracy'] == 0.95
        assert len(report.recommendations) == 1
    
    @patch('builtins.open')
    @patch('json.dump')
    def test_save_report(self, mock_json_dump, mock_open):
        """Test saving generalization report."""
        baseline_metrics = GeneralizationMetrics(
            accuracy=0.95,
            precision={'normal': 0.95},
            recall={'normal': 0.95},
            f1_score={'normal': 0.95},
            confusion_matrix=np.array([[10, 0], [0, 10]]),
            confidence_scores=[0.9, 0.95],
            uncertainty_scores=[0.1, 0.05],
            physics_consistency_scores=[0.95, 0.98]
        )
        
        report = GeneralizationReport(
            test_name="Test Report",
            baseline_metrics=baseline_metrics,
            generalization_metrics={},
            motor_configurations={},
            summary_statistics={},
            recommendations=[],
            timestamp='2024-01-01T00:00:00'
        )
        
        report.save_report('test_report.json')
        
        mock_open.assert_called_once_with('test_report.json', 'w')
        mock_json_dump.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])