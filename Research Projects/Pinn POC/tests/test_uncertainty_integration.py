"""
Integration tests for complete uncertainty quantification system.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock
from torch.utils.data import DataLoader, TensorDataset

from src.physics.uncertainty_integration import (
    UncertaintyIntegratedPredictor, ReliabilityAssessor, SafetyDecisionEngine,
    ReliabilityLevel, PredictionResult, ReliabilityMetrics
)
from src.physics.uncertainty import UncertaintyConfig, SafetyThresholds


class SimpleTestModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


class TestUncertaintyIntegratedPredictor:
    """Test cases for UncertaintyIntegratedPredictor"""
    
    @pytest.fixture
    def config(self):
        return UncertaintyConfig(
            num_mc_samples=20,
            kl_weight=1e-3,
            calibration_bins=10
        )
    
    @pytest.fixture
    def base_model(self):
        return SimpleTestModel(input_dim=8, output_dim=4)
    
    @pytest.fixture
    def integrated_predictor(self, base_model, config):
        return UncertaintyIntegratedPredictor(
            base_model=base_model,
            uncertainty_config=config,
            input_dim=8,
            output_dim=4
        )
    
    def test_initialization(self, integrated_predictor, config):
        """Test proper initialization of integrated predictor"""
        assert integrated_predictor.config == config
        assert integrated_predictor.base_model is not None
        assert integrated_predictor.vb_uq is not None
        assert integrated_predictor.confidence_calibration is not None
        assert integrated_predictor.safety_thresholds is not None
        assert integrated_predictor.reliability_assessor is not None
        assert not integrated_predictor.is_calibrated
        assert integrated_predictor.training_mode
    
    def test_training_forward(self, integrated_predictor):
        """Test forward pass during training"""
        batch_size = 16
        x = torch.randn(batch_size, 8)
        
        # Set to training mode
        integrated_predictor.set_training_mode(True)
        
        output = integrated_predictor(x, return_uncertainty=False)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 4)
    
    def test_simple_inference(self, integrated_predictor):
        """Test simple inference without uncertainty"""
        batch_size = 8
        x = torch.randn(batch_size, 8)
        
        # Set to inference mode
        integrated_predictor.set_training_mode(False)
        
        output = integrated_predictor(x, return_uncertainty=False)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 4)
    
    def test_inference_with_uncertainty(self, integrated_predictor):
        """Test inference with complete uncertainty analysis"""
        batch_size = 4
        x = torch.randn(batch_size, 8)
        
        # Set to inference mode
        integrated_predictor.set_training_mode(False)
        
        results = integrated_predictor(x, return_uncertainty=True)
        
        assert isinstance(results, list)
        assert len(results) == batch_size
        
        for result in results:
            assert isinstance(result, PredictionResult)
            assert result.prediction.shape == (4,)
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0
            assert result.uncertainty.shape == (4,)
            assert isinstance(result.reliability_level, ReliabilityLevel)
            assert isinstance(result.safety_assessment, dict)
            assert isinstance(result.physics_consistency, float)
            assert isinstance(result.recommended_action, str)
    
    def test_confidence_computation(self, integrated_predictor):
        """Test confidence score computation"""
        prediction = torch.randn(1, 4)
        uncertainty = torch.abs(torch.randn(1, 4)) * 0.1  # Small positive uncertainty
        
        confidence = integrated_predictor._compute_confidence(prediction, uncertainty)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # Test with high uncertainty
        high_uncertainty = torch.ones(1, 4) * 10.0
        low_confidence = integrated_predictor._compute_confidence(prediction, high_uncertainty)
        
        assert low_confidence < confidence
    
    def test_reliability_level_determination(self, integrated_predictor):
        """Test reliability level determination"""
        # Mock reliability metrics
        high_reliability_metrics = ReliabilityMetrics(
            prediction_reliability=0.95,
            uncertainty_reliability=0.90,
            physics_consistency_score=0.85,
            calibration_quality=0.80,
            overall_reliability=0.95,
            confidence_intervals={}
        )
        
        # Test critical reliability
        level = integrated_predictor._determine_reliability_level(0.96, high_reliability_metrics)
        assert level == ReliabilityLevel.CRITICAL
        
        # Test moderate reliability
        moderate_reliability_metrics = ReliabilityMetrics(
            prediction_reliability=0.75,
            uncertainty_reliability=0.70,
            physics_consistency_score=0.65,
            calibration_quality=0.60,
            overall_reliability=0.75,
            confidence_intervals={}
        )
        
        level = integrated_predictor._determine_reliability_level(0.75, moderate_reliability_metrics)
        assert level == ReliabilityLevel.MODERATE
    
    def test_safety_assessment(self, integrated_predictor):
        """Test safety assessment functionality"""
        prediction = torch.tensor([[0.1, 0.2, 0.7, 0.0]])  # Critical fault prediction
        confidence = 0.85
        reliability_level = ReliabilityLevel.HIGH
        
        safety_assessment = integrated_predictor._perform_safety_assessment(
            prediction, confidence, reliability_level
        )
        
        assert isinstance(safety_assessment, dict)
        assert 'overall_safe' in safety_assessment
        assert 'fault_type' in safety_assessment
        assert 'reliability_level' in safety_assessment
        
        # Check safety margins
        for decision_type in ['critical_fault', 'warning_fault', 'normal_operation']:
            assert f"{decision_type}_safe" in safety_assessment
            assert f"{decision_type}_margin" in safety_assessment
    
    def test_fault_type_classification(self, integrated_predictor):
        """Test fault type classification"""
        # Normal operation (index 0)
        normal_pred = torch.tensor([[0.8, 0.1, 0.1]])
        fault_type = integrated_predictor._classify_fault_type(normal_pred)
        assert fault_type == 'normal'
        
        # Warning fault (index 1)
        warning_pred = torch.tensor([[0.1, 0.8, 0.1]])
        fault_type = integrated_predictor._classify_fault_type(warning_pred)
        assert fault_type == 'warning'
        
        # Critical fault (index 2)
        critical_pred = torch.tensor([[0.1, 0.1, 0.8]])
        fault_type = integrated_predictor._classify_fault_type(critical_pred)
        assert fault_type == 'critical'
    
    def test_calibration_process(self, integrated_predictor):
        """Test uncertainty calibration process"""
        # Create mock validation data
        n_samples = 100
        x_val = torch.randn(n_samples, 8)
        y_val = torch.randint(0, 4, (n_samples, 4)).float()
        
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # Set to inference mode for calibration
        integrated_predictor.set_training_mode(False)
        
        # Perform calibration
        metrics = integrated_predictor.calibrate_uncertainty(val_loader)
        
        assert integrated_predictor.is_calibrated
        assert hasattr(metrics, 'expected_calibration_error')
        assert hasattr(metrics, 'maximum_calibration_error')
        assert hasattr(metrics, 'brier_score')
    
    def test_training_mode_switching(self, integrated_predictor):
        """Test switching between training and inference modes"""
        # Initially in training mode
        assert integrated_predictor.training_mode
        
        # Switch to inference mode
        integrated_predictor.set_training_mode(False)
        assert not integrated_predictor.training_mode
        
        # Switch back to training mode
        integrated_predictor.set_training_mode(True)
        assert integrated_predictor.training_mode


class TestReliabilityAssessor:
    """Test cases for ReliabilityAssessor"""
    
    @pytest.fixture
    def config(self):
        return UncertaintyConfig(confidence_levels=[0.68, 0.95, 0.99])
    
    @pytest.fixture
    def reliability_assessor(self, config):
        return ReliabilityAssessor(config)
    
    def test_reliability_assessment(self, reliability_assessor):
        """Test comprehensive reliability assessment"""
        prediction = torch.randn(4)
        uncertainty = torch.abs(torch.randn(4)) * 0.1
        confidence = 0.85
        
        metrics = reliability_assessor.assess_reliability(prediction, uncertainty, confidence)
        
        assert isinstance(metrics, ReliabilityMetrics)
        assert 0.0 <= metrics.prediction_reliability <= 1.0
        assert 0.0 <= metrics.uncertainty_reliability <= 1.0
        assert 0.0 <= metrics.physics_consistency_score <= 1.0
        assert 0.0 <= metrics.calibration_quality <= 1.0
        assert 0.0 <= metrics.overall_reliability <= 1.0
        assert isinstance(metrics.confidence_intervals, dict)
    
    def test_prediction_reliability_assessment(self, reliability_assessor):
        """Test prediction reliability assessment"""
        # High confidence prediction
        high_conf_pred = torch.tensor([1.0, 2.0, 3.0])
        high_reliability = reliability_assessor._assess_prediction_reliability(high_conf_pred, 0.9)
        
        # Low confidence prediction
        low_conf_pred = torch.tensor([0.1, 0.1, 0.1])
        low_reliability = reliability_assessor._assess_prediction_reliability(low_conf_pred, 0.3)
        
        assert high_reliability > low_reliability
        assert 0.0 <= high_reliability <= 1.0
        assert 0.0 <= low_reliability <= 1.0
    
    def test_uncertainty_reliability_assessment(self, reliability_assessor):
        """Test uncertainty reliability assessment"""
        # Consistent low uncertainty
        consistent_uncertainty = torch.tensor([0.1, 0.1, 0.1, 0.1])
        consistent_reliability = reliability_assessor._assess_uncertainty_reliability(consistent_uncertainty)
        
        # Inconsistent high uncertainty
        inconsistent_uncertainty = torch.tensor([0.1, 0.5, 1.0, 2.0])
        inconsistent_reliability = reliability_assessor._assess_uncertainty_reliability(inconsistent_uncertainty)
        
        assert consistent_reliability > inconsistent_reliability
        assert 0.0 <= consistent_reliability <= 1.0
        assert 0.0 <= inconsistent_reliability <= 1.0
    
    def test_confidence_intervals_computation(self, reliability_assessor):
        """Test confidence intervals computation"""
        prediction = torch.tensor([1.0, 2.0, 3.0])
        uncertainty = torch.tensor([0.1, 0.2, 0.3])
        
        intervals = reliability_assessor._compute_confidence_intervals(prediction, uncertainty)
        
        assert isinstance(intervals, dict)
        assert len(intervals) == len(reliability_assessor.config.confidence_levels)
        
        for level_str, (lower, upper) in intervals.items():
            assert isinstance(lower, float)
            assert isinstance(upper, float)
            assert lower < upper  # Lower bound should be less than upper bound


class TestSafetyDecisionEngine:
    """Test cases for SafetyDecisionEngine"""
    
    @pytest.fixture
    def safety_thresholds(self):
        config = UncertaintyConfig()
        return SafetyThresholds(config)
    
    @pytest.fixture
    def decision_engine(self, safety_thresholds):
        return SafetyDecisionEngine(safety_thresholds)
    
    @pytest.fixture
    def sample_prediction_result(self):
        return PredictionResult(
            prediction=torch.tensor([0.1, 0.2, 0.7]),
            confidence=0.85,
            uncertainty=torch.tensor([0.05, 0.1, 0.15]),
            reliability_level=ReliabilityLevel.HIGH,
            safety_assessment={
                'critical_fault_safe': False,
                'warning_fault_safe': True,
                'normal_operation_safe': True,
                'overall_safe': True,
                'fault_type': 'warning'
            },
            physics_consistency=0.80,
            calibrated_confidence=0.82,
            recommended_action='maintenance_alert',
            timestamp=0.0
        )
    
    def test_safety_decision_making(self, decision_engine, sample_prediction_result):
        """Test safety decision making process"""
        decision_record = decision_engine.make_safety_decision(sample_prediction_result)
        
        assert isinstance(decision_record, dict)
        assert 'decision' in decision_record
        assert 'justification' in decision_record
        assert 'confidence' in decision_record
        assert 'reliability_level' in decision_record
        assert 'timestamp' in decision_record
        assert 'safe_for_autonomous' in decision_record
        
        # Check that decision history is updated
        assert len(decision_engine.decision_history) == 1
    
    def test_critical_reliability_decision(self, decision_engine):
        """Test decision making for critical reliability level"""
        critical_result = PredictionResult(
            prediction=torch.tensor([0.0, 0.0, 1.0]),
            confidence=0.98,
            uncertainty=torch.tensor([0.01, 0.01, 0.01]),
            reliability_level=ReliabilityLevel.CRITICAL,
            safety_assessment={
                'critical_fault_safe': True,
                'overall_safe': True,
                'fault_type': 'critical'
            },
            physics_consistency=0.95,
            calibrated_confidence=0.97,
            recommended_action='emergency_shutdown',
            timestamp=0.0
        )
        
        decision_record = decision_engine.make_safety_decision(critical_result)
        
        assert decision_record['decision'] == 'proceed_with_high_confidence'
        assert decision_record['safe_for_autonomous'] == True
    
    def test_uncertain_reliability_decision(self, decision_engine):
        """Test decision making for uncertain reliability level"""
        uncertain_result = PredictionResult(
            prediction=torch.tensor([0.3, 0.3, 0.4]),
            confidence=0.45,
            uncertainty=torch.tensor([0.5, 0.5, 0.5]),
            reliability_level=ReliabilityLevel.UNCERTAIN,
            safety_assessment={
                'overall_safe': False,
                'fault_type': 'unknown'
            },
            physics_consistency=0.40,
            calibrated_confidence=0.42,
            recommended_action='uncertain',
            timestamp=0.0
        )
        
        decision_record = decision_engine.make_safety_decision(uncertain_result)
        
        assert decision_record['decision'] == 'manual_inspection_required'
        assert decision_record['safe_for_autonomous'] == False
    
    def test_decision_history_tracking(self, decision_engine, sample_prediction_result):
        """Test decision history tracking"""
        # Make multiple decisions
        for i in range(5):
            decision_engine.make_safety_decision(sample_prediction_result)
        
        history = decision_engine.get_decision_history()
        assert len(history) == 5
        
        # Each decision should have required fields
        for record in history:
            assert 'decision' in record
            assert 'confidence' in record
            assert 'reliability_level' in record
    
    def test_decision_pattern_analysis(self, decision_engine, sample_prediction_result):
        """Test decision pattern analysis"""
        # Make multiple decisions with different reliability levels
        results = [
            sample_prediction_result,  # HIGH reliability
            PredictionResult(
                prediction=torch.tensor([0.5, 0.3, 0.2]),
                confidence=0.60,
                uncertainty=torch.tensor([0.2, 0.2, 0.2]),
                reliability_level=ReliabilityLevel.MODERATE,
                safety_assessment={'overall_safe': False, 'fault_type': 'warning'},
                physics_consistency=0.70,
                calibrated_confidence=0.58,
                recommended_action='collect_more_data',
                timestamp=0.0
            ),
            PredictionResult(
                prediction=torch.tensor([0.4, 0.4, 0.2]),
                confidence=0.30,
                uncertainty=torch.tensor([0.8, 0.8, 0.8]),
                reliability_level=ReliabilityLevel.UNCERTAIN,
                safety_assessment={'overall_safe': False, 'fault_type': 'unknown'},
                physics_consistency=0.30,
                calibrated_confidence=0.28,
                recommended_action='manual_inspection_required',
                timestamp=0.0
            )
        ]
        
        for result in results:
            decision_engine.make_safety_decision(result)
        
        analysis = decision_engine.analyze_decision_patterns()
        
        assert 'total_decisions' in analysis
        assert 'autonomous_decision_rate' in analysis
        assert 'average_confidence' in analysis
        assert 'decision_distribution' in analysis
        
        assert analysis['total_decisions'] == 3
        assert 0.0 <= analysis['autonomous_decision_rate'] <= 1.0
        assert 0.0 <= analysis['average_confidence'] <= 1.0


class TestIntegrationWorkflow:
    """Integration tests for complete uncertainty quantification workflow"""
    
    @pytest.fixture
    def complete_system(self):
        """Set up complete integrated system"""
        config = UncertaintyConfig(num_mc_samples=10, calibration_bins=5)
        base_model = SimpleTestModel(input_dim=6, output_dim=3)
        
        integrated_predictor = UncertaintyIntegratedPredictor(
            base_model=base_model,
            uncertainty_config=config,
            input_dim=6,
            output_dim=3
        )
        
        safety_thresholds = SafetyThresholds(config)
        decision_engine = SafetyDecisionEngine(safety_thresholds)
        
        return integrated_predictor, decision_engine
    
    def test_end_to_end_workflow(self, complete_system):
        """Test complete end-to-end uncertainty quantification workflow"""
        integrated_predictor, decision_engine = complete_system
        
        # Generate test data
        test_input = torch.randn(1, 6)
        
        # Set to inference mode
        integrated_predictor.set_training_mode(False)
        
        # Get prediction with uncertainty
        results = integrated_predictor(test_input, return_uncertainty=True)
        assert len(results) == 1
        
        prediction_result = results[0]
        
        # Make safety decision
        decision_record = decision_engine.make_safety_decision(prediction_result)
        
        # Verify complete workflow
        assert isinstance(prediction_result, PredictionResult)
        assert isinstance(decision_record, dict)
        assert 'decision' in decision_record
        assert 'safe_for_autonomous' in decision_record
    
    def test_batch_processing_workflow(self, complete_system):
        """Test batch processing with uncertainty quantification"""
        integrated_predictor, decision_engine = complete_system
        
        # Generate batch test data
        batch_size = 5
        test_inputs = torch.randn(batch_size, 6)
        
        # Set to inference mode
        integrated_predictor.set_training_mode(False)
        
        # Process batch
        results = integrated_predictor(test_inputs, return_uncertainty=True)
        assert len(results) == batch_size
        
        # Make decisions for each result
        decisions = []
        for result in results:
            decision = decision_engine.make_safety_decision(result)
            decisions.append(decision)
        
        assert len(decisions) == batch_size
        
        # Analyze decision patterns
        analysis = decision_engine.analyze_decision_patterns()
        assert analysis['total_decisions'] == batch_size
    
    def test_calibration_integration(self, complete_system):
        """Test integration of calibration with prediction workflow"""
        integrated_predictor, _ = complete_system
        
        # Create calibration data
        n_cal = 50
        x_cal = torch.randn(n_cal, 6)
        y_cal = torch.randint(0, 3, (n_cal, 3)).float()
        
        cal_dataset = TensorDataset(x_cal, y_cal)
        cal_loader = DataLoader(cal_dataset, batch_size=10)
        
        # Set to inference mode
        integrated_predictor.set_training_mode(False)
        
        # Perform calibration
        metrics = integrated_predictor.calibrate_uncertainty(cal_loader)
        
        # Verify calibration is applied
        assert integrated_predictor.is_calibrated
        
        # Test prediction with calibration
        test_input = torch.randn(1, 6)
        results = integrated_predictor(test_input, return_uncertainty=True)
        
        result = results[0]
        # Calibrated confidence should be different from raw confidence
        # (unless they happen to be the same by chance)
        assert hasattr(result, 'calibrated_confidence')
        assert isinstance(result.calibrated_confidence, float)


if __name__ == '__main__':
    pytest.main([__file__])