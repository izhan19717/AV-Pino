"""
Complete Uncertainty Integration for AV-PINO Motor Fault Diagnosis

This module integrates uncertainty quantification into the main prediction pipeline
and implements reliability assessment for safety-critical decisions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from .uncertainty import (
    VariationalBayesianUQ, ConfidenceCalibration, SafetyThresholds,
    UncertaintyConfig, CalibrationMetrics
)

logger = logging.getLogger(__name__)


class ReliabilityLevel(Enum):
    """Reliability levels for safety-critical decisions"""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class PredictionResult:
    """Complete prediction result with uncertainty quantification"""
    prediction: torch.Tensor
    confidence: float
    uncertainty: torch.Tensor
    reliability_level: ReliabilityLevel
    safety_assessment: Dict[str, Union[bool, float, str]]
    physics_consistency: float
    calibrated_confidence: float
    recommended_action: str
    timestamp: float


@dataclass
class ReliabilityMetrics:
    """Metrics for reliability assessment"""
    prediction_reliability: float
    uncertainty_reliability: float
    physics_consistency_score: float
    calibration_quality: float
    overall_reliability: float
    confidence_intervals: Dict[str, Tuple[float, float]]


class UncertaintyIntegratedPredictor(nn.Module):
    """
    Main prediction pipeline with integrated uncertainty quantification.
    
    This class combines the neural operator predictions with variational Bayesian
    uncertainty quantification, confidence calibration, and safety assessment.
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 uncertainty_config: UncertaintyConfig,
                 input_dim: int,
                 output_dim: int):
        super().__init__()
        
        self.base_model = base_model
        self.config = uncertainty_config
        
        # Initialize uncertainty quantification components
        self.vb_uq = VariationalBayesianUQ(uncertainty_config, input_dim, output_dim)
        self.confidence_calibration = ConfidenceCalibration(uncertainty_config)
        self.safety_thresholds = SafetyThresholds(uncertainty_config)
        
        # Reliability assessment components
        self.reliability_assessor = ReliabilityAssessor(uncertainty_config)
        
        # Integration state
        self.is_calibrated = False
        self.training_mode = True
        
        logger.info("Initialized UncertaintyIntegratedPredictor")
    
    def forward(self, x: torch.Tensor, 
                return_uncertainty: bool = True) -> Union[torch.Tensor, PredictionResult]:
        """
        Forward pass with integrated uncertainty quantification.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            return_uncertainty: Whether to return full uncertainty analysis
            
        Returns:
            Predictions with uncertainty if return_uncertainty=True, else just predictions
        """
        if self.training_mode:
            # During training, use variational inference
            return self._training_forward(x)
        else:
            # During inference, provide complete uncertainty analysis
            if return_uncertainty:
                return self._inference_with_uncertainty(x)
            else:
                return self._simple_inference(x)
    
    def _training_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass during training with variational inference"""
        # Sample weights from variational distribution
        weight_samples = self.vb_uq.sample_weights(1)
        
        # Forward pass through base model with sampled weights
        # Note: This is a simplified version - actual implementation would
        # depend on how the base model integrates with variational weights
        predictions = self.base_model(x)
        
        return predictions
    
    def _simple_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Simple inference without uncertainty analysis"""
        with torch.no_grad():
            predictions = self.base_model(x)
        return predictions
    
    def _inference_with_uncertainty(self, x: torch.Tensor) -> List[PredictionResult]:
        """Complete inference with uncertainty quantification"""
        with torch.no_grad():
            batch_size = x.shape[0]
            results = []
            
            for i in range(batch_size):
                single_input = x[i:i+1]
                result = self._single_prediction_with_uncertainty(single_input)
                results.append(result)
            
            return results
    
    def _single_prediction_with_uncertainty(self, x: torch.Tensor) -> PredictionResult:
        """Single prediction with complete uncertainty analysis"""
        # Get prediction with uncertainty from variational Bayesian UQ
        def forward_fn(inputs, weights):
            # Simplified forward function - actual implementation would
            # integrate weights into the neural operator
            return self.base_model(inputs)
        
        mean_pred, uncertainty = self.vb_uq.predict_with_uncertainty(
            x, forward_fn, num_samples=self.config.num_mc_samples
        )
        
        # Compute confidence score
        confidence = self._compute_confidence(mean_pred, uncertainty)
        
        # Apply calibration if available
        calibrated_confidence = confidence
        if self.is_calibrated:
            calibrated_confidence = self.confidence_calibration.apply_calibration(
                torch.tensor([confidence])
            ).item()
        
        # Assess reliability
        reliability_metrics = self.reliability_assessor.assess_reliability(
            mean_pred, uncertainty, confidence
        )
        
        # Determine reliability level
        reliability_level = self._determine_reliability_level(
            calibrated_confidence, reliability_metrics
        )
        
        # Safety assessment
        safety_assessment = self._perform_safety_assessment(
            mean_pred, calibrated_confidence, reliability_level
        )
        
        # Get recommended action
        fault_type = self._classify_fault_type(mean_pred)
        recommended_action, _ = self.safety_thresholds.get_recommended_action(
            fault_type, calibrated_confidence
        )
        
        # Compute physics consistency (placeholder - would integrate with physics constraints)
        physics_consistency = self._compute_physics_consistency(mean_pred, x)
        
        return PredictionResult(
            prediction=mean_pred.squeeze(0),
            confidence=confidence,
            uncertainty=uncertainty.squeeze(0),
            reliability_level=reliability_level,
            safety_assessment=safety_assessment,
            physics_consistency=physics_consistency,
            calibrated_confidence=calibrated_confidence,
            recommended_action=recommended_action,
            timestamp=torch.tensor(0.0).item()  # Would use actual timestamp
        )
    
    def _compute_confidence(self, prediction: torch.Tensor, 
                          uncertainty: torch.Tensor) -> float:
        """Compute confidence score from prediction and uncertainty"""
        # Simple confidence metric: inverse of normalized uncertainty
        mean_uncertainty = torch.mean(uncertainty).item()
        confidence = 1.0 / (1.0 + mean_uncertainty)
        return min(max(confidence, 0.0), 1.0)
    
    def _determine_reliability_level(self, confidence: float, 
                                   reliability_metrics: ReliabilityMetrics) -> ReliabilityLevel:
        """Determine overall reliability level"""
        overall_reliability = reliability_metrics.overall_reliability
        
        if overall_reliability >= 0.95 and confidence >= 0.95:
            return ReliabilityLevel.CRITICAL
        elif overall_reliability >= 0.85 and confidence >= 0.85:
            return ReliabilityLevel.HIGH
        elif overall_reliability >= 0.70 and confidence >= 0.70:
            return ReliabilityLevel.MODERATE
        elif overall_reliability >= 0.50 and confidence >= 0.50:
            return ReliabilityLevel.LOW
        else:
            return ReliabilityLevel.UNCERTAIN
    
    def _perform_safety_assessment(self, prediction: torch.Tensor, 
                                 confidence: float, 
                                 reliability_level: ReliabilityLevel) -> Dict[str, Union[bool, float, str]]:
        """Perform comprehensive safety assessment"""
        fault_type = self._classify_fault_type(prediction)
        
        # Check if decision is safe for different scenarios
        safety_checks = {}
        for decision_type in ['critical_fault', 'warning_fault', 'normal_operation']:
            is_safe = self.safety_thresholds.is_decision_safe(decision_type, confidence)
            safety_margin = self.safety_thresholds.get_safety_margin(decision_type, confidence)
            
            safety_checks[f"{decision_type}_safe"] = is_safe
            safety_checks[f"{decision_type}_margin"] = safety_margin
        
        # Overall safety assessment
        safety_checks['overall_safe'] = reliability_level in [ReliabilityLevel.CRITICAL, ReliabilityLevel.HIGH]
        safety_checks['fault_type'] = fault_type
        safety_checks['reliability_level'] = reliability_level.value
        
        return safety_checks
    
    def _classify_fault_type(self, prediction: torch.Tensor) -> str:
        """Classify fault type from prediction (simplified)"""
        # This is a placeholder - actual implementation would depend on
        # the specific output format of the neural operator
        pred_max = torch.argmax(prediction, dim=-1).item()
        
        fault_types = ['normal', 'warning', 'critical']
        if pred_max < len(fault_types):
            return fault_types[pred_max]
        else:
            return 'unknown'
    
    def _compute_physics_consistency(self, prediction: torch.Tensor, 
                                   input_data: torch.Tensor) -> float:
        """Compute physics consistency score (placeholder)"""
        # This would integrate with the physics constraint system
        # For now, return a placeholder value
        return 0.85
    
    def calibrate_uncertainty(self, validation_data: torch.utils.data.DataLoader) -> CalibrationMetrics:
        """Calibrate uncertainty estimates using validation data"""
        logger.info("Starting uncertainty calibration")
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        self.eval()
        with torch.no_grad():
            for batch_x, batch_y in validation_data:
                # Get predictions with uncertainty
                results = self._inference_with_uncertainty(batch_x)
                
                for i, result in enumerate(results):
                    all_predictions.append(result.prediction.unsqueeze(0))
                    all_targets.append(batch_y[i:i+1])
                    all_confidences.append(result.confidence)
        
        # Stack all data
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        confidences = torch.tensor(all_confidences)
        
        # Perform calibration
        self.confidence_calibration.calibrate(predictions, targets, confidences)
        self.is_calibrated = True
        
        # Compute calibration metrics
        metrics = self.confidence_calibration.compute_calibration_metrics(
            predictions, targets, confidences
        )
        
        logger.info(f"Calibration completed. ECE: {metrics.expected_calibration_error:.4f}")
        return metrics
    
    def set_training_mode(self, training: bool = True) -> None:
        """Set training mode"""
        self.training_mode = training
        self.train(training)
    
    def get_uncertainty_config(self) -> UncertaintyConfig:
        """Get uncertainty configuration"""
        return self.config


class ReliabilityAssessor:
    """
    Assesses the reliability of predictions and uncertainty estimates.
    
    This class provides comprehensive reliability assessment for safety-critical
    motor fault diagnosis decisions.
    """
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        
    def assess_reliability(self, prediction: torch.Tensor, 
                         uncertainty: torch.Tensor, 
                         confidence: float) -> ReliabilityMetrics:
        """
        Assess the reliability of a prediction with uncertainty.
        
        Args:
            prediction: Model prediction
            uncertainty: Uncertainty estimate
            confidence: Confidence score
            
        Returns:
            Comprehensive reliability metrics
        """
        # Prediction reliability based on confidence and uncertainty consistency
        pred_reliability = self._assess_prediction_reliability(prediction, confidence)
        
        # Uncertainty reliability based on uncertainty estimate quality
        uncertainty_reliability = self._assess_uncertainty_reliability(uncertainty)
        
        # Physics consistency score (placeholder)
        physics_consistency = 0.85  # Would integrate with physics constraints
        
        # Calibration quality (placeholder - would use actual calibration data)
        calibration_quality = 0.80
        
        # Overall reliability as weighted combination
        overall_reliability = (
            0.4 * pred_reliability +
            0.3 * uncertainty_reliability +
            0.2 * physics_consistency +
            0.1 * calibration_quality
        )
        
        # Compute confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            prediction, uncertainty
        )
        
        return ReliabilityMetrics(
            prediction_reliability=pred_reliability,
            uncertainty_reliability=uncertainty_reliability,
            physics_consistency_score=physics_consistency,
            calibration_quality=calibration_quality,
            overall_reliability=overall_reliability,
            confidence_intervals=confidence_intervals
        )
    
    def _assess_prediction_reliability(self, prediction: torch.Tensor, 
                                     confidence: float) -> float:
        """Assess reliability of the prediction itself"""
        # Higher confidence generally indicates higher reliability
        # Also consider prediction magnitude and consistency
        pred_magnitude = torch.norm(prediction).item()
        
        # Normalize prediction magnitude to [0, 1] range
        normalized_magnitude = min(pred_magnitude / 10.0, 1.0)
        
        # Combine confidence and prediction characteristics
        reliability = 0.7 * confidence + 0.3 * normalized_magnitude
        return min(max(reliability, 0.0), 1.0)
    
    def _assess_uncertainty_reliability(self, uncertainty: torch.Tensor) -> float:
        """Assess reliability of the uncertainty estimate"""
        # Lower uncertainty variance indicates more reliable uncertainty estimates
        uncertainty_mean = torch.mean(uncertainty).item()
        uncertainty_std = torch.std(uncertainty).item()
        
        # Reliability is higher when uncertainty is consistent (low std)
        # and not too high or too low
        consistency_score = 1.0 / (1.0 + uncertainty_std)
        magnitude_score = 1.0 - min(uncertainty_mean, 1.0)
        
        reliability = 0.6 * consistency_score + 0.4 * magnitude_score
        return min(max(reliability, 0.0), 1.0)
    
    def _compute_confidence_intervals(self, prediction: torch.Tensor, 
                                    uncertainty: torch.Tensor) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for different confidence levels"""
        pred_np = prediction.detach().cpu().numpy().flatten()
        unc_np = uncertainty.detach().cpu().numpy().flatten()
        
        confidence_intervals = {}
        
        for level in self.config.confidence_levels:
            # Compute z-score for confidence level
            z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + level) / 2)).item()
            
            # Compute intervals for each output dimension
            lower_bounds = pred_np - z_score * unc_np
            upper_bounds = pred_np + z_score * unc_np
            
            confidence_intervals[f"{level:.0%}"] = (
                float(np.mean(lower_bounds)),
                float(np.mean(upper_bounds))
            )
        
        return confidence_intervals


class SafetyDecisionEngine:
    """
    Engine for making safety-critical decisions based on uncertainty-aware predictions.
    """
    
    def __init__(self, safety_thresholds: SafetyThresholds):
        self.safety_thresholds = safety_thresholds
        self.decision_history = []
        
    def make_safety_decision(self, prediction_result: PredictionResult) -> Dict[str, Union[str, bool, float]]:
        """
        Make a safety-critical decision based on prediction result.
        
        Args:
            prediction_result: Complete prediction result with uncertainty
            
        Returns:
            Safety decision with justification
        """
        # Extract key information
        confidence = prediction_result.calibrated_confidence
        reliability_level = prediction_result.reliability_level
        safety_assessment = prediction_result.safety_assessment
        
        # Determine decision based on reliability and safety thresholds
        if reliability_level == ReliabilityLevel.CRITICAL:
            if safety_assessment.get('critical_fault_safe', False):
                decision = "proceed_with_high_confidence"
                justification = "High reliability and confidence for critical decision"
            else:
                decision = "require_human_oversight"
                justification = "Critical fault detected but confidence below threshold"
        
        elif reliability_level == ReliabilityLevel.HIGH:
            if safety_assessment.get('warning_fault_safe', False):
                decision = "proceed_with_monitoring"
                justification = "High reliability for warning-level decision"
            else:
                decision = "increase_monitoring"
                justification = "Warning condition with moderate confidence"
        
        elif reliability_level in [ReliabilityLevel.MODERATE, ReliabilityLevel.LOW]:
            decision = "collect_more_data"
            justification = "Insufficient reliability for autonomous decision"
        
        else:  # UNCERTAIN
            decision = "manual_inspection_required"
            justification = "Uncertainty too high for automated decision"
        
        # Create decision record
        decision_record = {
            'decision': decision,
            'justification': justification,
            'confidence': confidence,
            'reliability_level': reliability_level.value,
            'timestamp': prediction_result.timestamp,
            'safe_for_autonomous': reliability_level in [ReliabilityLevel.CRITICAL, ReliabilityLevel.HIGH]
        }
        
        # Record decision
        self.decision_history.append(decision_record)
        
        return decision_record
    
    def get_decision_history(self) -> List[Dict]:
        """Get history of safety decisions"""
        return self.decision_history.copy()
    
    def analyze_decision_patterns(self) -> Dict[str, float]:
        """Analyze patterns in safety decisions"""
        if not self.decision_history:
            return {}
        
        total_decisions = len(self.decision_history)
        
        # Count decision types
        decision_counts = {}
        autonomous_count = 0
        
        for record in self.decision_history:
            decision = record['decision']
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            if record['safe_for_autonomous']:
                autonomous_count += 1
        
        # Compute statistics
        analysis = {
            'total_decisions': total_decisions,
            'autonomous_decision_rate': autonomous_count / total_decisions,
            'average_confidence': np.mean([r['confidence'] for r in self.decision_history]),
            'decision_distribution': {k: v/total_decisions for k, v in decision_counts.items()}
        }
        
        return analysis