"""
Generalization Testing Module for AV-PINO Motor Fault Diagnosis.

This module implements comprehensive generalization testing including cross-motor
configuration validation and performance evaluation on unseen motor types and
operating conditions.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from ..data.cwru_loader import CWRUDataset, CWRUDataSample
from ..inference.fault_classifier import FaultClassificationSystem, FaultPrediction
from ..physics.constraints import PhysicsConstraintLayer

logger = logging.getLogger(__name__)


class MotorType(Enum):
    """Different motor types for generalization testing."""
    INDUCTION_MOTOR = "induction"
    SYNCHRONOUS_MOTOR = "synchronous"
    BRUSHLESS_DC = "brushless_dc"
    STEPPER_MOTOR = "stepper"


class OperatingCondition(Enum):
    """Operating condition categories."""
    LOW_LOAD = "low_load"
    MEDIUM_LOAD = "medium_load"
    HIGH_LOAD = "high_load"
    VARIABLE_SPEED = "variable_speed"
    TRANSIENT = "transient"


@dataclass
class MotorConfiguration:
    """Motor configuration specification."""
    motor_type: MotorType
    power_rating: float  # kW
    voltage: float  # V
    frequency: float  # Hz
    pole_pairs: int
    bearing_type: str
    operating_condition: OperatingCondition
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'motor_type': self.motor_type.value,
            'power_rating': self.power_rating,
            'voltage': self.voltage,
            'frequency': self.frequency,
            'pole_pairs': self.pole_pairs,
            'bearing_type': self.bearing_type,
            'operating_condition': self.operating_condition.value
        }


@dataclass
class GeneralizationMetrics:
    """Metrics for generalization performance assessment."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: np.ndarray
    confidence_scores: List[float]
    uncertainty_scores: List[float]
    physics_consistency_scores: List[float]
    
    # Cross-motor specific metrics
    cross_motor_accuracy: Dict[str, float] = field(default_factory=dict)
    operating_condition_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Degradation metrics
    accuracy_drop: float = 0.0
    confidence_degradation: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'confidence_scores': self.confidence_scores,
            'uncertainty_scores': self.uncertainty_scores,
            'physics_consistency_scores': self.physics_consistency_scores,
            'cross_motor_accuracy': self.cross_motor_accuracy,
            'operating_condition_accuracy': self.operating_condition_accuracy,
            'accuracy_drop': self.accuracy_drop,
            'confidence_degradation': self.confidence_degradation
        }


@dataclass
class GeneralizationReport:
    """Comprehensive generalization testing report."""
    test_name: str
    baseline_metrics: GeneralizationMetrics
    generalization_metrics: Dict[str, GeneralizationMetrics]
    motor_configurations: Dict[str, MotorConfiguration]
    summary_statistics: Dict[str, float]
    recommendations: List[str]
    timestamp: str
    
    def save_report(self, filepath: str):
        """Save report to JSON file."""
        report_data = {
            'test_name': self.test_name,
            'baseline_metrics': self.baseline_metrics.to_dict(),
            'generalization_metrics': {
                k: v.to_dict() for k, v in self.generalization_metrics.items()
            },
            'motor_configurations': {
                k: v.to_dict() for k, v in self.motor_configurations.items()
            },
            'summary_statistics': self.summary_statistics,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Generalization report saved to {filepath}")


class CrossMotorValidator:
    """Validates model performance across different motor configurations."""
    
    def __init__(self, model: FaultClassificationSystem, 
                 physics_constraints: Optional[PhysicsConstraintLayer] = None):
        self.model = model
        self.physics_constraints = physics_constraints
        self.motor_configs = self._create_motor_configurations()
        
    def _create_motor_configurations(self) -> Dict[str, MotorConfiguration]:
        """Create diverse motor configurations for testing."""
        configs = {
            'baseline_cwru': MotorConfiguration(
                motor_type=MotorType.INDUCTION_MOTOR,
                power_rating=2.0,
                voltage=460,
                frequency=60,
                pole_pairs=2,
                bearing_type='6205-2RS',
                operating_condition=OperatingCondition.MEDIUM_LOAD
            ),
            'high_power_induction': MotorConfiguration(
                motor_type=MotorType.INDUCTION_MOTOR,
                power_rating=10.0,
                voltage=480,
                frequency=60,
                pole_pairs=4,
                bearing_type='6308',
                operating_condition=OperatingCondition.HIGH_LOAD
            ),
            'low_power_induction': MotorConfiguration(
                motor_type=MotorType.INDUCTION_MOTOR,
                power_rating=0.5,
                voltage=230,
                frequency=50,
                pole_pairs=2,
                bearing_type='6203',
                operating_condition=OperatingCondition.LOW_LOAD
            ),
            'synchronous_motor': MotorConfiguration(
                motor_type=MotorType.SYNCHRONOUS_MOTOR,
                power_rating=5.0,
                voltage=400,
                frequency=50,
                pole_pairs=3,
                bearing_type='6206',
                operating_condition=OperatingCondition.MEDIUM_LOAD
            ),
            'brushless_dc': MotorConfiguration(
                motor_type=MotorType.BRUSHLESS_DC,
                power_rating=1.5,
                voltage=48,
                frequency=0,  # DC motor
                pole_pairs=4,
                bearing_type='6204',
                operating_condition=OperatingCondition.VARIABLE_SPEED
            )
        }
        return configs
    
    def validate_cross_motor_performance(self, test_data: Dict[str, torch.Tensor],
                                       test_labels: Dict[str, torch.Tensor]
                                       ) -> Dict[str, GeneralizationMetrics]:
        """Validate performance across different motor configurations."""
        results = {}
        
        for config_name, config in self.motor_configs.items():
            if config_name in test_data:
                logger.info(f"Testing on motor configuration: {config_name}")
                
                # Get test data for this configuration
                features = test_data[config_name]
                labels = test_labels[config_name]
                
                # Make predictions
                predictions = self.model.predict(features)
                
                # Extract prediction data
                pred_labels = [pred.fault_type.value for pred in predictions]
                confidences = [pred.confidence for pred in predictions]
                uncertainties = [pred.uncertainty for pred in predictions]
                physics_scores = [pred.physics_consistency for pred in predictions]
                
                # Convert labels to strings for consistency
                true_labels = [self.model.fault_mapper.index_to_type(idx.item()).value 
                              for idx in labels]
                
                # Compute metrics
                metrics = self._compute_generalization_metrics(
                    true_labels, pred_labels, confidences, 
                    uncertainties, physics_scores
                )
                
                results[config_name] = metrics
                
        return results
    
    def _compute_generalization_metrics(self, true_labels: List[str], 
                                      pred_labels: List[str],
                                      confidences: List[float],
                                      uncertainties: List[float],
                                      physics_scores: List[float]
                                      ) -> GeneralizationMetrics:
        """Compute comprehensive generalization metrics."""
        # Basic classification metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, zero_division=0
        )
        
        # Get unique labels for mapping
        unique_labels = sorted(list(set(true_labels + pred_labels)))
        
        # Create per-class metrics dictionaries
        precision_dict = {label: precision[i] if i < len(precision) else 0.0 
                         for i, label in enumerate(unique_labels)}
        recall_dict = {label: recall[i] if i < len(recall) else 0.0 
                      for i, label in enumerate(unique_labels)}
        f1_dict = {label: f1[i] if i < len(f1) else 0.0 
                  for i, label in enumerate(unique_labels)}
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
        
        return GeneralizationMetrics(
            accuracy=accuracy,
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            confusion_matrix=cm,
            confidence_scores=confidences,
            uncertainty_scores=uncertainties,
            physics_consistency_scores=physics_scores
        )


class OperatingConditionValidator:
    """Validates model performance under different operating conditions."""
    
    def __init__(self, model: FaultClassificationSystem):
        self.model = model
        
    def validate_operating_conditions(self, test_data: Dict[str, torch.Tensor],
                                    test_labels: Dict[str, torch.Tensor]
                                    ) -> Dict[str, GeneralizationMetrics]:
        """Validate performance under different operating conditions."""
        results = {}
        
        # Define operating condition test scenarios
        operating_conditions = {
            'low_load': {'load_range': (0, 1), 'rpm_range': (1700, 1800)},
            'medium_load': {'load_range': (1, 2), 'rpm_range': (1750, 1800)},
            'high_load': {'load_range': (2, 3), 'rpm_range': (1720, 1750)},
            'variable_speed': {'load_range': (0, 3), 'rpm_range': (1500, 2000)},
            'transient': {'load_range': (0, 3), 'rpm_range': (1000, 2500)}
        }
        
        for condition_name, condition_params in operating_conditions.items():
            if condition_name in test_data:
                logger.info(f"Testing operating condition: {condition_name}")
                
                features = test_data[condition_name]
                labels = test_labels[condition_name]
                
                # Make predictions
                predictions = self.model.predict(features)
                
                # Extract prediction data
                pred_labels = [pred.fault_type.value for pred in predictions]
                confidences = [pred.confidence for pred in predictions]
                uncertainties = [pred.uncertainty for pred in predictions]
                physics_scores = [pred.physics_consistency for pred in predictions]
                
                # Convert labels
                true_labels = [self.model.fault_mapper.index_to_type(idx.item()).value 
                              for idx in labels]
                
                # Compute metrics
                metrics = self._compute_operating_condition_metrics(
                    true_labels, pred_labels, confidences, 
                    uncertainties, physics_scores, condition_params
                )
                
                results[condition_name] = metrics
                
        return results
    
    def _compute_operating_condition_metrics(self, true_labels: List[str],
                                           pred_labels: List[str],
                                           confidences: List[float],
                                           uncertainties: List[float],
                                           physics_scores: List[float],
                                           condition_params: Dict[str, Any]
                                           ) -> GeneralizationMetrics:
        """Compute metrics for specific operating conditions."""
        # Use the same computation as cross-motor validation
        validator = CrossMotorValidator(self.model)
        metrics = validator._compute_generalization_metrics(
            true_labels, pred_labels, confidences, uncertainties, physics_scores
        )
        
        # Add operating condition specific analysis
        # Analyze performance degradation under challenging conditions
        if condition_params.get('load_range', [0, 0])[1] > 2:  # High load
            metrics.accuracy_drop = max(0, 0.95 - metrics.accuracy)  # Assume 95% baseline
        
        if condition_params.get('rpm_range', [0, 0])[1] > 2000:  # High speed
            metrics.confidence_degradation = max(0, 0.9 - np.mean(confidences))
            
        return metrics


class GeneralizationTester:
    """Main class for comprehensive generalization testing."""
    
    def __init__(self, model: FaultClassificationSystem,
                 physics_constraints: Optional[PhysicsConstraintLayer] = None):
        self.model = model
        self.physics_constraints = physics_constraints
        self.cross_motor_validator = CrossMotorValidator(model, physics_constraints)
        self.operating_condition_validator = OperatingConditionValidator(model)
        
    def run_comprehensive_generalization_test(self, 
                                            baseline_data: Tuple[torch.Tensor, torch.Tensor],
                                            test_data: Dict[str, torch.Tensor],
                                            test_labels: Dict[str, torch.Tensor],
                                            test_name: str = "Generalization Test"
                                            ) -> GeneralizationReport:
        """Run comprehensive generalization testing."""
        logger.info(f"Starting comprehensive generalization test: {test_name}")
        
        # Compute baseline metrics
        baseline_features, baseline_labels = baseline_data
        baseline_predictions = self.model.predict(baseline_features)
        
        baseline_pred_labels = [pred.fault_type.value for pred in baseline_predictions]
        baseline_confidences = [pred.confidence for pred in baseline_predictions]
        baseline_uncertainties = [pred.uncertainty for pred in baseline_predictions]
        baseline_physics = [pred.physics_consistency for pred in baseline_predictions]
        
        baseline_true_labels = [self.model.fault_mapper.index_to_type(idx.item()).value 
                               for idx in baseline_labels]
        
        baseline_metrics = self.cross_motor_validator._compute_generalization_metrics(
            baseline_true_labels, baseline_pred_labels, baseline_confidences,
            baseline_uncertainties, baseline_physics
        )
        
        # Cross-motor validation
        logger.info("Running cross-motor validation...")
        cross_motor_results = self.cross_motor_validator.validate_cross_motor_performance(
            test_data, test_labels
        )
        
        # Operating condition validation
        logger.info("Running operating condition validation...")
        operating_condition_results = self.operating_condition_validator.validate_operating_conditions(
            test_data, test_labels
        )
        
        # Combine results
        all_results = {**cross_motor_results, **operating_condition_results}
        
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(baseline_metrics, all_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(baseline_metrics, all_results)
        
        # Create report
        report = GeneralizationReport(
            test_name=test_name,
            baseline_metrics=baseline_metrics,
            generalization_metrics=all_results,
            motor_configurations=self.cross_motor_validator.motor_configs,
            summary_statistics=summary_stats,
            recommendations=recommendations,
            timestamp=str(np.datetime64('now'))
        )
        
        logger.info("Generalization testing completed")
        return report
    
    def _compute_summary_statistics(self, baseline_metrics: GeneralizationMetrics,
                                  test_results: Dict[str, GeneralizationMetrics]
                                  ) -> Dict[str, float]:
        """Compute summary statistics across all tests."""
        accuracies = [metrics.accuracy for metrics in test_results.values()]
        confidences = []
        uncertainties = []
        physics_scores = []
        
        for metrics in test_results.values():
            confidences.extend(metrics.confidence_scores)
            uncertainties.extend(metrics.uncertainty_scores)
            physics_scores.extend(metrics.physics_consistency_scores)
        
        return {
            'baseline_accuracy': baseline_metrics.accuracy,
            'mean_generalization_accuracy': np.mean(accuracies),
            'std_generalization_accuracy': np.std(accuracies),
            'min_generalization_accuracy': np.min(accuracies),
            'max_generalization_accuracy': np.max(accuracies),
            'mean_accuracy_drop': baseline_metrics.accuracy - np.mean(accuracies),
            'mean_confidence': np.mean(confidences),
            'mean_uncertainty': np.mean(uncertainties),
            'mean_physics_consistency': np.mean(physics_scores),
            'generalization_robustness': np.min(accuracies) / baseline_metrics.accuracy
        }
    
    def _generate_recommendations(self, baseline_metrics: GeneralizationMetrics,
                                test_results: Dict[str, GeneralizationMetrics]
                                ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze accuracy drops
        accuracies = [metrics.accuracy for metrics in test_results.values()]
        mean_accuracy = np.mean(accuracies)
        accuracy_drop = baseline_metrics.accuracy - mean_accuracy
        
        if accuracy_drop > 0.1:
            recommendations.append(
                "Significant accuracy drop detected in generalization tests. "
                "Consider domain adaptation techniques or additional training data."
            )
        
        # Analyze confidence degradation
        all_confidences = []
        for metrics in test_results.values():
            all_confidences.extend(metrics.confidence_scores)
        
        if np.mean(all_confidences) < 0.8:
            recommendations.append(
                "Low confidence scores in generalization tests. "
                "Consider uncertainty calibration or confidence threshold adjustment."
            )
        
        # Analyze physics consistency
        all_physics_scores = []
        for metrics in test_results.values():
            all_physics_scores.extend(metrics.physics_consistency_scores)
        
        if np.mean(all_physics_scores) < 0.7:
            recommendations.append(
                "Physics consistency degradation detected. "
                "Review physics constraint weights and PDE formulations."
            )
        
        # Motor-specific recommendations
        worst_motor = min(test_results.keys(), 
                         key=lambda k: test_results[k].accuracy)
        if test_results[worst_motor].accuracy < 0.7:
            recommendations.append(
                f"Poor performance on {worst_motor} configuration. "
                "Consider motor-specific fine-tuning or feature engineering."
            )
        
        if not recommendations:
            recommendations.append(
                "Generalization performance is satisfactory across all test conditions."
            )
        
        return recommendations
    
    def assess_model_generalization(self, test_data: Dict[str, torch.Tensor],
                                  test_labels: Dict[str, torch.Tensor]
                                  ) -> Dict[str, float]:
        """Quick assessment of model generalization capability."""
        results = {}
        
        for config_name in test_data.keys():
            features = test_data[config_name]
            labels = test_labels[config_name]
            
            predictions = self.model.predict(features)
            pred_labels = [pred.fault_type.value for pred in predictions]
            true_labels = [self.model.fault_mapper.index_to_type(idx.item()).value 
                          for idx in labels]
            
            accuracy = accuracy_score(true_labels, pred_labels)
            results[config_name] = accuracy
            
        return results