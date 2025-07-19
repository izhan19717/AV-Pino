"""
Physics Consistency Validation for AV-PINO System.

This module implements validation tests for PDE constraint satisfaction
and physics consistency analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod

from ..physics.constraints import PDEConstraint, PhysicsConstraintLayer

logger = logging.getLogger(__name__)


@dataclass
class PhysicsValidationMetrics:
    """Metrics for physics consistency validation."""
    constraint_name: str
    mean_residual: float
    max_residual: float
    residual_std: float
    violation_percentage: float
    consistency_score: float  # 0-1, higher is better
    
    # Detailed residual statistics
    residual_distribution: Dict[str, float] = field(default_factory=dict)
    spatial_residual_pattern: Optional[np.ndarray] = None
    temporal_residual_pattern: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'constraint_name': self.constraint_name,
            'mean_residual': self.mean_residual,
            'max_residual': self.max_residual,
            'residual_std': self.residual_std,
            'violation_percentage': self.violation_percentage,
            'consistency_score': self.consistency_score,
            'residual_distribution': self.residual_distribution,
            'spatial_residual_pattern': (self.spatial_residual_pattern.tolist() 
                                       if self.spatial_residual_pattern is not None else None),
            'temporal_residual_pattern': (self.temporal_residual_pattern.tolist() 
                                        if self.temporal_residual_pattern is not None else None)
        }


@dataclass
class PhysicsValidationReport:
    """Comprehensive physics validation report."""
    test_name: str
    constraint_metrics: Dict[str, PhysicsValidationMetrics]
    overall_consistency_score: float
    physics_loss_evolution: List[float]
    violation_analysis: Dict[str, Any]
    recommendations: List[str]
    timestamp: str
    
    def save_report(self, filepath: str):
        """Save physics validation report."""
        import json
        
        report_data = {
            'test_name': self.test_name,
            'constraint_metrics': {k: v.to_dict() for k, v in self.constraint_metrics.items()},
            'overall_consistency_score': self.overall_consistency_score,
            'physics_loss_evolution': self.physics_loss_evolution,
            'violation_analysis': self.violation_analysis,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Physics validation report saved to {filepath}")


class PDEResidualAnalyzer:
    """Analyzes PDE residuals for physics consistency validation."""
    
    def __init__(self, tolerance_threshold: float = 1e-3):
        self.tolerance_threshold = tolerance_threshold
        
    def analyze_residual(self, residual: torch.Tensor, 
                        constraint_name: str,
                        coords: Optional[torch.Tensor] = None) -> PhysicsValidationMetrics:
        """Analyze PDE residual for a specific constraint."""
        # Convert to numpy for analysis
        residual_np = residual.detach().cpu().numpy()
        
        # Basic statistics
        mean_residual = float(np.mean(np.abs(residual_np)))
        max_residual = float(np.max(np.abs(residual_np)))
        residual_std = float(np.std(residual_np))
        
        # Violation percentage (residuals above threshold)
        violations = np.abs(residual_np) > self.tolerance_threshold
        violation_percentage = float(np.mean(violations) * 100)
        
        # Consistency score (1 - normalized mean residual)
        consistency_score = max(0.0, 1.0 - mean_residual / (mean_residual + self.tolerance_threshold))
        
        # Residual distribution analysis
        residual_distribution = self._analyze_residual_distribution(residual_np)
        
        # Spatial and temporal patterns (if coordinates provided)
        spatial_pattern = None
        temporal_pattern = None
        if coords is not None:
            spatial_pattern, temporal_pattern = self._analyze_residual_patterns(
                residual_np, coords.detach().cpu().numpy()
            )
        
        return PhysicsValidationMetrics(
            constraint_name=constraint_name,
            mean_residual=mean_residual,
            max_residual=max_residual,
            residual_std=residual_std,
            violation_percentage=violation_percentage,
            consistency_score=consistency_score,
            residual_distribution=residual_distribution,
            spatial_residual_pattern=spatial_pattern,
            temporal_residual_pattern=temporal_pattern
        )
    
    def _analyze_residual_distribution(self, residual: np.ndarray) -> Dict[str, float]:
        """Analyze the distribution of residual values."""
        abs_residual = np.abs(residual)
        
        return {
            'min': float(np.min(abs_residual)),
            'max': float(np.max(abs_residual)),
            'mean': float(np.mean(abs_residual)),
            'median': float(np.median(abs_residual)),
            'std': float(np.std(abs_residual)),
            'percentile_90': float(np.percentile(abs_residual, 90)),
            'percentile_95': float(np.percentile(abs_residual, 95)),
            'percentile_99': float(np.percentile(abs_residual, 99))
        }
    
    def _analyze_residual_patterns(self, residual: np.ndarray, 
                                  coords: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Analyze spatial and temporal patterns in residuals."""
        spatial_pattern = None
        temporal_pattern = None
        
        try:
            if coords.shape[1] >= 2:  # Has spatial coordinates
                # Compute spatial correlation of residuals
                spatial_pattern = np.correlate(residual.flatten(), coords[:, 0], mode='same')
            
            if coords.shape[1] >= 3:  # Has temporal coordinate
                # Compute temporal evolution of residuals
                temporal_pattern = np.correlate(residual.flatten(), coords[:, -1], mode='same')
        except Exception as e:
            logger.warning(f"Failed to analyze residual patterns: {e}")
        
        return spatial_pattern, temporal_pattern


class PhysicsConsistencyValidator:
    """Validates physics consistency of neural operator predictions."""
    
    def __init__(self, physics_constraints: PhysicsConstraintLayer,
                 tolerance_threshold: float = 1e-3):
        self.physics_constraints = physics_constraints
        self.tolerance_threshold = tolerance_threshold
        self.residual_analyzer = PDEResidualAnalyzer(tolerance_threshold)
        
    def validate_physics_consistency(self, 
                                   predictions: torch.Tensor,
                                   input_data: torch.Tensor,
                                   coords: torch.Tensor,
                                   test_name: str = "Physics Consistency Test"
                                   ) -> PhysicsValidationReport:
        """Validate physics consistency of predictions."""
        logger.info(f"Starting physics consistency validation: {test_name}")
        
        # Get physics residuals
        _, residuals_dict = self.physics_constraints(predictions, input_data, coords)
        
        # Analyze each constraint
        constraint_metrics = {}
        for constraint_name, residual in residuals_dict.items():
            if constraint_name != 'total_physics_loss':  # Skip total loss
                metrics = self.residual_analyzer.analyze_residual(
                    residual, constraint_name, coords
                )
                constraint_metrics[constraint_name] = metrics
        
        # Compute overall consistency score
        overall_score = self._compute_overall_consistency_score(constraint_metrics)
        
        # Analyze violations
        violation_analysis = self._analyze_violations(constraint_metrics)
        
        # Generate recommendations
        recommendations = self._generate_physics_recommendations(
            constraint_metrics, overall_score
        )
        
        # Create report
        report = PhysicsValidationReport(
            test_name=test_name,
            constraint_metrics=constraint_metrics,
            overall_consistency_score=overall_score,
            physics_loss_evolution=[],  # Would be populated during training
            violation_analysis=violation_analysis,
            recommendations=recommendations,
            timestamp=str(np.datetime64('now'))
        )
        
        logger.info("Physics consistency validation completed")
        return report
    
    def validate_conservation_laws(self, predictions: torch.Tensor,
                                 input_data: torch.Tensor) -> Dict[str, float]:
        """Validate conservation laws (energy, momentum, etc.)."""
        conservation_violations = {}
        
        try:
            # Energy conservation check
            energy_violation = self._check_energy_conservation(predictions, input_data)
            conservation_violations['energy'] = energy_violation
            
            # Momentum conservation check
            momentum_violation = self._check_momentum_conservation(predictions, input_data)
            conservation_violations['momentum'] = momentum_violation
            
            # Charge conservation check (for electromagnetic fields)
            charge_violation = self._check_charge_conservation(predictions, input_data)
            conservation_violations['charge'] = charge_violation
            
        except Exception as e:
            logger.error(f"Failed to validate conservation laws: {e}")
        
        return conservation_violations
    
    def _check_energy_conservation(self, predictions: torch.Tensor, 
                                 input_data: torch.Tensor) -> float:
        """Check energy conservation violation."""
        # Simplified energy conservation check
        # In practice, would compute proper energy terms from predictions
        
        # Extract relevant fields (assuming specific prediction format)
        if predictions.shape[1] >= 6:
            E_field = predictions[:, :3]  # Electric field
            B_field = predictions[:, 3:6]  # Magnetic field
            
            # Electromagnetic energy density: (ε₀E² + B²/μ₀)/2
            epsilon_0 = 8.854e-12
            mu_0 = 4e-7 * np.pi
            
            energy_density = (epsilon_0 * torch.sum(E_field**2, dim=1) + 
                            torch.sum(B_field**2, dim=1) / mu_0) / 2
            
            # Check energy conservation (simplified)
            energy_change = torch.std(energy_density)
            return float(energy_change)
        
        return 0.0
    
    def _check_momentum_conservation(self, predictions: torch.Tensor,
                                   input_data: torch.Tensor) -> float:
        """Check momentum conservation violation."""
        # Simplified momentum conservation check
        if predictions.shape[1] >= 9:
            velocity = predictions[:, 6:9]  # Velocity field
            momentum_change = torch.std(torch.sum(velocity, dim=1))
            return float(momentum_change)
        
        return 0.0
    
    def _check_charge_conservation(self, predictions: torch.Tensor,
                                 input_data: torch.Tensor) -> float:
        """Check charge conservation violation."""
        # Simplified charge conservation check
        # ∇ · J + ∂ρ/∂t = 0 (continuity equation)
        
        if input_data.shape[1] >= 3:
            current_density = input_data[:, :3]  # Current density J
            charge_change = torch.std(torch.sum(current_density, dim=1))
            return float(charge_change)
        
        return 0.0
    
    def _compute_overall_consistency_score(self, 
                                         constraint_metrics: Dict[str, PhysicsValidationMetrics]
                                         ) -> float:
        """Compute overall physics consistency score."""
        if not constraint_metrics:
            return 0.0
        
        # Weighted average of individual consistency scores
        scores = [metrics.consistency_score for metrics in constraint_metrics.values()]
        return float(np.mean(scores))
    
    def _analyze_violations(self, constraint_metrics: Dict[str, PhysicsValidationMetrics]
                          ) -> Dict[str, Any]:
        """Analyze physics constraint violations."""
        violation_analysis = {
            'total_constraints': len(constraint_metrics),
            'constraints_with_violations': 0,
            'worst_violating_constraint': None,
            'violation_severity': 'low'
        }
        
        worst_violation = 0.0
        worst_constraint = None
        
        for constraint_name, metrics in constraint_metrics.items():
            if metrics.violation_percentage > 0:
                violation_analysis['constraints_with_violations'] += 1
                
                if metrics.violation_percentage > worst_violation:
                    worst_violation = metrics.violation_percentage
                    worst_constraint = constraint_name
        
        violation_analysis['worst_violating_constraint'] = worst_constraint
        
        # Determine violation severity
        if worst_violation > 50:
            violation_analysis['violation_severity'] = 'high'
        elif worst_violation > 20:
            violation_analysis['violation_severity'] = 'medium'
        else:
            violation_analysis['violation_severity'] = 'low'
        
        return violation_analysis
    
    def _generate_physics_recommendations(self, 
                                        constraint_metrics: Dict[str, PhysicsValidationMetrics],
                                        overall_score: float) -> List[str]:
        """Generate recommendations based on physics validation results."""
        recommendations = []
        
        # Overall consistency assessment
        if overall_score > 0.9:
            recommendations.append(
                "Excellent physics consistency achieved. Model satisfies PDE constraints well."
            )
        elif overall_score > 0.7:
            recommendations.append(
                "Good physics consistency. Minor constraint violations detected."
            )
        else:
            recommendations.append(
                "Poor physics consistency. Significant constraint violations require attention."
            )
        
        # Constraint-specific recommendations
        for constraint_name, metrics in constraint_metrics.items():
            if metrics.violation_percentage > 20:
                recommendations.append(
                    f"High violation rate ({metrics.violation_percentage:.1f}%) in {constraint_name}. "
                    "Consider increasing constraint weight or reviewing PDE formulation."
                )
            elif metrics.consistency_score < 0.5:
                recommendations.append(
                    f"Low consistency score ({metrics.consistency_score:.2f}) for {constraint_name}. "
                    "Review constraint implementation and numerical stability."
                )
        
        # Residual pattern analysis
        high_residual_constraints = [
            name for name, metrics in constraint_metrics.items() 
            if metrics.max_residual > 10 * self.tolerance_threshold
        ]
        
        if high_residual_constraints:
            recommendations.append(
                f"High residuals detected in: {', '.join(high_residual_constraints)}. "
                "Consider adaptive constraint weighting or numerical regularization."
            )
        
        # General recommendations
        if len(recommendations) == 1 and overall_score > 0.9:
            recommendations.append(
                "Consider reducing physics constraint weights to improve computational efficiency "
                "while maintaining physics consistency."
            )
        
        return recommendations
    
    def monitor_physics_during_training(self, predictions: torch.Tensor,
                                      input_data: torch.Tensor,
                                      coords: torch.Tensor) -> Dict[str, float]:
        """Monitor physics consistency during training (lightweight version)."""
        # Get physics residuals
        _, residuals_dict = self.physics_constraints(predictions, input_data, coords)
        
        # Compute simple metrics for monitoring
        monitoring_metrics = {}
        for constraint_name, residual in residuals_dict.items():
            if constraint_name != 'total_physics_loss':
                residual_np = residual.detach().cpu().numpy()
                monitoring_metrics[f"{constraint_name}_mean_residual"] = float(np.mean(np.abs(residual_np)))
                monitoring_metrics[f"{constraint_name}_max_residual"] = float(np.max(np.abs(residual_np)))
        
        # Overall physics loss
        if 'total_physics_loss' in residuals_dict:
            monitoring_metrics['total_physics_loss'] = float(residuals_dict['total_physics_loss'])
        
        return monitoring_metrics