"""
Variational Bayesian Uncertainty Quantification for AV-PINO Motor Fault Diagnosis

This module implements physics-informed uncertainty quantification using variational
Bayesian methods with Gaussian process priors for neural operator parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification"""
    prior_mean: float = 0.0
    prior_std: float = 1.0
    num_mc_samples: int = 100
    kl_weight: float = 1e-3
    calibration_bins: int = 15
    safety_threshold: float = 0.8
    confidence_levels: List[float] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.68, 0.95, 0.99]


@dataclass
class CalibrationMetrics:
    """Metrics for uncertainty calibration assessment"""
    expected_calibration_error: float
    maximum_calibration_error: float
    average_confidence: float
    average_accuracy: float
    reliability_diagram: Dict[str, np.ndarray]
    brier_score: float


class ConfidenceCalibration:
    """
    Implements uncertainty calibration for neural operator predictions.
    
    This class provides methods to calibrate prediction confidence scores
    and assess the reliability of uncertainty estimates.
    """
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.calibration_map = None
        self.is_calibrated = False
        
    def calibrate(self, predictions: torch.Tensor, targets: torch.Tensor, 
                  confidences: torch.Tensor) -> None:
        """
        Calibrate confidence scores using Platt scaling or isotonic regression.
        
        Args:
            predictions: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            confidences: Confidence scores [N]
        """
        logger.info("Starting confidence calibration")
        
        # Convert to numpy for calibration
        pred_probs = F.softmax(predictions, dim=1).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        conf_np = confidences.detach().cpu().numpy()
        
        # Create calibration mapping using binning
        self.calibration_map = self._create_calibration_map(
            pred_probs, targets_np, conf_np
        )
        self.is_calibrated = True
        logger.info("Confidence calibration completed")
    
    def _create_calibration_map(self, predictions: np.ndarray, 
                               targets: np.ndarray, 
                               confidences: np.ndarray) -> Dict[float, float]:
        """Create calibration mapping from confidence to accuracy"""
        calibration_map = {}
        
        # Create bins for confidence scores
        bin_boundaries = np.linspace(0, 1, self.config.calibration_bins + 1)
        
        for i in range(len(bin_boundaries) - 1):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this confidence bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Calculate accuracy for this bin
                bin_predictions = predictions[in_bin]
                bin_targets = targets[in_bin]
                # Handle multi-dimensional targets by converting to class labels
                if bin_targets.ndim > 1:
                    # For multi-dimensional targets, use argmax to get class labels
                    bin_targets_labels = np.argmax(bin_targets, axis=1)
                else:
                    bin_targets_labels = bin_targets
                
                bin_accuracy = np.mean(
                    np.argmax(bin_predictions, axis=1) == bin_targets_labels
                )
                
                # Map bin center to accuracy
                bin_center = (bin_lower + bin_upper) / 2
                calibration_map[bin_center] = bin_accuracy
        
        return calibration_map
    
    def apply_calibration(self, confidences: torch.Tensor) -> torch.Tensor:
        """Apply calibration mapping to confidence scores"""
        if not self.is_calibrated:
            logger.warning("Calibration not performed, returning original confidences")
            return confidences
        
        calibrated = torch.zeros_like(confidences)
        conf_np = confidences.detach().cpu().numpy()
        
        for i, conf in enumerate(conf_np):
            # Find closest calibration point
            closest_bin = min(self.calibration_map.keys(), 
                            key=lambda x: abs(x - conf))
            calibrated[i] = self.calibration_map[closest_bin]
        
        return calibrated
    
    def compute_calibration_metrics(self, predictions: torch.Tensor, 
                                   targets: torch.Tensor, 
                                   confidences: torch.Tensor) -> CalibrationMetrics:
        """Compute comprehensive calibration metrics"""
        # Check if predictions are already probabilities (sum to 1)
        pred_sums = torch.sum(predictions, dim=1)
        if torch.allclose(pred_sums, torch.ones_like(pred_sums), atol=1e-6):
            # Already probabilities, don't apply softmax
            pred_probs = predictions.detach().cpu().numpy()
        else:
            # Apply softmax to convert logits to probabilities
            pred_probs = F.softmax(predictions, dim=1).detach().cpu().numpy()
        
        targets_np = targets.detach().cpu().numpy()
        conf_np = confidences.detach().cpu().numpy()
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(pred_probs, targets_np, conf_np)
        
        # Maximum Calibration Error (MCE)
        mce = self._compute_mce(pred_probs, targets_np, conf_np)
        
        # Average confidence and accuracy
        avg_conf = np.mean(conf_np)
        pred_labels = np.argmax(pred_probs, axis=1)
        
        # Handle multi-dimensional targets
        if targets_np.ndim > 1:
            targets_labels = np.argmax(targets_np, axis=1)
        else:
            targets_labels = targets_np
            
        avg_acc = np.mean(pred_labels == targets_labels)
        
        # Reliability diagram data
        reliability_data = self._compute_reliability_diagram(
            pred_probs, targets_np, conf_np
        )
        
        # Brier score
        brier_score = self._compute_brier_score(pred_probs, targets_np)
        
        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            average_confidence=avg_conf,
            average_accuracy=avg_acc,
            reliability_diagram=reliability_data,
            brier_score=brier_score
        )
    
    def _compute_ece(self, predictions: np.ndarray, targets: np.ndarray, 
                     confidences: np.ndarray) -> float:
        """Compute Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, self.config.calibration_bins + 1)
        ece = 0.0
        
        for i in range(len(bin_boundaries) - 1):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Handle multi-dimensional targets
                bin_targets = targets[in_bin]
                if bin_targets.ndim > 1:
                    bin_targets_labels = np.argmax(bin_targets, axis=1)
                else:
                    bin_targets_labels = bin_targets
                
                accuracy_in_bin = np.mean(
                    np.argmax(predictions[in_bin], axis=1) == bin_targets_labels
                )
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_mce(self, predictions: np.ndarray, targets: np.ndarray, 
                     confidences: np.ndarray) -> float:
        """Compute Maximum Calibration Error"""
        bin_boundaries = np.linspace(0, 1, self.config.calibration_bins + 1)
        mce = 0.0
        
        for i in range(len(bin_boundaries) - 1):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Handle multi-dimensional targets
                bin_targets = targets[in_bin]
                if bin_targets.ndim > 1:
                    bin_targets_labels = np.argmax(bin_targets, axis=1)
                else:
                    bin_targets_labels = bin_targets
                
                accuracy_in_bin = np.mean(
                    np.argmax(predictions[in_bin], axis=1) == bin_targets_labels
                )
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _compute_reliability_diagram(self, predictions: np.ndarray, 
                                   targets: np.ndarray, 
                                   confidences: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute data for reliability diagram"""
        bin_boundaries = np.linspace(0, 1, self.config.calibration_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        accuracies = []
        confidences_binned = []
        counts = []
        
        for i in range(len(bin_boundaries) - 1):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            count = np.sum(in_bin)
            
            if count > 0:
                # Handle multi-dimensional targets
                bin_targets = targets[in_bin]
                if bin_targets.ndim > 1:
                    bin_targets_labels = np.argmax(bin_targets, axis=1)
                else:
                    bin_targets_labels = bin_targets
                
                accuracy = np.mean(
                    np.argmax(predictions[in_bin], axis=1) == bin_targets_labels
                )
                avg_conf = np.mean(confidences[in_bin])
            else:
                accuracy = 0.0
                avg_conf = bin_centers[i]
            
            accuracies.append(accuracy)
            confidences_binned.append(avg_conf)
            counts.append(count)
        
        return {
            'bin_centers': np.array(bin_centers),
            'accuracies': np.array(accuracies),
            'confidences': np.array(confidences_binned),
            'counts': np.array(counts)
        }
    
    def _compute_brier_score(self, predictions: np.ndarray, 
                           targets: np.ndarray) -> float:
        """Compute Brier score for probabilistic predictions"""
        num_classes = predictions.shape[1]
        
        # Handle multi-dimensional targets
        if targets.ndim > 1:
            # Convert to class labels using argmax
            targets_labels = np.argmax(targets, axis=1)
        else:
            targets_labels = targets
        
        # Convert to integer indices for one-hot encoding
        targets_labels = targets_labels.astype(int)
        
        # Create one-hot encoded targets
        targets_one_hot = np.eye(num_classes)[targets_labels]
        
        # Brier score: average of squared differences between predicted probabilities and true labels
        return np.mean(np.sum((predictions - targets_one_hot) ** 2, axis=1))


class SafetyThresholds:
    """
    Manages safety thresholds for confidence-based decision making.
    
    This class defines and manages confidence thresholds for different
    types of safety-critical decisions in motor fault diagnosis.
    """
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.thresholds = self._initialize_default_thresholds()
        
    def _initialize_default_thresholds(self) -> Dict[str, float]:
        """Initialize default safety thresholds"""
        return {
            'critical_fault': 0.95,  # Very high confidence required
            'warning_fault': 0.80,   # High confidence required
            'normal_operation': 0.70, # Moderate confidence required
            'maintenance_alert': 0.85, # High confidence for maintenance decisions
            'emergency_shutdown': 0.98 # Extremely high confidence for shutdown
        }
    
    def set_threshold(self, decision_type: str, threshold: float) -> None:
        """Set threshold for specific decision type"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.thresholds[decision_type] = threshold
        logger.info(f"Set {decision_type} threshold to {threshold}")
    
    def get_threshold(self, decision_type: str) -> float:
        """Get threshold for specific decision type"""
        return self.thresholds.get(decision_type, self.config.safety_threshold)
    
    def is_decision_safe(self, decision_type: str, confidence: float) -> bool:
        """Check if confidence meets safety threshold for decision"""
        threshold = self.get_threshold(decision_type)
        return confidence >= threshold
    
    def get_safety_margin(self, decision_type: str, confidence: float) -> float:
        """Get safety margin (confidence - threshold)"""
        threshold = self.get_threshold(decision_type)
        return confidence - threshold
    
    def validate_thresholds(self) -> Dict[str, bool]:
        """Validate all thresholds are within acceptable ranges"""
        validation_results = {}
        
        for decision_type, threshold in self.thresholds.items():
            is_valid = (0.0 <= threshold <= 1.0 and 
                       threshold >= 0.5)  # Minimum 50% confidence
            validation_results[decision_type] = is_valid
            
            if not is_valid:
                logger.warning(f"Invalid threshold for {decision_type}: {threshold}")
        
        return validation_results
    
    def get_recommended_action(self, fault_type: str, 
                             confidence: float) -> Tuple[str, str]:
        """Get recommended action based on fault type and confidence"""
        if fault_type == 'critical' and confidence >= self.get_threshold('critical_fault'):
            return 'emergency_shutdown', 'Immediate shutdown required'
        elif fault_type == 'warning' and confidence >= self.get_threshold('warning_fault'):
            return 'maintenance_alert', 'Schedule maintenance soon'
        elif fault_type == 'normal' and confidence >= self.get_threshold('normal_operation'):
            return 'continue_operation', 'Normal operation'
        else:
            return 'uncertain', 'Insufficient confidence for decision'

class VariationalBayesianUQ(nn.Module):
    """
    Implements Variational Bayesian Uncertainty Quantification for neural operators.
    
    This class implements physics-informed prior distributions G~GP(μ_G, k_G) for 
    neural operator parameters, likelihood functions incorporating observed motor data,
    and variational distributions q(G) with factorized modes for ELBO optimization.
    """
    
    def __init__(self, config: UncertaintyConfig, input_dim: int, output_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize variational parameters
        self.variational_mean = nn.Parameter(
            torch.zeros(input_dim, output_dim) + config.prior_mean
        )
        self.variational_logvar = nn.Parameter(
            torch.ones(input_dim, output_dim) * np.log(config.prior_std ** 2)
        )
        
        # Physics-informed prior parameters
        self.prior_mean = config.prior_mean
        self.prior_std = config.prior_std
        
        logger.info(f"Initialized Variational Bayesian UQ with input_dim={input_dim}, output_dim={output_dim}")
    
    def sample_weights(self, num_samples: int = None) -> torch.Tensor:
        """
        Sample weights from variational distribution q(G).
        
        Args:
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Sampled weights [num_samples, input_dim, output_dim]
        """
        if num_samples is None:
            num_samples = self.config.num_mc_samples
        
        # Reparameterization trick: G = μ + σ * ε, where ε ~ N(0, I)
        epsilon = torch.randn(num_samples, self.input_dim, self.output_dim)
        std = torch.exp(0.5 * self.variational_logvar)
        
        samples = self.variational_mean.unsqueeze(0) + std.unsqueeze(0) * epsilon
        return samples
    
    def compute_kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between variational distribution q(G) and prior p(G).
        
        For Gaussian distributions:
        KL(q||p) = 0.5 * [tr(Σ_p^-1 Σ_q) + (μ_p - μ_q)^T Σ_p^-1 (μ_p - μ_q) - k + log(|Σ_p|/|Σ_q|)]
        
        Returns:
            KL divergence scalar
        """
        # Variational parameters
        var_mean = self.variational_mean
        var_logvar = self.variational_logvar
        var_var = torch.exp(var_logvar)
        
        # Prior parameters (assumed diagonal covariance)
        prior_var = self.prior_std ** 2
        
        # KL divergence for diagonal Gaussian
        kl_div = 0.5 * torch.sum(
            var_var / prior_var +  # tr(Σ_p^-1 Σ_q)
            (var_mean - self.prior_mean) ** 2 / prior_var +  # (μ_p - μ_q)^T Σ_p^-1 (μ_p - μ_q)
            torch.log(torch.tensor(prior_var)) - var_logvar - 1  # log(|Σ_p|/|Σ_q|) - k
        )
        
        return kl_div
    
    def compute_log_likelihood(self, predictions: torch.Tensor, 
                              targets: torch.Tensor, 
                              noise_var: float = 1e-3) -> torch.Tensor:
        """
        Compute log likelihood p(D|G) for observed motor data.
        
        Args:
            predictions: Model predictions [batch_size, output_dim]
            targets: Ground truth targets [batch_size, output_dim]
            noise_var: Observation noise variance
            
        Returns:
            Log likelihood scalar
        """
        # Gaussian likelihood: log p(y|x, G) = -0.5 * ||y - f(x, G)||^2 / σ^2 - 0.5 * log(2πσ^2)
        mse = torch.mean((predictions - targets) ** 2)
        log_likelihood = -0.5 * mse / noise_var - 0.5 * np.log(2 * np.pi * noise_var)
        
        return log_likelihood * predictions.shape[0]  # Scale by batch size
    
    def compute_elbo(self, predictions: torch.Tensor, 
                     targets: torch.Tensor, 
                     noise_var: float = 1e-3) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Evidence Lower BOund (ELBO) for variational inference.
        
        ELBO = E_q[log p(D|G)] - KL(q(G)||p(G))
        
        Args:
            predictions: Model predictions [batch_size, output_dim]
            targets: Ground truth targets [batch_size, output_dim]
            noise_var: Observation noise variance
            
        Returns:
            ELBO value and component breakdown
        """
        # Compute likelihood term
        log_likelihood = self.compute_log_likelihood(predictions, targets, noise_var)
        
        # Compute KL divergence term
        kl_divergence = self.compute_kl_divergence()
        
        # ELBO = likelihood - KL divergence
        elbo = log_likelihood - self.config.kl_weight * kl_divergence
        
        components = {
            'log_likelihood': log_likelihood,
            'kl_divergence': kl_divergence,
            'elbo': elbo
        }
        
        return elbo, components
    
    def predict_with_uncertainty(self, inputs: torch.Tensor, 
                                forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                num_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty quantification using Monte Carlo sampling.
        
        Args:
            inputs: Input data [batch_size, input_dim]
            forward_fn: Function that takes (inputs, weights) and returns predictions
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Mean predictions and uncertainty estimates
        """
        if num_samples is None:
            num_samples = self.config.num_mc_samples
        
        # Sample weights from variational distribution
        weight_samples = self.sample_weights(num_samples)
        
        predictions = []
        for i in range(num_samples):
            pred = forward_fn(inputs, weight_samples[i])
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
        
        # Compute mean and uncertainty
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty
    
    def get_variational_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current variational parameters"""
        return {
            'mean': self.variational_mean.clone(),
            'logvar': self.variational_logvar.clone(),
            'std': torch.exp(0.5 * self.variational_logvar)
        }
    
    def set_variational_parameters(self, mean: torch.Tensor, logvar: torch.Tensor) -> None:
        """Set variational parameters"""
        with torch.no_grad():
            self.variational_mean.copy_(mean)
            self.variational_logvar.copy_(logvar)


class PhysicsInformedPrior:
    """
    Implements physics-informed Gaussian process priors for neural operator parameters.
    
    This class defines prior distributions that incorporate physical constraints
    and domain knowledge about motor dynamics.
    """
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        
    def electromagnetic_prior_kernel(self, x1: torch.Tensor, x2: torch.Tensor, 
                                   length_scale: float = 1.0) -> torch.Tensor:
        """
        Kernel function for electromagnetic field parameters.
        
        Uses RBF kernel with physics-informed length scales based on
        electromagnetic field correlation lengths.
        """
        # Squared exponential kernel
        dist_sq = torch.sum((x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2, dim=-1)
        kernel = torch.exp(-0.5 * dist_sq / (length_scale ** 2))
        
        # Add small regularization to ensure positive definiteness
        n = kernel.shape[0]
        kernel = kernel + 1e-4 * torch.eye(n)
        return kernel
    
    def thermal_prior_kernel(self, x1: torch.Tensor, x2: torch.Tensor, 
                           length_scale: float = 2.0) -> torch.Tensor:
        """
        Kernel function for thermal dynamics parameters.
        
        Uses Matérn kernel appropriate for thermal diffusion processes.
        """
        # Simplified Matérn 3/2 kernel
        dist = torch.sqrt(torch.sum((x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2, dim=-1) + 1e-8)
        scaled_dist = np.sqrt(3) * dist / length_scale
        kernel = (1 + scaled_dist) * torch.exp(-scaled_dist)
        
        # Add small regularization to ensure positive definiteness
        n = kernel.shape[0]
        kernel = kernel + 1e-4 * torch.eye(n)
        return kernel
    
    def mechanical_prior_kernel(self, x1: torch.Tensor, x2: torch.Tensor, 
                              length_scale: float = 0.5) -> torch.Tensor:
        """
        Kernel function for mechanical vibration parameters.
        
        Uses periodic kernel to capture oscillatory nature of vibrations.
        """
        # Periodic kernel for vibration patterns
        dist = torch.sum((x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2, dim=-1)
        periodic_component = torch.sin(np.pi * torch.sqrt(dist) / length_scale) ** 2
        kernel = torch.exp(-2 * periodic_component / (length_scale ** 2))
        
        # Add small regularization to ensure positive definiteness
        n = kernel.shape[0]
        kernel = kernel + 1e-4 * torch.eye(n)
        return kernel
    
    def compute_physics_informed_prior(self, coordinates: torch.Tensor, 
                                     physics_type: str) -> torch.Tensor:
        """
        Compute physics-informed prior covariance matrix.
        
        Args:
            coordinates: Spatial/temporal coordinates [N, dim]
            physics_type: Type of physics ('electromagnetic', 'thermal', 'mechanical')
            
        Returns:
            Prior covariance matrix [N, N]
        """
        if physics_type == 'electromagnetic':
            return self.electromagnetic_prior_kernel(coordinates, coordinates)
        elif physics_type == 'thermal':
            return self.thermal_prior_kernel(coordinates, coordinates)
        elif physics_type == 'mechanical':
            return self.mechanical_prior_kernel(coordinates, coordinates)
        else:
            raise ValueError(f"Unknown physics type: {physics_type}")


class VariationalInferenceTrainer:
    """
    Trainer for variational Bayesian neural operators with physics constraints.
    """
    
    def __init__(self, vb_uq: VariationalBayesianUQ, 
                 physics_prior: PhysicsInformedPrior,
                 optimizer: torch.optim.Optimizer):
        self.vb_uq = vb_uq
        self.physics_prior = physics_prior
        self.optimizer = optimizer
        self.training_history = []
        
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor,
                   forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                   physics_loss_fn: Optional[Callable] = None) -> Dict[str, float]:
        """
        Perform one training step with variational inference.
        
        Args:
            inputs: Input data [batch_size, input_dim]
            targets: Target data [batch_size, output_dim]
            forward_fn: Forward function for the neural operator
            physics_loss_fn: Optional physics constraint loss function
            
        Returns:
            Dictionary of loss components
        """
        self.optimizer.zero_grad()
        
        # Sample weights and make predictions
        weight_samples = self.vb_uq.sample_weights(1)
        predictions = forward_fn(inputs, weight_samples[0])
        
        # Compute ELBO
        elbo, components = self.vb_uq.compute_elbo(predictions, targets)
        
        # Add physics constraints if provided
        physics_loss = torch.tensor(0.0)
        if physics_loss_fn is not None:
            physics_loss = physics_loss_fn(predictions, inputs)
        
        # Total loss (negative ELBO + physics loss)
        total_loss = -elbo + physics_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Record training metrics
        metrics = {
            'total_loss': total_loss.item(),
            'elbo': elbo.item(),
            'log_likelihood': components['log_likelihood'].item(),
            'kl_divergence': components['kl_divergence'].item(),
            'physics_loss': physics_loss.item()
        }
        
        self.training_history.append(metrics)
        return metrics
    
    def get_training_history(self) -> List[Dict[str, float]]:
        """Get training history"""
        return self.training_history.copy()