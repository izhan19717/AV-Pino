"""
Fault Classification Implementation with Uncertainty-Aware Predictions.

This module implements the fault classification head with uncertainty quantification
for the AV-PINO motor fault diagnosis system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """CWRU dataset fault types."""
    NORMAL = "normal"
    INNER_RACE = "inner_race"
    OUTER_RACE = "outer_race"
    BALL = "ball"


@dataclass
class FaultPrediction:
    """Fault prediction with uncertainty information."""
    fault_type: FaultType
    confidence: float
    uncertainty: float
    probabilities: Dict[FaultType, float]
    physics_consistency: float
    timestamp: Optional[float] = None


@dataclass
class ClassificationMetrics:
    """Classification performance metrics."""
    accuracy: float
    precision: Dict[FaultType, float]
    recall: Dict[FaultType, float]
    f1_score: Dict[FaultType, float]
    confusion_matrix: np.ndarray
    per_class_accuracy: Dict[FaultType, float]


class FaultTypeMapper:
    """Maps between fault types and class indices."""
    
    def __init__(self):
        self.fault_types = list(FaultType)
        self.type_to_idx = {fault_type: idx for idx, fault_type in enumerate(self.fault_types)}
        self.idx_to_type = {idx: fault_type for fault_type, idx in self.type_to_idx.items()}
        self.n_classes = len(self.fault_types)
    
    def type_to_index(self, fault_type: Union[FaultType, str]) -> int:
        """Convert fault type to class index."""
        if isinstance(fault_type, str):
            fault_type = FaultType(fault_type)
        return self.type_to_idx[fault_type]
    
    def index_to_type(self, index: int) -> FaultType:
        """Convert class index to fault type."""
        return self.idx_to_type[index]
    
    def get_all_types(self) -> List[FaultType]:
        """Get all fault types."""
        return self.fault_types.copy()


class UncertaintyAwareClassifier(nn.Module):
    """
    Fault classification head with uncertainty quantification.
    
    Implements both aleatoric (data) and epistemic (model) uncertainty
    estimation for safety-critical fault diagnosis decisions.
    """
    
    def __init__(self, input_dim: int, n_classes: int = 4, 
                 hidden_dim: int = 256, dropout_rate: float = 0.1,
                 enable_mc_dropout: bool = True, n_mc_samples: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.enable_mc_dropout = enable_mc_dropout
        self.n_mc_samples = n_mc_samples
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head (mean prediction)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        
        # Aleatoric uncertainty head (data uncertainty)
        self.uncertainty_head = nn.Linear(hidden_dim, n_classes)
        
        # Epistemic uncertainty via MC Dropout
        self.mc_dropout = nn.Dropout(dropout_rate)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, return_uncertainty: bool = True
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional uncertainty estimation.
        
        Args:
            x: Input features (batch_size, input_dim)
            return_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            If return_uncertainty=False: logits (batch_size, n_classes)
            If return_uncertainty=True: (logits, uncertainty) tuple
        """
        # Extract features
        features = self.feature_extractor(x)
        
        if not return_uncertainty:
            # Simple forward pass
            logits = self.classifier(features)
            return logits / self.temperature
        
        # Compute mean prediction
        logits = self.classifier(features)
        
        # Compute aleatoric uncertainty (data uncertainty)
        log_var = self.uncertainty_head(features)
        aleatoric_uncertainty = torch.exp(log_var)
        
        # Compute epistemic uncertainty via MC Dropout
        if self.enable_mc_dropout and self.training:
            epistemic_uncertainty = self._compute_epistemic_uncertainty(features)
        else:
            epistemic_uncertainty = torch.zeros_like(aleatoric_uncertainty)
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        # Temperature scaling for calibration
        calibrated_logits = logits / self.temperature
        
        return calibrated_logits, total_uncertainty
    
    def _compute_epistemic_uncertainty(self, features: torch.Tensor) -> torch.Tensor:
        """Compute epistemic uncertainty using MC Dropout."""
        predictions = []
        
        # Multiple forward passes with dropout
        for _ in range(self.n_mc_samples):
            dropped_features = self.mc_dropout(features)
            pred = self.classifier(dropped_features)
            predictions.append(F.softmax(pred, dim=-1))
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch_size, n_classes)
        
        # Compute variance across samples (epistemic uncertainty)
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        return epistemic_uncertainty
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with full uncertainty quantification.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Tuple of (predictions, probabilities, uncertainties)
        """
        self.eval()
        with torch.no_grad():
            logits, uncertainty = self.forward(x, return_uncertainty=True)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            
        return predictions, probabilities, uncertainty


class FaultClassificationSystem:
    """
    Complete fault classification system with CWRU dataset integration.
    
    Handles fault type mapping, uncertainty-aware predictions, and
    comprehensive evaluation metrics.
    """
    
    def __init__(self, input_dim: int, device: str = 'cpu'):
        self.device = device
        self.fault_mapper = FaultTypeMapper()
        
        # Initialize classifier
        self.classifier = UncertaintyAwareClassifier(
            input_dim=input_dim,
            n_classes=self.fault_mapper.n_classes
        ).to(device)
        
        # Confidence thresholds for safety-critical decisions
        self.confidence_thresholds = {
            FaultType.NORMAL: 0.9,
            FaultType.INNER_RACE: 0.85,
            FaultType.OUTER_RACE: 0.85,
            FaultType.BALL: 0.85
        }
        
        # Physics consistency weight
        self.physics_weight = 0.1
        
    def predict(self, features: torch.Tensor, 
                physics_consistency: Optional[torch.Tensor] = None) -> List[FaultPrediction]:
        """
        Make fault predictions with uncertainty quantification.
        
        Args:
            features: Input features (batch_size, input_dim)
            physics_consistency: Physics consistency scores (batch_size,)
            
        Returns:
            List of FaultPrediction objects
        """
        self.classifier.eval()
        
        with torch.no_grad():
            predictions, probabilities, uncertainties = self.classifier.predict_with_uncertainty(features)
            
        results = []
        batch_size = features.shape[0]
        
        for i in range(batch_size):
            pred_idx = predictions[i].item()
            fault_type = self.fault_mapper.index_to_type(pred_idx)
            
            # Extract probabilities for this sample
            sample_probs = probabilities[i]
            prob_dict = {
                self.fault_mapper.index_to_type(j): sample_probs[j].item()
                for j in range(self.fault_mapper.n_classes)
            }
            
            # Confidence is the maximum probability
            confidence = sample_probs[pred_idx].item()
            
            # Uncertainty is the total uncertainty for the predicted class
            uncertainty = uncertainties[i, pred_idx].item()
            
            # Physics consistency (if provided)
            phys_consistency = physics_consistency[i].item() if physics_consistency is not None else 1.0
            
            result = FaultPrediction(
                fault_type=fault_type,
                confidence=confidence,
                uncertainty=uncertainty,
                probabilities=prob_dict,
                physics_consistency=phys_consistency
            )
            
            results.append(result)
        
        return results
    
    def is_prediction_reliable(self, prediction: FaultPrediction) -> bool:
        """
        Determine if a prediction is reliable for safety-critical decisions.
        
        Args:
            prediction: FaultPrediction object
            
        Returns:
            True if prediction is reliable, False otherwise
        """
        threshold = self.confidence_thresholds.get(prediction.fault_type, 0.8)
        
        # Check confidence threshold
        confidence_ok = prediction.confidence >= threshold
        
        # Check uncertainty threshold (lower is better)
        uncertainty_ok = prediction.uncertainty <= (1.0 - threshold)
        
        # Check physics consistency
        physics_ok = prediction.physics_consistency >= 0.7
        
        return confidence_ok and uncertainty_ok and physics_ok
    
    def calibrate_temperature(self, val_features: torch.Tensor, val_labels: torch.Tensor):
        """
        Calibrate temperature parameter for better confidence estimates.
        
        Args:
            val_features: Validation features
            val_labels: Validation labels
        """
        self.classifier.eval()
        
        # Get uncalibrated logits
        with torch.no_grad():
            logits, _ = self.classifier(val_features, return_uncertainty=True)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.classifier.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            loss = F.cross_entropy(logits / self.classifier.temperature, val_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        logger.info(f"Temperature calibrated to: {self.classifier.temperature.item():.3f}")
    
    def get_fault_type_mapping(self) -> Dict[str, int]:
        """Get mapping from fault type strings to class indices."""
        return {fault_type.value: idx for fault_type, idx in self.fault_mapper.type_to_idx.items()}
    
    def save_model(self, filepath: str):
        """Save the trained classifier."""
        torch.save({
            'model_state_dict': self.classifier.state_dict(),
            'fault_mapper': self.fault_mapper,
            'confidence_thresholds': self.confidence_thresholds
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained classifier."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.fault_mapper = checkpoint['fault_mapper']
        self.confidence_thresholds = checkpoint['confidence_thresholds']
        logger.info(f"Model loaded from {filepath}")


def create_cwru_fault_classifier(input_dim: int, device: str = 'cpu') -> FaultClassificationSystem:
    """
    Create a fault classification system configured for CWRU dataset.
    
    Args:
        input_dim: Input feature dimension
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Configured FaultClassificationSystem
    """
    return FaultClassificationSystem(input_dim=input_dim, device=device)