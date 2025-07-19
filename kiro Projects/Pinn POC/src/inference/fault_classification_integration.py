"""
Fault Classification Integration Module.

Integrates the fault classification head with the AGT-NO architecture
and provides a unified interface for uncertainty-aware fault diagnosis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging

from .fault_classifier import (
    FaultType, FaultPrediction, FaultClassificationSystem, 
    UncertaintyAwareClassifier, create_cwru_fault_classifier
)
from .classification_validator import ClassificationValidator, ValidationResults
from ..physics.agt_no_architecture import AGTNO, create_motor_agtno

logger = logging.getLogger(__name__)


class IntegratedFaultDiagnosisSystem(nn.Module):
    """
    Integrated fault diagnosis system combining AGT-NO with fault classification.
    
    This system provides end-to-end fault diagnosis with physics-informed
    neural operators and uncertainty-aware classification.
    """
    
    def __init__(self, 
                 input_dim: int = 12,
                 hidden_dim: int = 256,
                 output_dim: int = 64,
                 n_classes: int = 4,
                 device: str = 'cpu',
                 enable_physics_constraints: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_classes = n_classes
        self.device = device
        
        # Core AGT-NO architecture (simplified for integration)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Fault classification system
        self.fault_classifier = UncertaintyAwareClassifier(
            input_dim=output_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim // 2,
            dropout_rate=0.1,
            enable_mc_dropout=True
        )
        
        # Physics consistency estimator
        self.physics_estimator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Classification system for high-level interface
        self.classification_system = FaultClassificationSystem(
            input_dim=output_dim, 
            device=device
        )
        # Replace the classifier with our integrated one
        self.classification_system.classifier = self.fault_classifier
        
    def forward(self, x: torch.Tensor, 
                return_features: bool = False,
                return_uncertainty: bool = True) -> Union[
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                ]:
        """
        Forward pass through integrated system.
        
        Args:
            x: Input motor signals (batch_size, input_dim) or (batch_size, seq_len, input_dim)
            return_features: Whether to return extracted features
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Tuple of (classification_logits, physics_consistency, uncertainty, [features])
        """
        # Handle different input shapes
        if x.dim() == 3:
            # (batch_size, seq_len, input_dim) -> average over sequence
            x = torch.mean(x, dim=1)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Get classification with uncertainty
        if return_uncertainty:
            logits, uncertainty = self.fault_classifier(features, return_uncertainty=True)
        else:
            logits = self.fault_classifier(features, return_uncertainty=False)
            uncertainty = torch.zeros_like(logits)
        
        # Estimate physics consistency
        physics_consistency = self.physics_estimator(features).squeeze(-1)
        
        if return_features:
            return logits, physics_consistency, uncertainty, features
        else:
            return logits, physics_consistency, uncertainty
    
    def predict_faults(self, x: torch.Tensor) -> List[FaultPrediction]:
        """
        High-level fault prediction interface.
        
        Args:
            x: Input motor signals
            
        Returns:
            List of FaultPrediction objects with uncertainty quantification
        """
        self.eval()
        
        with torch.no_grad():
            logits, physics_consistency, uncertainty, features = self.forward(
                x, return_features=True, return_uncertainty=True
            )
            
            # Use the classification system for prediction
            predictions = self.classification_system.predict(features, physics_consistency)
            
        return predictions
    
    def evaluate_on_dataset(self, 
                           features: torch.Tensor,
                           true_labels: List[FaultType],
                           output_dir: Optional[str] = None) -> ValidationResults:
        """
        Evaluate the integrated system on a dataset.
        
        Args:
            features: Input features (batch_size, input_dim)
            true_labels: True fault labels
            output_dir: Optional directory to save results
            
        Returns:
            ValidationResults object
        """
        # Make predictions
        predictions = self.predict_faults(features)
        
        # Evaluate using validator
        validator = ClassificationValidator()
        results = validator.evaluate_predictions(predictions, true_labels)
        
        if output_dir:
            validator.save_results(results, output_dir)
            
        return results
    
    def calibrate_uncertainty(self, val_features: torch.Tensor, val_labels: torch.Tensor):
        """Calibrate uncertainty estimates using validation data."""
        # Process features through the feature extractor first
        self.eval()
        with torch.no_grad():
            processed_features = self.feature_extractor(val_features)
        
        self.classification_system.calibrate_temperature(processed_features, val_labels)
        logger.info("Uncertainty calibration completed")
    
    def save_model(self, filepath: str):
        """Save the complete integrated model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'n_classes': self.n_classes,
            'device': self.device,
            'classification_system': self.classification_system
        }, filepath)
        logger.info(f"Integrated model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a complete integrated model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Recreate model with same dimensions
        self.__init__(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim'],
            n_classes=checkpoint['n_classes'],
            device=checkpoint['device']
        )
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.classification_system = checkpoint['classification_system']
        logger.info(f"Integrated model loaded from {filepath}")


class CWRUFaultDiagnosisSystem:
    """
    CWRU-specific fault diagnosis system with complete evaluation pipeline.
    
    Provides a high-level interface for CWRU dataset fault diagnosis
    with comprehensive evaluation and validation capabilities.
    """
    
    def __init__(self, input_dim: int = 12, device: str = 'cpu'):
        self.device = device
        self.input_dim = input_dim
        
        # Create integrated system
        self.diagnosis_system = IntegratedFaultDiagnosisSystem(
            input_dim=input_dim,
            device=device
        ).to(device)
        
        # Validator for evaluation
        self.validator = ClassificationValidator()
        
        # CWRU fault type mapping
        self.fault_type_mapping = {
            'normal': FaultType.NORMAL,
            'inner_race': FaultType.INNER_RACE,
            'outer_race': FaultType.OUTER_RACE,
            'ball': FaultType.BALL
        }
        
    def train_on_cwru_data(self, 
                          train_features: torch.Tensor,
                          train_labels: torch.Tensor,
                          val_features: Optional[torch.Tensor] = None,
                          val_labels: Optional[torch.Tensor] = None,
                          n_epochs: int = 100,
                          learning_rate: float = 1e-3) -> Dict[str, List[float]]:
        """
        Train the system on CWRU dataset.
        
        Args:
            train_features: Training features (n_samples, input_dim)
            train_labels: Training labels (n_samples,) - class indices
            val_features: Validation features
            val_labels: Validation labels
            n_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history dictionary
        """
        self.diagnosis_system.train()
        
        optimizer = torch.optim.Adam(self.diagnosis_system.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(n_epochs):
            # Training
            optimizer.zero_grad()
            
            logits, physics_consistency, uncertainty = self.diagnosis_system(train_features)
            
            # Combined loss: classification + physics consistency + uncertainty regularization
            class_loss = criterion(logits, train_labels)
            physics_loss = torch.mean((1.0 - physics_consistency) ** 2)  # Encourage high consistency
            uncertainty_reg = torch.mean(uncertainty)  # Regularize uncertainty
            
            total_loss = class_loss + 0.1 * physics_loss + 0.01 * uncertainty_reg
            
            total_loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            with torch.no_grad():
                train_pred = torch.argmax(logits, dim=1)
                train_acc = (train_pred == train_labels).float().mean().item()
            
            history['train_loss'].append(total_loss.item())
            history['train_acc'].append(train_acc)
            
            # Validation
            if val_features is not None and val_labels is not None:
                self.diagnosis_system.eval()
                with torch.no_grad():
                    val_logits, val_physics, val_uncertainty = self.diagnosis_system(val_features)
                    val_loss = criterion(val_logits, val_labels)
                    val_pred = torch.argmax(val_logits, dim=1)
                    val_acc = (val_pred == val_labels).float().mean().item()
                
                history['val_loss'].append(val_loss.item())
                history['val_acc'].append(val_acc)
                self.diagnosis_system.train()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss={total_loss.item():.4f}, Acc={train_acc:.4f}")
        
        # Calibrate uncertainty on validation data
        if val_features is not None and val_labels is not None:
            self.diagnosis_system.calibrate_uncertainty(val_features, val_labels)
        
        logger.info("Training completed")
        return history
    
    def evaluate_system(self, 
                       test_features: torch.Tensor,
                       test_labels: List[FaultType],
                       output_dir: Optional[str] = None) -> ValidationResults:
        """
        Comprehensive evaluation of the fault diagnosis system.
        
        Args:
            test_features: Test features
            test_labels: True fault types
            output_dir: Directory to save results
            
        Returns:
            ValidationResults with comprehensive metrics
        """
        results = self.diagnosis_system.evaluate_on_dataset(
            test_features, test_labels, output_dir
        )
        
        # Log key metrics
        logger.info(f"Overall Accuracy: {results.overall_metrics.accuracy:.4f}")
        logger.info(f"Reliability Ratio: {results.reliability_analysis.get('reliability_ratio', 0):.4f}")
        
        return results
    
    def predict_single_sample(self, sample: torch.Tensor) -> FaultPrediction:
        """
        Predict fault for a single sample.
        
        Args:
            sample: Single sample (input_dim,) or (1, input_dim)
            
        Returns:
            FaultPrediction object
        """
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)
        
        predictions = self.diagnosis_system.predict_faults(sample)
        return predictions[0]
    
    def get_system_performance_summary(self) -> Dict[str, float]:
        """Get a summary of system performance metrics."""
        # This would typically be computed from validation results
        # For now, return placeholder metrics
        return {
            'accuracy': 0.0,  # To be updated after training/evaluation
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'reliability_ratio': 0.0
        }


def create_cwru_diagnosis_system(input_dim: int = 12, device: str = 'cpu') -> CWRUFaultDiagnosisSystem:
    """
    Create a complete CWRU fault diagnosis system.
    
    Args:
        input_dim: Input feature dimension
        device: Device to run on
        
    Returns:
        Configured CWRUFaultDiagnosisSystem
    """
    return CWRUFaultDiagnosisSystem(input_dim=input_dim, device=device)


# Convenience functions for common use cases
def diagnose_motor_fault(signal_features: torch.Tensor, 
                        model_path: Optional[str] = None,
                        device: str = 'cpu') -> List[FaultPrediction]:
    """
    Convenience function for motor fault diagnosis.
    
    Args:
        signal_features: Motor signal features
        model_path: Path to trained model (optional)
        device: Device to run on
        
    Returns:
        List of fault predictions
    """
    system = create_cwru_diagnosis_system(
        input_dim=signal_features.shape[-1], 
        device=device
    )
    
    if model_path:
        system.diagnosis_system.load_model(model_path)
    
    return system.diagnosis_system.predict_faults(signal_features)


def evaluate_fault_diagnosis_system(features: torch.Tensor,
                                  true_labels: List[FaultType],
                                  model_path: Optional[str] = None,
                                  output_dir: Optional[str] = None,
                                  device: str = 'cpu') -> ValidationResults:
    """
    Convenience function for system evaluation.
    
    Args:
        features: Test features
        true_labels: True fault labels
        model_path: Path to trained model
        output_dir: Directory to save results
        device: Device to run on
        
    Returns:
        ValidationResults object
    """
    system = create_cwru_diagnosis_system(
        input_dim=features.shape[-1],
        device=device
    )
    
    if model_path:
        system.diagnosis_system.load_model(model_path)
    
    return system.evaluate_system(features, true_labels, output_dir)