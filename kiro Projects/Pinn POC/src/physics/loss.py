"""
Physics-Informed Loss System for AV-PINO Motor Fault Diagnosis.

Implements comprehensive loss functions combining data-driven losses with 
physics-based constraints for neural operator training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from .constraints import PDEConstraint, PhysicsConstraintLayer


class BaseLoss(ABC):
    """Base class for all loss components."""
    
    def __init__(self, weight: float = 1.0, name: str = "base_loss"):
        self.weight = weight
        self.name = name
    
    @abstractmethod
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        """Compute the loss value."""
        pass
    
    def get_weight(self) -> float:
        """Get the loss weight."""
        return self.weight
    
    def set_weight(self, weight: float):
        """Set the loss weight."""
        self.weight = weight


class DataLoss(BaseLoss):
    """Standard classification loss computation for fault diagnosis."""
    
    def __init__(self, loss_type: str = "cross_entropy", weight: float = 1.0, 
                 label_smoothing: float = 0.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__(weight, "data_loss")
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        
        # Initialize loss function based on type
        if loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights, 
                label_smoothing=label_smoothing,
                reduction='mean'
            )
        elif loss_type == "focal":
            self.loss_fn = self._focal_loss
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor, 
                    **kwargs) -> torch.Tensor:
        """Compute data-driven loss.
        
        Args:
            prediction: Model predictions [batch_size, num_classes] or [batch_size, ...]
            target: Ground truth labels [batch_size] or [batch_size, ...]
            
        Returns:
            Computed loss value
        """
        if self.loss_type == "focal":
            return self.loss_fn(prediction, target)
        else:
            return self.loss_fn(prediction, target)
    
    def _focal_loss(self, prediction: torch.Tensor, target: torch.Tensor, 
                   alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(prediction, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()


class PhysicsLoss(BaseLoss):
    """PDE residual-based loss terms for physics consistency."""
    
    def __init__(self, constraints: List[PDEConstraint], weight: float = 1.0,
                 adaptive_weighting: bool = True):
        super().__init__(weight, "physics_loss")
        self.constraints = constraints
        self.adaptive_weighting = adaptive_weighting
        self.constraint_layer = PhysicsConstraintLayer(constraints)
        
        # Initialize constraint weights
        self.constraint_weights = {c.name: c.get_constraint_weight() for c in constraints}
        
        # For adaptive weighting
        self.loss_history = {c.name: [] for c in constraints}
        self.adaptation_rate = 0.1
    
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                    input_data: torch.Tensor, coords: torch.Tensor, 
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute physics-based loss from PDE residuals.
        
        Args:
            prediction: Neural operator prediction
            target: Ground truth (not used for physics loss but kept for interface consistency)
            input_data: Input data to the operator
            coords: Spatial/temporal coordinates
            
        Returns:
            Tuple of (total_physics_loss, individual_residuals)
        """
        # Compute physics residuals
        _, residuals = self.constraint_layer(prediction, input_data, coords)
        
        # Update loss history for adaptive weighting
        if self.adaptive_weighting:
            for constraint_name, residual_value in residuals.items():
                if constraint_name != 'total_physics_loss' and constraint_name in self.loss_history:
                    self.loss_history[constraint_name].append(residual_value.item())
                    # Keep only recent history
                    if len(self.loss_history[constraint_name]) > 100:
                        self.loss_history[constraint_name] = self.loss_history[constraint_name][-100:]
            
            # Update constraint weights adaptively
            self._update_adaptive_weights()
        
        total_loss = residuals['total_physics_loss']
        return total_loss, residuals
    
    def _update_adaptive_weights(self):
        """Update constraint weights based on loss history."""
        if not self.adaptive_weighting:
            return
        
        # Compute relative loss magnitudes
        avg_losses = {}
        for name, history in self.loss_history.items():
            if len(history) > 10:  # Need sufficient history
                avg_losses[name] = np.mean(history[-10:])
        
        if len(avg_losses) > 1:
            # Normalize weights inversely proportional to loss magnitude
            total_avg = sum(avg_losses.values())
            if total_avg > 0:
                new_weights = {}
                for name, avg_loss in avg_losses.items():
                    # Higher loss gets lower weight to balance training
                    new_weights[name] = (total_avg / avg_loss) / len(avg_losses)
                
                # Update constraint weights with momentum
                for constraint in self.constraints:
                    if constraint.name in new_weights:
                        old_weight = self.constraint_weights[constraint.name]
                        new_weight = new_weights[constraint.name]
                        updated_weight = (1 - self.adaptation_rate) * old_weight + self.adaptation_rate * new_weight
                        self.constraint_weights[constraint.name] = updated_weight
                        constraint.set_constraint_weight(updated_weight)
    
    def get_constraint_weights(self) -> Dict[str, float]:
        """Get current constraint weights."""
        return self.constraint_weights.copy()
    
    def set_constraint_weights(self, weights: Dict[str, float]):
        """Set constraint weights manually."""
        for constraint in self.constraints:
            if constraint.name in weights:
                self.constraint_weights[constraint.name] = weights[constraint.name]
                constraint.set_constraint_weight(weights[constraint.name])
        
        # Update constraint layer weights
        self.constraint_layer.update_constraint_weights(weights)


class ConsistencyLoss(BaseLoss):
    """Multi-physics coupling constraints loss for electromagnetic-thermal-mechanical consistency."""
    
    def __init__(self, weight: float = 1.0, coupling_strength: float = 1.0):
        super().__init__(weight, "consistency_loss")
        self.coupling_strength = coupling_strength
        
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                    electromagnetic_features: torch.Tensor,
                    thermal_features: torch.Tensor,
                    mechanical_features: torch.Tensor,
                    **kwargs) -> torch.Tensor:
        """Compute multi-physics coupling consistency loss.
        
        Args:
            prediction: Neural operator prediction
            target: Ground truth (for interface consistency)
            electromagnetic_features: EM field features
            thermal_features: Temperature/heat features
            mechanical_features: Vibration/structural features
            
        Returns:
            Consistency loss value
        """
        # Energy conservation constraint: electromagnetic energy should couple with thermal
        em_energy = torch.sum(electromagnetic_features ** 2, dim=-1)
        thermal_energy = torch.sum(thermal_features ** 2, dim=-1)
        
        # Electromagnetic-thermal coupling loss
        em_thermal_coupling = F.mse_loss(em_energy, thermal_energy)
        
        # Thermal-mechanical coupling: thermal expansion affects mechanical vibrations
        thermal_mechanical_coupling = self._compute_thermal_mechanical_coupling(
            thermal_features, mechanical_features
        )
        
        # Electromagnetic-mechanical coupling: magnetic forces affect vibrations
        em_mechanical_coupling = self._compute_em_mechanical_coupling(
            electromagnetic_features, mechanical_features
        )
        
        # Total consistency loss
        total_consistency = (
            em_thermal_coupling + 
            thermal_mechanical_coupling + 
            em_mechanical_coupling
        ) * self.coupling_strength
        
        return total_consistency
    
    def _compute_thermal_mechanical_coupling(self, thermal_features: torch.Tensor,
                                           mechanical_features: torch.Tensor) -> torch.Tensor:
        """Compute thermal-mechanical coupling constraint."""
        # Thermal expansion should correlate with mechanical stress
        thermal_gradient = torch.gradient(thermal_features, dim=-1)[0]
        mechanical_stress = torch.gradient(mechanical_features, dim=-1)[0]
        
        # Cross-correlation as coupling measure
        coupling_loss = 1.0 - F.cosine_similarity(
            thermal_gradient.flatten(1), 
            mechanical_stress.flatten(1), 
            dim=1
        ).mean()
        
        return coupling_loss
    
    def _compute_em_mechanical_coupling(self, em_features: torch.Tensor,
                                      mechanical_features: torch.Tensor) -> torch.Tensor:
        """Compute electromagnetic-mechanical coupling constraint."""
        # Magnetic forces should correlate with mechanical vibrations
        em_force = torch.gradient(em_features, dim=-1)[0]
        mechanical_acceleration = torch.gradient(mechanical_features, dim=-1)[0]
        
        # Force-acceleration relationship (F = ma)
        coupling_loss = F.mse_loss(
            F.normalize(em_force.flatten(1), dim=1),
            F.normalize(mechanical_acceleration.flatten(1), dim=1)
        )
        
        return coupling_loss


class VariationalLoss(BaseLoss):
    """Variational loss for uncertainty quantification training using ELBO."""
    
    def __init__(self, weight: float = 1.0, kl_weight: float = 1.0, 
                 prior_std: float = 1.0):
        super().__init__(weight, "variational_loss")
        self.kl_weight = kl_weight
        self.prior_std = prior_std
        
    def compute_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                    mu: torch.Tensor, log_var: torch.Tensor,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute variational loss (negative ELBO).
        
        Args:
            prediction: Model predictions (reconstruction)
            target: Ground truth
            mu: Mean of variational distribution
            log_var: Log variance of variational distribution
            
        Returns:
            Tuple of (total_variational_loss, loss_components)
        """
        # Reconstruction loss (likelihood term)
        # Handle different target types (classification vs regression)
        if target.dim() == 1 and prediction.dim() == 2:
            # Classification case: use cross-entropy
            reconstruction_loss = F.cross_entropy(prediction, target, reduction='mean')
        else:
            # Regression case: use MSE
            reconstruction_loss = F.mse_loss(prediction, target, reduction='mean')
        
        # KL divergence loss (regularization term)
        # KL(q(z|x) || p(z)) where p(z) = N(0, prior_std^2)
        kl_loss = self._compute_kl_divergence(mu, log_var)
        
        # Total variational loss (negative ELBO)
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        loss_components = {
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'total_variational_loss': total_loss
        }
        
        return total_loss, loss_components
    
    def _compute_kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between variational posterior and prior."""
        # KL(q(z|x) || N(0, σ²)) = 0.5 * Σ(σ²_q/σ²_p + μ²/σ²_p - 1 - log(σ²_q/σ²_p))
        var = torch.exp(log_var)
        prior_var = self.prior_std ** 2
        
        kl_div = 0.5 * torch.sum(
            var / prior_var + 
            mu ** 2 / prior_var - 
            1 - 
            log_var + 
            2 * np.log(self.prior_std),
            dim=-1
        )
        
        return kl_div.mean()
    
    def update_kl_weight(self, epoch: int, total_epochs: int, 
                        annealing_type: str = "linear"):
        """Update KL weight for annealing schedule."""
        if annealing_type == "linear":
            self.kl_weight = min(1.0, epoch / max(1, total_epochs * 0.5))
        elif annealing_type == "cosine":
            self.kl_weight = 0.5 * (1 + np.cos(np.pi * epoch / max(1, total_epochs)))
        elif annealing_type == "exponential":
            self.kl_weight = 1.0 - np.exp(-epoch / max(1, total_epochs * 0.3))


class AdaptiveLossWeighting:
    """Adaptive loss weighting mechanism for balancing data and physics terms."""
    
    def __init__(self, initial_weights: Dict[str, float], 
                 adaptation_method: str = "uncertainty_weighting",
                 update_frequency: int = 10):
        self.weights = initial_weights.copy()
        self.adaptation_method = adaptation_method
        self.update_frequency = update_frequency
        self.loss_history = {name: [] for name in initial_weights.keys()}
        self.step_count = 0
        
        # Method-specific parameters
        self.temperature = 1.0  # For uncertainty weighting
        self.momentum = 0.9     # For momentum-based adaptation
        self.weight_momentum = {name: 0.0 for name in initial_weights.keys()}
        
    def update_weights(self, loss_values: Dict[str, torch.Tensor]):
        """Update loss weights based on current loss values."""
        self.step_count += 1
        
        # Store loss history
        for name, loss_value in loss_values.items():
            if name in self.loss_history:
                self.loss_history[name].append(loss_value.item())
                # Keep only recent history
                if len(self.loss_history[name]) > 100:
                    self.loss_history[name] = self.loss_history[name][-100:]
        
        # Update weights at specified frequency
        if self.step_count % self.update_frequency == 0:
            if self.adaptation_method == "uncertainty_weighting":
                self._uncertainty_weighting()
            elif self.adaptation_method == "gradient_norm_balancing":
                self._gradient_norm_balancing(loss_values)
            elif self.adaptation_method == "homoscedastic_uncertainty":
                self._homoscedastic_uncertainty()
    
    def _uncertainty_weighting(self):
        """Uncertainty-based adaptive weighting."""
        if len(self.loss_history[list(self.loss_history.keys())[0]]) < 10:
            return
        
        # Compute uncertainty (variance) for each loss
        uncertainties = {}
        for name, history in self.loss_history.items():
            if len(history) >= 10:
                recent_losses = history[-10:]
                uncertainties[name] = max(np.var(recent_losses), 1e-8)  # Avoid zero variance
        
        if len(uncertainties) > 1:
            # Weight inversely proportional to uncertainty
            total_uncertainty = sum(uncertainties.values())
            if total_uncertainty > 0:
                for name in uncertainties:
                    # Avoid division by zero and overflow
                    uncertainty_val = max(uncertainties[name], 1e-8)
                    new_weight = (total_uncertainty / uncertainty_val) / len(uncertainties)
                    
                    # Clip to avoid overflow in exp
                    new_weight = np.clip(new_weight, 0.1, 10.0)
                    
                    # Apply temperature scaling with clipping
                    scaled_weight = np.clip(new_weight / self.temperature, -10, 10)
                    new_weight = np.exp(scaled_weight)
                    
                    # Momentum update
                    self.weight_momentum[name] = (
                        self.momentum * self.weight_momentum[name] + 
                        (1 - self.momentum) * new_weight
                    )
                    self.weights[name] = self.weight_momentum[name]
                
                # Normalize weights
                total_weight = sum(self.weights.values())
                if total_weight > 1e-8:
                    for name in self.weights:
                        self.weights[name] /= total_weight
                else:
                    # Reset to equal weights if normalization fails
                    equal_weight = 1.0 / len(self.weights)
                    for name in self.weights:
                        self.weights[name] = equal_weight
    
    def _gradient_norm_balancing(self, loss_values: Dict[str, torch.Tensor]):
        """Balance weights based on gradient norms."""
        # This would require gradients, simplified version using loss magnitudes
        if len(loss_values) > 1:
            # Normalize weights inversely proportional to loss magnitude
            total_loss = sum(loss.item() for loss in loss_values.values())
            if total_loss > 0:
                for name, loss_value in loss_values.items():
                    if name in self.weights:
                        new_weight = (total_loss / loss_value.item()) / len(loss_values)
                        # Momentum update
                        self.weight_momentum[name] = (
                            self.momentum * self.weight_momentum[name] + 
                            (1 - self.momentum) * new_weight
                        )
                        self.weights[name] = self.weight_momentum[name]
    
    def _homoscedastic_uncertainty(self):
        """Homoscedastic uncertainty-based weighting."""
        # Simplified version - in practice would use learned uncertainty parameters
        if len(self.loss_history[list(self.loss_history.keys())[0]]) < 10:
            return
        
        # Estimate homoscedastic uncertainty from loss variance
        for name, history in self.loss_history.items():
            if len(history) >= 10:
                recent_losses = history[-10:]
                uncertainty = np.var(recent_losses)
                # Weight inversely proportional to uncertainty
                new_weight = 1.0 / (2 * uncertainty + 1e-8)
                
                # Momentum update
                self.weight_momentum[name] = (
                    self.momentum * self.weight_momentum[name] + 
                    (1 - self.momentum) * new_weight
                )
                self.weights[name] = self.weight_momentum[name]
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for name in self.weights:
                self.weights[name] /= total_weight
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.weights.copy()
    
    def set_weights(self, weights: Dict[str, float]):
        """Set loss weights manually."""
        self.weights.update(weights)
    
    def get_weight_history(self) -> Dict[str, List[float]]:
        """Get history of weight changes."""
        return {name: [self.weights[name]] for name in self.weights}


class PhysicsInformedLoss(nn.Module):
    """Complete physics-informed loss system combining all loss components."""
    
    def __init__(self, 
                 data_loss_config: Dict = None,
                 physics_loss_config: Dict = None,
                 consistency_loss_config: Dict = None,
                 variational_loss_config: Dict = None,
                 adaptive_weighting_config: Dict = None):
        super().__init__()
        
        # Initialize loss components with default configs if not provided
        data_config = data_loss_config or {'loss_type': 'cross_entropy', 'weight': 1.0}
        physics_config = physics_loss_config or {'constraints': [], 'weight': 1.0}
        consistency_config = consistency_loss_config or {'weight': 0.5, 'coupling_strength': 1.0}
        variational_config = variational_loss_config or {'weight': 0.1, 'kl_weight': 1.0}
        adaptive_config = adaptive_weighting_config or {
            'initial_weights': {'data_loss': 1.0, 'physics_loss': 1.0, 'consistency_loss': 0.5, 'variational_loss': 0.1},
            'adaptation_method': 'uncertainty_weighting'
        }
        
        # Initialize individual loss components
        self.data_loss = DataLoss(**data_config)
        self.physics_loss = PhysicsLoss(**physics_config)
        self.consistency_loss = ConsistencyLoss(**consistency_config)
        self.variational_loss = VariationalLoss(**variational_config)
        
        # Initialize adaptive weighting
        self.adaptive_weighting = AdaptiveLossWeighting(**adaptive_config)
        
        # Loss scheduling parameters
        self.epoch = 0
        self.total_epochs = 1000
        self.warmup_epochs = 100
        self.physics_ramp_epochs = 200
        
        # Loss history for monitoring
        self.loss_history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'consistency_loss': [],
            'variational_loss': []
        }
        
        # Enable/disable components
        self.use_physics_loss = len(physics_config.get('constraints', [])) > 0
        self.use_consistency_loss = True
        self.use_variational_loss = True
        self.use_adaptive_weighting = True
    
    def forward(self, 
                prediction: torch.Tensor,
                target: torch.Tensor,
                input_data: torch.Tensor = None,
                coords: torch.Tensor = None,
                electromagnetic_features: torch.Tensor = None,
                thermal_features: torch.Tensor = None,
                mechanical_features: torch.Tensor = None,
                mu: torch.Tensor = None,
                log_var: torch.Tensor = None,
                **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute complete physics-informed loss.
        
        Args:
            prediction: Model predictions
            target: Ground truth targets
            input_data: Input data for physics constraints
            coords: Spatial/temporal coordinates
            electromagnetic_features: EM field features for consistency
            thermal_features: Thermal features for consistency
            mechanical_features: Mechanical features for consistency
            mu: Mean for variational loss
            log_var: Log variance for variational loss
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        loss_components = {}
        
        # 1. Data loss (always computed)
        data_loss_value = self.data_loss.compute_loss(prediction, target)
        loss_components['data_loss'] = data_loss_value
        
        # 2. Physics loss (if constraints are provided)
        physics_loss_value = torch.tensor(0.0, device=prediction.device)
        if self.use_physics_loss and input_data is not None and coords is not None:
            physics_loss_value, physics_residuals = self.physics_loss.compute_loss(
                prediction, target, input_data, coords
            )
            loss_components['physics_loss'] = physics_loss_value
            loss_components.update(physics_residuals)
        else:
            loss_components['physics_loss'] = physics_loss_value
        
        # 3. Consistency loss (if multi-physics features are provided)
        consistency_loss_value = torch.tensor(0.0, device=prediction.device)
        if (self.use_consistency_loss and 
            electromagnetic_features is not None and 
            thermal_features is not None and 
            mechanical_features is not None):
            consistency_loss_value = self.consistency_loss.compute_loss(
                prediction, target, electromagnetic_features, 
                thermal_features, mechanical_features
            )
            loss_components['consistency_loss'] = consistency_loss_value
        else:
            loss_components['consistency_loss'] = consistency_loss_value
        
        # 4. Variational loss (if variational parameters are provided)
        variational_loss_value = torch.tensor(0.0, device=prediction.device)
        if self.use_variational_loss and mu is not None and log_var is not None:
            variational_loss_value, var_components = self.variational_loss.compute_loss(
                prediction, target, mu, log_var
            )
            loss_components['variational_loss'] = variational_loss_value
            loss_components.update(var_components)
        else:
            loss_components['variational_loss'] = variational_loss_value
        
        # 5. Apply loss scheduling
        scheduled_weights = self._compute_scheduled_weights()
        
        # 6. Apply adaptive weighting if enabled
        if self.use_adaptive_weighting:
            # Update adaptive weights based on current loss values
            adaptive_loss_values = {
                'data_loss': data_loss_value,
                'physics_loss': physics_loss_value,
                'consistency_loss': consistency_loss_value,
                'variational_loss': variational_loss_value
            }
            self.adaptive_weighting.update_weights(adaptive_loss_values)
            adaptive_weights = self.adaptive_weighting.get_weights()
            
            # Combine scheduled and adaptive weights
            final_weights = {}
            for key in scheduled_weights:
                if key in adaptive_weights:
                    final_weights[key] = scheduled_weights[key] * adaptive_weights[key]
                else:
                    final_weights[key] = scheduled_weights[key]
        else:
            final_weights = scheduled_weights
        
        # 7. Compute weighted total loss
        total_loss = (
            final_weights.get('data_loss', 1.0) * data_loss_value +
            final_weights.get('physics_loss', 1.0) * physics_loss_value +
            final_weights.get('consistency_loss', 0.5) * consistency_loss_value +
            final_weights.get('variational_loss', 0.1) * variational_loss_value
        )
        
        loss_components['total_loss'] = total_loss
        loss_components['weights'] = final_weights
        
        # Update loss history
        self._update_loss_history(loss_components)
        
        return total_loss, loss_components
    
    def _compute_scheduled_weights(self) -> Dict[str, float]:
        """Compute loss weights based on training schedule."""
        weights = {}
        
        # Data loss: constant throughout training
        weights['data_loss'] = 1.0
        
        # Physics loss: ramp up during early training
        if self.epoch < self.warmup_epochs:
            physics_weight = 0.1
        elif self.epoch < self.physics_ramp_epochs:
            # Linear ramp from 0.1 to 1.0
            ramp_progress = (self.epoch - self.warmup_epochs) / (self.physics_ramp_epochs - self.warmup_epochs)
            physics_weight = 0.1 + 0.9 * ramp_progress
        else:
            physics_weight = 1.0
        weights['physics_loss'] = physics_weight
        
        # Consistency loss: gradually increase importance
        if self.epoch < self.warmup_epochs:
            consistency_weight = 0.1
        else:
            # Sigmoid ramp
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            consistency_weight = 0.5 * (1 + np.tanh(5 * (progress - 0.5)))
        weights['consistency_loss'] = consistency_weight
        
        # Variational loss: KL annealing
        if hasattr(self.variational_loss, 'update_kl_weight'):
            self.variational_loss.update_kl_weight(self.epoch, self.total_epochs, "linear")
        variational_weight = 0.1 * self.variational_loss.kl_weight
        weights['variational_loss'] = variational_weight
        
        return weights
    
    def _update_loss_history(self, loss_components: Dict[str, torch.Tensor]):
        """Update loss history for monitoring."""
        for key in self.loss_history:
            if key in loss_components:
                value = loss_components[key]
                if isinstance(value, torch.Tensor):
                    self.loss_history[key].append(value.item())
                else:
                    self.loss_history[key].append(value)
        
        # Keep only recent history
        max_history = 1000
        for key in self.loss_history:
            if len(self.loss_history[key]) > max_history:
                self.loss_history[key] = self.loss_history[key][-max_history:]
    
    def set_epoch(self, epoch: int, total_epochs: int = None):
        """Set current training epoch for loss scheduling."""
        self.epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
    
    def get_loss_history(self) -> Dict[str, List[float]]:
        """Get loss history for monitoring."""
        return self.loss_history.copy()
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        scheduled_weights = self._compute_scheduled_weights()
        if self.use_adaptive_weighting:
            adaptive_weights = self.adaptive_weighting.get_weights()
            final_weights = {}
            for key in scheduled_weights:
                if key in adaptive_weights:
                    final_weights[key] = scheduled_weights[key] * adaptive_weights[key]
                else:
                    final_weights[key] = scheduled_weights[key]
            return final_weights
        return scheduled_weights
    
    def enable_component(self, component: str, enable: bool = True):
        """Enable or disable specific loss components."""
        if component == 'physics_loss':
            self.use_physics_loss = enable
        elif component == 'consistency_loss':
            self.use_consistency_loss = enable
        elif component == 'variational_loss':
            self.use_variational_loss = enable
        elif component == 'adaptive_weighting':
            self.use_adaptive_weighting = enable
        else:
            raise ValueError(f"Unknown component: {component}")
    
    def reset_adaptive_weights(self):
        """Reset adaptive weighting to initial state."""
        if hasattr(self, 'adaptive_weighting'):
            initial_weights = {
                'data_loss': 1.0,
                'physics_loss': 1.0,
                'consistency_loss': 0.5,
                'variational_loss': 0.1
            }
            self.adaptive_weighting.set_weights(initial_weights)
    
    def get_loss_summary(self) -> Dict[str, float]:
        """Get summary statistics of recent losses."""
        summary = {}
        for key, history in self.loss_history.items():
            if len(history) > 0:
                recent_losses = history[-10:] if len(history) >= 10 else history
                summary[f"{key}_mean"] = np.mean(recent_losses)
                summary[f"{key}_std"] = np.std(recent_losses)
                summary[f"{key}_latest"] = history[-1]
        return summary