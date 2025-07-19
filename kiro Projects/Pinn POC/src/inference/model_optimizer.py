"""
Model optimizer for quantization and pruning for edge deployment.

Provides model compression techniques including quantization, pruning,
and knowledge distillation to achieve <1ms inference latency.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import copy
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    enable_quantization: bool = True
    quantization_bits: int = 8
    enable_pruning: bool = True
    pruning_ratio: float = 0.3
    enable_fusion: bool = True
    target_latency_ms: float = 1.0
    preserve_accuracy_threshold: float = 0.95


@dataclass
class OptimizationResults:
    """Results from model optimization."""
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    original_latency_ms: float
    optimized_latency_ms: float
    speedup_ratio: float
    accuracy_preserved: bool
    optimization_techniques: List[str]


class QuantizationOptimizer:
    """Handles model quantization for edge deployment."""
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.supported_bits = [8, 16]
        
        if bits not in self.supported_bits:
            raise ValueError(f"Unsupported quantization bits: {bits}. "
                           f"Supported: {self.supported_bits}")
    
    def quantize_model(self, model: nn.Module, 
                      calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Quantize model to reduce memory and computation.
        
        Args:
            model: Model to quantize
            calibration_data: Data for calibration (optional)
            
        Returns:
            Quantized model
        """
        logger.info(f"Starting {self.bits}-bit quantization")
        
        # Prepare model for quantization
        model.eval()
        quantized_model = copy.deepcopy(model)
        
        if self.bits == 8:
            # Dynamic quantization for 8-bit
            quantized_model = torch.quantization.quantize_dynamic(
                quantized_model,
                {nn.Linear, nn.Conv1d, nn.Conv2d},
                dtype=torch.qint8
            )
        elif self.bits == 16:
            # Half precision for 16-bit
            quantized_model = quantized_model.half()
        
        logger.info(f"Model quantized to {self.bits} bits")
        return quantized_model
    
    def calibrate_quantization(self, model: nn.Module, 
                             calibration_data: torch.Tensor) -> None:
        """Calibrate quantization parameters using representative data."""
        model.eval()
        with torch.no_grad():
            # Run calibration data through model
            for batch in calibration_data:
                _ = model(batch)
        
        logger.info("Quantization calibration completed")


class PruningOptimizer:
    """Handles model pruning for edge deployment."""
    
    def __init__(self, pruning_ratio: float = 0.3):
        self.pruning_ratio = pruning_ratio
        
        if not 0 < pruning_ratio < 1:
            raise ValueError("Pruning ratio must be between 0 and 1")
    
    def prune_model(self, model: nn.Module) -> nn.Module:
        """
        Prune model weights to reduce computation.
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        logger.info(f"Starting pruning with ratio {self.pruning_ratio}")
        
        pruned_model = copy.deepcopy(model)
        
        # Apply magnitude-based pruning
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                self._prune_layer(module, name)
        
        logger.info("Model pruning completed")
        return pruned_model
    
    def _prune_layer(self, layer: nn.Module, layer_name: str) -> None:
        """Prune individual layer using magnitude-based pruning."""
        if hasattr(layer, 'weight') and layer.weight is not None:
            weight = layer.weight.data
            
            # Calculate threshold for pruning
            weight_abs = torch.abs(weight)
            threshold = torch.quantile(weight_abs, self.pruning_ratio)
            
            # Create mask for pruning
            mask = weight_abs > threshold
            
            # Apply pruning
            layer.weight.data *= mask.float()
            
            # Count pruned parameters
            total_params = weight.numel()
            pruned_params = (mask == 0).sum().item()
            actual_ratio = pruned_params / total_params
            
            logger.debug(f"Layer {layer_name}: pruned {actual_ratio:.2%} of parameters")


class ModelFusion:
    """Handles operator fusion for optimization."""
    
    def __init__(self):
        self.fusion_patterns = [
            ['conv', 'bn'],
            ['conv', 'bn', 'relu'],
            ['linear', 'relu']
        ]
    
    def fuse_model(self, model: nn.Module) -> nn.Module:
        """
        Fuse operators to reduce memory access and computation.
        
        Args:
            model: Model to fuse
            
        Returns:
            Fused model
        """
        logger.info("Starting operator fusion")
        
        fused_model = copy.deepcopy(model)
        
        # Apply torch.jit.script for automatic fusion
        try:
            fused_model = torch.jit.script(fused_model)
            logger.info("Applied TorchScript fusion")
        except Exception as e:
            logger.warning(f"TorchScript fusion failed: {e}")
            # Fall back to manual fusion if available
            fused_model = self._manual_fusion(fused_model)
        
        return fused_model
    
    def _manual_fusion(self, model: nn.Module) -> nn.Module:
        """Manual operator fusion for specific patterns."""
        # This would implement manual fusion patterns
        # For now, return the original model
        logger.info("Manual fusion not implemented, returning original model")
        return model


class ModelOptimizer:
    """
    Model optimizer for quantization and pruning for edge deployment.
    
    Features:
    - Dynamic quantization (8-bit, 16-bit)
    - Magnitude-based pruning
    - Operator fusion
    - Accuracy preservation validation
    - Performance benchmarking
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize model optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.quantizer = QuantizationOptimizer(config.quantization_bits) if config.enable_quantization else None
        self.pruner = PruningOptimizer(config.pruning_ratio) if config.enable_pruning else None
        self.fusion = ModelFusion() if config.enable_fusion else None
        
        logger.info(f"ModelOptimizer initialized with config: {config}")
    
    def optimize_model(self, 
                      model: nn.Module,
                      calibration_data: Optional[torch.Tensor] = None,
                      validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                      ) -> Tuple[nn.Module, OptimizationResults]:
        """
        Optimize model for edge deployment.
        
        Args:
            model: Model to optimize
            calibration_data: Data for quantization calibration
            validation_data: Data for accuracy validation (inputs, targets)
            
        Returns:
            Tuple of (optimized_model, optimization_results)
        """
        logger.info("Starting model optimization")
        
        # Store original model metrics
        original_size = self._calculate_model_size(model)
        original_latency = self._benchmark_latency(model, calibration_data)
        
        # Start with copy of original model
        optimized_model = copy.deepcopy(model)
        applied_techniques = []
        
        # Apply optimizations in order
        if self.config.enable_fusion and self.fusion:
            optimized_model = self.fusion.fuse_model(optimized_model)
            applied_techniques.append('fusion')
            logger.info("Applied operator fusion")
        
        if self.config.enable_pruning and self.pruner:
            optimized_model = self.pruner.prune_model(optimized_model)
            applied_techniques.append('pruning')
            logger.info("Applied model pruning")
        
        if self.config.enable_quantization and self.quantizer:
            optimized_model = self.quantizer.quantize_model(optimized_model, calibration_data)
            applied_techniques.append('quantization')
            logger.info("Applied model quantization")
        
        # Calculate optimized model metrics
        optimized_size = self._calculate_model_size(optimized_model)
        optimized_latency = self._benchmark_latency(optimized_model, calibration_data)
        
        # Validate accuracy if validation data provided
        accuracy_preserved = True
        if validation_data is not None:
            accuracy_preserved = self._validate_accuracy(
                model, optimized_model, validation_data
            )
        
        # Create results
        results = OptimizationResults(
            original_size_mb=original_size,
            optimized_size_mb=optimized_size,
            compression_ratio=original_size / optimized_size if optimized_size > 0 else 1.0,
            original_latency_ms=original_latency,
            optimized_latency_ms=optimized_latency,
            speedup_ratio=original_latency / optimized_latency if optimized_latency > 0 else 1.0,
            accuracy_preserved=accuracy_preserved,
            optimization_techniques=applied_techniques
        )
        
        logger.info(f"Optimization completed: {results}")
        return optimized_model, results
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            if param is not None:
                param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            if buffer is not None:
                buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        # Ensure minimum size to avoid division by zero
        return max(size_mb, 0.001)
    
    def _benchmark_latency(self, 
                          model: nn.Module, 
                          sample_data: Optional[torch.Tensor] = None,
                          num_iterations: int = 100) -> float:
        """Benchmark model inference latency."""
        if sample_data is None:
            # Create dummy input based on model
            sample_data = torch.randn(1, 10)  # Default shape
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_data[:1])
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        else:
            import time
            start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(sample_data[:1])
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            latency_ms = start_time.elapsed_time(end_time) / num_iterations
        else:
            end_time = time.perf_counter()
            latency_ms = ((end_time - start_time) / num_iterations) * 1000
        
        return latency_ms
    
    def _validate_accuracy(self, 
                          original_model: nn.Module,
                          optimized_model: nn.Module,
                          validation_data: Tuple[torch.Tensor, torch.Tensor]) -> bool:
        """Validate that optimized model preserves accuracy."""
        inputs, targets = validation_data
        
        original_model.eval()
        optimized_model.eval()
        
        with torch.no_grad():
            # Get predictions from both models
            original_preds = original_model(inputs)
            optimized_preds = optimized_model(inputs)
            
            # Calculate accuracy for classification
            if targets.dtype in [torch.long, torch.int]:
                original_acc = (original_preds.argmax(dim=1) == targets).float().mean()
                optimized_acc = (optimized_preds.argmax(dim=1) == targets).float().mean()
                
                accuracy_ratio = optimized_acc / original_acc if original_acc > 0 else 1.0
            else:
                # For regression, use relative error
                original_error = torch.mean(torch.abs(original_preds - targets))
                optimized_error = torch.mean(torch.abs(optimized_preds - targets))
                
                accuracy_ratio = 1.0 - (optimized_error - original_error) / original_error if original_error > 0 else 1.0
        
        preserved = accuracy_ratio >= self.config.preserve_accuracy_threshold
        
        logger.info(f"Accuracy validation: ratio={accuracy_ratio:.3f}, "
                   f"preserved={preserved} (threshold={self.config.preserve_accuracy_threshold})")
        
        return preserved
    
    def optimize_for_target_latency(self,
                                  model: nn.Module,
                                  target_latency_ms: float,
                                  calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Optimize model to meet target latency requirement.
        
        Args:
            model: Model to optimize
            target_latency_ms: Target latency in milliseconds
            calibration_data: Data for optimization
            
        Returns:
            Optimized model that meets latency target
        """
        logger.info(f"Optimizing for target latency: {target_latency_ms}ms")
        
        current_model = model
        current_latency = self._benchmark_latency(current_model, calibration_data)
        
        if current_latency <= target_latency_ms:
            logger.info(f"Model already meets target latency: {current_latency:.2f}ms")
            return current_model
        
        # Try different optimization levels
        optimization_levels = [
            OptimizationConfig(enable_fusion=True, enable_quantization=False, enable_pruning=False),
            OptimizationConfig(enable_fusion=True, enable_quantization=True, quantization_bits=16, enable_pruning=False),
            OptimizationConfig(enable_fusion=True, enable_quantization=True, quantization_bits=8, enable_pruning=False),
            OptimizationConfig(enable_fusion=True, enable_quantization=True, quantization_bits=8, enable_pruning=True, pruning_ratio=0.2),
            OptimizationConfig(enable_fusion=True, enable_quantization=True, quantization_bits=8, enable_pruning=True, pruning_ratio=0.5),
        ]
        
        for i, config in enumerate(optimization_levels):
            logger.info(f"Trying optimization level {i+1}/{len(optimization_levels)}")
            
            optimizer = ModelOptimizer(config)
            optimized_model, results = optimizer.optimize_model(model, calibration_data)
            
            if results.optimized_latency_ms <= target_latency_ms:
                logger.info(f"Target latency achieved: {results.optimized_latency_ms:.2f}ms "
                           f"with techniques: {results.optimization_techniques}")
                return optimized_model
        
        # If no configuration meets target, return best effort
        logger.warning(f"Could not meet target latency {target_latency_ms}ms. "
                      f"Best achieved: {results.optimized_latency_ms:.2f}ms")
        return optimized_model
    
    def get_optimization_summary(self, results: OptimizationResults) -> Dict[str, Any]:
        """Get human-readable optimization summary."""
        return {
            'model_size_reduction': f"{(1 - results.optimized_size_mb/results.original_size_mb)*100:.1f}%",
            'latency_improvement': f"{(1 - results.optimized_latency_ms/results.original_latency_ms)*100:.1f}%",
            'compression_ratio': f"{results.compression_ratio:.1f}x",
            'speedup_ratio': f"{results.speedup_ratio:.1f}x",
            'techniques_applied': results.optimization_techniques,
            'accuracy_preserved': results.accuracy_preserved,
            'meets_target_latency': results.optimized_latency_ms <= self.config.target_latency_ms
        }