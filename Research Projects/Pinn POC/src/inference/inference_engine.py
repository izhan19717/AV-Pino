"""
Inference engine with optimized forward pass for <1ms latency.

Provides high-performance inference capabilities with memory management,
batch processing, and hardware optimization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
import logging
import numpy as np
from contextlib import contextmanager

from .memory_manager import MemoryManager
from .hardware_profiler import HardwareProfiler
from .model_optimizer import ModelOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    batch_size: int = 1
    max_batch_size: int = 32
    enable_batching: bool = True
    target_latency_ms: float = 1.0
    enable_optimization: bool = True
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    precision: str = 'float32'  # 'float32', 'float16', 'int8'
    enable_profiling: bool = False


@dataclass
class InferenceResult:
    """Result from inference operation."""
    predictions: torch.Tensor
    latency_ms: float
    batch_size: int
    confidence: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchProcessor:
    """Handles dynamic batching for improved throughput."""
    
    def __init__(self, max_batch_size: int = 32, timeout_ms: float = 10.0):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests = []
        self.batch_ready_event = None
    
    def add_request(self, input_data: torch.Tensor, request_id: str) -> None:
        """Add request to batch queue."""
        self.pending_requests.append({
            'input': input_data,
            'id': request_id,
            'timestamp': time.time()
        })
    
    def get_batch(self) -> Optional[Tuple[torch.Tensor, List[str]]]:
        """Get next batch for processing."""
        if not self.pending_requests:
            return None
        
        # Determine batch size
        batch_size = min(len(self.pending_requests), self.max_batch_size)
        
        # Extract batch
        batch_requests = self.pending_requests[:batch_size]
        self.pending_requests = self.pending_requests[batch_size:]
        
        # Stack inputs
        inputs = torch.stack([req['input'] for req in batch_requests])
        request_ids = [req['id'] for req in batch_requests]
        
        return inputs, request_ids
    
    def should_process_batch(self) -> bool:
        """Check if batch should be processed now."""
        if not self.pending_requests:
            return False
        
        # Process if batch is full
        if len(self.pending_requests) >= self.max_batch_size:
            return True
        
        # Process if oldest request is timing out
        oldest_time = self.pending_requests[0]['timestamp']
        if (time.time() - oldest_time) * 1000 >= self.timeout_ms:
            return True
        
        return False


class InferenceEngine:
    """
    Inference engine with optimized forward pass for <1ms latency.
    
    Features:
    - Optimized model execution
    - Dynamic batching
    - Memory management
    - Hardware profiling
    - Multi-precision support
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: InferenceConfig,
                 memory_manager: Optional[MemoryManager] = None,
                 profiler: Optional[HardwareProfiler] = None):
        """
        Initialize inference engine.
        
        Args:
            model: Neural network model
            config: Inference configuration
            memory_manager: Memory manager instance
            profiler: Hardware profiler instance
        """
        self.config = config
        self.original_model = model
        self.optimized_model = None
        
        # Initialize components
        self.memory_manager = memory_manager or MemoryManager()
        self.profiler = profiler or HardwareProfiler() if config.enable_profiling else None
        self.batch_processor = BatchProcessor(config.max_batch_size) if config.enable_batching else None
        
        # Setup device
        self.device = self._setup_device()
        
        # Optimize model if requested
        if config.enable_optimization:
            self._optimize_model()
        else:
            self.optimized_model = self.original_model
        
        # Move model to device
        self.optimized_model = self.optimized_model.to(self.device)
        self.optimized_model.eval()
        
        # Performance tracking
        self.inference_count = 0
        self.total_latency = 0.0
        
        logger.info(f"InferenceEngine initialized on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _optimize_model(self) -> None:
        """Optimize model for inference."""
        logger.info("Optimizing model for inference")
        
        # Create optimization config
        opt_config = OptimizationConfig(
            enable_quantization=self.config.precision in ['int8', 'float16'],
            quantization_bits=8 if self.config.precision == 'int8' else 16,
            enable_pruning=True,
            pruning_ratio=0.2,
            enable_fusion=True,
            target_latency_ms=self.config.target_latency_ms
        )
        
        # Optimize model
        optimizer = ModelOptimizer(opt_config)
        
        # Create sample data for optimization
        sample_input = self._create_sample_input()
        
        self.optimized_model, optimization_results = optimizer.optimize_model(
            self.original_model,
            calibration_data=sample_input
        )
        
        logger.info(f"Model optimization completed: "
                   f"size reduction={optimization_results.compression_ratio:.1f}x, "
                   f"speedup={optimization_results.speedup_ratio:.1f}x")
    
    def _create_sample_input(self) -> torch.Tensor:
        """Create sample input for model optimization."""
        # Try to infer input shape from model
        try:
            # Get first layer to determine input size
            first_layer = None
            for module in self.original_model.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    first_layer = module
                    break
            
            if first_layer and hasattr(first_layer, 'in_features'):
                input_dim = first_layer.in_features
                return torch.randn(self.config.batch_size, input_dim)
            elif first_layer and hasattr(first_layer, 'in_channels'):
                # For conv layers, create appropriate shape
                return torch.randn(self.config.batch_size, first_layer.in_channels, 32)
        except:
            pass
        
        # Default fallback
        return torch.randn(self.config.batch_size, 10)
    
    def predict(self, 
                input_data: torch.Tensor,
                return_uncertainty: bool = False) -> InferenceResult:
        """
        Perform single prediction.
        
        Args:
            input_data: Input tensor
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Inference result with predictions and metadata
        """
        start_time = time.perf_counter()
        
        # Ensure input is on correct device
        input_data = input_data.to(self.device)
        
        # Profile if enabled
        profiler_context = None
        if self.profiler:
            profiler_context = self.profiler.profile_operation('inference')
            profiler_context.__enter__()
        
        try:
            with self.memory_manager.managed_inference():
                with torch.no_grad():
                    # Forward pass
                    predictions = self.optimized_model(input_data)
                    
                    # Handle uncertainty if requested
                    uncertainty = None
                    confidence = None
                    
                    if return_uncertainty and hasattr(self.optimized_model, 'predict_with_uncertainty'):
                        predictions, uncertainty = self.optimized_model.predict_with_uncertainty(input_data)
                        confidence = 1.0 - uncertainty  # Simple confidence estimate
        
        finally:
            if profiler_context:
                profiler_context.__exit__(None, None, None)
        
        # Calculate latency
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        # Update statistics
        self.inference_count += 1
        self.total_latency += latency_ms
        
        # Create result
        result = InferenceResult(
            predictions=predictions,
            latency_ms=latency_ms,
            batch_size=input_data.shape[0],
            confidence=confidence,
            uncertainty=uncertainty,
            metadata={
                'device': str(self.device),
                'precision': self.config.precision,
                'inference_count': self.inference_count
            }
        )
        
        return result
    
    def predict_batch(self, 
                     batch_data: torch.Tensor,
                     return_uncertainty: bool = False) -> InferenceResult:
        """
        Perform batch prediction.
        
        Args:
            batch_data: Batch of input tensors
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Inference result for the batch
        """
        if batch_data.shape[0] > self.config.max_batch_size:
            # Split large batches
            results = []
            for i in range(0, batch_data.shape[0], self.config.max_batch_size):
                batch_slice = batch_data[i:i + self.config.max_batch_size]
                result = self.predict(batch_slice, return_uncertainty)
                results.append(result)
            
            # Combine results
            combined_predictions = torch.cat([r.predictions for r in results], dim=0)
            combined_latency = sum(r.latency_ms for r in results)
            
            combined_uncertainty = None
            combined_confidence = None
            if return_uncertainty and results[0].uncertainty is not None:
                combined_uncertainty = torch.cat([r.uncertainty for r in results], dim=0)
                combined_confidence = torch.cat([r.confidence for r in results], dim=0)
            
            return InferenceResult(
                predictions=combined_predictions,
                latency_ms=combined_latency,
                batch_size=batch_data.shape[0],
                confidence=combined_confidence,
                uncertainty=combined_uncertainty,
                metadata={'split_batches': len(results)}
            )
        else:
            return self.predict(batch_data, return_uncertainty)
    
    @contextmanager
    def benchmark_mode(self):
        """Context manager for benchmarking mode."""
        original_profiling = self.config.enable_profiling
        self.config.enable_profiling = True
        
        if not self.profiler:
            self.profiler = HardwareProfiler()
        
        try:
            yield self
        finally:
            self.config.enable_profiling = original_profiling
    
    def benchmark_performance(self, 
                            sample_input: torch.Tensor,
                            num_iterations: int = 100,
                            warmup_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            sample_input: Sample input for benchmarking
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Performance metrics
        """
        logger.info(f"Benchmarking performance with {num_iterations} iterations")
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = self.predict(sample_input)
        
        # Benchmark
        latencies = []
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            result = self.predict(sample_input)
            latencies.append(result.latency_ms)
        
        total_time = time.perf_counter() - start_time
        
        # Calculate statistics
        metrics = {
            'avg_latency_ms': np.mean(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_ops_per_sec': num_iterations / total_time,
            'total_time_sec': total_time,
            'meets_target_latency': np.mean(latencies) <= self.config.target_latency_ms
        }
        
        logger.info(f"Benchmark results: avg={metrics['avg_latency_ms']:.2f}ms, "
                   f"p95={metrics['p95_latency_ms']:.2f}ms, "
                   f"throughput={metrics['throughput_ops_per_sec']:.1f} ops/sec")
        
        return metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        avg_latency = self.total_latency / self.inference_count if self.inference_count > 0 else 0
        
        stats = {
            'total_inferences': self.inference_count,
            'average_latency_ms': avg_latency,
            'meets_target_latency': avg_latency <= self.config.target_latency_ms,
            'device': str(self.device),
            'precision': self.config.precision,
            'optimization_enabled': self.config.enable_optimization
        }
        
        # Add memory stats if available
        if self.memory_manager:
            memory_summary = self.memory_manager.get_memory_summary()
            stats['memory_usage_mb'] = memory_summary.get('current_memory_mb', 0)
        
        # Add profiler stats if available
        if self.profiler:
            profiler_summary = self.profiler.get_performance_summary()
            if 'performance_stats' in profiler_summary:
                stats.update(profiler_summary['performance_stats'])
        
        return stats
    
    def optimize_for_latency(self, target_latency_ms: float) -> bool:
        """
        Optimize engine to meet target latency.
        
        Args:
            target_latency_ms: Target latency in milliseconds
            
        Returns:
            True if target latency can be achieved
        """
        logger.info(f"Optimizing for target latency: {target_latency_ms}ms")
        
        # Update config
        self.config.target_latency_ms = target_latency_ms
        
        # Re-optimize model with new target
        if self.config.enable_optimization:
            optimizer = ModelOptimizer(OptimizationConfig(
                target_latency_ms=target_latency_ms
            ))
            
            sample_input = self._create_sample_input()
            optimized_model = optimizer.optimize_for_target_latency(
                self.original_model,
                target_latency_ms,
                sample_input
            )
            
            self.optimized_model = optimized_model.to(self.device)
            self.optimized_model.eval()
        
        # Test current performance
        sample_input = self._create_sample_input()
        result = self.predict(sample_input)
        
        meets_target = result.latency_ms <= target_latency_ms
        logger.info(f"Latency optimization result: {result.latency_ms:.2f}ms "
                   f"(target: {target_latency_ms}ms, meets_target: {meets_target})")
        
        return meets_target
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.inference_count = 0
        self.total_latency = 0.0
        logger.info("Performance statistics reset")
    
    def save_model(self, filepath: str) -> None:
        """Save optimized model to file."""
        torch.save({
            'model_state_dict': self.optimized_model.state_dict(),
            'config': self.config,
            'performance_stats': self.get_performance_stats()
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load optimized model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.optimized_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimized_model.eval()
        
        logger.info(f"Model loaded from {filepath}")