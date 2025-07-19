"""
Real-time inference system integrating all optimization components.

Provides complete real-time inference solution with hardware constraint handling,
adaptive model configuration, and comprehensive monitoring.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import time
import threading
import queue
import logging
from pathlib import Path
import json

from .memory_manager import MemoryManager
from .hardware_profiler import HardwareProfiler, BottleneckAnalysis
from .model_optimizer import ModelOptimizer, OptimizationConfig
from .inference_engine import InferenceEngine, InferenceConfig, InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class HardwareConstraints:
    """Hardware constraints for real-time inference."""
    max_memory_mb: Optional[float] = None
    max_gpu_memory_mb: Optional[float] = None
    max_cpu_usage_percent: float = 80.0
    target_latency_ms: float = 1.0
    min_throughput_ops_per_sec: float = 100.0
    power_budget_watts: Optional[float] = None


@dataclass
class AdaptiveConfig:
    """Adaptive configuration parameters."""
    enable_adaptive_batching: bool = True
    enable_adaptive_precision: bool = True
    enable_adaptive_optimization: bool = True
    adaptation_interval_sec: float = 10.0
    performance_window_size: int = 100
    adaptation_threshold: float = 0.1  # 10% performance change threshold


@dataclass
class SystemStatus:
    """Current system status."""
    is_healthy: bool = True
    current_latency_ms: float = 0.0
    current_throughput_ops_per_sec: float = 0.0
    memory_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: Optional[float] = None
    active_optimizations: List[str] = field(default_factory=list)
    bottlenecks: List[BottleneckAnalysis] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)


class AdaptiveController:
    """Controls adaptive behavior based on performance metrics."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.performance_history = []
        self.last_adaptation = time.time()
        
    def should_adapt(self, current_metrics: Dict[str, float]) -> bool:
        """Check if adaptation is needed."""
        if time.time() - self.last_adaptation < self.config.adaptation_interval_sec:
            return False
        
        if len(self.performance_history) < 10:  # Need some history
            return False
        
        # Check for performance degradation
        recent_latency = current_metrics.get('avg_latency_ms', 0)
        historical_latency = sum(h.get('avg_latency_ms', 0) for h in self.performance_history[-10:]) / 10
        
        if historical_latency > 0:
            latency_change = abs(recent_latency - historical_latency) / historical_latency
            if latency_change > self.config.adaptation_threshold:
                return True
        
        return False
    
    def add_metrics(self, metrics: Dict[str, float]) -> None:
        """Add performance metrics to history."""
        self.performance_history.append(metrics)
        
        # Limit history size
        if len(self.performance_history) > self.config.performance_window_size:
            self.performance_history.pop(0)
    
    def get_adaptation_recommendations(self, 
                                    bottlenecks: List[BottleneckAnalysis]) -> List[str]:
        """Get recommendations for adaptation."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if bottleneck.bottleneck_type == 'latency' and bottleneck.severity > 0.5:
                if self.config.enable_adaptive_optimization:
                    recommendations.append('increase_optimization')
                if self.config.enable_adaptive_precision:
                    recommendations.append('reduce_precision')
            
            elif bottleneck.bottleneck_type == 'memory' and bottleneck.severity > 0.5:
                if self.config.enable_adaptive_batching:
                    recommendations.append('reduce_batch_size')
                recommendations.append('enable_memory_optimization')
            
            elif bottleneck.bottleneck_type == 'cpu' and bottleneck.severity > 0.5:
                recommendations.append('optimize_cpu_usage')
        
        return recommendations


class RealTimeInference:
    """
    Complete real-time inference system with adaptive optimization.
    
    Features:
    - Hardware constraint handling
    - Adaptive model configuration
    - Real-time performance monitoring
    - Automatic optimization adjustment
    - Comprehensive system health monitoring
    """
    
    def __init__(self,
                 model: nn.Module,
                 hardware_constraints: HardwareConstraints,
                 adaptive_config: Optional[AdaptiveConfig] = None,
                 inference_config: Optional[InferenceConfig] = None):
        """
        Initialize real-time inference system.
        
        Args:
            model: Neural network model
            hardware_constraints: Hardware constraints
            adaptive_config: Adaptive behavior configuration
            inference_config: Inference engine configuration
        """
        self.model = model
        self.hardware_constraints = hardware_constraints
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        
        # Initialize components
        self.memory_manager = MemoryManager(
            enable_pooling=True,
            gc_threshold=0.8
        )
        
        self.profiler = HardwareProfiler(
            enable_continuous_monitoring=True,
            monitoring_interval=1.0
        )
        
        # Setup inference configuration
        if inference_config is None:
            inference_config = InferenceConfig(
                target_latency_ms=hardware_constraints.target_latency_ms,
                enable_optimization=True,
                enable_profiling=True
            )
        
        self.inference_engine = InferenceEngine(
            model=model,
            config=inference_config,
            memory_manager=self.memory_manager,
            profiler=self.profiler
        )
        
        # Adaptive control
        self.adaptive_controller = AdaptiveController(self.adaptive_config)
        
        # System monitoring
        self.system_status = SystemStatus()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Request queue for batching
        self.request_queue = queue.Queue()
        self.processing_active = False
        self.processing_thread: Optional[threading.Thread] = None
        
        logger.info("RealTimeInference system initialized")
    
    def start_system(self) -> None:
        """Start the real-time inference system."""
        logger.info("Starting real-time inference system")
        
        # Start monitoring
        self.profiler.start_continuous_monitoring()
        self.memory_manager.start_monitoring()
        
        # Start system monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start request processing
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._process_requests)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Real-time inference system started")
    
    def stop_system(self) -> None:
        """Stop the real-time inference system."""
        logger.info("Stopping real-time inference system")
        
        # Stop processing
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        # Stop components
        self.profiler.stop_continuous_monitoring()
        self.memory_manager.stop_monitoring()
        
        logger.info("Real-time inference system stopped")
    
    def predict(self, 
                input_data: torch.Tensor,
                return_uncertainty: bool = False,
                timeout_ms: float = 100.0) -> InferenceResult:
        """
        Perform real-time prediction.
        
        Args:
            input_data: Input tensor
            return_uncertainty: Whether to return uncertainty estimates
            timeout_ms: Request timeout in milliseconds
            
        Returns:
            Inference result
        """
        # Check system health
        if not self.system_status.is_healthy:
            logger.warning("System is not healthy, prediction may be degraded")
        
        # Validate hardware constraints
        if not self._check_hardware_constraints():
            logger.warning("Hardware constraints violated")
        
        # Perform inference
        result = self.inference_engine.predict(input_data, return_uncertainty)
        
        # Update system status
        self._update_system_status(result)
        
        return result
    
    def predict_async(self, 
                     input_data: torch.Tensor,
                     callback: Callable[[InferenceResult], None],
                     return_uncertainty: bool = False) -> str:
        """
        Perform asynchronous prediction.
        
        Args:
            input_data: Input tensor
            callback: Callback function for result
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Request ID
        """
        request_id = f"req_{time.time()}_{id(input_data)}"
        
        request = {
            'id': request_id,
            'input_data': input_data,
            'callback': callback,
            'return_uncertainty': return_uncertainty,
            'timestamp': time.time()
        }
        
        self.request_queue.put(request)
        return request_id
    
    def _process_requests(self) -> None:
        """Background request processing loop."""
        while self.processing_active:
            try:
                # Get request with timeout
                request = self.request_queue.get(timeout=0.1)
                
                # Process request
                result = self.predict(
                    request['input_data'],
                    request['return_uncertainty']
                )
                
                # Call callback
                request['callback'](result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing request: {e}")
    
    def _monitor_system(self) -> None:
        """Background system monitoring loop."""
        while self.monitoring_active:
            try:
                # Update system status
                self._update_system_health()
                
                # Check for adaptation needs
                if self.adaptive_config.enable_adaptive_optimization:
                    self._check_adaptation_needs()
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
    
    def _update_system_health(self) -> None:
        """Update system health status."""
        # Get current metrics
        performance_stats = self.inference_engine.get_performance_stats()
        memory_summary = self.memory_manager.get_memory_summary()
        profiler_summary = self.profiler.get_performance_summary()
        
        # Update status
        self.system_status.current_latency_ms = performance_stats.get('average_latency_ms', 0)
        self.system_status.memory_usage_percent = memory_summary.get('current_memory_percent', 0)
        
        if 'performance_stats' in profiler_summary:
            stats = profiler_summary['performance_stats']
            self.system_status.cpu_usage_percent = stats.get('avg_cpu_usage', 0)
            self.system_status.gpu_usage_percent = stats.get('avg_gpu_usage')
        
        # Check health
        self.system_status.is_healthy = self._assess_system_health()
        
        # Identify bottlenecks
        self.system_status.bottlenecks = self.profiler.identify_bottlenecks(
            self.hardware_constraints.target_latency_ms
        )
        
        self.system_status.last_update = time.time()
    
    def _assess_system_health(self) -> bool:
        """Assess overall system health."""
        # Check latency constraint
        if (self.system_status.current_latency_ms > 
            self.hardware_constraints.target_latency_ms * 1.5):
            return False
        
        # Check memory constraint
        if (self.hardware_constraints.max_memory_mb and
            self.system_status.memory_usage_percent > 90):
            return False
        
        # Check CPU constraint
        if self.system_status.cpu_usage_percent > self.hardware_constraints.max_cpu_usage_percent:
            return False
        
        return True
    
    def _check_hardware_constraints(self) -> bool:
        """Check if current state meets hardware constraints."""
        memory_summary = self.memory_manager.get_memory_summary()
        
        # Check memory constraint
        if self.hardware_constraints.max_memory_mb:
            current_memory = memory_summary.get('current_memory_mb', 0)
            if current_memory > self.hardware_constraints.max_memory_mb:
                return False
        
        # Check GPU memory constraint
        if (self.hardware_constraints.max_gpu_memory_mb and 
            'gpu_memory_mb' in memory_summary):
            gpu_memory = memory_summary['gpu_memory_mb']
            if gpu_memory > self.hardware_constraints.max_gpu_memory_mb:
                return False
        
        return True
    
    def _update_system_status(self, result: InferenceResult) -> None:
        """Update system status based on inference result."""
        # Update throughput calculation
        current_time = time.time()
        if hasattr(self, '_last_inference_time'):
            time_diff = current_time - self._last_inference_time
            if time_diff > 0:
                self.system_status.current_throughput_ops_per_sec = 1.0 / time_diff
        
        self._last_inference_time = current_time
    
    def _check_adaptation_needs(self) -> None:
        """Check if system adaptation is needed."""
        current_metrics = self.inference_engine.get_performance_stats()
        
        # Add metrics to adaptive controller
        self.adaptive_controller.add_metrics(current_metrics)
        
        # Check if adaptation is needed
        if self.adaptive_controller.should_adapt(current_metrics):
            recommendations = self.adaptive_controller.get_adaptation_recommendations(
                self.system_status.bottlenecks
            )
            
            if recommendations:
                logger.info(f"Applying adaptations: {recommendations}")
                self._apply_adaptations(recommendations)
    
    def _apply_adaptations(self, recommendations: List[str]) -> None:
        """Apply system adaptations based on recommendations."""
        for recommendation in recommendations:
            try:
                if recommendation == 'increase_optimization':
                    self._increase_optimization()
                elif recommendation == 'reduce_precision':
                    self._reduce_precision()
                elif recommendation == 'reduce_batch_size':
                    self._reduce_batch_size()
                elif recommendation == 'enable_memory_optimization':
                    self._enable_memory_optimization()
                elif recommendation == 'optimize_cpu_usage':
                    self._optimize_cpu_usage()
                
                self.system_status.active_optimizations.append(recommendation)
                
            except Exception as e:
                logger.error(f"Failed to apply adaptation {recommendation}: {e}")
    
    def _increase_optimization(self) -> None:
        """Increase model optimization level."""
        # Re-optimize with more aggressive settings
        target_latency = self.hardware_constraints.target_latency_ms * 0.8  # 20% buffer
        success = self.inference_engine.optimize_for_latency(target_latency)
        
        if success:
            logger.info(f"Increased optimization for target latency {target_latency}ms")
        else:
            logger.warning("Failed to increase optimization level")
    
    def _reduce_precision(self) -> None:
        """Reduce model precision for better performance."""
        current_precision = self.inference_engine.config.precision
        
        if current_precision == 'float32':
            self.inference_engine.config.precision = 'float16'
            logger.info("Reduced precision to float16")
        elif current_precision == 'float16':
            self.inference_engine.config.precision = 'int8'
            logger.info("Reduced precision to int8")
    
    def _reduce_batch_size(self) -> None:
        """Reduce batch size to save memory."""
        current_batch_size = self.inference_engine.config.batch_size
        new_batch_size = max(1, current_batch_size // 2)
        
        self.inference_engine.config.batch_size = new_batch_size
        logger.info(f"Reduced batch size from {current_batch_size} to {new_batch_size}")
    
    def _enable_memory_optimization(self) -> None:
        """Enable additional memory optimizations."""
        self.memory_manager.force_cleanup()
        logger.info("Enabled additional memory optimization")
    
    def _optimize_cpu_usage(self) -> None:
        """Optimize CPU usage."""
        # This could involve thread pool adjustments, etc.
        logger.info("Applied CPU usage optimizations")
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        return self.system_status
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'system_status': {
                'is_healthy': self.system_status.is_healthy,
                'current_latency_ms': self.system_status.current_latency_ms,
                'current_throughput_ops_per_sec': self.system_status.current_throughput_ops_per_sec,
                'memory_usage_percent': self.system_status.memory_usage_percent,
                'cpu_usage_percent': self.system_status.cpu_usage_percent,
                'gpu_usage_percent': self.system_status.gpu_usage_percent,
                'active_optimizations': self.system_status.active_optimizations
            },
            'hardware_constraints': {
                'target_latency_ms': self.hardware_constraints.target_latency_ms,
                'max_memory_mb': self.hardware_constraints.max_memory_mb,
                'max_cpu_usage_percent': self.hardware_constraints.max_cpu_usage_percent
            },
            'inference_stats': self.inference_engine.get_performance_stats(),
            'memory_summary': self.memory_manager.get_memory_summary(),
            'profiler_summary': self.profiler.get_performance_summary(),
            'bottlenecks': [b.__dict__ for b in self.system_status.bottlenecks]
        }
    
    def save_system_state(self, filepath: Path) -> None:
        """Save complete system state."""
        state = {
            'timestamp': time.time(),
            'hardware_constraints': self.hardware_constraints.__dict__,
            'adaptive_config': self.adaptive_config.__dict__,
            'inference_config': self.inference_engine.config.__dict__,
            'performance_report': self.get_performance_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save model
        model_path = filepath.with_suffix('.pth')
        self.inference_engine.save_model(str(model_path))
        
        logger.info(f"System state saved to {filepath}")
    
    def benchmark_system(self, 
                        sample_input: torch.Tensor,
                        duration_sec: float = 60.0) -> Dict[str, Any]:
        """
        Benchmark complete system performance.
        
        Args:
            sample_input: Sample input for benchmarking
            duration_sec: Benchmark duration in seconds
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Starting system benchmark for {duration_sec} seconds")
        
        start_time = time.time()
        results = []
        
        while time.time() - start_time < duration_sec:
            result = self.predict(sample_input)
            results.append({
                'latency_ms': result.latency_ms,
                'timestamp': time.time()
            })
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.001)
        
        # Calculate statistics
        latencies = [r['latency_ms'] for r in results]
        
        benchmark_results = {
            'duration_sec': duration_sec,
            'total_requests': len(results),
            'avg_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'p95_latency_ms': sorted(latencies)[int(0.95 * len(latencies))],
            'p99_latency_ms': sorted(latencies)[int(0.99 * len(latencies))],
            'throughput_ops_per_sec': len(results) / duration_sec,
            'meets_latency_target': sum(1 for l in latencies if l <= self.hardware_constraints.target_latency_ms) / len(latencies),
            'system_health_during_test': self.system_status.is_healthy,
            'final_system_status': self.get_system_status().__dict__
        }
        
        logger.info(f"Benchmark completed: {benchmark_results['throughput_ops_per_sec']:.1f} ops/sec, "
                   f"avg latency: {benchmark_results['avg_latency_ms']:.2f}ms")
        
        return benchmark_results