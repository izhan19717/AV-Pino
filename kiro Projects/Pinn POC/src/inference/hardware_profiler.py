"""
Hardware profiler for performance monitoring and bottleneck identification.

Provides comprehensive profiling of CPU, GPU, memory, and I/O performance
during inference operations.
"""

import time
import psutil
import torch
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation_name: str
    duration_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    throughput_ops_per_sec: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class HardwareSpecs:
    """Hardware specifications."""
    cpu_count: int
    cpu_freq_mhz: float
    total_memory_gb: float
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    gpu_compute_capability: Optional[Tuple[int, int]] = None


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks."""
    bottleneck_type: str  # 'cpu', 'memory', 'gpu', 'io'
    severity: float  # 0-1 scale
    description: str
    recommendations: List[str]


class PerformanceProfiler:
    """Context manager for profiling individual operations."""
    
    def __init__(self, operation_name: str, profiler: 'HardwareProfiler'):
        self.operation_name = operation_name
        self.profiler = profiler
        self.start_time = None
        self.start_cpu = None
        self.start_memory = None
        self.start_gpu_memory = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_cpu = psutil.cpu_percent()
        self.start_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        if torch.cuda.is_available():
            self.start_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration_ms = (end_time - self.start_time) * 1000
        
        # Calculate resource usage
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)
        
        gpu_usage = None
        gpu_memory = None
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            except:
                pass
        
        # Create metrics
        metrics = PerformanceMetrics(
            operation_name=self.operation_name,
            duration_ms=duration_ms,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            gpu_usage_percent=gpu_usage,
            gpu_memory_mb=gpu_memory
        )
        
        # Add to profiler
        self.profiler.add_metrics(metrics)


class HardwareProfiler:
    """
    Hardware profiler for performance monitoring and bottleneck identification.
    
    Features:
    - Real-time performance monitoring
    - Bottleneck identification
    - Hardware capability detection
    - Performance regression detection
    - Optimization recommendations
    """
    
    def __init__(self, 
                 enable_continuous_monitoring: bool = False,
                 monitoring_interval: float = 0.1,
                 max_history_size: int = 10000):
        """
        Initialize hardware profiler.
        
        Args:
            enable_continuous_monitoring: Enable background monitoring
            monitoring_interval: Monitoring interval in seconds
            max_history_size: Maximum number of metrics to store
        """
        self.enable_continuous_monitoring = enable_continuous_monitoring
        self.monitoring_interval = monitoring_interval
        self.max_history_size = max_history_size
        
        # Performance metrics storage
        self.metrics_history: List[PerformanceMetrics] = []
        self._metrics_lock = threading.Lock()
        
        # Monitoring thread
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Hardware specifications
        self.hardware_specs = self._detect_hardware_specs()
        
        # Baseline performance
        self._baseline_metrics: Optional[Dict[str, float]] = None
        
        logger.info(f"HardwareProfiler initialized: {self.hardware_specs}")
    
    def _detect_hardware_specs(self) -> HardwareSpecs:
        """Detect hardware specifications."""
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0
        
        # Memory information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024 ** 3)
        
        # GPU information
        gpu_name = None
        gpu_memory_gb = None
        gpu_compute_capability = None
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024 ** 3)
            
            # Get compute capability
            props = torch.cuda.get_device_properties(0)
            gpu_compute_capability = (props.major, props.minor)
        
        return HardwareSpecs(
            cpu_count=cpu_count,
            cpu_freq_mhz=cpu_freq_mhz,
            total_memory_gb=total_memory_gb,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            gpu_compute_capability=gpu_compute_capability
        )
    
    def profile_operation(self, operation_name: str) -> PerformanceProfiler:
        """Create a profiler context for an operation."""
        return PerformanceProfiler(operation_name, self)
    
    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add performance metrics to history."""
        with self._metrics_lock:
            self.metrics_history.append(metrics)
            
            # Limit history size
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
    
    def benchmark_inference(self, 
                          inference_fn: Callable,
                          input_data: torch.Tensor,
                          num_iterations: int = 100,
                          warmup_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            inference_fn: Function to benchmark
            input_data: Input tensor for inference
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Starting inference benchmark with {num_iterations} iterations")
        
        # Warmup
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = inference_fn(input_data)
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        start_time = time.perf_counter()
        
        for i in range(num_iterations):
            iter_start = time.perf_counter()
            
            with torch.no_grad():
                _ = inference_fn(input_data)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            iter_end = time.perf_counter()
            latencies.append((iter_end - iter_start) * 1000)  # Convert to ms
        
        total_time = time.perf_counter() - start_time
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
        throughput = num_iterations / total_time
        
        results = {
            'avg_latency_ms': avg_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'throughput_ops_per_sec': throughput,
            'total_time_sec': total_time,
            'num_iterations': num_iterations
        }
        
        logger.info(f"Benchmark completed: avg={avg_latency:.2f}ms, "
                   f"p95={p95_latency:.2f}ms, throughput={throughput:.1f} ops/sec")
        
        return results
    
    def identify_bottlenecks(self, 
                           target_latency_ms: float = 1.0) -> List[BottleneckAnalysis]:
        """
        Identify performance bottlenecks.
        
        Args:
            target_latency_ms: Target latency threshold
            
        Returns:
            List of identified bottlenecks
        """
        if not self.metrics_history:
            return []
        
        bottlenecks = []
        recent_metrics = self.metrics_history[-100:]  # Last 100 operations
        
        # Analyze latency
        avg_latency = sum(m.duration_ms for m in recent_metrics) / len(recent_metrics)
        if avg_latency > target_latency_ms:
            severity = min(avg_latency / target_latency_ms - 1, 1.0)
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type='latency',
                severity=severity,
                description=f'Average latency {avg_latency:.2f}ms exceeds target {target_latency_ms}ms',
                recommendations=[
                    'Enable model quantization',
                    'Use model pruning',
                    'Optimize batch size',
                    'Consider hardware acceleration'
                ]
            ))
        
        # Analyze CPU usage
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        if avg_cpu > 80:
            severity = (avg_cpu - 80) / 20
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type='cpu',
                severity=severity,
                description=f'High CPU usage: {avg_cpu:.1f}%',
                recommendations=[
                    'Reduce model complexity',
                    'Use GPU acceleration',
                    'Optimize preprocessing',
                    'Enable multi-threading'
                ]
            ))
        
        # Analyze memory usage
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        memory_percent = (avg_memory / (self.hardware_specs.total_memory_gb * 1024)) * 100
        if memory_percent > 80:
            severity = (memory_percent - 80) / 20
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type='memory',
                severity=severity,
                description=f'High memory usage: {memory_percent:.1f}%',
                recommendations=[
                    'Enable memory pooling',
                    'Reduce batch size',
                    'Use gradient checkpointing',
                    'Optimize data loading'
                ]
            ))
        
        # Analyze GPU usage if available
        gpu_metrics = [m for m in recent_metrics if m.gpu_usage_percent is not None]
        if gpu_metrics:
            avg_gpu = sum(m.gpu_usage_percent for m in gpu_metrics) / len(gpu_metrics)
            if avg_gpu < 50:  # Low GPU utilization
                severity = (50 - avg_gpu) / 50
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type='gpu',
                    severity=severity,
                    description=f'Low GPU utilization: {avg_gpu:.1f}%',
                    recommendations=[
                        'Increase batch size',
                        'Optimize data transfer',
                        'Use mixed precision',
                        'Profile GPU kernels'
                    ]
                ))
        
        return bottlenecks
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        recent_metrics = self.metrics_history[-100:]
        
        # Calculate statistics
        latencies = [m.duration_ms for m in recent_metrics]
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        
        summary = {
            'hardware_specs': {
                'cpu_count': self.hardware_specs.cpu_count,
                'cpu_freq_mhz': self.hardware_specs.cpu_freq_mhz,
                'total_memory_gb': self.hardware_specs.total_memory_gb,
                'gpu_name': self.hardware_specs.gpu_name,
                'gpu_memory_gb': self.hardware_specs.gpu_memory_gb
            },
            'performance_stats': {
                'avg_latency_ms': sum(latencies) / len(latencies),
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'p95_latency_ms': sorted(latencies)[int(0.95 * len(latencies))],
                'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
                'avg_memory_mb': sum(memory_usage) / len(memory_usage),
                'total_operations': len(self.metrics_history)
            }
        }
        
        # Add GPU stats if available
        gpu_metrics = [m for m in recent_metrics if m.gpu_usage_percent is not None]
        if gpu_metrics:
            gpu_usage = [m.gpu_usage_percent for m in gpu_metrics]
            gpu_memory = [m.gpu_memory_mb for m in gpu_metrics]
            summary['performance_stats'].update({
                'avg_gpu_usage': sum(gpu_usage) / len(gpu_usage),
                'avg_gpu_memory_mb': sum(gpu_memory) / len(gpu_memory)
            })
        
        return summary
    
    def save_profile_report(self, filepath: Path) -> None:
        """Save detailed profiling report to file."""
        report = {
            'timestamp': time.time(),
            'hardware_specs': self.hardware_specs.__dict__,
            'performance_summary': self.get_performance_summary(),
            'bottleneck_analysis': [b.__dict__ for b in self.identify_bottlenecks()],
            'metrics_history': [m.__dict__ for m in self.metrics_history[-1000:]]  # Last 1000
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Profile report saved to {filepath}")
    
    def start_continuous_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_performance)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        logger.info("Continuous monitoring started")
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        logger.info("Continuous monitoring stopped")
    
    def _monitor_performance(self) -> None:
        """Background performance monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_usage_mb = memory.used / (1024 * 1024)
                
                gpu_usage = None
                gpu_memory = None
                if torch.cuda.is_available():
                    try:
                        gpu_usage = torch.cuda.utilization()
                        gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    except:
                        pass
                
                # Create monitoring metrics
                metrics = PerformanceMetrics(
                    operation_name='system_monitoring',
                    duration_ms=self.monitoring_interval * 1000,
                    cpu_usage_percent=cpu_usage,
                    memory_usage_mb=memory_usage_mb,
                    gpu_usage_percent=gpu_usage,
                    gpu_memory_mb=gpu_memory
                )
                
                self.add_metrics(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                break