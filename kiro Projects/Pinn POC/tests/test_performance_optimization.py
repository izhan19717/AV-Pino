"""
Unit tests for performance optimization components.

Tests memory management, hardware profiling, and performance benchmarking
functionality for real-time inference.
"""

import pytest
import torch
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.inference.memory_manager import MemoryManager, MemoryPool, MemoryStats
from src.inference.hardware_profiler import (
    HardwareProfiler, PerformanceMetrics, HardwareSpecs, BottleneckAnalysis
)


class TestMemoryPool:
    """Test memory pool functionality."""
    
    def test_memory_pool_creation(self):
        """Test memory pool initialization."""
        pool = MemoryPool(max_pool_size=50)
        assert pool.max_pool_size == 50
        assert len(pool.pools) == 0
    
    def test_tensor_get_and_return(self):
        """Test tensor allocation and return to pool."""
        pool = MemoryPool(max_pool_size=10)
        device = torch.device('cpu')
        dtype = torch.float32
        shape = (2, 3, 4)
        
        # Get tensor from empty pool (should create new)
        tensor1 = pool.get_tensor(shape, dtype, device)
        assert tensor1.shape == shape
        assert tensor1.dtype == dtype
        assert tensor1.device == device
        assert torch.all(tensor1 == 0)  # Should be zeroed
        
        # Return tensor to pool
        tensor1.fill_(1.0)  # Fill with non-zero values
        pool.return_tensor(tensor1)
        
        # Get tensor again (should reuse from pool)
        tensor2 = pool.get_tensor(shape, dtype, device)
        assert torch.all(tensor2 == 0)  # Should be zeroed again
    
    def test_pool_size_limit(self):
        """Test pool size limitation."""
        pool = MemoryPool(max_pool_size=2)
        device = torch.device('cpu')
        dtype = torch.float32
        shape = (2, 2)
        
        # Create and return multiple tensors
        tensors = []
        for i in range(5):
            tensor = pool.get_tensor(shape, dtype, device)
            tensors.append(tensor)
        
        # Return all tensors
        for tensor in tensors:
            pool.return_tensor(tensor)
        
        # Pool should only keep max_pool_size tensors
        key = (dtype, device)
        assert len(pool.pools[key]) == 2
    
    def test_clear_pool(self):
        """Test pool clearing."""
        pool = MemoryPool(max_pool_size=10)
        device = torch.device('cpu')
        dtype = torch.float32
        shape = (2, 2)
        
        # Add some tensors to pool
        for i in range(3):
            tensor = pool.get_tensor(shape, dtype, device)
            pool.return_tensor(tensor)
        
        # Clear pool
        pool.clear_pool()
        assert len(pool.pools) == 0


class TestMemoryManager:
    """Test memory manager functionality."""
    
    def test_memory_manager_creation(self):
        """Test memory manager initialization."""
        manager = MemoryManager(
            enable_pooling=True,
            pool_size=50,
            gc_threshold=0.7
        )
        
        assert manager.enable_pooling is True
        assert manager.gc_threshold == 0.7
        assert manager.memory_pool is not None
    
    def test_memory_stats(self):
        """Test memory statistics collection."""
        manager = MemoryManager()
        stats = manager.get_memory_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.total_memory_mb > 0
        assert stats.used_memory_mb > 0
        assert stats.available_memory_mb > 0
    
    def test_managed_inference_context(self):
        """Test managed inference context manager."""
        manager = MemoryManager()
        
        with manager.managed_inference() as mgr:
            assert mgr is manager
            # Should have optimized settings during context
        
        # Settings should be restored after context
    
    def test_tensor_management(self):
        """Test tensor allocation and management."""
        manager = MemoryManager(enable_pooling=True)
        shape = (10, 20)
        
        # Get tensor
        tensor = manager.get_tensor(shape)
        assert tensor.shape == shape
        assert torch.all(tensor == 0)
        
        # Return tensor
        tensor.fill_(1.0)
        manager.return_tensor(tensor)
        
        # Get another tensor (should be reused and zeroed)
        tensor2 = manager.get_tensor(shape)
        assert torch.all(tensor2 == 0)
    
    def test_memory_pressure_detection(self):
        """Test memory pressure detection."""
        manager = MemoryManager(gc_threshold=0.1)  # Very low threshold
        
        # Should detect high memory usage
        is_pressure = manager.check_memory_pressure()
        # Result depends on actual system memory usage
        assert isinstance(is_pressure, bool)
    
    def test_force_cleanup(self):
        """Test forced memory cleanup."""
        manager = MemoryManager(enable_pooling=True)
        
        # Add some tensors to pool
        for i in range(5):
            tensor = manager.get_tensor((10, 10))
            manager.return_tensor(tensor)
        
        # Force cleanup
        manager.force_cleanup()
        
        # Pool should be cleared
        if manager.memory_pool:
            assert len(manager.memory_pool.pools) == 0
    
    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        manager = MemoryManager(monitoring_interval=0.1)
        
        # Start monitoring
        manager.start_monitoring()
        assert manager._monitoring_active is True
        
        # Wait a bit for some stats to be collected
        time.sleep(0.3)
        
        # Stop monitoring
        manager.stop_monitoring()
        assert manager._monitoring_active is False
        
        # Should have collected some stats
        history = manager.get_memory_history()
        assert len(history) > 0
    
    def test_memory_summary(self):
        """Test memory summary generation."""
        manager = MemoryManager()
        summary = manager.get_memory_summary()
        
        assert 'current_memory_mb' in summary
        assert 'current_memory_percent' in summary
        assert 'available_memory_mb' in summary
        assert 'pooling_enabled' in summary
        assert 'gc_threshold' in summary


class TestHardwareProfiler:
    """Test hardware profiler functionality."""
    
    def test_hardware_specs_detection(self):
        """Test hardware specification detection."""
        profiler = HardwareProfiler()
        specs = profiler.hardware_specs
        
        assert isinstance(specs, HardwareSpecs)
        assert specs.cpu_count > 0
        assert specs.total_memory_gb > 0
    
    def test_performance_profiling(self):
        """Test operation profiling."""
        profiler = HardwareProfiler()
        
        # Profile a simple operation
        with profiler.profile_operation('test_operation'):
            time.sleep(0.01)  # Simulate work
        
        # Should have recorded metrics
        assert len(profiler.metrics_history) == 1
        metrics = profiler.metrics_history[0]
        assert metrics.operation_name == 'test_operation'
        assert metrics.duration_ms >= 10  # At least 10ms
        assert metrics.cpu_usage_percent >= 0
        assert metrics.memory_usage_mb > 0
    
    def test_inference_benchmarking(self):
        """Test inference benchmarking."""
        profiler = HardwareProfiler()
        
        # Create a simple inference function
        def dummy_inference(x):
            return torch.relu(x)
        
        input_data = torch.randn(1, 10)
        
        # Run benchmark
        results = profiler.benchmark_inference(
            dummy_inference, 
            input_data, 
            num_iterations=10,
            warmup_iterations=2
        )
        
        # Check results
        assert 'avg_latency_ms' in results
        assert 'min_latency_ms' in results
        assert 'max_latency_ms' in results
        assert 'p95_latency_ms' in results
        assert 'p99_latency_ms' in results
        assert 'throughput_ops_per_sec' in results
        assert results['num_iterations'] == 10
        assert results['avg_latency_ms'] > 0
        assert results['throughput_ops_per_sec'] > 0
    
    def test_bottleneck_identification(self):
        """Test bottleneck identification."""
        profiler = HardwareProfiler()
        
        # Add some high-latency metrics
        for i in range(10):
            metrics = PerformanceMetrics(
                operation_name='slow_operation',
                duration_ms=5.0,  # 5ms - above 1ms target
                cpu_usage_percent=90.0,  # High CPU usage
                memory_usage_mb=1000.0
            )
            profiler.add_metrics(metrics)
        
        # Identify bottlenecks
        bottlenecks = profiler.identify_bottlenecks(target_latency_ms=1.0)
        
        # Should identify latency and CPU bottlenecks
        bottleneck_types = [b.bottleneck_type for b in bottlenecks]
        assert 'latency' in bottleneck_types
        assert 'cpu' in bottleneck_types
        
        # Check bottleneck details
        for bottleneck in bottlenecks:
            assert isinstance(bottleneck, BottleneckAnalysis)
            assert 0 <= bottleneck.severity <= 1
            assert len(bottleneck.recommendations) > 0
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        profiler = HardwareProfiler()
        
        # Add some metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                operation_name=f'operation_{i}',
                duration_ms=1.0 + i * 0.5,
                cpu_usage_percent=50.0 + i * 5,
                memory_usage_mb=100.0 + i * 10
            )
            profiler.add_metrics(metrics)
        
        # Get summary
        summary = profiler.get_performance_summary()
        
        assert 'hardware_specs' in summary
        assert 'performance_stats' in summary
        
        stats = summary['performance_stats']
        assert 'avg_latency_ms' in stats
        assert 'avg_cpu_usage' in stats
        assert 'avg_memory_mb' in stats
        assert 'total_operations' in stats
        assert stats['total_operations'] == 5
    
    def test_profile_report_saving(self):
        """Test saving profile report to file."""
        profiler = HardwareProfiler()
        
        # Add some metrics
        metrics = PerformanceMetrics(
            operation_name='test_operation',
            duration_ms=2.0,
            cpu_usage_percent=60.0,
            memory_usage_mb=200.0
        )
        profiler.add_metrics(metrics)
        
        # Save report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            profiler.save_profile_report(filepath)
            
            # Verify file was created and contains expected data
            assert filepath.exists()
            
            with open(filepath, 'r') as f:
                report = json.load(f)
            
            assert 'timestamp' in report
            assert 'hardware_specs' in report
            assert 'performance_summary' in report
            assert 'bottleneck_analysis' in report
            assert 'metrics_history' in report
            
        finally:
            if filepath.exists():
                filepath.unlink()
    
    def test_continuous_monitoring(self):
        """Test continuous performance monitoring."""
        profiler = HardwareProfiler(
            enable_continuous_monitoring=True,
            monitoring_interval=0.1
        )
        
        # Start monitoring
        profiler.start_continuous_monitoring()
        assert profiler._monitoring_active is True
        
        # Wait for some monitoring data
        time.sleep(0.3)
        
        # Stop monitoring
        profiler.stop_continuous_monitoring()
        assert profiler._monitoring_active is False
        
        # Should have collected monitoring metrics
        monitoring_metrics = [
            m for m in profiler.metrics_history 
            if m.operation_name == 'system_monitoring'
        ]
        assert len(monitoring_metrics) > 0


class TestPerformanceIntegration:
    """Integration tests for performance optimization components."""
    
    def test_memory_manager_with_profiler(self):
        """Test memory manager integration with profiler."""
        memory_manager = MemoryManager(enable_pooling=True)
        profiler = HardwareProfiler()
        
        # Profile memory operations
        with profiler.profile_operation('memory_allocation'):
            with memory_manager.managed_inference():
                # Allocate some tensors
                tensors = []
                for i in range(10):
                    tensor = memory_manager.get_tensor((100, 100))
                    tensors.append(tensor)
                
                # Return tensors
                for tensor in tensors:
                    memory_manager.return_tensor(tensor)
        
        # Check that operation was profiled
        assert len(profiler.metrics_history) == 1
        metrics = profiler.metrics_history[0]
        assert metrics.operation_name == 'memory_allocation'
        assert metrics.duration_ms > 0
    
    def test_end_to_end_performance_optimization(self):
        """Test complete performance optimization workflow."""
        # Initialize components
        memory_manager = MemoryManager(
            enable_pooling=True,
            gc_threshold=0.8
        )
        profiler = HardwareProfiler()
        
        # Simulate inference workload
        def simulate_inference():
            with memory_manager.managed_inference():
                with profiler.profile_operation('inference'):
                    # Simulate model inference
                    input_tensor = memory_manager.get_tensor((1, 224, 224, 3))
                    
                    # Simulate processing
                    result = torch.relu(input_tensor)
                    time.sleep(0.001)  # Simulate computation
                    
                    # Return tensor
                    memory_manager.return_tensor(input_tensor)
                    memory_manager.return_tensor(result)
        
        # Run multiple inference operations
        for i in range(5):
            simulate_inference()
        
        # Analyze performance
        summary = profiler.get_performance_summary()
        bottlenecks = profiler.identify_bottlenecks(target_latency_ms=1.0)
        memory_summary = memory_manager.get_memory_summary()
        
        # Verify results
        assert summary['performance_stats']['total_operations'] == 5
        assert 'avg_latency_ms' in summary['performance_stats']
        assert 'current_memory_mb' in memory_summary
        
        # Should have some performance data
        assert len(profiler.metrics_history) == 5


if __name__ == '__main__':
    pytest.main([__file__])