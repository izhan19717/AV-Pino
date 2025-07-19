"""
Integration tests for complete real-time inference system.

Tests the integration of all optimization components including model optimization,
inference engine, memory management, hardware profiling, and adaptive control.
"""

import pytest
import torch
import torch.nn as nn
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.inference import (
    RealTimeInference, HardwareConstraints, AdaptiveConfig,
    InferenceEngine, InferenceConfig,
    ModelOptimizer, OptimizationConfig,
    MemoryManager, HardwareProfiler
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class TestModelOptimizer:
    """Test model optimizer functionality."""
    
    def test_model_optimizer_creation(self):
        """Test model optimizer initialization."""
        config = OptimizationConfig(
            enable_quantization=True,
            quantization_bits=8,
            enable_pruning=True,
            pruning_ratio=0.3
        )
        
        optimizer = ModelOptimizer(config)
        assert optimizer.config == config
        assert optimizer.quantizer is not None
        assert optimizer.pruner is not None
    
    def test_model_optimization(self):
        """Test complete model optimization."""
        model = SimpleTestModel()
        config = OptimizationConfig(
            enable_quantization=True,
            quantization_bits=8,
            enable_pruning=True,
            pruning_ratio=0.2,
            enable_fusion=False  # Disable fusion for testing
        )
        
        optimizer = ModelOptimizer(config)
        sample_data = torch.randn(10, 10)
        
        optimized_model, results = optimizer.optimize_model(
            model, 
            calibration_data=sample_data
        )
        
        # Check results
        assert optimized_model is not None
        assert results.original_size_mb > 0
        assert results.optimized_size_mb > 0
        assert results.compression_ratio >= 1.0
        assert 'quantization' in results.optimization_techniques
        assert 'pruning' in results.optimization_techniques
    
    def test_target_latency_optimization(self):
        """Test optimization for target latency."""
        model = SimpleTestModel()
        config = OptimizationConfig(target_latency_ms=1.0)
        optimizer = ModelOptimizer(config)
        
        sample_data = torch.randn(1, 10)
        optimized_model = optimizer.optimize_for_target_latency(
            model, 
            target_latency_ms=1.0,
            calibration_data=sample_data
        )
        
        assert optimized_model is not None


class TestInferenceEngine:
    """Test inference engine functionality."""
    
    def test_inference_engine_creation(self):
        """Test inference engine initialization."""
        model = SimpleTestModel()
        config = InferenceConfig(
            target_latency_ms=1.0,
            enable_optimization=True
        )
        
        engine = InferenceEngine(model, config)
        assert engine.config == config
        assert engine.optimized_model is not None
        assert engine.device is not None
    
    def test_single_prediction(self):
        """Test single prediction."""
        model = SimpleTestModel()
        config = InferenceConfig(enable_optimization=False)  # Disable for faster testing
        
        engine = InferenceEngine(model, config)
        input_data = torch.randn(1, 10)
        
        result = engine.predict(input_data)
        
        assert result.predictions is not None
        assert result.predictions.shape[0] == 1
        assert result.latency_ms > 0
        assert result.batch_size == 1
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        model = SimpleTestModel()
        config = InferenceConfig(
            enable_optimization=False,
            max_batch_size=8
        )
        
        engine = InferenceEngine(model, config)
        batch_data = torch.randn(4, 10)
        
        result = engine.predict_batch(batch_data)
        
        assert result.predictions.shape[0] == 4
        assert result.batch_size == 4
        assert result.latency_ms > 0
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking."""
        model = SimpleTestModel()
        config = InferenceConfig(enable_optimization=False)
        
        engine = InferenceEngine(model, config)
        sample_input = torch.randn(1, 10)
        
        metrics = engine.benchmark_performance(
            sample_input,
            num_iterations=10,
            warmup_iterations=2
        )
        
        assert 'avg_latency_ms' in metrics
        assert 'throughput_ops_per_sec' in metrics
        assert 'p95_latency_ms' in metrics
        assert metrics['avg_latency_ms'] > 0
        assert metrics['throughput_ops_per_sec'] > 0
    
    def test_latency_optimization(self):
        """Test latency optimization."""
        model = SimpleTestModel()
        config = InferenceConfig(enable_optimization=True)
        
        engine = InferenceEngine(model, config)
        
        # Try to optimize for very aggressive target
        success = engine.optimize_for_latency(0.5)  # 0.5ms target
        
        # Should return boolean indicating success/failure
        assert isinstance(success, bool)


class TestRealTimeInference:
    """Test complete real-time inference system."""
    
    def test_realtime_system_creation(self):
        """Test real-time system initialization."""
        model = SimpleTestModel()
        constraints = HardwareConstraints(
            target_latency_ms=1.0,
            max_memory_mb=1000.0,
            max_cpu_usage_percent=80.0
        )
        
        system = RealTimeInference(model, constraints)
        
        assert system.model is model
        assert system.hardware_constraints == constraints
        assert system.memory_manager is not None
        assert system.profiler is not None
        assert system.inference_engine is not None
    
    def test_system_start_stop(self):
        """Test system startup and shutdown."""
        model = SimpleTestModel()
        constraints = HardwareConstraints(target_latency_ms=1.0)
        
        system = RealTimeInference(model, constraints)
        
        # Start system
        system.start_system()
        assert system.monitoring_active is True
        assert system.processing_active is True
        
        # Wait a bit for threads to start
        time.sleep(0.1)
        
        # Stop system
        system.stop_system()
        assert system.monitoring_active is False
        assert system.processing_active is False
    
    def test_realtime_prediction(self):
        """Test real-time prediction."""
        model = SimpleTestModel()
        constraints = HardwareConstraints(target_latency_ms=10.0)  # Relaxed for testing
        
        system = RealTimeInference(model, constraints)
        input_data = torch.randn(1, 10)
        
        result = system.predict(input_data)
        
        assert result.predictions is not None
        assert result.latency_ms > 0
        assert result.batch_size == 1
    
    def test_async_prediction(self):
        """Test asynchronous prediction."""
        model = SimpleTestModel()
        constraints = HardwareConstraints(target_latency_ms=10.0)
        
        system = RealTimeInference(model, constraints)
        system.start_system()
        
        try:
            input_data = torch.randn(1, 10)
            result_received = threading.Event()
            received_result = None
            
            def callback(result):
                nonlocal received_result
                received_result = result
                result_received.set()
            
            # Submit async request
            request_id = system.predict_async(input_data, callback)
            assert isinstance(request_id, str)
            
            # Wait for result
            success = result_received.wait(timeout=5.0)
            assert success, "Async prediction timed out"
            assert received_result is not None
            assert received_result.predictions is not None
            
        finally:
            system.stop_system()
    
    def test_system_status_monitoring(self):
        """Test system status monitoring."""
        model = SimpleTestModel()
        constraints = HardwareConstraints(target_latency_ms=1.0)
        
        system = RealTimeInference(model, constraints)
        system.start_system()
        
        try:
            # Wait for monitoring to collect some data
            time.sleep(1.5)
            
            status = system.get_system_status()
            
            assert hasattr(status, 'is_healthy')
            assert hasattr(status, 'current_latency_ms')
            assert hasattr(status, 'memory_usage_percent')
            assert hasattr(status, 'last_update')
            assert status.last_update > 0
            
        finally:
            system.stop_system()
    
    def test_performance_report(self):
        """Test performance report generation."""
        model = SimpleTestModel()
        constraints = HardwareConstraints(target_latency_ms=1.0)
        
        system = RealTimeInference(model, constraints)
        
        # Make some predictions to generate data
        input_data = torch.randn(1, 10)
        for _ in range(3):
            system.predict(input_data)
        
        report = system.get_performance_report()
        
        assert 'system_status' in report
        assert 'hardware_constraints' in report
        assert 'inference_stats' in report
        assert 'memory_summary' in report
        assert 'profiler_summary' in report
    
    def test_system_benchmarking(self):
        """Test system benchmarking."""
        model = SimpleTestModel()
        constraints = HardwareConstraints(target_latency_ms=10.0)  # Relaxed for testing
        
        system = RealTimeInference(model, constraints)
        sample_input = torch.randn(1, 10)
        
        # Run short benchmark
        results = system.benchmark_system(sample_input, duration_sec=2.0)
        
        assert 'duration_sec' in results
        assert 'total_requests' in results
        assert 'avg_latency_ms' in results
        assert 'throughput_ops_per_sec' in results
        assert results['total_requests'] > 0
        assert results['avg_latency_ms'] > 0
        assert results['throughput_ops_per_sec'] > 0
    
    def test_system_state_saving(self):
        """Test system state saving and loading."""
        model = SimpleTestModel()
        constraints = HardwareConstraints(target_latency_ms=1.0)
        
        system = RealTimeInference(model, constraints)
        
        # Make some predictions to generate state
        input_data = torch.randn(1, 10)
        system.predict(input_data)
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            system.save_system_state(filepath)
            
            # Verify files were created
            assert filepath.exists()
            model_path = filepath.with_suffix('.pth')
            assert model_path.exists()
            
            # Verify state file content
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            assert 'timestamp' in state
            assert 'hardware_constraints' in state
            assert 'performance_report' in state
            
        finally:
            if filepath.exists():
                filepath.unlink()
            model_path = filepath.with_suffix('.pth')
            if model_path.exists():
                model_path.unlink()


class TestAdaptiveControl:
    """Test adaptive control functionality."""
    
    def test_adaptive_config(self):
        """Test adaptive configuration."""
        config = AdaptiveConfig(
            enable_adaptive_batching=True,
            enable_adaptive_precision=True,
            adaptation_interval_sec=5.0
        )
        
        assert config.enable_adaptive_batching is True
        assert config.enable_adaptive_precision is True
        assert config.adaptation_interval_sec == 5.0
    
    def test_hardware_constraints(self):
        """Test hardware constraints."""
        constraints = HardwareConstraints(
            max_memory_mb=2000.0,
            max_cpu_usage_percent=75.0,
            target_latency_ms=0.5,
            min_throughput_ops_per_sec=200.0
        )
        
        assert constraints.max_memory_mb == 2000.0
        assert constraints.max_cpu_usage_percent == 75.0
        assert constraints.target_latency_ms == 0.5
        assert constraints.min_throughput_ops_per_sec == 200.0


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_complete_inference_pipeline(self):
        """Test complete inference pipeline from model to result."""
        # Create model
        model = SimpleTestModel(input_dim=20, output_dim=3)
        
        # Setup constraints
        constraints = HardwareConstraints(
            target_latency_ms=5.0,  # Relaxed for testing
            max_memory_mb=2000.0,
            max_cpu_usage_percent=80.0
        )
        
        # Setup adaptive config
        adaptive_config = AdaptiveConfig(
            enable_adaptive_optimization=True,
            adaptation_interval_sec=1.0
        )
        
        # Create system
        system = RealTimeInference(
            model=model,
            hardware_constraints=constraints,
            adaptive_config=adaptive_config
        )
        
        # Start system
        system.start_system()
        
        try:
            # Test various input sizes
            test_inputs = [
                torch.randn(1, 20),   # Single sample
                torch.randn(4, 20),   # Small batch
                torch.randn(8, 20),   # Medium batch
            ]
            
            results = []
            for input_data in test_inputs:
                result = system.predict(input_data, return_uncertainty=False)
                results.append(result)
                
                # Verify result structure
                assert result.predictions is not None
                assert result.predictions.shape[0] == input_data.shape[0]
                assert result.predictions.shape[1] == 3  # Output dim
                assert result.latency_ms > 0
                assert result.batch_size == input_data.shape[0]
            
            # Check that system adapted over time
            time.sleep(2.0)  # Allow adaptation
            
            # Get final performance report
            final_report = system.get_performance_report()
            
            assert final_report['system_status']['is_healthy'] is not None
            assert final_report['inference_stats']['total_inferences'] >= len(test_inputs)
            
            # Verify latency requirements
            avg_latency = final_report['inference_stats'].get('average_latency_ms', 0)
            if avg_latency > 0:
                # Should be reasonable for test system
                assert avg_latency < 100.0  # 100ms is very generous for testing
            
        finally:
            system.stop_system()
    
    def test_stress_testing(self):
        """Test system under stress conditions."""
        model = SimpleTestModel()
        constraints = HardwareConstraints(
            target_latency_ms=10.0,  # Relaxed for stress test
            max_cpu_usage_percent=90.0
        )
        
        system = RealTimeInference(model, constraints)
        system.start_system()
        
        try:
            # Submit many concurrent requests
            input_data = torch.randn(1, 10)
            results = []
            
            # Stress test with rapid requests
            start_time = time.time()
            while time.time() - start_time < 3.0:  # 3 second stress test
                result = system.predict(input_data)
                results.append(result)
                time.sleep(0.01)  # Small delay between requests
            
            # Verify system handled stress
            assert len(results) > 10  # Should have processed multiple requests
            
            # Check system health
            status = system.get_system_status()
            # System might be stressed but should still be functional
            
            # Verify all results are valid
            for result in results:
                assert result.predictions is not None
                assert result.latency_ms > 0
            
        finally:
            system.stop_system()
    
    def test_memory_constraint_handling(self):
        """Test handling of memory constraints."""
        model = SimpleTestModel()
        
        # Set very tight memory constraints
        constraints = HardwareConstraints(
            target_latency_ms=5.0,
            max_memory_mb=100.0,  # Very tight constraint
            max_cpu_usage_percent=80.0
        )
        
        system = RealTimeInference(model, constraints)
        
        # Test constraint checking
        constraint_check = system._check_hardware_constraints()
        assert isinstance(constraint_check, bool)
        
        # Make predictions and monitor memory
        input_data = torch.randn(1, 10)
        
        for i in range(5):
            result = system.predict(input_data)
            assert result.predictions is not None
            
            # Check if memory management is working
            memory_summary = system.memory_manager.get_memory_summary()
            assert 'current_memory_mb' in memory_summary


if __name__ == '__main__':
    pytest.main([__file__])