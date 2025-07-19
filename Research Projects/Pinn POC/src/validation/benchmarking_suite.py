"""
Comprehensive Benchmarking Suite for AV-PINO System.

This module integrates all validation components into a comprehensive evaluation
system with automated performance regression detection.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import json
from pathlib import Path
import psutil
import gc

from .generalization_testing import GeneralizationTester, GeneralizationReport
from .baseline_comparisons import BaselineComparator, ComparisonReport
from .physics_validation import PhysicsConsistencyValidator, PhysicsValidationReport
from ..inference.fault_classifier import FaultClassificationSystem
from ..physics.constraints import PhysicsConstraintLayer

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    accuracy: float
    inference_latency: float  # milliseconds
    throughput: float  # samples per second
    memory_usage: float  # MB
    cpu_usage: float  # percentage
    gpu_usage: float  # percentage (if available)
    model_size: int  # number of parameters
    
    # Physics-specific metrics
    physics_consistency_score: float
    constraint_violation_rate: float
    
    # Uncertainty metrics
    mean_confidence: float
    uncertainty_calibration_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'accuracy': self.accuracy,
            'inference_latency': self.inference_latency,
            'throughput': self.throughput,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'gpu_usage': self.gpu_usage,
            'model_size': self.model_size,
            'physics_consistency_score': self.physics_consistency_score,
            'constraint_violation_rate': self.constraint_violation_rate,
            'mean_confidence': self.mean_confidence,
            'uncertainty_calibration_score': self.uncertainty_calibration_score
        }


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report."""
    test_name: str
    performance_metrics: PerformanceMetrics
    generalization_report: Optional[GeneralizationReport]
    comparison_report: Optional[ComparisonReport]
    physics_validation_report: Optional[PhysicsValidationReport]
    regression_analysis: Dict[str, Any]
    hardware_info: Dict[str, Any]
    recommendations: List[str]
    timestamp: str
    
    def save_report(self, filepath: str):
        """Save comprehensive benchmark report."""
        report_data = {
            'test_name': self.test_name,
            'performance_metrics': self.performance_metrics.to_dict(),
            'generalization_report': (self.generalization_report.__dict__ 
                                    if self.generalization_report else None),
            'comparison_report': (self.comparison_report.__dict__ 
                                if self.comparison_report else None),
            'physics_validation_report': (self.physics_validation_report.__dict__ 
                                        if self.physics_validation_report else None),
            'regression_analysis': self.regression_analysis,
            'hardware_info': self.hardware_info,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Comprehensive benchmark report saved to {filepath}")


class PerformanceProfiler:
    """Profiles system performance during benchmarking."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        self.start_cpu = psutil.cpu_percent()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def profile_inference(self, model: FaultClassificationSystem, 
                         test_data: torch.Tensor, 
                         num_runs: int = 100) -> Dict[str, float]:
        """Profile inference performance."""
        model.classifier.eval()
        
        # Warmup runs
        for _ in range(10):
            with torch.no_grad():
                _ = model.predict(test_data[:1])
        
        # Timed runs
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                predictions = model.predict(test_data)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
        
        # Compute statistics
        mean_latency = np.mean(latencies)
        throughput = len(test_data) / (mean_latency / 1000)  # samples per second
        
        return {
            'mean_latency': mean_latency,
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'std_latency': np.std(latencies),
            'throughput': throughput
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = psutil.virtual_memory()
        
        usage = {
            'total_memory': memory_info.total / 1024 / 1024,  # MB
            'used_memory': memory_info.used / 1024 / 1024,    # MB
            'available_memory': memory_info.available / 1024 / 1024,  # MB
            'memory_percent': memory_info.percent
        }
        
        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            usage.update({
                'gpu_memory_allocated': gpu_memory,
                'gpu_memory_cached': gpu_memory_cached
            })
        
        return usage
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def get_gpu_usage(self) -> float:
        """Get GPU usage percentage (if available)."""
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return float(utilization.gpu)
            except ImportError:
                logger.warning("pynvml not available for GPU monitoring")
                return 0.0
        return 0.0
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'total_memory': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            'torch_version': torch.__version__
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_available': True,
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            })
        else:
            info['cuda_available'] = False
        
        return info


class ValidationPipeline:
    """Automated validation pipeline with regression detection."""
    
    def __init__(self, model: FaultClassificationSystem,
                 physics_constraints: Optional[PhysicsConstraintLayer] = None,
                 baseline_threshold: Optional[Dict[str, float]] = None):
        self.model = model
        self.physics_constraints = physics_constraints
        self.baseline_threshold = baseline_threshold or {
            'accuracy': 0.9,
            'inference_latency': 1.0,  # ms
            'physics_consistency': 0.8,
            'memory_usage': 1000.0  # MB
        }
        
        self.profiler = PerformanceProfiler()
        
        # Initialize validators
        self.generalization_tester = GeneralizationTester(model, physics_constraints)
        self.baseline_comparator = BaselineComparator(model)
        if physics_constraints:
            self.physics_validator = PhysicsConsistencyValidator(physics_constraints)
        else:
            self.physics_validator = None
    
    def run_validation_pipeline(self, 
                              test_data: Dict[str, torch.Tensor],
                              test_labels: Dict[str, torch.Tensor],
                              baseline_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                              coords: Optional[torch.Tensor] = None,
                              test_name: str = "Validation Pipeline"
                              ) -> BenchmarkReport:
        """Run complete validation pipeline."""
        logger.info(f"Starting validation pipeline: {test_name}")
        
        # Start profiling
        self.profiler.start_profiling()
        
        # Core performance metrics
        performance_metrics = self._measure_core_performance(test_data, test_labels)
        
        # Generalization testing
        generalization_report = None
        if len(test_data) > 1:  # Multiple test configurations
            try:
                baseline_features = list(test_data.values())[0]
                baseline_labels = list(test_labels.values())[0]
                generalization_report = self.generalization_tester.run_comprehensive_generalization_test(
                    (baseline_features, baseline_labels), test_data, test_labels, test_name
                )
            except Exception as e:
                logger.error(f"Generalization testing failed: {e}")
        
        # Baseline comparison
        comparison_report = None
        if baseline_data is not None:
            try:
                X_train, y_train = baseline_data
                # Use first test configuration for comparison
                X_test = test_data[list(test_data.keys())[0]].numpy()
                y_test = test_labels[list(test_labels.keys())[0]].numpy()
                X_test_torch = test_data[list(test_data.keys())[0]]
                y_test_torch = test_labels[list(test_labels.keys())[0]]
                
                comparison_report = self.baseline_comparator.run_comprehensive_comparison(
                    X_train, y_train, X_test, y_test, X_test_torch, y_test_torch
                )
            except Exception as e:
                logger.error(f"Baseline comparison failed: {e}")
        
        # Physics validation
        physics_report = None
        if self.physics_validator and coords is not None:
            try:
                # Use first test configuration for physics validation
                test_features = list(test_data.values())[0]
                predictions = self.model.predict(test_features)
                
                # Convert predictions to tensor format expected by physics validator
                pred_tensor = torch.stack([
                    torch.tensor([pred.confidence for pred in predictions])
                ]).T
                
                physics_report = self.physics_validator.validate_physics_consistency(
                    pred_tensor, test_features, coords, test_name
                )
            except Exception as e:
                logger.error(f"Physics validation failed: {e}")
        
        # Regression analysis
        regression_analysis = self._analyze_performance_regression(performance_metrics)
        
        # Hardware info
        hardware_info = self.profiler.get_hardware_info()
        
        # Generate recommendations
        recommendations = self._generate_pipeline_recommendations(
            performance_metrics, generalization_report, comparison_report, 
            physics_report, regression_analysis
        )
        
        # Create comprehensive report
        report = BenchmarkReport(
            test_name=test_name,
            performance_metrics=performance_metrics,
            generalization_report=generalization_report,
            comparison_report=comparison_report,
            physics_validation_report=physics_report,
            regression_analysis=regression_analysis,
            hardware_info=hardware_info,
            recommendations=recommendations,
            timestamp=str(np.datetime64('now'))
        )
        
        logger.info("Validation pipeline completed")
        return report
    
    def _measure_core_performance(self, test_data: Dict[str, torch.Tensor],
                                test_labels: Dict[str, torch.Tensor]) -> PerformanceMetrics:
        """Measure core performance metrics."""
        # Use first test configuration
        features = list(test_data.values())[0]
        labels = list(test_labels.values())[0]
        
        # Profile inference
        inference_profile = self.profiler.profile_inference(self.model, features)
        
        # Make predictions for accuracy
        predictions = self.model.predict(features)
        pred_labels = [pred.fault_type.value for pred in predictions]
        true_labels = [self.model.fault_mapper.index_to_type(idx.item()).value for idx in labels]
        
        # Compute accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Extract confidence and uncertainty
        confidences = [pred.confidence for pred in predictions]
        uncertainties = [pred.uncertainty for pred in predictions]
        physics_scores = [pred.physics_consistency for pred in predictions]
        
        # System metrics
        memory_usage = self.profiler.get_memory_usage()['used_memory']
        cpu_usage = self.profiler.get_cpu_usage()
        gpu_usage = self.profiler.get_gpu_usage()
        
        # Model size
        model_size = sum(p.numel() for p in self.model.classifier.parameters())
        
        return PerformanceMetrics(
            accuracy=accuracy,
            inference_latency=inference_profile['mean_latency'],
            throughput=inference_profile['throughput'],
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            gpu_usage=gpu_usage,
            model_size=model_size,
            physics_consistency_score=np.mean(physics_scores),
            constraint_violation_rate=np.mean([1.0 - score for score in physics_scores]),
            mean_confidence=np.mean(confidences),
            uncertainty_calibration_score=1.0 - np.std(uncertainties)  # Simplified calibration
        )
    
    def _analyze_performance_regression(self, current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze performance regression against baseline thresholds."""
        regression_analysis = {
            'regression_detected': False,
            'failing_metrics': [],
            'performance_changes': {}
        }
        
        # Check against thresholds
        if current_metrics.accuracy < self.baseline_threshold['accuracy']:
            regression_analysis['failing_metrics'].append('accuracy')
            regression_analysis['regression_detected'] = True
        
        if current_metrics.inference_latency > self.baseline_threshold['inference_latency']:
            regression_analysis['failing_metrics'].append('inference_latency')
            regression_analysis['regression_detected'] = True
        
        if current_metrics.physics_consistency_score < self.baseline_threshold['physics_consistency']:
            regression_analysis['failing_metrics'].append('physics_consistency')
            regression_analysis['regression_detected'] = True
        
        if current_metrics.memory_usage > self.baseline_threshold['memory_usage']:
            regression_analysis['failing_metrics'].append('memory_usage')
            regression_analysis['regression_detected'] = True
        
        # Compute performance changes (would compare against historical data in practice)
        regression_analysis['performance_changes'] = {
            'accuracy_vs_threshold': current_metrics.accuracy - self.baseline_threshold['accuracy'],
            'latency_vs_threshold': current_metrics.inference_latency - self.baseline_threshold['inference_latency'],
            'physics_vs_threshold': current_metrics.physics_consistency_score - self.baseline_threshold['physics_consistency'],
            'memory_vs_threshold': current_metrics.memory_usage - self.baseline_threshold['memory_usage']
        }
        
        return regression_analysis
    
    def _generate_pipeline_recommendations(self, 
                                         performance_metrics: PerformanceMetrics,
                                         generalization_report: Optional[GeneralizationReport],
                                         comparison_report: Optional[ComparisonReport],
                                         physics_report: Optional[PhysicsValidationReport],
                                         regression_analysis: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # Performance regression recommendations
        if regression_analysis['regression_detected']:
            recommendations.append(
                f"Performance regression detected in: {', '.join(regression_analysis['failing_metrics'])}. "
                "Review recent changes and consider model optimization."
            )
        
        # Accuracy recommendations
        if performance_metrics.accuracy < 0.9:
            recommendations.append(
                f"Accuracy ({performance_metrics.accuracy:.3f}) below target. "
                "Consider additional training data or model architecture improvements."
            )
        
        # Latency recommendations
        if performance_metrics.inference_latency > 1.0:
            recommendations.append(
                f"Inference latency ({performance_metrics.inference_latency:.2f}ms) exceeds 1ms target. "
                "Consider model quantization or pruning for edge deployment."
            )
        
        # Physics consistency recommendations
        if performance_metrics.physics_consistency_score < 0.8:
            recommendations.append(
                f"Physics consistency score ({performance_metrics.physics_consistency_score:.3f}) is low. "
                "Review constraint weights and PDE formulations."
            )
        
        # Memory usage recommendations
        if performance_metrics.memory_usage > 1000:
            recommendations.append(
                f"High memory usage ({performance_metrics.memory_usage:.1f}MB). "
                "Consider memory optimization techniques."
            )
        
        # Generalization recommendations
        if generalization_report and generalization_report.summary_statistics.get('generalization_robustness', 1.0) < 0.8:
            recommendations.append(
                "Poor generalization robustness detected. Consider domain adaptation or transfer learning."
            )
        
        # Baseline comparison recommendations
        if comparison_report and comparison_report.summary_statistics.get('accuracy_advantage', 0.0) < 0.05:
            recommendations.append(
                "Limited accuracy advantage over traditional methods. "
                "Emphasize physics consistency and uncertainty quantification benefits."
            )
        
        # Physics validation recommendations
        if physics_report and physics_report.overall_consistency_score < 0.7:
            recommendations.append(
                "Poor physics consistency detected. Review constraint implementation and numerical stability."
            )
        
        # Positive recommendations
        if not recommendations:
            recommendations.append(
                "All validation metrics meet targets. System performance is satisfactory."
            )
        
        return recommendations


class BenchmarkingSuite:
    """Main benchmarking suite that orchestrates all validation components."""
    
    def __init__(self, model: FaultClassificationSystem,
                 physics_constraints: Optional[PhysicsConstraintLayer] = None):
        self.model = model
        self.physics_constraints = physics_constraints
        self.validation_pipeline = ValidationPipeline(model, physics_constraints)
        
    def run_comprehensive_benchmark(self, 
                                  test_data: Dict[str, torch.Tensor],
                                  test_labels: Dict[str, torch.Tensor],
                                  baseline_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                  coords: Optional[torch.Tensor] = None,
                                  save_reports: bool = True,
                                  output_dir: str = "benchmark_results"
                                  ) -> BenchmarkReport:
        """Run comprehensive benchmarking suite."""
        logger.info("Starting comprehensive benchmarking suite...")
        
        # Create output directory
        if save_reports:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run validation pipeline
        report = self.validation_pipeline.run_validation_pipeline(
            test_data, test_labels, baseline_data, coords, "Comprehensive Benchmark"
        )
        
        # Save reports
        if save_reports:
            report.save_report(f"{output_dir}/comprehensive_benchmark_report.json")
            
            if report.generalization_report:
                report.generalization_report.save_report(f"{output_dir}/generalization_report.json")
            
            if report.comparison_report:
                report.comparison_report.save_report(f"{output_dir}/baseline_comparison_report.json")
            
            if report.physics_validation_report:
                report.physics_validation_report.save_report(f"{output_dir}/physics_validation_report.json")
        
        logger.info("Comprehensive benchmarking completed")
        return report
    
    def run_regression_test(self, test_data: Dict[str, torch.Tensor],
                          test_labels: Dict[str, torch.Tensor],
                          baseline_thresholds: Dict[str, float]) -> bool:
        """Run automated regression test."""
        logger.info("Running automated regression test...")
        
        # Update baseline thresholds
        self.validation_pipeline.baseline_threshold.update(baseline_thresholds)
        
        # Run validation
        report = self.validation_pipeline.run_validation_pipeline(
            test_data, test_labels, test_name="Regression Test"
        )
        
        # Check for regression
        regression_detected = report.regression_analysis['regression_detected']
        
        if regression_detected:
            logger.warning(f"Regression detected in: {report.regression_analysis['failing_metrics']}")
        else:
            logger.info("No regression detected. All metrics pass thresholds.")
        
        return not regression_detected