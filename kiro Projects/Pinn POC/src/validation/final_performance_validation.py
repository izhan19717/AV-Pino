"""
Final Performance Validation for AV-PINO Motor Fault Diagnosis System

This module implements comprehensive end-to-end validation including:
- >90% fault classification accuracy validation
- <1ms inference latency validation  
- Physics consistency validation across all test scenarios
- Complete system integration testing
"""

import time
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from ..data.cwru_loader import CWRUDataLoader
from ..data.preprocessor import DataPreprocessor
from ..physics.agt_no_architecture import AGTNO
from ..inference.realtime_inference import RealTimeInference
from ..inference.fault_classifier import FaultClassificationSystem
from ..physics.uncertainty_integration import UncertaintyIntegration
from ..validation.physics_validation import PhysicsValidator
from ..validation.benchmarking_suite import BenchmarkingSuite
from ..reporting.technical_report_generator import TechnicalReportGenerator
from ..config.config_manager import ConfigManager


@dataclass
class PerformanceMetrics:
    """Container for comprehensive performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_latency_ms: float
    physics_consistency_score: float
    uncertainty_calibration_score: float
    memory_usage_mb: float
    throughput_samples_per_sec: float


@dataclass
class ValidationResults:
    """Container for complete validation results"""
    overall_metrics: PerformanceMetrics
    per_fault_metrics: Dict[str, PerformanceMetrics]
    physics_validation_results: Dict[str, float]
    benchmark_comparisons: Dict[str, float]
    edge_hardware_results: Dict[str, Any]
    requirements_compliance: Dict[str, bool]


class FinalPerformanceValidator:
    """
    Comprehensive performance validation for the complete AV-PINO system
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the final performance validator"""
        self.config = ConfigManager(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.data_loader = CWRUDataLoader()
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.realtime_inference = None
        self.fault_classifier = None
        self.uncertainty_module = None
        self.physics_validator = PhysicsValidator()
        self.benchmarking_suite = BenchmarkingSuite()
        self.report_generator = TechnicalReportGenerator()
        
        # Performance tracking
        self.validation_results = None
        
    def setup_complete_system(self) -> None:
        """Initialize and integrate all system components"""
        self.logger.info("Setting up complete AV-PINO system...")
        
        # Load and initialize the trained model
        model_config = self.config.get_model_config()
        self.model = AGTNO(
            input_dim=model_config['input_dim'],
            output_dim=model_config['output_dim'],
            n_modes=model_config.get('modes', 32),
            hidden_dim=model_config.get('width', 256),
            n_classes=model_config.get('n_classes', 4)
        )
        
        # Load trained weights if available
        model_path = self.config.get_model_path()
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))
            self.logger.info(f"Loaded trained model from {model_path}")
        else:
            self.logger.warning("No trained model found, using random initialization")
        
        # Initialize inference components
        self.realtime_inference = RealTimeInference(self.model)
        self.fault_classifier = FaultClassificationSystem(input_dim=model_config['input_dim'])
        self.uncertainty_module = UncertaintyIntegration(self.model)
        
        self.logger.info("Complete system setup finished")
    
    def load_test_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess the complete CWRU test dataset"""
        self.logger.info("Loading CWRU test dataset...")
        
        # Load all fault types for comprehensive testing
        fault_types = ['normal', 'ball_fault', 'inner_race_fault', 'outer_race_fault']
        all_data = []
        all_labels = []
        
        for fault_type in fault_types:
            data, labels = self.data_loader.load_fault_data(fault_type)
            processed_data = self.preprocessor.preprocess_signals(data)
            
            all_data.append(processed_data)
            all_labels.extend(labels)
        
        test_data = torch.cat(all_data, dim=0)
        test_labels = torch.tensor(all_labels)
        
        self.logger.info(f"Loaded {len(test_data)} test samples across {len(fault_types)} fault types")
        return test_data, test_labels
    
    def validate_classification_accuracy(self, test_data: torch.Tensor, 
                                       test_labels: torch.Tensor) -> Dict[str, float]:
        """Validate >90% fault classification accuracy requirement"""
        self.logger.info("Validating fault classification accuracy...")
        
        # Run inference on test dataset
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for batch_start in range(0, len(test_data), 32):
                batch_end = min(batch_start + 32, len(test_data))
                batch_data = test_data[batch_start:batch_end]
                
                # Get predictions with uncertainty
                pred, uncertainty = self.uncertainty_module.predict_with_uncertainty(batch_data)
                predictions.append(pred)
                uncertainties.append(uncertainty)
        
        predictions = torch.cat(predictions, dim=0)
        uncertainties = torch.cat(uncertainties, dim=0)
        
        # Calculate accuracy metrics
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == test_labels).float().mean().item()
        
        # Calculate per-class metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels.numpy(), predicted_classes.numpy(), average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'meets_90_percent_requirement': accuracy >= 0.90
        }
        
        self.logger.info(f"Classification accuracy: {accuracy:.4f} ({'PASS' if accuracy >= 0.90 else 'FAIL'})")
        return metrics
    
    def validate_inference_latency(self, test_data: torch.Tensor) -> Dict[str, float]:
        """Validate <1ms inference latency requirement"""
        self.logger.info("Validating inference latency...")
        
        # Warm up the model
        warmup_samples = test_data[:10]
        for _ in range(5):
            _ = self.realtime_inference.predict_realtime(warmup_samples[0:1])
        
        # Measure inference latency on single samples
        latencies = []
        num_samples = min(100, len(test_data))
        
        for i in range(num_samples):
            sample = test_data[i:i+1]
            
            start_time = time.perf_counter()
            _ = self.realtime_inference.predict_realtime(sample)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        metrics = {
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'p95_latency_ms': p95_latency,
            'meets_1ms_requirement': avg_latency <= 1.0
        }
        
        self.logger.info(f"Average inference latency: {avg_latency:.4f}ms ({'PASS' if avg_latency <= 1.0 else 'FAIL'})")
        return metrics
    
    def validate_physics_consistency(self, test_data: torch.Tensor) -> Dict[str, float]:
        """Validate physics consistency across all test scenarios"""
        self.logger.info("Validating physics consistency...")
        
        # Run physics validation on test samples
        consistency_scores = []
        constraint_violations = []
        
        for i in range(0, len(test_data), 10):  # Sample every 10th for efficiency
            sample = test_data[i:i+1]
            
            # Get model prediction and physics residuals
            with torch.no_grad():
                prediction, physics_residuals = self.model(sample)
            
            # Validate physics constraints
            consistency_score = self.physics_validator.validate_physics_consistency(
                prediction, physics_residuals
            )
            consistency_scores.append(consistency_score)
            
            # Check for constraint violations
            violations = self.physics_validator.check_constraint_violations(physics_residuals)
            constraint_violations.extend(violations)
        
        avg_consistency = np.mean(consistency_scores)
        violation_rate = len(constraint_violations) / len(consistency_scores)
        
        metrics = {
            'avg_physics_consistency': avg_consistency,
            'constraint_violation_rate': violation_rate,
            'maxwell_consistency': np.mean([s.get('maxwell', 0) for s in consistency_scores]),
            'thermal_consistency': np.mean([s.get('thermal', 0) for s in consistency_scores]),
            'mechanical_consistency': np.mean([s.get('mechanical', 0) for s in consistency_scores]),
            'meets_consistency_requirement': avg_consistency >= 0.85
        }
        
        self.logger.info(f"Physics consistency score: {avg_consistency:.4f}")
        return metrics
    
    def run_comprehensive_benchmarks(self, test_data: torch.Tensor, 
                                   test_labels: torch.Tensor) -> Dict[str, float]:
        """Run comprehensive benchmarks against baseline methods"""
        self.logger.info("Running comprehensive benchmarks...")
        
        # Compare against baseline methods
        benchmark_results = self.benchmarking_suite.run_all_benchmarks(
            test_data, test_labels, self.model
        )
        
        return benchmark_results
    
    def validate_edge_hardware_performance(self, test_data: torch.Tensor) -> Dict[str, Any]:
        """Validate performance on edge hardware constraints"""
        self.logger.info("Validating edge hardware performance...")
        
        # Simulate edge hardware constraints
        edge_results = {}
        
        # Memory usage validation
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run inference batch
        batch_size = 32
        sample_batch = test_data[:batch_size]
        
        with torch.no_grad():
            _ = self.realtime_inference.predict_realtime(sample_batch)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # Throughput measurement
        start_time = time.time()
        num_processed = 0
        
        for i in range(0, min(1000, len(test_data)), batch_size):
            batch = test_data[i:i+batch_size]
            with torch.no_grad():
                _ = self.realtime_inference.predict_realtime(batch)
            num_processed += len(batch)
        
        end_time = time.time()
        throughput = num_processed / (end_time - start_time)
        
        edge_results = {
            'memory_usage_mb': memory_usage,
            'throughput_samples_per_sec': throughput,
            'meets_memory_constraints': memory_usage <= 512,  # 512MB limit
            'meets_throughput_requirements': throughput >= 1000  # 1000 samples/sec
        }
        
        return edge_results
    
    def run_complete_validation(self) -> ValidationResults:
        """Run complete end-to-end validation of the AV-PINO system"""
        self.logger.info("Starting complete AV-PINO system validation...")
        
        # Setup the complete system
        self.setup_complete_system()
        
        # Load test dataset
        test_data, test_labels = self.load_test_dataset()
        
        # Run all validation components
        accuracy_metrics = self.validate_classification_accuracy(test_data, test_labels)
        latency_metrics = self.validate_inference_latency(test_data)
        physics_metrics = self.validate_physics_consistency(test_data)
        benchmark_results = self.run_comprehensive_benchmarks(test_data, test_labels)
        edge_results = self.validate_edge_hardware_performance(test_data)
        
        # Compile overall metrics
        overall_metrics = PerformanceMetrics(
            accuracy=accuracy_metrics['accuracy'],
            precision=accuracy_metrics['precision'],
            recall=accuracy_metrics['recall'],
            f1_score=accuracy_metrics['f1_score'],
            inference_latency_ms=latency_metrics['avg_latency_ms'],
            physics_consistency_score=physics_metrics['avg_physics_consistency'],
            uncertainty_calibration_score=0.85,  # Placeholder - would be computed from uncertainty module
            memory_usage_mb=edge_results['memory_usage_mb'],
            throughput_samples_per_sec=edge_results['throughput_samples_per_sec']
        )
        
        # Check requirements compliance
        requirements_compliance = {
            'accuracy_90_percent': accuracy_metrics['meets_90_percent_requirement'],
            'latency_1ms': latency_metrics['meets_1ms_requirement'],
            'physics_consistency': physics_metrics['meets_consistency_requirement'],
            'memory_constraints': edge_results['meets_memory_constraints'],
            'throughput_requirements': edge_results['meets_throughput_requirements']
        }
        
        # Compile complete results
        self.validation_results = ValidationResults(
            overall_metrics=overall_metrics,
            per_fault_metrics={},  # Would be populated with per-fault analysis
            physics_validation_results=physics_metrics,
            benchmark_comparisons=benchmark_results,
            edge_hardware_results=edge_results,
            requirements_compliance=requirements_compliance
        )
        
        # Log summary
        self.log_validation_summary()
        
        return self.validation_results
    
    def log_validation_summary(self) -> None:
        """Log comprehensive validation summary"""
        if not self.validation_results:
            return
        
        self.logger.info("=== AV-PINO SYSTEM VALIDATION SUMMARY ===")
        
        metrics = self.validation_results.overall_metrics
        compliance = self.validation_results.requirements_compliance
        
        self.logger.info(f"Classification Accuracy: {metrics.accuracy:.4f} ({'✓' if compliance['accuracy_90_percent'] else '✗'})")
        self.logger.info(f"Inference Latency: {metrics.inference_latency_ms:.4f}ms ({'✓' if compliance['latency_1ms'] else '✗'})")
        self.logger.info(f"Physics Consistency: {metrics.physics_consistency_score:.4f} ({'✓' if compliance['physics_consistency'] else '✗'})")
        self.logger.info(f"Memory Usage: {metrics.memory_usage_mb:.2f}MB ({'✓' if compliance['memory_constraints'] else '✗'})")
        self.logger.info(f"Throughput: {metrics.throughput_samples_per_sec:.2f} samples/sec ({'✓' if compliance['throughput_requirements'] else '✗'})")
        
        # Overall system status
        all_requirements_met = all(compliance.values())
        self.logger.info(f"Overall System Status: {'PASS' if all_requirements_met else 'FAIL'}")
        
        if not all_requirements_met:
            failed_requirements = [req for req, passed in compliance.items() if not passed]
            self.logger.warning(f"Failed requirements: {', '.join(failed_requirements)}")
    
    def generate_performance_report(self, output_path: str = "validation_report.md") -> None:
        """Generate comprehensive technical report of validation results"""
        if not self.validation_results:
            self.logger.error("No validation results available for report generation")
            return
        
        self.report_generator.generate_validation_report(
            self.validation_results, output_path
        )
        self.logger.info(f"Validation report generated: {output_path}")


def main():
    """Main function for running final performance validation"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run complete validation
    validator = FinalPerformanceValidator()
    results = validator.run_complete_validation()
    
    # Generate technical report
    validator.generate_performance_report("final_validation_report.md")
    
    return results


if __name__ == "__main__":
    main()