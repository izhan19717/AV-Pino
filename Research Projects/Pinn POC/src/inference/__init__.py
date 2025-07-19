"""
Real-time inference engine for AV-PINO motor fault diagnosis.

This module provides optimized inference capabilities with <1ms latency
for edge deployment, including model optimization, memory management,
hardware profiling, and fault classification with uncertainty quantification.
"""

from .memory_manager import MemoryManager
from .hardware_profiler import HardwareProfiler
from .model_optimizer import ModelOptimizer, OptimizationConfig
from .inference_engine import InferenceEngine, InferenceConfig
from .realtime_inference import RealTimeInference, HardwareConstraints, AdaptiveConfig
from .fault_classifier import (
    FaultType, FaultPrediction, FaultTypeMapper, 
    UncertaintyAwareClassifier, FaultClassificationSystem,
    create_cwru_fault_classifier
)
from .classification_validator import (
    ClassificationValidator, ValidationResults, evaluate_fault_classification
)
from .fault_classification_integration import (
    IntegratedFaultDiagnosisSystem, CWRUFaultDiagnosisSystem,
    create_cwru_diagnosis_system, diagnose_motor_fault,
    evaluate_fault_diagnosis_system
)

__all__ = [
    'MemoryManager',
    'HardwareProfiler', 
    'ModelOptimizer',
    'OptimizationConfig',
    'InferenceEngine',
    'InferenceConfig',
    'RealTimeInference',
    'HardwareConstraints',
    'AdaptiveConfig',
    'FaultType',
    'FaultPrediction',
    'FaultTypeMapper',
    'UncertaintyAwareClassifier',
    'FaultClassificationSystem',
    'create_cwru_fault_classifier',
    'ClassificationValidator',
    'ValidationResults',
    'evaluate_fault_classification',
    'IntegratedFaultDiagnosisSystem',
    'CWRUFaultDiagnosisSystem',
    'create_cwru_diagnosis_system',
    'diagnose_motor_fault',
    'evaluate_fault_diagnosis_system'
]