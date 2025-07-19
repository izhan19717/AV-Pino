"""
Validation and benchmarking module for AV-PINO system.

This module provides comprehensive validation, benchmarking, and generalization
testing capabilities for the motor fault diagnosis system.
"""

from .generalization_testing import (
    GeneralizationTester,
    CrossMotorValidator,
    OperatingConditionValidator,
    GeneralizationMetrics,
    GeneralizationReport
)

from .baseline_comparisons import (
    BaselineComparator,
    TraditionalMLBaseline,
    BaselineMetrics,
    ComparisonReport
)

from .physics_validation import (
    PhysicsConsistencyValidator,
    PDEResidualAnalyzer,
    PhysicsValidationMetrics,
    PhysicsValidationReport
)

from .benchmarking_suite import (
    BenchmarkingSuite,
    PerformanceProfiler,
    ValidationPipeline,
    BenchmarkReport,
    PerformanceMetrics
)

__all__ = [
    'GeneralizationTester',
    'CrossMotorValidator', 
    'OperatingConditionValidator',
    'GeneralizationMetrics',
    'GeneralizationReport',
    'BaselineComparator',
    'TraditionalMLBaseline',
    'BaselineMetrics',
    'ComparisonReport',
    'PhysicsConsistencyValidator',
    'PDEResidualAnalyzer',
    'PhysicsValidationMetrics',
    'PhysicsValidationReport',
    'BenchmarkingSuite',
    'PerformanceProfiler',
    'ValidationPipeline',
    'BenchmarkReport',
    'PerformanceMetrics'
]