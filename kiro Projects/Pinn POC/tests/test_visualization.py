"""
Unit tests for visualization components.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
import tempfile
import os

from src.visualization.prediction_visualizer import (
    PredictionVisualizer, PredictionResult, UncertaintyMetrics
)
from src.visualization.physics_analysis_visualizer import (
    PhysicsAnalysisVisualizer, PhysicsResiduals, PDEConstraintViolations
)
from src.visualization.diagnostic_visualizer import (
    DiagnosticVisualizer, ModelDiagnostics, PerformanceMetrics
)
from src.visualization.comparative_visualizer import (
    ComparativeVisualizer, MethodResults, BenchmarkResults
)


class TestPredictionVisualizer:
    """Test cases for PredictionVisualizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = PredictionVisualizer()
        
        # Create sample prediction results
        n_samples = 100
        self.prediction_result = PredictionResult(
            predictions=np.random.randint(0, 4, n_samples),
            uncertainties=np.random.exponential(0.1, n_samples),
            confidence_intervals=np.random.rand(n_samples, 2),
            fault_types=['Normal', 'Inner Race', 'Outer Race', 'Ball'],
            timestamps=np.linspace(0, 10, n_samples),
            true_labels=np.random.randint(0, 4, n_samples)
        )
        
        self.uncertainty_metrics = UncertaintyMetrics(
            epistemic_uncertainty=np.random.exponential(0.05, n_samples),
            aleatoric_uncertainty=np.random.exponential(0.08, n_samples),
            total_uncertainty=np.random.exponential(0.1, n_samples),
            confidence_scores=np.random.beta(8, 2, n_samples)
        )
    
    def test_prediction_visualizer_initialization(self):
        """Test PredictionVisualizer initialization."""
        assert self.visualizer.figsize == (12, 8)
        assert len(self.visualizer.fault_colors) >= 4
        assert 'Normal' in self.visualizer.fault_colors
    
    def test_plot_prediction_timeline(self):
        """Test prediction timeline plotting."""
        fig = self.visualizer.plot_prediction_timeline(self.prediction_result)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Main plot and uncertainty plot
        
        # Check that axes have proper labels
        axes = fig.axes
        assert axes[0].get_ylabel() == 'Fault Type Index'
        assert axes[1].get_xlabel() == 'Time'
        assert axes[1].get_ylabel() == 'Uncertainty'
        
        plt.close(fig)
    
    def test_plot_uncertainty_decomposition(self):
        """Test uncertainty decomposition plotting."""
        timestamps = np.linspace(0, 10, 100)
        fig = self.visualizer.plot_uncertainty_decomposition(
            self.uncertainty_metrics, timestamps
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot grid
        
        # Check subplot titles
        expected_titles = [
            'Epistemic Uncertainty (Model)',
            'Aleatoric Uncertainty (Data)', 
            'Total Uncertainty',
            'Confidence Scores'
        ]
        
        for ax, expected_title in zip(fig.axes, expected_titles):
            assert expected_title in ax.get_title()
        
        plt.close(fig)
    
    def test_plot_prediction_confidence_matrix(self):
        """Test prediction confidence matrix plotting."""
        predictions = self.prediction_result.predictions
        confidences = np.random.beta(8, 2, len(predictions))
        
        fig = self.visualizer.plot_prediction_confidence_matrix(
            predictions, confidences, self.prediction_result.fault_types
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # Heatmap and histogram
        
        plt.close(fig)
    
    def test_plot_reliability_assessment(self):
        """Test reliability assessment plotting."""
        fig = self.visualizer.plot_reliability_assessment(
            self.prediction_result.predictions,
            self.prediction_result.uncertainties,
            self.prediction_result.true_labels
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot grid
        
        plt.close(fig)
    
    def test_plot_fault_type_analysis(self):
        """Test fault type analysis plotting."""
        fig = self.visualizer.plot_fault_type_analysis(
            self.prediction_result.predictions,
            self.prediction_result.uncertainties,
            self.prediction_result.fault_types,
            self.prediction_result.true_labels
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4  # 2x2 subplot grid (may include colorbar)
        
        plt.close(fig)


class TestPhysicsAnalysisVisualizer:
    """Test cases for PhysicsAnalysisVisualizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = PhysicsAnalysisVisualizer()
        
        # Create sample physics residuals
        n_samples = 200
        self.physics_residuals = PhysicsResiduals(
            maxwell_residual=np.random.exponential(0.01, n_samples),
            heat_equation_residual=np.random.exponential(0.005, n_samples),
            structural_dynamics_residual=np.random.exponential(0.02, n_samples),
            coupling_residual=np.random.exponential(0.008, n_samples),
            timestamps=np.linspace(0, 20, n_samples),
            constraint_weights={'Maxwell': 1.0, 'Heat': 0.8, 'Structural': 1.2, 'Coupling': 0.9}
        )
        
        self.constraint_violations = PDEConstraintViolations(
            violation_magnitudes={
                'Maxwell': np.random.exponential(0.01, n_samples),
                'Heat': np.random.exponential(0.005, n_samples),
                'Structural': np.random.exponential(0.02, n_samples)
            },
            violation_frequencies={
                'Maxwell': np.random.poisson(2, n_samples),
                'Heat': np.random.poisson(1, n_samples),
                'Structural': np.random.poisson(3, n_samples)
            },
            critical_violations={
                'Maxwell': [10, 25, 67, 89],
                'Heat': [15, 45],
                'Structural': [5, 30, 55, 78, 95]
            },
            conservation_errors={
                'Energy': np.random.normal(0, 0.01, n_samples),
                'Momentum': np.random.normal(0, 0.005, n_samples),
                'Charge': np.random.normal(0, 0.002, n_samples)
            }
        )
    
    def test_physics_analysis_visualizer_initialization(self):
        """Test PhysicsAnalysisVisualizer initialization."""
        assert self.visualizer.figsize == (14, 10)
        assert len(self.visualizer.constraint_colors) >= 4
        assert 'Maxwell' in self.visualizer.constraint_colors
    
    def test_plot_pde_residuals_overview(self):
        """Test PDE residuals overview plotting."""
        fig = self.visualizer.plot_pde_residuals_overview(self.physics_residuals)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot grid
        
        # Check that all axes have log scale
        for ax in fig.axes:
            assert ax.get_yscale() == 'log'
        
        plt.close(fig)
    
    def test_plot_constraint_violation_analysis(self):
        """Test constraint violation analysis plotting."""
        fig = self.visualizer.plot_constraint_violation_analysis(self.constraint_violations)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 6  # 2x3 subplot grid
        
        plt.close(fig)
    
    def test_plot_conservation_law_analysis(self):
        """Test conservation law analysis plotting."""
        fig = self.visualizer.plot_conservation_law_analysis(self.constraint_violations)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot grid
        
        plt.close(fig)
    
    def test_plot_physics_consistency_evolution(self):
        """Test physics consistency evolution plotting."""
        fig = self.visualizer.plot_physics_consistency_evolution(
            self.physics_residuals, window_size=50
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # 3x1 subplot grid
        
        plt.close(fig)
    
    def test_plot_multiphysics_coupling_analysis(self):
        """Test multi-physics coupling analysis plotting."""
        n_samples = 200
        em_field = np.random.normal(0, 1, n_samples)
        thermal_field = np.random.normal(20, 5, n_samples)
        mechanical_disp = np.random.normal(0, 0.1, n_samples)
        timestamps = np.linspace(0, 20, n_samples)
        
        fig = self.visualizer.plot_multiphysics_coupling_analysis(
            em_field, thermal_field, mechanical_disp, timestamps
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 6  # 2x3 subplot grid
        
        plt.close(fig)
    
    def test_plot_pde_residual_statistics(self):
        """Test PDE residual statistics plotting."""
        fig = self.visualizer.plot_pde_residual_statistics(self.physics_residuals)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4  # 2x2 subplot grid (may include colorbar)
        
        plt.close(fig)


class TestDiagnosticVisualizer:
    """Test cases for DiagnosticVisualizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = DiagnosticVisualizer()
        
        # Create sample model diagnostics
        self.model_diagnostics = ModelDiagnostics(
            layer_activations={
                'encoder': np.random.normal(0, 1, (100, 64)),
                'processor': np.random.normal(0, 0.5, (100, 128)),
                'decoder': np.random.normal(0, 0.8, (100, 32))
            },
            gradient_norms={
                'encoder': np.random.exponential(0.1, 50),
                'processor': np.random.exponential(0.05, 50),
                'decoder': np.random.exponential(0.08, 50)
            },
            weight_distributions={
                'encoder': np.random.normal(0, 0.1, (64, 32)),
                'processor': np.random.normal(0, 0.05, (128, 64)),
                'decoder': np.random.normal(0, 0.08, (32, 4))
            },
            loss_components={
                'data_loss': np.random.exponential(0.5, 100),
                'physics_loss': np.random.exponential(0.1, 100),
                'consistency_loss': np.random.exponential(0.05, 100)
            },
            training_metrics={
                'accuracy': list(np.random.beta(8, 2, 100)),
                'loss': list(np.random.exponential(0.3, 100))
            }
        )
        
        self.performance_metrics = PerformanceMetrics(
            inference_times=np.random.exponential(0.0008, 1000),  # ~0.8ms average
            memory_usage=np.random.normal(512*1024*1024, 50*1024*1024, 100),  # ~512MB
            accuracy_over_time=np.random.beta(8, 2, 100),
            loss_over_time=np.random.exponential(0.3, 100),
            learning_rates=np.logspace(-4, -2, 100),
            epochs=np.arange(100)
        )
    
    def test_diagnostic_visualizer_initialization(self):
        """Test DiagnosticVisualizer initialization."""
        assert self.visualizer.figsize == (14, 10)
        assert len(self.visualizer.layer_colors) == 10
    
    def test_plot_training_diagnostics(self):
        """Test training diagnostics plotting."""
        fig = self.visualizer.plot_training_diagnostics(self.performance_metrics)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 6  # 2x3 subplot grid
        
        # Check that loss plot has log scale
        loss_ax = fig.axes[0]
        assert loss_ax.get_yscale() == 'log'
        
        plt.close(fig)
    
    def test_plot_model_architecture_analysis(self):
        """Test model architecture analysis plotting."""
        fig = self.visualizer.plot_model_architecture_analysis(self.model_diagnostics)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot grid
        
        plt.close(fig)
    
    def test_plot_gradient_flow_analysis(self):
        """Test gradient flow analysis plotting."""
        fig = self.visualizer.plot_gradient_flow_analysis(self.model_diagnostics)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot grid
        
        plt.close(fig)
    
    def test_plot_activation_analysis(self):
        """Test activation analysis plotting."""
        fig = self.visualizer.plot_activation_analysis(self.model_diagnostics)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 6  # 2x3 subplot grid
        
        plt.close(fig)
    
    def test_plot_performance_bottleneck_analysis(self):
        """Test performance bottleneck analysis plotting."""
        layer_times = {
            'encoder': np.random.exponential(0.0002, 100),
            'processor': np.random.exponential(0.0004, 100),
            'decoder': np.random.exponential(0.0001, 100)
        }
        
        fig = self.visualizer.plot_performance_bottleneck_analysis(
            self.performance_metrics, layer_times
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot grid
        
        plt.close(fig)


class TestComparativeVisualizer:
    """Test cases for ComparativeVisualizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = ComparativeVisualizer()
        
        # Create sample method results
        fault_types = ['Normal', 'Inner Race', 'Outer Race', 'Ball']
        n_fault_types = len(fault_types)
        
        self.benchmark_results = BenchmarkResults(
            methods=[
                MethodResults(
                    method_name='AV-PINO',
                    accuracy=0.95,
                    precision=np.random.beta(9, 1, n_fault_types),
                    recall=np.random.beta(9, 1, n_fault_types),
                    f1_score=np.random.beta(9, 1, n_fault_types),
                    inference_time=0.0008,
                    training_time=3600,
                    memory_usage=512*1024*1024,
                    physics_consistency=0.92,
                    uncertainty_quality=0.88
                ),
                MethodResults(
                    method_name='Traditional ML',
                    accuracy=0.87,
                    precision=np.random.beta(7, 2, n_fault_types),
                    recall=np.random.beta(7, 2, n_fault_types),
                    f1_score=np.random.beta(7, 2, n_fault_types),
                    inference_time=0.002,
                    training_time=1800,
                    memory_usage=256*1024*1024,
                    physics_consistency=None,
                    uncertainty_quality=None
                ),
                MethodResults(
                    method_name='Deep Learning',
                    accuracy=0.91,
                    precision=np.random.beta(8, 1.5, n_fault_types),
                    recall=np.random.beta(8, 1.5, n_fault_types),
                    f1_score=np.random.beta(8, 1.5, n_fault_types),
                    inference_time=0.005,
                    training_time=7200,
                    memory_usage=1024*1024*1024,
                    physics_consistency=None,
                    uncertainty_quality=0.75
                )
            ],
            fault_types=fault_types,
            dataset_name='CWRU',
            test_size=1000
        )
    
    def test_comparative_visualizer_initialization(self):
        """Test ComparativeVisualizer initialization."""
        assert self.visualizer.figsize == (14, 10)
        assert len(self.visualizer.method_colors) >= 3
        assert 'AV-PINO' in self.visualizer.method_colors
    
    def test_plot_method_comparison_overview(self):
        """Test method comparison overview plotting."""
        fig = self.visualizer.plot_method_comparison_overview(self.benchmark_results)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 6  # 2x3 subplot grid
        
        plt.close(fig)
    
    def test_plot_performance_radar_chart(self):
        """Test performance radar chart plotting."""
        fig = self.visualizer.plot_performance_radar_chart(self.benchmark_results)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1  # Single polar plot
        
        # Check that it's a polar plot
        assert fig.axes[0].name == 'polar'
        
        plt.close(fig)
    
    def test_plot_precision_recall_comparison(self):
        """Test precision-recall comparison plotting."""
        fig = self.visualizer.plot_precision_recall_comparison(self.benchmark_results)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Precision and recall plots
        
        plt.close(fig)
    
    def test_plot_efficiency_analysis(self):
        """Test efficiency analysis plotting."""
        fig = self.visualizer.plot_efficiency_analysis(self.benchmark_results)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot grid
        
        plt.close(fig)
    
    def test_plot_statistical_significance_analysis(self):
        """Test statistical significance analysis plotting."""
        confidence_intervals = {
            'AV-PINO': (0.93, 0.97),
            'Traditional ML': (0.85, 0.89),
            'Deep Learning': (0.89, 0.93)
        }
        
        fig = self.visualizer.plot_statistical_significance_analysis(
            self.benchmark_results, confidence_intervals
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4  # 2x2 subplot grid (may include colorbar)
        
        plt.close(fig)


class TestVisualizationManager:
    """Test cases for VisualizationManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.visualization.visualization_manager import (
            VisualizationManager, VisualizationConfig, ComprehensiveResults
        )
        
        self.config = VisualizationConfig(
            output_dir="test_outputs",
            save_figures=False,  # Don't save during tests
            show_figures=False   # Don't display during tests
        )
        self.manager = VisualizationManager(self.config)
        
        # Create comprehensive test results
        n_samples = 50
        self.comprehensive_results = ComprehensiveResults(
            prediction_results=PredictionResult(
                predictions=np.random.randint(0, 4, n_samples),
                uncertainties=np.random.exponential(0.1, n_samples),
                confidence_intervals=np.random.rand(n_samples, 2),
                fault_types=['Normal', 'Inner Race', 'Outer Race', 'Ball'],
                timestamps=np.linspace(0, 10, n_samples),
                true_labels=np.random.randint(0, 4, n_samples)
            ),
            uncertainty_metrics=UncertaintyMetrics(
                epistemic_uncertainty=np.random.exponential(0.05, n_samples),
                aleatoric_uncertainty=np.random.exponential(0.08, n_samples),
                total_uncertainty=np.random.exponential(0.1, n_samples),
                confidence_scores=np.random.beta(8, 2, n_samples)
            ),
            physics_residuals=PhysicsResiduals(
                maxwell_residual=np.random.exponential(0.01, n_samples),
                heat_equation_residual=np.random.exponential(0.005, n_samples),
                structural_dynamics_residual=np.random.exponential(0.02, n_samples),
                coupling_residual=np.random.exponential(0.008, n_samples),
                timestamps=np.linspace(0, 10, n_samples)
            ),
            performance_metrics=PerformanceMetrics(
                inference_times=np.random.exponential(0.0008, 100),
                memory_usage=np.random.normal(512*1024*1024, 50*1024*1024, 50),
                accuracy_over_time=np.random.beta(8, 2, 50),
                loss_over_time=np.random.exponential(0.3, 50),
                learning_rates=np.logspace(-4, -2, 50),
                epochs=np.arange(50)
            ),
            metadata={"test_run": True, "version": "1.0"}
        )
    
    def test_visualization_manager_initialization(self):
        """Test VisualizationManager initialization."""
        assert self.manager.config.output_dir == "test_outputs"
        assert hasattr(self.manager, 'prediction_viz')
        assert hasattr(self.manager, 'physics_viz')
        assert hasattr(self.manager, 'diagnostic_viz')
        assert hasattr(self.manager, 'comparative_viz')
    
    def test_generate_prediction_plots(self):
        """Test prediction plots generation."""
        plots = self.manager._generate_prediction_plots(
            self.comprehensive_results.prediction_results,
            self.comprehensive_results.uncertainty_metrics,
            "test_predictions"
        )
        
        assert isinstance(plots, dict)
        assert len(plots) >= 4  # Should have multiple plot types
        assert "prediction_timeline" in plots
        assert "uncertainty_decomposition" in plots
        assert "confidence_matrix" in plots
        assert "reliability_assessment" in plots
    
    def test_generate_physics_plots(self):
        """Test physics plots generation."""
        plots = self.manager._generate_physics_plots(
            self.comprehensive_results.physics_residuals,
            None,  # No constraint violations for this test
            "test_physics"
        )
        
        assert isinstance(plots, dict)
        assert len(plots) >= 4  # Should have multiple plot types
        assert "pde_residuals" in plots
        assert "consistency_evolution" in plots
        assert "residual_statistics" in plots
        assert "multiphysics_coupling" in plots
    
    def test_generate_diagnostic_plots(self):
        """Test diagnostic plots generation."""
        plots = self.manager._generate_diagnostic_plots(
            None,  # No model diagnostics for this test
            self.comprehensive_results.performance_metrics,
            "test_diagnostics"
        )
        
        assert isinstance(plots, dict)
        assert len(plots) >= 1  # Should have performance plots
        assert "training_diagnostics" in plots
        assert "bottleneck_analysis" in plots
    
    def test_generate_summary_dashboard(self):
        """Test summary dashboard generation."""
        dashboard_file = self.manager._generate_summary_dashboard(
            self.comprehensive_results,
            "test_dashboard"
        )
        
        assert isinstance(dashboard_file, str)
        assert "test_dashboard" in dashboard_file
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation."""
        plot_files = self.manager.generate_comprehensive_report(
            self.comprehensive_results,
            "test_report"
        )
        
        assert isinstance(plot_files, dict)
        assert len(plot_files) > 0
        assert "dashboard" in plot_files
    
    def test_export_analysis_summary(self):
        """Test analysis summary export."""
        summary_file = self.manager.export_analysis_summary(
            self.comprehensive_results,
            "test_summary"
        )
        
        assert isinstance(summary_file, str)
        assert "test_summary" in summary_file
    
    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation."""
        dashboard_file = self.manager.create_interactive_dashboard(
            self.comprehensive_results,
            "test_interactive"
        )
        
        assert isinstance(dashboard_file, str)
        assert "test_interactive" in dashboard_file
        assert dashboard_file.endswith(".html")


class TestVisualizationIntegration:
    """Integration tests for visualization components."""
    
    def test_all_visualizers_work_together(self):
        """Test that all visualizers can be used together without conflicts."""
        pred_viz = PredictionVisualizer()
        physics_viz = PhysicsAnalysisVisualizer()
        diag_viz = DiagnosticVisualizer()
        comp_viz = ComparativeVisualizer()
        
        # Test that they all have different default colors/styles
        assert pred_viz.figsize == (12, 8)
        assert physics_viz.figsize == (14, 10)
        assert diag_viz.figsize == (14, 10)
        assert comp_viz.figsize == (14, 10)
    
    def test_matplotlib_backend_compatibility(self):
        """Test compatibility with different matplotlib backends."""
        # Test that plots can be created without display
        with patch('matplotlib.pyplot.show'):
            pred_viz = PredictionVisualizer()
            
            # Create minimal test data
            result = PredictionResult(
                predictions=np.array([0, 1, 2]),
                uncertainties=np.array([0.1, 0.2, 0.15]),
                confidence_intervals=np.array([[0.05, 0.15], [0.15, 0.25], [0.1, 0.2]]),
                fault_types=['A', 'B', 'C'],
                timestamps=np.array([0, 1, 2])
            )
            
            fig = pred_viz.plot_prediction_timeline(result)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_memory_cleanup(self):
        """Test that figures are properly cleaned up to prevent memory leaks."""
        initial_figs = len(plt.get_fignums())
        
        # Create and close multiple figures
        for _ in range(5):
            pred_viz = PredictionVisualizer()
            result = PredictionResult(
                predictions=np.array([0, 1]),
                uncertainties=np.array([0.1, 0.2]),
                confidence_intervals=np.array([[0.05, 0.15], [0.15, 0.25]]),
                fault_types=['A', 'B'],
                timestamps=np.array([0, 1])
            )
            
            fig = pred_viz.plot_prediction_timeline(result)
            plt.close(fig)
        
        final_figs = len(plt.get_fignums())
        assert final_figs == initial_figs  # No figure leaks
    
    def test_end_to_end_visualization_workflow(self):
        """Test complete end-to-end visualization workflow."""
        from src.visualization import VisualizationManager, VisualizationConfig, ComprehensiveResults
        
        # Create test configuration
        config = VisualizationConfig(
            output_dir="test_workflow",
            save_figures=False,
            show_figures=False
        )
        
        # Initialize manager
        manager = VisualizationManager(config)
        
        # Create minimal test data
        n_samples = 20
        results = ComprehensiveResults(
            prediction_results=PredictionResult(
                predictions=np.random.randint(0, 3, n_samples),
                uncertainties=np.random.exponential(0.1, n_samples),
                confidence_intervals=np.random.rand(n_samples, 2),
                fault_types=['Normal', 'Fault1', 'Fault2'],
                timestamps=np.linspace(0, 5, n_samples),
                true_labels=np.random.randint(0, 3, n_samples)
            ),
            physics_residuals=PhysicsResiduals(
                maxwell_residual=np.random.exponential(0.01, n_samples),
                heat_equation_residual=np.random.exponential(0.005, n_samples),
                structural_dynamics_residual=np.random.exponential(0.02, n_samples),
                coupling_residual=np.random.exponential(0.008, n_samples),
                timestamps=np.linspace(0, 5, n_samples)
            )
        )
        
        # Generate comprehensive report
        plot_files = manager.generate_comprehensive_report(results, "workflow_test")
        
        # Verify report generation
        assert isinstance(plot_files, dict)
        assert len(plot_files) > 0
        
        # Generate summary
        summary_file = manager.export_analysis_summary(results, "workflow_summary")
        assert isinstance(summary_file, str)


if __name__ == '__main__':
    pytest.main([__file__])