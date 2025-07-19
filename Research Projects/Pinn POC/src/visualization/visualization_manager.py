"""
Comprehensive visualization manager for AV-PINO motor fault diagnosis system.

This module provides a unified interface for all visualization capabilities,
integrating prediction visualization, physics analysis, diagnostic tools,
and comparative analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import json

from .prediction_visualizer import (
    PredictionVisualizer, PredictionResult, UncertaintyMetrics
)
from .physics_analysis_visualizer import (
    PhysicsAnalysisVisualizer, PhysicsResiduals, PDEConstraintViolations
)
from .diagnostic_visualizer import (
    DiagnosticVisualizer, ModelDiagnostics, PerformanceMetrics
)
from .comparative_visualizer import (
    ComparativeVisualizer, MethodResults, BenchmarkResults
)

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    output_dir: str = "visualization_outputs"
    figure_format: str = "png"
    figure_dpi: int = 300
    style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    save_figures: bool = True
    show_figures: bool = False

@dataclass
class ComprehensiveResults:
    """Container for all analysis results."""
    prediction_results: Optional[PredictionResult] = None
    uncertainty_metrics: Optional[UncertaintyMetrics] = None
    physics_residuals: Optional[PhysicsResiduals] = None
    constraint_violations: Optional[PDEConstraintViolations] = None
    model_diagnostics: Optional[ModelDiagnostics] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    benchmark_results: Optional[BenchmarkResults] = None
    metadata: Optional[Dict[str, Any]] = None

class VisualizationManager:
    """
    Comprehensive visualization manager for AV-PINO system.
    
    Provides unified interface for all visualization capabilities including:
    - Prediction visualization with confidence intervals and uncertainty displays
    - Physics consistency analysis plots showing PDE residuals
    - Diagnostic visualization tools for model debugging and analysis
    - Comparative performance visualization between methods
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualization manager.
        
        Args:
            config: Visualization configuration settings
        """
        self.config = config or VisualizationConfig()
        
        # Initialize individual visualizers
        self.prediction_viz = PredictionVisualizer(style=self.config.style)
        self.physics_viz = PhysicsAnalysisVisualizer(style=self.config.style)
        self.diagnostic_viz = DiagnosticVisualizer(style=self.config.style)
        self.comparative_viz = ComparativeVisualizer(style=self.config.style)
        
        # Set up output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
        
    def generate_comprehensive_report(self, 
                                    results: ComprehensiveResults,
                                    report_name: str = "av_pino_analysis_report") -> Dict[str, str]:
        """
        Generate comprehensive visualization report with all available analyses.
        
        Args:
            results: ComprehensiveResults containing all analysis data
            report_name: Base name for the report files
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        plot_files = {}
        
        # 1. Prediction Analysis
        if results.prediction_results is not None:
            plot_files.update(self._generate_prediction_plots(
                results.prediction_results, 
                results.uncertainty_metrics,
                f"{report_name}_predictions"
            ))
        
        # 2. Physics Analysis
        if results.physics_residuals is not None:
            plot_files.update(self._generate_physics_plots(
                results.physics_residuals,
                results.constraint_violations,
                f"{report_name}_physics"
            ))
        
        # 3. Diagnostic Analysis
        if results.model_diagnostics is not None or results.performance_metrics is not None:
            plot_files.update(self._generate_diagnostic_plots(
                results.model_diagnostics,
                results.performance_metrics,
                f"{report_name}_diagnostics"
            ))
        
        # 4. Comparative Analysis
        if results.benchmark_results is not None:
            plot_files.update(self._generate_comparative_plots(
                results.benchmark_results,
                f"{report_name}_comparison"
            ))
        
        # 5. Generate summary dashboard
        if len(plot_files) > 0:
            dashboard_file = self._generate_summary_dashboard(
                results, f"{report_name}_dashboard"
            )
            plot_files["dashboard"] = dashboard_file
        
        # Save metadata
        self._save_report_metadata(results, plot_files, report_name)
        
        return plot_files
    
    def _generate_prediction_plots(self, 
                                 prediction_results: PredictionResult,
                                 uncertainty_metrics: Optional[UncertaintyMetrics],
                                 base_name: str) -> Dict[str, str]:
        """Generate all prediction-related plots."""
        plots = {}
        
        # Prediction timeline
        fig = self.prediction_viz.plot_prediction_timeline(prediction_results)
        plots["prediction_timeline"] = self._save_figure(fig, f"{base_name}_timeline")
        
        # Uncertainty decomposition (if available)
        if uncertainty_metrics is not None:
            fig = self.prediction_viz.plot_uncertainty_decomposition(
                uncertainty_metrics, prediction_results.timestamps
            )
            plots["uncertainty_decomposition"] = self._save_figure(fig, f"{base_name}_uncertainty")
        
        # Confidence matrix
        confidences = 1.0 - prediction_results.uncertainties  # Convert uncertainty to confidence
        fig = self.prediction_viz.plot_prediction_confidence_matrix(
            prediction_results.predictions, confidences, prediction_results.fault_types
        )
        plots["confidence_matrix"] = self._save_figure(fig, f"{base_name}_confidence")
        
        # Reliability assessment
        fig = self.prediction_viz.plot_reliability_assessment(
            prediction_results.predictions,
            prediction_results.uncertainties,
            prediction_results.true_labels
        )
        plots["reliability_assessment"] = self._save_figure(fig, f"{base_name}_reliability")
        
        # Fault type analysis
        fig = self.prediction_viz.plot_fault_type_analysis(
            prediction_results.predictions,
            prediction_results.uncertainties,
            prediction_results.fault_types,
            prediction_results.true_labels
        )
        plots["fault_analysis"] = self._save_figure(fig, f"{base_name}_fault_analysis")
        
        return plots
    
    def _generate_physics_plots(self,
                              physics_residuals: PhysicsResiduals,
                              constraint_violations: Optional[PDEConstraintViolations],
                              base_name: str) -> Dict[str, str]:
        """Generate all physics-related plots."""
        plots = {}
        
        # PDE residuals overview
        fig = self.physics_viz.plot_pde_residuals_overview(physics_residuals)
        plots["pde_residuals"] = self._save_figure(fig, f"{base_name}_residuals")
        
        # Physics consistency evolution
        fig = self.physics_viz.plot_physics_consistency_evolution(physics_residuals)
        plots["consistency_evolution"] = self._save_figure(fig, f"{base_name}_consistency")
        
        # PDE residual statistics
        fig = self.physics_viz.plot_pde_residual_statistics(physics_residuals)
        plots["residual_statistics"] = self._save_figure(fig, f"{base_name}_statistics")
        
        # Constraint violations (if available)
        if constraint_violations is not None:
            fig = self.physics_viz.plot_constraint_violation_analysis(constraint_violations)
            plots["constraint_violations"] = self._save_figure(fig, f"{base_name}_violations")
            
            fig = self.physics_viz.plot_conservation_law_analysis(constraint_violations)
            plots["conservation_laws"] = self._save_figure(fig, f"{base_name}_conservation")
        
        # Multi-physics coupling analysis (synthetic example)
        n_samples = len(physics_residuals.timestamps)
        em_field = np.random.normal(0, 1, n_samples)
        thermal_field = np.random.normal(20, 5, n_samples)
        mechanical_disp = np.random.normal(0, 0.1, n_samples)
        
        fig = self.physics_viz.plot_multiphysics_coupling_analysis(
            em_field, thermal_field, mechanical_disp, physics_residuals.timestamps
        )
        plots["multiphysics_coupling"] = self._save_figure(fig, f"{base_name}_coupling")
        
        return plots
    
    def _generate_diagnostic_plots(self,
                                 model_diagnostics: Optional[ModelDiagnostics],
                                 performance_metrics: Optional[PerformanceMetrics],
                                 base_name: str) -> Dict[str, str]:
        """Generate all diagnostic plots."""
        plots = {}
        
        # Performance metrics
        if performance_metrics is not None:
            fig = self.diagnostic_viz.plot_training_diagnostics(performance_metrics)
            plots["training_diagnostics"] = self._save_figure(fig, f"{base_name}_training")
            
            fig = self.diagnostic_viz.plot_performance_bottleneck_analysis(performance_metrics)
            plots["bottleneck_analysis"] = self._save_figure(fig, f"{base_name}_bottlenecks")
        
        # Model diagnostics
        if model_diagnostics is not None:
            fig = self.diagnostic_viz.plot_model_architecture_analysis(model_diagnostics)
            plots["architecture_analysis"] = self._save_figure(fig, f"{base_name}_architecture")
            
            fig = self.diagnostic_viz.plot_gradient_flow_analysis(model_diagnostics)
            plots["gradient_flow"] = self._save_figure(fig, f"{base_name}_gradients")
            
            fig = self.diagnostic_viz.plot_activation_analysis(model_diagnostics)
            plots["activation_analysis"] = self._save_figure(fig, f"{base_name}_activations")
        
        return plots
    
    def _generate_comparative_plots(self,
                                  benchmark_results: BenchmarkResults,
                                  base_name: str) -> Dict[str, str]:
        """Generate all comparative analysis plots."""
        plots = {}
        
        # Method comparison overview
        fig = self.comparative_viz.plot_method_comparison_overview(benchmark_results)
        plots["method_comparison"] = self._save_figure(fig, f"{base_name}_overview")
        
        # Performance radar chart
        fig = self.comparative_viz.plot_performance_radar_chart(benchmark_results)
        plots["performance_radar"] = self._save_figure(fig, f"{base_name}_radar")
        
        # Precision-recall comparison
        fig = self.comparative_viz.plot_precision_recall_comparison(benchmark_results)
        plots["precision_recall"] = self._save_figure(fig, f"{base_name}_precision_recall")
        
        # Efficiency analysis
        fig = self.comparative_viz.plot_efficiency_analysis(benchmark_results)
        plots["efficiency_analysis"] = self._save_figure(fig, f"{base_name}_efficiency")
        
        # Statistical significance analysis
        fig = self.comparative_viz.plot_statistical_significance_analysis(benchmark_results)
        plots["statistical_analysis"] = self._save_figure(fig, f"{base_name}_statistics")
        
        return plots
    
    def _generate_summary_dashboard(self,
                                  results: ComprehensiveResults,
                                  base_name: str) -> str:
        """Generate summary dashboard with key metrics and plots."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('AV-PINO System Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # Row 1: Prediction Performance
        if results.prediction_results is not None:
            # Accuracy over time
            if results.prediction_results.true_labels is not None:
                accuracy = (results.prediction_results.predictions == 
                           results.prediction_results.true_labels).astype(float)
                axes[0, 0].plot(results.prediction_results.timestamps, accuracy, 'b-', linewidth=2)
                axes[0, 0].set_title('Prediction Accuracy Over Time')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Uncertainty distribution
            axes[0, 1].hist(results.prediction_results.uncertainties, bins=30, 
                           alpha=0.7, color='orange', edgecolor='black')
            axes[0, 1].set_title('Uncertainty Distribution')
            axes[0, 1].set_xlabel('Uncertainty')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Fault type distribution
            fault_counts = np.bincount(results.prediction_results.predictions.astype(int))
            fault_labels = results.prediction_results.fault_types[:len(fault_counts)]
            axes[0, 2].bar(range(len(fault_counts)), fault_counts, alpha=0.7)
            axes[0, 2].set_title('Fault Type Distribution')
            axes[0, 2].set_xticks(range(len(fault_labels)))
            axes[0, 2].set_xticklabels(fault_labels, rotation=45)
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Row 2: Physics Consistency
        if results.physics_residuals is not None:
            # Combined residual evolution
            combined_residual = (results.physics_residuals.maxwell_residual + 
                               results.physics_residuals.heat_equation_residual +
                               results.physics_residuals.structural_dynamics_residual +
                               results.physics_residuals.coupling_residual) / 4
            
            axes[1, 0].semilogy(results.physics_residuals.timestamps, combined_residual, 
                               'r-', linewidth=2)
            axes[1, 0].set_title('Physics Consistency (Combined Residual)')
            axes[1, 0].set_ylabel('Residual Magnitude (log)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Individual residuals
            residual_names = ['Maxwell', 'Heat', 'Structural', 'Coupling']
            residual_values = [
                np.mean(results.physics_residuals.maxwell_residual),
                np.mean(results.physics_residuals.heat_equation_residual),
                np.mean(results.physics_residuals.structural_dynamics_residual),
                np.mean(results.physics_residuals.coupling_residual)
            ]
            
            axes[1, 1].bar(residual_names, residual_values, alpha=0.7, 
                          color=['red', 'orange', 'blue', 'green'])
            axes[1, 1].set_title('Average PDE Residuals')
            axes[1, 1].set_ylabel('Mean Residual')
            axes[1, 1].set_yscale('log')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Physics consistency score
            consistency_score = 1.0 / (1.0 + combined_residual)
            axes[1, 2].plot(results.physics_residuals.timestamps, consistency_score, 
                           'g-', linewidth=2)
            axes[1, 2].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='High')
            axes[1, 2].axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Medium')
            axes[1, 2].set_title('Physics Consistency Score')
            axes[1, 2].set_ylabel('Consistency Score')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # Row 3: Performance Metrics
        if results.performance_metrics is not None:
            # Inference time distribution
            inference_times_ms = results.performance_metrics.inference_times * 1000
            axes[2, 0].hist(inference_times_ms, bins=30, alpha=0.7, 
                           color='purple', edgecolor='black')
            axes[2, 0].axvline(1.0, color='red', linestyle='--', linewidth=2, 
                              label='1ms Target')
            axes[2, 0].axvline(np.mean(inference_times_ms), color='green', 
                              linestyle='--', linewidth=2, 
                              label=f'Mean: {np.mean(inference_times_ms):.2f}ms')
            axes[2, 0].set_title('Inference Time Distribution')
            axes[2, 0].set_xlabel('Inference Time (ms)')
            axes[2, 0].set_ylabel('Frequency')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Training progress
            axes[2, 1].plot(results.performance_metrics.epochs, 
                           results.performance_metrics.accuracy_over_time, 
                           'b-', linewidth=2, label='Accuracy')
            axes[2, 1].set_title('Training Progress')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Accuracy')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
            # Memory usage
            memory_mb = results.performance_metrics.memory_usage / (1024**2)
            axes[2, 2].plot(range(len(memory_mb)), memory_mb, 'purple', linewidth=2)
            axes[2, 2].set_title('Memory Usage')
            axes[2, 2].set_xlabel('Training Step')
            axes[2, 2].set_ylabel('Memory (MB)')
            axes[2, 2].grid(True, alpha=0.3)
        
        # Fill empty subplots with summary text
        for i in range(3):
            for j in range(3):
                if not axes[i, j].has_data():
                    axes[i, j].text(0.5, 0.5, 'Data not available\nfor this analysis',
                                   ha='center', va='center', transform=axes[i, j].transAxes,
                                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                                         facecolor="lightgray"))
        
        plt.tight_layout()
        return self._save_figure(fig, base_name)
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> str:
        """Save figure to file and optionally display."""
        filepath = self.output_dir / f"{filename}.{self.config.figure_format}"
        
        if self.config.save_figures:
            fig.savefig(filepath, dpi=self.config.figure_dpi, bbox_inches='tight')
        
        if self.config.show_figures:
            plt.show()
        else:
            plt.close(fig)
        
        return str(filepath)
    
    def _save_report_metadata(self, 
                            results: ComprehensiveResults,
                            plot_files: Dict[str, str],
                            report_name: str) -> None:
        """Save report metadata to JSON file."""
        metadata = {
            "report_name": report_name,
            "generated_plots": list(plot_files.keys()),
            "plot_files": plot_files,
            "config": {
                "output_dir": self.config.output_dir,
                "figure_format": self.config.figure_format,
                "figure_dpi": self.config.figure_dpi,
                "style": self.config.style
            }
        }
        
        # Add results metadata if available
        if results.metadata is not None:
            metadata["results_metadata"] = results.metadata
        
        # Add data availability summary
        metadata["data_availability"] = {
            "prediction_results": results.prediction_results is not None,
            "uncertainty_metrics": results.uncertainty_metrics is not None,
            "physics_residuals": results.physics_residuals is not None,
            "constraint_violations": results.constraint_violations is not None,
            "model_diagnostics": results.model_diagnostics is not None,
            "performance_metrics": results.performance_metrics is not None,
            "benchmark_results": results.benchmark_results is not None
        }
        
        metadata_file = self.output_dir / f"{report_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def create_interactive_dashboard(self, 
                                   results: ComprehensiveResults,
                                   dashboard_name: str = "interactive_dashboard") -> str:
        """
        Create interactive HTML dashboard (placeholder for future implementation).
        
        Args:
            results: ComprehensiveResults containing all analysis data
            dashboard_name: Name for the dashboard file
            
        Returns:
            Path to the generated HTML dashboard
        """
        # This is a placeholder for future interactive dashboard implementation
        # Could use libraries like Plotly Dash, Bokeh, or Streamlit
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AV-PINO Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                          background-color: #e8f4f8; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AV-PINO Motor Fault Diagnosis Analysis Dashboard</h1>
                <p>Comprehensive analysis results for physics-informed neural operator system</p>
            </div>
            
            <div class="section">
                <h2>System Overview</h2>
                <div class="metric">
                    <strong>Prediction Data:</strong> 
                    {'Available' if results.prediction_results else 'Not Available'}
                </div>
                <div class="metric">
                    <strong>Physics Analysis:</strong> 
                    {'Available' if results.physics_residuals else 'Not Available'}
                </div>
                <div class="metric">
                    <strong>Performance Metrics:</strong> 
                    {'Available' if results.performance_metrics else 'Not Available'}
                </div>
                <div class="metric">
                    <strong>Benchmark Results:</strong> 
                    {'Available' if results.benchmark_results else 'Not Available'}
                </div>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                <p>This interactive dashboard would contain dynamic visualizations and analysis.</p>
                <p>Future implementation could include:</p>
                <ul>
                    <li>Interactive plots with zoom and pan capabilities</li>
                    <li>Real-time data updates</li>
                    <li>Customizable analysis parameters</li>
                    <li>Export capabilities for reports</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        dashboard_file = self.output_dir / f"{dashboard_name}.html"
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
        
        return str(dashboard_file)
    
    def export_analysis_summary(self, 
                              results: ComprehensiveResults,
                              summary_name: str = "analysis_summary") -> str:
        """
        Export analysis summary to text file.
        
        Args:
            results: ComprehensiveResults containing all analysis data
            summary_name: Name for the summary file
            
        Returns:
            Path to the generated summary file
        """
        summary_lines = []
        summary_lines.append("AV-PINO Motor Fault Diagnosis System - Analysis Summary")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        
        # Prediction Analysis Summary
        if results.prediction_results is not None:
            summary_lines.append("PREDICTION ANALYSIS")
            summary_lines.append("-" * 20)
            summary_lines.append(f"Total Predictions: {len(results.prediction_results.predictions)}")
            summary_lines.append(f"Fault Types: {', '.join(results.prediction_results.fault_types)}")
            summary_lines.append(f"Mean Uncertainty: {np.mean(results.prediction_results.uncertainties):.4f}")
            summary_lines.append(f"Max Uncertainty: {np.max(results.prediction_results.uncertainties):.4f}")
            
            if results.prediction_results.true_labels is not None:
                accuracy = np.mean(results.prediction_results.predictions == 
                                 results.prediction_results.true_labels)
                summary_lines.append(f"Overall Accuracy: {accuracy:.4f}")
            summary_lines.append("")
        
        # Physics Analysis Summary
        if results.physics_residuals is not None:
            summary_lines.append("PHYSICS ANALYSIS")
            summary_lines.append("-" * 15)
            summary_lines.append(f"Maxwell Residual (mean): {np.mean(results.physics_residuals.maxwell_residual):.2e}")
            summary_lines.append(f"Heat Equation Residual (mean): {np.mean(results.physics_residuals.heat_equation_residual):.2e}")
            summary_lines.append(f"Structural Dynamics Residual (mean): {np.mean(results.physics_residuals.structural_dynamics_residual):.2e}")
            summary_lines.append(f"Coupling Residual (mean): {np.mean(results.physics_residuals.coupling_residual):.2e}")
            
            combined_residual = (results.physics_residuals.maxwell_residual + 
                               results.physics_residuals.heat_equation_residual +
                               results.physics_residuals.structural_dynamics_residual +
                               results.physics_residuals.coupling_residual) / 4
            consistency_score = np.mean(1.0 / (1.0 + combined_residual))
            summary_lines.append(f"Physics Consistency Score: {consistency_score:.4f}")
            summary_lines.append("")
        
        # Performance Analysis Summary
        if results.performance_metrics is not None:
            summary_lines.append("PERFORMANCE ANALYSIS")
            summary_lines.append("-" * 20)
            mean_inference_time = np.mean(results.performance_metrics.inference_times) * 1000
            summary_lines.append(f"Mean Inference Time: {mean_inference_time:.3f} ms")
            summary_lines.append(f"Target Achievement (<1ms): {'Yes' if mean_inference_time < 1.0 else 'No'}")
            
            mean_memory = np.mean(results.performance_metrics.memory_usage) / (1024**2)
            summary_lines.append(f"Mean Memory Usage: {mean_memory:.1f} MB")
            
            final_accuracy = results.performance_metrics.accuracy_over_time[-1]
            summary_lines.append(f"Final Training Accuracy: {final_accuracy:.4f}")
            summary_lines.append("")
        
        # Benchmark Analysis Summary
        if results.benchmark_results is not None:
            summary_lines.append("BENCHMARK ANALYSIS")
            summary_lines.append("-" * 18)
            summary_lines.append(f"Dataset: {results.benchmark_results.dataset_name}")
            summary_lines.append(f"Test Size: {results.benchmark_results.test_size}")
            summary_lines.append("Method Performance:")
            
            for method in results.benchmark_results.methods:
                summary_lines.append(f"  {method.method_name}:")
                summary_lines.append(f"    Accuracy: {method.accuracy:.4f}")
                summary_lines.append(f"    Inference Time: {method.inference_time*1000:.3f} ms")
                summary_lines.append(f"    Memory Usage: {method.memory_usage/(1024**2):.1f} MB")
                if method.physics_consistency is not None:
                    summary_lines.append(f"    Physics Consistency: {method.physics_consistency:.4f}")
            summary_lines.append("")
        
        # System Requirements Compliance
        summary_lines.append("REQUIREMENTS COMPLIANCE")
        summary_lines.append("-" * 24)
        
        # Check >90% accuracy requirement
        if results.prediction_results and results.prediction_results.true_labels is not None:
            accuracy = np.mean(results.prediction_results.predictions == 
                             results.prediction_results.true_labels)
            summary_lines.append(f"Accuracy Requirement (>90%): {'✓ PASS' if accuracy > 0.9 else '✗ FAIL'} ({accuracy:.1%})")
        
        # Check <1ms inference requirement
        if results.performance_metrics is not None:
            mean_time = np.mean(results.performance_metrics.inference_times) * 1000
            summary_lines.append(f"Inference Time Requirement (<1ms): {'✓ PASS' if mean_time < 1.0 else '✗ FAIL'} ({mean_time:.3f}ms)")
        
        # Check physics consistency
        if results.physics_residuals is not None:
            combined_residual = (results.physics_residuals.maxwell_residual + 
                               results.physics_residuals.heat_equation_residual +
                               results.physics_residuals.structural_dynamics_residual +
                               results.physics_residuals.coupling_residual) / 4
            consistency_score = np.mean(1.0 / (1.0 + combined_residual))
            summary_lines.append(f"Physics Consistency: {'✓ GOOD' if consistency_score > 0.8 else '⚠ MODERATE' if consistency_score > 0.6 else '✗ POOR'} ({consistency_score:.3f})")
        
        summary_content = "\n".join(summary_lines)
        
        summary_file = self.output_dir / f"{summary_name}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        return str(summary_file)