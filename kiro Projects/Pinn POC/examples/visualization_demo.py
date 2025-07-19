"""
Demonstration of AV-PINO Visualization System.

This script demonstrates the comprehensive visualization capabilities
for the AV-PINO motor fault diagnosis system.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import visualization components
from src.visualization import (
    VisualizationManager, VisualizationConfig, ComprehensiveResults,
    PredictionResult, UncertaintyMetrics, PhysicsResiduals, 
    PDEConstraintViolations, ModelDiagnostics, PerformanceMetrics,
    MethodResults, BenchmarkResults
)

def create_sample_data():
    """Create sample data for demonstration."""
    n_samples = 200
    n_epochs = 100
    fault_types = ['Normal', 'Inner Race Fault', 'Outer Race Fault', 'Ball Fault']
    
    # Create prediction results
    prediction_results = PredictionResult(
        predictions=np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.2, 0.2, 0.2]),
        uncertainties=np.random.exponential(0.1, n_samples),
        confidence_intervals=np.random.rand(n_samples, 2) * 0.2 + 0.8,
        fault_types=fault_types,
        timestamps=np.linspace(0, 20, n_samples),
        true_labels=np.random.choice([0, 1, 2, 3], n_samples, p=[0.35, 0.25, 0.25, 0.15])
    )
    
    # Create uncertainty metrics
    uncertainty_metrics = UncertaintyMetrics(
        epistemic_uncertainty=np.random.exponential(0.05, n_samples),
        aleatoric_uncertainty=np.random.exponential(0.08, n_samples),
        total_uncertainty=np.random.exponential(0.1, n_samples),
        confidence_scores=np.random.beta(8, 2, n_samples)
    )
    
    # Create physics residuals
    physics_residuals = PhysicsResiduals(
        maxwell_residual=np.random.exponential(0.01, n_samples),
        heat_equation_residual=np.random.exponential(0.005, n_samples),
        structural_dynamics_residual=np.random.exponential(0.02, n_samples),
        coupling_residual=np.random.exponential(0.008, n_samples),
        timestamps=np.linspace(0, 20, n_samples),
        constraint_weights={'Maxwell': 1.0, 'Heat': 0.8, 'Structural': 1.2, 'Coupling': 0.9}
    )
    
    # Create constraint violations
    constraint_violations = PDEConstraintViolations(
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
            'Maxwell': [10, 25, 67, 89, 120, 145],
            'Heat': [15, 45, 78, 110],
            'Structural': [5, 30, 55, 78, 95, 125, 160]
        },
        conservation_errors={
            'Energy': np.random.normal(0, 0.01, n_samples),
            'Momentum': np.random.normal(0, 0.005, n_samples),
            'Charge': np.random.normal(0, 0.002, n_samples)
        }
    )
    
    # Create model diagnostics
    model_diagnostics = ModelDiagnostics(
        layer_activations={
            'encoder_layer1': np.random.normal(0, 1, (n_samples, 64)),
            'encoder_layer2': np.random.normal(0, 0.8, (n_samples, 128)),
            'processor_layer1': np.random.normal(0, 0.6, (n_samples, 256)),
            'processor_layer2': np.random.normal(0, 0.5, (n_samples, 256)),
            'decoder_layer1': np.random.normal(0, 0.7, (n_samples, 128)),
            'decoder_layer2': np.random.normal(0, 0.9, (n_samples, 4))
        },
        gradient_norms={
            'encoder_layer1': np.random.exponential(0.1, n_epochs),
            'encoder_layer2': np.random.exponential(0.08, n_epochs),
            'processor_layer1': np.random.exponential(0.05, n_epochs),
            'processor_layer2': np.random.exponential(0.04, n_epochs),
            'decoder_layer1': np.random.exponential(0.06, n_epochs),
            'decoder_layer2': np.random.exponential(0.09, n_epochs)
        },
        weight_distributions={
            'encoder_layer1': np.random.normal(0, 0.1, (64, 32)),
            'encoder_layer2': np.random.normal(0, 0.08, (128, 64)),
            'processor_layer1': np.random.normal(0, 0.05, (256, 128)),
            'processor_layer2': np.random.normal(0, 0.04, (256, 256)),
            'decoder_layer1': np.random.normal(0, 0.06, (128, 256)),
            'decoder_layer2': np.random.normal(0, 0.09, (4, 128))
        },
        loss_components={
            'data_loss': np.random.exponential(0.5, n_epochs),
            'physics_loss': np.random.exponential(0.1, n_epochs),
            'consistency_loss': np.random.exponential(0.05, n_epochs),
            'variational_loss': np.random.exponential(0.02, n_epochs)
        },
        training_metrics={
            'accuracy': list(0.5 + 0.4 * (1 - np.exp(-np.arange(n_epochs) / 20)) + np.random.normal(0, 0.02, n_epochs)),
            'loss': list(2.0 * np.exp(-np.arange(n_epochs) / 15) + 0.1 + np.random.exponential(0.05, n_epochs))
        }
    )
    
    # Create performance metrics
    performance_metrics = PerformanceMetrics(
        inference_times=np.random.exponential(0.0008, 1000),  # ~0.8ms average
        memory_usage=np.random.normal(512*1024*1024, 50*1024*1024, n_epochs),  # ~512MB
        accuracy_over_time=0.5 + 0.4 * (1 - np.exp(-np.arange(n_epochs) / 20)) + np.random.normal(0, 0.02, n_epochs),
        loss_over_time=2.0 * np.exp(-np.arange(n_epochs) / 15) + 0.1 + np.random.exponential(0.05, n_epochs),
        learning_rates=0.001 * np.exp(-np.arange(n_epochs) / 50),  # Exponential decay
        epochs=np.arange(n_epochs)
    )
    
    # Create benchmark results
    benchmark_results = BenchmarkResults(
        methods=[
            MethodResults(
                method_name='AV-PINO',
                accuracy=0.94,
                precision=np.array([0.95, 0.92, 0.93, 0.96]),
                recall=np.array([0.93, 0.94, 0.95, 0.92]),
                f1_score=np.array([0.94, 0.93, 0.94, 0.94]),
                inference_time=0.0008,
                training_time=3600,
                memory_usage=512*1024*1024,
                physics_consistency=0.92,
                uncertainty_quality=0.88
            ),
            MethodResults(
                method_name='Traditional SVM',
                accuracy=0.85,
                precision=np.array([0.87, 0.82, 0.84, 0.88]),
                recall=np.array([0.83, 0.86, 0.87, 0.84]),
                f1_score=np.array([0.85, 0.84, 0.85, 0.86]),
                inference_time=0.002,
                training_time=1200,
                memory_usage=128*1024*1024,
                physics_consistency=None,
                uncertainty_quality=None
            ),
            MethodResults(
                method_name='Random Forest',
                accuracy=0.88,
                precision=np.array([0.89, 0.86, 0.87, 0.90]),
                recall=np.array([0.87, 0.88, 0.89, 0.87]),
                f1_score=np.array([0.88, 0.87, 0.88, 0.88]),
                inference_time=0.0015,
                training_time=900,
                memory_usage=256*1024*1024,
                physics_consistency=None,
                uncertainty_quality=None
            ),
            MethodResults(
                method_name='Deep CNN',
                accuracy=0.91,
                precision=np.array([0.92, 0.89, 0.90, 0.93]),
                recall=np.array([0.90, 0.91, 0.92, 0.91]),
                f1_score=np.array([0.91, 0.90, 0.91, 0.92]),
                inference_time=0.005,
                training_time=7200,
                memory_usage=1024*1024*1024,
                physics_consistency=None,
                uncertainty_quality=0.75
            ),
            MethodResults(
                method_name='Physics-Informed NN',
                accuracy=0.89,
                precision=np.array([0.90, 0.87, 0.88, 0.91]),
                recall=np.array([0.88, 0.89, 0.90, 0.89]),
                f1_score=np.array([0.89, 0.88, 0.89, 0.90]),
                inference_time=0.003,
                training_time=5400,
                memory_usage=768*1024*1024,
                physics_consistency=0.78,
                uncertainty_quality=0.65
            )
        ],
        fault_types=fault_types,
        dataset_name='CWRU Bearing Dataset',
        test_size=2000
    )
    
    # Create comprehensive results
    comprehensive_results = ComprehensiveResults(
        prediction_results=prediction_results,
        uncertainty_metrics=uncertainty_metrics,
        physics_residuals=physics_residuals,
        constraint_violations=constraint_violations,
        model_diagnostics=model_diagnostics,
        performance_metrics=performance_metrics,
        benchmark_results=benchmark_results,
        metadata={
            'experiment_name': 'AV-PINO Demonstration',
            'dataset': 'CWRU Bearing Dataset',
            'model_version': '1.0',
            'training_date': '2024-01-15',
            'hardware': 'NVIDIA RTX 4090',
            'framework': 'PyTorch 2.0'
        }
    )
    
    return comprehensive_results

def main():
    """Main demonstration function."""
    print("AV-PINO Visualization System Demonstration")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Configure visualization
    config = VisualizationConfig(
        output_dir=str(output_dir),
        figure_format="png",
        figure_dpi=300,
        save_figures=True,
        show_figures=False  # Set to True to display plots
    )
    
    # Initialize visualization manager
    print("Initializing visualization manager...")
    viz_manager = VisualizationManager(config)
    
    # Create sample data
    print("Creating sample data...")
    results = create_sample_data()
    
    # Generate comprehensive report
    print("Generating comprehensive visualization report...")
    plot_files = viz_manager.generate_comprehensive_report(
        results, 
        "av_pino_demo_report"
    )
    
    print(f"Generated {len(plot_files)} visualization plots:")
    for plot_type, file_path in plot_files.items():
        print(f"  - {plot_type}: {file_path}")
    
    # Export analysis summary
    print("\nExporting analysis summary...")
    summary_file = viz_manager.export_analysis_summary(
        results,
        "av_pino_demo_summary"
    )
    print(f"Analysis summary saved to: {summary_file}")
    
    # Create interactive dashboard
    print("\nCreating interactive dashboard...")
    dashboard_file = viz_manager.create_interactive_dashboard(
        results,
        "av_pino_demo_dashboard"
    )
    print(f"Interactive dashboard saved to: {dashboard_file}")
    
    # Demonstrate individual visualizers
    print("\nDemonstrating individual visualizers...")
    
    # Prediction visualization
    print("  - Creating prediction timeline...")
    pred_fig = viz_manager.prediction_viz.plot_prediction_timeline(
        results.prediction_results,
        title="AV-PINO Fault Prediction Timeline"
    )
    pred_file = output_dir / "individual_prediction_timeline.png"
    pred_fig.savefig(pred_file, dpi=300, bbox_inches='tight')
    plt.close(pred_fig)
    print(f"    Saved to: {pred_file}")
    
    # Physics analysis
    print("  - Creating physics residuals overview...")
    physics_fig = viz_manager.physics_viz.plot_pde_residuals_overview(
        results.physics_residuals,
        title="AV-PINO Physics Consistency Analysis"
    )
    physics_file = output_dir / "individual_physics_residuals.png"
    physics_fig.savefig(physics_file, dpi=300, bbox_inches='tight')
    plt.close(physics_fig)
    print(f"    Saved to: {physics_file}")
    
    # Diagnostic analysis
    print("  - Creating training diagnostics...")
    diag_fig = viz_manager.diagnostic_viz.plot_training_diagnostics(
        results.performance_metrics,
        title="AV-PINO Training Diagnostics"
    )
    diag_file = output_dir / "individual_training_diagnostics.png"
    diag_fig.savefig(diag_file, dpi=300, bbox_inches='tight')
    plt.close(diag_fig)
    print(f"    Saved to: {diag_file}")
    
    # Comparative analysis
    print("  - Creating method comparison...")
    comp_fig = viz_manager.comparative_viz.plot_method_comparison_overview(
        results.benchmark_results,
        title="AV-PINO vs Traditional Methods"
    )
    comp_file = output_dir / "individual_method_comparison.png"
    comp_fig.savefig(comp_file, dpi=300, bbox_inches='tight')
    plt.close(comp_fig)
    print(f"    Saved to: {comp_file}")
    
    print(f"\nDemonstration complete! All outputs saved to: {output_dir}")
    print("\nKey Features Demonstrated:")
    print("✓ Prediction visualization with confidence intervals and uncertainty displays")
    print("✓ Physics consistency analysis plots showing PDE residuals")
    print("✓ Diagnostic visualization tools for model debugging and analysis")
    print("✓ Comparative performance visualization between methods")
    print("✓ Unified visualization management and reporting")
    print("✓ Comprehensive analysis summary generation")
    print("✓ Interactive dashboard creation")
    
    # Print some key metrics from the demo
    print(f"\nDemo Results Summary:")
    print(f"- Model Accuracy: {np.mean(results.prediction_results.predictions == results.prediction_results.true_labels):.1%}")
    print(f"- Mean Inference Time: {np.mean(results.performance_metrics.inference_times)*1000:.2f}ms")
    print(f"- Physics Consistency Score: {np.mean(1.0 / (1.0 + (results.physics_residuals.maxwell_residual + results.physics_residuals.heat_equation_residual + results.physics_residuals.structural_dynamics_residual + results.physics_residuals.coupling_residual) / 4)):.3f}")
    print(f"- Best Competing Method: {max(results.benchmark_results.methods, key=lambda x: x.accuracy).method_name} ({max(results.benchmark_results.methods, key=lambda x: x.accuracy).accuracy:.1%})")

if __name__ == "__main__":
    main()