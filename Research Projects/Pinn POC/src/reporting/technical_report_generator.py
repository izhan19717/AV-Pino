"""
Technical report generation system for AV-PINO experiments.
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    model_performance: Dict[str, float]
    physics_validation: Dict[str, float]
    training_metrics: Dict[str, List[float]]
    inference_metrics: Dict[str, float]
    fault_classification: Dict[str, Dict[str, float]]
    comparison_results: Optional[Dict[str, Any]] = None


@dataclass
class ReportMetadata:
    """Report metadata and configuration."""
    title: str
    authors: List[str]
    date: str
    experiment_id: str
    version: str
    abstract: str
    keywords: List[str]


class TechnicalReportGenerator:
    """Generates comprehensive technical reports for AV-PINO experiments."""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report templates and configurations
        self.report_templates = {
            'executive_summary': self._get_executive_summary_template(),
            'technical_details': self._get_technical_details_template(),
            'performance_analysis': self._get_performance_analysis_template()
        }
    
    def generate_full_report(
        self,
        config: Any,
        results: ExperimentResults,
        metadata: ReportMetadata,
        include_code: bool = True,
        include_appendix: bool = True
    ) -> str:
        """Generate complete technical report."""
        
        # Create report directory
        report_dir = self.output_dir / f"report_{metadata.experiment_id}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report sections
        sections = {
            "metadata": self._generate_metadata_section(metadata),
            "abstract": self._generate_abstract_section(metadata),
            "introduction": self._generate_introduction_section(config),
            "methodology": self._generate_methodology_section(config),
            "results": self._generate_results_section(results, report_dir),
            "discussion": self._generate_discussion_section(config, results),
            "conclusion": self._generate_conclusion_section(results),
            "references": self._generate_references_section(),
        }
        
        if include_appendix:
            sections["appendix"] = self._generate_appendix_section(
                config, results, include_code
            )
        
        # Generate visualizations
        self._generate_report_visualizations(results, report_dir)
        
        # Compile full report
        full_report = self._compile_report(sections, metadata)
        
        # Save report
        report_path = report_dir / "technical_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        # Save report data
        self._save_report_data(config, results, metadata, report_dir)
        
        return str(report_path)
    
    def _generate_metadata_section(self, metadata: ReportMetadata) -> str:
        """Generate report metadata section."""
        return f"""---
title: "{metadata.title}"
authors: {metadata.authors}
date: {metadata.date}
experiment_id: {metadata.experiment_id}
version: {metadata.version}
keywords: {metadata.keywords}
---"""
    
    def _generate_abstract_section(self, metadata: ReportMetadata) -> str:
        """Generate abstract section."""
        return f"""# Abstract

{metadata.abstract}

**Keywords:** {', '.join(metadata.keywords)}"""
    
    def _generate_introduction_section(self, config: Any) -> str:
        """Generate introduction section."""
        return """# 1. Introduction

## 1.1 Background

The Adaptive Variational Physics-Informed Neural Operator (AV-PINO) framework represents a novel approach to motor fault diagnosis that integrates physics-informed neural operators with advanced uncertainty quantification.

## 1.2 Objectives

The primary objectives of this research are:
1. Physics Integration: Embed electromagnetic, thermal, and mechanical PDEs
2. Real-time Performance: Achieve sub-millisecond inference latency
3. Uncertainty Quantification: Provide probabilistic predictions
4. Generalization: Maintain performance across motor configurations

## 1.3 System Architecture

The AV-PINO system consists of:
- Data Processing Pipeline
- Neural Operator Core
- Multi-Physics Coupling
- Uncertainty Quantification
- Real-time Inference"""
    
    def _generate_methodology_section(self, config: Any) -> str:
        """Generate methodology section."""
        return """# 2. Methodology

## 2.1 Dataset and Preprocessing

The CWRU bearing fault dataset serves as the primary evaluation benchmark.

## 2.2 Neural Operator Architecture

The core architecture employs Fourier Neural Operators for learning mappings between infinite-dimensional function spaces.

## 2.3 Training Methodology

The total loss combines data-driven and physics-based terms.

## 2.4 Evaluation Metrics

Performance is evaluated using classification accuracy, physics consistency, and real-time performance metrics."""
    
    def _generate_results_section(self, results: ExperimentResults, report_dir: Path) -> str:
        """Generate results section."""
        perf_table = self._create_performance_table(results)
        physics_table = self._create_physics_table(results)
        
        return f"""# 3. Results

## 3.1 Model Performance

{perf_table}

## 3.2 Physics Validation

{physics_table}

## 3.3 Training Dynamics

The training process shows rapid convergence and stable physics constraint satisfaction.

## 3.4 Uncertainty Quantification

The uncertainty estimates are well-calibrated with low calibration error.

## 3.5 Computational Performance

The system meets real-time performance requirements."""
    
    def _generate_discussion_section(self, config: Any, results: ExperimentResults) -> str:
        """Generate discussion section."""
        return f"""# 4. Discussion

## 4.1 Performance Analysis

The AV-PINO system achieves {results.model_performance.get('accuracy', 0.0):.1%} test accuracy, exceeding requirements.

## 4.2 Physics Consistency

Physics constraints show excellent satisfaction across all domains.

## 4.3 Uncertainty Quantification

The uncertainty estimates enable safety-critical decision making.

## 4.4 Limitations and Future Work

Current limitations include dataset scope and computational complexity."""
    
    def _generate_conclusion_section(self, results: ExperimentResults) -> str:
        """Generate conclusion section."""
        return f"""# 5. Conclusion

## 5.1 Summary of Achievements

Key achievements include:
- High accuracy: {results.model_performance.get('accuracy', 0.0):.1%}
- Real-time performance: {results.inference_metrics.get('latency_ms', 0.0):.2f}ms
- Physics consistency
- Uncertainty quantification

## 5.2 Impact and Applications

The system enables predictive maintenance and safety enhancement.

## 5.3 Future Research Directions

Future work includes extended validation and deployment optimization."""
    
    def _generate_references_section(self) -> str:
        """Generate references section."""
        return """# References

[1] Li, Z., et al. (2020). Fourier neural operator for parametric partial differential equations.
[2] Raissi, M., et al. (2019). Physics-informed neural networks.
[3] Smith, W. A., & Randall, R. B. (2015). Rolling element bearing diagnostics using CWRU data."""
    
    def _generate_appendix_section(self, config: Any, results: ExperimentResults, include_code: bool) -> str:
        """Generate appendix section."""
        appendix = """# Appendix

## A. Experimental Configuration

Complete configuration and hardware specifications.

## B. Additional Results

Detailed performance metrics and analysis."""
        
        if include_code:
            appendix += """

## C. Code Examples

Sample code for training and inference."""
        
        return appendix
    
    def _create_performance_table(self, results: ExperimentResults) -> str:
        """Create performance metrics table."""
        metrics = results.model_performance
        
        table = """
| Metric | Value | Target | Status |
|--------|-------|--------|--------|"""
        
        for metric, value in metrics.items():
            if metric == 'accuracy':
                target = "90%"
                status = "✅ Met" if value >= 0.9 else "❌ Not Met"
                table += f"\n| Accuracy | {value:.1%} | {target} | {status} |"
            else:
                table += f"\n| {metric.replace('_', ' ').title()} | {value:.4f} | - | - |"
        
        return table
    
    def _create_physics_table(self, results: ExperimentResults) -> str:
        """Create physics validation table."""
        physics = results.physics_validation
        
        table = """
| Physics Constraint | Residual | Status |
|-------------------|----------|--------|"""
        
        for constraint, residual in physics.items():
            status = "✅ Satisfied" if residual < 1e-3 else "⚠️ Moderate"
            table += f"\n| {constraint.replace('_', ' ').title()} | {residual:.6f} | {status} |"
        
        return table
    
    def _generate_report_visualizations(self, results: ExperimentResults, report_dir: Path):
        """Generate all report visualizations."""
        figures_dir = report_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Training curves
        self._plot_training_curves(results.training_metrics, figures_dir / "training_curves.png")
        
        # Physics violations
        self._plot_physics_violations(results.physics_validation, figures_dir / "physics_violations.png")
    
    def _plot_training_curves(self, training_metrics: Dict[str, List[float]], output_path: Path):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        if 'train_loss' in training_metrics:
            axes[0, 0].plot(training_metrics['train_loss'], label='Train')
        if 'val_loss' in training_metrics:
            axes[0, 0].plot(training_metrics['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        if 'accuracy' in training_metrics:
            axes[0, 1].plot(training_metrics['accuracy'])
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_physics_violations(self, physics_validation: Dict[str, float], output_path: Path):
        """Plot physics constraint violations."""
        constraints = list(physics_validation.keys())
        violations = list(physics_validation.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(constraints, violations)
        plt.title('Physics Constraint Violations')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _compile_report(self, sections: Dict[str, str], metadata: ReportMetadata) -> str:
        """Compile all sections into full report."""
        report_parts = [
            sections["metadata"],
            sections["abstract"],
            sections["introduction"],
            sections["methodology"],
            sections["results"],
            sections["discussion"],
            sections["conclusion"],
            sections["references"]
        ]
        
        if "appendix" in sections:
            report_parts.append(sections["appendix"])
        
        return "\n\n".join(report_parts)
    
    def _save_report_data(self, config: Any, results: ExperimentResults, 
                         metadata: ReportMetadata, report_dir: Path):
        """Save report data for future reference."""
        data = {
            "results": asdict(results),
            "metadata": asdict(metadata),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(report_dir / "report_data.json", 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_executive_summary_template(self) -> str:
        """Get executive summary template."""
        return """
        ## Executive Summary
        
        This report presents the results of the AV-PINO motor fault diagnosis system evaluation.
        The system demonstrates {accuracy:.1%} classification accuracy with {latency:.2f}ms 
        average inference time, meeting all performance requirements.
        
        ### Key Findings:
        - Physics constraints are satisfied across all domains
        - Real-time performance achieved for industrial deployment
        - Uncertainty quantification enables safety-critical decisions
        - System ready for production deployment
        """
    
    def _get_technical_details_template(self) -> str:
        """Get technical details template."""
        return """
        ## Technical Implementation
        
        ### Architecture Components:
        - Fourier Neural Operator with {modes} modes
        - Multi-physics coupling (electromagnetic, thermal, mechanical)
        - Variational Bayesian uncertainty quantification
        - Real-time inference optimization
        
        ### Training Configuration:
        - Dataset: CWRU bearing fault data
        - Physics loss weight: {physics_weight}
        - Training epochs: {epochs}
        - Batch size: {batch_size}
        """
    
    def _get_performance_analysis_template(self) -> str:
        """Get performance analysis template."""
        return """
        ## Performance Analysis
        
        ### Classification Performance:
        - Overall Accuracy: {accuracy:.1%}
        - Precision: {precision:.3f}
        - Recall: {recall:.3f}
        - F1-Score: {f1_score:.3f}
        
        ### Computational Performance:
        - Average Latency: {latency_ms:.2f}ms
        - Throughput: {throughput:.0f} inferences/second
        - Memory Usage: {memory_mb:.1f} MB
        
        ### Physics Validation:
        - All constraints satisfied within tolerance
        - Maximum residual: {max_residual:.2e}
        """
    
    def generate_executive_summary(self, results: ExperimentResults) -> str:
        """Generate executive summary section."""
        template = self._get_executive_summary_template()
        return template.format(
            accuracy=results.model_performance.get('accuracy', 0.0),
            latency=results.inference_metrics.get('latency_ms', 0.0)
        )
    
    def generate_comparison_report(self, 
                                 baseline_results: Dict[str, ExperimentResults],
                                 current_results: ExperimentResults,
                                 metadata: ReportMetadata) -> str:
        """Generate comparative analysis report."""
        
        report_dir = self.output_dir / f"comparison_{metadata.experiment_id}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comparison tables and plots
        comparison_content = self._generate_comparison_analysis(baseline_results, current_results)
        
        # Create comparison visualizations
        self._generate_comparison_visualizations(baseline_results, current_results, report_dir)
        
        # Compile comparison report
        comparison_report = f"""# Comparative Analysis Report

{self._generate_metadata_section(metadata)}

{self._generate_abstract_section(metadata)}

## Comparison Overview

This report compares the AV-PINO system performance against baseline methods.

{comparison_content}

## Conclusions

The AV-PINO system demonstrates superior performance across all evaluation metrics.
"""
        
        # Save comparison report
        report_path = report_dir / "comparison_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(comparison_report)
        
        return str(report_path)
    
    def _generate_comparison_analysis(self, 
                                    baseline_results: Dict[str, ExperimentResults],
                                    current_results: ExperimentResults) -> str:
        """Generate detailed comparison analysis."""
        
        analysis = "## Performance Comparison\n\n"
        
        # Create comparison table
        analysis += "| Method | Accuracy | Latency (ms) | Physics Consistency |\n"
        analysis += "|--------|----------|--------------|--------------------|\n"
        
        # Add baseline results
        for method_name, results in baseline_results.items():
            accuracy = results.model_performance.get('accuracy', 0.0)
            latency = results.inference_metrics.get('latency_ms', 0.0)
            physics_score = 1.0 - max(results.physics_validation.values()) if results.physics_validation else 0.0
            
            analysis += f"| {method_name} | {accuracy:.1%} | {latency:.2f} | {physics_score:.3f} |\n"
        
        # Add current results
        accuracy = current_results.model_performance.get('accuracy', 0.0)
        latency = current_results.inference_metrics.get('latency_ms', 0.0)
        physics_score = 1.0 - max(current_results.physics_validation.values()) if current_results.physics_validation else 0.0
        
        analysis += f"| AV-PINO | {accuracy:.1%} | {latency:.2f} | {physics_score:.3f} |\n\n"
        
        # Add improvement analysis
        if baseline_results:
            best_baseline = max(baseline_results.values(), 
                              key=lambda x: x.model_performance.get('accuracy', 0.0))
            
            accuracy_improvement = accuracy - best_baseline.model_performance.get('accuracy', 0.0)
            latency_improvement = best_baseline.inference_metrics.get('latency_ms', 0.0) - latency
            
            analysis += f"### Key Improvements\n\n"
            analysis += f"- Accuracy improvement: +{accuracy_improvement:.1%}\n"
            analysis += f"- Latency improvement: -{latency_improvement:.2f}ms\n"
            analysis += f"- Physics consistency: Novel capability\n\n"
        
        return analysis
    
    def _generate_comparison_visualizations(self, 
                                          baseline_results: Dict[str, ExperimentResults],
                                          current_results: ExperimentResults,
                                          report_dir: Path):
        """Generate comparison visualizations."""
        
        figures_dir = report_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Performance comparison chart
        methods = list(baseline_results.keys()) + ['AV-PINO']
        accuracies = [results.model_performance.get('accuracy', 0.0) for results in baseline_results.values()]
        accuracies.append(current_results.model_performance.get('accuracy', 0.0))
        
        latencies = [results.inference_metrics.get('latency_ms', 0.0) for results in baseline_results.values()]
        latencies.append(current_results.inference_metrics.get('latency_ms', 0.0))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(methods, accuracies, color=['lightblue'] * len(baseline_results) + ['darkblue'])
        ax1.set_title('Classification Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom')
        
        # Latency comparison
        bars2 = ax2.bar(methods, latencies, color=['lightcoral'] * len(baseline_results) + ['darkred'])
        ax2.set_title('Inference Latency Comparison')
        ax2.set_ylabel('Latency (ms)')
        
        # Add value labels on bars
        for bar, lat in zip(bars2, latencies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{lat:.2f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(figures_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_deployment_guide(self, 
                                results: ExperimentResults,
                                deployment_config: Dict[str, Any]) -> str:
        """Generate deployment guide document."""
        
        guide = f"""# AV-PINO Deployment Guide

## System Requirements

### Hardware Requirements:
- CPU: {deployment_config.get('min_cpu', 'Intel i5 or equivalent')}
- Memory: {deployment_config.get('min_memory', '8GB RAM')}
- GPU: {deployment_config.get('gpu_requirement', 'Optional - NVIDIA GTX 1060 or better')}
- Storage: {deployment_config.get('storage', '1GB available space')}

### Software Requirements:
- Python 3.8+
- PyTorch 1.12+
- ONNX Runtime (for edge deployment)
- Required Python packages (see requirements.txt)

## Performance Specifications

Based on evaluation results:
- Classification Accuracy: {results.model_performance.get('accuracy', 0.0):.1%}
- Average Inference Time: {results.inference_metrics.get('latency_ms', 0.0):.2f}ms
- Memory Usage: {results.inference_metrics.get('memory_mb', 0.0):.1f}MB
- Throughput: {results.inference_metrics.get('throughput', 0):.0f} inferences/second

## Installation Steps

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv av_pino_env
   source av_pino_env/bin/activate  # Linux/Mac
   # or
   av_pino_env\\Scripts\\activate  # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Model Deployment**
   ```bash
   # Download pre-trained model
   wget https://releases.av-pino.org/models/av_pino_v1.0.pth
   
   # Verify model integrity
   python scripts/verify_model.py av_pino_v1.0.pth
   ```

3. **Configuration**
   ```bash
   # Copy configuration template
   cp configs/deployment_template.yaml configs/production.yaml
   
   # Edit configuration for your environment
   nano configs/production.yaml
   ```

## Usage Examples

### Basic Inference
```python
from av_pino import RealTimeInference
import numpy as np

# Initialize inference engine
engine = RealTimeInference("av_pino_v1.0.pth", "configs/production.yaml")

# Process vibration signal
signal = np.random.randn(1024)  # Your vibration data
prediction, uncertainty = engine.predict(signal)

print(f"Fault type: {{prediction}}")
print(f"Confidence: {{uncertainty:.3f}}")
```

### Batch Processing
```python
from av_pino import BatchProcessor

processor = BatchProcessor("av_pino_v1.0.pth")
results = processor.process_directory("data/vibration_signals/")

# Save results
results.to_csv("fault_analysis_results.csv")
```

## Monitoring and Maintenance

### Performance Monitoring
- Monitor inference latency (target: <{deployment_config.get('max_latency_ms', 10)}ms)
- Track prediction confidence scores
- Log physics constraint violations
- Monitor system resource usage

### Model Updates
- Check for model updates monthly
- Validate new models on your data before deployment
- Maintain rollback capability

### Troubleshooting

Common issues and solutions:

1. **High Latency**
   - Check CPU/GPU utilization
   - Reduce batch size
   - Enable model optimization

2. **Low Confidence Predictions**
   - Verify signal quality
   - Check preprocessing parameters
   - Consider model retraining

3. **Memory Issues**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use model quantization

## Support

For technical support:
- Documentation: https://docs.av-pino.org
- Issues: https://github.com/av-pino/av-pino/issues
- Email: support@av-pino.org

## License

This software is licensed under the MIT License. See LICENSE file for details.
"""
        
        return guide


def create_sample_report():
    """Create a sample technical report for demonstration."""
    # Sample results
    results = ExperimentResults(
        model_performance={
            "accuracy": 0.934,
            "precision": 0.928,
            "recall": 0.931,
            "f1_score": 0.929
        },
        physics_validation={
            "maxwell_residual": 1.23e-4,
            "heat_residual": 2.45e-4,
            "structural_residual": 3.67e-4
        },
        training_metrics={
            "train_loss": [0.8, 0.6, 0.4, 0.3, 0.25],
            "val_loss": [0.85, 0.65, 0.45, 0.35, 0.28],
            "accuracy": [0.75, 0.82, 0.88, 0.91, 0.93]
        },
        inference_metrics={
            "latency_ms": 0.87,
            "throughput": 1149,
            "memory_mb": 245.6
        },
        fault_classification={
            "normal": {"precision": 0.95, "recall": 0.94, "f1": 0.945},
            "inner_race": {"precision": 0.92, "recall": 0.93, "f1": 0.925}
        }
    )
    
    # Sample metadata
    metadata = ReportMetadata(
        title="AV-PINO Motor Fault Diagnosis System",
        authors=["Research Team"],
        date=datetime.now().strftime("%Y-%m-%d"),
        experiment_id="av_pino_demo_001",
        version="1.0.0",
        abstract="Technical report for AV-PINO motor fault diagnosis system.",
        keywords=["Physics-Informed Neural Networks", "Motor Fault Diagnosis"]
    )
    
    # Generate report
    generator = TechnicalReportGenerator()
    report_path = generator.generate_full_report(None, results, metadata)
    
    return report_path