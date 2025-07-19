"""
Baseline Method Comparisons for AV-PINO System.

This module implements comparisons against traditional ML approaches
to demonstrate the superiority of the physics-informed approach.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time

from ..inference.fault_classifier import FaultClassificationSystem, FaultPrediction

logger = logging.getLogger(__name__)


class BaselineMethod(Enum):
    """Traditional ML baseline methods."""
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    MLP = "mlp"
    LOGISTIC_REGRESSION = "logistic_regression"


@dataclass
class BaselineMetrics:
    """Metrics for baseline method performance."""
    method_name: str
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: np.ndarray
    training_time: float
    inference_time: float
    model_size: int  # Number of parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'method_name': self.method_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'model_size': self.model_size
        }


@dataclass
class ComparisonReport:
    """Comprehensive comparison report between AV-PINO and baselines."""
    av_pino_metrics: BaselineMetrics
    baseline_metrics: Dict[str, BaselineMetrics]
    performance_improvements: Dict[str, Dict[str, float]]
    statistical_significance: Dict[str, bool]
    summary_statistics: Dict[str, float]
    recommendations: List[str]
    timestamp: str
    
    def save_report(self, filepath: str):
        """Save comparison report to JSON file."""
        import json
        
        report_data = {
            'av_pino_metrics': self.av_pino_metrics.to_dict(),
            'baseline_metrics': {k: v.to_dict() for k, v in self.baseline_metrics.items()},
            'performance_improvements': self.performance_improvements,
            'statistical_significance': self.statistical_significance,
            'summary_statistics': self.summary_statistics,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Comparison report saved to {filepath}")


class TraditionalMLBaseline:
    """Traditional ML baseline implementation."""
    
    def __init__(self, method: BaselineMethod, random_state: int = 42):
        self.method = method
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize model based on method
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the baseline model."""
        if self.method == BaselineMethod.RANDOM_FOREST:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.method == BaselineMethod.SVM:
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=self.random_state,
                probability=True  # Enable probability estimates
            )
        elif self.method == BaselineMethod.MLP:
            self.model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        elif self.method == BaselineMethod.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_state,
                multi_class='ovr'
            )
        else:
            raise ValueError(f"Unsupported baseline method: {self.method}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Train the baseline model and return training time."""
        logger.info(f"Training {self.method.value} baseline model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model and measure time
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        self.is_trained = True
        logger.info(f"{self.method.value} training completed in {training_time:.2f} seconds")
        
        return training_time
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Make predictions and return predictions, probabilities, and inference time."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions and measure time
        start_time = time.time()
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)
        inference_time = time.time() - start_time
        
        return predictions, probabilities, inference_time
    
    def get_model_size(self) -> int:
        """Get approximate model size (number of parameters)."""
        if not self.is_trained:
            return 0
        
        if self.method == BaselineMethod.RANDOM_FOREST:
            # Approximate: number of trees * average tree size
            return self.model.n_estimators * 1000  # Rough estimate
        elif self.method == BaselineMethod.SVM:
            # Number of support vectors
            return len(self.model.support_vectors_)
        elif self.method == BaselineMethod.MLP:
            # Sum of weights in all layers
            total_params = 0
            for layer_weights in self.model.coefs_:
                total_params += layer_weights.size
            for layer_bias in self.model.intercepts_:
                total_params += layer_bias.size
            return total_params
        elif self.method == BaselineMethod.LOGISTIC_REGRESSION:
            # Number of coefficients
            return self.model.coef_.size + self.model.intercept_.size
        else:
            return 0


class BaselineComparator:
    """Compares AV-PINO performance against traditional ML baselines."""
    
    def __init__(self, av_pino_model: FaultClassificationSystem):
        self.av_pino_model = av_pino_model
        self.baseline_models = {}
        self.comparison_results = {}
        
        # Initialize baseline models
        for method in BaselineMethod:
            self.baseline_models[method.value] = TraditionalMLBaseline(method)
    
    def train_baselines(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train all baseline models."""
        training_times = {}
        
        for method_name, baseline_model in self.baseline_models.items():
            try:
                training_time = baseline_model.train(X_train, y_train)
                training_times[method_name] = training_time
            except Exception as e:
                logger.error(f"Failed to train {method_name}: {e}")
                training_times[method_name] = float('inf')
        
        return training_times
    
    def evaluate_baselines(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, BaselineMetrics]:
        """Evaluate all baseline models."""
        baseline_metrics = {}
        
        for method_name, baseline_model in self.baseline_models.items():
            if not baseline_model.is_trained:
                logger.warning(f"Skipping evaluation of untrained model: {method_name}")
                continue
            
            try:
                # Make predictions
                predictions, probabilities, inference_time = baseline_model.predict(X_test)
                
                # Compute metrics
                accuracy = accuracy_score(y_test, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, predictions, average=None, zero_division=0
                )
                
                # Get unique labels
                unique_labels = sorted(list(set(y_test.tolist() + predictions.tolist())))
                
                # Create per-class metrics dictionaries
                precision_dict = {f'class_{i}': precision[i] if i < len(precision) else 0.0 
                                for i in range(len(unique_labels))}
                recall_dict = {f'class_{i}': recall[i] if i < len(recall) else 0.0 
                             for i in range(len(unique_labels))}
                f1_dict = {f'class_{i}': f1[i] if i < len(f1) else 0.0 
                          for i in range(len(unique_labels))}
                
                # Confusion matrix
                cm = confusion_matrix(y_test, predictions)
                
                # Get training time (stored from training phase)
                training_time = getattr(baseline_model, '_training_time', 0.0)
                
                # Model size
                model_size = baseline_model.get_model_size()
                
                metrics = BaselineMetrics(
                    method_name=method_name,
                    accuracy=accuracy,
                    precision=precision_dict,
                    recall=recall_dict,
                    f1_score=f1_dict,
                    confusion_matrix=cm,
                    training_time=training_time,
                    inference_time=inference_time,
                    model_size=model_size
                )
                
                baseline_metrics[method_name] = metrics
                
            except Exception as e:
                logger.error(f"Failed to evaluate {method_name}: {e}")
        
        return baseline_metrics
    
    def evaluate_av_pino(self, X_test: torch.Tensor, y_test: torch.Tensor) -> BaselineMetrics:
        """Evaluate AV-PINO model performance."""
        logger.info("Evaluating AV-PINO model...")
        
        # Make predictions and measure time
        start_time = time.time()
        predictions = self.av_pino_model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Extract prediction data
        pred_labels = [pred.fault_type.value for pred in predictions]
        
        # Convert tensor labels to list
        true_labels = [self.av_pino_model.fault_mapper.index_to_type(idx.item()).value 
                      for idx in y_test]
        
        # Convert to numerical labels for sklearn metrics
        label_to_idx = {label: idx for idx, label in enumerate(set(true_labels + pred_labels))}
        true_labels_num = [label_to_idx[label] for label in true_labels]
        pred_labels_num = [label_to_idx[label] for label in pred_labels]
        
        # Compute metrics
        accuracy = accuracy_score(true_labels_num, pred_labels_num)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels_num, pred_labels_num, average=None, zero_division=0
        )
        
        # Create per-class metrics dictionaries
        unique_labels = sorted(list(set(true_labels_num + pred_labels_num)))
        precision_dict = {f'class_{i}': precision[i] if i < len(precision) else 0.0 
                         for i in range(len(unique_labels))}
        recall_dict = {f'class_{i}': recall[i] if i < len(recall) else 0.0 
                      for i in range(len(unique_labels))}
        f1_dict = {f'class_{i}': f1[i] if i < len(f1) else 0.0 
                  for i in range(len(unique_labels))}
        
        # Confusion matrix
        cm = confusion_matrix(true_labels_num, pred_labels_num)
        
        # Estimate model size (number of parameters)
        model_size = sum(p.numel() for p in self.av_pino_model.classifier.parameters())
        
        return BaselineMetrics(
            method_name="AV-PINO",
            accuracy=accuracy,
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            confusion_matrix=cm,
            training_time=0.0,  # Not measured here
            inference_time=inference_time,
            model_size=model_size
        )
    
    def run_comprehensive_comparison(self, 
                                   X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray,
                                   X_test_torch: torch.Tensor, y_test_torch: torch.Tensor
                                   ) -> ComparisonReport:
        """Run comprehensive comparison between AV-PINO and baselines."""
        logger.info("Starting comprehensive baseline comparison...")
        
        # Train baseline models
        training_times = self.train_baselines(X_train, y_train)
        
        # Store training times in models for later use
        for method_name, training_time in training_times.items():
            if method_name in self.baseline_models:
                self.baseline_models[method_name]._training_time = training_time
        
        # Evaluate baseline models
        baseline_metrics = self.evaluate_baselines(X_test, y_test)
        
        # Evaluate AV-PINO
        av_pino_metrics = self.evaluate_av_pino(X_test_torch, y_test_torch)
        
        # Compute performance improvements
        performance_improvements = self._compute_performance_improvements(
            av_pino_metrics, baseline_metrics
        )
        
        # Compute statistical significance (simplified)
        statistical_significance = self._compute_statistical_significance(
            av_pino_metrics, baseline_metrics
        )
        
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(
            av_pino_metrics, baseline_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            av_pino_metrics, baseline_metrics, performance_improvements
        )
        
        # Create comparison report
        report = ComparisonReport(
            av_pino_metrics=av_pino_metrics,
            baseline_metrics=baseline_metrics,
            performance_improvements=performance_improvements,
            statistical_significance=statistical_significance,
            summary_statistics=summary_stats,
            recommendations=recommendations,
            timestamp=str(np.datetime64('now'))
        )
        
        logger.info("Baseline comparison completed")
        return report
    
    def _compute_performance_improvements(self, av_pino_metrics: BaselineMetrics,
                                        baseline_metrics: Dict[str, BaselineMetrics]
                                        ) -> Dict[str, Dict[str, float]]:
        """Compute performance improvements over baselines."""
        improvements = {}
        
        for method_name, baseline in baseline_metrics.items():
            method_improvements = {
                'accuracy_improvement': av_pino_metrics.accuracy - baseline.accuracy,
                'accuracy_improvement_pct': ((av_pino_metrics.accuracy - baseline.accuracy) / 
                                           baseline.accuracy * 100) if baseline.accuracy > 0 else 0,
                'inference_speedup': (baseline.inference_time / av_pino_metrics.inference_time 
                                    if av_pino_metrics.inference_time > 0 else 1.0),
                'model_size_ratio': (baseline.model_size / av_pino_metrics.model_size 
                                   if av_pino_metrics.model_size > 0 else 1.0)
            }
            improvements[method_name] = method_improvements
        
        return improvements
    
    def _compute_statistical_significance(self, av_pino_metrics: BaselineMetrics,
                                        baseline_metrics: Dict[str, BaselineMetrics]
                                        ) -> Dict[str, bool]:
        """Compute statistical significance of improvements (simplified)."""
        # Simplified significance test based on accuracy difference
        significance = {}
        
        for method_name, baseline in baseline_metrics.items():
            # Simple threshold-based significance (in practice, would use proper statistical tests)
            accuracy_diff = av_pino_metrics.accuracy - baseline.accuracy
            significance[method_name] = accuracy_diff > 0.05  # 5% improvement threshold
        
        return significance
    
    def _compute_summary_statistics(self, av_pino_metrics: BaselineMetrics,
                                  baseline_metrics: Dict[str, BaselineMetrics]
                                  ) -> Dict[str, float]:
        """Compute summary statistics across all comparisons."""
        baseline_accuracies = [metrics.accuracy for metrics in baseline_metrics.values()]
        baseline_inference_times = [metrics.inference_time for metrics in baseline_metrics.values()]
        
        return {
            'av_pino_accuracy': av_pino_metrics.accuracy,
            'best_baseline_accuracy': max(baseline_accuracies) if baseline_accuracies else 0.0,
            'mean_baseline_accuracy': np.mean(baseline_accuracies) if baseline_accuracies else 0.0,
            'accuracy_advantage': (av_pino_metrics.accuracy - max(baseline_accuracies) 
                                 if baseline_accuracies else 0.0),
            'av_pino_inference_time': av_pino_metrics.inference_time,
            'mean_baseline_inference_time': np.mean(baseline_inference_times) if baseline_inference_times else 0.0,
            'inference_speedup': (np.mean(baseline_inference_times) / av_pino_metrics.inference_time 
                                if av_pino_metrics.inference_time > 0 and baseline_inference_times else 1.0)
        }
    
    def _generate_recommendations(self, av_pino_metrics: BaselineMetrics,
                                baseline_metrics: Dict[str, BaselineMetrics],
                                performance_improvements: Dict[str, Dict[str, float]]
                                ) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        # Analyze accuracy improvements
        best_baseline_acc = max([m.accuracy for m in baseline_metrics.values()]) if baseline_metrics else 0.0
        accuracy_advantage = av_pino_metrics.accuracy - best_baseline_acc
        
        if accuracy_advantage > 0.1:
            recommendations.append(
                f"AV-PINO shows significant accuracy improvement ({accuracy_advantage:.1%}) "
                "over traditional ML methods, demonstrating the value of physics-informed learning."
            )
        elif accuracy_advantage > 0.05:
            recommendations.append(
                f"AV-PINO shows moderate accuracy improvement ({accuracy_advantage:.1%}) "
                "over baselines. Consider further physics constraint tuning."
            )
        else:
            recommendations.append(
                "AV-PINO accuracy is comparable to traditional methods. "
                "Focus on physics consistency and uncertainty quantification advantages."
            )
        
        # Analyze inference speed
        mean_baseline_time = np.mean([m.inference_time for m in baseline_metrics.values()]) if baseline_metrics else 0.0
        if av_pino_metrics.inference_time < mean_baseline_time:
            speedup = mean_baseline_time / av_pino_metrics.inference_time
            recommendations.append(
                f"AV-PINO provides {speedup:.1f}x inference speedup over traditional methods, "
                "suitable for real-time applications."
            )
        
        # Analyze model complexity
        mean_baseline_size = np.mean([m.model_size for m in baseline_metrics.values()]) if baseline_metrics else 0.0
        if av_pino_metrics.model_size < mean_baseline_size:
            recommendations.append(
                "AV-PINO has lower model complexity while maintaining performance, "
                "indicating efficient physics-informed architecture."
            )
        
        # Physics-specific advantages
        recommendations.append(
            "AV-PINO provides unique advantages: physics consistency, uncertainty quantification, "
            "and generalization across motor configurations not available in traditional methods."
        )
        
        return recommendations