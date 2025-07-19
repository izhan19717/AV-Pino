# Inference API Reference

This document provides detailed API documentation for the inference components of the AV-PINO system.

## RealTimeInference

Real-time inference engine optimized for production deployment.

### Class Definition

```python
class RealTimeInference:
    """Real-time inference engine for AV-PINO models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device = None,
        optimization_level: str = "standard"
    )
```

### Methods

#### `predict(signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Perform real-time prediction with uncertainty quantification.

**Parameters:**
- `signal` (torch.Tensor): Input vibration signal [batch, sequence, channels]

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor]`: Predictions and uncertainties

**Example:**
```python
from src.inference.realtime_inference import RealTimeInference

inference_engine = RealTimeInference(model, config, device)
signal = torch.randn(1, 1024, 1)  # Single signal
prediction, uncertainty = inference_engine.predict(signal)

print(f"Predicted class: {torch.argmax(prediction)}")
print(f"Confidence: {torch.softmax(prediction, dim=1).max():.3f}")
print(f"Uncertainty: {uncertainty.item():.4f}")
```

#### `predict_batch(signals: torch.Tensor) -> Dict[str, torch.Tensor]`

Batch prediction for multiple signals.

**Parameters:**
- `signals` (torch.Tensor): Batch of input signals

**Returns:**
- `Dict[str, torch.Tensor]`: Batch predictions and metadata

#### `optimize_for_edge(model: nn.Module) -> nn.Module`

Optimize model for edge deployment.

**Parameters:**
- `model` (nn.Module): Model to optimize

**Returns:**
- `nn.Module`: Optimized model

**Example:**
```python
optimized_model = inference_engine.optimize_for_edge(trained_model)
# Model is now optimized for faster inference
```

#### `profile_performance(test_signals: List[torch.Tensor]) -> Dict[str, float]`

Profile inference performance metrics.

**Parameters:**
- `test_signals` (List[torch.Tensor]): Test signals for profiling

**Returns:**
- `Dict[str, float]`: Performance metrics

**Example:**
```python
test_data = [torch.randn(1, 1024, 1) for _ in range(100)]
metrics = inference_engine.profile_performance(test_data)

print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
print(f"Throughput: {metrics['throughput_hz']:.0f} Hz")
print(f"Memory usage: {metrics['memory_mb']:.1f} MB")
```

## FaultClassifier

Specialized fault classification with interpretability.

### Class Definition

```python
class FaultClassifier:
    """Fault classifier with interpretability features."""
    
    def __init__(
        self,
        model: nn.Module,
        fault_classes: List[str],
        confidence_threshold: float = 0.8
    )
```

### Methods

#### `classify_fault(signal: torch.Tensor) -> Dict[str, Any]`

Classify fault type with confidence and explanation.

**Parameters:**
- `signal` (torch.Tensor): Input vibration signal

**Returns:**
- `Dict[str, Any]`: Classification results with interpretability

**Example:**
```python
from src.inference.fault_classifier import FaultClassifier

classifier = FaultClassifier(
    model=trained_model,
    fault_classes=['normal', 'inner_race', 'outer_race', 'ball'],
    confidence_threshold=0.8
)

result = classifier.classify_fault(signal)
print(f"Fault type: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Is reliable: {result['is_reliable']}")
print(f"Key features: {result['important_features']}")
```

#### `explain_prediction(signal: torch.Tensor) -> Dict[str, Any]`

Generate explanation for prediction.

**Parameters:**
- `signal` (torch.Tensor): Input signal

**Returns:**
- `Dict[str, Any]`: Explanation data

#### `get_feature_importance(signal: torch.Tensor) -> torch.Tensor`

Get feature importance scores.

**Parameters:**
- `signal` (torch.Tensor): Input signal

**Returns:**
- `torch.Tensor`: Feature importance scores

## ModelOptimizer

Model optimization for deployment.

### Class Definition

```python
class ModelOptimizer:
    """Optimizes models for deployment scenarios."""
    
    def __init__(self, optimization_target: str = "latency")
```

### Methods

#### `quantize_model(model: nn.Module, calibration_data: DataLoader) -> nn.Module`

Apply quantization for faster inference.

**Parameters:**
- `model` (nn.Module): Model to quantize
- `calibration_data` (DataLoader): Calibration dataset

**Returns:**
- `nn.Module`: Quantized model

**Example:**
```python
from src.inference.model_optimizer import ModelOptimizer

optimizer = ModelOptimizer(optimization_target="latency")
quantized_model = optimizer.quantize_model(model, calibration_loader)

# Test quantized model performance
original_time = profile_inference_time(model, test_data)
quantized_time = profile_inference_time(quantized_model, test_data)
speedup = original_time / quantized_time
print(f"Quantization speedup: {speedup:.2f}x")
```

#### `prune_model(model: nn.Module, sparsity: float = 0.3) -> nn.Module`

Apply structured pruning to reduce model size.

**Parameters:**
- `model` (nn.Module): Model to prune
- `sparsity` (float): Target sparsity ratio

**Returns:**
- `nn.Module`: Pruned model

#### `export_onnx(model: nn.Module, sample_input: torch.Tensor, output_path: str)`

Export model to ONNX format.

**Parameters:**
- `model` (nn.Module): Model to export
- `sample_input` (torch.Tensor): Sample input for tracing
- `output_path` (str): Output file path

**Example:**
```python
sample_input = torch.randn(1, 1024, 1)
optimizer.export_onnx(model, sample_input, "model.onnx")
```

## Usage Examples

### Production Inference Pipeline

```python
import torch
from src.inference.realtime_inference import RealTimeInference
from src.inference.fault_classifier import FaultClassifier
from src.data.signal_processor import SignalProcessor

class ProductionInferencePipeline:
    """Complete production inference pipeline."""
    
    def __init__(self, model_path: str, config_path: str):
        # Load model and config
        self.model = torch.load(model_path)
        self.config = load_config(config_path)
        
        # Setup components
        self.signal_processor = SignalProcessor()
        self.inference_engine = RealTimeInference(self.model, self.config)
        self.fault_classifier = FaultClassifier(
            self.model,
            fault_classes=['normal', 'inner_race', 'outer_race', 'ball']
        )
        
        # Optimize for production
        self.model = self.inference_engine.optimize_for_edge(self.model)
    
    def process_signal(self, raw_signal: np.ndarray) -> Dict[str, Any]:
        """Process raw signal and return fault diagnosis."""
        
        # Preprocess signal
        clean_signal = self.signal_processor.preprocess_signal(raw_signal)
        
        # Convert to tensor
        signal_tensor = torch.FloatTensor(clean_signal).unsqueeze(0).unsqueeze(-1)
        
        # Classify fault
        result = self.fault_classifier.classify_fault(signal_tensor)
        
        # Add timing information
        start_time = time.time()
        prediction, uncertainty = self.inference_engine.predict(signal_tensor)
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'fault_type': result['predicted_class'],
            'confidence': result['confidence'],
            'uncertainty': uncertainty.item(),
            'is_reliable': result['is_reliable'],
            'inference_time_ms': inference_time,
            'timestamp': time.time()
        }

# Usage
pipeline = ProductionInferencePipeline("model.pth", "config.yaml")

# Process incoming signal
raw_vibration_data = np.random.randn(2048)  # Simulated sensor data
diagnosis = pipeline.process_signal(raw_vibration_data)

print(f"Fault diagnosis: {diagnosis['fault_type']}")
print(f"Confidence: {diagnosis['confidence']:.3f}")
print(f"Processing time: {diagnosis['inference_time_ms']:.2f}ms")
```

### Batch Processing for Historical Data

```python
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

class BatchInferenceProcessor:
    """Batch processing for historical data analysis."""
    
    def __init__(self, model_path: str, config_path: str):
        self.pipeline = ProductionInferencePipeline(model_path, config_path)
    
    def process_batch(self, signal_batch: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process batch of signals."""
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.pipeline.process_signal, signal)
                for signal in signal_batch
            ]
            
            for future in futures:
                results.append(future.result())
        
        return results
    
    def analyze_historical_data(self, data_path: str) -> pd.DataFrame:
        """Analyze historical vibration data."""
        
        # Load historical data
        historical_data = load_historical_signals(data_path)
        
        # Process in batches
        batch_size = 32
        all_results = []
        
        for i in range(0, len(historical_data), batch_size):
            batch = historical_data[i:i+batch_size]
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)
            
            print(f"Processed {min(i+batch_size, len(historical_data))}/{len(historical_data)} signals")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_results)
        
        # Add analysis metrics
        df['fault_detected'] = df['fault_type'] != 'normal'
        df['high_confidence'] = df['confidence'] > 0.8
        df['reliable_prediction'] = df['is_reliable']
        
        return df

# Usage
batch_processor = BatchInferenceProcessor("model.pth", "config.yaml")
results_df = batch_processor.analyze_historical_data("historical_data/")

# Analyze results
print(f"Total signals processed: {len(results_df)}")
print(f"Faults detected: {results_df['fault_detected'].sum()}")
print(f"High confidence predictions: {results_df['high_confidence'].sum()}")
print(f"Average inference time: {results_df['inference_time_ms'].mean():.2f}ms")
```

### Edge Deployment with ONNX

```python
import onnxruntime as ort
import numpy as np

class ONNXInferenceEngine:
    """ONNX-based inference engine for edge deployment."""
    
    def __init__(self, onnx_model_path: str):
        # Load ONNX model
        self.session = ort.InferenceSession(onnx_model_path)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Loaded ONNX model: {onnx_model_path}")
        print(f"Input: {self.input_name}")
        print(f"Outputs: {self.output_names}")
    
    def predict(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference using ONNX Runtime."""
        
        # Ensure correct input shape
        if signal.ndim == 1:
            signal = signal.reshape(1, -1, 1)
        elif signal.ndim == 2:
            signal = signal.reshape(1, signal.shape[0], signal.shape[1])
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: signal.astype(np.float32)}
        )
        
        return {name: output for name, output in zip(self.output_names, outputs)}
    
    def benchmark_performance(self, test_signals: List[np.ndarray], num_runs: int = 100):
        """Benchmark inference performance."""
        
        times = []
        
        for _ in range(num_runs):
            signal = np.random.choice(test_signals)
            
            start_time = time.time()
            _ = self.predict(signal)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'avg_latency_ms': np.mean(times),
            'std_latency_ms': np.std(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'throughput_hz': 1000 / np.mean(times)
        }

# Export PyTorch model to ONNX
def export_to_onnx(pytorch_model, sample_input, output_path):
    """Export PyTorch model to ONNX format."""
    
    pytorch_model.eval()
    
    torch.onnx.export(
        pytorch_model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['predictions', 'uncertainties'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'predictions': {0: 'batch_size'},
            'uncertainties': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")

# Usage
sample_input = torch.randn(1, 1024, 1)
export_to_onnx(trained_model, sample_input, "av_pino_model.onnx")

# Load and test ONNX model
onnx_engine = ONNXInferenceEngine("av_pino_model.onnx")

# Test prediction
test_signal = np.random.randn(1024)
result = onnx_engine.predict(test_signal)
print(f"ONNX prediction shape: {result['predictions'].shape}")

# Benchmark performance
test_signals = [np.random.randn(1024) for _ in range(10)]
perf_metrics = onnx_engine.benchmark_performance(test_signals)
print(f"ONNX inference latency: {perf_metrics['avg_latency_ms']:.2f}ms")
```

## Error Handling and Monitoring

### Robust Inference with Error Handling

```python
import logging
from typing import Optional

class RobustInferenceEngine:
    """Inference engine with comprehensive error handling."""
    
    def __init__(self, model_path: str, config_path: str):
        self.logger = logging.getLogger(__name__)
        
        try:
            self.model = torch.load(model_path, map_location='cpu')
            self.config = load_config(config_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def safe_predict(self, signal: np.ndarray) -> Optional[Dict[str, Any]]:
        """Safe prediction with comprehensive error handling."""
        
        try:
            # Input validation
            if not isinstance(signal, np.ndarray):
                raise ValueError("Input must be numpy array")
            
            if signal.size == 0:
                raise ValueError("Input signal is empty")
            
            if np.isnan(signal).any() or np.isinf(signal).any():
                raise ValueError("Input contains NaN or infinite values")
            
            # Preprocess signal
            if signal.ndim == 1:
                signal = signal.reshape(1, -1, 1)
            
            # Convert to tensor
            signal_tensor = torch.FloatTensor(signal).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(signal_tensor)
                
                # Validate outputs
                predictions = outputs.get('predictions')
                if predictions is None:
                    raise RuntimeError("Model did not return predictions")
                
                if torch.isnan(predictions).any():
                    raise RuntimeError("Model returned NaN predictions")
                
                # Extract results
                predicted_class = torch.argmax(predictions, dim=1).cpu().numpy()[0]
                confidence = torch.softmax(predictions, dim=1).max().cpu().numpy()
                
                return {
                    'predicted_class': int(predicted_class),
                    'confidence': float(confidence),
                    'status': 'success',
                    'timestamp': time.time()
                }
                
        except ValueError as e:
            self.logger.warning(f"Input validation error: {e}")
            return {
                'status': 'input_error',
                'error': str(e),
                'timestamp': time.time()
            }
            
        except RuntimeError as e:
            self.logger.error(f"Model inference error: {e}")
            return {
                'status': 'inference_error',
                'error': str(e),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return {
                'status': 'unknown_error',
                'error': str(e),
                'timestamp': time.time()
            }

# Usage with monitoring
robust_engine = RobustInferenceEngine("model.pth", "config.yaml")

# Process signals with error tracking
success_count = 0
error_count = 0

for signal in test_signals:
    result = robust_engine.safe_predict(signal)
    
    if result['status'] == 'success':
        success_count += 1
        print(f"Prediction: {result['predicted_class']}, Confidence: {result['confidence']:.3f}")
    else:
        error_count += 1
        print(f"Error: {result['error']}")

print(f"Success rate: {success_count/(success_count + error_count):.1%}")
```

## Best Practices

### Inference Optimization

1. **Model Optimization**: Use quantization, pruning, and ONNX export
2. **Batch Processing**: Process multiple signals together when possible
3. **Memory Management**: Use torch.no_grad() and clear cache regularly
4. **Error Handling**: Implement comprehensive error handling and logging

### Production Deployment

1. **Monitoring**: Track inference latency, accuracy, and error rates
2. **Fallback**: Implement fallback mechanisms for model failures
3. **Versioning**: Use model versioning for safe deployments
4. **Testing**: Comprehensive testing on target hardware

### Example Production Configuration

```python
# Production inference configuration
PRODUCTION_CONFIG = {
    'model': {
        'optimization_level': 'aggressive',
        'use_quantization': True,
        'use_pruning': False,
        'export_onnx': True
    },
    'inference': {
        'batch_size': 1,
        'max_latency_ms': 10,
        'confidence_threshold': 0.8,
        'uncertainty_threshold': 0.1
    },
    'monitoring': {
        'log_predictions': True,
        'track_performance': True,
        'alert_on_errors': True
    },
    'fallback': {
        'enable_fallback': True,
        'fallback_model_path': 'fallback_model.onnx',
        'max_consecutive_errors': 5
    }
}
```