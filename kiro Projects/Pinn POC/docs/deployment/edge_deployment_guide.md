# Edge Hardware Deployment Guide

This guide covers deploying the AV-PINO Motor Fault Diagnosis system on edge hardware for real-time inference.

## Supported Edge Platforms

### NVIDIA Jetson Series
- Jetson AGX Xavier
- Jetson Xavier NX
- Jetson Nano
- Jetson Orin

### Intel Edge Devices
- Intel NUC with Neural Compute Stick
- Intel Movidius VPU
- Intel OpenVINO compatible devices

### ARM-based Devices
- Raspberry Pi 4 (with limitations)
- NVIDIA Jetson (ARM64)
- Custom ARM SoCs

## Prerequisites

### Hardware Requirements
- **Minimum**: 4GB RAM, ARM64 or x86_64 processor
- **Recommended**: 8GB+ RAM, GPU acceleration
- **Storage**: 16GB+ available space
- **Network**: Ethernet or WiFi for data transmission

### Software Requirements
- Ubuntu 18.04+ or compatible Linux distribution
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- Docker (optional but recommended)

## Installation Steps

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/av-pino/motor-fault-diagnosis.git
cd motor-fault-diagnosis

# Run setup script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

### 2. Model Optimization for Edge

```python
from src.inference.realtime_inference import RealTimeInference
from src.config.experiment_config import ExperimentManager

# Load configuration
config_manager = ExperimentManager()
config = config_manager.load_config("configs/edge_deployment.yaml")

# Initialize inference engine
inference_engine = RealTimeInference(config)

# Optimize model for edge deployment
optimized_model = inference_engine.optimize_for_edge(
    quantization=True,
    pruning=True,
    onnx_export=True
)
```

### 3. Performance Validation

```bash
# Run performance benchmarks
python -m src.validation.benchmarking_suite --config configs/edge_deployment.yaml --profile-hardware

# Validate inference latency
python -m src.inference.realtime_inference --benchmark --target-latency 1ms
```

## Configuration for Edge Deployment

### Edge-Specific Configuration (`configs/edge_deployment.yaml`)

```yaml
# Edge deployment configuration
reproducibility:
  seed: 42
  deterministic: true

model:
  architecture: "AGT-NO-Lite"  # Lightweight version
  modes: 8                     # Reduced complexity
  width: 32
  layers: 2

inference:
  batch_size: 1
  uncertainty_samples: 50      # Reduced for speed
  confidence_threshold: 0.8
  optimization:
    quantization: true         # Enable INT8 quantization
    pruning: true             # Enable model pruning
    onnx_export: true         # Export to ONNX format

hardware:
  device: "cuda"              # or "cpu" for CPU-only
  memory_limit: "4GB"
  target_latency: "1ms"
  power_budget: "15W"
```

## Docker Deployment

### Build Docker Image

```dockerfile
# Dockerfile for edge deployment
FROM nvcr.io/nvidia/pytorch:22.08-py3

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Install package
RUN pip install -e .

# Expose inference port
EXPOSE 8080

# Run inference server
CMD ["python", "-m", "src.inference.realtime_inference", "--serve", "--port", "8080"]
```

### Deploy with Docker

```bash
# Build image
docker build -t av-pino-edge .

# Run container
docker run -d \
  --name av-pino-inference \
  --gpus all \
  -p 8080:8080 \
  -v /data:/app/data \
  av-pino-edge
```

## Performance Optimization

### Model Compression Techniques

1. **Quantization**: Convert FP32 to INT8
   ```python
   # Enable quantization in config
   config.inference.optimization.quantization = True
   ```

2. **Pruning**: Remove redundant parameters
   ```python
   # Enable pruning in config
   config.inference.optimization.pruning = True
   ```

3. **Knowledge Distillation**: Train smaller student model
   ```python
   from src.training.model_compression import KnowledgeDistillation
   
   distiller = KnowledgeDistillation(teacher_model, student_model)
   compressed_model = distiller.train(train_data)
   ```

### Hardware-Specific Optimizations

#### NVIDIA Jetson
```bash
# Enable maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Install TensorRT for optimization
pip install tensorrt
```

#### Intel OpenVINO
```bash
# Install OpenVINO toolkit
pip install openvino-dev

# Convert model to OpenVINO format
mo --input_model model.onnx --output_dir openvino_model/
```

## Monitoring and Maintenance

### Performance Monitoring

```python
from src.inference.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_monitoring()

# Monitor key metrics
metrics = monitor.get_metrics()
print(f"Inference latency: {metrics.latency_ms}ms")
print(f"Memory usage: {metrics.memory_mb}MB")
print(f"CPU usage: {metrics.cpu_percent}%")
```

### Health Checks

```bash
# Check system health
python -m src.validation.system_health_check

# Monitor inference accuracy
python -m src.validation.accuracy_monitor --continuous
```

## Troubleshooting

### Common Issues

1. **High Latency**
   - Enable model quantization
   - Reduce model complexity
   - Check hardware utilization

2. **Memory Issues**
   - Reduce batch size to 1
   - Enable gradient checkpointing
   - Use memory-mapped data loading

3. **Accuracy Degradation**
   - Validate quantization settings
   - Check input preprocessing
   - Monitor physics constraint violations

### Debug Commands

```bash
# Profile inference performance
python -m src.inference.realtime_inference --profile

# Validate model accuracy
python -m src.validation.accuracy_validator --edge-config

# Check hardware compatibility
python -m src.utils.hardware_checker
```

## Security Considerations

### Network Security
- Use HTTPS for API endpoints
- Implement authentication tokens
- Enable firewall rules

### Model Security
- Encrypt model files
- Validate input data
- Monitor for adversarial attacks

## Support and Resources

- **Documentation**: [docs/](../README.md)
- **Issues**: GitHub Issues
- **Community**: Discussion Forums
- **Commercial Support**: Contact support team