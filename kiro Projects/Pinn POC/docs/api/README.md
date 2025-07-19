# AV-PINO API Documentation

This directory contains comprehensive API documentation for the Adaptive Variational Physics-Informed Neural Operator (AV-PINO) Motor Fault Diagnosis System.

## Documentation Structure

- **[Core Components](core_components.md)**: Main system components and their APIs
- **[Data Processing](data_processing.md)**: Data loading, preprocessing, and feature extraction
- **[Neural Operators](neural_operators.md)**: Fourier Neural Operator and physics-informed layers
- **[Training](training.md)**: Training pipeline and optimization
- **[Inference](inference.md)**: Real-time inference and uncertainty quantification
- **[Validation](validation.md)**: Benchmarking and validation tools
- **[Visualization](visualization.md)**: Analysis and plotting utilities
- **[Configuration](configuration.md)**: System configuration and experiment management

## Quick Start

```python
from src.config.experiment_config import ExperimentManager
from src.training.training_engine import TrainingEngine
from src.inference.realtime_inference import RealTimeInference

# Setup experiment
config_manager = ExperimentManager()
config = config_manager.setup_experiment("configs/experiment_template.yaml")

# Train model
trainer = TrainingEngine(config)
model = trainer.train()

# Run inference
inference_engine = RealTimeInference(model, config)
prediction, uncertainty = inference_engine.predict(signal_data)
```

## API Reference

### Core Classes

- [`ExperimentManager`](configuration.md#experimentmanager): Experiment configuration and reproducibility
- [`TrainingEngine`](training.md#trainingengine): Model training and optimization
- [`RealTimeInference`](inference.md#realtimeinference): Real-time fault prediction
- [`VisualizationManager`](visualization.md#visualizationmanager): Comprehensive result visualization

### Data Processing

- [`CWRUDataLoader`](data_processing.md#cwrudataloader): CWRU dataset loading and preprocessing
- [`SignalProcessor`](data_processing.md#signalprocessor): Signal processing and feature extraction
- [`PhysicsFeatureExtractor`](data_processing.md#physicsfeatureextractor): Physics-based feature computation

### Neural Architecture

- [`FourierNeuralOperator`](neural_operators.md#fourierneuraloperator): Core neural operator implementation
- [`PhysicsConstraintLayer`](neural_operators.md#physicsconstraintlayer): Physics constraint enforcement
- [`MultiPhysicsCoupling`](neural_operators.md#multiphysicscoupling): Multi-physics domain coupling

## Usage Examples

See the [examples](../examples/) directory for complete usage examples and tutorials.

## Support

For API questions and support:
- Check the detailed documentation for each component
- Review the example notebooks
- Submit issues on GitHub
- Contact the development team