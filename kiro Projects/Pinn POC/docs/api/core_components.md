# Core Components API Reference

This document provides detailed API documentation for the core components of the AV-PINO system.

## ExperimentManager

The `ExperimentManager` class handles experiment configuration, reproducibility, and execution setup.

### Class Definition

```python
class ExperimentManager:
    """Manages reproducible experiment configuration and execution."""
    
    def __init__(self, config_path: Optional[str] = None)
```

### Methods

#### `load_config(config_path: str) -> ExperimentConfig`

Load experiment configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to the configuration YAML file

**Returns:**
- `ExperimentConfig`: Loaded configuration object

**Example:**
```python
manager = ExperimentManager()
config = manager.load_config("configs/experiment_template.yaml")
```

#### `save_config(config: ExperimentConfig, output_path: str)`

Save experiment configuration to YAML file.

**Parameters:**
- `config` (ExperimentConfig): Configuration object to save
- `output_path` (str): Output file path

**Example:**
```python
manager.save_config(config, "outputs/experiment_config.yaml")
```

#### `setup_reproducibility(config: Optional[ReproducibilityConfig] = None)`

Set up reproducible random seeds and deterministic behavior.

**Parameters:**
- `config` (ReproducibilityConfig, optional): Reproducibility configuration

**Example:**
```python
repro_config = ReproducibilityConfig(seed=42, deterministic=True)
manager.setup_reproducibility(repro_config)
```

#### `create_default_config() -> ExperimentConfig`

Create default experiment configuration.

**Returns:**
- `ExperimentConfig`: Default configuration object

**Example:**
```python
default_config = manager.create_default_config()
```

#### `validate_config(config: ExperimentConfig) -> bool`

Validate experiment configuration.

**Parameters:**
- `config` (ExperimentConfig): Configuration to validate

**Returns:**
- `bool`: True if valid, False otherwise

**Example:**
```python
is_valid = manager.validate_config(config)
if not is_valid:
    raise ValueError("Invalid configuration")
```

#### `setup_experiment(config_path: Optional[str] = None) -> ExperimentConfig`

Set up complete experiment with reproducibility.

**Parameters:**
- `config_path` (str, optional): Path to configuration file

**Returns:**
- `ExperimentConfig`: Validated and initialized configuration

**Example:**
```python
config = manager.setup_experiment("configs/my_experiment.yaml")
```

## Configuration Data Classes

### ReproducibilityConfig

Configuration for reproducible experiments.

```python
@dataclass
class ReproducibilityConfig:
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    use_deterministic_algorithms: bool = True
```

**Fields:**
- `seed` (int): Random seed for reproducibility
- `deterministic` (bool): Enable deterministic CUDA operations
- `benchmark` (bool): Enable CUDA benchmark mode
- `use_deterministic_algorithms` (bool): Use deterministic PyTorch algorithms

### ExperimentConfig

Complete experiment configuration.

```python
@dataclass
class ExperimentConfig:
    reproducibility: ReproducibilityConfig
    model: Dict[str, Any]
    training: Dict[str, Any]
    data: Dict[str, Any]
    physics: Dict[str, Any]
    inference: Dict[str, Any]
    logging: Dict[str, Any]
```

**Fields:**
- `reproducibility`: Reproducibility settings
- `model`: Model architecture configuration
- `training`: Training parameters
- `data`: Data processing configuration
- `physics`: Physics constraints configuration
- `inference`: Inference settings
- `logging`: Logging and monitoring configuration

## Usage Examples

### Basic Experiment Setup

```python
from src.config.experiment_config import ExperimentManager

# Initialize manager
manager = ExperimentManager()

# Create and validate configuration
config = manager.create_default_config()
config.training["epochs"] = 200
config.model["width"] = 128

if manager.validate_config(config):
    # Setup experiment with reproducibility
    final_config = manager.setup_experiment()
    print(f"Experiment setup complete with seed: {final_config.reproducibility.seed}")
```

### Custom Configuration

```python
from src.config.experiment_config import ExperimentManager, ReproducibilityConfig

# Custom reproducibility settings
repro_config = ReproducibilityConfig(
    seed=123,
    deterministic=True,
    use_deterministic_algorithms=True
)

# Setup manager with custom config
manager = ExperimentManager()
manager.setup_reproducibility(repro_config)

# Load and modify configuration
config = manager.load_config("configs/custom_experiment.yaml")
config.training["learning_rate"] = 0.0005

# Save modified configuration
manager.save_config(config, "outputs/modified_config.yaml")
```

### Configuration Validation

```python
from src.config.experiment_config import ExperimentManager

manager = ExperimentManager()
config = manager.load_config("configs/experiment.yaml")

# Validate configuration
try:
    if manager.validate_config(config):
        print("Configuration is valid")
        # Proceed with experiment
        validated_config = manager.setup_experiment()
    else:
        print("Configuration validation failed")
except Exception as e:
    print(f"Configuration error: {e}")
```

## Error Handling

### Common Exceptions

- `FileNotFoundError`: Configuration file not found
- `yaml.YAMLError`: Invalid YAML syntax
- `ValueError`: Invalid configuration values
- `AssertionError`: Configuration validation failure

### Error Handling Example

```python
from src.config.experiment_config import ExperimentManager
import yaml

manager = ExperimentManager()

try:
    config = manager.load_config("configs/experiment.yaml")
    validated_config = manager.setup_experiment()
except FileNotFoundError:
    print("Configuration file not found")
    config = manager.create_default_config()
except yaml.YAMLError as e:
    print(f"YAML parsing error: {e}")
except ValueError as e:
    print(f"Configuration validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### Configuration Management

1. **Use Templates**: Start with the provided template and modify as needed
2. **Validate Early**: Always validate configuration before training
3. **Version Control**: Track configuration changes in version control
4. **Documentation**: Document custom configuration parameters

### Reproducibility

1. **Set Seeds**: Always set random seeds for reproducible results
2. **Deterministic Mode**: Enable deterministic operations for exact reproducibility
3. **Environment**: Document Python and library versions
4. **Hardware**: Note hardware specifications for reproducibility

### Example Best Practice Implementation

```python
from src.config.experiment_config import ExperimentManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_reproducible_experiment(config_path: str):
    """Setup a reproducible experiment with proper error handling."""
    manager = ExperimentManager()
    
    try:
        # Load and validate configuration
        config = manager.load_config(config_path)
        
        if not manager.validate_config(config):
            raise ValueError("Configuration validation failed")
        
        # Setup reproducibility
        final_config = manager.setup_experiment(config_path)
        
        # Log experiment details
        logger.info(f"Experiment setup complete:")
        logger.info(f"  Seed: {final_config.reproducibility.seed}")
        logger.info(f"  Model: {final_config.model['architecture']}")
        logger.info(f"  Epochs: {final_config.training['epochs']}")
        
        return final_config
        
    except Exception as e:
        logger.error(f"Experiment setup failed: {e}")
        raise

# Usage
config = setup_reproducible_experiment("configs/my_experiment.yaml")
```