# Data Processing API Reference

This document provides detailed API documentation for the data processing components of the AV-PINO system.

## CWRUDataLoader

The `CWRUDataLoader` class handles loading and preprocessing of the CWRU bearing fault dataset.

### Class Definition

```python
class CWRUDataLoader:
    """Loads and preprocesses CWRU bearing fault dataset."""
    
    def __init__(self, data_dir: str = "data/cwru", download: bool = True)
```

### Methods

#### `load_dataset() -> Dict[str, Any]`

Load the complete CWRU dataset with all fault types.

**Returns:**
- `Dict[str, Any]`: Dictionary containing loaded data with keys:
  - `'normal'`: Normal bearing data
  - `'inner_race'`: Inner race fault data
  - `'outer_race'`: Outer race fault data
  - `'ball'`: Ball fault data

**Example:**
```python
loader = CWRUDataLoader(data_dir="data/cwru")
dataset = loader.load_dataset()
print(f"Loaded {len(dataset)} fault categories")
```

#### `load_fault_type(fault_type: str, load_size: str = "12k") -> np.ndarray`

Load specific fault type data.

**Parameters:**
- `fault_type` (str): Type of fault ('normal', 'inner_race', 'outer_race', 'ball')
- `load_size` (str): Load size ('12k', '48k')

**Returns:**
- `np.ndarray`: Vibration signal data

**Example:**
```python
normal_data = loader.load_fault_type('normal', '12k')
inner_fault_data = loader.load_fault_type('inner_race', '12k')
```

#### `get_metadata() -> Dict[str, Any]`

Get dataset metadata and information.

**Returns:**
- `Dict[str, Any]`: Metadata including sampling rates, fault descriptions

**Example:**
```python
metadata = loader.get_metadata()
print(f"Sampling rate: {metadata['sampling_rate']} Hz")
```

#### `create_train_test_split(test_size: float = 0.2, random_state: int = 42) -> Tuple[Dict, Dict]`

Create train/test split of the dataset.

**Parameters:**
- `test_size` (float): Fraction of data for testing
- `random_state` (int): Random seed for reproducibility

**Returns:**
- `Tuple[Dict, Dict]`: Training and testing datasets

**Example:**
```python
train_data, test_data = loader.create_train_test_split(test_size=0.2)
```

## SignalProcessor

The `SignalProcessor` class provides signal processing and feature extraction capabilities.

### Class Definition

```python
class SignalProcessor:
    """Processes vibration signals and extracts features."""
    
    def __init__(self, sampling_rate: float = 12000.0)
```

### Methods

#### `extract_time_domain_features(signal: np.ndarray) -> Dict[str, float]`

Extract time-domain statistical features.

**Parameters:**
- `signal` (np.ndarray): Input vibration signal

**Returns:**
- `Dict[str, float]`: Time-domain features including RMS, peak, kurtosis, etc.

**Example:**
```python
processor = SignalProcessor(sampling_rate=12000)
time_features = processor.extract_time_domain_features(signal_data)
print(f"RMS: {time_features['rms']:.4f}")
```

#### `extract_frequency_domain_features(signal: np.ndarray) -> Dict[str, Any]`

Extract frequency-domain features using FFT.

**Parameters:**
- `signal` (np.ndarray): Input vibration signal

**Returns:**
- `Dict[str, Any]`: Frequency-domain features and spectrum

**Example:**
```python
freq_features = processor.extract_frequency_domain_features(signal_data)
spectrum = freq_features['spectrum']
dominant_freq = freq_features['dominant_frequency']
```

#### `extract_time_frequency_features(signal: np.ndarray) -> Dict[str, Any]`

Extract time-frequency features using wavelet transform.

**Parameters:**
- `signal` (np.ndarray): Input vibration signal

**Returns:**
- `Dict[str, Any]`: Time-frequency features and scalogram

**Example:**
```python
tf_features = processor.extract_time_frequency_features(signal_data)
scalogram = tf_features['scalogram']
```

#### `preprocess_signal(signal: np.ndarray, normalize: bool = True, filter_noise: bool = True) -> np.ndarray`

Preprocess signal with normalization and filtering.

**Parameters:**
- `signal` (np.ndarray): Input signal
- `normalize` (bool): Apply normalization
- `filter_noise` (bool): Apply noise filtering

**Returns:**
- `np.ndarray`: Preprocessed signal

**Example:**
```python
clean_signal = processor.preprocess_signal(raw_signal, normalize=True)
```

## PhysicsFeatureExtractor

The `PhysicsFeatureExtractor` class computes physics-based features from motor signals.

### Class Definition

```python
class PhysicsFeatureExtractor:
    """Extracts physics-informed features from motor signals."""
    
    def __init__(self, motor_params: Dict[str, float])
```

### Methods

#### `extract_electromagnetic_features(signal: np.ndarray, current: np.ndarray) -> Dict[str, float]`

Extract electromagnetic field-related features.

**Parameters:**
- `signal` (np.ndarray): Vibration signal
- `current` (np.ndarray): Motor current signal

**Returns:**
- `Dict[str, float]`: Electromagnetic features

**Example:**
```python
extractor = PhysicsFeatureExtractor(motor_params={'poles': 4, 'frequency': 60})
em_features = extractor.extract_electromagnetic_features(vibration, current)
```

#### `extract_thermal_features(signal: np.ndarray, temperature: Optional[np.ndarray] = None) -> Dict[str, float]`

Extract thermal dynamics features.

**Parameters:**
- `signal` (np.ndarray): Vibration signal
- `temperature` (np.ndarray, optional): Temperature measurements

**Returns:**
- `Dict[str, float]`: Thermal features

**Example:**
```python
thermal_features = extractor.extract_thermal_features(signal, temp_data)
```

#### `extract_mechanical_features(signal: np.ndarray) -> Dict[str, float]`

Extract mechanical vibration features.

**Parameters:**
- `signal` (np.ndarray): Vibration signal

**Returns:**
- `Dict[str, float]`: Mechanical features

**Example:**
```python
mech_features = extractor.extract_mechanical_features(vibration_signal)
```

#### `extract_all_physics_features(signals: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]`

Extract all physics features from multi-modal signals.

**Parameters:**
- `signals` (Dict[str, np.ndarray]): Dictionary of signal types

**Returns:**
- `Dict[str, Dict[str, float]]`: All physics features organized by domain

**Example:**
```python
all_signals = {
    'vibration': vib_data,
    'current': current_data,
    'temperature': temp_data
}
physics_features = extractor.extract_all_physics_features(all_signals)
```

## DataPreprocessor

The `DataPreprocessor` class handles data preparation for neural operator training.

### Class Definition

```python
class DataPreprocessor:
    """Preprocesses data for neural operator training."""
    
    def __init__(self, sequence_length: int = 1024, overlap: float = 0.5)
```

### Methods

#### `create_sequences(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`

Create overlapping sequences for training.

**Parameters:**
- `data` (np.ndarray): Input signal data
- `labels` (np.ndarray): Corresponding labels

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Sequences and labels

**Example:**
```python
preprocessor = DataPreprocessor(sequence_length=1024, overlap=0.5)
sequences, seq_labels = preprocessor.create_sequences(signal_data, labels)
```

#### `normalize_data(data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict]`

Normalize data using specified method.

**Parameters:**
- `data` (np.ndarray): Input data
- `method` (str): Normalization method ('standard', 'minmax', 'robust')

**Returns:**
- `Tuple[np.ndarray, Dict]`: Normalized data and normalization parameters

**Example:**
```python
normalized_data, norm_params = preprocessor.normalize_data(data, 'standard')
```

#### `augment_data(data: np.ndarray, labels: np.ndarray, augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]`

Apply data augmentation techniques.

**Parameters:**
- `data` (np.ndarray): Input data
- `labels` (np.ndarray): Labels
- `augmentation_factor` (int): Augmentation multiplier

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Augmented data and labels

**Example:**
```python
aug_data, aug_labels = preprocessor.augment_data(data, labels, augmentation_factor=3)
```

## Usage Examples

### Complete Data Processing Pipeline

```python
from src.data.cwru_loader import CWRUDataLoader
from src.data.signal_processor import SignalProcessor
from src.data.preprocessor import DataPreprocessor
from src.physics.feature_extractor import PhysicsFeatureExtractor

# Load dataset
loader = CWRUDataLoader(data_dir="data/cwru")
dataset = loader.load_dataset()

# Process signals
processor = SignalProcessor(sampling_rate=12000)
processed_data = {}

for fault_type, signals in dataset.items():
    # Extract features
    time_features = processor.extract_time_domain_features(signals)
    freq_features = processor.extract_frequency_domain_features(signals)
    
    processed_data[fault_type] = {
        'time_features': time_features,
        'freq_features': freq_features,
        'processed_signal': processor.preprocess_signal(signals)
    }

# Extract physics features
motor_params = {'poles': 4, 'frequency': 60, 'power': 1000}
physics_extractor = PhysicsFeatureExtractor(motor_params)

physics_features = {}
for fault_type, data in processed_data.items():
    signal = data['processed_signal']
    physics_features[fault_type] = physics_extractor.extract_all_physics_features({
        'vibration': signal
    })

# Prepare for training
preprocessor = DataPreprocessor(sequence_length=1024)
train_data, test_data = loader.create_train_test_split()

# Create sequences
all_sequences = []
all_labels = []

for fault_type, signals in train_data.items():
    sequences, labels = preprocessor.create_sequences(
        signals, 
        np.full(len(signals), fault_type)
    )
    all_sequences.append(sequences)
    all_labels.append(labels)

# Combine and normalize
combined_sequences = np.concatenate(all_sequences)
combined_labels = np.concatenate(all_labels)

normalized_sequences, norm_params = preprocessor.normalize_data(combined_sequences)

print(f"Prepared {len(normalized_sequences)} sequences for training")
```

### Real-time Data Processing

```python
import numpy as np
from src.data.signal_processor import SignalProcessor
from src.physics.feature_extractor import PhysicsFeatureExtractor

# Setup for real-time processing
processor = SignalProcessor(sampling_rate=12000)
physics_extractor = PhysicsFeatureExtractor({'poles': 4, 'frequency': 60})

def process_realtime_signal(raw_signal: np.ndarray) -> Dict[str, Any]:
    """Process a real-time signal sample."""
    
    # Preprocess signal
    clean_signal = processor.preprocess_signal(raw_signal)
    
    # Extract features
    time_features = processor.extract_time_domain_features(clean_signal)
    freq_features = processor.extract_frequency_domain_features(clean_signal)
    
    # Extract physics features
    physics_features = physics_extractor.extract_mechanical_features(clean_signal)
    
    return {
        'processed_signal': clean_signal,
        'time_features': time_features,
        'frequency_features': freq_features,
        'physics_features': physics_features,
        'timestamp': time.time()
    }

# Example usage
signal_sample = np.random.randn(1024)  # Simulated signal
features = process_realtime_signal(signal_sample)
```

## Error Handling

### Common Exceptions

- `FileNotFoundError`: Dataset files not found
- `ValueError`: Invalid signal dimensions or parameters
- `RuntimeError`: Processing failures

### Error Handling Example

```python
from src.data.cwru_loader import CWRUDataLoader
import logging

logger = logging.getLogger(__name__)

def safe_data_loading(data_dir: str):
    """Safely load CWRU dataset with error handling."""
    try:
        loader = CWRUDataLoader(data_dir=data_dir)
        dataset = loader.load_dataset()
        logger.info(f"Successfully loaded {len(dataset)} fault categories")
        return dataset
    
    except FileNotFoundError:
        logger.error("CWRU dataset not found. Please download the dataset.")
        return None
    
    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}")
        return None

# Usage
dataset = safe_data_loading("data/cwru")
if dataset is not None:
    print("Dataset loaded successfully")
```

## Best Practices

### Data Processing Guidelines

1. **Signal Quality**: Always check signal quality before processing
2. **Normalization**: Apply consistent normalization across train/test sets
3. **Feature Selection**: Use domain knowledge for physics feature selection
4. **Validation**: Validate processed data before training

### Performance Optimization

1. **Batch Processing**: Process signals in batches for efficiency
2. **Caching**: Cache processed features to avoid recomputation
3. **Memory Management**: Use generators for large datasets
4. **Parallel Processing**: Utilize multiprocessing for feature extraction

### Example Optimized Pipeline

```python
from concurrent.futures import ProcessPoolExecutor
from src.data.signal_processor import SignalProcessor

def process_signal_batch(signal_batch: List[np.ndarray]) -> List[Dict]:
    """Process a batch of signals in parallel."""
    processor = SignalProcessor()
    
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(processor.extract_time_domain_features, signal)
            for signal in signal_batch
        ]
        
        results = [future.result() for future in futures]
    
    return results

# Usage
signal_batches = [signals[i:i+32] for i in range(0, len(signals), 32)]
all_features = []

for batch in signal_batches:
    batch_features = process_signal_batch(batch)
    all_features.extend(batch_features)
```