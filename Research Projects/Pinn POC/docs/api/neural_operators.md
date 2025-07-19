# Neural Operators API Reference

This document provides detailed API documentation for the neural operator components of the AV-PINO system.

## FourierNeuralOperator

The core neural operator implementation using Fourier transforms for learning operators in function spaces.

### Class Definition

```python
class FourierNeuralOperator(nn.Module):
    """Fourier Neural Operator for learning operators between function spaces."""
    
    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        input_dim: int = 1,
        output_dim: int = 1,
        n_layers: int = 4,
        physics_constraints: Optional[List[PDEConstraint]] = None
    )
```

### Parameters

- `modes` (int): Number of Fourier modes to keep
- `width` (int): Hidden dimension width
- `input_dim` (int): Input function dimension
- `output_dim` (int): Output function dimension
- `n_layers` (int): Number of Fourier layers
- `physics_constraints` (List[PDEConstraint], optional): Physics constraints to enforce

### Methods

#### `forward(x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]`

Forward pass through the neural operator.

**Parameters:**
- `x` (torch.Tensor): Input function values [batch, sequence, channels]

**Returns:**
- `Tuple[torch.Tensor, Dict[str, torch.Tensor]]`: Output predictions and physics residuals

**Example:**
```python
model = FourierNeuralOperator(modes=16, width=64)
input_signal = torch.randn(32, 1024, 1)  # batch_size=32, seq_len=1024, channels=1
output, residuals = model(input_signal)
```

#### `get_fourier_modes() -> torch.Tensor`

Get the learned Fourier mode weights.

**Returns:**
- `torch.Tensor`: Fourier mode weights

**Example:**
```python
fourier_weights = model.get_fourier_modes()
print(f"Fourier weights shape: {fourier_weights.shape}")
```

#### `compute_operator_norm() -> float`

Compute the operator norm for stability analysis.

**Returns:**
- `float`: Operator norm value

**Example:**
```python
op_norm = model.compute_operator_norm()
print(f"Operator norm: {op_norm:.6f}")
```

## SpectralConv1d

Spectral convolution layer in Fourier domain.

### Class Definition

```python
class SpectralConv1d(nn.Module):
    """1D Spectral convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int)
```

### Methods

#### `forward(x: torch.Tensor) -> torch.Tensor`

Apply spectral convolution in Fourier domain.

**Parameters:**
- `x` (torch.Tensor): Input tensor [batch, channels, sequence]

**Returns:**
- `torch.Tensor`: Convolved output

**Example:**
```python
spectral_conv = SpectralConv1d(in_channels=64, out_channels=64, modes=16)
x = torch.randn(32, 64, 1024)
output = spectral_conv(x)
```

## PhysicsConstraintLayer

Layer that enforces physics constraints during forward pass.

### Class Definition

```python
class PhysicsConstraintLayer(nn.Module):
    """Enforces physics constraints in neural operator."""
    
    def __init__(self, constraints: List[PDEConstraint], weight: float = 1.0)
```

### Methods

#### `forward(x: torch.Tensor, prediction: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]`

Apply physics constraints and compute residuals.

**Parameters:**
- `x` (torch.Tensor): Input data
- `prediction` (torch.Tensor): Model prediction

**Returns:**
- `Tuple[torch.Tensor, Dict[str, torch.Tensor]]`: Constrained prediction and residuals

**Example:**
```python
from src.physics.constraints import MaxwellConstraint, HeatEquationConstraint

constraints = [MaxwellConstraint(), HeatEquationConstraint()]
physics_layer = PhysicsConstraintLayer(constraints, weight=0.1)

constrained_pred, residuals = physics_layer(input_data, raw_prediction)
```

## MultiPhysicsCoupling

Handles coupling between different physics domains.

### Class Definition

```python
class MultiPhysicsCoupling(nn.Module):
    """Multi-physics domain coupling module."""
    
    def __init__(
        self,
        electromagnetic_dim: int = 64,
        thermal_dim: int = 64,
        mechanical_dim: int = 64,
        coupling_strength: float = 0.1
    )
```

### Methods

#### `forward(em_field: torch.Tensor, thermal_field: torch.Tensor, mechanical_field: torch.Tensor) -> Dict[str, torch.Tensor]`

Compute multi-physics coupling.

**Parameters:**
- `em_field` (torch.Tensor): Electromagnetic field representation
- `thermal_field` (torch.Tensor): Thermal field representation
- `mechanical_field` (torch.Tensor): Mechanical field representation

**Returns:**
- `Dict[str, torch.Tensor]`: Coupled field representations

**Example:**
```python
coupling = MultiPhysicsCoupling(
    electromagnetic_dim=64,
    thermal_dim=64,
    mechanical_dim=64
)

coupled_fields = coupling(em_data, thermal_data, mechanical_data)
```

#### `compute_coupling_energy(fields: Dict[str, torch.Tensor]) -> torch.Tensor`

Compute total coupling energy for conservation.

**Parameters:**
- `fields` (Dict[str, torch.Tensor]): Field representations

**Returns:**
- `torch.Tensor`: Total coupling energy

**Example:**
```python
total_energy = coupling.compute_coupling_energy(coupled_fields)
```

## AGTNOArchitecture

Complete Adaptive Graph-Transformer Neural Operator architecture.

### Class Definition

```python
class AGTNOArchitecture(nn.Module):
    """Complete AGT-NO architecture with encoder-processor-decoder."""
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 10,
        n_modes: int = 16,
        n_layers: int = 4,
        physics_constraints: Optional[List[PDEConstraint]] = None
    )
```

### Methods

#### `forward(x: torch.Tensor) -> Dict[str, torch.Tensor]`

Complete forward pass through AGT-NO architecture.

**Parameters:**
- `x` (torch.Tensor): Input signal data

**Returns:**
- `Dict[str, torch.Tensor]`: Predictions, uncertainties, and physics residuals

**Example:**
```python
model = AGTNOArchitecture(
    input_dim=1,
    hidden_dim=128,
    output_dim=10,  # 10 fault classes
    n_modes=16
)

input_signal = torch.randn(32, 1024, 1)
outputs = model(input_signal)

predictions = outputs['predictions']
uncertainties = outputs['uncertainties']
physics_residuals = outputs['physics_residuals']
```

#### `encode(x: torch.Tensor) -> torch.Tensor`

Encoder module: lift input to higher dimensional space.

**Parameters:**
- `x` (torch.Tensor): Input data

**Returns:**
- `torch.Tensor`: Encoded representation

#### `process(x: torch.Tensor) -> torch.Tensor`

Processor module: apply neural operator transformations.

**Parameters:**
- `x` (torch.Tensor): Encoded input

**Returns:**
- `torch.Tensor`: Processed representation

#### `decode(x: torch.Tensor) -> torch.Tensor`

Decoder module: project to output space.

**Parameters:**
- `x` (torch.Tensor): Processed representation

**Returns:**
- `torch.Tensor`: Final predictions

## Usage Examples

### Basic Neural Operator Training

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.physics.agt_no_architecture import AGTNOArchitecture
from src.physics.constraints import MaxwellConstraint

# Setup model
constraints = [MaxwellConstraint()]
model = AGTNOArchitecture(
    input_dim=1,
    hidden_dim=128,
    output_dim=10,
    n_modes=16,
    physics_constraints=constraints
)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for batch_idx, (data, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(data)
    predictions = outputs['predictions']
    physics_residuals = outputs['physics_residuals']
    
    # Compute losses
    data_loss = criterion(predictions, targets)
    physics_loss = sum(residuals.mean() for residuals in physics_residuals.values())
    total_loss = data_loss + 0.1 * physics_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    if batch_idx % 100 == 0:
        print(f'Batch {batch_idx}, Loss: {total_loss.item():.6f}')
```

### Physics-Informed Inference

```python
import torch
from src.physics.agt_no_architecture import AGTNOArchitecture
from src.physics.constraints import MaxwellConstraint, HeatEquationConstraint

# Load trained model
model = AGTNOArchitecture(
    input_dim=1,
    hidden_dim=128,
    output_dim=10,
    physics_constraints=[MaxwellConstraint(), HeatEquationConstraint()]
)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Inference with physics validation
with torch.no_grad():
    test_signal = torch.randn(1, 1024, 1)
    outputs = model(test_signal)
    
    predictions = outputs['predictions']
    uncertainties = outputs['uncertainties']
    physics_residuals = outputs['physics_residuals']
    
    # Check physics consistency
    max_residual = max(residual.max().item() for residual in physics_residuals.values())
    
    if max_residual < 1e-3:
        print("Physics constraints satisfied")
        fault_class = torch.argmax(predictions, dim=1)
        confidence = torch.softmax(predictions, dim=1).max()
        print(f"Predicted fault class: {fault_class.item()}")
        print(f"Confidence: {confidence.item():.3f}")
    else:
        print(f"Physics constraint violation: {max_residual:.6f}")
```

### Multi-Physics Coupling Example

```python
import torch
from src.physics.multi_physics_coupling import MultiPhysicsCoupling

# Setup multi-physics coupling
coupling = MultiPhysicsCoupling(
    electromagnetic_dim=64,
    thermal_dim=64,
    mechanical_dim=64,
    coupling_strength=0.1
)

# Simulate multi-physics fields
em_field = torch.randn(32, 64, 1024)      # Electromagnetic
thermal_field = torch.randn(32, 64, 1024)  # Thermal
mechanical_field = torch.randn(32, 64, 1024)  # Mechanical

# Compute coupling
coupled_fields = coupling(em_field, thermal_field, mechanical_field)

# Check energy conservation
total_energy = coupling.compute_coupling_energy(coupled_fields)
print(f"Total coupling energy: {total_energy.mean().item():.6f}")

# Access coupled representations
em_coupled = coupled_fields['electromagnetic']
thermal_coupled = coupled_fields['thermal']
mechanical_coupled = coupled_fields['mechanical']
```

### Custom Physics Constraints

```python
import torch
import torch.nn as nn
from src.physics.constraints import PDEConstraint

class CustomMotorConstraint(PDEConstraint):
    """Custom physics constraint for motor dynamics."""
    
    def __init__(self, motor_frequency: float = 60.0):
        super().__init__()
        self.motor_frequency = motor_frequency
    
    def compute_residual(self, prediction: torch.Tensor, input_data: torch.Tensor) -> torch.Tensor:
        """Compute residual for motor-specific physics."""
        # Example: enforce frequency domain constraints
        fft_pred = torch.fft.fft(prediction, dim=-1)
        fft_input = torch.fft.fft(input_data, dim=-1)
        
        # Motor frequency should be preserved
        freq_bins = torch.fft.fftfreq(prediction.size(-1))
        motor_bin = torch.argmin(torch.abs(freq_bins - self.motor_frequency))
        
        residual = torch.abs(fft_pred[..., motor_bin] - fft_input[..., motor_bin])
        return residual.mean()

# Use custom constraint
custom_constraint = CustomMotorConstraint(motor_frequency=60.0)
model = AGTNOArchitecture(
    input_dim=1,
    hidden_dim=128,
    output_dim=10,
    physics_constraints=[custom_constraint]
)
```

### Model Analysis and Visualization

```python
import torch
import matplotlib.pyplot as plt
from src.physics.agt_no_architecture import AGTNOArchitecture

# Load trained model
model = AGTNOArchitecture(input_dim=1, hidden_dim=128, output_dim=10)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Analyze Fourier modes
fourier_weights = model.get_fourier_modes()
print(f"Fourier weights shape: {fourier_weights.shape}")

# Plot learned modes
plt.figure(figsize=(12, 4))
for i in range(min(4, fourier_weights.size(0))):
    plt.subplot(1, 4, i+1)
    plt.plot(fourier_weights[i].detach().cpu().numpy())
    plt.title(f'Mode {i+1}')
    plt.grid(True)

plt.tight_layout()
plt.savefig('learned_fourier_modes.png')

# Compute operator norm for stability
op_norm = model.compute_operator_norm()
print(f"Operator norm: {op_norm:.6f}")

# Analyze prediction uncertainty
test_data = torch.randn(100, 1024, 1)
with torch.no_grad():
    outputs = model(test_data)
    uncertainties = outputs['uncertainties']
    
    plt.figure(figsize=(10, 6))
    plt.hist(uncertainties.cpu().numpy(), bins=50, alpha=0.7)
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution')
    plt.grid(True)
    plt.savefig('uncertainty_distribution.png')
```

## Error Handling

### Common Issues and Solutions

1. **CUDA Memory Errors**: Reduce batch size or use gradient checkpointing
2. **NaN Values**: Check learning rate and gradient clipping
3. **Physics Constraint Violations**: Adjust constraint weights
4. **Convergence Issues**: Verify data preprocessing and model initialization

### Error Handling Example

```python
import torch
from src.physics.agt_no_architecture import AGTNOArchitecture

def safe_model_forward(model, input_data):
    """Safely perform model forward pass with error handling."""
    try:
        # Check input validity
        if torch.isnan(input_data).any():
            raise ValueError("Input contains NaN values")
        
        if torch.isinf(input_data).any():
            raise ValueError("Input contains infinite values")
        
        # Forward pass
        outputs = model(input_data)
        
        # Check output validity
        predictions = outputs['predictions']
        if torch.isnan(predictions).any():
            raise RuntimeError("Model produced NaN predictions")
        
        return outputs
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA out of memory. Try reducing batch size.")
            torch.cuda.empty_cache()
        else:
            print(f"Runtime error: {e}")
        return None
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
model = AGTNOArchitecture(input_dim=1, hidden_dim=128, output_dim=10)
test_input = torch.randn(32, 1024, 1)

outputs = safe_model_forward(model, test_input)
if outputs is not None:
    print("Forward pass successful")
```

## Best Practices

### Model Design

1. **Mode Selection**: Choose Fourier modes based on signal characteristics
2. **Width Scaling**: Scale hidden dimensions with problem complexity
3. **Physics Integration**: Balance data and physics loss terms
4. **Regularization**: Use appropriate regularization for generalization

### Training Guidelines

1. **Learning Rate**: Start with 1e-3 and adjust based on convergence
2. **Batch Size**: Use largest batch size that fits in memory
3. **Physics Weights**: Gradually increase physics constraint weights
4. **Monitoring**: Track both accuracy and physics residuals

### Performance Optimization

1. **Mixed Precision**: Use automatic mixed precision for faster training
2. **Gradient Checkpointing**: Save memory for large models
3. **Data Loading**: Use efficient data loaders with multiple workers
4. **Model Compilation**: Use torch.compile for inference speedup

### Example Optimized Training

```python
import torch
from torch.cuda.amp import GradScaler, autocast
from src.physics.agt_no_architecture import AGTNOArchitecture

# Setup for mixed precision training
model = AGTNOArchitecture(input_dim=1, hidden_dim=128, output_dim=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = GradScaler()

# Training with mixed precision
model.train()
for batch_idx, (data, targets) in enumerate(train_loader):
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(data)
        predictions = outputs['predictions']
        physics_residuals = outputs['physics_residuals']
        
        data_loss = nn.CrossEntropyLoss()(predictions, targets)
        physics_loss = sum(residuals.mean() for residuals in physics_residuals.values())
        total_loss = data_loss + 0.1 * physics_loss
    
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if batch_idx % 100 == 0:
        print(f'Batch {batch_idx}, Loss: {total_loss.item():.6f}')
```