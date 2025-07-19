# Training API Reference

This document provides detailed API documentation for the training components of the AV-PINO system.

## TrainingEngine

The main training engine that orchestrates the physics-informed training process.

### Class Definition

```python
class TrainingEngine:
    """Main training engine for AV-PINO models with physics constraints."""
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device = None
    )
```

### Methods

#### `train(train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Tuple[nn.Module, Dict[str, List[float]]]`

Train the model with physics-informed loss.

**Parameters:**
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader, optional): Validation data loader

**Returns:**
- `Tuple[nn.Module, Dict[str, List[float]]]`: Trained model and training history

**Example:**
```python
from src.training.training_engine import TrainingEngine

engine = TrainingEngine(model, config, device)
trained_model, history = engine.train(train_loader, val_loader)

print(f"Final loss: {history['train_loss'][-1]:.6f}")
print(f"Physics loss: {history['physics_loss'][-1]:.6f}")
```

#### `train_epoch(data_loader: DataLoader) -> Dict[str, float]`

Train for one epoch.

**Parameters:**
- `data_loader` (DataLoader): Data loader for training

**Returns:**
- `Dict[str, float]`: Epoch metrics

**Example:**
```python
epoch_metrics = engine.train_epoch(train_loader)
print(f"Epoch loss: {epoch_metrics['loss']:.6f}")
```

#### `validate(data_loader: DataLoader) -> Dict[str, float]`

Validate the model.

**Parameters:**
- `data_loader` (DataLoader): Validation data loader

**Returns:**
- `Dict[str, float]`: Validation metrics

**Example:**
```python
val_metrics = engine.validate(val_loader)
print(f"Validation accuracy: {val_metrics['accuracy']:.3f}")
```

#### `save_checkpoint(path: str, epoch: int, metrics: Dict[str, float])`

Save training checkpoint.

**Parameters:**
- `path` (str): Checkpoint save path
- `epoch` (int): Current epoch
- `metrics` (Dict[str, float]): Current metrics

**Example:**
```python
engine.save_checkpoint("checkpoints/model_epoch_10.pth", 10, val_metrics)
```

#### `load_checkpoint(path: str) -> Dict[str, Any]`

Load training checkpoint.

**Parameters:**
- `path` (str): Checkpoint file path

**Returns:**
- `Dict[str, Any]`: Checkpoint data

**Example:**
```python
checkpoint = engine.load_checkpoint("checkpoints/model_epoch_10.pth")
start_epoch = checkpoint['epoch']
```

## PhysicsInformedLoss

Physics-informed loss function combining data and physics terms.

### Class Definition

```python
class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss combining data loss and physics constraints."""
    
    def __init__(
        self,
        data_loss_fn: nn.Module = nn.CrossEntropyLoss(),
        physics_weight: float = 0.1,
        adaptive_weighting: bool = True
    )
```

### Methods

#### `forward(predictions: torch.Tensor, targets: torch.Tensor, physics_residuals: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`

Compute total physics-informed loss.

**Parameters:**
- `predictions` (torch.Tensor): Model predictions
- `targets` (torch.Tensor): Ground truth targets
- `physics_residuals` (Dict[str, torch.Tensor]): Physics constraint residuals

**Returns:**
- `Dict[str, torch.Tensor]`: Loss components

**Example:**
```python
from src.training.physics_informed_loss import PhysicsInformedLoss

loss_fn = PhysicsInformedLoss(physics_weight=0.1)
outputs = model(batch_data)

losses = loss_fn(
    predictions=outputs['predictions'],
    targets=batch_targets,
    physics_residuals=outputs['physics_residuals']
)

total_loss = losses['total_loss']
data_loss = losses['data_loss']
physics_loss = losses['physics_loss']
```

#### `compute_data_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor`

Compute data-driven loss component.

**Parameters:**
- `predictions` (torch.Tensor): Model predictions
- `targets` (torch.Tensor): Ground truth targets

**Returns:**
- `torch.Tensor`: Data loss

#### `compute_physics_loss(physics_residuals: Dict[str, torch.Tensor]) -> torch.Tensor`

Compute physics constraint loss.

**Parameters:**
- `physics_residuals` (Dict[str, torch.Tensor]): Physics residuals

**Returns:**
- `torch.Tensor`: Physics loss

#### `update_weights(epoch: int, data_loss: float, physics_loss: float)`

Update adaptive loss weights.

**Parameters:**
- `epoch` (int): Current epoch
- `data_loss` (float): Current data loss
- `physics_loss` (float): Current physics loss

**Example:**
```python
# Adaptive weight adjustment during training
loss_fn.update_weights(epoch, data_loss.item(), physics_loss.item())
```

## AdvancedLossManager

Manager for advanced loss functions including consistency and variational losses.

### Class Definition

```python
class AdvancedLossManager:
    """Manages advanced loss functions for physics-informed training."""
    
    def __init__(
        self,
        consistency_weight: float = 0.05,
        variational_weight: float = 0.02,
        enable_adaptive: bool = True
    )
```

### Methods

#### `compute_consistency_loss(coupled_fields: Dict[str, torch.Tensor]) -> torch.Tensor`

Compute multi-physics consistency loss.

**Parameters:**
- `coupled_fields` (Dict[str, torch.Tensor]): Coupled physics fields

**Returns:**
- `torch.Tensor`: Consistency loss

**Example:**
```python
from src.training.advanced_loss_functions import AdvancedLossManager

loss_manager = AdvancedLossManager()
consistency_loss = loss_manager.compute_consistency_loss(coupled_fields)
```

#### `compute_variational_loss(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor`

Compute variational loss for uncertainty quantification.

**Parameters:**
- `mean` (torch.Tensor): Variational mean
- `log_var` (torch.Tensor): Variational log variance

**Returns:**
- `torch.Tensor`: Variational loss (KL divergence)

#### `get_total_advanced_loss(outputs: Dict[str, torch.Tensor]) -> torch.Tensor`

Compute total advanced loss.

**Parameters:**
- `outputs` (Dict[str, torch.Tensor]): Model outputs

**Returns:**
- `torch.Tensor`: Total advanced loss

## TrainingMonitor

Training progress monitoring and visualization.

### Class Definition

```python
class TrainingMonitor:
    """Monitors and visualizes training progress."""
    
    def __init__(self, log_dir: str = "logs", save_frequency: int = 10)
```

### Methods

#### `log_metrics(epoch: int, metrics: Dict[str, float])`

Log training metrics.

**Parameters:**
- `epoch` (int): Current epoch
- `metrics` (Dict[str, float]): Metrics to log

**Example:**
```python
from src.training.monitoring import TrainingMonitor

monitor = TrainingMonitor(log_dir="training_logs")
monitor.log_metrics(epoch, {
    'train_loss': 0.234,
    'val_loss': 0.267,
    'accuracy': 0.89,
    'physics_loss': 0.001
})
```

#### `plot_training_curves(save_path: Optional[str] = None)`

Plot training progress curves.

**Parameters:**
- `save_path` (str, optional): Path to save plot

**Example:**
```python
monitor.plot_training_curves("training_curves.png")
```

#### `log_physics_residuals(epoch: int, residuals: Dict[str, float])`

Log physics constraint residuals.

**Parameters:**
- `epoch` (int): Current epoch
- `residuals` (Dict[str, float]): Physics residuals

#### `check_convergence(patience: int = 10) -> bool`

Check if training has converged.

**Parameters:**
- `patience` (int): Patience for early stopping

**Returns:**
- `bool`: True if converged

**Example:**
```python
if monitor.check_convergence(patience=15):
    print("Training converged, stopping early")
    break
```

## Usage Examples

### Complete Training Pipeline

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.training_engine import TrainingEngine
from src.training.physics_informed_loss import PhysicsInformedLoss
from src.training.monitoring import TrainingMonitor
from src.physics.agt_no_architecture import AGTNOArchitecture

# Setup model and training components
model = AGTNOArchitecture(input_dim=1, hidden_dim=128, output_dim=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Setup training engine
config = load_experiment_config("config.yaml")
training_engine = TrainingEngine(model, config, device)

# Setup monitoring
monitor = TrainingMonitor(log_dir="training_logs")

# Training loop with monitoring
for epoch in range(config.training['epochs']):
    # Train epoch
    train_metrics = training_engine.train_epoch(train_loader)
    
    # Validate
    val_metrics = training_engine.validate(val_loader)
    
    # Log metrics
    all_metrics = {**train_metrics, **val_metrics}
    monitor.log_metrics(epoch, all_metrics)
    
    # Check convergence
    if monitor.check_convergence():
        print(f"Converged at epoch {epoch}")
        break
    
    # Save checkpoint
    if epoch % 10 == 0:
        training_engine.save_checkpoint(
            f"checkpoints/epoch_{epoch}.pth", 
            epoch, 
            val_metrics
        )

# Plot final results
monitor.plot_training_curves("final_training_curves.png")
```

### Custom Loss Function Training

```python
from src.training.physics_informed_loss import PhysicsInformedLoss
from src.training.advanced_loss_functions import AdvancedLossManager

# Setup custom loss functions
physics_loss = PhysicsInformedLoss(
    data_loss_fn=nn.CrossEntropyLoss(),
    physics_weight=0.1,
    adaptive_weighting=True
)

advanced_loss = AdvancedLossManager(
    consistency_weight=0.05,
    variational_weight=0.02
)

# Custom training loop
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Compute losses
        loss_components = physics_loss(
            predictions=outputs['predictions'],
            targets=targets,
            physics_residuals=outputs['physics_residuals']
        )
        
        # Add advanced losses
        advanced_losses = advanced_loss.get_total_advanced_loss(outputs)
        
        # Total loss
        total_loss = loss_components['total_loss'] + advanced_losses
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.6f}')
    
    print(f'Epoch {epoch} completed, Average Loss: {epoch_loss/len(train_loader):.6f}')
```

### Distributed Training

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def train_distributed(rank, world_size, config):
    """Distributed training function."""
    setup_distributed(rank, world_size)
    
    # Setup model
    model = AGTNOArchitecture(input_dim=1, hidden_dim=128, output_dim=10)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Setup training engine
    training_engine = TrainingEngine(model, config, torch.device(f'cuda:{rank}'))
    
    # Train
    trained_model, history = training_engine.train(train_loader, val_loader)
    
    cleanup_distributed()

# Launch distributed training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_distributed, args=(world_size, config), nprocs=world_size, join=True)
```

### Training with Mixed Precision

```python
from torch.cuda.amp import GradScaler, autocast

# Setup mixed precision training
model = AGTNOArchitecture(input_dim=1, hidden_dim=128, output_dim=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()

physics_loss = PhysicsInformedLoss()

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            outputs = model(data)
            loss_components = physics_loss(
                predictions=outputs['predictions'],
                targets=targets,
                physics_residuals=outputs['physics_residuals']
            )
            total_loss = loss_components['total_loss']
        
        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.6f}')
```

## Error Handling

### Common Training Issues

1. **Gradient Explosion**: Use gradient clipping
2. **NaN Values**: Check learning rate and data preprocessing
3. **Memory Issues**: Reduce batch size or use gradient checkpointing
4. **Physics Constraint Violations**: Adjust physics loss weights

### Error Handling Example

```python
import torch.nn.utils as nn_utils

def safe_training_step(model, data, targets, optimizer, loss_fn, max_grad_norm=1.0):
    """Safe training step with error handling."""
    try:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Check for NaN in outputs
        if torch.isnan(outputs['predictions']).any():
            raise ValueError("NaN detected in model predictions")
        
        # Compute loss
        loss_components = loss_fn(
            predictions=outputs['predictions'],
            targets=targets,
            physics_residuals=outputs['physics_residuals']
        )
        
        total_loss = loss_components['total_loss']
        
        # Check for NaN in loss
        if torch.isnan(total_loss):
            raise ValueError("NaN detected in loss computation")
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        nn_utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'data_loss': loss_components['data_loss'].item(),
            'physics_loss': loss_components['physics_loss'].item()
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA out of memory. Clearing cache and skipping batch.")
            torch.cuda.empty_cache()
            return None
        else:
            raise e
    
    except ValueError as e:
        print(f"Training error: {e}")
        return None

# Usage in training loop
for batch_idx, (data, targets) in enumerate(train_loader):
    metrics = safe_training_step(model, data, targets, optimizer, loss_fn)
    
    if metrics is not None:
        print(f"Batch {batch_idx}: Loss = {metrics['loss']:.6f}")
    else:
        print(f"Batch {batch_idx}: Skipped due to error")
```

## Best Practices

### Training Guidelines

1. **Learning Rate**: Start with 1e-3 and use learning rate scheduling
2. **Batch Size**: Use the largest batch size that fits in memory
3. **Physics Weights**: Start small (0.01-0.1) and increase gradually
4. **Monitoring**: Track both accuracy and physics residuals
5. **Checkpointing**: Save checkpoints regularly for recovery

### Performance Optimization

1. **Mixed Precision**: Use automatic mixed precision for faster training
2. **Gradient Checkpointing**: Save memory for large models
3. **Data Loading**: Use multiple workers and pin memory
4. **Model Compilation**: Use torch.compile for PyTorch 2.0+

### Example Optimized Training Setup

```python
# Optimized training configuration
def setup_optimized_training(model, config):
    """Setup optimized training configuration."""
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training['learning_rate'],
        weight_decay=config.training['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training['epochs'],
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.training['mixed_precision'] else None
    
    # Compile model for PyTorch 2.0+
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    return optimizer, scheduler, scaler

# Usage
optimizer, scheduler, scaler = setup_optimized_training(model, config)
```