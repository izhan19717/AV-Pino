"""
Configuration management system for hyperparameters and physics constraints.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Data processing configuration."""
    window_size: int = 1024
    overlap_ratio: float = 0.5
    sequence_length: int = 256
    sampling_rate: int = 12000
    normalization_method: str = 'standard'
    augmentation_factor: int = 2
    noise_std_ratio: float = 0.05

@dataclass
class PhysicsConfig:
    """Physics constraints configuration."""
    # Motor parameters
    pole_pairs: int = 2
    rotor_inertia: float = 0.01  # kg⋅m²
    stator_resistance: float = 1.5  # Ω
    rotor_resistance: float = 1.2  # Ω
    mutual_inductance: float = 0.3  # H
    bearing_stiffness: float = 1e8  # N/m
    damping_coefficient: float = 100  # N⋅s/m
    thermal_capacity: float = 500  # J/K
    thermal_conductivity: float = 50  # W/(m⋅K)
    
    # Physics constraint weights
    maxwell_constraint_weight: float = 1.0
    heat_equation_weight: float = 1.0
    structural_dynamics_weight: float = 1.0
    coupling_constraint_weight: float = 0.5
    
    # Tolerance for constraint violations
    constraint_tolerance: float = 1e-6

@dataclass
class ModelConfig:
    """Neural operator model configuration."""
    # Architecture parameters
    modes: int = 16
    width: int = 64
    n_layers: int = 4
    activation: str = 'gelu'
    dropout_rate: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    
    # Loss function weights
    data_loss_weight: float = 1.0
    physics_loss_weight: float = 0.1
    consistency_loss_weight: float = 0.05
    
    # Uncertainty quantification
    enable_uncertainty: bool = True
    n_mc_samples: int = 100
    uncertainty_threshold: float = 0.1

@dataclass
class InferenceConfig:
    """Real-time inference configuration."""
    target_latency_ms: float = 1.0
    batch_size: int = 1
    enable_optimization: bool = True
    quantization_bits: int = 8
    enable_pruning: bool = True
    pruning_ratio: float = 0.2
    
    # Hardware constraints
    max_memory_mb: int = 512
    target_device: str = 'cpu'  # 'cpu', 'cuda', 'edge'

@dataclass
class AVPINOConfig:
    """Complete AV-PINO system configuration."""
    data: DataConfig
    physics: PhysicsConfig
    model: ModelConfig
    inference: InferenceConfig
    
    # System settings
    random_seed: int = 42
    log_level: str = 'INFO'
    output_dir: str = 'outputs'
    checkpoint_dir: str = 'checkpoints'
    
    # Experiment tracking
    experiment_name: str = 'av_pino_experiment'
    tags: list = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class ConfigManager:
    """
    Configuration manager for AV-PINO system.
    
    Handles loading, saving, and validation of configuration files
    with support for JSON and YAML formats.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = None
        
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> AVPINOConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            AVPINOConfig object
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self.config = self.get_default_config()
            return self.config
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                elif self.config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_dict = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
            self.config = self._dict_to_config(config_dict)
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            logger.info("Using default configuration")
            self.config = self.get_default_config()
        
        return self.config
    
    def save_config(self, config: AVPINOConfig, 
                   config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: AVPINOConfig object to save
            config_path: Path to save configuration file
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path:
            raise ValueError("No config path specified")
        
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict(config)
        
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                elif self.config_path.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
            raise
    
    def get_default_config(self) -> AVPINOConfig:
        """
        Get default configuration.
        
        Returns:
            Default AVPINOConfig object
        """
        return AVPINOConfig(
            data=DataConfig(),
            physics=PhysicsConfig(),
            model=ModelConfig(),
            inference=InferenceConfig()
        )
    
    def update_config(self, updates: Dict[str, Any]) -> AVPINOConfig:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates (nested keys supported with dots)
            
        Returns:
            Updated AVPINOConfig object
        """
        if self.config is None:
            self.config = self.get_default_config()
        
        config_dict = self._config_to_dict(self.config)
        
        # Apply updates
        for key, value in updates.items():
            self._set_nested_value(config_dict, key, value)
        
        self.config = self._dict_to_config(config_dict)
        return self.config
    
    def validate_config(self, config: AVPINOConfig) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate data config
            assert config.data.window_size > 0, "Window size must be positive"
            assert 0 < config.data.overlap_ratio < 1, "Overlap ratio must be between 0 and 1"
            assert config.data.sequence_length > 0, "Sequence length must be positive"
            assert config.data.sampling_rate > 0, "Sampling rate must be positive"
            
            # Validate physics config
            assert config.physics.pole_pairs > 0, "Pole pairs must be positive"
            assert config.physics.rotor_inertia > 0, "Rotor inertia must be positive"
            assert config.physics.constraint_tolerance > 0, "Constraint tolerance must be positive"
            
            # Validate model config
            assert config.model.modes > 0, "Number of modes must be positive"
            assert config.model.width > 0, "Model width must be positive"
            assert config.model.n_layers > 0, "Number of layers must be positive"
            assert config.model.learning_rate > 0, "Learning rate must be positive"
            assert config.model.batch_size > 0, "Batch size must be positive"
            
            # Validate inference config
            assert config.inference.target_latency_ms > 0, "Target latency must be positive"
            assert config.inference.batch_size > 0, "Inference batch size must be positive"
            
            logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return False
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AVPINOConfig:
        """Convert dictionary to AVPINOConfig object."""
        # Extract nested dictionaries
        data_dict = config_dict.get('data', {})
        physics_dict = config_dict.get('physics', {})
        model_dict = config_dict.get('model', {})
        inference_dict = config_dict.get('inference', {})
        
        # Create config objects
        data_config = DataConfig(**data_dict)
        physics_config = PhysicsConfig(**physics_dict)
        model_config = ModelConfig(**model_dict)
        inference_config = InferenceConfig(**inference_dict)
        
        # Extract top-level parameters
        top_level = {k: v for k, v in config_dict.items() 
                    if k not in ['data', 'physics', 'model', 'inference']}
        
        return AVPINOConfig(
            data=data_config,
            physics=physics_config,
            model=model_config,
            inference=inference_config,
            **top_level
        )
    
    def _config_to_dict(self, config: AVPINOConfig) -> Dict[str, Any]:
        """Convert AVPINOConfig object to dictionary."""
        return asdict(config)
    
    def _set_nested_value(self, dictionary: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested dictionary value using dot notation."""
        keys = key.split('.')
        current = dictionary
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get_config_summary(self) -> str:
        """
        Get configuration summary as formatted string.
        
        Returns:
            Formatted configuration summary
        """
        if self.config is None:
            return "No configuration loaded"
        
        summary = []
        summary.append("=== AV-PINO Configuration Summary ===")
        summary.append(f"Experiment: {self.config.experiment_name}")
        summary.append(f"Random Seed: {self.config.random_seed}")
        summary.append("")
        
        summary.append("Data Configuration:")
        summary.append(f"  Window Size: {self.config.data.window_size}")
        summary.append(f"  Sequence Length: {self.config.data.sequence_length}")
        summary.append(f"  Sampling Rate: {self.config.data.sampling_rate}")
        summary.append(f"  Normalization: {self.config.data.normalization_method}")
        summary.append("")
        
        summary.append("Model Configuration:")
        summary.append(f"  Modes: {self.config.model.modes}")
        summary.append(f"  Width: {self.config.model.width}")
        summary.append(f"  Layers: {self.config.model.n_layers}")
        summary.append(f"  Learning Rate: {self.config.model.learning_rate}")
        summary.append(f"  Batch Size: {self.config.model.batch_size}")
        summary.append("")
        
        summary.append("Physics Configuration:")
        summary.append(f"  Pole Pairs: {self.config.physics.pole_pairs}")
        summary.append(f"  Maxwell Weight: {self.config.physics.maxwell_constraint_weight}")
        summary.append(f"  Heat Equation Weight: {self.config.physics.heat_equation_weight}")
        summary.append("")
        
        summary.append("Inference Configuration:")
        summary.append(f"  Target Latency: {self.config.inference.target_latency_ms} ms")
        summary.append(f"  Target Device: {self.config.inference.target_device}")
        summary.append(f"  Optimization Enabled: {self.config.inference.enable_optimization}")
        
        return "\n".join(summary)