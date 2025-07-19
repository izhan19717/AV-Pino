"""
Reproducible experiment configuration with proper random seeding.
"""

import os
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducible experiments."""
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    use_deterministic_algorithms: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Reproducibility
    reproducibility: ReproducibilityConfig
    
    # Model configuration
    model: Dict[str, Any]
    
    # Training configuration
    training: Dict[str, Any]
    
    # Data configuration
    data: Dict[str, Any]
    
    # Physics configuration
    physics: Dict[str, Any]
    
    # Inference configuration
    inference: Dict[str, Any]
    
    # Logging and output
    logging: Dict[str, Any]


class ExperimentManager:
    """Manages reproducible experiment configuration and execution."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize experiment manager."""
        self.config_path = config_path
        self.config = None
        
    def load_config(self, config_path: str) -> ExperimentConfig:
        """Load experiment configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to dataclass
        self.config = ExperimentConfig(**config_dict)
        return self.config
    
    def save_config(self, config: ExperimentConfig, output_path: str):
        """Save experiment configuration to YAML file."""
        config_dict = asdict(config)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def setup_reproducibility(self, config: Optional[ReproducibilityConfig] = None):
        """Set up reproducible random seeds and deterministic behavior."""
        if config is None:
            config = ReproducibilityConfig()
        
        # Set random seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
        
        # Set deterministic behavior
        if config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = config.benchmark
        
        if config.use_deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
        
        # Set environment variables for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        
        logger.info(f"Reproducibility setup complete with seed: {config.seed}")
    
    def create_default_config(self) -> ExperimentConfig:
        """Create default experiment configuration."""
        return ExperimentConfig(
            reproducibility=ReproducibilityConfig(),
            model={
                "architecture": "AGT-NO",
                "modes": 16,
                "width": 64,
                "layers": 4,
                "activation": "gelu",
                "dropout": 0.1
            },
            training={
                "batch_size": 32,
                "learning_rate": 1e-3,
                "epochs": 100,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "weight_decay": 1e-4,
                "gradient_clip": 1.0
            },
            data={
                "dataset": "CWRU",
                "data_path": "data/cwru",
                "sequence_length": 1024,
                "overlap": 0.5,
                "normalization": "standard",
                "augmentation": True
            },
            physics={
                "constraints": ["maxwell", "heat", "structural"],
                "loss_weights": {
                    "data": 1.0,
                    "physics": 0.1,
                    "consistency": 0.05
                },
                "coupling_strength": 0.1
            },
            inference={
                "batch_size": 1,
                "uncertainty_samples": 100,
                "confidence_threshold": 0.8,
                "optimization": {
                    "quantization": False,
                    "pruning": False,
                    "onnx_export": False
                }
            },
            logging={
                "level": "INFO",
                "log_dir": "logs",
                "tensorboard": True,
                "wandb": False,
                "save_frequency": 10
            }
        )
    
    def validate_config(self, config: ExperimentConfig) -> bool:
        """Validate experiment configuration."""
        try:
            # Check required fields
            assert config.reproducibility.seed >= 0, "Seed must be non-negative"
            assert config.training["batch_size"] > 0, "Batch size must be positive"
            assert config.training["learning_rate"] > 0, "Learning rate must be positive"
            assert config.training["epochs"] > 0, "Epochs must be positive"
            
            # Check physics constraints
            valid_constraints = ["maxwell", "heat", "structural"]
            for constraint in config.physics["constraints"]:
                assert constraint in valid_constraints, f"Invalid constraint: {constraint}"
            
            logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def setup_experiment(self, config_path: Optional[str] = None) -> ExperimentConfig:
        """Set up complete experiment with reproducibility."""
        if config_path:
            config = self.load_config(config_path)
        else:
            config = self.create_default_config()
        
        # Validate configuration
        if not self.validate_config(config):
            raise ValueError("Invalid experiment configuration")
        
        # Setup reproducibility
        self.setup_reproducibility(config.reproducibility)
        
        # Create output directories
        os.makedirs(config.logging["log_dir"], exist_ok=True)
        
        return config


def create_experiment_template(output_path: str = "configs/experiment_template.yaml"):
    """Create a template experiment configuration file."""
    manager = ExperimentManager()
    config = manager.create_default_config()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    manager.save_config(config, output_path)
    
    logger.info(f"Experiment template created at: {output_path}")


if __name__ == "__main__":
    # Create default experiment template
    create_experiment_template()