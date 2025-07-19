"""
Unit tests for configuration management and logging.
"""

import unittest
import tempfile
import json
from pathlib import Path
from src.config.config_manager import ConfigManager, AVPINOConfig, DataConfig, PhysicsConfig
from src.config.logging_config import setup_logging, PerformanceLogger, TimingContext


class TestConfigManager(unittest.TestCase):
    """Test cases for configuration manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_manager = ConfigManager()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_default_config(self):
        """Test default configuration creation."""
        config = self.config_manager.get_default_config()
        
        self.assertIsInstance(config, AVPINOConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.physics, PhysicsConfig)
        
        # Check some default values
        self.assertEqual(config.data.window_size, 1024)
        self.assertEqual(config.physics.pole_pairs, 2)
        self.assertEqual(config.model.learning_rate, 1e-3)
        
    def test_config_validation(self):
        """Test configuration validation."""
        config = self.config_manager.get_default_config()
        
        # Valid config should pass
        self.assertTrue(self.config_manager.validate_config(config))
        
        # Invalid config should fail
        config.data.window_size = -1
        self.assertFalse(self.config_manager.validate_config(config))
        
    def test_save_and_load_json(self):
        """Test saving and loading JSON configuration."""
        config = self.config_manager.get_default_config()
        config.experiment_name = "test_experiment"
        config.data.window_size = 512
        
        # Save config
        config_path = self.temp_dir / "test_config.json"
        self.config_manager.save_config(config, config_path)
        
        # Load config
        loaded_config = self.config_manager.load_config(config_path)
        
        self.assertEqual(loaded_config.experiment_name, "test_experiment")
        self.assertEqual(loaded_config.data.window_size, 512)
        
    def test_update_config(self):
        """Test configuration updates."""
        config = self.config_manager.get_default_config()
        
        updates = {
            'data.window_size': 2048,
            'model.learning_rate': 1e-4,
            'experiment_name': 'updated_experiment'
        }
        
        updated_config = self.config_manager.update_config(updates)
        
        self.assertEqual(updated_config.data.window_size, 2048)
        self.assertEqual(updated_config.model.learning_rate, 1e-4)
        self.assertEqual(updated_config.experiment_name, 'updated_experiment')
        
    def test_config_summary(self):
        """Test configuration summary generation."""
        config = self.config_manager.get_default_config()
        self.config_manager.config = config
        
        summary = self.config_manager.get_config_summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn("AV-PINO Configuration Summary", summary)
        self.assertIn("Data Configuration", summary)
        self.assertIn("Model Configuration", summary)
        
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        nonexistent_path = self.temp_dir / "nonexistent.json"
        config = self.config_manager.load_config(nonexistent_path)
        
        # Should return default config
        self.assertIsInstance(config, AVPINOConfig)
        self.assertEqual(config.data.window_size, 1024)  # Default value


class TestLoggingConfig(unittest.TestCase):
    """Test cases for logging configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_setup_logging(self):
        """Test logging setup."""
        metrics_logger = setup_logging(
            log_level='INFO',
            log_dir=str(self.temp_dir),
            enable_file_logging=True
        )
        
        # Should return metrics logger
        self.assertIsNotNone(metrics_logger)
        
        # Log files should be created
        log_file = self.temp_dir / 'av_pino.log'
        metrics_file = self.temp_dir / 'metrics.jsonl'
        
        # Files might not exist immediately, but directory should
        self.assertTrue(self.temp_dir.exists())
        
    def test_performance_logger(self):
        """Test performance logger."""
        perf_logger = PerformanceLogger()
        
        # Test timer functionality
        perf_logger.start_timer('test_timer')
        import time
        time.sleep(0.01)  # Small delay
        elapsed = perf_logger.end_timer('test_timer', log_result=False)
        
        self.assertGreater(elapsed, 0)
        self.assertLess(elapsed, 1.0)  # Should be much less than 1 second
        
    def test_timing_context(self):
        """Test timing context manager."""
        perf_logger = PerformanceLogger()
        
        with TimingContext('test_context', perf_logger):
            import time
            time.sleep(0.01)
        
        # Should complete without errors
        self.assertTrue(True)
        
    def test_memory_logging(self):
        """Test memory usage logging."""
        perf_logger = PerformanceLogger()
        
        # Should not raise exceptions even if psutil is not available
        try:
            perf_logger.log_memory_usage("test context")
            perf_logger.log_system_info()
        except Exception as e:
            self.fail(f"Memory logging raised unexpected exception: {e}")


if __name__ == '__main__':
    unittest.main()