"""
Logging configuration for AV-PINO system.

Provides centralized logging setup for training and inference monitoring.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class MetricsLogger:
    """Logger for training and inference metrics."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup metrics logger
        self.metrics_logger = logging.getLogger('av_pino.metrics')
        self.metrics_logger.setLevel(logging.INFO)
        
        # Create file handler for metrics
        metrics_file = self.log_dir / 'metrics.jsonl'
        metrics_handler = logging.FileHandler(metrics_file)
        metrics_handler.setFormatter(JSONFormatter())
        self.metrics_logger.addHandler(metrics_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.metrics_logger.propagate = False
    
    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log training metrics.
        
        Args:
            epoch: Training epoch
            metrics: Dictionary of metrics
        """
        self.metrics_logger.info("Training metrics", extra={
            'type': 'training',
            'epoch': epoch,
            **metrics
        })
    
    def log_validation_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log validation metrics.
        
        Args:
            epoch: Training epoch
            metrics: Dictionary of metrics
        """
        self.metrics_logger.info("Validation metrics", extra={
            'type': 'validation',
            'epoch': epoch,
            **metrics
        })
    
    def log_inference_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log inference metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        self.metrics_logger.info("Inference metrics", extra={
            'type': 'inference',
            **metrics
        })
    
    def log_physics_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log physics constraint metrics.
        
        Args:
            metrics: Dictionary of physics metrics
        """
        self.metrics_logger.info("Physics metrics", extra={
            'type': 'physics',
            **metrics
        })

def setup_logging(log_level: str = 'INFO',
                 log_dir: Optional[str] = None,
                 enable_file_logging: bool = True,
                 enable_json_logging: bool = False,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> MetricsLogger:
    """
    Setup logging configuration for AV-PINO system.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_dir: Directory for log files (optional)
        enable_file_logging: Whether to enable file logging
        enable_json_logging: Whether to use JSON formatting
        max_file_size: Maximum size of log files in bytes
        backup_count: Number of backup log files to keep
        
    Returns:
        MetricsLogger instance for metrics logging
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if enable_json_logging:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Setup file logging if enabled
    metrics_logger = None
    if enable_file_logging and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Main log file with rotation
        log_file = log_path / 'av_pino.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        
        if enable_json_logging:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Setup metrics logger
        metrics_logger = MetricsLogger(log_path)
        
        # Log startup message
        logging.info(f"Logging initialized - Level: {log_level}, Dir: {log_dir}")
    
    # Setup specific loggers
    _setup_library_loggers()
    
    return metrics_logger

def _setup_library_loggers():
    """Setup logging for third-party libraries."""
    # Reduce verbosity of common libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Set specific levels for scientific libraries
    logging.getLogger('scipy').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)

class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self, logger_name: str = 'av_pino.performance'):
        """
        Initialize performance logger.
        
        Args:
            logger_name: Name of the logger
        """
        self.logger = logging.getLogger(logger_name)
        self.timers = {}
    
    def start_timer(self, name: str) -> None:
        """
        Start a performance timer.
        
        Args:
            name: Timer name
        """
        import time
        self.timers[name] = time.time()
    
    def end_timer(self, name: str, log_result: bool = True) -> float:
        """
        End a performance timer and optionally log the result.
        
        Args:
            name: Timer name
            log_result: Whether to log the timing result
            
        Returns:
            Elapsed time in seconds
        """
        import time
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        if log_result:
            self.logger.info(f"Timer '{name}': {elapsed:.4f} seconds")
        
        return elapsed
    
    def log_memory_usage(self, context: str = "") -> None:
        """
        Log current memory usage.
        
        Args:
            context: Context description
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.info(
                f"Memory usage {context}: "
                f"RSS={memory_info.rss / 1024 / 1024:.1f}MB, "
                f"VMS={memory_info.vms / 1024 / 1024:.1f}MB"
            )
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
    
    def log_system_info(self) -> None:
        """Log system information."""
        try:
            import platform
            import psutil
            
            self.logger.info(f"System: {platform.system()} {platform.release()}")
            self.logger.info(f"Python: {platform.python_version()}")
            self.logger.info(f"CPU cores: {psutil.cpu_count()}")
            self.logger.info(f"Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
            
        except ImportError:
            self.logger.warning("System info logging requires psutil")
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")

# Context manager for performance timing
class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, logger: Optional[PerformanceLogger] = None):
        """
        Initialize timing context.
        
        Args:
            name: Timer name
            logger: Performance logger instance
        """
        self.name = name
        self.logger = logger or PerformanceLogger()
    
    def __enter__(self):
        self.logger.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.end_timer(self.name)