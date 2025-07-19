"""
Memory management for efficient real-time inference.

Provides memory optimization strategies including memory pooling,
garbage collection control, and memory usage monitoring.
"""

import gc
import psutil
import torch
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_used_mb: Optional[float] = None
    gpu_available_mb: Optional[float] = None


class MemoryPool:
    """Memory pool for tensor reuse to reduce allocation overhead."""
    
    def __init__(self, max_pool_size: int = 100):
        self.max_pool_size = max_pool_size
        self.pools: Dict[Tuple[torch.dtype, torch.device], List[torch.Tensor]] = {}
        self._lock = threading.Lock()
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, 
                   device: torch.device) -> torch.Tensor:
        """Get a tensor from pool or create new one."""
        key = (dtype, device)
        
        with self._lock:
            if key in self.pools and self.pools[key]:
                tensor = self.pools[key].pop()
                if tensor.shape == shape:
                    tensor.zero_()
                    return tensor
                else:
                    # Reshape if possible, otherwise create new
                    if tensor.numel() >= torch.prod(torch.tensor(shape)):
                        return tensor.view(shape).zero_()
        
        # Create new tensor if pool is empty or no suitable tensor found
        return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to pool for reuse."""
        key = (tensor.dtype, tensor.device)
        
        with self._lock:
            if key not in self.pools:
                self.pools[key] = []
            
            if len(self.pools[key]) < self.max_pool_size:
                # Detach tensor to avoid gradient tracking
                tensor = tensor.detach()
                self.pools[key].append(tensor)
    
    def clear_pool(self) -> None:
        """Clear all tensors from pool."""
        with self._lock:
            self.pools.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class MemoryManager:
    """
    Efficient memory management for real-time inference.
    
    Features:
    - Memory pooling for tensor reuse
    - Automatic garbage collection control
    - Memory usage monitoring and alerts
    - GPU memory optimization
    """
    
    def __init__(self, 
                 enable_pooling: bool = True,
                 pool_size: int = 100,
                 gc_threshold: float = 0.8,
                 monitoring_interval: float = 1.0):
        """
        Initialize memory manager.
        
        Args:
            enable_pooling: Enable tensor memory pooling
            pool_size: Maximum number of tensors in pool
            gc_threshold: Memory usage threshold to trigger GC (0-1)
            monitoring_interval: Memory monitoring interval in seconds
        """
        self.enable_pooling = enable_pooling
        self.gc_threshold = gc_threshold
        self.monitoring_interval = monitoring_interval
        
        # Initialize memory pool
        self.memory_pool = MemoryPool(pool_size) if enable_pooling else None
        
        # Memory monitoring
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._memory_stats: List[MemoryStats] = []
        self._max_stats_history = 1000
        
        # Memory optimization settings
        self._original_gc_settings = None
        
        logger.info(f"MemoryManager initialized with pooling={enable_pooling}, "
                   f"gc_threshold={gc_threshold}")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        # System memory
        memory = psutil.virtual_memory()
        total_mb = memory.total / (1024 * 1024)
        used_mb = memory.used / (1024 * 1024)
        available_mb = memory.available / (1024 * 1024)
        
        # GPU memory if available
        gpu_total_mb = gpu_used_mb = gpu_available_mb = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_total_mb = gpu_memory / (1024 * 1024)
            gpu_used_mb = gpu_allocated / (1024 * 1024)
            gpu_available_mb = gpu_total_mb - gpu_used_mb
        
        return MemoryStats(
            total_memory_mb=total_mb,
            used_memory_mb=used_mb,
            available_memory_mb=available_mb,
            gpu_memory_mb=gpu_total_mb,
            gpu_used_mb=gpu_used_mb,
            gpu_available_mb=gpu_available_mb
        )
    
    def optimize_for_inference(self) -> None:
        """Optimize memory settings for inference."""
        # Disable automatic garbage collection during inference
        self._original_gc_settings = gc.get_threshold()
        gc.set_threshold(0, 0, 0)
        gc.disable()
        
        # Set PyTorch memory management for inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        logger.info("Memory optimized for inference")
    
    def restore_memory_settings(self) -> None:
        """Restore original memory settings."""
        if self._original_gc_settings is not None:
            gc.set_threshold(*self._original_gc_settings)
            gc.enable()
            self._original_gc_settings = None
        
        logger.info("Memory settings restored")
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        stats = self.get_memory_stats()
        memory_usage = stats.used_memory_mb / stats.total_memory_mb
        
        if memory_usage > self.gc_threshold:
            logger.warning(f"High memory usage detected: {memory_usage:.2%}")
            return True
        
        if stats.gpu_memory_mb and stats.gpu_used_mb:
            gpu_usage = stats.gpu_used_mb / stats.gpu_memory_mb
            if gpu_usage > self.gc_threshold:
                logger.warning(f"High GPU memory usage: {gpu_usage:.2%}")
                return True
        
        return False
    
    def force_cleanup(self) -> None:
        """Force memory cleanup."""
        if self.memory_pool:
            self.memory_pool.clear_pool()
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Forced memory cleanup completed")
    
    @contextmanager
    def managed_inference(self):
        """Context manager for managed inference with automatic cleanup."""
        try:
            self.optimize_for_inference()
            yield self
        finally:
            # Check if cleanup is needed
            if self.check_memory_pressure():
                self.force_cleanup()
            self.restore_memory_settings()
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                   device: Optional[torch.device] = None) -> torch.Tensor:
        """Get tensor with memory management."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.memory_pool:
            return self.memory_pool.get_tensor(shape, dtype, device)
        else:
            return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to memory manager."""
        if self.memory_pool:
            self.memory_pool.return_tensor(tensor)
    
    def start_monitoring(self) -> None:
        """Start memory monitoring in background thread."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        logger.info("Memory monitoring stopped")
    
    def _monitor_memory(self) -> None:
        """Background memory monitoring loop."""
        import time
        
        while self._monitoring_active:
            try:
                stats = self.get_memory_stats()
                
                # Store stats with size limit
                self._memory_stats.append(stats)
                if len(self._memory_stats) > self._max_stats_history:
                    self._memory_stats.pop(0)
                
                # Check for memory pressure
                if self.check_memory_pressure():
                    logger.warning("Memory pressure detected during monitoring")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                break
    
    def get_memory_history(self) -> List[MemoryStats]:
        """Get memory usage history."""
        return self._memory_stats.copy()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        current_stats = self.get_memory_stats()
        
        summary = {
            'current_memory_mb': current_stats.used_memory_mb,
            'current_memory_percent': (current_stats.used_memory_mb / 
                                     current_stats.total_memory_mb * 100),
            'available_memory_mb': current_stats.available_memory_mb,
            'pooling_enabled': self.enable_pooling,
            'gc_threshold': self.gc_threshold
        }
        
        if current_stats.gpu_memory_mb:
            summary.update({
                'gpu_memory_mb': current_stats.gpu_used_mb,
                'gpu_memory_percent': (current_stats.gpu_used_mb / 
                                     current_stats.gpu_memory_mb * 100),
                'gpu_available_mb': current_stats.gpu_available_mb
            })
        
        # Add history statistics if available
        if self._memory_stats:
            memory_usage = [s.used_memory_mb for s in self._memory_stats[-100:]]
            summary.update({
                'avg_memory_mb': sum(memory_usage) / len(memory_usage),
                'max_memory_mb': max(memory_usage),
                'min_memory_mb': min(memory_usage)
            })
        
        return summary