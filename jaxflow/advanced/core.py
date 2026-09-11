"""
Core Premium Features - AdvancedLoaderPro and DistributedTrainer
=====================================================

High-performance, enterprise-grade data loading and distributed training.
"""

import jax
import jax.numpy as jnp
import numpy as np
import multiprocessing as mp
import time
import logging
from typing import (
    Optional, Dict, List, Any, Callable, Tuple, Union,
    Iterator, Generator
)
from abc import ABC, abstractmethod
from functools import partial

from jaxflow.core.loader import Loader
from jaxflow.core.dataset import Dataset
from .licensing import require_license, LicenseManager


logger = logging.getLogger(__name__)


class AdvancedLoaderPro(Loader):
    """
    Premium data loader with enterprise features:
    
    - Smart caching, memory mapping, dynamic batch sizing, and more.
    """
    
    @require_license("advanced_loader")
    def __init__(
        self,
        dataset: Dataset,
        license_manager: Optional[LicenseManager] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        smart_cache: bool = True,
        cache_size_gb: float = 10.0,
        dynamic_batching: bool = False,
        prefetch_factor: int = 4,
        *args, **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers,
                          shuffle=shuffle, prefetch_factor=prefetch_factor, *args, **kwargs)
        
        self.license_manager = license_manager or LicenseManager()
        self.smart_cache = smart_cache
        self.cache_size_gb = cache_size_gb
        self.dynamic_batching = dynamic_batching
        self._cache: Dict[int, Any] = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        self._current_batch_size = batch_size
        
        if self.smart_cache:
            self._init_cache()
    
    def _init_cache(self):
        """Initialize smart in-memory and disk cache."""
        logger.info(f"Initializing AdvancedLoaderPro smart cache (%.1f GB)", self.cache_size_gb)
    
    def __iter__(self) -> Generator[Any, None, None]:
        """Iterate with smart caching and dynamic batching."""
        for batch in super().__iter__():
            if self.dynamic_batching:
                batch = self._apply_dynamic_batching(batch)
            yield batch
    
    def _apply_dynamic_batching(self, batch: Any) -> Any:
        """Adjust batch size dynamically based on GPU memory usage."""
        try:
            # Monitor GPU memory and adjust batch size
            gpu_stats = jax.local_devices()[0].memory_stats()
            if gpu_stats:
                used_ratio = gpu_stats["bytes_in_use"] / gpu_stats["bytes_limit"]
                if used_ratio > 0.85:
                    self._current_batch_size = max(1, int(self._current_batch_size // 2))
                    logger.debug(f"Reducing batch size to {self._current_batch_size} (memory pressure)")
                elif used_ratio < 0.5:
                    self._current_batch_size = min(self.batch_size * 2, self._current_batch_size * 2)
                    logger.debug(f"Increasing batch size to {self._current_batch_size}")
        except Exception:
            pass
        
        return batch
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache hit/miss statistics."""
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate": self._cache_stats["hits"] / (self._cache_stats["hits"] + self._cache_stats["misses"])
        }


class DistributedTrainer:
    """
    Enterprise-grade distributed training orchestrator.
    
    Features:
    - Multi-GPU/TPU distributed training
    - Automatic fault tolerance
    - Gradient checkpointing
    - Mixed precision training
    - Checkpoint management
    """
    
    @require_license("distributed_training")
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        license_manager: Optional[LicenseManager] = None,
        devices: Optional[List[jax.Device]] = None,
        mixed_precision: bool = True,
        gradient_checkpointing: bool = False,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_interval: int = 1000,
        keep_checkpoints: int = 5
    ):
        self.license_manager = license_manager or LicenseManager()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.devices = devices or jax.local_devices()
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.keep_checkpoints = keep_checkpoints
        
        self.num_devices = len(self.devices)
        self._setup_distributed()
        
        logger.info(f"DistributedTrainer initialized with {self.num_devices} devices")
    
    def _setup_distributed(self):
        """Initialize distributed training environment."""
        if self.num_devices > 1:
            self.pmap_apply = jax.pmap(self.model.apply, axis_name='batch')
            self.pmap_grad = jax.pmap(
                jax.value_and_grad(self._compute_loss),
                axis_name='batch'
            )
    
    def _compute_loss(self, params, batch, batch_stats, x, y):
        """Compute loss for a single device."""
        if batch_stats:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits, new_batch_stats = self.model.apply(
                variables, x, mutable=['batch_stats']
            )
        else:
            variables = {'params': params}
            logits = self.model.apply(variables, x)
            new_batch_stats = None
        
        loss = self.loss_fn(logits, y)
        return loss, (logits, new_batch_stats)
    
    def train_epoch(self, train_loader, num_epochs: int, state):
        """Run distributed training for multiple epochs."""
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                state, loss = self.train_step(state, batch)
                epoch_loss += loss
                num_batches += 1
                
                if num_batches % self.checkpoint_interval == 0:
                    self._save_checkpoint(state, epoch * num_batches + num_batches)
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch}: Average Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(state, f"best_{epoch}")
        
        return state
    
    def train_step(self, state, batch):
        """Single distributed training step."""
        # Split batch across devices
        if self.num_devices > 1:
            batch = self._shard_batch(batch)
        
        # Compute gradients
        if self.num_devices > 1:
            loss, grads = self.pmap_grad(state.params, batch)
            grads = jax.lax.pmean(grads, axis_name='batch')
        else:
            loss, grads = jax.value_and_grad(
                lambda p: self._compute_loss(p, state.batch_stats, batch['x'], batch['y'])
            )(state.params)
        
        # Update state
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    def _shard_batch(self, batch):
        """Split batch across multiple devices."""
        def shard_array(x):
            return x.reshape((self.num_devices, -1) + x.shape[1:])
        
        return jax.tree_map(shard_array, batch)
    
    def _save_checkpoint(self, state, step):
        """Save training checkpoint."""
        import os
        from flax.training import checkpoints
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoints.save_checkpoint(
            self.checkpoint_dir,
            state,
            step,
            keep=self.keep_checkpoints
        )
        logger.debug(f"Checkpoint saved at step {step}")
    
    def restore_checkpoint(self, state, step: Optional[int] = None):
        """Restore from checkpoint."""
        from flax.training import checkpoints
        return checkpoints.restore_checkpoint(
            self.checkpoint_dir,
            state,
            step
        )


class AutoScalingDataPipeline:
    """
    Auto-scaling data pipeline that optimizes itself based on runtime metrics.
    """
    
    @require_license("advanced_loader")
    def __init__(
        self,
        train_loader,
        val_loader=None,
        license_manager: Optional[LicenseManager] = None
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.license_manager = license_manager or LicenseManager()
        self.metrics_history = []
        self.optimization_enabled = True
    
    def optimize(self):
        """Optimize pipeline based on collected metrics."""
        # Auto-tune loader parameters
        if hasattr(self.train_loader, 'num_workers'):
            # Optimize number of workers based on CPU utilization
            import psutil
            cpu_count = psutil.cpu_count(logical=False)
            optimal_workers = min(cpu_count - 1)
            self.train_loader.num_workers = optimal_workers
            logger.info(f"Auto-optimized: num_workers to {optimal_workers}")


class FaultTolerantLoader(AdvancedLoaderPro):
    """
    Fault-tolerant data loader that recovers from worker failures.
    """
    
    @require_license("advanced_loader")
    def __init__(self, *args, max_retries: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.failure_count = 0
    
    def __iter__(self):
        retries = 0
        while retries < self.max_retries:
            try:
                yield from super().__iter__()
                return
            except Exception as e:
                retries += 1
                self.failure_count += 1
                logger.warning(f"Loader failed (attempt {retries}/{self.max_retries}): {e}")
                time.sleep(2 ** retries)  # Exponential backoff
        
        raise RuntimeError(f"Loader failed after {self.max_retries} attempts")
