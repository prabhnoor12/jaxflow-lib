"""
Enterprise Monitoring & Debugging Tools
======================================

Comprehensive monitoring, profiling, and debugging capabilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import logging
import threading
import psutil
import GPUtil
from typing import Optional, Dict, List, Any, Callable, Tuple
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .licensing import require_license, LicenseManager


logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Represents a tracked metric."""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    min_val: float = float('inf')
    max_val: float = -float('inf')
    avg_val: float = 0.0
    
    def record(self, value: float):
        self.values.append(value)
        self.timestamps.append(time.time())
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.avg_val = sum(self.values) / len(self.values)


class SystemMonitor:
    """
    Real-time system performance monitor.
    
    Features:
    - CPU, GPU, memory, and disk monitoring
    - Real-time metrics dashboard
    - Alerting system
    - Performance bottleneck detection
    """
    
    @require_license("monitoring")
    def __init__(
        self,
        license_manager: Optional[LicenseManager] = None,
        sample_interval: float = 1.0,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.license_manager = license_manager or LicenseManager()
        self.sample_interval = sample_interval
        self.metrics: Dict[str, Metric] = defaultdict(lambda: Metric(name=""))
        self.alert_thresholds = alert_thresholds or {
            'cpu_usage': 90.0,
            'gpu_memory': 90.0,
            'system_memory': 85.0,
        }
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._alert_callbacks: List[Callable] = []
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize metric tracking."""
        self.metrics['cpu_usage'] = Metric(name='cpu_usage')
        self.metrics['memory_usage'] = Metric(name='memory_usage')
        self.metrics['disk_usage'] = Metric(name='disk_usage')
        
        if self._has_gpu():
            for i in range(len(GPUtil.getGPUs())):
                self.metrics[f'gpu_{i}_usage'] = Metric(name=f'gpu_{i}_usage')
                self.metrics[f'gpu_{i}_memory'] = Metric(name=f'gpu_{i}_memory')
                self.metrics[f'gpu_{i}_temperature'] = Metric(name=f'gpu_{i}_temperature')
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            return len(GPUtil.getGPUs()) > 0
        except Exception:
            return False
    
    def start(self):
        """Start the monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop(self):
        """Stop the monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._sample_metrics()
                self._check_alerts()
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _sample_metrics(self):
        """Sample current system metrics."""
        # CPU metrics
        self.metrics['cpu_usage'].record(psutil.cpu_percent(interval=0.1))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics['memory_usage'].record(memory.percent)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics['disk_usage'].record(disk.percent)
        
        # GPU metrics
        if self._has_gpu():
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                self.metrics[f'gpu_{i}_usage'].record(gpu.load * 100)
                self.metrics[f'gpu_{i}_memory'].record(gpu.memoryUtil * 100)
                self.metrics[f'gpu_{i}_temperature'].record(gpu.temperature)
    
    def _check_alerts(self):
        """Check if any metrics exceed thresholds."""
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                if metric.values and metric.values[-1] > threshold:
                    self._trigger_alert(metric_name, metric.values[-1], threshold)
    
    def _trigger_alert(self, metric_name: str, value: float, threshold: float):
        """Trigger an alert."""
        alert_msg = f"ALERT: {metric_name} = {value:.1f}% (threshold: {threshold}%)"
        logger.warning(alert_msg)
        
        for callback in self._alert_callbacks:
            try:
                callback(metric_name, value, threshold)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """Register an alert callback."""
        self._alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        return {
            name: metric.values[-1] if metric.values else 0.0
            for name, metric in self.metrics.items()
        }
    
    def get_metric_history(self, metric_name: str) -> Tuple[List[float], List[float]]:
        """Get historical metric data."""
        if metric_name not in self.metrics:
            return [], []
        
        metric = self.metrics[metric_name]
        return list(metric.values), list(metric.timestamps)
    
    def report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'current': self.get_current_metrics(),
            'statistics': {},
        }
        
        for name, metric in self.metrics.items():
            if metric.values:
                report['statistics'][name] = {
                    'min': metric.min_val,
                    'max': metric.max_val,
                    'avg': metric.avg_val,
                    'latest': metric.values[-1]
                }
        
        return report


class JAXProfiler:
    """
    Advanced JAX profiler for debugging and performance optimization.
    """
    
    @require_license("monitoring")
    def __init__(
        self,
        license_manager: Optional[LicenseManager] = None,
        trace_dir: str = "./jax_traces"
    ):
        self.license_manager = license_manager or LicenseManager()
        self.trace_dir = trace_dir
        self._profiles: Dict[str, List[float]] = defaultdict(list)
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Profile a single function call."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        
        func_name = getattr(func, '__name__', 'anonymous')
        self._profiles[func_name].append(duration)
        
        return result, duration
    
    def jit_profile(self, func: Callable, *args, **kwargs):
        """Profile a JIT-compiled function with detailed timing."""
        jitted = jax.jit(func)
        
        # Warmup
        _ = jitted(*args, **kwargs)
        
        # Profile multiple runs
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            _ = jitted(*args, **kwargs)
            jax.block_until_ready(_)
            timings.append(time.perf_counter() - start)
        
        return {
            'mean': np.mean(timings),
            'std': np.std(timings),
            'min': np.min(timings),
            'max': np.max(timings),
            'raw': timings
        }
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get profiling summary."""
        summary = {}
        for name, times in self._profiles.items():
            if times:
                summary[name] = {
                    'calls': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times)
                }
        return summary


class TrainingMonitor:
    """
    Comprehensive training monitoring with real-time insights.
    """
    
    @require_license("monitoring")
    def __init__(
        self,
        license_manager: Optional[LicenseManager] = None,
        log_dir: str = "./training_logs",
        window_size: int = 100
    ):
        self.license_manager = license_manager or LicenseManager()
        self.log_dir = log_dir
        self.window_size = window_size
        
        self.losses = deque(maxlen=window_size)
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.grad_norms = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)
        
        self._start_time: Optional[float] = None
        self._last_step_time: Optional[float] = None
        self._step_count = 0
    
    def on_train_start(self):
        """Called when training starts."""
        self._start_time = time.time()
        logger.info("Training monitoring started")
    
    def on_step_end(self, loss: float, metrics: Optional[Dict[str, float]] = None,
                    grad_norm: Optional[float] = None):
        """Record training step data."""
        self._step_count += 1
        
        self.losses.append(loss)
        
        if metrics:
            for name, value in metrics.items():
                self.metrics[name].append(value)
        
        if grad_norm:
            self.grad_norms.append(grad_norm)
        
        if self._last_step_time:
            self.step_times.append(time.time() - self._last_step_time)
        
        self._last_step_time = time.time()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        stats = {
            'step': self._step_count,
            'elapsed_time': elapsed,
            'steps_per_second': self._step_count / elapsed if elapsed > 0 else 0,
        }
        
        if self.losses:
            stats['loss'] = {
                'latest': self.losses[-1],
                'avg': np.mean(self.losses),
                'min': np.min(self.losses),
                'std': np.std(self.losses)
            }
        
        if self.step_times:
            stats['step_time'] = {
                'avg': np.mean(self.step_times),
                'std': np.std(self.step_times)
            }
        
        if self.grad_norms:
            stats['grad_norm'] = {
                'latest': self.grad_norms[-1],
                'avg': np.mean(self.grad_norms),
                'max': np.max(self.grad_norms)
            }
        
        stats['metrics'] = {}
        for name, values in self.metrics.items():
            if values:
                stats['metrics'][name] = {
                    'latest': values[-1],
                    'avg': np.mean(values)
                }
        
        return stats
    
    def detect_anomalies(self) -> List[str]:
        """Detect training anomalies."""
        anomalies = []
        
        if len(self.losses) >= 10:
            # Check for exploding loss
            recent_losses = list(self.losses)[-10:]
            if np.mean(recent_losses[-3:]) > 2 * np.mean(recent_losses[:7]):
                anomalies.append("Possible exploding loss detected")
            
            # Check for stagnating loss
            if np.std(recent_losses) < 0.001 and len(self.losses) > 50:
                anomalies.append("Loss appears to be stagnating")
        
        if len(self.grad_norms) >= 10:
            recent_grads = list(self.grad_norms)[-10:]
            if np.max(recent_grads) > 10 * np.mean(recent_grads[:7]):
                anomalies.append("Possible gradient explosion detected")
        
        return anomalies
