import numpy as np
import jax.numpy as jnp
from typing import Optional, List, Union, Tuple, Dict
import jax
import os

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

def _check_matplotlib():
    if plt is None:
        raise ImportError("matplotlib and seaborn are required for visualization. Please install them via 'pip install matplotlib seaborn'.")

def show_batch(
    batch: Union[np.ndarray, jnp.ndarray], 
    n_samples: int = 4, 
    figsize: tuple = (12, 4),
    title: Optional[str] = None,
    mean: Optional[Union[float, Tuple[float, ...]]] = None,
    std: Optional[Union[float, Tuple[float, ...]]] = None,
    channel_first: bool = False,
    save_path: Optional[str] = None
):
    """
    Visualize a batch of images.
    
    Args:
        batch: Batch of images (B, H, W, C) or (B, C, H, W) or (B, H, W).
        n_samples: Number of samples to show.
        figsize: Figure size.
        title: Title for the plot.
        mean: Mean for denormalization.
        std: Std for denormalization.
        channel_first: If True, assumes (B, C, H, W). Defaults to False (B, H, W, C).
        save_path: If provided, saves the plot to this path instead of showing it.
    """
    _check_matplotlib()
        
    if isinstance(batch, jnp.ndarray):
        batch = np.array(batch)
    
    if channel_first and batch.ndim == 4:
        batch = np.transpose(batch, (0, 2, 3, 1))
        
    n = min(n_samples, len(batch))
    
    # Denormalize
    if mean is not None and std is not None:
        batch = batch * np.array(std) + np.array(mean)
        
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
        
    for i in range(n):
        img = batch[i]
        
        # Clip to valid range
        if img.dtype == np.float32 or img.dtype == np.float64:
             img = np.clip(img, 0, 1)
        
        # Handle grayscale
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
            
        axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[i].axis('off')
        
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
        plt.close(fig)

def smooth_curve(points: List[float], factor: float = 0.8) -> List[float]:
    """Smoothing using exponential moving average."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_stats(
    losses: Union[List[float], Dict[str, List[float]]], 
    val_losses: Optional[Union[List[float], Dict[str, List[float]]]] = None,
    metrics: Optional[Dict[str, List[float]]] = None,
    title: str = "Training Progress",
    log_scale: bool = False,
    smoothing: float = 0.0,
    save_path: Optional[str] = None
):
    """
    Plot training statistics including losses and arbitrary metrics.
    
    Args:
        losses: List of training losses or dict of named losses.
        val_losses: List of validation losses or dict of named losses.
        metrics: Dictionary of other metrics to plot (e.g. {'accuracy': [...]}).
        title: Title of the plot.
        log_scale: Whether to use log scale for y-axis.
        smoothing: Smoothing factor (0.0 to 1.0) for loss curves.
        save_path: If provided, saves the plot to this path.
    """
    _check_matplotlib()
        
    n_plots = 1 + (1 if metrics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Helper to plot with optional smoothing
    def plot_with_smoothing(ax, data, label, style='-'):
        if smoothing > 0.0:
            # Plot raw data transparently
            line, = ax.plot(data, alpha=0.3, linestyle=style)
            color = line.get_color()
            
            # Plot smoothed data with same color
            smoothed = smooth_curve(data, smoothing)
            ax.plot(smoothed, label=f'{label}', color=color, linestyle=style)
        else:
            ax.plot(data, label=label, linestyle=style)

    # Plot Losses
    ax = axes[0]
    if isinstance(losses, dict):
        for name, vals in losses.items():
            plot_with_smoothing(ax, vals, f'Train {name}')
    else:
        plot_with_smoothing(ax, losses, 'Train Loss')
        
    if val_losses:
        if isinstance(val_losses, dict):
             for name, vals in val_losses.items():
                plot_with_smoothing(ax, vals, f'Val {name}', linestyle='--')
        else:
            plot_with_smoothing(ax, val_losses, 'Val Loss', linestyle='--')
            
    ax.set_title(f'{title} - Loss')
    ax.set_xlabel('Epoch/Step')
    ax.set_ylabel('Loss')
    if log_scale:
        ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    
    # Plot Metrics
    if metrics:
        ax = axes[1]
        for name, vals in metrics.items():
            ax.plot(vals, label=name)
        ax.set_title(f'{title} - Metrics')
        ax.set_xlabel('Epoch/Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
        plt.close(fig)
