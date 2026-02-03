
# JaxFlow

[![PyPI version](https://badge.fury.io/py/jaxflow.svg)](https://badge.fury.io/py/jaxflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**JaxFlow** is a high-performance, PyTree-native data loading and processing library designed specifically for the JAX and Flax ecosystem. 

Unlike generic data loaders, JaxFlow is built from the ground up to handle JAX's specific needs—like handling arbitrary PyTrees, efficient prefetching to devices (GPU/TPU), and seamless integration with `jax.jit` and `jax.pmap`. It also provides a rich set of integrations for **Flax**, **Optax**, and **HuggingFace Transformers**, making it a complete toolkit for JAX-based deep learning.

## 🚀 Key Features

*   **PyTree Native**: Data loaders yield PyTrees (dicts, tuples, lists, custom classes) directly, ready for `jax.tree_map`.
*   **JAX Device Prefetching**: Automatically prefetches batches to the target device (GPU/TPU) to minimize host-device transfer bottlenecks.
*   **Full Ecosystem Integration**: Built-in helpers for **Flax** (TrainState, checkpoints), **Optax** (optimizers, schedulers), and **HuggingFace** (model/tokenizer loading).
*   **Torch-like API**: Familiar `Dataset` and `Loader` API for those coming from PyTorch, but optimized for JAX.
*   **Multiprocessing**: Robust multiprocessing workers for parallel data loading and augmentation.
*   **Composability**: Flexible `transforms` module for composing image and data augmentations.
*   **Visualization**: Built-in tools in `jaxflow.viz` to quickly inspect batches and training curves.
*   **CLI Tools**: Includes a command-line interface for benchmarking system performance (`python -m jaxflow.cli benchmark`).

## 📦 Installation

```bash
# Install from PyPI
pip install jaxflow-lib

# Install with visualization support
pip install jaxflow-lib[viz]

# Install from source
pip install .
```

## ⚡ Quick Start

### 1. Data Loading

Here's how to create a simple dataset and iterate over it:

```python
import jax.numpy as jnp
import numpy as np
from jaxflow import Dataset, Loader, transforms

# Define a custom dataset
class RandomDataset(Dataset):
    def __init__(self, length=1000):
        self.length = length
        self.transform = transforms.Compose([
            transforms.ToArray(dtype=jnp.float32),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a dict (PyTree)
        image = np.random.rand(28, 28, 1)
        label = np.random.randint(0, 10)
        return {
            "image": self.transform(image),
            "label": label
        }

# Create a loader
dataset = RandomDataset()
loader = Loader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=2,
    drop_last=True
)

# Iterate (batches are automatically prefetched to device if available)
print("Starting training loop...")
for batch in loader:
    images = batch["image"] # Shape: (32, 28, 28, 1)
    labels = batch["label"] # Shape: (32,)
    # Your JAX training step here...
```

### 2. Flax & Optax Integration

JaxFlow simplifies the boilerplate often associated with JAX training loops.

```python
from jaxflow.integrations import create_train_state, get_optimizer, TrainStep
import flax.linen as nn

# Define a simple model
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x

# Initialize
model = CNN()
optimizer = get_optimizer("adamw", learning_rate=1e-3)
state = create_train_state(model, optimizer, input_shape=(1, 28, 28, 1))

# Define training step
train_step = TrainStep(state)

# Training loop
for batch in loader:
    state, metrics = train_step(state, batch)
    print(f"Loss: {metrics['loss']}")
```

### 3. HuggingFace Integration

Load models and tokenizers directly into JAX/Flax format.

```python
from jaxflow.integrations import load_model, load_tokenizer

# Load a BERT model and tokenizer
model = load_model("bert-base-uncased")
tokenizer = load_tokenizer("bert-base-uncased")

# Use in your pipeline...
```

### 4. Visualization

Inspect your data and training progress easily.

```python
from jaxflow.viz import show_batch, plot_loss

# Visualize a batch of images
batch = next(iter(loader))
show_batch(batch["image"], save_path="batch_sample.png")

# Plot training loss with smoothing
losses = [0.5, 0.4, 0.35, 0.3, 0.25] # ... collected during training
plot_loss(losses, smooth=0.9, save_path="training_curve.png")
```

## 🛠️ CLI Usage

JaxFlow comes with a handy CLI to check your environment and run benchmarks.

```bash
# Check system info and output as JSON
jaxflow info --json

# Run a matrix multiplication benchmark to test device performance
jaxflow benchmark --device gpu --size 4096 --iters 100 --save-path benchmark_results.json
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
