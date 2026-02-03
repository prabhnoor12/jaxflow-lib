import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, List, Optional, TypeVar, Tuple, Union

T = TypeVar("T")

def tree_collate(batch: List[Any]) -> Any:
    """
    Stacks a list of pytrees into a single pytree with an extra batch dimension.
    Assumes all elements in the batch have the same structure.
    
    Args:
        batch: A list of pytrees (e.g., dicts, lists, tuples, or custom classes registered with JAX).
    
    Returns:
        A single pytree where each leaf is a stacked numpy array of the corresponding leaves in the batch.
        Returns None if the batch is empty.
    """
    if not batch:
        return None
        
    return jax.tree_util.tree_map(lambda *leaves: np.stack(leaves), *batch)

def tree_stack(trees: List[Any], axis: int = 0) -> Any:
    """
    Stack a list of pytrees along a new axis.
    
    Args:
        trees: List of pytrees.
        axis: Axis to stack along.
        
    Returns:
        Stacked pytree.
    """
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=axis), *trees)

def tree_concat(trees: List[Any], axis: int = 0) -> Any:
    """
    Concatenate a list of pytrees along an existing axis.
    
    Args:
        trees: List of pytrees.
        axis: Axis to concatenate along.
        
    Returns:
        Concatenated pytree.
    """
    return jax.tree_util.tree_map(lambda *leaves: jnp.concatenate(leaves, axis=axis), *trees)

def tree_slice(tree: Any, start: int, end: int, axis: int = 0) -> Any:
    """
    Slice a pytree along an axis.
    
    Args:
        tree: The pytree.
        start: Start index.
        end: End index.
        axis: Axis to slice.
        
    Returns:
        Sliced pytree.
    """
    def _slice(x):
        return jax.lax.dynamic_slice_in_dim(x, start, end - start, axis=axis)
        
    return jax.tree_util.tree_map(_slice, tree)

def tree_flatten(tree: Any) -> Tuple[List[Any], Any]:
    """
    Flatten a pytree.
    
    Args:
        tree: The pytree to flatten.
        
    Returns:
        A tuple containing a list of leaves and the treedef.
    """
    return jax.tree_util.tree_flatten(tree)

def tree_unflatten(treedef: Any, leaves: List[Any]) -> Any:
    """
    Unflatten a pytree.
    
    Args:
        treedef: The tree definition.
        leaves: The list of leaves.
        
    Returns:
        The reconstructed pytree.
    """
    return jax.tree_util.tree_unflatten(treedef, leaves)

def tree_shape(tree: Any) -> Any:
    """
    Get the shape of each leaf in the pytree.
    """
    return jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else (), tree)
