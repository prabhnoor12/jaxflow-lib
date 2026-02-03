import random
import numpy as np
import os
import jax

def seed_everything(seed: int = 42):
    """
    Set seeds for random, numpy, and JAX to ensure reproducibility.
    
    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Note: JAX is functional and doesn't rely on global random state in the same way 
    # PyTorch does (via set_seed). Instead, it uses explicit PRNGKeys.
    # However, setting a predictable environment is still good practice.
    
    # We can print a confirmation or log it
    print(f"Global seed set to {seed}")

def get_jax_key(seed: int = 42):
    """
    Helper to get a JAX PRNGKey.
    
    Args:
        seed: The integer seed.
        
    Returns:
        jax.random.PRNGKey: The initialized PRNG key.
    """
    return jax.random.PRNGKey(seed)

def split_key(key, num=2):
    """
    Wrapper around jax.random.split for convenience.
    """
    return jax.random.split(key, num=num)
