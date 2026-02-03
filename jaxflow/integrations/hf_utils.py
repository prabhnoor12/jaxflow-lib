from typing import Any, Optional, Tuple, Union, List, Dict
import os
import json

try:
    from transformers import FlaxAutoModel, AutoTokenizer, AutoConfig, PreTrainedTokenizerBase
except ImportError:
    FlaxAutoModel = None
    AutoTokenizer = None
    AutoConfig = None
    PreTrainedTokenizerBase = None

def _check_transformers():
    if FlaxAutoModel is None:
        raise ImportError("transformers library is required. Please install it via 'pip install transformers'.")

def load_model(
    model_name_or_path: str, 
    from_pt: bool = False, 
    dtype: Any = None,
    **kwargs
) -> Any:
    """
    Load a Flax model from Hugging Face Hub.
    
    Args:
        model_name_or_path: Model ID or path.
        from_pt: Load from PyTorch checkpoint if True.
        dtype: Data type (e.g., jnp.float32, jnp.bfloat16).
        **kwargs: Additional arguments for from_pretrained.
        
    Returns:
        Flax model.
    """
    _check_transformers()
    return FlaxAutoModel.from_pretrained(model_name_or_path, from_pt=from_pt, dtype=dtype, **kwargs)

def load_tokenizer(model_name_or_path: str, **kwargs) -> Any:
    """
    Load a tokenizer from Hugging Face Hub.
    
    Args:
        model_name_or_path: Model ID or path.
        **kwargs: Additional arguments for from_pretrained.
        
    Returns:
        Tokenizer.
    """
    _check_transformers()
    return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

def load_config(model_name_or_path: str, **kwargs) -> Any:
    """
    Load configuration from Hugging Face Hub.
    
    Args:
        model_name_or_path: Model ID or path.
        **kwargs: Additional arguments for from_pretrained.
        
    Returns:
        Configuration object.
    """
    _check_transformers()
    return AutoConfig.from_pretrained(model_name_or_path, **kwargs)

def save_model(
    model: Any, 
    tokenizer: Optional[Any], 
    save_directory: str,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    **kwargs
):
    """
    Save model and tokenizer, optionally pushing to Hub.
    
    Args:
        model: Flax model instance.
        tokenizer: Tokenizer instance.
        save_directory: Directory to save to.
        push_to_hub: Whether to push to Hugging Face Hub.
        repo_id: Repository ID for Hub (required if push_to_hub is True).
        **kwargs: Additional arguments for save_pretrained or push_to_hub.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    model.save_pretrained(save_directory, push_to_hub=push_to_hub, repo_id=repo_id, **kwargs)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory, push_to_hub=push_to_hub, repo_id=repo_id, **kwargs)

def create_model_card(
    save_directory: str,
    model_name: str,
    dataset_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs
):
    """
    Create a basic model card (README.md).
    """
    content = f"""---
language: en
tags:
{json.dumps(tags or [], indent=2)}
---

# {model_name}

This model was trained using JaxFlow.
"""
    if dataset_name:
        content += f"\nDataset: {dataset_name}\n"
        
    with open(os.path.join(save_directory, "README.md"), "w") as f:
        f.write(content)
