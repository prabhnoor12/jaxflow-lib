import optax
from typing import Any, Callable, Dict, Optional, Union, List

def get_optimizer(
    name: str = "adam", 
    learning_rate: Union[float, Callable[[int], float]] = 1e-3, 
    weight_decay: float = 0.0,
    clip_norm: Optional[float] = None,
    gradient_accumulation_steps: int = 1,
    **kwargs
) -> optax.GradientTransformation:
    """
    Get an optax optimizer by name.
    
    Args:
        name: Name of the optimizer (adam, adamw, sgd, rmsprop, adabelief, etc.).
        learning_rate: Learning rate or schedule.
        weight_decay: Weight decay factor.
        clip_norm: Global norm clipping threshold.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        **kwargs: Additional arguments for the optimizer.
        
    Returns:
        Optax GradientTransformation.
    """
    name = name.lower()
    
    # Common optimizers
    if name == "adam":
        opt = optax.adam(learning_rate, **kwargs)
    elif name == "adamw":
        opt = optax.adamw(learning_rate, weight_decay=weight_decay, **kwargs)
    elif name == "sgd":
        opt = optax.sgd(learning_rate, **kwargs)
    elif name == "rmsprop":
        opt = optax.rmsprop(learning_rate, **kwargs)
    elif name == "adabelief":
        opt = optax.adabelief(learning_rate, **kwargs)
    elif name == "lamb":
        opt = optax.lamb(learning_rate, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
        
    # Chain transformations
    transforms = []
    
    # Gradient Accumulation
    if gradient_accumulation_steps > 1:
        transforms.append(optax.MultiSteps(opt, every_k_schedule=gradient_accumulation_steps))
    else:
        transforms.append(opt)
    
    # Clipping
    if clip_norm is not None:
        # Prepend clipping before optimizer update? 
        # Usually clipping is done on gradients before they are applied.
        # But optax.chain order matters.
        # We usually clip, then update.
        # However, MultiSteps wraps the optimizer.
        # If we use MultiSteps, the inner optimizer runs every k steps.
        # We probably want to clip the *accumulated* gradients?
        # Or clip every step?
        # Standard practice: Clip -> Update.
        # If we use MultiSteps(opt), opt is the update.
        # So we should chain: Clip -> MultiSteps(opt) or MultiSteps(Clip -> Opt)?
        # Actually MultiSteps accumulates gradients.
        # If we wrap the whole thing: Clip -> MultiSteps(Opt) -> this clips the *accumulated* gradient.
        # This seems correct for global norm clipping on the effective batch.
        transforms.insert(0, optax.clip_by_global_norm(clip_norm))

    # Weight decay (if not handled by optimizer)
    if weight_decay > 0.0 and name not in ["adamw", "lamb"]:
        # Add decay before update
        transforms.insert(0, optax.add_decayed_weights(weight_decay))
        
    # If we have multiple transforms (besides the main opt), chain them
    # But wait, opt is already in transforms.
    # Let's reconstruct the chain properly.
    
    chain_ops = []
    
    # 1. Weight Decay (if external)
    if weight_decay > 0.0 and name not in ["adamw", "lamb"]:
        chain_ops.append(optax.add_decayed_weights(weight_decay))
        
    # 2. Clipping
    if clip_norm is not None:
        chain_ops.append(optax.clip_by_global_norm(clip_norm))
        
    # 3. Optimizer (potentially wrapped in MultiSteps)
    if gradient_accumulation_steps > 1:
        # If we want to clip gradients *before* accumulation? No, usually accumulate then clip.
        # But MultiSteps wraps the *update* function.
        # It accumulates gradients, then calls the inner update.
        # So if we want to clip the accumulated gradient, we need to pass the clipper *inside* MultiSteps?
        # No, MultiSteps accumulates gradients, and then applies the inner transformation.
        # If we want to clip the accumulated gradient, the clipper should be part of the inner transformation.
        
        # Let's keep it simple:
        # If we use MultiSteps, it takes an inner transformation.
        # That inner transformation is applied to the accumulated gradients.
        # So we should chain Clip -> Optimizer, and wrap THAT in MultiSteps.
        
        inner_ops = []
        if clip_norm is not None:
            inner_ops.append(optax.clip_by_global_norm(clip_norm))
        inner_ops.append(opt)
        
        inner_opt = optax.chain(*inner_ops)
        final_opt = optax.MultiSteps(inner_opt, every_k_schedule=gradient_accumulation_steps)
        
        # But wait, if we added weight decay above, that's also a gradient transform.
        # Should that be accumulated? Yes.
        # So:
        all_inner = []
        if weight_decay > 0.0 and name not in ["adamw", "lamb"]:
            all_inner.append(optax.add_decayed_weights(weight_decay))
        if clip_norm is not None:
            all_inner.append(optax.clip_by_global_norm(clip_norm))
        all_inner.append(opt)
        
        final_opt = optax.MultiSteps(optax.chain(*all_inner), every_k_schedule=gradient_accumulation_steps)
        
    else:
        # Standard chain
        ops = []
        if weight_decay > 0.0 and name not in ["adamw", "lamb"]:
            ops.append(optax.add_decayed_weights(weight_decay))
        if clip_norm is not None:
            ops.append(optax.clip_by_global_norm(clip_norm))
        ops.append(opt)
        final_opt = optax.chain(*ops)
        
    return final_opt

def create_scheduler(
    name: str,
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0,
    warmup_steps: int = 0
) -> Callable[[int], float]:
    """
    Create a learning rate schedule.
    
    Args:
        name: Schedule name (cosine, linear, constant).
        init_value: Initial learning rate.
        decay_steps: Number of steps for decay.
        alpha: Minimum learning rate multiplier.
        warmup_steps: Number of warmup steps.
        
    Returns:
        Schedule function.
    """
    if name == "constant":
        schedule = optax.constant_schedule(init_value)
    elif name == "cosine":
        schedule = optax.cosine_decay_schedule(init_value, decay_steps, alpha=alpha)
    elif name == "linear":
        schedule = optax.linear_schedule(init_value, init_value * alpha, decay_steps)
    else:
        raise ValueError(f"Unknown schedule: {name}")
        
    if warmup_steps > 0:
        warmup = optax.linear_schedule(0.0, init_value, warmup_steps)
        schedule = optax.join_schedules([warmup, schedule], [warmup_steps])
        
    return schedule
