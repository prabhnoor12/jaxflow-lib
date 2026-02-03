import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state
from typing import Any, Dict, Optional, Tuple, Callable, List, Union

class TrainState(train_state.TrainState):
    """
    Extended TrainState that handles batch statistics (for BatchNorm).
    """
    batch_stats: Any = None
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, batch_stats=None, **kwargs):
        return super().create(apply_fn=apply_fn, params=params, tx=tx, batch_stats=batch_stats, **kwargs)

def create_train_state(
    rng: jax.random.PRNGKey, 
    model: nn.Module, 
    input_shape: Tuple[int, ...], 
    optimizer: Any, # optax optimizer
    learning_rate: Optional[float] = None,
    mutable: List[str] = ['batch_stats']
) -> TrainState:
    """
    Creates a robust Flax TrainState.
    
    Args:
        rng: JAX PRNGKey.
        model: Flax Module.
        input_shape: Shape of input (excluding batch dim if needed, depends on model).
        optimizer: Optax optimizer.
        learning_rate: Optional learning rate to configure optimizer if it's a callable.
        mutable: List of mutable collections (e.g., ['batch_stats']).
        
    Returns:
        Initial TrainState.
    """
    variables = model.init(rng, jnp.ones(input_shape))
    params = variables['params']
    batch_stats = variables.get('batch_stats')
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats
    )

def save_checkpoint(
    ckpt_dir: str, 
    state: train_state.TrainState, 
    step: int, 
    keep: int = 3,
    prefix: str = "checkpoint_"
):
    """
    Saves a checkpoint using flax.training.checkpoints.
    """
    from flax.training import checkpoints
    checkpoints.save_checkpoint(
        ckpt_dir=ckpt_dir,
        target=state,
        step=step,
        keep=keep,
        prefix=prefix
    )

def restore_checkpoint(
    ckpt_dir: str, 
    target: Optional[train_state.TrainState] = None, 
    step: Optional[int] = None,
    prefix: str = "checkpoint_"
) -> train_state.TrainState:
    """
    Restores a checkpoint.
    """
    from flax.training import checkpoints
    return checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=target,
        step=step,
        prefix=prefix
    )

class TrainStep:
    """
    A robust helper class to wrap training steps with support for mutable state (BatchNorm).

    This class simplifies the JAX training loop by handling:
    1. Gradient computation (value_and_grad).
    2. State updates (applying gradients).
    3. Mutable state management (e.g., updating batch statistics for BatchNorm).

    Assumption:
        The `batch` argument passed to `__call__` must be a dictionary (or PyTree) 
        containing keys 'x' (inputs) and 'y' (targets).

    Example:
        ```python
        import jax
        import jax.numpy as jnp
        import optax
        from jaxflow.integrations.flax_utils import TrainStep, create_train_state

        # 1. Define Loss Function
        def cross_entropy_loss(logits, labels):
            one_hot = jax.nn.one_hot(labels, num_classes=10)
            return optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()

        # 2. Create TrainStep
        train_step = TrainStep(loss_fn=cross_entropy_loss)

        # 3. Inside Training Loop
        # state = create_train_state(...)
        # for batch in loader:
        #     # batch must be {'x': ..., 'y': ...}
        #     state, metrics = train_step(state, batch)
        #     print(f"Loss: {metrics['loss']}")
        ```
    """
    def __init__(self, loss_fn: Callable, has_aux: bool = False, mutables: List[str] = ['batch_stats']):
        self.loss_fn = loss_fn
        self.has_aux = has_aux
        self.mutables = mutables
        
    @jax.jit
    def __call__(self, state: TrainState, batch: Any) -> Tuple[TrainState, Dict[str, Any]]:
        """
        Performs a single training step.
        """
        def loss_func(params):
            variables = {'params': params}
            if state.batch_stats is not None:
                variables['batch_stats'] = state.batch_stats
                
            # If we have mutable state, we need to capture updates
            if state.batch_stats is not None:
                (logits, new_model_state) = state.apply_fn(
                    variables, 
                    batch['x'], 
                    mutable=self.mutables
                )
            else:
                logits = state.apply_fn(variables, batch['x'])
                new_model_state = None
                
            loss = self.loss_fn(logits, batch['y'])
            return loss, (logits, new_model_state)
            
        grad_fn = jax.value_and_grad(loss_func, has_aux=True)
        (loss, (logits, new_model_state)), grads = grad_fn(state.params)
        
        new_state = state.apply_gradients(grads=grads)
        
        # Update batch_stats if they exist
        if new_model_state is not None and 'batch_stats' in new_model_state:
            new_state = new_state.replace(batch_stats=new_model_state['batch_stats'])
            
        metrics = {'loss': loss}
        return new_state, metrics

class PredictStep:
    """
    Helper for prediction/evaluation.
    """
    @staticmethod
    @jax.jit
    def apply(state: TrainState, batch: Any, train: bool = False) -> Any:
        variables = {'params': state.params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats
            
        return state.apply_fn(
            variables, 
            batch['x'], 
            train=train, 
            mutable=False
        )
