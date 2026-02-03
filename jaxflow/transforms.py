import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union

class Transform:
    """Base class for all transforms."""
    def __call__(self, x: Any) -> Any:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

class Compose(Transform):
    """
    Composes several transforms together.
    
    Args:
        transforms (List[Callable]): List of transforms to compose.
    
    Example:
        >>> transforms.Compose([
        >>>     transforms.Resize((256, 256)),
        >>>     transforms.ToArray(),
        >>> ])
    """
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ToArray(Transform):
    """Convert a numpy array or list to a JAX array."""
    def __init__(self, dtype: Optional[Any] = None):
        self.dtype = dtype
        
    def __call__(self, x: Union[np.ndarray, List, Any]) -> jax.Array:
        return jnp.array(x, dtype=self.dtype)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dtype={self.dtype})"

class Lambda(Transform):
    """Apply a user-defined lambda as a transform."""
    def __init__(self, lambd: Callable):
        self.lambd = lambd

    def __call__(self, x: Any) -> Any:
        return self.lambd(x)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Resize(Transform):
    """
    Resize the input array to the given size.
    
    Args:
        size (sequence): Desired output size. If size is a sequence like (h, w), output size will be matched to this.
                         If size is an int, smaller edge of the image will be matched to this number.
                         i.e, if height > width, then image will be rescaled to (size * height / width, size).
        method (str): Resize method. One of "nearest", "linear", "bilinear", "bicubic", "lanczos3", "lanczos5".
    """
    def __init__(self, size: Union[int, Tuple[int, ...]], method: str = 'bilinear'):
        self.size = size
        self.method = method

    def __call__(self, x: jax.Array) -> jax.Array:
        # Assumes channel-last format (H, W, C) for images
        if isinstance(self.size, int):
            h, w = x.shape[:2]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return x
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
            new_shape = (oh, ow) + x.shape[2:]
        else:
            new_shape = tuple(self.size) + x.shape[2:]
            
        return jax.image.resize(x, new_shape, method=self.method)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, method='{self.method}')"

class CenterCrop(Transform):
    """Crops the given image at the center."""
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, x: jax.Array) -> jax.Array:
        h, w = x.shape[:2]
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return x[i:i+th, j:j+tw, ...]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class RandomCrop(Transform):
    """
    Crop the given image at a random location.
    
    Args:
        size (sequence or int): Desired output size of the crop.
        pad_if_needed (boolean): It will pad the image if smaller than the desired size to avoid raising an exception.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0.
    """
    def __init__(self, size: Union[int, Tuple[int, int]], pad_if_needed: bool = False, fill: int = 0):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.fill = fill

    def __call__(self, x: jax.Array) -> jax.Array:
        h, w = x.shape[:2]
        th, tw = self.size
        
        if w == tw and h == th:
            return x
            
        if w < tw or h < th:
             if self.pad_if_needed:
                 ph = max(0, th - h)
                 pw = max(0, tw - w)
                 # Pad (top, bottom), (left, right), (channels...)
                 x = jnp.pad(x, ((0, ph), (0, pw), (0, 0)), constant_values=self.fill)
                 h, w = x.shape[:2]
             else:
                 raise ValueError(f"Image size {(h,w)} is smaller than crop size {self.size}")

        # Use numpy random for worker safety
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        
        return x[i:i+th, j:j+tw, ...]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

class RandomHorizontalFlip(Transform):
    """Horizontally flip the given image randomly with a given probability."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: jax.Array) -> jax.Array:
        # Use numpy random to be compatible with DataLoader workers seeding
        if np.random.rand() < self.p:
            return jnp.fliplr(x)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

class Normalize(Transform):
    """
    Normalize a tensor image with mean and standard deviation.
    
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean: Union[float, Tuple[float, ...]], std: Union[float, Tuple[float, ...]]):
        self.mean = jnp.array(mean)
        self.std = jnp.array(std)

    def __call__(self, x: jax.Array) -> jax.Array:
        return (x - self.mean) / self.std

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
