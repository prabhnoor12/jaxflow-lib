import jax.numpy as jnp
import numpy as np
from jaxflow import transforms

def test_transforms():
    print("Testing transforms...")
    
    # 1. RandomHorizontalFlip
    img = jnp.array([[1, 2, 3], [4, 5, 6]]) # 2x3 image
    # We can't deterministic test random easily without mocking, but we can check it runs
    flip = transforms.RandomHorizontalFlip(p=1.0)
    out = flip(img)
    print("Flip output shape:", out.shape)
    
    # 2. RandomCrop
    img2 = jnp.zeros((10, 10, 3))
    crop = transforms.RandomCrop(size=(5, 5))
    out2 = crop(img2)
    print("Crop output shape:", out2.shape)
    assert out2.shape == (5, 5, 3)
    
    # 3. Normalize
    img3 = jnp.array([[[1.0, 2.0, 3.0]]]) # 1x1x3
    norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    out3 = norm(img3)
    print("Normalize output:", out3)
    expected = (img3 - 0.5) / 0.5
    assert jnp.allclose(out3, expected)
    
    print("All transform tests passed!")

if __name__ == "__main__":
    test_transforms()
