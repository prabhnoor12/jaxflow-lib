import pytest
from jaxflow.core.loader import Loader
from jaxflow.core.dataset import Dataset

def test_loader_init():
    ds = Dataset()
    loader = Loader(ds)
    assert loader.dataset == ds
