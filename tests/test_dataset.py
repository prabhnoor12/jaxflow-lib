import pytest
from jaxflow.core.dataset import Dataset

def test_dataset_init():
    ds = Dataset()
    assert len(ds) == 0
    with pytest.raises(NotImplementedError):
        _ = ds[0]
