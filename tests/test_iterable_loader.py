
import pytest
import multiprocessing as mp
import time
from typing import Iterator
from jaxflow import Loader, IterableDataset
import jax.numpy as jnp
import numpy as np

class NumberIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __iter__(self) -> Iterator[int]:
        worker_info = mp.current_process()
        # Simple yield
        for i in range(self.start, self.end):
            yield {"value": i, "worker": worker_info.name}

def test_iterable_loader_multiprocessing():
    # Dataset yields 0..9
    ds = NumberIterableDataset(0, 10)
    
    # Loader with 2 workers
    # Each worker will yield 0..9, so we expect 20 items total (10 from each)
    # Batch size 2
    loader = Loader(ds, batch_size=2, num_workers=2)
    
    batches = []
    for batch in loader:
        batches.append(batch)
        
    # Verify
    # Total items = 20
    # Batch size = 2
    # Total batches = 10
    assert len(batches) == 10
    
    # Check structure
    b0 = batches[0]
    assert "value" in b0
    assert b0["value"].shape == (2,)
    
    # Collect all values
    all_values = []
    for b in batches:
        all_values.extend(np.array(b["value"]).tolist())
        
    assert len(all_values) == 20
    # We should have two 0s, two 1s, ..., two 9s
    all_values.sort()
    expected = sorted(list(range(10)) * 2)
    assert all_values == expected

def test_iterable_loader_single_process():
    ds = NumberIterableDataset(0, 10)
    loader = Loader(ds, batch_size=2, num_workers=0)
    
    batches = list(loader)
    # Single process, iterated once -> 10 items -> 5 batches
    assert len(batches) == 5
    
    all_values = []
    for b in batches:
        all_values.extend(np.array(b["value"]).tolist())
        
    assert sorted(all_values) == list(range(10))
