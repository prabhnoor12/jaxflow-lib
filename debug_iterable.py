
import multiprocessing as mp
import time
from typing import Iterator
from jaxflow import Loader, IterableDataset, get_worker_info
import jax.numpy as jnp
import numpy as np
import sys
import math

# Ensure this is protected for Windows
if __name__ == "__main__":
    mp.freeze_support()

class NumberIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __iter__(self) -> Iterator[int]:
        worker_info = get_worker_info()
        if worker_info is None:
            # Single process
            iter_start = self.start
            iter_end = self.end
        else:
            # Shard data
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
            
        for i in range(iter_start, iter_end):
             yield {"value": i}

def test_iterable_loader_multiprocessing():
    print("Starting test...")
    # Dataset yields 0..9
    ds = NumberIterableDataset(0, 10)
    
    # Loader with 2 workers
    print("Creating Loader...")
    loader = Loader(ds, batch_size=2, num_workers=2)
    
    batches = []
    print("Iterating...")
    seen_values = []
    for i, batch in enumerate(loader):
        print(f"Batch {i}")
        batches.append(batch)
        # Check values
        # batch is a dict of arrays
        vals = batch['value']
        seen_values.extend(vals.tolist())
        
    print(f"Total batches: {len(batches)}")
    print(f"Seen values: {sorted(seen_values)}")
    
    # Verify
    # With 2 workers and 10 items total (5 per worker), and batch_size=2:
    # Worker 0: [0,1], [2,3], [4] -> 3 batches
    # Worker 1: [5,6], [7,8], [9] -> 3 batches
    # Total 6 batches.
    assert len(batches) == 6, f"Expected 6 batches, got {len(batches)}"
    assert len(seen_values) == 10, f"Expected 10 items, got {len(seen_values)}"
    assert sorted(seen_values) == list(range(10))
    print("Test Passed!")

if __name__ == "__main__":
    try:
        test_iterable_loader_multiprocessing()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
