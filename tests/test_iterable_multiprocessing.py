
import multiprocessing as mp
import pytest
import math
from jaxflow import Loader, IterableDataset, get_worker_info
import sys

# Mock dataset for testing
class NumberIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __iter__(self):
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

@pytest.mark.skipif(sys.platform == "win32", reason="Multiprocessing on Windows can be flaky in CI/tests without proper guard")
def test_iterable_multiprocessing():
    # Dataset yields 0..9
    ds = NumberIterableDataset(0, 10)
    
    # Loader with 2 workers
    # On Windows, we need to be careful with spawn/fork. 
    # But Loader uses 'spawn' by default on Windows? 
    # Actually multiprocessing default is 'spawn' on Windows.
    
    loader = Loader(ds, batch_size=2, num_workers=2)
    
    batches = []
    seen_values = []
    for batch in loader:
        batches.append(batch)
        vals = batch['value']
        seen_values.extend(vals.tolist())
        
    # With 2 workers and 10 items total (5 per worker), and batch_size=2:
    # Worker 0: [0,1], [2,3], [4] -> 3 batches
    # Worker 1: [5,6], [7,8], [9] -> 3 batches
    # Total 6 batches.
    assert len(batches) == 6
    assert len(seen_values) == 10
    assert sorted(seen_values) == list(range(10))

def test_iterable_single_process():
    ds = NumberIterableDataset(0, 10)
    loader = Loader(ds, batch_size=2, num_workers=0)
    
    batches = []
    seen_values = []
    for batch in loader:
        batches.append(batch)
        vals = batch['value']
        seen_values.extend(vals.tolist())
        
    # Single process: 0..9 -> 5 batches
    assert len(batches) == 5
    assert len(seen_values) == 10
    assert sorted(seen_values) == list(range(10))
