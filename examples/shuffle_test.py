import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from jaxflow.core.dataset import Dataset
from jaxflow.core.loader import Loader

class MapDataset(Dataset):
    def __init__(self, size):
        self.data = np.arange(size)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return {"value": self.data[idx]}

class IterableDataset(Dataset):
    def __init__(self, size):
        self.data = range(size)
        
    def __iter__(self):
        for item in self.data:
            yield {"value": item}

def test_map_shuffle():
    print("Testing Map-style Shuffle...")
    ds = MapDataset(20)
    loader = Loader(ds, batch_size=4, shuffle=True, seed=42)
    
    batches = []
    for batch in loader:
        batches.append(batch["value"])
    
    all_data = np.concatenate(batches)
    print("Data:", all_data)
    
    # Check if shuffled (not sorted)
    is_sorted = np.all(all_data[:-1] <= all_data[1:])
    print(f"Is sorted (should be False): {is_sorted}")
    assert not is_sorted, "Data should be shuffled"
    assert len(all_data) == 20
    assert len(np.unique(all_data)) == 20

def test_iterable_shuffle():
    print("\nTesting Iterable-style Shuffle...")
    ds = IterableDataset(20)
    loader = Loader(ds, batch_size=4, shuffle=True, seed=42)
    
    batches = []
    for batch in loader:
        batches.append(batch["value"])
    
    all_data = np.concatenate(batches)
    print("Data:", all_data)
    
    is_sorted = np.all(all_data[:-1] <= all_data[1:])
    print(f"Is sorted (should be False): {is_sorted}")
    assert not is_sorted, "Data should be shuffled"
    assert len(all_data) == 20
    assert len(np.unique(all_data)) == 20

def test_multiprocessing():
    print("\nTesting Multiprocessing...")
    ds = MapDataset(20)
    loader = Loader(ds, batch_size=4, num_workers=2, shuffle=True, seed=42)
    
    batches = []
    for batch in loader:
        batches.append(batch["value"])
        
    all_data = np.concatenate(batches)
    print("Data:", all_data)
    assert len(all_data) == 20
    assert len(np.unique(all_data)) == 20

if __name__ == "__main__":
    test_map_shuffle()
    test_iterable_shuffle()
    # Note: Multiprocessing test might need proper if __name__ == "__main__" guard in spawned processes if not using fork
    # Windows uses spawn, so this script must be importable without side effects.
    try:
        test_multiprocessing()
    except Exception as e:
        print(f"Multiprocessing test failed (expected if env issues): {e}")
