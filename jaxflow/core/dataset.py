from typing import Any, Iterator, Optional, TypeVar, Generic, Sequence, List
from abc import ABC, abstractmethod

T_co = TypeVar('T_co', covariant=True)

class Dataset(Generic[T_co], ABC):
    """
    Abstract Base Class for all map-style datasets.
    
    Subclasses must override __getitem__ and __len__.
    """
    
    @abstractmethod
    def __getitem__(self, index: int) -> T_co:
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

class IterableDataset(Generic[T_co], ABC):
    """
    Abstract Base Class for all iterable-style datasets.
    
    Subclasses must override __iter__.
    """
    
    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

class ArrayDataset(Dataset[T_co]):
    """
    Dataset wrapping a sequence (list, numpy array, etc.).
    """
    def __init__(self, data: Sequence[T_co]) -> None:
        self.data = data
        
    def __getitem__(self, index: int) -> T_co:
        return self.data[index]
        
    def __len__(self) -> int:
        return len(self.data)

class Subset(Dataset[T_co]):
    """
    Subset of a dataset at specified indices.
    
    Args:
        dataset: The whole Dataset
        indices: Indices in the whole set selected for subset
    """
    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        
    def __getitem__(self, idx: int) -> T_co:
        return self.dataset[self.indices[idx]]
        
    def __len__(self) -> int:
        return len(self.indices)

class ConcatDataset(Dataset[T_co]):
    """
    Dataset as a concatenation of multiple datasets.
    
    This class is useful to assemble different existing datasets.
    """
    def __init__(self, datasets: Sequence[Dataset[T_co]]) -> None:
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"
        self.cumulative_sizes = self.cumsum(self.datasets)
        
    @staticmethod
    def cumsum(sequence: Sequence[Dataset]) -> List[int]:
        r, s = [], 0
        for e in sequence:
            l = len(e)
            s += l
            r.append(s)
        return r
        
    def __len__(self) -> int:
        return self.cumulative_sizes[-1]
        
    def __getitem__(self, idx: int) -> T_co:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        
        # Binary search or simple loop to find the right dataset
        # Since number of datasets is usually small, simple loop is fine
        dataset_idx = 0
        sample_idx = idx
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                if i > 0:
                    sample_idx = idx - self.cumulative_sizes[i - 1]
                dataset_idx = i
                break
        else:
             raise IndexError("list index out of range")
             
        return self.datasets[dataset_idx][sample_idx]
