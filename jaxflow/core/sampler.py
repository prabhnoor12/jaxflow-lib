from typing import Iterator, Sequence, List, Sized, Optional
import numpy as np
from abc import ABC, abstractmethod
import math

class Sampler(ABC):
    """
    Base class for all Samplers.
    
    Every Sampler subclass has to provide an __iter__() method, providing a
    way to iterate over indices of dataset elements, and a __len__() method
    that returns the length of the returned iterators.
    """
    def __init__(self, data_source: Optional[Sized]):
        pass
        
    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError
        
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

class SequentialSampler(Sampler):
    """
    Samples elements sequentially, always in the same order.
    """
    def __init__(self, data_source: Sized):
        self.data_source = data_source
        
    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))
        
    def __len__(self) -> int:
        return len(self.data_source)

class RandomSampler(Sampler):
    """
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    """
    def __init__(self, data_source: Sized, replacement: bool = False, 
                 num_samples: Optional[int] = None, seed: Optional[int] = None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))
                             
        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")
                             
        if self._num_samples is not None:
             if not isinstance(self.num_samples, int) or self.num_samples <= 0:
                raise ValueError("num_samples should be a positive integer "
                                 "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            for _ in range(self.num_samples):
                yield self.rng.randint(high=n, low=0)
        else:
            indices = self.rng.permutation(n).tolist()
            for i in indices:
                yield i
                
    def __len__(self) -> int:
        return self.num_samples

class SubsetRandomSampler(Sampler):
    """
    Samples elements randomly from a given list of indices, without replacement.
    
    Args:
        indices: a sequence of indices
        seed: random seed
    """
    def __init__(self, indices: Sequence[int], seed: Optional[int] = None):
        self.indices = indices
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        
    def __iter__(self) -> Iterator[int]:
        # Copy and shuffle
        indices = list(self.indices)
        self.rng.shuffle(indices)
        return iter(indices)
        
    def __len__(self) -> int:
        return len(self.indices)

class WeightedRandomSampler(Sampler):
    """
    Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).
    
    Args:
        weights: a sequence of weights, not necessary summing up to one
        num_samples: number of samples to draw
        replacement: if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        seed: random seed
    """
    def __init__(self, weights: Sequence[float], num_samples: int, 
                 replacement: bool = True, seed: Optional[int] = None):
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
                             
        self.weights = np.array(weights, dtype=np.float64)
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        
    def __iter__(self) -> Iterator[int]:
        n = len(self.weights)
        if self.replacement:
            # Normalize weights
            p = self.weights / self.weights.sum()
            indices = self.rng.choice(n, size=self.num_samples, replace=True, p=p)
            for idx in indices:
                yield idx
        else:
            # Weighted sampling without replacement is tricky/slow for large N
            # For now, use simple choice without replacement
            # Note: This might be slow if len(weights) is huge
             p = self.weights / self.weights.sum()
             indices = self.rng.choice(n, size=self.num_samples, replace=False, p=p)
             for idx in indices:
                 yield idx
                 
    def __len__(self) -> int:
        return self.num_samples

class BatchSampler(Sampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    """
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
            
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
