import multiprocessing as mp
import numpy as np
import jax
import itertools
import time
import queue
import random
import logging
from typing import Iterator, Optional, List, Any, Union, Generator, Callable

from .dataset import Dataset, IterableDataset
from .pytree import tree_collate
from .sampler import Sampler, SequentialSampler, RandomSampler, BatchSampler

# Configure logger
logger = logging.getLogger(__name__)

class _MultiProcessingDataLoaderIter:
    """
    Iterator for DataLoader that handles multiprocessing.
    """
    def __init__(self, loader: 'Loader'):
        self.loader = loader
        self.dataset = loader.dataset
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.prefetch_factor = loader.prefetch_factor
        self.collate_fn = loader.collate_fn
        self.worker_init_fn = loader.worker_init_fn
        self.seed = loader.seed
        self.timeout = 30.0 
        
        self.worker_result_queue = mp.Queue(maxsize=self.prefetch_factor * self.num_workers)
        self.workers: List[mp.Process] = []
        self.num_batches = 0
        self.index_queue: Optional[mp.Queue] = None
        
        # Setup run
        self._reset()
        
    def _reset(self) -> None:
        # Determine if map-style or iterable-style
        self.is_map_style = hasattr(self.dataset, "__getitem__") and hasattr(self.dataset, "__len__")
        
        if self.is_map_style:
            self._start_map_style_workers()
        else:
            raise NotImplementedError("Multiprocessing for iterable-style datasets is not fully implemented yet.")

    def _start_map_style_workers(self) -> None:
        # 1. Get batches from batch_sampler
        batches = list(self.batch_sampler)
        self.num_batches = len(batches)
        
        # 2. Distribute work
        self.index_queue = mp.Queue()
        for b in batches:
            self.index_queue.put(b)
            
        # Add poison pills
        for _ in range(self.num_workers):
            self.index_queue.put(None)
            
        # 3. Start workers
        for i in range(self.num_workers):
            p = mp.Process(
                target=_worker_loop,
                args=(
                    self.dataset, 
                    self.index_queue, 
                    self.worker_result_queue, 
                    self.collate_fn, 
                    self.worker_init_fn,
                    i, # worker_id
                    self.seed
                )
            )
            p.daemon = True
            p.start()
            self.workers.append(p)
            
    def __iter__(self) -> '_MultiProcessingDataLoaderIter':
        return self
        
    def __next__(self) -> Any:
        if self.is_map_style:
            if self.num_batches == 0:
                 self._shutdown()
                 raise StopIteration
                 
            # Fetch from result queue
            try:
                # Get item with timeout
                item = self.worker_result_queue.get(timeout=self.timeout) 
            except queue.Empty:
                self._shutdown()
                logger.error("Timeout waiting for data. Workers may have died.")
                raise StopIteration("Timeout waiting for data. Workers may have died.")
                
            if isinstance(item, Exception):
                self._shutdown()
                raise item
                
            self.num_batches -= 1
            
            # Prefetch to device (JIT-Ready Streaming)
            return jax.device_put(item)
            
    def _shutdown(self) -> None:
        # Terminate workers
        for p in self.workers:
            if p.is_alive():
                p.terminate()
        self.workers = []

def _worker_loop(
    dataset: Any, 
    index_queue: mp.Queue, 
    result_queue: mp.Queue, 
    collate_fn: Callable,
    worker_init_fn: Optional[Callable],
    worker_id: int,
    seed: Optional[int]
) -> None:
    """
    Worker function to fetch data and put into result queue.
    """
    # Initialize worker
    if seed is not None:
        # Set seed for this worker to ensure reproducibility but difference between workers
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
    
    if worker_init_fn is not None:
        worker_init_fn(worker_id)

    try:
        while True:
            try:
                indices = index_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            if indices is None:
                break
                
            # Fetch data
            samples = [dataset[i] for i in indices]
            
            # Collate
            batch = collate_fn(samples)
            
            # Send
            result_queue.put(batch)
            
    except Exception as e:
        result_queue.put(e)

class _SingleProcessDataLoaderIter:
    """
    Iterator for DataLoader that runs in a single process.
    """
    def __init__(self, loader: 'Loader'):
        self.loader = loader
        self.dataset = loader.dataset
        self.batch_sampler = loader.batch_sampler
        self.collate_fn = loader.collate_fn
        
        self.is_map_style = hasattr(self.dataset, "__getitem__") and hasattr(self.dataset, "__len__")
        self.iterator = self._create_iterator()
        
    def _create_iterator(self) -> Generator[Any, None, None]:
        if self.is_map_style:
            # Use batch_sampler
            for batch_indices in self.batch_sampler:
                samples = [self.dataset[idx] for idx in batch_indices]
                yield self.collate_fn(samples)
        else:
            # Iterable style (fallback to simple logic, bypassing sampler for now as samplers are index-based)
            # A true IterableDataset handling would use an IteratorSampler but that's complex.
            # We keep the old logic for iterable datasets for now.
            iter_data = iter(self.dataset) # type: ignore
            
            if self.loader.shuffle:
                iter_data = self._shuffle_stream(iter_data)
            
            batch = []
            for item in iter_data:
                batch.append(item)
                if len(batch) == self.loader.batch_size:
                    yield self.collate_fn(batch)
                    batch = []
            if batch and not self.loader.drop_last:
                yield self.collate_fn(batch)

    def _shuffle_stream(self, iterator: Iterator[Any]) -> Generator[Any, None, None]:
        buffer_size = 1000 
        buffer = []
        try:
            for _ in range(buffer_size):
                buffer.append(next(iterator))
        except StopIteration:
            pass 
            
        rng = random.Random(self.loader.seed) if self.loader.seed else random
        
        if not buffer:
             return

        while True:
            try:
                new_item = next(iterator)
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer[idx]
                buffer[idx] = new_item
            except StopIteration:
                break
                
        rng.shuffle(buffer)
        for item in buffer:
            yield item

    def __iter__(self) -> '_SingleProcessDataLoaderIter':
        return self
        
    def __next__(self) -> Any:
        item = next(self.iterator)
        return jax.device_put(item)

class Loader:
    """
    Data Loader that handles batching, shuffling, and multiprocessing.
    """
    def __init__(
        self, 
        dataset: Union[Dataset, IterableDataset], 
        batch_size: int = 1, 
        num_workers: int = 0, 
        shuffle: bool = False, 
        drop_last: bool = False, 
        seed: Optional[int] = None, 
        prefetch_factor: int = 2,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.collate_fn = collate_fn if collate_fn is not None else tree_collate
        self.worker_init_fn = worker_init_fn
        
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if batch_size <= 0 and batch_sampler is None:
             raise ValueError("batch_size must be positive")

        # Setup Samplers
        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and drop_last')
            self.batch_sampler = batch_sampler
        else:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset, seed=seed) # type: ignore
                else:
                    sampler = SequentialSampler(dataset) # type: ignore
            self.sampler = sampler
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self) -> Union[_MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter]:
        if self.num_workers > 0:
            return _MultiProcessingDataLoaderIter(self)
        else:
            return _SingleProcessDataLoaderIter(self)
