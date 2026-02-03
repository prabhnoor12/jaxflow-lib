import time
import functools
from contextlib import contextmanager
from typing import Callable, Any
from .logging import get_logger

logger = get_logger(__name__)

@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager to measure execution time.
    
    Usage:
        with timer("Processing"):
            do_something()
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(f"{name} took {elapsed:.4f} seconds")

def time_execution(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a function.
    
    Usage:
        @time_execution
        def my_func():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(f"Function '{func.__name__}' took {elapsed:.4f} seconds")
        return result
    return wrapper
