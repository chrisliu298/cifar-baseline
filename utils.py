from contextlib import contextmanager
from time import perf_counter

import numpy as np


@contextmanager
def timer() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


@contextmanager
def temp_seed(seed):
    """Used a a context manager to temporarily set the seed of the random number generator."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
