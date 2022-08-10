from contextlib import contextmanager

import numpy as np


@contextmanager
def temp_seed(seed):
    """Used a a context manager to temporarily set the seed of the random number generator."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
