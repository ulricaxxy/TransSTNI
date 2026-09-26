import os, numpy as np, torch
from typing import Optional
import pickle


def build_sizes(num=8, min_n=153, max_n=307, *, rng=None):
    """Sample Stage2 subgraph node counts.

    Default path consumes the **global** NumPy RNG so ``np.random.seed`` /
    ``seed_everything`` control the schedule. Previously this used
    ``np.random.default_rng()`` with no seed, which ignored ``seed_everything``
    and made Stage2 irreproducible across runs.

    ``high=max_n`` is exclusive (same as the old ``Generator.integers`` API).
    Callers that want an inclusive CLI range must pass ``max_n = hi + 1``
    (see ``sage_nosubgraph_patch._force_random_subgraph_args``).
    """
    # assert min_n >= 1 and max_n >= min_n + 1
    k = int(num)
    if rng is None:
        sizes = np.random.randint(low=min_n, high=max_n, size=k).tolist()
        np.random.shuffle(sizes)
        return sizes

    sizes = rng.integers(low=min_n, high=max_n, size=k).tolist()
    rng.shuffle(sizes)
    return sizes  # len = num
