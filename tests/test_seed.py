from __future__ import annotations

import random

import numpy as np

from manga_animation.core.seed import set_global_seed


def test_set_global_seed_makes_random_reproducible():
    set_global_seed(123)
    first = [random.random() for _ in range(5)]
    set_global_seed(123)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_global_seed_makes_numpy_reproducible():
    set_global_seed(7)
    first = np.random.rand(5)
    set_global_seed(7)
    second = np.random.rand(5)
    assert np.array_equal(first, second)


def test_different_seeds_produce_different_sequences():
    set_global_seed(1)
    a = [random.random() for _ in range(5)]
    set_global_seed(2)
    b = [random.random() for _ in range(5)]
    assert a != b
