"""Deterministic seeding, so pipeline runs are reproducible given the same config.seed."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed `random` and `numpy`, plus `torch` if it happens to be installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
