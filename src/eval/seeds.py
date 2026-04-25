"""Global seed-setting utility. Call once at process entry.

Seeds the four nondeterminism sources we care about: Python's random,
NumPy, PyTorch CPU, and PyTorch CUDA (all devices). Also sets
`PYTHONHASHSEED` so dict ordering is deterministic in subprocesses
launched via multiprocessing.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seeds(seed: int, deterministic_cudnn: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (CPU + all CUDA devices).

    Parameters
    ----------
    seed: int
        The seed to use across all RNGs.
    deterministic_cudnn: bool
        If True, set `cudnn.deterministic=True` and `benchmark=False`.
        This hurts throughput a bit but makes SigLIP/LLaVA/Qwen forward
        passes bit-stable across runs with the same inputs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
