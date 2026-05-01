"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA).

    Args:
        seed: integer seed.
        deterministic: if True, force cuDNN deterministic algorithms (slower
            but bitwise reproducible). Default False matches the paper, which
            does not require strict determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
