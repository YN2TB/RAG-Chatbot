"""Reproducibility.

An ablation table is only evidence if the runs differ by the thing you changed
and not by the RNG. Seed every run; report the seed in the results table.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    `deterministic=True` also pins cuDNN to reproducible kernels. It costs roughly
    10-20% throughput, so use it for the runs you report and leave it off while
    iterating.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic reductions in some CUDA kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, run-reproducible seed."""
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
