"""Shared utility helpers for seeding and bookkeeping."""

from __future__ import annotations

import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: Optional[int]) -> None:
    """Seed `random`, NumPy, and PyTorch (CPU + CUDA if available)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_id(prefix: str = "run") -> str:
    """Return a timestamped identifier such as `run_20240406-153012`."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{stamp}"


__all__ = ["make_run_id", "set_global_seed"]
