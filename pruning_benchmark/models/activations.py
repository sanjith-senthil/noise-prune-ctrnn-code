"""Shared activation helpers for CTRNN dynamics and analysis code."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


ActivationName = Literal["relu", "tanh", "shifted_tanh", "softplus"]


def activation_torch(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "relu":
        return F.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "shifted_tanh":
        return 0.5 * (torch.tanh(x) + 1.0)
    if name == "softplus":
        return F.softplus(x)
    raise ValueError(f"unknown activation {name}")


def activation_np(x: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return np.tanh(x)
    if name == "shifted_tanh":
        return 0.5 * (np.tanh(x) + 1.0)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "softplus":
        return np.logaddexp(x, 0.0)
    raise ValueError(f"unknown activation {name}")


def activation_derivative_np(x: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return 1.0 - np.tanh(x) ** 2
    if name == "shifted_tanh":
        return 0.5 * (1.0 - np.tanh(x) ** 2)
    if name == "relu":
        return (x > 0.0).astype(np.float64)
    if name == "softplus":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"unknown activation {name}")


def supports_gain_linearization(name: str) -> bool:
    return name in {"tanh", "shifted_tanh"}


__all__ = [
    "ActivationName",
    "activation_derivative_np",
    "activation_np",
    "activation_torch",
    "supports_gain_linearization",
]
