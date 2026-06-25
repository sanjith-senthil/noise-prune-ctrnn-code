"""Base classes and registry for pruning strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from pruning_benchmark.models import CTRNN


Batch = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class PruneContext:
    """Container passed to pruners with all runtime information."""

    model: CTRNN
    amount: float
    criterion: nn.Module
    last_only: bool
    device: str
    batches: Optional[List[Batch]] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    prune_feedforward: bool = False


class BasePruner:
    """Abstract base class for pruning strategies."""

    name: str = "base"
    description: str = ""
    requires_batches: bool = False
    default_batch_count: int = 0
    aliases: Sequence[str] = ()
    supports_pretrain: bool = False

    def resolved_batch_count(self, requested: Optional[int]) -> int:
        if not self.requires_batches:
            return 0
        if requested is not None and requested > 0:
            return requested
        return max(1, self.default_batch_count or 0)

    def prepare(self, context: PruneContext) -> Mapping[str, object]:
        """Optional hook for pre-computing scores/statistics before pruning."""
        return {}

    def apply(
        self,
        context: PruneContext,
        state: Mapping[str, object],
        **kwargs,
    ) -> Mapping[str, float]:
        """Perform pruning (override in subclasses)."""
        raise NotImplementedError

    def run(self, context: PruneContext, **kwargs) -> Mapping[str, float]:
        state = self.prepare(context)
        stats = self.apply(context, state, **kwargs)
        return dict(stats) if stats else {}

    def pretrain(self, context: PruneContext, **kwargs) -> Mapping[str, float]:
        if not self.supports_pretrain:
            raise ValueError(f"Pruner '{self.name}' does not support pre-training pruning.")
        return self.run(context, **kwargs)


_PRUNERS: Dict[str, BasePruner] = {}
_ALIASES: Dict[str, str] = {}


def register_pruner(pruner: BasePruner) -> None:
    key = pruner.name.lower()
    if key in _PRUNERS:
        raise ValueError(f"Pruner '{pruner.name}' already registered.")
    _PRUNERS[key] = pruner
    for alias in pruner.aliases:
        alias_key = alias.lower()
        if alias_key in _ALIASES:
            raise ValueError(f"Alias '{alias}' already registered.")
        _ALIASES[alias_key] = key


def get_pruner(name: str) -> BasePruner:
    key = name.lower()
    key = _ALIASES.get(key, key)
    if key not in _PRUNERS:
        raise KeyError(f"Unknown pruning strategy '{name}'.")
    return _PRUNERS[key]


def available_pruning_strategies() -> Dict[str, BasePruner]:
    return dict(_PRUNERS)


def apply_registered_pruner(
    name: str,
    context: PruneContext,
    **kwargs,
) -> Mapping[str, float]:
    pruner = get_pruner(name)
    return pruner.run(context, **kwargs)


__all__ = [
    "Batch",
    "BasePruner",
    "PruneContext",
    "available_pruning_strategies",
    "apply_registered_pruner",
    "get_pruner",
    "register_pruner",
]
