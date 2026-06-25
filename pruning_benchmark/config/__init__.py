"""Dataclasses describing experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ..utils import make_run_id


@dataclass
class ExperimentConfig:
    """Container for experiment-level settings to aid reproducibility."""

    strategy: str
    amount: float
    train_steps: int = 600
    ft_steps: int = 200
    last_only: bool = True
    seed: int = 0
    device: str = "cpu"
    model_type: str = "ctrnn"
    movement_batches: int = 20
    task: str = "modcog:ctxdlydm1seql"
    no_prune: bool = False
    prune_phase: str = "post"
    run_id: str = field(default_factory=make_run_id)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        """Return a flat dict suitable for logging."""
        data = {
            "strategy": self.strategy,
            "amount": self.amount,
            "train_steps": self.train_steps,
            "ft_steps": self.ft_steps,
            "last_only": self.last_only,
            "seed": self.seed,
            "device": self.device,
            "model_type": self.model_type,
            "movement_batches": self.movement_batches,
            "task": self.task,
            "no_prune": self.no_prune,
            "prune_phase": self.prune_phase,
            "run_id": self.run_id,
        }
        data.update({f"extra_{k}": v for k, v in self.extra.items()})
        return data


__all__ = ["ExperimentConfig"]
