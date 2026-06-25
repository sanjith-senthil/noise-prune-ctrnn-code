"""Minimal vendored NeuroGym compatibility layer for Mod-Cog tasks."""

from .core import TrialEnv, TrialWrapper
from .utils import scheduler, spaces

__version__ = "2.2.0-modcog-subset"

__all__ = ["TrialEnv", "TrialWrapper", "scheduler", "spaces"]
