"""Utilities for resolving Mod_Cog task builders."""

from __future__ import annotations

import importlib
import math
import inspect
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

ModCogBuilder = Callable[..., object]

_PREFIXES = (
    "mod_cog-",
    "modcog-",
    "mod_cog:",
    "modcog:",
    "modcog/",
)
_SUFFIXES = (
    "-v0",
    "-v1",
    "-v2",
    "_v0",
    "_v1",
    "_v2",
)

_ALIASES = {
    "delaygo": "dlygo",
}

_REGISTERED: Dict[str, str] = {}
_GYM_AVAILABLE = False
try:
    import gymnasium as gym  # type: ignore

    _GYM_AVAILABLE = True
except Exception:  # pragma: no cover
    try:
        import gym  # type: ignore

        _GYM_AVAILABLE = True
    except Exception:  # pragma: no cover
        gym = None


def _normalize_name(name: str) -> str:
    token = (name or "").strip().lower()
    if not token:
        return ""
    for prefix in _PREFIXES:
        if token.startswith(prefix):
            token = token[len(prefix):]
            break
    for suffix in _SUFFIXES:
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    normalized = "".join(ch for ch in token if ch.isalnum())
    return normalized


@lru_cache(maxsize=1)
def _builder_registry() -> Dict[str, Tuple[str, ModCogBuilder]]:
    try:
        module = importlib.import_module("Mod_Cog.mod_cog_tasks")
    except ImportError as exc:  # pragma: no cover - handled by caller
        raise ImportError(
            "Mod_Cog tasks are unavailable. Install the Mod_Cog package or ensure it is on PYTHONPATH."
        ) from exc

    registry: Dict[str, Tuple[str, ModCogBuilder]] = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        key = _normalize_name(name)
        if not key:
            continue
        registry[key] = (name, obj)
    return registry


def _registry_contains(env_id: str) -> bool:
    registry = getattr(gym, "envs", None)
    if registry is None:
        return False
    reg_obj = getattr(registry, "registry", None)
    if reg_obj is None:
        return False
    try:
        return env_id in reg_obj
    except TypeError:
        keys = getattr(reg_obj, "keys", None)
        if callable(keys):
            try:
                return env_id in keys()
            except TypeError:
                return env_id in list(keys())
    except Exception:
        return False
    return False


def _register_env(raw: str, builder: ModCogBuilder) -> str:
    if not _GYM_AVAILABLE:
        raise RuntimeError(
            "Gym is required to register Mod_Cog environments. Ensure `gym` is installed in the environment."
        )
    name = raw.lower().replace("modcog:", "").replace("mod_cog-", "")
    base = f"Mod_Cog-{name.replace('-v0', '')}"
    versioned = f"{base}-v0"
    if versioned not in _REGISTERED and not _registry_contains(versioned):
        gym.register(id=versioned, entry_point=builder)
        _REGISTERED[versioned] = name
    return versioned


def resolve_modcog_callable(name: str) -> Optional[Tuple[str, ModCogBuilder]]:
    """Return (canonical_name, builder) for a Mod_Cog task name, if available."""
    key = _normalize_name(name)
    if not key:
        return None
    key = _ALIASES.get(key, key)
    registry = _builder_registry()
    return registry.get(key)


def ensure_modcog_env_id(name: str) -> Optional[str]:
    if not _GYM_AVAILABLE:
        raise RuntimeError(
            "Gym is required to instantiate Mod_Cog tasks. Install gym before requesting Mod_Cog environments."
        )
    info = resolve_modcog_callable(name)
    if info is None:
        return None
    canonical, builder = info
    return _register_env(name, builder)


def list_modcog_tasks() -> Tuple[str, ...]:
    """List canonical Mod_Cog builder names."""
    registry = _builder_registry()
    names = sorted({canonical for canonical, _ in registry.values()})
    return tuple(names)

def estimate_modcog_T(env, *, sample_calls: int = 10, safety_steps: int = 10) -> int:
    """
    Estimate the required sequence length (T) to cover a full Mod-Cog trial.

    Uses env.timing values (ms) divided by env.dt to get steps. Callables are
    sampled a few times and the max is used to avoid truncating the trial.
    """
    timing = getattr(env, "timing", None)
    dt = float(getattr(env, "dt", 100.0))
    if not isinstance(timing, dict):
        return int(400)
    total_ms = 0.0
    for value in timing.values():
        if callable(value):
            samples = [float(value()) for _ in range(max(1, sample_calls))]
            total_ms += max(samples)
        else:
            total_ms += float(value)
    if dt <= 0:
        return int(400)
    steps = int(math.ceil(total_ms / dt))
    return max(1, steps + int(safety_steps))


def ensure_modcog_available(task_names: Optional[Tuple[str, ...]] = None) -> None:
    """
    Validate that the Mod_Cog builders can be imported.

    Optionally accepts a tuple of requested Mod_Cog specifiers (e.g. "modcog:go")
    to provide clearer error messages when a builder cannot be located. This does
    not guarantee dataset-only variants exist, but it ensures the package itself
    is importable before suites are launched.
    """
    try:
        registry = _builder_registry()
    except ImportError as exc:
        raise ImportError(
            "Mod_Cog support is requested but the package is not installed. "
            "Install it via `pip install -e Mod_Cog/` before running suites."
        ) from exc
    if not task_names:
        return
    missing = []
    for raw in task_names:
        suffix = (raw or "").split("modcog:", 1)[-1]
        if not suffix:
            continue
        normalized = _normalize_name(suffix)
        normalized = _ALIASES.get(normalized, normalized)
        if normalized not in registry:
            missing.append(raw)
    if missing:
        raise ValueError(
            "The following Mod_Cog builders are unknown: "
            + ", ".join(sorted(set(missing)))
        )


__all__ = [
    "resolve_modcog_callable",
    "list_modcog_tasks",
    "ensure_modcog_env_id",
    "ensure_modcog_available",
    "estimate_modcog_T",
]
