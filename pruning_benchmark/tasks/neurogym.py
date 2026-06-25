"""Helpers to sample Mod-Cog/NeuroGym trials into RNN-ready tensors."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
import copy

try:
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover - gym fallback
    try:
        import gym  # type: ignore
    except Exception:  # pragma: no cover
        gym = None


def _seed_env(env, seed: int) -> None:
    try:
        if hasattr(env, "seed"):
            env.seed(seed)
    except Exception:
        pass
    try:
        env.reset(seed=seed)
    except TypeError:
        try:
            if hasattr(env, "seed"):
                env.seed(seed)
            else:
                env.reset()
        except Exception:
            pass
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        try:
            env.action_space.seed(seed)
        except Exception:
            pass


def _labels_from_targets(targets: np.ndarray) -> np.ndarray:
    arr = np.asarray(targets)
    if arr.ndim >= 2 and arr.shape[-1] > 1:
        nan_mask = np.isnan(arr).any(axis=-1) if np.issubdtype(arr.dtype, np.floating) else None
        safe = np.where(np.isnan(arr), -np.inf, arr) if nan_mask is not None else arr
        labels = np.argmax(safe, axis=-1)
        if nan_mask is not None:
            labels = labels.astype(np.float32, copy=False)
            labels[nan_mask] = -1.0
        return labels.astype(np.int64, copy=False)
    if arr.ndim >= 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=False)
        arr[np.isnan(arr)] = -1.0
    return arr.astype(np.int64, copy=False)


class ModCogTrialDM:
    """Trial-aligned Mod-Cog dataset to avoid slicing mid-trial."""

    def __init__(
        self,
        env: str | Any,
        T: int,
        B: int,
        *,
        device: str = "cpu",
        last_only: bool = False,
        seed: int = 0,
        env_kwargs: dict | None = None,
        mask_fixation: bool = True,
    ):
        if last_only:
            raise ValueError("Mod-Cog trial datasets do not support last_only=True.")
        if gym is None and isinstance(env, str):
            raise RuntimeError("Gym is required to instantiate Mod-Cog environments.")
        self.T = int(T)
        self.B = int(B)
        self.device = device
        self.last_only = bool(last_only)
        self.mask_fixation = bool(mask_fixation)
        self._trial_kwargs = dict(env_kwargs or {})

        if isinstance(env, str):
            self.envs = [gym.make(env, **(env_kwargs or {})) for _ in range(self.B)]
        else:
            self.envs = [copy.deepcopy(env) for _ in range(self.B)]

        for idx, env_i in enumerate(self.envs):
            _seed_env(env_i, seed + idx)

        obs_shape = getattr(self.envs[0].observation_space, "shape", None)
        if not obs_shape:
            raise ValueError(f"Environment {env} has no observation shape.")
        self.input_dim = int(np.prod(obs_shape))
        action_space = self.envs[0].action_space
        if hasattr(action_space, "n"):
            self.n_classes = int(action_space.n)
        else:
            shape = getattr(action_space, "shape", None)
            if not shape:
                raise ValueError(f"Environment {env} has no action shape.")
            self.n_classes = int(np.prod(shape))

    def _sample_single(self, env) -> Tuple[np.ndarray, np.ndarray]:
        try:
            new_trial_fn = env.get_wrapper_attr("new_trial")
        except Exception:
            new_trial_fn = getattr(getattr(env, "unwrapped", env), "new_trial")
        try:
            new_trial_fn(**self._trial_kwargs)
        except TypeError:
            new_trial_fn()
        try:
            ob = env.get_wrapper_attr("ob")
        except Exception:
            ob = getattr(getattr(env, "unwrapped", env), "ob")
        ob = np.asarray(ob)
        try:
            gt = env.get_wrapper_attr("gt")
        except Exception:
            gt = getattr(getattr(env, "unwrapped", env), "gt")
        gt = np.asarray(gt)
        ob = ob.reshape(ob.shape[0], -1)
        labels = _labels_from_targets(gt)
        if labels.ndim > 1:
            labels = labels.reshape(labels.shape[0], -1)
            labels = labels[:, 0]
        if self.mask_fixation:
            labels = labels.astype(np.int64, copy=False)
            labels[labels == 0] = -1
        return ob, labels

    def sample_batch(self):
        T, B, I = self.T, self.B, self.input_dim
        X = torch.zeros(T, B, I, device=self.device)
        Y = torch.full((T, B), -1, dtype=torch.long, device=self.device)
        for b, env in enumerate(self.envs):
            ob, labels = self._sample_single(env)
            t_len = min(T, ob.shape[0])
            if t_len > 0:
                X[:t_len, b] = torch.from_numpy(ob[:t_len]).float().to(self.device)
                Y[:t_len, b] = torch.from_numpy(labels[:t_len]).to(torch.long).to(self.device)
        return X, Y
