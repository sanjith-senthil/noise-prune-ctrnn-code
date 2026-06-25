"""Mask-only vanilla noise-prune for task/evaluation suites.

This module implements the practical ranking-based variant used in the
CTRNN dynamics study: compute Lyapunov-derived edge scores on the vanilla
continuous-time operator ``A = W_rec - I`` and apply the resulting exact
top-k recurrent mask directly to the original recurrent weights.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.utils.prune as prune
from scipy.linalg import LinAlgError, solve_continuous_lyapunov

from .pruners import BasePruner, PruneContext


def _consolidate_if_pruned(layer: torch.nn.Module) -> None:
    try:
        prune.remove(layer, "weight")
    except (ValueError, AttributeError):
        pass


@torch.no_grad()
def _enforce_hidden_constraints(model: torch.nn.Module) -> None:
    layer = getattr(model, "hidden_layer", None)
    if layer is None:
        return
    weight = getattr(layer, "weight_orig", layer.weight)
    if getattr(model, "use_dale", False):
        weight.mul_(model.dale_sign).abs_().mul_(model.dale_sign)
    if getattr(model, "no_self_connections", False):
        weight.fill_diagonal_(0.0)
        mask = getattr(layer, "weight_mask", None)
        if mask is not None:
            mask.fill_diagonal_(0.0)


def _exact_offdiag_keep_mask_from_scores(
    scores: np.ndarray,
    *,
    amount: float,
    no_self_connections: bool,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Expected square recurrent score matrix.")
    H = scores.shape[0]
    off_mask = ~np.eye(H, dtype=bool)
    if not no_self_connections:
        off_mask = np.ones_like(scores, dtype=bool)
    flat_scores = scores[off_mask]
    total = flat_scores.size
    k_prune = int(round(float(amount) * total))
    k_keep = max(0, total - k_prune)
    keep_flat = np.zeros(total, dtype=bool)
    if k_keep > 0:
        order = np.argsort(flat_scores)[::-1]
        keep_flat[order[:k_keep]] = True
    keep_mask = np.zeros_like(scores, dtype=bool)
    keep_mask[off_mask] = keep_flat
    if no_self_connections:
        np.fill_diagonal(keep_mask, False)
    return keep_mask


def _apply_keep_mask_to_model(
    model: torch.nn.Module,
    keep_mask: np.ndarray,
) -> Dict[str, float]:
    layer = model.hidden_layer
    weight = layer.weight.detach().clone()
    mask = torch.as_tensor(keep_mask, dtype=weight.dtype, device=weight.device)
    _consolidate_if_pruned(layer)
    prune.custom_from_mask(layer, name="weight", mask=mask)
    _enforce_hidden_constraints(model)
    total = float(mask.numel() - mask.shape[0])
    kept = float(mask.sum().item())
    density = kept / total if total > 0 else 0.0
    return {
        "masked_kept_edges": int(kept),
        "density_achieved": float(density),
    }


def _noise_prune_edge_scores(
    operator: np.ndarray,
    *,
    sigma: float,
    eps: float,
    target_density: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if sigma <= 0.0 or eps <= 0.0:
        raise ValueError("sigma and eps must be positive.")
    n = operator.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    rows, cols = np.where(off_mask)
    try:
        cov = solve_continuous_lyapunov(operator.astype(np.float64, copy=False), -(sigma**2) * np.eye(n))
    except LinAlgError as exc:
        raise ValueError("Lyapunov solver failed; ensure operator is Hurwitz.") from exc
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eigvals))
    max_abs = float(np.max(np.abs(eigvals)))
    if min_eig < -1e-5 * max(1.0, max_abs):
        raise ValueError("Covariance from Lyapunov solve is not positive semidefinite.")

    weights = operator[rows, cols]
    abs_weights = np.abs(weights)
    sign_weights = np.sign(weights)
    diag_cov = np.diag(cov)
    diff_cov = diag_cov[rows] + diag_cov[cols] - sign_weights * 2.0 * cov[rows, cols]
    diff_cov = np.maximum(diff_cov, 0.0)

    Kdeg = 4.0 * np.log(n) / (eps**2)
    K = 8.0 * np.log(n) / (eps**2 * sigma**2)
    score_vec = K * abs_weights * diff_cov
    total_edges = float(n * (n - 1))
    scale_factor = 1.0
    if score_vec.size > 0:
        sum_scores = float(np.sum(score_vec))
        desired_kept = float(target_density) * total_edges
        if sum_scores > 0.0 and desired_kept >= 0.0:
            scale_factor = desired_kept / sum_scores
            score_vec = score_vec * scale_factor
    capped_probs = int(np.count_nonzero(score_vec > 1.0))
    score_vec = np.clip(score_vec, 0.0, 1.0)

    score_mat = np.zeros_like(operator, dtype=np.float64)
    score_mat[rows, cols] = score_vec
    return score_mat, {
        "K": float(K),
        "Kdeg": float(Kdeg),
        "scale_factor": float(scale_factor),
        "capped_probs": int(capped_probs),
        "score_sum": float(np.sum(score_vec)),
        "score_max": float(np.max(score_vec)) if score_vec.size else 0.0,
        "score_mean": float(np.mean(score_vec)) if score_vec.size else 0.0,
    }


def _noise_prune_scores_with_adaptive_shift(
    operator: np.ndarray,
    *,
    amount: float,
    sigma: float,
    eps: float,
    shift_buffer: float = 0.0,
    max_attempts: int = 6,
) -> Tuple[np.ndarray, Dict[str, float]]:
    current_shift = float(max(0.0, shift_buffer))
    eye = np.eye(operator.shape[0], dtype=np.float64)
    target_density = 1.0 - float(amount)
    for attempt in range(max_attempts):
        shifted = operator - current_shift * eye
        try:
            score_mat, stats = _noise_prune_edge_scores(
                shifted,
                sigma=sigma,
                eps=eps,
                target_density=target_density,
            )
            stats = dict(stats)
            stats["adaptive_shift"] = float(current_shift)
            stats["adaptive_attempt"] = int(attempt)
            return score_mat, stats
        except ValueError:
            if attempt >= max_attempts - 1:
                raise
            current_shift = 0.5 if current_shift == 0.0 else current_shift * 2.0
    raise RuntimeError("adaptive shift loop exhausted unexpectedly")


def vanilla_mask_prune_recurrent(
    model: torch.nn.Module,
    amount: float,
    *,
    sigma: float = 1.0,
    eps: float = 0.3,
    leak_shift: float = 0.0,
    include_feedforward: bool = False,
) -> Dict[str, float]:
    if include_feedforward:
        raise NotImplementedError("vanilla_mask_only currently supports recurrent pruning only.")
    layer = getattr(model, "hidden_layer", None)
    if layer is None or layer.weight.ndim != 2 or layer.weight.shape[0] != layer.weight.shape[1]:
        raise NotImplementedError("vanilla_mask_only currently supports square CTRNN recurrent layers only.")
    wrec = layer.weight.detach().cpu().numpy().astype(np.float64, copy=True)
    operator = wrec - np.eye(wrec.shape[0], dtype=np.float64)
    score_operator, stats = _noise_prune_scores_with_adaptive_shift(
        operator,
        amount=float(amount),
        sigma=float(sigma),
        eps=float(eps),
        shift_buffer=float(leak_shift),
    )
    scored = np.abs(score_operator)
    if getattr(model, "no_self_connections", False):
        np.fill_diagonal(scored, 0.0)
    keep_mask = _exact_offdiag_keep_mask_from_scores(
        scored,
        amount=float(amount),
        no_self_connections=bool(getattr(model, "no_self_connections", False)),
    )
    mask_stats = _apply_keep_mask_to_model(model, keep_mask)
    stats.update(mask_stats)
    stats.update({
        "variant": "vanilla_A_mask_only",
        "candidate_rec_abs_mean": float(np.mean(np.abs(wrec * keep_mask))),
        "score_rec_abs_mean": float(np.mean(scored)),
        "amount": float(amount),
        "target_density": float(1.0 - amount),
    })
    return stats


class VanillaMaskNoisePruneStrategy(BasePruner):
    name = "vanilla_mask_only"
    aliases = ("noise_prune_mask", "vanilla_mask")
    description = "Mask-only vanilla noise-prune using Lyapunov edge saliency on A = W - I."

    def apply(
        self,
        context: PruneContext,
        state: Mapping[str, object],
        **kwargs,
    ) -> Mapping[str, float]:
        return vanilla_mask_prune_recurrent(
            context.model,
            context.amount,
            sigma=float(kwargs.get("sigma", 1.0)),
            eps=float(kwargs.get("eps", 0.3)),
            leak_shift=float(kwargs.get("leak_shift", 0.0)),
            include_feedforward=context.prune_feedforward,
        )


__all__ = [
    "VanillaMaskNoisePruneStrategy",
    "vanilla_mask_prune_recurrent",
    "_noise_prune_scores_with_adaptive_shift",
    "_exact_offdiag_keep_mask_from_scores",
    "_apply_keep_mask_to_model",
]
