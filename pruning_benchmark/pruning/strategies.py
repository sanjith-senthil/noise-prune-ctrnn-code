"""Core pruning utilities focused on the primary RNN baselines."""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from ..models import CTRNN
from ..models.activations import activation_derivative_np
from .pruners import (
    BasePruner,
    PruneContext,
    register_pruner,
)
from .simulation_noise_prune import SimulationNoisePruneStrategy
from .simulation_noise_prune_rescale import (
    SimulationNoisePruneCappedRescaleStrategy,
    SimulationNoisePruneRescaleStrategy,
)
from .vanilla_mask_prune import VanillaMaskNoisePruneStrategy

PRUNE_AMOUNT_STEP = 0.1
_STEP_EPS = 1e-6


def _prune_stats_mapping(stats: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in stats.items():
        if isinstance(value, (bool, str)) or value is None:
            out[key] = value
        else:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = value
    return out


def validate_prune_fraction(amount: float, *, step: float = PRUNE_AMOUNT_STEP) -> float:
    """Ensure pruning fractions follow the global step granularity (defaults to 10%)."""
    if math.isnan(amount):
        raise ValueError("Pruning amount cannot be NaN")
    if amount < 0.0 or amount > 1.0:
        raise ValueError(f"Pruning amount must be in [0, 1], got {amount}")
    scaled = round(amount / step)
    normalized = round(scaled * step, 10)
    if abs(normalized - amount) > _STEP_EPS:
        raise ValueError(
            f"Pruning amount {amount} must be a multiple of {step:.2f}. "
            f"Suggested value: {normalized:.1f}."
        )
    return float(normalized)


def _layer_names(include_feedforward: bool) -> Tuple[str, ...]:
    names = ["hidden_layer"]
    if include_feedforward:
        names.extend(["input_layer", "readout_layer"])
    return tuple(names)


@torch.no_grad()
def enforce_constraints(model: CTRNN) -> None:
    """Apply Dale's law and remove self-connections when requested by the model."""
    if not hasattr(model, "hidden_layer"):
        return
    layer = model.hidden_layer
    weight = getattr(layer, "weight_orig", layer.weight)

    if getattr(model, "use_dale", False):
        weight.mul_(model.dale_sign).abs_().mul_(model.dale_sign)

    if getattr(model, "no_self_connections", False):
        weight.fill_diagonal_(0.0)
        mask = getattr(layer, "weight_mask", None)
        if mask is not None:
            mask.fill_diagonal_(0.0)


def _consolidate_if_pruned(layer: nn.Module) -> None:
    try:
        prune.remove(layer, "weight")
    except (ValueError, AttributeError):
        pass


def _neuron_keep_mask_from_scores(scores: torch.Tensor, amount: float) -> torch.Tensor:
    amount = validate_prune_fraction(float(amount))
    H = scores.numel()
    k_prune = int(round(amount * H))
    if k_prune <= 0:
        return torch.ones(H, dtype=torch.uint8, device=scores.device)
    keep = torch.zeros(H, dtype=torch.uint8, device=scores.device)
    idx = torch.argsort(scores, descending=True)
    keep[idx[: H - k_prune]] = 1
    return keep


def _weight_scores_to_mask(scores: torch.Tensor, amount: float) -> torch.Tensor:
    amount = validate_prune_fraction(float(amount))
    flat = scores.flatten()
    k_prune = int(round(amount * flat.numel()))
    if k_prune <= 0:
        return torch.ones_like(scores, dtype=torch.uint8)
    if k_prune >= flat.numel():
        return torch.zeros_like(scores, dtype=torch.uint8)
    sorted_vals, sorted_idx = torch.sort(flat)
    thresh = sorted_vals[k_prune - 1]
    mask = (scores > thresh).to(torch.uint8)
    # bring mask exactly to target sparsity if many scores equal threshold
    current_kept = mask.sum().item()
    target_kept = scores.numel() - k_prune
    if current_kept < target_kept:
        equal_mask = (scores == thresh).flatten()
        equal_indices = torch.nonzero(equal_mask, as_tuple=False).view(-1)
        needed = target_kept - current_kept
        if needed > 0 and equal_indices.numel() > 0:
            perm = torch.randperm(equal_indices.numel(), device=equal_indices.device)
            keep_indices = equal_indices[perm[:needed]]
            flat_mask = mask.view(-1)
            flat_mask[keep_indices] = 1
            mask = flat_mask.view_as(scores)
    return mask


def _apply_neuron_keep_mask(model: CTRNN, keep: torch.Tensor, *, include_feedforward: bool = True) -> None:
    H = model.H
    if keep.numel() != H:
        raise ValueError(f"Neuron mask has shape {keep.shape}, expected {H}")

    keep = keep.to(dtype=torch.uint8, device=model.hidden_layer.weight.device)
    keep_r = keep.view(-1, 1)
    keep_c = keep.view(1, -1)

    layers = [model.hidden_layer]
    if include_feedforward:
        layers.extend([model.input_layer, model.readout_layer])

    for layer in layers:
        _consolidate_if_pruned(layer)

    mask_hh = (keep_r & keep_c).to(dtype=model.hidden_layer.weight.dtype)
    if getattr(model, "no_self_connections", False):
        mask_hh.fill_diagonal_(0.0)
    prune.custom_from_mask(model.hidden_layer, name="weight", mask=mask_hh)

    if include_feedforward:
        input_mask = keep_r.expand_as(model.input_layer.weight).to(dtype=model.input_layer.weight.dtype)
        prune.custom_from_mask(model.input_layer, name="weight", mask=input_mask)

        readout_mask = keep_c.expand_as(model.readout_layer.weight).to(dtype=model.readout_layer.weight.dtype)
        prune.custom_from_mask(model.readout_layer, name="weight", mask=readout_mask)
    enforce_constraints(model)


def _prune_layer(layer: nn.Module, fn, amount: float) -> None:
    if layer is None or not hasattr(layer, "weight"):
        return
    fn(layer, name="weight", amount=amount)


def prune_random_unstructured(model: CTRNN, amount: float, *, include_feedforward: bool = False) -> None:
    amount = validate_prune_fraction(float(amount))
    for layer_name in _layer_names(include_feedforward):
        layer = getattr(model, layer_name, None)
        _prune_layer(layer, prune.random_unstructured, amount)
    enforce_constraints(model)


def prune_l1_unstructured(model: CTRNN, amount: float, *, include_feedforward: bool = False) -> None:
    amount = validate_prune_fraction(float(amount))
    for layer_name in _layer_names(include_feedforward):
        layer = getattr(model, layer_name, None)
        _prune_layer(layer, prune.l1_unstructured, amount)
    enforce_constraints(model)


def prune_scores_unstructured(
    model: CTRNN,
    scores: torch.Tensor,
    amount: float,
    *,
    include_feedforward: bool = False,
) -> None:
    mask = _weight_scores_to_mask(scores, amount).to(dtype=model.hidden_layer.weight.dtype)
    _consolidate_if_pruned(model.hidden_layer)
    prune.custom_from_mask(model.hidden_layer, name="weight", mask=mask)
    if include_feedforward:
        for layer_name in ("input_layer", "readout_layer"):
            layer = getattr(model, layer_name, None)
            _prune_layer(layer, prune.l1_unstructured, amount)
    enforce_constraints(model)


def noise_prune_recurrent(
    model: CTRNN,
    amount: float,
    *,
    sigma: float = 1.0,
    eps: float = 0.3,
    leak_shift: float = 0.0,
    matched_diagonal: bool = True,
    rescale_weights: bool = True,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 5,
    rescale_cap: Optional[float] = None,
    rescale_cap_quantile: Optional[float] = None,
    include_feedforward: bool = False,
) -> Dict[str, float]:
    amount = validate_prune_fraction(float(amount))
    stats: Dict[str, float] = {}
    if amount <= 0.0:
        return stats

    rng = rng or np.random.default_rng()
    weight_tensor = model.hidden_layer.weight
    if weight_tensor.ndim != 2 or weight_tensor.shape[0] != weight_tensor.shape[1]:
        raise NotImplementedError("noise_prune currently supports square recurrent matrices (CTRNN/GRU).")
    weight = weight_tensor.detach().cpu().numpy()
    desired_density = float(max(0.0, min(1.0, 1.0 - amount)))
    current_shift = float(leak_shift)
    used_shift = current_shift

    for attempt in range(max_attempts):
        base_shift = 1.0 + current_shift
        shifted = weight - base_shift * np.eye(weight.shape[0], dtype=weight.dtype)
        from .noise_prune import noise_prune as ct_noise_prune
        try:
            pruned, noise_stats = ct_noise_prune(
                shifted,
                sigma=float(sigma),
                eps=float(eps),
                matched_diagonal=bool(matched_diagonal),
                rescale_weights=bool(rescale_weights),
                rng=rng,
                target_density=desired_density,
                rescale_cap=rescale_cap,
                rescale_cap_quantile=rescale_cap_quantile,
            )
            stats = _prune_stats_mapping(noise_stats)
            stats["leak_shift"] = float(current_shift)
            used_shift = current_shift
            break
        except ValueError as exc:
            if attempt >= max_attempts - 1:
                raise
            current_shift = max(0.5, current_shift * 2.0 if current_shift > 0 else 0.5)
    else:  # pragma: no cover - loop exhaustion defensive clause
        raise RuntimeError("noise_prune failed to converge")

    restored = pruned + (1.0 + used_shift) * np.eye(pruned.shape[0], dtype=pruned.dtype)
    tensor = torch.tensor(restored, dtype=model.hidden_layer.weight.dtype, device=model.hidden_layer.weight.device)
    if getattr(model, "no_self_connections", False):
        tensor.fill_diagonal_(0.0)
    _consolidate_if_pruned(model.hidden_layer)
    model.hidden_layer.weight.data.copy_(tensor)
    # Use noise-prune output as saliency scores, then apply an exact sparsity mask.
    scores = tensor.abs()
    if getattr(model, "no_self_connections", False):
        scores.fill_diagonal_(-1.0)
    mask = _weight_scores_to_mask(scores, amount).to(dtype=model.hidden_layer.weight.dtype)
    if getattr(model, "no_self_connections", False):
        mask.fill_diagonal_(0.0)
    prune.custom_from_mask(model.hidden_layer, name="weight", mask=mask)
    if include_feedforward:
        for layer_name in ("input_layer", "readout_layer"):
            layer = getattr(model, layer_name, None)
            _prune_layer(layer, prune.l1_unstructured, amount)
    enforce_constraints(model)

    stats.update({
        "amount": float(amount),
        "target_density": desired_density,
        "enforced_density": float(mask.sum().item()) / float(mask.numel()),
    })
    return stats


def _recurrent_input_inverse_hessian(
    model: CTRNN,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    damping: float = 1e-3,
    max_samples: int = 25_000,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    if not hasattr(model, "hidden_layer"):
        raise ValueError("Layerwise compensated OBS requires a recurrent hidden_layer.")
    batches = list(batches)
    if not batches:
        raise ValueError("Layerwise compensated OBS requires sampled batches.")
    if damping <= 0.0:
        raise ValueError("OBS damping must be positive.")

    weight = model.hidden_layer.weight
    device = weight.device
    dtype = weight.dtype
    h = int(weight.shape[1])
    second_moment = torch.zeros((h, h), device=device, dtype=dtype)
    sample_count = 0
    max_samples = max(1, int(max_samples))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for x_batch, _ in batches:
            fr, v = model.init_state(int(x_batch.shape[1]), device=device)
            for t in range(int(x_batch.shape[0])):
                recurrent_input = fr.detach()
                remaining = max_samples - sample_count
                if remaining <= 0:
                    break
                if recurrent_input.shape[0] > remaining:
                    recurrent_input = recurrent_input[:remaining]
                second_moment += recurrent_input.T @ recurrent_input
                sample_count += int(recurrent_input.shape[0])
                if sample_count >= max_samples:
                    break
                fr, v = model.step(fr, v, x_batch[t].to(device))
            if sample_count >= max_samples:
                break
    if was_training:
        model.train()
    else:
        model.eval()
    if sample_count <= 0:
        raise ValueError("No recurrent-input samples collected for compensated OBS.")

    hessian = second_moment / float(sample_count)
    hessian = 0.5 * (hessian + hessian.T)
    hessian = hessian + float(damping) * torch.eye(h, device=device, dtype=dtype)
    h_inv = torch.linalg.pinv(hessian)
    h_inv = 0.5 * (h_inv + h_inv.T)
    stats = {
        "obs_compensated_damping": float(damping),
        "obs_compensated_max_samples": float(max_samples),
        "obs_compensated_sample_count": float(sample_count),
        "obs_compensated_hessian_trace": float(torch.trace(hessian).detach().cpu()),
        "obs_compensated_hinv_diag_min": float(torch.diagonal(h_inv).min().detach().cpu()),
        "obs_compensated_hinv_diag_mean": float(torch.diagonal(h_inv).mean().detach().cpu()),
        "obs_compensated_hinv_diag_max": float(torch.diagonal(h_inv).max().detach().cpu()),
    }
    return h_inv, stats


def prune_obs_compensated_recurrent(
    model: CTRNN,
    amount: float,
    *,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    damping: float = 1e-3,
    max_samples: int = 25_000,
    compensation_mode: str = "diagonal",
    exact_block_threshold: int = 128,
    include_feedforward: bool = False,
) -> Mapping[str, float]:
    """Layerwise compensated OBS for recurrent weights.

    This is the tractable one-shot OBS form used for large layers: the Hessian
    is approximated by the recurrent-input second moment, and surviving weights
    are compensated row-wise with the inverse layer Hessian. ``diagonal`` mode
    uses the additive single-weight OBS update, while ``block`` solves the exact
    OBS block update for each row's pruned set. ``auto`` uses the exact block
    update only when the row's pruned set is small enough.
    """

    amount = validate_prune_fraction(float(amount))
    mode = str(compensation_mode).lower()
    if mode not in {"diagonal", "block", "auto"}:
        raise ValueError("obs compensation_mode must be one of: diagonal, block, auto")
    if amount <= 0.0:
        return {}

    h_inv, stats = _recurrent_input_inverse_hessian(
        model,
        batches,
        damping=float(damping),
        max_samples=int(max_samples),
    )
    weight = model.hidden_layer.weight.detach()
    h_inv_diag = torch.diagonal(h_inv).abs().clamp_min(1e-8)
    scores = 0.5 * weight.pow(2) / h_inv_diag.view(1, -1)
    if getattr(model, "no_self_connections", False) and scores.shape[0] == scores.shape[1]:
        scores = scores.clone()
        scores.fill_diagonal_(float("-inf"))

    mask = _weight_scores_to_mask(scores, amount).to(dtype=weight.dtype, device=weight.device)
    if getattr(model, "no_self_connections", False) and mask.shape[0] == mask.shape[1]:
        mask.fill_diagonal_(0.0)

    compensated = weight.clone()
    pruned_bool = mask == 0
    block_rows = 0
    diagonal_rows = 0
    skipped_rows = 0
    exact_block_threshold = max(0, int(exact_block_threshold))

    for row_idx in range(weight.shape[0]):
        pruned_cols = torch.nonzero(pruned_bool[row_idx], as_tuple=False).flatten()
        if pruned_cols.numel() == 0:
            skipped_rows += 1
            continue
        w_pruned = weight[row_idx, pruned_cols]
        if torch.count_nonzero(w_pruned).item() == 0:
            compensated[row_idx, pruned_cols] = 0.0
            skipped_rows += 1
            continue

        use_block = mode == "block" or (mode == "auto" and pruned_cols.numel() <= exact_block_threshold)
        if use_block:
            sub = h_inv.index_select(0, pruned_cols).index_select(1, pruned_cols)
            eye = torch.eye(sub.shape[0], device=sub.device, dtype=sub.dtype)
            try:
                coeff = torch.linalg.solve(sub + 1e-8 * eye, w_pruned)
            except RuntimeError:
                coeff = torch.linalg.lstsq(sub + 1e-6 * eye, w_pruned).solution
            delta = h_inv.index_select(1, pruned_cols) @ coeff
            block_rows += 1
        else:
            coeff = torch.zeros(weight.shape[1], device=weight.device, dtype=weight.dtype)
            coeff[pruned_cols] = w_pruned / h_inv_diag[pruned_cols]
            delta = h_inv @ coeff
            diagonal_rows += 1
        compensated[row_idx] = weight[row_idx] - delta
        compensated[row_idx, pruned_cols] = 0.0

    if getattr(model, "no_self_connections", False) and compensated.shape[0] == compensated.shape[1]:
        compensated.fill_diagonal_(0.0)

    _consolidate_if_pruned(model.hidden_layer)
    model.hidden_layer.weight.data.copy_(compensated)
    prune.custom_from_mask(model.hidden_layer, name="weight", mask=mask)
    if include_feedforward:
        for layer_name in ("input_layer", "readout_layer"):
            layer = getattr(model, layer_name, None)
            _prune_layer(layer, prune.l1_unstructured, amount)
    enforce_constraints(model)

    mode_code = {"diagonal": 0.0, "block": 1.0, "auto": 2.0}[mode]
    stats.update({
        "obs_compensated_amount": float(amount),
        "obs_compensated_mode_code": mode_code,
        "obs_compensated_block_rows": float(block_rows),
        "obs_compensated_diagonal_rows": float(diagonal_rows),
        "obs_compensated_skipped_rows": float(skipped_rows),
        "obs_compensated_kept_edges": float(mask.sum().item()),
        "obs_compensated_density_achieved": float(mask.sum().item()) / float(mask.numel()),
        "obs_compensated_weight_abs_mean": float(compensated.abs().mean().detach().cpu()),
        "obs_compensated_weight_abs_max": float(compensated.abs().max().detach().cpu()),
    })
    return stats


def finalize_pruning(model: CTRNN) -> None:
    """Remove pruning reparameterisations so saved checkpoints are dense tensors."""
    for layer_name in ("input_layer", "hidden_layer", "readout_layer"):
        if not hasattr(model, layer_name):
            continue
        layer = getattr(model, layer_name)
        try:
            prune.remove(layer, "weight")
        except (ValueError, AttributeError):
            continue
    enforce_constraints(model)


class NoisePruneStrategy(BasePruner):
    name = "noise_prune"
    description = "Covariance-guided pruning on the continuous-time operator."

    def apply(
        self,
        context: PruneContext,
        state: Mapping[str, object],
        **kwargs,
    ) -> Mapping[str, float]:
        try:
            return noise_prune_recurrent(
                context.model,
                context.amount,
                include_feedforward=context.prune_feedforward,
                **kwargs,
            )
        except NotImplementedError as exc:
            warnings.warn(f"{exc} Falling back to magnitude pruning for this model.")
            prune_l1_unstructured(
                context.model,
                context.amount,
                include_feedforward=context.prune_feedforward,
            )
            return {"fallback": "l1_unstructured"}


class NoisePruneCappedRescaleStrategy(BasePruner):
    name = "noise_prune_capped_rescale"
    aliases = ("vnp_capped_rescale", "vanilla_capped_rescale")
    description = "Vanilla noise-prune with capped rescale amplification."

    def apply(
        self,
        context: PruneContext,
        state: Mapping[str, object],
        **kwargs,
    ) -> Mapping[str, float]:
        return noise_prune_recurrent(
            context.model,
            context.amount,
            include_feedforward=context.prune_feedforward,
            **kwargs,
        )


class RandomUnstructuredPruner(BasePruner):
    name = "random_unstructured"
    aliases = ("random",)

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        prune_random_unstructured(
            context.model,
            context.amount,
            include_feedforward=context.prune_feedforward,
        )
        return {}


class L1UnstructuredPruner(BasePruner):
    name = "l1_unstructured"
    aliases = ("l1",)

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        prune_l1_unstructured(
            context.model,
            context.amount,
            include_feedforward=context.prune_feedforward,
        )
        return {}


class OBSCompensatedPruner(BasePruner):
    name = "obs_compensated"
    description = "Layerwise one-shot OBS with recurrent-input Hessian compensation."
    requires_batches = True
    default_batch_count = 20

    def apply(self, context: PruneContext, state: Mapping[str, object], **kwargs) -> Mapping[str, float]:
        if not context.batches:
            raise ValueError("Compensated OBS requires sampled batches.")
        return prune_obs_compensated_recurrent(
            context.model,
            context.amount,
            batches=context.batches,
            damping=float(kwargs.get("damping", 1e-3)),
            max_samples=int(kwargs.get("max_samples", 25_000)),
            compensation_mode=str(kwargs.get("compensation_mode", "diagonal")),
            exact_block_threshold=int(kwargs.get("exact_block_threshold", 128)),
            include_feedforward=context.prune_feedforward,
        )


# Register built-in strategies
register_pruner(NoisePruneStrategy())
register_pruner(NoisePruneCappedRescaleStrategy())
register_pruner(VanillaMaskNoisePruneStrategy())
register_pruner(SimulationNoisePruneStrategy())
register_pruner(SimulationNoisePruneRescaleStrategy())
register_pruner(SimulationNoisePruneCappedRescaleStrategy())
register_pruner(RandomUnstructuredPruner())
register_pruner(L1UnstructuredPruner())
register_pruner(OBSCompensatedPruner())


__all__ = [
    "PRUNE_AMOUNT_STEP",
    "enforce_constraints",
    "finalize_pruning",
    "noise_prune_recurrent",
    "prune_l1_unstructured",
    "prune_obs_compensated_recurrent",
    "prune_random_unstructured",
    "validate_prune_fraction",
]
