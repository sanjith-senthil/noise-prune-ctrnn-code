"""Simulation-based noise-prune with expectation-preserving rescaling."""

from __future__ import annotations

import json
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils import prune

from .pruners import BasePruner, PruneContext
from .simulation_noise_prune import (
    _as_numpy_batches,
    _collect_centered_samples,
    _empirical_covariance,
    _extract_ctrnn,
    _natural_variability_stats,
    _resolve_noise_scale,
    empirical_noise_prune_scores,
)


def simulation_noise_prune_rescale_recurrent(
    model: torch.nn.Module,
    amount: float,
    *,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    sigma: float | None = None,
    sigma_source: str = "natural_voltage",
    eps: float = 0.3,
    observable_space: str = "rate",
    inject_space: str = "rate",
    centering: str = "trajectory_mean",
    max_samples: int = 25_000,
    burn_in_steps: int = 300,
    rng_seed: int | None = None,
    rescale_cap: float | None = None,
    rescale_cap_quantile: float | None = None,
    include_feedforward: bool = False,
) -> Dict[str, float]:
    if include_feedforward:
        raise NotImplementedError("simulation_noise_prune_rescale currently supports recurrent pruning only.")
    if not batches:
        raise ValueError("simulation_noise_prune_rescale requires sampled batches.")
    if rescale_cap is not None and float(rescale_cap) <= 0.0:
        raise ValueError("rescale_cap must be positive when provided.")
    if rescale_cap_quantile is not None and not (0.0 < float(rescale_cap_quantile) <= 1.0):
        raise ValueError("rescale_cap_quantile must lie in (0, 1].")

    rng = np.random.default_rng(rng_seed)
    net = _extract_ctrnn(model)
    batch_arrays = _as_numpy_batches(batches)
    natural = _natural_variability_stats(net, batch_arrays, burn_in_steps=int(burn_in_steps))
    sigma_used = _resolve_noise_scale(natural, sigma=sigma, sigma_source=sigma_source)
    X, sim_stats = _collect_centered_samples(
        net,
        batch_arrays,
        observable_space=observable_space,
        inject_space=inject_space,
        centering=centering,
        noise_scale=sigma_used,
        max_samples=int(max_samples),
        burn_in_steps=int(burn_in_steps),
        rng=rng,
    )
    C_emp = _empirical_covariance(X, center_assumed=True)
    empirical_mean_gain = np.asarray(json.loads(natural["empirical_mean_gain_json"]), dtype=np.float64)
    score_weights = net.wrec
    if observable_space == "voltage":
        score_weights = net.wrec * empirical_mean_gain[np.newaxis, :]
    probs, score_stats = empirical_noise_prune_scores(
        score_weights,
        C_emp,
        sigma=float(sigma_used),
        eps=float(eps),
        target_density=float(1.0 - amount),
    )

    off_mask = ~np.eye(net.wrec.shape[0], dtype=bool)
    if net.no_self_connections:
        np.fill_diagonal(off_mask, False)
    rows, cols = np.where(off_mask)
    flat_probs = probs[rows, cols]
    draws = rng.random(size=flat_probs.shape[0])
    keep = draws < flat_probs

    w_candidate = np.zeros_like(net.wrec, dtype=np.float64)
    kept_rows = rows[keep]
    kept_cols = cols[keep]
    kept_probs = flat_probs[keep]
    positive_probs = flat_probs[flat_probs > 0.0]
    inv_p = 1.0 / positive_probs if positive_probs.size else np.array([], dtype=np.float64)
    cap_value = None
    cap_mode = "none"
    if rescale_cap is not None:
        cap_value = float(rescale_cap)
        cap_mode = "fixed"
    elif rescale_cap_quantile is not None and inv_p.size:
        cap_value = float(np.quantile(inv_p, float(rescale_cap_quantile)))
        cap_mode = "quantile"
    if kept_rows.size > 0:
        amp = 1.0 / kept_probs
        if cap_value is not None:
            amp = np.minimum(amp, cap_value)
        w_candidate[kept_rows, kept_cols] = net.wrec[kept_rows, kept_cols] * amp
    else:
        amp = np.array([], dtype=np.float64)

    tensor = torch.tensor(w_candidate, dtype=model.hidden_layer.weight.dtype, device=model.hidden_layer.weight.device)
    if getattr(model, "no_self_connections", False):
        tensor.fill_diagonal_(0.0)

    from .strategies import _consolidate_if_pruned, _weight_scores_to_mask, enforce_constraints

    _consolidate_if_pruned(model.hidden_layer)
    model.hidden_layer.weight.data.copy_(tensor)
    scores = tensor.abs()
    if getattr(model, "no_self_connections", False):
        scores.fill_diagonal_(-1.0)
    mask = _weight_scores_to_mask(scores, float(amount)).to(dtype=model.hidden_layer.weight.dtype)
    if getattr(model, "no_self_connections", False):
        mask.fill_diagonal_(0.0)
    prune.custom_from_mask(model.hidden_layer, name="weight", mask=mask)
    enforce_constraints(model)

    stats: Dict[str, float] = {}
    stats.update(natural)
    stats.update(sim_stats)
    stats.update(score_stats)
    stats.update({
        "variant": "simulation_empirical_cov_rescale",
        "observable_space": observable_space,
        "inject_space": inject_space,
        "centering": centering,
        "sigma_source": sigma_source if sigma is None else "manual",
        "sigma_used": float(sigma_used),
        "amount": float(amount),
        "target_density": float(1.0 - amount),
        "enforced_density": float(mask.sum().item()) / float(mask.numel()),
        "burn_in_steps": int(burn_in_steps),
        "empirical_cov_trace": float(np.trace(C_emp)),
        "empirical_cov_diag_mean": float(np.mean(np.diag(C_emp))),
        "u_shape_json": json.dumps([list(x.shape) for x in batch_arrays]),
        "frac_amp_gt10": float(np.mean(inv_p > 10.0)) if inv_p.size else 0.0,
        "frac_amp_gt20": float(np.mean(inv_p > 20.0)) if inv_p.size else 0.0,
        "inv_p_mean": float(np.mean(inv_p)) if inv_p.size else 0.0,
        "inv_p_max": float(np.max(inv_p)) if inv_p.size else 0.0,
        "rescale_capped": bool(cap_value is not None),
        "rescale_cap_mode": cap_mode,
        "rescale_cap_value": float(cap_value) if cap_value is not None else 0.0,
        "rescale_cap_quantile": float(rescale_cap_quantile) if rescale_cap_quantile is not None else 0.0,
        "frac_positive_amp_capped": (
            float(np.mean(inv_p > cap_value))
            if cap_value is not None and inv_p.size
            else 0.0
        ),
        "kept_amp_mean": float(np.mean(amp)) if amp.size else 0.0,
        "kept_amp_max": float(np.max(amp)) if amp.size else 0.0,
    })
    return stats


class SimulationNoisePruneRescaleStrategy(BasePruner):
    name = "simulation_noise_prune_rescale"
    aliases = ("sim_noise_prune_rescale", "empirical_cov_noise_prune_rescale")
    description = "Simulation-based noise-prune with expectation-preserving rescaling."
    requires_batches = True
    default_batch_count = 20

    def apply(
        self,
        context: PruneContext,
        state: Mapping[str, object],
        **kwargs,
    ) -> Mapping[str, float]:
        return simulation_noise_prune_rescale_recurrent(
            context.model,
            context.amount,
            batches=context.batches,
            sigma=kwargs.get("sigma"),
            sigma_source=str(kwargs.get("sigma_source", "natural_voltage")),
            eps=float(kwargs.get("eps", 0.3)),
            observable_space=str(kwargs.get("observable_space", "rate")),
            inject_space=str(kwargs.get("inject_space", "rate")),
            centering=str(kwargs.get("centering", "trajectory_mean")),
            max_samples=int(kwargs.get("max_samples", 25_000)),
            burn_in_steps=int(kwargs.get("burn_in_steps", 300)),
            rng_seed=kwargs.get("rng_seed"),
            rescale_cap=kwargs.get("rescale_cap"),
            rescale_cap_quantile=kwargs.get("rescale_cap_quantile"),
            include_feedforward=context.prune_feedforward,
        )


class SimulationNoisePruneCappedRescaleStrategy(BasePruner):
    name = "simulation_noise_prune_capped_rescale"
    aliases = ("sim_noise_prune_capped_rescale", "snp_capped_rescale")
    description = "Simulation-based noise-prune with capped rescale amplification."
    requires_batches = True
    default_batch_count = 20

    def apply(
        self,
        context: PruneContext,
        state: Mapping[str, object],
        **kwargs,
    ) -> Mapping[str, float]:
        return simulation_noise_prune_rescale_recurrent(
            context.model,
            context.amount,
            batches=context.batches,
            sigma=kwargs.get("sigma"),
            sigma_source=str(kwargs.get("sigma_source", "natural_voltage")),
            eps=float(kwargs.get("eps", 0.3)),
            observable_space=str(kwargs.get("observable_space", "rate")),
            inject_space=str(kwargs.get("inject_space", "rate")),
            centering=str(kwargs.get("centering", "trajectory_mean")),
            max_samples=int(kwargs.get("max_samples", 25_000)),
            burn_in_steps=int(kwargs.get("burn_in_steps", 300)),
            rng_seed=kwargs.get("rng_seed"),
            rescale_cap=kwargs.get("rescale_cap"),
            rescale_cap_quantile=kwargs.get("rescale_cap_quantile"),
            include_feedforward=context.prune_feedforward,
        )


__all__ = [
    "SimulationNoisePruneCappedRescaleStrategy",
    "SimulationNoisePruneRescaleStrategy",
    "simulation_noise_prune_rescale_recurrent",
]
