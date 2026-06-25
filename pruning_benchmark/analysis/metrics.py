"""Convenience metrics and run logging helpers for characterising pruned CTRNNs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

from ..models import CTRNN


def _has_ctrnn_layers(model: torch.nn.Module) -> bool:
    return hasattr(model, "hidden_layer") and hasattr(model.hidden_layer, "weight")


def count_nonzero_and_total(t: torch.Tensor):
    nz = int((t != 0).sum().item())
    tot = t.numel()
    return nz, tot


def count_mask_nonzero_and_total(layer: torch.nn.Module):
    mask = getattr(layer, "weight_mask", None)
    if mask is None:
        return None
    nz = int((mask != 0).sum().item())
    tot = mask.numel()
    return nz, tot


@torch.no_grad()
def recurrent_sparsity(model: CTRNN):
    layer = model.hidden_layer
    mask_counts = count_mask_nonzero_and_total(layer)
    if mask_counts is not None:
        nz, tot = mask_counts
        return 1.0 - (nz / tot)
    W = layer.weight
    nz, tot = count_nonzero_and_total(W)
    return 1.0 - (nz / tot)


@torch.no_grad()
def spectral_radius(W: torch.Tensor) -> float:
    W = W.detach().to("cpu", dtype=torch.float32)
    W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        eigvals = np.linalg.eigvals(W.numpy())
        return float(np.max(np.abs(eigvals)))
    except Exception:
        return float("nan")


@torch.no_grad()
def spectral_abscissa(W: torch.Tensor) -> float:
    """Return max real part of eigenvalues."""
    W = W.detach().to("cpu", dtype=torch.float32)
    W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        eigvals = np.linalg.eigvals(W.numpy())
        return float(np.max(np.real(eigvals)))
    except Exception:
        return float("nan")


@torch.no_grad()
def ctrnn_stability_proxy(model: CTRNN):
    if not _has_ctrnn_layers(model):
        return float("nan")
    W = model.hidden_layer.weight.detach()
    rho = spectral_radius(W)
    return model.alpha * rho


@torch.no_grad()
def layer_sparsities(model: CTRNN):
    """Sparsity per layer (fraction of zeros), preferring pruning masks when present."""
    layers = {}
    if hasattr(model, "input_layer"):
        layers["input"] = model.input_layer
    if _has_ctrnn_layers(model):
        layers["recurrent"] = model.hidden_layer
    if hasattr(model, "readout_layer"):
        layers["readout"] = model.readout_layer
    out = {}
    for name, layer in layers.items():
        mask_counts = count_mask_nonzero_and_total(layer)
        if mask_counts is not None:
            nz, tot = mask_counts
            out[name] = 1.0 - (nz / tot)
            continue
        W = layer.weight
        nz = int((W != 0).sum().item())
        tot = W.numel()
        out[name] = 1.0 - (nz / tot)
    return out


@torch.no_grad()
def neuron_keep_fraction(model: CTRNN):
    """Fraction of hidden neurons effectively kept (not isolated)."""
    if not _has_ctrnn_layers(model):
        return float("nan")
    W = model.hidden_layer.weight.detach()
    if W.size(0) != W.size(1):
        H = W.size(1)
        gates = max(1, W.size(0) // H)
        W = W.view(gates, H, H).abs().sum(dim=0)
    row_zero = (W.abs().sum(dim=1) == 0)
    col_zero = (W.abs().sum(dim=0) == 0)
    removed = int((row_zero & col_zero).sum().item())
    H = W.size(0)
    kept = H - removed
    return kept / max(1, H)


@torch.no_grad()
def neuron_pruning_stats(model: CTRNN):
    """
    Returns counts for neuron-level pruning on the recurrent matrix (H x H).

    rows_zero: neurons whose outgoing weights are all zero (row == 0)
    cols_zero: neurons whose incoming weights are all zero (col == 0)
    isolated:  neurons that are both rows_zero AND cols_zero (fully disconnected)
    """
    if not _has_ctrnn_layers(model):
        return {"rows_zero": float("nan"), "cols_zero": float("nan"), "isolated": float("nan")}
    W = model.hidden_layer.weight.detach()
    if W.size(0) != W.size(1):
        H = W.size(1)
        gates = max(1, W.size(0) // H)
        W = W.view(gates, H, H).abs().sum(dim=0)
    row_zero = (W.abs().sum(dim=1) == 0)
    col_zero = (W.abs().sum(dim=0) == 0)
    rows_zero = int(row_zero.sum().item())
    cols_zero = int(col_zero.sum().item())
    isolated = int((row_zero & col_zero).sum().item())
    return {"rows_zero": rows_zero, "cols_zero": cols_zero, "isolated": isolated}


def snapshot_model(model: CTRNN) -> Dict[str, float]:
    """Collect core structural metrics for the current model state."""
    stats = {
        "sparsity": recurrent_sparsity(model),
        "alpha_rho": ctrnn_stability_proxy(model),
        "neuron_keep_fraction": neuron_keep_fraction(model),
    }
    layer_stats = layer_sparsities(model)
    for layer_name, value in layer_stats.items():
        stats[f"sparsity_{layer_name}"] = float(value)

    if _has_ctrnn_layers(model):
        W = model.hidden_layer.weight.detach()
        abs_w = W.abs()
        stats["rec_weight_abs_mean"] = float(abs_w.mean())
        stats["rec_weight_abs_std"] = float(abs_w.std(unbiased=False))
        stats["rec_weight_l2"] = float(torch.linalg.vector_norm(W, ord=2))
        nz_mask = W != 0
        if nz_mask.any():
            abs_nz = abs_w[nz_mask]
            stats["rec_weight_abs_mean_nz"] = float(abs_nz.mean())
            stats["rec_weight_abs_std_nz"] = float(abs_nz.std(unbiased=False))
            stats["rec_weight_nz_count"] = float(nz_mask.sum().item())
        else:
            stats["rec_weight_abs_mean_nz"] = float("nan")
            stats["rec_weight_abs_std_nz"] = float("nan")
            stats["rec_weight_nz_count"] = 0.0
        H = W.size(0)
        eye = torch.eye(H, dtype=W.dtype, device=W.device)
        # Discrete-time linearized Jacobian proxy for the Euler step.
        J_lin = model.oneminusalpha * eye + model.alpha * W
        rho_lin = spectral_radius(J_lin)
        stats["rec_linear_rho"] = float(rho_lin)
        stats["rec_linear_margin"] = float(1.0 - rho_lin)
        # Continuous-time operator proxy: v_dot ≈ (-I + W) v.
        A_ct = W - eye
        abscissa = spectral_abscissa(A_ct)
        stats["rec_ct_abscissa"] = float(abscissa)
        stats["rec_ct_margin"] = float(-abscissa)
    else:
        stats["rec_weight_abs_mean"] = float("nan")
        stats["rec_weight_abs_std"] = float("nan")
        stats["rec_weight_l2"] = float("nan")
        stats["rec_linear_rho"] = float("nan")
        stats["rec_linear_margin"] = float("nan")
        stats["rec_ct_abscissa"] = float("nan")
        stats["rec_ct_margin"] = float("nan")

    if hasattr(model, "readout_layer"):
        readout_abs = model.readout_layer.weight.detach().abs()
        stats["readout_weight_abs_mean"] = float(readout_abs.mean())
        stats["readout_weight_abs_std"] = float(readout_abs.std(unbiased=False))

    if _has_ctrnn_layers(model):
        abs_w = model.hidden_layer.weight.detach().abs()
    else:
        abs_w = None

    if abs_w is not None and getattr(model, "use_dale", False) and hasattr(model, "dale_sign"):
        excit = abs_w[:, model.dale_sign.squeeze() > 0]
        inhib = abs_w[:, model.dale_sign.squeeze() < 0]
        stats["dale_excit_mean"] = float(excit.mean()) if excit.numel() else 0.0
        stats["dale_inhib_mean"] = float(inhib.mean()) if inhib.numel() else 0.0

    for key, value in neuron_pruning_stats(model).items():
        stats[f"neurons_{key}"] = float(value)
    return {k: float(v) for k, v in stats.items()}


def _normalize_metric(value: Any):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def compile_run_metrics(
    phases: Mapping[str, Mapping[str, Any]],
    *,
    extras: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Flatten per-phase metrics into a single dictionary for logging.
    """
    flat: Dict[str, Any] = {}
    for phase, values in phases.items():
        for key, value in values.items():
            flat[f"{phase}_{key}"] = _normalize_metric(value)
    if extras:
        for key, value in extras.items():
            flat[key] = _normalize_metric(value)
    return flat


def save_metrics(
    run_dir: Union[str, Path],
    metrics: Mapping[str, Any],
    filename: str = "metrics.json",
) -> Path:
    """Write metrics to JSON inside the run directory."""
    path = Path(run_dir) / filename
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return path


__all__ = [
    "compile_run_metrics",
    "count_nonzero_and_total",
    "ctrnn_stability_proxy",
    "layer_sparsities",
    "neuron_keep_fraction",
    "neuron_pruning_stats",
    "recurrent_sparsity",
    "save_metrics",
    "spectral_abscissa",
    "snapshot_model",
]
