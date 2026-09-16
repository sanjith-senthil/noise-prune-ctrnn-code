"""Simulation-based noise-prune for nonlinear tanh-like CTRNNs.

This module replaces the Lyapunov covariance in noise-prune with an empirical
covariance estimated from noisy rollouts of the nonlinear network itself.

Paper-oriented defaults:
- observe fluctuations in ``rate`` space
- use ``trajectory_mean`` centering on post-burn-in noisy rollouts
- match injected noise scale to the trained network's natural voltage
  variability along the task trajectory

The implementation intentionally keeps the original edge-magnitude term from
noise-prune and only swaps the covariance estimate.  For tanh-like CTRNNs without
self-connections, the off-diagonal entries of ``A = W - I`` equal the raw
recurrent weights, so the original weight term is recovered by scoring the
off-diagonal entries of ``W_rec`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import root
import torch

from ..models.activations import activation_derivative_np, activation_np, supports_gain_linearization
from .pruners import BasePruner, PruneContext
from .vanilla_mask_prune import (
    _apply_keep_mask_to_model,
    _exact_offdiag_keep_mask_from_scores,
    _noise_prune_scores_with_adaptive_shift,
)


Batch = Tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class _ExtractedCTRNN:
    win: np.ndarray
    bin: np.ndarray
    wrec: np.ndarray
    brec: np.ndarray
    alpha: float
    tau: float
    no_self_connections: bool
    activation: str


def _extract_ctrnn(model: torch.nn.Module) -> _ExtractedCTRNN:
    activation = str(getattr(model, "_activation_name", ""))
    if not supports_gain_linearization(activation):
        raise ValueError("simulation_noise_prune currently supports tanh-like CTRNNs only.")
    return _ExtractedCTRNN(
        win=model.input_layer.weight.detach().cpu().numpy().copy(),
        bin=model.input_layer.bias.detach().cpu().numpy().copy(),
        wrec=model.hidden_layer.weight.detach().cpu().numpy().copy(),
        brec=model.hidden_layer.bias.detach().cpu().numpy().copy(),
        alpha=float(model.alpha),
        tau=float(model.tau),
        no_self_connections=bool(model.no_self_connections),
        activation=activation,
    )


def _step(net: _ExtractedCTRNN, v_t: np.ndarray, u_star: np.ndarray) -> np.ndarray:
    fr_t = activation_np(v_t, net.activation)
    drive = net.win @ u_star + net.bin + net.wrec @ fr_t + net.brec
    return (1.0 - net.alpha) * v_t + net.alpha * drive


def _fixed_point_residual(v: np.ndarray, net: _ExtractedCTRNN, u_star: np.ndarray) -> np.ndarray:
    return _step(net, v, u_star) - v


def _find_operating_point(
    net: _ExtractedCTRNN,
    *,
    u_star: np.ndarray,
    rollout_steps: int,
    tol: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    v = np.zeros(net.wrec.shape[0], dtype=np.float64)
    for _ in range(max(1, int(rollout_steps))):
        v = _step(net, v, u_star)
    sol = root(
        lambda z: _fixed_point_residual(z, net, u_star),
        v,
        method="hybr",
        tol=tol,
    )
    v_star = np.asarray(sol.x, dtype=np.float64)
    resid = _fixed_point_residual(v_star, net, u_star)
    stats = {
        "fixed_point_resid_l2": float(np.linalg.norm(resid)),
        "fixed_point_resid_max": float(np.max(np.abs(resid))),
        "root_success": bool(sol.success),
        "root_status": int(sol.status),
        "root_nfev": int(getattr(sol, "nfev", -1)),
        "mean_abs_gain": float(np.mean(activation_derivative_np(v_star, net.activation))),
    }
    return v_star, stats


def _effective_operator_for_activation(
    wrec: np.ndarray,
    v_star: np.ndarray,
    activation: str,
) -> Tuple[np.ndarray, np.ndarray]:
    gains = activation_derivative_np(v_star, activation)
    return wrec @ np.diag(gains) - np.eye(wrec.shape[0], dtype=np.float64), gains


def _as_numpy_batches(batches: Sequence[Batch]) -> List[np.ndarray]:
    xs: List[np.ndarray] = []
    for x_batch, _ in batches:
        xb = x_batch.detach().cpu().numpy().astype(np.float64, copy=False)
        if xb.ndim != 3:
            raise ValueError(f"Expected batch inputs with shape (T,B,I), got {xb.shape}.")
        xs.append(np.asarray(xb, dtype=np.float64))
    if not xs:
        raise ValueError("simulation_noise_prune requires sampled batches.")
    return xs


def _mean_input_from_numpy_batches(batches: Sequence[np.ndarray]) -> np.ndarray:
    total = None
    count = 0
    for xb in batches:
        batch_mean = xb.mean(axis=(0, 1))
        total = batch_mean if total is None else total + batch_mean
        count += 1
    if total is None or count <= 0:
        raise ValueError("No usable batches available for operating input estimation.")
    return np.asarray(total / count, dtype=np.float64)


def _step_deterministic(
    net: _ExtractedCTRNN,
    *,
    fr_t: np.ndarray,
    v_t: np.ndarray,
    u_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    drive = u_t @ net.win.T + net.bin + fr_t @ net.wrec.T + net.brec
    v_next = (1.0 - net.alpha) * v_t + net.alpha * drive
    fr_next = activation_np(v_next, net.activation)
    return fr_next, v_next


def _step_noisy(
    net: _ExtractedCTRNN,
    *,
    fr_t: np.ndarray,
    v_t: np.ndarray,
    u_t: np.ndarray,
    inject_space: str,
    noise_scale: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    fr_next, v_next = _step_deterministic(net, fr_t=fr_t, v_t=v_t, u_t=u_t)
    if inject_space == "voltage":
        # Euler-Maruyama discretization of dv = f(v) dt + sigma dW_t
        # contributes sqrt(alpha) * sigma * xi per step, not alpha * sigma * xi.
        v_next = v_next + np.sqrt(net.alpha) * rng.normal(scale=noise_scale, size=v_next.shape)
        fr_next = activation_np(v_next, net.activation)
    elif inject_space == "rate":
        fr_next = fr_next + rng.normal(scale=noise_scale, size=fr_next.shape)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported inject_space '{inject_space}'.")
    return fr_next, v_next


def _rollout_noise_free(
    net: _ExtractedCTRNN,
    inputs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    T, B, _ = inputs.shape
    v = np.zeros((B, net.wrec.shape[0]), dtype=np.float64)
    fr = activation_np(v, net.activation)
    fr_seq = np.zeros((T, B, net.wrec.shape[0]), dtype=np.float64)
    v_seq = np.zeros((T, B, net.wrec.shape[0]), dtype=np.float64)
    for t in range(T):
        fr, v = _step_deterministic(net, fr_t=fr, v_t=v, u_t=inputs[t])
        fr_seq[t] = fr
        v_seq[t] = v
    return fr_seq, v_seq


def _rollout_noisy(
    net: _ExtractedCTRNN,
    inputs: np.ndarray,
    *,
    inject_space: str,
    noise_scale: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    T, B, _ = inputs.shape
    v = np.zeros((B, net.wrec.shape[0]), dtype=np.float64)
    fr = activation_np(v, net.activation)
    fr_seq = np.zeros((T, B, net.wrec.shape[0]), dtype=np.float64)
    v_seq = np.zeros((T, B, net.wrec.shape[0]), dtype=np.float64)
    for t in range(T):
        fr, v = _step_noisy(
            net,
            fr_t=fr,
            v_t=v,
            u_t=inputs[t],
            inject_space=inject_space,
            noise_scale=noise_scale,
            rng=rng,
        )
        fr_seq[t] = fr
        v_seq[t] = v
    return fr_seq, v_seq


def _rollout_noise_free_identity(
    net: _ExtractedCTRNN,
    inputs: np.ndarray,
) -> np.ndarray:
    T, B, _ = inputs.shape
    v = np.zeros((B, net.wrec.shape[0]), dtype=np.float64)
    v_seq = np.zeros((T, B, net.wrec.shape[0]), dtype=np.float64)
    for t in range(T):
        drive = inputs[t] @ net.win.T + net.bin + v @ net.wrec.T + net.brec
        v = (1.0 - net.alpha) * v + net.alpha * drive
        v_seq[t] = v
    return v_seq


def _rollout_noisy_identity(
    net: _ExtractedCTRNN,
    inputs: np.ndarray,
    *,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    T, B, _ = inputs.shape
    v = np.zeros((B, net.wrec.shape[0]), dtype=np.float64)
    v_seq = np.zeros((T, B, net.wrec.shape[0]), dtype=np.float64)
    for t in range(T):
        drive = inputs[t] @ net.win.T + net.bin + v @ net.wrec.T + net.brec
        v = (1.0 - net.alpha) * v + net.alpha * drive
        v = v + np.sqrt(net.alpha) * rng.normal(scale=noise_scale, size=v.shape)
        v_seq[t] = v
    return v_seq


def _flatten_samples(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected trajectory tensor with shape (T,B,H), got {x.shape}.")
    return np.reshape(x, (-1, x.shape[-1]))


def _empirical_covariance(
    X: np.ndarray,
    *,
    center_assumed: bool,
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"Expected matrix of centered samples, got {X.shape}.")
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least two samples to estimate covariance.")
    if center_assumed:
        return (X.T @ X) / float(n - 1)
    centered = X - np.mean(X, axis=0, keepdims=True)
    return (centered.T @ centered) / float(n - 1)


def _natural_variability_stats(
    net: _ExtractedCTRNN,
    batches: Sequence[np.ndarray],
    *,
    burn_in_steps: int,
) -> Dict[str, float]:
    all_v: List[np.ndarray] = []
    all_fr: List[np.ndarray] = []
    for xb in batches:
        fr_seq, v_seq = _rollout_noise_free(net, xb)
        start = min(max(0, int(burn_in_steps)), v_seq.shape[0] - 1)
        all_v.append(_flatten_samples(v_seq[start:]))
        all_fr.append(_flatten_samples(fr_seq[start:]))
    v_task = np.concatenate(all_v, axis=0)
    fr_task = np.concatenate(all_fr, axis=0)
    cov_v = _empirical_covariance(v_task, center_assumed=False)
    cov_fr = _empirical_covariance(fr_task, center_assumed=False)
    gain_task = activation_derivative_np(v_task, net.activation)
    sigma_natural_v = float(np.sqrt(np.mean(np.diag(cov_v))))
    sigma_natural_fr = float(np.sqrt(np.mean(np.diag(cov_fr))))
    return {
        "sigma_natural_v": sigma_natural_v,
        "sigma_natural_rate": sigma_natural_fr,
        "natural_cov_trace_v": float(np.trace(cov_v)),
        "natural_cov_trace_rate": float(np.trace(cov_fr)),
        "empirical_mean_gain_json": json.dumps(np.mean(gain_task, axis=0).tolist()),
        "empirical_mean_gain": float(np.mean(gain_task)),
        "empirical_min_gain": float(np.min(gain_task)),
        "empirical_max_gain": float(np.max(gain_task)),
    }


def _resolve_noise_scale(
    natural: Mapping[str, float],
    *,
    sigma: Optional[float],
    sigma_source: str,
    sigma_factor: float = 1.0,
) -> float:
    """Resolve the injected-noise scale.

    ``sigma_factor`` multiplies the network-calibrated scale, so a suite can
    sweep noise relative to each network's own natural variability rather than
    at a shared absolute value.  It is ignored when an absolute ``sigma`` is
    supplied.  The factor must be strictly positive: the probability formula
    divides by ``sigma**2``, so sigma = 0 is undefined.  Use a small factor
    (e.g. 1e-3) to approximate the noise-free limit.
    """
    if sigma_factor <= 0.0:
        raise ValueError("sigma_factor must be positive; use a small value to approach zero noise.")
    if sigma is not None:
        if sigma <= 0.0:
            raise ValueError("sigma must be positive when provided explicitly.")
        return float(sigma)
    if sigma_source == "natural_voltage":
        return float(natural["sigma_natural_v"]) * float(sigma_factor)
    if sigma_source == "natural_rate":
        return float(natural["sigma_natural_rate"]) * float(sigma_factor)
    raise ValueError(f"Unsupported sigma_source '{sigma_source}'.")


def _collect_centered_samples(
    net: _ExtractedCTRNN,
    batches: Sequence[np.ndarray],
    *,
    observable_space: str,
    inject_space: str,
    centering: str,
    noise_scale: float,
    max_samples: int,
    burn_in_steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if observable_space not in {"rate", "voltage"}:
        raise ValueError("observable_space must be 'rate' or 'voltage'.")
    if inject_space not in {"rate", "voltage"}:
        raise ValueError("inject_space must be 'rate' or 'voltage'.")
    if centering not in {"trajectory_mean", "conditional"}:
        raise ValueError("centering must be 'trajectory_mean' or 'conditional'.")
    if max_samples < 2:
        raise ValueError("max_samples must be >= 2.")

    references: List[np.ndarray] = []
    if centering == "conditional":
        for xb in batches:
            fr_ref, v_ref = _rollout_noise_free(net, xb)
            references.append(fr_ref if observable_space == "rate" else v_ref)

    gathered: List[np.ndarray] = []
    gathered_count = 0
    batch_index = 0
    while gathered_count < max_samples:
        xb = batches[batch_index % len(batches)]
        fr_seq, v_seq = _rollout_noisy(
            net,
            xb,
            inject_space=inject_space,
            noise_scale=noise_scale,
            rng=rng,
        )
        obs = fr_seq if observable_space == "rate" else v_seq
        start = min(max(0, int(burn_in_steps)), obs.shape[0] - 1)
        if centering == "trajectory_mean":
            flat = _flatten_samples(obs[start:])
            centered = flat - np.mean(flat, axis=0, keepdims=True)
        else:
            ref = references[batch_index % len(batches)]
            centered = _flatten_samples((obs - ref)[start:])
        remaining = max_samples - gathered_count
        if centered.shape[0] > remaining:
            centered = centered[:remaining]
        gathered.append(centered)
        gathered_count += centered.shape[0]
        batch_index += 1
    X = np.concatenate(gathered, axis=0)
    return X, {
        "sample_count": int(X.shape[0]),
        "num_rollouts": int(batch_index),
        "burn_in_steps": int(burn_in_steps),
    }


def _accumulate_covariance(
    net: _ExtractedCTRNN,
    batches: Sequence[np.ndarray],
    *,
    observable_space: str,
    inject_space: str,
    centering: str,
    noise_scale: float,
    num_rollouts: int,
    burn_in_steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Empirical covariance at a fixed *rollout* budget.

    The sample-capped path (``_collect_centered_samples``) ties the number of
    rollouts to the retained window.  At the paper's burn-in the window is a
    single timestep, so the 25,000-sample cap costs 98 rollouts; a whole-trial
    window reaches the same cap in 3, and would see only 3 of the 20 frozen
    score batches.  Comparing windows under a sample cap therefore confounds
    *what* is measured with *how many* independent noise realisations went into
    it.  This path fixes the rollouts instead and lets the sample count follow,
    so window variants are compared at matched simulation cost.

    Covariance accumulates as a running sum of outer products, so peak memory is
    one rollout's trajectory regardless of the window.  The per-rollout centering
    convention matches ``_collect_centered_samples`` exactly: ``trajectory_mean``
    subtracts each rollout's own window mean, ``conditional`` subtracts the
    matched noise-free reference sample by sample.
    """
    if observable_space not in {"rate", "voltage"}:
        raise ValueError("observable_space must be 'rate' or 'voltage'.")
    if inject_space not in {"rate", "voltage"}:
        raise ValueError("inject_space must be 'rate' or 'voltage'.")
    if centering not in {"trajectory_mean", "conditional"}:
        raise ValueError("centering must be 'trajectory_mean' or 'conditional'.")
    if int(num_rollouts) < 1:
        raise ValueError("num_rollouts must be >= 1.")

    # The noise-free reference is recomputed per rollout rather than cached for
    # every batch: caching costs (n_batches x T x B x H) floats -- 839 MB at
    # T = 40, B = 256, H = 512 -- per worker, which is what drove twelve
    # concurrent workers into tens of GB of resident memory. A noise-free
    # rollout is ~0.7% of the cost of the noisy rollout plus its outer-product
    # accumulation, so recomputing is far cheaper than holding it.
    H = net.wrec.shape[0]
    scatter = np.zeros((H, H), dtype=np.float64)
    total = 0
    window_steps = 0
    for index in range(int(num_rollouts)):
        xb = batches[index % len(batches)]
        fr_seq, v_seq = _rollout_noisy(
            net,
            xb,
            inject_space=inject_space,
            noise_scale=noise_scale,
            rng=rng,
        )
        obs = fr_seq if observable_space == "rate" else v_seq
        start = min(max(0, int(burn_in_steps)), obs.shape[0] - 1)
        window_steps = obs.shape[0] - start
        if centering == "trajectory_mean":
            flat = _flatten_samples(obs[start:])
            centered = flat - np.mean(flat, axis=0, keepdims=True)
        else:
            fr_ref, v_ref = _rollout_noise_free(net, xb)
            ref = fr_ref if observable_space == "rate" else v_ref
            centered = _flatten_samples((obs - ref)[start:])
        scatter += centered.T @ centered
        total += centered.shape[0]
    if total < 2:
        raise ValueError("Need at least two samples to estimate covariance.")
    return scatter / float(total - 1), {
        "sample_count": int(total),
        "num_rollouts": int(num_rollouts),
        "burn_in_steps": int(burn_in_steps),
        "window_steps": int(window_steps),
    }


def injection_floor(
    net: _ExtractedCTRNN,
    *,
    observable_space: str,
    inject_space: str,
    noise_scale: float,
    trial_steps: int | None = None,
    window_start: int = 0,
) -> float:
    """Variance the injected noise contributes to the observable *directly*.

    The noise drawn at step ``t`` is independent of everything before it, so when
    it enters the observed variable without passing through ``W`` it contributes
    an exactly isotropic term to the covariance:

        C_observed = C_propagated + floor * I

    Only ``C_propagated`` has been through the recurrent weights, so only it
    carries connectivity information.  The floor is not noise in the estimator --
    it is a large, flat, perfectly-known constant, and because the score uses
    ``C`` multiplicatively (through ``C_ii + C_jj -/+ 2 C_ij``) a constant offset
    dilutes the informative variation rather than adding to it.  Measured on the
    frozen score batches it is 27-84% of the covariance diagonal, and on the
    low-sigma tasks that share is sigma-independent (0.797 -> 0.784 as sigma
    drops 20x): in the linear regime floor and propagated response both scale as
    sigma^2, so reducing the noise cannot improve the ratio.  Subtracting it can.

    Rate noise observed in rate contributes ``sigma^2``.  Voltage noise observed
    in voltage contributes ``alpha * sigma^2`` -- the Euler-Maruyama step adds
    ``sqrt(alpha) * sigma * xi``.  When the injection and observation spaces
    differ the noise reaches the observable only through ``W`` or through the
    activation, so there is no direct term and the floor is zero.
    """
    sig2 = float(noise_scale) ** 2
    if observable_space == inject_space == "rate":
        # Exact. The rate is rebuilt from v each step and the noise is added
        # after the activation, so fr(t) = f(v(t)) + sigma*xi(t) with xi(t)
        # independent of everything earlier. Verified against a W = 0 rollout:
        # measured 0.089995 against a predicted 0.090000.
        return sig2
    if observable_space == "voltage" and inject_space == "rate":
        # Exact, and zero: rate noise reaches the voltage only through W.
        # Verified against a W = 0 rollout: measured 0.000000.
        return 0.0
    if observable_space == inject_space == "voltage":
        # The Euler-Maruyama step adds sqrt(alpha)*sigma*xi to v, but the leak
        # also carries each neuron's *own* past noise forward without it passing
        # through W, so the floor is the leak-accumulated own-noise and not the
        # per-step injection. It is therefore window-dependent: at step t the
        # accumulated variance is alpha*sigma^2 (1 - lam^t)/(1 - lam) with
        # lam = (1-alpha)^2, and the retained window averages that over t.
        # Using the per-step value alpha*sigma^2 understates it 4.5x at T = 30.
        if trial_steps is None:
            raise ValueError(
                "voltage->voltage injection_floor is window-dependent; pass trial_steps."
            )
        alpha = float(net.alpha)
        lam = (1.0 - alpha) ** 2
        start = min(max(0, int(window_start)), int(trial_steps) - 1)
        t = np.arange(start + 1, int(trial_steps) + 1, dtype=np.float64)
        return float(alpha * sig2 / (1.0 - lam) * np.mean(1.0 - lam ** t))
    # voltage -> rate: the own-noise floor is passed through the activation, so
    # it is g_i^2-weighted rather than isotropic and there is no scalar for it.
    raise ValueError(
        "injection_floor is not available for inject_space='voltage' with "
        "observable_space='rate': the floor is per-neuron (gain-weighted), not isotropic."
    )


def subtract_injection_floor(C: np.ndarray, floor: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """Remove the direct-injection term from a covariance estimate.

    Only the diagonal is touched: the floor is isotropic by construction.  A
    diagonal entry can land slightly below zero from sampling error, which would
    make the matrix indefinite, so entries are clipped at zero and the number
    clipped is reported.
    """
    floor = float(floor)
    if floor <= 0.0:
        return C, {"injection_floor": 0.0, "injection_floor_share": 0.0,
                   "injection_floor_neg_mass": 0.0, "cov_eff_rank": 0.0}
    diag_before = float(np.mean(np.diag(C)))
    out = np.array(C, dtype=np.float64, copy=True)
    out[np.diag_indices_from(out)] -= floor

    # The propagated covariance is PSD in expectation, but it is close to
    # rank-deficient -- the whole point of the diagnosis is that most of the
    # observed variance was the isotropic injection -- so sampling error leaves
    # many eigenvalues slightly negative once the floor is removed. Project back
    # onto the PSD cone rather than relaxing the validator, and report how much
    # spectral mass was negative: a large value means the floor estimate and the
    # data disagree, not merely that the estimate is noisy.
    out = 0.5 * (out + out.T)
    evals, evecs = np.linalg.eigh(out)
    neg_mass = float(np.sum(np.abs(evals[evals < 0.0])) / max(np.sum(np.abs(evals)), 1e-300))
    clipped = np.maximum(evals, 0.0)
    out = (evecs * clipped) @ evecs.T
    out = 0.5 * (out + out.T)
    total = float(np.sum(clipped))
    eff_rank = float(total ** 2 / max(float(np.sum(clipped ** 2)), 1e-300)) if total > 0 else 0.0
    return out, {
        "injection_floor": floor,
        "injection_floor_share": floor / diag_before if diag_before > 0.0 else 0.0,
        "injection_floor_neg_mass": neg_mass,
        # participation ratio of the propagated covariance: how many dimensions
        # the network's noise response actually occupies, out of N.
        "cov_eff_rank": eff_rank,
    }


def _validate_empirical_covariance(C: np.ndarray, tol: float = 1e-8) -> None:
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Covariance must be square, got {C.shape}.")
    if not np.allclose(C, C.T, atol=tol, rtol=0.0):
        raise ValueError("Empirical covariance is not symmetric.")
    eigvals = np.linalg.eigvalsh(C)
    min_eig = float(np.min(eigvals))
    max_abs = float(np.max(np.abs(eigvals)))
    if min_eig < -1e-4 * max(1.0, max_abs):
        raise ValueError("Empirical covariance has significantly negative eigenvalues.")


def _waterfill_to_target(raw: np.ndarray, target: float) -> Tuple[np.ndarray, int]:
    """Scale ``raw`` into probabilities with ``sum(p) == target`` and ``p <= 1``.

    Plain proportional scaling followed by a clip silently loses the mass of
    every entry pushed above 1, so the expected retained density falls below the
    target -- and it falls *further* the more structured the score vector is,
    because structure means a heavier tail. That biases any comparison between
    covariance estimators directly against the more informative ones. Measured
    on dm1seqr at 60% sparsity, the published estimator lands 2.2% below its
    edge budget while the rate-inject/voltage-observe estimator lands 23.7%
    below it, i.e. is really pruned to 69.5% while nominally at 60%.

    Water-filling instead pins each over-1 entry at 1 and redistributes its
    excess over the entries still below 1, in proportion to their raw scores.
    The free set stays proportional to ``raw`` throughout (it is only ever
    multiplied by a common factor), and each pass pins at least one more entry,
    so the loop terminates. Mirrors ``score_controls._magnitude_proportional_probs``.

    Returns the probability vector and the number of entries pinned at 1.
    """
    w = np.asarray(raw, dtype=np.float64).ravel()
    total = float(w.sum())
    if target <= 0.0 or total <= 0.0:
        return np.zeros_like(w), 0
    if target >= float(w.size):
        return np.ones_like(w), int(w.size)

    out = w * (target / total)
    pinned = np.zeros(out.shape, dtype=bool)
    for _ in range(64):
        newly = (out > 1.0) & ~pinned
        if not newly.any():
            break
        pinned |= newly
        excess = float(out[pinned].sum() - float(pinned.sum()))
        out[pinned] = 1.0
        free = ~pinned & (w > 0.0)
        free_mass = float(w[free].sum())
        if excess <= 0.0 or not free.any() or free_mass <= 0.0:
            break
        out[free] += excess * w[free] / free_mass
    return np.clip(out, 0.0, 1.0), int(pinned.sum())


def empirical_noise_prune_scores(
    weight_matrix: np.ndarray,
    covariance: np.ndarray,
    *,
    sigma: float,
    eps: float,
    target_density: float,
    normalize: str = "clip",
) -> Tuple[np.ndarray, Dict[str, float]]:
    if normalize not in {"clip", "waterfill"}:
        raise ValueError("normalize must be 'clip' or 'waterfill'.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError("weight_matrix must be square.")
    if covariance.shape != weight_matrix.shape:
        raise ValueError(
            f"covariance shape {covariance.shape} does not match weight matrix {weight_matrix.shape}."
        )
    if not (0.0 <= target_density <= 1.0):
        raise ValueError("target_density must lie in [0, 1].")

    W = np.asarray(weight_matrix, dtype=np.float64)
    C = np.asarray(covariance, dtype=np.float64)
    _validate_empirical_covariance(C)

    N = W.shape[0]
    off_mask = ~np.eye(N, dtype=bool)
    rows, cols = np.where(off_mask)
    weights = W[rows, cols]
    nonzero_mask = weights != 0.0
    rows = rows[nonzero_mask]
    cols = cols[nonzero_mask]
    weights = weights[nonzero_mask]

    abs_weights = np.abs(weights)
    sign_weights = np.sign(weights)
    diagC = np.diag(C)
    diff_cov = diagC[rows] + diagC[cols] - sign_weights * 2.0 * C[rows, cols]
    diff_cov = np.maximum(diff_cov, 0.0)

    Kdeg = 4.0 * np.log(N) / (eps**2)
    K = 8.0 * np.log(N) / (eps**2 * sigma**2)
    raw_probs = K * abs_weights * diff_cov

    total_edges = float(N * (N - 1))
    num_available = raw_probs.size
    sum_raw = raw_probs.sum()
    scale_factor = 1.0
    desired_kept_total = target_density * total_edges
    pinned_at_one = 0
    if normalize == "waterfill":
        # Hold the expected retained density exactly instead of letting the clip
        # eat it; `scale_factor` is reported as the first-pass factor so it stays
        # comparable with the legacy path.
        if num_available > 0 and sum_raw > 0.0:
            scale_factor = 0.0 if desired_kept_total == 0.0 else desired_kept_total / sum_raw
            probs, pinned_at_one = _waterfill_to_target(raw_probs, desired_kept_total)
        else:
            probs = np.zeros_like(raw_probs)
        capped_probs = int(pinned_at_one)
    else:
        if num_available > 0 and sum_raw > 0.0:
            scale_factor = 0.0 if desired_kept_total == 0.0 else desired_kept_total / sum_raw
            raw_probs = raw_probs * scale_factor
        capped_probs = int(np.count_nonzero(raw_probs > 1.0))
        probs = np.clip(raw_probs, 0.0, 1.0)

    score_mat = np.zeros_like(W, dtype=np.float64)
    score_mat[rows, cols] = probs
    return score_mat, {
        "K": float(K),
        "Kdeg": float(Kdeg),
        "prob_normalize": normalize,
        "prob_target_sum": float(desired_kept_total),
        "prob_sum_shortfall": float(desired_kept_total - float(np.sum(probs))),
        "prob_pinned_at_one": int(pinned_at_one),
        "scale_factor": float(scale_factor),
        "capped_probs": int(capped_probs),
        "score_sum": float(np.sum(probs)),
        "score_max": float(np.max(probs)) if probs.size else 0.0,
        "score_mean": float(np.mean(probs)) if probs.size else 0.0,
        "positive_prob_count": int(np.count_nonzero(probs > 0.0)),
    }


def simulation_noise_prune_mask(
    model: torch.nn.Module,
    amount: float,
    *,
    batches: Sequence[Batch],
    sigma: Optional[float] = None,
    sigma_source: str = "natural_voltage",
    sigma_factor: float = 1.0,
    eps: float = 0.3,
    observable_space: str = "rate",
    inject_space: str = "rate",
    centering: str = "trajectory_mean",
    max_samples: int = 25_000,
    num_rollouts: Optional[int] = None,
    burn_in_steps: int = 300,
    rng_seed: Optional[int] = None,
    prob_normalize: str = "clip",
    subtract_floor: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    net = _extract_ctrnn(model)
    batch_arrays = _as_numpy_batches(batches)
    natural = _natural_variability_stats(net, batch_arrays, burn_in_steps=int(burn_in_steps))
    sigma_used = _resolve_noise_scale(natural, sigma=sigma, sigma_source=sigma_source, sigma_factor=sigma_factor)
    rng = np.random.default_rng(rng_seed)
    if num_rollouts is None:
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
    else:
        C_emp, sim_stats = _accumulate_covariance(
            net,
            batch_arrays,
            observable_space=observable_space,
            inject_space=inject_space,
            centering=centering,
            noise_scale=sigma_used,
            num_rollouts=int(num_rollouts),
            burn_in_steps=int(burn_in_steps),
            rng=rng,
        )
    floor_stats = {"injection_floor": 0.0, "injection_floor_share": 0.0,
                   "injection_floor_clipped": 0}
    if subtract_floor:
        C_emp, floor_stats = subtract_injection_floor(
            C_emp,
            injection_floor(net, observable_space=observable_space,
                            inject_space=inject_space, noise_scale=float(sigma_used),
                            trial_steps=int(batch_arrays[0].shape[0]),
                            window_start=min(int(burn_in_steps),
                                             int(batch_arrays[0].shape[0]) - 1)),
        )
    empirical_mean_gain = np.asarray(json.loads(natural["empirical_mean_gain_json"]), dtype=np.float64)
    score_weights = net.wrec
    if observable_space == "voltage":
        # Voltage-space scoring should use the effective operator magnitude
        # |W_ij g_j|, with g_j averaged along the underlying noise-free trajectory.
        score_weights = net.wrec * empirical_mean_gain[np.newaxis, :]
    score_mat, score_stats = empirical_noise_prune_scores(
        score_weights,
        C_emp,
        sigma=float(sigma_used),
        eps=float(eps),
        target_density=float(1.0 - amount),
        normalize=str(prob_normalize),
    )
    # score_mat already contains clipped non-negative probabilities. The
    # deterministic top-k is unaffected by which normaliser produced them: both
    # are monotone in the raw score below 1, and the entries pinned at 1 (at
    # most 39k here) never exceed the keep budget at 50-70% sparsity, so the
    # ties they create never reach the cut.
    scored = score_mat
    if net.no_self_connections:
        np.fill_diagonal(scored, 0.0)
    keep_mask = _exact_offdiag_keep_mask_from_scores(
        scored,
        amount=float(amount),
        no_self_connections=net.no_self_connections,
    )
    stats: Dict[str, float] = {}
    stats.update(natural)
    stats.update(sim_stats)
    stats.update(score_stats)
    stats.update({
        "variant": "simulation_empirical_cov_mask_only",
        "observable_space": observable_space,
        "inject_space": inject_space,
        "centering": centering,
        "sigma_source": sigma_source if sigma is None else "manual",
        "sigma_used": float(sigma_used),
        "amount": float(amount),
        "target_density": float(1.0 - amount),
        "burn_in_steps": int(burn_in_steps),
        "num_rollouts_requested": int(num_rollouts) if num_rollouts is not None else 0,
        "cov_budget_mode": "rollouts" if num_rollouts is not None else "samples",
        "prob_normalize_mode": str(prob_normalize),
        **floor_stats,
        "candidate_rec_abs_mean": float(np.mean(np.abs(net.wrec * keep_mask))),
        "score_rec_abs_mean": float(np.mean(scored)),
        "empirical_cov_trace": float(np.trace(C_emp)),
        "empirical_cov_diag_mean": float(np.mean(np.diag(C_emp))),
        "u_shape_json": json.dumps([list(x.shape) for x in batch_arrays]),
    })
    return keep_mask, stats


def simulation_noise_prune_recurrent(
    model: torch.nn.Module,
    amount: float,
    *,
    batches: Sequence[Batch],
    sigma: Optional[float] = None,
    sigma_source: str = "natural_voltage",
    sigma_factor: float = 1.0,
    eps: float = 0.3,
    observable_space: str = "rate",
    inject_space: str = "rate",
    centering: str = "trajectory_mean",
    max_samples: int = 25_000,
    num_rollouts: Optional[int] = None,
    burn_in_steps: int = 300,
    rng_seed: Optional[int] = None,
    prob_normalize: str = "clip",
    subtract_floor: bool = False,
    include_feedforward: bool = False,
) -> Dict[str, float]:
    if include_feedforward:
        raise NotImplementedError("simulation_noise_prune currently supports recurrent pruning only.")
    keep_mask, stats = simulation_noise_prune_mask(
        model,
        amount,
        batches=batches,
        sigma=sigma,
        sigma_source=sigma_source,
        sigma_factor=sigma_factor,
        eps=eps,
        observable_space=observable_space,
        inject_space=inject_space,
        centering=centering,
        max_samples=max_samples,
        num_rollouts=num_rollouts,
        burn_in_steps=burn_in_steps,
        rng_seed=rng_seed,
        prob_normalize=prob_normalize,
        subtract_floor=subtract_floor,
    )
    mask_stats = _apply_keep_mask_to_model(model, keep_mask)
    stats.update(mask_stats)
    return stats


def empirical_mask_overlap_convergence(
    model: torch.nn.Module,
    amount: float,
    *,
    batches: Sequence[Batch],
    M_values: Sequence[int] = (5_000, 10_000, 25_000, 50_000, 100_000),
    sigma: Optional[float] = None,
    sigma_source: str = "natural_voltage",
    eps: float = 0.3,
    observable_space: str = "rate",
    inject_space: str = "rate",
    centering: str = "trajectory_mean",
    burn_in_steps: int = 300,
    seed_a: int = 0,
    seed_b: int = 1,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for M in M_values:
        mask_a, stats_a = simulation_noise_prune_mask(
            model,
            amount,
            batches=batches,
            sigma=sigma,
            sigma_source=sigma_source,
            eps=eps,
            observable_space=observable_space,
            inject_space=inject_space,
            centering=centering,
            max_samples=int(M),
            burn_in_steps=int(burn_in_steps),
            rng_seed=int(seed_a),
        )
        mask_b, stats_b = simulation_noise_prune_mask(
            model,
            amount,
            batches=batches,
            sigma=sigma,
            sigma_source=sigma_source,
            eps=eps,
            observable_space=observable_space,
            inject_space=inject_space,
            centering=centering,
            max_samples=int(M),
            burn_in_steps=int(burn_in_steps),
            rng_seed=int(seed_b),
        )
        kept_a = int(np.count_nonzero(mask_a))
        intersect = int(np.count_nonzero(np.logical_and(mask_a, mask_b)))
        overlap = float(intersect / kept_a) if kept_a > 0 else 0.0
        results.append({
            "M": int(M),
            "mask_overlap": overlap,
            "kept_edges": kept_a,
            "sigma_used_a": float(stats_a["sigma_used"]),
            "sigma_used_b": float(stats_b["sigma_used"]),
            "sample_count_a": int(stats_a["sample_count"]),
            "sample_count_b": int(stats_b["sample_count"]),
        })
    return results


class SimulationNoisePruneStrategy(BasePruner):
    name = "simulation_noise_prune_mask_only"
    aliases = ("simulation_noise_prune", "sim_noise_prune", "empirical_cov_noise_prune")
    description = "Mask-only noise-prune using empirical covariance from noisy nonlinear CTRNN rollouts."
    requires_batches = True
    default_batch_count = 20

    def apply(
        self,
        context: PruneContext,
        state: Mapping[str, object],
        **kwargs,
    ) -> Mapping[str, float]:
        if not context.batches:
            raise ValueError("simulation_noise_prune requires sampled batches.")
        return simulation_noise_prune_recurrent(
            context.model,
            context.amount,
            batches=context.batches,
            sigma=kwargs.get("sigma"),
            sigma_source=str(kwargs.get("sigma_source", "natural_voltage")),
            sigma_factor=float(kwargs.get("sigma_factor", 1.0)),
            eps=float(kwargs.get("eps", 0.3)),
            observable_space=str(kwargs.get("observable_space", "rate")),
            inject_space=str(kwargs.get("inject_space", "rate")),
            centering=str(kwargs.get("centering", "trajectory_mean")),
            max_samples=int(kwargs.get("max_samples", 25_000)),
            num_rollouts=(None if kwargs.get("num_rollouts") is None
                          else int(kwargs["num_rollouts"])),
            burn_in_steps=int(kwargs.get("burn_in_steps", 300)),
            rng_seed=kwargs.get("rng_seed"),
            prob_normalize=str(kwargs.get("prob_normalize", "clip")),
            subtract_floor=bool(kwargs.get("subtract_floor", False)),
            include_feedforward=context.prune_feedforward,
        )


def linearized_identity_covariance_sanity(
    model: torch.nn.Module,
    *,
    batches: Sequence[Batch],
    sigma: float,
    max_samples: int = 25_000,
    burn_in_steps: int = 300,
    rng_seed: int = 0,
) -> Dict[str, float]:
    """Compare empirical identity-dynamics covariance to analytic Lyapunov covariance.

    This sanity check replaces ``tanh`` with the identity map in simulation,
    uses voltage-space noise and conditional centering, and compares the
    empirical covariance of ``v`` against ``solve_continuous_lyapunov(A, -σ²I)``
    with ``A = W - I``.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    net = _extract_ctrnn(model)
    batch_arrays = _as_numpy_batches(batches)
    rng = np.random.default_rng(int(rng_seed))

    gathered: List[np.ndarray] = []
    gathered_count = 0
    rollout_count = 0
    while gathered_count < int(max_samples):
        xb = batch_arrays[rollout_count % len(batch_arrays)]
        v_ref = _rollout_noise_free_identity(net, xb)
        v_noisy = _rollout_noisy_identity(net, xb, noise_scale=float(sigma), rng=rng)
        start = min(max(0, int(burn_in_steps)), v_noisy.shape[0] - 1)
        centered = _flatten_samples((v_noisy - v_ref)[start:])
        remaining = int(max_samples) - gathered_count
        if centered.shape[0] > remaining:
            centered = centered[:remaining]
        gathered.append(centered)
        gathered_count += centered.shape[0]
        rollout_count += 1
    X = np.concatenate(gathered, axis=0)
    C_emp = _empirical_covariance(X, center_assumed=True)

    A = net.wrec - np.eye(net.wrec.shape[0], dtype=np.float64)
    C_analytic = solve_continuous_lyapunov(A, -(float(sigma) ** 2) * np.eye(A.shape[0], dtype=np.float64))
    C_analytic = 0.5 * (C_analytic + C_analytic.T)

    diff = C_emp - C_analytic
    rel_fro = float(np.linalg.norm(diff, ord="fro") / max(1e-12, np.linalg.norm(C_analytic, ord="fro")))
    diag_emp = np.diag(C_emp)
    diag_analytic = np.diag(C_analytic)
    diag_corr = float(np.corrcoef(diag_emp, diag_analytic)[0, 1]) if diag_emp.size > 1 else 1.0
    trace_ratio = float(np.trace(C_emp) / max(1e-12, np.trace(C_analytic)))
    return {
        "sigma": float(sigma),
        "sample_count": int(X.shape[0]),
        "burn_in_steps": int(burn_in_steps),
        "alpha": float(net.alpha),
        "empirical_cov_trace": float(np.trace(C_emp)),
        "analytic_cov_trace": float(np.trace(C_analytic)),
        "trace_ratio_empirical_to_analytic": trace_ratio,
        "cov_rel_fro_error": rel_fro,
        "cov_diag_corr": diag_corr,
    }


__all__ = [
    "SimulationNoisePruneStrategy",
    "empirical_mask_overlap_convergence",
    "empirical_noise_prune_scores",
    "linearized_identity_covariance_sanity",
    "simulation_noise_prune_mask",
    "simulation_noise_prune_recurrent",
]
