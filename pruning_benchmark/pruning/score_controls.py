"""Uninformative-score and gain-matched controls for the noise-prune ablation.

These controls exist to separate two factors that the main task-preservation
suite varies together:

* whether the retention probabilities carry covariance information, and
* whether edges are selected stochastically and survivors are rescaled.

``apply_probability_control`` destroys the covariance information while leaving
every other part of the sample-and-rescale pipeline untouched:

``shuffle``
    Permutes the probability vector across candidate edges.  The multiset of
    probabilities is preserved exactly, so the sum (hence the expected number
    of retained edges) and the amplification distribution ``{1 / p_ij}`` are
    identical to the informative run.  Only the score-to-edge assignment is
    destroyed.

``uniform``
    Replaces every probability by the mean of the real probability vector.
    The sum is again preserved exactly, so the expected retained density
    matches the informative run, and every survivor receives the same
    amplification ``1 / mean(p)``.  This is "stochastic mask plus constant gain
    restoration" with no per-edge information at all.

``magnitude``
    Replaces the probabilities by ones proportional to ``|w_ij|`` alone, i.e.
    noise-prune with the diff-covariance factor dropped from
    ``p_ij = K |w_ij| (C_ii + C_jj -/+ 2 C_ij)``.  The sum is again preserved
    exactly (by water-filling, so the ``p <= 1`` constraint holds without
    changing the expected retained density), which makes this the "is the
    covariance term doing any work?" control: everything about the pipeline is
    kept, and only the covariance factor is removed.  Note the algebraic
    consequence -- with ``p_ij proportional to |w_ij|`` the rescaled survivor
    ``w_ij / p_ij`` has constant magnitude, so unclipped survivors all carry the
    same weight and differ only in sign.  That degeneracy is a property of the
    control, not a bug; ties at the exact-sparsity threshold are broken at
    random by ``_weight_scores_to_mask``.

``restore_uniform_gain`` applies the matching gain correction to a
deterministically masked network (magnitude or random) so that every method is
compared at matched expected recurrent gain.  Three conventions are supported:
``inv_density`` (multiply survivors by ``1 / retained density``) and ``match_l1``
(rescale so the retained absolute weight sum equals the unpruned value).  The two
coincide for an unbiased mask such as random pruning, but not for magnitude
pruning, whose survivors are the large weights.

Neither restores gain for magnitude pruning, and the failure is large enough to
matter.  Magnitude pruning removes the small weights, which carry most of the
count but little of the spectrum, so it loses far more L1 mass than recurrent
gain.  Measured over the eight seed-0 networks, at 70% sparsity magnitude pruning
retains 0.57 of the off-diagonal L1 but 0.84 of ``rho(W_rec)`` (at 80%: 0.42 and
0.74).  ``inv_density`` (x3.3 at 70%, x5.0 at 80%) and ``match_l1`` (x1.8 and
x2.4) therefore do not restore gain: they inflate it several-fold past the
unpruned value and drive the network into saturation.  ``match_l1_rowwise`` and
``match_l2`` are the local-homeostatic and mean-field readings of the same idea.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils import prune


PROBABILITY_CONTROLS = ("none", "shuffle", "uniform", "magnitude")


def _magnitude_proportional_probs(
    probs: np.ndarray, magnitudes: Optional[np.ndarray]
) -> Tuple[np.ndarray, int]:
    """Probabilities proportional to ``|w|`` with ``sum(p)`` and ``p <= 1`` both held.

    Plain proportional scaling can push the largest-magnitude edges above 1,
    where clipping would silently lower ``sum(p)`` and so lower the expected
    retained density -- which is exactly the quantity every control is supposed
    to hold fixed.  Water-filling instead pins each over-1 entry at 1 and
    redistributes its excess over the entries still below 1, in proportion to
    their magnitudes.  The free set stays proportional to ``|w|`` throughout (it
    is only ever multiplied by a common factor), and each pass pins at least one
    more entry, so the loop terminates.

    Returns the probability vector and the number of entries pinned at 1.
    """
    if magnitudes is None:
        raise ValueError("prob_control='magnitude' requires the candidate-edge magnitudes.")
    w = np.abs(np.asarray(magnitudes, dtype=np.float64)).ravel()
    if w.shape != probs.shape:
        raise ValueError(
            f"magnitudes must match the probability vector: {w.shape} vs {probs.shape}."
        )

    target = float(probs.sum())
    total = float(w.sum())
    if target <= 0.0 or total <= 0.0:
        return np.zeros_like(probs), 0

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


def apply_probability_control(
    probs: np.ndarray,
    *,
    control: str = "none",
    seed: Optional[int] = None,
    magnitudes: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return a control version of ``probs`` plus audit statistics.

    Parameters
    ----------
    probs:
        One-dimensional array of retention probabilities for the candidate
        edges, already density-normalized and clipped to ``[0, 1]``.
    control:
        One of ``"none"``, ``"shuffle"``, ``"uniform"`` or ``"magnitude"``.
    seed:
        Seed for the permutation generator.  A dedicated generator is used so
        that the Bernoulli draw downstream consumes the same random stream as
        the corresponding informative run.

    Notes
    -----
    magnitudes:
        Absolute weights of the same candidate edges, in the same order.
        Required by ``control="magnitude"`` and ignored otherwise.

    Notes
    -----
    All three controls preserve ``probs.sum()`` exactly, so the expected number
    of retained edges is unchanged.  ``uniform`` additionally collapses the
    amplification distribution to a single value; ``magnitude`` collapses the
    *rescaled survivor weights* to a single magnitude instead.
    """
    control = str(control).lower()
    if control not in PROBABILITY_CONTROLS:
        raise ValueError(
            f"prob_control must be one of {PROBABILITY_CONTROLS}, got {control!r}."
        )

    flat = np.asarray(probs, dtype=np.float64)
    if flat.ndim != 1:
        raise ValueError("apply_probability_control expects a 1-D probability vector.")

    magnitude_pinned = 0
    sum_before = float(flat.sum())
    stats: Dict[str, float] = {
        "prob_control": control,
        "prob_control_seed": -1 if seed is None else int(seed),
        "prob_control_prob_sum_before": sum_before,
        "prob_control_prob_mean_before": float(flat.mean()) if flat.size else 0.0,
    }

    if control == "none" or flat.size == 0:
        controlled = flat
    elif control == "shuffle":
        control_rng = np.random.default_rng(0 if seed is None else int(seed))
        controlled = control_rng.permutation(flat)
    elif control == "uniform":
        controlled = np.full_like(flat, float(flat.mean()))
    else:  # magnitude
        controlled, magnitude_pinned = _magnitude_proportional_probs(flat, magnitudes)

    stats.update({
        "prob_control_prob_sum_after": float(controlled.sum()),
        "prob_control_prob_mean_after": float(controlled.mean()) if controlled.size else 0.0,
        "prob_control_uniform_value": (
            float(controlled[0]) if control == "uniform" and controlled.size else 0.0
        ),
        "prob_control_uniform_amp": (
            float(1.0 / controlled[0])
            if control == "uniform" and controlled.size and controlled[0] > 0.0
            else 0.0
        ),
        "prob_control_pinned_at_one": float(magnitude_pinned),
        "prob_control_pinned_frac": (
            float(magnitude_pinned) / float(controlled.size) if controlled.size else 0.0
        ),
    })
    return controlled, stats


GAIN_MODES = (
    "inv_density", "match_l1", "match_l1_rowwise", "match_l2",
    "match_rho_jlin", "fixed",
)


def _jlin_rho(eigvals: np.ndarray, alpha: float, gain: float) -> float:
    """rho((1-alpha) I + alpha * gain * W) from the eigenvalues of W.

    The eigenvalues of ``(1-a) I + a g W`` are ``(1-a) + a g lambda_i``, so the
    whole gain response is available from a single eigendecomposition of the
    pruned matrix rather than one per candidate gain.
    """
    return float(np.max(np.abs((1.0 - alpha) + alpha * gain * eigvals)))


def solve_jlin_gain(
    pruned: torch.Tensor, target_rho: float, alpha: float, *, hi: float = 50.0
) -> float:
    """Scalar ``g`` making ``rho((1-a)I + a g W_pruned)`` equal ``target_rho``.

    ``rho`` is a max of moduli of functions affine in ``g``, hence convex and,
    beyond the point where the largest-modulus eigenvalue dominates, increasing.
    Bisection on that branch is therefore well posed.  Returns 1.0 when the
    target is already met or cannot be reached within ``hi``.
    """
    ev = np.linalg.eigvals(pruned.detach().to(torch.float64).cpu().numpy())
    if _jlin_rho(ev, alpha, hi) < target_rho:
        return 1.0
    lo = 1.0 if _jlin_rho(ev, alpha, 1.0) <= target_rho else 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _jlin_rho(ev, alpha, mid) < target_rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spectral_radius(weight: torch.Tensor) -> float:
    """Largest eigenvalue modulus of a square weight matrix.

    Computed in float64 on the CPU regardless of the model dtype: the pruned
    recurrent matrices are ill-conditioned enough that a float32 eigensolve
    moves the third decimal place, and this value sets a multiplicative gain
    applied to every surviving weight.
    """
    w = weight.detach().to(torch.float64).cpu().numpy()
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("spectral_radius expects a square matrix.")
    return float(np.max(np.abs(np.linalg.eigvals(w))))


@torch.no_grad()
def restore_uniform_gain(
    model: torch.nn.Module,
    *,
    gain_mode: str = "inv_density",
    reference_offdiag_l1: Optional[float] = None,
    reference_spectral_radius: Optional[float] = None,
    gain_value: Optional[float] = None,
    reference_offdiag_l2: Optional[float] = None,
    reference_rho_jlin: Optional[float] = None,
    alpha: Optional[float] = None,
    reference_rowwise_l1: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Rescale surviving recurrent weights to restore lost recurrent gain.

    Intended to be called immediately after a deterministic masking pruner
    (magnitude or random) so the pruned network is compared at matched
    expected recurrent gain.  The existing mask is preserved; only the
    surviving weight magnitudes change.

    ``inv_density``
        Multiply survivors by ``1 / retained off-diagonal density``.  This is
        the literal correction for an unbiased mask and restores the expected
        weight exactly for random pruning.  For magnitude pruning the mask is
        *not* unbiased -- survivors are the large weights -- so this
        systematically overshoots the original recurrent gain.

    ``match_l1``
        Multiply survivors so the retained off-diagonal absolute weight sum
        equals that of the unpruned matrix.  This restores total *synaptic*
        mass for any mask, biased or not.  Requires ``reference_offdiag_l1``.

    ``match_l1_rowwise``
        Rescale each row independently so that neuron's total absolute input
        weight is restored -- i.e. synaptic scaling, the homeostatic rule in
        which a neuron normalises its incoming drive.  Unlike every other mode
        here this is a *local* rule: the factor for row ``i`` depends only on
        the weights that neuron ``i`` receives, so it needs no global spectral
        quantity and has a mechanistic interpretation.  That makes it the
        natural biological baseline, and the one a reviewer invoking
        homeostatic plasticity would have in mind.  Returns a per-row vector
        rather than a scalar; ``gain_restore_factor`` reports its mean.

    ``match_l2``
        Multiply survivors so the retained sum of squared off-diagonal weights
        equals the unpruned value.  In mean-field RNN theory the gain parameter
        is defined through the second moment (``g^2 = N Var(w)``), so this is
        the mean-field reading of "matched gain".  Requires
        ``reference_offdiag_l2``.

    ``match_rho_jlin``
        Multiply survivors so ``rho((1-alpha) I + alpha W)`` equals the unpruned
        value -- the spectral radius of the network's actual discrete-time
        state-transition operator rather than of the weight matrix alone.  For a
        CTRNN this is the operative notion of recurrent gain: it is the operator
        that iterates, and it is already the quantity the paper reports as
        ``post_rec_linear_rho``.  Because ``alpha = dt/tau = 0.1`` here, the
        identity term dominates and the required factor is small (x1.02 at 50%
        sparsity to x1.08 at 80%), far below the x1.05-x1.36 that matching
        ``rho(W)`` demands.  Requires ``reference_rho_jlin`` and ``alpha``.

    ``fixed``
        Multiply survivors by an externally supplied ``gain_value``.  This is
        not a matched-gain convention; it exists so a gain *sweep* can be run
        through the same code path as the three conventions, which answers a
        question none of them can: whether matching a particular quantity is
        what helps, or whether performance simply increases with gain over the
        whole range.  ``gain_value=1.0`` is also the identity control -- it must
        reproduce the corresponding un-rescaled baseline exactly, which proves
        the only difference between the arms is the scalar.
    """
    from .strategies import _consolidate_if_pruned, enforce_constraints

    gain_mode = str(gain_mode).lower()
    if gain_mode not in GAIN_MODES:
        raise ValueError(f"gain_mode must be one of {GAIN_MODES}, got {gain_mode!r}.")

    layer = getattr(model, "hidden_layer", None)
    if layer is None:
        return {}

    mask = getattr(layer, "weight_mask", None)
    if mask is None:
        raise ValueError("restore_uniform_gain requires an already-pruned hidden layer.")
    mask = mask.detach().clone()

    H = mask.shape[0]
    offdiag = ~torch.eye(H, dtype=torch.bool, device=mask.device)
    total_offdiag = float(offdiag.sum().item())
    kept_offdiag = float((mask.to(torch.bool) & offdiag).sum().item())
    density = kept_offdiag / total_offdiag if total_offdiag > 0 else 0.0

    _consolidate_if_pruned(layer)
    pruned_l1 = float(layer.weight.data[offdiag].abs().sum().item())
    pruned_rho = spectral_radius(layer.weight.data)

    if gain_mode == "inv_density":
        gain = 1.0 / density if density > 0.0 else 1.0
    elif gain_mode == "match_l1":
        if reference_offdiag_l1 is None:
            raise ValueError("gain_mode='match_l1' requires reference_offdiag_l1.")
        gain = (float(reference_offdiag_l1) / pruned_l1) if pruned_l1 > 0.0 else 1.0
    elif gain_mode == "match_l1_rowwise":
        w = layer.weight.data
        row_ref = (reference_rowwise_l1 if reference_rowwise_l1 is not None
                   else None)
        if row_ref is None:
            raise ValueError("gain_mode='match_l1_rowwise' requires reference_rowwise_l1.")
        row_now = (w * offdiag).abs().sum(dim=1)
        row_gain = torch.where(row_now > 0, row_ref / row_now.clamp_min(1e-30),
                               torch.ones_like(row_now))
        w.mul_(row_gain.unsqueeze(1))
        prune.custom_from_mask(layer, name="weight", mask=mask)
        enforce_constraints(model)
        restored = layer.weight_orig.data * layer.weight_mask.data
        return {
            "gain_restored": True,
            "gain_restore_mode": gain_mode,
            "gain_restore_offdiag_density": float(density),
            "gain_restore_factor": float(row_gain.mean().item()),
            "gain_restore_rowwise_gain_min": float(row_gain.min().item()),
            "gain_restore_rowwise_gain_max": float(row_gain.max().item()),
            "gain_restore_kept_offdiag": float(kept_offdiag),
            "gain_restore_offdiag_l1_before": pruned_l1,
            "gain_restore_offdiag_l1_after": float(restored[offdiag].abs().sum().item()),
            "gain_restore_offdiag_l1_ratio": (
                float(restored[offdiag].abs().sum().item()) / float(reference_offdiag_l1)
                if reference_offdiag_l1 not in (None, 0.0) else 0.0),
            "gain_restore_spectral_radius_before": pruned_rho,
            "gain_restore_spectral_radius_after": spectral_radius(restored),
            "gain_restore_spectral_radius_reference": float(reference_spectral_radius or 0.0),
            "gain_restore_spectral_radius_ratio": (
                spectral_radius(restored) / float(reference_spectral_radius)
                if reference_spectral_radius not in (None, 0.0) else 0.0),
        }
    elif gain_mode == "match_l2":
        if reference_offdiag_l2 is None:
            raise ValueError("gain_mode='match_l2' requires reference_offdiag_l2.")
        pruned_l2 = float((layer.weight.data[offdiag] ** 2).sum().item())
        gain = ((float(reference_offdiag_l2) / pruned_l2) ** 0.5) if pruned_l2 > 0.0 else 1.0
    elif gain_mode == "match_rho_jlin":
        if reference_rho_jlin is None or alpha is None:
            raise ValueError(
                "gain_mode='match_rho_jlin' requires reference_rho_jlin and alpha."
            )
        gain = solve_jlin_gain(layer.weight.data, float(reference_rho_jlin), float(alpha))
    else:
        if gain_value is None:
            raise ValueError("gain_mode='fixed' requires gain_value.")
        gain = float(gain_value)
        if gain <= 0.0:
            raise ValueError("gain_value must be positive.")

    layer.weight.data.mul_(gain)
    prune.custom_from_mask(layer, name="weight", mask=mask)
    enforce_constraints(model)

    restored = layer.weight_orig.data * layer.weight_mask.data
    restored_l1 = float(restored[offdiag].abs().sum().item())
    restored_rho = spectral_radius(restored)
    return {
        "gain_restored": True,
        "gain_restore_mode": gain_mode,
        "gain_restore_offdiag_density": float(density),
        "gain_restore_factor": float(gain),
        "gain_restore_kept_offdiag": float(kept_offdiag),
        "gain_restore_offdiag_l1_before": pruned_l1,
        "gain_restore_offdiag_l1_after": restored_l1,
        "gain_restore_offdiag_l1_ratio": (
            restored_l1 / float(reference_offdiag_l1)
            if reference_offdiag_l1 not in (None, 0.0)
            else 0.0
        ),
        "gain_restore_spectral_radius_reference": float(reference_spectral_radius or 0.0),
        "gain_restore_spectral_radius_before": pruned_rho,
        "gain_restore_spectral_radius_after": restored_rho,
        "gain_restore_alpha": float(alpha) if alpha is not None else 0.0,
        "gain_restore_rho_jlin_reference": float(reference_rho_jlin or 0.0),
        "gain_restore_rho_jlin_after": (
            _jlin_rho(np.linalg.eigvals(restored.detach().to(torch.float64).cpu().numpy()),
                      float(alpha), 1.0)
            if alpha is not None else 0.0
        ),
        "gain_restore_offdiag_l2_ratio": (
            float((restored[offdiag] ** 2).sum().item()) / float(reference_offdiag_l2)
            if reference_offdiag_l2 not in (None, 0.0) else 0.0
        ),
        "gain_restore_spectral_radius_ratio": (
            restored_rho / float(reference_spectral_radius)
            if reference_spectral_radius not in (None, 0.0)
            else 0.0
        ),
    }


__all__ = [
    "GAIN_MODES",
    "PROBABILITY_CONTROLS",
    "apply_probability_control",
    "restore_uniform_gain",
    "spectral_radius",
]
