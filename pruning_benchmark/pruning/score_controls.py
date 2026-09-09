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

``restore_uniform_gain`` applies the matching gain correction to a
deterministically masked network (magnitude or random) so that every method is
compared at matched expected recurrent gain.  Two conventions are supported:
``inv_density`` (multiply survivors by ``1 / retained density``) and
``match_l1`` (rescale so the retained absolute weight sum equals the unpruned
value).  They coincide for an unbiased mask such as random pruning, but not for
magnitude pruning, whose survivors are the large weights.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils import prune


PROBABILITY_CONTROLS = ("none", "shuffle", "uniform")


def apply_probability_control(
    probs: np.ndarray,
    *,
    control: str = "none",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return a control version of ``probs`` plus audit statistics.

    Parameters
    ----------
    probs:
        One-dimensional array of retention probabilities for the candidate
        edges, already density-normalized and clipped to ``[0, 1]``.
    control:
        One of ``"none"``, ``"shuffle"`` or ``"uniform"``.
    seed:
        Seed for the permutation generator.  A dedicated generator is used so
        that the Bernoulli draw downstream consumes the same random stream as
        the corresponding informative run.

    Notes
    -----
    Both controls preserve ``probs.sum()`` exactly, so the expected number of
    retained edges is unchanged.  ``uniform`` additionally collapses the
    amplification distribution to a single value.
    """
    control = str(control).lower()
    if control not in PROBABILITY_CONTROLS:
        raise ValueError(
            f"prob_control must be one of {PROBABILITY_CONTROLS}, got {control!r}."
        )

    flat = np.asarray(probs, dtype=np.float64)
    if flat.ndim != 1:
        raise ValueError("apply_probability_control expects a 1-D probability vector.")

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
    else:  # uniform
        controlled = np.full_like(flat, float(flat.mean()))

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
    })
    return controlled, stats


GAIN_MODES = ("inv_density", "match_l1")


@torch.no_grad()
def restore_uniform_gain(
    model: torch.nn.Module,
    *,
    gain_mode: str = "inv_density",
    reference_offdiag_l1: Optional[float] = None,
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
        equals that of the unpruned matrix.  This restores total recurrent
        gain for any mask, biased or not, and is the correct matched-gain
        control for magnitude pruning.  Requires ``reference_offdiag_l1``.
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

    if gain_mode == "inv_density":
        gain = 1.0 / density if density > 0.0 else 1.0
    else:
        if reference_offdiag_l1 is None:
            raise ValueError("gain_mode='match_l1' requires reference_offdiag_l1.")
        gain = (float(reference_offdiag_l1) / pruned_l1) if pruned_l1 > 0.0 else 1.0

    layer.weight.data.mul_(gain)
    prune.custom_from_mask(layer, name="weight", mask=mask)
    enforce_constraints(model)

    restored_l1 = float(
        (layer.weight_orig.data * layer.weight_mask.data)[offdiag].abs().sum().item()
    )
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
    }


__all__ = [
    "GAIN_MODES",
    "PROBABILITY_CONTROLS",
    "apply_probability_control",
    "restore_uniform_gain",
]
