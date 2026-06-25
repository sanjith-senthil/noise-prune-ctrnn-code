"""Covariance-guided noise pruning for continuous-time recurrent operators.

This module implements the "noise probe" synapse pruning rule described by
Professor [Mentor] (private communication).  Given a continuous-time operator
``A`` (typically ``-D + W``) that is Hurwitz, we solve the continuous-time
Lyapunov equation to obtain the stationary covariance and then sample a sparse
operator whose expectation matches the original off-diagonal entries.  The
sampling probabilities depend on the diff-covariance term derived from the
stationary covariance.

The main entry point is :func:`noise_prune`.  The implementation uses only
``numpy`` and ``scipy`` (specifically ``scipy.linalg.solve_continuous_lyapunov``).

Complexity
----------
The Lyapunov solve is cubic in ``N`` for dense matrices (``O(N^3)``) and
generally dominates the runtime.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import LinAlgError, solve_continuous_lyapunov


def _validate_covariance(C: np.ndarray, tol: float = 1e-8) -> None:
    """Validate that ``C`` is (numerically) symmetric positive semidefinite."""
    if not np.allclose(C, C.T, atol=tol, rtol=0.0):
        raise ValueError(
            "Covariance from Lyapunov solve is not symmetric; "
            "ensure A is Hurwitz and well-conditioned."
        )
    eigvals = np.linalg.eigvalsh(C)
    min_eig = eigvals.min()
    max_abs = np.max(np.abs(eigvals))
    if min_eig < -1e3 * tol * max(1.0, max_abs):
        raise ValueError(
            "Covariance from Lyapunov solve has significantly negative eigenvalues; "
            "ensure A is Hurwitz or better conditioned."
        )


def _ensure_square_matrix(A: ArrayLike) -> np.ndarray:
    """Convert to 2-D numpy array and ensure it is square."""
    mat = np.asarray(A)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix A must be a square 2-D array.")
    if not np.issubdtype(mat.dtype, np.number):
        raise ValueError("Input matrix A must contain numeric entries.")
    return mat


def noise_prune(
    A: ArrayLike,
    *,
    sigma: float = 1.0,
    eps: float = 0.3,
    matched_diagonal: bool = True,
    rescale_weights: bool = True,
    rng: Optional[np.random.Generator] = None,
    target_density: Optional[float] = None,
    rescale_damping: Optional[ArrayLike] = None,
    rescale_cap: Optional[float] = None,
    rescale_cap_quantile: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Prune continuous-time operator using the covariance "noise probe" rule.

    Parameters
    ----------
    A:
        Square ``N × N`` continuous-time operator (typically ``-D + W``).  Must
        be Hurwitz (all eigenvalues with negative real part) so that the
        Lyapunov equation has a unique positive semidefinite solution.
    sigma:
        Standard deviation of independent white noise injected per node.
    eps:
        Target relative spectral error; lower values keep more edges.
    matched_diagonal:
        When ``True``, adjust the diagonal to preserve (approximately) the
        absolute input magnitude per row after pruning.
    rng:
        Optional ``numpy.random.Generator`` for reproducibility.  If ``None``,
        a default generator is created.

    Returns
    -------
    pruned_A:
        Pruned operator with the same shape and dtype as ``A``.
    stats:
        Dictionary describing the pruning process with keys:
        ``N``, ``eps``, ``sigma``, ``K``, ``Kdeg``,
        ``kept_edges``, ``capped_probs`` (probabilities clipped to 1),
        and ``density`` (kept off-diagonal edges divided by ``N*(N-1)``).

    Raises
    ------
    ValueError
        If the inputs are invalid, the covariance is not positive semidefinite,
        or ``A`` appears to be unstable.

    Notes
    -----
    The Lyapunov solve is ``O(N^3)`` for dense matrices and typically dominates
    the runtime.  The pruning preserves the expectation of each off-diagonal
    entry by reweighting the sampled edges.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if target_density is not None:
        if not (0.0 <= target_density <= 1.0):
            raise ValueError("target_density must lie in [0, 1].")
    if rescale_cap is not None and rescale_cap <= 0.0:
        raise ValueError("rescale_cap must be positive when provided.")
    if rescale_cap_quantile is not None:
        if not (0.0 < rescale_cap_quantile <= 1.0):
            raise ValueError("rescale_cap_quantile must lie in (0, 1].")

    mat = _ensure_square_matrix(A)
    orig_dtype = mat.dtype
    A_float = mat.astype(np.float64, copy=True)
    N = A_float.shape[0]
    damping_float: Optional[np.ndarray] = None
    if rescale_damping is not None:
        damping_float = np.asarray(rescale_damping, dtype=np.float64)
        if damping_float.shape != A_float.shape:
            raise ValueError(
                f"rescale_damping must have shape {A_float.shape}, got {damping_float.shape}."
            )
        damping_float = np.clip(damping_float, 0.0, 1.0)

    # Solve continuous-time Lyapunov equation: A C + C A^T = -(sigma^2) I
    try:
        C = solve_continuous_lyapunov(A_float, -(sigma**2) * np.eye(N))
    except LinAlgError as exc:
        raise ValueError(
            "Lyapunov solver failed; ensure A is Hurwitz (stable)."
        ) from exc

    C = 0.5 * (C + C.T)
    _validate_covariance(C)

    if rng is None:
        rng = np.random.default_rng()

    diagC = np.diag(C)
    off_mask = ~np.eye(N, dtype=bool)
    rows, cols = np.where(off_mask)
    weights = A_float[rows, cols]
    nonzero_mask = weights != 0.0

    rows = rows[nonzero_mask]
    cols = cols[nonzero_mask]
    weights = weights[nonzero_mask]

    abs_weights = np.abs(weights)
    sign_weights = np.sign(weights)

    diff_cov = diagC[rows] + diagC[cols] - sign_weights * 2.0 * C[rows, cols]
    diff_cov = np.maximum(diff_cov, 0.0)  # eliminate tiny negatives from roundoff

    Kdeg = 4.0 * np.log(N) / (eps**2)
    K = 8.0 * np.log(N) / (eps**2 * sigma**2)

    raw_probs = K * abs_weights * diff_cov
    capped_probs = np.count_nonzero(raw_probs > 1.0)
    scale_factor = 1.0
    total_edges = float(N * (N - 1))
    num_available = raw_probs.size
    sum_raw = raw_probs.sum()

    if target_density is not None and num_available > 0 and sum_raw > 0.0:
        zero_edges = max(0.0, total_edges - num_available)
        desired_kept_total = target_density * total_edges
        desired_kept_nonzero = np.clip(desired_kept_total - zero_edges, 0.0, float(num_available))
        if desired_kept_nonzero == 0.0:
            scale_factor = 0.0
        else:
            scale_factor = desired_kept_nonzero / sum_raw
        raw_probs = raw_probs * scale_factor

    probs = np.clip(raw_probs, 0.0, 1.0)
    positive_probs = probs[probs > 0.0]
    inv_positive_probs = 1.0 / positive_probs if positive_probs.size else np.array([], dtype=np.float64)
    cap_value: Optional[float] = None
    cap_mode = "none"
    if rescale_cap is not None:
        cap_value = float(rescale_cap)
        cap_mode = "fixed"
    elif rescale_cap_quantile is not None and inv_positive_probs.size:
        cap_value = float(np.quantile(inv_positive_probs, float(rescale_cap_quantile)))
        cap_mode = "quantile"

    random_draws = rng.random(size=probs.shape[0])
    keep_mask = random_draws < probs
    kept_indices = np.where(keep_mask)[0]

    pruned_A = A_float.copy()
    pruned_A[off_mask] = 0.0
    kept_probs = probs[kept_indices]
    kept_rows = rows[kept_indices]
    kept_cols = cols[kept_indices]
    kept_uncapped_amp = 1.0 / kept_probs if kept_probs.size else np.array([], dtype=np.float64)
    if rescale_weights:
        if damping_float is None:
            amp = kept_uncapped_amp.copy()
            kept_damping = np.ones_like(kept_probs, dtype=np.float64)
        else:
            kept_damping = damping_float[kept_rows, kept_cols]
            amp = 1.0 + kept_damping * (kept_uncapped_amp - 1.0)
        if cap_value is not None:
            amp = np.minimum(amp, cap_value)
        pruned_A[kept_rows, kept_cols] = A_float[kept_rows, kept_cols] * amp
    else:
        pruned_A[kept_rows, kept_cols] = A_float[kept_rows, kept_cols]
        kept_damping = np.zeros_like(kept_probs, dtype=np.float64)
        amp = np.ones_like(kept_probs, dtype=np.float64)

    if matched_diagonal:
        abs_orig = np.sum(np.abs(A_float), axis=1) - np.abs(np.diag(A_float))
        abs_pruned = np.sum(np.abs(pruned_A), axis=1) - np.abs(np.diag(pruned_A))
        delta = abs_pruned - abs_orig
        np.fill_diagonal(pruned_A, np.diag(A_float) - delta)
    else:
        np.fill_diagonal(pruned_A, np.diag(A_float))

    num_kept = int(keep_mask.sum())
    density = num_kept / total_edges if total_edges > 0 else 0.0

    stats = {
        "N": N,
        "eps": float(eps),
        "sigma": float(sigma),
        "K": float(K),
        "Kdeg": float(Kdeg),
        "kept_edges": num_kept,
        "capped_probs": int(capped_probs),
        "density": density,
        "scale_factor": float(scale_factor),
        "rescale_weights": bool(rescale_weights),
        "prob_sum": float(np.sum(probs)),
        "prob_max": float(np.max(probs)) if probs.size else 0.0,
        "prob_mean": float(np.mean(probs)) if probs.size else 0.0,
        "positive_prob_count": int(positive_probs.size),
        "frac_amp_gt10": float(np.mean(inv_positive_probs > 10.0)) if inv_positive_probs.size else 0.0,
        "frac_amp_gt20": float(np.mean(inv_positive_probs > 20.0)) if inv_positive_probs.size else 0.0,
        "inv_p_mean": float(np.mean(inv_positive_probs)) if inv_positive_probs.size else 0.0,
        "inv_p_max": float(np.max(inv_positive_probs)) if inv_positive_probs.size else 0.0,
        "rescale_capped": bool(cap_value is not None and rescale_weights),
        "rescale_cap_mode": cap_mode,
        "rescale_cap_value": float(cap_value) if cap_value is not None else 0.0,
        "rescale_cap_quantile": float(rescale_cap_quantile) if rescale_cap_quantile is not None else 0.0,
        "frac_positive_amp_capped": (
            float(np.mean(inv_positive_probs > cap_value))
            if cap_value is not None and inv_positive_probs.size
            else 0.0
        ),
        "kept_amp_mean": float(np.mean(amp)) if amp.size else 0.0,
        "kept_amp_max": float(np.max(amp)) if amp.size else 0.0,
        "rescale_damped": bool(damping_float is not None and rescale_weights),
        "rescale_damping_mean": float(np.mean(damping_float[off_mask])) if damping_float is not None else 1.0,
        "rescale_damping_min": float(np.min(damping_float[off_mask])) if damping_float is not None else 1.0,
        "rescale_damping_max": float(np.max(damping_float[off_mask])) if damping_float is not None else 1.0,
        "kept_rescale_damping_mean": float(np.mean(kept_damping)) if kept_damping.size else 0.0,
    }

    return pruned_A.astype(orig_dtype, copy=False), stats


if __name__ == "__main__":
    import numpy.random as npr

    rng = npr.default_rng(0)
    N = 8
    base = -np.eye(N)
    perturb = rng.normal(scale=0.05, size=(N, N))
    A_example = base + 0.5 * (perturb + perturb.T)

    pruned, stats = noise_prune(A_example, rng=rng)
    print("Example stats:", stats)
