#!/usr/bin/env python3
"""Export noise-prune rescale-value distributions from official benchmark rows.

The official task-preservation CSV stores per-run summaries of the native
rescale amplitudes ``1 / p_ij``.  This script reconstructs the underlying
probability maps for L-NP/V-NP rescale and S-NP rescale, then writes compact
histogram and quantile CSVs suitable for paper figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.linalg import LinAlgError, solve_continuous_lyapunov

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.models import CTRNN
from pruning_benchmark.pruning.noise_prune import _validate_covariance
from pruning_benchmark.pruning.simulation_noise_prune import (
    _as_numpy_batches,
    _collect_centered_samples,
    _empirical_covariance,
    _extract_ctrnn,
    _natural_variability_stats,
    _resolve_noise_scale,
    empirical_noise_prune_scores,
)


DEFAULT_INPUT = (
    "paper_artifacts/official_h512_24net/data/task_preservation/"
    "official_task_preservation_h512_24net_p50_80_allmethods.csv"
)
DEFAULT_OUTDIR = (
    "paper_artifacts/official_h512_24net/data/rescale_value_distributions"
)


def _load_state(path: str) -> Mapping[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _model_from_checkpoint(path: str, *, dt: float, tau: float, activation: str) -> CTRNN:
    state = _load_state(path)
    input_dim = int(state["input_layer.weight"].shape[1])
    hidden_size = int(state["hidden_layer.weight"].shape[0])
    output_dim = int(state["readout_layer.weight"].shape[0])
    model = CTRNN(
        input_dim=input_dim,
        hidden_size=hidden_size,
        output_dim=output_dim,
        dt=float(dt),
        tau=float(tau),
        activation=activation,
        preact_noise=0.0,
        postact_noise=0.0,
        use_dale=False,
        no_self_connections=True,
        scaling=1.0,
        bias=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def _load_batches(path: str) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    except Exception:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"Expected non-empty batch list in {path}")
    batches = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Expected (x, y) batch tuples in {path}")
        batches.append((item[0].cpu(), item[1].cpu()))
    return batches


def _linear_np_probabilities(
    wrec: np.ndarray,
    *,
    amount: float,
    sigma: float,
    eps: float,
    leak_shift: float,
    max_attempts: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    current_shift = float(leak_shift)
    eye = np.eye(wrec.shape[0], dtype=np.float64)
    for attempt in range(int(max_attempts)):
        base_shift = 1.0 + current_shift
        operator = np.asarray(wrec, dtype=np.float64) - base_shift * eye
        try:
            probs, stats = _linear_np_probabilities_once(
                operator,
                amount=float(amount),
                sigma=float(sigma),
                eps=float(eps),
            )
        except ValueError:
            if attempt >= int(max_attempts) - 1:
                raise
            current_shift = max(0.5, current_shift * 2.0 if current_shift > 0.0 else 0.5)
            continue
        stats["leak_shift"] = float(current_shift)
        stats["adaptive_attempt"] = int(attempt)
        return probs, stats
    raise RuntimeError("adaptive shift loop exhausted")


def _linear_np_probabilities_once(
    operator: np.ndarray,
    *,
    amount: float,
    sigma: float,
    eps: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n = operator.shape[0]
    try:
        cov = solve_continuous_lyapunov(
            operator.astype(np.float64, copy=False),
            -(float(sigma) ** 2) * np.eye(n),
        )
    except LinAlgError as exc:
        raise ValueError("Lyapunov solver failed") from exc
    cov = 0.5 * (cov + cov.T)
    _validate_covariance(cov)

    off_mask = ~np.eye(n, dtype=bool)
    rows, cols = np.where(off_mask)
    weights = np.asarray(operator, dtype=np.float64)[rows, cols]
    nonzero = weights != 0.0
    rows = rows[nonzero]
    cols = cols[nonzero]
    weights = weights[nonzero]

    diag_cov = np.diag(cov)
    diff_cov = diag_cov[rows] + diag_cov[cols] - 2.0 * np.sign(weights) * cov[rows, cols]
    diff_cov = np.maximum(diff_cov, 0.0)
    k_const = 8.0 * np.log(n) / (float(eps) ** 2 * float(sigma) ** 2)
    raw = k_const * np.abs(weights) * diff_cov

    total_edges = float(n * (n - 1))
    desired_kept = float(1.0 - amount) * total_edges
    scale_factor = 1.0
    if raw.size and float(np.sum(raw)) > 0.0:
        scale_factor = desired_kept / float(np.sum(raw))
        raw = raw * scale_factor
    capped_probs = int(np.count_nonzero(raw > 1.0))
    prob_vec = np.clip(raw, 0.0, 1.0)
    probs = np.zeros_like(operator, dtype=np.float64)
    probs[rows, cols] = prob_vec
    return probs, {
        "K": float(k_const),
        "scale_factor": float(scale_factor),
        "capped_probs": int(capped_probs),
        "score_sum": float(np.sum(prob_vec)),
        "score_max": float(np.max(prob_vec)) if prob_vec.size else 0.0,
        "positive_prob_count": int(np.count_nonzero(prob_vec > 0.0)),
    }


def _simulation_np_probabilities(
    model: CTRNN,
    *,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    amount: float,
    sigma: float | None,
    sigma_source: str,
    eps: float,
    observable_space: str,
    inject_space: str,
    centering: str,
    max_samples: int,
    burn_in_steps: int,
    rng_seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(int(rng_seed))
    net = _extract_ctrnn(model)
    batch_arrays = _as_numpy_batches(batches)
    natural = _natural_variability_stats(net, batch_arrays, burn_in_steps=int(burn_in_steps))
    sigma_used = _resolve_noise_scale(natural, sigma=sigma, sigma_source=str(sigma_source))
    samples, sim_stats = _collect_centered_samples(
        net,
        batch_arrays,
        observable_space=str(observable_space),
        inject_space=str(inject_space),
        centering=str(centering),
        noise_scale=float(sigma_used),
        max_samples=int(max_samples),
        burn_in_steps=int(burn_in_steps),
        rng=rng,
    )
    cov = _empirical_covariance(samples, center_assumed=True)
    mean_gain = np.asarray(json.loads(natural["empirical_mean_gain_json"]), dtype=np.float64)
    score_weights = net.wrec
    if str(observable_space) == "voltage":
        score_weights = net.wrec * mean_gain[np.newaxis, :]
    probs, score_stats = empirical_noise_prune_scores(
        score_weights,
        cov,
        sigma=float(sigma_used),
        eps=float(eps),
        target_density=float(1.0 - amount),
    )
    stats: Dict[str, float] = {}
    stats.update(natural)
    stats.update(sim_stats)
    stats.update(score_stats)
    stats["sigma_used"] = float(sigma_used)
    stats["empirical_cov_trace"] = float(np.trace(cov))
    stats["empirical_cov_diag_mean"] = float(np.mean(np.diag(cov)))
    return probs, stats


def _positive_rescale_values(probs: np.ndarray) -> np.ndarray:
    n = probs.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    positive = probs[off_mask]
    positive = positive[positive > 0.0]
    return 1.0 / positive


def _histogram_rows(
    values: np.ndarray,
    *,
    bins: np.ndarray,
    row_base: Mapping[str, object],
) -> List[Dict[str, object]]:
    log_values = np.log10(values)
    counts, edges = np.histogram(log_values, bins=bins)
    rows: List[Dict[str, object]] = []
    for idx, count in enumerate(counts):
        if int(count) == 0:
            continue
        left = float(edges[idx])
        right = float(edges[idx + 1])
        out = dict(row_base)
        out.update({
            "log10_rescale_left": left,
            "log10_rescale_right": right,
            "rescale_left": float(10.0 ** left),
            "rescale_right": float(10.0 ** right),
            "count": int(count),
        })
        rows.append(out)
    return rows


def _summary_row(
    values: np.ndarray,
    *,
    row_base: Mapping[str, object],
    stats: Mapping[str, object],
) -> Dict[str, object]:
    quantiles = np.quantile(values, [0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    row = dict(row_base)
    row.update({
        "positive_prob_count": int(values.size),
        "rescale_min": float(np.min(values)),
        "rescale_mean": float(np.mean(values)),
        "rescale_std": float(np.std(values, ddof=0)),
        "rescale_q50": float(quantiles[0]),
        "rescale_q75": float(quantiles[1]),
        "rescale_q90": float(quantiles[2]),
        "rescale_q95": float(quantiles[3]),
        "rescale_q99": float(quantiles[4]),
        "rescale_q999": float(quantiles[5]),
        "rescale_max": float(np.max(values)),
        "frac_rescale_gt10": float(np.mean(values > 10.0)),
        "frac_rescale_gt20": float(np.mean(values > 20.0)),
        "score_sum": float(stats.get("score_sum", np.nan)),
        "score_max": float(stats.get("score_max", np.nan)),
        "scale_factor": float(stats.get("scale_factor", np.nan)),
        "capped_probs": int(stats.get("capped_probs", 0)),
        "sigma_used": float(stats.get("sigma_used", np.nan)),
        "adaptive_shift": float(stats.get("leak_shift", np.nan)),
        "adaptive_attempt": int(stats.get("adaptive_attempt", -1)),
    })
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_histogram(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, object, object, object, object], int] = defaultdict(int)
    meta: Dict[Tuple[object, object, object, object, object], Dict[str, object]] = {}
    for row in rows:
        key = (
            row["method_family"],
            row["method"],
            row["amount"],
            row["log10_rescale_left"],
            row["log10_rescale_right"],
        )
        grouped[key] += int(row["count"])
        if key not in meta:
            meta[key] = {
                "method_family": row["method_family"],
                "method": row["method"],
                "amount": row["amount"],
                "log10_rescale_left": row["log10_rescale_left"],
                "log10_rescale_right": row["log10_rescale_right"],
                "rescale_left": row["rescale_left"],
                "rescale_right": row["rescale_right"],
            }
    out = []
    for key, count in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][3])):
        row = dict(meta[key])
        row["count"] = int(count)
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", default=DEFAULT_INPUT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--amount", type=float, default=0.80)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=100.0)
    parser.add_argument("--activation", default="tanh")
    parser.add_argument("--linear_max_attempts", type=int, default=5)
    parser.add_argument("--bin_width_log10", type=float, default=0.05)
    parser.add_argument("--max_log10", type=float, default=9.0)
    parser.add_argument("--max_rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    outdir = Path(args.outdir)
    df = pd.read_csv(input_csv)
    rows = df[
        (df["method"].isin(["S-NP rescale", "V-NP rescale"]))
        & np.isclose(df["amount"].astype(float), float(args.amount))
    ].copy()
    rows = rows.sort_values(["method", "task", "source_model_label", "pruning_seed"])
    if args.max_rows is not None:
        rows = rows.head(int(args.max_rows))
    if rows.empty:
        raise ValueError(f"No S-NP/V-NP rescale rows found at amount={args.amount} in {input_csv}")

    bins = np.arange(0.0, float(args.max_log10) + float(args.bin_width_log10), float(args.bin_width_log10))
    if bins[-1] < float(args.max_log10):
        bins = np.append(bins, float(args.max_log10))

    model_cache: Dict[str, CTRNN] = {}
    batch_cache: Dict[str, Sequence[Tuple[torch.Tensor, torch.Tensor]]] = {}
    prob_cache: Dict[Tuple[object, ...], Tuple[np.ndarray, Dict[str, float]]] = {}
    hist_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    started = time.time()
    total = len(rows)
    for ordinal, (_, row) in enumerate(rows.iterrows(), start=1):
        method = str(row["method"])
        method_family = "S-NP" if method == "S-NP rescale" else "L-NP"
        model_path = str(row["load_model_path"])
        if model_path not in model_cache:
            model_cache[model_path] = _model_from_checkpoint(
                model_path,
                dt=float(args.dt),
                tau=float(args.tau),
                activation=str(args.activation),
            )
        model = model_cache[model_path]
        pruning_seed = int(float(row.get("pruning_seed", 0)))
        amount = float(row["amount"])
        task = str(row["task"])
        label = str(row["source_model_label"])

        base = {
            "method_family": method_family,
            "method": method,
            "amount": amount,
            "task": task,
            "source_model_label": label,
            "pruning_seed": pruning_seed,
            "score_batch_seed": int(float(row["score_batch_seed"])) if not pd.isna(row.get("score_batch_seed")) else "",
        }
        if method_family == "L-NP":
            cache_key = (method, model_path, amount)
            if cache_key not in prob_cache:
                wrec = model.hidden_layer.weight.detach().cpu().numpy().copy()
                probs, stats = _linear_np_probabilities(
                    wrec,
                    amount=amount,
                    sigma=float(row.get("noise_sigma", 1.0) if not pd.isna(row.get("noise_sigma")) else 1.0),
                    eps=float(row.get("noise_eps", 0.3) if not pd.isna(row.get("noise_eps")) else 0.3),
                    leak_shift=float(row.get("noise_leak_shift", 0.0) if not pd.isna(row.get("noise_leak_shift", np.nan)) else 0.0),
                    max_attempts=int(args.linear_max_attempts),
                )
                prob_cache[cache_key] = (probs, stats)
            probs, stats = prob_cache[cache_key]
        else:
            score_path = str(row["score_batches_path"])
            if score_path not in batch_cache:
                batch_cache[score_path] = _load_batches(score_path)
            cache_key = (
                method,
                model_path,
                amount,
                score_path,
                pruning_seed,
                str(row.get("sim_np_observable_space", "rate")),
                str(row.get("sim_np_inject_space", "rate")),
                str(row.get("sim_np_centering", "trajectory_mean")),
            )
            if cache_key not in prob_cache:
                explicit_sigma = None
                if not pd.isna(row.get("noise_sigma")):
                    explicit_sigma = float(row.get("noise_sigma"))
                probs, stats = _simulation_np_probabilities(
                    model,
                    batches=batch_cache[score_path],
                    amount=amount,
                    sigma=explicit_sigma,
                    sigma_source=str(row.get("sim_np_sigma_source", "natural_voltage")),
                    eps=float(row.get("noise_eps", 0.3) if not pd.isna(row.get("noise_eps")) else 0.3),
                    observable_space=str(row.get("sim_np_observable_space", "rate")),
                    inject_space=str(row.get("sim_np_inject_space", "rate")),
                    centering=str(row.get("sim_np_centering", "trajectory_mean")),
                    max_samples=int(float(row.get("sim_np_max_samples", 25000))),
                    burn_in_steps=int(float(row.get("sim_np_burn_in_steps", 300))),
                    rng_seed=pruning_seed,
                )
                prob_cache[cache_key] = (probs, stats)
            probs, stats = prob_cache[cache_key]

        values = _positive_rescale_values(probs)
        if values.size == 0:
            raise RuntimeError(f"No positive probabilities for {method} {label} seed={pruning_seed}")
        if np.max(values) > 10.0 ** float(args.max_log10):
            raise RuntimeError(
                f"Increase --max_log10; max rescale {np.max(values):.6g} exceeds current range."
            )
        summary_rows.append(_summary_row(values, row_base=base, stats=stats))
        hist_rows.extend(_histogram_rows(values, bins=bins, row_base=base))
        elapsed = time.time() - started
        print(
            f"[{ordinal}/{total}] {method_family} {task} {label} prune_seed={pruning_seed} "
            f"n={values.size} mean={float(np.mean(values)):.4g} q95={float(np.quantile(values, 0.95)):.4g} "
            f"max={float(np.max(values)):.4g} elapsed={elapsed/60:.1f}m",
            flush=True,
        )

    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"p{int(round(float(args.amount) * 100)):02d}"
    by_run_path = outdir / f"rescale_value_histogram_snp_lnp_{suffix}_by_run.csv"
    overall_path = outdir / f"rescale_value_histogram_snp_lnp_{suffix}_overall.csv"
    summary_path = outdir / f"rescale_value_run_summary_snp_lnp_{suffix}.csv"
    _write_csv(by_run_path, hist_rows)
    _write_csv(overall_path, _aggregate_histogram(hist_rows))
    _write_csv(summary_path, summary_rows)
    print(f"Wrote {by_run_path}")
    print(f"Wrote {overall_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
