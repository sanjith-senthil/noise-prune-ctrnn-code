#!/usr/bin/env python3
"""Calibrate and stress-test sigma for simulation-based noise-prune.

This script uses trained CTRNN checkpoints from a suite CSV, computes the
network's natural voltage variability under noise-free task input,

    sigma_natural_v = sqrt(mean(diag(Cov(v_task))))

and then measures how much the simulation-noise-prune mask changes when sigma
is scaled by factors around that baseline.  The main sensitivity metric is
mask overlap relative to the factor-1.0 baseline mask.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pruning_benchmark.analysis.replay import resolve_modcog_B, resolve_modcog_T
from pruning_benchmark.models import CTRNN
from pruning_benchmark.pruning.simulation_noise_prune import (
    _as_numpy_batches,
    _extract_ctrnn,
    _natural_variability_stats,
    simulation_noise_prune_mask,
)
from pruning_benchmark.tasks.modcog import ensure_modcog_env_id
from pruning_benchmark.tasks.neurogym import ModCogTrialDM


@dataclass(frozen=True)
class Target:
    task: str
    seed: int
    checkpoint_path: str
    T: int
    B: int
    activation: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sigma sensitivity for simulation-based noise-prune.")
    p.add_argument("--suite_csv", required=True)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--checkpoint_limit", type=int, default=None)
    p.add_argument("--activation", choices=("tanh", "shifted_tanh"), default="tanh")
    p.add_argument("--sample_batches", type=int, default=8)
    p.add_argument(
        "--batch_seed",
        type=int,
        default=100000,
        help="Seed for task batches used to estimate natural sigma and S-NP covariance.",
    )
    p.add_argument("--input_device", default="cpu")
    p.add_argument("--amounts", type=float, nargs="+", default=[0.8])
    p.add_argument("--sigma_factors", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.5, 2.0])
    p.add_argument("--rng_seeds", type=int, nargs="+", default=[0])
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--observable_space", choices=("rate", "voltage"), default="rate")
    p.add_argument("--inject_space", choices=("rate", "voltage"), default="rate")
    p.add_argument("--centering", choices=("conditional", "trajectory_mean"), default="conditional")
    p.add_argument("--max_samples", type=int, default=25000)
    p.add_argument("--burn_in_steps", type=int, default=300)
    p.add_argument("--output_rows_csv", required=True)
    p.add_argument("--output_summary_csv", required=True)
    p.add_argument("--output_plot", required=True)
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute all rows even if output_rows_csv already contains matching rows.",
    )
    return p.parse_args()


def _normalize_task_filters(task_filters: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not task_filters:
        return None
    return {t.strip().lower() for t in task_filters if t.strip()}


def _resolve_checkpoint_from_row(row: Dict[str, str]) -> Optional[str]:
    for key in ("save_model_path", "load_model_path"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return None


def _load_targets(args: argparse.Namespace) -> List[Target]:
    rows = list(csv.DictReader(Path(args.suite_csv).open()))
    task_filters = _normalize_task_filters(args.tasks)
    targets: List[Target] = []
    seen: set[str] = set()
    for row in rows:
        if (row.get("model_type") or "ctrnn").strip().lower() != "ctrnn":
            continue
        task = (row.get("task") or "").strip()
        if not task:
            continue
        if task_filters is not None and task.lower() not in task_filters:
            continue
        ckpt = _resolve_checkpoint_from_row(row)
        if not ckpt or ckpt in seen or not Path(ckpt).exists():
            continue
        targets.append(
            Target(
                task=task,
                seed=int(row.get("seed") or 0),
                checkpoint_path=ckpt,
                T=resolve_modcog_T(row, task),
                B=resolve_modcog_B(row),
                activation=str(row.get("activation") or args.activation),
            )
        )
        seen.add(ckpt)
        if args.checkpoint_limit is not None and len(targets) >= int(args.checkpoint_limit):
            break
    if not targets:
        raise ValueError(f"No usable CTRNN checkpoints found in {args.suite_csv}")
    return targets


def _load_ctrnn_model(checkpoint_path: str, *, activation: str) -> CTRNN:
    state = torch.load(checkpoint_path, map_location="cpu")
    win = state["input_layer.weight"]
    wrec = state["hidden_layer.weight"]
    wout = None
    for key in ("readout_layer.weight", "output_layer.weight"):
        if key in state:
            wout = state[key]
            break
    if wout is None:
        raise KeyError(
            f"Could not find a readout weight in checkpoint {checkpoint_path}. "
            "Expected one of: readout_layer.weight, output_layer.weight."
        )
    model = CTRNN(
        input_dim=int(win.shape[1]),
        hidden_size=int(wrec.shape[0]),
        output_dim=int(wout.shape[0]),
        dt=10.0,
        tau=100.0,
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


def _sample_batches(
    target: Target,
    *,
    sample_batches: int,
    device: str,
    batch_seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    env_id = ensure_modcog_env_id(target.task)
    if env_id is None:
        raise ValueError(f"Could not resolve Mod-Cog env for task {target.task}")
    data = ModCogTrialDM(
        env_id,
        T=target.T,
        B=target.B,
        device=device,
        last_only=False,
        seed=int(batch_seed),
        mask_fixation=True,
    )
    return [data.sample_batch() for _ in range(max(1, int(sample_batches)))]


def _mask_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float]:
    kept_a = int(np.count_nonzero(mask_a))
    intersect = int(np.count_nonzero(np.logical_and(mask_a, mask_b)))
    union = int(np.count_nonzero(np.logical_or(mask_a, mask_b)))
    overlap = float(intersect / kept_a) if kept_a > 0 else 0.0
    jaccard = float(intersect / union) if union > 0 else 1.0
    return overlap, jaccard


def _write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            try:
                fieldnames = next(reader)
            except StopIteration:
                fieldnames = list(row.keys())
    else:
        fieldnames = list(row.keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _row_key(row: Dict[str, object]) -> Tuple[object, ...]:
    return (
        str(row.get("task", "")),
        str(row.get("checkpoint_path", "")),
        round(float(row.get("amount", 0.0)), 8),
        int(float(row.get("rng_seed", 0))),
        round(float(row.get("sigma_factor", 0.0)), 8),
        str(row.get("observable_space", "")),
        str(row.get("inject_space", "")),
        str(row.get("centering", "")),
        int(float(row.get("max_samples", 0))),
        int(float(row.get("burn_in_steps", 0))),
        int(float(row.get("batch_seed", -1))),
    )


def _planned_row_key(
    *,
    task: str,
    checkpoint_path: str,
    amount: float,
    rng_seed: int,
    sigma_factor: float,
    observable_space: str,
    inject_space: str,
    centering: str,
    max_samples: int,
    burn_in_steps: int,
    batch_seed: int,
) -> Tuple[object, ...]:
    return _row_key(
        {
            "task": task,
            "checkpoint_path": checkpoint_path,
            "amount": amount,
            "rng_seed": rng_seed,
            "sigma_factor": sigma_factor,
            "observable_space": observable_space,
            "inject_space": inject_space,
            "centering": centering,
            "max_samples": max_samples,
            "burn_in_steps": burn_in_steps,
            "batch_seed": batch_seed,
        }
    )


def _read_existing_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _summarize(rows_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        rows_df.groupby(["amount", "sigma_factor"], dropna=False)
        .agg(
            overlap_mean=("overlap_vs_factor1", "mean"),
            overlap_std=("overlap_vs_factor1", "std"),
            jaccard_mean=("jaccard_vs_factor1", "mean"),
            jaccard_std=("jaccard_vs_factor1", "std"),
            sigma_used_mean=("sigma_used", "mean"),
            sigma_natural_v_mean=("sigma_natural_v", "mean"),
            natural_trace_v_mean=("natural_cov_trace_v", "mean"),
            n=("task", "count"),
        )
        .reset_index()
    )
    overall = (
        rows_df.groupby(["sigma_factor"], dropna=False)
        .agg(
            overlap_mean=("overlap_vs_factor1", "mean"),
            overlap_std=("overlap_vs_factor1", "std"),
            jaccard_mean=("jaccard_vs_factor1", "mean"),
            jaccard_std=("jaccard_vs_factor1", "std"),
            sigma_used_mean=("sigma_used", "mean"),
            sigma_natural_v_mean=("sigma_natural_v", "mean"),
            natural_trace_v_mean=("natural_cov_trace_v", "mean"),
            n=("task", "count"),
        )
        .reset_index()
    )
    overall.insert(0, "amount", "all")
    return pd.concat([grouped, overall], ignore_index=True)


def _plot_summary(summary_df: pd.DataFrame, output_plot: Path) -> None:
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.8))
    for amount, amount_df in summary_df.groupby("amount", dropna=False):
        amount_df = amount_df.sort_values("sigma_factor")
        label = f"amount={amount}"
        plt.errorbar(
            amount_df["sigma_factor"],
            amount_df["overlap_mean"],
            yerr=amount_df["overlap_std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=label,
        )
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xscale("log")
    plt.xticks([0.5, 0.75, 1.0, 1.5, 2.0], ["0.5", "0.75", "1.0", "1.5", "2.0"])
    plt.ylim(0.0, 1.02)
    plt.xlabel(r"$\sigma / \sigma_{\mathrm{natural},v}$")
    plt.ylabel("Mask overlap vs factor-1 baseline")
    plt.title("Simulation NP sigma sensitivity")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=180)
    plt.close()


def main() -> None:
    args = _parse_args()
    targets = _load_targets(args)
    sigma_factors = [float(x) for x in args.sigma_factors]
    if 1.0 not in sigma_factors:
        sigma_factors = sorted(set(sigma_factors + [1.0]))
    rows_path = Path(args.output_rows_csv)
    summary_path = Path(args.output_summary_csv)
    plot_path = Path(args.output_plot)

    existing_rows: List[Dict[str, object]] = []
    completed_keys: set[Tuple[object, ...]] = set()
    if rows_path.exists() and args.force:
        rows_path.unlink()
    elif rows_path.exists():
        existing_rows = _read_existing_rows(rows_path)
        completed_keys = {_row_key(row) for row in existing_rows}
        if completed_keys:
            print(f"Resuming from {rows_path}: {len(completed_keys)} completed rows")

    for target in targets:
        model = _load_ctrnn_model(target.checkpoint_path, activation=target.activation)
        batches = _sample_batches(
            target,
            sample_batches=args.sample_batches,
            device=args.input_device,
            batch_seed=int(args.batch_seed),
        )
        net = _extract_ctrnn(model)
        batch_arrays = _as_numpy_batches(batches)
        natural = _natural_variability_stats(net, batch_arrays, burn_in_steps=int(args.burn_in_steps))
        sigma_natural_v = float(natural["sigma_natural_v"])

        for amount in args.amounts:
            remaining_by_rng: Dict[int, List[float]] = {}
            for rng_seed in args.rng_seeds:
                remaining = []
                for sigma_factor in sigma_factors:
                    planned_key = _planned_row_key(
                        task=target.task,
                        checkpoint_path=target.checkpoint_path,
                        amount=float(amount),
                        rng_seed=int(rng_seed),
                        sigma_factor=float(sigma_factor),
                        observable_space=args.observable_space,
                        inject_space=args.inject_space,
                        centering=args.centering,
                        max_samples=int(args.max_samples),
                        burn_in_steps=int(args.burn_in_steps),
                        batch_seed=int(args.batch_seed),
                    )
                    if planned_key not in completed_keys:
                        remaining.append(float(sigma_factor))
                if remaining:
                    remaining_by_rng[int(rng_seed)] = remaining
            if not remaining_by_rng:
                continue

            baseline_masks: Dict[int, np.ndarray] = {}
            baseline_stats: Dict[int, Dict[str, float]] = {}
            for rng_seed in remaining_by_rng:
                mask, stats = simulation_noise_prune_mask(
                    model,
                    float(amount),
                    batches=batches,
                    sigma=sigma_natural_v,
                    eps=float(args.eps),
                    observable_space=args.observable_space,
                    inject_space=args.inject_space,
                    centering=args.centering,
                    max_samples=int(args.max_samples),
                    burn_in_steps=int(args.burn_in_steps),
                    rng_seed=int(rng_seed),
                )
                baseline_masks[int(rng_seed)] = mask
                baseline_stats[int(rng_seed)] = stats

            for sigma_factor in sigma_factors:
                sigma_value = float(sigma_factor) * sigma_natural_v
                for rng_seed in remaining_by_rng:
                    planned_key = _planned_row_key(
                        task=target.task,
                        checkpoint_path=target.checkpoint_path,
                        amount=float(amount),
                        rng_seed=int(rng_seed),
                        sigma_factor=float(sigma_factor),
                        observable_space=args.observable_space,
                        inject_space=args.inject_space,
                        centering=args.centering,
                        max_samples=int(args.max_samples),
                        burn_in_steps=int(args.burn_in_steps),
                        batch_seed=int(args.batch_seed),
                    )
                    if planned_key in completed_keys:
                        continue
                    if abs(float(sigma_factor) - 1.0) < 1e-12:
                        mask = baseline_masks[int(rng_seed)]
                        stats = baseline_stats[int(rng_seed)]
                    else:
                        mask, stats = simulation_noise_prune_mask(
                            model,
                            float(amount),
                            batches=batches,
                            sigma=sigma_value,
                            eps=float(args.eps),
                            observable_space=args.observable_space,
                            inject_space=args.inject_space,
                            centering=args.centering,
                            max_samples=int(args.max_samples),
                            burn_in_steps=int(args.burn_in_steps),
                            rng_seed=int(rng_seed),
                        )
                    overlap, jaccard = _mask_overlap(mask, baseline_masks[int(rng_seed)])
                    row = {
                        "task": target.task,
                        "model_seed": target.seed,
                        "checkpoint_path": target.checkpoint_path,
                        "amount": float(amount),
                        "rng_seed": int(rng_seed),
                        "sigma_factor": float(sigma_factor),
                        "sigma_natural_v": sigma_natural_v,
                        "sigma_used": float(stats["sigma_used"]),
                        "batch_seed": int(args.batch_seed),
                        "observable_space": args.observable_space,
                        "inject_space": args.inject_space,
                        "centering": args.centering,
                        "max_samples": int(args.max_samples),
                        "burn_in_steps": int(args.burn_in_steps),
                        "kept_edges": int(np.count_nonzero(mask)),
                        "overlap_vs_factor1": overlap,
                        "jaccard_vs_factor1": jaccard,
                        "natural_cov_trace_v": float(natural["natural_cov_trace_v"]),
                        "natural_cov_trace_rate": float(natural["natural_cov_trace_rate"]),
                        "empirical_mean_gain": float(natural["empirical_mean_gain"]),
                        "empirical_min_gain": float(natural["empirical_min_gain"]),
                        "empirical_max_gain": float(natural["empirical_max_gain"]),
                        "sample_count": int(stats["sample_count"]),
                        "empirical_cov_trace": float(stats["empirical_cov_trace"]),
                        "empirical_cov_diag_mean": float(stats["empirical_cov_diag_mean"]),
                    }
                    key = _row_key(row)
                    if key in completed_keys:
                        continue
                    _append_row(rows_path, row)
                    completed_keys.add(key)
                    print(
                        "wrote "
                        f"task={target.task} model_seed={target.seed} amount={amount} "
                        f"rng_seed={rng_seed} sigma_factor={sigma_factor} "
                        f"overlap={overlap:.4f} jaccard={jaccard:.4f}"
                    )

    rows_df = pd.DataFrame(_read_existing_rows(rows_path))
    if rows_df.empty:
        raise ValueError(f"No rows available in {rows_path}")
    numeric_cols = [
        "amount",
        "rng_seed",
        "sigma_factor",
        "sigma_natural_v",
        "sigma_used",
        "batch_seed",
        "max_samples",
        "burn_in_steps",
        "kept_edges",
        "overlap_vs_factor1",
        "jaccard_vs_factor1",
        "natural_cov_trace_v",
        "natural_cov_trace_rate",
        "empirical_mean_gain",
        "empirical_min_gain",
        "empirical_max_gain",
        "sample_count",
        "empirical_cov_trace",
        "empirical_cov_diag_mean",
    ]
    for col in numeric_cols:
        if col in rows_df.columns:
            rows_df[col] = pd.to_numeric(rows_df[col], errors="coerce")
    summary_df = _summarize(rows_df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    plot_df = summary_df[summary_df["amount"] != "all"].copy()
    _plot_summary(plot_df, plot_path)
    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
