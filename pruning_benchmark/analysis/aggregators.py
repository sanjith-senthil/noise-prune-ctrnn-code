"""Helpers for aggregating pruning experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

MODEL_COLUMNS = [
    "class",
    "input_dim",
    "hidden_size",
    "output_dim",
    "alpha",
    "activation",
    "preact_noise",
    "postact_noise",
    "use_dale",
    "no_self_connections",
    "hidden_size_override",
]

COLS_TO_DROP = {"config_json", "config_yaml", "metrics_json"}


def load_experiment_records(csv_path: str) -> pd.DataFrame:
    """Load the pruning CSV written by sweeps or suites."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(path)
    drop_cols = [c for c in COLS_TO_DROP if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def stats_by_strategy(
    df: pd.DataFrame,
    metrics: Iterable[str] = ("post_acc_sequence", "delta_post_acc_sequence", "post_loss", "sparsity"),
    group_fields: Iterable[str] = ("task", "strategy", "amount"),
) -> pd.DataFrame:
    """Summaries of metrics grouped by task, strategy, amount."""
    agg = {}
    for metric in metrics:
        if metric in df.columns:
            agg[metric] = ["mean", "std", "count"]
    if not agg:
        return pd.DataFrame()
    grouped = df.groupby(list(group_fields), dropna=False).agg(agg)
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]
    return grouped.reset_index()


def pairwise_deltas(df: pd.DataFrame, baseline: str = "noise_prune") -> pd.DataFrame:
    """Compute delta vs. a baseline strategy per seed."""
    metric = "post_acc_sequence" if "post_acc_sequence" in df.columns else "post_acc"
    required = ["task", "amount", "seed", "strategy", metric]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    filtered = df.dropna(subset=[metric]).copy()
    base = filtered[filtered["strategy"] == baseline]
    non_baseline = filtered[filtered["strategy"] != baseline]
    if non_baseline.empty or base.empty:
        return pd.DataFrame()
    merged = non_baseline.merge(
        base[["task", "amount", "seed", metric]].rename(columns={metric: "baseline_post_acc"}),
        on=["task", "amount", "seed"],
        suffixes=("", "_baseline"),
    )
    merged["delta_vs_baseline"] = merged[metric] - merged["baseline_post_acc"]
    return merged


def load_metrics_jsons(root_dir: str) -> pd.DataFrame:
    """Flatten the metrics.json files beneath results/<run_id>."""
    base = Path(root_dir)
    records = []
    for path in base.rglob("metrics.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data["run_dir"] = str(path.parent)
        records.append(data)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


__all__ = [
    "load_experiment_records",
    "stats_by_strategy",
    "pairwise_deltas",
    "load_metrics_jsons",
    "paired_ttest_vs_baseline",
]


def paired_ttest_vs_baseline(
    df: pd.DataFrame,
    *,
    metric: str = "post_acc_sequence",
    baseline: str = "noise_prune",
) -> pd.DataFrame:
    """Compute paired t-tests (strategy vs baseline) across seeds for each task/amount."""
    try:
        from scipy.stats import ttest_rel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required for paired_ttest_vs_baseline") from exc

    required = {"task", "amount", "seed", "strategy", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    baseline_df = df[df["strategy"] == baseline]
    records = []
    for (task, amount), group in df.groupby(["task", "amount"]):
        base = baseline_df[(baseline_df["task"] == task) & (baseline_df["amount"] == amount)]
        if base.empty:
            continue
        base_series = base.set_index("seed")[metric]
        for strategy, strat_group in group.groupby("strategy"):
            if strategy == baseline:
                continue
            strat_series = strat_group.set_index("seed")[metric]
            common = base_series.index.intersection(strat_series.index)
            if len(common) < 2:
                continue
            stat, pval = ttest_rel(strat_series.loc[common], base_series.loc[common])
            records.append(
                {
                    "task": task,
                    "amount": amount,
                    "strategy": strategy,
                    f"ttest_stat_{metric}": stat,
                    f"ttest_p_{metric}": pval,
                    "n": len(common),
                }
            )
    return pd.DataFrame(records)
