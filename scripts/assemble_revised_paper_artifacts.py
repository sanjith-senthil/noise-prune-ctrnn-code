#!/usr/bin/env python3
"""Create canonical revised-paper task artifacts without modifying history."""

from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "paper_artifacts/official_h512_24net"
OUT = OFFICIAL / "data/revised_scope"
MAIN_SOURCE = OFFICIAL / "data/task_preservation/official_task_preservation_h512_24net_p50_80_allmethods.csv"
CAP_SOURCE = (
    OFFICIAL
    / "data/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "snp_capped_rescale_q50_q75_q90_full8_p50_80.csv"
)
HIDDEN_SIZE = 512
OFFDIAG_EDGES = HIDDEN_SIZE * (HIDDEN_SIZE - 1)
FULL_EDGES = HIDDEN_SIZE * HIDDEN_SIZE
METHOD_ORDER = {
    "Baseline": 0,
    "Random": 1,
    "Magnitude": 2,
    "OBS compensated": 3,
    "L-NP mask": 4,
    "L-NP rescale": 5,
    "S-NP mask": 6,
    "S-NP rescale": 7,
}
SUMMARY_METRICS = (
    "post_acc_sequence",
    "sequence_retention",
    "effective_nonzero_recurrent_density_offdiag",
    "post_rec_linear_rho",
    "post_rec_linear_rho_ratio_to_unpruned",
)


def add_density_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["nominal_requested_sparsity"] = out["amount"].astype(float)
    out["nominal_requested_density"] = 1.0 - out["nominal_requested_sparsity"]
    out["recorded_mask_density_full_matrix"] = 1.0 - out["post_sparsity_recurrent"].astype(float)
    out["effective_nonzero_recurrent_density_full_matrix"] = (
        out["post_rec_weight_nz_count"].astype(float) / FULL_EDGES
    )
    out["effective_nonzero_recurrent_density_offdiag"] = (
        out["post_rec_weight_nz_count"].astype(float) / OFFDIAG_EDGES
    )
    out["post_rec_linear_rho_ratio_to_unpruned"] = (
        out["post_rec_linear_rho"].astype(float) / out["pre_rec_linear_rho"].astype(float)
    )
    return out


def add_baseline_retention(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=["baseline_acc_sequence", "sequence_retention"], errors="ignore").copy()
    baseline = (
        out[out["method"] == "Baseline"][["source_model_label", "post_acc_sequence"]]
        .rename(columns={"post_acc_sequence": "baseline_acc_sequence"})
    )
    if baseline["source_model_label"].duplicated().any():
        raise ValueError("expected one loaded baseline row per source model")
    if len(baseline) != out["source_model_label"].nunique():
        raise ValueError(
            f"baseline/source-model mismatch: baselines={len(baseline)} "
            f"source_models={out['source_model_label'].nunique()}"
        )
    if (baseline["baseline_acc_sequence"].astype(float) <= 0.0).any():
        raise ValueError("baseline sequence accuracy must be positive for retention normalization")
    out = out.merge(baseline, on="source_model_label", how="left", validate="many_to_one")
    out["sequence_retention"] = (
        out["post_acc_sequence"].astype(float) / out["baseline_acc_sequence"].astype(float)
    )
    return out


def sem(series: pd.Series) -> float:
    return float(series.std(ddof=1) / np.sqrt(len(series))) if len(series) > 1 else np.nan


def sd(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else np.nan


def summarize_network_cells(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    extra_metrics: tuple[str, ...] = (),
) -> pd.DataFrame:
    metrics = (*SUMMARY_METRICS, *extra_metrics)
    cell_aggs: dict[str, tuple[str, str]] = {
        metric: (metric, "mean") for metric in metrics
    }
    cell_aggs["raw_run_count"] = ("post_acc_sequence", "size")
    cell = (
        df[df["method"] != "Baseline"]
        .groupby([*group_cols, "task_short", "source_network_seed"], as_index=False)
        .agg(**cell_aggs)
    )
    summary_aggs: dict[str, tuple[str, object]] = {
        "n": ("post_acc_sequence", "count"),
        "n_trained_networks": ("post_acc_sequence", "count"),
        "raw_run_count": ("raw_run_count", "sum"),
        "pruning_seed_replicates_per_network_min": ("raw_run_count", "min"),
        "pruning_seed_replicates_per_network_max": ("raw_run_count", "max"),
    }
    for metric in metrics:
        summary_aggs[f"{metric}_mean"] = (metric, "mean")
        summary_aggs[f"{metric}_sd"] = (metric, sd)
        summary_aggs[f"{metric}_sem"] = (metric, sem)
    summary = cell.groupby(group_cols, as_index=False).agg(**summary_aggs)

    task_cell = (
        cell.groupby([*group_cols, "task_short"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in metrics})
    )
    task_aggs: dict[str, tuple[str, object]] = {
        "n_tasks": ("task_short", "count"),
    }
    for metric in metrics:
        task_aggs[f"{metric}_taskmean_sd"] = (metric, sd)
        task_aggs[f"{metric}_taskmean_sem"] = (metric, sem)
    task_summary = task_cell.groupby(group_cols, as_index=False).agg(**task_aggs)
    return summary.merge(task_summary, on=group_cols, how="left", validate="one_to_one")


def write_main(out_root: Path = OUT) -> None:
    df = pd.read_csv(MAIN_SOURCE)
    df = df[~df["method"].isin(["OP-NP mask", "OP-NP rescale"])].copy()
    df["method"] = df["method"].replace({"V-NP mask": "L-NP mask", "V-NP rescale": "L-NP rescale"})
    df["method_order"] = df["method"].map(METHOD_ORDER)
    df = add_density_fields(df)
    df = add_baseline_retention(df)
    df = df.sort_values(
        ["task_short", "source_network_seed", "pruning_pct", "method_order", "pruning_seed"],
        na_position="first",
    )
    expected = {
        "Baseline": 24,
        "Random": 288,
        "Magnitude": 96,
        "OBS compensated": 96,
        "L-NP mask": 96,
        "L-NP rescale": 288,
        "S-NP mask": 288,
        "S-NP rescale": 288,
    }
    counts = df.groupby("method").size().to_dict()
    if counts != expected or len(df) != 1464:
        raise ValueError(f"unexpected revised main counts: rows={len(df)} counts={counts}")
    main_dir = out_root / "task_preservation"
    main_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(main_dir / "revised_task_preservation_h512_24net_p50_80.csv", index=False)
    # Pruning seeds are technical replicates. Average them within each trained
    # network before computing paper-facing SEM across the 24 trained networks.
    # Also export SEM across the eight task means as a clustered sensitivity view.
    summary = summarize_network_cells(
        df,
        group_cols=["method_order", "method", "pruning_pct"],
    ).sort_values(["method_order", "pruning_pct"])
    summary.to_csv(main_dir / "revised_task_preservation_h512_24net_p50_80_summary.csv", index=False)


def cap_method(row: pd.Series) -> str:
    if row["strategy"] == "none":
        return "Baseline"
    if row["strategy"] == "simulation_noise_prune_rescale":
        return "S-NP rescale"
    q = int(round(float(row["prune_rescale_cap_quantile"]) * 100.0))
    return f"S-NP capped q{q:02d}"


def write_capped(out_root: Path = OUT) -> None:
    df = pd.read_csv(CAP_SOURCE).copy()
    df["method"] = df.apply(cap_method, axis=1)
    df["task_short"] = df["task"].astype(str).str.replace("modcog:", "", regex=False)
    df["pruning_pct"] = (df["amount"].astype(float) * 100.0).round().astype(int)
    df = add_density_fields(df)
    df = add_baseline_retention(df)
    df["candidate_edge_cap_value"] = df["prune_rescale_cap_value"].astype(float)
    expected = {
        "Baseline": 24,
        "S-NP rescale": 96,
        "S-NP capped q50": 96,
        "S-NP capped q75": 96,
        "S-NP capped q90": 96,
    }
    counts = df.groupby("method").size().to_dict()
    if counts != expected or len(df) != 408:
        raise ValueError(f"unexpected capped-probe counts: rows={len(df)} counts={counts}")
    cap_dir = out_root / "capped_rescale"
    cap_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cap_dir / "revised_snp_capped_rescale_probe_h512_24net_p50_80.csv", index=False)
    summary = summarize_network_cells(
        df,
        group_cols=["method", "pruning_pct"],
        extra_metrics=("candidate_edge_cap_value",),
    ).sort_values(["method", "pruning_pct"])
    summary.to_csv(cap_dir / "revised_snp_capped_rescale_probe_h512_24net_p50_80_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT,
        help="destination for the assembled artifacts. Defaults to the frozen revised_scope "
             "tree, so pass an alternative directory to regenerate without overwriting it.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_main(args.out_dir)
    write_capped(args.out_dir)
    print(f"Wrote revised-scope task artifacts under {args.out_dir}")


if __name__ == "__main__":
    main()
