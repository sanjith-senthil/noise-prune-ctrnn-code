#!/usr/bin/env python3
"""Validate and summarize the matched L-NP/S-NP q50 capped-rescale suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SUITE_ID = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_q50_capped_rescale_3seed_full8_p50_80"
DEFAULT_INPUT = Path(f"results/{SUITE_ID}.csv")
DEFAULT_SUMMARY = Path(f"results/{SUITE_ID}_summary.csv")
DEFAULT_DIFFS = Path(f"results/{SUITE_ID}_paired_diffs.csv")
OFFDIAG_EDGES = 512 * 511
METHOD_MAP = {
    "noise_prune": "L-NP rescale",
    "noise_prune_capped_rescale": "L-NP capped q50",
    "simulation_noise_prune_rescale": "S-NP rescale",
    "simulation_noise_prune_capped_rescale": "S-NP capped q50",
}
EXPECTED_COUNTS = {"none": 24, **{strategy: 288 for strategy in METHOD_MAP}}


def sem(values: pd.Series) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def sd(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else np.nan


def validate(df: pd.DataFrame) -> None:
    counts = df.groupby("strategy").size().to_dict()
    if counts != EXPECTED_COUNTS or len(df) != 1176:
        raise ValueError(f"incomplete or unexpected suite: rows={len(df)} strategy_counts={counts}")
    capped = df[df["strategy"].isin({"noise_prune_capped_rescale", "simulation_noise_prune_capped_rescale"})]
    quantiles = set(pd.to_numeric(capped["prune_rescale_cap_quantile"], errors="coerce").dropna().round(6))
    if quantiles != {0.5}:
        raise ValueError(f"expected q50 capped rows, found quantiles={sorted(quantiles)}")
    if set(pd.to_numeric(df["pruning_seed"], errors="coerce").dropna().astype(int)) != {0, 1, 2}:
        raise ValueError("expected pruning seeds {0,1,2}")
    if set((pd.to_numeric(df.loc[df["strategy"] != "none", "amount"]) * 100).round().astype(int)) != {50, 60, 70, 80}:
        raise ValueError("expected pruning percentages {50,60,70,80}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--diff-output", type=Path, default=DEFAULT_DIFFS)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    validate(df)
    baseline = (
        df[df["strategy"] == "none"][["source_model_label", "post_acc_sequence"]]
        .drop_duplicates("source_model_label")
        .rename(columns={"post_acc_sequence": "baseline_acc_sequence"})
    )
    work = df[df["strategy"] != "none"].copy()
    work["method"] = work["strategy"].map(METHOD_MAP)
    work["task_short"] = work["task"].astype(str).str.replace("modcog:", "", regex=False)
    work["pruning_pct"] = (work["amount"].astype(float) * 100.0).round().astype(int)
    work["effective_nonzero_recurrent_density_offdiag"] = work["post_rec_weight_nz_count"].astype(float) / OFFDIAG_EDGES
    work["post_rec_linear_rho_ratio_to_unpruned"] = (
        work["post_rec_linear_rho"].astype(float) / work["pre_rec_linear_rho"].astype(float)
    )
    work = work.merge(baseline, on="source_model_label", how="left", validate="many_to_one")
    work["sequence_retention"] = work["post_acc_sequence"] / work["baseline_acc_sequence"].replace({0.0: np.nan})

    # Pruning seeds are technical replicates. Average them within each trained
    # network before computing paper-facing SEM across the 24 trained networks.
    cell = (
        work.groupby(["method", "pruning_pct", "task_short", "source_network_seed"], as_index=False)
        .agg(
            raw_run_count=("post_acc_sequence", "size"),
            post_acc_sequence=("post_acc_sequence", "mean"),
            sequence_retention=("sequence_retention", "mean"),
            effective_nonzero_recurrent_density_offdiag=("effective_nonzero_recurrent_density_offdiag", "mean"),
            candidate_edge_cap_value=("prune_rescale_cap_value", "mean"),
            post_rec_linear_rho=("post_rec_linear_rho", "mean"),
            post_rec_linear_rho_ratio_to_unpruned=("post_rec_linear_rho_ratio_to_unpruned", "mean"),
        )
    )
    summary = (
        cell.groupby(["method", "pruning_pct"], as_index=False)
        .agg(
            n=("post_acc_sequence", "count"),
            n_trained_networks=("post_acc_sequence", "count"),
            raw_run_count=("raw_run_count", "sum"),
            pruning_seed_replicates_per_network_min=("raw_run_count", "min"),
            pruning_seed_replicates_per_network_max=("raw_run_count", "max"),
            post_acc_sequence_mean=("post_acc_sequence", "mean"),
            post_acc_sequence_sd=("post_acc_sequence", sd),
            post_acc_sequence_sem=("post_acc_sequence", sem),
            sequence_retention_mean=("sequence_retention", "mean"),
            sequence_retention_sd=("sequence_retention", sd),
            sequence_retention_sem=("sequence_retention", sem),
            effective_nonzero_recurrent_density_offdiag_mean=("effective_nonzero_recurrent_density_offdiag", "mean"),
            effective_nonzero_recurrent_density_offdiag_sd=("effective_nonzero_recurrent_density_offdiag", sd),
            effective_nonzero_recurrent_density_offdiag_sem=("effective_nonzero_recurrent_density_offdiag", sem),
            candidate_edge_cap_value_mean=("candidate_edge_cap_value", "mean"),
            candidate_edge_cap_value_sd=("candidate_edge_cap_value", sd),
            candidate_edge_cap_value_sem=("candidate_edge_cap_value", sem),
            post_rec_linear_rho_mean=("post_rec_linear_rho", "mean"),
            post_rec_linear_rho_sd=("post_rec_linear_rho", sd),
            post_rec_linear_rho_sem=("post_rec_linear_rho", sem),
            post_rec_linear_rho_ratio_to_unpruned_mean=("post_rec_linear_rho_ratio_to_unpruned", "mean"),
            post_rec_linear_rho_ratio_to_unpruned_sd=("post_rec_linear_rho_ratio_to_unpruned", sd),
            post_rec_linear_rho_ratio_to_unpruned_sem=("post_rec_linear_rho_ratio_to_unpruned", sem),
        )
        .sort_values(["method", "pruning_pct"])
    )
    task_cell = (
        cell.groupby(["method", "pruning_pct", "task_short"], as_index=False)
        .agg(
            post_acc_sequence=("post_acc_sequence", "mean"),
            sequence_retention=("sequence_retention", "mean"),
            effective_nonzero_recurrent_density_offdiag=("effective_nonzero_recurrent_density_offdiag", "mean"),
            candidate_edge_cap_value=("candidate_edge_cap_value", "mean"),
            post_rec_linear_rho=("post_rec_linear_rho", "mean"),
            post_rec_linear_rho_ratio_to_unpruned=("post_rec_linear_rho_ratio_to_unpruned", "mean"),
        )
    )
    task_summary = (
        task_cell.groupby(["method", "pruning_pct"], as_index=False)
        .agg(
            n_tasks=("task_short", "count"),
            post_acc_sequence_taskmean_sd=("post_acc_sequence", sd),
            post_acc_sequence_taskmean_sem=("post_acc_sequence", sem),
            sequence_retention_taskmean_sd=("sequence_retention", sd),
            sequence_retention_taskmean_sem=("sequence_retention", sem),
            effective_nonzero_recurrent_density_offdiag_taskmean_sd=("effective_nonzero_recurrent_density_offdiag", sd),
            effective_nonzero_recurrent_density_offdiag_taskmean_sem=("effective_nonzero_recurrent_density_offdiag", sem),
            candidate_edge_cap_value_taskmean_sd=("candidate_edge_cap_value", sd),
            candidate_edge_cap_value_taskmean_sem=("candidate_edge_cap_value", sem),
            post_rec_linear_rho_taskmean_sd=("post_rec_linear_rho", sd),
            post_rec_linear_rho_taskmean_sem=("post_rec_linear_rho", sem),
            post_rec_linear_rho_ratio_to_unpruned_taskmean_sd=("post_rec_linear_rho_ratio_to_unpruned", sd),
            post_rec_linear_rho_ratio_to_unpruned_taskmean_sem=("post_rec_linear_rho_ratio_to_unpruned", sem),
        )
    )
    summary = summary.merge(
        task_summary,
        on=["method", "pruning_pct"],
        how="left",
        validate="one_to_one",
    )
    key = ["task_short", "source_network_seed", "pruning_pct", "pruning_seed"]
    wide = work.pivot(index=key, columns="method", values=["post_acc_sequence", "effective_nonzero_recurrent_density_offdiag"])
    diffs = []
    for family in ("L-NP", "S-NP"):
        capped = f"{family} capped q50"
        uncapped = f"{family} rescale"
        part = pd.DataFrame(index=wide.index)
        part["family"] = family
        part["capped_post_acc_sequence"] = wide["post_acc_sequence", capped]
        part["uncapped_post_acc_sequence"] = wide["post_acc_sequence", uncapped]
        part["delta_post_acc_sequence_capped_minus_uncapped"] = (
            part["capped_post_acc_sequence"] - part["uncapped_post_acc_sequence"]
        )
        part["capped_effective_nonzero_density_offdiag"] = wide["effective_nonzero_recurrent_density_offdiag", capped]
        part["uncapped_effective_nonzero_density_offdiag"] = wide["effective_nonzero_recurrent_density_offdiag", uncapped]
        part["delta_effective_nonzero_density_offdiag_capped_minus_uncapped"] = (
            part["capped_effective_nonzero_density_offdiag"] - part["uncapped_effective_nonzero_density_offdiag"]
        )
        diffs.append(part.reset_index())
    paired = pd.concat(diffs, ignore_index=True)
    if len(paired) != 576:
        raise ValueError(f"expected 576 paired delta rows, found {len(paired)}")

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.diff_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    paired.to_csv(args.diff_output, index=False)
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.diff_output}")


if __name__ == "__main__":
    main()
