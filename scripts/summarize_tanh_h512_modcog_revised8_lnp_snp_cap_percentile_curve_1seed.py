#!/usr/bin/env python3
"""Summarize one-seed q10-to-q90 L-NP/S-NP capped-rescale curve data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MISSING_SUITE_ID = (
    "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_plus_lnp_q90_"
    "1seed_full8_p50_80"
)
Q50_SUITE_ID = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_q50_capped_rescale_3seed_full8_p50_80"
SNP_Q90_SUITE_ID = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_snp_capped_rescale_q50_q75_q90_full8_p50_80"
DEFAULT_MISSING_INPUT = Path(f"results/{MISSING_SUITE_ID}.csv")
DEFAULT_Q50_INPUT = Path(f"results/{Q50_SUITE_ID}.csv")
DEFAULT_SNP_Q90_INPUT = Path(f"results/{SNP_Q90_SUITE_ID}.csv")
DEFAULT_MAIN_SUMMARY_INPUT = Path(
    "paper_artifacts/official_h512_24net/data/revised_scope/task_preservation/"
    "revised_task_preservation_h512_24net_p50_80_summary.csv"
)
DEFAULT_COMBINED_OUTPUT = Path(
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_1seed_curve_rows.csv"
)
DEFAULT_SUMMARY_OUTPUT = Path(
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_1seed_curve_summary.csv"
)
DEFAULT_PLOT_POINTS_OUTPUT = Path(
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q0_to_q100_1seed_plot_points.csv"
)
OFFDIAG_EDGES = 512 * 511
ALL_QUANTILES = {round(q, 2) for q in np.arange(0.1, 1.0, 0.1)}
AMOUNTS = {50, 60, 70, 80}


def sem(values: pd.Series) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def sd(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else np.nan


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def completed_only(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "error" in df.columns:
        errors = df["error"].notna() & (df["error"].astype(str) != "")
        if errors.any():
            bad = df.loc[errors, "run_id"].head(10).tolist()
            raise ValueError(f"{path} contains failed runs, examples={bad}")
    return df


def baseline_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = pd.concat(
        [
            df[df["strategy"] == "none"][["source_model_label", "post_acc_sequence"]]
            for df in frames
        ],
        ignore_index=True,
    )
    base = base.drop_duplicates("source_model_label").rename(
        columns={"post_acc_sequence": "baseline_acc_sequence"}
    )
    if len(base) != 24:
        raise ValueError(f"expected 24 unique baselines, found {len(base)}")
    return base


def select_rows(missing: pd.DataFrame, q50: pd.DataFrame, snp_q90: pd.DataFrame) -> pd.DataFrame:
    parts = []
    missing_keep = missing[
        missing["strategy"].isin({"noise_prune_capped_rescale", "simulation_noise_prune_capped_rescale"})
    ].copy()
    parts.append(missing_keep)

    q50_keep = q50[
        q50["strategy"].isin({"noise_prune_capped_rescale", "simulation_noise_prune_capped_rescale"})
        & (pd.to_numeric(q50["pruning_seed"], errors="coerce") == 0)
        & (pd.to_numeric(q50["prune_rescale_cap_quantile"], errors="coerce").round(6) == 0.5)
    ].copy()
    parts.append(q50_keep)

    snp_q90_keep = snp_q90[
        (snp_q90["strategy"] == "simulation_noise_prune_capped_rescale")
        & (pd.to_numeric(snp_q90["pruning_seed"], errors="coerce") == 0)
        & (pd.to_numeric(snp_q90["prune_rescale_cap_quantile"], errors="coerce").round(6) == 0.9)
    ].copy()
    parts.append(snp_q90_keep)

    work = pd.concat(parts, ignore_index=True, sort=False)
    work["cap_quantile"] = pd.to_numeric(work["prune_rescale_cap_quantile"], errors="coerce").round(2)
    work["cap_percentile"] = (work["cap_quantile"] * 100).round().astype(int)
    work["method"] = work["strategy"].map(
        {
            "noise_prune_capped_rescale": "L-NP capped rescale",
            "simulation_noise_prune_capped_rescale": "S-NP capped rescale",
        }
    )
    work["method_label"] = work.apply(
        lambda row: f"{row['method'].replace(' capped rescale', '')} capped q{int(row['cap_percentile'])}",
        axis=1,
    )
    work["task_short"] = work["task"].astype(str).str.replace("modcog:", "", regex=False)
    work["pruning_pct"] = (pd.to_numeric(work["amount"], errors="coerce") * 100.0).round().astype(int)
    work["effective_nonzero_recurrent_density_offdiag"] = (
        pd.to_numeric(work["post_rec_weight_nz_count"], errors="coerce") / OFFDIAG_EDGES
    )
    work["post_rec_linear_rho_ratio_to_unpruned"] = (
        pd.to_numeric(work["post_rec_linear_rho"], errors="coerce")
        / pd.to_numeric(work["pre_rec_linear_rho"], errors="coerce")
    )
    return work


def validate_curve(work: pd.DataFrame) -> None:
    if len(work) != 1728:
        raise ValueError(f"expected 1728 capped rows for two methods x nine quantiles x four sparsities x 24 networks; found {len(work)}")
    if set(pd.to_numeric(work["pruning_seed"], errors="coerce").astype(int)) != {0}:
        raise ValueError("curve summary expects one pruning seed: pruning_seed=0")
    if set(work["pruning_pct"]) != AMOUNTS:
        raise ValueError(f"expected pruning percentages {sorted(AMOUNTS)}, found {sorted(set(work['pruning_pct']))}")
    for method in ("L-NP capped rescale", "S-NP capped rescale"):
        quantiles = set(work.loc[work["method"] == method, "cap_quantile"].round(2))
        if quantiles != ALL_QUANTILES:
            raise ValueError(f"{method} has quantiles {sorted(quantiles)}, expected {sorted(ALL_QUANTILES)}")
    counts = work.groupby(["method", "cap_quantile", "pruning_pct"]).size()
    bad = counts[counts != 24]
    if not bad.empty:
        raise ValueError(f"expected 24 rows per method x quantile x sparsity cell, bad counts={bad.to_dict()}")


def build_plot_points(summary: pd.DataFrame, q50: pd.DataFrame, main_summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method, family in (
        ("L-NP capped rescale", "L-NP"),
        ("S-NP capped rescale", "S-NP"),
    ):
        sub = summary[summary["method"] == method]
        for q, group in sub.groupby("cap_percentile"):
            records.append(
                {
                    "family": family,
                    "plot_x": int(q),
                    "plot_label": f"q{int(q)}",
                    "point_type": "cap_quantile",
                    "source": "one_pruning_seed_curve",
                    "mean_retention_across_p50_80": group["sequence_retention_mean"].mean(),
                    "mean_rho_ratio_across_p50_80": group[
                        "post_rec_linear_rho_ratio_to_unpruned_mean"
                    ].mean(),
                }
            )

    q50_base = (
        q50[q50["strategy"] == "none"][["source_model_label", "post_acc_sequence"]]
        .drop_duplicates("source_model_label")
        .rename(columns={"post_acc_sequence": "baseline_acc_sequence"})
    )
    q100 = q50[
        q50["strategy"].isin({"noise_prune", "simulation_noise_prune_rescale"})
        & (pd.to_numeric(q50["pruning_seed"], errors="coerce") == 0)
    ].copy()
    q100 = q100.merge(q50_base, on="source_model_label", how="left", validate="many_to_one")
    q100["sequence_retention"] = pd.to_numeric(q100["post_acc_sequence"], errors="coerce") / q100[
        "baseline_acc_sequence"
    ].replace({0.0: np.nan})
    q100["post_rec_linear_rho_ratio_to_unpruned"] = (
        pd.to_numeric(q100["post_rec_linear_rho"], errors="coerce")
        / pd.to_numeric(q100["pre_rec_linear_rho"], errors="coerce")
    )
    q100["pruning_pct"] = (pd.to_numeric(q100["amount"], errors="coerce") * 100.0).round().astype(int)
    q100["family"] = q100["strategy"].map({"noise_prune": "L-NP", "simulation_noise_prune_rescale": "S-NP"})
    for family, group in q100.groupby("family"):
        by_sparsity = group.groupby("pruning_pct", as_index=False).agg(
            sequence_retention_mean=("sequence_retention", "mean"),
            rho_ratio_mean=("post_rec_linear_rho_ratio_to_unpruned", "mean"),
        )
        records.append(
            {
                "family": family,
                "plot_x": 100,
                "plot_label": "q100 no cap",
                "point_type": "uncapped_rescale",
                "source": "matched_q50_suite_pruning_seed0_uncapped",
                "mean_retention_across_p50_80": by_sparsity["sequence_retention_mean"].mean(),
                "mean_rho_ratio_across_p50_80": by_sparsity["rho_ratio_mean"].mean(),
            }
        )

    for method, family in (("L-NP mask", "L-NP"), ("S-NP mask", "S-NP")):
        group = main_summary[main_summary["method"] == method]
        if len(group) != 4:
            raise ValueError(f"expected four mask summary rows for {method}, found {len(group)}")
        records.append(
            {
                "family": family,
                "plot_x": 0,
                "plot_label": "q0 mask-only",
                "point_type": "mask_only_not_cap_quantile",
                "source": "canonical_main_mask_summary",
                "mean_retention_across_p50_80": group["sequence_retention_mean"].mean(),
                "mean_rho_ratio_across_p50_80": group[
                    "post_rec_linear_rho_ratio_to_unpruned_mean"
                ].mean(),
            }
        )

    out = pd.DataFrame(records).sort_values(["family", "plot_x"]).reset_index(drop=True)
    counts = out.groupby("family").size().to_dict()
    if counts != {"L-NP": 11, "S-NP": 11}:
        raise ValueError(f"unexpected plot-point counts: {counts}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing-input", type=Path, default=DEFAULT_MISSING_INPUT)
    parser.add_argument("--q50-input", type=Path, default=DEFAULT_Q50_INPUT)
    parser.add_argument("--snp-q90-input", type=Path, default=DEFAULT_SNP_Q90_INPUT)
    parser.add_argument("--main-summary-input", type=Path, default=DEFAULT_MAIN_SUMMARY_INPUT)
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--plot-points-output", type=Path, default=DEFAULT_PLOT_POINTS_OUTPUT)
    args = parser.parse_args()

    missing = completed_only(read_required(args.missing_input), args.missing_input)
    q50 = completed_only(read_required(args.q50_input), args.q50_input)
    snp_q90 = completed_only(read_required(args.snp_q90_input), args.snp_q90_input)
    main_summary = read_required(args.main_summary_input)
    baseline = baseline_table([missing, q50, snp_q90])
    work = select_rows(missing, q50, snp_q90)
    work = work.merge(baseline, on="source_model_label", how="left", validate="many_to_one")
    work["sequence_retention"] = pd.to_numeric(work["post_acc_sequence"], errors="coerce") / work[
        "baseline_acc_sequence"
    ].replace({0.0: np.nan})
    validate_curve(work)

    summary = (
        work.groupby(["method", "method_label", "cap_quantile", "cap_percentile", "pruning_pct"], as_index=False)
        .agg(
            n=("sequence_retention", "count"),
            n_trained_networks=("source_model_label", "nunique"),
            raw_run_count=("run_id", "count"),
            sequence_retention_mean=("sequence_retention", "mean"),
            sequence_retention_sd=("sequence_retention", sd),
            sequence_retention_sem=("sequence_retention", sem),
            post_acc_sequence_mean=("post_acc_sequence", "mean"),
            post_acc_sequence_sd=("post_acc_sequence", sd),
            post_acc_sequence_sem=("post_acc_sequence", sem),
            effective_nonzero_recurrent_density_offdiag_mean=("effective_nonzero_recurrent_density_offdiag", "mean"),
            effective_nonzero_recurrent_density_offdiag_sd=("effective_nonzero_recurrent_density_offdiag", sd),
            effective_nonzero_recurrent_density_offdiag_sem=("effective_nonzero_recurrent_density_offdiag", sem),
            candidate_edge_cap_value_mean=("prune_rescale_cap_value", "mean"),
            candidate_edge_cap_value_sd=("prune_rescale_cap_value", sd),
            candidate_edge_cap_value_sem=("prune_rescale_cap_value", sem),
            post_rec_linear_rho_mean=("post_rec_linear_rho", "mean"),
            post_rec_linear_rho_sd=("post_rec_linear_rho", sd),
            post_rec_linear_rho_sem=("post_rec_linear_rho", sem),
            post_rec_linear_rho_ratio_to_unpruned_mean=("post_rec_linear_rho_ratio_to_unpruned", "mean"),
            post_rec_linear_rho_ratio_to_unpruned_sd=("post_rec_linear_rho_ratio_to_unpruned", sd),
            post_rec_linear_rho_ratio_to_unpruned_sem=("post_rec_linear_rho_ratio_to_unpruned", sem),
        )
        .sort_values(["method", "cap_quantile", "pruning_pct"])
    )

    args.combined_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.plot_points_output.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(args.combined_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    build_plot_points(summary, q50, main_summary).to_csv(args.plot_points_output, index=False)
    print(f"wrote {args.combined_output}")
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.plot_points_output}")


if __name__ == "__main__":
    main()
