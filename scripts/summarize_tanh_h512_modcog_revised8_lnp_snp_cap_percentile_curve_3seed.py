#!/usr/bin/env python3
"""Summarize full three-seed q10-to-q90 L-NP/S-NP cap-percentile curve."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


SEED0_SUITE_ID = (
    "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_plus_lnp_q90_"
    "1seed_full8_p50_80"
)
SEEDS12_SUITE_ID = (
    "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_q90_"
    "pruneseeds1_2_full8_p50_80"
)
Q50_SUITE_ID = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_q50_capped_rescale_3seed_full8_p50_80"
SNP_Q90_SUITE_ID = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_snp_capped_rescale_q50_q75_q90_full8_p50_80"
DEFAULT_SEED0_INPUT = Path(f"results/{SEED0_SUITE_ID}.csv")
DEFAULT_SEEDS12_INPUT = Path(f"results/{SEEDS12_SUITE_ID}.csv")
DEFAULT_Q50_INPUT = Path(f"results/{Q50_SUITE_ID}.csv")
DEFAULT_SNP_Q90_INPUT = Path(f"results/{SNP_Q90_SUITE_ID}.csv")
DEFAULT_MAIN_SUMMARY_INPUT = Path(
    "paper_artifacts/official_h512_24net/data/revised_scope/task_preservation/"
    "revised_task_preservation_h512_24net_p50_80_summary.csv"
)
DEFAULT_MAIN_RAW_INPUT = Path(
    "paper_artifacts/official_h512_24net/data/revised_scope/task_preservation/"
    "revised_task_preservation_h512_24net_p50_80.csv"
)
DEFAULT_COMBINED_OUTPUT = Path(
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_3seed_curve_rows.csv"
)
DEFAULT_SUMMARY_OUTPUT = Path(
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_3seed_curve_summary.csv"
)
DEFAULT_PLOT_POINTS_OUTPUT = Path(
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q0_to_q100_3seed_plot_points.csv"
)
DEFAULT_PLOT_POINTS_EXACT_OUTPUT = Path(
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q0_to_q100_3seed_plot_points_with_sem.csv"
)
DEFAULT_TESTS_OUTPUT = Path(
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_3seed_vs_uncapped_tests.csv"
)
OFFDIAG_EDGES = 512 * 511
ALL_QUANTILES = {round(q, 2) for q in np.arange(0.1, 1.0, 0.1)}
AMOUNTS = {50, 60, 70, 80}
METHOD_MAP = {
    "noise_prune_capped_rescale": "L-NP capped rescale",
    "simulation_noise_prune_capped_rescale": "S-NP capped rescale",
}
UNCAPPED_MAP = {
    "noise_prune": "L-NP uncapped rescale",
    "simulation_noise_prune_rescale": "S-NP uncapped rescale",
}


def sem(values: pd.Series) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def sd(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else np.nan


def completed_only(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "error" in df.columns:
        errors = df["error"].notna() & (df["error"].astype(str) != "")
        if errors.any():
            bad = df.loc[errors, "run_id"].head(10).tolist()
            raise ValueError(f"{path} contains failed runs, examples={bad}")
    return df


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return completed_only(pd.read_csv(path), path)


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


def capped_rows(seed0: pd.DataFrame, seeds12: pd.DataFrame, q50: pd.DataFrame, snp_q90: pd.DataFrame) -> pd.DataFrame:
    parts = [
        seed0[seed0["strategy"].isin(METHOD_MAP)].copy(),
        seeds12[seeds12["strategy"].isin(METHOD_MAP)].copy(),
        q50[
            q50["strategy"].isin(METHOD_MAP)
            & (pd.to_numeric(q50["prune_rescale_cap_quantile"], errors="coerce").round(6) == 0.5)
        ].copy(),
        snp_q90[
            (snp_q90["strategy"] == "simulation_noise_prune_capped_rescale")
            & (pd.to_numeric(snp_q90["pruning_seed"], errors="coerce") == 0)
            & (pd.to_numeric(snp_q90["prune_rescale_cap_quantile"], errors="coerce").round(6) == 0.9)
        ].copy(),
    ]
    work = pd.concat(parts, ignore_index=True, sort=False)
    work["cap_quantile"] = pd.to_numeric(work["prune_rescale_cap_quantile"], errors="coerce").round(2)
    work["cap_percentile"] = (work["cap_quantile"] * 100).round().astype(int)
    work["method"] = work["strategy"].map(METHOD_MAP)
    work["family"] = work["method"].str.replace(" capped rescale", "", regex=False)
    work["method_label"] = work.apply(
        lambda row: f"{row['family']} capped q{int(row['cap_percentile'])}",
        axis=1,
    )
    return add_common_fields(work)


def uncapped_rows(q50: pd.DataFrame) -> pd.DataFrame:
    work = q50[q50["strategy"].isin(UNCAPPED_MAP)].copy()
    work["method"] = work["strategy"].map(UNCAPPED_MAP)
    work["family"] = work["method"].str.replace(" uncapped rescale", "", regex=False)
    work["method_label"] = work["method"]
    return add_common_fields(work)


def add_common_fields(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    out["task_short"] = out["task"].astype(str).str.replace("modcog:", "", regex=False)
    out["pruning_pct"] = (pd.to_numeric(out["amount"], errors="coerce") * 100.0).round().astype(int)
    out["pruning_seed"] = pd.to_numeric(out["pruning_seed"], errors="coerce").astype(int)
    out["effective_nonzero_recurrent_density_offdiag"] = (
        pd.to_numeric(out["post_rec_weight_nz_count"], errors="coerce") / OFFDIAG_EDGES
    )
    out["post_rec_linear_rho_ratio_to_unpruned"] = (
        pd.to_numeric(out["post_rec_linear_rho"], errors="coerce")
        / pd.to_numeric(out["pre_rec_linear_rho"], errors="coerce")
    )
    return out


def validate_capped(work: pd.DataFrame) -> None:
    if len(work) != 5184:
        raise ValueError(f"expected 5184 capped rows, found {len(work)}")
    if set(work["pruning_seed"]) != {0, 1, 2}:
        raise ValueError(f"expected pruning seeds {{0,1,2}}, found {sorted(set(work['pruning_seed']))}")
    if set(work["pruning_pct"]) != AMOUNTS:
        raise ValueError(f"expected pruning percentages {sorted(AMOUNTS)}, found {sorted(set(work['pruning_pct']))}")
    for method in ("L-NP capped rescale", "S-NP capped rescale"):
        quantiles = set(work.loc[work["method"] == method, "cap_quantile"].round(2))
        if quantiles != ALL_QUANTILES:
            raise ValueError(f"{method} has quantiles {sorted(quantiles)}, expected {sorted(ALL_QUANTILES)}")
    counts = work.groupby(["method", "cap_quantile", "pruning_pct"]).size()
    bad = counts[counts != 72]
    if not bad.empty:
        raise ValueError(f"expected 72 raw rows per method x quantile x sparsity cell, bad counts={bad.to_dict()}")


def attach_retention(work: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    out = work.merge(baseline, on="source_model_label", how="left", validate="many_to_one")
    out["sequence_retention"] = pd.to_numeric(out["post_acc_sequence"], errors="coerce") / out[
        "baseline_acc_sequence"
    ].replace({0.0: np.nan})
    return out


def cell_means(work: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        work.groupby([*group_cols, "task_short", "source_network_seed"], as_index=False)
        .agg(
            source_model_label=("source_model_label", "first"),
            raw_run_count=("post_acc_sequence", "size"),
            post_acc_sequence=("post_acc_sequence", "mean"),
            sequence_retention=("sequence_retention", "mean"),
            effective_nonzero_recurrent_density_offdiag=("effective_nonzero_recurrent_density_offdiag", "mean"),
            candidate_edge_cap_value=("prune_rescale_cap_value", "mean"),
            post_rec_linear_rho=("post_rec_linear_rho", "mean"),
            post_rec_linear_rho_ratio_to_unpruned=("post_rec_linear_rho_ratio_to_unpruned", "mean"),
        )
    )


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "family", "method_label", "cap_quantile", "cap_percentile", "pruning_pct"]
    return (
        cells.groupby(group_cols, as_index=False)
        .agg(
            n=("sequence_retention", "count"),
            n_trained_networks=("source_model_label", "count"),
            raw_run_count=("raw_run_count", "sum"),
            pruning_seed_replicates_per_network_min=("raw_run_count", "min"),
            pruning_seed_replicates_per_network_max=("raw_run_count", "max"),
            sequence_retention_mean=("sequence_retention", "mean"),
            sequence_retention_sd=("sequence_retention", sd),
            sequence_retention_sem=("sequence_retention", sem),
            post_acc_sequence_mean=("post_acc_sequence", "mean"),
            post_acc_sequence_sd=("post_acc_sequence", sd),
            post_acc_sequence_sem=("post_acc_sequence", sem),
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
        .sort_values(["method", "cap_quantile", "pruning_pct"])
    )


def expected_holm(pvalues: pd.Series) -> pd.Series:
    order = pvalues.sort_values().index
    out = pd.Series(index=pvalues.index, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(order) - rank) * float(pvalues.loc[idx]))
        out.loc[idx] = min(running, 1.0)
    return out


def paired_tests(capped_cells: pd.DataFrame, uncapped_cells: pd.DataFrame) -> pd.DataFrame:
    uncapped = uncapped_cells.rename(
        columns={
            "sequence_retention": "uncapped_sequence_retention",
            "post_rec_linear_rho_ratio_to_unpruned": "uncapped_rho_ratio",
        }
    )
    records = []
    keys = ["family", "pruning_pct", "task_short", "source_network_seed"]
    for (family, quantile, pruning_pct), subset in capped_cells.groupby(
        ["family", "cap_quantile", "pruning_pct"]
    ):
        paired = subset.merge(
            uncapped[keys + ["uncapped_sequence_retention", "uncapped_rho_ratio"]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != 24:
            raise ValueError(f"expected 24 paired cells for {family} q{quantile} p{pruning_pct}, found {len(paired)}")
        diff = paired["sequence_retention"] - paired["uncapped_sequence_retention"]
        nonzero = diff[diff != 0.0]
        wins = int((diff > 0.0).sum())
        losses = int((diff < 0.0).sum())
        records.append(
            {
                "family": f"{family}_capped_vs_uncapped_p{int(pruning_pct)}",
                "method_family": family,
                "cap_quantile": float(quantile),
                "cap_percentile": int(round(float(quantile) * 100)),
                "pruning_pct": int(pruning_pct),
                "metric": "sequence_retention",
                "unit": "task_network_cell_after_pruning_seed_mean",
                "test_name": "paired Wilcoxon signed-rank test",
                "alternative": "two-sided",
                "tails": 2,
                "alpha": 0.05,
                "multiple_comparison_method": "Holm correction within method-family and pruning-level family",
                "normality_required": False,
                "robustness_test_name": "exact binomial sign test",
                "n": int(len(diff)),
                "n_nonzero_differences": int(len(nonzero)),
                "mean_capped": float(paired["sequence_retention"].mean()),
                "mean_uncapped": float(paired["uncapped_sequence_retention"].mean()),
                "mean_diff_capped_minus_uncapped": float(diff.mean()),
                "median_diff_capped_minus_uncapped": float(diff.median()),
                "wins_capped_gt_uncapped": wins,
                "losses_capped_lt_uncapped": losses,
                "ties": int((diff == 0.0).sum()),
                "wilcoxon_p": float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue),
                "sign_test_p": float(binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue)
                if wins + losses
                else 1.0,
            }
        )
    out = pd.DataFrame(records).sort_values(["family", "cap_quantile"]).reset_index(drop=True)
    out["holm_p_within_family"] = np.nan
    out["sign_test_holm_p_within_family"] = np.nan
    for family, idx in out.groupby("family").groups.items():
        out.loc[idx, "holm_p_within_family"] = expected_holm(out.loc[idx, "wilcoxon_p"])
        out.loc[idx, "sign_test_holm_p_within_family"] = expected_holm(out.loc[idx, "sign_test_p"])
    out["reject_holm_alpha_0_05"] = out["holm_p_within_family"] < 0.05
    out["sign_test_reject_holm_alpha_0_05"] = out["sign_test_holm_p_within_family"] < 0.05
    return out


def plot_points(summary: pd.DataFrame, uncapped_cells: pd.DataFrame, main_summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for family, subset in summary.groupby("family"):
        for quantile, group in subset.groupby("cap_percentile"):
            records.append(
                {
                    "family": family,
                    "plot_x": int(quantile),
                    "plot_label": f"q{int(quantile)}",
                    "point_type": "cap_quantile",
                    "source": "three_pruning_seed_curve",
                    "mean_retention_across_p50_80": group["sequence_retention_mean"].mean(),
                    "mean_rho_ratio_across_p50_80": group["post_rec_linear_rho_ratio_to_unpruned_mean"].mean(),
                }
            )
    for family, group in uncapped_cells.groupby("family"):
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
                "source": "matched_q50_suite_uncapped_3_pruning_seeds",
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
                "mean_rho_ratio_across_p50_80": group["post_rec_linear_rho_ratio_to_unpruned_mean"].mean(),
            }
        )
    return pd.DataFrame(records).sort_values(["family", "plot_x"]).reset_index(drop=True)


def mask_cells(main_raw: pd.DataFrame) -> pd.DataFrame:
    work = main_raw[main_raw["method"].isin({"L-NP mask", "S-NP mask"})].copy()
    if work.empty:
        raise ValueError("main raw input contains no L-NP/S-NP mask rows")
    work["family"] = work["method"].map({"L-NP mask": "L-NP", "S-NP mask": "S-NP"})
    work["plot_x"] = 0
    work["plot_label"] = "q0 mask-only"
    work["point_type"] = "mask_only_not_cap_quantile"
    work["source"] = "canonical_main_mask_raw"
    work["raw_run_count"] = 1
    return (
        work.groupby(
            ["family", "plot_x", "plot_label", "point_type", "source", "pruning_pct", "task_short", "source_network_seed"],
            as_index=False,
        )
        .agg(
            source_model_label=("source_model_label", "first"),
            raw_run_count=("raw_run_count", "sum"),
            sequence_retention=("sequence_retention", "mean"),
            post_rec_linear_rho_ratio_to_unpruned=("post_rec_linear_rho_ratio_to_unpruned", "mean"),
        )
    )


def exact_plot_points(
    capped_cells: pd.DataFrame,
    uncapped_cells: pd.DataFrame,
    main_raw: pd.DataFrame,
) -> pd.DataFrame:
    capped = capped_cells.copy()
    capped["plot_x"] = capped["cap_percentile"].astype(int)
    capped["plot_label"] = "q" + capped["plot_x"].astype(str)
    capped["point_type"] = "cap_quantile"
    capped["source"] = "three_pruning_seed_curve"

    uncapped = uncapped_cells.copy()
    uncapped["plot_x"] = 100
    uncapped["plot_label"] = "q100 no cap"
    uncapped["point_type"] = "uncapped_rescale"
    uncapped["source"] = "matched_q50_suite_uncapped_3_pruning_seeds"

    masks = mask_cells(main_raw)
    point_cells = pd.concat(
        [
            capped[
                [
                    "family",
                    "plot_x",
                    "plot_label",
                    "point_type",
                    "source",
                    "pruning_pct",
                    "task_short",
                    "source_network_seed",
                    "source_model_label",
                    "raw_run_count",
                    "sequence_retention",
                    "post_rec_linear_rho_ratio_to_unpruned",
                ]
            ],
            uncapped[
                [
                    "family",
                    "plot_x",
                    "plot_label",
                    "point_type",
                    "source",
                    "pruning_pct",
                    "task_short",
                    "source_network_seed",
                    "source_model_label",
                    "raw_run_count",
                    "sequence_retention",
                    "post_rec_linear_rho_ratio_to_unpruned",
                ]
            ],
            masks,
        ],
        ignore_index=True,
        sort=False,
    )

    per_network = (
        point_cells.groupby(
            ["family", "plot_x", "plot_label", "point_type", "source", "source_model_label"],
            as_index=False,
        )
        .agg(
            task_short=("task_short", "first"),
            source_network_seed=("source_network_seed", "first"),
            sparsity_level_count=("pruning_pct", "nunique"),
            raw_run_count=("raw_run_count", "sum"),
            sequence_retention_across_p50_80=("sequence_retention", "mean"),
            post_rec_linear_rho_ratio_to_unpruned_across_p50_80=(
                "post_rec_linear_rho_ratio_to_unpruned",
                "mean",
            ),
        )
    )
    bad = per_network[per_network["sparsity_level_count"] != 4]
    if not bad.empty:
        raise ValueError(f"expected four sparsity levels per plotted network cell, examples={bad.head().to_dict('records')}")

    out = (
        per_network.groupby(["family", "plot_x", "plot_label", "point_type", "source"], as_index=False)
        .agg(
            n=("sequence_retention_across_p50_80", "count"),
            raw_run_count=("raw_run_count", "sum"),
            sparsity_level_count_min=("sparsity_level_count", "min"),
            sparsity_level_count_max=("sparsity_level_count", "max"),
            mean_retention_across_p50_80=("sequence_retention_across_p50_80", "mean"),
            sd_retention_across_p50_80=("sequence_retention_across_p50_80", sd),
            sem_retention_across_p50_80=("sequence_retention_across_p50_80", sem),
            mean_rho_ratio_across_p50_80=(
                "post_rec_linear_rho_ratio_to_unpruned_across_p50_80",
                "mean",
            ),
            sd_rho_ratio_across_p50_80=(
                "post_rec_linear_rho_ratio_to_unpruned_across_p50_80",
                sd,
            ),
            sem_rho_ratio_across_p50_80=(
                "post_rec_linear_rho_ratio_to_unpruned_across_p50_80",
                sem,
            ),
        )
        .sort_values(["family", "plot_x"])
        .reset_index(drop=True)
    )
    if set(out["n"].astype(int)) != {24}:
        raise ValueError(f"expected n=24 for every exact plot point, found {sorted(set(out['n']))}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-input", type=Path, default=DEFAULT_SEED0_INPUT)
    parser.add_argument("--seeds12-input", type=Path, default=DEFAULT_SEEDS12_INPUT)
    parser.add_argument("--q50-input", type=Path, default=DEFAULT_Q50_INPUT)
    parser.add_argument("--snp-q90-input", type=Path, default=DEFAULT_SNP_Q90_INPUT)
    parser.add_argument("--main-summary-input", type=Path, default=DEFAULT_MAIN_SUMMARY_INPUT)
    parser.add_argument("--main-raw-input", type=Path, default=DEFAULT_MAIN_RAW_INPUT)
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--plot-points-output", type=Path, default=DEFAULT_PLOT_POINTS_OUTPUT)
    parser.add_argument("--plot-points-exact-output", type=Path, default=DEFAULT_PLOT_POINTS_EXACT_OUTPUT)
    parser.add_argument("--tests-output", type=Path, default=DEFAULT_TESTS_OUTPUT)
    args = parser.parse_args()

    seed0 = read_required(args.seed0_input)
    seeds12 = read_required(args.seeds12_input)
    q50 = read_required(args.q50_input)
    snp_q90 = read_required(args.snp_q90_input)
    main_summary = pd.read_csv(args.main_summary_input)
    main_raw = pd.read_csv(args.main_raw_input)
    baseline = baseline_table([seed0, seeds12, q50, snp_q90])

    capped = attach_retention(capped_rows(seed0, seeds12, q50, snp_q90), baseline)
    uncapped = attach_retention(uncapped_rows(q50), baseline)
    validate_capped(capped)
    capped_cells = cell_means(
        capped,
        ["method", "family", "method_label", "cap_quantile", "cap_percentile", "pruning_pct"],
    )
    uncapped_cells = cell_means(uncapped, ["method", "family", "method_label", "pruning_pct"])
    summary = summarize(capped_cells)
    tests = paired_tests(capped_cells, uncapped_cells)
    points = plot_points(summary, uncapped_cells, main_summary)
    exact_points = exact_plot_points(capped_cells, uncapped_cells, main_raw)

    for path in (
        args.combined_output,
        args.summary_output,
        args.plot_points_output,
        args.plot_points_exact_output,
        args.tests_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    capped.to_csv(args.combined_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    points.to_csv(args.plot_points_output, index=False)
    exact_points.to_csv(args.plot_points_exact_output, index=False)
    tests.to_csv(args.tests_output, index=False)
    print(f"wrote {args.combined_output}")
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.plot_points_output}")
    print(f"wrote {args.plot_points_exact_output}")
    print(f"wrote {args.tests_output}")


if __name__ == "__main__":
    main()
