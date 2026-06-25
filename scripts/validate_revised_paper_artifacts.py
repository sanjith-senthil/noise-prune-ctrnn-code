#!/usr/bin/env python3
"""Independently validate the canonical revised-paper artifact set."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_artifacts/official_h512_24net/data"
REVISED = DATA / "revised_scope"
MAIN = REVISED / "task_preservation/revised_task_preservation_h512_24net_p50_80.csv"
MAIN_SUMMARY = REVISED / "task_preservation/revised_task_preservation_h512_24net_p50_80_summary.csv"
TASK_SPECS = REVISED / "task_preservation/modcog_revised8_task_specifications.csv"
CAP = REVISED / "capped_rescale/revised_snp_capped_rescale_probe_h512_24net_p50_80.csv"
CAP_SUMMARY = REVISED / "capped_rescale/revised_snp_capped_rescale_probe_h512_24net_p50_80_summary.csv"
MATCH = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_q50_capped_rescale_3seed_full8_p50_80.csv"
)
MATCH_SUMMARY = MATCH.with_name(f"{MATCH.stem}_summary.csv")
CURVE_MISSING = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_plus_lnp_q90_"
    "1seed_full8_p50_80.csv"
)
CURVE_ROWS = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_1seed_curve_rows.csv"
)
CURVE_SUMMARY = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_1seed_curve_summary.csv"
)
CURVE_PLOT_POINTS = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q0_to_q100_1seed_plot_points.csv"
)
CURVE_MISSING_SEEDS12 = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_q90_"
    "pruneseeds1_2_full8_p50_80.csv"
)
CURVE_ROWS_3SEED = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_3seed_curve_rows.csv"
)
CURVE_SUMMARY_3SEED = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_3seed_curve_summary.csv"
)
CURVE_PLOT_POINTS_3SEED = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q0_to_q100_3seed_plot_points.csv"
)
CURVE_PLOT_POINTS_3SEED_WITH_SEM = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q0_to_q100_3seed_plot_points_with_sem.csv"
)
CURVE_TESTS_3SEED = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_3seed_vs_uncapped_tests.csv"
)
CURVE_PLOT_TESTS_3SEED = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_3seed_vs_q100_plot_level_holm_tests.csv"
)
TESTS = REVISED / "significance/revised_scope_task_only_significance_tests.csv"
MEANS = REVISED / "significance/revised_scope_task_only_significance_means.csv"
RESCALE_DIR = DATA / "rescale_value_distributions"
RESCALE_RUN = RESCALE_DIR / "rescale_value_run_summary_snp_lnp_p80.csv"
RESCALE_BY_RUN = RESCALE_DIR / "rescale_value_histogram_snp_lnp_p80_by_run.csv"
RESCALE_OVERALL = RESCALE_DIR / "rescale_value_histogram_snp_lnp_p80_overall.csv"
RESCALE_BY_RUN_WIDE = RESCALE_DIR / "rescale_value_histogram_snp_lnp_p80_by_run_logbin0p20.csv"
RESCALE_OVERALL_WIDE = RESCALE_DIR / "rescale_value_histogram_snp_lnp_p80_overall_logbin0p20.csv"
MANIFEST = REVISED / "artifact_manifest.json"
OFFDIAG_EDGES = 512 * 511
FULL_EDGES = 512 * 512
SUMMARY_METRICS = (
    "post_acc_sequence",
    "sequence_retention",
    "effective_nonzero_recurrent_density_offdiag",
    "post_rec_linear_rho",
    "post_rec_linear_rho_ratio_to_unpruned",
)
MATCH_METHODS = {
    "noise_prune": "L-NP rescale",
    "noise_prune_capped_rescale": "L-NP capped q50",
    "simulation_noise_prune_rescale": "S-NP rescale",
    "simulation_noise_prune_capped_rescale": "S-NP capped q50",
}
CURVE_QUANTILES = {round(q, 2) for q in np.arange(0.1, 1.0, 0.1)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_close(actual: object, expected: object, label: str, *, atol: float = 1e-12) -> None:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    if not np.allclose(left, right, rtol=1e-10, atol=atol, equal_nan=True):
        diff = float(np.nanmax(np.abs(left - right)))
        raise AssertionError(f"{label}: maximum absolute difference {diff}")


def sd(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else math.nan


def sem(values: pd.Series) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else math.nan


def add_retention_and_density(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=["baseline_acc_sequence", "sequence_retention"], errors="ignore").copy()
    baseline = (
        out[out["method"] == "Baseline"][["source_model_label", "post_acc_sequence"]]
        .rename(columns={"post_acc_sequence": "baseline_acc_sequence"})
    )
    require(len(baseline) == 24, f"expected 24 baselines, found {len(baseline)}")
    require(not baseline["source_model_label"].duplicated().any(), "duplicate source-model baseline")
    out = out.merge(baseline, on="source_model_label", how="left", validate="many_to_one")
    out["sequence_retention"] = out["post_acc_sequence"].astype(float) / out["baseline_acc_sequence"].astype(float)
    out["effective_nonzero_recurrent_density_full_matrix"] = out["post_rec_weight_nz_count"].astype(float) / FULL_EDGES
    out["effective_nonzero_recurrent_density_offdiag"] = out["post_rec_weight_nz_count"].astype(float) / OFFDIAG_EDGES
    out["post_rec_linear_rho_ratio_to_unpruned"] = out["post_rec_linear_rho"].astype(float) / out["pre_rec_linear_rho"].astype(float)
    return out


def validate_raw_derivations(df: pd.DataFrame, label: str) -> None:
    derived = add_retention_and_density(df)
    assert_close(df["baseline_acc_sequence"], derived["baseline_acc_sequence"], f"{label} baseline accuracy")
    assert_close(df["sequence_retention"], derived["sequence_retention"], f"{label} retention")
    assert_close(
        df["recorded_mask_density_full_matrix"],
        1.0 - df["post_sparsity_recurrent"].astype(float),
        f"{label} recorded mask density",
    )
    assert_close(
        df["effective_nonzero_recurrent_density_full_matrix"],
        derived["effective_nonzero_recurrent_density_full_matrix"],
        f"{label} full-matrix nonzero density",
    )
    assert_close(
        df["effective_nonzero_recurrent_density_offdiag"],
        derived["effective_nonzero_recurrent_density_offdiag"],
        f"{label} off-diagonal nonzero density",
    )
    assert_close(
        df["post_rec_linear_rho_ratio_to_unpruned"],
        derived["post_rec_linear_rho_ratio_to_unpruned"],
        f"{label} rho ratio",
    )


def independently_summarize(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    extra_metrics: tuple[str, ...] = (),
) -> pd.DataFrame:
    metrics = (*SUMMARY_METRICS, *extra_metrics)
    cell_aggs: dict[str, tuple[str, str]] = {metric: (metric, "mean") for metric in metrics}
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
    task_aggs: dict[str, tuple[str, object]] = {"n_tasks": ("task_short", "count")}
    for metric in metrics:
        task_aggs[f"{metric}_taskmean_sd"] = (metric, sd)
        task_aggs[f"{metric}_taskmean_sem"] = (metric, sem)
    task_summary = task_cell.groupby(group_cols, as_index=False).agg(**task_aggs)
    return summary.merge(task_summary, on=group_cols, how="left", validate="one_to_one")


def compare_summaries(
    raw: pd.DataFrame,
    summary_path: Path,
    *,
    group_cols: list[str],
    extra_metrics: tuple[str, ...] = (),
) -> None:
    expected = independently_summarize(raw, group_cols=group_cols, extra_metrics=extra_metrics)
    actual = pd.read_csv(summary_path)
    expected = expected.sort_values(group_cols).reset_index(drop=True)
    actual = actual.sort_values(group_cols).reset_index(drop=True)
    require(len(actual) == len(expected), f"{summary_path}: row count mismatch")
    for col in expected.columns:
        require(col in actual.columns, f"{summary_path}: missing column {col}")
        if pd.api.types.is_numeric_dtype(expected[col]):
            assert_close(actual[col], expected[col], f"{summary_path.name} {col}")
        else:
            require(actual[col].tolist() == expected[col].tolist(), f"{summary_path}: mismatch in {col}")


def enrich_match(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["method"] = out["strategy"].map({**MATCH_METHODS, "none": "Baseline"})
    out["task_short"] = out["task"].astype(str).str.replace("modcog:", "", regex=False)
    out["pruning_pct"] = (out["amount"].astype(float) * 100.0).round().astype(int)
    out["candidate_edge_cap_value"] = out["prune_rescale_cap_value"].astype(float)
    out = add_retention_and_density(out)
    return out


def cell_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = (
        df[df["method"] != "Baseline"]
        .groupby(["task_short", "source_network_seed", "pruning_pct", "method"], as_index=False)
        .post_acc_sequence.mean()
    )
    wide = cell.pivot(index=["task_short", "source_network_seed", "pruning_pct"], columns="method", values="post_acc_sequence")
    task = cell.groupby(["task_short", "method"], as_index=False).post_acc_sequence.mean()
    return wide, task.pivot(index="task_short", columns="method", values="post_acc_sequence")


def expected_holm(pvalues: pd.Series) -> pd.Series:
    order = pvalues.sort_values().index
    out = pd.Series(index=pvalues.index, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(order) - rank) * float(pvalues.loc[idx]))
        out.loc[idx] = min(running, 1.0)
    return out


def validate_task_specs() -> None:
    specs = pd.read_csv(TASK_SPECS)
    expected_ng_t = {
        "ctxdlydm2intseq": 40,
        "ctxdlydm1intseq": 40,
        "dlydm1intseq": 40,
        "dlydm2intseq": 40,
        "multidlydmintseq": 40,
        "dm1seqr": 30,
        "dm2seql": 30,
        "dmsintseq": 38,
    }
    require(len(specs) == 8, f"unexpected task-spec row count {len(specs)}")
    require(specs.set_index("task")["ng_T"].astype(int).to_dict() == expected_ng_t, "task-spec ng_T map mismatch")
    require(set(specs["env_dt_ms"].astype(int)) == {100}, "task-spec Mod-Cog env dt should be 100 ms")
    require(set(specs["dim_ring"].astype(int)) == {16}, "task-spec dim_ring should be 16")


def validate_significance(main: pd.DataFrame, cap: pd.DataFrame, match: pd.DataFrame) -> None:
    tests = pd.read_csv(TESTS)
    means = pd.read_csv(MEANS)
    expected_counts = {
        "capped_rescale_probe_cell": 6,
        "capped_rescale_probe_taskmean": 6,
        "matched_q50_cell": 3,
        "matched_q50_taskmean": 3,
        "revised_task_cell": 8,
        "revised_task_rescale_vs_baseline_by_sparsity_cell": 16,
        "revised_task_rescale_vs_baseline_by_sparsity_taskmean": 16,
        "revised_task_taskmean": 8,
    }
    require(tests.groupby("family").size().to_dict() == expected_counts, "unexpected significance family counts")
    datasets = {
        "revised_task": main,
        "capped_rescale_probe": cap,
        "matched_q50": match,
    }
    for idx, row in tests.iterrows():
        dataset = next(name for name in datasets if str(row["family"]).startswith(name))
        wide, task_wide = cell_tables(datasets[dataset])
        if row["unit"] == "task_network_pruning_pct_cell":
            table = wide
        elif row["unit"] == "task_network_cell":
            table = wide.xs(int(row["pruning_pct"]), level="pruning_pct")
        elif row["unit"] == "task_mean" and not pd.isna(row.get("pruning_pct")):
            table = (
                wide.xs(int(row["pruning_pct"]), level="pruning_pct")
                .reset_index()
                .groupby("task_short", as_index=True)
                .mean(numeric_only=True)
            )
        elif row["unit"] == "task_mean":
            table = task_wide
        else:
            raise AssertionError(f"unknown significance unit {row['unit']}")
        pair = table[[row["method_a"], row["method_b"]]].dropna()
        diff = pair[row["method_a"]] - pair[row["method_b"]]
        nonzero = diff[diff != 0.0]
        expected_p = float(wilcoxon(nonzero, alternative=row["alternative"], zero_method="wilcox").pvalue)
        wins = int(((diff > 0.0) if row["alternative"] == "greater" else (diff < 0.0)).sum())
        losses = int(((diff < 0.0) if row["alternative"] == "greater" else (diff > 0.0)).sum())
        expected_sign = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)
        assert_close(row["n"], len(diff), f"significance row {idx} n")
        assert_close(row["mean_a"], pair[row["method_a"]].mean(), f"significance row {idx} mean_a")
        assert_close(row["mean_b"], pair[row["method_b"]].mean(), f"significance row {idx} mean_b")
        assert_close(row["wilcoxon_p"], expected_p, f"significance row {idx} Wilcoxon p")
        assert_close(row["sign_test_p"], expected_sign, f"significance row {idx} sign p")
    for family, group in tests.groupby("family"):
        expected = expected_holm(group["wilcoxon_p"])
        assert_close(group["holm_p_within_family"], expected.loc[group.index], f"{family} Holm p")
    require(len(means) == 30, f"expected 30 descriptive means, found {len(means)}")
    for dataset, df in datasets.items():
        wide, task_wide = cell_tables(df)
        for unit, table in [("task_network_pruning_pct_cell", wide), ("task_mean", task_wide)]:
            for method in table.columns:
                row = means[(means["dataset"] == dataset) & (means["unit"] == unit) & (means["method"] == method)]
                require(len(row) == 1, f"missing descriptive mean for {dataset} {unit} {method}")
                values = table[method].dropna()
                assert_close(row.iloc[0]["n"], len(values), f"{dataset} {unit} {method} n")
                assert_close(row.iloc[0]["mean"], values.mean(), f"{dataset} {unit} {method} mean")
                assert_close(row.iloc[0]["sd"], sd(values), f"{dataset} {unit} {method} sd")
                assert_close(row.iloc[0]["sem"], sem(values), f"{dataset} {unit} {method} sem")


def validate_rescale_distributions(main: pd.DataFrame, match: pd.DataFrame) -> None:
    run = pd.read_csv(RESCALE_RUN)
    by_run = pd.read_csv(RESCALE_BY_RUN)
    overall = pd.read_csv(RESCALE_OVERALL)
    by_run_wide = pd.read_csv(RESCALE_BY_RUN_WIDE)
    overall_wide = pd.read_csv(RESCALE_OVERALL_WIDE)
    require(run.groupby("method_family").size().to_dict() == {"L-NP": 72, "S-NP": 72}, "rescale run counts")
    hist_counts = (
        by_run.groupby(["method_family", "task", "source_model_label", "pruning_seed"], as_index=False)["count"].sum()
    )
    joined = run.merge(
        hist_counts,
        on=["method_family", "task", "source_model_label", "pruning_seed"],
        how="left",
        validate="one_to_one",
    )
    assert_close(joined["positive_prob_count"], joined["count"], "by-run histogram totals")
    overall_counts = overall.groupby("method_family")["count"].sum().sort_index()
    run_counts = run.groupby("method_family")["positive_prob_count"].sum().sort_index()
    assert_close(overall_counts, run_counts, "overall histogram totals")
    wide_counts = overall_wide.groupby("method_family")["count"].sum().sort_index()
    assert_close(wide_counts, run_counts, "wide-bin overall histogram totals")
    wide_by_run_counts = by_run_wide.groupby("method_family")["count"].sum().sort_index()
    assert_close(wide_by_run_counts, run_counts, "wide-bin by-run histogram totals")
    require(set(overall_wide["bin_width_log10"].dropna().astype(float)) == {0.20}, "unexpected overall wide-bin width")
    require(set(by_run_wide["bin_width_log10"].dropna().astype(float)) == {0.20}, "unexpected by-run wide-bin width")
    aggregated_wide = (
        by_run_wide.groupby(
            ["method_family", "method", "amount", "log10_rescale_left", "log10_rescale_right"],
            as_index=False,
        )["count"]
        .sum()
        .sort_values(["method_family", "method", "amount", "log10_rescale_left"])
        .reset_index(drop=True)
    )
    overall_wide_check = (
        overall_wide[
            ["method_family", "method", "amount", "log10_rescale_left", "log10_rescale_right", "count"]
        ]
        .sort_values(["method_family", "method", "amount", "log10_rescale_left"])
        .reset_index(drop=True)
    )
    require(len(aggregated_wide) == len(overall_wide_check), "wide-bin overall row-count mismatch")
    for col in overall_wide_check.columns:
        if pd.api.types.is_numeric_dtype(overall_wide_check[col]):
            assert_close(overall_wide_check[col], aggregated_wide[col], f"wide-bin overall {col}")
        else:
            require(
                overall_wide_check[col].tolist() == aggregated_wide[col].tolist(),
                f"wide-bin overall mismatch in {col}",
            )

    canonical = main[(main["pruning_pct"] == 80) & main["method"].isin(["L-NP rescale", "S-NP rescale"])].copy()
    canonical["method_family"] = canonical["method"].str.replace(" rescale", "", regex=False)
    checked = canonical.merge(
        run,
        on=["method_family", "task", "source_model_label", "pruning_seed"],
        how="left",
        validate="one_to_one",
    )
    assert_close(checked["prune_positive_prob_count"], checked["positive_prob_count"], "candidate-edge counts")
    assert_close(checked["prune_inv_p_mean"], checked["rescale_mean"], "candidate inv-p means")
    assert_close(checked["prune_inv_p_max"], checked["rescale_max"], "candidate inv-p maxima")
    assert_close(checked["prune_frac_amp_gt10"], checked["frac_rescale_gt10"], "candidate fraction >10")
    assert_close(checked["prune_frac_amp_gt20"], checked["frac_rescale_gt20"], "candidate fraction >20")

    capped = match[
        (match["pruning_pct"] == 80) & match["method"].isin(["L-NP capped q50", "S-NP capped q50"])
    ].copy()
    capped["method_family"] = capped["method"].str.replace(" capped q50", "", regex=False)
    q50 = capped.merge(
        run[["method_family", "task", "source_model_label", "pruning_seed", "rescale_q50"]],
        on=["method_family", "task", "source_model_label", "pruning_seed"],
        how="left",
        validate="one_to_one",
    )
    assert_close(q50["prune_rescale_cap_value"], q50["rescale_q50"], "matched q50 candidate-edge caps")


def validate_protocol(main: pd.DataFrame, cap: pd.DataFrame, match: pd.DataFrame) -> None:
    require(main.groupby("method").size().to_dict() == {
        "Baseline": 24,
        "Random": 288,
        "Magnitude": 96,
        "OBS compensated": 96,
        "L-NP mask": 96,
        "L-NP rescale": 288,
        "S-NP mask": 288,
        "S-NP rescale": 288,
    }, "unexpected main method counts")
    require(len(main) == 1464, f"unexpected main rows {len(main)}")
    require(cap.groupby("method").size().to_dict() == {
        "Baseline": 24,
        "S-NP rescale": 96,
        "S-NP capped q50": 96,
        "S-NP capped q75": 96,
        "S-NP capped q90": 96,
    }, "unexpected exploratory cap counts")
    require(len(cap) == 408, f"unexpected exploratory cap rows {len(cap)}")
    require(match.groupby("strategy").size().to_dict() == {
        "none": 24,
        "noise_prune": 288,
        "noise_prune_capped_rescale": 288,
        "simulation_noise_prune_rescale": 288,
        "simulation_noise_prune_capped_rescale": 288,
    }, "unexpected matched q50 counts")
    require(len(match) == 1176, f"unexpected matched q50 rows {len(match)}")
    for label, df in [("main", main), ("cap", cap), ("matched q50", match)]:
        require(df["task"].nunique() == 8, f"{label}: expected 8 tasks")
        require(df["source_model_label"].nunique() == 24, f"{label}: expected 24 trained networks")
        require(df["eval_batches_path"].nunique() == 8, f"{label}: expected one fixed eval batch path per task")
        require(set(df["eval_seed"].dropna().astype(int)) == {200_000}, f"{label}: unexpected eval seed")
        require(set(df["eval_sample_batches"].dropna().astype(int)) == {128}, f"{label}: unexpected eval batch count")
    require(set(cap.loc[cap["method"] != "Baseline", "pruning_seed"].dropna().astype(int)) == {0}, "cap probe is not one-seed")
    require(
        set(match.loc[match["strategy"] != "none", "pruning_seed"].dropna().astype(int)) == {0, 1, 2},
        "matched q50 is not three-seed",
    )
    snp = match[match["strategy"].str.contains("simulation")]
    require(set(snp["sim_np_sigma_source"].dropna()) == {"natural_voltage"}, "unexpected S-NP sigma calibration")
    require(set(snp["sim_np_observable_space"].dropna()) == {"rate"}, "unexpected S-NP observable space")
    require(set(snp["sim_np_inject_space"].dropna()) == {"rate"}, "unexpected S-NP injection space")
    require(set(snp["sim_np_centering"].dropna()) == {"trajectory_mean"}, "unexpected S-NP centering")
    require(set(snp["sim_np_max_samples"].dropna().astype(int)) == {25_000}, "unexpected S-NP max samples")
    require(set(snp["sim_np_burn_in_steps"].dropna().astype(int)) == {300}, "unexpected S-NP burn-in")


def validate_cap_percentile_curve(main_summary: pd.DataFrame, match: pd.DataFrame) -> None:
    missing = pd.read_csv(CURVE_MISSING)
    rows = pd.read_csv(CURVE_ROWS)
    summary = pd.read_csv(CURVE_SUMMARY)
    plot = pd.read_csv(CURVE_PLOT_POINTS)

    if "error" in missing.columns:
        errors = missing["error"].notna() & (missing["error"].astype(str) != "")
        require(not errors.any(), f"cap-percentile suite contains failed runs: {missing.loc[errors, 'run_id'].head().tolist()}")
    require(
        missing.groupby("strategy").size().to_dict()
        == {"none": 24, "noise_prune_capped_rescale": 768, "simulation_noise_prune_capped_rescale": 672},
        "unexpected cap-percentile missing-suite counts",
    )
    require(len(missing) == 1464, f"unexpected cap-percentile missing-suite rows {len(missing)}")
    for label, df in [("missing", missing), ("curve rows", rows)]:
        require(df["task"].nunique() == 8, f"{label}: expected 8 tasks")
        require(df["source_model_label"].nunique() == 24, f"{label}: expected 24 trained networks")
        require(set(df["eval_seed"].dropna().astype(int)) == {200_000}, f"{label}: unexpected eval seed")
        require(set(df["eval_sample_batches"].dropna().astype(int)) == {128}, f"{label}: unexpected eval batch count")

    missing_nonbase = missing[missing["strategy"] != "none"]
    require(set(missing_nonbase["pruning_seed"].dropna().astype(int)) == {0}, "missing curve suite is not one-seed")
    missing_quantiles = {
        key: sorted(set(group["prune_rescale_cap_quantile"].dropna().round(2)))
        for key, group in missing_nonbase.groupby("strategy")
    }
    require(
        missing_quantiles == {
            "noise_prune_capped_rescale": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            "simulation_noise_prune_capped_rescale": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
        },
        f"unexpected missing cap quantiles: {missing_quantiles}",
    )
    snp = missing[missing["strategy"] == "simulation_noise_prune_capped_rescale"]
    require(set(snp["prune_sigma_source"].dropna()) == {"natural_voltage"}, "curve S-NP sigma calibration")
    require(set(snp["prune_observable_space"].dropna()) == {"rate"}, "curve S-NP observable space")
    require(set(snp["prune_inject_space"].dropna()) == {"rate"}, "curve S-NP injection space")
    require(set(snp["prune_centering"].dropna()) == {"trajectory_mean"}, "curve S-NP centering")
    require(set(snp["prune_sample_count"].dropna().astype(int)) == {25_000}, "curve S-NP sample count")
    require(set(snp["prune_burn_in_steps"].dropna().astype(int)) == {300}, "curve S-NP burn-in")

    require(len(rows) == 1728, f"unexpected cap-percentile curve row count {len(rows)}")
    require(len(summary) == 72, f"unexpected cap-percentile summary row count {len(summary)}")
    require(len(plot) == 22, f"unexpected cap-percentile plot-point row count {len(plot)}")
    require(set(rows["method"]) == {"L-NP capped rescale", "S-NP capped rescale"}, "unexpected curve methods")
    for method in ("L-NP capped rescale", "S-NP capped rescale"):
        method_rows = rows[rows["method"] == method]
        require(set(method_rows["cap_quantile"].round(2)) == CURVE_QUANTILES, f"{method}: missing q10-q90 rows")
        counts = method_rows.groupby(["cap_quantile", "pruning_pct"]).size()
        require((counts == 24).all(), f"{method}: expected 24 rows per quantile/sparsity cell")

    expected_summary = (
        rows.groupby(["method", "method_label", "cap_quantile", "cap_percentile", "pruning_pct"], as_index=False)
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
        .reset_index(drop=True)
    )
    actual_summary = summary.sort_values(["method", "cap_quantile", "pruning_pct"]).reset_index(drop=True)
    require(len(actual_summary) == len(expected_summary), "cap-percentile summary row count mismatch")
    for col in expected_summary.columns:
        require(col in actual_summary.columns, f"cap-percentile summary missing column {col}")
        if pd.api.types.is_numeric_dtype(expected_summary[col]):
            assert_close(actual_summary[col], expected_summary[col], f"cap-percentile summary {col}")
        else:
            require(actual_summary[col].tolist() == expected_summary[col].tolist(), f"cap-percentile summary mismatch in {col}")

    records = []
    for method, family in (("L-NP capped rescale", "L-NP"), ("S-NP capped rescale", "S-NP")):
        subset = summary[summary["method"] == method]
        for quantile, group in subset.groupby("cap_percentile"):
            records.append({
                "family": family,
                "plot_x": int(quantile),
                "plot_label": f"q{int(quantile)}",
                "point_type": "cap_quantile",
                "source": "one_pruning_seed_curve",
                "mean_retention_across_p50_80": group["sequence_retention_mean"].mean(),
                "mean_rho_ratio_across_p50_80": group["post_rec_linear_rho_ratio_to_unpruned_mean"].mean(),
            })
    q100 = match[
        match["strategy"].isin({"noise_prune", "simulation_noise_prune_rescale"})
        & (match["pruning_seed"].astype(float) == 0)
    ].copy()
    q100["family"] = q100["strategy"].map({"noise_prune": "L-NP", "simulation_noise_prune_rescale": "S-NP"})
    for family, group in q100.groupby("family"):
        by_sparsity = group.groupby("pruning_pct", as_index=False).agg(
            sequence_retention_mean=("sequence_retention", "mean"),
            rho_ratio_mean=("post_rec_linear_rho_ratio_to_unpruned", "mean"),
        )
        records.append({
            "family": family,
            "plot_x": 100,
            "plot_label": "q100 no cap",
            "point_type": "uncapped_rescale",
            "source": "matched_q50_suite_pruning_seed0_uncapped",
            "mean_retention_across_p50_80": by_sparsity["sequence_retention_mean"].mean(),
            "mean_rho_ratio_across_p50_80": by_sparsity["rho_ratio_mean"].mean(),
        })
    for method, family in (("L-NP mask", "L-NP"), ("S-NP mask", "S-NP")):
        subset = main_summary[main_summary["method"] == method]
        records.append({
            "family": family,
            "plot_x": 0,
            "plot_label": "q0 mask-only",
            "point_type": "mask_only_not_cap_quantile",
            "source": "canonical_main_mask_summary",
            "mean_retention_across_p50_80": subset["sequence_retention_mean"].mean(),
            "mean_rho_ratio_across_p50_80": subset["post_rec_linear_rho_ratio_to_unpruned_mean"].mean(),
        })
    expected_plot = pd.DataFrame(records).sort_values(["family", "plot_x"]).reset_index(drop=True)
    actual_plot = plot.sort_values(["family", "plot_x"]).reset_index(drop=True)
    for col in expected_plot.columns:
        require(col in actual_plot.columns, f"plot points missing column {col}")
        if pd.api.types.is_numeric_dtype(expected_plot[col]):
            assert_close(actual_plot[col], expected_plot[col], f"cap-percentile plot {col}")
        else:
            require(actual_plot[col].tolist() == expected_plot[col].tolist(), f"cap-percentile plot mismatch in {col}")


def validate_three_seed_cap_percentile_curve() -> None:
    missing = pd.read_csv(CURVE_MISSING_SEEDS12)
    rows = pd.read_csv(CURVE_ROWS_3SEED)
    summary = pd.read_csv(CURVE_SUMMARY_3SEED)
    plot = pd.read_csv(CURVE_PLOT_POINTS_3SEED)
    plot_sem = pd.read_csv(CURVE_PLOT_POINTS_3SEED_WITH_SEM)
    tests = pd.read_csv(CURVE_TESTS_3SEED)
    plot_tests = pd.read_csv(CURVE_PLOT_TESTS_3SEED)

    if "error" in missing.columns:
        errors = missing["error"].notna() & (missing["error"].astype(str) != "")
        require(not errors.any(), f"three-seed cap-percentile suite contains failed runs: {missing.loc[errors, 'run_id'].head().tolist()}")
    require(len(missing) == 3096, f"unexpected three-seed missing-suite rows {len(missing)}")
    require(
        missing.groupby("strategy").size().to_dict()
        == {"none": 24, "noise_prune_capped_rescale": 1536, "simulation_noise_prune_capped_rescale": 1536},
        "unexpected three-seed missing-suite strategy counts",
    )
    require(not missing["run_id"].duplicated().any(), "duplicate run IDs in three-seed missing suite")
    require(set(missing.loc[missing["strategy"] != "none", "pruning_seed"].dropna().astype(int)) == {1, 2}, "missing suite should contain pruning seeds 1 and 2")

    require(len(rows) == 5184, f"unexpected three-seed curve row count {len(rows)}")
    require(len(summary) == 72, f"unexpected three-seed curve summary row count {len(summary)}")
    require(len(plot) == 22, f"unexpected three-seed plot-point row count {len(plot)}")
    require(len(plot_sem) == 22, f"unexpected three-seed plot-point-with-SEM row count {len(plot_sem)}")
    require(len(tests) == 72, f"unexpected three-seed cap-vs-uncapped test row count {len(tests)}")
    require(set(rows["method"]) == {"L-NP capped rescale", "S-NP capped rescale"}, "unexpected three-seed curve methods")
    require(set(rows["pruning_seed"].dropna().astype(int)) == {0, 1, 2}, "combined three-seed curve should contain pruning seeds 0, 1, and 2")
    require(set(rows["cap_quantile"].round(2)) == CURVE_QUANTILES, "combined three-seed curve missing q10-q90 rows")
    counts = rows.groupby(["method", "cap_quantile", "pruning_pct"]).size()
    require((counts == 72).all(), "three-seed curve should have 72 raw rows per method x quantile x sparsity cell")
    require(set(summary["n"].astype(int)) == {24}, "three-seed curve summary should use n=24 trained networks")
    require(set(summary["raw_run_count"].astype(int)) == {72}, "three-seed curve summary should have 72 raw rows per summary cell")
    require(set(plot["plot_x"].astype(int)) == {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}, "three-seed plot points should include q0, q10-q90, and q100")
    require(set(plot_sem["plot_x"].astype(int)) == {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}, "three-seed exact-SEM plot points should include q0, q10-q90, and q100")
    require(set(plot_sem["n"].astype(int)) == {24}, "three-seed exact-SEM plot points should use n=24 trained networks")
    require(set(plot_sem["sparsity_level_count_min"].astype(int)) == {4}, "exact-SEM plot points should average four sparsity levels per network")
    require(set(plot_sem["sparsity_level_count_max"].astype(int)) == {4}, "exact-SEM plot points should average four sparsity levels per network")
    merged_plot = plot.merge(
        plot_sem,
        on=["family", "plot_x", "plot_label", "point_type"],
        how="inner",
        validate="one_to_one",
        suffixes=("_mean_only", "_with_sem"),
    )
    require(len(merged_plot) == 22, "mean-only and exact-SEM plot points do not align")
    assert_close(
        merged_plot["mean_retention_across_p50_80_mean_only"],
        merged_plot["mean_retention_across_p50_80_with_sem"],
        "three-seed exact-SEM retention means",
    )
    assert_close(
        merged_plot["mean_rho_ratio_across_p50_80_mean_only"],
        merged_plot["mean_rho_ratio_across_p50_80_with_sem"],
        "three-seed exact-SEM rho-ratio means",
    )
    require(set(tests["n"].astype(int)) == {24}, "three-seed cap-vs-uncapped tests should use n=24 trained networks")
    require(len(plot_tests) == 18, f"unexpected plot-level cap-vs-q100 test row count {len(plot_tests)}")
    require(set(plot_tests["method_family"]) == {"L-NP", "S-NP"}, "plot-level tests should cover L-NP and S-NP")
    require(set(plot_tests["cap_percentile"].astype(int)) == set(range(10, 100, 10)), "plot-level tests should cover q10-q90")
    require(set(plot_tests["n"].astype(int)) == {24}, "plot-level tests should use n=24 trained networks")
    require(set(plot_tests["multiple_comparison_family_size"].astype(int)) == {18}, "plot-level tests should use the 18-comparison cap family")
    assert_close(
        plot_tests["holm_p_within_family"],
        expected_holm(plot_tests["wilcoxon_p"]),
        "plot-level cap-vs-q100 Holm p-values",
    )
    assert_close(
        plot_tests["sign_test_holm_p_within_family"],
        expected_holm(plot_tests["sign_test_p"]),
        "plot-level cap-vs-q100 sign-test Holm p-values",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest() -> None:
    payload = json.loads(MANIFEST.read_text())
    require(payload["artifact_set"] == "official_h512_24net_revised_scope", "unexpected artifact manifest label")
    require(payload["hash_algorithm"] == "sha256", "unexpected artifact manifest hash algorithm")
    require(len(payload["files"]) == 45, "unexpected artifact manifest file count")
    for record in payload["files"]:
        path = ROOT / "paper_artifacts/official_h512_24net" / record["path"]
        require(path.is_file(), f"manifest file missing: {path}")
        require(path.stat().st_size == int(record["bytes"]), f"manifest byte-size mismatch: {path}")
        require(sha256(path) == record["sha256"], f"manifest SHA-256 mismatch: {path}")


def main() -> None:
    main = pd.read_csv(MAIN)
    main_summary = pd.read_csv(MAIN_SUMMARY)
    cap = pd.read_csv(CAP)
    match = enrich_match(pd.read_csv(MATCH))
    validate_task_specs()
    validate_protocol(main, cap, match)
    validate_raw_derivations(main, "main")
    validate_raw_derivations(cap, "exploratory cap")
    compare_summaries(main, MAIN_SUMMARY, group_cols=["method_order", "method", "pruning_pct"])
    compare_summaries(
        cap,
        CAP_SUMMARY,
        group_cols=["method", "pruning_pct"],
        extra_metrics=("candidate_edge_cap_value",),
    )
    compare_summaries(
        match,
        MATCH_SUMMARY,
        group_cols=["method", "pruning_pct"],
        extra_metrics=("candidate_edge_cap_value",),
    )
    validate_significance(main, cap, match)
    validate_rescale_distributions(main, match)
    validate_cap_percentile_curve(main_summary, match)
    validate_three_seed_cap_percentile_curve()
    validate_manifest()
    print("validated revised paper artifacts")
    print("main rows: 1464; exploratory cap rows: 408; matched q50 rows: 1176")
    print("summary units: trained-network n=24; clustered task-mean n=8")
    print("significance rows: 66; descriptive mean rows: 30")
    print("rescale-distribution runs: 144; histogram candidate-edge totals verified")
    print("wide-bin rescale histograms: log10 bin width=0.20; totals verified")
    print("cap-percentile curve rows: one-seed raw=1464; three-seed raw=3096; combined=5184; summary=72; plot points=22")
    print("release manifest: 45 SHA-256 hashes verified")


if __name__ == "__main__":
    main()
