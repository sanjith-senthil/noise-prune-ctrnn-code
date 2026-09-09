#!/usr/bin/env python3
"""Independently validate the reporting-grade H=512 statistical exports."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_artifacts/official_h512_24net/data"
REVISED = DATA / "revised_scope"
SIG = REVISED / "significance"
MAIN = REVISED / "task_preservation/revised_task_preservation_h512_24net_p50_80.csv"
CAP = REVISED / "capped_rescale/revised_snp_capped_rescale_probe_h512_24net_p50_80.csv"
MATCHED = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_q50_capped_rescale_3seed_full8_p50_80.csv"
)
RESCALE_RUN = DATA / "rescale_value_distributions/rescale_value_run_summary_snp_lnp_p80.csv"
TESTS = SIG / "paper_relevant_h512_comparison_tests.csv"
DESCRIPTIVES = SIG / "paper_relevant_h512_descriptive_statistics.csv"
AMPLIFICATION = SIG / "paper_relevant_amplification_descriptive_statistics.csv"
PCTS = {50, 60, 70, 80}
METRICS = (
    "sequence_retention",
    "post_rec_linear_rho",
    "post_rec_linear_rho_ratio_to_unpruned",
    "effective_nonzero_recurrent_density_offdiag",
)
MATCHED_METHOD_MAP = {
    "none": "Baseline",
    "noise_prune": "L-NP rescale",
    "noise_prune_capped_rescale": "L-NP capped q50",
    "simulation_noise_prune_rescale": "S-NP rescale",
    "simulation_noise_prune_capped_rescale": "S-NP capped q50",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: object, expected: object, label: str) -> None:
    if not np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=1e-10, atol=1e-12, equal_nan=True):
        raise AssertionError(label)


def sd(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else math.nan


def sem(values: pd.Series) -> float:
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else math.nan


def enrich_matched() -> pd.DataFrame:
    df = pd.read_csv(MATCHED)
    df["method"] = df["strategy"].map(MATCHED_METHOD_MAP)
    df["task_short"] = df["task"].astype(str).str.replace("modcog:", "", regex=False)
    df["pruning_pct"] = (df["amount"].astype(float) * 100.0).round().astype(int)
    baseline = (
        df[df["method"] == "Baseline"][["source_model_label", "post_acc_sequence"]]
        .rename(columns={"post_acc_sequence": "baseline_acc_sequence"})
    )
    df = df.merge(baseline, on="source_model_label", how="left", validate="many_to_one")
    df["sequence_retention"] = df["post_acc_sequence"] / df["baseline_acc_sequence"]
    df["post_rec_linear_rho_ratio_to_unpruned"] = df["post_rec_linear_rho"] / df["pre_rec_linear_rho"]
    df["effective_nonzero_recurrent_density_offdiag"] = df["post_rec_weight_nz_count"] / (512 * 511)
    return df


def network_cells(df: pd.DataFrame) -> pd.DataFrame:
    pruned = (
        df[df["method"] != "Baseline"]
        .groupby(["task_short", "source_network_seed", "pruning_pct", "method"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS})
    )
    baseline = (
        df[df["method"] == "Baseline"]
        .groupby(["task_short", "source_network_seed", "method"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS})
    )
    expanded = pd.concat([baseline.assign(pruning_pct=pct) for pct in sorted(PCTS)], ignore_index=True)
    return pd.concat([pruned, expanded], ignore_index=True, sort=False)


def task_cells(network: pd.DataFrame) -> pd.DataFrame:
    return (
        network.groupby(["task_short", "pruning_pct", "method"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS})
    )


def holm(pvalues: pd.Series) -> pd.Series:
    order = pvalues.sort_values().index
    adjusted = pd.Series(index=pvalues.index, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(order) - rank) * float(pvalues.loc[idx]))
        adjusted.loc[idx] = min(running, 1.0)
    return adjusted


def validate_tests(datasets: dict[str, pd.DataFrame]) -> None:
    tests = pd.read_csv(TESTS)
    require(len(tests) == 768, f"expected 768 reporting tests, found {len(tests)}")
    require(set(tests["alpha"]) == {0.05}, "unexpected alpha")
    require(set(tests["alternative"]) == {"two-sided"}, "reporting tests must be two-sided")
    require(set(tests["tails"]) == {2}, "reporting tests must record two tails")
    require(set(tests["test_name"]) == {"paired Wilcoxon signed-rank test"}, "unexpected primary test")
    require(set(tests["robustness_test_name"]) == {"exact binomial sign test"}, "unexpected robustness test")
    require(not tests["normality_required"].any(), "normality must not be required")
    require(tests["wilcoxon_paired_difference_symmetry_assumption"].all(), "missing Wilcoxon symmetry note")
    require(not tests.duplicated(["family", "pruning_pct", "method_a", "method_b"]).any(), "duplicate test row")
    cell_cache = {name: network_cells(df) for name, df in datasets.items()}
    task_cache = {name: task_cells(cells) for name, cells in cell_cache.items()}
    for idx, row in tests.iterrows():
        cells = cell_cache[row["dataset"]] if row["unit"] == "task_network_cell" else task_cache[row["dataset"]]
        index = ["task_short", "source_network_seed"] if row["unit"] == "task_network_cell" else ["task_short"]
        table = cells[cells["pruning_pct"] == int(row["pruning_pct"])].pivot(
            index=index, columns="method", values=row["metric"]
        )
        pair = table[[row["method_a"], row["method_b"]]].dropna()
        diff = pair[row["method_a"]] - pair[row["method_b"]]
        nonzero = diff[diff != 0.0]
        expected_wilcoxon = (
            float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
            if len(nonzero)
            else 1.0
        )
        wins = int((diff > 0.0).sum())
        losses = int((diff < 0.0).sum())
        expected_sign = float(binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue) if wins + losses else 1.0
        close(row["n"], len(diff), f"test row {idx} n")
        close(row["n_nonzero_differences"], len(nonzero), f"test row {idx} nonzero n")
        close(row["mean_a"], pair[row["method_a"]].mean(), f"test row {idx} mean a")
        close(row["median_a"], pair[row["method_a"]].median(), f"test row {idx} median a")
        close(row["mean_b"], pair[row["method_b"]].mean(), f"test row {idx} mean b")
        close(row["median_b"], pair[row["method_b"]].median(), f"test row {idx} median b")
        close(row["mean_diff_a_minus_b"], diff.mean(), f"test row {idx} mean diff")
        close(row["median_diff_a_minus_b"], diff.median(), f"test row {idx} median diff")
        close(row["min_diff_a_minus_b"], diff.min(), f"test row {idx} min diff")
        close(row["max_diff_a_minus_b"], diff.max(), f"test row {idx} max diff")
        close(row["wins_a_gt_b"], wins, f"test row {idx} wins")
        close(row["losses_a_lt_b"], losses, f"test row {idx} losses")
        close(row["ties"], int((diff == 0.0).sum()), f"test row {idx} ties")
        close(row["wilcoxon_p"], expected_wilcoxon, f"test row {idx} Wilcoxon p")
        close(row["sign_test_p"], expected_sign, f"test row {idx} sign p")
    for family, group in tests.groupby("family"):
        require(set(group["family_size"]) == {len(group)}, f"{family}: family-size mismatch")
        close(group["holm_p_within_family"], holm(group["wilcoxon_p"]).loc[group.index], f"{family}: Holm p")
        close(
            group["sign_test_holm_p_within_family"],
            holm(group["sign_test_p"]).loc[group.index],
            f"{family}: sign Holm p",
        )


def validate_descriptives(datasets: dict[str, pd.DataFrame]) -> None:
    desc = pd.read_csv(DESCRIPTIVES)
    require(len(desc) == 576, f"expected 576 descriptive rows, found {len(desc)}")
    for dataset, df in datasets.items():
        network = network_cells(df)
        task = task_cells(network)
        for unit, cells in [("task_network_cell", network), ("task_mean", task)]:
            for (pct, method), group in cells.groupby(["pruning_pct", "method"]):
                for metric in METRICS:
                    row = desc[
                        (desc["dataset"] == dataset)
                        & (desc["unit"] == unit)
                        & (desc["pruning_pct"] == pct)
                        & (desc["method"] == method)
                        & (desc["metric"] == metric)
                    ]
                    require(len(row) == 1, f"missing descriptive row: {dataset} {unit} {pct} {method} {metric}")
                    values = group[metric].dropna()
                    actual = row.iloc[0]
                    close(actual["n"], len(values), "descriptive n")
                    close(actual["mean"], values.mean(), "descriptive mean")
                    close(actual["median"], values.median(), "descriptive median")
                    close(actual["sd"], sd(values), "descriptive sd")
                    close(actual["sem"], sem(values), "descriptive sem")
                    close(actual["min"], values.min(), "descriptive min")
                    close(actual["max"], values.max(), "descriptive max")
                    close(actual["range"], values.max() - values.min(), "descriptive range")


def validate_amplification() -> None:
    raw = pd.read_csv(RESCALE_RUN)
    desc = pd.read_csv(AMPLIFICATION)
    require(len(desc) == 22, f"expected 22 amplification rows, found {len(desc)}")
    for _, row in desc.iterrows():
        values = raw.loc[raw["method_family"] == row["method_family"], row["metric"]].dropna()
        close(row["n"], len(values), "amplification n")
        close(row["mean"], values.mean(), "amplification mean")
        close(row["median"], values.median(), "amplification median")
        close(row["sd"], sd(values), "amplification sd")
        close(row["sem"], sem(values), "amplification sem")
        close(row["min"], values.min(), "amplification min")
        close(row["max"], values.max(), "amplification max")
        close(row["range"], values.max() - values.min(), "amplification range")


def main() -> None:
    datasets = {
        "main_h512": pd.read_csv(MAIN),
        "exploratory_capped_probe_h512": pd.read_csv(CAP),
        "matched_q50_h512": enrich_matched(),
    }
    validate_tests(datasets)
    validate_descriptives(datasets)
    validate_amplification()
    print("validated paper-relevant statistical reporting package")
    print("comparison tests: 768; descriptive rows: 576; amplification rows: 22")
    print("primary test: paired two-sided Wilcoxon signed-rank; Holm correction within family")
    print("robustness test: exact two-sided binomial sign test; Holm correction within family")


if __name__ == "__main__":
    main()
