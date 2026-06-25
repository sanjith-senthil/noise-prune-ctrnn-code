#!/usr/bin/env python3
"""Generate reporting-grade statistics for every supported H=512 paper comparison."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_artifacts/official_h512_24net/data"
REVISED = DATA / "revised_scope"
OUT = REVISED / "significance"
MAIN = REVISED / "task_preservation/revised_task_preservation_h512_24net_p50_80.csv"
CAP = REVISED / "capped_rescale/revised_snp_capped_rescale_probe_h512_24net_p50_80.csv"
MATCHED = (
    REVISED
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_q50_capped_rescale_3seed_full8_p50_80.csv"
)
RESCALE_RUN = DATA / "rescale_value_distributions/rescale_value_run_summary_snp_lnp_p80.csv"
TESTS_OUT = OUT / "paper_relevant_h512_comparison_tests.csv"
DESCRIPTIVES_OUT = OUT / "paper_relevant_h512_descriptive_statistics.csv"
AMPLIFICATION_OUT = OUT / "paper_relevant_amplification_descriptive_statistics.csv"
ALPHA = 0.05
PCTS = (50, 60, 70, 80)
METRICS = (
    "sequence_retention",
    "post_rec_linear_rho",
    "post_rec_linear_rho_ratio_to_unpruned",
    "effective_nonzero_recurrent_density_offdiag",
)
MAIN_METHODS = (
    "Random",
    "Magnitude",
    "OBS compensated",
    "L-NP mask",
    "L-NP rescale",
    "S-NP mask",
    "S-NP rescale",
)
CAP_METHODS = (
    "S-NP rescale",
    "S-NP capped q50",
    "S-NP capped q75",
    "S-NP capped q90",
)
MATCHED_METHODS = (
    "L-NP rescale",
    "L-NP capped q50",
    "S-NP rescale",
    "S-NP capped q50",
)
MATCHED_METHOD_MAP = {
    "none": "Baseline",
    "noise_prune": "L-NP rescale",
    "noise_prune_capped_rescale": "L-NP capped q50",
    "simulation_noise_prune_rescale": "S-NP rescale",
    "simulation_noise_prune_capped_rescale": "S-NP capped q50",
}


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
    if len(baseline) != 24 or baseline["source_model_label"].duplicated().any():
        raise ValueError("matched q50 source must contain one baseline row per trained network")
    df = df.merge(baseline, on="source_model_label", how="left", validate="many_to_one")
    df["sequence_retention"] = (
        df["post_acc_sequence"].astype(float) / df["baseline_acc_sequence"].astype(float)
    )
    df["post_rec_linear_rho_ratio_to_unpruned"] = (
        df["post_rec_linear_rho"].astype(float) / df["pre_rec_linear_rho"].astype(float)
    )
    df["effective_nonzero_recurrent_density_offdiag"] = (
        df["post_rec_weight_nz_count"].astype(float) / (512 * 511)
    )
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
    expanded = pd.concat(
        [baseline.assign(pruning_pct=pruning_pct) for pruning_pct in PCTS],
        ignore_index=True,
    )
    return pd.concat([pruned, expanded], ignore_index=True, sort=False)


def task_cells(network: pd.DataFrame) -> pd.DataFrame:
    return (
        network.groupby(["task_short", "pruning_pct", "method"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS})
    )


def holm_adjust(rows: list[dict[str, object]], p_col: str, out_col: str) -> None:
    for family in sorted({str(row["family"]) for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        ordered = sorted(selected, key=lambda row: float(row[p_col]))
        running = 0.0
        for rank, row in enumerate(ordered):
            running = max(running, (len(ordered) - rank) * float(row[p_col]))
            row[out_col] = min(running, 1.0)
            row["family_size"] = len(ordered)


def test_row(
    *,
    dataset: str,
    family: str,
    metric: str,
    unit: str,
    pruning_pct: int,
    method_a: str,
    method_b: str,
    table: pd.DataFrame,
) -> dict[str, object]:
    pair = table[[method_a, method_b]].dropna()
    diff = pair[method_a] - pair[method_b]
    nonzero = diff[diff != 0.0]
    wilcoxon_p = (
        float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
        if len(nonzero)
        else 1.0
    )
    wins = int((diff > 0.0).sum())
    losses = int((diff < 0.0).sum())
    sign_n = wins + losses
    sign_p = float(binomtest(wins, sign_n, 0.5, alternative="two-sided").pvalue) if sign_n else 1.0
    return {
        "dataset": dataset,
        "family": family,
        "metric": metric,
        "unit": unit,
        "pruning_pct": pruning_pct,
        "method_a": method_a,
        "method_b": method_b,
        "test_name": "paired Wilcoxon signed-rank test",
        "wilcoxon_zero_method": "wilcox",
        "wilcoxon_method": "auto",
        "alternative": "two-sided",
        "tails": 2,
        "alpha": ALPHA,
        "multiple_comparison_method": "Holm correction within family",
        "normality_required": False,
        "wilcoxon_paired_difference_symmetry_assumption": True,
        "robustness_test_name": "exact binomial sign test",
        "n": int(len(diff)),
        "n_nonzero_differences": int(len(nonzero)),
        "mean_a": float(pair[method_a].mean()),
        "median_a": float(pair[method_a].median()),
        "mean_b": float(pair[method_b].mean()),
        "median_b": float(pair[method_b].median()),
        "mean_diff_a_minus_b": float(diff.mean()),
        "median_diff_a_minus_b": float(diff.median()),
        "min_diff_a_minus_b": float(diff.min()),
        "max_diff_a_minus_b": float(diff.max()),
        "wins_a_gt_b": wins,
        "losses_a_lt_b": losses,
        "ties": int((diff == 0.0).sum()),
        "wilcoxon_p": wilcoxon_p,
        "sign_test_p": sign_p,
    }


def add_comparisons(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    metric: str,
    methods: tuple[str, ...],
    network: pd.DataFrame,
    task: pd.DataFrame,
    vs_unpruned: bool,
) -> None:
    comparison_label = "vs_unpruned" if vs_unpruned else "pairwise"
    pairs = [(method, "Baseline") for method in methods] if vs_unpruned else list(itertools.combinations(methods, 2))
    for unit, cells, index in [
        ("task_network_cell", network, ["task_short", "source_network_seed"]),
        ("task_mean", task, ["task_short"]),
    ]:
        family = f"{dataset}_{metric}_{comparison_label}_by_sparsity_{unit}"
        for pruning_pct in PCTS:
            wide = (
                cells[cells["pruning_pct"] == pruning_pct]
                .pivot(index=index, columns="method", values=metric)
            )
            for method_a, method_b in pairs:
                rows.append(
                    test_row(
                        dataset=dataset,
                        family=family,
                        metric=metric,
                        unit=unit,
                        pruning_pct=pruning_pct,
                        method_a=method_a,
                        method_b=method_b,
                        table=wide,
                    )
                )


def descriptive_rows(dataset: str, network: pd.DataFrame, task: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for unit, cells in [("task_network_cell", network), ("task_mean", task)]:
        for (pruning_pct, method), group in cells.groupby(["pruning_pct", "method"]):
            for metric in METRICS:
                values = group[metric].dropna()
                rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "unit": unit,
                    "pruning_pct": int(pruning_pct),
                    "method": method,
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "sd": sd(values),
                    "sem": sem(values),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "range": float(values.max() - values.min()),
                })
    return rows


def amplification_descriptives() -> pd.DataFrame:
    df = pd.read_csv(RESCALE_RUN)
    metrics = (
        "positive_prob_count",
        "rescale_mean",
        "rescale_q50",
        "rescale_q75",
        "rescale_q90",
        "rescale_q95",
        "rescale_q99",
        "rescale_q999",
        "rescale_max",
        "frac_rescale_gt10",
        "frac_rescale_gt20",
    )
    rows = []
    for method_family, group in df.groupby("method_family"):
        for metric in metrics:
            values = group[metric].dropna()
            rows.append({
                "dataset": "candidate_edge_rescale_distribution_p80",
                "method_family": method_family,
                "metric": metric,
                "unit": "reconstructed_run",
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "sd": sd(values),
                "sem": sem(values),
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    datasets = {
        "main_h512": (pd.read_csv(MAIN), MAIN_METHODS),
        "exploratory_capped_probe_h512": (pd.read_csv(CAP), CAP_METHODS),
        "matched_q50_h512": (enrich_matched(), MATCHED_METHODS),
    }
    tests: list[dict[str, object]] = []
    descriptives: list[dict[str, object]] = []
    for dataset, (df, methods) in datasets.items():
        network = network_cells(df)
        task = task_cells(network)
        descriptives.extend(descriptive_rows(dataset, network, task))
        for metric in ("sequence_retention", "post_rec_linear_rho_ratio_to_unpruned"):
            add_comparisons(
                tests,
                dataset=dataset,
                metric=metric,
                methods=methods,
                network=network,
                task=task,
                vs_unpruned=False,
            )
            add_comparisons(
                tests,
                dataset=dataset,
                metric=metric,
                methods=methods,
                network=network,
                task=task,
                vs_unpruned=True,
            )
    holm_adjust(tests, "wilcoxon_p", "holm_p_within_family")
    holm_adjust(tests, "sign_test_p", "sign_test_holm_p_within_family")
    test_df = pd.DataFrame(tests)
    test_df["wilcoxon_reject_holm_alpha_0_05"] = test_df["holm_p_within_family"] <= ALPHA
    test_df["sign_test_reject_holm_alpha_0_05"] = test_df["sign_test_holm_p_within_family"] <= ALPHA
    OUT.mkdir(parents=True, exist_ok=True)
    test_df.sort_values(["dataset", "metric", "unit", "pruning_pct", "method_a", "method_b"]).to_csv(
        TESTS_OUT, index=False
    )
    pd.DataFrame(descriptives).sort_values(
        ["dataset", "metric", "unit", "pruning_pct", "method"]
    ).to_csv(DESCRIPTIVES_OUT, index=False)
    amplification_descriptives().sort_values(["method_family", "metric"]).to_csv(
        AMPLIFICATION_OUT, index=False
    )
    print(f"wrote {TESTS_OUT}")
    print(f"wrote {DESCRIPTIVES_OUT}")
    print(f"wrote {AMPLIFICATION_OUT}")
    print(f"comparison tests: {len(test_df)}")
    print(f"descriptive rows: {len(descriptives)}")


if __name__ == "__main__":
    main()
