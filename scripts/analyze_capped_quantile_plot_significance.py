#!/usr/bin/env python3
"""Plot-level significance tests for the L-NP/S-NP capped-rescale curve."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "paper_artifacts" / "official_h512_24net" / "data" / "revised_scope" / "capped_rescale"

ROWS_CSV = DATA_DIR / (
    "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_3seed_curve_rows.csv"
)
PLOT_CSV = DATA_DIR / (
    "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q0_to_q100_3seed_plot_points_with_sem.csv"
)
MAIN_CSV = DATA_DIR.parent / "task_preservation" / "revised_task_preservation_h512_24net_p50_80.csv"
OUT_CSV = DATA_DIR / (
    "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_3seed_vs_q100_plot_level_holm_tests.csv"
)


def sem(values: pd.Series) -> float:
    n = int(values.count())
    if n <= 1:
        return math.nan
    return float(values.std(ddof=1) / math.sqrt(n))


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        value = min(1.0, (m - rank) * float(p_value))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def family_from_method(method: str) -> str:
    if method.startswith("L-NP"):
        return "L-NP"
    if method.startswith("S-NP"):
        return "S-NP"
    raise ValueError(f"Cannot infer capped-rescale family from method {method!r}")


def cap_quantile(row: pd.Series) -> float:
    if pd.notna(row.get("cap_quantile")):
        return float(row["cap_quantile"])
    if pd.notna(row.get("rescale_cap_quantile")):
        return float(row["rescale_cap_quantile"])
    raise ValueError("Capped row is missing cap_quantile/rescale_cap_quantile")


def load_unit_table() -> pd.DataFrame:
    rows = pd.read_csv(ROWS_CSV)
    if "method_family" not in rows.columns:
        rows["method_family"] = rows["method"].map(family_from_method)
    if "cap_quantile" not in rows.columns:
        rows["cap_quantile"] = rows.apply(cap_quantile, axis=1)

    key_cols = ["method_family", "cap_quantile", "source_model_label", "pruning_pct"]
    seed_means = (
        rows.groupby(key_cols + ["pruning_seed"], dropna=False)["sequence_retention"]
        .mean()
        .reset_index()
    )
    cell_means = (
        seed_means.groupby(key_cols, dropna=False)["sequence_retention"]
        .mean()
        .reset_index()
    )
    unit = (
        cell_means.groupby(["method_family", "cap_quantile", "source_model_label"], dropna=False)[
            "sequence_retention"
        ]
        .mean()
        .reset_index()
        .rename(columns={"sequence_retention": "retention_mean_across_p50_80"})
    )
    return unit


def add_q100_units(unit: pd.DataFrame) -> pd.DataFrame:
    q100_units = []
    raw = pd.read_csv(MAIN_CSV)
    raw = raw[raw["method"].isin(["L-NP rescale", "S-NP rescale"])].copy()
    raw["method_family"] = raw["method"].map(family_from_method)
    # Recover q100 per-network plot units from the canonical uncapped rescale
    # rows with the same averaging order as the plot-points-with-sem CSV.
    for family in ["L-NP", "S-NP"]:
        family_rows = raw[raw["method_family"] == family]
        if family_rows.empty:
            raise ValueError(f"Missing q100 rows for {family}")
        seed_means = (
            family_rows.groupby(["source_model_label", "pruning_pct", "pruning_seed"], dropna=False)[
                "sequence_retention"
            ]
            .mean()
            .reset_index()
        )
        cell_means = (
            seed_means.groupby(["source_model_label", "pruning_pct"], dropna=False)["sequence_retention"]
            .mean()
            .reset_index()
        )
        collapsed = (
            cell_means.groupby("source_model_label", dropna=False)["sequence_retention"]
            .mean()
            .reset_index()
        )
        collapsed["method_family"] = family
        collapsed["cap_quantile"] = 1.0
        collapsed = collapsed.rename(columns={"sequence_retention": "retention_mean_across_p50_80"})
        q100_units.append(collapsed[unit.columns])
    return pd.concat([unit, *q100_units], ignore_index=True)


def main() -> None:
    unit = add_q100_units(load_unit_table())
    records = []
    for family in ["L-NP", "S-NP"]:
        baseline = unit[(unit["method_family"] == family) & (unit["cap_quantile"] == 1.0)]
        for q in sorted(unit.loc[(unit["method_family"] == family) & (unit["cap_quantile"] < 1), "cap_quantile"].unique()):
            capped = unit[(unit["method_family"] == family) & (unit["cap_quantile"] == q)]
            paired = capped.merge(
                baseline,
                on=["method_family", "source_model_label"],
                suffixes=("_capped", "_q100"),
            )
            if len(paired) != 24:
                raise ValueError(f"Expected n=24 for {family} q={q}, found {len(paired)}")
            a = paired["retention_mean_across_p50_80_capped"]
            b = paired["retention_mean_across_p50_80_q100"]
            diff = a - b
            nonzero = diff[diff != 0]
            wilcox_p = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
            wins = int((diff > 0).sum())
            losses = int((diff < 0).sum())
            ties = int((diff == 0).sum())
            sign_p = float(binomtest(min(wins, losses), wins + losses, 0.5, alternative="two-sided").pvalue)
            records.append(
                {
                    "dataset": "lnp_snp_capped_rescale_q10_to_q90_3seed_plot_points",
                    "metric": "sequence_retention",
                    "method_family": family,
                    "method_a": f"{family} capped rescale q{int(round(q * 100))}",
                    "method_b": f"{family} uncapped rescale q100",
                    "cap_quantile": q,
                    "cap_percentile": int(round(q * 100)),
                    "analysis_unit": "trained_network_after_pruning_seed_and_sparsity_mean",
                    "n": int(len(paired)),
                    "test_name": "paired two-sided Wilcoxon signed-rank test",
                    "wilcoxon_zero_method": "wilcox",
                    "wilcoxon_method": "auto",
                    "alternative": "two-sided",
                    "alpha": 0.05,
                    "multiple_comparison_method": "Holm",
                    "multiple_comparison_family": "all L-NP/S-NP q10-q90 capped-vs-q100 plot-level comparisons",
                    "multiple_comparison_family_size": 18,
                    "normality_assumed": False,
                    "mean_a": float(a.mean()),
                    "sd_a": float(a.std(ddof=1)),
                    "sem_a": sem(a),
                    "mean_b": float(b.mean()),
                    "sd_b": float(b.std(ddof=1)),
                    "sem_b": sem(b),
                    "mean_diff_a_minus_b": float(diff.mean()),
                    "median_diff_a_minus_b": float(diff.median()),
                    "min_diff_a_minus_b": float(diff.min()),
                    "max_diff_a_minus_b": float(diff.max()),
                    "n_nonzero_differences": int(nonzero.count()),
                    "wins_a_gt_b": wins,
                    "losses_a_lt_b": losses,
                    "ties": ties,
                    "wilcoxon_p": wilcox_p,
                    "sign_test_name": "exact two-sided binomial sign test",
                    "sign_test_p": sign_p,
                }
            )

    out = pd.DataFrame.from_records(records)
    out["holm_p_within_family"] = holm_adjust(out["wilcoxon_p"].tolist())
    out["sign_test_holm_p_within_family"] = holm_adjust(out["sign_test_p"].tolist())
    out["reject_holm_alpha_0p05"] = out["holm_p_within_family"] <= 0.05
    out["sign_test_reject_holm_alpha_0p05"] = out["sign_test_holm_p_within_family"] <= 0.05
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
