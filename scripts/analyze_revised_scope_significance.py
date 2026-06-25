#!/usr/bin/env python3
"""Task-only significance analysis for the narrowed revised-paper scope."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_artifacts/official_h512_24net/data/revised_scope"
MAIN = DATA / "task_preservation/revised_task_preservation_h512_24net_p50_80.csv"
CAP = DATA / "capped_rescale/revised_snp_capped_rescale_probe_h512_24net_p50_80.csv"
MATCHED_CAP = (
    DATA
    / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_q50_capped_rescale_3seed_full8_p50_80.csv"
)
OUT = DATA / "significance"


def holm(rows: list[dict[str, object]]) -> None:
    for family in sorted({str(row["family"]) for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        ordered = sorted(selected, key=lambda row: float(row["wilcoxon_p"]))
        running = 0.0
        for rank, row in enumerate(ordered):
            running = max(running, (len(ordered) - rank) * float(row["wilcoxon_p"]))
            row["holm_p_within_family"] = min(running, 1.0)


def paired(
    wide: pd.DataFrame,
    a: str,
    b: str,
    alternative: str,
    family: str,
    unit: str,
    *,
    pruning_pct: int | None = None,
) -> dict[str, object]:
    sub = wide[[a, b]].dropna()
    diff = sub[a] - sub[b]
    nonzero = diff[diff != 0.0]
    p = (
        float(wilcoxon(nonzero, alternative=alternative, zero_method="wilcox").pvalue)
        if len(nonzero)
        else math.nan
    )
    wins = int(((diff > 0.0) if alternative == "greater" else (diff < 0.0)).sum())
    losses = int(((diff < 0.0) if alternative == "greater" else (diff > 0.0)).sum())
    sign_n = wins + losses
    row = {
        "family": family,
        "metric": "post_acc_sequence",
        "unit": unit,
        "method_a": a,
        "method_b": b,
        "alternative": alternative,
        "n": len(diff),
        "mean_a": float(sub[a].mean()),
        "mean_b": float(sub[b].mean()),
        "mean_diff_a_minus_b": float(diff.mean()),
        "median_diff_a_minus_b": float(diff.median()),
        "wins_for_alternative": wins,
        "losses_against_alternative": losses,
        "ties": int((diff == 0.0).sum()),
        "wilcoxon_p": p,
        "sign_test_p": float(binomtest(wins, sign_n, 0.5, alternative="greater").pvalue) if sign_n else math.nan,
    }
    if pruning_pct is not None:
        row["pruning_pct"] = int(pruning_pct)
    return row


def widths(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = (
        df[df["method"] != "Baseline"]
        .groupby(["task_short", "source_network_seed", "pruning_pct", "method"], as_index=False)
        .post_acc_sequence.mean()
    )
    wide = cell.pivot(index=["task_short", "source_network_seed", "pruning_pct"], columns="method", values="post_acc_sequence")
    task = cell.groupby(["task_short", "method"], as_index=False).post_acc_sequence.mean()
    task_wide = task.pivot(index="task_short", columns="method", values="post_acc_sequence")
    return wide, task_wide


def add_family(rows: list[dict[str, object]], df: pd.DataFrame, family: str, comparisons: list[tuple[str, str, str]]) -> None:
    wide, task_wide = widths(df)
    for a, b, alternative in comparisons:
        rows.append(paired(wide, a, b, alternative, f"{family}_cell", "task_network_pruning_pct_cell"))
        rows.append(paired(task_wide, a, b, alternative, f"{family}_taskmean", "task_mean"))


def add_per_sparsity_family(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    family: str,
    comparisons: list[tuple[str, str, str]],
) -> None:
    wide, _ = widths(df)
    pruning_pcts = df.loc[df["method"] != "Baseline", "pruning_pct"].dropna().unique()
    for pruning_pct in sorted(int(value) for value in pruning_pcts):
        pct_wide = wide.xs(pruning_pct, level="pruning_pct")
        pct_task_wide = (
            pct_wide.reset_index()
            .groupby("task_short", as_index=True)
            .mean(numeric_only=True)
        )
        for a, b, alternative in comparisons:
            rows.append(
                paired(
                    pct_wide,
                    a,
                    b,
                    alternative,
                    f"{family}_cell",
                    "task_network_cell",
                    pruning_pct=pruning_pct,
                )
            )
            rows.append(
                paired(
                    pct_task_wide,
                    a,
                    b,
                    alternative,
                    f"{family}_taskmean",
                    "task_mean",
                    pruning_pct=pruning_pct,
                )
            )


def matched_cap_rows() -> pd.DataFrame:
    df = pd.read_csv(MATCHED_CAP)
    method_map = {
        "noise_prune": "L-NP rescale",
        "noise_prune_capped_rescale": "L-NP capped q50",
        "simulation_noise_prune_rescale": "S-NP rescale",
        "simulation_noise_prune_capped_rescale": "S-NP capped q50",
    }
    df = df[df["strategy"].isin(method_map)].copy()
    df["method"] = df["strategy"].map(method_map)
    df["task_short"] = df["task"].astype(str).str.replace("modcog:", "", regex=False)
    df["pruning_pct"] = (df["amount"].astype(float) * 100.0).round().astype(int)
    return df


def main() -> None:
    rows: list[dict[str, object]] = []
    main = pd.read_csv(MAIN)
    cap = pd.read_csv(CAP)
    matched_cap = matched_cap_rows()
    add_family(
        rows,
        main,
        "revised_task",
        [
            ("S-NP mask", "Magnitude", "greater"),
            ("S-NP mask", "L-NP mask", "greater"),
            ("L-NP mask", "Magnitude", "greater"),
            ("L-NP rescale", "Magnitude", "greater"),
            ("L-NP rescale", "S-NP mask", "greater"),
            ("Magnitude", "Random", "greater"),
            ("OBS compensated", "Magnitude", "greater"),
            ("S-NP rescale", "Magnitude", "greater"),
        ],
    )
    add_family(
        rows,
        cap,
        "capped_rescale_probe",
        [
            ("S-NP capped q50", "S-NP rescale", "greater"),
            ("S-NP capped q75", "S-NP rescale", "greater"),
            ("S-NP capped q90", "S-NP rescale", "greater"),
            ("S-NP capped q50", "S-NP capped q75", "greater"),
            ("S-NP capped q50", "S-NP capped q90", "greater"),
            ("S-NP capped q75", "S-NP capped q90", "greater"),
        ],
    )
    add_per_sparsity_family(
        rows,
        main,
        "revised_task_rescale_vs_baseline_by_sparsity",
        [
            ("L-NP rescale", "OBS compensated", "greater"),
            ("S-NP rescale", "OBS compensated", "greater"),
            ("L-NP rescale", "Magnitude", "greater"),
            ("S-NP rescale", "Magnitude", "greater"),
        ],
    )
    add_family(
        rows,
        matched_cap,
        "matched_q50",
        [
            ("L-NP capped q50", "L-NP rescale", "greater"),
            ("S-NP capped q50", "S-NP rescale", "greater"),
            ("S-NP capped q50", "L-NP capped q50", "greater"),
        ],
    )
    holm(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tests = pd.DataFrame(rows)
    tests.to_csv(OUT / "revised_scope_task_only_significance_tests.csv", index=False)
    means = []
    for dataset, df in [
        ("revised_task", main),
        ("capped_rescale_probe", cap),
        ("matched_q50", matched_cap),
    ]:
        wide, task_wide = widths(df)
        for unit, table in [
            ("task_network_pruning_pct_cell", wide),
            ("task_mean", task_wide),
        ]:
            for method in table.columns:
                values = table[method].dropna()
                means.append({
                    "dataset": dataset,
                    "metric": "post_acc_sequence",
                    "unit": unit,
                    "method": method,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else math.nan,
                })
    pd.DataFrame(means).to_csv(OUT / "revised_scope_task_only_significance_means.csv", index=False)
    print(f"Wrote revised-scope significance artifacts under {OUT}")


if __name__ == "__main__":
    main()
