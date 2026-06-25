#!/usr/bin/env python3
"""Summarize S-NP rate/voltage and centering ablation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _fmt(value: float) -> str:
    return "" if pd.isna(value) else f"{float(value):.4f}"


def _print_markdown_table(df: pd.DataFrame) -> None:
    cols = [str(c) for c in df.columns]
    rows = [[_fmt(v) if isinstance(v, float) else str(v) for v in row] for row in df.to_numpy()]
    widths = [
        max(len(cols[i]), *(len(row[i]) for row in rows)) if rows else len(cols[i])
        for i in range(len(cols))
    ]
    print("| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |")
    print("| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |")
    for row in rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        default="results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_simnp_ablation_p50_80_fixedbatches.csv",
    )
    args = parser.parse_args()
    path = Path(args.csv)
    df = pd.read_csv(path)
    snp = df[df["strategy"] == "simulation_noise_prune_mask_only"].copy()
    if snp.empty:
        raise SystemExit(f"No simulation_noise_prune_mask_only rows found in {path}")

    group_cols = [
        "prune_sim_np_observable_space",
        "prune_sim_np_inject_space",
        "prune_sim_np_centering",
    ]
    overall = (
        snp.groupby(group_cols, dropna=False)
        .agg(
            rows=("post_acc_sequence", "size"),
            mean_post_acc_sequence=("post_acc_sequence", "mean"),
            sem_post_acc_sequence=(
                "post_acc_sequence",
                lambda s: s.std(ddof=1) / (len(s) ** 0.5) if len(s) > 1 else 0.0,
            ),
            mean_sequence_retention=(
                "post_acc_sequence",
                lambda s: float("nan"),
            ),
        )
        .reset_index()
    )
    if "pre_acc_sequence" in snp.columns:
        snp["sequence_retention"] = snp["post_acc_sequence"] / snp["pre_acc_sequence"].where(
            snp["pre_acc_sequence"] != 0
        )
        overall = (
            snp.groupby(group_cols, dropna=False)
            .agg(
                rows=("post_acc_sequence", "size"),
                mean_post_acc_sequence=("post_acc_sequence", "mean"),
                sem_post_acc_sequence=(
                    "post_acc_sequence",
                    lambda s: s.std(ddof=1) / (len(s) ** 0.5) if len(s) > 1 else 0.0,
                ),
                mean_sequence_retention=("sequence_retention", "mean"),
            )
            .reset_index()
        )

    by_amount = snp.pivot_table(
        index=group_cols,
        columns="amount",
        values="post_acc_sequence",
        aggfunc="mean",
    ).reset_index()

    print("\nS-NP ablation overall")
    _print_markdown_table(overall)
    print("\nS-NP ablation by pruning amount")
    _print_markdown_table(by_amount)


if __name__ == "__main__":
    main()
