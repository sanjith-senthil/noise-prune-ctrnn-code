#!/usr/bin/env python3
"""Summarize the dm1/anti training baselines and compare them to the paper's eight tasks.

Reports selected-checkpoint validation sequence accuracy per task, in the same
form as Table 1, so the exclusion of dm1 and anti at ll. 92-95 / 259-261 can be
documented rather than asserted.

The comparison row set is read from the paper's own training outcomes, so the
two are aggregated identically: mean, sample SD (ddof=1), min and max across the
three network seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULT_ROOT = Path("results")
PHASE2_STEM = "continue_tanh_h512_modcog_dm1anti_to12k_lr0006_seqbest_no_l2"
SEEDS = (0, 1, 2)
PAPER_OUTCOMES = Path(
    "../data_release_staging/results/training_outcomes/training_outcome_summary_by_task.csv"
)
ACC_COL_CANDIDATES = ("train_best_val_acc_sequence", "pre_acc_sequence", "post_acc_sequence")


def load_runs() -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        f = RESULT_ROOT / f"{PHASE2_STEM}_seed{seed}.csv"
        if not f.exists():
            raise SystemExit(f"missing {f} -- run the training script first")
        frames.append(pd.read_csv(f, low_memory=False))
    d = pd.concat(frames, ignore_index=True)
    d["task_short"] = d["task"].str.replace("modcog:", "", regex=False)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=RESULT_ROOT / "dm1anti_baseline_summary.csv")
    args = ap.parse_args()

    d = load_runs()
    acc_col = next((c for c in ACC_COL_CANDIDATES if c in d.columns and d[c].notna().any()), None)
    if acc_col is None:
        raise SystemExit(f"no accuracy column found among {ACC_COL_CANDIDATES}")
    loss_col = "train_best_val_loss" if "train_best_val_loss" in d.columns else None

    agg = {"n_seeds": (acc_col, "count"),
           "sequence_accuracy_mean": (acc_col, "mean"),
           "sequence_accuracy_sd": (acc_col, lambda s: s.std(ddof=1)),
           "sequence_accuracy_min": (acc_col, "min"),
           "sequence_accuracy_max": (acc_col, "max")}
    if loss_col:
        agg["selected_checkpoint_loss_mean"] = (loss_col, "mean")
    out = d.groupby("task_short").agg(**agg).reset_index()
    out["accuracy_column"] = acc_col
    out.to_csv(args.out, index=False)

    print(f"accuracy column: {acc_col}   (n = {len(d)} runs, {d.task_short.nunique()} tasks)")
    print(f"\n{'task':<20}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
    for _, r in out.iterrows():
        print(f"{r.task_short:<20}{r.sequence_accuracy_mean:>9.4f}{r.sequence_accuracy_sd:>9.4f}"
              f"{r.sequence_accuracy_min:>9.4f}{r.sequence_accuracy_max:>9.4f}")

    if PAPER_OUTCOMES.exists():
        paper = pd.read_csv(PAPER_OUTCOMES)
        col = next((c for c in paper.columns if "accuracy_mean" in c or c == "sequence_accuracy_mean"), None)
        if col:
            print(f"\nthe paper's eight tasks, same metric ({col}):")
            for _, r in paper.sort_values(col).iterrows():
                print(f"{r.iloc[0]:<20}{r[col]:>9.4f}")
            lo, hi = paper[col].min(), paper[col].max()
            print(f"\npaper range: {lo:.4f} - {hi:.4f}   "
                  f"dm1/anti: {out.sequence_accuracy_mean.min():.4f} - {out.sequence_accuracy_mean.max():.4f}")
            if out.sequence_accuracy_mean.min() > hi:
                print("=> both excluded tasks exceed every retained task: 'trivially solvable' is supported.")
            else:
                print("=> NOT uniformly above the retained tasks; scope the exclusion claim accordingly.")
    print(f"\nwrote {args.out}")
    print("NOTE: unpruned accuracy supports 'trivially solvable'. It does not by itself show that")
    print("      pruning comparisons on these tasks are uninformative -- that needs a pruning suite.")


if __name__ == "__main__":
    main()
