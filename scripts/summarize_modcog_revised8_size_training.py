#!/usr/bin/env python3
"""Summarize a size-matched training set and compare it to the paper's H=512 networks.

Reports selected-checkpoint validation sequence accuracy per task, in the same
form as Table 1, so the new size can be placed alongside the paper's set before
any pruning is run.

What to check before pruning
----------------------------
The size comparison is only clean if the two sets are comparably trained. Two
failure modes matter:

* **Undertrained** -- the larger network needs more than 12,000 steps to reach
  the same accuracy, so a later pruning difference would confound size with
  degree of convergence.
* **Ceiling** -- the larger network lands near perfect accuracy, which
  compresses retention (normalised to the unpruned network) and makes the
  pruning curves incomparable at the top end.

Both are visible here rather than after a pruning suite has been spent, which
is why this runs first. The script prints the per-task delta against H=512 and
flags any task within 0.02 of 1.0.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_OUTCOMES = Path("../data_release_staging/results/training_outcomes/"
                      "training_outcome_summary_by_task.csv")
ACC = "train_best_val_acc_sequence"
CEILING = 0.98


def suite_stem(h: int) -> str:
    return f"train_tanh_h{h}_modcog_revised8_to12k_seqbest_no_l2"


def load(h: int) -> pd.DataFrame:
    files = sorted(glob.glob(f"results/{suite_stem(h)}/*_seed*.csv"))
    if not files:
        raise SystemExit(f"no per-network CSVs under results/{suite_stem(h)}/")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    # keep the phase-2 row for each network; that is the checkpoint pruning uses
    d = d[d.run_id.str.startswith("continue_")].copy()
    d["task_short"] = d.run_id.str.extract(r"continue_modcog_([a-z0-9]+)_tanh")
    d["network_seed"] = d.run_id.str.extract(r"seed(\d+)$").astype(int)
    print(f"H={h}: {len(files)} unit CSVs -> {len(d)} trained networks "
          f"({d.task_short.nunique()} tasks x {d.network_seed.nunique()} seeds)")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=1024)
    args = ap.parse_args()
    h = args.hidden_size
    d = load(h)

    g = (d.groupby("task_short")[ACC]
         .agg(n="count", mean="mean", sd=lambda s: s.std(ddof=1), min="min", max="max")
         .reset_index())
    out = Path(f"results/{suite_stem(h)}/{suite_stem(h)}_summary_by_task.csv")
    g.assign(hidden_size=h).to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== H={h} selected-checkpoint validation sequence accuracy ===")
    print(f"{'task':<20}{'n':>3}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
    for _, r in g.sort_values("mean", ascending=False).iterrows():
        print(f"{r.task_short:<20}{int(r['n']):>3}{r['mean']:>9.4f}{r['sd']:>9.4f}"
              f"{r['min']:>9.4f}{r['max']:>9.4f}")

    if not PAPER_OUTCOMES.exists():
        print(f"\npaper H=512 outcomes not found at {PAPER_OUTCOMES}; skipping comparison")
        return
    paper = pd.read_csv(PAPER_OUTCOMES)
    col = next(c for c in paper.columns if "accuracy_mean" in c)
    paper["task_short"] = paper.iloc[:, 0].astype(str).str.replace("modcog:", "", regex=False)
    m = g.merge(paper[["task_short", col]].rename(columns={col: "h512_mean"}), on="task_short")
    m["delta"] = m["mean"] - m.h512_mean
    print(f"\n=== H={h} vs the paper's H=512, same protocol ===")
    print(f"{'task':<20}{'H'+str(h):>9}{'H512':>9}{'delta':>9}")
    for _, r in m.sort_values("delta", ascending=False).iterrows():
        print(f"{r.task_short:<20}{r['mean']:>9.4f}{r.h512_mean:>9.4f}{r.delta:>+9.4f}")
    print(f"\nmean delta {m.delta.mean():+.4f}   range {m.delta.min():+.4f} to {m.delta.max():+.4f}")

    ceil = m[m["mean"] >= CEILING]
    if len(ceil):
        print(f"\n⚠️  {len(ceil)} task(s) at or above {CEILING} — retention is normalised to the "
              f"unpruned network, so a ceiling compresses the pruning curves:")
        print("   " + ", ".join(f"{r.task_short} {r['mean']:.4f}" for _, r in ceil.iterrows()))
    else:
        print(f"\nno task at or above {CEILING}: retention remains well defined at this size.")
    if m.delta.mean() < -0.02:
        print("⚠️  H=%d trains WORSE on average than H=512 under the same budget; it may be "
              "undertrained, which would confound a later pruning comparison." % h)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
