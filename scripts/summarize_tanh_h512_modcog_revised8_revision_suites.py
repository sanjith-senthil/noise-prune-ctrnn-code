#!/usr/bin/env python3
"""Summarize the two peer-review revision suites.

Aggregation follows the conventions locked in PAPER_RESULTS_PROVENANCE.md:
pruning-seed replicates are averaged first within each
``(arm, sparsity, task, network seed)`` cell, the analysis unit is the trained
network (n = 24), and error bars are mean +/- SEM with sample SD (ddof=1)
across the 24 networks -- never across the pruning-seed rows.

Also reports, for the low-sparsity suite, the floor-corrected retention
``(acc - chance)/(acc_unpruned - chance)``.  Chance is 1/15: the readout has 17
units (1 fixation + 16 ring) but only 15 are ever targets, because ring
position 0 coincides with the fixation label and is masked from scoring.  This
was measured directly from the frozen evaluation batches and is identical for
all eight tasks.

Outputs, per suite, next to the per-task CSVs:
  <stem>_raw.csv                 merged rows with derived retention
  <stem>_summary.csv             per (arm, sparsity), n = 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOW_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_low_sparsity_p10_40"
NOISE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_necessity_p50_80"
RESULT_ROOT = Path("results")

# 15 attainable target classes; see module docstring.
CHANCE_ACCURACY = 1.0 / 15.0


def load_suite(stem: str) -> pd.DataFrame:
    d = RESULT_ROOT / stem
    files = [f for f in sorted(d.glob("*.csv")) if not f.name.startswith(stem)]
    if not files:
        raise SystemExit(f"no per-task CSVs in {d}")
    frame = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    baseline = (frame[frame.strategy == "none"]
                .set_index("source_model_label")["post_acc_sequence"])
    p = frame[frame.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(baseline)
    if p["baseline_acc_sequence"].isna().any():
        raise SystemExit("missing unpruned baseline for some networks")
    p["sequence_retention"] = p["post_acc_sequence"] / p["baseline_acc_sequence"]
    p["floor_corrected_retention"] = (
        (p["post_acc_sequence"] - CHANCE_ACCURACY)
        / (p["baseline_acc_sequence"] - CHANCE_ACCURACY)
    )
    p["pruning_pct"] = (p["amount"] * 100).round().astype(int)
    p["task_short"] = p["run_id"].str.extract(r"^[a-z]+_([a-z0-9]+)_netseed")
    p["arm"] = p["run_id"].str.extract(r"netseed\d+_(.+)_p\d\d(?:_pruneseed\d+)?$")
    if p["arm"].isna().any():
        raise SystemExit("could not parse arm from some run_ids")
    return p


def summarize(p: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keys = ["arm", "pruning_pct", "task_short", "source_network_seed"]
    cells = p.groupby(keys, as_index=False)[metrics].mean()
    out = []
    for metric in metrics:
        g = cells.groupby(["arm", "pruning_pct"])[metric]
        s = g.agg(n="count", mean="mean", sd=lambda x: x.std(ddof=1)).reset_index()
        s["sem"] = s["sd"] / np.sqrt(s["n"])
        s["metric"] = metric
        out.append(s)
    return pd.concat(out, ignore_index=True)


def report(stem: str, metrics: list[str], extra: list[str]) -> None:
    p = load_suite(stem)
    s = summarize(p, metrics)
    d = RESULT_ROOT / stem
    keep = ["run_id", "arm", "strategy", "task_short", "source_network_seed", "pruning_seed",
            "pruning_pct", "amount", "post_acc_sequence", "baseline_acc_sequence",
            "sequence_retention", "floor_corrected_retention", "post_rec_weight_nz_count",
            "post_rec_ct_abscissa"]
    keep += [c for c in extra if c in p.columns]
    p[[c for c in keep if c in p.columns]].to_csv(d / f"{stem}_raw.csv", index=False)
    s.to_csv(d / f"{stem}_summary.csv", index=False)

    print(f"\n=== {stem} ===")
    print(f"merged {len(p)} pruned rows; n = {int(s.n.max())} trained networks per cell")
    main = s[s.metric == "sequence_retention"]
    piv = main.pivot(index="arm", columns="pruning_pct", values="mean")
    sem = main.pivot(index="arm", columns="pruning_pct", values="sem")
    pcts = sorted(piv.columns)
    print(f"{'arm':<24}" + "".join(f"{f'p{c}':>16}" for c in pcts))
    for arm in piv.index:
        print(f"{arm:<24}" + "".join(
            f"{f'{piv.loc[arm,c]:.3f}+/-{sem.loc[arm,c]:.3f}':>16}"
            if pd.notna(piv.loc[arm, c]) else f"{'-':>16}" for c in pcts))
    print(f"wrote {d/f'{stem}_raw.csv'} and {d/f'{stem}_summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("low_sparsity", "noise_necessity", "both"), default="both")
    args = ap.parse_args()
    if args.suite in ("low_sparsity", "both"):
        report(LOW_STEM, ["sequence_retention", "floor_corrected_retention"],
               ["prune_kept_edges", "prune_enforced_density"])
    if args.suite in ("noise_necessity", "both"):
        report(NOISE_STEM, ["sequence_retention", "floor_corrected_retention"],
               ["prune_sim_np_sigma_factor", "prune_sigma_used", "prune_sigma_factor"])


if __name__ == "__main__":
    main()
