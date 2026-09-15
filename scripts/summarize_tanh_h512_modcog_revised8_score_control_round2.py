#!/usr/bin/env python3
"""Merge and summarize the round-2 reviewer control suite.

Scope caveat
------------
This suite is one trained-network seed and one pruning seed, so the analysis
unit is the *task* (n = 8), not the trained network (n = 24) that the paper's
inferential statistics are built on.  Nothing here is Holm-corrected or
significance-tested: at n = 8 the exact two-sided Wilcoxon floor is 2/2^8 =
0.0078, which a family of comparisons this size cannot clear, and quoting
uncorrected p-values from a reduced-scope suite alongside the paper's corrected
ones would be worse than quoting none.  The suite exists to read the direction
and magnitude of two controls quickly.  Promote an arm to the full n = 24
protocol before it appears in the manuscript with statistics attached.

Retention conventions
---------------------
Both are reported, because the reviewers raised each one:

``sequence_retention``
    ``post / unpruned``, matching the round-1 tables so the regression arms can
    be compared value-for-value.

``floor_corrected_retention``
    ``(post - chance) / (unpruned - chance)`` with ``chance = 1/15``, which is
    the quantity that does not credit an arm for sitting at chance.  The two
    diverge sharply exactly where an arm has collapsed, which is the regime
    several of these controls occupy.

Regression check
----------------
``magnitude``, ``lnp_rescale`` and ``snp_rescale`` are re-run here at the same
seeds as round 1 and must reproduce the frozen values.  The check is fatal by
default: a mismatch means a shared code path moved under the round-1 results,
and every number in both suites would be suspect.

Outputs (written next to the merged CSV):
  <stem>_raw.csv                 merged per-run rows with both retention metrics
  <stem>_summary_by_sparsity.csv per (arm, sparsity), n = 8 tasks
  <stem>_gain_audit.csv          per-run gain factors and spectral/L1 ratios
  <stem>_regression_check.csv    round-2 vs frozen round-1 for the shared arms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_round2_p50_80"
ROUND1_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_full_p50_80"
RESULT_DIR = Path(f"results/{STEM}")
ROUND1_RAW = Path(f"results/{ROUND1_STEM}/{ROUND1_STEM}_raw.csv")
ROUND1_FALLBACK = Path(
    f"../data_release_staging/results/revision_2026/score_controls/{ROUND1_STEM}_raw.csv"
)
CHANCE = 1.0 / 15.0
REGRESSION_ARMS = ("magnitude", "lnp_rescale", "snp_rescale")
REGRESSION_TOL = 1e-9

ARM_ORDER = [
    "lnp_rescale", "snp_rescale",
    "lnp_magnitude", "snp_magnitude",
    "magnitude_gain_rowl1", "magnitude_gain_jlin", "magnitude_gain_l2",
    "magnitude",
]


def load_raw(result_dir: Path) -> pd.DataFrame:
    files = [f for f in sorted(result_dir.glob("*.csv")) if not f.name.startswith(STEM)]
    if not files:
        raise SystemExit(f"No per-task CSVs found in {result_dir}")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"merged {len(files)} per-task CSVs -> {len(d)} rows")

    baseline = (
        d[d.strategy == "none"]
        .set_index("source_model_label")["post_acc_sequence"]
        .rename("baseline_acc_sequence")
    )
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(baseline)
    if p["baseline_acc_sequence"].isna().any():
        missing = sorted(p.loc[p.baseline_acc_sequence.isna(), "source_model_label"].unique())
        raise SystemExit(f"missing unpruned baseline for: {missing}")

    p["sequence_retention"] = p["post_acc_sequence"] / p["baseline_acc_sequence"]
    p["floor_corrected_retention"] = (
        (p["post_acc_sequence"] - CHANCE) / (p["baseline_acc_sequence"] - CHANCE)
    )
    p["arm"] = p["run_id"].str.extract(r"netseed\d+_(.+)_p\d\d_pruneseed")
    p["pruning_pct"] = (p["amount"] * 100).round().astype(int)
    p["task_short"] = p["run_id"].str.extract(r"ctrl_r2_([a-z0-9]+)_netseed")
    if p["arm"].isna().any():
        raise SystemExit("could not parse arm label from some run_ids")
    return p


def summarize(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, pct), g in p.groupby(["arm", "pruning_pct"]):
        row = {"arm": arm, "pruning_pct": pct, "n_tasks": g.task_short.nunique()}
        for metric in ("sequence_retention", "floor_corrected_retention"):
            v = g[metric]
            row[f"{metric}_mean"] = v.mean()
            row[f"{metric}_sd"] = v.std(ddof=1)
            row[f"{metric}_sem"] = v.std(ddof=1) / np.sqrt(len(v))
        if "post_rec_ct_abscissa" in g:
            row["abscissa_mean"] = g["post_rec_ct_abscissa"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    return out.sort_values(
        ["pruning_pct", "arm"], key=lambda s: s.map(order) if s.name == "arm" else s
    ).reset_index(drop=True)


def gain_audit(p: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in (
        "arm", "task_short", "pruning_pct", "prune_gain_restore_mode",
        "prune_gain_restore_factor", "prune_gain_restore_spectral_radius_ratio",
        "prune_gain_restore_offdiag_l1_ratio", "prune_prob_control",
        "prune_prob_control_prob_sum_before", "prune_prob_control_prob_sum_after",
        "prune_prob_control_pinned_at_one", "prune_kept_edges", "post_rec_weight_nz_count",
    ) if c in p.columns]
    audit = p[cols].copy()

    ctrl = audit[audit.get("prune_prob_control").eq("magnitude")] \
        if "prune_prob_control" in audit else audit.iloc[:0]
    if len(ctrl):
        delta = (ctrl["prune_prob_control_prob_sum_after"]
                 - ctrl["prune_prob_control_prob_sum_before"]).abs()
        rel = float((delta / ctrl["prune_prob_control_prob_sum_before"]).max())
        print(f"magnitude-probability arms: {len(ctrl)} runs, "
              f"max relative change in sum(p) = {rel:.2e}")
        if rel > 1e-9:
            raise SystemExit("magnitude probability control did not preserve sum(p)")
    return audit


def regression_check(p: pd.DataFrame, strict: bool) -> pd.DataFrame:
    source = ROUND1_RAW if ROUND1_RAW.exists() else ROUND1_FALLBACK
    if not source.exists():
        print(f"round-1 raw table not found ({ROUND1_RAW} or {ROUND1_FALLBACK}); "
              f"skipping regression check")
        return pd.DataFrame()
    r1 = pd.read_csv(source, low_memory=False)
    r1["arm"] = r1["run_id"].str.extract(r"netseed\d+_(.+)_p\d\d_pruneseed")
    r1["task_short"] = r1["run_id"].str.extract(r"ctrl_full_([a-z0-9]+)_netseed")
    r1["pruning_pct"] = (r1["amount"] * 100).round().astype(int)
    r1 = r1[(r1.source_network_seed == 0) & (r1.pruning_seed == 0)
            & r1.arm.isin(REGRESSION_ARMS)]

    key = ["arm", "task_short", "pruning_pct"]
    merged = (p[p.arm.isin(REGRESSION_ARMS)][key + ["post_acc_sequence"]]
              .merge(r1[key + ["post_acc_sequence"]], on=key, suffixes=("_round2", "_round1")))
    if merged.empty:
        print("no overlapping rows for the regression check")
        return merged
    merged["abs_diff"] = (merged.post_acc_sequence_round2
                          - merged.post_acc_sequence_round1).abs()
    worst = float(merged.abs_diff.max())
    n_bad = int((merged.abs_diff > REGRESSION_TOL).sum())
    print(f"\nregression vs frozen round 1 ({source}):")
    print(f"  {len(merged)} shared runs, max |delta post_acc_sequence| = {worst:.3e}, "
          f"{n_bad} exceed {REGRESSION_TOL:g}")
    if n_bad:
        print(merged[merged.abs_diff > REGRESSION_TOL].to_string(index=False))
        if strict:
            raise SystemExit(
                "REGRESSION: shared arms no longer reproduce the frozen round-1 values. "
                "A shared code path changed; do not trust either suite until this is resolved."
            )
    else:
        print("  reproduces exactly -- the shared code paths are unchanged")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--allow-regression", action="store_true",
                    help="report rather than fail on a round-1 mismatch")
    args = ap.parse_args()

    p = load_raw(args.result_dir)
    out = args.result_dir
    p.to_csv(out / f"{STEM}_raw.csv", index=False)

    audit = gain_audit(p)
    audit.to_csv(out / f"{STEM}_gain_audit.csv", index=False)

    reg = regression_check(p, strict=not args.allow_regression)
    if len(reg):
        reg.to_csv(out / f"{STEM}_regression_check.csv", index=False)

    s = summarize(p)
    s.to_csv(out / f"{STEM}_summary_by_sparsity.csv", index=False)

    print(f"\nsequence retention (post / unpruned), mean over n = 8 tasks, "
          f"network seed 0 / pruning seed 0")
    pivot = s.pivot(index="arm", columns="pruning_pct", values="sequence_retention_mean")
    pivot = pivot.reindex([a for a in ARM_ORDER if a in pivot.index])
    print(pivot.round(3).to_string())

    print(f"\nfloor-corrected retention ((post - 1/15) / (unpruned - 1/15))")
    pivot2 = s.pivot(index="arm", columns="pruning_pct",
                     values="floor_corrected_retention_mean")
    pivot2 = pivot2.reindex([a for a in ARM_ORDER if a in pivot2.index])
    print(pivot2.round(3).to_string())

    print(f"\nwrote {out}/{STEM}_{{raw,summary_by_sparsity,gain_audit,regression_check}}.csv")
    print("SCOPE: n = 8 tasks, one network seed, one pruning seed -- descriptive only.")


if __name__ == "__main__":
    main()
