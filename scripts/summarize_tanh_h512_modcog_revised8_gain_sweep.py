#!/usr/bin/env python3
"""Summarize the magnitude gain sweep.

The question the sweep answers is whether restoring ``rho(W_rec)`` to its
unpruned value is a meaningful operating point for a magnitude-pruned network,
or whether performance simply rises with gain and the matched point happens to
sit somewhere on that ramp.  Concretely: where is the argmax of retention over
gain, and how far is it from the spectral-matching factor?

Two checks run first and are fatal by default:

identity
    ``gain = 1.00`` must reproduce the plain ``l1_unstructured`` arm from the
    round-2 suite exactly.  If it does, then every arm in this family differs
    from plain magnitude by the scalar and nothing else, and no other code path
    is implicated in the effect.

spectral bracket
    The per-network spectral factor must lie inside the swept range at both
    sparsities, otherwise the argmax reported below is an artifact of where the
    sweep was truncated rather than a property of the response.

Scope: one network seed, one pruning seed, eight tasks.  Descriptive only.

Outputs (written next to the merged CSV):
  <stem>_raw.csv        merged per-run rows with retention
  <stem>_by_gain.csv    per (sparsity, gain), n = 8 tasks
  <stem>_argmax.csv     per (task, sparsity): best gain vs the spectral factor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_gain_sweep_p50_80"
ROUND2_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_round2_p50_80"
RESULT_DIR = Path(f"results/{STEM}")
ROUND2_RAW = Path(f"results/{ROUND2_STEM}/{ROUND2_STEM}_raw.csv")
CHANCE = 1.0 / 15.0
TOL = 1e-9


def load_raw(result_dir: Path) -> pd.DataFrame:
    files = [f for f in sorted(result_dir.glob("*.csv")) if not f.name.startswith(STEM)]
    if not files:
        raise SystemExit(f"No per-task CSVs found in {result_dir}")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"merged {len(files)} per-task CSVs -> {len(d)} rows")

    baseline = (d[d.strategy == "none"].set_index("source_model_label")["post_acc_sequence"])
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(baseline)
    if p["baseline_acc_sequence"].isna().any():
        raise SystemExit("missing unpruned baseline for some networks")
    p["sequence_retention"] = p["post_acc_sequence"] / p["baseline_acc_sequence"]
    p["floor_corrected_retention"] = (
        (p["post_acc_sequence"] - CHANCE) / (p["baseline_acc_sequence"] - CHANCE))
    p["task_short"] = p["run_id"].str.extract(r"gainsweep_([a-z0-9]+)_netseed")
    p["arm"] = p["run_id"].str.extract(r"netseed\d+_(spectral|g\d{3})_p\d\d_")
    p["pruning_pct"] = (p["amount"] * 100).round().astype(int)
    # the applied scalar, whichever mode produced it
    p["gain"] = p["prune_gain_restore_factor"].astype(float)
    if p["arm"].isna().any():
        raise SystemExit("could not parse arm label from some run_ids")
    return p


def identity_check(p: pd.DataFrame, strict: bool) -> None:
    if not ROUND2_RAW.exists():
        print(f"round-2 raw table not found ({ROUND2_RAW}); skipping identity check")
        return
    r2 = pd.read_csv(ROUND2_RAW, low_memory=False)
    r2 = r2[r2.arm == "magnitude"][["task_short", "pruning_pct", "post_acc_sequence"]]
    g1 = p[p.arm == "g100"][["task_short", "pruning_pct", "post_acc_sequence"]]
    m = g1.merge(r2, on=["task_short", "pruning_pct"], suffixes=("_sweep", "_round2"))
    if m.empty:
        print("identity check: no overlapping rows")
        return
    m["abs_diff"] = (m.post_acc_sequence_sweep - m.post_acc_sequence_round2).abs()
    worst, n_bad = float(m.abs_diff.max()), int((m.abs_diff > TOL).sum())
    print(f"identity check (gain = 1.00 vs plain magnitude): {len(m)} runs, "
          f"max |delta| = {worst:.3e}, {n_bad} exceed {TOL:g}")
    if n_bad:
        print(m[m.abs_diff > TOL].to_string(index=False))
        if strict:
            raise SystemExit(
                "IDENTITY FAILED: gain=1.00 does not reproduce plain magnitude, so the gain "
                "pruner differs from the plain pruner by more than the scalar. The sweep "
                "cannot be interpreted as isolating gain until this is explained.")
    else:
        print("  exact -- the gain arms differ from plain magnitude by the scalar alone")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--allow-identity-failure", action="store_true")
    args = ap.parse_args()

    p = load_raw(args.result_dir)
    p.to_csv(args.result_dir / f"{STEM}_raw.csv", index=False)
    identity_check(p, strict=not args.allow_identity_failure)

    swept = p[p.arm.str.startswith("g")].copy()
    spectral = p[p.arm == "spectral"].copy()

    # bracket check: is the spectral factor inside the swept range?
    lo, hi = swept.gain.min(), swept.gain.max()
    print(f"\nswept gain range [{lo:.2f}, {hi:.2f}]")
    for pct, g in spectral.groupby("pruning_pct"):
        inside = bool((g.gain > lo).all() and (g.gain < hi).all())
        print(f"  {pct}%: spectral factor {g.gain.min():.3f}-{g.gain.max():.3f}  "
              f"inside swept range: {inside}")
        if not inside:
            raise SystemExit(
                f"spectral factor at {pct}% falls outside the swept range; widen GAINS "
                f"before reading an argmax off this sweep.")

    by_gain = (swept.groupby(["pruning_pct", "gain"])
               .agg(n_tasks=("task_short", "nunique"),
                    retention_mean=("sequence_retention", "mean"),
                    retention_sem=("sequence_retention",
                                   lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                    floor_mean=("floor_corrected_retention", "mean"))
               .reset_index())
    by_gain.to_csv(args.result_dir / f"{STEM}_by_gain.csv", index=False)

    # per-network argmax vs the spectral factor
    rows = []
    for (task, pct), g in swept.groupby(["task_short", "pruning_pct"]):
        best = g.loc[g.sequence_retention.idxmax()]
        s = spectral[(spectral.task_short == task) & (spectral.pruning_pct == pct)]
        rows.append({
            "task_short": task, "pruning_pct": pct,
            "best_gain": float(best.gain),
            "best_retention": float(best.sequence_retention),
            "spectral_gain": float(s.gain.iloc[0]) if len(s) else np.nan,
            "spectral_retention": float(s.sequence_retention.iloc[0]) if len(s) else np.nan,
        })
    am = pd.DataFrame(rows)
    am["best_over_spectral"] = am.best_retention - am.spectral_retention
    am.to_csv(args.result_dir / f"{STEM}_argmax.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nretention vs fixed gain (mean over n = 8 tasks)")
    print(by_gain.pivot(index="gain", columns="pruning_pct",
                        values="retention_mean").round(3).to_string())
    print("\nspectral-convention reference point")
    print(spectral.groupby("pruning_pct")
          .agg(gain_mean=("gain", "mean"),
               retention_mean=("sequence_retention", "mean")).round(3).to_string())
    print("\nper-network argmax vs spectral factor")
    print(am.round(3).to_string(index=False))
    for pct, g in am.groupby("pruning_pct"):
        print(f"\n{pct}%: argmax gain median {g.best_gain.median():.2f} "
              f"(spectral median {g.spectral_gain.median():.2f}); "
              f"mean retention gap best - spectral = {g.best_over_spectral.mean():+.3f}")
    print("\nSCOPE: n = 8 tasks, one network seed, one pruning seed -- descriptive only.")


if __name__ == "__main__":
    main()
