#!/usr/bin/env python3
"""Summarize the S-NP covariance-estimator screen (50-70% sparsity).

Eighteen estimator variants, all paired within trained network against the
published estimator (``paper``). Reported per sparsity: mean retention at
n = 24, the Hodges-Lehmann median paired shift against ``paper`` with a
distribution-free CI, the sign count, and a Wilcoxon signed-rank p-value with a
Holm step-down over the whole family of arm x sparsity comparisons.

This is a screen at one pruning seed: it ranks variants, it does not certify
them. Whatever wins here earns the full three-pruning-seed treatment.

Regression gate (fatal unless --allow-regression)
-------------------------------------------------
The ``paper`` arm is configured identically to the frozen task-preservation
suite -- same frozen score batches, same eval batches, same seeds, same
sample-capped covariance path -- so its rows must reproduce those results
exactly. Any drift means the new estimator plumbing perturbed the published
path, which would invalidate every comparison in the table.

``budget_bridge`` is the same estimator moved onto the fixed-rollout budget. It
is not expected to be bit-identical (98 whole rollouts against 97 whole plus one
truncated), but it must be statistically indistinguishable from ``paper``; if it
is not, the budget change is doing work on its own and the window and centering
arms cannot be read cleanly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_covquality_p50_70"
RESULT_DIR = Path(f"results/{STEM}")
FROZEN = Path("paper_artifacts/official_h512_24net/data/revised_scope/task_preservation/"
              "revised_task_preservation_h512_24net_p50_80.csv")
REF = "paper"
AMOUNTS = (50, 60, 70)
TASKS = ("ctxdlydm2intseq", "ctxdlydm1intseq", "dlydm1intseq", "dlydm2intseq",
         "multidlydmintseq", "dm1seqr", "dm2seql", "dmsintseq")
UNIT_NAMES = frozenset(f"{t}_p{p}.csv" for t in TASKS for p in AMOUNTS)
TOL = 1e-9

ESTIMATORS = (
    "paper", "budget_bridge",
    "cond_term", "traj_full", "cond_full",
    "cond_full_vv", "cond_full_rv", "cond_full_vr",
    "cond_full_rr_sf050", "cond_full_rr_sf025",
    "cond_full_rv_sf050", "cond_full_rv_sf025", "cond_full_vv_sf025",
    "cond_term_rv",
    "cond_full_rr_sigrate", "cond_full_rv_sigrate",
    "cond_full_rr_noinput", "cond_full_rv_noinput",
)
# Clip first (as published), then the water-filled twin of each, so the table
# reads down the estimator axis within each normaliser.
ARM_ORDER = tuple(ESTIMATORS) + tuple(f"{e}__wf" for e in ESTIMATORS)


def hodges_lehmann_ci(d: np.ndarray, alpha: float = 0.05):
    """Hodges-Lehmann point estimate and distribution-free CI of the median shift."""
    w = np.array([(d[i] + d[j]) / 2.0 for i in range(len(d)) for j in range(i, len(d))])
    w.sort()
    n = len(d)
    z = 1.959963984540054   # normal approximation; adequate at n = 24
    m = n * (n + 1) / 2
    k = int(np.floor(m / 2 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(0, min(k, len(w) - 1))
    return float(np.median(w)), float(w[k]), float(w[len(w) - 1 - k])


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals)); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def load(result_dir: Path) -> pd.DataFrame:
    files = [f for f in sorted(result_dir.glob("*.csv")) if f.name in UNIT_NAMES]
    if not files:
        raise SystemExit(f"No per-unit CSVs found in {result_dir}")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"merged {len(files)}/{len(UNIT_NAMES)} per-unit CSVs -> {len(d)} rows")
    base = d[d.strategy == "none"].set_index("source_model_label")["post_acc_sequence"]
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(base)
    if p["baseline_acc_sequence"].isna().any():
        raise SystemExit("missing unpruned baseline for some rows")
    p["sequence_retention"] = p["post_acc_sequence"] / p["baseline_acc_sequence"]
    p["task_short"] = p["run_id"].str.extract(r"covq_([a-z0-9]+)_netseed")
    p["arm"] = p["run_id"].str.extract(r"netseed\d+_(.+)_p\d+_pruneseed")
    p["pruning_pct"] = (p["amount"] * 100).round().astype(int)
    if p["arm"].isna().any():
        raise SystemExit("could not parse arm from run_ids")
    unknown = sorted(set(p["arm"]) - set(ARM_ORDER))
    if unknown:
        raise SystemExit(f"unexpected arms: {unknown}")
    return p


def regression_gate(p: pd.DataFrame, allow: bool) -> None:
    if not FROZEN.exists():
        print(f"WARNING: frozen artifact not found at {FROZEN}; regression gate skipped")
        return
    f = pd.read_csv(FROZEN, low_memory=False)
    f = f[(f.strategy == "simulation_noise_prune_rescale") & (f.pruning_seed == 0)
          & (f.amount.round(2).isin([a / 100 for a in AMOUNTS]))]
    key = ["source_model_label", "amount"]
    a = (p[p.arm == REF].set_index(key)["post_acc_sequence"].sort_index())
    b = (f.set_index(key)["post_acc_sequence"].sort_index())
    common = a.index.intersection(b.index)
    if len(common) == 0:
        print("WARNING: no overlapping rows for the regression gate")
        return
    diff = (a.loc[common] - b.loc[common]).abs()
    worst = float(diff.max())
    print(f"\nregression gate: '{REF}' vs frozen task-preservation, {len(common)} paired rows, "
          f"max |delta post_acc_sequence| = {worst:.3e}")
    if worst > TOL:
        bad = diff[diff > TOL]
        msg = (f"REGRESSION: {len(bad)} rows differ from the frozen artifact "
               f"(worst {worst:.3e}). The published path has been perturbed.")
        if allow:
            print("  " + msg + "  [--allow-regression]")
        else:
            raise SystemExit("  " + msg)
    else:
        print("  exact -- the published estimator is unchanged.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--allow-regression", action="store_true")
    args = ap.parse_args()

    p = load(args.result_dir)
    p.to_csv(args.result_dir / f"{STEM}_raw.csv", index=False)

    counts = p.groupby(["arm", "pruning_pct"]).size().unstack(fill_value=0)
    if not (counts == 24).all().all():
        print("\nWARNING: not every (arm, sparsity) cell has 24 networks:")
        print(counts[(counts != 24).any(axis=1)].to_string())
    else:
        print("cell counts: every (arm, sparsity) has n = 24 trained networks")

    regression_gate(p, args.allow_regression)

    unit = ["task_short", "source_network_seed"]
    wide = p.pivot_table(index=unit + ["pruning_pct"], columns="arm",
                         values="sequence_retention")

    rows = []
    for pct in AMOUNTS:
        sub = wide.xs(pct, level="pruning_pct")
        ref = sub[REF]
        for arm in ARM_ORDER:
            if arm not in sub.columns:
                continue
            v = sub[arm]
            d = (v - ref).dropna().to_numpy()
            rec = {"sparsity": pct, "arm": arm, "n": int(v.notna().sum()),
                   "retention": float(v.mean()), "sd": float(v.std(ddof=1))}
            if arm == REF or len(d) == 0 or np.allclose(d, 0.0):
                rec.update({"delta": 0.0, "lo": 0.0, "hi": 0.0, "wins": "-", "p": np.nan})
            else:
                hl, lo, hi = hodges_lehmann_ci(d)
                rec.update({"delta": hl, "lo": lo, "hi": hi,
                            "wins": f"{int((d > 0).sum())}/{len(d)}",
                            "p": float(wilcoxon(d).pvalue)})
            rows.append(rec)
    res = pd.DataFrame(rows)
    mask = res["p"].notna()
    res.loc[mask, "p_holm"] = holm(res.loc[mask, "p"].to_numpy())

    print(f"\nS-NP covariance-estimator screen -- sequence retention, n = 24, 1 pruning seed")
    print(f"paired shifts are vs '{REF}'; Holm over all {int(mask.sum())} comparisons\n")
    for pct in AMOUNTS:
        sub = res[res.sparsity == pct]
        print(f"--- {pct}% sparsity " + "-" * 62)
        print(f"{'arm':24s} {'reten':>7s} {'sd':>6s} {'vs paper':>9s} "
              f"{'95% CI':>18s} {'wins':>6s} {'p_holm':>9s}")
        for _, r in sub.iterrows():
            ci = "" if r.arm == REF else f"[{r.lo:+.3f}, {r.hi:+.3f}]"
            ph = "" if pd.isna(r.get("p_holm")) else f"{r.p_holm:.2e}"
            star = " *" if (not pd.isna(r.get("p_holm")) and r.p_holm < 0.05) else ""
            print(f"{r.arm:24s} {r.retention:7.4f} {r.sd:6.3f} {r.delta:+9.4f} "
                  f"{ci:>18s} {str(r.wins):>6s} {ph:>9s}{star}")
        print()

    print("--- water-filling alone: each estimator, clip -> waterfill " + "-" * 18)
    print(f"{'estimator':24s} " + "  ".join(f"{a:>16d}%" for a in AMOUNTS))
    for est in ESTIMATORS:
        cells = []
        for pct in AMOUNTS:
            sub = wide.xs(pct, level="pruning_pct")
            if est not in sub.columns or f"{est}__wf" not in sub.columns:
                cells.append(f"{'-':>17s}"); continue
            d = (sub[f"{est}__wf"] - sub[est]).dropna().to_numpy()
            if len(d) == 0 or np.allclose(d, 0.0):
                cells.append(f"{0.0:+8.4f}{'':>9s}"); continue
            pv = float(wilcoxon(d).pvalue)
            cells.append(f"{np.mean(d):+8.4f} ({int((d > 0).sum()):2d}/{len(d)}) {'*' if pv < 0.05 else ' '}")
        print(f"{est:24s} " + "  ".join(cells))
    print()

    diag = (p.groupby("arm")
             .agg(sigma_used=("prune_sigma_used", "mean"),
                  cov_trace=("prune_empirical_cov_trace", "mean"),
                  cov_diag=("prune_empirical_cov_diag_mean", "mean"),
                  samples=("prune_sample_count", "mean"),
                  kept_edges=("prune_kept_edges", "mean"),
                  nz_edges=("post_rec_weight_nz_count", "mean"),
                  shortfall=("prune_prob_sum_shortfall", "mean"),
                  capped=("prune_capped_probs", "mean") if "prune_capped_probs" in p else ("amount", "size"))
             .reindex(ARM_ORDER).dropna(how="all"))
    print("estimator diagnostics (mean over all runs of the arm)")
    print(diag.to_string(float_format=lambda x: f"{x:,.4g}"))

    out = args.result_dir / f"{STEM}_summary.csv"
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
