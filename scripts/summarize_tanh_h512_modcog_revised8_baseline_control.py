#!/usr/bin/env python3
"""Does a *less trained* network of the same width retain more?

Pairs each 6k-step H=512 network against the 12k-step network of the same task
and seed, both pruned by the same protocol on the same eval and score batches.
Width, task and pruning are held fixed; only unpruned accuracy differs. If
baseline accuracy is what drives retention -- the explanation offered for the
apparent H=1024 collapse -- the 6k networks must retain more.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr

STEM = ("task_preservation_tanh_h512_modcog_revised8_6k_seqbest"
        "_baseline_control_p50_80")
NEW = Path(f"results/{STEM}")
FROZEN = Path("paper_artifacts/official_h512_24net/data/revised_scope/task_preservation/"
              "revised_task_preservation_h512_24net_p50_80.csv")
TASKS = ("ctxdlydm2intseq", "ctxdlydm1intseq", "dlydm1intseq", "dlydm2intseq",
         "multidlydmintseq", "dm1seqr", "dm2seql", "dmsintseq")
AMOUNTS = (50, 60, 70, 80)
METHOD_OF = {"simulation_noise_prune_rescale": "snp_rescale",
             "simulation_noise_prune_mask_only": "snp_mask",
             "l1_unstructured": "magnitude"}


def hl(d):
    w = np.array([(d[i] + d[j]) / 2.0 for i in range(len(d)) for j in range(i, len(d))])
    return float(np.median(w))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.parse_args()
    files = [NEW / f"{t}_p{a}.csv" for t in TASKS for a in AMOUNTS]
    have = [f for f in files if f.exists()]
    print(f"6k suite: {len(have)}/{len(files)} unit files")
    n = pd.concat([pd.read_csv(f, low_memory=False) for f in have], ignore_index=True)
    n["task_short"] = n["task"].str.replace("modcog:", "", regex=False)
    base6 = (n[n.strategy == "none"].drop_duplicates("source_model_label")
             .set_index("source_model_label")["post_acc_sequence"])
    n = n[n.strategy != "none"].copy()
    n["baseline"] = n["source_model_label"].map(base6)
    n["retention"] = n["post_acc_sequence"] / n["baseline"]
    n["method"] = n["strategy"].map(METHOD_OF)
    n["pruning_pct"] = (n["amount"] * 100).round().astype(int)
    n["train"] = "6k"

    f = pd.read_csv(FROZEN, low_memory=False)
    f = f[(f.pruning_seed == 0) & (f.amount > 0)].copy()
    f["task_short"] = f["task"].str.replace("modcog:", "", regex=False)
    f["method"] = f["strategy"].map(METHOD_OF)
    f = f[f["method"].notna()]
    f["baseline"] = f["pre_acc_sequence"]
    f["retention"] = f["post_acc_sequence"] / f["baseline"]
    f["pruning_pct"] = (f["amount"] * 100).round().astype(int)
    f["train"] = "12k"

    cols = ["train", "method", "task_short", "source_network_seed", "pruning_pct",
            "baseline", "retention", "post_acc_sequence"]
    d = pd.concat([n[cols], f[cols]], ignore_index=True)

    print("\n=== unpruned accuracy ===")
    b = d.groupby(["train", "task_short"])["baseline"].mean().unstack(0)
    b["delta"] = b["12k"] - b["6k"]
    print(b.round(4).to_string())
    print(f"\nmean over 24 networks: 6k {d[d.train=='6k'].baseline.mean():.4f}   "
          f"12k {d[d.train=='12k'].baseline.mean():.4f}")

    unit = ["task_short", "source_network_seed"]
    print("\n=== retention, paired within (task, seed): 6k minus 12k ===")
    print(f"{'method':13s}{'spars':>7s}{'6k':>9s}{'12k':>9s}{'6k-12k':>10s}{'wins':>9s}{'p':>10s}")
    out = []
    for m in ("snp_rescale", "snp_mask", "magnitude"):
        for pct in AMOUNTS:
            a = d[(d.train == "6k") & (d.method == m) & (d.pruning_pct == pct)].set_index(unit)
            c = d[(d.train == "12k") & (d.method == m) & (d.pruning_pct == pct)].set_index(unit)
            a = a[~a.index.duplicated()]; c = c[~c.index.duplicated()]
            k = a.index.intersection(c.index)
            if len(k) < 4:
                continue
            dd = (a.loc[k, "retention"] - c.loc[k, "retention"]).to_numpy()
            pv = float(wilcoxon(dd).pvalue)
            print(f"{m:13s}{pct:6d}%{a.loc[k,'retention'].mean():9.4f}"
                  f"{c.loc[k,'retention'].mean():9.4f}{hl(dd):+10.4f}"
                  f"{int((dd>0).sum()):6d}/{len(k):<3d}{pv:10.2e}")
            out.append(dict(method=m, sparsity=pct, ret6k=a.loc[k, "retention"].mean(),
                            ret12k=c.loc[k, "retention"].mean(), delta=hl(dd),
                            wins=int((dd > 0).sum()), n=len(k), p=pv))

    print("\n=== pooled over both training lengths: Spearman(baseline, retention) ===")
    for m in ("snp_rescale", "snp_mask", "magnitude"):
        row = []
        for pct in AMOUNTS:
            s = d[(d.method == m) & (d.pruning_pct == pct)]
            r, p = spearmanr(s.baseline, s.retention)
            row.append(f"p{pct}: {r:+.3f}({p:.1e})")
        print(f"  {m:13s} " + "  ".join(row))

    pd.DataFrame(out).to_csv(NEW / f"{STEM}_summary.csv", index=False)
    d.to_csv(NEW / f"{STEM}_merged.csv", index=False)
    print(f"\nwrote {NEW / f'{STEM}_summary.csv'}")


if __name__ == "__main__":
    main()
