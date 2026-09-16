#!/usr/bin/env python3
"""Does capping the rescale give the covariance term its job back?

Uncapped, E[W_hat] = W for any retention score -- the probability cancels out of
the expectation -- so the score can only move variance, and magnitude-proportional
retention already minimises that. Capping at c breaks the cancellation:

    E[W_hat_ij] = w_ij * min(1, p_ij * c)

biased low on exactly the edges whose retention probability falls below 1/c. The
score then decides which connections get under-restored, which is a first-order
job. Prediction: the covariance's contribution grows as the cap tightens.

The contrast at each cap level is

    full covariance score   minus   magnitude-only score (covariance dropped)

paired within trained network. `uncapped_full` doubles as a regression anchor
against the frozen task-preservation suite.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_cappedcov_p50_70"
RESULT_DIR = Path(f"results/{STEM}")
FROZEN = Path("paper_artifacts/official_h512_24net/data/revised_scope/task_preservation/"
              "revised_task_preservation_h512_24net_p50_80.csv")
TASKS = ("ctxdlydm2intseq","ctxdlydm1intseq","dlydm1intseq","dlydm2intseq",
         "multidlydmintseq","dm1seqr","dm2seql","dmsintseq")
AMOUNTS = (50, 60, 70)
UNITS = frozenset(f"{t}_p{a}.csv" for t in TASKS for a in AMOUNTS)
CAPS = ("uncapped", "q90", "q70", "q50", "q30")
TOL = 1e-9


def hl_ci(d, alpha=0.05):
    w = np.array([(d[i]+d[j])/2.0 for i in range(len(d)) for j in range(i, len(d))]); w.sort()
    n = len(d); z = 1.959963984540054
    k = int(np.floor(n*(n+1)/4 - z*np.sqrt(n*(n+1)*(2*n+1)/24)))
    k = max(0, min(k, len(w)-1))
    return float(np.median(w)), float(w[k]), float(w[len(w)-1-k])


def holm(p):
    o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p)-r)*p[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--allow-regression", action="store_true")
    a = ap.parse_args()

    files = [f for f in sorted(a.result_dir.glob("*.csv")) if f.name in UNITS]
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"merged {len(files)}/{len(UNITS)} unit CSVs -> {len(d)} rows")
    # Each sparsity shard carries its own unpruned baselines, so the same
    # source_model_label appears once per shard. They are the same network run
    # with no pruning, so they must agree -- check that, then dedupe.
    b = d[d.strategy == "none"][["source_model_label", "post_acc_sequence"]]
    spread = b.groupby("source_model_label")["post_acc_sequence"].agg(lambda v: v.max() - v.min())
    if float(spread.max()) > 1e-9:
        raise SystemExit(f"baselines disagree across shards, max spread {float(spread.max()):.3e}")
    base = b.drop_duplicates("source_model_label").set_index("source_model_label")["post_acc_sequence"]
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(base)
    if p["baseline_acc_sequence"].isna().any(): raise SystemExit("missing baseline")
    p["sequence_retention"] = p["post_acc_sequence"] / p["baseline_acc_sequence"]
    p["arm"] = p["run_id"].str.extract(r"netseed\d+_(.+)_p\d+_pruneseed")
    p["pruning_pct"] = (p["amount"] * 100).round().astype(int)
    if p["arm"].isna().any(): raise SystemExit("arm parse failed")
    p.to_csv(a.result_dir / f"{STEM}_raw.csv", index=False)

    counts = p.groupby(["arm", "pruning_pct"]).size()
    print("cell counts:", "all 24" if (counts == 24).all() else counts.to_string())

    if FROZEN.exists():
        f = pd.read_csv(FROZEN, low_memory=False)
        fr = f[(f.strategy == "simulation_noise_prune_rescale") & (f.pruning_seed == 0)
               & (f.amount.round(2).isin([a / 100 for a in AMOUNTS]))]
        key = ["source_model_label", "amount"]
        x = p[p.arm == "uncapped_full"].set_index(key)["post_acc_sequence"].sort_index()
        x = x[~x.index.duplicated()]
        y = fr.set_index(key)["post_acc_sequence"].sort_index(); y = y[~y.index.duplicated()]
        c = x.index.intersection(y.index).unique()
        worst = float((x.loc[c] - y.loc[c]).abs().max())
        print(f"regression: uncapped_full vs frozen, {len(c)} rows, max |delta| = {worst:.3e}"
              f"  {'exact' if worst <= TOL else '*** DRIFT ***'}")
        if worst > TOL and not a.allow_regression:
            raise SystemExit("regression against frozen artifact")

    unit = ["task", "source_network_seed"]
    wide = p.pivot_table(index=unit + ["pruning_pct"], columns="arm", values="sequence_retention")
    capinfo = p.groupby(["arm", "pruning_pct"])[["prune_rescale_cap_value",
                                                 "prune_frac_positive_amp_capped"]].mean()

    print("\nmean sequence retention, n = 24\n")
    print(f"{'cap':10s} {'sparsity':>9s} {'cap value':>10s} {'frac capped':>12s} "
          f"{'full':>8s} {'mag':>8s} {'covariance':>11s}")
    rows = []; ps = []
    for pct in AMOUNTS:
        sub_w = wide.xs(pct, level="pruning_pct")
        for cap in CAPS:
            fa, ma = f"{cap}_full", f"{cap}_mag"
            if fa not in sub_w.columns or ma not in sub_w.columns: continue
            s = sub_w[[fa, ma]].dropna()
            if not len(s): continue
            dd = (s[fa] - s[ma]).to_numpy()
            hl, lo, hi = hl_ci(dd); pv = float(wilcoxon(dd).pvalue)
            ps.append(pv)
            cv = capinfo.loc[(fa, pct), "prune_rescale_cap_value"]
            fc = capinfo.loc[(fa, pct), "prune_frac_positive_amp_capped"]
            rows.append((cap, pct, cv, fc, s[fa].mean(), s[ma].mean(),
                         hl, lo, hi, int((dd > 0).sum()), len(dd), pv))
    adj = holm(np.array(ps))
    for (cap, pct, cv, fc, fm, mm, hl, lo, hi, w, n, pv) in rows:
        print(f"{cap:10s} {pct:8d}% {cv:10.3f} {fc:12.2f} {fm:8.4f} {mm:8.4f} {hl:+11.4f}")
    print(f"\ncovariance contribution (full - magnitude), paired within network, "
          f"Holm over {len(rows)} cells\n")
    print(f"{'cap':10s} {'sparsity':>9s} {'HL shift':>9s} {'95% CI':>20s} {'wins':>8s} "
          f"{'p_raw':>9s} {'p_holm':>9s}")
    for i, (cap, pct, cv, fc, fm, mm, hl, lo, hi, w, n, pv) in enumerate(rows):
        star = " *" if adj[i] < 0.05 else ""
        print(f"{cap:10s} {pct:8d}% {hl:+9.4f} [{lo:+.4f}, {hi:+.4f}] {w:3d}/{n:<4d} "
              f"{pv:9.2e} {adj[i]:9.2e}{star}")
    out = pd.DataFrame(rows, columns=["cap","sparsity","cap_value","frac_capped","full","mag",
                                      "delta","lo","hi","wins","n","p"])
    out["p_holm"] = adj
    out.to_csv(a.result_dir / f"{STEM}_summary.csv", index=False)
    print(f"\nwrote {a.result_dir / f'{STEM}_summary.csv'}")


if __name__ == "__main__":
    main()
