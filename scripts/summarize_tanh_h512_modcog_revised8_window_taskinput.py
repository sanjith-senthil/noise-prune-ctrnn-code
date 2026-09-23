#!/usr/bin/env python3
"""The 2x2 of how the covariance is estimated: window x drive.

Two choices go into the empirical covariance the noise score uses, and reviewer 2
asks about both:

    window   end-step only            vs  the whole trial
    drive    injected noise only      vs  task input + injected noise

The published estimator is **end-step, task input + noise**. This suite runs all
four combinations through S-NP sample-and-rescale at matched cost and asks
whether any of the three alternatives is better. Retention is normalised to each
network's own unpruned accuracy, paired within network, n = 24.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_windowdrive_p50_70"
RESULT_DIR = Path(f"results/{STEM}")
PUBLISHED = "endstep_taskplusnoise"
ARMS = ("endstep_taskplusnoise", "endstep_noiseonly",
        "wholetrial_taskplusnoise", "wholetrial_noiseonly")
LABEL = {"endstep_taskplusnoise": "end-step, task+noise  (published)",
         "endstep_noiseonly": "end-step, noise only",
         "wholetrial_taskplusnoise": "whole-trial, task+noise",
         "wholetrial_noiseonly": "whole-trial, noise only"}
AMOUNTS = (50, 60, 70)


def hl_ci(d):
    w = np.array([(d[i] + d[j]) / 2.0 for i in range(len(d)) for j in range(i, len(d))])
    w.sort(); n = len(d); z = 1.959963984540054
    k = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(0, min(k, len(w) - 1))
    return float(np.median(w)), float(w[k]), float(w[len(w) - 1 - k])


def holm(p):
    o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    a = ap.parse_args()
    files = [f for f in sorted(glob.glob(str(a.result_dir / "*.csv")))
             if "_summary" not in f and "_raw" not in f]
    if not files:
        raise SystemExit(f"no unit CSVs under {a.result_dir}")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)

    base = (d[d.strategy == "none"].drop_duplicates("source_model_label")
            .set_index("source_model_label")["post_acc_sequence"])
    p = d[d.strategy != "none"].copy()
    p["baseline"] = p["source_model_label"].map(base)
    if p["baseline"].isna().any():
        raise SystemExit("missing unpruned baseline for some runs")
    p["retention"] = p["post_acc_sequence"] / p["baseline"]
    p["arm"] = p["run_id"].str.extract(r"netseed\d+_(.+)_p\d+_pruneseed")
    p["pct"] = (p["amount"] * 100).round().astype(int)
    p["task_short"] = p["task"].str.replace("modcog:", "", regex=False)
    unit = ["task_short", "source_network_seed"]

    counts = p.groupby(["arm", "pct"]).size()
    print("cell counts:", "all 24" if (counts == 24).all() else f"\n{counts.to_string()}")

    print("\n=== mean sequence retention, n = 24 ===\n")
    piv = p.pivot_table(index="arm", columns="pct", values="retention")
    print(piv.reindex(ARMS).rename(index=LABEL).round(4).to_string())

    print("\n=== each alternative minus the published estimator, paired ===\n")
    rows, ps = [], []
    for pct in AMOUNTS:
        ref = p[(p.arm == PUBLISHED) & (p.pct == pct)].set_index(unit)["retention"]
        ref = ref[~ref.index.duplicated()]
        for arm in ARMS:
            if arm == PUBLISHED:
                continue
            v = p[(p.arm == arm) & (p.pct == pct)].set_index(unit)["retention"]
            v = v[~v.index.duplicated()]
            c = ref.index.intersection(v.index)
            dd = (v.loc[c] - ref.loc[c]).to_numpy()
            hl, lo, hi = hl_ci(dd); pv = float(wilcoxon(dd).pvalue)
            ps.append(pv)
            rows.append((arm, pct, hl, lo, hi, int((dd > 0).sum()), len(c), pv))
    adj = holm(np.array(ps))
    print(f"{'estimator':34s}{'spars':>7s}{'HL shift':>10s}{'95% CI':>22s}{'wins':>8s}{'p_holm':>10s}")
    for (arm, pct, hl, lo, hi, w, n, pv), q in zip(rows, adj):
        star = " *" if q < 0.05 else ""
        print(f"{LABEL[arm]:34s}{pct:6d}%{hl:+10.4f}  [{lo:+.4f},{hi:+.4f}]{w:5d}/{n:<3d}{q:10.2e}{star}")
    print("\n(a negative shift means the published end-step, task+noise estimator wins)")

    out = pd.DataFrame(rows, columns=["arm", "sparsity", "delta", "lo", "hi",
                                      "wins", "n", "p"])
    out["p_holm"] = adj
    out.to_csv(a.result_dir / f"{STEM}_summary.csv", index=False)
    p.to_csv(a.result_dir / f"{STEM}_raw.csv", index=False)
    print(f"\nwrote {a.result_dir / f'{STEM}_summary.csv'}")


if __name__ == "__main__":
    main()
