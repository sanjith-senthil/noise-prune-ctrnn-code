#!/usr/bin/env python3
"""Can a constant cap replace the per-network q50 quantile cap?

Two questions, because the quantile cap varies very differently along two axes:
  across NETWORKS at fixed sparsity  -- only 1.15x min-to-max, sd ~4%
  across SPARSITY                    -- 1.67x (2.37 -> 3.95)

`fixedps_*`   one constant per sparsity (2.369 / 2.961 / 3.948) -- does the cap
              need to adapt to the network?
`fixedglob_*` one constant everywhere (3.092) -- does it need to adapt to
              sparsity as well? That value is ~30% high at 50% and ~22% low at 70%.

Both score arms run at each cap, so the covariance contribution stays measurable
under a fixed cap and not just the absolute retention -- a constant that recovers
retention while killing the covariance advantage would be no use.

The q50 quantile reference is joined from the cappedcov suite. Both suites share
checkpoints, eval batches and seeds, so their unpruned baselines must agree; that
is checked and is what licenses the join.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_fixedcap_p50_70"
REF_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_cappedcov_p50_70"
AMOUNTS = (50, 60, 70)
TASKS = ("ctxdlydm2intseq","ctxdlydm1intseq","dlydm1intseq","dlydm2intseq",
         "multidlydmintseq","dm1seqr","dm2seql","dmsintseq")
TOL = 1e-9


def hl_ci(d):
    w = np.array([(d[i]+d[j])/2.0 for i in range(len(d)) for j in range(i, len(d))]); w.sort()
    n = len(d); z = 1.959963984540054
    k = max(0, min(int(np.floor(n*(n+1)/4 - z*np.sqrt(n*(n+1)*(2*n+1)/24))), len(w)-1))
    return float(np.median(w)), float(w[k]), float(w[len(w)-1-k])


def holm(p):
    o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p)-r)*p[i]); adj[i] = min(1.0, run)
    return adj


def load(stem):
    files = [f for f in glob.glob(f"results/{stem}/*.csv")
             if Path(f).name in {f"{t}_p{a}.csv" for t in TASKS for a in AMOUNTS}]
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    b = d[d.strategy == "none"][["source_model_label", "post_acc_sequence"]]
    sp = b.groupby("source_model_label")["post_acc_sequence"].agg(lambda v: v.max()-v.min())
    if float(sp.max()) > TOL:
        raise SystemExit(f"{stem}: baselines disagree across shards, {float(sp.max()):.3e}")
    base = b.drop_duplicates("source_model_label").set_index("source_model_label")["post_acc_sequence"]
    p = d[d.strategy != "none"].copy()
    p["ret"] = p.post_acc_sequence / p.source_model_label.map(base)
    p["arm"] = p.run_id.str.extract(r"netseed\d+_(.+)_p\d+_pruneseed")
    p["pct"] = pd.to_numeric(p["amount"], errors="coerce").mul(100).round().astype("Int64")
    return p, base


def main():
    ap = argparse.ArgumentParser(); ap.parse_args()
    p, base = load(STEM)
    r, rbase = load(REF_STEM)

    shared = base.index.intersection(rbase.index)
    drift = float((base.loc[shared] - rbase.loc[shared]).abs().max())
    print(f"join gate: unpruned baselines across the two suites, {len(shared)} networks, "
          f"max |delta| = {drift:.3e}  {'exact' if drift <= TOL else '*** DRIFT ***'}")
    if drift > TOL:
        raise SystemExit("baselines differ; the cross-suite join is not licensed")

    counts = p.groupby(["arm", "pct"]).size()
    print("cell counts:", "all 24" if (counts == 24).all() else counts.to_string())

    idx = ["task", "source_network_seed", "pct"]
    w = p.pivot_table(index=idx, columns="arm", values="ret")
    wr = r.pivot_table(index=idx, columns="arm", values="ret")
    both = w.join(wr[["q50_full", "q50_mag", "uncapped_full"]], how="inner")

    print("\n=== mean sequence retention, n = 24 ===\n")
    cols = ["uncapped_full", "q50_full", "fixedps_full", "fixedglob_full",
            "q50_mag", "fixedps_mag", "fixedglob_mag"]
    print(f"{'arm':>16s} |" + "".join(f"{a:>10d}%" for a in AMOUNTS))
    for c in cols:
        if c not in both.columns: continue
        print(f"{c:>16s} |" + "".join(f"{both.xs(a, level='pct')[c].mean():11.4f}" for a in AMOUNTS))

    CONTRASTS = (
        ("fixedps_full",   "q50_full",      "per-sparsity constant vs q50 quantile (covariance)"),
        ("fixedglob_full", "q50_full",      "global constant vs q50 quantile (covariance)"),
        ("fixedglob_full", "fixedps_full",  "global vs per-sparsity constant (covariance)"),
        ("fixedps_full",   "fixedps_mag",   "COVARIANCE CONTRIBUTION under per-sparsity constant"),
        ("fixedglob_full", "fixedglob_mag", "COVARIANCE CONTRIBUTION under global constant"),
        ("fixedps_full",   "uncapped_full", "per-sparsity constant vs no cap (covariance)"),
    )
    rows = []; ps = []
    for a, b_, lab in CONTRASTS:
        for pct in AMOUNTS:
            s = both.xs(pct, level="pct")[[a, b_]].dropna()
            if not len(s): continue
            dd = (s[a] - s[b_]).to_numpy(); h, lo, hi = hl_ci(dd)
            pv = float(wilcoxon(dd).pvalue); ps.append(pv)
            rows.append((lab, a, b_, pct, h, lo, hi, int((dd > 0).sum()), len(dd), pv))
    adj = holm(np.array(ps))
    print(f"\n=== paired contrasts, Holm over {len(rows)} cells ===\n")
    last = None
    for i, (lab, a, b_, pct, h, lo, hi, win, n, pv) in enumerate(rows):
        if lab != last:
            print(f"\n  {lab}   [{a} - {b_}]"); last = lab
        star = " *" if adj[i] < 0.05 else ""
        print(f"    {pct}%  {h:+.4f} [{lo:+.4f}, {hi:+.4f}]  {win:2d}/{n}  p_holm={adj[i]:.2e}{star}")
    out = pd.DataFrame(rows, columns=["contrast","test","ref","sparsity","delta","lo","hi","wins","n","p"])
    out["p_holm"] = adj
    out.to_csv(f"results/{STEM}/{STEM}_summary.csv", index=False)
    print(f"\nwrote results/{STEM}/{STEM}_summary.csv")


if __name__ == "__main__":
    main()
