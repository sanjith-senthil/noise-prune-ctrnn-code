#!/usr/bin/env python3
"""Does a better-estimated covariance help S-NP, and where?

Six arms at n = 24 (8 tasks x 3 network seeds), one pruning seed, 50-70%:

  mag_mask          magnitude top-k
  mask_paper        S-NP deterministic mask, covariance as published
  mask_rv           S-NP deterministic mask, covariance estimated over the whole
                    trial, conditional centering, noise injected into rate and
                    observed in voltage
  rescale_paper     S-NP sample-and-rescale, covariance as published
  rescale_paper_wf  same, with water-filled instead of clipped probabilities
  rescale_rv_wf     sample-and-rescale with the improved covariance, water-filled

The deterministic pair is the informative one. Under 1/p rescaling the pruned
matrix has W as its mask-average for any score, so the score controls only the
variance -- and magnitude-proportional retention already minimises that. The
score therefore has first-order leverage on the deterministic mask and only
second-order leverage after rescaling, so an improved covariance should show up
in mask_rv - mask_paper and not in rescale_rv_wf - rescale_paper.

mag_mask, mask_paper and rescale_paper are configured identically to the frozen
task-preservation suite and must reproduce it exactly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_covverdict_p50_70"
RESULT_DIR = Path(f"results/{STEM}")
FROZEN = Path("paper_artifacts/official_h512_24net/data/revised_scope/task_preservation/"
              "revised_task_preservation_h512_24net_p50_80.csv")
AMOUNTS = (50, 60, 70)
TASKS = ("ctxdlydm2intseq","ctxdlydm1intseq","dlydm1intseq","dlydm2intseq",
         "multidlydmintseq","dm1seqr","dm2seql","dmsintseq")
UNITS = frozenset(f"{t}_p{p}.csv" for t in TASKS for p in AMOUNTS)
ARMS = ("mag_mask","mask_paper","mask_rv","rescale_paper","rescale_paper_wf","rescale_rv_wf")
# (test, reference, what the contrast isolates)
CONTRASTS = (
    ("mask_paper",       "mag_mask",      "covariance, deterministic (published estimator)"),
    ("mask_rv",          "mask_paper",    "IMPROVED covariance, deterministic"),
    ("mask_rv",          "mag_mask",      "covariance, deterministic (improved estimator)"),
    ("rescale_paper_wf", "rescale_paper", "water-filling alone, rescaled"),
    ("rescale_rv_wf",    "rescale_paper", "IMPROVED covariance + water-fill, rescaled"),
    ("rescale_rv_wf",    "rescale_paper_wf", "IMPROVED covariance alone, rescaled"),
)
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
    base = d[d.strategy == "none"].set_index("source_model_label")["post_acc_sequence"]
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(base)
    if p["baseline_acc_sequence"].isna().any():
        raise SystemExit("missing unpruned baseline")
    p["sequence_retention"] = p["post_acc_sequence"] / p["baseline_acc_sequence"]
    p["arm"] = p["run_id"].str.extract(r"netseed\d+_(.+)_p\d+_pruneseed")
    p["pruning_pct"] = (p["amount"]*100).round().astype(int)
    if p["arm"].isna().any() or set(p["arm"]) - set(ARMS):
        raise SystemExit(f"arm parse problem: {sorted(set(p['arm']))}")
    p.to_csv(a.result_dir / f"{STEM}_raw.csv", index=False)

    counts = p.groupby(["arm","pruning_pct"]).size().unstack(fill_value=0)
    print("cell counts all 24" if (counts == 24).all().all() else counts.to_string())

    # regression gate on the three published anchors
    if FROZEN.exists():
        f = pd.read_csv(FROZEN, low_memory=False)
        print("\nregression against the frozen task-preservation suite:")
        for arm, strat in (("mag_mask","l1_unstructured"),
                           ("mask_paper","simulation_noise_prune_mask_only"),
                           ("rescale_paper","simulation_noise_prune_rescale")):
            fr = f[(f.strategy == strat) & (f.amount.round(2).isin([x/100 for x in AMOUNTS]))]
            if strat != "l1_unstructured":
                fr = fr[fr.pruning_seed == 0]
            key = ["source_model_label","amount"]
            x = p[p.arm == arm].set_index(key)["post_acc_sequence"].sort_index()
            y = fr.set_index(key)["post_acc_sequence"].sort_index()
            y = y[~y.index.duplicated()]
            common = x.index.intersection(y.index)
            worst = float((x.loc[common]-y.loc[common]).abs().max())
            flag = "exact" if worst <= TOL else f"*** DRIFT {worst:.3e} ***"
            print(f"  {arm:18s} {len(common):3d} paired rows   max |delta| = {worst:.3e}   {flag}")
            if worst > TOL and not a.allow_regression:
                raise SystemExit("regression against frozen artifact")

    unit = ["task","source_network_seed"]
    wide = p.pivot_table(index=unit+["pruning_pct"], columns="arm", values="sequence_retention")

    print(f"\nmean sequence retention, n = 24\n")
    print(f"{'arm':18s}" + "".join(f"{x:>10d}%" for x in AMOUNTS))
    for arm in ARMS:
        print(f"{arm:18s}" + "".join(
            f"{wide.xs(x, level='pruning_pct')[arm].mean():11.4f}" for x in AMOUNTS))

    rows = []
    for test, ref, label in CONTRASTS:
        for pct in AMOUNTS:
            s = wide.xs(pct, level="pruning_pct")
            dd = (s[test]-s[ref]).dropna().to_numpy()
            hl, lo, hi = hl_ci(dd)
            rows.append(dict(contrast=label, test=test, ref=ref, sparsity=pct, delta=hl,
                             lo=lo, hi=hi, wins=int((dd > 0).sum()), n=len(dd),
                             p=float(wilcoxon(dd).pvalue)))
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r["p"].to_numpy())

    print(f"\npaired contrasts, Holm over all {len(r)} tests\n")
    for label in dict.fromkeys(r.contrast):
        sub = r[r.contrast == label]
        print(f"  {label}   [{sub.test.iloc[0]} - {sub.ref.iloc[0]}]")
        for _, x in sub.iterrows():
            star = " *" if x.p_holm < 0.05 else ""
            print(f"    {x.sparsity:3d}%  {x.delta:+.4f}  [{x.lo:+.4f}, {x.hi:+.4f}]  "
                  f"{x.wins:2d}/{x.n}  p_holm={x.p_holm:.2e}{star}")
        print()
    r.to_csv(a.result_dir / f"{STEM}_summary.csv", index=False)
    print(f"wrote {a.result_dir / f'{STEM}_summary.csv'}")


if __name__ == "__main__":
    main()
