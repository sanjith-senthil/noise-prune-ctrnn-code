#!/usr/bin/env python3
"""Summarize the task-preservation suite at one hidden size, and test size scaling.

Two questions, both answered *within* a single size so the unpruned baseline is
common to every arm and cancels out of the comparison:

1. **Method ranking.** Does noise-prune's margin over the baselines widen at the
   larger size? Reported as retention per arm plus paired contrasts against the
   strongest non-noise-prune baseline.

2. **Does the covariance contribution grow with size?** The covariance-ablation
   arms (``lnp_magnitude`` / ``snp_magnitude``) drop the covariance factor from
   ``p_ij``, leaving it proportional to ``|w_ij|``. The contrast against the
   corresponding rescale arm isolates the covariance term. At H = 512 it is
   +0.025 / +0.041 / +0.032 / +0.006 for L-NP across 50-80%, significant except
   at 80%. Theory says ``K = 8 ln N / (eps^2 sigma^2)`` tightens with N, so the
   contribution should be *larger* here. It may not be -- a flat or shrinking
   delta is a real result and is reported as such, not buried.

Why cross-size retention is NOT compared directly
-------------------------------------------------
H = 1024 trains far better than H = 512 under the same budget (+0.268 mean
sequence accuracy; five of eight tasks at or above 0.997). Retention is
normalised to the unpruned network, so the larger networks have more headroom
before retention visibly falls. Comparing retention across sizes would conflate
method quality with that headroom. Every number here is a within-size contrast;
the cross-size claim is about how those contrasts *change*, which is well defined.

Aggregation follows the paper: pruning-seed replicates averaged within each
(arm, sparsity, task, network seed) cell, analysis unit = trained network
(n = 24), sample SD (ddof=1), paired two-sided Wilcoxon with Holm correction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TASKS = ("ctxdlydm2intseq", "ctxdlydm1intseq", "dlydm1intseq", "dlydm2intseq",
         "multidlydmintseq", "dm1seqr", "dm2seql", "dmsintseq")
UNIT_NAMES = frozenset(f"{t}_p{p}.csv" for t in TASKS for p in (50, 60, 70, 80))
ALPHA = 0.05
ARM_ORDER = ["snp_rescale", "lnp_rescale", "snp_magnitude", "lnp_magnitude",
             "obs_compensated", "snp_mask", "lnp_mask", "magnitude", "random"]
# L-NP and S-NP covariance contributions measured at H=512, for the size comparison
H512_COVARIANCE_DELTA = {
    ("lnp", 50): 0.025, ("lnp", 60): 0.041, ("lnp", 70): 0.032, ("lnp", 80): 0.006,
    ("snp", 50): 0.014, ("snp", 60): 0.022, ("snp", 70): 0.017, ("snp", 80): 0.003,
}


def suite_stem(h: int) -> str:
    return f"task_preservation_tanh_h{h}_modcog_revised8_12k_seqbest_revised_task_only_p50_80"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty_like(p); r = 0.0
    for k, i in enumerate(o):
        r = max(r, (p.size - k) * p[i]); a[i] = min(1.0, r)
    return a


def load(h: int) -> pd.DataFrame:
    d_dir = Path(f"results/{suite_stem(h)}")
    files = [f for f in sorted(d_dir.glob("*.csv")) if f.name in UNIT_NAMES]
    if not files:
        raise SystemExit(f"no per-unit CSVs in {d_dir}")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"H={h}: merged {len(files)} unit CSVs -> {len(d)} rows")
    if d.post_acc_sequence.isna().any():
        n = int(d.post_acc_sequence.isna().sum())
        raise SystemExit(
            f"{n} rows have no result. Failed runs still carry their run_id, so `resume` "
            f"would skip them on a re-run -- delete the unit CSVs and re-run rather than "
            f"summarizing a partial suite.")
    base = d[d.strategy == "none"].set_index("source_model_label")["post_acc_sequence"]
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(base)
    if p.baseline_acc_sequence.isna().any():
        raise SystemExit("missing unpruned baseline for some networks")
    p["sequence_retention"] = p.post_acc_sequence / p.baseline_acc_sequence
    p["arm"] = p.run_id.str.extract(rf"taskpres_h{h}_[a-z0-9]+_netseed\d+_(.+)_p\d\d_pruneseed")
    p["task_short"] = p.run_id.str.extract(rf"taskpres_h{h}_([a-z0-9]+)_netseed")
    p["pruning_pct"] = (p.amount * 100).round().astype(int)
    if p.arm.isna().any():
        raise SystemExit("could not parse arm from some run_ids")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=1024)
    args = ap.parse_args()
    h = args.hidden_size
    p = load(h)
    out = Path(f"results/{suite_stem(h)}")
    p.to_csv(out / f"{suite_stem(h)}_raw.csv", index=False)

    cells = (p.groupby(["arm", "pruning_pct", "task_short", "source_network_seed"],
                       as_index=False).sequence_retention.mean())
    n = cells.groupby(["arm", "pruning_pct"]).size()
    print(f"trained networks per (arm, sparsity): {n.min()}-{n.max()} (target 24)")

    summ = (cells.groupby(["arm", "pruning_pct"]).sequence_retention
            .agg(n="count", mean="mean", sd=lambda s: s.std(ddof=1)).reset_index())
    summ["sem"] = summ["sd"] / np.sqrt(summ["n"])
    summ.assign(hidden_size=h).to_csv(out / f"{suite_stem(h)}_summary_by_sparsity.csv", index=False)

    pd.set_option("display.width", 200)
    piv = summ.pivot(index="arm", columns="pruning_pct", values="mean")
    piv = piv.reindex([a for a in ARM_ORDER if a in piv.index])
    print(f"\n=== H={h} sequence retention, n = 24 trained networks ===")
    print(piv.round(3).to_string())

    # ---- the covariance contribution, and how it compares with H=512 --------
    rows = []
    for fam, ref, abl in (("lnp", "lnp_rescale", "lnp_magnitude"),
                          ("snp", "snp_rescale", "snp_magnitude")):
        for pct in sorted(cells.pruning_pct.unique()):
            u = cells[cells.pruning_pct == pct].pivot_table(
                index=["task_short", "source_network_seed"], columns="arm",
                values="sequence_retention")
            if ref not in u.columns or abl not in u.columns:
                continue
            pr = u[[ref, abl]].dropna()
            x, y = pr[ref].to_numpy(), pr[abl].to_numpy(); d = x - y
            rows.append(dict(family=fam.upper(), pruning_pct=pct, n=len(pr),
                             mean_full=x.mean(), mean_ablated=y.mean(),
                             covariance_delta=d.mean(), wins=int((d > 0).sum()),
                             h512_delta=H512_COVARIANCE_DELTA.get((fam, pct), np.nan),
                             wilcoxon_p=float(wilcoxon(x, y, zero_method="wilcox",
                                                       alternative="two-sided",
                                                       method="auto").pvalue)))
    t = pd.DataFrame(rows)
    if len(t):
        t["holm_p"] = holm(t.wilcoxon_p.to_numpy())
        t["sig"] = t.holm_p <= ALPHA
        t["grew_vs_h512"] = t.covariance_delta > t.h512_delta
        t.assign(hidden_size=h).to_csv(out / f"{suite_stem(h)}_covariance_contribution.csv",
                                       index=False)
        print(f"\n=== covariance contribution at H={h} vs H=512 ===")
        print(f"{'fam':<5}{'sp%':>5}{'full':>8}{'ablated':>9}{'delta':>8}"
              f"{'H512':>8}{'grew?':>7}{'wins':>7}{'holm p':>10}")
        for _, r in t.iterrows():
            print(f"{r.family:<5}{r.pruning_pct:>4}%{r.mean_full:>8.3f}{r.mean_ablated:>9.3f}"
                  f"{r.covariance_delta:>+8.3f}{r.h512_delta:>+8.3f}"
                  f"{'yes' if r.grew_vs_h512 else 'no':>7}{r.wins:>4}/{r.n}"
                  f"{r.holm_p:>10.3g}{' *' if r.sig else ''}")
        grew = int(t.grew_vs_h512.sum())
        print(f"\ncovariance contribution larger than at H=512 in {grew}/{len(t)} cells")
        if grew <= len(t) // 2:
            print("=> the contribution does NOT clearly grow with size over this range; "
                  "report that plainly rather than as a null.")
    print(f"\nwrote {out}/{suite_stem(h)}_{{raw,summary_by_sparsity,covariance_contribution}}.csv")


if __name__ == "__main__":
    main()
