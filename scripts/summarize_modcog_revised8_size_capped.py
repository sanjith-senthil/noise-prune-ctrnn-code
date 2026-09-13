#!/usr/bin/env python3
"""Does the rescale cap transfer to a new network size?

The paper's cap-percentile curve peaks at q50-q60 at H = 512. The revision plan
argues the cap has a **size-independent anchor**: probabilities are normalised so
``sum(p) = density * N(N-1)``, hence ``1/p = (1/density) * (1/shape)`` and the
cap at quantile q is ``c(q)/density`` with density a design parameter rather than
a function of N. ``cap_value * density`` is constant to machine precision within
a network (spread 6.7e-16). This tests that prediction directly at H = 1024,
where it should matter more: the amplification tail is far heavier there
(``prune_inv_p_max`` reaching 2.3e6), so a cap has more to bite on.

Three questions, all answered within H = 1024 so the unpruned baseline cancels:

1. **Does capping help at all here?** capped q50 / q60 against the uncapped
   reference, paired by trained network.
2. **Is q50-q60 still the useful range?** i.e. does the H = 512 optimum transfer.
3. **Is the size-independent anchor empirically true?** ``cap_value * density``
   is reported per quantile; if the argument holds it is constant within a
   network and comparable across sizes.

The uncapped reference is re-run inside this suite at pruning seed 0 and must
reproduce the completed H = 1024 task-preservation suite exactly -- that
regression is what licenses comparing capped arms here against uncapped arms
recorded there.

Interpretation note: ``prune_leak_shift`` is 4.0 or 8.0 for every L-NP arm at
this size (against a mean of 1.60 at H = 512), so L-NP's covariance is computed
for A = W - 5I or W - 9I rather than W - I. S-NP uses no Lyapunov solve and is
unaffected. Prefer S-NP when reading the cap result as a statement about the
method rather than about the solver.
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
ALPHA, TOL = 0.05, 1e-9
# H=512 cap curve (L-NP), from the frozen capped_rescale artifacts
H512_CAP = {("q50", 50): 0.915, ("q50", 60): 0.821, ("q50", 70): 0.620, ("q50", 80): 0.397,
            ("q60", 50): 0.927, ("q60", 60): 0.828, ("q60", 70): 0.616, ("q60", 80): 0.394,
            ("uncapped", 50): 0.913, ("uncapped", 60): 0.800,
            ("uncapped", 70): 0.581, ("uncapped", 80): 0.378}


def suite_stem(h): return f"task_preservation_tanh_h{h}_modcog_revised8_12k_seqbest_capped_q50_q60_p50_80"
def ref_stem(h): return f"task_preservation_tanh_h{h}_modcog_revised8_12k_seqbest_revised_task_only_p50_80"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty_like(p); r = 0.0
    for k, i in enumerate(o):
        r = max(r, (p.size - k) * p[i]); a[i] = min(1.0, r)
    return a


def load(h, allow_missing=False):
    P = Path(f"results/{suite_stem(h)}")
    files = [f for f in sorted(P.glob("*.csv")) if f.name in UNIT_NAMES]
    if not files:
        raise SystemExit(f"no per-unit CSVs in {P}")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"merged {len(files)} unit CSVs -> {len(d)} rows")
    empty = d[d.run_id.notna() & d.post_acc_sequence.isna()]
    if len(empty):
        print(f"!! {len(empty)} runs produced no result")
        if not allow_missing:
            raise SystemExit("refusing to summarize a partial suite; pass --allow-missing to proceed")
        d = d[d.post_acc_sequence.notna()].copy()
    base = d[d.strategy == "none"].set_index("source_model_label")["post_acc_sequence"]
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(base)
    p["sequence_retention"] = p.post_acc_sequence / p.baseline_acc_sequence
    p["arm"] = p.run_id.str.extract(rf"cap{h}_[a-z0-9]+_netseed\d+_(.+)_p\d\d_pruneseed")
    p["task_short"] = p.run_id.str.extract(rf"cap{h}_([a-z0-9]+)_netseed")
    p["pruning_pct"] = (p.amount * 100).round().astype(int)
    if p.arm.isna().any():
        raise SystemExit("could not parse arm from some run_ids")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=1024)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    h = args.hidden_size
    p = load(h, args.allow_missing)
    out = Path(f"results/{suite_stem(h)}")
    p.to_csv(out / f"{suite_stem(h)}_raw.csv", index=False)

    # ---- regression: uncapped reference vs the completed task-preservation suite
    ref_raw = Path(f"results/{ref_stem(h)}/{ref_stem(h)}_raw.csv")
    if ref_raw.exists():
        r = pd.read_csv(ref_raw, low_memory=False)
        r = r[(r.arm == "snp_rescale") & (r.pruning_seed == 0)]
        key = ["task_short", "pruning_pct", "source_network_seed"]
        a = p[(p.arm == "snp_rescale") & (p.pruning_seed == 0)][key + ["post_acc_sequence"]]
        m = a.merge(r[key + ["post_acc_sequence"]], on=key, suffixes=("_cap", "_ref"))
        if len(m):
            m["d"] = (m.post_acc_sequence_cap - m.post_acc_sequence_ref).abs()
            print(f"\nregression vs {ref_stem(h)}: {len(m)} runs, max |delta| = {m.d.max():.3e}")
            if m.d.max() > TOL:
                raise SystemExit("REGRESSION: the two suites do not agree; capped vs uncapped "
                                 "comparisons across them are not licensed.")
            print("  reproduces exactly -- cross-suite comparison is licensed")
    else:
        print(f"\nreference raw not found at {ref_raw}; skipping regression check")

    cells = (p.groupby(["arm", "pruning_pct", "task_short", "source_network_seed"],
                       as_index=False).sequence_retention.mean())
    n = cells.groupby(["arm", "pruning_pct"]).size()
    print(f"trained networks per (arm, sparsity): {n.min()}-{n.max()} (target 24)")
    summ = (cells.groupby(["arm", "pruning_pct"]).sequence_retention
            .agg(n="count", mean="mean", sd=lambda s: s.std(ddof=1)).reset_index())
    summ["sem"] = summ["sd"] / np.sqrt(summ["n"])
    summ.to_csv(out / f"{suite_stem(h)}_summary_by_sparsity.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== H={h} retention, capped vs uncapped (n = 24) ===")
    print(summ.pivot(index="arm", columns="pruning_pct", values="mean").round(3).to_string())

    # ---- does capping beat uncapped? paired within network -------------------
    rows = []
    for fam, unc in (("lnp", None), ("snp", "snp_rescale")):
        for q in ("q50", "q60"):
            arm = f"{fam}_capped_{q}"
            if unc is None:
                continue    # L-NP uncapped lives in the other suite; compare S-NP here
            for pct in sorted(cells.pruning_pct.unique()):
                u = cells[cells.pruning_pct == pct].pivot_table(
                    index=["task_short", "source_network_seed"], columns="arm",
                    values="sequence_retention")
                if arm not in u.columns or unc not in u.columns:
                    continue
                pr = u[[arm, unc]].dropna()
                x, y = pr[arm].to_numpy(), pr[unc].to_numpy(); d = x - y
                rows.append(dict(family=fam.upper(), quantile=q, pruning_pct=pct, n=len(pr),
                                 capped=x.mean(), uncapped=y.mean(), delta=d.mean(),
                                 wins=int((d > 0).sum()),
                                 h512_capped=H512_CAP.get((q, pct), np.nan),
                                 h512_uncapped=H512_CAP.get(("uncapped", pct), np.nan),
                                 wilcoxon_p=float(wilcoxon(x, y, zero_method="wilcox",
                                                           alternative="two-sided",
                                                           method="auto").pvalue)))
    t = pd.DataFrame(rows)
    if len(t):
        t["holm_p"] = holm(t.wilcoxon_p.to_numpy()); t["sig"] = t.holm_p <= ALPHA
        t["h512_delta"] = t.h512_capped - t.h512_uncapped
        t.to_csv(out / f"{suite_stem(h)}_cap_vs_uncapped.csv", index=False)
        print(f"\n=== capping vs uncapped at H={h}, with the H=512 gain for comparison ===")
        print(f"{'fam':<5}{'q':>5}{'sp%':>5}{'capped':>9}{'uncap':>8}{'delta':>8}"
              f"{'H512 d':>9}{'wins':>7}{'holm p':>10}")
        for _, r in t.iterrows():
            print(f"{r.family:<5}{r.quantile:>5}{r.pruning_pct:>4}%{r.capped:>9.3f}"
                  f"{r.uncapped:>8.3f}{r.delta:>+8.3f}{r.h512_delta:>+9.3f}"
                  f"{r.wins:>4}/{r.n}{r.holm_p:>10.3g}{' *' if r.sig else ''}")

    # ---- the size-independence anchor ---------------------------------------
    cv = [c for c in p.columns if "cap_value" in c]
    if cv and "post_rec_weight_nz_count" in p.columns:
        c = p[p[cv[0]].notna() & (p[cv[0]] > 0)].copy()
        c["density"] = c.post_rec_weight_nz_count / (h * h)
        c["anchor"] = c[cv[0]] * c["density"]
        print(f"\n=== size-independence anchor: cap_value x density ===")
        print(c.groupby(["arm", "pruning_pct"]).anchor.agg(["mean", "std"]).round(4).to_string())
        print("If the argument holds this is ~constant across sparsity within an arm.")
    print(f"\nwrote {out}/{suite_stem(h)}_{{raw,summary_by_sparsity,cap_vs_uncapped}}.csv")


if __name__ == "__main__":
    main()
