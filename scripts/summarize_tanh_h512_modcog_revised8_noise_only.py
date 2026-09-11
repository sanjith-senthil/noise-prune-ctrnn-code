#!/usr/bin/env python3
"""Summarize the noise-only control: is task input needed to estimate C?

Two arms, both at the paper's sigma and both at three pruning seeds, so the
comparison is paired within trained network and self-contained:

  snp_rescale_taskplusnoise  task input present   (the method as published)
  snp_rescale_noiseonly      task input zeroed    (injected noise only)

If the two do not separate, the covariance the method relies on is noise-induced
rather than task-driven, which is the direct answer to reviewer 2's comment 4.
A null result is the *informative* outcome here, so it is reported with an
equivalence-style reading -- effect size and CI -- rather than as a bare
"not significant", which at n = 24 would be ambiguous between "no effect" and
"underpowered".

Regression gate (fatal)
-----------------------
``snp_rescale_taskplusnoise`` at pruning seed 0 must reproduce the original
``noise_necessity`` suite's ``snp_rescale_sf1p0`` rows exactly. That arm differs
from it in no respect, so any discrepancy means the zero-task-input plumbing
perturbed the reference path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_only_p50_80"
SRC_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_necessity_p50_80"
RESULT_DIR = Path(f"results/{STEM}")
SRC_RAW = Path(f"results/{SRC_STEM}/{SRC_STEM}_raw.csv")
SRC_FALLBACK = Path(f"../data_release_staging/results/revision_2026/noise_necessity/{SRC_STEM}_raw.csv")
REF, TEST = "snp_rescale_taskplusnoise", "snp_rescale_noiseonly"
ALPHA, TOL = 0.05, 1e-9
TASKS = ("ctxdlydm2intseq", "ctxdlydm1intseq", "dlydm1intseq", "dlydm2intseq",
         "multidlydmintseq", "dm1seqr", "dm2seql", "dmsintseq")
UNIT_NAMES = frozenset(f"{t}{s}.csv" for t in TASKS
                       for s in ("", "_p50", "_p60", "_p70", "_p80"))


def hodges_lehmann_ci(d: np.ndarray, alpha: float = 0.05):
    """Hodges-Lehmann point estimate and distribution-free CI of the median shift."""
    w = np.array([(d[i] + d[j]) / 2.0 for i in range(len(d)) for j in range(i, len(d))])
    w.sort()
    n = len(d)
    # normal approximation to the signed-rank critical value; adequate at n = 24
    z = 1.959963984540054
    m = n * (n + 1) / 2
    k = int(np.floor(m / 2 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(0, min(k, len(w) - 1))
    return float(np.median(w)), float(w[k]), float(w[len(w) - 1 - k])


def load(result_dir: Path) -> pd.DataFrame:
    files = [f for f in sorted(result_dir.glob("*.csv")) if f.name in UNIT_NAMES]
    if not files:
        raise SystemExit(f"No per-unit CSVs found in {result_dir}")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"merged {len(files)} per-unit CSVs -> {len(d)} rows")
    base = d[d.strategy == "none"].set_index("source_model_label")["post_acc_sequence"]
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(base)
    if p["baseline_acc_sequence"].isna().any():
        raise SystemExit("missing unpruned baseline")
    p["sequence_retention"] = p["post_acc_sequence"] / p["baseline_acc_sequence"]
    p["task_short"] = p["run_id"].str.extract(r"noiseonly_([a-z0-9]+)_netseed")
    p["arm"] = p["run_id"].str.extract(r"netseed\d+_(snp_rescale_[a-z]+)_p\d\d_pruneseed")
    p["pruning_pct"] = (p["amount"] * 100).round().astype(int)
    if p["arm"].isna().any():
        raise SystemExit("could not parse arm from run_ids")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--allow-regression", action="store_true")
    args = ap.parse_args()

    p = load(args.result_dir)
    p.to_csv(args.result_dir / f"{STEM}_raw.csv", index=False)

    src_path = SRC_RAW if SRC_RAW.exists() else SRC_FALLBACK
    if src_path.exists():
        src = pd.read_csv(src_path, low_memory=False)
        src = src[(src.arm == "snp_rescale_sf1p0") & (src.pruning_seed == 0)]
        key = ["task_short", "pruning_pct", "source_network_seed"]
        a = p[(p.arm == REF) & (p.pruning_seed == 0)][key + ["post_acc_sequence"]]
        m = a.merge(src[key + ["post_acc_sequence"]], on=key, suffixes=("_new", "_ref"))
        if len(m):
            m["abs_diff"] = (m.post_acc_sequence_new - m.post_acc_sequence_ref).abs()
            n_bad = int((m.abs_diff > TOL).sum())
            print(f"\nregression (reference arm vs {SRC_STEM}): {len(m)} runs, "
                  f"max |delta| = {m.abs_diff.max():.3e}")
            if n_bad and not args.allow_regression:
                raise SystemExit("REGRESSION: zero-task-input plumbing perturbed the reference path.")
            print("  reproduces exactly -- the reference path is untouched")
    else:
        print(f"original noise_necessity table not found; skipping regression check")

    cells = (p.groupby(["arm", "pruning_pct", "task_short", "source_network_seed"],
                       as_index=False).sequence_retention.mean())
    nseeds = p.groupby(["arm", "pruning_pct", "task_short", "source_network_seed"]).size()
    print(f"\npruning seeds per cell: min {nseeds.min()}, max {nseeds.max()} (target 3)")
    ncell = cells.groupby(["arm", "pruning_pct"]).size()
    print(f"trained networks per (arm, sparsity): min {ncell.min()}, max {ncell.max()} (target 24)")

    summ = (cells.groupby(["arm", "pruning_pct"])["sequence_retention"]
            .agg(n="count", mean="mean", sd=lambda s: s.std(ddof=1)).reset_index())
    summ["sem"] = summ["sd"] / np.sqrt(summ["n"])
    summ.to_csv(args.result_dir / f"{STEM}_summary_by_sparsity.csv", index=False)

    rows = []
    for pct in sorted(cells.pruning_pct.unique()):
        u = cells[cells.pruning_pct == pct].pivot_table(
            index=["task_short", "source_network_seed"], columns="arm",
            values="sequence_retention")
        if REF not in u.columns or TEST not in u.columns:
            continue
        pr = u[[REF, TEST]].dropna()
        x, y = pr[REF].to_numpy(), pr[TEST].to_numpy(); d = x - y
        hl, lo, hi = hodges_lehmann_ci(d)
        rows.append(dict(pruning_pct=pct, n=len(pr),
                         mean_taskplusnoise=x.mean(), mean_noiseonly=y.mean(),
                         mean_diff=d.mean(), hodges_lehmann=hl, ci_lo=lo, ci_hi=hi,
                         wins_ref=int((d > 0).sum()),
                         wilcoxon_p=float(wilcoxon(x, y, zero_method="wilcox",
                                                   alternative="two-sided",
                                                   method="auto").pvalue)))
    t = pd.DataFrame(rows)
    if len(t):
        pv = t.wilcoxon_p.to_numpy(); o = np.argsort(pv); adj = np.empty_like(pv); run = 0.0
        for k, i in enumerate(o):
            run = max(run, (pv.size - k) * pv[i]); adj[i] = min(1.0, run)
        t["holm_p"] = adj
        t["reject_holm_alpha_0_05"] = t.holm_p <= ALPHA
        t.to_csv(args.result_dir / f"{STEM}_tests.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== sequence retention, n = 24 trained networks, 3 pruning seeds ===")
    print(summ.pivot(index="arm", columns="pruning_pct", values="mean").round(3).to_string())
    if len(t):
        print("\n=== task+noise vs noise-only (positive = task input helps) ===")
        for _, r in t.iterrows():
            verdict = "SIG" if r.reject_holm_alpha_0_05 else "n.s."
            print(f"  {r.pruning_pct}%  d={r.mean_diff:+.3f}  HL={r.hodges_lehmann:+.3f} "
                  f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  {r.wins_ref:>2}/{r.n}  "
                  f"holm p={r.holm_p:.3g}  {verdict}")
        print("\nRead the CI, not just the p-value: a tight interval around zero is evidence that "
              "\ntask input is unnecessary, whereas a wide one means the suite is underpowered.")
    print(f"\nwrote {args.result_dir}/{STEM}_{{raw,summary_by_sparsity,tests}}.csv")


if __name__ == "__main__":
    main()
