#!/usr/bin/env python3
"""Merge the noise-necessity seed extension with the original suite and test it.

The original ``noise_necessity`` suite ran one pruning seed per cell; this merges
its pruning-seed-0 rows for the rescale variant with the extension's seeds 1 and
2, giving the three replicates the official protocol averages before reducing to
n = 24 trained networks.

Regression gate (fatal)
-----------------------
The extension re-runs ``sf1p0`` at pruning seed 0, where the original suite
already has it. Those rows must match exactly; otherwise the two suites are not
on the same footing and must not be merged.

Claims tested (paired Wilcoxon on network means, Holm within family)
-------------------------------------------------------------------
A. noise necessary  -- sigma_nat vs ~noise-free (0.001x)
B. interior optimum -- sigma_nat vs 2x and vs 0.5x

Claim B is why the extension exists: on single-seed data it was significant for
the mask variant at 60-80% but absent at 50% and absent at every sparsity for the
rescale variant, on differences of 0.01-0.06 that one stochastic draw per cell
cannot resolve.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_necessity_seedext_p50_80"
SRC_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_necessity_p50_80"
RESULT_DIR = Path(f"results/{STEM}")
SRC_RAW = Path(f"results/{SRC_STEM}/{SRC_STEM}_raw.csv")
SRC_FALLBACK = Path(f"../data_release_staging/results/revision_2026/noise_necessity/{SRC_STEM}_raw.csv")
FACTORS = (0.001, 0.25, 0.5, 1.0, 2.0)
ALPHA, TOL = 0.05, 1e-9
TASKS = ("ctxdlydm2intseq", "ctxdlydm1intseq", "dlydm1intseq", "dlydm2intseq",
         "multidlydmintseq", "dm1seqr", "dm2seql", "dmsintseq")
UNIT_NAMES = frozenset(f"{t}{s}.csv" for t in TASKS
                       for s in ("", "_p50", "_p60", "_p70", "_p80"))


def tag(f: float) -> str:
    return f"sf{str(f).replace('.', 'p')}"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (p.size - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def load_extension(result_dir: Path) -> pd.DataFrame:
    files = [f for f in sorted(result_dir.glob("*.csv")) if f.name in UNIT_NAMES]
    if not files:
        raise SystemExit(f"No per-unit CSVs found in {result_dir}")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"extension: merged {len(files)} per-unit CSVs -> {len(d)} rows")
    base = d[d.strategy == "none"].set_index("source_model_label")["post_acc_sequence"]
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(base)
    if p["baseline_acc_sequence"].isna().any():
        raise SystemExit("missing unpruned baseline in the extension")
    p["sequence_retention"] = p["post_acc_sequence"] / p["baseline_acc_sequence"]
    p["task_short"] = p["run_id"].str.extract(r"noisenecext_([a-z0-9]+)_netseed")
    p["arm"] = p["run_id"].str.extract(r"netseed\d+_(snp_rescale_sf[0-9p]+)_p\d\d_pruneseed")
    p["pruning_pct"] = (p["amount"] * 100).round().astype(int)
    if p["arm"].isna().any():
        raise SystemExit("could not parse arm from extension run_ids")
    return p


def load_source() -> pd.DataFrame:
    src = SRC_RAW if SRC_RAW.exists() else SRC_FALLBACK
    if not src.exists():
        raise SystemExit(f"original noise_necessity table not found ({SRC_RAW} / {SRC_FALLBACK})")
    d = pd.read_csv(src, low_memory=False)
    print(f"original: {src} ({len(d)} rows)")
    return d[d.arm.str.startswith("snp_rescale")].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--allow-regression", action="store_true")
    args = ap.parse_args()

    ext = load_extension(args.result_dir)
    ext.to_csv(args.result_dir / f"{STEM}_raw.csv", index=False)
    src = load_source()

    # ---- regression: extension's seed-0 sf1p0 vs the original suite ----
    key = ["arm", "task_short", "pruning_pct", "source_network_seed"]
    a = ext[(ext.pruning_seed == 0) & (ext.arm == "snp_rescale_sf1p0")][key + ["post_acc_sequence"]]
    b = src[(src.pruning_seed == 0) & (src.arm == "snp_rescale_sf1p0")][key + ["post_acc_sequence"]]
    m = a.merge(b, on=key, suffixes=("_new", "_ref"))
    if m.empty:
        print("WARNING: no overlapping rows for the regression check")
    else:
        m["abs_diff"] = (m.post_acc_sequence_new - m.post_acc_sequence_ref).abs()
        n_bad = int((m.abs_diff > TOL).sum())
        print(f"\nregression vs {SRC_STEM}: {len(m)} shared runs, "
              f"max |delta| = {m.abs_diff.max():.3e}, {n_bad} exceed {TOL:g}")
        if n_bad and not args.allow_regression:
            print(m[m.abs_diff > TOL].head(15).to_string(index=False))
            raise SystemExit("REGRESSION: the suites are not on the same footing; do not merge.")
        print("  reproduces exactly -- merging is safe" if not n_bad else "  MISMATCH (allowed)")
        m.to_csv(args.result_dir / f"{STEM}_regression_check.csv", index=False)

    # ---- merge: original seed 0 + extension seeds 1,2 ----
    cols = ["arm", "task_short", "pruning_pct", "source_network_seed",
            "pruning_seed", "sequence_retention"]
    merged = pd.concat([src[src.pruning_seed == 0][cols],
                        ext[ext.pruning_seed.isin([1, 2])][cols]], ignore_index=True)
    merged = merged.drop_duplicates(["arm", "task_short", "pruning_pct",
                                     "source_network_seed", "pruning_seed"])
    nseeds = merged.groupby(["arm", "pruning_pct", "task_short", "source_network_seed"]).size()
    print(f"\npruning seeds per cell after merge: min {nseeds.min()}, max {nseeds.max()} "
          f"(target 3)")
    cells = (merged.groupby(["arm", "pruning_pct", "task_short", "source_network_seed"],
                            as_index=False).sequence_retention.mean())
    ncell = cells.groupby(["arm", "pruning_pct"]).size()
    print(f"trained networks per (arm, sparsity): min {ncell.min()}, max {ncell.max()} (target 24)")

    summ = (cells.groupby(["arm", "pruning_pct"])["sequence_retention"]
            .agg(n="count", mean="mean", sd=lambda s: s.std(ddof=1)).reset_index())
    summ["sem"] = summ["sd"] / np.sqrt(summ["n"])
    summ.to_csv(args.result_dir / f"{STEM}_summary_by_sparsity.csv", index=False)

    # ---- tests ----
    rows = []
    for pct in sorted(cells.pruning_pct.unique()):
        u = cells[cells.pruning_pct == pct].pivot_table(
            index=["task_short", "source_network_seed"], columns="arm",
            values="sequence_retention")
        ref = f"snp_rescale_{tag(1.0)}"
        for f_, label in ((0.001, "A. noise necessary: sigma_nat vs ~noise-free"),
                          (2.0, "B. interior optimum: sigma_nat vs 2x"),
                          (0.5, "B. interior optimum: sigma_nat vs 0.5x")):
            other = f"snp_rescale_{tag(f_)}"
            if ref not in u.columns or other not in u.columns:
                continue
            pr = u[[ref, other]].dropna()
            x, y = pr[ref].to_numpy(), pr[other].to_numpy(); d = x - y
            rows.append(dict(contrast=label, pruning_pct=pct, n=len(pr),
                             mean_ref=x.mean(), mean_other=y.mean(), mean_diff=d.mean(),
                             wins=int((d > 0).sum()),
                             wilcoxon_p=float(wilcoxon(x, y, zero_method="wilcox",
                                                       alternative="two-sided",
                                                       method="auto").pvalue)))
    t = pd.DataFrame(rows)
    t["holm_p"] = holm(t.wilcoxon_p.to_numpy())
    t["reject_holm_alpha_0_05"] = t.holm_p <= ALPHA
    t.to_csv(args.result_dir / f"{STEM}_tests.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== retention vs injected-noise scale, rescale variant, n = 24, 3 pruning seeds ===")
    piv = summ.pivot(index="arm", columns="pruning_pct", values="mean")
    piv = piv.reindex([f"snp_rescale_{tag(f)}" for f in FACTORS if f"snp_rescale_{tag(f)}" in piv.index])
    print(piv.round(3).to_string())
    print(f"\n=== paired Wilcoxon, Holm across all {len(t)} contrasts ===")
    for c, g in t.groupby("contrast"):
        print(f"\n{c}")
        for _, r in g.iterrows():
            print(f"  {r.pruning_pct}%  d={r.mean_diff:+.3f}  {r.wins:>2}/{r.n}  "
                  f"holm p={r.holm_p:.3g}  {'SIG' if r.reject_holm_alpha_0_05 else 'n.s.'}")
    print(f"\nwrote {args.result_dir}/{STEM}_{{raw,summary_by_sparsity,tests,regression_check}}.csv")


if __name__ == "__main__":
    main()
