#!/usr/bin/env python3
"""Regenerate every statistic the paper reports. Run from the repo root:

    .venv/bin/python analysis/paper_statistics/run_paper_stats.py

Writes, beside this script:
    curves.csv                mean retention per method and sparsity, with
                              hierarchical-bootstrap 95% and 68% intervals
    task_curves.csv           the per-task retention the SI panels plot
    auc_by_method.csv         the normalised 10-80% AUC summary per method
    contrasts_auc.csv         noise-prune and capped rescale against every
                              competitor, on the AUC summary
    contrasts_by_sparsity.csv the same contrasts at each sparsity
    mechanism_contrasts.csv   the score / rescale / cap decomposition

Every effect size is a mean over 8 task-level differences, every interval is a
percentile of 10,000 shared hierarchical resamples, and every p-value is an
exact sign-flip test over all 256 sign assignments. See paper_stats.py for why.

Holm is applied WITHIN declared families and reported next to the uncorrected
p-value; the paper can then state which it uses. A family here is a set of
comparisons the paper makes together -- one reference arm against its
competitors, or one contrast across the sparsity range -- not all 104 tests the
tables happen to contain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_stats import (ARMS, AUC_HI, AUC_LO, GRID, NOISE_PRUNE_ARMS, N_BOOT,
                         auc_table, build_cube, contrast, curve_table, holm,
                         make_surrogates, task_curve_table)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 46)

CAP, RES = NOISE_PRUNE_ARMS
MECHANISM = [
    ("does the noise score alone help?", "noise-prune score only", "magnitude"),
    ("does rescaling alone help?", "magnitude + rescale", "magnitude"),
    ("does the score add beyond rescaling?", RES, "magnitude + rescale"),
    ("does capping help the noise score?", CAP, RES),
    ("does capping help magnitude too?", "magnitude + capped rescale", "magnitude + rescale"),
]


def stars(p):
    return "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else "ns"


def fmt(rows, family_col="family"):
    R = pd.DataFrame(rows)
    R["holm"] = np.nan
    for _, g in R.groupby(family_col):
        R.loc[g.index, "holm"] = holm(g.p_two.values)
    R["CI95"] = R.apply(lambda r: f"[{r.lo95:+.3f}, {r.hi95:+.3f}]", axis=1)
    R["CI68"] = R.apply(lambda r: f"[{r.lo68:+.3f}, {r.hi68:+.3f}]", axis=1)
    R["signs"] = R.apply(lambda r: f"{r.n_positive}/{r.n_tasks}", axis=1)
    R["sig"] = R.p_two.map(stars)
    R["sig_holm"] = np.where(R.holm <= 0.05, "yes", "no")
    return R


def show(R, cols):
    V = R[cols].copy()
    for c in ("effect", "lo95", "hi95"):
        if c in V:
            V[c] = V[c].round(4)
    for c in ("p_two", "holm"):
        if c in V:
            V[c] = V[c].map(lambda v: f"{v:.4f}")
    print(V.to_string(index=False))


def main():
    # =====================================================================
    cube = build_cube()
    print(f"cube: {len(cube.tasks)} tasks x {len(cube.seeds)} seeds x "
          f"{len(cube.methods)} methods x {len(cube.qs)} sparsities, "
          f"{int(cube.draws.sum())} pruning draws averaged in")
    print(f"drawing {N_BOOT} shared hierarchical resamples ...")
    sur = make_surrogates(cube, n_boot=N_BOOT)

    # ---- 1. curves with bootstrap intervals ------------------------------
    curves = curve_table(cube, sur)
    curves.to_csv(HERE / "curves.csv", index=False)
    task_curve_table(cube).to_csv(HERE / "task_curves.csv", index=False)

    print("\n" + "=" * 110)
    print("1. MEAN RETENTION CURVES, hierarchical-bootstrap intervals")
    print("=" * 110)
    piv = curves.pivot(index="sparsity", columns="method", values="mean")
    piv = piv[[m for m in cube.methods]]
    print("\npoint estimates")
    print(piv.round(3).to_string())
    print("\n95% interval half-widths (the error bars the figures should carry)")
    hw = curves.assign(h=(curves.hi95 - curves.lo95) / 2).pivot(
        index="sparsity", columns="method", values="h")[cube.methods]
    print(hw.round(3).to_string())

    # ---- 2. the summary measure ------------------------------------------
    A = auc_table(cube, sur)
    A.to_csv(HERE / "auc_by_method.csv", index=False)
    print("\n" + "=" * 110)
    print(f"2. SUMMARY MEASURE: normalised area under the retention curve, "
          f"{AUC_LO}-{AUC_HI}% (1.0 = no loss anywhere)")
    print("=" * 110)
    B = A.copy()
    B["CI95"] = B.apply(lambda r: f"[{r.lo95:.3f}, {r.hi95:.3f}]", axis=1)
    B["CI68"] = B.apply(lambda r: f"[{r.lo68:.3f}, {r.hi68:.3f}]", axis=1)
    print(B[["method", "AUC", "CI95", "CI68", "sd_across_tasks"]].round(4).to_string(index=False))

    # ---- 3. contrasts on the summary measure -----------------------------
    rows = []
    for ref in (CAP, RES):
        for other in cube.methods:
            if other == ref or (ref == RES and other == CAP):
                continue
            r = contrast(cube, sur, ref, other)
            r.pop("task_diffs")
            rows.append(dict(family=f"vs {ref}", **r))
    C = fmt(rows)
    C.to_csv(HERE / "contrasts_auc.csv", index=False)
    print("\n" + "=" * 110)
    print("3. NOISE-PRUNE AGAINST EVERY COMPETITOR, on the AUC summary")
    print("   effect = mean over 8 tasks of (A_ref - A_other); p = exact sign-flip; "
          "Holm within each reference arm's family")
    print("=" * 110)
    show(C.sort_values(["family", "effect"], ascending=[True, False]),
         ["family", "arm_b", "effect", "CI95", "signs", "p_two", "sig", "holm", "sig_holm"])

    # ---- 4. contrasts at each sparsity -----------------------------------
    rows = []
    for ref in (CAP, RES):
        for other in cube.methods:
            if other == ref or (ref == RES and other == CAP):
                continue
            for q in GRID:
                r = contrast(cube, sur, ref, other, q=q)
                r.pop("task_diffs")
                rows.append(dict(family=f"{ref} vs {other}", **r))
    S = fmt(rows)
    S.to_csv(HERE / "contrasts_by_sparsity.csv", index=False)
    print("\n" + "=" * 110)
    print("4. THE SAME CONTRASTS AT EACH SPARSITY  (Holm within each contrast's "
          "8-sparsity family)")
    print("=" * 110)
    for ref in (CAP, RES):
        print(f"\n--- {ref} ---")
        sub = S[S.arm_a == ref]
        e = sub.pivot(index="sparsity", columns="arm_b", values="effect")
        g = sub.pivot(index="sparsity", columns="arm_b", values="sig")
        k = sub.pivot(index="sparsity", columns="arm_b", values="signs")
        order = [m for m in cube.methods if m in e.columns]
        print("effect size")
        print(e[order].round(3).to_string())
        print("significance (exact sign-flip, uncorrected)")
        print(g[order].to_string())
        print("tasks favouring the noise-prune arm")
        print(k[order].to_string())

    # ---- the mechanism decomposition -------------------------------------
    rows = []
    for label, a, b in MECHANISM:
        r = contrast(cube, sur, a, b)
        r.pop("task_diffs")
        rows.append(dict(family="mechanism", question=label, **r))
        for q in GRID:
            rq = contrast(cube, sur, a, b, q=q)
            rq.pop("task_diffs")
            rows.append(dict(family=f"mechanism @ {label}", question=label, **rq))
    M = fmt(rows)
    M.to_csv(HERE / "mechanism_contrasts.csv", index=False)
    print("\n" + "=" * 110)
    print("THE DECOMPOSITION: what each ingredient contributes (AUC summary)")
    print("=" * 110)
    show(M[M.measure.str.startswith("AUC")],
         ["question", "arm_a", "arm_b", "effect", "CI95", "signs", "p_two", "sig"])

    # ---- what is strong, what is not -------------------------------------
    print("\n" + "=" * 110)
    print("VERDICT")
    print("=" * 110)
    allc = pd.concat([C.assign(kind="AUC"), S.assign(kind="per-sparsity"),
                      M.assign(kind="mechanism")], ignore_index=True)
    floorhit = allc[np.isclose(allc.p_two, allc.p_floor)]
    print(f"contrasts tested            : {len(allc)}")
    print(f"CI excludes zero (95%)      : {((allc.lo95 > 0) | (allc.hi95 < 0)).sum()}")
    print(f"significant uncorrected     : {(allc.p_two <= 0.05).sum()}")
    print(f"significant after Holm      : {(allc.holm <= 0.05).sum()}")
    print(f"unanimous across all 8 tasks: {(allc.n_positive == allc.n_tasks).sum()}")
    print(f"at the exact-test floor     : {len(floorhit)}  "
          f"(p = {2/256:.4f}, the smallest 8 tasks can produce)")
    print("\nAUC contrasts that do NOT clear zero at 95%:")
    weak = C[(C.lo95 <= 0) & (C.hi95 >= 0)]
    print("   none" if weak.empty else
          weak[["family", "arm_b", "effect", "CI95", "signs", "p_two"]].to_string(index=False))
    print("\nper-sparsity contrasts that do NOT clear zero at 95%:")
    weakS = S[(S.lo95 <= 0) & (S.hi95 >= 0)]
    print("   none" if weakS.empty else
          weakS[["arm_a", "arm_b", "sparsity", "effect", "CI95", "signs", "p_two"]]
          .to_string(index=False))
    print(f"\nwritten to {HERE}")


if __name__ == "__main__":
    main()
