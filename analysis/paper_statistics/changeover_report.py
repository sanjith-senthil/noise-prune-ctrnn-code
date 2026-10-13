#!/usr/bin/env python3
"""What the corrected probability normalisation changed.

    .venv/bin/python analysis/paper_statistics/changeover_report.py

Runs the whole procedure twice -- once on the legacy data, once on the re-run
suite -- and reports the difference. This is the before/after the revision has
to state: the fix altered the method, so every number it touched has to be
shown to have moved, or shown not to have.

It needs the re-run suite to be complete; until then build_cube("renorm")
raises with a count of the missing cells rather than reporting a partial
answer.

Read it in this order:
  1. realized sparsity -- did the fix do what it claims
  2. the curves -- where the arms moved, by how much
  3. the summary measure and the key claims -- did any verdict change
Only 3 can change the paper. 1 and 2 are there so that a reader can see why.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_stats import (GRID, N_BOOT, SAMPLING_ARMS, auc, build_cube, contrast,
                         load_runs, make_surrogates, select, RENORM_SOURCES, ARMS)
from key_comparisons import CLAIMS, FULL_ARMS

pd.set_option("display.width", 250)


def verdict(r):
    if r["expected"] == "~0":
        return "null" if r["p_two"] > 0.05 else "not null"
    return "holds" if (r["lo95"] > 0 and r["p_two"] <= 0.05) else "fails"


def main() -> None:
    print("building both cubes (the renorm one raises if the suite is incomplete) ...")
    old = build_cube(FULL_ARMS, source_set="legacy")
    new = build_cube(FULL_ARMS, source_set="renorm")
    s_old = make_surrogates(old, n_boot=N_BOOT)
    s_new = make_surrogates(new, n_boot=N_BOOT)

    # ---- 1. did the fix do what it claims ----------------------------
    print("\n" + "=" * 104)
    print("1. REALIZED SPARSITY -- mean % of recurrent weights actually removed")
    print("=" * 104)
    rows = {}
    for label, pool in (("legacy", load_runs()), ("renorm", load_runs(sources=RENORM_SOURCES))):
        for name, spec in FULL_ARMS.items():
            if label == "renorm" and name not in SAMPLING_ARMS:
                continue
            s = select(pool, **spec)
            if s.empty:
                continue
            rows[(name, label)] = (s.groupby(["pct", "task_short", "source_network_seed"])
                                   .removed.mean().groupby(level=0).mean())
    R = pd.DataFrame(rows).round(1)
    R.columns = pd.MultiIndex.from_tuples(R.columns)
    print(R.to_string())
    print("\nAn arm whose renorm row is not within a few tenths of the requested value "
          "has not been fixed; say so rather than reporting its curve.")

    # ---- 2. the curves ------------------------------------------------
    print("\n" + "=" * 104)
    print("2. MEAN RETENTION, before and after")
    print("=" * 104)
    co = old.task_level().mean(axis=0)
    cn = new.task_level().mean(axis=0)
    for mi, name in enumerate(old.methods):
        tag = "  <- re-run" if name in SAMPLING_ARMS else ""
        print(f"\n{name}{tag}")
        t = pd.DataFrame({"before": co[mi], "after": cn[mi],
                          "delta": cn[mi] - co[mi]}, index=old.qs).T
        print(t.round(3).to_string())

    # ---- 3. the summary measure and the claims ------------------------
    print("\n" + "=" * 104)
    print("3. SUMMARY MEASURE (normalised AUC 10-80%)")
    print("=" * 104)
    ao = auc(old.task_level(), old.qs).mean(axis=0)
    an = auc(new.task_level(), new.qs).mean(axis=0)
    A = pd.DataFrame({"method": old.methods, "before": ao, "after": an,
                      "delta": an - ao,
                      "re_run": [m in SAMPLING_ARMS for m in old.methods]})
    print(A.round(4).sort_values("after", ascending=False).to_string(index=False))

    print("\n" + "=" * 104)
    print("4. THE KEY CLAIMS -- did any verdict change?")
    print("=" * 104)
    out = []
    for fig, text, a, b, direction in CLAIMS:
        ro = contrast(old, s_old, a, b); ro["expected"] = direction
        rn = contrast(new, s_new, a, b); rn["expected"] = direction
        out.append(dict(
            figure=fig, claim=text,
            eff_before=ro["effect"], eff_after=rn["effect"],
            delta=rn["effect"] - ro["effect"],
            CI_after=f"[{rn['lo95']:+.3f}, {rn['hi95']:+.3f}]",
            signs_after=f"{rn['n_positive']}/{rn['n_tasks']}",
            p_after=rn["p_two"],
            before=verdict(ro), after=verdict(rn)))
    V = pd.DataFrame(out)
    V["changed"] = np.where(V.before != V.after, "*** CHANGED ***", "")
    show = V.copy()
    for c in ("eff_before", "eff_after", "delta"):
        show[c] = show[c].round(3)
    show["p_after"] = show.p_after.map(lambda v: f"{v:.4f}")
    print(show[["figure", "claim", "eff_before", "eff_after", "delta", "CI_after",
                "signs_after", "p_after", "before", "after", "changed"]]
          .to_string(index=False, max_colwidth=44))

    changed = V[V.before != V.after]
    print(f"\nclaims whose verdict changed: {len(changed)} of {len(V)}")
    if len(changed):
        for _, r in changed.iterrows():
            print(f"   {r.claim}: {r.before} -> {r.after} "
                  f"(effect {r.eff_before:+.3f} -> {r.eff_after:+.3f})")
    V.to_csv(HERE / "changeover_report.csv", index=False)
    print(f"\nwritten to {HERE / 'changeover_report.csv'}")


if __name__ == "__main__":
    main()
