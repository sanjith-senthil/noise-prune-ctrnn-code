#!/usr/bin/env python3
"""The comparisons the manuscript actually makes, with the finalised statistics.

    .venv/bin/python analysis/paper_statistics/key_comparisons.py

The claim list is taken from docs/PAPER_NARRATIVE_PLAN.md section 1 and the
figure-by-figure audit, not invented here. Each row is one sentence the paper
asserts, and the numbers beside it are what the procedure in paper_stats.py
returns for that sentence.

No multiple-comparison correction is applied. Each row is a single
pre-specified comparison, so the p-value beside it is the p-value for that
comparison.

The full 10-80% sparsity range is used throughout. Note that the four arms
which draw a Bernoulli mask have removed about 33% of the recurrent weights at
a requested 10% rather than 10%, so over 10-40% they are compared against
denser competitors; diagnostics.py prints the realized densities. Arms that
exist only at 50-80% are reported in a separate block over their own range,
because an area under a curve is only comparable over a common range.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_stats import (ARMS, GRID, N_BOOT, auc, build_cube, contrast,
                         make_surrogates)

pd.set_option("display.width", 260)

# the two single-factor magnitude controls that carry Fig 3's claim, added to
# the eight arms paper_stats declares
FULL_ARMS = dict(ARMS)
FULL_ARMS["magnitude + global 1/density"] = dict(strategy="l1_unstructured_gain",
                                                 gain="inv_density")
FULL_ARMS["magnitude + global L1 match"] = dict(strategy="l1_unstructured_gain",
                                                gain="match_l1")

LATE_ARMS = dict(ARMS)
LATE_ARMS["noise-prune shuffled-p"] = dict(strategy="simulation_noise_prune_shuffled_rescale")
LATE_ARMS["noise-prune uniform-p"] = dict(strategy="simulation_noise_prune_uniform_rescale")
LATE_ARMS["random + global 1/density"] = dict(strategy="random_unstructured_gain",
                                              gain="inv_density")

CAP = "noise-prune + capped rescale"
RES = "noise-prune + rescale"
SCORE = "noise-prune score only"
MAG = "magnitude"
MAGRES = "magnitude + rescale"
MAGCAP = "magnitude + capped rescale"

# (figure, the sentence the paper asserts, arm A, arm B, expected direction)
CLAIMS = [
    ("Fig 1", "capped noise-prune retains more than magnitude",        CAP, MAG, "+"),
    ("Fig 1", "capped noise-prune retains more than OBS",              CAP, "OBS", "+"),
    ("Fig 1", "capped noise-prune retains more than random",           CAP, "random", "+"),
    ("Fig 1", "the original rescale rule beats magnitude",             RES, MAG, "+"),
    ("Fig 1", "the original rescale rule beats OBS",                   RES, "OBS", "+"),
    ("Fig 2", "the noise score alone picks better weights than magnitude",
     SCORE, MAG, "+"),
    ("Fig 2", "the score alone is not enough: rescaling adds to it",   RES, SCORE, "+"),
    ("Fig 3", "sample-and-rescale lifts even a weak rule (magnitude)", MAGRES, MAG, "+"),
    ("Fig 3", "a derived single factor does not substitute: 1/density",
     MAGRES, "magnitude + global 1/density", "+"),
    ("Fig 3", "a derived single factor does not substitute: global L1",
     MAGRES, "magnitude + global L1 match", "+"),
    ("Fig 3", "THE PUZZLE: magnitude largely catches up once rescaled",
     RES, MAGRES, "~0"),
    ("Fig 5", "capping improves the rescale rule itself",              CAP, RES, "+"),
    ("Fig 5", "capping restores noise-prune's advantage over magnitude",
     CAP, MAGCAP, "+"),
    ("Fig 5", "capping helps other rules too",                         MAGCAP, MAGRES, "+"),
]

LATE_CLAIMS = [
    ("Fig 3", "the sampling matters, not just the factors: shuffled-p fails",
     RES, "noise-prune shuffled-p", "+"),
    ("Fig 3", "the sampling matters: uniform-p fails",
     RES, "noise-prune uniform-p", "+"),
    ("Fig 3", "1/density does not rescue random either",
     RES, "random + global 1/density", "+"),
]

HEADLINE = [(CAP, MAG), (CAP, "OBS"), (CAP, RES), (CAP, MAGCAP), (RES, MAGRES),
            (SCORE, MAG), (MAGRES, MAG)]


def rows_for(cube, sur, claims, lo, hi):
    obs = cube.task_level()
    A = auc(obs, cube.qs, lo, hi)
    out = []
    for fig, text, a, b, direction in claims:
        r = contrast(cube, sur, a, b, lo=lo, hi=hi)
        out.append(dict(
            figure=fig, claim=text, arm_a=a, arm_b=b, expected=direction,
            mean_a=A[:, cube.m(a)].mean(), mean_b=A[:, cube.m(b)].mean(),
            effect=r["effect"], lo95=r["lo95"], hi95=r["hi95"],
            lo68=r["lo68"], hi68=r["hi68"], boot_sd=r["boot_sd"],
            signs=f"{r['n_positive']}/{r['n_tasks']}",
            p_two=r["p_two"], p_one=r["p_one"], at_floor=np.isclose(r["p_two"], r["p_floor"]),
        ))
    return pd.DataFrame(out)


def render(R, title, note=""):
    V = R.copy()
    V["A"] = V.mean_a.round(3)
    V["B"] = V.mean_b.round(3)
    V["effect"] = V.effect.round(3)
    V["CI95"] = V.apply(lambda r: f"[{r.lo95:+.3f}, {r.hi95:+.3f}]", axis=1)
    V["CI68"] = V.apply(lambda r: f"[{r.lo68:+.3f}, {r.hi68:+.3f}]", axis=1)
    V["p"] = V.apply(lambda r: ("= 0.0078 (floor)" if r.at_floor else f"= {r.p_two:.4f}"), axis=1)
    V["holds"] = np.where(V.expected == "~0",
                          np.where(V.p_two > 0.05, "null, as claimed", "NOT null"),
                          np.where((V.lo95 > 0) & (V.p_two <= 0.05), "yes",
                                   np.where(V.effect > 0, "direction only", "NO")))
    print("\n" + "=" * 128)
    print(title)
    if note:
        print(note)
    print("=" * 128)
    print(V[["figure", "claim", "A", "B", "effect", "CI95", "signs", "p", "holds"]]
          .to_string(index=False, max_colwidth=52))
    return R


def main():
    cube = build_cube(FULL_ARMS)
    sur = make_surrogates(cube, n_boot=N_BOOT)
    R = rows_for(cube, sur, CLAIMS, 10, 80)
    render(R, "KEY COMPARISONS -- summary measure, normalised AUC over 10-80% sparsity",
           "A and B are the two arms' AUC; effect = mean over the 8 tasks of (A - B); "
           "signs = tasks favouring A; p = exact two-sided sign-flip, uncorrected")
    R.to_csv(HERE / "key_comparisons.csv", index=False)

    late = build_cube(LATE_ARMS, grid=(50, 60, 70, 80))
    sur_l = make_surrogates(late, n_boot=N_BOOT)
    L = rows_for(late, sur_l, LATE_CLAIMS, 50, 80)
    render(L, "CONTROLS THAT EXIST ONLY AT 50-80% -- normalised AUC over 50-80%",
           "a separate range, so these effects are NOT comparable in size with the table above")
    L.to_csv(HERE / "key_comparisons_late.csv", index=False)

    # ---- the same headline comparisons sparsity by sparsity ----
    rows = []
    for a, b in HEADLINE:
        for q in GRID:
            r = contrast(cube, sur, a, b, q=q)
            rows.append(dict(contrast=f"{a}  -  {b}", sparsity=q, effect=r["effect"],
                             lo95=r["lo95"], hi95=r["hi95"],
                             signs=f"{r['n_positive']}/8", p_two=r["p_two"]))
    S = pd.DataFrame(rows)
    S.to_csv(HERE / "key_comparisons_by_sparsity.csv", index=False)
    print("\n" + "=" * 128)
    print("THE SAME COMPARISONS AT EACH SPARSITY  (retention difference, not AUC)")
    print("the 10-40% columns compare a sampling arm against a denser competitor; see the docstring")
    print("=" * 128)
    for lab, g in S.groupby("contrast", sort=False):
        g = g.set_index("sparsity")
        print(f"\n{lab}")
        t = pd.DataFrame({
            "effect": g.effect.round(3),
            "95% CI": g.apply(lambda r: f"[{r.lo95:+.3f}, {r.hi95:+.3f}]", axis=1),
            "tasks": g.signs,
            "p": g.p_two.map(lambda v: f"{v:.4f}"),
            "": np.where(g.p_two <= 0.05, "*", ""),
        })
        print(t.T.to_string())
    print(f"\nwritten to {HERE}")



if __name__ == "__main__":
    main()
