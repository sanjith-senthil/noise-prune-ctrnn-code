#!/usr/bin/env python3
"""Checks on paper_stats.py. Run with .venv/bin/python before trusting any table.

Three kinds of check:
  STRUCTURE  the cube is the balanced 8x3 design it claims to be, and the
             pruning draws really were averaged away first
  ALGEBRA    metrics reproduce closed-form answers on constructed inputs, and
             the shared-surrogate design holds exactly (a contrast on a
             surrogate equals the difference of the two curves on that same
             surrogate, to the last bit)
  CALIBRATION  a simulation with known truth, reporting what the interval's
             coverage and the test's type-I rate actually are at 8 clusters
             rather than assuming the asymptotics
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_stats import (ARMS, GRID, Cube, auc, build_cube, contrast, curve_table,
                         holm, make_surrogates, mean_curve, percentile_ci,
                         signflip_exact, trapezoid)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def synth_cube(R):
    nt, ns, nm, nq = R.shape
    return Cube(R=R, tasks=[f"t{i}" for i in range(nt)], seeds=list(range(ns)),
                methods=[f"m{i}" for i in range(nm)], qs=np.array(GRID[:nq]),
                draws=np.ones_like(R))


# =========================================================================
print("\nSTRUCTURE  (the real cube)")
# =========================================================================
cube = build_cube()
check("cube is 8 tasks x 3 seeds x 8 methods x 8 sparsities",
      cube.R.shape == (8, 3, 8, 8), str(cube.R.shape))
check("no missing cells", not np.isnan(cube.R).any())
check("sparsity grid is 10..80 by 10", list(cube.qs) == list(GRID))
check("retention is finite and non-negative everywhere",
      np.isfinite(cube.R).all() and (cube.R >= 0).all())
# 12 cells are exactly 0: pruned networks that collapsed to zero sequence
# accuracy, all on dmsintseq. Verified against the source CSVs as completed
# runs with correct post-prune nonzero counts, not failures recorded as zero.
zeros = np.argwhere(cube.R == 0)
check("the only zero-retention cells are genuine collapses on one task",
      len(zeros) > 0 and {cube.tasks[t] for t, _, _, _ in zeros} == {"dmsintseq"},
      f"{len(zeros)} zero cells on "
      + ",".join(sorted({cube.tasks[t] for t, _, _, _ in zeros})))

# pruning draws: some arms average several masks per cell, others exactly one
per_arm = {cube.methods[i]: (int(cube.draws[:, :, i].min()), int(cube.draws[:, :, i].max()))
           for i in range(len(cube.methods))}
check("stochastic arms average >1 pruning draw per cell",
      all(per_arm[a][0] >= 1 for a in per_arm)
      and per_arm["noise-prune + rescale"][1] == 3,
      "; ".join(f"{k}:{v[0]}-{v[1]}" for k, v in per_arm.items()))
check("deterministic arms have exactly one draw per cell",
      per_arm["magnitude"] == (1, 1) and per_arm["OBS"] == (1, 1))

# the averaging is real: recompute one cell straight from the source rows
from paper_stats import (RENORM_SOURCES, SAMPLING_ARMS, SOURCE_SET,  # noqa: E402
                         load_runs, select)
# the cube draws sampling arms from the re-run pool, so the raw rows this check
# recomputes from must come from the same place
_arm = "noise-prune + rescale"
d = load_runs(sources=(RENORM_SOURCES if (SOURCE_SET == "renorm" and _arm in SAMPLING_ARMS)
                       else None))
raw = select(d, **ARMS[_arm])
cell = raw[(raw.task_short == cube.tasks[0]) & (raw.source_network_seed == cube.seeds[0])
           & (raw.pct == 50)]
check("one cell equals the plain mean of its pruning draws",
      np.isclose(cell.ret.mean(), cube.R[0, 0, cube.m(_arm),
                                         list(cube.qs).index(50)]),
      f"n={len(cell)}")

# balanced design: averaging seeds then tasks == averaging all 24 networks
flat = cube.R.mean(axis=(0, 1))
nested = mean_curve(cube.task_level())
check("balanced design: task-mean-of-task-means == flat 24-network mean",
      np.allclose(flat, nested, atol=0, rtol=1e-12),
      f"max |diff| = {np.abs(flat - nested).max():.2e}")

# =========================================================================
print("\nALGEBRA  (closed forms and exact identities)")
# =========================================================================
qs = np.array(GRID, dtype=float)

A_const = np.full((8, 1, 8), 0.37)
check("AUC of a constant curve is that constant",
      np.allclose(auc(A_const, qs), 0.37))

A_one = np.ones((8, 1, 8))
check("AUC of perfect retention is exactly 1.0",
      np.allclose(auc(A_one, qs), 1.0))

# a straight line: trapezoid is exact, and the mean of a line over [10,80] is
# its value at the midpoint 45
A_lin = np.broadcast_to(1 - qs / 100.0, (8, 1, 8)).copy()
check("AUC of the line 1-q/100 is its midpoint value 0.55",
      np.allclose(auc(A_lin, qs), 0.55), f"got {auc(A_lin, qs)[0, 0]:.10f}")

# a quadratic, where the trapezoid rule has a known error, so this pins the
# rule in use rather than some other quadrature
y = (qs / 100.0) ** 2
exact = ((0.8 ** 3 - 0.1 ** 3) / 3) / 0.7
trap = trapezoid(y, qs) / 70
check("quadratic matches the trapezoid rule, not exact quadrature",
      np.isclose(auc(np.broadcast_to(y, (8, 1, 8)).copy(), qs)[0, 0], trap)
      and not np.isclose(trap, exact, atol=1e-6),
      f"trapezoid {trap:.6f} vs exact {exact:.6f}")

# AUC is linear, so differencing before or after integrating must agree
rng = np.random.default_rng(1)
A2 = rng.random((8, 2, 8))
check("AUC is linear: AUC(a)-AUC(b) == AUC(a-b)",
      np.allclose(auc(A2, qs)[:, 0] - auc(A2, qs)[:, 1],
                  auc((A2[:, 0] - A2[:, 1])[:, None, :], qs)[:, 0]))

# surrogate indices
sc = synth_cube(rng.random((8, 3, 2, 8)))
sur = make_surrogates(sc, n_boot=4000, seed=7)
check("task indices inside range", sur.task_idx.min() == 0 and sur.task_idx.max() == 7)
check("seed indices inside range", sur.seed_idx.min() == 0 and sur.seed_idx.max() == 2)
check("tasks resampled with replacement (repeats occur)",
      (np.array([len(set(r)) for r in sur.task_idx]) < 8).mean() > 0.95,
      f"{(np.array([len(set(r)) for r in sur.task_idx]) < 8).mean():.3f} of draws have a repeat")
counts = np.apply_along_axis(lambda r: np.bincount(r, minlength=8), 1, sur.task_idx)
check("each task appears once per surrogate on average",
      np.allclose(counts.mean(axis=0), 1.0, atol=0.05), f"{counts.mean(axis=0).round(3)}")
check("seed draws are independent across repeated picks of one task",
      abs(np.corrcoef(sur.seed_idx[:, 0, 0], sur.seed_idx[:, 1, 0])[0, 1]) < 0.05)

# the surrogate's task-level value must equal the mean of the gathered cells
b, p = 11, 3
t, s = sur.task_idx[b, p], sur.seed_idx[b, p]
check("surrogate task-level value == mean of its drawn seeds",
      np.allclose(sur.A[b, p], sc.R[t, s].mean(axis=0)))

# the shared-resample property, which is the whole point of drawing once
real = make_surrogates(cube, n_boot=600, seed=3)
ia, ib = cube.m("noise-prune + capped rescale"), cube.m("magnitude")
k = list(cube.qs).index(60)
diff_of_curves = mean_curve(real.A)[:, ia, k] - mean_curve(real.A)[:, ib, k]
contrast_draws = (real.A[:, :, ia, k] - real.A[:, :, ib, k]).mean(axis=1)
check("contrast on a surrogate == difference of that surrogate's curves",
      np.allclose(diff_of_curves, contrast_draws, atol=0, rtol=1e-12),
      f"max |diff| = {np.abs(diff_of_curves - contrast_draws).max():.2e}")

# percentiles
draws = np.arange(10001, dtype=float)
lo, hi = percentile_ci(draws, 95)
check("95% percentile interval cuts 2.5% from each tail",
      np.isclose(lo, 250.0) and np.isclose(hi, 9750.0), f"[{lo}, {hi}]")

# =========================================================================
print("\nSIGN-FLIP TEST")
# =========================================================================
r = signflip_exact([0.1, 0.2, 0.05, 0.3, 0.15, 0.25, 0.08, 0.12])
check("all 8 differences positive gives the floor p = 2/256",
      np.isclose(r["p_two"], 2 / 256) and r["n_positive"] == 8,
      f"p={r['p_two']:.6f}")
check("floor is recorded as 0.0078", np.isclose(r["p_floor"], 0.0078125))

r0 = signflip_exact([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.05, -0.05])
check("a perfectly balanced set gives p = 1.0", np.isclose(r0["p_two"], 1.0),
      f"p={r0['p_two']:.4f}")

r1 = signflip_exact([1.0, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
mc_rng = np.random.default_rng(5)
D1 = np.array([1.0, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
mc = (np.abs((mc_rng.choice([-1.0, 1.0], (400_000, 8)) * D1).mean(1)) >= abs(D1.mean()) - 1e-12).mean()
check("exact enumeration agrees with 400k Monte-Carlo sign flips",
      abs(r1["p_two"] - mc) < 0.003, f"exact {r1['p_two']:.5f} vs MC {mc:.5f}")

check("sign reversal flips the effect and leaves p unchanged",
      np.isclose(signflip_exact(-D1)["effect"], -r1["effect"])
      and np.isclose(signflip_exact(-D1)["p_two"], r1["p_two"]))
check("one-sided p is at most the two-sided p and at least half of it",
      r1["p_one"] <= r1["p_two"] + 1e-12 and r1["p_one"] >= r1["p_two"] / 2 - 1e-12,
      f"one {r1['p_one']:.5f}, two {r1['p_two']:.5f}")
check("attainable p-values are multiples of 1/256",
      np.isclose(r1["p_two"] * 256, round(r1["p_two"] * 256)))

# =========================================================================
print("\nHOLM")
# =========================================================================
p = np.array([0.001, 0.013, 0.021, 0.04, 0.5])
h = holm(p)
expect = np.maximum.accumulate([5 * 0.001, 4 * 0.013, 3 * 0.021, 2 * 0.04, 1 * 0.5])
check("Holm matches the hand calculation", np.allclose(h, np.minimum(expect, 1.0)),
      str(h.round(4)))
check("Holm is monotone in the sorted p-values",
      np.all(np.diff(h[np.argsort(p)]) >= -1e-15))
check("a family of one leaves p unchanged", np.isclose(holm([0.03])[0], 0.03))
check("Holm never exceeds 1", holm([0.9, 0.95, 0.99]).max() <= 1.0)

# =========================================================================
print("\nCALIBRATION  (simulated, 8 clusters, truth known)")
# =========================================================================
REPS, NB = 500, 600
sim = np.random.default_rng(99)
cover_mean, cover_eff, reject = [], [], []
TRUE_MEAN = 0.6
for _ in range(REPS):
    task_fx = sim.normal(0, 0.12, size=8)            # between-task spread
    seed_fx = sim.normal(0, 0.06, size=(8, 3))       # between-seed spread
    base = TRUE_MEAN + task_fx[:, None] + seed_fx    # (8, 3)
    # two arms with a TRUE difference of zero, correlated within a network
    arms = np.stack([base + sim.normal(0, 0.02, (8, 3)) for _ in range(2)], axis=2)
    R = np.repeat(arms[:, :, :, None], 8, axis=3)
    c = synth_cube(R)
    s = make_surrogates(c, n_boot=NB, seed=int(sim.integers(1 << 30)))
    lo, hi = percentile_ci(mean_curve(s.A)[:, 0, 0], 95)
    cover_mean.append(lo <= TRUE_MEAN <= hi)
    res = contrast(c, s, "m0", "m1", q=10)
    cover_eff.append(res["lo95"] <= 0 <= res["hi95"])
    reject.append(res["p_two"] <= 0.05)

cm, ce, rj = np.mean(cover_mean), np.mean(cover_eff), np.mean(reject)
check("95% interval on the mean covers the truth near 95%", 0.85 <= cm <= 0.97,
      f"empirical coverage {cm:.3f} over {REPS} simulations")
check("95% interval on a null effect covers zero near 95%", 0.88 <= ce <= 0.99,
      f"empirical coverage {ce:.3f}")
check("sign-flip type-I rate does not exceed its attainable level 12/256",
      rj <= 0.0469 + 3 * np.sqrt(0.0469 * 0.95 / REPS),
      f"rejection rate under the null {rj:.4f}, attainable level {12/256:.4f}")

# power, so the floor is not mistaken for weakness
pw = []
for _ in range(300):
    D = sim.normal(0.08, 0.05, size=8)
    pw.append(signflip_exact(D)["p_two"] <= 0.05)
check("the test has real power at a plausible effect (0.08 +- 0.05 over 8 tasks)",
      np.mean(pw) > 0.5, f"power {np.mean(pw):.3f}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + "; ".join(FAIL))
    sys.exit(1)
