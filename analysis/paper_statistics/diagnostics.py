#!/usr/bin/env python3
"""Robustness checks on the finalised procedure. Nothing here replaces a
reported number; each item either bounds a known weakness or shows that a
choice in the procedure does not matter.

  A  the multiple-comparison arithmetic. With 8 tasks the exact sign-flip test
     has a p-value floor, so Holm's power depends on family size in a way that
     has nothing to do with the data.
  B  realized versus requested sparsity. The arms that sample do not remove the
     requested fraction below 50%, so the low-sparsity contrasts compare arms at
     unequal density.
  C  the summary measure restricted to 50-80%, where all arms do hit the
     requested sparsity -- a sensitivity check on B.
  D  the one ambiguity in the resampling spec: whether a task drawn twice gets
     one seed draw or two independent ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_stats import (ARMS, GRID, NOISE_PRUNE_ARMS, N_BOOT, SEED, Surrogates,
                         auc, build_cube, contrast, holm, make_surrogates,
                         percentile_ci, signflip_exact)

pd.set_option("display.width", 250)
CAP, RES = NOISE_PRUNE_ARMS

def main():
    cube = build_cube()
    sur = make_surrogates(cube, n_boot=N_BOOT)

    # =====================================================================
    print("=" * 104)
    print("A. WHY HOLM BEHAVES ODDLY HERE: the exact test has a floor")
    print("=" * 104)
    print("With n tasks the sign-flip test enumerates 2^n sign assignments, so the")
    print("smallest two-sided p-value it can return is 2/2^n. Holm multiplies the")
    print("smallest p in a family by the family size m. So for n = 8:\n")
    floor2, floor1 = 2 / 256, 1 / 256
    rows = []
    for m in range(1, 15):
        rows.append(dict(family_size=m,
                         two_sided=f"{min(1, m*floor2):.4f}",
                         two_ok="yes" if m * floor2 <= 0.05 else "NO",
                         one_sided=f"{min(1, m*floor1):.4f}",
                         one_ok="yes" if m * floor1 <= 0.05 else "NO"))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSo with the two-sided test, a family of 7 or more can NEVER produce a")
    print(f"Holm-significant result -- 7 x {floor2:.4f} = {7*floor2:.4f} > 0.05 -- however")
    print("large the effect and even if all 8 tasks agree. That is arithmetic, not")
    print("evidence. One-sided raises the usable family size to 12.")

    print("\nthe AUC families as run, under both tails:")
    rows = []
    for ref in (CAP, RES):
        fam = [m for m in cube.methods if m != ref and not (ref == RES and m == CAP)]
        r2, r1 = [], []
        for other in fam:
            c = contrast(cube, sur, ref, other)
            r2.append(c["p_two"]); r1.append(c["p_one"])
        h2, h1 = holm(np.array(r2)), holm(np.array(r1))
        rows.append(dict(reference=ref, m=len(fam),
                         holm_two_sided_pass=int((h2 <= 0.05).sum()),
                         holm_one_sided_pass=int((h1 <= 0.05).sum()),
                         uncorrected_pass=int((np.array(r2) <= 0.05).sum())))
    print(pd.DataFrame(rows).to_string(index=False))

    # =====================================================================
    print("\n" + "=" * 104)
    print("B. REALIZED VERSUS REQUESTED SPARSITY")
    print("=" * 104)
    from paper_stats import load_runs, select  # noqa: E402
    d = load_runs()
    rows = {}
    for name, spec in ARMS.items():
        s = select(d, **spec)
        rows[name] = (s[s.pct.isin(GRID)]
                      .groupby(["pct", "task_short", "source_network_seed"]).removed.mean()
                      .groupby(level=0).mean())
    Rz = pd.DataFrame(rows)
    print("mean % of recurrent weights actually removed, by requested sparsity")
    print(Rz.round(1).to_string())
    print("\nThe arms that draw a Bernoulli mask cannot reach a low requested")
    print("sparsity: the per-edge keep probabilities saturate at 1 and the excess")
    print("probability mass is discarded. At a requested 10% they have removed")
    print(f"{Rz.loc[10, RES]:.0f}%. Magnitude and OBS remove exactly what is asked.")
    print("\nConsequence for the tables: every per-sparsity contrast at 10-40%")
    print("compares a sparser noise-prune network against a denser competitor, so a")
    print("negative effect there is not evidence against the method. The AUC")
    print("integrates over that region too, which C bounds.")

    # =====================================================================
    print("\n" + "=" * 104)
    print("C. SENSITIVITY: the summary measure over 50-80% only, where every arm")
    print("   does reach the requested sparsity")
    print("=" * 104)
    obs = cube.task_level()
    rows = []
    for lo, hi in ((10, 80), (50, 80)):
        a_obs = auc(obs, cube.qs, lo, hi)
        a_sur = auc(sur.A, cube.qs, lo, hi).mean(axis=1)
        for mi, name in enumerate(cube.methods):
            l, h = percentile_ci(a_sur[:, mi], 95)
            rows.append(dict(range=f"{lo}-{hi}%", method=name, AUC=a_obs[:, mi].mean(),
                             lo95=l, hi95=h))
    print(pd.DataFrame(rows).pivot(index="method", columns="range", values="AUC")
          .loc[cube.methods].round(4).to_string())

    print("\ncontrasts on the 50-80% summary (compare with the 10-80% table)")
    rows = []
    for ref in (CAP, RES):
        for other in cube.methods:
            if other == ref or (ref == RES and other == CAP):
                continue
            ia, ib = cube.m(ref), cube.m(other)
            D = auc(obs, cube.qs, 50, 80)[:, ia] - auc(obs, cube.qs, 50, 80)[:, ib]
            boots = (auc(sur.A, cube.qs, 50, 80)[:, :, ia]
                     - auc(sur.A, cube.qs, 50, 80)[:, :, ib]).mean(axis=1)
            l, h = percentile_ci(boots, 95)
            r = signflip_exact(D)
            full = contrast(cube, sur, ref, other)
            rows.append(dict(reference=ref, competitor=other,
                             eff_10_80=full["effect"], eff_50_80=r["effect"],
                             CI=f"[{l:+.3f}, {h:+.3f}]",
                             signs=f"{r['n_positive']}/8", p=f"{r['p_two']:.4f}"))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # =====================================================================
    print("\n" + "=" * 104)
    print("D. DOES THE SEED-RESAMPLING CONVENTION MATTER?")
    print("=" * 104)
    print("The spec says to resample seeds 'separately for each task'. Read one way,")
    print("a task drawn twice gets two independent seed draws (what we do); read the")
    print("other, both copies share one draw. Both are run here.\n")


    def shared_seed_surrogates(cube, n_boot, seed):
        """Variant: seeds drawn once per DISTINCT task, shared by its repeats."""
        rng = np.random.default_rng(seed)
        nt, ns = cube.R.shape[0], cube.R.shape[1]
        task_idx = rng.integers(0, nt, size=(n_boot, nt))
        per_task = rng.integers(0, ns, size=(n_boot, nt, ns))   # indexed by TASK id
        seed_idx = np.take_along_axis(per_task, task_idx[:, :, None]
                                      * np.ones((1, 1, ns), dtype=int), axis=1)
        A = np.empty((n_boot, nt, len(cube.methods), len(cube.qs)))
        for lo in range(0, n_boot, 1000):
            hi = min(lo + 1000, n_boot)
            A[lo:hi] = cube.R[task_idx[lo:hi, :, None], seed_idx[lo:hi]].mean(axis=2)
        return Surrogates(task_idx=task_idx, seed_idx=seed_idx, A=A)


    alt = shared_seed_surrogates(cube, N_BOOT, SEED_ALT := 777)
    rows = []
    for ref, other in ((CAP, "magnitude"), (CAP, "OBS"), (CAP, RES),
                       (RES, "magnitude + rescale"), (RES, "OBS")):
        a = contrast(cube, sur, ref, other)
        b = contrast(cube, alt, ref, other)
        rows.append(dict(contrast=f"{ref} - {other}", effect=a["effect"],
                         independent=f"[{a['lo95']:+.3f}, {a['hi95']:+.3f}]",
                         shared=f"[{b['lo95']:+.3f}, {b['hi95']:+.3f}]",
                         width_change=f"{((b['hi95']-b['lo95'])/(a['hi95']-a['lo95'])-1)*100:+.1f}%"))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nThe effect sizes are identical by construction (they come from the")
    print("observed data, not the surrogates). Only the interval width moves, and")
    print("if it moves by a few percent the convention is immaterial.")

    # Monte-Carlo stability of the intervals themselves
    print("\n" + "=" * 104)
    print("E. IS 10,000 RESAMPLES ENOUGH?")
    print("=" * 104)
    rows = []
    for s in (1, 2, 3):
        sx = make_surrogates(cube, n_boot=N_BOOT, seed=1000 + s)
        c = contrast(cube, sx, CAP, "OBS")
        rows.append(dict(run=s, lo95=c["lo95"], hi95=c["hi95"],
                         width=c["hi95"] - c["lo95"]))
    V = pd.DataFrame(rows)
    print(V.round(5).to_string(index=False))
    print(f"spread of the interval endpoints across independent resample sets: "
          f"{(V.lo95.max()-V.lo95.min()):.5f} (low), {(V.hi95.max()-V.hi95.min()):.5f} (high)")


if __name__ == "__main__":
    main()
