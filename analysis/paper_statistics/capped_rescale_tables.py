#!/usr/bin/env python3
"""Data tables for the capped-rescale claims.

    .venv/bin/python analysis/paper_statistics/capped_rescale_tables.py

Four claims, four tables. Every number uses the paper's procedure: pruning
draws averaged within a network, then the 3 network seeds within a task, then
the 8 tasks; effect sizes are the mean of the 8 paired task-level differences;
intervals are percentiles of 10,000 hierarchical resamples (tasks, then seeds
within task) shared across every contrast; p-values are the exact sign-flip
test over all 256 sign assignments.

  1  capping helps, and the score still matters once both scores are capped.
     The second contrast IS the covariance contribution: same cap, same
     sampling, the only difference is whether the covariance term entered the
     score. It is carried as its own column so the two claims can be read
     against each other.
  2  the same thing per task, since 8/8 is the strongest part of the evidence.
  3  the cap percentile sweep: is q50 the best choice?
  4  does the constant factor reproduce the q50 quantile cap?

A note on the p-value floor. With 8 tasks the exact test cannot return less
than 2/256 = 0.0078, reached whenever all 8 differences share a sign. A value
AT the floor means the data are as extreme as this test can register, not that
the effect is marginal -- read the effect size and the sign count alongside.
"""
from __future__ import annotations

import glob
import pathlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_stats import (CAP_CONSTANT, Cube, N_BOOT, auc, build_cube, contrast,
                         make_surrogates, percentile_ci, signflip_exact)

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
R = f"{ROOT}/results"
SUITE = {
    "renorm":   f"{R}/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_renorm_p10_80/*.csv",
    "recap":    f"{R}/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_renorm_recap_p10_80/*.csv",
    "capq50":   f"{R}/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_capq50_p10_80/*.csv",
    "capsweep": f"{R}/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_capsweep_p50_80/*.csv",
}
NEED = ["strategy", "amount", "post_acc_sequence", "pre_acc_sequence", "task",
        "source_network_seed", "prune_rescale_cap_quantile", "prune_rescale_cap_value"]
NOISE = "simulation_noise_prune_capped_rescale"
MAGN = "simulation_noise_prune_capped_magnitude_rescale"
_POOL: dict = {}


def pool(name: str) -> pd.DataFrame:
    if name in _POOL:
        return _POOL[name]
    fr = []
    for f in sorted(glob.glob(SUITE[name])):
        cols = set(pd.read_csv(f, nrows=0).columns)
        fr.append(pd.read_csv(f, usecols=[c for c in NEED if c in cols], low_memory=False))
    d = pd.concat(fr, ignore_index=True)
    d = d[d.strategy != "none"].copy()
    d["t"] = d.task.str.replace("modcog:", "", regex=False)
    d["pct"] = (d.amount * 100).round().astype(int)
    d["ret"] = d.post_acc_sequence / d.pre_acc_sequence
    _POOL[name] = d
    return d


def make_cube(arms: dict, qs) -> Cube:
    """arms: label -> dataframe already filtered to one arm."""
    qs = np.array(sorted(qs))
    tasks = sorted(set.intersection(*[set(df.t) for df in arms.values()]))
    seeds = sorted(set.intersection(*[set(df.source_network_seed) for df in arms.values()]))
    names = list(arms)
    Rm = np.full((len(tasks), len(seeds), len(names), len(qs)), np.nan)
    D = np.zeros_like(Rm)
    for mi, n in enumerate(names):
        g = arms[n].groupby(["t", "source_network_seed", "pct"]).ret.agg(["mean", "size"])
        for (t, s, q), row in g.iterrows():
            if t in tasks and s in seeds and q in qs:
                Rm[tasks.index(t), seeds.index(s), mi, list(qs).index(q)] = row["mean"]
                D[tasks.index(t), seeds.index(s), mi, list(qs).index(q)] = row["size"]
    miss = np.argwhere(np.isnan(Rm))
    if len(miss):
        raise SystemExit(f"{len(miss)} empty cells, e.g. "
                         f"{[f'{tasks[a]}/{names[c]}/{qs[e]}' for a, b, c, e in miss[:4]]}")
    return Cube(R=Rm, tasks=tasks, seeds=seeds, methods=names, qs=qs, draws=D)


def fmt(r, pct=None):
    return dict(sparsity=("AUC 10-80%" if pct is None else f"{pct}%"),
                effect=round(r["effect"], 4),
                CI95=f"[{r['lo95']:+.3f}, {r['hi95']:+.3f}]",
                tasks=f"{r['n_positive']}/{r['n_tasks']}",
                p=f"{r['p_two']:.4f}" + ("*" if np.isclose(r["p_two"], r["p_floor"]) else ""))


def main() -> None:
    out = {}

    # ============ 1 + 2: capping helps, and the covariance term still pays ====
    cube = build_cube()
    sur = make_surrogates(cube, n_boot=N_BOOT)
    tl = cube.task_level()
    CAP, RES, MCAP = ("noise-prune + capped rescale", "noise-prune + rescale",
                      "magnitude + capped rescale")
    rows = []
    for qi, q in enumerate(list(cube.qs) + [None]):
        a = contrast(cube, sur, CAP, RES, q=q)
        b = contrast(cube, sur, CAP, MCAP, q=q)
        if q is None:
            mc, mr, mm = (auc(tl, cube.qs)[:, cube.m(x)].mean() for x in (CAP, RES, MCAP))
        else:
            k = list(cube.qs).index(q)
            mc, mr, mm = (tl[:, cube.m(x), k].mean() for x in (CAP, RES, MCAP))
        rows.append(dict(
            sparsity=("AUC 10-80%" if q is None else f"{q}%"),
            capped_NP=round(mc, 4), NP_rescale=round(mr, 4),
            d_capping=round(a["effect"], 4),
            capping_CI=f"[{a['lo95']:+.3f}, {a['hi95']:+.3f}]",
            capping_tasks=f"{a['n_positive']}/8",
            capping_p=f"{a['p_two']:.4f}" + ("*" if np.isclose(a["p_two"], a["p_floor"]) else ""),
            magnitude_capped=round(mm, 4),
            d_covariance=round(b["effect"], 4),
            covariance_CI=f"[{b['lo95']:+.3f}, {b['hi95']:+.3f}]",
            covariance_tasks=f"{b['n_positive']}/8",
            covariance_p=f"{b['p_two']:.4f}" + ("*" if np.isclose(b["p_two"], b["p_floor"]) else ""),
        ))
    T1 = pd.DataFrame(rows)
    print("=" * 150)
    print("TABLE 1  Capping helps the rescale rule, and the covariance term still pays once both scores are capped")
    print("  d_capping    = capped noise-prune  minus  noise-prune sample-and-rescale   (what the cap buys)")
    print("  d_covariance = capped noise-prune  minus  capped magnitude                 (what the covariance term buys)")
    print("  same cap (%.3f), same sampling; the covariance contrast differs only in whether C entered the score" % CAP_CONSTANT)
    print("=" * 150)
    print(T1.to_string(index=False))
    out["table1_capping_and_covariance"] = T1

    # ============ 2: per task, because 8/8 is the strongest evidence =========
    rows = []
    for ti, t in enumerate(cube.tasks):
        rows.append(dict(task=t,
                         d_capping_AUC=round(float(auc(tl, cube.qs)[ti, cube.m(CAP)]
                                                   - auc(tl, cube.qs)[ti, cube.m(RES)]), 4),
                         d_covariance_AUC=round(float(auc(tl, cube.qs)[ti, cube.m(CAP)]
                                                      - auc(tl, cube.qs)[ti, cube.m(MCAP)]), 4)))
    T2 = pd.DataFrame(rows)
    T2.loc[len(T2)] = dict(task="MEAN", d_capping_AUC=round(T2.d_capping_AUC.mean(), 4),
                           d_covariance_AUC=round(T2.d_covariance_AUC.mean(), 4))
    print("\n" + "=" * 90)
    print("TABLE 2  The same two differences, task by task (AUC 10-80%)")
    print("=" * 90)
    print(T2.to_string(index=False))
    out["table2_per_task"] = T2

    # ============ 3: is q50 the best cap percentile? =========================
    sw = pool("capsweep")
    QS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    arms = {f"q{q}": sw[np.isclose(sw.prune_rescale_cap_quantile, q / 100)] for q in QS}
    unc = pool("renorm")
    arms["uncapped"] = unc[unc.strategy == "simulation_noise_prune_rescale"]
    cs = make_cube(arms, (50, 60, 70, 80))
    ss = make_surrogates(cs, n_boot=N_BOOT)
    tlc = cs.task_level()
    rows = []
    for name in list(arms):
        k = cs.m(name)
        r = dict(cap=name)
        for qi, q in enumerate(cs.qs):
            r[f"{q}%"] = round(float(tlc[:, k, qi].mean()), 4)
        r["pooled"] = round(float(tlc[:, k, :].mean()), 4)
        if name != "q50":
            c = contrast(cs, ss, "q50", name, lo=50, hi=80)
            r["q50 minus this"] = round(c["effect"], 4)
            r["CI95"] = f"[{c['lo95']:+.3f}, {c['hi95']:+.3f}]"
            r["tasks"] = f"{c['n_positive']}/8"
            r["p"] = f"{c['p_two']:.4f}" + ("*" if np.isclose(c["p_two"], c["p_floor"]) else "")
        else:
            r.update({"q50 minus this": "--", "CI95": "--", "tasks": "--", "p": "--"})
        rows.append(r)
    T3 = pd.DataFrame(rows)
    print("\n" + "=" * 130)
    print("TABLE 3  Cap percentile sweep, noise score, 24 networks x 3 draws per cell")
    print("  'q50 minus this' is the paired AUC 50-80% difference against the q50 cap; positive means q50 is better")
    print("=" * 130)
    print(T3.to_string(index=False))
    out["table3_cap_percentile_sweep"] = T3

    # ============ 4: does the constant reproduce the q50 cap? ===============
    q50, con = pool("capq50"), pool("recap")
    rows = []
    for score, strat in (("noise", NOISE), ("magnitude", MAGN)):
        arms = {"q50 quantile": q50[q50.strategy == strat],
                f"constant {CAP_CONSTANT}": con[con.strategy == strat]}
        cc = make_cube(arms, range(10, 90, 10))
        sc = make_surrogates(cc, n_boot=N_BOOT)
        t = cc.task_level()
        for qi, q in list(enumerate(cc.qs)) + [(None, None)]:
            r = contrast(cc, sc, "q50 quantile", f"constant {CAP_CONSTANT}", q=q)
            if q is None:
                a, b = (auc(t, cc.qs)[:, i].mean() for i in (0, 1))
            else:
                a, b = t[:, 0, qi].mean(), t[:, 1, qi].mean()
            rows.append(dict(score=score, sparsity=("AUC 10-80%" if q is None else f"{q}%"),
                             q50_cap=round(float(a), 4),
                             constant_cap=round(float(b), 4),
                             difference=round(r["effect"], 4),
                             CI95=f"[{r['lo95']:+.3f}, {r['hi95']:+.3f}]",
                             tasks=f"{r['n_positive']}/8",
                             p=f"{r['p_two']:.4f}" + ("*" if np.isclose(r["p_two"], r["p_floor"]) else "")))
    T4 = pd.DataFrame(rows)
    print("\n" + "=" * 120)
    print(f"TABLE 4  The constant factor {CAP_CONSTANT} against the per-network q50 quantile cap")
    print("  difference = q50 minus constant; near zero means the constant reproduces the adaptive cap")
    print("=" * 120)
    print(T4.to_string(index=False))
    out["table4_constant_vs_q50"] = T4

    for name, df in out.items():
        df.to_csv(HERE / f"capped_{name}.csv", index=False)
    print(f"\n* = at the exact test's floor, 2/256, the smallest 8 tasks can return")
    print(f"written to {HERE}/capped_table*.csv")


if __name__ == "__main__":
    main()
