#!/usr/bin/env python3
"""Does capping restore the covariance contribution at H = 1024?

Merges three H=1024 suites into one cap-tightness axis, pairing the full
covariance score against the covariance-dropped (magnitude-only) score within
trained network at each cap level:

    uncapped   task-preservation suite      snp_rescale      vs snp_magnitude
    q70        capcov suite (both halves)   q70_full         vs q70_mag
    q60        capped suite + capcov        snp_capped_q60   vs q60_mag
    q50        capped suite + capcov        snp_capped_q50   vs q50_mag
    q30        capcov suite (both halves)   q30_full         vs q30_mag

Pruning seed 0 throughout, so n = 24 = 8 tasks x 3 network seeds, matching the
H=512 Suite F protocol. The H=512 reference numbers are +0.011 uncapped,
+0.034 at q70, +0.060 at q50 and +0.137 at q30 (50% sparsity).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

H = 1024
CAPCOV = Path(f"results/task_preservation_tanh_h{H}_modcog_revised8_12k_seqbest_capcov_p50_80")
CAPPED = Path(f"results/task_preservation_tanh_h{H}_modcog_revised8_12k_seqbest"
              "_capped_q50_q60_p50_80")
TASKONLY = Path(f"results/task_preservation_tanh_h{H}_modcog_revised8_12k_seqbest"
                "_revised_task_only_p50_80")
TASKS = ("ctxdlydm2intseq", "ctxdlydm1intseq", "dlydm1intseq", "dlydm2intseq",
         "multidlydmintseq", "dm1seqr", "dm2seql", "dmsintseq")
AMOUNTS = (50, 60, 70, 80)
# cap label -> (full arm source, magnitude arm source)
PAIRS = (
    ("uncapped", ("taskonly", "simulation_noise_prune_rescale"),
                 ("taskonly", "simulation_noise_prune_magnitude_rescale")),
    ("q70",      ("capcov", "q70_full"), ("capcov", "q70_mag")),
    ("q60",      ("capped", "snp_capped_q60"), ("capcov", "q60_mag")),
    ("q50",      ("capped", "snp_capped_q50"), ("capcov", "q50_mag")),
    ("q30",      ("capcov", "q30_full"), ("capcov", "q30_mag")),
)


def hl_ci(d):
    w = np.array([(d[i] + d[j]) / 2.0 for i in range(len(d)) for j in range(i, len(d))])
    w.sort(); n = len(d); z = 1.959963984540054
    k = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    k = max(0, min(k, len(w) - 1))
    return float(np.median(w)), float(w[k]), float(w[len(w) - 1 - k])


def holm(p):
    o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def load(d: Path) -> pd.DataFrame:
    files = [d / f"{t}_p{a}.csv" for t in TASKS for a in AMOUNTS]
    have = [f for f in files if f.exists()]
    if not have:
        raise SystemExit(f"no unit CSVs under {d}")
    if len(have) < len(files):
        print(f"  {d.name}: {len(have)}/{len(files)} unit files present")
    x = pd.concat([pd.read_csv(f, low_memory=False) for f in have], ignore_index=True)
    x["task_short"] = x["task"].str.replace("modcog:", "", regex=False)
    x["pruning_pct"] = (x["amount"] * 100).round().astype(int)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="report on whatever cells are complete instead of failing")
    a = ap.parse_args()

    src = {}
    print("loading:")
    src["capcov"] = load(CAPCOV)
    src["capped"] = load(CAPPED)
    src["taskonly"] = load(TASKONLY)

    # arm labels: capcov/capped carry them in the run_id; taskonly is keyed by strategy
    for k in ("capcov", "capped"):
        src[k]["arm"] = src[k]["run_id"].str.extract(r"netseed\d+_(.+)_p\d+_pruneseed")
    src["taskonly"]["arm"] = src["taskonly"]["strategy"]

    # baselines: every suite carries its own, on the same checkpoints -- check agreement
    bases = {}
    for k, x in src.items():
        b = x[x.strategy == "none"]
        if len(b):
            bases[k] = b.drop_duplicates("source_model_label").set_index(
                "source_model_label")["post_acc_sequence"]
    # The three suites re-evaluate the same unpruned checkpoints, so their
    # baselines must agree exactly; check that, then take the union. Using only
    # this suite's own baselines would cap n at however many of ITS unpruned
    # runs have finished -- and those live in the p50 shards, which are not the
    # shards that answer the question, so a partial read would be needlessly
    # underpowered.
    ref = bases["capcov"]
    for k, b in bases.items():
        c = ref.index.intersection(b.index)
        drift = float((ref.loc[c] - b.loc[c]).abs().max()) if len(c) else float("nan")
        print(f"  baseline check {k:9s}: {len(b)} networks, {len(c)} shared, "
              f"max |delta| = {drift:.3e}")
        if len(c) and drift > 1e-9:
            raise SystemExit(f"baselines disagree between capcov and {k}")
    ref = pd.concat([bases[k] for k in ("capcov", "capped", "taskonly") if k in bases])
    ref = ref[~ref.index.duplicated()]
    print(f"  baseline pool: {len(ref)} networks")

    rows = []
    for k, x in src.items():
        p = x[(x.strategy != "none") & (x.get("pruning_seed", 0) == 0)].copy()
        p["baseline"] = p["source_model_label"].map(ref)
        p = p[p["baseline"].notna()]
        p["retention"] = p["post_acc_sequence"] / p["baseline"]
        p["src"] = k
        rows.append(p[["src", "arm", "task_short", "source_network_seed", "pruning_pct",
                       "retention", "post_acc_sequence", "prune_rescale_cap_value",
                       "prune_frac_positive_amp_capped", "post_rec_ct_abscissa",
                       "post_rec_linear_rho", "prune_inv_p_mean"]])
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(CAPCOV / "size_capcov_merged.csv", index=False)

    unit = ["task_short", "source_network_seed"]
    out, ps = [], []
    for cap, (sf, af), (sm, am) in PAIRS:
        for pct in AMOUNTS:
            f = d[(d.src == sf) & (d.arm == af) & (d.pruning_pct == pct)].set_index(unit)
            m = d[(d.src == sm) & (d.arm == am) & (d.pruning_pct == pct)].set_index(unit)
            f = f[~f.index.duplicated()]; m = m[~m.index.duplicated()]
            c = f.index.intersection(m.index)
            if len(c) < 24:
                if not a.partial and len(c) == 0:
                    continue
                if len(c) == 0:
                    continue
            dd = (f.loc[c, "retention"] - m.loc[c, "retention"]).to_numpy()
            hl, lo, hi = hl_ci(dd); pv = float(wilcoxon(dd).pvalue)
            ps.append(pv)
            out.append(dict(cap=cap, sparsity=pct, n=len(c),
                            cap_value=float(f.loc[c, "prune_rescale_cap_value"].mean()),
                            frac_capped=float(f.loc[c, "prune_frac_positive_amp_capped"].mean()),
                            full=float(f.loc[c, "retention"].mean()),
                            mag=float(m.loc[c, "retention"].mean()),
                            delta=hl, lo=lo, hi=hi, wins=int((dd > 0).sum()), p=pv,
                            abscissa_full=float(f.loc[c, "post_rec_ct_abscissa"].mean()),
                            abscissa_mag=float(m.loc[c, "post_rec_ct_abscissa"].mean())))
    if not out:
        raise SystemExit("no complete cells yet")
    res = pd.DataFrame(out)
    res["p_holm"] = holm(res["p"].to_numpy())

    print("\n=== retention by cap level, n = 24 (pruning seed 0) ===")
    print(f"{'cap':9s}{'spars':>7s}{'cap val':>9s}{'frac cap':>9s}"
          f"{'full':>9s}{'mag':>9s}{'cov delta':>11s}{'wins':>8s}{'p_holm':>10s}")
    for _, r in res.sort_values(["sparsity", "cap"]).iterrows():
        star = " *" if r.p_holm < 0.05 else ""
        print(f"{r.cap:9s}{int(r.sparsity):6d}%{r.cap_value:9.3f}{r.frac_capped:9.2f}"
              f"{r.full:9.4f}{r['mag']:9.4f}{r.delta:+11.4f}{r.wins:5d}/{int(r.n):<3d}"
              f"{r.p_holm:10.2e}{star}")

    # Cap optimum. This table compares cap levels against each other, so it must
    # be built on the networks every cap level has finished -- otherwise a
    # partial read compares a complete uncapped arm on 24 networks against a
    # capped arm on whichever 4 tasks happened to run first, and reports the
    # task subset as a cap effect.
    print("\n=== cap optimum: mean retention of the full-score arm, common networks only ===")
    order = ["uncapped", "q70", "q60", "q50", "q30"]
    full_src = {c: (sf, af) for c, (sf, af), _ in PAIRS}
    opt_rows = []
    for pct in AMOUNTS:
        sets, series = [], {}
        for cap in order:
            sf, af = full_src[cap]
            x = d[(d.src == sf) & (d.arm == af) & (d.pruning_pct == pct)].set_index(unit)
            x = x[~x.index.duplicated()]
            if not len(x):
                continue
            series[cap] = x["retention"]; sets.append(set(x.index))
        if not sets:
            continue
        common = set.intersection(*sets)
        for cap, v in series.items():
            opt_rows.append(dict(cap=cap, sparsity=pct, n=len(common),
                                 retention=float(v.loc[sorted(common)].mean())))
    opt = pd.DataFrame(opt_rows)
    piv = opt.pivot_table(index="cap", columns="sparsity", values="retention")
    ns = opt.pivot_table(index="cap", columns="sparsity", values="n").max()
    piv["mean"] = piv.mean(axis=1)
    print(piv.reindex([c for c in order if c in piv.index]).round(4).to_string())
    print("n per sparsity: " + "  ".join(f"{int(k)}%: {int(v)}" for k, v in ns.items()))
    opt.to_csv(CAPCOV / "size_capcov_cap_optimum.csv", index=False)

    res.to_csv(CAPCOV / "size_capcov_summary.csv", index=False)
    print(f"\nwrote {CAPCOV/'size_capcov_summary.csv'}")


if __name__ == "__main__":
    main()
