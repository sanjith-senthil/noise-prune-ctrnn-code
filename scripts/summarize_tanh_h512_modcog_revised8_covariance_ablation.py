#!/usr/bin/env python3
"""Summarize the covariance-ablation suite and compile all five reviewer controls.

Aggregation follows the conventions locked in
``paper_artifacts/official_h512_24net/data/revised_scope/PAPER_RESULTS_PROVENANCE.md``
and used by ``summarize_..._score_control_full.py``:

* pruning-seed technical replicates are averaged first, within each
  ``(arm, sparsity, task, network seed)`` cell;
* the analysis unit is the trained network, giving ``n = 24``;
* error bars are ``mean +/- SEM`` with sample SD (``ddof=1``) across the 24
  trained networks -- never across the 72 pruning-seed rows.

Because this suite's defaults are byte-identical to ``score_control_full``'s
(asserted by the generator), its arms merge with that suite's directly, and the
combined table is the deliverable: all five controls reviewer 1 asked for, at
one standard, against the reference arms.

Regression gate (fatal)
-----------------------
``magnitude`` and ``lnp_rescale`` are re-run here at ``score_control_full``'s
seeds and must reproduce it exactly. ``magnitude`` checks the evaluation
protocol; ``lnp_rescale`` checks the *stochastic* code path, which is the one
that carried the unseeded-RNG defect (errata E12) -- a deterministic arm alone
would not have caught it.

Outputs (written next to the merged CSV):
  <stem>_raw.csv                    merged per-run rows with retention
  <stem>_summary_by_sparsity.csv    per (arm, sparsity), n = 24
  <stem>_regression_check.csv       this suite vs score_control_full
  reviewer1_controls_combined.csv   all five controls + references, n = 24
  reviewer1_controls_tests.csv      paired Wilcoxon + Holm per sparsity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_covariance_ablation_full_p50_80"
REF_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_full_p50_80"
RESULT_DIR = Path(f"results/{STEM}")
REF_RAW = Path(f"results/{REF_STEM}/{REF_STEM}_raw.csv")
REF_FALLBACK = Path(f"../data_release_staging/results/revision_2026/score_controls/{REF_STEM}_raw.csv")
ALPHA = 0.05
TOL = 1e-9
REGRESSION_ARMS = ("magnitude", "lnp_rescale")

# The five controls the reviewer asked for, plus references, in reporting order.
COMBINED_ORDER = [
    "lnp_rescale", "snp_rescale",
    "lnp_magnitude", "snp_magnitude",
    "lnp_uniform", "snp_uniform",
    "lnp_shuffled", "snp_shuffled",
    "magnitude_gain_invdensity", "random_gain_invdensity",
    "magnitude_gain_matchl1",
    "magnitude", "random",
]
# (better_arm, worse_arm, rationale)
COMPARISONS = [
    ("lnp_rescale", "lnp_magnitude", "covariance term alone (L-NP)"),
    ("snp_rescale", "snp_magnitude", "covariance term alone (S-NP)"),
    ("lnp_rescale", "lnp_uniform", "covariance vs uniform p (L-NP)"),
    ("snp_rescale", "snp_uniform", "covariance vs uniform p (S-NP)"),
    ("lnp_rescale", "lnp_shuffled", "covariance vs shuffled assignment (L-NP)"),
    ("snp_rescale", "snp_shuffled", "covariance vs shuffled assignment (S-NP)"),
    ("lnp_rescale", "magnitude_gain_invdensity", "L-NP vs gain-matched magnitude (reviewer's convention)"),
    ("lnp_rescale", "random_gain_invdensity", "L-NP vs gain-matched random (reviewer's convention)"),
    ("lnp_magnitude", "lnp_uniform", "magnitude-weighted p vs uniform p"),
]


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _derive(d: pd.DataFrame, arm_re: str, task_re: str) -> pd.DataFrame:
    baseline = (d[d.strategy == "none"]
                .set_index("source_model_label")["post_acc_sequence"])
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(baseline)
    if p["baseline_acc_sequence"].isna().any():
        missing = sorted(p.loc[p.baseline_acc_sequence.isna(), "source_model_label"].unique())
        raise SystemExit(f"missing unpruned baseline for: {missing}")
    p["sequence_retention"] = p["post_acc_sequence"] / p["baseline_acc_sequence"]
    p["arm"] = p["run_id"].str.extract(arm_re)
    p["task_short"] = p["run_id"].str.extract(task_re)
    p["pruning_pct"] = (p["amount"] * 100).round().astype(int)
    if p["arm"].isna().any() or p["task_short"].isna().any():
        raise SystemExit("could not parse arm/task from some run_ids")
    return p


# Per-unit outputs are named "<task>.csv" or "<task>_p50.csv". Matching on that
# rather than on "not the stem" matters: this script writes its own tables into
# the same directory, and reviewer1_controls_*.csv do not start with the stem, so
# a looser glob silently re-ingests them on the second run.
UNIT_NAMES = frozenset(
    f"{t}{suffix}.csv"
    for t in ("ctxdlydm2intseq", "ctxdlydm1intseq", "dlydm1intseq", "dlydm2intseq",
              "multidlydmintseq", "dm1seqr", "dm2seql", "dmsintseq")
    for suffix in ("", "_p50", "_p60", "_p70", "_p80")
)


def load_this(result_dir: Path) -> pd.DataFrame:
    files = [f for f in sorted(result_dir.glob("*.csv")) if f.name in UNIT_NAMES]
    if not files:
        raise SystemExit(f"No per-unit CSVs found in {result_dir}")
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"merged {len(files)} per-unit CSVs -> {len(d)} rows")
    return _derive(d, r"netseed\d+_(.+)_p\d\d_pruneseed", r"covabl_([a-z0-9]+)_netseed")


def load_reference() -> pd.DataFrame | None:
    src = REF_RAW if REF_RAW.exists() else REF_FALLBACK
    if not src.exists():
        print(f"reference suite not found ({REF_RAW} or {REF_FALLBACK})")
        return None
    d = pd.read_csv(src, low_memory=False)
    print(f"reference suite: {src} ({len(d)} rows)")
    # score_control_full's "_raw.csv" is its summarizer's OUTPUT, not the runner's:
    # baseline rows are already stripped and retention/arm/task columns already
    # derived. Re-deriving would fail on the absent source_model_label, so take the
    # derived form when present and only derive from a genuine runner CSV.
    derived = {"arm", "task_short", "pruning_pct", "sequence_retention"}
    if derived.issubset(d.columns):
        print(f"  already derived by its own summarizer; using as-is "
              f"({d.arm.nunique()} arms, {len(d)} pruned rows)")
        return d
    return _derive(d, r"netseed\d+_(.+)_p\d\d_pruneseed", r"ctrl_full_([a-z0-9]+)_netseed")


def network_cells(p: pd.DataFrame) -> pd.DataFrame:
    """Average pruning-seed replicates within each trained-network cell."""
    return (p.groupby(["arm", "pruning_pct", "task_short", "source_network_seed"],
                      as_index=False)["sequence_retention"].mean())


def summarize(cells: pd.DataFrame, by: list) -> pd.DataFrame:
    g = cells.groupby(by)["sequence_retention"]
    out = g.agg(n="count", mean="mean", sd=lambda s: s.std(ddof=1)).reset_index()
    out["sem"] = out["sd"] / np.sqrt(out["n"])
    return out


def regression_check(p: pd.DataFrame, ref: pd.DataFrame | None, strict: bool) -> pd.DataFrame:
    if ref is None:
        return pd.DataFrame()
    key = ["arm", "task_short", "pruning_pct", "source_network_seed", "pruning_seed"]
    a = p[p.arm.isin(REGRESSION_ARMS)][key + ["post_acc_sequence"]]
    b = ref[ref.arm.isin(REGRESSION_ARMS)][key + ["post_acc_sequence"]]
    m = a.merge(b, on=key, suffixes=("_new", "_ref"))
    if m.empty:
        print("no overlapping rows for the regression check")
        return m
    m["abs_diff"] = (m.post_acc_sequence_new - m.post_acc_sequence_ref).abs()
    n_bad = int((m.abs_diff > TOL).sum())
    print(f"\nregression vs {REF_STEM}:")
    for arm, g in m.groupby("arm"):
        print(f"  {arm:<14} {len(g):>4} shared runs, max |delta| = {g.abs_diff.max():.3e}")
    if n_bad:
        print(m[m.abs_diff > TOL].head(20).to_string(index=False))
        if strict:
            raise SystemExit(
                "REGRESSION: this suite no longer reproduces score_control_full. The two "
                "cannot be merged until this is explained.")
    else:
        print("  reproduces exactly -- the suites sit on the same footing and merge safely")
    return m


def run_tests(units: pd.DataFrame, family: str) -> list:
    rows = []
    for a, b, why in COMPARISONS:
        if a not in units.columns or b not in units.columns:
            continue
        paired = units[[a, b]].dropna()
        if paired.empty:
            continue
        x, y = paired[a].to_numpy(), paired[b].to_numpy()
        diff = x - y
        wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
        w = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
        nz = wins + losses
        rows.append({
            "family": family, "arm_a": a, "arm_b": b, "rationale": why, "n": len(paired),
            "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
            "wins_a": wins, "losses_a": losses, "ties": int((diff == 0).sum()),
            "wilcoxon_p": float(w.pvalue),
            "sign_test_p": float(binomtest(wins, nz, 0.5).pvalue) if nz else 1.0,
            "test": "paired two-sided Wilcoxon signed-rank (zero_method=wilcox, method=auto)",
            "alpha": ALPHA,
        })
    if not rows:
        return rows
    f = pd.DataFrame(rows)
    f["multiple_comparison_family_size"] = len(f)
    f["holm_p"] = holm(f["wilcoxon_p"].to_numpy())
    f["reject_holm_alpha_0_05"] = f["holm_p"] <= ALPHA
    return f.to_dict("records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--allow-regression", action="store_true")
    args = ap.parse_args()

    p = load_this(args.result_dir)
    p.to_csv(args.result_dir / f"{STEM}_raw.csv", index=False)
    ref = load_reference()

    reg = regression_check(p, ref, strict=not args.allow_regression)
    if len(reg):
        reg.to_csv(args.result_dir / f"{STEM}_regression_check.csv", index=False)

    cells = network_cells(p)
    summarize(cells, ["arm", "pruning_pct"]).to_csv(
        args.result_dir / f"{STEM}_summary_by_sparsity.csv", index=False)

    # ---- combined table: the five controls plus references, one standard ----
    new_arms = {"lnp_magnitude", "snp_magnitude"}
    combined = cells[cells.arm.isin(new_arms)]
    if ref is not None:
        combined = pd.concat([combined, network_cells(ref)], ignore_index=True)
    combined = combined.drop_duplicates(["arm", "pruning_pct", "task_short", "source_network_seed"])

    n_by = combined.groupby(["arm", "pruning_pct"]).size()
    bad = n_by[n_by != 24]
    if len(bad):
        print(f"\nWARNING: {len(bad)} (arm, sparsity) cells do not have n = 24:")
        print(bad.to_string())

    summ = summarize(combined, ["arm", "pruning_pct"])
    summ.to_csv(args.result_dir / "reviewer1_controls_combined.csv", index=False)

    tests = []
    for pct, g in combined.groupby("pruning_pct"):
        units = g.pivot_table(index=["task_short", "source_network_seed"],
                              columns="arm", values="sequence_retention")
        for r in run_tests(units, f"sparsity_{pct}"):
            r["pruning_pct"] = pct
            tests.append(r)
    if tests:
        pd.DataFrame(tests).to_csv(args.result_dir / "reviewer1_controls_tests.csv", index=False)

    pd.set_option("display.width", 200)
    piv = summ.pivot(index="arm", columns="pruning_pct", values="mean")
    piv = piv.reindex([a for a in COMBINED_ORDER if a in piv.index])
    print("\n=== reviewer-1 controls, sequence retention, n = 24 trained networks ===")
    print(piv.round(3).to_string())
    if tests:
        t = pd.DataFrame(tests)
        print("\n=== paired Wilcoxon, Holm-corrected within each sparsity ===")
        for pct, g in t.groupby("pruning_pct"):
            print(f"\n-- {pct}% --")
            for _, r in g.iterrows():
                flag = "*" if r.reject_holm_alpha_0_05 else " "
                print(f" {flag} {r.arm_a:<14} > {r.arm_b:<26} d={r.mean_diff:+.3f}  "
                      f"{r.wins_a:>2}/{r.n}  holm p={r.holm_p:.3g}")
    print(f"\nwrote {args.result_dir}/reviewer1_controls_{{combined,tests}}.csv")


if __name__ == "__main__":
    main()
