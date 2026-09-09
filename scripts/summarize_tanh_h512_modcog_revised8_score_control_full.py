#!/usr/bin/env python3
"""Merge and summarize the full score-control suite.

Aggregation follows the conventions locked in
``paper_artifacts/official_h512_24net/data/revised_scope/PAPER_RESULTS_PROVENANCE.md``:

* pruning-seed technical replicates are averaged first, within each
  ``(arm, sparsity, task, network seed)`` cell;
* the analysis unit is the trained network, giving ``n = 24``;
* error bars are ``mean +/- SEM`` with sample SD (``ddof=1``) across the 24
  trained networks -- never across the 72 pruning-seed rows;
* the plot-level view averages the four sparsities within each trained network
  before computing SD/SEM, matching the Fig. 5 convention.

Significance uses paired two-sided Wilcoxon signed-rank tests
(``zero_method='wilcox'``, ``method='auto'``) with an exact two-sided binomial
sign test as a distribution-free robustness check, Holm-corrected within each
named comparison family.

Outputs (written next to the merged CSV):
  <stem>_raw.csv                   merged per-run rows with derived retention
  <stem>_summary_by_sparsity.csv   per (arm, sparsity), n=24
  <stem>_plot_points.csv           sparsity-averaged per arm, n=24
  <stem>_holm_tests.csv            plot-level + per-sparsity comparison families
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_full_p50_80"
RESULT_DIR = Path(f"results/{STEM}")
ALPHA = 0.05

ARM_ORDER = [
    "lnp_rescale", "snp_rescale",
    "lnp_uniform", "snp_uniform", "random_gain_invdensity",
    "lnp_shuffled", "snp_shuffled",
    "magnitude", "magnitude_gain_invdensity", "magnitude_gain_matchl1",
    "random",
]

# (arm_a, arm_b, rationale) -- arm_a is the hypothesised better arm.
COMPARISONS = [
    # informative score vs uninformative score, procedure held fixed
    ("lnp_rescale", "lnp_shuffled", "covariance vs shuffled assignment (L-NP)"),
    ("lnp_rescale", "lnp_uniform", "covariance vs uniform probability (L-NP)"),
    ("snp_rescale", "snp_shuffled", "covariance vs shuffled assignment (S-NP)"),
    ("snp_rescale", "snp_uniform", "covariance vs uniform probability (S-NP)"),
    ("lnp_rescale", "random_gain_invdensity", "L-NP vs gain-matched random"),
    ("snp_rescale", "random_gain_invdensity", "S-NP vs gain-matched random"),
    ("lnp_rescale", "magnitude_gain_invdensity", "L-NP vs gain-matched magnitude (inv_density)"),
    ("snp_rescale", "magnitude_gain_invdensity", "S-NP vs gain-matched magnitude (inv_density)"),
    ("lnp_rescale", "magnitude_gain_matchl1", "L-NP vs gain-matched magnitude (match_l1)"),
    ("snp_rescale", "magnitude_gain_matchl1", "S-NP vs gain-matched magnitude (match_l1)"),
    # mechanism contrasts
    ("lnp_uniform", "lnp_shuffled", "uniform gain vs misassigned heavy tail (L-NP)"),
    ("snp_uniform", "snp_shuffled", "uniform gain vs misassigned heavy tail (S-NP)"),
    ("random_gain_invdensity", "random", "gain restoration effect on random"),
    ("magnitude_gain_invdensity", "magnitude", "gain restoration effect on magnitude (inv_density)"),
    ("magnitude_gain_matchl1", "magnitude", "gain restoration effect on magnitude (match_l1)"),
]


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def load_raw(result_dir: Path) -> pd.DataFrame:
    files = sorted(result_dir.glob("*.csv"))
    files = [f for f in files if not f.name.startswith(STEM)]
    if not files:
        raise SystemExit(f"No per-task CSVs found in {result_dir}")
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    d = pd.concat(frames, ignore_index=True)
    print(f"merged {len(files)} per-task CSVs -> {len(d)} rows")

    baseline = (
        d[d.strategy == "none"]
        .set_index("source_model_label")["post_acc_sequence"]
        .rename("baseline_acc_sequence")
    )
    p = d[d.strategy != "none"].copy()
    p["baseline_acc_sequence"] = p["source_model_label"].map(baseline)
    if p["baseline_acc_sequence"].isna().any():
        missing = sorted(p.loc[p.baseline_acc_sequence.isna(), "source_model_label"].unique())
        raise SystemExit(f"missing unpruned baseline for: {missing}")
    p["sequence_retention"] = p["post_acc_sequence"] / p["baseline_acc_sequence"]
    p["arm"] = p["run_id"].str.extract(r"netseed\d+_(.+)_p\d\d_pruneseed")
    p["pruning_pct"] = (p["amount"] * 100).round().astype(int)
    p["task_short"] = p["run_id"].str.extract(r"ctrl_full_([a-z0-9]+)_netseed")
    if p["arm"].isna().any():
        raise SystemExit("could not parse arm label from some run_ids")
    return p


def network_cells(p: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average pruning-seed replicates within each trained-network cell."""
    return (
        p.groupby(["arm", "pruning_pct", "task_short", "source_network_seed"], as_index=False)[metric]
        .mean()
    )


def summarize(cells: pd.DataFrame, metric: str, by: list[str]) -> pd.DataFrame:
    g = cells.groupby(by)[metric]
    out = g.agg(n="count", mean="mean", sd=lambda s: s.std(ddof=1)).reset_index()
    out["sem"] = out["sd"] / np.sqrt(out["n"])
    return out


def run_family(units: pd.DataFrame, family: str, extra: dict | None = None) -> list[dict]:
    """units: index = network id, columns = arm, values = metric."""
    rows = []
    for a, b, rationale in COMPARISONS:
        if a not in units.columns or b not in units.columns:
            continue
        paired = units[[a, b]].dropna()
        x, y = paired[a].to_numpy(), paired[b].to_numpy()
        diff = x - y
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
        ties = int((diff == 0).sum())
        w = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
        nonzero = wins + losses
        sign_p = (
            binomtest(wins, nonzero, 0.5, alternative="two-sided").pvalue
            if nonzero > 0 else 1.0
        )
        row = {
            "family": family,
            "arm_a": a,
            "arm_b": b,
            "rationale": rationale,
            "n": len(paired),
            "mean_a": float(x.mean()),
            "mean_b": float(y.mean()),
            "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)),
            "wins_a": wins,
            "losses_a": losses,
            "ties": ties,
            "wilcoxon_p": float(w.pvalue),
            "sign_test_p": float(sign_p),
            "test": "paired two-sided Wilcoxon signed-rank (zero_method=wilcox, method=auto)",
            "alpha": ALPHA,
        }
        row.update(extra or {})
        rows.append(row)
    if not rows:
        return rows
    frame = pd.DataFrame(rows)
    frame["multiple_comparison_family_size"] = len(frame)
    frame["holm_p"] = holm(frame["wilcoxon_p"].to_numpy())
    frame["sign_test_holm_p"] = holm(frame["sign_test_p"].to_numpy())
    frame["reject_holm_alpha_0_05"] = frame["holm_p"] <= ALPHA
    return frame.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()

    p = load_raw(args.result_dir)
    metric = "sequence_retention"
    cells = network_cells(p, metric)

    n_networks = cells.groupby(["arm", "pruning_pct"])[["task_short", "source_network_seed"]].apply(
        lambda f: len(f.drop_duplicates())
    )
    print(f"trained-network cells per (arm, sparsity): min {n_networks.min()}, max {n_networks.max()}")

    by_sparsity = summarize(cells, metric, ["arm", "pruning_pct"])
    plot_cells = cells.groupby(["arm", "task_short", "source_network_seed"], as_index=False)[metric].mean()
    plot_points = summarize(plot_cells, metric, ["arm"])

    # spectral abscissa, same aggregation
    abscissa_cells = network_cells(p, "post_rec_ct_abscissa")
    abscissa = summarize(abscissa_cells, "post_rec_ct_abscissa", ["arm", "pruning_pct"])
    by_sparsity = by_sparsity.merge(
        abscissa.rename(columns={"mean": "abscissa_mean", "sd": "abscissa_sd", "sem": "abscissa_sem"})
        [["arm", "pruning_pct", "abscissa_mean", "abscissa_sd", "abscissa_sem"]],
        on=["arm", "pruning_pct"], how="left",
    )

    units_plot = plot_cells.pivot_table(
        index=["task_short", "source_network_seed"], columns="arm", values=metric
    )
    tests = run_family(units_plot, "score_control_plot_level_sparsity_averaged")
    for pct in sorted(cells["pruning_pct"].unique()):
        sub = cells[cells.pruning_pct == pct].pivot_table(
            index=["task_short", "source_network_seed"], columns="arm", values=metric
        )
        tests.extend(run_family(sub, f"score_control_per_sparsity_p{pct}", {"pruning_pct": pct}))

    out_raw = args.result_dir / f"{STEM}_raw.csv"
    out_sum = args.result_dir / f"{STEM}_summary_by_sparsity.csv"
    out_plot = args.result_dir / f"{STEM}_plot_points.csv"
    out_tests = args.result_dir / f"{STEM}_holm_tests.csv"
    keep = [
        "run_id", "arm", "strategy", "task_short", "source_network_seed", "pruning_seed",
        "pruning_pct", "amount", "post_acc_sequence", "baseline_acc_sequence",
        "sequence_retention", "post_rec_weight_nz_count", "post_rec_ct_abscissa",
    ]
    keep += [c for c in ("prune_prob_control", "prune_kept_amp_mean", "prune_kept_amp_max",
                         "prune_gain_restore_factor", "prune_gain_restore_offdiag_l1_ratio",
                         "prune_leak_shift") if c in p.columns]
    p[keep].to_csv(out_raw, index=False)
    by_sparsity.to_csv(out_sum, index=False)
    plot_points.to_csv(out_plot, index=False)
    pd.DataFrame(tests).to_csv(out_tests, index=False)

    order = [a for a in ARM_ORDER if a in set(by_sparsity["arm"])]
    piv = by_sparsity.pivot(index="arm", columns="pruning_pct", values="mean")
    sem = by_sparsity.pivot(index="arm", columns="pruning_pct", values="sem")
    pcts = sorted(piv.columns)
    print(f"\nretention, n = {plot_points['n'].iloc[0]} trained networks (pruning seeds averaged first)\n")
    header = "".join(f"{f'p{c}':>17}" for c in pcts)
    print(f"{'arm':<28}{header}{'avg':>9}")
    pp = plot_points.set_index("arm")
    for a in order:
        cellstr = "".join(f"{f'{piv.loc[a, c]:.3f}+/-{sem.loc[a, c]:.3f}':>17}" for c in pcts)
        print(f"{a:<28}{cellstr}{pp.loc[a, 'mean']:>9.3f}")

    plot_tests = pd.DataFrame([t for t in tests if t["family"].endswith("sparsity_averaged")])
    print(f"\nplot-level family (sparsity-averaged, Holm across {len(plot_tests)} comparisons):\n")
    print(f"{'comparison':<52}{'A':>7}{'B':>7}{'diff':>8}{'wins':>8}{'holm p':>11}{'sig':>5}")
    for _, r in plot_tests.iterrows():
        print(f"{r.arm_a + ' vs ' + r.arm_b:<52}{r.mean_a:>7.3f}{r.mean_b:>7.3f}{r.mean_diff:>8.3f}"
              f"{str(r.wins_a) + '/' + str(r.n):>8}{r.holm_p:>11.2e}{'  *' if r.reject_holm_alpha_0_05 else '   ':>5}")
    print(f"\nwrote:\n  {out_raw}\n  {out_sum}\n  {out_plot}\n  {out_tests}")


if __name__ == "__main__":
    main()
