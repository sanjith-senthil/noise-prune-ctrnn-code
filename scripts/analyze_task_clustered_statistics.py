#!/usr/bin/env python3
"""Task-clustered confirmatory statistics and per-task retention curves.

The main analysis treats the 24 trained networks as the unit of inference.  The
networks are nested: 8 Mod-Cog tasks x 3 weight-initialisation seeds, and three
networks trained on the same task are more alike than networks from different
tasks.  This script quantifies that clustering and re-tests every comparison at
the task level, so a reader can check that no conclusion depends on pooling
task-level and seed-level variance together.

What is fitted
--------------
The reviewer asked for *either* aggregation to n = 8 *or* a mixed-effects model with
task as a random effect.  Both are provided, and for this design they coincide.

``random_intercept_fit`` fits ``d_ij = mu + u_i + e_ij`` with task as a random
intercept.  The design is balanced -- three trained networks per task for every method
-- so the REML solution is closed form and is computed directly rather than by
iterative optimisation; no extra dependency is required.  Variance components
(``var_task``, ``var_residual``) and the resulting ICC are reported so the fit can be
inspected, not just its p-value.  For balanced data the fixed-effect test uses the
between-task mean square and therefore reduces exactly to a paired t on the task means;
the script asserts this agreement rather than assuming it.

``leave_one_task_out`` re-runs the task-level test dropping each task in turn, which
answers the reviewer's stated reason for wanting per-task curves -- whether a result is
carried by a handful of tasks -- rather than leaving the reader to eyeball a figure.

Why not the exact signed-rank test at the task level
----------------------------------------------------
Aggregating to 8 task means and re-running the exact two-sided Wilcoxon gives a
smallest attainable p of ``2 / 2^8 = 0.0078``.  Holm-corrected inside the
84-comparison family that is ``84 x 0.0078 = 0.656``, so *no* comparison can
reach alpha = 0.05 regardless of effect size: the test is quantised, not
uninformative about the data.  The confirmatory analysis therefore uses two
statistics whose p-values are not floored:

* a paired t-test on the 8 task means, and
* a cluster bootstrap that resamples whole tasks (keeping their 3 networks
  together) and inverts the percentile interval.

The exact task-level Wilcoxon p is still reported for transparency, flagged
against its floor.

Outputs
-------
``task_clustered_h512_comparison_tests.csv``
    Supplementary Table S4: per-comparison ICC, design effect, effective n,
    task-level t and cluster-bootstrap inference, Holm-corrected within the
    same 84-comparison family used by Table S2.
``per_task_retention_curves.csv``
    Supplementary figure source: retention by task x method x sparsity
    (mean, SD, SEM over the 3 trained networks per task).
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_artifacts/official_h512_24net/data"
REVISED = DATA / "revised_scope"
MAIN = REVISED / "task_preservation/revised_task_preservation_h512_24net_p50_80.csv"
OUT = REVISED / "significance"
TESTS_NAME = "task_clustered_h512_comparison_tests.csv"
CURVES_NAME = "per_task_retention_curves.csv"

ALPHA = 0.05
PCTS = (50, 60, 70, 80)
METRIC = "sequence_retention"
METHODS = (
    "Random",
    "Magnitude",
    "OBS compensated",
    "L-NP mask",
    "L-NP rescale",
    "S-NP mask",
    "S-NP rescale",
)
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260908


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment; identical to the released implementation."""
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        value = min(1.0, (m - rank) * float(p_value))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def network_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Average pruning-seed replicates within each trained-network cell."""
    return frame.groupby(
        ["method", "pruning_pct", "task_short", "source_network_seed"], as_index=False
    )[METRIC].mean()


def icc_one_way(values: np.ndarray, tasks: np.ndarray) -> tuple[float, float, float]:
    """One-way random-effects ICC of paired differences, plus design effect and effective n."""
    frame = pd.DataFrame({"d": values, "task": tasks})
    sizes = frame.groupby("task").size()
    n_tasks = int(sizes.size)
    n_total = int(sizes.sum())
    if n_tasks < 2 or sizes.min() < 2:
        return (float("nan"),) * 3
    k = float(sizes.mean())
    grand = frame["d"].mean()
    ms_between = (
        frame.groupby("task")["d"].mean().sub(grand).pow(2).mul(sizes).sum() / (n_tasks - 1)
    )
    ms_within = (
        frame.groupby("task")["d"].apply(lambda s: s.sub(s.mean()).pow(2).sum()).sum()
        / (n_total - n_tasks)
    )
    denom = ms_between + (k - 1.0) * ms_within
    icc = float((ms_between - ms_within) / denom) if denom > 0 else 0.0
    icc = max(icc, 0.0)
    design_effect = 1.0 + (k - 1.0) * icc
    return icc, float(design_effect), float(n_total / design_effect)


def random_intercept_fit(values: np.ndarray, tasks: np.ndarray) -> dict:
    """Fit ``d_ij = mu + u_i + e_ij`` with task as a random intercept.

    Returns the fixed-effect estimate with its standard error, t, df and p, plus the
    REML variance components and their ICC. Balanced data only: the closed-form
    solution below assumes an equal number of networks per task, which holds here
    (three seeds per task for every method). Raises if that is violated, rather than
    returning a silently wrong fit.
    """
    frame = pd.DataFrame({"d": values, "task": tasks})
    sizes = frame.groupby("task").size()
    if sizes.nunique() != 1:
        raise ValueError(
            f"random_intercept_fit requires a balanced design; group sizes were {sorted(sizes)}"
        )
    n_tasks = int(sizes.size)
    per_task = int(sizes.iloc[0])
    grand = float(frame["d"].mean())
    task_means = frame.groupby("task")["d"].mean()
    ms_between = per_task * float(((task_means - grand) ** 2).sum()) / (n_tasks - 1)
    ms_within = float(
        frame.groupby("task")["d"].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
    ) / (n_tasks * (per_task - 1))
    var_task = max((ms_between - ms_within) / per_task, 0.0)
    var_residual = ms_within
    total = var_task + var_residual
    standard_error = math.sqrt(ms_between / (n_tasks * per_task))
    t_stat = grand / standard_error if standard_error > 0 else math.nan
    df = n_tasks - 1
    p_value = float(2.0 * student_t.sf(abs(t_stat), df)) if standard_error > 0 else 1.0
    return {
        "mixed_mu": grand,
        "mixed_se": standard_error,
        "mixed_t": float(t_stat),
        "mixed_df": df,
        "mixed_p": p_value,
        "mixed_var_task": var_task,
        "mixed_var_residual": var_residual,
        "mixed_icc": float(var_task / total) if total > 0 else 0.0,
    }


def leave_one_task_out(values: np.ndarray, tasks: np.ndarray) -> dict:
    """Re-run the task-level test dropping each task in turn.

    Reports the worst (largest) p-value over the leave-one-out refits and whether the
    alpha-level verdict is unchanged throughout, i.e. whether the conclusion depends on
    any single task.
    """
    frame = pd.DataFrame({"d": values, "task": tasks})
    task_means = frame.groupby("task")["d"].mean()
    full_p = float(ttest_1samp(task_means.to_numpy(), 0.0).pvalue)
    worst_p, worst_task = full_p, ""
    for dropped in task_means.index:
        kept = task_means.drop(dropped)
        p_value = float(ttest_1samp(kept.to_numpy(), 0.0).pvalue)
        if p_value > worst_p:
            worst_p, worst_task = p_value, str(dropped)
    return {
        "loo_worst_p": worst_p,
        "loo_worst_dropped_task": worst_task,
        "loo_verdict_stable": bool((worst_p <= ALPHA) == (full_p <= ALPHA)),
    }


def cluster_bootstrap(values: np.ndarray, tasks: np.ndarray, rng: np.random.Generator):
    """Percentile CI and inverted p from resampling whole tasks with replacement."""
    groups = [g.to_numpy() for _, g in pd.DataFrame({"d": values, "t": tasks}).groupby("t")["d"]]
    n_tasks = len(groups)
    draws = rng.integers(0, n_tasks, size=(BOOTSTRAP_RESAMPLES, n_tasks))
    means = np.array([np.concatenate([groups[j] for j in row]).mean() for row in draws])
    lo, hi = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return float(lo), float(hi), float(max(p, 1.0 / BOOTSTRAP_RESAMPLES))


def build_tests(cells: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n8_floor = 2.0 / 2**8
    records = []
    for pct in PCTS:
        table = cells[cells.pruning_pct == pct].pivot_table(
            index=["task_short", "source_network_seed"], columns="method", values=METRIC
        )
        for method_a, method_b in itertools.combinations(METHODS, 2):
            if method_a not in table.columns or method_b not in table.columns:
                continue
            pair = table[[method_a, method_b]].dropna()
            diff = pair[method_a] - pair[method_b]
            tasks = np.array([t for t, _ in diff.index])
            icc, deff, eff_n = icc_one_way(diff.to_numpy(), tasks)
            task_means = diff.groupby(level=0).mean()
            t_res = ttest_1samp(task_means.to_numpy(), 0.0)
            ci = t_res.confidence_interval(confidence_level=1 - ALPHA)
            boot_lo, boot_hi, boot_p = cluster_bootstrap(diff.to_numpy(), tasks, rng)
            nonzero8 = task_means[task_means != 0]
            w8 = (
                float(wilcoxon(nonzero8, alternative="two-sided", zero_method="wilcox",
                               method="auto").pvalue)
                if len(nonzero8) else 1.0
            )
            mixed = random_intercept_fit(diff.to_numpy(), tasks)
            loo = leave_one_task_out(diff.to_numpy(), tasks)
            if not math.isclose(mixed["mixed_t"], float(t_res.statistic), rel_tol=1e-9,
                                abs_tol=1e-12):
                raise AssertionError(
                    "random-intercept fixed effect disagrees with the paired t on task means: "
                    f"{mixed['mixed_t']} vs {t_res.statistic}"
                )
            records.append({
                "dataset": "main_h512",
                "family": "task_clustered_sequence_retention_pairwise_by_sparsity",
                "metric": METRIC,
                "unit": "task_mean_of_trained_network_cells",
                "pruning_pct": pct,
                "method_a": method_a,
                "method_b": method_b,
                "n_tasks": int(task_means.size),
                "n_networks": int(diff.size),
                "icc_paired_differences": icc,
                "design_effect": deff,
                "effective_n": eff_n,
                "mean_diff_network_level": float(diff.mean()),
                "mean_diff_task_level": float(task_means.mean()),
                "sd_diff_task_level": float(task_means.std(ddof=1)),
                "tasks_favouring_a": int((task_means > 0).sum()),
                "tasks_favouring_b": int((task_means < 0).sum()),
                "primary_test_name": "paired t-test on task means",
                "t_statistic": float(t_res.statistic),
                "t_df": int(task_means.size - 1),
                "t_p": float(t_res.pvalue),
                "t_ci_lower": float(ci.low),
                "t_ci_upper": float(ci.high),
                "cluster_bootstrap_name": "task-level cluster bootstrap (percentile)",
                "cluster_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "cluster_bootstrap_ci_lower": boot_lo,
                "cluster_bootstrap_ci_upper": boot_hi,
                "cluster_bootstrap_p": boot_p,
                "wilcoxon_task_level_p": w8,
                "wilcoxon_task_level_floor": n8_floor,
                "wilcoxon_task_level_at_floor": bool(abs(w8 - n8_floor) < 1e-12),
                "alpha": ALPHA,
                "multiple_comparison_method": "Holm correction within family",
                **mixed,
                **loo,
            })
    out = pd.DataFrame.from_records(records)
    out["family_size"] = len(out)
    out["holm_p_within_family"] = holm_adjust(out["t_p"].tolist())
    out["cluster_bootstrap_holm_p"] = holm_adjust(out["cluster_bootstrap_p"].tolist())
    out["reject_holm_alpha_0_05"] = out["holm_p_within_family"] <= ALPHA
    out["cluster_bootstrap_reject_holm_alpha_0_05"] = out["cluster_bootstrap_holm_p"] <= ALPHA
    return out


def build_curves(cells: pd.DataFrame) -> pd.DataFrame:
    grouped = cells.groupby(["task_short", "method", "pruning_pct"])[METRIC]
    out = grouped.agg(
        n_networks="count", retention_mean="mean", retention_sd=lambda s: s.std(ddof=1)
    ).reset_index()
    out["retention_sem"] = out["retention_sd"] / np.sqrt(out["n_networks"])
    return out.sort_values(["task_short", "method", "pruning_pct"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--main-csv", type=Path, default=MAIN)
    args = parser.parse_args()

    frame = pd.read_csv(args.main_csv, low_memory=False)
    frame = frame[frame.method.isin(METHODS)]
    cells = network_cells(frame)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    tests = build_tests(cells, rng)
    curves = build_curves(cells)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tests_out = args.out_dir / TESTS_NAME
    curves_out = args.out_dir / CURVES_NAME
    tests.sort_values(["pruning_pct", "method_a", "method_b"]).to_csv(tests_out, index=False)
    curves.to_csv(curves_out, index=False)

    print(f"wrote {tests_out}  ({len(tests)} comparisons)")
    print(f"wrote {curves_out}  ({len(curves)} rows: "
          f"{curves.task_short.nunique()} tasks x {curves.method.nunique()} methods x "
          f"{curves.pruning_pct.nunique()} sparsities)")
    print(f"\nICC of paired differences by sparsity (median across comparisons):")
    for pct in PCTS:
        sub = tests[tests.pruning_pct == pct]
        print(f"  {pct}%  ICC {sub.icc_paired_differences.median():.3f}   "
              f"design effect {sub.design_effect.median():.2f}   "
              f"effective n {sub.effective_n.median():.1f}")
    n_floor = int(tests.wilcoxon_task_level_at_floor.sum())
    print(f"\nexact task-level Wilcoxon sitting at its {2/2**8:.4f} floor: "
          f"{n_floor}/{len(tests)} comparisons (why the t / bootstrap are primary)")
    print(f"Holm-significant at task level: {int(tests.reject_holm_alpha_0_05.sum())}/{len(tests)} "
          f"(t), {int(tests.cluster_bootstrap_reject_holm_alpha_0_05.sum())}/{len(tests)} (bootstrap)")


if __name__ == "__main__":
    main()
