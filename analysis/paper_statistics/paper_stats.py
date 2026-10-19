#!/usr/bin/env python3
"""The paper's statistical procedure. Finalised 2026-10-09; this module is the
single implementation, and every reported interval, effect size and p-value in
the manuscript comes from here.

Run the tests with
    .venv/bin/python analysis/paper_statistics/test_paper_stats.py
and the tables with
    .venv/bin/python analysis/paper_statistics/run_paper_stats.py


THE DESIGN
----------
8 tasks x 3 network seeds = 24 trained networks, fully balanced. The unpruned
accuracy P_unpruned[t, s] is one number per network, shared by every method and
every sparsity, so retention

    R[t, s, m, q] = P_pruned[t, s, m, q] / P_unpruned[t, s]

is comparable across methods within a network. Pruning draws (the repeated
stochastic masks some methods have and others do not) are AVERAGED AWAY FIRST,
before anything else happens, so every arm enters the procedure with exactly one
number per (task, seed, sparsity). That is what makes the arms commensurable:
were the draws left in, the sampling arms would carry a noise source the
deterministic arms do not have.

Reduction, in order:
    R[t, s, m, q]  --mean over the 3 seeds-->  A[t, m, q]   (8 tasks)
                   --mean over the 8 tasks-->    [m, q]     (the plotted curve)


THE HIERARCHICAL BOOTSTRAP
--------------------------
One surrogate dataset is built in two stages:
    1. draw 8 task positions from the 8 tasks, with replacement;
    2. for each drawn position independently, draw 3 seeds from that task's 3
       seeds, with replacement.
Repeated tasks get independently redrawn seeds, which is the standard two-stage
cluster bootstrap. 10,000 surrogates are built.

The resample is drawn ONCE and shared by every method, every sparsity and every
contrast. If surrogate 7 picked tasks (2,2,4,5,7,7,8,1), then all eight methods
at all eight sparsities are evaluated on those same tasks and seeds. This is
what keeps the paired structure intact: a contrast on a surrogate is a
difference between two arms measured on the *same* resampled networks, so
between-task variance cancels from the contrast exactly as it does in the data.

Confidence intervals are percentiles of the surrogate distribution.


THE SUMMARY MEASURE
-------------------
Normalised area under the retention curve over the full 10-80% range,

    AUC[t, m] = (1 / (80 - 10)) * trapezoid( A[t, m, .] , q )

so 1.0 is perfect retention across the whole range. Computed per task and
method, then averaged over tasks.


SIGNIFICANCE
------------
Sign-flip (randomisation) test on the 8 task-level differences D_t. With 8
tasks there are only 2^8 = 256 sign assignments, so the null distribution is
ENUMERATED EXHAUSTIVELY rather than sampled: the p-value is exact and carries no
Monte-Carlo error. The two-sided p-value is the fraction of sign assignments
whose mean is at least as extreme in absolute value as the observed mean.

This makes the smallest attainable two-sided p-value 2/256 = 0.0078, reached
when all 8 differences share a sign. `P_FLOOR` records it. A p-value AT the
floor means the data are as extreme as this test can register, not that the
effect is weak; the number of tasks favouring the method is reported alongside
for exactly this reason, since 8/8 is strong evidence on its own.

Because the attainable p-values are multiples of 2/256, a nominal alpha of 0.05
is really a test at 12/256 = 0.0469. The test is therefore mildly conservative,
which is the direction that costs us nothing to admit.


WHAT IS DELIBERATELY NOT HERE
-----------------------------
No t-intervals, no Wilcoxon, no normal approximation. The procedure above is the
one the authors settled on, and nothing in this module silently substitutes a
different estimator or picks between two of them post hoc.
"""
from __future__ import annotations

import pathlib
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# the deposit root, resolved from this file so a clone runs anywhere
REPO = str(pathlib.Path(__file__).resolve().parents[2])

N_BOOT = 10_000
SEED = 20261009
GRID = (10, 20, 30, 40, 50, 60, 70, 80)
OFFDIAG = 512 * 511          # off-diagonal recurrent entries at H = 512

# The constant amplification cap, recalibrated from the corrected 1/p
# distribution as the mean q50 over 50-80% -- the same construction that
# produced the published 3.8 from the old distribution, which measured 3.823.
CAP_CONSTANT = 3.695
# The magnitude score's own constant, derived by the IDENTICAL construction from
# its own cap-percentile sweep: pooled optimum q60 -> 4.549, against the noise
# score's q50 -> 3.695. Holding one number fixed for both scores tunes one side
# only; holding the RULE fixed and letting each score set its own constant is
# the matched comparison, and it is the one the paper reports. The magnitude
# score wants a looser cap because it removes the small weights, which costs
# more L1 mass than spectral gain and needs more amplification headroom.
MAG_CAP_CONSTANT = 4.549
# what the capped arms were run at BEFORE the recalibration; the legacy cube
# has to ask for that value or it finds nothing
LEGACY_CAP_CONSTANT = 3.8
AUC_LO, AUC_HI = 10, 80

trapezoid = getattr(np, "trapezoid", None) or np.trapz

# The arms, by the names the paper uses. `cap` and `gain` pin the rescaling
# variant where the strategy name alone is ambiguous.
ARMS = {
    # cap_value pins the constant-factor arm the paper reports; see select()
    "noise-prune + capped rescale":  dict(strategy="simulation_noise_prune_capped_rescale",
                                          cap="fixed", cap_value=CAP_CONSTANT),
    "noise-prune + rescale":         dict(strategy="simulation_noise_prune_rescale"),
    "noise-prune score only":        dict(strategy="simulation_noise_prune_mask_only"),
    "magnitude + capped rescale":    dict(strategy="simulation_noise_prune_capped_magnitude_rescale",
                                          cap="fixed", cap_value=MAG_CAP_CONSTANT),
    "magnitude + rescale":           dict(strategy="simulation_noise_prune_magnitude_rescale"),
    "magnitude":                     dict(strategy="l1_unstructured"),
    "OBS":                           dict(strategy="obs_compensated"),
    "random":                        dict(strategy="random_unstructured"),
}

NOISE_PRUNE_ARMS = ("noise-prune + capped rescale", "noise-prune + rescale")

# The arms that draw a Bernoulli mask, and so were re-run. Confirmed by walking
# the strategy registry, not by eye.
SAMPLING_ARMS = frozenset({
    "noise-prune + capped rescale",
    "noise-prune + rescale",
    "magnitude + capped rescale",
    "magnitude + rescale",
    # named as key_comparisons declares them; both were re-run at 10-80%,
    # having previously existed only at 50-80%
    "noise-prune shuffled-p",
    "noise-prune uniform-p",
})


# --------------------------------------------------------------------------
# the data cube
# --------------------------------------------------------------------------
@dataclass
class Cube:
    """R[task, seed, method, sparsity], pruning draws already averaged out."""
    R: np.ndarray
    tasks: list
    seeds: list
    methods: list
    qs: np.ndarray
    draws: np.ndarray          # how many pruning draws each cell averaged

    def m(self, name: str) -> int:
        return self.methods.index(name)

    def task_level(self) -> np.ndarray:
        """A[task, method, sparsity] -- the 3 seeds averaged."""
        return self.R.mean(axis=1)


# WHICH DATA EACH ARM COMES FROM
#
# The six arms that sample were re-run under the corrected probability
# normalisation (suite `renorm_p10_80`, 2026-10-10). The legacy directories
# still hold their OLD cells, computed when the sampling rule did not reach the
# requested sparsity, so a flat load over everything would mix two
# normalisations inside one arm. Each arm therefore declares its pool:
#
#   "renorm"  the six sampling arms, from the re-run suite only
#   "legacy"  magnitude, OBS, random, the score-only mask -- deterministic,
#             never affected, and not re-run
#
# SOURCE_SET switches the sampling arms between the two pools. It stays
# "legacy" until the re-run finishes; flipping it to "renorm" is the whole of
# the changeover. Keeping both reachable means the before/after comparison is
# one constant apart, which is what the revision needs to report.
SOURCE_SET = "renorm"         # the re-run suite is complete as of 2026-10-10

RENORM_SOURCES = [
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_renorm_p10_80/*.csv",
]

# The constant amplification cap was recalibrated from the corrected 1/p
# distribution: the mean q50 over 50-80%, measured the same way the published
# 3.8 was, came out at 3.695. The two cap-dependent arms were therefore re-run
# at the new constant and come from here instead. The other four sampling arms
# do not depend on the cap and would reproduce byte for byte, so they were not
# re-run and still come from RENORM_SOURCES.
RECAP_SOURCES = [
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_renorm_recap_p10_80/*.csv",
]
# the magnitude arm at ITS optimum lives in its own suite, run at 4.549
MAGOPT_SOURCES = [
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_renorm_magopt_p10_80/*.csv",
]
# split by which constant each arm is entitled to, because they no longer share
# one: the noise arm comes from the recap pool at 3.695, the magnitude arm from
# the magopt pool at 4.549
NOISE_CAPPED_ARMS = frozenset({"noise-prune + capped rescale"})
MAG_CAPPED_ARMS = frozenset({"magnitude + capped rescale"})
CAPPED_ARMS = NOISE_CAPPED_ARMS | MAG_CAPPED_ARMS

# Every H=512 source the analysis draws on, named explicitly. The results tree
# holds 18 tasks and many exploratory suites; globbing it would silently pull in
# arms the paper does not report.
SOURCES = [
    "paper_artifacts/official_h512_24net/data/revised_scope/task_preservation/*.csv",
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_low_sparsity_p10_40/*.csv",
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_fixedcapfull_p50_80/*.csv",
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_covariance_ablation_full_p50_80/*.csv",
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_fig3_lowsparsity_p10_40/*.csv",
    "results/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_full_p50_80/*.csv",
]

TASKS = ("ctxdlydm2intseq", "ctxdlydm1intseq", "dlydm1intseq", "dlydm2intseq",
         "multidlydmintseq", "dm1seqr", "dm2seql", "dmsintseq")

NEED = {"strategy", "amount", "post_acc_sequence", "pre_acc_sequence", "task",
        "source_network_seed", "post_rec_weight_nz_count"}
OPTIONAL = ("prune_rescale_cap_mode", "prune_gain_restore_mode", "pruning_seed",
            "prune_rescale_cap_value")

# De-duplication. The low-sparsity suites carry 50-80% regression anchors that
# re-state the frozen artifact, and one aggregate file re-states its own
# per-task files, so the same run can appear more than once and must be counted
# once. The key is PROVENANCE PLUS VALUE: a row is a duplicate only if it
# matches another in which run produced it *and* in what that run measured.
#
# Two near-misses this key gets right and a value-only key does not:
#   * the capped arms at 80% hold two measurements under one pruning seed -- a
#     genuine re-run that differs in accuracy. Keyed on the value they survive
#     as the separate draws they are; keyed on provenance alone one would be
#     thrown away.
#   * several draws of random and of the score-only arm land on exactly 0.0
#     accuracy. Keyed on the value alone they collapse into one, which would
#     pull the cell mean up off the floor. Keyed with the pruning seed they stay
#     distinct.
# Switching from the value-only key the figure loader uses moves one cell of
# magnitude + rescale by 6.4e-4 and no other cell of any reported arm by more
# than 1e-6, so no reported number depends on the choice. It is made correctly
# here regardless.
DEDUP_KEY = ["strategy", "amount", "task", "source_network_seed", "pruning_seed",
             "_cap", "_gain", "post_acc_sequence"]


def load_runs(grid=GRID, sources=None) -> pd.DataFrame:
    """One row per pruning run, duplicates removed, 8 paper tasks only."""
    import glob

    frames = []
    for pat in (SOURCES if sources is None else sources):
        for f in sorted(glob.glob(f"{REPO}/{pat}")):
            try:
                cols = set(pd.read_csv(f, nrows=0).columns)
            except Exception:
                continue
            if not NEED <= cols:
                continue
            use = list(NEED | {c for c in OPTIONAL if c in cols})
            frames.append(pd.read_csv(f, usecols=use, low_memory=False))
    if not frames:
        raise RuntimeError("no source CSVs matched; is the working directory the repo root?")
    d = pd.concat(frames, ignore_index=True)

    d["task_short"] = d.task.str.replace("modcog:", "", regex=False)
    d = d[d.task_short.isin(TASKS)].copy()
    d["pct"] = (d.amount * 100).round().astype(int)
    d = d[d.pct.isin(grid)].copy()
    for c in ("prune_rescale_cap_mode", "prune_gain_restore_mode"):
        if c not in d:
            d[c] = ""
        d[c] = d[c].fillna("")
    # the frozen artifact omits these columns entirely while the results tree
    # writes "none"; both mean "not capped" / "no gain restore"
    d["_cap"] = d.prune_rescale_cap_mode.replace({"none": ""})
    d["_gain"] = d.prune_gain_restore_mode.replace({"none": ""})
    if "pruning_seed" not in d:
        raise RuntimeError("pruning_seed is required to de-duplicate correctly")
    if d.pruning_seed.isna().any():
        raise RuntimeError(f"{int(d.pruning_seed.isna().sum())} rows lack a pruning_seed")
    d = d.drop_duplicates(subset=DEDUP_KEY)
    d["ret"] = d.post_acc_sequence / d.pre_acc_sequence
    # realized density, for the diagnostic that the requested sparsity is not
    # what the sampling arms actually remove
    d["removed"] = 100 * (1 - d.post_rec_weight_nz_count / OFFDIAG)
    return d


def select(d: pd.DataFrame, strategy: str, cap: str | None = None,
           gain: str | None = None, cap_value: float | None = None) -> pd.DataFrame:
    """Rows for one arm.

    `cap_value` matters and must not be omitted for a capped arm. The
    fixedcapfull suite carries TWO capped arms at 80% sparsity -- the constant
    factor 3.8 the paper reports, and a per-sparsity constant of 5.922 -- and
    both record cap_mode "fixed". Selecting on the mode alone silently averages
    two different methods into one curve at that one sparsity.
    """
    s = d[d.strategy == strategy]
    if cap is not None:
        s = s[s.prune_rescale_cap_mode == cap]
    if gain is not None:
        s = s[s.prune_gain_restore_mode == gain]
    if cap_value is not None:
        s = s[np.isclose(s.prune_rescale_cap_value.astype(float), cap_value)]
    return s


def build_cube(arms: dict | None = None, grid=GRID, source_set: str | None = None) -> Cube:
    """Assemble the cube from the pinned H=512 sources, and prove it is complete.

    Sampling arms are drawn from the re-run suite when `source_set` is
    "renorm", and from the legacy directories otherwise; deterministic arms
    always come from the legacy directories, which is the only place they
    exist. See SOURCE_SET.
    """
    arms = arms or ARMS
    source_set = SOURCE_SET if source_set is None else source_set
    if source_set not in {"legacy", "renorm"}:
        raise ValueError(f"source_set must be 'legacy' or 'renorm', got {source_set!r}")
    d = load_runs(grid)
    d_renorm = load_runs(grid, RENORM_SOURCES) if source_set == "renorm" else None
    d_recap = load_runs(grid, RECAP_SOURCES) if source_set == "renorm" else None
    d_magopt = load_runs(grid, MAGOPT_SOURCES) if source_set == "renorm" else None

    cells = {}
    for name, spec in arms.items():
        if source_set == "renorm" and name in MAG_CAPPED_ARMS:
            pool = d_magopt
        elif source_set == "renorm" and name in NOISE_CAPPED_ARMS:
            pool = d_recap
        elif source_set == "renorm" and name in SAMPLING_ARMS:
            pool = d_renorm
        else:
            pool = d
        if pool is d and name in CAPPED_ARMS:
            spec = {**spec, "cap_value": LEGACY_CAP_CONSTANT}
        s = select(pool, **spec)
        if s.empty:
            raise ValueError(
                f"arm {name!r} selected no rows from the "
                f"{'magopt' if pool is d_magopt else 'recap' if pool is d_recap else 'renorm' if pool is d_renorm else 'legacy'} pool")
        g = s.groupby(["task_short", "source_network_seed", "pct"])
        cells[name] = pd.DataFrame({"ret": g.ret.mean(), "n": g.size()})

    tasks = sorted({t for c in cells.values() for t, _, _ in c.index})
    seeds = sorted({s for c in cells.values() for _, s, _ in c.index})
    methods = list(arms)
    qs = np.array(sorted(grid))

    R = np.full((len(tasks), len(seeds), len(methods), len(qs)), np.nan)
    D = np.zeros_like(R)
    ti = {t: i for i, t in enumerate(tasks)}
    si = {s: i for i, s in enumerate(seeds)}
    qi = {q: i for i, q in enumerate(qs)}
    for mi, name in enumerate(methods):
        for (t, s, q), row in cells[name].iterrows():
            R[ti[t], si[s], mi, qi[q]] = row.ret
            D[ti[t], si[s], mi, qi[q]] = row.n

    missing = np.argwhere(np.isnan(R))
    if len(missing):
        human = [f"{tasks[a]}/seed{seeds[b]}/{methods[c]}/{qs[e]}%" for a, b, c, e in missing[:10]]
        raise ValueError(f"cube has {len(missing)} empty cells, e.g. {human}")
    if R.shape[:2] != (8, 3):
        raise ValueError(f"expected 8 tasks x 3 seeds, got {R.shape[:2]}")
    return Cube(R=R, tasks=tasks, seeds=seeds, methods=methods, qs=qs, draws=D)


# --------------------------------------------------------------------------
# the shared surrogate datasets
# --------------------------------------------------------------------------
@dataclass
class Surrogates:
    """One set of hierarchical resamples, shared by every metric.

    task_idx[b, p]      which task sits at position p of surrogate b
    seed_idx[b, p, k]   which of that task's seeds is its k-th draw
    A[b, p, m, q]       the surrogate's task-level retention
    """
    task_idx: np.ndarray
    seed_idx: np.ndarray
    A: np.ndarray

    @property
    def n_boot(self) -> int:
        return self.A.shape[0]


def make_surrogates(cube: Cube, n_boot: int = N_BOOT, seed: int = SEED,
                    chunk: int = 1000) -> Surrogates:
    """Draw the resamples and evaluate the task-level cube on each.

    Stage 1 draws task positions; stage 2 draws seeds *within each position*,
    so a task picked twice gets two independent seed draws.
    """
    rng = np.random.default_rng(seed)
    nt, ns = cube.R.shape[0], cube.R.shape[1]
    task_idx = rng.integers(0, nt, size=(n_boot, nt))
    seed_idx = rng.integers(0, ns, size=(n_boot, nt, ns))

    A = np.empty((n_boot, nt, len(cube.methods), len(cube.qs)))
    for lo in range(0, n_boot, chunk):
        hi = min(lo + chunk, n_boot)
        t = task_idx[lo:hi, :, None]               # (c, nt, 1)
        s = seed_idx[lo:hi]                        # (c, nt, ns)
        # gather R[t, s] -> (c, nt, ns, m, q), then average the drawn seeds
        A[lo:hi] = cube.R[t, s].mean(axis=2)
    return Surrogates(task_idx=task_idx, seed_idx=seed_idx, A=A)


# --------------------------------------------------------------------------
# metrics -- each takes a task-level array and nothing else, so the observed
# data and a surrogate go through identical code
# --------------------------------------------------------------------------
def mean_curve(A: np.ndarray) -> np.ndarray:
    """A[..., task, method, q] -> [..., method, q]. Mean over tasks."""
    return A.mean(axis=-3)


def auc(A: np.ndarray, qs: np.ndarray, lo: int = AUC_LO, hi: int = AUC_HI) -> np.ndarray:
    """Normalised trapezoid AUC. A[..., task, method, q] -> [..., task, method]."""
    keep = (qs >= lo) & (qs <= hi)
    x = qs[keep].astype(float)
    if x[0] != lo or x[-1] != hi:
        raise ValueError(f"grid {qs} does not span [{lo}, {hi}]")
    return trapezoid(A[..., keep], x, axis=-1) / (hi - lo)


def percentile_ci(draws: np.ndarray, level: float = 95, axis: int = 0):
    a = (100 - level) / 2
    return np.percentile(draws, [a, 100 - a], axis=axis)


# --------------------------------------------------------------------------
# the sign-flip test, enumerated exactly
# --------------------------------------------------------------------------
P_FLOOR = {n: 2.0 / 2 ** n for n in range(2, 21)}


def signflip_exact(D: np.ndarray) -> dict:
    """Exact two-sided sign-flip test on the task-level differences D.

    Enumerates all 2^n sign assignments. `p_two` is the fraction whose mean is
    at least as far from zero as the observed mean; `p_one` the fraction at
    least as large in the observed direction.
    """
    D = np.asarray(D, dtype=float).ravel()
    n = D.size
    if n == 0:
        raise ValueError("no differences to test")
    if n > 20:
        raise ValueError(f"{n} tasks would need 2^{n} enumerations; use a sampled test")
    signs = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    null = (signs * D).mean(axis=1)
    obs = D.mean()
    tol = 1e-12 * max(1.0, abs(obs))
    return dict(
        effect=obs,
        n_tasks=n,
        p_two=float((np.abs(null) >= abs(obs) - tol).mean()),
        p_one=float((null >= obs - tol).mean()),
        p_floor=2.0 / 2 ** n,
        n_positive=int((D > 0).sum()),
        n_negative=int((D < 0).sum()),
    )


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, monotone and capped at 1."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------
# the two reported comparisons
# --------------------------------------------------------------------------
def contrast(cube: Cube, sur: Surrogates, a: str, b: str, q=None,
             levels=(95, 68), lo: int = AUC_LO, hi: int = AUC_HI) -> dict:
    """Arm `a` minus arm `b`. `q=None` compares the AUC summary, otherwise the
    retention at that one sparsity.

    Effect size and significance come from the 8 observed task-level
    differences; the interval comes from the shared surrogates, where the same
    difference is recomputed on each resampled dataset.
    """
    ia, ib = cube.m(a), cube.m(b)
    obs = cube.task_level()
    if q is None:
        va, vb = auc(obs, cube.qs, lo, hi)[:, ia], auc(obs, cube.qs, lo, hi)[:, ib]
        sa = auc(sur.A, cube.qs, lo, hi)[:, :, ia]
        sb = auc(sur.A, cube.qs, lo, hi)[:, :, ib]
        measure = f"AUC {lo}-{hi}%"
    else:
        k = int(np.where(cube.qs == q)[0][0])
        va, vb = obs[:, ia, k], obs[:, ib, k]
        sa, sb = sur.A[:, :, ia, k], sur.A[:, :, ib, k]
        measure = f"{q}% sparsity"

    D = va - vb
    out = dict(arm_a=a, arm_b=b, measure=measure, sparsity=q)
    out.update(signflip_exact(D))
    boots = (sa - sb).mean(axis=1)
    out["boot_sd"] = float(boots.std(ddof=1))
    for L in levels:
        lo, hi = percentile_ci(boots, L)
        out[f"lo{L}"] = float(lo)
        out[f"hi{L}"] = float(hi)
    out["task_diffs"] = D
    return out


def curve_table(cube: Cube, sur: Surrogates, levels=(95, 68)) -> pd.DataFrame:
    """The plotted curves with hierarchical-bootstrap intervals."""
    point = mean_curve(cube.task_level())                 # (m, q)
    draws = mean_curve(sur.A)                             # (b, m, q)
    rows = []
    for mi, name in enumerate(cube.methods):
        for qi, q in enumerate(cube.qs):
            r = dict(method=name, sparsity=int(q), mean=point[mi, qi],
                     boot_sd=draws[:, mi, qi].std(ddof=1))
            for L in levels:
                lo, hi = percentile_ci(draws[:, mi, qi], L)
                r[f"lo{L}"], r[f"hi{L}"] = lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


def auc_table(cube: Cube, sur: Surrogates, levels=(95, 68)) -> pd.DataFrame:
    """The summary measure per method, with intervals."""
    per_task = auc(cube.task_level(), cube.qs)            # (task, m)
    draws = auc(sur.A, cube.qs).mean(axis=1)              # (b, m)
    rows = []
    for mi, name in enumerate(cube.methods):
        r = dict(method=name, AUC=per_task[:, mi].mean(),
                 sd_across_tasks=per_task[:, mi].std(ddof=1),
                 boot_sd=draws[:, mi].std(ddof=1), n_tasks=per_task.shape[0])
        for L in levels:
            lo, hi = percentile_ci(draws[:, mi], L)
            r[f"lo{L}"], r[f"hi{L}"] = lo, hi
        rows.append(r)
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


def task_curve_table(cube: Cube) -> pd.DataFrame:
    """Per-task, per-method retention -- what the SI task-by-task panels plot."""
    A = cube.task_level()
    rows = [dict(task=cube.tasks[t], method=cube.methods[m], sparsity=int(cube.qs[k]),
                 retention=A[t, m, k])
            for t in range(A.shape[0]) for m in range(A.shape[1]) for k in range(A.shape[2])]
    return pd.DataFrame(rows)
