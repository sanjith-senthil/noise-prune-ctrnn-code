#!/usr/bin/env python3
"""Stage the revision artifacts into the data release and regenerate its manifests.

The peer-review revision produced new result tables, statistics and checkpoints. They
belong in the data deposit alongside the original artifacts, and the deposit's two
checksum manifests have to be rebuilt so every file it ships is covered.

What is deposited
-----------------
Per suite, the **merged** raw rows and the summary tables, not the eight per-task shards
the runner writes.  The shards are intermediates: they carry ~290 raw columns each and
total ~59 MB, whereas the merged tables carry the analysis columns and total ~1.2 MB.
This matches how the original deposit already treats the main benchmark, which ships one
merged raw CSV rather than its per-run sources.

For dm1/anti, only the selected final (12,000-step) checkpoints are deposited, not the
6,000-step phase-1 intermediates -- again matching the original deposit, which ships the
24 selected checkpoints only.

Manifest regeneration
---------------------
Both ``MANIFEST.sha256`` and ``manifest.json`` are rebuilt over every file in the deposit
apart from VCS metadata, the manifests themselves, and macOS ``.DS_Store`` droppings.
Note this *widens* coverage: the previous manifests carried 70 records and omitted
``results/weight_distributions/`` entirely, so those seven files shipped unchecksummed.

The script only ever adds or overwrites; it never deletes from the deposit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DEFAULT_DEPOSIT = PROJECT / "data_release_staging"
RESULTS = PROJECT / "results"
ANALYSIS = PROJECT / "analysis/revision_2026"

# label -> (result stem, config directory under release_code_staging/configs)
#
# The config directory is named explicitly rather than derived from the stem.
# Most suite generators put their per-task configs in ``configs/<stem>/``, but
# the two score-control suites use short directory names instead, and deriving
# the path silently found nothing for them -- which is why the original deposit
# shipped the round-1 score-control results with no configs alongside them.
SUITE_STEMS = {
    "score_controls": (
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_full_p50_80",
        "score_control_full",
    ),
    "low_sparsity": (
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_low_sparsity_p10_40",
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_low_sparsity_p10_40",
    ),
    "noise_necessity": (
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_necessity_p50_80",
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_necessity_p50_80",
    ),
    "score_controls_round2": (
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_round2_p50_80",
        "score_control_round2",
    ),
    "covariance_ablation": (
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_covariance_ablation_full_p50_80",
        "covariance_ablation",
    ),
    "noise_only": (
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_only_p50_80",
        "noise_only",
    ),
    # already ships its own merged _raw and summaries, so it goes through the
    # ordinary copy path rather than being re-merged from shards
    "size_h1024_p50_80": (
        "task_preservation_tanh_h1024_modcog_revised8_12k_seqbest_revised_task_only_p50_80",
        "task_preservation_tanh_h1024_modcog_revised8_12k_seqbest_revised_task_only_p50_80",
    ),
}
# The revision reports results at hidden size 1024 as well as 512, so the
# deposit has to ship those trained networks too. Without them the size
# comparison is not reproducible at all: the H=1024 suites load these 24
# checkpoints and nothing else in either deposit provides them. 101 MB.
H1024_CHECKPOINT_DIRS = tuple(
    f"tanh_h1024_modcog_revised8_to12k_lr0006_seqbest_no_l2_seed{seed}" for seed in (0, 1, 2)
)
DM1ANTI_CHECKPOINT_DIRS = tuple(
    f"tanh_h512_modcog_dm1anti_to12k_lr0006_seqbest_no_l2_seed{seed}" for seed in (0, 1, 2)
)
DM1ANTI_RESULT_GLOBS = (
    "train_tanh_h512_modcog_dm1anti_6k_seqbest_no_l2_seed*.csv",
    "continue_tanh_h512_modcog_dm1anti_to12k_lr0006_seqbest_no_l2_seed*.csv",
    "dm1anti_baseline_summary.csv",
)
MANIFEST_EXCLUDE_NAMES = {"MANIFEST.sha256", "manifest.json", ".DS_Store"}
MANIFEST_EXCLUDE_DIRS = {".git"}

# ---------------------------------------------------------------------------
# THE RENORMALISATION RE-RUN (2026-10-10)
#
# The sampling rule's probability normalisation was corrected: it scaled scores
# by K = M/sum(q) and then clipped at 1, discarding the mass above 1, so it
# retained fewer edges than asked for -- at a requested 10% sparsity it removed
# 33%. The six strategies that draw a Bernoulli mask were re-run over 10-80% at
# both network sizes, and those re-runs are what every figure and every
# statistic in the paper now reads.
#
# The deposit therefore has to do two things it has never done before: ship the
# re-run, and stop shipping the rows it supersedes. The second is why this
# script grew a --prune mode; until now it only ever added.
# ---------------------------------------------------------------------------
SAMPLING_STRATEGIES = (
    "simulation_noise_prune_rescale",
    "simulation_noise_prune_capped_rescale",
    "simulation_noise_prune_magnitude_rescale",
    "simulation_noise_prune_capped_magnitude_rescale",
    "simulation_noise_prune_shuffled_rescale",
    "simulation_noise_prune_uniform_rescale",
)

# label -> result stem. Shipped MERGED, one table per suite, matching how the
# rest of the deposit treats raw rows.
RENORM_SUITES = {
    "h512_p10_80": "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_renorm_p10_80",
    # the two cap-dependent arms, re-run at the recalibrated constant 3.695;
    # these supersede the capped rows in h512_p10_80, which used 3.8
    "h512_recap_p10_80": "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_renorm_recap_p10_80",
    "h1024_p10_80": "task_preservation_tanh_h1024_modcog_revised8_12k_seqbest_renorm_p10_80",
    "h1024_rest_p10_80": "task_preservation_tanh_h1024_modcog_revised8_12k_seqbest_renorm_rest_p10_80",
    "capq50_h512_p10_80": "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_capq50_p10_80",
    # the cap-percentile sweep, re-run on the corrected rule; this is what
    # Figure 5 panel a plots and what establishes that the optimum is interior
    # and sits at the 50th percentile, which is where the constant 3.695 comes
    # from
    "capsweep_h512_p50_80": "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_capsweep_p50_80",

    # --- each score capped at its OWN constant -----------------------------
    # The same sweep applied to the magnitude score: its pooled optimum is q60,
    # a decile looser than the noise score's q50, giving 4.549 against 3.695.
    "capsweep_mag_h512_p50_80":
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_capsweep_mag_p50_80",
    # The same sweep at H = 1024, which asks whether the optimal percentile is a
    # property of the rule or of the size. It is of the size: the optimum moves
    # from q50 to q60 and the constant it implies from 3.695 to 4.821, with the
    # cap value scaling by 1.29-1.31 at every sparsity. That is why the constant
    # does not transfer -- at 80% sparsity H = 1024 wants 7.74 and the paper
    # applies 3.695 -- and why capping reverses there at the larger size.
    "capsweep_h1024_p50_80":
        "task_preservation_tanh_h1024_modcog_revised8_12k_seqbest_capsweep_p50_80",
    # The capped arms re-run at 4.549. This is the magnitude arm the paper
    # reports, and the noise arm at 4.549 completes the {score} x {cap} 2x2.
    "magopt_h512_p10_80":
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_renorm_magopt_p10_80",
    # The same arms at 3.584, the magnitude score's q50. Superseded as the
    # reported cap -- the sweep put magnitude's optimum at q60 -- but kept
    # because it is the third point that establishes how little the constant
    # matters within a score.
    #
    # IT MUST STAY ITS OWN TABLE. Its run_ids are identical to magopt's, because
    # the generator builds run_id from (task, netseed, arm, sparsity, draw) and
    # the cap is not in it. All 1,152 collide if the two are merged, which is
    # exactly the duplication this deposit was cleaned of once before.
    "magcap_h512_p10_80":
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_renorm_magcap_p10_80",

    # --- controls ----------------------------------------------------------
    # The noise score with a global factor, so Figure 3's sampling contrast is
    # within-score rather than confounded with a change of score.
    "score_matched_gain_h512_p10_80":
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_matched_gain_p10_80",
    # The same for the random mask. This is the control that shows a global
    # factor is not uniformly useless: it lifts an unbiased mask and destroys a
    # biased one.
    "random_gain_h512_p10_80":
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_random_gain_p10_80",
    # Task input x injected noise, re-run on the corrected rule. The original
    # 2x2 predates the probability fix and both its cells are rescale variants,
    # so both were affected.
    "cov2x2_h512_p10_80":
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_cov2x2_p10_80",
    # Pruning on dm1 and anti, the two excluded tasks. Establishes what the
    # baselines could not: that the methods stop separating on them.
    "dm1anti_pruning_p10_80":
        "task_preservation_tanh_h512_modcog_dm1anti_12k_seqbest_pruning_p10_80",
}

# The paper's statistics, which also changed -- from Wilcoxon/Holm to a
# hierarchical bootstrap with an exact sign-flip test.
# Suites the runner shards by (task, sparsity) and that the deposit therefore
# ships merged, like the re-run. Without the first, two rows of
# key_comparisons.csv cannot be recomputed from the deposit and
# build_cube(FULL_ARMS) cannot run at all. Without the other two, three of the
# seven arms in the published two-size figure -- magnitude, OBS and random at
# H=1024 -- have no data here.
SHARDED_SUITES = {
    "fig3_lowsparsity_h512_p10_40":
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_fig3_lowsparsity_p10_40",
    "size_h1024_p10_40":
        "task_preservation_tanh_h1024_modcog_revised8_12k_seqbest_revised_task_only_p10_40",

    # --- three suites the live code reads that were never deposited ---------
    # Found by checking every source path in final_style.SOURCES and
    # paper_stats.SOURCES against the manifest. The first is the worst of them:
    # it is a source for BOTH the figures and the statistics, so a referee
    # re-running the deposited analysis on the deposited data hits a missing
    # path rather than a missing interval.
    "fixedcapfull_h512_p50_80":
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_fixedcapfull_p50_80",
    # The H=1024 half of the same arm, which the two-size SI figure draws.
    "fixedcapfull_h1024_p50_80":
        "task_preservation_tanh_h1024_modcog_revised8_12k_seqbest_fixedcapfull_p50_80",
    # Gain restoration on the NOISE-PRUNE mask at 50-80%: row-wise (synaptic
    # scaling proper), its column-wise mirror, and 1/density. The density-
    # matched control's interval cannot be resampled without it -- only the
    # per-sparsity means were deposited, which give the average but not the
    # bootstrap.
    "synaptic_h512_p50_80":
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_synaptic_p50_80",
    # The neuron-wise rules over the full 10-80% range, on BOTH masks: the
    # magnitude mask at every sparsity (the arm the manuscript sentence
    # describes, previously n = 8 and 50-80% for row-wise and absent for
    # column-wise) and the noise-prune mask at 10-40%, which completes that
    # mask's range against the suite above.
    "neuronwise_h512_p10_80":
        "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_neuronwise_p10_80",
}

NEW_STATS_DIR = PROJECT / "analysis/paper_statistics"

# Deposit-relative files the re-run REPLACES outright. Deliberately empty, and
# the reasoning is worth recording because the obvious candidates are all wrong:
#
#   * The two big quantile-cap sweep tables are only half sampling-arm rows; the
#     rest are L-NP. Deleting them would take valid data with them. (The --prune
#     guard refused exactly this when it was first attempted.)
#   * noise_necessity and noise_only ARE 100% sampling-arm rows, but they are
#     reviewer-control ablations -- a sigma sweep and a zero-task-input
#     condition -- that were not part of the re-run. Nothing replaces them, so
#     deleting them would lose the controls rather than refresh them.
#
# "Affected by the correction" and "superseded by something better" are not the
# same property, and only the second justifies a deletion from a deposit that is
# already published. Everything affected-but-not-replaced is recorded in
# SUPERSEDED.md instead.
SUPERSEDED_WHOLE: tuple[str, ...] = ()

# Statistical outputs superseded by the change of PROCEDURE, independent of the
# data. These are Wilcoxon/Holm families the paper no longer reports.
SUPERSEDED_STATISTICS = (
    "results/statistics/main_task_retention_84_wilcoxon_holm_sign_tests.csv",
    "results/statistics/paper_relevant_h512_comparison_tests.csv",
    "results/statistics/revised_scope_task_only_significance_tests.csv",
    "results/statistics/revised_scope_task_only_significance_means.csv",
)


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks so large checkpoints do not load into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_copies(deposit: Path) -> List[Tuple[Path, Path]]:
    """Build the (source, destination) list for everything the revision adds.

    Sources that do not exist are skipped rather than raising, so the script stays
    usable when only some suites have been run.
    """
    pairs: List[Tuple[Path, Path]] = []
    revision = deposit / "results/revision_2026"

    for label, (stem, _config_dir) in SUITE_STEMS.items():
        suite_dir = RESULTS / stem
        for source in sorted(suite_dir.glob(f"{stem}_*.csv")):
            pairs.append((source, revision / label / source.name))
        # cross-suite deliverables are named for their purpose, not the stem
        for source in sorted(suite_dir.glob("reviewer1_controls_*.csv")):
            pairs.append((source, revision / label / source.name))

    for pattern in DM1ANTI_RESULT_GLOBS:
        for source in sorted(RESULTS.glob(pattern)):
            pairs.append((source, revision / "dm1anti_baselines" / source.name))

    # "suites" is deliberately omitted: everything in it duplicates a file already staged
    # from its primary location, and a deposit should not ship the same table twice.
    for sub in ("reference_tables", "statistics"):
        for source in sorted((ANALYSIS / sub).glob("*.csv")):
            pairs.append((source, revision / sub / source.name))

    for name in DM1ANTI_CHECKPOINT_DIRS + H1024_CHECKPOINT_DIRS:
        for source in sorted((PROJECT / "checkpoints" / name).glob("*.pt")):
            pairs.append((source, deposit / "checkpoints" / name / source.name))

    config_root = ROOT / "configs"
    for stem, config_dir in SUITE_STEMS.values():
        found = sorted((config_root / config_dir).glob("*.json"))
        if not found:
            raise SystemExit(
                f"no configs found for suite '{stem}' in {config_root / config_dir}. "
                f"A deposit that ships results without the configs that produced them is "
                f"not reproducible; fix the path in SUITE_STEMS rather than skipping it."
            )
        for source in found:
            pairs.append((source, revision / "configs" / stem / source.name))
    for pattern in ("*modcog_dm1anti*.json",):
        for source in sorted(config_root.glob(pattern)):
            pairs.append((source, revision / "configs" / "dm1anti" / source.name))

    # --- the renormalisation re-run, merged, plus the configs that produced it
    renorm = deposit / "results/renormalized_2026_10"
    staging = PROJECT / ".staging_merged"
    for label, stem in RENORM_SUITES.items():
        merged = merge_suite(stem, staging)
        if merged is None:
            print(f"  note: suite {stem} has no results yet; skipped")
            continue
        pairs.append((merged, renorm / f"{label}.csv"))
        for source in sorted((config_root / stem).glob("*.json")):
            pairs.append((source, renorm / "configs" / label / source.name))

    # --- the sharded suites the paper's figures and statistics still read
    sharded = deposit / "results/revision_2026/sharded_suites"
    for label, stem in SHARDED_SUITES.items():
        merged = merge_suite(stem, staging)
        if merged is None:
            print(f"  note: suite {stem} has no results yet; skipped")
            continue
        pairs.append((merged, sharded / f"{label}.csv"))
        for source in sorted((config_root / stem).glob("*.json")):
            pairs.append((source, sharded / "configs" / label / source.name))

    # --- the paper's statistics under the new procedure
    for source in sorted(NEW_STATS_DIR.glob("*.csv")):
        pairs.append((source, deposit / "results/statistics_2026_10" / source.name))
    for source in sorted(NEW_STATS_DIR.glob("*.py")):
        pairs.append((source, deposit / "results/statistics_2026_10" / "code" / source.name))

    return [(src, dst) for src, dst in pairs if src.is_file()]


# The two cap-dependent arms were re-run at the recalibrated constant, under the
# SAME run_ids. If the primary H=512 table kept its capped rows, the deposit
# would carry each of those 1,152 runs twice at two different caps, with
# colliding run_ids -- anyone concatenating the tables would double-count them
# and silently average two caps. The primary table therefore sheds the arms the
# recap table supersedes, so every run_id in the deposit appears exactly once.
SUPERSEDED_BY_RECAP = {
    "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_renorm_p10_80": (
        "simulation_noise_prune_capped_rescale",
        "simulation_noise_prune_capped_magnitude_rescale",
    ),
}


def merge_suite(stem: str, out_dir: Path) -> Path | None:
    """Merge a suite's per-task shards into one table, as the deposit expects.

    The runner writes 64 shards per suite; the deposit ships merged tables, so
    the two stay consistent with how every other suite here is deposited.
    """
    import pandas as pd

    src = RESULTS / stem
    # Only the per-task shards. A suite that already ships its own merged
    # `_raw.csv` or a summary would otherwise be concatenated with itself --
    # which is exactly what the duplicate-run_id guard caught for the H=1024
    # p50-80 suite, where shards plus _raw gave 2,059 collisions.
    DERIVED = ("_summary", "_raw", "_merged", "_vs_", "_paired", "_plot_points",
               "_holm_tests", "_regression_check", "_gain_audit", "_contribution")
    shards = sorted(f for f in src.glob("*.csv")
                    if "logs" not in f.parts and not any(k in f.name for k in DERIVED))
    if not shards:
        return None
    frames = [pd.read_csv(f, low_memory=False) for f in shards]
    merged = pd.concat(frames, ignore_index=True)
    drop = SUPERSEDED_BY_RECAP.get(stem)
    if drop and "strategy" in merged:
        before = len(merged)
        merged = merged[~merged.strategy.isin(drop)]
        print(f"  {stem}: dropped {before - len(merged)} rows superseded by the "
              f"recalibrated-cap re-run")
    # A run emitted identically by two shards is not two runs. Suites run in
    # more than one pass re-emit the unpruned baseline in each pass -- the
    # H=512 fixedcapfull suite carries its 24 baselines twice, once from the
    # p10 pass and once from p50, agreeing to the last digit on every measured
    # column. Collapsing those is correct; collapsing two runs that DISAGREE
    # would hide a real re-run, so the key is run_id PLUS what was measured,
    # the same key paper_stats.DEDUP_KEY uses and for the same reason.
    if "run_id" in merged:
        value_cols = [c for c in ("post_acc_sequence", "post_rec_weight_nz_count",
                                  "pre_acc_sequence") if c in merged]
        before = len(merged)
        merged = merged.drop_duplicates(subset=["run_id"] + value_cols)
        if before != len(merged):
            print(f"  {stem}: collapsed {before - len(merged)} rows re-emitted "
                  f"identically by more than one shard")
        dupes = int(merged.run_id.duplicated().sum())
        if dupes:
            raise SystemExit(
                f"{stem}: {dupes} run_ids appear more than once with DIFFERENT "
                f"measured values; refusing to deposit a table that double-counts runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{stem}_merged.csv"
    merged.to_csv(dst, index=False)
    return dst


def affected_fraction(path: Path) -> tuple[int, int]:
    """(rows naming a superseded sampling arm, total rows) for one deposit CSV.

    Looks in every column that could carry a method label, because the summary
    and statistics tables use display names rather than strategy strings.
    """
    import pandas as pd

    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return (0, 0)
    keys = [c for c in ("strategy", "arm", "method", "Method", "label",
                        "arm_a", "arm_b", "comparison") if c in cols]
    if not keys:
        return (0, 0)
    try:
        d = pd.read_csv(path, usecols=keys, low_memory=False)
    except Exception:
        return (0, 0)
    if d.empty:
        return (0, 0)
    hit = pd.Series(False, index=d.index)
    for c in keys:
        col = d[c].astype(str)
        hit |= col.isin(SAMPLING_STRATEGIES)
        hit |= col.str.contains(r"S-NP|snp_|fixedg38", regex=True, na=False)
    return (int(hit.sum()), len(d))


# The re-run and the statistics computed from it ARE the corrected data. They
# must never be listed as containing pre-correction rows: a reader following
# that instruction would filter out the paper's primary evidence and keep the
# superseded tables instead, the exact opposite of the intent.
AUDIT_SKIP_DIRS = ("results/renormalized_2026_10/", "results/statistics_2026_10/")


def audit_deposit(deposit: Path) -> dict:
    """Classify every deposit CSV by how much of it the re-run supersedes."""
    whole, mixed = [], []
    for path in sorted(deposit.rglob("*.csv")):
        if any(path.relative_to(deposit).as_posix().startswith(d) for d in AUDIT_SKIP_DIRS):
            continue
        aff, tot = affected_fraction(path)
        if not aff:
            continue
        rel = path.relative_to(deposit).as_posix()
        (whole if aff >= tot else mixed).append((rel, aff, tot))
    return {"whole": whole, "mixed": mixed}


def write_superseded_note(deposit: Path, audit: dict) -> Path:
    """Record exactly which rows of which files predate the correction.

    The mixed files cannot simply be deleted -- most of their rows describe
    deterministic arms that never changed -- and rewriting 38 derived summary
    tables row by row would be a far larger and more error-prone change than
    stating plainly which rows are stale. A reader filtering any of these
    tables to a sampling arm needs to know to use the re-run instead, so that
    is written down rather than left to be discovered.
    """
    lines = [
        "# Superseded rows",
        "",
        "The sampling rule's probability normalisation was corrected on 2026-10-09.",
        "Before the fix it scaled its unnormalised scores by `K = M / sum(q)` and then",
        "clipped them at 1, discarding the probability mass above 1, so it retained fewer",
        "connections than requested -- at a requested 10% sparsity it removed 33% of the",
        "recurrent weights. It now solves `sum(min(1, K q)) = M`, so the expected retained",
        "count equals the budget.",
        "",
        "Six strategies draw a Bernoulli mask and were therefore affected:",
        "",
    ] + [f"  - `{s}`" for s in SAMPLING_STRATEGIES] + [
        "",
        "All six were re-run over 10-80% at H=512. At H=1024 four were re-run -- the",
        "plain and capped rescales on both scores -- while the shuffled and uniform",
        "probability controls were not, because the size comparison does not use them.",
        "**The re-run is the data the paper reports**, and it is deposited under",
        "`results/renormalized_2026_10/`.",
        "",
        "The Lyapunov family (`noise_prune*`, called L-NP or V-NP in older tables) is",
        "affected by the identical defect and was NOT re-run. It is not reported in the",
        "revised paper. Treat any L-NP row anywhere in this deposit as pre-correction.",
        "",
        "Four strategies are deterministic and were NOT affected; their numbers are",
        "bit-identical before and after: `l1_unstructured`, `obs_compensated`,",
        "`random_unstructured`, `simulation_noise_prune_mask_only`.",
        "",
        "## Files that still contain pre-correction rows",
        "",
        "These tables mix affected and unaffected rows, so they are kept whole rather than",
        "rewritten. Filter out the six strategies above and take them from the re-run",
        "instead; the remaining rows are unchanged and still valid.",
        "",
        "| file | pre-correction rows | total rows |",
        "|---|---|---|",
    ] + [f"| `{rel}` | {aff} | {tot} |" for rel, aff, tot in audit["mixed"]] + [
        "",
        "## Affected but NOT re-run",
        "",
        "These are reviewer-control ablations of the sampling rule -- a sigma sweep and a",
        "zero-task-input condition -- that the re-run did not cover. Every row in them",
        "predates the correction and nothing replaces them, so they are kept and labelled",
        "rather than deleted. Read them as evidence about the OLD rule.",
        "",
        "  - `results/revision_2026/noise_necessity/`",
        "  - `results/revision_2026/noise_only/`",
        "",
        "## Files removed",
        "",
    ] + ([f"  - `{rel}`  ({aff} rows)" for rel, aff, _t in audit["whole"]]
         or ["  (none -- see the note above)"]) + [
        "",
        "These statistical outputs are superseded by the change of PROCEDURE, not only of",
        "data: the paper now reports a hierarchical bootstrap over tasks with an exact",
        "sign-flip test, in place of Wilcoxon signed-rank with Holm correction. The",
        "replacements are in `results/statistics_2026_10/`:",
        "",
    ] + [f"  - `{p}`" for p in SUPERSEDED_STATISTICS] + [""]
    dst = deposit / "SUPERSEDED.md"
    dst.write_text("\n".join(lines))
    return dst


def deposit_files(deposit: Path) -> Iterable[Path]:
    """Every file the deposit ships, excluding VCS metadata and the manifests."""
    for path in sorted(deposit.rglob("*")):
        if not path.is_file():
            continue
        if path.name in MANIFEST_EXCLUDE_NAMES:
            continue
        if MANIFEST_EXCLUDE_DIRS & set(path.relative_to(deposit).parts):
            continue
        yield path


def write_manifests(deposit: Path) -> int:
    """Rebuild both checksum manifests over the whole deposit; return the record count."""
    records = []
    lines = []
    for path in deposit_files(deposit):
        rel = path.relative_to(deposit).as_posix()
        digest = sha256(path)
        records.append({"path": rel, "bytes": path.stat().st_size, "sha256": digest})
        lines.append(f"{digest}  {rel}")
    (deposit / "manifest.json").write_text(json.dumps({"files": records}, indent=2) + "\n")
    (deposit / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deposit-dir", type=Path, default=DEFAULT_DEPOSIT,
                        help="data release staging directory to update")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be copied without writing anything")
    parser.add_argument("--audit", action="store_true",
                        help="classify every deposit CSV by how much the re-run "
                             "supersedes, and exit without writing")
    parser.add_argument("--prune", action="store_true",
                        help="also REMOVE the wholly superseded files and the "
                             "superseded statistical outputs. Off by default: this "
                             "script has only ever added, and a deletion from a "
                             "deposit that is already published should be deliberate.")
    args = parser.parse_args()

    deposit = args.deposit_dir
    if not deposit.is_dir():
        raise SystemExit(f"deposit directory not found: {deposit}")

    if args.audit:
        a = audit_deposit(deposit)
        print(f"wholly superseded: {len(a['whole'])} files")
        for rel, aff, tot in a["whole"]:
            print(f"   {rel}  ({aff}/{tot} rows)")
        print(f"\npartly superseded: {len(a['mixed'])} files")
        for rel, aff, tot in a["mixed"]:
            print(f"   {rel}  ({aff}/{tot} rows)")
        return

    pairs = plan_copies(deposit)
    total_bytes = sum(src.stat().st_size for src, _ in pairs)
    print(f"{len(pairs)} files to stage ({total_bytes / 1048576:.1f} MB)")
    if args.dry_run:
        for src, dst in pairs:
            print(f"  would copy {src.relative_to(PROJECT)} -> {dst.relative_to(deposit)}")
        return

    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"staged into {deposit}")

    if args.prune:
        audit = audit_deposit(deposit)
        whole = {rel for rel, _a, _t in audit["whole"]}
        removed = []
        for rel in SUPERSEDED_WHOLE:
            path = deposit / rel
            if not path.is_file():
                print(f"  prune: {rel} already absent")
                continue
            if rel not in whole:
                aff, tot = affected_fraction(path)
                raise SystemExit(
                    f"refusing to remove {rel}: only {aff} of {tot} rows are "
                    f"superseded, so deleting it would take valid data with it")
            path.unlink(); removed.append(rel)
        for rel in SUPERSEDED_STATISTICS:
            path = deposit / rel
            if path.is_file():
                path.unlink(); removed.append(rel)
            else:
                print(f"  prune: {rel} already absent")
        print(f"removed {len(removed)} superseded files")
        for rel in removed:
            print(f"   - {rel}")
    before = 0
    manifest_path = deposit / "manifest.json"
    if manifest_path.is_file():
        before = len(json.loads(manifest_path.read_text()).get("files", []))
    # after staging, so the counts describe what actually shipped
    note = write_superseded_note(deposit, audit_deposit(deposit))
    print(f"wrote {note.relative_to(deposit)} from the post-staging audit")
    count = write_manifests(deposit)
    print(f"manifests rebuilt: {before} -> {count} records "
          f"(MANIFEST.sha256 and manifest.json)")


if __name__ == "__main__":
    main()
