#!/usr/bin/env python3
"""Every sampling arm, re-run at 10-80% under the corrected probability
normalisation.

WHY THIS EXISTS. Until 2026-10-09 the sampling rule scaled its unnormalised
scores by K = M / sum(q) and then clipped to 1, where M is the requested number
of retained edges. Clipping discards the probability mass above 1, so the rule
retained fewer edges than it was asked for -- at H = 512 a requested 10%
sparsity removed 33.0%, a 25.6% shortfall, falling to 0.7% at 80%. Every arm
that draws a Bernoulli mask was therefore being compared at a sparsity it never
asked for, and the comparison was worst exactly where the shortfall was worst.

`prob_normalize: "bisect"` now solves for the single K with
sum(min(1, K q)) = M, so the expected retained count equals the budget. The
configs set it explicitly rather than relying on the default, so they stay
self-describing if the default ever moves again.

WHAT IS RE-RUN. Exactly the six registered strategies that sample, confirmed by
walking the strategy registry. The deterministic arms -- magnitude, OBS, random,
the noise-prune score-only mask -- already hit the requested sparsity exactly
and are NOT re-run; their existing cells remain valid and will be joined to this
suite's output.

    snp_rescale       simulation_noise_prune_rescale
    snp_capped        simulation_noise_prune_capped_rescale               cap 3.8
    snp_magnitude     simulation_noise_prune_magnitude_rescale
    snp_capped_mag    simulation_noise_prune_capped_magnitude_rescale     cap 3.8
    snp_shuffled      simulation_noise_prune_shuffled_rescale
    snp_uniform       simulation_noise_prune_uniform_rescale

TWO DESIGN POINTS.

The full 10-80% range is run for every arm, not just the 10-40% band where the
shortfall was large. Re-running only the affected range would leave one curve
built from two normalisations joined at 50%, and no reader could tell which
side of the join a difference came from. The 50-80% cells will barely move --
the old shortfall there was 6.8% down to 0.7% -- and that is the point: they
are the regression check on the fix.

`snp_shuffled` and `snp_uniform` previously existed only at 50-80%, so Figure
3B could not span the same range as the rest. They are run at all eight
sparsities here, which closes that gap at the same time.

THE CAP IS A SINGLE CONSTANT, 3.8, AT EVERY SPARSITY. The old fixedcapfull
suite mixed two capped arms at 80% -- the constant 3.8 and a per-sparsity
constant of 5.922 -- both recording cap_mode "fixed", which is a trap for any
analysis that selects on the mode alone. One constant everywhere removes it.
The constant was calibrated against the OLD amplification distribution and will
need recalibrating once these curves are in; that is deliberate and deferred,
so this suite measures the normalisation change alone.

SIZE. H = 512: 6 arms x 8 sparsities x 8 tasks x 3 network seeds x 3 pruning
seeds = 3,456 pruned runs plus 24 baselines. H = 1024: snp_rescale only, same
grid = 576 pruned runs plus 24 baselines, to show the larger network follows the
same pattern once it is pruned to the sparsity it was asked for.

Sharded by (task, sparsity): 64 units per size, so an interruption costs at
most one unit and the suite resumes per run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(f"generate_{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rs = _load("tanh_h512_modcog_revised8_revision_suites")      # H = 512 helpers
_st = _load("modcog_revised8_size_taskpres_suite")            # H = 1024 helpers

AMOUNTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
NETWORK_SEEDS = (0, 1, 2)
PRUNING_SEEDS = (0, 1, 2)
CAP_VALUE = 3.8
PROB_CONTROL_SEED_BASE = 900_000
NORMALIZE = "bisect"

# (label, strategy, extras)
ARMS_512 = (
    ("snp_rescale",    "simulation_noise_prune_rescale", {}),
    ("snp_capped",     "simulation_noise_prune_capped_rescale",
     {"rescale_cap_mode": "fixed", "rescale_cap_value": CAP_VALUE}),
    ("snp_magnitude",  "simulation_noise_prune_magnitude_rescale", {}),
    ("snp_capped_mag", "simulation_noise_prune_capped_magnitude_rescale",
     {"rescale_cap_mode": "fixed", "rescale_cap_value": CAP_VALUE}),
    ("snp_shuffled",   "simulation_noise_prune_shuffled_rescale", {}),
    ("snp_uniform",    "simulation_noise_prune_uniform_rescale", {}),
)
# The first pass ran only the plain rescale rule at H = 1024, which is what the
# size question needed. ARM_SET="rest" runs the other three, so the size figure
# can show every arm on one normalisation.
ARMS_1024 = (("snp_rescale", "simulation_noise_prune_rescale", {}),)
ARMS_1024_REST = (
    ("snp_capped", "simulation_noise_prune_capped_rescale",
     {"rescale_cap_mode": "fixed", "rescale_cap_value": CAP_VALUE}),
    ("snp_magnitude", "simulation_noise_prune_magnitude_rescale", {}),
    ("snp_capped_mag", "simulation_noise_prune_capped_magnitude_rescale",
     {"rescale_cap_mode": "fixed", "rescale_cap_value": CAP_VALUE}),
)
# Only the two capped arms depend on the cap constant, so a recalibration needs
# only those re-run. The other four sampling arms would reproduce byte for byte
# -- same seeds, same code, cap irrelevant -- and re-running them would waste
# about three hours and invite confusion about why two suites hold identical
# numbers.
ARMS_512_CAPPED = tuple(a for a in ARMS_512 if "capped" in a[1])
ARM_SET = "first"          # "first" | "rest" | "capped"; set by --arm-set

# every arm here drives a probability control except the two plain rescales
CONTROLLED = {
    "simulation_noise_prune_magnitude_rescale",
    "simulation_noise_prune_capped_magnitude_rescale",
    "simulation_noise_prune_shuffled_rescale",
    "simulation_noise_prune_uniform_rescale",
}


STEM_TAG = ""              # set by --stem-tag; keeps passes in separate stems


def suite_stem(h: int) -> str:
    """Stem for this pass.

    MUST differ between passes that could produce different numbers for the
    same run_id. The harness resumes on run_id alone, and the run_id carries
    neither the cap value nor the arm set, so a second pass written to the same
    stem would find every row already present, skip all of them, and leave the
    configs on disk advertising a cap the rows were never produced with --
    silently.
    """
    tag = f"_{STEM_TAG}" if STEM_TAG else ""
    return f"task_preservation_tanh_h{h}_modcog_revised8_12k_seqbest_renorm{tag}_p10_80"


def shard(amount: float) -> str:
    return f"_p{int(round(amount * 100))}"


def arms(h: int):
    if h == 512:
        return ARMS_512_CAPPED if ARM_SET == "capped" else ARMS_512
    return ARMS_1024 if ARM_SET == "first" else ARMS_1024_REST


def common(h: int, task_label: str, ng_t: int, seed: int) -> dict:
    if h == 512:
        return _rs._common(task_label, ng_t, seed)
    return {
        "task": f"modcog:{task_label}", "ng_T": ng_t,
        "load_model_path": _st.checkpoint_path(h, task_label, seed),
        "source_model_label": f"{task_label}_tanh_h{h}_12k_lr0006_seqbest_no_l2_seed{seed}",
        "source_run_id": (f"continue_modcog_{task_label}_tanh_h{h}_to12k_lr0006"
                          f"_seqbest_no_l2_seed{seed}"),
        "source_network_seed": seed, "source_recurrent_l2_lambda": 0.0,
        "eval_seed": 200_000, "eval_batches_path": _st.eval_batches_path(task_label),
    }


def score_path(h: int, task_label: str, pseed: int) -> str:
    return (_rs.score_batches_path(task_label, pseed) if h == 512
            else _st.score_batches_path(task_label, pseed))


def defaults(h: int) -> dict:
    d = _rs.defaults_block() if h == 512 else _st.defaults_block(h)
    # Explicit, so the config records the normalisation it was run under rather
    # than inheriting whatever the code's default happens to be later.
    d = dict(d)
    d["sim_np_prob_normalize"] = NORMALIZE
    return d


def make_unit(h: int, task_label: str, ng_t: int, amount: float,
              include_baseline: bool) -> dict:
    runs = []
    # only the primary pass records baselines; see the note on STEM_TAG
    include_baseline = include_baseline and not STEM_TAG
    for seed in NETWORK_SEEDS:
        c = common(h, task_label, ng_t, seed)
        if include_baseline:
            runs.append({
                "run_id": f"renorm_h{h}_{task_label}_netseed{seed}_baseline",
                "strategy": "none", "amount": 0.0, "no_prune": True,
                "seed": seed, **c,
            })
        for label, strategy, extras in arms(h):
            for pseed in PRUNING_SEEDS:
                run = {
                    "run_id": (f"renorm_h{h}_{task_label}_netseed{seed}_{label}"
                               f"_p{int(round(amount * 100))}_pruneseed{pseed}"),
                    "strategy": strategy, "amount": amount,
                    "seed": pseed, "pruning_seed": pseed, "noise_rng_seed": pseed,
                    "score_batch_seed": 100_000 + pseed,
                    "score_batches_path": score_path(h, task_label, pseed),
                    **extras, **c,
                }
                if strategy in CONTROLLED:
                    run["prob_control_seed"] = PROB_CONTROL_SEED_BASE + pseed
                runs.append(run)
    stem = suite_stem(h)
    return {
        "run_id": f"{stem}_{task_label}{shard(amount)}",
        "output_csv": f"results/{stem}/{task_label}{shard(amount)}.csv",
        "defaults": defaults(h),
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    global ARM_SET, CAP_VALUE, ARMS_512, ARMS_1024_REST, STEM_TAG
    ap.add_argument("--hidden-size", type=int, default=512, choices=(512, 1024))
    ap.add_argument("--arm-set", default="first", choices=("first", "rest", "capped"),
                    help="'first' the primary set; 'rest' the three H=1024 arms the "
                         "first pass skipped; 'capped' only the two cap-dependent arms")
    ap.add_argument("--cap-value", type=float, default=None,
                    help="override the constant amplification cap")
    ap.add_argument("--stem-tag", default="",
                    help="suffix for the suite stem; REQUIRED for any pass after "
                         "the first, since resume keys on run_id alone")
    args = ap.parse_args()
    h = args.hidden_size
    ARM_SET = args.arm_set
    STEM_TAG = args.stem_tag
    if ARM_SET != "first" and not STEM_TAG:
        raise SystemExit("a non-primary arm set needs --stem-tag, or it would "
                         "write into the primary pass's stem and be skipped")
    if args.cap_value is not None and args.cap_value != CAP_VALUE:
        CAP_VALUE = float(args.cap_value)
        ARMS_512 = tuple(
            (l, s, ({**e, "rescale_cap_value": CAP_VALUE} if "rescale_cap_value" in e else e))
            for l, s, e in ARMS_512)
        ARMS_1024_REST = tuple(
            (l, s, ({**e, "rescale_cap_value": CAP_VALUE} if "rescale_cap_value" in e else e))
            for l, s, e in ARMS_1024_REST)
        globals()["ARMS_512_CAPPED"] = tuple(
            (l, s, ({**e, "rescale_cap_value": CAP_VALUE} if "rescale_cap_value" in e else e))
            for l, s, e in ARMS_512 if "capped" in s)

    tasks = _rs.TASKS if h == 512 else _st.TASKS
    if h == 512:
        _rs.validate_inputs()
    else:
        _st.validate(h, require_checkpoints=True)

    stem = suite_stem(h)
    cfg_dir = Path(f"configs/{stem}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    Path(f"results/{stem}").mkdir(parents=True, exist_ok=True)

    total = n_cfg = 0
    for entry in tasks:
        task_label, _task, ng_t = entry
        for i, amount in enumerate(AMOUNTS):
            unit = make_unit(h, task_label, ng_t, amount, include_baseline=(i == 0))
            (cfg_dir / f"{stem}_{task_label}{shard(amount)}.json").write_text(
                json.dumps(unit, indent=2) + "\n")
            total += len(unit["runs"])
            n_cfg += 1

    pruned = len(arms(h)) * len(AMOUNTS) * len(tasks) * len(NETWORK_SEEDS) * len(PRUNING_SEEDS)
    print(f"wrote {n_cfg} configs to {cfg_dir}")
    print(f"  H = {h}; arms {[a[0] for a in arms(h)]}")
    print(f"  sparsities {[int(a * 100) for a in AMOUNTS]}; "
          f"{len(tasks)} tasks x {len(NETWORK_SEEDS)} net seeds x {len(PRUNING_SEEDS)} prune seeds")
    n_base = total - pruned
    print(f"  {pruned} pruned runs + {n_base} baselines = {total} total"
          + ("   (baselines live in the primary pass)" if STEM_TAG else ""))
    print(f"  normalisation: {NORMALIZE}; cap {CAP_VALUE} at every sparsity")


if __name__ == "__main__":
    main()
