#!/usr/bin/env python3
"""Pruning on dm1 and anti -- the two tasks the paper excludes as uninformative.

The baseline suite established that these tasks are *trivially solvable*: anti
reaches 1.0000 with s.d. 0.0000 over three seeds and dm1 0.9500 with s.d.
0.0003, against 0.602-0.954 and s.d. 0.0056-0.0995 for the eight retained
tasks. It did not establish the claim the paper actually makes, which is that
they are uninformative *for comparing pruning methods* -- and the revision plan
flags exactly that gap. This suite closes it.

The same arms as the main retention figure, over the same 10-80% grid, so the
supplementary panel can be read directly against it. What makes a task
uninformative is not that pruning leaves it untouched but that the methods stop
SEPARATING on it: if a trivially solvable task is held up by so much redundant
structure that any mask preserves it, every curve lies on top of every other
and the task cannot discriminate between rules. That is the comparison to plot,
so random and magnitude matter here as much as noise-prune does.

Batch caches live in their own directory. The eval and score batches are keyed
by task and the harness regenerates them when the file is absent, so pointing
these runs at the revised8 cache would either miss or, worse, silently collide
with the paper's own cached batches.

The guard in the baseline generator -- refusing any path containing "revised8"
-- is kept for the same reason: the paper's 24 networks must not be touched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "revision_suites",
    Path(__file__).with_name("generate_tanh_h512_modcog_revised8_revision_suites.py"))
_rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rs)

SUITE_STEM = "task_preservation_tanh_h512_modcog_dm1anti_12k_seqbest_pruning_p10_80"
CONFIG_DIR = Path(f"configs/{SUITE_STEM}")
BATCH_CACHE = Path("results/fixed_batches/tanh_h512_modcog_dm1anti_taskpres")
CKPT_DIR = "checkpoints/tanh_h512_modcog_dm1anti_to12k_lr0006_seqbest_no_l2_seed{seed}"

# (label, modcog id, ng_T) -- timings derived exactly as the baseline suite did
TASKS = (("dm1", "modcog:dm1", 22), ("anti", "modcog:anti", 30))
NETWORK_SEEDS = (0, 1, 2)
AMOUNTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
STOCHASTIC_SEEDS = (0, 1, 2)
DETERMINISTIC_SEEDS = (0,)

CAP_VALUE = 3.695

# the arms the main retention figure draws, same names, same settings
ARMS = (
    ("snp_capped", "simulation_noise_prune_capped_rescale", STOCHASTIC_SEEDS,
     {"rescale_cap_mode": "fixed", "rescale_cap_value": CAP_VALUE}),
    ("snp_rescale", "simulation_noise_prune_rescale", STOCHASTIC_SEEDS, {}),
    ("magnitude", "l1_unstructured", DETERMINISTIC_SEEDS, {}),
    ("obs", "obs_compensated", DETERMINISTIC_SEEDS, {}),
    ("random", "random_unstructured", DETERMINISTIC_SEEDS, {}),
)


def checkpoint_path(task: str, seed: int) -> str:
    p = f"{CKPT_DIR.format(seed=seed)}/modcog_{task}_seed{seed}.pt"
    if "revised8" in p:                     # the paper's networks are off limits
        raise SystemExit(f"refusing a revised8 checkpoint path: {p}")
    return p


def eval_batches_path(task: str) -> str:
    return str(BATCH_CACHE / "eval" / f"{task}_eval_seed200000.pt")


def score_batches_path(task: str, seed: int) -> str:
    return str(BATCH_CACHE / "score" / f"{task}_score_seed{100_000 + seed}.pt")


def _common(label: str, ng_t: int, seed: int) -> dict:
    return {
        "task": f"modcog:{label}", "ng_T": ng_t,
        "load_model_path": checkpoint_path(label, seed),
        "source_model_label": f"{label}_tanh_h512_dm1anti_12k_seqbest_no_l2_seed{seed}",
        "source_run_id": f"continue_modcog_{label}_tanh_h512_dm1anti_to12k_seed{seed}",
        "source_network_seed": seed, "source_recurrent_l2_lambda": 0.0,
        "eval_seed": 200_000, "eval_batches_path": eval_batches_path(label),
    }


def make_unit(label: str, ng_t: int, amount: float) -> dict:
    runs = []
    for seed in NETWORK_SEEDS:
        c = _common(label, ng_t, seed)
        for arm, strategy, pseeds, extras in ARMS:
            for ps in pseeds:
                runs.append({
                    "run_id": (f"dm1anti_{label}_netseed{seed}_{arm}_"
                               f"p{int(round(amount * 100))}_pruneseed{ps}"),
                    "strategy": strategy, "amount": amount,
                    "seed": ps, "pruning_seed": ps, "noise_rng_seed": ps,
                    "score_batch_seed": 100_000 + ps,
                    "score_batches_path": score_batches_path(label, ps),
                    **extras, **c})
    return {"run_id": f"{SUITE_STEM}_{label}_p{int(round(amount * 100))}",
            "output_csv": f"results/{SUITE_STEM}/{label}_p{int(round(amount * 100))}.csv",
            "defaults": _baseline_defaults(), "runs": runs}


def _baseline_defaults() -> dict:
    d = dict(_rs.defaults_block())
    d["sim_np_prob_normalize"] = "bisect"       # the corrected rule
    return d


def make_baseline_unit(label: str, ng_t: int) -> dict:
    runs = [{"run_id": f"dm1anti_{label}_netseed{seed}_baseline", "strategy": "none",
             "amount": 0.0, "no_prune": True, "seed": seed,
             **_common(label, ng_t, seed)} for seed in NETWORK_SEEDS]
    return {"run_id": f"{SUITE_STEM}_{label}_baseline",
            "output_csv": f"results/{SUITE_STEM}/{label}_baseline.csv",
            "defaults": _baseline_defaults(), "runs": runs}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    for label, _t, _n in TASKS:
        for seed in NETWORK_SEEDS:
            if not Path(checkpoint_path(label, seed)).exists():
                raise SystemExit(f"missing checkpoint: {checkpoint_path(label, seed)}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(f"results/{SUITE_STEM}").mkdir(parents=True, exist_ok=True)
    for sub in ("eval", "score"):
        (BATCH_CACHE / sub).mkdir(parents=True, exist_ok=True)
    total = n = 0
    for label, _t, ng_t in TASKS:
        u = make_baseline_unit(label, ng_t)
        (CONFIG_DIR / f"{SUITE_STEM}_{label}_baseline.json").write_text(
            json.dumps(u, indent=2) + "\n")
        total += len(u["runs"]); n += 1
        for amount in AMOUNTS:
            u = make_unit(label, ng_t, amount)
            (CONFIG_DIR / f"{SUITE_STEM}_{label}_p{int(round(amount*100))}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs to {CONFIG_DIR}; total runs {total}")
    print(f"  tasks {[t[0] for t in TASKS]} x {len(NETWORK_SEEDS)} seeds; "
          f"sparsities {[int(a*100) for a in AMOUNTS]}")
    print(f"  arms {[a[0] for a in ARMS]}")


if __name__ == "__main__":
    main()
