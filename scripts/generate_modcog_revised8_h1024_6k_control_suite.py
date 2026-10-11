#!/usr/bin/env python3
"""Does the H=1024 fragility appear early, or accumulate with training?

At H=512 the 6k checkpoints sit well below the 12k ones (0.534 vs 0.676 unpruned)
and prune *better*, which is what established that baseline accuracy drives
retention. At H=1024 both are already at ceiling -- 0.968 at 6k against 0.974 at
12k -- so the same contrast asks a different and sharper question: does the extra
6k steps spent at ceiling damage prunability, or is the network's fragility
already fixed by the time it first solves the task?

  no difference  -> fragility is a property of the solution H=1024 finds, present
                    as soon as it works; prolonged training is not the mechanism
  6k prunes better -> training past ceiling sharpens the solution in a way that
                    pruning destroys, and the H=1024 deficit is partly a
                    training-duration effect after all

Two methods so the answer is not specific to one rule, and magnitude needs no
covariance estimate, which separates score quality from network property.

2 methods x 8 tasks x 3 network seeds x 4 sparsities = 192 runs, pruning seed 0.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "size_taskpres", Path(__file__).with_name("generate_modcog_revised8_size_taskpres_suite.py"))
_st = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_st)

H = 1024
SUITE_STEM = f"task_preservation_tanh_h{H}_modcog_revised8_6k_seqbest_ceiling_control_p50_80"
CONFIG_DIR = Path(f"configs/{SUITE_STEM}")
TASKS, NETWORK_SEEDS, AMOUNTS = _st.TASKS, _st.NETWORK_SEEDS, _st.AMOUNTS
PRUNING_SEED = 0
ARMS = (("snp_rescale", "simulation_noise_prune_rescale", True),
        ("magnitude", "l1_unstructured", False))


def ckpt(task, seed):
    return (f"checkpoints/tanh_h{H}_modcog_revised8_6k_seqbest_no_l2_seed{seed}/"
            f"modcog_{task}_seed{seed}.pt")


def shard(a): return f"_p{int(a*100)}"


def make_unit(task, ng_t, amount, baseline):
    runs = []
    for seed in NETWORK_SEEDS:
        common = {
            "task": f"modcog:{task}", "ng_T": ng_t,
            "load_model_path": ckpt(task, seed),
            "source_model_label": f"{task}_tanh_h{H}_6k_seqbest_no_l2_seed{seed}",
            "source_run_id": f"train_modcog_{task}_tanh_h{H}_6k_seqbest_no_l2_seed{seed}",
            "source_network_seed": seed, "source_recurrent_l2_lambda": 0.0,
            "eval_seed": 200_000, "eval_batches_path": _st.eval_batches_path(task),
        }
        if baseline:
            runs.append({"run_id": f"h1k6k_{task}_netseed{seed}_baseline", "strategy": "none",
                         "amount": 0.0, "no_prune": True, "seed": seed, **common})
        for label, strat, needs_score in ARMS:
            r = {"run_id": (f"h1k6k_{task}_netseed{seed}_{label}_"
                            f"p{int(amount*100)}_pruneseed{PRUNING_SEED}"),
                 "strategy": strat, "amount": amount,
                 "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                 "noise_rng_seed": PRUNING_SEED, **common}
            if needs_score:
                r["score_batch_seed"] = 100_000 + PRUNING_SEED
                r["score_batches_path"] = _st.score_batches_path(task, PRUNING_SEED)
            runs.append(r)
    return {"run_id": f"{SUITE_STEM}_{task}{shard(amount)}",
            "output_csv": f"results/{SUITE_STEM}/{task}{shard(amount)}.csv",
            "defaults": _st.defaults_block(H), "runs": runs}


def main():
    argparse.ArgumentParser().parse_args()
    missing = [ckpt(t, s) for t, _x, _n in TASKS for s in NETWORK_SEEDS
               if not Path(ckpt(t, s)).exists()]
    if missing:
        raise SystemExit("missing checkpoints:\n" + "\n".join(missing))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(f"results/{SUITE_STEM}").mkdir(parents=True, exist_ok=True)
    total = n = 0
    for task, _t, ng_t in TASKS:
        for i, amount in enumerate(AMOUNTS):
            u = make_unit(task, ng_t, amount, baseline=(i == 0))
            (CONFIG_DIR / f"{SUITE_STEM}_{task}{shard(amount)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs; total runs {total}")


if __name__ == "__main__":
    main()
