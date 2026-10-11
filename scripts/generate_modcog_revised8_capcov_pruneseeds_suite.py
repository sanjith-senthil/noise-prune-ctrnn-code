#!/usr/bin/env python3
"""Extra pruning draws for the H = 1024 capped-covariance arms.

Reviewer 1's comment 5 moves the primary inference to task level (n = 8 tasks),
and at that level the H = 1024 covariance-contribution-under-a-cap result becomes
a non-significant trend (p = 0.17 at 50% sparsity, 0.24 at 60%).

That is partly self-inflicted. The capped *magnitude* arms were run at a single
pruning seed while the capped full-score arms they pair against already carry
three, so the contrast inherits a full pruning-draw's worth of noise on one side
and none of the averaging on the other. Measured on the three-seed arm, that
noise is **64% of the task-level variance at 50% sparsity**. Averaging three
draws instead of one raises the standardised effect from 0.54 to 0.71 and cuts
the tasks needed for 80% power from 29 to 18.

This adds pruning seeds 1 and 2 for `q50_mag` and `q60_mag` at 50% and 60%
sparsity, writing into the existing suite directory so the analysis simply sees
three draws where it saw one. Nothing already on disk is re-run: `resume` keys on
run_id and only seeds 1 and 2 are new.

It does not change the effect, only its precision -- so it may still not cross
0.05, and the honest outcome is a tighter interval either way.

2 arms x 2 sparsities x 8 tasks x 3 network seeds x 2 new pruning seeds = 192 runs.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "capcov", Path(__file__).with_name("generate_modcog_revised8_size_capcov_suite.py"))
_cc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cc)
_st = _cc._st

H = 1024
TASKS, NETWORK_SEEDS = _st.TASKS, _st.NETWORK_SEEDS
AMOUNTS = (0.5, 0.6)
NEW_PRUNING_SEEDS = (1, 2)
ARMS = (("q50_mag", _cc.MAG_CAPPED, 0.50), ("q60_mag", _cc.MAG_CAPPED, 0.60))
STEM = _cc.suite_stem(H)          # write into the existing capcov suite


def make_unit(task, ng_t, amount):
    runs = []
    for label, strategy, cap_q in ARMS:
        for seed in NETWORK_SEEDS:
            common = {
                "task": task, "ng_T": ng_t,
                "load_model_path": _st.checkpoint_path(H, task.replace("modcog:", ""), seed),
                "source_model_label": (f"{task.replace('modcog:', '')}_tanh_h{H}"
                                       f"_12k_lr0006_seqbest_no_l2_seed{seed}"),
                "source_run_id": (f"continue_modcog_{task.replace('modcog:', '')}_tanh_h{H}"
                                  f"_to12k_lr0006_seqbest_no_l2_seed{seed}"),
                "source_network_seed": seed, "source_recurrent_l2_lambda": 0.0,
                "eval_seed": 200_000,
                "eval_batches_path": _st.eval_batches_path(task.replace("modcog:", "")),
            }
            for pseed in NEW_PRUNING_SEEDS:
                runs.append({
                    "run_id": (f"capcov{H}_{task.replace('modcog:', '')}_netseed{seed}_"
                               f"{label}_p{int(amount*100)}_pruneseed{pseed}"),
                    "strategy": strategy, "amount": amount,
                    "seed": pseed, "pruning_seed": pseed,
                    "noise_rng_seed": pseed, "prob_control_seed": pseed,
                    "score_batch_seed": 100_000 + pseed,
                    "score_batches_path": _st.score_batches_path(
                        task.replace("modcog:", ""), pseed),
                    "rescale_cap_mode": "quantile", "rescale_cap_quantile": cap_q,
                    **common})
    label_t = task.replace("modcog:", "")
    return {"run_id": f"{STEM}_{label_t}_p{int(amount*100)}_pruneseeds12",
            "output_csv": f"results/{STEM}/{label_t}_p{int(amount*100)}.csv",
            "defaults": _st.defaults_block(H), "runs": runs}


def main():
    ap = argparse.ArgumentParser(); ap.parse_args()
    cfg = Path(f"configs/{STEM}_pruneseeds12"); cfg.mkdir(parents=True, exist_ok=True)
    total = n = 0
    for task_label, task, ng_t in TASKS:
        for amount in AMOUNTS:
            u = make_unit(task, ng_t, amount)
            (cfg / f"{STEM}_{task_label}_p{int(amount*100)}_pruneseeds12.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs to {cfg}; total runs {total}")


if __name__ == "__main__":
    main()
