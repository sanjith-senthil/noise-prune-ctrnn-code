#!/usr/bin/env python3
"""Neuron-by-neuron homeostatic rescaling over the FULL 10-80% range.

The manuscript reports the homeostatic controls as an average over 10-80%, the
same window as every other retention number in the paper. The neuron-wise arms
did not exist over that window:

  * on the MAGNITUDE mask, which is the mask the manuscript sentence describes,
    row-wise ran at 50-80% and at ONE network seed (n = 8, descriptive only),
    and column-wise was never run at all;
  * on the NOISE-PRUNE mask both directions ran at n = 24 but only at 50-80%,
    because that suite was built to answer reviewer 2's mechanism objection
    rather than to sit in this comparison.

So the numbers could be quoted only over 50-80%, which is a different window
from the rest of the paragraph. This suite closes that.

ARMS

    magnitude mask   l1_unstructured_gain         10-80%  (8 sparsities)
      match_l1_rowwise   restore each neuron's total INCOMING |w| -- synaptic
                         scaling proper, and the only local rule here: row i's
                         factor depends solely on what neuron i receives
      match_l1_colwise   the presynaptic mirror, restoring total OUTGOING |w|

    noise-prune mask  simulation_noise_prune_gain  10-40%  (4 sparsities)
      the same two directions. 50-80% is already on disk at n = 24 in
      ..._synaptic_p50_80 and is NOT re-run: these are deterministic
      mask-plus-scalar arms, untouched by the probability-normalisation
      correction, so the existing cells stay valid and only the gap is filled.
      Re-running the overlap would deposit the same cells twice under two
      stems, which is the convention ..._score_matched_gain already follows.

The magnitude mask IS run over the whole 10-80% range rather than stitched,
because its existing 50-80% rows carry one network seed and the paper's unit is
24 networks. These rows supersede them.

(2 arms x 8 sparsities + 2 arms x 4 sparsities) x 8 tasks x 3 network seeds
= 576 runs, n = 24. One pruning seed: every arm here is a deterministic mask
plus a closed-form per-neuron scalar, so repeated draws would be identical.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "revision_suites",
    Path(__file__).with_name("generate_tanh_h512_modcog_revised8_revision_suites.py"))
_rs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rs)

SUITE_STEM = "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_neuronwise_p10_80"
CONFIG_DIR = Path(f"configs/{SUITE_STEM}")
TASKS, NETWORK_SEEDS = _rs.TASKS, _rs.NETWORK_SEEDS
PRUNING_SEED = 0

FULL = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
LOW = (0.1, 0.2, 0.3, 0.4)

# (label, strategy, gain_mode, amounts)
ARMS = (
    ("maggain_rowwise", "l1_unstructured_gain", "match_l1_rowwise", FULL),
    ("maggain_colwise", "l1_unstructured_gain", "match_l1_colwise", FULL),
    ("snpgain_rowwise", "simulation_noise_prune_gain", "match_l1_rowwise", LOW),
    ("snpgain_colwise", "simulation_noise_prune_gain", "match_l1_colwise", LOW),
)
# only the noise-score arms read the fixed score batches; the magnitude mask is
# computed from the weights alone
SNP = {"simulation_noise_prune_gain"}
ALL_AMOUNTS = tuple(sorted({a for *_, amts in ARMS for a in amts}))


def shard(a): return "" if a is None else f"_p{int(round(a * 100))}"


def make_unit(task, ng_t, amounts, baseline, sh):
    runs = []
    if baseline:
        for seed in NETWORK_SEEDS:
            runs.append({"run_id": f"nw_{task}_netseed{seed}_baseline", "strategy": "none",
                         "amount": 0.0, "no_prune": True, "seed": seed,
                         **_rs._common(task, ng_t, seed)})
    for label, strategy, gain_mode, arm_amounts in ARMS:
        for amount in amounts:
            if amount not in arm_amounts:
                continue
            for seed in NETWORK_SEEDS:
                run = {
                    "run_id": (f"nw_{task}_netseed{seed}_{label}_"
                               f"p{int(round(amount * 100))}_pruneseed{PRUNING_SEED}"),
                    "strategy": strategy, "amount": amount,
                    "gain_mode": gain_mode,
                    "seed": PRUNING_SEED, "pruning_seed": PRUNING_SEED,
                    "noise_rng_seed": PRUNING_SEED,
                    **_rs._common(task, ng_t, seed)}
                if strategy in SNP:
                    run["score_batch_seed"] = 100_000 + PRUNING_SEED
                    run["score_batches_path"] = _rs.score_batches_path(task, PRUNING_SEED)
                runs.append(run)
    return {"run_id": f"{SUITE_STEM}_{task}{shard(sh)}",
            "output_csv": f"results/{SUITE_STEM}/{task}{shard(sh)}.csv",
            "defaults": _rs.defaults_block(), "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-by-sparsity", action="store_true")
    a = ap.parse_args()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(f"results/{SUITE_STEM}").mkdir(parents=True, exist_ok=True)
    total = n = 0
    for task, _t, ng_t in TASKS:
        shards = [(x,) for x in ALL_AMOUNTS] if a.shard_by_sparsity else [ALL_AMOUNTS]
        for i, amounts in enumerate(shards):
            sh = amounts[0] if a.shard_by_sparsity else None
            u = make_unit(task, ng_t, amounts, baseline=(i == 0), sh=sh)
            if not u["runs"]:
                continue
            (CONFIG_DIR / f"{SUITE_STEM}_{task}{shard(sh)}.json").write_text(
                json.dumps(u, indent=2) + "\n")
            total += len(u["runs"]); n += 1
    print(f"wrote {n} configs to {CONFIG_DIR}; total runs {total}")
    for label, strategy, gain_mode, amts in ARMS:
        print(f"  {label:<18} {strategy:<28} {gain_mode:<18} "
              f"{[int(x*100) for x in amts]}")


if __name__ == "__main__":
    main()
