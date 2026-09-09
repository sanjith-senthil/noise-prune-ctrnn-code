#!/usr/bin/env python3
"""Export the derived tables the revision cites, so no number is transcribed by hand.

Several quantities used in the reviewer responses are simple derivations of the frozen
artifacts rather than new experiments -- the chance level, floor-corrected retention,
the recurrent-gain accounting behind the energy argument, the theory gap in ``K``, and
the cap-value anchor.  Computing them ad hoc leaves the manuscript depending on numbers
that exist only in prose.  This script writes each one to a CSV so the text and the
figures draw from the same source.

Outputs (one CSV each, written to ``--out-dir``):

``chance_level_by_task.csv``
    Chance accuracy and, because baselines differ, chance expressed in retention units
    per task.
``floor_corrected_retention_summary.csv``
    ``(acc - chance) / (acc_unpruned - chance)`` by method and sparsity, with a
    one-sample test of whether each method beats chance.
``recurrent_weight_retention.csv``
    Edge fraction versus retained absolute recurrent weight -- the distinction the
    energetic-efficiency argument turns on.
``retention_above_one_counts.csv``
    How often pruning improved on the unpruned network.
``theory_gap_by_sparsity.csv``
    Required ``K`` against the ``K`` the density normalisation actually realises, and
    the epsilon that realised value would certify.
``topk_binding_summary.csv``
    How often the exact-density top-k actually removes edges, for the L-NP rescale arm.
``cap_value_anchor.csv``
    Cap value by quantile and sparsity, and ``cap_value x density``, which is invariant.

Chance is ``1/15``: the readout has 17 units (1 fixation + 16 ring positions) but only
15 are ever targets, because ring position 0 coincides with the fixation label and is
masked from scoring.  This was measured from the frozen evaluation batches and is
identical across all eight tasks; ``--verify-chance-from-batches`` re-derives it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "paper_artifacts/official_h512_24net"
REVISED = OFFICIAL / "data/revised_scope"
MAIN = REVISED / "task_preservation/revised_task_preservation_h512_24net_p50_80.csv"
CAP_CURVE = (
    REVISED / "capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_"
    "lnp_snp_capped_rescale_q10_to_q90_3seed_curve_rows.csv"
)
EVAL_BATCH_DIR = ROOT / "results/fixed_batches/tanh_h512_modcog_revised8_taskpres/eval"

CHANCE_ACCURACY = 1.0 / 15.0
OFFDIAG_EDGES = 512 * 511
HIDDEN_SIZE = 512
NOISE_EPS = 0.3
ALPHA = 0.05
CELL_KEYS = ["method", "pruning_pct", "task_short", "source_network_seed"]


def _sem(values: pd.Series) -> float:
    """Standard error of the mean using the sample SD (ddof=1)."""
    n = values.count()
    return float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")


def verify_chance_from_batches() -> Optional[float]:
    """Re-derive chance accuracy from the frozen evaluation batches.

    Returns the uniform-guess rate ``1/K`` over the target classes that actually
    occur, or ``None`` if torch or the batch files are unavailable.
    """
    try:
        import torch
    except ImportError:
        return None
    files = sorted(EVAL_BATCH_DIR.glob("*_eval_seed200000.pt"))
    if not files:
        return None
    classes: set[int] = set()
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        batches = payload.get("batches", payload) if isinstance(payload, dict) else payload
        for _inputs, targets in list(batches):
            flat = targets.reshape(-1)
            classes.update(int(c) for c in flat[flat >= 0].unique().tolist())
    return 1.0 / len(classes) if classes else None


def load_main() -> pd.DataFrame:
    """Load the frozen main benchmark and attach floor-corrected retention."""
    frame = pd.read_csv(MAIN, low_memory=False)
    frame = frame[frame["method"] != "Baseline"].copy()
    frame["floor_corrected_retention"] = (
        (frame["post_acc_sequence"] - CHANCE_ACCURACY)
        / (frame["baseline_acc_sequence"] - CHANCE_ACCURACY)
    )
    return frame


def chance_by_task(frame: pd.DataFrame) -> pd.DataFrame:
    """Chance accuracy and chance-in-retention-units for each task."""
    out = (
        frame.groupby("task_short", as_index=False)["baseline_acc_sequence"]
        .mean()
        .rename(columns={"baseline_acc_sequence": "unpruned_accuracy_mean"})
    )
    out["chance_accuracy"] = CHANCE_ACCURACY
    out["chance_in_retention_units"] = CHANCE_ACCURACY / out["unpruned_accuracy_mean"]
    out["n_target_classes"] = 15
    return out.sort_values("task_short")


def _holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment; identical to the released ``holm_adjust``."""
    indexed = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [math.nan] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, (idx, value) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * float(value)))
        adjusted[idx] = running
    return adjusted


def floor_corrected_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Floor-corrected retention by method and sparsity, tested against chance.

    Pruning-seed replicates are averaged within each trained network first, so the
    analysis unit is the network (n = 24), matching the paper's convention.

    The above-chance tests are Holm-corrected within each sparsity level, since one
    test is run per method there.  Both the raw and adjusted p-values are reported:
    a method can clear the uncorrected threshold and still fail correction, which is
    the case for S-NP mask at 80% and should not be reported as a firm result.
    """
    cells = frame.groupby(CELL_KEYS, as_index=False)[
        ["sequence_retention", "floor_corrected_retention"]
    ].mean()
    rows = []
    for (method, pct), group in cells.groupby(["method", "pruning_pct"]):
        values = group["floor_corrected_retention"]
        nonzero = values[values != 0.0]
        pvalue = (
            float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox",
                           method="auto").pvalue)
            if len(nonzero) else 1.0
        )
        mean = float(values.mean())
        if pvalue > ALPHA:
            verdict = "indistinguishable from chance"
        else:
            verdict = "above chance" if mean > 0 else "below chance"
        rows.append({
            "method": method,
            "pruning_pct": int(pct),
            "n_networks": int(values.count()),
            "sequence_retention_mean": float(group["sequence_retention"].mean()),
            "floor_corrected_mean": mean,
            "floor_corrected_sd": float(values.std(ddof=1)),
            "floor_corrected_sem": _sem(values),
            "networks_at_or_below_chance": int((values <= 0).sum()),
            "vs_chance_wilcoxon_p": pvalue,
            "vs_chance_verdict_uncorrected": verdict,
        })
    table = pd.DataFrame(rows)
    adjusted: list[float] = [math.nan] * len(table)
    for pct in table["pruning_pct"].unique():
        idx = table.index[table["pruning_pct"] == pct].tolist()
        for position, value in zip(idx, _holm(table.loc[idx, "vs_chance_wilcoxon_p"].tolist())):
            adjusted[position] = value
    table["vs_chance_holm_p"] = adjusted
    table["vs_chance_family_size"] = table.groupby("pruning_pct")["method"].transform("size")
    table["vs_chance_verdict_holm"] = [
        "indistinguishable from chance" if p > ALPHA
        else ("above chance" if m > 0 else "below chance")
        for p, m in zip(table["vs_chance_holm_p"], table["floor_corrected_mean"])
    ]
    return table.sort_values(["pruning_pct", "method"])


def recurrent_weight_retention(frame: pd.DataFrame) -> pd.DataFrame:
    """Edge fraction versus retained absolute recurrent weight.

    The energetic argument depends on which of these metabolic cost tracks: the
    rescale variants remove most synapses while preserving total weight.
    """
    rows = []
    for (method, pct), group in frame.groupby(["method", "pruning_pct"]):
        nonzero = group["post_rec_weight_nz_count"].mean()
        retained_l1 = (group["post_rec_weight_abs_mean_nz"] * group["post_rec_weight_nz_count"]).mean()
        unpruned_l1 = (group["pre_rec_weight_abs_mean"] * OFFDIAG_EDGES).mean()
        rows.append({
            "method": method,
            "pruning_pct": int(pct),
            "nonzero_offdiag_edges": float(nonzero),
            "edge_fraction_retained": float(nonzero / OFFDIAG_EDGES),
            "abs_weight_fraction_retained": float(retained_l1 / unpruned_l1),
        })
    return pd.DataFrame(rows).sort_values(["pruning_pct", "method"])


def retention_above_one(frame: pd.DataFrame) -> pd.DataFrame:
    """Count trained-network cells where pruning beat the unpruned network."""
    cells = frame.groupby(CELL_KEYS, as_index=False)["sequence_retention"].mean()
    rows = []
    for (method, pct), group in cells.groupby(["method", "pruning_pct"]):
        values = group["sequence_retention"]
        rows.append({
            "method": method,
            "pruning_pct": int(pct),
            "n_networks": int(values.count()),
            "n_above_unpruned": int((values > 1.0).sum()),
            "max_retention": float(values.max()),
        })
    return pd.DataFrame(rows).sort_values(["pruning_pct", "method"])


def theory_gap(frame: pd.DataFrame) -> pd.DataFrame:
    """Required K against the K the density normalisation realises.

    The guarantee needs ``K >= 8 ln(N) / eps^2``.  Probabilities are then rescaled to
    hit the target density, so the operating point is ``K * scale_factor``; the epsilon
    that value would certify follows by inverting the same expression.
    """
    lnp = frame[frame["method"] == "L-NP rescale"]
    k_theory = 8.0 * math.log(HIDDEN_SIZE) / (NOISE_EPS ** 2)
    rows = []
    for pct, group in lnp.groupby("pruning_pct"):
        scale = float(group["prune_scale_factor"].mean())
        k_eff = k_theory * scale
        rows.append({
            "pruning_pct": int(pct),
            "hidden_size": HIDDEN_SIZE,
            "eps_nominal": NOISE_EPS,
            "K_theory_required": k_theory,
            "density_scale_factor": scale,
            "K_effective": k_eff,
            "K_effective_over_required": k_eff / k_theory,
            "eps_implied_by_K_effective": math.sqrt(8.0 * math.log(HIDDEN_SIZE) / k_eff),
        })
    return pd.DataFrame(rows).sort_values("pruning_pct")


def topk_binding(frame: pd.DataFrame) -> pd.DataFrame:
    """How often the exact-density top-k removes edges, for L-NP rescale.

    ``prune_kept_edges`` is the Bernoulli survivor count before the top-k, so a value
    below target means the step padded the mask rather than cutting anything.
    """
    lnp = frame[(frame["method"] == "L-NP rescale") & frame["prune_kept_edges"].notna()].copy()
    lnp["target_kept"] = (1.0 - lnp["amount"]) * OFFDIAG_EDGES
    lnp["excess_over_target"] = lnp["prune_kept_edges"] - lnp["target_kept"]
    rows = []
    for pct, group in lnp.groupby("pruning_pct"):
        binding = group[group["excess_over_target"] > 0]
        rows.append({
            "method": "L-NP rescale",
            "pruning_pct": int(pct),
            "n_runs": int(len(group)),
            "bernoulli_kept_mean": float(group["prune_kept_edges"].mean()),
            "target_kept": float(group["target_kept"].mean()),
            "mean_excess_over_target": float(group["excess_over_target"].mean()),
            "runs_where_topk_cuts": int(len(binding)),
            "mean_edges_removed_when_binding": float(binding["excess_over_target"].mean())
            if len(binding) else 0.0,
        })
    return pd.DataFrame(rows).sort_values("pruning_pct")


def cap_value_anchor() -> Optional[pd.DataFrame]:
    """Cap value by quantile and sparsity, plus the invariant ``cap x density``.

    Probabilities are normalised so ``sum(p) = density * N(N-1)``, hence density
    factors out of every quantile of ``1/p`` exactly.  The product is therefore a
    shape constant of the score distribution, independent of the chosen sparsity.
    """
    if not CAP_CURVE.exists():
        return None
    curve = pd.read_csv(CAP_CURVE, low_memory=False)
    curve["cap_percentile"] = (curve["cap_quantile"] * 100).round().astype(int)
    curve["retained_density"] = 1.0 - curve["amount"]
    curve["cap_times_density"] = curve["prune_rescale_cap_value"] * curve["retained_density"]
    out = (
        curve.groupby(["family", "cap_percentile", "amount"], as_index=False)
        .agg(cap_value_mean=("prune_rescale_cap_value", "mean"),
             cap_times_density=("cap_times_density", "mean"))
    )
    out["pruning_pct"] = (out["amount"] * 100).round().astype(int)
    return out.drop(columns="amount").sort_values(["family", "cap_percentile", "pruning_pct"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/revision_reference_tables",
                        help="destination directory for the exported CSVs")
    parser.add_argument("--verify-chance-from-batches", action="store_true",
                        help="re-derive chance accuracy from the frozen evaluation batches "
                             "and fail if it disagrees with the documented value")
    args = parser.parse_args()

    if args.verify_chance_from_batches:
        measured = verify_chance_from_batches()
        if measured is None:
            print("chance verification skipped: torch or the evaluation batches are unavailable")
        elif not math.isclose(measured, CHANCE_ACCURACY, rel_tol=1e-12):
            raise SystemExit(f"measured chance {measured} != documented {CHANCE_ACCURACY}")
        else:
            print(f"chance verified from evaluation batches: {measured:.6f} (1/15)")

    frame = load_main()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "chance_level_by_task.csv": chance_by_task(frame),
        "floor_corrected_retention_summary.csv": floor_corrected_summary(frame),
        "recurrent_weight_retention.csv": recurrent_weight_retention(frame),
        "retention_above_one_counts.csv": retention_above_one(frame),
        "theory_gap_by_sparsity.csv": theory_gap(frame),
        "topk_binding_summary.csv": topk_binding(frame),
    }
    anchor = cap_value_anchor()
    if anchor is not None:
        tables["cap_value_anchor.csv"] = anchor
    else:
        print(f"skipped cap_value_anchor.csv: {CAP_CURVE.name} not found")

    for name, table in tables.items():
        table.to_csv(args.out_dir / name, index=False)
        print(f"wrote {args.out_dir / name}  ({len(table)} rows)")


if __name__ == "__main__":
    main()
