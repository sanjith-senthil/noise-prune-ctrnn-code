#!/usr/bin/env python3
"""Write SHA-256 hashes for the frozen revised-paper artifact set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "paper_artifacts/official_h512_24net"
OUTPUT = OFFICIAL / "data/revised_scope/artifact_manifest.json"
FILES = (
    "README.md",
    "data/revised_scope/README.md",
    "data/revised_scope/DERIVATION_AUDIT.md",
    "data/revised_scope/PAPER_RESULTS_PROVENANCE.md",
    "data/revised_scope/task_preservation/revised_task_preservation_h512_24net_p50_80.csv",
    "data/revised_scope/task_preservation/revised_task_preservation_h512_24net_p50_80_summary.csv",
    "data/revised_scope/task_preservation/modcog_revised8_task_specifications.csv",
    "data/revised_scope/capped_rescale/revised_snp_capped_rescale_probe_h512_24net_p50_80.csv",
    "data/revised_scope/capped_rescale/revised_snp_capped_rescale_probe_h512_24net_p50_80_summary.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_q50_capped_rescale_3seed_full8_p50_80.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_q50_capped_rescale_3seed_full8_p50_80_summary.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_q50_capped_rescale_3seed_full8_p50_80_paired_diffs.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_plus_lnp_q90_1seed_full8_p50_80.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_q10_to_q90_1seed_curve_rows.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_q10_to_q90_1seed_curve_summary.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_q0_to_q100_1seed_plot_points.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_q90_pruneseeds1_2_full8_p50_80.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_q10_to_q90_3seed_curve_rows.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_q10_to_q90_3seed_curve_summary.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_q0_to_q100_3seed_plot_points.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_q0_to_q100_3seed_plot_points_with_sem.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_q10_to_q90_3seed_vs_uncapped_tests.csv",
    "data/revised_scope/capped_rescale/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_q10_to_q90_3seed_vs_q100_plot_level_holm_tests.csv",
    "data/revised_scope/significance/README.md",
    "data/revised_scope/significance/STATISTICAL_REPORTING.md",
    "data/revised_scope/significance/revised_scope_task_only_significance_tests.csv",
    "data/revised_scope/significance/revised_scope_task_only_significance_means.csv",
    "data/revised_scope/significance/paper_relevant_h512_comparison_tests.csv",
    "data/revised_scope/significance/paper_relevant_h512_descriptive_statistics.csv",
    "data/revised_scope/significance/paper_relevant_amplification_descriptive_statistics.csv",
    "data/rescale_value_distributions/rescale_value_histogram_snp_lnp_p80_by_run.csv",
    "data/rescale_value_distributions/rescale_value_histogram_snp_lnp_p80_overall.csv",
    "data/rescale_value_distributions/rescale_value_histogram_snp_lnp_p80_by_run_logbin0p20.csv",
    "data/rescale_value_distributions/rescale_value_histogram_snp_lnp_p80_overall_logbin0p20.csv",
    "data/rescale_value_distributions/rescale_value_run_summary_snp_lnp_p80.csv",
    "configs/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_plus_lnp_q90_1seed_full8_p50_80.json",
    "configs/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_lnp_snp_capped_rescale_missing_q10_q20_q30_q40_q60_q70_q80_q90_pruneseeds1_2_full8_p50_80.json",
    "scripts/generate_tanh_h512_modcog_revised8_lnp_snp_capped_rescale_missing_quantiles_1seed_suite.py",
    "scripts/generate_tanh_h512_modcog_revised8_lnp_snp_capped_rescale_missing_quantiles_pruneseeds1_2_suite.py",
    "scripts/analyze_capped_quantile_plot_significance.py",
    "scripts/coarsen_rescale_value_histograms.py",
    "scripts/run_tanh_h512_modcog_revised8_lnp_snp_capped_rescale_missing_quantiles_1seed.sh",
    "scripts/run_tanh_h512_modcog_revised8_lnp_snp_capped_rescale_missing_quantiles_pruneseeds1_2.sh",
    "scripts/summarize_tanh_h512_modcog_revised8_lnp_snp_cap_percentile_curve_1seed.py",
    "scripts/summarize_tanh_h512_modcog_revised8_lnp_snp_cap_percentile_curve_3seed.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    records = []
    for relative in FILES:
        path = OFFICIAL / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    payload = {
        "artifact_set": "official_h512_24net_revised_scope",
        "frozen_on": "2026-06-06",
        "hash_algorithm": "sha256",
        "files": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT}")
    print(f"files: {len(records)}")


if __name__ == "__main__":
    main()
