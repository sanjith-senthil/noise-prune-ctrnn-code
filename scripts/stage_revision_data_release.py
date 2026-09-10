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

SUITE_STEMS = {
    "score_controls": "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_score_control_full_p50_80",
    "low_sparsity": "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_low_sparsity_p10_40",
    "noise_necessity": "task_preservation_tanh_h512_modcog_revised8_12k_seqbest_noise_necessity_p50_80",
}
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

    for label, stem in SUITE_STEMS.items():
        suite_dir = RESULTS / stem
        for source in sorted(suite_dir.glob(f"{stem}_*.csv")):
            pairs.append((source, revision / label / source.name))

    for pattern in DM1ANTI_RESULT_GLOBS:
        for source in sorted(RESULTS.glob(pattern)):
            pairs.append((source, revision / "dm1anti_baselines" / source.name))

    # "suites" is deliberately omitted: everything in it duplicates a file already staged
    # from its primary location, and a deposit should not ship the same table twice.
    for sub in ("reference_tables", "statistics"):
        for source in sorted((ANALYSIS / sub).glob("*.csv")):
            pairs.append((source, revision / sub / source.name))

    for name in DM1ANTI_CHECKPOINT_DIRS:
        for source in sorted((PROJECT / "checkpoints" / name).glob("*.pt")):
            pairs.append((source, deposit / "checkpoints" / name / source.name))

    config_root = ROOT / "configs"
    for stem in SUITE_STEMS.values():
        for source in sorted((config_root / stem).glob("*.json")):
            pairs.append((source, revision / "configs" / stem / source.name))
    for pattern in ("*modcog_dm1anti*.json",):
        for source in sorted(config_root.glob(pattern)):
            pairs.append((source, revision / "configs" / "dm1anti" / source.name))

    return [(src, dst) for src, dst in pairs if src.is_file()]


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
    args = parser.parse_args()

    deposit = args.deposit_dir
    if not deposit.is_dir():
        raise SystemExit(f"deposit directory not found: {deposit}")

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

    before = 0
    manifest_path = deposit / "manifest.json"
    if manifest_path.is_file():
        before = len(json.loads(manifest_path.read_text()).get("files", []))
    count = write_manifests(deposit)
    print(f"manifests rebuilt: {before} -> {count} records "
          f"(MANIFEST.sha256 and manifest.json)")


if __name__ == "__main__":
    main()
