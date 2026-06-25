"""Configuration-driven evaluation harness for running pruning suites."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ..tasks.modcog import ensure_modcog_available
from ..utils import make_run_id
from .runner import append_results_csv, run_prune_experiment

SuiteConfig = Tuple[Dict, List[Dict], str, str]


def load_suite_config(path: str) -> SuiteConfig:
    """Load a JSON suite config returning (defaults, runs, output_csv, suite_id)."""
    with open(path, "r") as f:
        cfg = json.load(f)
    defaults = cfg.get("defaults", {})
    runs = cfg.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Suite config 'runs' must be a list of run specs.")
    output_csv = cfg.get("output_csv", "")
    suite_id = cfg.get("run_id", make_run_id("suite"))
    _validate_default_schema(defaults)
    return defaults, runs, output_csv, suite_id


def run_suite_from_config(path: str) -> str:
    """
    Execute a suite described by a JSON config file.

    Returns the path to the CSV containing aggregated results (if any).
    """
    defaults, runs, output_csv, suite_id = load_suite_config(path)
    if len(runs) == 0:
        raise ValueError("Suite config contains no runs.")

    _preflight_suite_runs(defaults, runs, suite_id)

    results_accum: List[Dict] = []
    csv_path = output_csv or os.path.join("results", f"{suite_id}.csv")
    Path(os.path.dirname(csv_path) or ".").mkdir(parents=True, exist_ok=True)

    reset_results = bool(defaults.pop("reset_results", False))
    resume = bool(defaults.pop("resume", True))
    if reset_results and os.path.exists(csv_path):
        os.remove(csv_path)

    base_models: Dict[Tuple, any] = {}
    quick_factor = defaults.pop("train_steps_factor", None)
    eval_steps_factor = defaults.pop("eval_steps_factor", None)

    # Build set of existing run_ids in CSV to support resuming
    completed_run_ids = set()
    if resume and os.path.exists(csv_path):
        import csv

        with open(csv_path, "r") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames and "run_id" in reader.fieldnames:
                for row in reader:
                    completed_run_ids.add(row.get("run_id"))

    for idx, spec in enumerate(runs, start=1):
        merged = {**defaults, **spec}
        run_id = merged.get("run_id") or f"{suite_id}_{idx}"
        merged["run_id"] = run_id
        merged.setdefault("model_type", defaults.get("model_type", "ctrnn") if isinstance(defaults, dict) else "ctrnn")

        if run_id in completed_run_ids:
            print(f"[suite:{suite_id}] ({idx}/{len(runs)}) skipping {run_id} (already completed)")
            continue

        if quick_factor is not None:
            if "train_steps" in merged:
                merged["train_steps"] = max(1, int(round(merged["train_steps"] * quick_factor)))
            if "ft_steps" in merged:
                merged["ft_steps"] = max(0, int(round(merged["ft_steps"] * quick_factor)))
        if eval_steps_factor is not None:
            for key_name, default_val in (
                ("eval_steps_pre0", 50),
                ("eval_steps_pre", 100),
                ("eval_steps_post0", 100),
                ("eval_steps_post", 100),
            ):
                base_val = merged.get(key_name, default_val)
                merged[key_name] = max(1, int(round(base_val * eval_steps_factor)))
        hidden_default = defaults.get("hidden_size") if isinstance(defaults, dict) else None
        hidden_size = merged.get("hidden_size", hidden_default)
        model_kwargs_key = tuple(
            sorted(
                (k, merged.get(k))
                for k in ("use_dale", "ei_ratio", "no_self_connections", "activation")
                if k in merged
            )
        )
        key = (
            merged.get("task"),
            merged.get("seed"),
            merged.get("load_model_path"),
            hidden_size,
            merged.get("model_type"),
            model_kwargs_key,
        )
        base_model = base_models.get(key)
        print(
            f"[suite:{suite_id}] ({idx}/{len(runs)}) running {merged['strategy']} "
            f"amount={merged.get('amount')} seed={merged.get('seed', 'NA')} run_id={run_id}"
        )
        try:
            row, model = run_prune_experiment(**merged, base_model=base_model, return_model=True)
        except Exception as exc:  # pragma: no cover - fail-safe for long suites
            err_row = {
                "run_id": run_id,
                "task": merged.get("task"),
                "strategy": merged.get("strategy"),
                "amount": merged.get("amount"),
                "seed": merged.get("seed"),
                "error": repr(exc),
            }
            append_results_csv([err_row], csv_path)
            print(f"[suite:{suite_id}] ({idx}/{len(runs)}) failed {run_id}: {exc}")
            continue
        results_accum.append(row)
        append_results_csv([row], csv_path)
        if (merged.get("strategy") == "none" or merged.get("no_prune")) and model is not None:
            base_models[key] = model
    return csv_path


def _preflight_suite_runs(defaults: Dict, runs: List[Dict], suite_id: str) -> None:
    """
    Validate optional dependencies and checkpoint availability before launching a suite.
    """
    merged_runs: List[Dict] = []
    modcog_tasks = set()
    manifest_cache: Dict[Path, Dict] = {}
    for idx, spec in enumerate(runs, start=1):
        merged = {**defaults, **spec}
        merged.setdefault("run_id", f"{suite_id}_{idx}")
        merged_runs.append(merged)
        task = str(merged.get("task", ""))
        if task.startswith("modcog:"):
            modcog_tasks.add(task)
    if modcog_tasks:
        ensure_modcog_available(tuple(modcog_tasks))

    produced_paths = set()
    missing: List[str] = []
    for merged in merged_runs:
        run_id = merged["run_id"]
        load_path = merged.get("load_model_path")
        if load_path:
            load_path = os.path.expanduser(load_path)
            if not (os.path.exists(load_path) or load_path in produced_paths):
                missing.append(f"{run_id} -> {load_path}")
            manifest = _load_manifest_for_checkpoint(load_path, manifest_cache)
            if manifest:
                entry = _find_manifest_entry(manifest, load_path)
                if entry is None:
                    raise FileNotFoundError(
                        f"Checkpoint {load_path} referenced by {run_id} is not recorded in the accompanying baseline_manifest.json"
                    )
        save_path = merged.get("save_model_path")
        if save_path:
            produced_paths.add(os.path.expanduser(save_path))
    if missing:
        raise FileNotFoundError(
            "Suite references checkpoints that do not exist and are not produced earlier "
            f"in the schedule: {', '.join(missing)}"
        )


def _validate_default_schema(defaults: Dict) -> None:
    """Best-effort validation/linting for suite defaults."""
    if not isinstance(defaults, dict):
        raise TypeError("Suite defaults must be a dictionary.")
    required = {
        "hidden_size",
        "train_steps",
        "ft_steps",
        "last_only",
        "device",
        "movement_batches",
        "ng_T",
        "ng_B",
        "eval_sample_batches",
    }
    missing = sorted(required - set(defaults.keys()))
    if missing:
        raise ValueError(f"Suite defaults missing required keys: {', '.join(missing)}")
    if defaults.get("eval_sample_batches", 0) <= 0:
        raise ValueError("eval_sample_batches must be > 0 for deterministic evaluation in suites.")


def _load_manifest_for_checkpoint(path: str, cache: Dict[Path, Dict]) -> Optional[Dict]:
    manifest_path = Path(path).expanduser().parent / "baseline_manifest.json"
    if not manifest_path.exists():
        return None
    manifest_key = manifest_path.resolve()
    if manifest_key not in cache:
        with open(manifest_path, "r") as fh:
            cache[manifest_key] = json.load(fh)
    return cache[manifest_key]


def _find_manifest_entry(manifest: Dict[str, Dict], checkpoint_path: str) -> Optional[Dict]:
    normalized = str(Path(checkpoint_path))
    resolved = str(Path(checkpoint_path).resolve())
    if normalized in manifest:
        return manifest[normalized]
    for record in manifest.values():
        if record.get("abs_path") == resolved or record.get("path") == normalized:
            return record
    return None


__all__ = ["load_suite_config", "run_suite_from_config"]
